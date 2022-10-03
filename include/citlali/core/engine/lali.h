#pragma once

#include <citlali/core/engine/engine.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

class Lali: public Engine {
public:
    void setup();
    auto run();

    template <class KidsProc, class RawObs>
    auto pipeline(KidsProc &, RawObs &);

    void output();
};

void Lali::setup() {

    // setup kernel
    if (rtcproc.run_kernel) {
        rtcproc.kernel.setup(n_maps);
    }

    // clear for each observation
    fits_io_vec.clear();

    for (Eigen::Index i=0; i<calib.n_arrays; i++) {
        auto array = calib.arrays[i];
        std::string array_name = toltec_io.array_name_map[array];
        auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                  engine_utils::toltecIO::map>(obsnum_dir_name, redu_type, array_name,
                                                                               obsnum, telescope.sim_obs);
        fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);

        fits_io_vec.push_back(std::move(fits_io));
    }

    if (run_tod_output) {
        for (const auto &stokes_param: rtcproc.polarization.stokes_params) {
            auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                     engine_utils::toltecIO::timestream>(obsnum_dir_name, redu_type, "",
                                                                                          obsnum, telescope.sim_obs);

            SPDLOG_INFO("tod_filename {}", tod_filename);
            tod_filename[stokes_param] = filename + "_" + stokes_param + ".nc";
            netCDF::NcFile fo(tod_filename[stokes_param], netCDF::NcFile::replace);
            netCDF::NcDim n_pts_dim = fo.addDim("n_pts");
            netCDF::NcDim n_scan_indices_dim = fo.addDim("n_scan_indices", telescope.scan_indices.rows());
            netCDF::NcDim n_scans_dim = fo.addDim("n_scans", telescope.scan_indices.cols());

            Eigen::Index n_dets;

            if (stokes_param=="I") {
                n_dets = calib.apt["array"].size();
            }

            else if ((stokes_param == "Q") || (stokes_param == "U")) {
                n_dets = (calib.apt["fg"].array() == 0).count() + (calib.apt["fg"].array() == 1).count();
            }

            netCDF::NcDim n_dets_dim = fo.addDim("n_dets", n_dets);

            std::vector<netCDF::NcDim> dims = {n_pts_dim, n_dets_dim};
            std::vector<netCDF::NcDim> scans_dims = {n_scan_indices_dim, n_scans_dim};

            // scan indices
            netCDF::NcVar scan_indices_v = fo.addVar("scan_indices",netCDF::ncInt, scans_dims);
            Eigen::MatrixXI scans_indices_transposed = telescope.scan_indices.transpose();
            scan_indices_v.putVar(scans_indices_transposed.data());

            // scans
            netCDF::NcVar scans_v = fo.addVar("scans",netCDF::ncDouble, dims);
            // flags
            netCDF::NcVar flags_v = fo.addVar("flags",netCDF::ncDouble, dims);
            // kernel
            netCDF::NcVar kernel_v = fo.addVar("kernel",netCDF::ncDouble, dims);

            // add apt table
            for (auto const& x: calib.apt) {
                netCDF::NcVar apt_v = fo.addVar(x.first,netCDF::ncDouble, n_dets_dim);
            }

            // add telescope parameters
            for (auto const& x: telescope.tel_data) {
                netCDF::NcVar tel_data_v = fo.addVar(x.first,netCDF::ncDouble, n_pts_dim);
            }

            // weights
            if (tod_output_type == "ptc") {
                netCDF::NcDim n_weights_dim = fo.addDim("n_weights");
                std::vector<netCDF::NcDim> weight_dims = {n_scans_dim, n_weights_dim};
                netCDF::NcVar weights_v = fo.addVar("weights",netCDF::ncDouble, n_weights_dim);
            }

            fo.close();
        }
    }
}

auto Lali::run() {
    auto farm = grppi::farm(n_threads,[&](auto &input_tuple) -> TCData<TCDataKind::PTC,Eigen::MatrixXd> {
        // RTCData input
        auto rtcdata = std::get<0>(input_tuple);
        // kidsproc
        auto kidsproc = std::get<1>(input_tuple);
        // start index input
        auto scan_rawobs = std::get<2>(input_tuple);

        // starting index for scan
        Eigen::Index si = rtcdata.scan_indices.data(2);

        // current length of outer scans
        Eigen::Index sl = rtcdata.scan_indices.data(3) - rtcdata.scan_indices.data(2) + 1;

        // copy scan's telescope vectors
        for (auto const& x: telescope.tel_data) {
            rtcdata.tel_data.data[x.first] = telescope.tel_data[x.first].segment(si,sl);
        }

        // get hwp
        if (rtcproc.run_polarization) {
            rtcdata.hwp_angle.data = calib.hwp_angle.segment(si, sl);
        }

        // get raw tod from files
        {
            tula::logging::scoped_loglevel<spdlog::level::off> _0;
            rtcdata.scans.data = kidsproc.populate_rtc(scan_rawobs,rtcdata.scan_indices.data, sl, calib.n_dets, tod_type);
        }

        // create PTCData
        TCData<TCDataKind::PTC,Eigen::MatrixXd> ptcdata;

        // loop through polarizations
        for (const auto &stokes_param: rtcproc.polarization.stokes_params) {
            SPDLOG_INFO("reducing {} timestream",stokes_param);
            // create a new rtcdata for each polarization
            TCData<TCDataKind::RTC,Eigen::MatrixXd> rtcdata_pol;
            // demodulate
            SPDLOG_INFO("demodulating polarization");
            auto [array_indices, nw_indices, det_indices] = rtcproc.polarization.demodulate_timestream(rtcdata, rtcdata_pol,
                                                                                                       stokes_param,
                                                                                                       redu_type, calib);

            // get indices for maps
            SPDLOG_INFO("calculating map indices");
            auto map_indices = calc_map_indices(det_indices, nw_indices, array_indices, stokes_param);

            // run rtcproc
            SPDLOG_INFO("rtcproc");
            rtcproc.run(rtcdata_pol, ptcdata, telescope.pixel_axes, redu_type, calib, pointing_offsets_arcsec, det_indices, array_indices,
                        map_indices, omb.pixel_size_rad);

            // write rtc timestreams
            if (run_tod_output) {
                SPDLOG_INFO("writing rtcdata");
                if (tod_output_type == "rtc") {
                    ptcproc.append_to_netcdf(ptcdata, tod_filename[stokes_param], det_indices, calib.apt);
                }
            }

            // remove flagged dets
            SPDLOG_INFO("removing flagged dets");
            ptcproc.remove_flagged_dets(ptcdata, calib.apt, det_indices);

            // run cleaning
            if (stokes_param == "I") {
                SPDLOG_INFO("ptcproc");
                ptcproc.run(ptcdata, ptcdata, calib);
            }
            // calculate weights
            SPDLOG_INFO("calculating weights");
            ptcproc.calc_weights(ptcdata, calib, telescope);

            // remove outliers
            SPDLOG_INFO("removing outlier weights");
            //ptcproc.remove_bad_dets(ptcdata, calib.apt, det_indices);

            // write ptc timestreams
            if (run_tod_output) {
                SPDLOG_INFO("writing ptcdata");
                if (tod_output_type == "ptc") {
                    ptcproc.append_to_netcdf(ptcdata, tod_filename[stokes_param], det_indices, calib.apt);
                }
            }

            // populate maps
            SPDLOG_INFO("populating maps");
            mapmaking::populate_maps_naive(ptcdata, omb, map_indices, det_indices, telescope.pixel_axes,
                                           redu_type, calib.apt, pointing_offsets_arcsec, telescope.d_fsmp);

        }

        return ptcdata;
    });

    return farm;
}

template <class KidsProc, class RawObs>
auto Lali::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    grppi::pipeline(tula::grppi_utils::dyn_ex(parallel_policy),
        [&]() -> std::optional<std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, KidsProc,
                                          std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>>> {

            // variable to hold current scan
            static auto scan = 0;
            while (scan < telescope.scan_indices.cols()) {
                TCData<TCDataKind::RTC, Eigen::MatrixXd> rtcdata;
                rtcdata.scan_indices.data = telescope.scan_indices.col(scan);
                rtcdata.index.data = scan;

                std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> scan_rawobs;
                {
                    tula::logging::scoped_loglevel<spdlog::level::off> _0;
                    scan_rawobs = kidsproc.load_rawobs(rawobs, scan, telescope.scan_indices, start_indices, end_indices);
                }

                scan++;
                return std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, KidsProc,
                                  std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>> (std::move(rtcdata), kidsproc,
                                                                                                  std::move(scan_rawobs));
            }
            scan = 0;
            return {};
        },

        run());

    // normalize maps
    omb.normalize_maps();
}

void Lali::output() {
    SPDLOG_INFO("writing maps");
    Eigen::Index k = 0;
    for (Eigen::Index i=0; i<rtcproc.polarization.stokes_params.size(); i++) {
        for (Eigen::Index j=0; j<n_maps/rtcproc.polarization.stokes_params.size(); j++) {
            fits_io_vec[j].add_hdu("signal_" + rtcproc.polarization.stokes_params[i], omb.signal[k]);
            fits_io_vec[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);

            fits_io_vec[j].add_hdu("weight_" + rtcproc.polarization.stokes_params[i], omb.weight[k]);
            fits_io_vec[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);

            if (rtcproc.run_kernel) {
                fits_io_vec[j].add_hdu("kernel_" + rtcproc.polarization.stokes_params[i], omb.kernel[k]);
                fits_io_vec[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);
            }

            fits_io_vec[j].add_hdu("coverage_" + rtcproc.polarization.stokes_params[i], omb.coverage[k]);
            fits_io_vec[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);

            Eigen::MatrixXd sig2noise = omb.signal[k].array()*sqrt(omb.weight[k].array());
            fits_io_vec[j].add_hdu("sig2noise_" + rtcproc.polarization.stokes_params[i], sig2noise);
            fits_io_vec[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);

            // add telescope file header information
            for (auto const& [key, val] : telescope.tel_header) {
                fits_io_vec[j].pfits->pHDU().addKey(key, val(0), key);
            }
            fits_io_vec[j].pfits->pHDU().addKey("OBSNUM", obsnum, "Observation Number");
            fits_io_vec[j].pfits->pHDU().addKey("CITLALI_GIT_VERSION", CITLALI_GIT_VERSION, "CITLALI_GIT_VERSION");

            k++;
        }
    }
}
