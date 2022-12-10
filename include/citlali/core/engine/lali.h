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
    void pipeline(KidsProc &, RawObs &);

    template <mapmaking::MapType map_type>
    void output();
};

void Lali::setup() {    
    // setup kernel
    if (rtcproc.run_kernel) {
        rtcproc.kernel.setup(n_maps);
    }

    // set despiker sample rate
    rtcproc.despiker.fsmp = telescope.fsmp;

    // if filter is requested, make it here
    if (rtcproc.run_tod_filter) {
        rtcproc.filter.make_filter(telescope.fsmp);
    }

    // set center pointing
    if (telescope.pixel_axes == "icrs") {
        omb.wcs.crval[0] = telescope.tel_header["Header.Source.Ra"](0);
        omb.wcs.crval[1] = telescope.tel_header["Header.Source.Dec"](0);

        cmb.wcs.crval[0] = telescope.tel_header["Header.Source.Ra"](0);
        cmb.wcs.crval[1] = telescope.tel_header["Header.Source.Dec"](0);
    }

    // create output map files
    if (run_mapmaking) {
        create_map_files();
    }
    // create timestream files
    if (run_tod_output) {
        create_tod_files();
    }

    // print basic info for obs reduction
    print_summary();
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
        for (const auto &[stokes_index,stokes_param]: rtcproc.polarization.stokes_params) {
            SPDLOG_INFO("starting scan {}. {}/{} scans completed", rtcdata.index.data + 1, n_scans_done, telescope.scan_indices.cols());

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
            rtcproc.run(rtcdata_pol, ptcdata, telescope.pixel_axes, redu_type, calib, telescope, pointing_offsets_arcsec, det_indices,
                        array_indices, map_indices, omb.pixel_size_rad);

            // write rtc timestreams
            if (run_tod_output) {
                if (tod_output_type == "rtc") {
                    SPDLOG_INFO("writing rtcdata");
                    ptcproc.append_to_netcdf(ptcdata, tod_filename[stokes_param], redu_type, telescope.pixel_axes, pointing_offsets_arcsec,
                                             det_indices, calib.apt, tod_output_type, verbose_mode, telescope.d_fsmp);
                }
            }

            // remove flagged dets
            SPDLOG_INFO("removing flagged dets");
            ptcproc.remove_flagged_dets(ptcdata, calib.apt, det_indices);

            // remove outliers
            SPDLOG_INFO("removing outlier weights");
            auto calib_scan = ptcproc.remove_bad_dets_nw(ptcdata, calib, det_indices, nw_indices, array_indices);

            // run cleaning
            if (stokes_param == "I") {
                SPDLOG_INFO("ptcproc");
                ptcproc.run(ptcdata, ptcdata, calib);
            }

            // calculate weights
            SPDLOG_INFO("calculating weights");
            ptcproc.calc_weights(ptcdata, calib.apt, telescope);

            if (verbose_mode) {
                write_chunk_summary(ptcdata);
            }

            // write ptc timestreams
            if (run_tod_output) {
                if (tod_output_type == "ptc") {
                    SPDLOG_INFO("writing ptcdata");
                    ptcproc.append_to_netcdf(ptcdata, tod_filename[stokes_param], redu_type, telescope.pixel_axes, pointing_offsets_arcsec,
                                             det_indices, calib.apt, tod_output_type, verbose_mode, telescope.d_fsmp);
                }
            }

            // populate maps
            if (run_mapmaking) {
                SPDLOG_INFO("populating maps");
                mapmaking::populate_maps_naive(ptcdata, omb, cmb, map_indices, det_indices, telescope.pixel_axes,
                                               redu_type, calib.apt, pointing_offsets_arcsec, telescope.d_fsmp, run_noise);
            }
        }

        n_scans_done++;
        SPDLOG_INFO("done with scan {}. {}/{} scans completed", ptcdata.index.data + 1, n_scans_done, telescope.scan_indices.cols());

        return ptcdata;
    });

    return farm;
}

template <class KidsProc, class RawObs>
void Lali::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    // initialize number of completed scans
    n_scans_done = 0;

    // progress bar
    tula::logging::progressbar pb(
        [](const auto &msg) { SPDLOG_INFO("{}", msg); }, 100, "citlali progress ");

    grppi::pipeline(tula::grppi_utils::dyn_ex(parallel_policy),
        [&]() -> std::optional<std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, KidsProc,
                                          std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>>> {
            // variable to hold current scan
            static auto scan = 0;
            // loop through scans
            while (scan < telescope.scan_indices.cols()) {

                // update progress bar
                pb.count(telescope.scan_indices.cols(), 1);

                // create rtcdata
                TCData<TCDataKind::RTC, Eigen::MatrixXd> rtcdata;
                // get scan indices
                rtcdata.scan_indices.data = telescope.scan_indices.col(scan);
                // current scan
                rtcdata.index.data = scan;

                // vector to store kids data
                std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> scan_rawobs;
                {
                    tula::logging::scoped_loglevel<spdlog::level::off> _0;
                    // get kids data
                    scan_rawobs = kidsproc.load_rawobs(rawobs, scan, telescope.scan_indices, start_indices, end_indices);
                }

                // increment scan
                scan++;
                return std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, KidsProc,
                                  std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>> (std::move(rtcdata), kidsproc,
                                                                                                  std::move(scan_rawobs));
            }
            // reset scan to zero for each obs
            scan = 0;
            return {};
        },

        // run the farm
        run());

    if (run_mapmaking) {
        // normalize maps
        SPDLOG_INFO("normalizing maps");
        omb.normalize_maps();
        // calculate map psds
        SPDLOG_INFO("calculating map psd");
        omb.calc_map_psd();
        // calculate map histograms
        SPDLOG_INFO("calculating map histogram");
        omb.calc_map_hist();

        // write map summary
        if (verbose_mode) {
            write_map_summary(omb);
        }
    }
}

template <mapmaking::MapType map_type>
void Lali::output() {
    // pointer to map buffer
    mapmaking::ObsMapBuffer* mb = NULL;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = NULL;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = NULL;

    // directory name
    std::string dir_name;

    // raw obs maps
    if constexpr (map_type == mapmaking::RawObs) {
        mb = &omb;
        f_io = &fits_io_vec;
        n_io = &noise_fits_io_vec;
        dir_name = obsnum_dir_name + "/raw/";
    }

    // filtered obs maps
    else if constexpr (map_type == mapmaking::FilteredObs) {
        mb = &omb;
        f_io = &filtered_fits_io_vec;
        n_io = &filtered_noise_fits_io_vec;
        dir_name = obsnum_dir_name + "/filtered/";
    }

    // raw coadded maps
    else if constexpr (map_type == mapmaking::RawCoadd) {
        mb = &cmb;
        f_io = &coadd_fits_io_vec;
        n_io = &coadd_noise_fits_io_vec;
        dir_name = coadd_dir_name + "/raw/";
    }

    // filtered coadded maps
    else if constexpr (map_type == mapmaking::FilteredCoadd) {
        mb = &cmb;
        f_io = &filtered_coadd_fits_io_vec;
        n_io = &filtered_coadd_noise_fits_io_vec;
        dir_name = coadd_dir_name + "/filtered/";
    }

    for (Eigen::Index i=0; i<n_maps; i++) {
        write_maps(f_io,n_io,mb,i);
    }

    // clear fits file vectors to ensure its closed.
    f_io->clear();
    n_io->clear();

    // write psd and histogram files
    write_psd<map_type>(mb, dir_name);
    write_hist<map_type>(mb, dir_name);

    mb = NULL;
    f_io = NULL;
    n_io = NULL;
}
