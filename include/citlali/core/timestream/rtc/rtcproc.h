#pragma once

#include <tula/algorithm/ei_stats.h>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/timestream/rtc/polarization_3.h>
#include <citlali/core/timestream/rtc/kernel.h>
#include <citlali/core/timestream/rtc/despike.h>
#include <citlali/core/timestream/rtc/filter.h>
#include <citlali/core/timestream/rtc/downsample.h>
#include <citlali/core/timestream/rtc/calibrate.h>

namespace timestream {

using timestream::TCData;

class RTCProc {
public:
    // controls for timestream reduction
    bool run_timestream;
    bool run_polarization;
    bool run_kernel;
    bool run_despike;
    bool run_tod_filter;
    bool run_downsample;
    bool run_calibrate;
    bool run_extinction;

    // rtc tod classes
    timestream::Polarization polarization;
    timestream::Kernel kernel;
    timestream::Despiker despiker;
    timestream::Filter filter;
    timestream::Downsampler downsampler;
    timestream::Calibration calibration;

    // upper and lower limits for outliers
    double lower_weight_factor, upper_weight_factor;
    // minimum allowed frequency distance between tones
    double delta_f_min_Hz;

    // number of weight outlier iterations
    int iter_lim = 0;

    template <typename Derived, class calib_t>
    auto get_grouping(std::string, Eigen::DenseBase<Derived> &, calib_t &, int);

    template <class calib_t, typename Derived>
    auto calc_map_indices(calib_t &, Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &,
                          Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string, std::string);


    template<typename calib_t, typename telescope_t>
    auto run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &,
             TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string &,
             std::string &, calib_t &, telescope_t &, double, std::string, std::string);

    template <typename calib_t, typename Derived>
    auto remove_nearby_tones(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                             std::string);

    template <typename calib_t, typename Derived>
    auto remove_bad_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                         Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string, std::string);
};

template <typename Derived, class calib_t>
auto RTCProc::get_grouping(std::string grp, Eigen::DenseBase<Derived> &det_indices, calib_t &calib, int n_dets) {
    std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grp_limits;

    Eigen::Index grp_i = calib.apt[grp](det_indices(0));
    grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{0, 0};
    Eigen::Index j = 0;
    // loop through apt table arrays, get highest index for current array
    for (Eigen::Index i=0; i<n_dets; i++) {
        auto det_index = det_indices(i);
        if (calib.apt[grp](det_index) == grp_i) {
            std::get<1>(grp_limits[grp_i]) = i + 1;
        }
        else {
            grp_i = calib.apt[grp](det_index);
            j += 1;
            grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{i, 0};
        }
    }
    return grp_limits;
}

template <class calib_t, typename Derived>
auto RTCProc::calc_map_indices(calib_t &calib, Eigen::DenseBase<Derived> &det_indices, Eigen::DenseBase<Derived> &nw_indices,
                               Eigen::DenseBase<Derived> &array_indices, Eigen::DenseBase<Derived> &fg_indices,
                               std::string stokes_param, std::string map_grouping) {
    // indices for maps
    Eigen::VectorXI indices(array_indices.size()), map_indices(array_indices.size());

    // number of maps from grouping
    int n_maps = 0;

    // overwrite map indices for networks
    if (map_grouping == "nw") {
        indices = nw_indices;
        n_maps = calib.n_nws;
    }

    // overwrite map indices for arrays
    else if (map_grouping == "array") {
        indices = array_indices;
        n_maps = calib.n_arrays;
    }

    // overwrite map indices for detectors
    else if (map_grouping == "detector") {
        indices = det_indices;
        n_maps = calib.n_dets;
    }

    // overwrite map indices for fg
    else if (map_grouping == "fg") {
        indices = fg_indices;
        n_maps = calib.fg.size()*calib.n_arrays;
    }

    // start at 0
    if (map_grouping != "fg") {
        Eigen::Index map_index = 0;
        map_indices(0) = 0;
        // loop through and populate map indices
        for (Eigen::Index i=0; i<indices.size()-1; i++) {
            // if next index is larger than current index, increment map index
            if (indices(i+1) > indices(i)) {
                map_index++;
            }
            map_indices(i+1) = map_index;
        }
    }
    else {
        // convert fg to indices
        std::map<Eigen::Index, Eigen::Index> fg_to_index, array_to_index;

        // get mapping from fg to map index
        for (Eigen::Index i=0; i<calib.fg.size(); i++) {
            fg_to_index[calib.fg(i)] = i;
        }

        // get mapping from fg to map index
        for (Eigen::Index i=0; i<calib.arrays.size(); i++) {
            array_to_index[calib.arrays(i)] = i;
        }

        // allocate map indices from fg
        for (Eigen::Index i=0; i<indices.size(); i++) {
            map_indices(i) = fg_to_index[indices(i)] + calib.fg.size()*array_to_index[array_indices(i)];
        }
    }

    // increment if polarization is enabled
    if (run_polarization) {
        if (stokes_param == "Q") {
            // stokes Q takes the second set of n_maps
            map_indices = map_indices.array() + n_maps;
        }
        else if (stokes_param == "U") {
            // stokes U takes the third set of n_maps
            map_indices = map_indices.array() + 2*n_maps;
        }
    }

    // return the map indices
    return std::move(map_indices);
}

template<class calib_t, typename telescope_t>
auto RTCProc::run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in,
                  TCData<TCDataKind::PTC, Eigen::MatrixXd> &out, std::string &pixel_axes,
                  std::string &redu_type, calib_t &calib, telescope_t &telescope, double pixel_size_rad,
                  std::string stokes_param, std::string map_grouping) {

    // new timechunk for current stokes parameter timestream
    TCData<TCDataKind::RTC,Eigen::MatrixXd> in_pol;

    // calculate the stokes timestream
    auto [array_indices, nw_indices, det_indices, fg_indices] = polarization.demodulate_timestream(in, in_pol, stokes_param,
                                                                                                   redu_type, calib, telescope.sim_obs);

    // resize fcf
    in_pol.fcf.data.setOnes(in_pol.scans.data.cols());

    // get indices for maps
    SPDLOG_DEBUG("calculating map indices");
    auto map_indices = calc_map_indices(calib, det_indices, nw_indices, array_indices, fg_indices, stokes_param, map_grouping);

    if (run_calibrate) {
        SPDLOG_DEBUG("calibrating timestream");
        // calibrate tod
        calibration.calibrate_tod(in_pol, det_indices, array_indices, calib);

        out.calibrated = true;
    }

    if (run_extinction) {
        SPDLOG_DEBUG("correcting extinction");
        // calc tau at toltec frequencies
        auto tau_freq = calibration.calc_tau(in_pol.tel_data.data["TelElAct"], telescope.tau_225_GHz);
        // correct for extinction
        calibration.extinction_correction(in_pol, det_indices, array_indices, calib, tau_freq);
    }

    // number of points in scan
    Eigen::Index n_pts = in_pol.scans.data.rows();

    // start index of inner scans
    auto si = filter.n_terms;
    // end index of inner scans
    auto sl = in_pol.scan_indices.data(1) - in_pol.scan_indices.data(0) + 1;

    // set up flags
    in_pol.flags.data.setZero(in_pol.scans.data.rows(), in_pol.scans.data.cols());

    // create kernel if requested
    if (run_kernel) {
        SPDLOG_DEBUG("creating kernel timestream");
        // symmetric gaussian kernel
        if (kernel.type == "gaussian") {
            SPDLOG_DEBUG("creating symmetric gaussian kernel");
            kernel.create_symmetric_gaussian_kernel(in_pol, pixel_axes, redu_type, calib.apt, in_pol.pointing_offsets_arcsec.data,
                                                    det_indices);
        }
        // airy kernel
        else if (kernel.type == "airy") {
            SPDLOG_DEBUG("creating airy kernel");
            kernel.create_airy_kernel(in_pol, pixel_axes, redu_type, calib.apt, in_pol.pointing_offsets_arcsec.data,
                                      det_indices);
        }
        // get kernel from fits
        else if (kernel.type == "fits") {
            SPDLOG_DEBUG("getting kernel from fits");
            kernel.create_kernel_from_fits(in_pol, pixel_axes, redu_type, calib.apt, in_pol.pointing_offsets_arcsec.data,
                                           pixel_size_rad, map_indices, det_indices);
        }

        out.kernel_generated = true;
    }

    // run despiking
    if (run_despike) {
        SPDLOG_DEBUG("despiking");
        // despike data
        despiker.despike(in_pol.scans.data, in_pol.flags.data, calib.apt);

        // we want to replace spikes on a per array or network basis
        auto grp_limits = get_grouping(despiker.grouping, det_indices, calib, in_pol.scans.data.cols());

        /*std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grp_limits;

        Eigen::Index grp_i = calib.apt[despiker.grouping](det_indices(0));
        grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{0, 0};
        Eigen::Index j = 0;
        // loop through apt table arrays, get highest index for current array
        for (Eigen::Index i=0; i<in_pol.scans.data.cols(); i++) {
            auto det_index = det_indices(i);
            if (calib.apt[despiker.grouping](det_index) == grp_i) {
                std::get<1>(grp_limits[grp_i]) = i + 1;
            }
            else {
                grp_i = calib.apt[despiker.grouping](det_index);
                j += 1;
                grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{i, 0};
            }
        }*/

        SPDLOG_DEBUG("replacing spikes");
        for (auto const& [key, val] : grp_limits) {
            // starting index
            auto start_index = std::get<0>(val);
            // size of block for each grouping
            auto n_dets = std::get<1>(val) - std::get<0>(val);

            // get the reference block of in scans that corresponds to the current array
            Eigen::Ref<Eigen::MatrixXd> in_scans_ref = in_pol.scans.data.block(0, start_index, n_pts, n_dets);

            Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>
                in_scans(in_scans_ref.data(), in_scans_ref.rows(), in_scans_ref.cols(),
                         Eigen::OuterStride<>(in_scans_ref.outerStride()));

            // get the block of in flags that corresponds to the current array
            Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags_ref =
                in_pol.flags.data.block(0, start_index, n_pts, n_dets);

            Eigen::Map<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>, 0, Eigen::OuterStride<> >
                in_flags(in_flags_ref.data(), in_flags_ref.rows(), in_flags_ref.cols(),
                         Eigen::OuterStride<>(in_flags_ref.outerStride()));

            // replace spikes
            despiker.replace_spikes(in_scans, in_flags, calib.apt, start_index);
        }

        out.despiked = true;
    }

    // timestream filtering
    if (run_tod_filter) {
        SPDLOG_DEBUG("convolving signal with tod filter");
        filter.convolve(in_pol.scans.data);
        //filter.iir(in_pol.scans.data);

        // filter kernel
        if (run_kernel) {
            SPDLOG_DEBUG("convolving kernel with tod filter");
            filter.convolve(in_pol.kernel.data);
        }

        out.tod_filtered = true;
    }

    if (run_downsample) {
        SPDLOG_DEBUG("downsampling data");
        // get the block of out scans that corresponds to the inner scan indices
        Eigen::Ref<Eigen::Map<Eigen::MatrixXd>> in_scans =
            in_pol.scans.data.block(si, 0, sl, in_pol.scans.data.cols());

        // get the block of in flags that corresponds to the inner scan indices
        Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags =
            in_pol.flags.data.block(si, 0, sl, in_pol.flags.data.cols());

        downsampler.downsample(in_scans, out.scans.data);
        downsampler.downsample(in_flags, out.flags.data);

        // loop through telescope meta data and downsample
        SPDLOG_DEBUG("downsampling telescope");
        for (auto const& x: in_pol.tel_data.data) {
            // get the block of in tel data that corresponds to the inner scan indices
            Eigen::Ref<Eigen::VectorXd> in_tel =
                in_pol.tel_data.data[x.first].segment(si, sl);

            downsampler.downsample(in_tel, out.tel_data.data[x.first]);
        }

        // downsample pointing
        for (auto const& x: in_pol.pointing_offsets_arcsec.data) {
        Eigen::Ref<Eigen::VectorXd> in_pointing =
            in_pol.pointing_offsets_arcsec.data[x.first].segment(si, sl);

            downsampler.downsample(in_pointing, out.pointing_offsets_arcsec.data[x.first]);
        }

        // downsample hwpr
        if (run_polarization && calib.run_hwp) {
            Eigen::Ref<Eigen::VectorXd> in_hwp =
                in_pol.hwp_angle.data.segment(si, sl);
            downsampler.downsample(in_hwp, in_pol.hwp_angle.data);
        }
        // downsample kernel if requested
        if (run_kernel) {
            SPDLOG_DEBUG("downsampling kernel");
            // get the block of in kernel scans that corresponds to the inner scan indices
            Eigen::Ref<Eigen::MatrixXd> in_kernel =
                in_pol.kernel.data.block(si, 0, sl, in_pol.kernel.data.cols());

            downsampler.downsample(in_kernel, out.kernel.data);
        }

        out.downsampled = true;
    }

    else {
        // copy data
        out.scans.data = in_pol.scans.data.block(si, 0, sl, in_pol.scans.data.cols());
        // copy flags
        out.flags.data = in_pol.flags.data.block(si, 0, sl, in_pol.flags.data.cols());
        // copy kernel
        if (run_kernel) {
            out.kernel.data = in_pol.kernel.data.block(si, 0, sl, in_pol.kernel.data.cols());
        }
        // copy telescope data
        for (auto const& x: in_pol.tel_data.data) {
            out.tel_data.data[x.first] = in_pol.tel_data.data[x.first].segment(si, sl);
        }

        // copy pointing offsets
        for (auto const& x: in_pol.pointing_offsets_arcsec.data) {
            out.pointing_offsets_arcsec.data[x.first] = in_pol.pointing_offsets_arcsec.data[x.first].segment(si, sl);
        }

        // copy hwpr angle
        if (run_polarization) {
            if (calib.run_hwp) {
                out.hwp_angle.data = in_pol.hwp_angle.data.segment(si, sl);
            }
        }
    }

    out.scan_indices.data = in_pol.scan_indices.data;
    out.index.data = in_pol.index.data;
    out.fcf.data = in_pol.fcf.data;

    return std::tuple<Eigen::VectorXI,Eigen::VectorXI,Eigen::VectorXI,Eigen::VectorXI>(map_indices, array_indices, nw_indices, det_indices);
}

template <typename calib_t, typename Derived>
auto RTCProc::remove_nearby_tones(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                                  std::string map_grouping) {

    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();

    int n_nearby_tones = 0;

    // loop through flag columns
    for (Eigen::Index i=0; i<in.flags.data.cols(); i++) {
        // map from data column to apt row
        Eigen::Index det_index = det_indices(i);
        // if closer than freq separation limit and unflagged, flag it
        if (calib.apt["duplicate_tone"](det_index) && calib_scan.apt["flag"](det_index)==0) {
            n_nearby_tones++;
            // increment number of nearby tones
            if (map_grouping!="detector") {
                in.flags.data.col(i).setOnes();
            }
            else {
                calib_scan.apt["flag"](det_index) = 1;
            }
        }
    }

    SPDLOG_INFO("removed {}/{} ({}%) unflagged tones closer than {} kHz", n_nearby_tones, n_dets,
                (static_cast<float>(n_nearby_tones)/static_cast<float>(n_dets))*100, delta_f_min_Hz/1000);

    // set up scan calib
    calib_scan.setup();

    return std::move(calib_scan);
}

template <typename calib_t, typename Derived>
auto RTCProc::remove_bad_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                              Eigen::DenseBase<Derived> &nw_indices, Eigen::DenseBase<Derived> &array_indices, std::string redu_type,
                              std::string map_grouping) {


    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    // only run if limits are not zero
    if (lower_weight_factor !=0 || upper_weight_factor !=0) {
        // number of detectors
        Eigen::Index n_dets = in.scans.data.cols();

        std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grp_limits;

        Eigen::Index grp_i = calib.apt["array"](det_indices(0));
        grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{0, 0};
        Eigen::Index j = 0;
        // loop through apt table arrays, get highest index for current array
        for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
            auto det_index = det_indices(i);
            if (calib.apt["array"](det_index) == grp_i) {
                std::get<1>(grp_limits[grp_i]) = i + 1;
            }
            else {
                grp_i = calib.apt["array"](det_index);
                j += 1;
                grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{i, 0};
            }
        }

        in.n_low_dets = 0;
        in.n_high_dets = 0;

        for (auto const& [key, val] : grp_limits) {
            bool keep_going = true;
            Eigen::Index n_iter = 0;

            while (keep_going) {
                // number of unflagged detectors
                Eigen::Index n_good_dets = 0;

                for (Eigen::Index j=std::get<0>(grp_limits[key]); j<std::get<1>(grp_limits[key]); j++) {
                    if (calib.apt["flag"](det_indices(j))==0) {
                        n_good_dets++;
                    }
                }

                Eigen::VectorXd det_std_dev(n_good_dets);
                Eigen::VectorXI dets(n_good_dets);
                Eigen::Index k = 0;

                // collect standard deviation from good detectors
                for (Eigen::Index j=std::get<0>(grp_limits[key]); j<std::get<1>(grp_limits[key]); j++) {
                    Eigen::Index det_index = det_indices(j);
                    if (calib.apt["flag"](det_index)==0) {
                        // make Eigen::Maps for each detector's scan
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                            in.scans.data.col(j).data(), in.scans.data.rows());
                        Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                            in.flags.data.col(j).data(), in.flags.data.rows());

                        // calc standard deviation
                        det_std_dev(k) = engine_utils::calc_std_dev(scans, flags);

                        // convert to 1/variance
                        if (det_std_dev(k)!=0) {
                            det_std_dev(k) = std::pow(det_std_dev(k),-2);
                        }
                        else {
                            det_std_dev(k) = 0;
                        }

                        dets(k) = j;
                        k++;
                    }
                }

                // get median standard deviation
                double mean_std_dev = tula::alg::median(det_std_dev);

                int n_low_dets = 0;
                int n_high_dets = 0;

                // loop through good detectors and flag those that have std devs beyond the limits
                for (Eigen::Index j=0; j<n_good_dets; j++) {
                    Eigen::Index det_index = det_indices(dets(j));
                    // flag those below limit
                    if (calib.apt["flag"](det_index)==0) {
                        if ((det_std_dev(j) < (lower_weight_factor*mean_std_dev)) && lower_weight_factor!=0) {
                            if (map_grouping!="detector") {
                                in.flags.data.col(dets(j)).setOnes();
                            }
                            else {
                                calib_scan.apt["flag"](det_index) = 1;
                            }
                            in.n_low_dets++;
                            n_low_dets++;
                        }

                        // flag those above limit
                        if ((det_std_dev(j) > (upper_weight_factor*mean_std_dev)) && upper_weight_factor!=0) {
                            if (map_grouping!="detector") {
                                in.flags.data.col(dets(j)).setOnes();
                            }
                            else {
                                calib_scan.apt["flag"](det_index) = 1;
                            }
                            in.n_high_dets++;
                            n_high_dets++;
                        }
                    }
                }

                SPDLOG_INFO("array {} iter {}: {}/{} dets below limit. {}/{} dets above limit.", key, n_iter,
                            n_low_dets, n_good_dets, n_high_dets, n_good_dets);

                // increment iteration
                n_iter++;
                // check if no more detectors are above limit
                if ((n_low_dets==0 && n_high_dets==0) || n_iter > iter_lim) {
                    keep_going = false;
                }
            }
        }
    }

    // set up scan calib
    calib_scan.setup();

    return std::move(calib_scan);
}

} // namespace timestream
