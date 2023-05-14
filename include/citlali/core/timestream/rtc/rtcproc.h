#pragma once

#include <tula/algorithm/ei_stats.h>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/timestream/rtc/polarization_2.h>
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

    template <class calib_t, typename Derived>
    auto calc_map_indices(calib_t &, Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &,
                          Eigen::DenseBase<Derived> &, std::string, std::string);


    template<typename calib_t, typename telescope_t>
    auto run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &,
             TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string &,
             std::string &, calib_t &, telescope_t &, double, std::string, std::string);

    template <typename calib_t, typename Derived>
    auto remove_nearby_tones(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                             Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string, std::string);
};

template <class calib_t, typename Derived>
auto RTCProc::calc_map_indices(calib_t &calib, Eigen::DenseBase<Derived> &det_indices, Eigen::DenseBase<Derived> &nw_indices,
                               Eigen::DenseBase<Derived> &array_indices, std::string stokes_param, std::string map_grouping) {
    // indices for maps
    Eigen::VectorXI indices(array_indices.size()), map_indices(array_indices.size());

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

    // start at 0
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

    if (run_polarization) {
        if (stokes_param == "Q") {
            map_indices = map_indices.array() + (3*n_maps)/3;
        }
        else if (stokes_param == "U") {
            map_indices = map_indices.array() + 2*(3*n_maps)/3;
        }
    }

    return std::move(map_indices);
}

template<class calib_t, typename telescope_t>
auto RTCProc::run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in,
                  TCData<TCDataKind::PTC, Eigen::MatrixXd> &out, std::string &pixel_axes,
                  std::string &redu_type, calib_t &calib, telescope_t &telescope, double pixel_size_rad,
                  std::string stokes_param, std::string map_grouping) {

    TCData<TCDataKind::RTC,Eigen::MatrixXd> in_pol;

    // demodulate
    auto [array_indices, nw_indices, det_indices] = polarization.demodulate_timestream(in, in_pol,
                                                                                       stokes_param,
                                                                                       redu_type, calib,
                                                                                       telescope.sim_obs);

    // resize fcf
    in_pol.fcf.data.setOnes(in_pol.scans.data.cols());

    // get indices for maps
    SPDLOG_INFO("calculating map indices");
    auto map_indices = calc_map_indices(calib, det_indices, nw_indices, array_indices, stokes_param, map_grouping);

    SPDLOG_DEBUG("array indices {}, nw indices {}, det indices {} map_indices {}",array_indices,nw_indices,det_indices,
                 map_indices);

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
        if (kernel.type == "gaussian") {
            SPDLOG_DEBUG("creating symmetric gaussian kernel");
            kernel.create_gaussian_kernel(in_pol, pixel_axes, redu_type, calib.apt, in_pol.pointing_offsets_arcsec.data,
                                                    det_indices);
        }
        else if (kernel.type == "airy") {
            SPDLOG_DEBUG("creating airy kernel");
            kernel.create_airy_kernel(in_pol, pixel_axes, redu_type, calib.apt, in_pol.pointing_offsets_arcsec.data,
                                      det_indices);
        }
        else if (kernel.type == "fits") {
            SPDLOG_DEBUG("getting kernel from fits");
            kernel.create_kernel_from_fits(in_pol, pixel_axes, redu_type, calib.apt, in_pol.pointing_offsets_arcsec.data,
                                           pixel_size_rad, map_indices, det_indices);
        }

        out.kernel_generated = true;
    }

    if (run_despike) {
        SPDLOG_DEBUG("despiking");
        despiker.despike(in_pol.scans.data, in_pol.flags.data, calib.apt);

        std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grouping_limits;

        // nw grouping for flag replacement
        if (despiker.grouping == "nw") {
            grouping_limits = calib.nw_limits;
        }
        // array grouping for flag replacement
        else if (despiker.grouping == "array") {
            grouping_limits = calib.array_limits;
        }

        for (auto const& [key, val] : grouping_limits) {
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

            SPDLOG_DEBUG("replacing spikes");
            //despiker.replace_spikes(in_scans, in_flags, calib.apt, start_index);
        }

        out.despiked = true;
    }

    // timestream filtering
    if (run_tod_filter) {
        SPDLOG_DEBUG("convolving signal with tod filter");
        filter.convolve(in_pol.scans.data);

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

        for (auto const& x: in_pol.pointing_offsets_arcsec.data) {
        Eigen::Ref<Eigen::VectorXd> in_pointing =
            in_pol.pointing_offsets_arcsec.data[x.first].segment(si, sl);

            downsampler.downsample(in_pointing, out.pointing_offsets_arcsec.data[x.first]);
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
    }

    SPDLOG_INFO("flags {}",out.flags.data);

    out.scan_indices.data = in_pol.scan_indices.data;
    out.index.data = in_pol.index.data;
    out.fcf.data = in_pol.fcf.data;

    return std::tuple<Eigen::VectorXI,Eigen::VectorXI,Eigen::VectorXI,Eigen::VectorXI>(map_indices, array_indices, nw_indices, det_indices);
}

template <typename calib_t, typename Derived>
auto RTCProc::remove_nearby_tones(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                                 Eigen::DenseBase<Derived> &nw_indices, Eigen::DenseBase<Derived> &array_indices, std::string redu_type,
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
        if (calib.apt["duplicate_tone"](det_index) && calib_scan.apt["flag"](det_index)!=1) {
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
} // namespace timestream
