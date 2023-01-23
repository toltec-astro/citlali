#pragma once

#include <tula/algorithm/ei_stats.h>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/timestream/rtc/polarization.h>
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

    // rtc tod classes
    timestream::Polarization polarization;
    timestream::Kernel kernel;
    timestream::Despiker despiker;
    timestream::Filter filter;
    timestream::Downsampler downsampler;
    timestream::Calibration calibration;

    double lower_std_dev, upper_std_dev;

    template<typename calib_t, typename telescope_t, typename pointing_offset_t, typename Derived>
    void run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &,
             TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string &,
             std::string &, calib_t &, telescope_t &, pointing_offset_t &,
             Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &,
             Eigen::DenseBase<Derived> &, double);

    template <typename calib_t, typename Derived>
    auto remove_bad_dets_nw(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                            Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string, std::string);

    template <typename calib_t, typename Derived>
    auto remove_nearby_tones(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                             Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string, std::string);
};

template<class calib_t, typename telescope_t, typename pointing_offset_t, typename Derived>
void RTCProc::run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in,
                  TCData<TCDataKind::PTC, Eigen::MatrixXd> &out, std::string &pixel_axes,
                  std::string &redu_type, calib_t &calib, telescope_t &telescope,
                  pointing_offset_t &pointing_offsets_arcsec, Eigen::DenseBase<Derived> &det_indices,
                  Eigen::DenseBase<Derived> &array_indices, Eigen::DenseBase<Derived> &map_indices,
                  double pixel_size_rad) {

    // set chunk as demodulated
    if (run_polarization) {
        out.demodulated = true;
    }

    // number of points in scan
    Eigen::Index n_pts = in.scans.data.rows();

    // start index of inner scans
    auto si = filter.n_terms;
    // end index of inner scans
    auto sl = in.scan_indices.data(1) - in.scan_indices.data(0) + 1;

    // set up flags
    in.flags.data.setOnes(in.scans.data.rows(), in.scans.data.cols());

    // create kernel if requested
    if (run_kernel) {
        SPDLOG_DEBUG("creating kernel timestream");
        if (kernel.type == "gaussian") {
            SPDLOG_DEBUG("creating symmetric gaussian kernel");
            kernel.create_symmetric_gaussian_kernel(in, pixel_axes, redu_type, calib.apt, pointing_offsets_arcsec,
                                                    det_indices);
        }
        else if (kernel.type == "airy") {
            SPDLOG_DEBUG("creating airy kernel");
            kernel.create_airy_kernel(in, pixel_axes, redu_type, calib.apt, pointing_offsets_arcsec,
                                      det_indices);
        }
        else if (kernel.type == "fits") {
            SPDLOG_DEBUG("getting kernel from fits");
            kernel.create_kernel_from_fits(in, pixel_axes, redu_type, calib.apt, pointing_offsets_arcsec,
                                           pixel_size_rad, map_indices, det_indices);
        }

        out.kernel_generated = true;
    }

    if (run_despike) {
        SPDLOG_DEBUG("despiking");
        despiker.despike(in.scans.data, in.flags.data, calib.apt);

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
            Eigen::Ref<Eigen::MatrixXd> in_scans_ref = in.scans.data.block(0, start_index, n_pts, n_dets);

            Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>
                in_scans(in_scans_ref.data(), in_scans_ref.rows(), in_scans_ref.cols(),
                         Eigen::OuterStride<>(in_scans_ref.outerStride()));

            // get the block of in flags that corresponds to the current array
            Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags_ref =
                in.flags.data.block(0, start_index, n_pts, n_dets);

            Eigen::Map<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>, 0, Eigen::OuterStride<> >
                in_flags(in_flags_ref.data(), in_flags_ref.rows(), in_flags_ref.cols(),
                         Eigen::OuterStride<>(in_flags_ref.outerStride()));

            SPDLOG_DEBUG("replacing spikes");
            despiker.replace_spikes(in_scans, in_flags, calib.apt, start_index);
        }

        out.despiked = true;
    }

    // timestream filtering
    if (run_tod_filter) {
        SPDLOG_DEBUG("convolving signal with tod filter");
        filter.convolve(in.scans.data);

        // filter kernel
        if (run_kernel) {
            SPDLOG_DEBUG("convolving kernel with tod filter");
            filter.convolve(in.kernel.data);
        }

        out.tod_filtered = true;
    }

    if (run_downsample) {
        SPDLOG_DEBUG("downsampling data");
        // get the block of out scans that corresponds to the inner scan indices
        Eigen::Ref<Eigen::Map<Eigen::MatrixXd>> in_scans =
            in.scans.data.block(si, 0, sl, in.scans.data.cols());

        // get the block of in flags that corresponds to the inner scan indices
        Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags =
            in.flags.data.block(si, 0, sl, in.flags.data.cols());

        downsampler.downsample(in_scans, out.scans.data);
        downsampler.downsample(in_flags, out.flags.data);

        // loop through telescope meta data and downsample
        SPDLOG_DEBUG("downsampling telescope");
        for (auto const& x: in.tel_data.data) {
            // get the block of in tel data that corresponds to the inner scan indices
            Eigen::Ref<Eigen::VectorXd> in_tel =
                in.tel_data.data[x.first].segment(si, sl);

            downsampler.downsample(in_tel, out.tel_data.data[x.first]);
        }

        // downsample kernel if requested
        if (run_kernel) {
            SPDLOG_DEBUG("downsampling kernel");
            // get the block of in kernel scans that corresponds to the inner scan indices
            Eigen::Ref<Eigen::MatrixXd> in_kernel =
                in.kernel.data.block(si, 0, sl, in.kernel.data.cols());

            downsampler.downsample(in_kernel, out.kernel.data);
        }

        out.downsampled =true;
    }

    else {
        // copy data
        out.scans.data = in.scans.data.block(si, 0, sl, in.scans.data.cols());
        // copy flags
        out.flags.data = in.flags.data.block(si, 0, sl, in.flags.data.cols());
        // copy kernel
        if (run_kernel) {
            out.kernel.data = in.kernel.data.block(si, 0, sl, in.kernel.data.cols());
        }
        // copy telescope data
        for (auto const& x: in.tel_data.data) {
            out.tel_data.data[x.first] = in.tel_data.data[x.first].segment(si, sl);
        }
    }

    out.scan_indices.data = in.scan_indices.data;
    out.index.data = in.index.data;

    if (run_calibrate) {
        SPDLOG_DEBUG("calibrating timestream");
        //calc tau at toltec frequencies
        auto tau_freq = calibration.calc_tau(out.tel_data.data["TelElAct"], telescope.tau_225_GHz);
        // calibrate tod
        calibration.calibrate_tod(out, det_indices, array_indices, calib, tau_freq);

        out.calibrated = true;
    }
}

template <typename calib_t, typename Derived>
auto RTCProc::remove_bad_dets_nw(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                                 Eigen::DenseBase<Derived> &nw_indices, Eigen::DenseBase<Derived> &array_indices, std::string redu_type,
                                 std::string map_grouping) {

    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();

    in.n_low_dets = 0;
    in.n_high_dets = 0;

    for (Eigen::Index i=0; i<calib.n_nws; i++) {
        // number of unflagged detectors
        Eigen::Index n_good_dets = 0;

        for (Eigen::Index j=0; j<n_dets; j++) {
            Eigen::Index det_index = det_indices(j);
            if (calib.apt["flag"](det_index) && calib.apt["nw"](det_index)==calib.nws(i)) {
                n_good_dets++;
            }
        }

        Eigen::VectorXd det_std_dev(n_good_dets);
        Eigen::VectorXI dets(n_good_dets);
        Eigen::Index k = 0;

        // collect standard deviation from good detectors
        for (Eigen::Index j=0; j<n_dets; j++) {
            Eigen::Index det_index = det_indices(j);
            if (calib.apt["flag"](det_index) && calib.apt["nw"](det_index)==calib.nws(i)) {

                     // make Eigen::Maps for each detector's scan
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                    in.scans.data.col(j).data(), in.scans.data.rows());
                Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                    in.flags.data.col(j).data(), in.flags.data.rows());

                det_std_dev(k) = engine_utils::calc_std_dev(scans, flags);

                if (det_std_dev(k) !=0) {
                    det_std_dev(k) = std::pow(det_std_dev(k),-2);
                }
                else {
                    det_std_dev(k) = 0;
                }
                //det_std_dev(k) = engine_utils::calc_rms(scans, flags);

                dets(k) = j;
                k++;
            }
        }

        // get mean standard deviation
        //double mean_std_dev = det_std_dev.mean();
        double mean_std_dev = tula::alg::median(det_std_dev);

        int n_low_dets = 0;
        int n_high_dets = 0;

        // loop through good detectors and flag those that have std devs beyond the limits
        for (Eigen::Index j=0; j<n_good_dets; j++) {
            Eigen::Index det_index = det_indices(dets(j));
            // flag those below limit
            if (calib.apt["flag"](det_index) && calib.apt["nw"](det_index)==calib.nws(i)) {
                if ((det_std_dev(j) < (lower_std_dev*mean_std_dev)) && lower_std_dev!=0) {
                    if (map_grouping!="detector") {
                        in.flags.data.col(dets(j)).setZero();
                    }
                    else {
                        calib_scan.apt["flag"](det_index) = 0;
                    }
                    in.n_low_dets++;
                    n_low_dets++;
                }

                // flag those above limit
                if ((det_std_dev(j) > (upper_std_dev*mean_std_dev)) && upper_std_dev!=0) {
                    if (map_grouping!="detector") {
                        in.flags.data.col(dets(j)).setZero();
                    }
                    else {
                        calib_scan.apt["flag"](det_index) = 0;
                    }
                    in.n_high_dets++;
                    n_high_dets++;
                }
            }
        }

        SPDLOG_INFO("nw{}: {}/{} dets below limit. {}/{} dets above limit.", calib.nws(i), n_low_dets, n_good_dets,
                    n_high_dets, n_good_dets);
    }

    calib_scan.setup();

    return std::move(calib_scan);
}

template <typename calib_t, typename Derived>
auto RTCProc::remove_nearby_tones(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                                 Eigen::DenseBase<Derived> &nw_indices, Eigen::DenseBase<Derived> &array_indices, std::string redu_type,
                                 std::string map_grouping) {

}

} // namespace timestream
