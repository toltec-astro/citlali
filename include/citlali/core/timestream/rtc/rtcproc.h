#pragma once

namespace timestream {

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

    template<typename calib_t, typename telescope_t, typename pointing_offset_t, typename Derived>
    void run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &,
             TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string &,
             std::string &, calib_t &, telescope_t &, pointing_offset_t &,
             Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &,
             Eigen::DenseBase<Derived> &, double);
};

template<class calib_t, typename telescope_t, typename pointing_offset_t, typename Derived>
void RTCProc::run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in,
                  TCData<TCDataKind::PTC, Eigen::MatrixXd> &out, std::string &pixel_axes,
                  std::string &redu_type, calib_t &calib, telescope_t &telescope,
                  pointing_offset_t &pointing_offsets_arcsec, Eigen::DenseBase<Derived> &det_indices,
                  Eigen::DenseBase<Derived> &array_indices, Eigen::DenseBase<Derived> &map_indices,
                  double pixel_size_rad) {

    if (run_polarization) {
        out.demodulated = true;
    }

    Eigen::Index n_pts = in.scans.data.rows();

    auto si = filter.n_terms;
    auto sl = in.scan_indices.data(1) - in.scan_indices.data(0) + 1;

    in.flags.data.setOnes(in.scans.data.rows(), in.scans.data.cols());

    if (run_kernel) {
        if (kernel.type == "gaussian") {
            kernel.create_symmetric_gaussian_kernel(in, pixel_axes, redu_type, calib.apt, pointing_offsets_arcsec,
                                                    det_indices);
        }
        else if (kernel.type == "airy") {
            kernel.create_airy_kernel(in, pixel_axes, redu_type, calib.apt, pointing_offsets_arcsec,
                                      det_indices);
        }
        else if (kernel.type == "fits") {
            kernel.create_kernel_from_fits(in, pixel_axes, redu_type, calib.apt, pointing_offsets_arcsec,
                                           pixel_size_rad, map_indices, det_indices);
        }

        out.kernel_generated = true;
    }

    if (run_despike) {
        SPDLOG_INFO("despiking");
        despiker.despike(in.scans.data, in.flags.data);

        std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grouping_limits;

        if (despiker.grouping == "nw") {
            grouping_limits = calib.nw_limits;
        }

        else if (despiker.grouping == "array") {
            grouping_limits = calib.array_limits;
            SPDLOG_INFO("array_limits {}", calib.array_limits);
        }

        for (auto const& [key, val] : grouping_limits) {
            // starting index
            auto start_index = std::get<0>(val);
            // size of block for each grouping
            auto n_dets = std::get<1>(val) - std::get<0>(val);

            SPDLOG_INFO("start_index {} ndets {}", start_index, n_dets);

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

            despiker.replace_spikes(in_scans, in_flags, calib.apt["responsivity"]);
        }

        out.despiked = true;
    }

    if (run_tod_filter) {
        SPDLOG_INFO("convolving with tod filter");
        filter.convolve(in.scans.data);

        if (run_kernel) {
            filter.convolve(in.kernel.data);
        }

        out.tod_filtered = true;
    }

    if (run_downsample) {
        // get the block of out scans that corresponds to the inner scan indices
        Eigen::Ref<Eigen::Map<Eigen::MatrixXd>> in_scans =
            in.scans.data.block(si, 0, sl, in.scans.data.cols());

        // get the block of in flags that corresponds to the inner scan indices
        Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags =
            in.flags.data.block(si, 0, sl, in.flags.data.cols());

        downsampler.downsample(in_scans, out.scans.data);
        downsampler.downsample(in_flags, out.flags.data);

        // loop through telescope meta data and downsample
        for (auto const& x: in.tel_data.data) {
            // get the block of in tel data that corresponds to the inner scan indices
            Eigen::Ref<Eigen::VectorXd> in_tel =
                in.tel_data.data[x.first].segment(si, sl);

            downsampler.downsample(in_tel, out.tel_data.data[x.first]);
        }

        // downsample kernel if requested
        if (run_kernel) {
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
        SPDLOG_INFO("calibrating timestream");
        //calc tau at toltec frequencies
        auto tau_freq = calibration.calc_tau(in.tel_data.data["TelElDes"], telescope.tau_225_GHz);
        // calibrate tod
        calibration.calibrate_tod(out, det_indices, array_indices, calib, tau_freq);

        out.calibrated = true;
    }
}

} // namespace timestream
