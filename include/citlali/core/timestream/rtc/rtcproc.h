#pragma once

#include <citlali/core/timestream/timestream.h>

namespace timestream {


class RTCProc {
public:
    template <class Engine>
    void run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &,
             TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
             Engine);
};

template <class Engine>
void RTCProc::run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in,
         TCData<TCDataKind::PTC, Eigen::MatrixXd> &out,
         Engine engine) {

    // start index for removing scan edges due to lowpassing
    auto si = engine->filter.nterms;
    // scan length for inner scans
    auto sl = in.scan_indices.data(1) - in.scan_indices.data(0) + 1;

    // generate kernel
    if (engine->run_kernel) {
        SPDLOG_INFO("making kernel for scan {}",in.index.data);
        if (engine->kernel.kernel_type == "internal_gaussian") {
            engine->kernel.gaussian_kernel(engine, in);
        }
        else if (engine->kernel.kernel_type == "internal_airy") {
            engine->kernel.airy_kernel(engine, in);
        }
        else if (engine->kernel.kernel_type == "image") {
            engine->kernel.kernel_from_fits(engine, in);
        }
    }

    // despike scan
    if (engine->run_despike) {
        SPDLOG_INFO("despiking signal for scan {}", in.index.data);
        engine->despiker.despike(in.scans.data, in.flags.data);
        // replace flagged data
        SPDLOG_INFO("replacing flagged signal for scan {}", in.index.data);
        engine->despiker.replace_spikes(in.scans.data, in.flags.data, engine->responsivity);
    }

    // lowpass and highpass filter scan convolution
    if (engine->run_filter) {
        SPDLOG_INFO("filtering signal for scan {}", in.index.data);
        engine->filter.convolve_filter(in.scans.data);

        // run lowpass and highpass filter on kernel if requested
        if (engine->run_kernel) {
            SPDLOG_INFO("filtering kernel scan {}", in.index.data);
          engine->filter.convolve_filter(in.kernel_scans.data);
        }
    }

    // downsample scans, flags, telescope meta data, and kernel and move to PTCData
    if (engine->run_downsample) {

        // get the block of out scans that corresponds to the inner scan indices
        Eigen::Ref<Eigen::Map<Eigen::MatrixXd>> in_scans =
                in.scans.data.block(si, 0, sl, in.scans.data.cols());

        // get the block of in flags that corresponds to the inner scan indices
        Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags =
                in.flags.data.block(si, 0, sl, in.flags.data.cols());

        SPDLOG_INFO("downsampling scan {}", in.index.data);
        engine->downsampler.downsample(in_scans, out.scans.data);
        SPDLOG_INFO("downsampling flags for scan {}", in.index.data);
        engine->downsampler.downsample(in_flags, out.flags.data);

        // loop through telescope meta data and downsample
        SPDLOG_INFO("downsampling telescope meta for scan {}", in.index.data);
        for (auto const& x: in.tel_meta_data.data) {

            // get the block of in tel data that corresponds to the inner scan indices
            Eigen::Ref<Eigen::VectorXd> in_tel =
                    in.tel_meta_data.data[x.first].segment(si, sl);

            engine->downsampler.downsample(in_tel, out.tel_meta_data.data[x.first]);

        }

        // downsample kernel if requested
        if (engine->run_kernel) {

            // get the block of in kernel scans that corresponds to the inner scan indices
            Eigen::Ref<Eigen::MatrixXd> in_kernel_scans =
                    in.kernel_scans.data.block(si, 0, sl, in.kernel_scans.data.cols());

            SPDLOG_INFO("downsampling kernel scan {}", in.index.data);
            engine->downsampler.downsample(in_kernel_scans, out.kernel_scans.data);
        }

        // set the sample rate to the downsampled sample rate
        engine->fsmp = engine->fsmp/engine->downsampler.dsf;
    }

    // if downsampling skipped, just copy scans, flags, and kernel over
    else {
        SPDLOG_INFO("downsampling skipped, copying scan {}", in.index.data);
        out.scans.data = in.scans.data.block(si, 0, sl, in.scans.data.cols());;
        if (engine->run_kernel) {
            SPDLOG_INFO("downsampling skipped, copying kernel scan {}", in.index.data);
            out.kernel_scans.data = in.kernel_scans.data.block(si, 0, sl, in.kernel_scans.data.cols());;
        }

        // copy flags
        out.flags.data = in.flags.data.block(si, 0, sl, in.flags.data.cols());
        // copy telescope data
        for (auto const& x: in.tel_meta_data.data) {
            out.tel_meta_data.data[x.first] = in.tel_meta_data.data[x.first].segment(si,sl);
        }
    }

    // copy scan indices and current index to PTCData
    out.scan_indices.data = in.scan_indices.data;
    out.index.data = in.index.data;

    // flux calibration
    if ((engine->reduction_type == "science") || (engine->reduction_type == "pointing")) {
        SPDLOG_INFO("calibrating flux scale for scan {}", in.index.data);
        calibrate(out.scans.data, engine->calib_data["flxscale"]);
    }
}

} // namespace
