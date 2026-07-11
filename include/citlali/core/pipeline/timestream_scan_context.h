#pragma once

#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <Eigen/Core>

#include <algorithm>
#include <cstdlib>
#include <tuple>

namespace citlali::pipeline {

struct RtcScanSampleWindow {
    Eigen::Index start = 0;
    Eigen::Index length = 0;
};

template <class RtcData, class Telescope, class PointingOffsets>
RtcScanSampleWindow copy_rtc_scan_context(
    RtcData &rtcdata, const Telescope &telescope,
    const PointingOffsets &pointing_offsets_arcsec) {
    const Eigen::Index start = rtcdata.scan_indices.data(2);
    const Eigen::Index length =
        rtcdata.scan_indices.data(3) - rtcdata.scan_indices.data(2) + 1;

    for (const auto &[key, values] : telescope.tel_data) {
        rtcdata.tel_data.data[key] = values.segment(start, length);
    }

    for (const auto &[axis, offset] : pointing_offsets_arcsec) {
        rtcdata.pointing_offsets_arcsec.data[axis] =
            offset.segment(start, length);
    }

    return {start, length};
}

template <class RtcData, class Calib>
void copy_hwpr_angle_if_enabled(
    RtcData &rtcdata, const Calib &calib, bool run_polarization,
    bool run_hwpr, Eigen::Index hwpr_start_index, Eigen::Index scan_start,
    Eigen::Index scan_length) {
    if (run_polarization && run_hwpr) {
        rtcdata.hwpr_angle.data =
            calib.hwpr_angle.segment(scan_start + hwpr_start_index,
                                     scan_length);
    }
}

template <class RtcData>
void initialize_rtc_flags(RtcData &rtcdata) {
    rtcdata.flags.data.resize(rtcdata.scans.data.rows(),
                              rtcdata.scans.data.cols());
    rtcdata.flags.data.setConstant(0);
}

template <class RtcData, class Calib, class NetworkMasks, class ContextSamples,
          class Logger>
void apply_gap_masks_to_rtc_flags(
    RtcData &rtcdata, const Calib &calib, const NetworkMasks &nw_masks,
    Eigen::Index scan_start, ContextSamples context_samples,
    const Logger &logger) {
    for (const auto &[network_id, limits] : calib.nw_limits) {
        auto mask_it = nw_masks.find(network_id);
        if (mask_it == nw_masks.end()) {
            logger->error(
                "missing gap mask for nw {}; cannot apply gap flagging",
                network_id);
            std::exit(EXIT_FAILURE);
        }
        const auto &mask = mask_it->second;

        const Eigen::Index start = std::get<0>(limits);
        const Eigen::Index end = std::get<1>(limits) - 1;

        for (Eigen::Index row = 0; row < rtcdata.flags.data.rows(); ++row) {
            Eigen::Index start_index = row;
            Eigen::Index size = 1;
            if (context_samples > 0) {
                const Eigen::Index context =
                    static_cast<Eigen::Index>(context_samples);
                start_index = std::max<Eigen::Index>(0, row - context);
                const Eigen::Index end_index = std::min<Eigen::Index>(
                    row + context, rtcdata.flags.data.rows() - 1);
                size = end_index - start_index + 1;
            }
            if (mask(row + scan_start) == 0) {
                rtcdata.flags.data
                    .block(start_index, start, size, end - start + 1)
                    .setOnes();
            }
        }
        logger->debug("{}/{} gaps flagged",
                      rtcdata.flags.data.col(start).template cast<int>().sum(),
                      rtcdata.flags.data.rows());
    }
}

template <class Engine, class RtcData>
RtcScanSampleWindow prepare_standard_rtc_scan_context(
    Engine &engine, RtcData &rtcdata) {
    const auto scan_window = copy_rtc_scan_context(
        rtcdata, engine.telescope, engine.pointing_offsets.arcsec);
    copy_hwpr_angle_if_enabled(
        rtcdata, engine.calib, engine.rtcproc.run_polarization,
        engine.calib.run_hwpr, engine.alignment.hwpr_start_index,
        scan_window.start, scan_window.length);
    initialize_rtc_flags(rtcdata);
    if (citlali::config::timing_gap_interpolation_active(
            effective_runtime_values(engine))) {
        apply_gap_masks_to_rtc_flags(
            rtcdata, engine.calib, engine.alignment.network_masks,
            scan_window.start, engine.rtcproc.filter_edge_guard.context_samples,
            engine.logger);
    }
    return scan_window;
}

}  // namespace citlali::pipeline
