#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <tuple>

namespace citlali::pipeline {

struct RtcScanSampleWindow {
    Eigen::Index start = 0;
    Eigen::Index length = 0;
};

struct FruitLoopWeightPolicy {
    bool use_noise_weights = false;
    bool keep_source_subtracted_weights = false;
};

template <class Logger, class Telescope, class Counter>
void log_scan_start(
    const std::shared_ptr<std::mutex> &scans_done_mutex,
    const Logger &logger, Eigen::Index scan_index, Counter n_scans_done,
    const Telescope &telescope) {
    std::lock_guard<std::mutex> lock(*scans_done_mutex);
    logger->info("starting scan {}. {}/{} scans completed",
                 scan_index + 1, n_scans_done,
                 telescope.scan_indices.cols());
}

template <class Logger, class Telescope, class Counter>
void log_scan_done(
    const std::shared_ptr<std::mutex> &scans_done_mutex,
    const Logger &logger, Eigen::Index scan_index, Counter &n_scans_done,
    const Telescope &telescope) {
    std::lock_guard<std::mutex> lock(*scans_done_mutex);
    n_scans_done++;
    logger->info("done with scan {}. {}/{} scans completed",
                 scan_index + 1, n_scans_done,
                 telescope.scan_indices.cols());
}

template <class PtcProc>
FruitLoopWeightPolicy fruit_loop_weight_policy(const PtcProc &ptcproc) {
    FruitLoopWeightPolicy policy;
    policy.use_noise_weights =
        ptcproc.run_fruit_loops && !ptcproc.tod_mb.signal.empty();
    policy.keep_source_subtracted_weights =
        policy.use_noise_weights &&
        !ptcproc.fruit_loops_recompute_weights_after_addback;
    return policy;
}

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

}  // namespace citlali::pipeline
