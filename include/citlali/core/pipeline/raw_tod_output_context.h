#pragma once

#include <citlali/core/config/timestream_config.h>

#include <Eigen/Core>

#include <algorithm>

namespace citlali::pipeline {

template <class Telescope, class RtcProc, class RawOutputConfig>
void configure_raw_tod_output_context(Telescope &telescope,
                                      const RtcProc &rtcproc,
                                      const RawOutputConfig &output_config) {
    telescope.inner_scans_chunk = rtcproc.filter_edge_guard.context_samples;
    telescope.outer_scans_chunk = telescope.inner_scans_chunk;
    if (citlali::config::is_outer_tod_stream_output_mode(
            output_config.mode)) {
        telescope.outer_scans_chunk = std::max<Eigen::Index>(
            telescope.outer_scans_chunk,
            std::max<Eigen::Index>(
                0, output_config.outer_context_samples));
    }
    if (rtcproc.line_audit.enabled &&
        rtcproc.line_audit.post_filter_enabled &&
        rtcproc.line_audit.post_filter_apply_detector_notches) {
        telescope.outer_scans_chunk = std::max<Eigen::Index>(
            telescope.outer_scans_chunk,
            std::max<Eigen::Index>(
                0, rtcproc.line_audit.detector_notch_context_samples));
    }
}

}  // namespace citlali::pipeline
