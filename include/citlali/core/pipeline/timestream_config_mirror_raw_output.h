#pragma once

// Included by timestream_config_mirror.h inside namespace citlali::pipeline.

template <class Telescope, class RtcProc>
void configure_raw_tod_output_context(Telescope &telescope,
                                      const RtcProc &rtcproc) {
    telescope.inner_scans_chunk = rtcproc.filter_edge_guard.context_samples;
    telescope.outer_scans_chunk = telescope.inner_scans_chunk;
    if (rtcproc.tod_output_outer) {
        telescope.outer_scans_chunk = std::max<Eigen::Index>(
            telescope.outer_scans_chunk,
            std::max<Eigen::Index>(
                0, rtcproc.tod_output_outer_context_samples));
    }
    if (rtcproc.line_audit.enabled && rtcproc.line_audit.post_filter_enabled &&
        rtcproc.line_audit.post_filter_apply_detector_notches) {
        telescope.outer_scans_chunk = std::max<Eigen::Index>(
            telescope.outer_scans_chunk,
            std::max<Eigen::Index>(
                0, rtcproc.line_audit.detector_notch_context_samples));
    }
}

