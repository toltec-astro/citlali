#pragma once

// Included by summary_log.h inside namespace citlali::pipeline.

template <class TimeChunk, class RawTimeChunkConfig, class RtcProc>
void write_chunk_summary_log(std::ostream &stream, const TimeChunk &chunk,
                             const std::string &citlali_version,
                             const std::string &kids_version,
                             const std::string &write_time,
                             citlali::config::ReductionType reduction_type,
                             citlali::config::TodType tod_type,
                             std::string_view signal_unit,
                             const RawTimeChunkConfig &raw_config,
                             const RtcProc &rtcproc,
                             int outer_context_samples,
                             long long n_apt_flagged,
                             double data_median,
                             double data_stddev) {
    const std::string reduction_type_name{
        citlali::config::to_string(reduction_type)};
    const std::string tod_type_name{citlali::config::to_string(tod_type)};

    stream << "Summary file for scan " << chunk.index.data << "\n";
    write_pipeline_version_summary(stream, citlali_version, kids_version);
    write_chunk_time_summary(stream, chunk.creation_time, write_time);
    write_chunk_identity_summary(
        stream, reduction_type_name, tod_type_name, signal_unit, chunk.name);
    write_chunk_processing_status_summary(stream, chunk.status);
    write_chunk_tod_filter_summary(
        stream, raw_config, rtcproc.filter_edge_guard,
        outer_context_samples);
    write_chunk_ptc_model_line_audit_summary(
        stream, raw_config.line_audit);
    write_chunk_scan_shape_summary(
        stream, chunk.scans.data.rows(), chunk.scans.data.cols());
    write_chunk_detector_flag_summary(
        stream, n_apt_flagged, chunk.n_dets_low, chunk.n_dets_high,
        chunk.scans.data.cols());
    write_chunk_nonfinite_summary(stream, chunk.scans.data);
    write_chunk_data_stat_summary(
        stream, chunk.scans.data.minCoeff(), chunk.scans.data.maxCoeff(),
        chunk.scans.data.mean(), data_median, data_stddev, signal_unit);
    write_chunk_kernel_summary_if_generated(
        stream, chunk.status.kernel_generated, chunk.kernel, signal_unit);
}
