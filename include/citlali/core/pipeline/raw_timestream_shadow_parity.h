#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/raw_timestream_config_serialization.h>
#include <citlali/core/pipeline/timestream_config_adapter_raw.h>
#include <citlali/core/pipeline/timestream_config_mirror.h>

#include <string>
#include <type_traits>

namespace citlali::pipeline {

template <class RtcProc>
citlali::config::RawTimeChunkConfig snapshot_raw_rtc_config(
    const RtcProc &source, double radians_to_arcsec) {
    citlali::config::RawTimeChunkConfig snapshot;
    mirror_raw_despike_config(snapshot.despike, source);
    mirror_raw_flagging_config(snapshot.flagging, source);
    mirror_raw_kernel_config(snapshot.kernel, source, radians_to_arcsec);
    mirror_raw_altaz_destripe_config(snapshot.altaz_destripe, source);
    mirror_raw_line_audit_config(snapshot.line_audit, source.line_audit);
    mirror_raw_downsample_config(snapshot.downsample, source);
    mirror_raw_filter_config(snapshot.filter, source);
    mirror_raw_iir_filter_config(snapshot.iir_filter, source);
    mirror_raw_correction_flags(snapshot, source);
    mirror_raw_filter_edge_guard_config(
        snapshot.filter.edge_guard, source.filter_edge_guard);
    return snapshot;
}

struct RawTimestreamShadowParityReport {
    bool exact = false;
    std::string legacy_snapshot;
    std::string typed_adapter_snapshot;
};

template <class RtcProc>
RawTimestreamShadowParityReport compare_raw_timestream_shadow(
    const citlali::config::RawTimeChunkConfig &request,
    const RtcProc &legacy, double native_sample_rate_hz,
    double arcsec_to_rad, double radians_to_arcsec,
    double fwhm_to_std) {
    using ShadowRtcProc = std::remove_cv_t<std::remove_reference_t<RtcProc>>;
    ShadowRtcProc shadow;
    adapt_raw_timestream_config_one_way(
        request, shadow, arcsec_to_rad, fwhm_to_std);
    shadow.configure_filter_edge_guard(native_sample_rate_hz);

    RawTimestreamShadowParityReport report;
    report.legacy_snapshot = YAML::Dump(raw_timestream_request_node(
        snapshot_raw_rtc_config(legacy, radians_to_arcsec)));
    report.typed_adapter_snapshot = YAML::Dump(raw_timestream_request_node(
        snapshot_raw_rtc_config(shadow, radians_to_arcsec)));
    report.exact =
        report.legacy_snapshot == report.typed_adapter_snapshot;
    return report;
}

}  // namespace citlali::pipeline
