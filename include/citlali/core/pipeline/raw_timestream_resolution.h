#pragma once

#include <citlali/core/config/timestream_config.h>

namespace citlali::pipeline {

enum class RawDownsampleRequestKind {
    disabled,
    explicit_factor,
    target_frequency,
};

struct RawFilterActivationResolution {
    bool fir_requested = false;
    bool fixed_notch_requested = false;
    bool fixed_notch_effective = false;
    bool iir_highpass_requested = false;
    bool edge_guard_requested = false;
    bool downsample_requested = false;
    bool downsample_filter_dependency_satisfied = true;
};

struct RawDownsampleRequestResolution {
    RawDownsampleRequestKind kind = RawDownsampleRequestKind::disabled;
    int requested_factor = 1;
    double requested_frequency_hz = 0.0;
};

struct RawSourceProtectionIntentResolution {
    bool despike_requested = false;
    bool source_protection_requested = false;
};

struct RawCorrectionIntentResolution {
    bool flux_calibration_requested = false;
    bool extinction_correction_requested = false;
};

struct RawTimestreamEffectiveResolutions {
    RawFilterActivationResolution filtering;
    RawDownsampleRequestResolution downsampling;
    RawSourceProtectionIntentResolution source_protection;
    RawCorrectionIntentResolution corrections;
};

inline RawFilterActivationResolution resolve_raw_filter_activation(
    const citlali::config::RawTimeChunkConfig &request) {
    return RawFilterActivationResolution{
        request.filter.enabled,
        request.filter.notch.enabled,
        request.filter.enabled && request.filter.notch.enabled,
        request.iir_filter.enabled,
        request.filter.edge_guard.enabled,
        request.downsample.enabled,
        !request.downsample.enabled || request.filter.enabled,
    };
}

inline RawDownsampleRequestResolution resolve_raw_downsample_request(
    const citlali::config::RawTimeChunkDownsampleConfig &request) {
    RawDownsampleRequestResolution resolution;
    resolution.requested_factor = request.factor;
    resolution.requested_frequency_hz = request.downsampled_freq_Hz;
    if (!request.enabled) {
        return resolution;
    }
    resolution.kind = request.factor > 0
                          ? RawDownsampleRequestKind::explicit_factor
                          : RawDownsampleRequestKind::target_frequency;
    return resolution;
}

inline RawSourceProtectionIntentResolution
resolve_raw_source_protection_intent(
    const citlali::config::RawTimeChunkDespikeConfig &request) {
    return RawSourceProtectionIntentResolution{
        request.enabled,
        request.enabled && request.source_protection.enabled,
    };
}

inline RawCorrectionIntentResolution resolve_raw_correction_intent(
    const citlali::config::RawTimeChunkConfig &request) {
    return RawCorrectionIntentResolution{
        request.flux_calibration_enabled,
        request.extinction_correction_enabled,
    };
}

inline RawTimestreamEffectiveResolutions
resolve_raw_timestream_effective_request(
    const citlali::config::RawTimeChunkConfig &request) {
    return RawTimestreamEffectiveResolutions{
        resolve_raw_filter_activation(request),
        resolve_raw_downsample_request(request.downsample),
        resolve_raw_source_protection_intent(request.despike),
        resolve_raw_correction_intent(request),
    };
}

}  // namespace citlali::pipeline
