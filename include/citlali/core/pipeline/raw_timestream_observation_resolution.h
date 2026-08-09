#pragma once

#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/raw_timestream_execution_plan.h>
#include <citlali/core/timestream/extinction_model_selection.h>
#include <citlali/core/timestream/filter_transient_samples.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>

namespace citlali::pipeline {

enum class RawSampleRateResolutionError {
    none,
    invalid_native_sample_rate,
    invalid_target_frequency,
    target_frequency_above_native,
    invalid_downsample_factor,
    antialias_filter_above_nyquist,
};

inline std::string_view to_string(RawSampleRateResolutionError error) {
    switch (error) {
        case RawSampleRateResolutionError::none:
            return "none";
        case RawSampleRateResolutionError::invalid_native_sample_rate:
            return "invalid_native_sample_rate";
        case RawSampleRateResolutionError::invalid_target_frequency:
            return "invalid_target_frequency";
        case RawSampleRateResolutionError::target_frequency_above_native:
            return "target_frequency_above_native";
        case RawSampleRateResolutionError::invalid_downsample_factor:
            return "invalid_downsample_factor";
        case RawSampleRateResolutionError::antialias_filter_above_nyquist:
            return "antialias_filter_above_nyquist";
    }
    return "unknown";
}

struct RawSampleRateObservationResolution {
    RawSampleRateResolutionError error =
        RawSampleRateResolutionError::none;
    double native_sample_rate_hz = 0.0;
    double effective_sample_rate_hz = 0.0;
    int downsample_factor = 1;
    double downsample_nyquist_hz = 0.0;

    bool valid() const {
        return error == RawSampleRateResolutionError::none;
    }
};

inline RawSampleRateObservationResolution resolve_raw_sample_rate(
    const citlali::config::RawTimeChunkConfig &request,
    double native_sample_rate_hz) {
    RawSampleRateObservationResolution resolution;
    resolution.native_sample_rate_hz = native_sample_rate_hz;
    if (!std::isfinite(native_sample_rate_hz) ||
        native_sample_rate_hz <= 0.0) {
        resolution.error =
            RawSampleRateResolutionError::invalid_native_sample_rate;
        return resolution;
    }

    if (!request.downsample.enabled) {
        resolution.effective_sample_rate_hz = native_sample_rate_hz;
        resolution.downsample_nyquist_hz = native_sample_rate_hz / 2.0;
        return resolution;
    }

    int factor = request.downsample.factor;
    if (factor <= 0) {
        const double target_hz = request.downsample.downsampled_freq_Hz;
        if (!std::isfinite(target_hz) || target_hz <= 0.0) {
            resolution.error =
                RawSampleRateResolutionError::invalid_target_frequency;
            return resolution;
        }
        if (target_hz > native_sample_rate_hz) {
            resolution.error =
                RawSampleRateResolutionError::target_frequency_above_native;
            return resolution;
        }
        const double derived_factor =
            std::floor(native_sample_rate_hz / target_hz);
        if (!std::isfinite(derived_factor) ||
            derived_factor > std::numeric_limits<int>::max()) {
            resolution.error =
                RawSampleRateResolutionError::invalid_downsample_factor;
            return resolution;
        }
        factor = static_cast<int>(derived_factor);
    }
    if (factor <= 0) {
        resolution.error =
            RawSampleRateResolutionError::invalid_downsample_factor;
        return resolution;
    }

    resolution.downsample_factor = factor;
    resolution.effective_sample_rate_hz =
        native_sample_rate_hz / static_cast<double>(factor);
    resolution.downsample_nyquist_hz =
        resolution.effective_sample_rate_hz / 2.0;
    if (request.filter.freq_high_Hz > resolution.downsample_nyquist_hz) {
        resolution.error =
            RawSampleRateResolutionError::antialias_filter_above_nyquist;
    }
    return resolution;
}

struct RawFilterEdgeGuardObservationResolution {
    int fir_samples = 0;
    int fixed_notch_samples = 0;
    int line_audit_fixed_notch_samples = 0;
    int line_audit_dynamic_notch_samples = 0;
    int iir_highpass_samples = 0;
    int downsample_samples = 0;
    int guard_samples = 0;
    int context_samples = 0;
};

inline int raw_transient_sample_count(std::ptrdiff_t samples) {
    return static_cast<int>(std::clamp<std::ptrdiff_t>(
        samples, 0, std::numeric_limits<int>::max()));
}

inline int add_raw_transient_samples(int left, int right) {
    return raw_transient_sample_count(
        static_cast<std::ptrdiff_t>(left) +
        static_cast<std::ptrdiff_t>(right));
}

inline int multiply_raw_transient_samples(int count, int samples) {
    return raw_transient_sample_count(
        static_cast<std::ptrdiff_t>(count) *
        static_cast<std::ptrdiff_t>(samples));
}

inline double raw_line_audit_fixed_notch_width_hz(
    const citlali::config::RawTimeChunkLineAuditConfig &audit,
    std::size_t index) {
    if (audit.fixed_notch_widths_hz.empty()) {
        return 0.25;
    }
    if (index < audit.fixed_notch_widths_hz.size()) {
        return audit.fixed_notch_widths_hz[index];
    }
    return audit.fixed_notch_widths_hz.back();
}

inline RawFilterEdgeGuardObservationResolution resolve_raw_filter_edge_guard(
    const citlali::config::RawTimeChunkConfig &request,
    const RawSampleRateObservationResolution &sample_rate) {
    RawFilterEdgeGuardObservationResolution resolution;
    if (!sample_rate.valid()) {
        return resolution;
    }

    const auto &guard = request.filter.edge_guard;
    const double fs_hz = sample_rate.native_sample_rate_hz;
    resolution.fir_samples = request.filter.enabled
                                 ? std::max(0, request.filter.n_terms)
                                 : 0;
    resolution.iir_highpass_samples = request.iir_filter.enabled
        ? raw_transient_sample_count(
              timestream::transient::iir_highpass_settle_samples(
                  fs_hz, request.iir_filter.freq_Hz,
                  request.iir_filter.order))
        : 0;
    resolution.context_samples = add_raw_transient_samples(
        resolution.fir_samples, resolution.iir_highpass_samples);

    if (!guard.enabled ||
        citlali::config::is_none_raw_filter_edge_guard_mode(guard.mode)) {
        return resolution;
    }

    auto combine = [&guard](int current, int next) {
        if (next <= 0) {
            return current;
        }
        return citlali::config::is_max_raw_filter_edge_guard_combine(
                   guard.combine)
                   ? std::max(current, next)
                   : add_raw_transient_samples(current, next);
    };

    if (request.filter.enabled && guard.apply_fir) {
        resolution.guard_samples =
            combine(resolution.guard_samples, resolution.fir_samples);
    }
    if (request.filter.enabled && request.filter.notch.enabled &&
        guard.apply_notch) {
        const auto &notch = request.filter.notch;
        for (std::size_t index = 0; index < notch.freqs_Hz.size(); ++index) {
            if (notch.delta_f_Hz.empty()) {
                break;
            }
            const double width_hz = notch.delta_f_Hz.size() == 1
                                        ? notch.delta_f_Hz.front()
                                        : notch.delta_f_Hz[index];
            resolution.fixed_notch_samples = add_raw_transient_samples(
                resolution.fixed_notch_samples,
                raw_transient_sample_count(
                    timestream::transient::notch_settle_samples_for_width(
                        fs_hz, width_hz,
                        guard.iir_settle_attenuation)));
        }
        resolution.guard_samples = combine(
            resolution.guard_samples, resolution.fixed_notch_samples);
    }

    const auto &audit = request.line_audit;
    if (audit.enabled && audit.pre_filter_enabled &&
        audit.fixed_notch_enabled && guard.apply_dynamic_notch) {
        int valid_notches = 0;
        double minimum_width_hz = std::numeric_limits<double>::infinity();
        const double nyquist_hz = fs_hz / 2.0;
        for (std::size_t index = 0;
             index < audit.fixed_notch_freqs_hz.size(); ++index) {
            const double frequency_hz = audit.fixed_notch_freqs_hz[index];
            const double width_hz =
                raw_line_audit_fixed_notch_width_hz(audit, index);
            if (!std::isfinite(frequency_hz) || frequency_hz <= 0.0 ||
                frequency_hz >= nyquist_hz || !std::isfinite(width_hz) ||
                width_hz <= 0.0) {
                continue;
            }
            ++valid_notches;
            minimum_width_hz = std::min(minimum_width_hz, width_hz);
        }
        if (valid_notches > 0) {
            resolution.line_audit_fixed_notch_samples =
                multiply_raw_transient_samples(
                    valid_notches,
                    raw_transient_sample_count(
                        timestream::transient::
                            notch_settle_samples_for_width(
                                fs_hz, minimum_width_hz,
                                guard.iir_settle_attenuation)));
            resolution.guard_samples = combine(
                resolution.guard_samples,
                resolution.line_audit_fixed_notch_samples);
        }
    }
    if (audit.enabled && audit.apply_shared_notches &&
        guard.apply_dynamic_notch) {
        const int sections =
            audit.apply_max_notches > 0 ? audit.apply_max_notches : 1;
        resolution.line_audit_dynamic_notch_samples =
            multiply_raw_transient_samples(
                sections,
                raw_transient_sample_count(
                    timestream::transient::notch_settle_samples_for_width(
                        fs_hz, audit.apply_min_width_hz,
                        guard.iir_settle_attenuation)));
        resolution.guard_samples = combine(
            resolution.guard_samples,
            resolution.line_audit_dynamic_notch_samples);
    }
    if (request.iir_filter.enabled && guard.apply_iir_highpass) {
        resolution.guard_samples = combine(
            resolution.guard_samples, resolution.iir_highpass_samples);
    }
    if (request.downsample.enabled && guard.apply_downsample &&
        sample_rate.downsample_factor > 1) {
        resolution.downsample_samples = sample_rate.downsample_factor - 1;
        resolution.guard_samples = combine(
            resolution.guard_samples, resolution.downsample_samples);
    }

    resolution.guard_samples =
        std::max(resolution.guard_samples, guard.min_samples);
    resolution.guard_samples = add_raw_transient_samples(
        resolution.guard_samples, guard.extra_samples);
    if (guard.max_samples > 0) {
        resolution.guard_samples =
            std::min(resolution.guard_samples, guard.max_samples);
    }
    resolution.guard_samples = std::max(0, resolution.guard_samples);
    resolution.context_samples = std::max(
        resolution.context_samples, resolution.guard_samples);
    return resolution;
}

struct RawSourceProtectionObservationResolution {
    bool requested = false;
    bool source_aware_reduction = false;
    bool active = false;
};

inline RawSourceProtectionObservationResolution
resolve_raw_source_protection_observation(
    citlali::config::ReductionType reduction_type,
    const citlali::config::RawTimeChunkDespikeConfig &request) {
    const bool requested =
        request.enabled && request.source_protection.enabled;
    const bool source_aware =
        citlali::config::is_pointing_reduction_type(reduction_type);
    return {requested, source_aware, requested && source_aware};
}

struct RawExtinctionObservationResolution {
    bool requested = false;
    bool active = false;
    std::string model{"N/A"};
};

template <class TransmissionMap>
RawExtinctionObservationResolution resolve_raw_extinction_observation(
    bool requested, double tau_225_ghz,
    const TransmissionMap &transmission_zenith) {
    if (!requested) {
        return {};
    }
    return RawExtinctionObservationResolution{
        true, true,
        timestream::select_extinction_model(
            tau_225_ghz, transmission_zenith)};
}

inline RawTimestreamObservationState make_raw_timestream_observation_state(
    const RawSampleRateObservationResolution &sample_rate,
    const RawFilterEdgeGuardObservationResolution &edge_guard,
    const RawSourceProtectionObservationResolution &source_protection,
    const RawExtinctionObservationResolution &extinction) {
    RawTimestreamObservationState state;
    state.rtc_contract = RawRtcContractState{};
    if (sample_rate.valid()) {
        state.native_sample_rate_hz = sample_rate.native_sample_rate_hz;
        state.effective_sample_rate_hz =
            sample_rate.effective_sample_rate_hz;
        state.downsample_factor = sample_rate.downsample_factor;
        state.filter_edge_guard_samples = edge_guard.guard_samples;
        state.filter_outer_context_samples = edge_guard.context_samples;
    }
    state.source_protection_active = source_protection.active;
    state.extinction_active = extinction.active;
    state.extinction_model = extinction.model;
    return state;
}

}  // namespace citlali::pipeline
