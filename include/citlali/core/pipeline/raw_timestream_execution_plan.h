#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/raw_timestream_resolution.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

struct RawTimestreamObservationState {
    std::optional<double> native_sample_rate_hz;
    std::optional<double> effective_sample_rate_hz;
    std::optional<int> downsample_factor;
    std::optional<int> filter_edge_guard_samples;
    std::optional<int> filter_outer_context_samples;
    bool filter_edge_guard_parity_deferred = false;
    std::optional<bool> source_protection_active;
    std::optional<bool> extinction_active;
    std::optional<std::string> extinction_model;
};

struct RawTimestreamRealizedState {
    std::size_t completed_scan_count = 0;
    std::size_t flagged_sample_count = 0;
    std::size_t dynamic_notch_count = 0;
    std::size_t required_output_count = 0;
};

struct RawTimestreamExecutionPlan {
    bool initialized = false;
    citlali::config::RawTimeChunkConfig requested;
    citlali::config::RawTimeChunkConfig effective;
    RawTimestreamEffectiveResolutions effective_resolutions;
    std::optional<RawTimestreamObservationState> observation;
    RawTimestreamRealizedState realized;

    void reset_from_request(
        const citlali::config::RawTimeChunkConfig &request) {
        initialized = true;
        requested = request;
        effective = request;
        effective_resolutions =
            resolve_raw_timestream_effective_request(request);
        observation.reset();
        realized = {};
    }

    RawTimestreamObservationState &begin_observation() {
        if (!initialized) {
            throw std::logic_error(
                "raw timestream plan is not initialized");
        }
        observation.emplace();
        realized = {};
        return *observation;
    }
};

}  // namespace citlali::pipeline
