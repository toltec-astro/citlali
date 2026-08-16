#pragma once

#include <citlali/core/config/interface_sync_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/native_cohort_product_provenance.h>
#include <citlali/core/pipeline/raw_timestream_resolution.h>

#include <cstddef>
#include <memory>
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
    // Populated only for admitted native Science/Pointing observations.  The
    // slots are observation-owned and intentionally do not publish a product.
    std::shared_ptr<NativeCohortObservationLineage> native_cohort_lineage;
};

struct RawTimestreamRealizedState {
    bool execution_completed = false;
    std::optional<std::size_t> completed_scan_count;
    std::optional<std::size_t> flagged_sample_count;
    std::optional<std::size_t> dynamic_notch_count;
    std::optional<std::size_t> required_timestream_write_count;
    // Snapshot captured only after every reserved native scan commits.  The
    // existing raw-timestream provenance writer serializes it; B3c owns any
    // separate final sidecar or index publication.
    std::optional<NativeCohortProductProvenance> native_cohort_provenance;
};

struct RawTimestreamExecutionPlan {
    bool initialized = false;
    citlali::config::RawTimeChunkConfig requested;
    citlali::config::RawTimeChunkConfig effective;
    citlali::config::InterfaceSyncOffsetConfig interface_sync_requested;
    citlali::config::InterfaceSyncOffsetConfig interface_sync_effective;
    RawTimestreamEffectiveResolutions effective_resolutions;
    std::optional<RawTimestreamObservationState> observation;
    RawTimestreamRealizedState realized;

    void reset_from_request(
        const citlali::config::RawTimeChunkConfig &request,
        const citlali::config::InterfaceSyncOffsetConfig
            &interface_sync_request = {}) {
        initialized = true;
        requested = request;
        effective = request;
        interface_sync_requested = interface_sync_request;
        interface_sync_effective = interface_sync_request;
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
