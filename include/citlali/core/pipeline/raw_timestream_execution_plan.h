#pragma once

#include <citlali/core/config/interface_sync_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/native_cohort_product_provenance_v3.h>
#include <citlali/core/pipeline/native_consumer_mode_policy.h>
#include <citlali/core/pipeline/raw_timestream_resolution.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::pipeline {

struct RawTimestreamConfigSourceIdentity {
    std::size_t precedence = 0;
    std::string path;
    std::uintmax_t size_bytes = 0;
    std::string sha256;
};

struct RawTimestreamCanonicalRunIdentity {
    std::string accepted_merged_config_sha256;
    std::string effective_configuration_identity;
    std::string runtime_effective_identity;
    std::vector<RawTimestreamConfigSourceIdentity> config_sources;

    static bool is_sha256_identity(const std::string &value) noexcept {
        constexpr std::size_t prefix_size = 7;
        if (value.size() != prefix_size + 64 ||
            value.compare(0, prefix_size, "sha256:") != 0) {
            return false;
        }
        for (auto index = prefix_size; index < value.size(); ++index) {
            const auto ch = value[index];
            if (!((ch >= '0' && ch <= '9') ||
                  (ch >= 'a' && ch <= 'f'))) {
                return false;
            }
        }
        return true;
    }

    bool complete() const noexcept {
        if (!is_sha256_identity(accepted_merged_config_sha256) ||
            !is_sha256_identity(effective_configuration_identity) ||
            !is_sha256_identity(runtime_effective_identity) ||
            config_sources.empty()) {
            return false;
        }
        for (std::size_t index = 0; index < config_sources.size(); ++index) {
            const auto &source = config_sources[index];
            if (source.precedence != index || source.path.empty() ||
                !is_sha256_identity(source.sha256)) {
                return false;
            }
        }
        return true;
    }
};

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
    NativeConsumerRoute native_consumer_route =
        NativeConsumerRoute::legacy_inactive;
    std::shared_ptr<NativeCohortObservationLineageV3>
        native_cohort_lineage;
};

struct RawTimestreamRealizedState {
    bool execution_completed = false;
    std::optional<std::size_t> completed_scan_count;
    std::optional<std::size_t> flagged_sample_count;
    std::optional<std::size_t> dynamic_notch_count;
    std::optional<std::size_t> required_timestream_write_count;
    std::optional<NativeCohortProductProvenanceV3>
        native_cohort_provenance;
};

struct RawTimestreamExecutionPlan {
    bool initialized = false;
    citlali::config::RawTimeChunkConfig requested;
    citlali::config::RawTimeChunkConfig effective;
    citlali::config::InterfaceSyncOffsetConfig interface_sync_requested;
    citlali::config::InterfaceSyncOffsetConfig interface_sync_effective;
    RawTimestreamEffectiveResolutions effective_resolutions;
    std::optional<RawTimestreamCanonicalRunIdentity>
        canonical_run_identity;
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
        canonical_run_identity.reset();
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
