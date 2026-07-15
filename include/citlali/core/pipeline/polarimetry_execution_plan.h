#pragma once

#include <citlali/core/config/timestream_config.h>

#include <stdexcept>
#include <string_view>

namespace citlali::pipeline {

inline constexpr bool enabled_polarimetry_available = false;
inline constexpr std::string_view polarimetry_capability_status{
    "planned-unavailable"};
inline constexpr std::string_view polarimetry_capability_reason{
    "enabled polarimetry is reserved for future development and has no "
    "approved scientific contract or reference validation dataset"};
inline constexpr std::string_view polarimetry_capability_exit_condition{
    "approve the polarimetry and HWPR contract and pass an enabled reference "
    "validation gate"};

struct PolarimetryCapabilityResolution {
    bool enabled_capability_available = enabled_polarimetry_available;
    bool requested_enabled = false;
    bool request_accepted = true;
    bool disabled_by_capability = false;
};

struct PolarimetryRealizedState {
    bool reduction_completed = false;
    bool polarimetry_executed = false;
    bool hwpr_loaded = false;
};

struct PolarimetryExecutionPlan {
    bool initialized = false;
    citlali::config::TimestreamPolarimetryConfig requested;
    citlali::config::TimestreamPolarimetryConfig effective;
    PolarimetryCapabilityResolution capability;
    PolarimetryRealizedState realized;

    void reset_from_request(
        const citlali::config::TimestreamPolarimetryConfig &request) {
        initialized = true;
        requested = request;
        effective = request;
        const bool accepted =
            enabled_polarimetry_available || !request.enabled;
        if (!accepted) {
            effective.enabled = false;
        }
        capability = PolarimetryCapabilityResolution{
            enabled_polarimetry_available,
            request.enabled,
            accepted,
            request.enabled && !enabled_polarimetry_available,
        };
        realized = {};
    }
};

inline void record_polarimetry_run_completed(
    PolarimetryExecutionPlan &plan) {
    if (!plan.initialized) {
        throw std::logic_error("polarimetry plan is not initialized");
    }
    if (!plan.capability.request_accepted) {
        throw std::logic_error(
            "unsupported polarimetry request cannot complete");
    }
    if (plan.effective.enabled) {
        throw std::logic_error(
            "enabled polarimetry cannot complete while unavailable");
    }
    plan.realized = PolarimetryRealizedState{true, false, false};
}

}  // namespace citlali::pipeline
