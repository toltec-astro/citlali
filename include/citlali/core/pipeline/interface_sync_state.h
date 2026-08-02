#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::pipeline {

enum class OffsetAvailability {
    observation_resolved,
    not_applicable,
    unavailable_authority,
};

inline const char *to_string(OffsetAvailability availability) noexcept {
    switch (availability) {
        case OffsetAvailability::observation_resolved:
            return "observation_resolved";
        case OffsetAvailability::not_applicable:
            return "not_applicable";
        case OffsetAvailability::unavailable_authority:
            return "unavailable_authority";
    }
    return "unavailable_authority";
}

struct InterfaceOffsetLifecycleRecord {
    std::string interface_id;
    double requested_sec = 0.0;
    double effective_sec = 0.0;
    double observation_resolved_sec = 0.0;
    double realized_sec = 0.0;
    std::string source = "schema_default_zero";
    std::string sign = "positive_add";
    std::string reference = "detector_clock";
    std::string unit = "s";
    std::string application_stage = "before_ordering_slotting_and_gaps";
    std::string uncertainty = "unavailable";
    OffsetAvailability availability = OffsetAvailability::not_applicable;
    bool applied_exactly_once = false;
};

struct InterfaceSyncState {
    std::map<std::string, double> offsets;
    std::vector<InterfaceOffsetLifecycleRecord> lifecycle;
};

inline void begin_interface_sync_observation(InterfaceSyncState &state) {
    for (auto &record : state.lifecycle) {
        record.observation_resolved_sec = 0.0;
        record.realized_sec = 0.0;
        record.availability = OffsetAvailability::not_applicable;
        record.applied_exactly_once = false;
    }
}

inline InterfaceOffsetLifecycleRecord &require_interface_offset_record(
    InterfaceSyncState &state, const std::string &interface_id) {
    for (auto &record : state.lifecycle) {
        if (record.interface_id == interface_id) {
            return record;
        }
    }
    throw std::runtime_error(
        "missing typed interface offset record for " + interface_id);
}

inline double realize_interface_offset(
    InterfaceSyncState &state, const std::string &interface_id,
    bool comparable_epoch_authority_available) {
    auto &record = require_interface_offset_record(state, interface_id);
    if (record.applied_exactly_once) {
        throw std::runtime_error(
            "interface offset was applied more than once for " + interface_id);
    }
    if (record.effective_sec != 0.0 &&
        !comparable_epoch_authority_available) {
        record.availability = OffsetAvailability::unavailable_authority;
        throw std::runtime_error(
            "nonzero interface offset lacks comparable clock/epoch authority for " +
            interface_id);
    }
    record.observation_resolved_sec = record.effective_sec;
    record.realized_sec = record.effective_sec;
    record.availability = OffsetAvailability::observation_resolved;
    record.applied_exactly_once = true;
    return record.realized_sec;
}

}  // namespace citlali::pipeline
