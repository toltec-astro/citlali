#pragma once

#include <citlali/core/config/interface_sync_config.h>
#include <citlali/core/pipeline/interface_sync_state.h>

#include <string>

namespace citlali::pipeline {

inline void adapt_interface_sync_config_one_way(
    const citlali::config::InterfaceSyncOffsetConfig &config,
    InterfaceSyncState &state) {
    state.offsets.clear();
    state.lifecycle.clear();
    state.lifecycle.reserve(citlali::config::toltec_interface_count + 2);
    for (std::size_t index = 0;
         index < citlali::config::toltec_interface_count; ++index) {
        const auto interface_id = "toltec" + std::to_string(index);
        const double value = config.toltec_offset_sec[index];
        state.offsets.emplace(interface_id, value);
        InterfaceOffsetLifecycleRecord record;
        record.interface_id = interface_id;
        record.requested_sec = value;
        record.effective_sec = value;
        record.source = config.toltec_configured[index]
            ? (value == 0.0 ? "configured_zero" : "configured_nonzero")
            : "schema_default_zero";
        state.lifecycle.push_back(std::move(record));
    }
    state.offsets.emplace("hwpr", config.hwpr_offset_sec);
    InterfaceOffsetLifecycleRecord hwpr;
    hwpr.interface_id = "hwpr";
    hwpr.requested_sec = config.hwpr_offset_sec;
    hwpr.effective_sec = config.hwpr_offset_sec;
    hwpr.source = config.hwpr_configured
        ? (config.hwpr_offset_sec == 0.0 ? "configured_zero"
                                         : "configured_nonzero")
        : "schema_default_zero";
    state.lifecycle.push_back(std::move(hwpr));

    InterfaceOffsetLifecycleRecord lmt;
    lmt.interface_id = "lmt";
    lmt.source = "schema_default_zero";
    state.lifecycle.push_back(std::move(lmt));
}

template <class OffsetMap>
void adapt_interface_sync_config_one_way(
    const citlali::config::InterfaceSyncOffsetConfig &config,
    OffsetMap &offsets) {
    offsets.clear();
    for (std::size_t index = 0;
         index < citlali::config::toltec_interface_count; ++index) {
        offsets.emplace(
            "toltec" + std::to_string(index),
            config.toltec_offset_sec[index]);
    }
    offsets.emplace("hwpr", config.hwpr_offset_sec);
}

}  // namespace citlali::pipeline
