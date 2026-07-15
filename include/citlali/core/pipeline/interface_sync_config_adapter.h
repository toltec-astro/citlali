#pragma once

#include <citlali/core/config/interface_sync_config.h>

#include <string>

namespace citlali::pipeline {

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
