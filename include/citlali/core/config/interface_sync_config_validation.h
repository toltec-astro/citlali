#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/interface_sync_config.h>

#include <string>

namespace citlali::config {

inline void validate(
    const InterfaceSyncOffsetConfig &config, ValidationReport &report) {
    for (std::size_t index = 0; index < toltec_interface_count; ++index) {
        check_finite_value(
            config.toltec_offset_sec[index],
            {"interface_sync_offset", "toltec" + std::to_string(index)},
            report);
    }
    check_finite_value(
        config.hwpr_offset_sec,
        {"interface_sync_offset", "hwpr"}, report);
}

}  // namespace citlali::config
