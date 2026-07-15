#pragma once

#include <array>
#include <cstddef>

namespace citlali::config {

inline constexpr std::size_t toltec_interface_count = 13;

struct InterfaceSyncOffsetConfig {
    std::array<double, toltec_interface_count> toltec_offset_sec{};
    double hwpr_offset_sec = 0.0;
};

}  // namespace citlali::config
