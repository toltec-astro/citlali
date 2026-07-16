#pragma once

#include <citlali/core/error/error.h>

#include <fmt/core.h>

#include <cstddef>
#include <string_view>

namespace citlali::pipeline {

inline void require_contiguous_fruit_loop_group(
    bool value_seen, std::string_view grouping, long long value) {
    if (value_seen) {
        throw citlali::error::io(fmt::format(
            "non-contiguous fruit-loop grouping '{}' value {}",
            grouping, value));
    }
}

inline void require_fruit_loop_array_identity(
    bool array_found, long long array_id) {
    if (!array_found) {
        throw citlali::error::io(fmt::format(
            "fruit-loop detector array {} is absent from calibration arrays",
            array_id));
    }
}

inline void require_fruit_loop_map_index(
    std::ptrdiff_t map_index, std::size_t map_count) {
    if (map_index < 0 ||
        static_cast<std::size_t>(map_index) >= map_count) {
        throw citlali::error::io(fmt::format(
            "fruit-loop map index {} is outside [0, {})",
            map_index, map_count));
    }
}

}  // namespace citlali::pipeline
