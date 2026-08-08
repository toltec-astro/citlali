#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace citlali::pipeline {

inline std::array<std::size_t, 2> tod_output_append_bounds(
    std::size_t existing_samples, std::size_t appended_samples) {
    if (appended_samples == 0) {
        throw std::invalid_argument("TOD output cannot append an empty scan");
    }
    if (appended_samples - 1 >
        std::numeric_limits<std::size_t>::max() - existing_samples) {
        throw std::overflow_error("TOD output append bounds overflow");
    }
    return {existing_samples, existing_samples + appended_samples - 1};
}

}  // namespace citlali::pipeline
