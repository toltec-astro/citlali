#pragma once

#include <citlali/core/error/error.h>

#include <fmt/core.h>

#include <cstddef>
#include <string_view>

namespace citlali::pipeline {

inline void require_group_value_not_seen(
    bool already_seen, std::string_view grouping, long long value) {
    if (already_seen) {
        throw citlali::error::io(fmt::format(
            "non-contiguous grouping detected for '{}' value {}",
            grouping, value));
    }
}

inline void require_valid_weight_counters(
    std::ptrdiff_t group_detector_count,
    std::ptrdiff_t unflagged_count,
    std::ptrdiff_t positive_unflagged_count,
    std::ptrdiff_t below_limit_count,
    std::ptrdiff_t above_limit_count) {
    if (unflagged_count < 0 ||
        unflagged_count > group_detector_count ||
        positive_unflagged_count < 0 ||
        positive_unflagged_count > unflagged_count ||
        below_limit_count < 0 ||
        below_limit_count > unflagged_count ||
        above_limit_count < 0 ||
        above_limit_count > unflagged_count) {
        throw citlali::error::internal(fmt::format(
            "invalid PTC weight counters: group_dets={} unflagged={} "
            "positive_unflagged={} below_limit={} above_limit={}",
            group_detector_count, unflagged_count,
            positive_unflagged_count, below_limit_count,
            above_limit_count));
    }
}

inline void require_kernel_image_cardinality(
    std::size_t image_count, std::ptrdiff_t map_count) {
    if (map_count < 0 ||
        (image_count != 1 &&
         image_count != static_cast<std::size_t>(map_count))) {
        throw citlali::error::invalid_config(fmt::format(
            "kernel image count {} must be 1 or match map count {}",
            image_count, map_count));
    }
}

}  // namespace citlali::pipeline
