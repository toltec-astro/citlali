#pragma once

#include <citlali/core/error/error.h>

#include <fmt/core.h>

#include <cstddef>
#include <string>
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

inline void require_filtered_fruit_loop_feedback_product_contract(
    bool feedback_contract_present, bool feedback_approved,
    bool filter_operator_present, std::string_view filter_operator,
    std::string_view product_name) {
    if (feedback_contract_present && !feedback_approved) {
        throw citlali::error::io(fmt::format(
            "fruit-loop product '{}' is explicitly withheld from feedback "
            "by FLFBACK=false",
            product_name));
    }
    if (filter_operator_present &&
        filter_operator == "unit_sum_convolution") {
        throw citlali::error::io(fmt::format(
            "fruit-loop product '{}' was produced by withheld unit-sum "
            "convolution",
            product_name));
    }
    if (feedback_contract_present && feedback_approved) {
        return;
    }
    if (filter_operator_present && filter_operator == "wiener_filter") {
        return;
    }
    if (filter_operator_present) {
        throw citlali::error::io(fmt::format(
            "filtered fruit-loop product '{}' has unapproved producer "
            "FILTEROP='{}'",
            product_name, filter_operator));
    }
    throw citlali::error::io(fmt::format(
        "filtered fruit-loop product '{}' lacks producer FILTEROP identity "
        "and an explicit FLFBACK approval contract",
        product_name));
}

}  // namespace citlali::pipeline
