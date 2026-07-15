#pragma once

#include <citlali/core/error/error.h>

#include <Eigen/Core>
#include <fmt/format.h>

#include <string_view>

namespace citlali::pipeline {

template <typename Derived>
void require_finite_kids_input(const Eigen::DenseBase<Derived> &data,
                               std::string_view context) {
    if (data.derived().array().isNaN().any()) {
        throw citlali::error::io(fmt::format(
            "{} contains NaN values; check the KIDs data directory",
            context));
    }
    if (data.derived().array().isInf().any()) {
        throw citlali::error::io(fmt::format(
            "{} contains infinite values; check the KIDs data directory",
            context));
    }
}

}  // namespace citlali::pipeline
