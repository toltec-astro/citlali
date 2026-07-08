#pragma once

#include <citlali/core/config/timestream_config.h>

#include <Eigen/Core>

#include <stdexcept>
#include <utility>

namespace citlali::pipeline {

template <class SolverResult, class Visitor>
decltype(auto) visit_kids_tod_channel(const SolverResult &result,
                                      citlali::config::TodType type,
                                      Visitor &&visitor) {
    switch (type) {
    case citlali::config::TodType::xs:
        return std::forward<Visitor>(visitor)(result.data_out.xs.data);
    case citlali::config::TodType::rs:
        return std::forward<Visitor>(visitor)(result.data_out.rs.data);
    case citlali::config::TodType::is:
        return std::forward<Visitor>(visitor)(result.data.is.data);
    case citlali::config::TodType::qs:
        return std::forward<Visitor>(visitor)(result.data.qs.data);
    }
    throw std::runtime_error("unsupported KIDs TOD channel");
}

}  // namespace citlali::pipeline
