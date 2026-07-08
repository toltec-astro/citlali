#pragma once

#include <citlali/core/config/timestream_config.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>

namespace timestream {

enum class TimestreamChannel {
    science_x,
    quadrature_r,
    synthetic_kernel
};

struct AuxiliaryMeasuredStream {
    using CalibrationPolicy =
        citlali::config::AuxiliaryMeasuredChannelCalibrationPolicy;

    Eigen::MatrixXd data;
    TimestreamChannel channel = TimestreamChannel::quadrature_r;
    std::string name = "r";
    citlali::config::TodType source_type = citlali::config::TodType::rs;
    std::string native_unit = "native";
    CalibrationPolicy calibration_policy = CalibrationPolicy::native;
    bool apply_primary_linear_transfer = true;
    bool use_for_science_map = false;
    bool diagnostics_enabled = false;

    [[nodiscard]] bool has_data() const {
        return data.size() != 0;
    }

    [[nodiscard]] Eigen::Index rows() const {
        return data.rows();
    }

    [[nodiscard]] Eigen::Index cols() const {
        return data.cols();
    }
};

using AuxiliaryMeasuredStreams = std::vector<AuxiliaryMeasuredStream>;

inline AuxiliaryMeasuredStream *find_auxiliary_measured_stream(
    AuxiliaryMeasuredStreams &streams, std::string_view name) {
    auto it = std::find_if(streams.begin(), streams.end(),
                           [name](const auto &stream) {
                               return std::string_view{stream.name} == name;
                           });
    return it == streams.end() ? nullptr : &(*it);
}

inline const AuxiliaryMeasuredStream *find_auxiliary_measured_stream(
    const AuxiliaryMeasuredStreams &streams, std::string_view name) {
    auto it = std::find_if(streams.begin(), streams.end(),
                           [name](const auto &stream) {
                               return std::string_view{stream.name} == name;
                           });
    return it == streams.end() ? nullptr : &(*it);
}

}  // namespace timestream
