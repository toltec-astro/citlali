#pragma once

// Private YAML-to-AST adapter. Offset values remain signed tangent-azimuth and
// elevation arcseconds. TolTECA owns User/Paddle correction composition.
#include <citlali/core/pipeline/timestream_native_pointing.h>
#include <citlali/core/utils/utils.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace citlali::cli::detail {
inline pipeline::NativePointingOffsetModel read_cal_pointing_offsets(
    const YAML::Node &entries, const Eigen::VectorXd &telescope_times,
    const Eigen::VectorXd *common_observation_times = nullptr) {
    auto require = [](bool ok, const char *message) {
        if (!ok) throw std::invalid_argument(message);
    };
    require(entries.IsSequence() && entries.size() > 0,
            "AST pointing offsets must be a nonempty sequence");
    pipeline::NativePointingOffsetsArcsec values;
    std::vector<double> mjd;
    bool has_mjd = false;
    for (const auto &entry : entries) {
        require(entry.IsMap(), "AST pointing offset entry must be a mapping");
        if (entry["modified_julian_date"]) {
            require(!has_mjd && entry.size() == 1,
                    "AST offset time support duplicated or ambiguous");
            has_mjd = true;
            mjd = entry["modified_julian_date"].as<std::vector<double>>();
            require((mjd.size() == 1 || mjd.size() == 2) &&
                        std::all_of(mjd.begin(), mjd.end(),
                                    [](double v) { return std::isfinite(v); }),
                    "AST offset epochs require one or two finite values");
        } else {
            require(entry.size() == 2 && entry["axes_name"] && entry["value_arcsec"],
                    "AST offset axis entry is incomplete or ambiguous");
            const auto axis = entry["axes_name"].as<std::string>();
            require((axis == "az" || axis == "alt") && !values.contains(axis),
                    "AST offset axis duplicate or unknown");
            const auto v = entry["value_arcsec"].as<std::vector<double>>();
            require((v.size() == 1 || v.size() == 2) &&
                        std::all_of(v.begin(), v.end(),
                                    [](double x) { return std::isfinite(x); }),
                    "AST offset axis requires one or two finite values");
            values[axis] = Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());
        }
    }
    require(values.contains("az") && values.contains("alt") &&
                values.at("az").size() == values.at("alt").size(),
            "AST pointing requires equal az and alt value counts");
    const bool constant = values.at("az").size() == 1;
    const bool sentinel = !has_mjd ||
        std::all_of(mjd.begin(), mjd.end(), [](double v) { return v <= 0.; });
    const bool actual_pair = mjd.size() == 2 && mjd[0] > 0. && mjd[1] > mjd[0];
    require(sentinel || actual_pair || (constant && mjd.size() == 1 && mjd[0] > 0.),
            "AST offsets require increasing actual epochs or an observation-span sentinel");

    Eigen::VectorXd support(2);
    if (!constant && actual_pair) {
        for (int i = 0; i < 2; ++i) {
            // Guard the integer conversion, then call the exact OG owner:
            // preserve its expression and truncation to whole Unix seconds.
            const double seconds = (mjd[i] - 2440587.5 + 2400000.5) * 86400.;
            require(std::isfinite(seconds) &&
                        seconds >= static_cast<double>(std::numeric_limits<long long>::lowest()) &&
                        seconds < static_cast<double>(std::numeric_limits<long long>::max()),
                    "AST offset epoch is outside representable Unix seconds");
            support[i] = static_cast<double>(engine_utils::modified_julian_date_to_unix(mjd[i]));
        }
    } else {
        // A constant does not interpolate against its PPT date. Two undated
        // values use OG's realized common-grid endpoints, never raw TEL bounds.
        require(constant || common_observation_times != nullptr,
                "AST observation-span offsets require realized common observation support");
        const auto &times = constant ? telescope_times : *common_observation_times;
        require(times.size() >= 2, "AST offsets require two observation support bounds");
        support << times[0], times[times.size() - 1];
    }
    // The AST model rejects nonfinite, reversed and integer-collapsed support;
    // it retains the existing no-extrapolation rule for two-value offsets.
    return pipeline::NativePointingOffsetModel{std::move(values), std::move(support)};
}
// Read a saved CAL model without reinterpreting an older time convention.
// Old one-value offsets are safe constants. Old two-value receipts lack the
// conversion/support declaration needed to distinguish fractional and OG time.
inline pipeline::NativePointingOffsetModel read_saved_cal_pointing_offsets(
    const YAML::Node &entries, const Eigen::VectorXd &telescope_times,
    const YAML::Node &receipt) {
    const auto saved = receipt["AST_pointing_offset_support_unix_sec"];
    if (!saved) {
        auto model = read_cal_pointing_offsets(entries, telescope_times);
        if (model.source_value_count() != 1)
            throw std::invalid_argument(
                "saved CAL two-value offsets lack an explicit epoch convention; regenerate with this candidate");
        return model;
    }
    const auto values = saved.as<std::vector<double>>();
    if (values.size() != 2 || !std::isfinite(values[0]) || !std::isfinite(values[1]) ||
        values[1] <= values[0] || !receipt["AST_pointing_offset_epoch_conversion"] ||
        receipt["AST_pointing_offset_epoch_conversion"].as<std::string>() != "OG-whole-Unix-seconds-v1")
        throw std::invalid_argument("saved CAL pointing support or epoch convention differs");
    const Eigen::VectorXd support = Eigen::Map<const Eigen::VectorXd>(values.data(), values.size());
    auto model = read_cal_pointing_offsets(entries, telescope_times, &support);
    if (!(model.support_times_unix_sec().array() == support.array()).all())
        throw std::invalid_argument("saved CAL pointing offset support differs from its configuration");
    return model;
}
} // namespace citlali::cli::detail
