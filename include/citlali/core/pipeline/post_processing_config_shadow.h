#pragma once

#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/pipeline/post_processing_execution_plan.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct PostProcessingConfigShadowReport {
    bool exact = true;
    bool compared_source_fitting_details = false;
    std::vector<std::string> mismatches;

    void add_mismatch(std::string mismatch) {
        exact = false;
        mismatches.push_back(std::move(mismatch));
    }

    std::string diagnostic() const {
        std::ostringstream stream;
        for (std::size_t index = 0; index < mismatches.size(); ++index) {
            if (index != 0) {
                stream << "; ";
            }
            stream << mismatches[index];
        }
        return stream.str();
    }
};

inline bool post_processing_shadow_double_equal(double expected,
                                                double actual) {
    if (expected == actual) {
        return true;
    }
    const double scale = std::max(
        {1.0, std::abs(expected), std::abs(actual)});
    return std::abs(expected - actual) <=
           8.0 * std::numeric_limits<double>::epsilon() * scale;
}

template <class Value>
std::string post_processing_shadow_value_string(const Value &value) {
    std::ostringstream stream;
    if constexpr (std::is_enum_v<Value>) {
        stream << citlali::config::to_string(value);
    } else {
        stream << value;
    }
    return stream.str();
}

template <class Value>
void compare_post_processing_shadow_value(
    PostProcessingConfigShadowReport &report, const char *field,
    const Value &expected, const Value &actual) {
    if (expected == actual) {
        return;
    }
    report.add_mismatch(
        std::string{field} + " expected=" +
        post_processing_shadow_value_string(expected) + " actual=" +
        post_processing_shadow_value_string(actual));
}

inline void compare_post_processing_shadow_value(
    PostProcessingConfigShadowReport &report, const char *field,
    double expected, double actual) {
    if (post_processing_shadow_double_equal(expected, actual)) {
        return;
    }
    report.add_mismatch(
        std::string{field} + " expected=" +
        post_processing_shadow_value_string(expected) + " actual=" +
        post_processing_shadow_value_string(actual));
}

inline PostProcessingConfigShadowReport compare_post_processing_config_shadow(
    const citlali::config::PostProcessingConfig &requested,
    const citlali::config::PostProcessingConfig &legacy,
    citlali::config::ReductionType reduction_type) {
    PostProcessingConfigShadowReport report;
    compare_post_processing_shadow_value(
        report, "map_filtering.enabled", requested.map_filtering.enabled,
        legacy.map_filtering.enabled);
    compare_post_processing_shadow_value(
        report, "map_histogram_n_bins", requested.map_histogram_n_bins,
        legacy.map_histogram_n_bins);
    compare_post_processing_shadow_value(
        report, "source_finding.enabled", requested.source_finding.enabled,
        legacy.source_finding.enabled);

    const bool fitting_required = post_processing_source_fitting_required(
        reduction_type, requested);
    compare_post_processing_shadow_value(
        report, "source_fitting.active", fitting_required,
        legacy.source_fitting.active);
    if (fitting_required) {
        report.compared_source_fitting_details = true;
        const auto &expected = requested.source_fitting;
        const auto &actual = legacy.source_fitting;
        compare_post_processing_shadow_value(
            report, "source_fitting.model", expected.model, actual.model);
        compare_post_processing_shadow_value(
            report, "source_fitting.bounding_box_arcsec",
            expected.bounding_box_arcsec, actual.bounding_box_arcsec);
        compare_post_processing_shadow_value(
            report, "source_fitting.fitting_radius_arcsec",
            expected.fitting_radius_arcsec, actual.fitting_radius_arcsec);
        compare_post_processing_shadow_value(
            report, "source_fitting.fit_rotation_angle",
            expected.fit_rotation_angle, actual.fit_rotation_angle);
        for (std::size_t index = 0; index < 2; ++index) {
            const auto amp_field =
                "source_fitting.amp_limit_factors." +
                std::to_string(index);
            compare_post_processing_shadow_value(
                report, amp_field.c_str(),
                expected.amp_limit_factors[index],
                actual.amp_limit_factors[index]);
            const auto fwhm_field =
                "source_fitting.fwhm_limit_factors." +
                std::to_string(index);
            compare_post_processing_shadow_value(
                report, fwhm_field.c_str(),
                expected.fwhm_limit_factors[index],
                actual.fwhm_limit_factors[index]);
        }
    }
    return report;
}

}  // namespace citlali::pipeline
