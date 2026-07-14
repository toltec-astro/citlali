#pragma once

#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/config/runtime_config.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct PostProcessingConfigShadowReport {
    bool exact = true;
    bool compared_map_filter_details = false;
    bool compared_source_finding_details = false;
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

inline void compare_post_processing_fwhm_shadow(
    PostProcessingConfigShadowReport &report,
    const std::map<std::string, double> &expected,
    const std::map<std::string, double> &actual) {
    if (expected.size() != actual.size()) {
        report.add_mismatch(
            "map_filtering.template_fwhm_arcsec size expected=" +
            std::to_string(expected.size()) + " actual=" +
            std::to_string(actual.size()));
        return;
    }
    for (const auto &[array_name, expected_fwhm] : expected) {
        const auto found = actual.find(array_name);
        if (found == actual.end()) {
            report.add_mismatch(
                "map_filtering.template_fwhm_arcsec missing array=" +
                array_name);
            continue;
        }
        const auto field =
            "map_filtering.template_fwhm_arcsec." + array_name;
        compare_post_processing_shadow_value(
            report, field.c_str(), expected_fwhm, found->second);
    }
}

inline bool post_processing_source_fitting_required(
    citlali::config::ReductionType reduction_type,
    const citlali::config::PostProcessingConfig &requested) {
    return citlali::config::is_pointing_reduction_type(reduction_type) ||
           citlali::config::is_beammap_reduction_type(reduction_type) ||
           requested.map_filtering.enabled ||
           requested.source_finding.enabled;
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

    if (requested.map_filtering.enabled) {
        report.compared_map_filter_details = true;
        const auto &expected = requested.map_filtering;
        const auto &actual = legacy.map_filtering;
        compare_post_processing_shadow_value(
            report, "map_filtering.type", expected.type, actual.type);
        compare_post_processing_shadow_value(
            report, "map_filtering.template_type", expected.template_type,
            actual.template_type);
        compare_post_processing_shadow_value(
            report, "map_filtering.kernel_template_tail_mode",
            expected.kernel_template_tail_mode,
            actual.kernel_template_tail_mode);
        compare_post_processing_shadow_value(
            report, "map_filtering.lowpass_only", expected.lowpass_only,
            actual.lowpass_only);
        compare_post_processing_shadow_value(
            report, "map_filtering.normalize_errors",
            expected.normalize_errors, actual.normalize_errors);
        compare_post_processing_shadow_value(
            report, "map_filtering.edge_guard.enabled",
            expected.edge_guard.enabled, actual.edge_guard.enabled);
        compare_post_processing_shadow_value(
            report, "map_filtering.edge_guard.weight_threshold_mode",
            expected.edge_guard.weight_threshold_mode,
            actual.edge_guard.weight_threshold_mode);
        compare_post_processing_shadow_value(
            report, "map_filtering.edge_guard.hits_threshold_mode",
            expected.edge_guard.hits_threshold_mode,
            actual.edge_guard.hits_threshold_mode);
        compare_post_processing_shadow_value(
            report, "map_filtering.edge_guard.hits_core_fraction",
            expected.edge_guard.hits_core_fraction,
            actual.edge_guard.hits_core_fraction);
        compare_post_processing_shadow_value(
            report, "map_filtering.edge_guard.guard_radius_fwhm",
            expected.edge_guard.guard_radius_fwhm,
            actual.edge_guard.guard_radius_fwhm);
        compare_post_processing_shadow_value(
            report, "map_filtering.edge_guard.fill_mode",
            expected.edge_guard.fill_mode, actual.edge_guard.fill_mode);
        compare_post_processing_shadow_value(
            report, "map_filtering.edge_guard.taper_mode",
            expected.edge_guard.taper_mode, actual.edge_guard.taper_mode);
        compare_post_processing_shadow_value(
            report, "map_filtering.edge_guard.taper_min_fraction",
            expected.edge_guard.taper_min_fraction,
            actual.edge_guard.taper_min_fraction);
        compare_post_processing_shadow_value(
            report, "map_filtering.denom_rel_tol", expected.denom_rel_tol,
            actual.denom_rel_tol);
        compare_post_processing_shadow_value(
            report, "map_filtering.tail_frac_tol", expected.tail_frac_tol,
            actual.tail_frac_tol);
        compare_post_processing_shadow_value(
            report, "map_filtering.max_loops", expected.max_loops,
            actual.max_loops);
        compare_post_processing_shadow_value(
            report, "map_filtering.denom_check_iters",
            expected.denom_check_iters, actual.denom_check_iters);
        compare_post_processing_shadow_value(
            report, "map_filtering.max_denom_iters",
            expected.max_denom_iters, actual.max_denom_iters);
        if (citlali::config::map_filter_template_uses_fwhm(
                expected.template_type)) {
            compare_post_processing_fwhm_shadow(
                report, expected.template_fwhm_arcsec,
                actual.template_fwhm_arcsec);
        }
    }

    if (requested.source_finding.enabled) {
        report.compared_source_finding_details = true;
        compare_post_processing_shadow_value(
            report, "source_finding.source_sigma",
            requested.source_finding.source_sigma,
            legacy.source_finding.source_sigma);
        compare_post_processing_shadow_value(
            report, "source_finding.source_window_arcsec",
            requested.source_finding.source_window_arcsec,
            legacy.source_finding.source_window_arcsec);
        compare_post_processing_shadow_value(
            report, "source_finding.mode", requested.source_finding.mode,
            legacy.source_finding.mode);
    }

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
