#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/mapmaking_config.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>

namespace citlali::config {

inline void validate(const MapmakingConfig &config, ValidationReport &report) {
    if (is_maximum_likelihood_map_method(config.method)) {
        report.add_error(
            {"mapmaking", "method"},
            "maximum_likelihood is under development and is not a supported "
            "production mapmaking method");
    }
    check_greater_than(config.pixel_size_arcsec, 0.0,
                       {"mapmaking", "pixel_size_arcsec"}, report);
    check_minimum(config.x_size_pix, 0, {"mapmaking", "x_size_pix"}, report);
    check_minimum(config.y_size_pix, 0, {"mapmaking", "y_size_pix"}, report);
    check_finite_value(
        config.coverage_cut, {"mapmaking", "coverage_cut"}, report);
    check_finite_value(config.crpix1, {"mapmaking", "crpix1"}, report);
    check_finite_value(config.crpix2, {"mapmaking", "crpix2"}, report);
    check_finite_value(
        config.crval1_j2000, {"mapmaking", "crval1_J2000"}, report);
    check_finite_value(
        config.crval2_j2000, {"mapmaking", "crval2_J2000"}, report);
    check_finite_value(config.tan_ra, {"mapmaking", "tan_ra"}, report);
    check_finite_value(config.tan_dec, {"mapmaking", "tan_dec"}, report);
    check_greater_than(
        config.jinc_filter.r_max, 0.0,
        {"mapmaking", "jinc_filter", "r_max"}, report);
    check_minimum(
        config.jinc_filter.subpixel_n, 1,
        {"mapmaking", "jinc_filter", "subpixel_n"}, report);
    for (const auto &[array_name, shape] :
         config.jinc_filter.shape_params) {
        for (std::size_t index = 0; index < shape.size(); ++index) {
            check_finite_value(
                shape[index],
                {"mapmaking", "jinc_filter", "shape_params",
                 array_name, std::to_string(index)},
                report);
        }
    }
    if (config.jinc_accounting.enabled) {
        if (!is_jinc_map_method(config.method)) {
            report.add_error(
                {"mapmaking", "jinc_accounting", "enabled"},
                "JINC accounting requires mapmaking.method='jinc'");
        }
        if (!is_automatic_map_grouping(config.grouping) &&
            !is_array_map_grouping(config.grouping)) {
            report.add_error(
                {"mapmaking", "jinc_accounting", "array"},
                "JINC accounting requires automatic or array grouping");
        }
        constexpr std::array<const char *, 3> arrays{
            "a1100", "a1400", "a2000"};
        if (std::find(arrays.begin(), arrays.end(),
                      config.jinc_accounting.array) == arrays.end()) {
            report.add_error(
                {"mapmaking", "jinc_accounting", "array"},
                "JINC accounting array must be a1100, a1400, or a2000");
        }
        check_minimum(
            config.jinc_accounting.uid, 0,
            {"mapmaking", "jinc_accounting", "uid"}, report);
        check_minimum(
            config.jinc_accounting.scan_index, 0,
            {"mapmaking", "jinc_accounting", "scan_index"}, report);
    }
    check_minimum(
        config.maximum_likelihood.max_iterations, 1,
        {"mapmaking", "maximum_likelihood", "max_iterations"}, report);
    check_greater_than(
        config.maximum_likelihood.tolerance, 0.0,
        {"mapmaking", "maximum_likelihood", "tolerance"}, report);
}

}  // namespace citlali::config
