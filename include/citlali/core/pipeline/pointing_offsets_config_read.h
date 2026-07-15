#pragma once

#include <citlali/core/config/calibration_config_validation.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/pointing_offsets_config.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline {

template <class Config, class Logger>
citlali::config::AstrometryConfig read_astrometry_config(
    Config &config, const Logger &logger) {
    if (!config.has("pointing_offsets")) {
        logger->error("pointing_offsets not found in config");
        throw citlali::error::invalid_config(
            "invalid astrometry pointing_offsets configuration");
    }

    citlali::config::AstrometryConfig astrometry;
    auto &request = astrometry.pointing_offsets;
    request.enabled = true;

    auto pointing_node = config.get_node(std::tuple{"pointing_offsets"});
    bool has_az = false;
    bool has_alt = false;
    bool has_mjd = false;
    std::vector<double> mjd_values;

    for (Eigen::Index i = 0; i < pointing_node.size(); ++i) {
        if (config.has(std::tuple{"pointing_offsets", i, "axes_name"})) {
            auto axis = normalized_pointing_axis_name(
                config.get_str(
                    std::tuple{"pointing_offsets", i, "axes_name"}));
            if (citlali::config::is_supported_pointing_axis(axis)) {
                auto offset = config.template get_typed<std::vector<double>>(
                    std::tuple{"pointing_offsets", i, "value_arcsec"});
                if ((citlali::config::is_pointing_axis_az(axis) && has_az) ||
                    (citlali::config::is_pointing_axis_alt(axis) && has_alt)) {
                    logger->warn(
                        "pointing_offsets {} specified multiple times; using last value",
                        axis);
                }
                if (citlali::config::is_pointing_axis_az(axis)) {
                    request.az_arcsec = std::move(offset);
                    has_az = true;
                }
                else {
                    request.alt_arcsec = std::move(offset);
                    has_alt = true;
                }
            }
            else {
                logger->warn(
                    "unknown pointing_offsets axis_name '{}' at entry {}",
                    axis, i);
            }
        }
        else if (config.has(
                     std::tuple{"pointing_offsets", i,
                                "modified_julian_date"})) {
            mjd_values = config.template get_typed<std::vector<double>>(
                std::tuple{"pointing_offsets", i, "modified_julian_date"});
            has_mjd = true;
        }
        else {
            logger->warn(
                "unrecognized pointing_offsets entry {}. expected axes_name/value_arcsec or modified_julian_date",
                i);
        }
    }

    if (!has_az && config.has(std::tuple{"pointing_offsets", 0,
                                         "value_arcsec"})) {
        auto offset = config.template get_typed<std::vector<double>>(
            std::tuple{"pointing_offsets", 0, "value_arcsec"});
        logger->warn(
            "pointing_offsets az parsed by positional index; consider setting axes_name: az");
        request.az_arcsec = std::move(offset);
        has_az = true;
    }
    if (!has_alt && config.has(std::tuple{"pointing_offsets", 1,
                                          "value_arcsec"})) {
        auto offset = config.template get_typed<std::vector<double>>(
            std::tuple{"pointing_offsets", 1, "value_arcsec"});
        logger->warn(
            "pointing_offsets alt parsed by positional index; consider setting axes_name: alt");
        request.alt_arcsec = std::move(offset);
        has_alt = true;
    }
    if (!has_mjd &&
        config.has(std::tuple{"pointing_offsets", 2,
                              "modified_julian_date"})) {
        mjd_values = config.template get_typed<std::vector<double>>(
            std::tuple{"pointing_offsets", 2, "modified_julian_date"});
        has_mjd = true;
    }

    const auto n_az = request.az_arcsec.size();

    if (has_mjd) {
        if (mjd_values.size() == 2) {
            request.modified_julian_date = std::move(mjd_values);
        }
        else if (!mjd_values.empty() &&
                 std::all_of(mjd_values.begin(), mjd_values.end(),
                             [](double value) {
                                 return std::isfinite(value) && value <= 0.0;
                             })) {
            request.modified_julian_date = {0.0, 0.0};
        }
        else if (mjd_values.size() == 1 && n_az == 1 &&
                 std::isfinite(mjd_values.front())) {
            logger->warn(
                "ignoring single pointing_offsets.modified_julian_date for single pointing offset; using a constant offset across the observation");
            request.modified_julian_date = {0.0, 0.0};
        }
        else {
            request.modified_julian_date = std::move(mjd_values);
        }
    }
    else {
        request.modified_julian_date = {0.0, 0.0};
    }

    return astrometry;
}

template <class Logger>
void require_valid_astrometry_config(
    const citlali::config::AstrometryConfig &config, const Logger &logger) {
    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    if (report.ok()) {
        return;
    }
    logger->error(
        "invalid astrometry pointing_offsets configuration:\n{}",
        report.format_for_cli());
    throw citlali::error::invalid_config(
        "invalid astrometry pointing_offsets configuration");
}

}  // namespace citlali::pipeline
