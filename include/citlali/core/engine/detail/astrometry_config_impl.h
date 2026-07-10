#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/pointing_offsets_config.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_astrometry_config(CT &config) {
    auto &astrometry_config = citlali::pipeline::astrometry_config(*this);
    astrometry_config = citlali::config::AstrometryConfig{};

    // check if config file has pointing_offsets
    if (config.has("pointing_offsets")) {
        // reset for each observation
        citlali::pipeline::clear_pointing_offsets(pointing_offsets);

        auto pointing_node = config.get_node(std::tuple{"pointing_offsets"});
        bool has_az = false;
        bool has_alt = false;
        bool has_mjd = false;
        std::vector<double> mjd_values;

        for (Eigen::Index i = 0; i < pointing_node.size(); ++i) {
            if (config.has(std::tuple{"pointing_offsets", i, "axes_name"})) {
                auto axis = citlali::pipeline::normalized_pointing_axis_name(
                    config.get_str(
                        std::tuple{"pointing_offsets", i, "axes_name"}));
                if (citlali::config::is_supported_pointing_axis(axis)) {
                    auto offset = config.template get_typed<std::vector<double>>(
                        std::tuple{"pointing_offsets", i, "value_arcsec"});
                    if (offset.empty()) {
                        logger->error("pointing_offsets {} has empty value_arcsec", axis);
                        std::exit(EXIT_FAILURE);
                    }
                    if (pointing_offsets.arcsec.find(axis) != pointing_offsets.arcsec.end()) {
                        logger->warn("pointing_offsets {} specified multiple times; using last value", axis);
                    }
                    pointing_offsets.arcsec[axis] =
                        Eigen::Map<Eigen::VectorXd>(offset.data(), offset.size());
                    if (citlali::config::is_pointing_axis_az(axis)) {
                        has_az = true;
                    }
                    else {
                        has_alt = true;
                    }
                }
                else {
                    logger->warn("unknown pointing_offsets axis_name '{}' at entry {}", axis, i);
                }
            }
            else if (config.has(std::tuple{"pointing_offsets", i, "modified_julian_date"})) {
                mjd_values = config.template get_typed<std::vector<double>>(
                    std::tuple{"pointing_offsets", i, "modified_julian_date"});
                has_mjd = true;
            }
            else {
                logger->warn("unrecognized pointing_offsets entry {}. expected axes_name/value_arcsec or modified_julian_date", i);
            }
        }

        // backward-compatible fallback for positional configs
        if (!has_az && config.has(std::tuple{"pointing_offsets", 0, "value_arcsec"})) {
            auto offset = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets", 0, "value_arcsec"});
            if (offset.empty()) {
                logger->error("pointing_offsets az has empty value_arcsec");
                std::exit(EXIT_FAILURE);
            }
            logger->warn("pointing_offsets az parsed by positional index; consider setting axes_name: az");
            pointing_offsets.arcsec[citlali::config::pointing_axis_az()] =
                Eigen::Map<Eigen::VectorXd>(offset.data(), offset.size());
            has_az = true;
        }
        if (!has_alt && config.has(std::tuple{"pointing_offsets", 1, "value_arcsec"})) {
            auto offset = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets", 1, "value_arcsec"});
            if (offset.empty()) {
                logger->error("pointing_offsets alt has empty value_arcsec");
                std::exit(EXIT_FAILURE);
            }
            logger->warn("pointing_offsets alt parsed by positional index; consider setting axes_name: alt");
            pointing_offsets.arcsec[citlali::config::pointing_axis_alt()] =
                Eigen::Map<Eigen::VectorXd>(offset.data(), offset.size());
            has_alt = true;
        }
        if (!has_mjd && config.has(std::tuple{"pointing_offsets", 2, "modified_julian_date"})) {
            mjd_values = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets", 2, "modified_julian_date"});
            has_mjd = true;
        }

        if (!has_az || !has_alt) {
            logger->error("pointing_offsets must include both az and alt entries");
            std::exit(EXIT_FAILURE);
        }

        const auto n_az =
            pointing_offsets.arcsec[citlali::config::pointing_axis_az()].size();
        const auto n_alt =
            pointing_offsets.arcsec[citlali::config::pointing_axis_alt()].size();
        if (n_az != n_alt) {
            logger->error("pointing_offsets az/alt lengths differ (az={} alt={})", n_az, n_alt);
            std::exit(EXIT_FAILURE);
        }
        if (n_az != 1 && n_az != 2) {
            logger->error("pointing_offsets supports only one or two values per axis (got {})", n_az);
            std::exit(EXIT_FAILURE);
        }

        if (has_mjd) {
            if (mjd_values.size() == 2) {
                pointing_offsets.modified_julian_date =
                    Eigen::Map<Eigen::VectorXd>(mjd_values.data(), mjd_values.size());
            }
            else if (!mjd_values.empty() &&
                     std::all_of(mjd_values.begin(), mjd_values.end(), [](double v){ return v <= 0.0; })) {
                // non-positive sentinel means "not specified"
                pointing_offsets.modified_julian_date.setZero(2);
            }
            else if (mjd_values.size() == 1 && n_az == 1) {
                logger->warn(
                    "ignoring single pointing_offsets.modified_julian_date for single pointing offset; using a constant offset across the observation");
                pointing_offsets.modified_julian_date.setZero(2);
            }
            else {
                logger->error(
                    "pointing_offsets.modified_julian_date must contain 2 values when interpolating two offsets, or non-positive sentinels");
                std::exit(EXIT_FAILURE);
            }
        }

        citlali::pipeline::mirror_typed_pointing_offsets(
            pointing_offsets.arcsec, pointing_offsets.modified_julian_date,
            astrometry_config.pointing_offsets);
    }
    else {
        logger->error("pointing_offsets not found in config");
        std::exit(EXIT_FAILURE);
    }
}
