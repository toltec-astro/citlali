#pragma once

#include <citlali/core/config/calibration_config.h>
#include <citlali/core/config/config_error.h>

#include <algorithm>
#include <cmath>

namespace citlali::config {

inline void validate(const AstrometryPointingOffsetsConfig &config,
                     ValidationReport &report) {
    const ConfigPath path{"inputs", "cal_items", "astrometry",
                          "pointing_offsets"};
    if (!config.enabled) {
        return;
    }

    if (config.az_arcsec.empty()) {
        report.add_error({"inputs", "cal_items", "astrometry",
                          "pointing_offsets", "az", "value_arcsec"},
                         "must not be empty");
    }
    if (config.alt_arcsec.empty()) {
        report.add_error({"inputs", "cal_items", "astrometry",
                          "pointing_offsets", "alt", "value_arcsec"},
                         "must not be empty");
    }
    if (config.az_arcsec.size() != config.alt_arcsec.size()) {
        report.add_error(path, "az and alt value_arcsec lengths must match");
    }
    if (std::any_of(config.az_arcsec.begin(), config.az_arcsec.end(),
                    [](double value) { return !std::isfinite(value); })) {
        report.add_error({"inputs", "cal_items", "astrometry",
                          "pointing_offsets", "az", "value_arcsec"},
                         "must contain only finite values");
    }
    if (std::any_of(config.alt_arcsec.begin(), config.alt_arcsec.end(),
                    [](double value) { return !std::isfinite(value); })) {
        report.add_error({"inputs", "cal_items", "astrometry",
                          "pointing_offsets", "alt", "value_arcsec"},
                         "must contain only finite values");
    }
    const auto n_offsets = config.az_arcsec.size();
    if (n_offsets != 1 && n_offsets != 2) {
        report.add_error(path,
                         "must contain one or two values per pointing axis");
    }
    if (config.modified_julian_date.size() != 2) {
        report.add_error({"inputs", "cal_items", "astrometry",
                          "pointing_offsets", "modified_julian_date"},
                         "must contain two values after legacy normalization");
    }
    if (std::any_of(config.modified_julian_date.begin(),
                    config.modified_julian_date.end(),
                    [](double value) { return !std::isfinite(value); })) {
        report.add_error({"inputs", "cal_items", "astrometry",
                          "pointing_offsets", "modified_julian_date"},
                         "must contain only finite values");
    }
}

inline void validate(const AstrometryConfig &config, ValidationReport &report) {
    validate(config.pointing_offsets, report);
}

inline void validate(const CalibrationReferenceConfig &config,
                     ValidationReport &report) {
    if (!config.spectral_index_alpha.has_value()) {
        return;
    }
    const double alpha = *config.spectral_index_alpha;
    if (!std::isfinite(alpha)
        || (alpha != -1.0 && alpha != 0.0
            && alpha != 2.0 && alpha != 4.0)) {
        report.add_error(
            {"calibration", "reference_spectral_index_alpha"},
            "must be finite and exactly one of -1, 0, 2, or 4");
    }
}

inline void validate(const CalibrationConfig &config,
                     ValidationReport &report) {
    validate(config.reference, report);
}

inline void validate(const BeammapArrayFluxConfig &config,
                     ValidationReport &report) {
    if (config.array_name.empty()) {
        report.add_error({"beammap_source", "fluxes", "array_name"},
                         "must not be empty");
    }
    if (!std::isfinite(config.value_mjy) || config.value_mjy <= 0.0) {
        report.add_error({"beammap_source", "fluxes", "value_mJy"},
                         "must be positive and finite");
    }
    if (!std::isfinite(config.uncertainty_mjy) ||
        config.uncertainty_mjy < 0.0) {
        report.add_error({"beammap_source", "fluxes", "uncertainty_mJy"},
                         "must be greater than or equal to 0 and finite");
    }
}

inline void validate(const BeammapPhotometryConfig &config,
                     ValidationReport &report) {
    for (const auto &flux : config.fluxes) {
        validate(flux, report);
    }
}

}  // namespace citlali::config
