#pragma once

#include <citlali/core/config/config_error.h>

#include <array>

namespace citlali::config {

struct SourceFittingConfig {
    bool active = false;
    double bounding_box_arcsec = 0.0;
    double fitting_radius_arcsec = 0.0;
    bool fit_rotation_angle = false;
    std::array<double, 2> amp_limit_factors{0.0, 0.0};
    std::array<double, 2> fwhm_limit_factors{0.0, 0.0};
};

struct PostProcessingConfig {
    bool map_filtering_enabled = false;
    bool source_finding_enabled = false;
    SourceFittingConfig source_fitting;
};

inline void validate(const SourceFittingConfig &config, ValidationReport &report) {
    if (!config.active) {
        return;
    }
    check_minimum(config.bounding_box_arcsec, 0.0,
                  {"post_processing", "source_fitting", "bounding_box_arcsec"},
                  report);
}

inline void validate(const PostProcessingConfig &config, ValidationReport &report) {
    validate(config.source_fitting, report);
}

}  // namespace citlali::config
