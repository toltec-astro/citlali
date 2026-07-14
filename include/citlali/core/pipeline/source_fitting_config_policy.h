#pragma once

#include <citlali/core/config/post_processing_config.h>

namespace citlali::pipeline {

template <class MapFitter>
void adapt_source_fitting_config_one_way(
    const citlali::config::SourceFittingConfig &config,
    double pixel_size_rad, double arcsec_to_rad, MapFitter &map_fitter) {
    const double arcsec_to_pixels = arcsec_to_rad / pixel_size_rad;
    map_fitter.bounding_box_pix =
        config.bounding_box_arcsec * arcsec_to_pixels;
    map_fitter.fitting_region_pix =
        config.fitting_radius_arcsec * arcsec_to_pixels;
    map_fitter.fit_angle = config.fit_rotation_angle;

    map_fitter.flux_limits.resize(2);
    map_fitter.fwhm_limits.resize(2);
    for (int index = 0; index < 2; ++index) {
        map_fitter.flux_limits(index) = config.amp_limit_factors[index];
        map_fitter.fwhm_limits(index) = config.fwhm_limit_factors[index];
    }

    if (config.amp_limit_factors[0] > 0.0) {
        map_fitter.flux_low = config.amp_limit_factors[0];
    }
    if (config.amp_limit_factors[1] > 0.0) {
        map_fitter.flux_high = config.amp_limit_factors[1];
    }
    if (config.fwhm_limit_factors[0] > 0.0) {
        map_fitter.fwhm_low = config.fwhm_limit_factors[0];
    }
    if (config.fwhm_limit_factors[1] > 0.0) {
        map_fitter.fwhm_high = config.fwhm_limit_factors[1];
    }
}

}  // namespace citlali::pipeline
