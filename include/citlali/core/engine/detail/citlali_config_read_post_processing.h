#pragma once

// Included by citlali_config_read.h inside namespace citlali::engine_detail {

template <class Config, class KeyList, class PostProcessingConfig>
void read_post_processing_activation_config(
    Config &config, bool &run_map_filter, bool &run_source_finder,
    PostProcessingConfig &typed_post_processing_config,
    KeyList &missing_keys, KeyList &invalid_keys) {
    read_config_value_if_clean(
        config, std::tuple{"post_processing", "map_filtering", "enabled"},
        run_map_filter,
        [&typed_post_processing_config](bool enabled) {
            citlali::config::set_map_filtering_enabled(
                typed_post_processing_config, enabled);
        },
        missing_keys, invalid_keys);

    read_config_value_if_clean(
        config, std::tuple{"post_processing", "source_finding", "enabled"},
        run_source_finder,
        [&typed_post_processing_config](bool enabled) {
            citlali::config::set_source_finding_enabled(
                typed_post_processing_config, enabled);
        },
        missing_keys, invalid_keys);
}

template <class Config, class MapFitter, class PostProcessingConfig,
          class KeyList>
void read_source_fitting_config(
    Config &config, citlali::config::ReductionType reduction_type,
    MapFitter &map_fitter, double pixel_size_rad, double arcsec_to_rad,
    PostProcessingConfig &typed_post_processing_config, KeyList &missing_keys,
    KeyList &invalid_keys) {
    if (!citlali::pipeline::source_fitting_config_needed(
            reduction_type, typed_post_processing_config)) {
        return;
    }

    typed_post_processing_config.source_fitting.active = true;

    read_mirrored_config_value(
        config,
        std::tuple{"post_processing", "source_fitting", "bounding_box_arcsec"},
        map_fitter.bounding_box_pix,
        typed_post_processing_config.source_fitting.bounding_box_arcsec,
        missing_keys, invalid_keys, {}, {0});

    read_mirrored_config_value(
        config,
        std::tuple{"post_processing", "source_fitting",
                   "fitting_radius_arcsec"},
        map_fitter.fitting_region_pix,
        typed_post_processing_config.source_fitting.fitting_radius_arcsec,
        missing_keys, invalid_keys);

    read_mirrored_config_value(
        config,
        std::tuple{"post_processing", "source_fitting", "gauss_model",
                   "fit_rotation_angle"},
        map_fitter.fit_angle,
        typed_post_processing_config.source_fitting.fit_rotation_angle,
        missing_keys, invalid_keys);

    map_fitter.bounding_box_pix =
        citlali::pipeline::source_fitting_arcsec_to_pixels(
            map_fitter.bounding_box_pix, arcsec_to_rad, pixel_size_rad);
    map_fitter.fitting_region_pix =
        citlali::pipeline::source_fitting_arcsec_to_pixels(
            map_fitter.fitting_region_pix, arcsec_to_rad, pixel_size_rad);

    map_fitter.flux_limits.resize(2);
    map_fitter.fwhm_limits.resize(2);
    for (Eigen::Index i = 0; i < map_fitter.flux_limits.size(); ++i) {
        map_fitter.flux_limits(i) =
            config.template get_typed<double>(
                std::tuple{"post_processing", "source_fitting", "gauss_model",
                           "amp_limit_factors", i});
        typed_post_processing_config.source_fitting
            .amp_limit_factors[static_cast<std::size_t>(i)] =
            map_fitter.flux_limits(i);

        map_fitter.fwhm_limits(i) =
            config.template get_typed<double>(
                std::tuple{"post_processing", "source_fitting", "gauss_model",
                           "fwhm_limit_factors", i});
        typed_post_processing_config.source_fitting
            .fwhm_limit_factors[static_cast<std::size_t>(i)] =
            map_fitter.fwhm_limits(i);
    }

    citlali::pipeline::apply_positive_source_fit_limits(map_fitter);
}
