#pragma once

// Included by citlali_config_read.h inside namespace citlali::engine_detail {

template <class Config, class KeyList, class PostProcessingConfig>
void read_post_processing_activation_config(
    Config &config, bool &run_map_filter, bool &run_source_finder,
    PostProcessingConfig &typed_post_processing_config,
    KeyList &missing_keys, KeyList &invalid_keys) {
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(config, run_map_filter, missing_keys, invalid_keys,
                           std::tuple{"post_processing", "map_filtering", "enabled"});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.map_filtering_enabled = run_map_filter;
            typed_post_processing_config.map_filtering.enabled = run_map_filter;
        }
    }

    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(config, run_source_finder, missing_keys, invalid_keys,
                           std::tuple{"post_processing", "source_finding", "enabled"});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.source_finding_enabled = run_source_finder;
            typed_post_processing_config.source_finding.enabled = run_source_finder;
        }
    }
}

template <class Config, class MapFitter, class PostProcessingConfig,
          class KeyList>
void read_source_fitting_config(
    Config &config, const std::string &reduction_type, bool run_map_filter,
    bool run_source_finder, MapFitter &map_fitter, double pixel_size_rad,
    double arcsec_to_rad, PostProcessingConfig &typed_post_processing_config,
    KeyList &missing_keys, KeyList &invalid_keys) {
    if (!citlali::pipeline::source_fitting_config_needed(
            reduction_type, run_map_filter, run_source_finder)) {
        return;
    }

    typed_post_processing_config.source_fitting.active = true;

    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(
            config, map_fitter.bounding_box_pix, missing_keys, invalid_keys,
            std::tuple{"post_processing", "source_fitting", "bounding_box_arcsec"},
            {}, {0});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.source_fitting.bounding_box_arcsec =
                map_fitter.bounding_box_pix;
        }
    }

    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(
            config, map_fitter.fitting_region_pix, missing_keys, invalid_keys,
            std::tuple{"post_processing", "source_fitting", "fitting_radius_arcsec"});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.source_fitting.fitting_radius_arcsec =
                map_fitter.fitting_region_pix;
        }
    }

    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(
            config, map_fitter.fit_angle, missing_keys, invalid_keys,
            std::tuple{"post_processing", "source_fitting", "gauss_model",
                       "fit_rotation_angle"});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.source_fitting.fit_rotation_angle =
                map_fitter.fit_angle;
        }
    }

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

