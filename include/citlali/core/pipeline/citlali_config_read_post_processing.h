#pragma once

// Included by citlali_config_read.h inside namespace citlali::pipeline {

template <class KeyList>
void append_source_fitting_limit_key(KeyList &keys, const char *name,
                                     Eigen::Index index) {
    keys.push_back({"post_processing", "source_fitting", "gauss_model", name,
                    std::to_string(index)});
}

template <class Config, class LimitVector, class LimitArray, class KeyList>
void read_source_fitting_limit_factor(Config &config,
                                      LimitVector &map_fitter_limits,
                                      LimitArray &typed_limits,
                                      const char *name, Eigen::Index index,
                                      KeyList &missing_keys,
                                      KeyList &invalid_keys) {
    try {
        const auto sequence_key =
            std::tuple{"post_processing", "source_fitting", "gauss_model",
                       name};
        if (!config.has(sequence_key) ||
            static_cast<std::size_t>(index) >=
                config.get_node(sequence_key).size()) {
            append_source_fitting_limit_key(missing_keys, name, index);
            return;
        }

        const auto value = config.template get_typed<double>(
            std::tuple{"post_processing", "source_fitting", "gauss_model",
                       name, index});
        map_fitter_limits(index) = value;
        typed_limits[static_cast<std::size_t>(index)] = value;
    }
    catch (YAML::TypedBadConversion<double>) {
        append_source_fitting_limit_key(invalid_keys, name, index);
    }
    catch (YAML::InvalidNode) {
        append_source_fitting_limit_key(invalid_keys, name, index);
    }
}

template <class Config, class KeyList, class PostProcessingConfig>
void read_post_processing_activation_config(
    Config &config, bool &run_map_filter, bool &run_source_finder,
    PostProcessingConfig &typed_post_processing_config,
    KeyList &missing_keys, KeyList &invalid_keys) {
    citlali::pipeline::read_config_value_if_clean(
        config, std::tuple{"post_processing", "map_filtering", "enabled"},
        run_map_filter,
        [&typed_post_processing_config](bool enabled) {
            citlali::config::set_map_filtering_enabled(
                typed_post_processing_config, enabled);
        },
        missing_keys, invalid_keys);

    citlali::pipeline::read_config_value_if_clean(
        config, std::tuple{"post_processing", "source_finding", "enabled"},
        run_source_finder,
        [&typed_post_processing_config](bool enabled) {
            citlali::config::set_source_finding_enabled(
                typed_post_processing_config, enabled);
        },
        missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class PostProcessingConfig>
void read_post_processing_activation_config(
    Config &config, bool &run_map_filter, bool &run_source_finder,
    PostProcessingConfig &typed_post_processing_config,
    Diagnostics &diagnostics) {
    read_post_processing_activation_config(
        config, run_map_filter, run_source_finder,
        typed_post_processing_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
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

    citlali::config::set_source_fitting_active(
        typed_post_processing_config, true);

    citlali::pipeline::read_mirrored_config_value(
        config,
        std::tuple{"post_processing", "source_fitting", "bounding_box_arcsec"},
        map_fitter.bounding_box_pix,
        typed_post_processing_config.source_fitting.bounding_box_arcsec,
        missing_keys, invalid_keys, {}, {0});

    citlali::pipeline::read_mirrored_config_value(
        config,
        std::tuple{"post_processing", "source_fitting",
                   "fitting_radius_arcsec"},
        map_fitter.fitting_region_pix,
        typed_post_processing_config.source_fitting.fitting_radius_arcsec,
        missing_keys, invalid_keys);

    citlali::pipeline::read_mirrored_config_value(
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
        read_source_fitting_limit_factor(
            config, map_fitter.flux_limits,
            typed_post_processing_config.source_fitting.amp_limit_factors,
            "amp_limit_factors", i, missing_keys, invalid_keys);

        read_source_fitting_limit_factor(
            config, map_fitter.fwhm_limits,
            typed_post_processing_config.source_fitting.fwhm_limit_factors,
            "fwhm_limit_factors", i, missing_keys, invalid_keys);
    }

    citlali::pipeline::apply_positive_source_fit_limits(map_fitter);
}

template <class Config, class MapFitter, class PostProcessingConfig,
          class Diagnostics>
void read_source_fitting_config(
    Config &config, citlali::config::ReductionType reduction_type,
    MapFitter &map_fitter, double pixel_size_rad, double arcsec_to_rad,
    PostProcessingConfig &typed_post_processing_config,
    Diagnostics &diagnostics) {
    read_source_fitting_config(
        config, reduction_type, map_fitter, pixel_size_rad, arcsec_to_rad,
        typed_post_processing_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}
