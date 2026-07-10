#pragma once

// Included by mapmaking_config_read.h inside namespace citlali::engine_detail {

template <class Config, class OutputMapBlock, class MissingKeys,
          class InvalidKeys, class PixelAxes, class MapmakingConfig,
          class PostProcessingConfig, class Logger>
void read_output_map_block_config(
    Config &config, OutputMapBlock &omb, MissingKeys &missing_keys,
    InvalidKeys &invalid_keys, const PixelAxes &pixel_axes,
    citlali::config::ReductionType reduction_type, double rad_to_arcsec,
    MapmakingConfig &typed_mapmaking_config,
    PostProcessingConfig &typed_post_processing_config,
    const Logger &logger) {
    logger->info("getting omb config options");
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    omb.get_config(
        config, missing_keys, invalid_keys, pixel_axes, reduction_type);
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        citlali::pipeline::mirror_output_map_block_config(
            typed_mapmaking_config, omb, rad_to_arcsec,
            typed_post_processing_config);
    }
}

template <class Config, class OutputMapBlock, class Diagnostics,
          class PixelAxes, class MapmakingConfig,
          class PostProcessingConfig, class Logger>
void read_output_map_block_config(
    Config &config, OutputMapBlock &omb, Diagnostics &diagnostics,
    const PixelAxes &pixel_axes,
    citlali::config::ReductionType reduction_type, double rad_to_arcsec,
    MapmakingConfig &typed_mapmaking_config,
    PostProcessingConfig &typed_post_processing_config,
    const Logger &logger) {
    read_output_map_block_config(
        config, omb, diagnostics.missing_keys, diagnostics.invalid_keys,
        pixel_axes, reduction_type, rad_to_arcsec, typed_mapmaking_config,
        typed_post_processing_config, logger);
}

template <class Config, class CoaddMapBlock, class CoaddConfig,
          class MissingKeys, class InvalidKeys, class PixelAxes,
          class Logger>
void read_coadd_map_block_config(
    Config &config, const CoaddConfig &typed_coadd_config, CoaddMapBlock &cmb,
    MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    const PixelAxes &pixel_axes, citlali::config::ReductionType reduction_type,
    const Logger &logger) {
    if (!citlali::config::coadd_active(typed_coadd_config)) {
        return;
    }
    logger->info("getting cmb config options");
    cmb.get_config(
        config, missing_keys, invalid_keys, pixel_axes, reduction_type);
}

template <class Config, class CoaddMapBlock, class CoaddConfig,
          class Diagnostics, class PixelAxes, class Logger>
void read_coadd_map_block_config(
    Config &config, const CoaddConfig &typed_coadd_config, CoaddMapBlock &cmb,
    Diagnostics &diagnostics, const PixelAxes &pixel_axes,
    citlali::config::ReductionType reduction_type, const Logger &logger) {
    read_coadd_map_block_config(
        config, typed_coadd_config, cmb, diagnostics.missing_keys,
        diagnostics.invalid_keys, pixel_axes, reduction_type, logger);
}
