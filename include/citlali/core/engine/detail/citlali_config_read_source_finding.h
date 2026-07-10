#pragma once

// Included by citlali_config_read.h inside namespace citlali::engine_detail {

template <class Config, class ObservationMapBuffer, class CoaddMapBuffer,
          class CoaddConfig, class PostProcessingConfig, class KeyList>
void read_source_finding_config(
    Config &config, ObservationMapBuffer &omb, CoaddMapBuffer &cmb,
    const CoaddConfig &typed_coadd_config, double arcsec_to_rad,
    PostProcessingConfig &typed_post_processing_config, KeyList &missing_keys,
    KeyList &invalid_keys) {
    if (!citlali::config::source_finding_active(
            typed_post_processing_config)) {
        return;
    }

    read_mirrored_config_value(
        config, std::tuple{"post_processing", "source_finding", "source_sigma"},
        omb.source_sigma,
        typed_post_processing_config.source_finding.source_sigma, missing_keys,
        invalid_keys);

    read_mirrored_config_value(
        config,
        std::tuple{"post_processing", "source_finding", "source_window_arcsec"},
        omb.source_window_rad,
        typed_post_processing_config.source_finding.source_window_arcsec,
        missing_keys, invalid_keys);

    read_mirrored_config_value(
        config, std::tuple{"post_processing", "source_finding", "mode"},
        omb.source_finder_mode,
        typed_post_processing_config.source_finding.mode, missing_keys,
        invalid_keys);

    omb.source_window_rad =
        citlali::pipeline::source_window_arcsec_to_rad(
            omb.source_window_rad, arcsec_to_rad);

    citlali::pipeline::mirror_source_finding_config_to_coadd(
        omb, cmb, citlali::config::coadd_active(typed_coadd_config));
}

template <class Config, class ObservationMapBuffer, class CoaddMapBuffer,
          class CoaddConfig, class PostProcessingConfig, class Diagnostics>
void read_source_finding_config(
    Config &config, ObservationMapBuffer &omb, CoaddMapBuffer &cmb,
    const CoaddConfig &typed_coadd_config, double arcsec_to_rad,
    PostProcessingConfig &typed_post_processing_config,
    Diagnostics &diagnostics) {
    read_source_finding_config(
        config, omb, cmb, typed_coadd_config, arcsec_to_rad,
        typed_post_processing_config, diagnostics.missing_keys,
        diagnostics.invalid_keys);
}
