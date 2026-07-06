#pragma once

// Included by citlali_config_read.h inside namespace citlali::engine_detail {

template <class Config, class ObservationMapBuffer, class CoaddMapBuffer,
          class PostProcessingConfig, class KeyList>
void read_source_finding_config(
    Config &config, bool run_source_finder, ObservationMapBuffer &omb,
    CoaddMapBuffer &cmb, bool run_coadd, double arcsec_to_rad,
    PostProcessingConfig &typed_post_processing_config,
    KeyList &missing_keys, KeyList &invalid_keys) {
    if (!run_source_finder) {
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
        omb, cmb, run_coadd);
}
