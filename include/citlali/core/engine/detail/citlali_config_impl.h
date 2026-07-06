#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/citlali_config_read.h>
#include <citlali/core/engine/detail/mapmaking_activation_policy.h>
#include <citlali/core/engine/detail/source_protection_activation.h>

template<typename CT>
void Engine::get_citlali_config(CT &config) {
    citlali::engine_detail::read_interface_sync_offsets(
        config, interface_sync_offset, logger);

    auto &runtime_config = typed_config.runtime;
    auto &timestream_config = typed_config.timestream;
    auto &post_processing_config = typed_config.post_processing;
    auto &coadd_config = typed_config.coadd;

    runtime_config = get_runtime_config(config);
    if (!runtime_config.interp_over_gaps) {
        logger->error("runtime.interp_over_gaps=false is unsupported; set runtime.interp_over_gaps: true");
        std::exit(EXIT_FAILURE);
    }

    /* get timestream config */
    get_timestream_config(config);
    citlali::engine_detail::apply_source_protection_activation(
        runtime_config.reduction_type, rtcproc, ptcproc, timestream_config,
        logger);

    /* get mapmaking config */
    post_processing_config = citlali::config::PostProcessingConfig{};
    get_mapmaking_config(config);

    citlali::engine_detail::read_post_processing_activation_config(
        config, run_map_filter, run_source_finder,
        post_processing_config, missing_keys, invalid_keys);

    // map fitter options if in pointing or beammap mode or if map filtering or source finding are enabled
    citlali::engine_detail::read_source_fitting_config(
        config, runtime_config.reduction_type, map_fitter,
        omb.pixel_size_rad, ASEC_TO_RAD,
        post_processing_config,
        missing_keys, invalid_keys);

    /* get wiener filter config */
    if (post_processing_config.map_filtering.enabled) {
        // needs map fitter config
        get_map_filter_config(config);
    }

    // get source finder config options
    citlali::engine_detail::read_source_finding_config(
        config, omb, cmb, coadd_config, ASEC_TO_RAD,
        post_processing_config, missing_keys, invalid_keys);

    /* get pointing config */
    if (runtime_config.reduction_type ==
        citlali::config::ReductionType::pointing) {
        get_pointing_config(config);
    }

    /* get beammap config */
    if (runtime_config.reduction_type ==
        citlali::config::ReductionType::beammap) {
        // needs redu_type config
        get_beammap_config(config);
    }

    // disable map related keys if map-making is disabled
    citlali::engine_detail::disable_map_products_if_mapmaking_disabled(
        run_coadd, run_noise, run_map_filter, run_source_finder,
        typed_config, beammap_iter_max);

    citlali::engine_detail::validate_typed_config_mirrors(typed_config, logger);
}
