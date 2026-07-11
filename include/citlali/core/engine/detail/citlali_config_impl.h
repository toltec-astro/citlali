#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/citlali_config_read.h>
#include <citlali/core/pipeline/mapmaking_activation_policy.h>
#include <citlali/core/pipeline/source_protection_activation.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_citlali_config(CT &config) {
    citlali::pipeline::read_interface_sync_offsets(
        config, interface_sync.offsets, logger);

    auto &runtime_config = citlali::pipeline::runtime_config(*this);
    auto &timestream_config = citlali::pipeline::timestream_config(*this);
    auto &post_processing_config =
        citlali::pipeline::post_processing_config(*this);
    auto &coadd_config = citlali::pipeline::coadd_config(*this);
    auto &reduction_config = citlali::pipeline::reduction_config(*this);
    auto &diagnostics = citlali::pipeline::config_diagnostics(*this);

    runtime_config = get_runtime_config(config);
    citlali::pipeline::runtime_config_provenance(*this) =
        citlali::config::make_runtime_config_provenance(
            runtime_config,
#if defined(CITLALI_USE_WIENER_FILTER_OMP)
            true
#else
            false
#endif
        );
    if (!runtime_config.interp_over_gaps) {
        logger->error("runtime.interp_over_gaps=false is unsupported; set runtime.interp_over_gaps: true");
        std::exit(EXIT_FAILURE);
    }

    /* get timestream config */
    get_timestream_config(config);
    citlali::pipeline::apply_source_protection_activation(
        runtime_config.reduction_type, rtcproc, ptcproc, timestream_config,
        logger);

    /* get mapmaking config */
    post_processing_config = citlali::config::PostProcessingConfig{};
    get_mapmaking_config(config);

    bool run_map_filter = post_processing_config.map_filtering.enabled;
    bool run_source_finder = post_processing_config.source_finding.enabled;
    citlali::pipeline::read_post_processing_activation_config(
        config, run_map_filter, run_source_finder,
        post_processing_config, diagnostics);

    // map fitter options if in pointing or beammap mode or if map filtering or source finding are enabled
    citlali::pipeline::read_source_fitting_config(
        config, runtime_config.reduction_type, map_fitter,
        omb.pixel_size_rad, ASEC_TO_RAD,
        post_processing_config, diagnostics);

    /* get wiener filter config */
    if (citlali::config::map_filtering_active(post_processing_config)) {
        // needs map fitter config
        get_map_filter_config(config);
    }

    // get source finder config options
    citlali::pipeline::read_source_finding_config(
        config, omb, cmb, coadd_config, ASEC_TO_RAD,
        post_processing_config, diagnostics);

    /* get pointing config */
    if (runtime_config.reduction_type ==
        citlali::config::ReductionType::pointing) {
        get_pointing_config(config);
    }

    /* get beammap config */
    if (runtime_config.reduction_type ==
        citlali::config::ReductionType::beammap) {
        // needs runtime reduction-type config
        get_beammap_config(config);
    }

    // disable map related keys if map-making is disabled
    citlali::pipeline::disable_map_products_if_mapmaking_disabled(
        reduction_config);

    citlali::pipeline::validate_typed_config_mirrors(
        reduction_config, logger);
}
