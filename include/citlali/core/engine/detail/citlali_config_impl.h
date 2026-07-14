#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/citlali_config_read.h>
#include <citlali/core/pipeline/mapmaking_activation_policy.h>
#include <citlali/core/pipeline/post_processing_config_read.h>
#include <citlali/core/pipeline/source_protection_activation.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/source_finding_config_policy.h>
#include <citlali/core/pipeline/source_fitting_config_policy.h>

#include <cstdlib>

template<typename CT>
void Engine::get_citlali_config(CT &config) {
    citlali::pipeline::read_interface_sync_offsets(
        config, interface_sync.offsets, logger);

    auto &runtime_config = citlali::pipeline::runtime_config(*this);
    auto &timestream_config = citlali::pipeline::timestream_config(*this);
    auto &post_processing_config =
        citlali::pipeline::post_processing_config(*this);
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
    const auto source_protection_resolution =
        citlali::pipeline::resolve_source_protection(
            runtime_config.reduction_type, timestream_config);
    citlali::pipeline::apply_source_protection_activation(
        runtime_config.reduction_type, rtcproc, ptcproc, timestream_config,
        logger);
    auto &processed_plan =
        citlali::pipeline::processed_timestream_plan(*this);
    if (processed_plan.initialized) {
        processed_plan.realized.source_protection =
            source_protection_resolution;
        processed_plan.effective.processed_time_chunk.flagging
            .second_pass_local.source_protection.active =
            source_protection_resolution.processed_active;
    }

    /* get mapmaking config */
    post_processing_config = citlali::config::PostProcessingConfig{};
    citlali::pipeline::read_post_processing_request_config(
        config, post_processing_config, diagnostics);
    get_mapmaking_config(config);

    auto &post_processing_plan =
        citlali::pipeline::post_processing_plan(*this);
    post_processing_plan.reset_from_request(
        post_processing_config, runtime_config.reduction_type,
        citlali::config::mapmaking_active(
            citlali::pipeline::mapmaking_config(*this)),
        citlali::config::coadd_active(
            citlali::pipeline::coadd_config(*this)));

    if (citlali::config::source_fitting_active(
            post_processing_plan.effective)) {
        citlali::pipeline::adapt_source_fitting_config_one_way(
            post_processing_plan.effective.source_fitting,
            omb.pixel_size_rad, ASEC_TO_RAD, map_fitter);
    }

    /* get wiener filter config */
    if (citlali::config::map_filtering_active(
            post_processing_plan.effective)) {
        get_map_filter_config();
    }

    if (citlali::config::source_finding_active(
            post_processing_plan.effective)) {
        citlali::pipeline::adapt_source_finding_config_one_way(
            post_processing_plan.effective.source_finding, ASEC_TO_RAD,
            citlali::config::coadd_active(
                citlali::pipeline::coadd_config(*this)),
            omb, cmb);
    }

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

    citlali::pipeline::normalize_beammap_iterations_if_mapmaking_disabled(
        reduction_config);

    citlali::pipeline::validate_typed_config_mirrors(
        reduction_config, logger);
}
