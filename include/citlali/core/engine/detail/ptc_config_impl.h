#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/config_parse_tracking.h>
#include <citlali/core/pipeline/fruit_loops_config_read.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/processed_timestream_compatibility_seed.h>
#include <citlali/core/pipeline/processed_clean_config_read.h>
#include <citlali/core/pipeline/processed_weighting_config_read.h>
#include <citlali/core/pipeline/processed_weighting_resolution.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/second_pass_local_config_read.h>
#include <citlali/core/pipeline/timestream_config_adapter_processed.h>

template<typename CT>
void Engine::get_ptc_config(CT &config) {
    logger->info("getting ptc config options");
    auto &config_diag = citlali::pipeline::config_diagnostics(*this);
    // get ptcproc config
    citlali::pipeline::read_processor_config(
        ptcproc, config, config_diag);
    auto &timestream_config = citlali::pipeline::timestream_config(*this);
    auto &fruit_loops_config = timestream_config.fruit_loops;
    auto &processed_config = timestream_config.processed_time_chunk;
    citlali::pipeline::seed_processed_timestream_config_from_legacy(
        timestream_config, ptcproc, toltec_io.array_name_map);
    citlali::pipeline::read_fruit_loops_core_config(
        config, fruit_loops_config, config_diag);
    citlali::pipeline::read_processed_clean_core_config(
        config, processed_config.clean, config_diag,
        toltec_io.array_name_map, logger);
    auto &typed_weighting = processed_config.weighting;
    auto &typed_flagging = processed_config.flagging;
    citlali::pipeline::read_processed_weighting_core_config(
        config, typed_weighting, typed_flagging, processed_config.clean,
        config_diag);
    citlali::pipeline::read_processed_weight_validation_config(
        config, typed_weighting.validation, config_diag);
    citlali::pipeline::read_processed_weighting_expert_config(
        config, typed_weighting, config_diag);

    auto &typed_second_pass = processed_config.flagging.second_pass_local;
    citlali::pipeline::read_second_pass_local_config(
        config, typed_second_pass, config_diag);

    auto &plan = citlali::pipeline::processed_timestream_plan(*this);
    citlali::pipeline::reset_processed_timestream_execution_plan(
        plan, timestream_config);
    auto &effective_fruit_loops = plan.effective.fruit_loops;
    auto &effective_processed = plan.effective.processed_time_chunk;

    plan.effective_resolutions.cleaner_mode =
        citlali::pipeline::resolve_processed_cleaner_mode(
            plan.requested.processed_time_chunk.clean);
    effective_processed.clean.active =
        plan.effective_resolutions.cleaner_mode->effective;

    const auto source_mask_key = std::tuple{
        "timestream", "processed_time_chunk", "weighting",
        "source_mask_radius_arcsec"};
    std::optional<double> requested_source_mask;
    if (config.template has_typed<double>(source_mask_key)) {
        requested_source_mask =
            effective_processed.weighting.source_mask_radius_arcsec;
    }
    plan.effective_resolutions.weighting_source_mask =
        citlali::pipeline::resolve_processed_weighting_source_mask(
            requested_source_mask,
            effective_processed.clean.mask_radius_arcsec);
    effective_processed.weighting.source_mask_radius_arcsec =
        plan.effective_resolutions.weighting_source_mask->effective;

    plan.effective_resolutions.weighting_dependencies =
        citlali::pipeline::resolve_processed_weighting(
            effective_processed.weighting,
            effective_processed.flagging);
    citlali::pipeline::log_processed_weighting_resolution(
        *plan.effective_resolutions.weighting_dependencies, logger);
    effective_processed.weighting =
        plan.effective_resolutions.weighting_dependencies->effective;

    citlali::pipeline::apply_fruit_loops_config_to_processor(
        effective_fruit_loops, ptcproc);
    citlali::pipeline::apply_processed_clean_config_to_processor(
        effective_processed.clean, toltec_io.array_name_map, ptcproc);
    citlali::pipeline::apply_processed_weighting_config_to_processor(
        effective_processed.weighting, effective_processed.flagging,
        ptcproc);
    citlali::pipeline::apply_second_pass_local_config_to_processor(
        effective_processed.flagging.second_pass_local, ptcproc);

    // Keep the transitional root typed config effective for compatibility
    // consumers that have not moved to the plan accessors yet.
    fruit_loops_config = effective_fruit_loops;
    processed_config = effective_processed;

    // copy tod output bool for eigenvalues
    ptcproc.run_tod_output =
        citlali::pipeline::tod_output_enabled(*this);
    ptcproc.write_evals = diagnostics.write_evals;
}
