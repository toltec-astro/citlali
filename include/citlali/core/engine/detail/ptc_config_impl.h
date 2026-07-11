#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/config_parse_tracking.h>
#include <citlali/core/pipeline/fruit_loops_config_read.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/processed_clean_config_read.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_config_mirror.h>
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
    citlali::pipeline::mirror_fruit_loops_config(
        fruit_loops_config, ptcproc);
    citlali::pipeline::read_fruit_loops_core_config(
        config, fruit_loops_config, config_diag);
    citlali::pipeline::apply_fruit_loops_config_to_processor(
        fruit_loops_config, ptcproc);
    citlali::pipeline::mirror_processed_clean_config(
        processed_config.clean, ptcproc,
        toltec_io.array_name_map);
    citlali::pipeline::read_processed_clean_core_config(
        config, processed_config.clean, config_diag);
    citlali::pipeline::apply_processed_clean_config_to_processor(
        processed_config.clean, toltec_io.array_name_map, ptcproc);
    auto &typed_weighting = processed_config.weighting;
    auto &typed_flagging = processed_config.flagging;
    citlali::pipeline::mirror_processed_weighting_config(
        typed_weighting, typed_flagging, ptcproc);
    const auto &weight_validation = ptcproc.weight_validation;
    citlali::pipeline::mirror_processed_weight_validation_config(
        typed_weighting.validation, weight_validation);

    const auto &weight_corr_penalty = ptcproc.weight_corr_penalty;
    citlali::pipeline::mirror_processed_weight_corr_penalty_config(
        typed_weighting.corr_penalty, weight_corr_penalty);

    auto &typed_second_pass = processed_config.flagging.second_pass_local;
    citlali::pipeline::mirror_second_pass_local_config(
        typed_second_pass, ptcproc.second_pass_local);

    // copy tod output bool for eigenvalues
    ptcproc.run_tod_output =
        citlali::pipeline::tod_output_enabled(*this);
    ptcproc.write_evals = diagnostics.write_evals;
}
