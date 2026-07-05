#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/timestream_config_mirror.h>

template<typename CT>
void Engine::get_ptc_config(CT &config) {
    logger->info("getting ptc config options");
    // get ptcproc config
    ptcproc.get_config(config, missing_keys, invalid_keys);
    citlali::pipeline::mirror_fruit_loops_config(
        typed_timestream_config.fruit_loops, ptcproc);
    citlali::pipeline::mirror_processed_clean_config(
        typed_timestream_config.processed_time_chunk.clean, ptcproc,
        toltec_io.array_name_map);
    auto &typed_weighting =
        typed_timestream_config.processed_time_chunk.weighting;
    auto &typed_flagging =
        typed_timestream_config.processed_time_chunk.flagging;
    citlali::pipeline::mirror_processed_weighting_config(
        typed_weighting, typed_flagging, ptcproc);
    const auto &weight_validation = ptcproc.weight_validation;
    citlali::pipeline::mirror_processed_weight_validation_config(
        typed_weighting.validation, weight_validation);

    const auto &weight_corr_penalty = ptcproc.weight_corr_penalty;
    citlali::pipeline::mirror_processed_weight_corr_penalty_config(
        typed_weighting.corr_penalty, weight_corr_penalty);

    auto &typed_second_pass =
        typed_timestream_config.processed_time_chunk.flagging.second_pass_local;
    citlali::pipeline::mirror_second_pass_local_config(
        typed_second_pass, ptcproc.second_pass_local);

    // copy tod output bool for eigenvalues
    ptcproc.run_tod_output = run_tod_output;
    ptcproc.write_evals = diagnostics.write_evals;
}
