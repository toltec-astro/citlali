#pragma once

// Engine learning implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/learning_config_logging.h>
#include <citlali/core/pipeline/learning_config_adapter.h>
#include <citlali/core/pipeline/learning_config_read.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_learning_config(CT &config) {
    auto &learning_config =
        citlali::pipeline::timestream_config(*this).learning;
    auto &diagnostics = citlali::pipeline::config_diagnostics(*this);
    learning_config = citlali::config::TimestreamLearningConfig{};

    citlali::pipeline::read_learning_config(
        config, learning_config, diagnostics);

    citlali::pipeline::adapt_learning_config_one_way(
        learning_config, learning);
    const bool map_contribution_diag =
        citlali::pipeline::learning_map_contribution_diagnostics_enabled(
            learning.options);
    citlali::pipeline::set_learning_map_contribution_diagnostics(
        map_contribution_diag, omb, cmb);
    citlali::pipeline::log_reduction_learning_config(
        learning.options, logger);
}
