#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/timestream_config_read.h>
#include <citlali/core/pipeline/timestream_output_config_read.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_timestream_config(CT &config) {
    logger->info("getting timestream config options");
    auto &timestream_config = citlali::pipeline::timestream_config(*this);
    auto &config_diag = citlali::pipeline::config_diagnostics(*this);
    timestream_config = citlali::config::TimestreamConfig{};

    if (!citlali::pipeline::read_timestream_core_config(
            config, timestream_config, config_diag)) {
        logger->error("timestream.enabled is false. This reduction requires TOD processing; set "
                      "low_level.timestream.enabled: true in your reduce config.");
        config_diag.invalid_key_paths().push_back(
            {"timestream", "enabled"});
        return;
    }

    bool run_tod_output = false;
    bool run_tod_output_rtc = false;
    bool run_tod_output_ptc = false;
    citlali::pipeline::read_tod_output_runtime_config(
        config, timestream_config, run_tod_output_rtc, run_tod_output_ptc,
        run_tod_output, config_diag);
    citlali::pipeline::read_timestream_output_metadata_config(
        config, timestream_config, diagnostics.write_evals, config_diag);

    citlali::pipeline::read_tod_output_selection_config(
        config, run_tod_output_rtc, run_tod_output_ptc,
        timestream_config.output, config_diag, logger);

    citlali::pipeline::read_timestream_chunking_config(
        config, timestream_config, config_diag);

    /* get raw time chunk config */
    get_rtc_config(config);

    /* get shared reduction-learning config */
    get_learning_config(config);

    /* get processed time chunk config */
    get_ptc_config(config);
}
