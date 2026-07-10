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

    bool run_tod = timestream_config.enabled;
    citlali::pipeline::read_timestream_enabled_config(
        config, run_tod, timestream_config, config_diag);
    if (!run_tod) {
        logger->error("timestream.enabled is false. This reduction requires TOD processing; set "
                      "low_level.timestream.enabled: true in your reduce config.");
        std::exit(EXIT_FAILURE);
    }
    std::string tod_type{
        std::string(citlali::config::to_string(timestream_config.type))};
    citlali::pipeline::read_timestream_type_config(
        config, tod_type, timestream_config, config_diag);
    citlali::pipeline::read_auxiliary_quadrature_channel_config(
        config, timestream_config, config_diag);

    bool run_tod_output = false;
    bool run_tod_output_rtc = false;
    bool run_tod_output_ptc = false;
    citlali::pipeline::read_tod_output_runtime_config(
        config, timestream_config, rtcproc, ptcproc, run_tod_output_rtc,
        run_tod_output_ptc, run_tod_output, config_diag);
    citlali::pipeline::read_timestream_output_metadata_config(
        config, timestream_config, diagnostics.write_evals, config_diag);

    citlali::pipeline::read_tod_output_selection_config(
        config, run_tod_output_rtc, run_tod_output_ptc,
        timestream_config.output, config_diag, logger);

    citlali::pipeline::read_timestream_chunking_config(
        config, timestream_config, telescope, config_diag);

    /* get raw time chunk config */
    get_rtc_config(config);

    /* get processed time chunk config */
    get_ptc_config(config);

    /* get shared reduction-learning config */
    get_learning_config(config);
}
