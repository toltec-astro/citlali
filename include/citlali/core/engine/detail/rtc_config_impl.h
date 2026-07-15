#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/config_parse_tracking.h>
#include <citlali/core/pipeline/polarimetry_config_read.h>
#include <citlali/core/pipeline/polarimetry_execution_plan.h>
#include <citlali/core/pipeline/raw_timestream_authority.h>
#include <citlali/core/pipeline/raw_timestream_config_read.h>
#include <citlali/core/pipeline/raw_tod_output_context.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_config_adapter_polarimetry.h>

#include <string>

template<typename CT>
void Engine::get_rtc_config(CT &config) {
    logger->info("getting rtc config options");
    auto &config_diag = citlali::pipeline::config_diagnostics(*this);
    citlali::config::RawTimeChunkConfig typed_request;
    citlali::pipeline::raw_timestream_plan(*this) = {};
    citlali::pipeline::read_raw_timestream_request_config(
        config, typed_request, config_diag);

    auto &polarimetry_request =
        citlali::pipeline::polarimetry_config(*this);
    citlali::pipeline::read_polarimetry_request_config(
        config, polarimetry_request, config_diag);
    auto &polarimetry_plan =
        citlali::pipeline::polarimetry_plan(*this);
    polarimetry_plan.reset_from_request(polarimetry_request);
    if (!polarimetry_plan.capability.request_accepted) {
        logger->error(
            "timestream.polarimetry.enabled=true is unavailable: {}. "
            "Exit condition: {}.",
            std::string{
                citlali::pipeline::polarimetry_capability_reason},
            std::string{
                citlali::pipeline::polarimetry_capability_exit_condition});
        citlali::pipeline::add_invalid_config_key(
            std::tuple{"timestream", "polarimetry", "enabled"},
            config_diag.invalid_key_paths());
    }

    auto &raw_config =
        citlali::pipeline::timestream_config(*this).raw_time_chunk;
    if (!config_diag.has_errors()) {
        citlali::pipeline::initialize_raw_timestream_authority(
            typed_request,
            citlali::pipeline::raw_timestream_plan(*this), raw_config,
            rtcproc, telescope.fsmp, ASEC_TO_RAD, FWHM_TO_STD);
    }

    citlali::pipeline::adapt_polarimetry_config(
        polarimetry_plan.effective, rtcproc, calib);

    citlali::pipeline::configure_raw_tod_output_context(
        telescope, rtcproc,
        citlali::pipeline::timestream_config(*this).output.raw_time_chunk);

}
