#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/polarimetry_compatibility_config.h>
#include <citlali/core/pipeline/raw_timestream_authority.h>
#include <citlali/core/pipeline/raw_timestream_config_read.h>
#include <citlali/core/pipeline/raw_tod_output_context.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_config_adapter_polarimetry.h>
#include <citlali/core/pipeline/timestream_config_mirror_polarimetry.h>
#include <citlali/core/pipeline/timestream_config_read.h>

#include <type_traits>

template<typename CT>
void Engine::get_rtc_config(CT &config) {
    logger->info("getting rtc config options");
    auto &config_diag = citlali::pipeline::config_diagnostics(*this);
    citlali::config::RawTimeChunkConfig typed_request;
    citlali::pipeline::raw_timestream_plan(*this) = {};
    citlali::pipeline::read_raw_timestream_request_config(
        config, typed_request, config_diag);

    using RtcProc = std::remove_reference_t<decltype(rtcproc)>;
    RtcProc legacy_polarimetry;
    citlali::pipeline::read_legacy_polarimetry_runtime_config(
        config, legacy_polarimetry, config_diag);

    auto &raw_config =
        citlali::pipeline::timestream_config(*this).raw_time_chunk;
    if (!config_diag.has_errors()) {
        citlali::pipeline::initialize_raw_timestream_authority(
            typed_request,
            citlali::pipeline::raw_timestream_plan(*this), raw_config,
            rtcproc, telescope.fsmp, ASEC_TO_RAD, FWHM_TO_STD);
    }

    citlali::pipeline::adapt_legacy_polarimetry_runtime(
        legacy_polarimetry, rtcproc);

    citlali::pipeline::configure_raw_tod_output_context(
        telescope, rtcproc,
        citlali::pipeline::timestream_config(*this).output.raw_time_chunk);

    // ignore hwpr?
    auto &polarimetry_config =
        citlali::pipeline::polarimetry_config(*this);
    citlali::pipeline::read_polarimetry_hwpr_policy_config(
        config, calib.ignore_hwpr, polarimetry_config, config_diag);
    citlali::pipeline::mirror_polarimetry_config(
        polarimetry_config, rtcproc);
}
