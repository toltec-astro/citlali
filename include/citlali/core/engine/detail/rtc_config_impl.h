#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/config_parse_tracking.h>
#include <citlali/core/pipeline/raw_timestream_authority.h>
#include <citlali/core/pipeline/raw_timestream_config_read.h>
#include <citlali/core/pipeline/raw_timestream_shadow_parity.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_config_read.h>
#include <citlali/core/pipeline/timestream_config_adapter_polarimetry.h>
#include <citlali/core/pipeline/timestream_config_mirror.h>

#include <stdexcept>
#include <type_traits>

template<typename CT>
void Engine::get_rtc_config(CT &config) {
    logger->info("getting rtc config options");
    auto &config_diag = citlali::pipeline::config_diagnostics(*this);
    citlali::pipeline::ConfigDiagnosticsState legacy_diagnostics;
    citlali::config::RawTimeChunkConfig typed_request;
    citlali::pipeline::raw_timestream_plan(*this) = {};
    citlali::pipeline::read_raw_timestream_request_config(
        config, typed_request, config_diag);

    using RtcProc = std::remove_reference_t<decltype(rtcproc)>;
    RtcProc legacy_rtcproc;
    citlali::pipeline::read_processor_config(
        legacy_rtcproc, config, legacy_diagnostics);
    citlali::config::RawTimeChunkConfig legacy_oracle;
    citlali::pipeline::mirror_raw_despike_config(
        legacy_oracle.despike, legacy_rtcproc);

    citlali::pipeline::mirror_raw_flagging_config(
        legacy_oracle.flagging, legacy_rtcproc);

    citlali::pipeline::mirror_raw_kernel_config(
        legacy_oracle.kernel, legacy_rtcproc, RAD_TO_ASEC);

    citlali::pipeline::mirror_raw_altaz_destripe_config(
        legacy_oracle.altaz_destripe, legacy_rtcproc);

    citlali::pipeline::mirror_raw_line_audit_config(
        legacy_oracle.line_audit, legacy_rtcproc.line_audit);

    citlali::pipeline::mirror_raw_downsample_config(
        legacy_oracle.downsample, legacy_rtcproc);

    citlali::pipeline::mirror_raw_filter_config(
        legacy_oracle.filter, legacy_rtcproc);

    citlali::pipeline::mirror_raw_iir_filter_config(
        legacy_oracle.iir_filter, legacy_rtcproc);

    citlali::pipeline::mirror_raw_correction_flags(
        legacy_oracle, legacy_rtcproc);

    legacy_rtcproc.configure_filter_edge_guard(telescope.fsmp);
    citlali::pipeline::mirror_raw_filter_edge_guard_config(
        legacy_oracle.filter.edge_guard,
        legacy_rtcproc.filter_edge_guard);

    auto &raw_config =
        citlali::pipeline::timestream_config(*this).raw_time_chunk;
    if (!config_diag.has_errors() && !legacy_diagnostics.has_errors()) {
        citlali::pipeline::initialize_raw_timestream_authority(
            typed_request,
            citlali::pipeline::raw_timestream_plan(*this), raw_config,
            rtcproc, telescope.fsmp, ASEC_TO_RAD, FWHM_TO_STD);

        const auto parity =
            citlali::pipeline::compare_raw_timestream_authority(
                legacy_oracle, rtcproc, RAD_TO_ASEC);
        if (!parity.exact) {
            logger->error(
                "typed raw RTC authority differs from legacy oracle\nlegacy:\n{}\ntyped authority:\n{}",
                parity.legacy_oracle_snapshot,
                parity.typed_authority_snapshot);
            throw std::runtime_error(
                "typed raw RTC authority parity failure");
        }
    } else if (!config_diag.has_errors()) {
        logger->error(
            "legacy raw RTC oracle rejected configuration accepted by typed parsing");
        throw std::runtime_error(
            "legacy raw RTC oracle diagnostics diverged from typed parsing");
    }

    citlali::pipeline::adapt_legacy_polarimetry_runtime(
        legacy_rtcproc, rtcproc);

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
