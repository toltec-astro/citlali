#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/mapmaking_config_read.h>
#include <citlali/core/pipeline/coadd_config_read.h>
#include <citlali/core/pipeline/mapmaking_config_policy.h>
#include <citlali/core/pipeline/noise_config_adapter.h>
#include <citlali/core/pipeline/noise_config_read.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_mapmaking_config(CT &config) {
    logger->info("getting mapmaking config options");
    auto &runtime_config = citlali::pipeline::runtime_config(*this);
    auto &timestream_config = citlali::pipeline::timestream_config(*this);
    auto &mapmaking_plan = citlali::pipeline::mapmaking_plan(*this);
    mapmaking_plan = {};
    auto &coadd_plan = citlali::pipeline::coadd_plan(*this);
    coadd_plan = {};
    auto &noise_plan = citlali::pipeline::noise_plan(*this);
    noise_plan = {};
    auto &mapmaking_config =
        citlali::pipeline::reduction_config(*this).mapmaking;
    auto &coadd_config = citlali::pipeline::coadd_config(*this);
    auto &noise_config = citlali::pipeline::noise_config(*this);
    auto &post_processing_config =
        citlali::pipeline::post_processing_config(*this);
    auto &diagnostics = citlali::pipeline::config_diagnostics(*this);
    mapmaking_config = citlali::config::MapmakingConfig{};
    coadd_config = citlali::config::CoaddConfig{};
    noise_config = citlali::config::NoiseConfig{};

    bool mapmaking_enabled = mapmaking_config.enabled;
    citlali::pipeline::read_mapmaking_enabled_config(
        config, mapmaking_enabled, mapmaking_config, diagnostics);
    std::string map_grouping{
        std::string(citlali::config::to_string(mapmaking_config.grouping))};
    citlali::pipeline::read_map_grouping_config(
        config, map_grouping, mapmaking_config, diagnostics);

    citlali::pipeline::read_map_regime_config(
        config, mapmaking_config, diagnostics);

    citlali::pipeline::enforce_map_grouping_polarization_policy(
        rtcproc.run_polarization, runtime_config.reduction_type,
        mapmaking_config.grouping, logger);

    std::string map_method{
        std::string(citlali::config::to_string(mapmaking_config.method))};
    citlali::pipeline::read_map_method_config(
        config, map_method, mapmaking_config, diagnostics);
    citlali::pipeline::configure_fruit_loop_interpolation_mode(
        *this, mapmaking_config.method, logger);
    citlali::pipeline::log_fruit_loop_runtime_policy(*this, logger);
    citlali::pipeline::reset_fruit_loop_jinc_kernel_config(ptcproc);

    citlali::pipeline::read_map_pixel_axes_config(
        config, telescope.pixel_axes, mapmaking_config, diagnostics);
    citlali::pipeline::enforce_beammap_pixel_axes_policy(
        runtime_config.reduction_type,
        mapmaking_config.pixel_axes_frame, logger);

    logger->info("getting omb config options");
    const auto output_missing_before =
        diagnostics.missing_key_paths().size();
    const auto output_invalid_before =
        diagnostics.invalid_key_paths().size();
    citlali::pipeline::read_mapmaking_output_request_config(
        config, mapmaking_config, diagnostics);
    const bool output_config_clean =
        citlali::pipeline::config_parse_clean(
            diagnostics.missing_key_paths(),
            diagnostics.invalid_key_paths(), output_missing_before,
            output_invalid_before);
    if (output_config_clean) {
        citlali::pipeline::adapt_mapmaking_output_config_one_way(
            mapmaking_config, post_processing_config,
            mapmaking_config.pixel_axes_frame,
            runtime_config.reduction_type, ASEC_TO_RAD, RAD_TO_DEG,
            RAD_TO_ASEC, omb);
    }

    citlali::pipeline::read_coadd_request_config(
        config, coadd_config, diagnostics);
    if (output_config_clean &&
        citlali::config::coadd_active(coadd_config)) {
        logger->info("getting cmb config options");
        citlali::pipeline::adapt_mapmaking_output_config_one_way(
            mapmaking_config, post_processing_config,
            mapmaking_config.pixel_axes_frame,
            runtime_config.reduction_type, ASEC_TO_RAD, RAD_TO_DEG,
            RAD_TO_ASEC, cmb);
    }

    const bool flux_calibration_enabled =
        citlali::pipeline::raw_flux_calibration_enabled(*this);
    citlali::pipeline::apply_uncalibrated_map_units(
        flux_calibration_enabled, timestream_config.type, omb, cmb);

    citlali::pipeline::sync_mapmaking_parallel_policy(
        citlali::pipeline::runtime_parallel_policy_name(*this),
        omb, cmb, jinc_mm);

    const auto method_missing_before =
        diagnostics.missing_key_paths().size();
    const auto method_invalid_before =
        diagnostics.invalid_key_paths().size();
    citlali::pipeline::read_mapmaking_method_request_config(
        config, mapmaking_config.method, toltec_io.array_name_map,
        mapmaking_config, diagnostics);
    if (citlali::pipeline::config_parse_clean(
            diagnostics.missing_key_paths(),
            diagnostics.invalid_key_paths(), method_missing_before,
            method_invalid_before)) {
        if (citlali::config::is_jinc_map_method(
                mapmaking_config.method)) {
            citlali::pipeline::adapt_jinc_filter_config_one_way(
                mapmaking_config.jinc_filter, toltec_io.array_name_map,
                jinc_mm);
            citlali::pipeline::finalize_jinc_filter_config(
                jinc_mm, ptcproc, omb.pixel_size_rad);
        } else if (citlali::config::is_maximum_likelihood_map_method(
                       mapmaking_config.method)) {
            citlali::pipeline::adapt_maximum_likelihood_config_one_way(
                mapmaking_config.maximum_likelihood, ml_mm);
        }
    }

    citlali::pipeline::read_noise_request_config(
        config, noise_config, diagnostics);

    citlali::pipeline::set_mapmaker_polarization(
        rtcproc.run_polarization, naive_mm, jinc_mm);

    mapmaking_plan.reset_from_request(
        mapmaking_config, runtime_config.reduction_type,
        flux_calibration_enabled, timestream_config.type);
    coadd_plan.reset_from_request(
        coadd_config, mapmaking_plan.effective.enabled);
    noise_plan.reset_from_request(
        noise_config, mapmaking_plan.effective.enabled);
    citlali::pipeline::adapt_noise_config_one_way(
        noise_plan.effective, coadd_plan.effective.enabled, omb, cmb);
    citlali::pipeline::sync_map_grouping_to_timestream_processors(
        mapmaking_plan.effective.grouping, rtcproc, ptcproc);
}
