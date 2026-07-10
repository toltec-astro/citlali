#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/mapmaking_config_read.h>
#include <citlali/core/pipeline/mapmaking_config_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_mapmaking_config(CT &config) {
    logger->info("getting mapmaking config options");
    auto &runtime_config = citlali::pipeline::runtime_config(*this);
    auto &timestream_config = citlali::pipeline::timestream_config(*this);
    auto &mapmaking_config = citlali::pipeline::mapmaking_config(*this);
    auto &coadd_config = citlali::pipeline::coadd_config(*this);
    auto &noise_config = citlali::pipeline::noise_config(*this);
    auto &post_processing_config =
        citlali::pipeline::post_processing_config(*this);
    auto &diagnostics = config_diagnostics;
    mapmaking_config = citlali::config::MapmakingConfig{};
    coadd_config = citlali::config::CoaddConfig{};
    noise_config = citlali::config::NoiseConfig{};

    bool mapmaking_enabled = mapmaking_config.enabled;
    citlali::engine_detail::read_mapmaking_enabled_config(
        config, mapmaking_enabled, mapmaking_config, diagnostics);
    std::string map_grouping{
        std::string(citlali::config::to_string(mapmaking_config.grouping))};
    citlali::engine_detail::read_map_grouping_config(
        config, map_grouping, mapmaking_config, diagnostics);

    citlali::engine_detail::read_map_regime_config(
        config, mapmaking_config, diagnostics);

    citlali::pipeline::enforce_map_grouping_polarization_policy(
        rtcproc.run_polarization, runtime_config.reduction_type,
        mapmaking_config.grouping, logger);

    citlali::pipeline::sync_map_grouping_to_timestream_processors(
        mapmaking_config.grouping, rtcproc, ptcproc);

    std::string map_method{
        std::string(citlali::config::to_string(mapmaking_config.method))};
    citlali::engine_detail::read_map_method_config(
        config, map_method, mapmaking_config, diagnostics);
    citlali::pipeline::configure_fruit_loop_interpolation_mode(
        ptcproc, mapmaking_config.method, logger);
    citlali::pipeline::log_fruit_loop_runtime_policy(ptcproc, logger);
    citlali::pipeline::reset_fruit_loop_jinc_kernel_config(ptcproc);

    citlali::engine_detail::read_map_pixel_axes_config(
        config, telescope.pixel_axes, mapmaking_config, diagnostics);
    citlali::pipeline::enforce_beammap_pixel_axes_policy(
        runtime_config.reduction_type,
        mapmaking_config.pixel_axes_frame, logger);

    citlali::engine_detail::read_output_map_block_config(
        config, omb, diagnostics,
        mapmaking_config.pixel_axes_frame, runtime_config.reduction_type,
        RAD_TO_ASEC, mapmaking_config,
        post_processing_config, logger);

    bool coadd_enabled = coadd_config.enabled;
    citlali::engine_detail::read_coadd_enabled_config(
        config, coadd_enabled, coadd_config, diagnostics);
    citlali::engine_detail::read_coadd_map_block_config(
        config, coadd_config, cmb, diagnostics,
        mapmaking_config.pixel_axes_frame, runtime_config.reduction_type,
        logger);

    citlali::pipeline::apply_uncalibrated_map_units(
        rtcproc.run_calibrate, timestream_config.type, omb, cmb);

    citlali::pipeline::sync_mapmaking_parallel_policy(
        citlali::pipeline::runtime_parallel_policy_name(*this),
        omb, cmb, jinc_mm);

    citlali::engine_detail::read_method_specific_mapmaker_config(
        config, mapmaking_config.method, jinc_mm, ml_mm,
        toltec_io.array_name_map, ptcproc, omb.pixel_size_rad, diagnostics);

    bool noise_maps_enabled = noise_config.enabled;
    citlali::engine_detail::read_noise_map_config(
        config, noise_maps_enabled, coadd_config, omb, cmb, noise_config,
        diagnostics);
    bool write_noise_realizations = noise_config.write_realizations;
    bool run_noise_products = noise_config.products_enabled;
    bool apply_empirical_noise_weights =
        noise_config.apply_empirical_weights;
    citlali::engine_detail::read_noise_product_config(
        config, write_noise_realizations, run_noise_products,
        apply_empirical_noise_weights, noise_config, diagnostics);

    citlali::pipeline::set_mapmaker_polarization(
        rtcproc.run_polarization, naive_mm, jinc_mm);
}
