#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/mapmaking_config_read.h>
#include <citlali/core/pipeline/mapmaking_config_policy.h>

template<typename CT>
void Engine::get_mapmaking_config(CT &config) {
    logger->info("getting mapmaking config options");
    auto &mapmaking_config = typed_config.mapmaking;
    auto &coadd_config = typed_config.coadd;
    auto &noise_config = typed_config.noise;
    auto &post_processing_config = typed_config.post_processing;
    mapmaking_config = citlali::config::MapmakingConfig{};
    coadd_config = citlali::config::CoaddConfig{};
    noise_config = citlali::config::NoiseConfig{};

    citlali::engine_detail::read_mapmaking_enabled_config(
        config, run_mapmaking, mapmaking_config, missing_keys,
        invalid_keys);
    citlali::engine_detail::read_map_grouping_config(
        config, map_grouping, mapmaking_config, missing_keys,
        invalid_keys);

    citlali::engine_detail::read_map_regime_config(
        config, map_regime, missing_keys, invalid_keys);

    citlali::pipeline::enforce_map_grouping_polarization_policy(
        rtcproc.run_polarization, typed_config.runtime.reduction_type,
        mapmaking_config.grouping, logger);

    citlali::pipeline::sync_map_grouping_to_timestream_processors(
        map_grouping, rtcproc, ptcproc);

    citlali::engine_detail::read_map_method_config(
        config, map_method, mapmaking_config, missing_keys,
        invalid_keys);
    citlali::pipeline::configure_fruit_loop_interpolation_mode(
        ptcproc, mapmaking_config.method, logger);
    citlali::pipeline::log_fruit_loop_runtime_policy(ptcproc, logger);
    citlali::pipeline::reset_fruit_loop_jinc_kernel_config(ptcproc);

    citlali::engine_detail::read_map_pixel_axes_config(
        config, telescope.pixel_axes, mapmaking_config, missing_keys,
        invalid_keys);
    citlali::pipeline::enforce_beammap_pixel_axes_policy(
        redu_type, telescope.pixel_axes, logger);

    citlali::engine_detail::read_output_map_block_config(
        config, omb, missing_keys, invalid_keys, telescope.pixel_axes,
        redu_type, RAD_TO_ASEC, mapmaking_config,
        post_processing_config, logger);

    citlali::engine_detail::read_coadd_enabled_config(
        config, run_coadd, coadd_config, missing_keys, invalid_keys);
    citlali::engine_detail::read_coadd_map_block_config(
        config, run_coadd, cmb, missing_keys, invalid_keys,
        telescope.pixel_axes, redu_type, logger);

    citlali::pipeline::apply_uncalibrated_map_units(
        rtcproc.run_calibrate, tod_type, omb, cmb);

    citlali::pipeline::sync_mapmaking_parallel_policy(
        parallel_policy, omb, cmb, jinc_mm);

    citlali::engine_detail::read_method_specific_mapmaker_config(
        config, map_method, jinc_mm, ml_mm, toltec_io.array_name_map,
        ptcproc, omb.pixel_size_rad, missing_keys, invalid_keys);

    citlali::engine_detail::read_noise_map_config(
        config, run_noise, run_coadd, omb, cmb, noise_config,
        missing_keys, invalid_keys);
    citlali::engine_detail::read_noise_product_config(
        config, run_noise, write_noise_realizations, run_noise_products,
        apply_empirical_noise_weights, noise_config, missing_keys,
        invalid_keys);

    citlali::pipeline::set_mapmaker_polarization(
        rtcproc.run_polarization, naive_mm, jinc_mm);
}
