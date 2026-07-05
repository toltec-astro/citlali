#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/mapmaking_config_read.h>
#include <citlali/core/pipeline/mapmaking_config_policy.h>

template<typename CT>
void Engine::get_mapmaking_config(CT &config) {
    logger->info("getting mapmaking config options");
    typed_mapmaking_config = citlali::config::MapmakingConfig{};
    typed_coadd_config = citlali::config::CoaddConfig{};
    typed_noise_config = citlali::config::NoiseConfig{};

    citlali::engine_detail::read_mapmaking_enabled_config(
        config, run_mapmaking, typed_mapmaking_config, missing_keys,
        invalid_keys);
    citlali::engine_detail::read_map_grouping_config(
        config, map_grouping, typed_mapmaking_config, missing_keys,
        invalid_keys);

    citlali::engine_detail::read_map_regime_config(
        config, map_regime, missing_keys, invalid_keys);

    citlali::pipeline::enforce_map_grouping_polarization_policy(
        rtcproc.run_polarization, redu_type, map_grouping, logger);

    citlali::pipeline::sync_map_grouping_to_timestream_processors(
        map_grouping, rtcproc, ptcproc);

    citlali::engine_detail::read_map_method_config(
        config, map_method, typed_mapmaking_config, missing_keys,
        invalid_keys);
    citlali::pipeline::configure_fruit_loop_interpolation_mode(
        ptcproc, map_method, logger);
    citlali::pipeline::log_fruit_loop_runtime_policy(ptcproc, logger);
    citlali::pipeline::reset_fruit_loop_jinc_kernel_config(ptcproc);

    citlali::engine_detail::read_map_pixel_axes_config(
        config, telescope.pixel_axes, typed_mapmaking_config, missing_keys,
        invalid_keys);
    citlali::pipeline::enforce_beammap_pixel_axes_policy(
        redu_type, telescope.pixel_axes, logger);

    citlali::engine_detail::read_output_map_block_config(
        config, omb, missing_keys, invalid_keys, telescope.pixel_axes,
        redu_type, RAD_TO_ASEC, typed_mapmaking_config,
        typed_post_processing_config, logger);

    citlali::engine_detail::read_coadd_enabled_config(
        config, run_coadd, typed_coadd_config, missing_keys, invalid_keys);
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

    citlali::engine_detail::read_noise_maps_enabled_config(
        config, run_noise, typed_noise_config, missing_keys, invalid_keys);
    if (run_noise) {
        citlali::engine_detail::read_noise_map_count_config(
            config, omb.n_noise, typed_noise_config, missing_keys,
            invalid_keys);
        citlali::engine_detail::read_noise_randomize_dets_config(
            config, omb.randomize_dets, typed_noise_config, missing_keys,
            invalid_keys);

        if (run_coadd) {
            citlali::pipeline::mirror_noise_map_settings_to_coadd(omb, cmb);
        }
    }
    // otherwise set number of noise maps to zero
    else {
        citlali::pipeline::disable_noise_map_settings(
            omb, cmb, typed_noise_config);
    }

    citlali::engine_detail::read_noise_write_realizations_config(
        config, write_noise_realizations, typed_noise_config, missing_keys,
        invalid_keys);
    citlali::engine_detail::read_noise_products_enabled_config(
        config, run_noise_products, run_noise, typed_noise_config,
        missing_keys, invalid_keys);
    citlali::engine_detail::read_noise_empirical_weights_config(
        config, apply_empirical_noise_weights, run_noise, typed_noise_config,
        missing_keys, invalid_keys);

    citlali::pipeline::set_mapmaker_polarization(
        rtcproc.run_polarization, naive_mm, jinc_mm);
}
