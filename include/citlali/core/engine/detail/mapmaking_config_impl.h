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

    if (map_method=="jinc") {
        // maximum radius for jinc filter
        get_config_value(config, jinc_mm.r_max, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","jinc_filter","r_max"});
        // get jinc filter shape params
        for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
            auto jinc_shape_vec = config.template get_typed<std::vector<double>>(std::tuple{"mapmaking","jinc_filter","shape_params",arr_name});
            if (jinc_shape_vec.size() != 3) {
                invalid_keys.push_back({"mapmaking","jinc_filter","shape_params",arr_name});
                jinc_shape_vec.resize(3, 0.0);
            }
            jinc_mm.shape_params[arr_index] = Eigen::Map<Eigen::VectorXd>(jinc_shape_vec.data(),jinc_shape_vec.size());
        }
        // optional: sub-pixel sampling for jinc kernel
        if (config.template has_typed<int>(std::tuple{"mapmaking","jinc_filter","subpixel_n"})) {
            get_config_value(config, jinc_mm.subpixel_n, missing_keys, invalid_keys,
                             std::tuple{"mapmaking","jinc_filter","subpixel_n"},{},{1});
        }
        citlali::pipeline::mirror_jinc_mapmaker_config_to_fruit_loops(
            jinc_mm, ptcproc);

        if (jinc_mm.mode=="matrix") {
            // allocate jinc matrix
            jinc_mm.allocate_jinc_matrix(omb.pixel_size_rad);
        }
        else if (jinc_mm.mode=="splines") {
            // precompute jinc spline
            jinc_mm.calculate_jinc_splines();
        }
    }

    else if (map_method=="maximum_likelihood") {
        get_config_value(config, ml_mm.tolerance, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","maximum_likelihood","tolerance"});
        get_config_value(config, ml_mm.max_iterations, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","maximum_likelihood","max_iterations"});
    }

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
