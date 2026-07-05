#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/config_parse_tracking.h>
#include <citlali/core/engine/detail/mapmaking_config_read.h>
#include <citlali/core/pipeline/mapmaking_config_policy.h>

template<typename CT>
void Engine::get_mapmaking_config(CT &config) {
    logger->info("getting mapmaking config options");
    typed_mapmaking_config = citlali::config::MapmakingConfig{};
    typed_coadd_config = citlali::config::CoaddConfig{};
    typed_noise_config = citlali::config::NoiseConfig{};

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return citlali::engine_detail::config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before);
    };

    citlali::engine_detail::read_mapmaking_enabled_config(
        config, run_mapmaking, typed_mapmaking_config, missing_keys,
        invalid_keys);
    citlali::engine_detail::read_map_grouping_config(
        config, map_grouping, typed_mapmaking_config, missing_keys,
        invalid_keys);

    // optional expected sky regime for interpreting map diagnostics
    map_regime = "unknown";
    if (config.template has_typed<std::string>(std::tuple{"source", "map_regime"})) {
        map_regime = config.template get_typed<std::string>(std::tuple{"source", "map_regime"});
        check_allowed(map_regime, missing_keys, invalid_keys,
                      citlali::pipeline::allowed_map_regimes(),
                      std::tuple{"source", "map_regime"});
    }

    // polarization is disabled for detector grouping
    if (rtcproc.run_polarization && ((redu_type=="beammap" && map_grouping=="auto") || map_grouping=="detector")) {
        logger->error("Detector grouping reductions do not currently support polarimetry mode");
        std::exit(EXIT_FAILURE);
    }

    // set rtcproc map_grouping
    rtcproc.kernel.map_grouping = map_grouping;
    ptcproc.active_map_grouping = map_grouping;

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
    if (redu_type == "beammap" && telescope.pixel_axes != "altaz") {
        logger->error(
            "beammap reductions require mapmaking.pixel_axes='altaz'; got '{}'",
            telescope.pixel_axes);
        std::exit(EXIT_FAILURE);
    }

    // get config for omb
    logger->info("getting omb config options");
    const auto omb_missing_before = missing_keys.size();
    const auto omb_invalid_before = invalid_keys.size();
    omb.get_config(config, missing_keys, invalid_keys, telescope.pixel_axes, redu_type);
    if (parsed_cleanly(omb_missing_before, omb_invalid_before)) {
        typed_mapmaking_config.coverage_cut = omb.cov_cut;
        typed_mapmaking_config.pixel_size_arcsec = omb.pixel_size_rad * RAD_TO_ASEC;
        typed_mapmaking_config.unit = omb.sig_unit;
        if (omb.wcs.naxis.size() >= 2) {
            typed_mapmaking_config.x_size_pix = static_cast<int>(omb.wcs.naxis[0]);
            typed_mapmaking_config.y_size_pix = static_cast<int>(omb.wcs.naxis[1]);
        }
        if (omb.wcs.crpix.size() >= 2) {
            typed_mapmaking_config.crpix1 = omb.wcs.crpix[0];
            typed_mapmaking_config.crpix2 = omb.wcs.crpix[1];
        }
        if (omb.crval_config.size() >= 2) {
            typed_mapmaking_config.crval1_j2000 = omb.crval_config[0];
            typed_mapmaking_config.crval2_j2000 = omb.crval_config[1];
        }
        typed_post_processing_config.map_histogram_n_bins = omb.hist_n_bins;
    }

    // run coaddition?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_coadd, missing_keys, invalid_keys,
                         std::tuple{"coadd","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_coadd_config.enabled = run_coadd;
        }
    }
    // re-run to get config for cmb
    if (run_coadd) {
        logger->info("getting cmb config options");
        cmb.get_config(config, missing_keys, invalid_keys, telescope.pixel_axes, redu_type);
    }

    // if flux calibration is not enabled, use tod type units (xs, rs, is, or qs)
    if (!rtcproc.run_calibrate) {
        omb.sig_unit = tod_type;
        cmb.sig_unit = tod_type;
    }

    // set parallelization for psd filter ffts (maintained with tod output/verbose mode)
    omb.parallel_policy = parallel_policy;
    cmb.parallel_policy = parallel_policy;
    jinc_mm.parallel_policy = parallel_policy;

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

    // make noise maps?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_noise, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_noise_config.enabled = run_noise;
        }
    }
    if (run_noise) {
        // number of noise maps
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.n_noise, missing_keys, invalid_keys,
                             std::tuple{"noise_maps","n_noise_maps"},{},{0},{});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_noise_config.n_noise_maps = static_cast<int>(omb.n_noise);
            }
        }
        // randomize noise maps on detector as well as time chunk
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.randomize_dets, missing_keys, invalid_keys,
                             std::tuple{"noise_maps","randomize_dets"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_noise_config.randomize_dets = omb.randomize_dets;
            }
        }

        if (run_coadd) {
            citlali::pipeline::mirror_noise_map_settings_to_coadd(omb, cmb);
        }
    }
    // otherwise set number of noise maps to zero
    else {
        citlali::pipeline::disable_noise_map_settings(
            omb, cmb, typed_noise_config);
    }

    write_noise_realizations = false;
    if (config.template has_typed<bool>(std::tuple{"noise_maps","write_realizations"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, write_noise_realizations, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","write_realizations"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_noise_config.write_realizations = write_noise_realizations;
        }
    }
    run_noise_products = run_noise;
    typed_noise_config.products_enabled = run_noise_products;
    if (config.template has_typed<bool>(std::tuple{"noise_maps","products","enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_noise_products, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","products","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_noise_config.products_enabled = run_noise_products;
        }
    }
    apply_empirical_noise_weights = run_noise;
    typed_noise_config.apply_empirical_weights = apply_empirical_noise_weights;
    if (config.template has_typed<bool>(std::tuple{"noise_maps","products","apply_empirical_weights"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, apply_empirical_noise_weights, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","products","apply_empirical_weights"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_noise_config.apply_empirical_weights = apply_empirical_noise_weights;
        }
    }

    citlali::pipeline::set_mapmaker_polarization(
        rtcproc.run_polarization, naive_mm, jinc_mm);
}
