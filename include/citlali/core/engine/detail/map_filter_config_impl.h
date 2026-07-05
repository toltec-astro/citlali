#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

template<typename CT>
void Engine::get_map_filter_config(CT &config) {
    logger->info("getting map filtering config options");
    // get wiener filter config options
    wiener_filter.get_config(config, missing_keys, invalid_keys);

    auto &typed_map_filter = typed_post_processing_config.map_filtering;
    typed_map_filter.enabled = run_map_filter;
    if (auto parsed = citlali::config::parse_map_filter_type(wiener_filter.filter_type)) {
        typed_map_filter.type = *parsed;
    }
    if (auto parsed = citlali::config::parse_map_filter_template_type(
            wiener_filter.template_type)) {
        typed_map_filter.template_type = *parsed;
    }
    typed_map_filter.lowpass_only = wiener_filter.run_lowpass;
    typed_map_filter.normalize_errors = wiener_filter.normalize_error;
    typed_map_filter.edge_guard.enabled = wiener_filter.edge_guard_enabled;
    typed_map_filter.edge_guard.weight_threshold_mode =
        wiener_filter.edge_weight_threshold_mode;
    typed_map_filter.edge_guard.hits_threshold_mode =
        wiener_filter.edge_hits_threshold_mode;
    typed_map_filter.edge_guard.hits_core_fraction =
        wiener_filter.edge_hits_core_fraction;
    typed_map_filter.edge_guard.guard_radius_fwhm =
        wiener_filter.edge_guard_radius_fwhm;
    typed_map_filter.edge_guard.fill_mode = wiener_filter.edge_fill_mode;
    if (auto parsed = citlali::config::parse_map_filter_edge_taper_mode(
            wiener_filter.edge_taper_mode)) {
        typed_map_filter.edge_guard.taper_mode = *parsed;
    }
    typed_map_filter.edge_guard.taper_min_fraction =
        wiener_filter.edge_taper_min_fraction;
    typed_map_filter.denom_rel_tol = wiener_filter.denom_rel_tol;
    typed_map_filter.tail_frac_tol = wiener_filter.tail_frac_tol;
    typed_map_filter.max_loops = wiener_filter.max_loops;
    typed_map_filter.denom_check_iters = wiener_filter.denom_check_iters;
    typed_map_filter.max_denom_iters = wiener_filter.max_denom_iters;
    typed_map_filter.template_fwhm_arcsec.clear();
    for (const auto &[array_name, fwhm_rad] : wiener_filter.template_fwhm_rad) {
        typed_map_filter.template_fwhm_arcsec[array_name] =
            fwhm_rad * RAD_TO_ASEC;
    }

    // if in science mode, write filtered maps as they complete
    if (redu_type=="science") {
        write_filtered_maps_partial = true;
    }
    // otherwise write at end
    else {
        write_filtered_maps_partial = false;
    }
    // check if kernel is enabled
    if (wiener_filter.template_type=="kernel") {
        if (!rtcproc.run_kernel) {
            logger->error("wiener filter kernel template requires kernel");
            std::exit(EXIT_FAILURE);
        }
        // copy the map fitter
        else {
            wiener_filter.map_fitter = map_fitter;
        }
    }
    // make sure noise maps were enabled
    if (!run_noise && (!wiener_filter.run_lowpass && wiener_filter.filter_type=="wiener_filter")) {
        logger->error("wiener filter requires noise maps");
        std::exit(EXIT_FAILURE);
    }

    // set parallelization for ffts (maintained with tod output/verbose mode)
    wiener_filter.parallel_policy = parallel_policy;
}

