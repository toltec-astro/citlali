#pragma once

// Included by reduction_config_netcdf.h inside namespace citlali::pipeline.

void add_fruit_loops_config_vars(netCDF::NcFile &fo,
                                 const citlali::config::TimestreamFruitLoopsConfig
                                     &config) {
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS", config.enabled);
    add_netcdf_var<std::string>(fo, "CONFIG.FRUITLOOPS.PATH",
                                config.path);
    add_netcdf_var<std::string>(fo, "CONFIG.FRUITLOOPS.RESTART_PATH",
                                config.restart_path);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.S2N",
                   config.sig2noise_limit);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.PEAKFRAC",
                   config.peak_fraction_limit);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSNR",
                   config.local_snr_floor);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_INNER",
                   config.local_sigma_inner_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_OUTER",
                   config.local_sigma_outer_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_EDGE",
                   config.local_sigma_edge_guard_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_MINPIX",
                   config.local_sigma_min_pixels);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.ADAPT_SUPPORT_RAD",
                   config.adaptive_support_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.ADAPT_SUPPORT_FWHM",
                   config.adaptive_support_radius_fwhm);
    add_netcdf_var<std::string>(
        fo, "CONFIG.FRUITLOOPS.SOURCE_CENTER_MODE",
        std::string{citlali::config::to_string(config.source_center_mode)});
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.ENABLED",
                   config.weight_feedback.enabled);
    add_netcdf_var<std::string>(
        fo, "CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.REFERENCE",
        std::string{
            citlali::config::to_string(config.weight_feedback.reference)});
    add_netcdf_var(
        fo, "CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.LOW_RELATIVE_WEIGHT",
        config.weight_feedback.low_relative_weight);
    add_netcdf_var(
        fo, "CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.HIGH_RELATIVE_WEIGHT",
        config.weight_feedback.high_relative_weight);
    add_netcdf_var(
        fo, "CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.ENABLED",
        config.injected_source_test.enabled);
    add_netcdf_var(
        fo, "CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.START_ITERATION",
        config.injected_source_test.start_iteration);
}

template <class Calib, class ArrayNameMap>
void add_fruit_loop_flux_config_vars(netCDF::NcFile &fo,
                                     const citlali::config::TimestreamFruitLoopsConfig
                                         &config,
                                     const Calib &calib,
                                     ArrayNameMap &array_name_map) {
    for (decltype(calib.arrays.size()) i=0; i<calib.arrays.size(); ++i) {
        double flux_limit = 0.0;
        if (config.enabled) {
            if (config.array_flux_limit.size() == calib.arrays.size()) {
                flux_limit = config.array_flux_limit[i];
            }
            else if (calib.arrays(i) < config.array_flux_limit.size()) {
                flux_limit = config.array_flux_limit[calib.arrays(i)];
            }
        }
        add_netcdf_var(
            fo, "CONFIG.FRUITLOOPS.FLUX_" + array_name_map[calib.arrays(i)],
            flux_limit);
        double injected_amplitude = 0.0;
        const auto array_id = calib.arrays(i);
        if (array_id >= 0 &&
            array_id < static_cast<Eigen::Index>(
                config.injected_source_test
                    .array_amplitude_mjy_beam.size())) {
            injected_amplitude =
                config.injected_source_test.array_amplitude_mjy_beam[
                    static_cast<std::size_t>(array_id)];
        }
        add_netcdf_var(
            fo,
            "CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.AMPLITUDE_MJY_BEAM_" +
                array_name_map[array_id],
            injected_amplitude);
    }
}

void add_fruit_loop_iteration_config_vars(netCDF::NcFile &fo,
                                          const citlali::config::TimestreamFruitLoopsConfig
                                              &config) {
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.MAXITER",
                   config.max_iters);
}

template <class Calib, class ArrayNameMap>
void add_fruit_loop_header_config_vars(netCDF::NcFile &fo,
                                       const citlali::config::TimestreamFruitLoopsConfig
                                           &config,
                                       const Calib &calib,
                                       ArrayNameMap &array_name_map) {
    add_fruit_loops_config_vars(fo, config);
    add_fruit_loop_flux_config_vars(fo, config, calib, array_name_map);
    add_fruit_loop_iteration_config_vars(fo, config);
}

void add_ptcdiag_compact_config_vars(netCDF::NcFile &fo,
                                     const citlali::config::ProcessedTimeChunkConfig
                                         &processed_config,
                                     const citlali::config::TimestreamFruitLoopsConfig
                                         &fruit_config) {
    const auto &clean = processed_config.clean;
    const auto &weighting = processed_config.weighting;
    const auto &second_pass = processed_config.flagging.second_pass_local;
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.ENABLED",
                   weighting.corr_penalty.enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.ENABLED",
                   weighting.busy_row_suppression.enabled);
    add_netcdf_var(fo, "CONFIG.CLEANED", clean.enabled);
    add_netcdf_var<std::string>(fo, "CONFIG.CLEANED.MODESEL",
                                std::string{citlali::config::to_string(
                                    clean.active)});
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.ENABLED",
                   clean.adaptive_selector.enabled);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.ENABLED",
                   second_pass.enabled);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_SPIKE_SIGMA",
                   second_pass.min_spike_sigma);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.HIGH_SCORE_EVENT_OVERRIDE",
                   second_pass.high_score_event_override);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_CLUSTER_DETECTORS",
                   second_pass.min_cluster_detectors);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MAX_AUTO_FLAG_CLUSTERS",
                   second_pass.max_auto_flag_clusters_per_network);
    add_fruit_loops_config_vars(fo, fruit_config);
}
