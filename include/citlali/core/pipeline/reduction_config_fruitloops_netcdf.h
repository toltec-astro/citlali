#pragma once

// Included by reduction_config_netcdf.h inside namespace citlali::pipeline.

template <class PtcProc>
void add_fruit_loops_config_vars(netCDF::NcFile &fo,
                                 const PtcProc &ptcproc) {
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS", ptcproc.run_fruit_loops);
    add_netcdf_var<std::string>(fo, "CONFIG.FRUITLOOPS.PATH",
                                ptcproc.fruit_loops_path);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.S2N",
                   ptcproc.fruit_loops_sig2noise);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.PEAKFRAC",
                   ptcproc.fruit_loops_peak_fraction_limit);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSNR",
                   ptcproc.fruit_loops_local_snr_floor);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_INNER",
                   ptcproc.fruit_loops_local_sigma_inner_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_OUTER",
                   ptcproc.fruit_loops_local_sigma_outer_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_EDGE",
                   ptcproc.fruit_loops_local_sigma_edge_guard_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_MINPIX",
                   ptcproc.fruit_loops_local_sigma_min_pixels);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.ADAPT_SUPPORT_RAD",
                   ptcproc.fruit_loops_adaptive_support_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.ADAPT_SUPPORT_FWHM",
                   ptcproc.fruit_loops_adaptive_support_radius_fwhm);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.ENABLED",
                   ptcproc.fruit_loops_weight_feedback_enabled);
    add_netcdf_var<std::string>(
        fo, "CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.REFERENCE",
        ptcproc.fruit_loops_weight_feedback_reference);
    add_netcdf_var(
        fo, "CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.LOW_RELATIVE_WEIGHT",
        ptcproc.fruit_loops_weight_feedback_low_relative_weight);
    add_netcdf_var(
        fo, "CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.HIGH_RELATIVE_WEIGHT",
        ptcproc.fruit_loops_weight_feedback_high_relative_weight);
}

template <class PtcProc, class Calib, class ArrayNameMap>
void add_fruit_loop_flux_config_vars(netCDF::NcFile &fo,
                                     const PtcProc &ptcproc,
                                     const Calib &calib,
                                     ArrayNameMap &array_name_map) {
    for (decltype(calib.arrays.size()) i=0; i<calib.arrays.size(); ++i) {
        double flux_limit = 0.0;
        if (ptcproc.run_fruit_loops) {
            if (ptcproc.fruit_loops_flux.size() == calib.arrays.size()) {
                flux_limit = ptcproc.fruit_loops_flux(i);
            }
            else if (calib.arrays(i) < ptcproc.fruit_loops_flux.size()) {
                flux_limit = ptcproc.fruit_loops_flux(calib.arrays(i));
            }
        }
        add_netcdf_var(
            fo, "CONFIG.FRUITLOOPS.FLUX_" + array_name_map[calib.arrays(i)],
            flux_limit);
    }
}

template <class PtcProc>
void add_fruit_loop_iteration_config_vars(netCDF::NcFile &fo,
                                          const PtcProc &ptcproc) {
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.MAXITER",
                   ptcproc.fruit_loops_iters);
}

template <class PtcProc, class Calib, class ArrayNameMap>
void add_fruit_loop_header_config_vars(netCDF::NcFile &fo,
                                       const PtcProc &ptcproc,
                                       const Calib &calib,
                                       ArrayNameMap &array_name_map) {
    add_fruit_loops_config_vars(fo, ptcproc);
    add_fruit_loop_flux_config_vars(fo, ptcproc, calib, array_name_map);
    add_fruit_loop_iteration_config_vars(fo, ptcproc);
}

template <class PtcProc>
void add_ptcdiag_compact_config_vars(netCDF::NcFile &fo,
                                     const PtcProc &ptcproc) {
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.ENABLED",
                   ptcproc.weight_corr_penalty.enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.ENABLED",
                   ptcproc.busy_row_suppression.enabled);
    add_netcdf_var(fo, "CONFIG.CLEANED", ptcproc.run_clean);
    add_netcdf_var<std::string>(fo, "CONFIG.CLEANED.MODESEL",
                                ptcproc.cleaner.active_cleaner_label());
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.ENABLED",
                   ptcproc.cleaner.adaptive_selector.enabled);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.ENABLED",
                   ptcproc.second_pass_local.enabled);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_SPIKE_SIGMA",
                   ptcproc.second_pass_local.min_spike_sigma);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.HIGH_SCORE_EVENT_OVERRIDE",
                   ptcproc.second_pass_local.high_score_event_override);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_CLUSTER_DETECTORS",
                   ptcproc.second_pass_local.min_cluster_detectors);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MAX_AUTO_FLAG_CLUSTERS",
                   ptcproc.second_pass_local.max_auto_flag_clusters_per_network);
    add_fruit_loops_config_vars(fo, ptcproc);
}

