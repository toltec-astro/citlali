#pragma once

// Included by reduction_config_netcdf.h inside namespace citlali::pipeline.

template <class WeightValidation>
void add_weight_validation_config_vars(
    netCDF::NcFile &fo, const WeightValidation &weight_validation) {
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.ENABLED",
                   weight_validation.enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.ACCUMULATION_ITERS",
                   weight_validation.accumulation_iters);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.APPLY_START_ITER",
                   weight_validation.apply_start_iter);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.MIN_FACTOR",
                   weight_validation.min_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.UPWARD_ENABLED",
                   weight_validation.upward_enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.UPWARD_MAX_FACTOR",
                   weight_validation.upward_max_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.UPWARD_POWER",
                   weight_validation.upward_power);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.UPWARD_MIN_BASE_FACTOR",
                   weight_validation.upward_min_base_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.UPWARD_REQUIRE_ATM",
                   weight_validation.upward_require_atmospheric);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.UPWARD_MIN_ATM_FACTOR",
                   weight_validation.upward_min_atmospheric_factor);
    add_netcdf_var<std::string>(
        fo, "CONFIG.WEIGHT.VALIDATION.ATM_GROUPING",
        std::string{citlali::config::to_string(
            weight_validation.atmospheric_grouping)});
}

void add_weight_selection_config_vars(netCDF::NcFile &fo,
                                      const citlali::config::ProcessedTimeChunkWeightingConfig
                                          &weighting) {
    add_netcdf_var<std::string>(fo, "CONFIG.WEIGHT.TYPE",
                                std::string{citlali::config::to_string(
                                    weighting.type)});
    add_netcdf_var(fo, "CONFIG.WEIGHT.SOURCE_MASK_RADIUS_ARCSEC",
                   weighting.source_mask_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.WEIGHT.HYBRID_MIN_FACTOR",
                   weighting.hybrid_correction_min_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.HYBRID_MAX_FACTOR",
                   weighting.hybrid_correction_max_factor);
    add_weight_validation_config_vars(fo, weighting.validation);
}

template <class RuntimeWindow>
void add_ptc_weight_cutoff_config_vars(netCDF::NcFile &fo,
                                       const citlali::config::ProcessedTimeChunkConfig
                                           &config,
                                       RuntimeWindow inv_var_window_sec,
                                       bool include_inv_var_window = false) {
    const auto &flagging = config.flagging;
    const auto &weighting = config.weighting;
    add_netcdf_var(fo, "CONFIG.INV_VAR.PTC.WTLOW",
                   flagging.lower_tod_inv_var_factor);
    add_netcdf_var(fo, "CONFIG.INV_VAR.PTC.WTHIGH",
                   flagging.upper_tod_inv_var_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.PTC.WTLOW",
                   weighting.lower_map_weight_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.PTC.WTHIGH",
                   weighting.upper_map_weight_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.MEDWTFACTOR",
                   weighting.median_map_weight_factor);
    if (include_inv_var_window) {
        add_netcdf_var(fo, "CONFIG.INV_VAR.WINDOW_SEC",
                       inv_var_window_sec);
    }
}

inline void add_tod_initial_runtime_config_vars(netCDF::NcFile &fo,
                                                bool verbose_mode,
                                                bool run_polarization,
                                                bool run_despike) {
    add_netcdf_var(fo, "CONFIG.VERBOSE", verbose_mode);
    add_netcdf_var(fo, "CONFIG.POLARIZED", run_polarization);
    add_netcdf_var(fo, "CONFIG.DESPIKED", run_despike);
}

template <class RawTimeChunkConfig>
void add_tod_filter_config_vars(netCDF::NcFile &fo,
                                const RawTimeChunkConfig &config,
                                bool run_any_tod_filter) {
    add_netcdf_var(fo, "CONFIG.TODFILTERED", run_any_tod_filter);
    add_netcdf_var(fo, "CONFIG.TODNOTCH", config.filter.notch.enabled);
    add_netcdf_var(fo, "CONFIG.TODIIRHP", config.iir_filter.enabled);
    add_netcdf_var(fo, "CONFIG.TODIIRHP.FREQ_HZ",
                   config.iir_filter.freq_Hz);
    add_netcdf_var(fo, "CONFIG.TODIIRHP.ORDER",
                   config.iir_filter.order);
    add_netcdf_var(fo, "CONFIG.TODIIRHP.ZEROPHASE",
                   config.iir_filter.zero_phase);
}

template <class RawTimeChunkConfig>
void add_tod_processing_config_vars(netCDF::NcFile &fo,
                                    const RawTimeChunkConfig &config) {
    add_netcdf_var(fo, "CONFIG.DOWNSAMPLED", config.downsample.enabled);
    add_netcdf_var(fo, "CONFIG.CALIBRATED",
                   config.flux_calibration_enabled);
    add_netcdf_var(fo, "CONFIG.EXTINCTION",
                   config.extinction_correction_enabled);
    add_netcdf_var<std::string>(fo, "CONFIG.EXTINCTION.EXTMODEL",
                                config.extinction_model);
}
