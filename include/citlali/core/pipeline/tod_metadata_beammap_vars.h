#pragma once

// Included by tod_output_reduction_metadata.h inside namespace citlali::pipeline.

template <class Arrays, class ArrayNameMap, class FluxMap>
void add_beammap_source_flux_vars(netCDF::NcFile &fo, const Arrays &arrays,
                                  ArrayNameMap &array_name_map,
                                  FluxMap &flux_mjy_beam,
                                  FluxMap &flux_mjy_sr) {
    for (const auto &arr: arrays) {
        const auto &name = array_name_map[arr];
        add_netcdf_var(fo, "HEADER.SOURCE.FLUX_MJYPERBEAM_" + name,
                       flux_mjy_beam[name]);
        add_netcdf_var(fo, "HEADER.SOURCE.FLUX_MJYPERSR_" + name,
                       flux_mjy_sr[name]);
    }
}

inline void add_beammap_tuning_vars(
    netCDF::NcFile &fo,
    const citlali::config::BeammapIterationConfig &iteration_config,
    const citlali::config::BeammapPhaseStrategyConfig &phase_config,
    const citlali::config::BeammapReferenceConfig &reference_config) {
    add_netcdf_var(fo, "BEAMMAP.ITER_TOLERANCE",
                   iteration_config.tolerance);
    add_netcdf_var(fo, "BEAMMAP.CONVERGENCE_RADIUS_ARCSEC",
                   iteration_config.convergence_radius_arcsec);
    add_netcdf_var(fo, "BEAMMAP.ITER_MAX",
                   iteration_config.max_iterations);
    add_netcdf_var(fo, "BEAMMAP.PHASE_SPLIT_ENABLED",
                   phase_config.enabled);
    add_netcdf_var(fo, "BEAMMAP.LOCATOR_ITER", phase_config.locator_iter);
    add_netcdf_var(fo, "BEAMMAP.MEASUREMENT_START_ITER",
                   phase_config.measurement_start_iter);
    add_netcdf_var(fo, "BEAMMAP.IS_DEROTATED", reference_config.derotate);
}

inline void add_beammap_reference_vars(netCDF::NcFile &fo, int det_index,
                                       double ref_x_t, double ref_y_t) {
    add_netcdf_var(fo, "BEAMMAP.REF_DET_INDEX", det_index);
    add_netcdf_var(fo, "BEAMMAP.REF_X_T", ref_x_t);
    add_netcdf_var(fo, "BEAMMAP.REF_Y_T", ref_y_t);
}

template <class Calib, class ArrayNameMap, class FluxMap>
void add_beammap_tod_header_vars(
    netCDF::NcFile &fo, Calib &calib, ArrayNameMap &array_name_map,
    FluxMap &flux_mjy_beam, FluxMap &flux_mjy_sr,
    const citlali::config::BeammapIterationConfig &iteration_config,
    const citlali::config::BeammapPhaseStrategyConfig &phase_config,
    const citlali::config::BeammapReferenceConfig &reference_config) {
    add_beammap_source_flux_vars(
        fo, calib.arrays, array_name_map, flux_mjy_beam, flux_mjy_sr);
    add_beammap_tuning_vars(
        fo, iteration_config, phase_config, reference_config);

    int ref_det_index = -99;
    double ref_x_t = -99.0;
    double ref_y_t = -99.0;
    if (reference_config.subtract_reference_detector) {
        const auto reference_values =
            beammap_reference_header_values(
                calib, static_cast<Eigen::Index>(
                    reference_config.reference_detector));
        ref_det_index = reference_values.det_index;
        ref_x_t = reference_values.x_t;
        ref_y_t = reference_values.y_t;
    }
    add_beammap_reference_vars(fo, ref_det_index, ref_x_t, ref_y_t);
}
