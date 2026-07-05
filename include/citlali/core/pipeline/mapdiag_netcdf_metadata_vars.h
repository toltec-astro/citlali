#pragma once

// Included by mapdiag_netcdf.h inside namespace citlali::pipeline.

inline void add_mapdiag_identity_vars(
    netCDF::NcFile &fo, const MapdiagIdentityVars &values) {
    add_netcdf_var<std::string>(fo, "MAP_STAGE", values.stage_name);
    add_netcdf_var<std::string>(fo, "MAP_BUFFER", values.buffer_name);
    add_netcdf_var<std::string>(fo, "MAP_REGIME", values.map_regime);
    add_netcdf_var<std::string>(fo, "SOURCE", values.source_name);
    add_netcdf_var<std::string>(fo, "PROJID", values.project_id);
    add_netcdf_var<std::string>(fo, "OBSGOAL", values.obs_goal);
}

inline void add_mapdiag_runtime_vars(
    netCDF::NcFile &fo, const MapdiagRuntimeVars &values) {
    add_netcdf_var(fo, "MAP_PIXEL_SIZE_RAD", values.pixel_size_rad);
    add_netcdf_var(fo, "MAP_COVERAGE_CUT", values.coverage_cut);
    add_netcdf_var<std::string>(fo, "MAP_SIG_UNIT", values.signal_unit);
}

inline void add_mapdiag_edge_guard_config_vars(
    netCDF::NcFile &fo, const MapdiagEdgeGuardConfigVars &values) {
    add_netcdf_var(fo, "MAP_EDGE_GUARD_ENABLED", values.enabled);
    add_netcdf_var<std::string>(
        fo, "MAP_EDGE_GUARD_WEIGHT_THRESHOLD_MODE",
        values.weight_threshold_mode);
    add_netcdf_var<std::string>(
        fo, "MAP_EDGE_GUARD_HITS_THRESHOLD_MODE",
        values.hits_threshold_mode);
    add_netcdf_var<std::string>(
        fo, "MAP_EDGE_GUARD_FILL_MODE", values.fill_mode);
    add_netcdf_var<std::string>(
        fo, "MAP_EDGE_GUARD_TAPER_MODE", values.taper_mode);
    add_netcdf_var(
        fo, "MAP_EDGE_GUARD_HITS_CORE_FRACTION",
        values.hits_core_fraction);
    add_netcdf_var(
        fo, "MAP_EDGE_GUARD_RADIUS_FWHM", values.radius_fwhm);
    add_netcdf_var(
        fo, "MAP_EDGE_GUARD_TAPER_MIN_FRACTION",
        values.taper_min_fraction);
}

inline void add_mapdiag_metadata_vars(
    netCDF::NcFile &fo, const MapdiagIdentityVars &identity,
    const MapdiagRuntimeVars &runtime,
    const MapdiagEdgeGuardConfigVars &edge_guard) {
    add_mapdiag_identity_vars(fo, identity);
    add_mapdiag_runtime_vars(fo, runtime);
    add_mapdiag_edge_guard_config_vars(fo, edge_guard);
}

inline void add_mapdiag_metadata_vars(
    netCDF::NcFile &fo, const MapdiagMetadataVars &values) {
    add_mapdiag_metadata_vars(
        fo, values.identity, values.runtime, values.edge_guard);
}

