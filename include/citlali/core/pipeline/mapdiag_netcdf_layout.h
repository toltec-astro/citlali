#pragma once

// Included by mapdiag_netcdf.h inside namespace citlali::pipeline.

inline double mapdiag_fill_double() {
    return std::numeric_limits<double>::quiet_NaN();
}

constexpr int mapdiag_fill_int() {
    return -2147483647;
}

struct MapdiagSizeContext {
    std::size_t n_maps;
    std::size_t n_obsnums;
    bool is_coadd;
};

inline std::string mapdiag_map_dim_name() {
    return "n_maps";
}

inline std::string mapdiag_obsnum_dim_name() {
    return "n_obsnums";
}

inline std::string mapdiag_netcdf_filename(
    const std::string &base_filename) {
    return base_filename + ".nc";
}

struct MapdiagNetcdfDims {
    netCDF::NcDim maps;
    netCDF::NcDim obsnums;
    std::vector<netCDF::NcDim> map_obs;
};

struct MapdiagIdentityVars {
    const std::string &stage_name;
    const std::string &buffer_name;
    const std::string &map_regime;
    const std::string &source_name;
    const std::string &project_id;
    const std::string &obs_goal;
};

struct MapdiagRuntimeVars {
    double pixel_size_rad;
    double coverage_cut;
    const std::string &signal_unit;
};

struct MapdiagEdgeGuardConfigVars {
    bool enabled;
    const std::string &weight_threshold_mode;
    const std::string &hits_threshold_mode;
    const std::string &fill_mode;
    const std::string &taper_mode;
    double hits_core_fraction;
    double radius_fwhm;
    double taper_min_fraction;
};

struct MapdiagMetadataVars {
    MapdiagIdentityVars identity;
    MapdiagRuntimeVars runtime;
    MapdiagEdgeGuardConfigVars edge_guard;
};

template <class MapBuffer, class MapFilter>
MapdiagMetadataVars make_mapdiag_metadata_vars(
    const std::string &stage_name, const MapBuffer &mb,
    const std::string &map_regime, const std::string &source_name,
    const std::string &project_id, const std::string &obs_goal,
    const MapFilter &map_filter) {
    return {
        {stage_name, mb->name, map_regime, source_name, project_id, obs_goal},
        {mb->pixel_size_rad, mb->cov_cut, mb->sig_unit},
        {map_filter.edge_guard_enabled,
         map_filter.edge_weight_threshold_mode,
         map_filter.edge_hits_threshold_mode,
         map_filter.edge_fill_mode,
         map_filter.edge_taper_mode,
         map_filter.edge_hits_core_fraction,
         map_filter.edge_guard_radius_fwhm,
         map_filter.edge_taper_min_fraction}};
}

struct MapdiagLabelVars {
    const std::vector<std::string> &array_names;
    const std::vector<std::string> &stokes_names;
    const std::vector<std::string> &map_names;
    const std::vector<std::string> &obsnums;
    const std::string &fallback_obsnum;
    const std::vector<std::string> &date_obs;
    std::size_t n_obsnums;
};

inline MapdiagLabelVars make_mapdiag_label_vars(
    const std::vector<std::string> &array_names,
    const std::vector<std::string> &stokes_names,
    const std::vector<std::string> &map_names,
    const std::vector<std::string> &obsnums,
    const std::string &fallback_obsnum,
    const std::vector<std::string> &date_obs,
    const MapdiagSizeContext &context) {
    return {
        array_names,
        stokes_names,
        map_names,
        obsnums,
        fallback_obsnum,
        date_obs,
        context.n_obsnums};
}

inline MapdiagLabelVars make_mapdiag_label_vars(
    const MapdiagMapLabelStorage &labels,
    const std::vector<std::string> &obsnums,
    const std::string &fallback_obsnum,
    const std::vector<std::string> &date_obs,
    const MapdiagSizeContext &context) {
    return make_mapdiag_label_vars(
        labels.array_names, labels.stokes_names, labels.map_names, obsnums,
        fallback_obsnum, date_obs, context);
}

inline MapdiagNetcdfDims add_mapdiag_netcdf_dims(
    netCDF::NcFile &fo, const MapdiagSizeContext &context) {
    netCDF::NcDim maps_dim =
        fo.addDim(mapdiag_map_dim_name(), context.n_maps);
    netCDF::NcDim obsnums_dim =
        fo.addDim(mapdiag_obsnum_dim_name(), context.n_obsnums);
    return {maps_dim, obsnums_dim, {maps_dim, obsnums_dim}};
}

MapdiagSizeContext make_mapdiag_size_context(std::size_t n_maps,
                                             std::size_t obsnum_count,
                                             bool is_coadd) {
    return {n_maps, obsnum_count, is_coadd};
}

inline std::size_t mapdiag_obs_table_size(const MapdiagSizeContext &context) {
    return context.n_maps * context.n_obsnums;
}

inline int mapdiag_obsnum_value(const MapdiagSizeContext &context,
                                const std::string &obsnum) {
    return context.is_coadd ? -1 : std::stoi(obsnum);
}

inline std::size_t mapdiag_obs_flat_index(const MapdiagSizeContext &context,
                                          std::size_t map_index,
                                          std::size_t obs_index) {
    return map_index * context.n_obsnums + obs_index;
}

