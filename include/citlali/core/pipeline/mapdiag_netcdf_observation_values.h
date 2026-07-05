#pragma once

// Included by mapdiag_netcdf.h inside namespace citlali::pipeline.

struct MapdiagObservationDoubleValues {
    const std::vector<double> &obs_weight_sum;
    const std::vector<double> &obs_weight_frac;
    const std::vector<double> &obs_core_weight_sum;
    const std::vector<double> &obs_core_weight_frac;
};

struct MapdiagObservationIntValues {
    const std::vector<int> &obs_valid_pixels;
    const std::vector<int> &obs_core_pixels;
};

struct MapdiagValueVars {
    MapdiagMapDoubleValues map_double;
    MapdiagMapIntValues map_int;
    MapdiagObservationDoubleValues observation_double;
    MapdiagObservationIntValues observation_int;
};

inline MapdiagValueVars make_mapdiag_value_vars(
    const MapdiagMapDoubleValues &map_double_values,
    const MapdiagMapIntValues &map_int_values,
    const MapdiagObservationDoubleValues &obs_double_values,
    const MapdiagObservationIntValues &obs_int_values) {
    return {
        map_double_values,
        map_int_values,
        obs_double_values,
        obs_int_values};
}

struct MapdiagNetcdfVars {
    const MapdiagSizeContext &size;
    const std::string &obsnum;
    const MapdiagMetadataVars &metadata;
    const MapdiagLabelVars &labels;
    const MapdiagValueVars &values;
};

inline MapdiagNetcdfVars make_mapdiag_netcdf_vars(
    const MapdiagSizeContext &size, const std::string &obsnum,
    const MapdiagMetadataVars &metadata, const MapdiagLabelVars &labels,
    const MapdiagValueVars &values) {
    return {size, obsnum, metadata, labels, values};
}

template <class AddDouble, class AddInt>
void add_mapdiag_observation_contribution_vars(
    const AddDouble &add_double, const AddInt &add_int,
    const MapdiagObservationDoubleValues &double_values,
    const MapdiagObservationIntValues &int_values) {
    add_double("coadd_obs_weight_sum",
               "sum of positive observation-level raw weight values aligned onto this map grid",
               double_values.obs_weight_sum);
    add_double("coadd_obs_weight_frac",
               "fractional contribution of each obsnum to coadd_obs_weight_sum for a given map",
               double_values.obs_weight_frac);
    add_double("coadd_obs_core_weight_sum",
               "sum of positive observation-level raw weight values within the final map core support",
               double_values.obs_core_weight_sum);
    add_double("coadd_obs_core_weight_frac",
               "fractional contribution of each obsnum within the final map core support",
               double_values.obs_core_weight_frac);
    add_int("coadd_obs_n_valid_pixels",
            "count of aligned observation pixels with positive raw weight",
            int_values.obs_valid_pixels);
    add_int("coadd_obs_n_core_pixels",
            "count of aligned observation pixels with positive raw weight inside the final map core support",
            int_values.obs_core_pixels);
}

inline void add_mapdiag_observation_contribution_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const MapdiagObservationDoubleValues &double_values,
    const MapdiagObservationIntValues &int_values) {
    auto add_double = [&](const std::string &name,
                          const std::string &comment,
                          const std::vector<double> &var_values) {
        add_mapdiag_obs_double_var(fo, dims, name, comment, var_values);
    };
    auto add_int = [&](const std::string &name,
                       const std::string &comment,
                       const std::vector<int> &var_values) {
        add_mapdiag_obs_int_var(fo, dims, name, comment, var_values);
    };
    add_mapdiag_observation_contribution_vars(
        add_double, add_int, double_values, int_values);
}

inline void add_mapdiag_value_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const MapdiagValueVars &values) {
    add_mapdiag_map_double_vars(fo, dims, values.map_double);
    add_mapdiag_map_int_vars(fo, dims, values.map_int);
    add_mapdiag_observation_contribution_vars(
        fo, dims, values.observation_double, values.observation_int);
}

inline void add_mapdiag_netcdf_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfVars &values) {
    add_obsnum_var(fo, mapdiag_obsnum_value(values.size, values.obsnum));

    const auto dims = add_mapdiag_netcdf_dims(fo, values.size);
    add_mapdiag_metadata_vars(fo, values.metadata);
    add_mapdiag_label_vars(fo, dims, values.labels);
    add_mapdiag_value_vars(fo, dims, values.values);
}

