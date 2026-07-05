#pragma once

// Included by mapdiag_netcdf_map_values.h inside namespace citlali::pipeline.

struct MapdiagMapIntValues {
    const std::vector<int> &n_valid_pixels;
    const std::vector<int> &n_core_pixels;
    const std::vector<int> &peak_row;
    const std::vector<int> &peak_col;
    const std::vector<int> &edge_guard_applied;
    const std::vector<int> &edge_guard_support_radius_pix;
    const std::vector<int> &edge_guard_science_npix;
    const std::vector<int> &edge_guard_support_npix;
    const std::vector<int> &edge_guard_guardband_npix;
};

template <class AddInt>
void add_mapdiag_map_int_vars(
    const AddInt &add_int, const MapdiagMapIntValues &values) {
    add_int("map_n_valid_pixels",
            "count of pixels with strictly positive weight",
            values.n_valid_pixels);
    add_int("map_n_core_pixels",
            "count of pixels with weight >= map_weight_threshold",
            values.n_core_pixels);
    add_int("map_peak_row",
            "row index of the maximum absolute signal-to-noise pixel",
            values.peak_row);
    add_int("map_peak_col",
            "column index of the maximum absolute signal-to-noise pixel",
            values.peak_col);
    add_int("map_edge_guard_applied",
            "1 when the filter edge guard was applied to this map, 0 otherwise",
            values.edge_guard_applied);
    add_int("map_edge_guard_support_radius_pix",
            "support-mask dilation radius in pixels used by the filter edge guard",
            values.edge_guard_support_radius_pix);
    add_int("map_edge_guard_science_npix",
            "number of pixels in the filter edge-guard science mask",
            values.edge_guard_science_npix);
    add_int("map_edge_guard_support_npix",
            "number of pixels in the filter edge-guard support mask",
            values.edge_guard_support_npix);
    add_int("map_edge_guard_guardband_npix",
            "number of pixels in the filter edge-guard guard band (support minus science)",
            values.edge_guard_guardband_npix);
}

inline void add_mapdiag_map_int_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const MapdiagMapIntValues &values) {
    auto add_int = [&](const std::string &name,
                       const std::string &comment,
                       const std::vector<int> &var_values) {
        add_mapdiag_map_int_var(fo, dims, name, comment, var_values);
    };
    add_mapdiag_map_int_vars(add_int, values);
}

