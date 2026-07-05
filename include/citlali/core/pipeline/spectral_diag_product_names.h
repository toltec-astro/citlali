#pragma once

// Included by spectral_diagnostics_netcdf.h inside namespace citlali::pipeline.

inline std::string spectral_product_base_name(const std::string &array_name,
                                              const std::string &map_name,
                                              const std::string &stokes_name) {
    return array_name + "_" + map_name + stokes_name;
}

template <class ArrayNameMap, class Arrays, class StokesParams, class MapIndex,
          class StokesIndex>
std::string spectral_product_name(ArrayNameMap &array_name_map,
                                  const Arrays &arrays,
                                  StokesParams &stokes_params,
                                  const std::string &map_name,
                                  MapIndex map_index,
                                  StokesIndex stokes_index) {
    const auto array = arrays[map_index];
    return spectral_product_base_name(
        array_name_map[array], map_name, stokes_params[stokes_index]);
}

inline std::string spectral_noise_product_base_name(
    const std::string &base_name) {
    return base_name + "_noise";
}

inline std::string spectral_noise_histogram_name(
    const std::string &base_name) {
    return base_name + "_noise_hist";
}

inline std::string spectral_histogram_bins_dim_name() {
    return "n_bins";
}

inline netCDF::NcDim add_spectral_histogram_bins_dim(netCDF::NcFile &fo,
                                                     std::size_t n_bins) {
    return fo.addDim(spectral_histogram_bins_dim_name(), n_bins);
}

template <class NoiseList>
bool has_spectral_noise_products(const NoiseList &noise) {
    return !noise.empty();
}

