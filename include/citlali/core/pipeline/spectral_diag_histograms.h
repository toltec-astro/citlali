#pragma once

// Included by spectral_diagnostics_netcdf.h inside namespace citlali::pipeline.

template <class Bins, class Counts>
void add_histogram_pair(netCDF::NcFile &fo, const std::string &base_name,
                        netCDF::NcDim dim, const Bins &bins,
                        const Counts &counts) {
    add_double_1d_var(fo, base_name + "_bins", dim, bins);
    add_double_1d_var(fo, base_name + "_hist", dim, counts);
}

template <class NoiseList, class Counts, class Index>
void add_noise_histogram_if_present(netCDF::NcFile &fo,
                                    const NoiseList &noise,
                                    const std::string &base_name,
                                    netCDF::NcDim dim,
                                    const Counts &counts,
                                    Index index) {
    if (has_spectral_noise_products(noise)) {
        add_double_1d_var(
            fo, spectral_noise_histogram_name(base_name), dim, counts[index]);
    }
}

template <class Bins, class Counts, class NoiseList, class NoiseCounts,
          class Index>
void add_spectral_histogram_product(
    netCDF::NcFile &fo, const NoiseList &noise,
    const std::string &base_name, netCDF::NcDim dim, const Bins &bins,
    const Counts &counts, const NoiseCounts &noise_counts, Index index) {
    add_histogram_pair(fo, base_name, dim, bins[index], counts[index]);
    add_noise_histogram_if_present(
        fo, noise, base_name, dim, noise_counts, index);
}

template <class MapBuffer, class ArrayNameMap, class Arrays,
          class StokesParams, class GetMapName, class ArrayToMap,
          class MapToStokes>
void add_spectral_histogram_products_for_maps(
    netCDF::NcFile &fo, const MapBuffer &mb, netCDF::NcDim hist_bins_dim,
    ArrayNameMap &array_name_map, const Arrays &arrays,
    StokesParams &stokes_params, const GetMapName &get_map_name,
    const ArrayToMap &arrays_to_maps, const MapToStokes &maps_to_stokes) {
    for (Eigen::Index i = 0; i < mb->hists.size(); ++i) {
        const std::string name = spectral_product_name(
            array_name_map, arrays, stokes_params, get_map_name(i),
            arrays_to_maps(i), maps_to_stokes(i));
        add_spectral_histogram_product(
            fo, mb->noise, name, hist_bins_dim, mb->hist_bins, mb->hists,
            mb->noise_hists, i);
    }
}

