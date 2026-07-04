#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

namespace citlali::pipeline {

inline std::string spectral_product_base_name(const std::string &array_name,
                                              const std::string &map_name,
                                              const std::string &stokes_name) {
    return array_name + "_" + map_name + stokes_name;
}

inline std::string spectral_noise_product_base_name(
    const std::string &base_name) {
    return base_name + "_noise";
}

inline std::string spectral_noise_histogram_name(
    const std::string &base_name) {
    return base_name + "_noise_hist";
}

template <class NoiseList>
bool has_spectral_noise_products(const NoiseList &noise) {
    return !noise.empty();
}

struct PsdNetcdfDims {
    netCDF::NcDim spectrum;
    std::vector<netCDF::NcDim> image;
};

inline PsdNetcdfDims add_psd_netcdf_dims(netCDF::NcFile &fo,
                                         const std::string &base_name,
                                         std::size_t nfreq,
                                         std::size_t n_rows,
                                         std::size_t n_cols) {
    netCDF::NcDim spectrum_dim = fo.addDim(base_name + "_nfreq", nfreq);
    netCDF::NcDim row_dim = fo.addDim(base_name + "_rows", n_rows);
    netCDF::NcDim col_dim = fo.addDim(base_name + "_cols", n_cols);
    return {spectrum_dim, {row_dim, col_dim}};
}

template <class Data>
void add_double_1d_var(netCDF::NcFile &fo, const std::string &name,
                       netCDF::NcDim dim, const Data &data) {
    netCDF::NcVar var = fo.addVar(name, netCDF::ncDouble, dim);
    var.putVar(data.data());
}

template <class Data>
void add_double_2d_var(netCDF::NcFile &fo, const std::string &name,
                       const std::vector<netCDF::NcDim> &dims,
                       const Data &data) {
    netCDF::NcVar var = fo.addVar(name, netCDF::ncDouble, dims);
    var.putVar(data.data());
}

template <class Spectrum, class Frequency>
void add_psd_vector_pair(netCDF::NcFile &fo, const std::string &base_name,
                         netCDF::NcDim dim, const Spectrum &spectrum,
                         const Frequency &frequency) {
    add_double_1d_var(fo, base_name + "_psd", dim, spectrum);
    add_double_1d_var(fo, base_name + "_psd_freq", dim, frequency);
}

template <class SpectrumImage, class FrequencyImage>
void add_psd_image_pair(netCDF::NcFile &fo, const std::string &base_name,
                        const std::vector<netCDF::NcDim> &dims,
                        const SpectrumImage &spectrum,
                        const FrequencyImage &frequency) {
    add_double_2d_var(fo, base_name + "_psd_2d", dims, spectrum);
    add_double_2d_var(fo, base_name + "_psd_2d_freq", dims, frequency);
}

template <class SpectrumImage, class FrequencyImage>
void add_transposed_psd_image_pair(
    netCDF::NcFile &fo, const std::string &base_name,
    const std::vector<netCDF::NcDim> &dims,
    const SpectrumImage &spectrum, const FrequencyImage &frequency) {
    const Eigen::MatrixXd spectrum_transposed = spectrum.transpose();
    const Eigen::MatrixXd frequency_transposed = frequency.transpose();
    add_psd_image_pair(
        fo, base_name, dims, spectrum_transposed, frequency_transposed);
}

template <class Bins, class Counts>
void add_histogram_pair(netCDF::NcFile &fo, const std::string &base_name,
                        netCDF::NcDim dim, const Bins &bins,
                        const Counts &counts) {
    add_double_1d_var(fo, base_name + "_bins", dim, bins);
    add_double_1d_var(fo, base_name + "_hist", dim, counts);
}

}  // namespace citlali::pipeline
