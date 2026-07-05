#pragma once

// Included by spectral_diagnostics_netcdf.h inside namespace citlali::pipeline.

struct PsdNetcdfDims {
    netCDF::NcDim spectrum;
    std::vector<netCDF::NcDim> image;
};

template <class Spectrum>
std::size_t psd_spectrum_size(const Spectrum &spectrum) {
    return spectrum.size();
}

template <class Image>
std::size_t psd_image_rows(const Image &image) {
    return static_cast<std::size_t>(image.rows());
}

template <class Image>
std::size_t psd_image_cols(const Image &image) {
    return static_cast<std::size_t>(image.cols());
}

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

template <class Spectrum, class Image>
PsdNetcdfDims add_psd_netcdf_dims_for_image(
    netCDF::NcFile &fo, const std::string &base_name,
    const Spectrum &spectrum, const Image &image) {
    return add_psd_netcdf_dims(
        fo, base_name, psd_spectrum_size(spectrum), psd_image_rows(image),
        psd_image_cols(image));
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

