#pragma once

// Included by spectral_diagnostics_netcdf.h inside namespace citlali::pipeline.

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

template <class Spectrum, class Frequency, class SpectrumImage,
          class FrequencyImage>
void add_psd_product(netCDF::NcFile &fo, const std::string &base_name,
                     const Spectrum &spectrum, const Frequency &frequency,
                     const SpectrumImage &spectrum_image,
                     const FrequencyImage &frequency_image) {
    const auto dims = add_psd_netcdf_dims_for_image(
        fo, base_name, spectrum, spectrum_image);
    add_psd_vector_pair(fo, base_name, dims.spectrum, spectrum, frequency);
    add_transposed_psd_image_pair(
        fo, base_name, dims.image, spectrum_image, frequency_image);
}

template <class NoiseList, class Spectra, class Frequencies,
          class SpectrumImages, class FrequencyImages, class Index>
void add_noise_psd_product_if_present(
    netCDF::NcFile &fo, const NoiseList &noise,
    const std::string &base_name, const Spectra &spectra,
    const Frequencies &frequencies, const SpectrumImages &spectrum_images,
    const FrequencyImages &frequency_images, Index index) {
    if (has_spectral_noise_products(noise)) {
        add_psd_product(
            fo, spectral_noise_product_base_name(base_name), spectra[index],
            frequencies[index], spectrum_images[index],
            frequency_images[index]);
    }
}

template <class Spectra, class Frequencies, class SpectrumImages,
          class FrequencyImages, class NoiseList, class NoiseSpectra,
          class NoiseFrequencies, class NoiseSpectrumImages,
          class NoiseFrequencyImages, class Index>
void add_spectral_psd_product(
    netCDF::NcFile &fo, const NoiseList &noise,
    const std::string &base_name, const Spectra &spectra,
    const Frequencies &frequencies, const SpectrumImages &spectrum_images,
    const FrequencyImages &frequency_images, const NoiseSpectra &noise_spectra,
    const NoiseFrequencies &noise_frequencies,
    const NoiseSpectrumImages &noise_spectrum_images,
    const NoiseFrequencyImages &noise_frequency_images, Index index) {
    add_psd_product(
        fo, base_name, spectra[index], frequencies[index],
        spectrum_images[index], frequency_images[index]);
    add_noise_psd_product_if_present(
        fo, noise, base_name, noise_spectra, noise_frequencies,
        noise_spectrum_images, noise_frequency_images, index);
}

template <class MapBuffer, class ArrayNameMap, class Arrays,
          class StokesParams, class GetMapName, class ArrayToMap,
          class MapToStokes>
void add_spectral_psd_products_for_maps(
    netCDF::NcFile &fo, const MapBuffer &mb, ArrayNameMap &array_name_map,
    const Arrays &arrays, StokesParams &stokes_params,
    const GetMapName &get_map_name, const ArrayToMap &arrays_to_maps,
    const MapToStokes &maps_to_stokes) {
    for (Eigen::Index i = 0; i < mb->psds.size(); ++i) {
        const std::string name = spectral_product_name(
            array_name_map, arrays, stokes_params, get_map_name(i),
            arrays_to_maps(i), maps_to_stokes(i));
        add_spectral_psd_product(
            fo, mb->noise, name, mb->psds, mb->psd_freqs, mb->psd_2ds,
            mb->psd_2d_freqs, mb->noise_psds, mb->noise_psd_freqs,
            mb->noise_psd_2ds, mb->noise_psd_2d_freqs, i);
    }
}

