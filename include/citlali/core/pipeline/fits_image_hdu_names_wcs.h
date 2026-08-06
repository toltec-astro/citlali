#pragma once

// Included by fits_image_metadata.h inside namespace citlali::pipeline.

inline std::string signal_map_hdu_name(const std::string &map_name,
                                       const std::string &stokes_suffix) {
    return "signal_" + map_name + stokes_suffix;
}

inline std::string weight_map_hdu_name(const std::string &map_name,
                                       const std::string &stokes_suffix) {
    return "weight_" + map_name + stokes_suffix;
}

inline std::string formal_weight_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "weight_formal_" + map_name + stokes_suffix;
}

inline std::string noise_variance_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "noise_variance_" + map_name + stokes_suffix;
}

inline std::string conditional_stack_scatter_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "conditional_stack_scatter_" + map_name + stokes_suffix;
}

inline std::string kernel_map_hdu_name(const std::string &map_name,
                                       const std::string &stokes_suffix) {
    return "kernel_" + map_name + stokes_suffix;
}

inline std::string coverage_map_hdu_name(const std::string &map_name,
                                         const std::string &stokes_suffix) {
    return "coverage_" + map_name + stokes_suffix;
}

inline std::string coverage_mask_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "coverage_bool_" + map_name + stokes_suffix;
}

inline std::string geometric_hits_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "geometric_hits_" + map_name + stokes_suffix;
}

inline std::string contributing_hits_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "contributing_hits_" + map_name + stokes_suffix;
}

inline std::string coadd_observation_count_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "coadd_observation_count_" + map_name + stokes_suffix;
}

inline std::string upstream_eligible_exposure_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "upstream_eligible_exposure_" + map_name + stokes_suffix;
}

inline std::string retained_exposure_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "retained_exposure_" + map_name + stokes_suffix;
}

inline std::string normalization_support_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "normalization_support_" + map_name + stokes_suffix;
}

inline std::string science_policy_support_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "science_policy_support_" + map_name + stokes_suffix;
}

inline std::string science_valid_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "science_valid_" + map_name + stokes_suffix;
}

inline std::string legacy_pixel_snr_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "sig2noise_" + map_name + stokes_suffix;
}

inline std::string pixel_snr_map_hdu_name(const std::string &map_name,
                                          const std::string &stokes_suffix) {
    return "sig2noise_pixel_" + map_name + stokes_suffix;
}

inline std::string coefficient_standardized_signal_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "coefficient_standardized_signal_" + map_name + stokes_suffix;
}

inline std::string formal_standardized_signal_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "formal_standardized_signal_" + map_name + stokes_suffix;
}

inline std::string point_source_flux_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "point_source_flux_" + map_name + stokes_suffix;
}

inline std::string point_source_uncertainty_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "point_source_uncertainty_" + map_name + stokes_suffix;
}

inline std::string filtered_pixel_stack_scatter_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "filtered_pixel_stack_scatter_" + map_name + stokes_suffix;
}

inline std::string point_source_snr_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "sig2noise_point_source_" + map_name + stokes_suffix;
}

inline std::string conditional_stack_scatter_ratio_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "conditional_stack_scatter_ratio_" + map_name + stokes_suffix;
}

inline std::string noise_signal_map_hdu_name(
    const std::string &map_name, Eigen::Index noise_index,
    const std::string &stokes_suffix) {
    return "signal_" + map_name + std::to_string(noise_index) + "_" +
           stokes_suffix;
}

inline double default_wcs_source_epoch() {
    return 2000.0;
}

template <class HeaderMap, class Logger>
double wcs_source_epoch_or_default(const HeaderMap &tel_header,
                                   const Logger &logger) {
    const double source_epoch = default_wcs_source_epoch();
    const auto epoch_it = tel_header.find("Header.Source.Epoch");
    if (epoch_it != tel_header.end() && epoch_it->second.size() > 0 &&
        std::isfinite(epoch_it->second(0))) {
        return epoch_it->second(0);
    }
    logger->warn("Header.Source.Epoch missing/invalid; using epoch={} for WCS",
                 source_epoch);
    return source_epoch;
}

template <class ArrayFreqMap>
double map_wcs_frequency(const ArrayFreqMap &array_freq_map,
                         Eigen::Index array_id) {
    const auto it = array_freq_map.find(array_id);
    if (it == array_freq_map.end() || !std::isfinite(it->second)) {
        throw std::runtime_error(
            "map WCS requires a declared finite array frequency");
    }
    return it->second;
}

template <class Wcs, class ArrayFreqMap>
void assign_map_wcs_spectral_axes(Wcs &wcs,
                                  const ArrayFreqMap &array_freq_map,
                                  Eigen::Index array_id,
                                  Eigen::Index stokes_index) {
    wcs.crval[2] = map_wcs_frequency(array_freq_map, array_id);
    wcs.crval[3] = stokes_index;
}

template <class ImageList>
bool has_map_image_slot(const ImageList &images, Eigen::Index i,
                        Eigen::Index n_rows, Eigen::Index n_cols) {
    return i >= 0 && i < static_cast<Eigen::Index>(images.size()) &&
           images[i].rows() == n_rows &&
           images[i].cols() == n_cols;
}

template <class FitsIo, class FitsIoContainer>
bool is_filtered_map_output(const FitsIo &fits_io,
                            const FitsIoContainer &filtered_fits_io,
                            const FitsIoContainer &filtered_coadd_fits_io) {
    return fits_io == &filtered_fits_io || fits_io == &filtered_coadd_fits_io;
}
