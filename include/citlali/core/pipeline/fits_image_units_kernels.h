#pragma once

// Included by fits_image_metadata.h inside namespace citlali::pipeline.

inline std::string map_weight_unit(const std::string &signal_unit) {
    return "1/(" + signal_unit + ")^2";
}

inline std::string map_variance_unit(const std::string &signal_unit) {
    return "(" + signal_unit + ")^2";
}

inline bool empirical_weight_calibration_enabled(
    bool run_noise_products, bool run_noise,
    bool apply_empirical_noise_weights) {
    return run_noise_products && run_noise && apply_empirical_noise_weights;
}

inline const char *weight_calibration_type(bool empirical_calibration) {
    return empirical_calibration ? "empirical" : "formal";
}

inline const char *weight_map_description(bool empirical_calibration) {
    return empirical_calibration
        ? "Jackknife-calibrated inverse variance weight map"
        : "Formal mapmaker inverse variance weight map";
}

inline const char *not_applicable_image_unit() {
    return "N/A";
}

inline const char *coverage_time_unit() {
    return "sec";
}

inline const char *formal_weight_calibration_type() {
    return "formal";
}

inline const char *pixel_snr_estimator_type() {
    return "pixel";
}

inline const char *point_source_snr_estimator_type() {
    return "point_source";
}

inline const char *convolved_amplitude_estimator_type() {
    return "convolved_amplitude";
}

inline const char *weight_calibration_type_comment() {
    return "Weight calibration type";
}

inline const char *kernel_type_comment() {
    return "Kernel type";
}

inline const char *snr_estimator_type_comment() {
    return "S/N estimator type";
}

inline const char *filtered_amplitude_estimator_type_comment() {
    return "Filtered amplitude estimator type";
}

inline const char *standardized_signal_estimator_type_comment() {
    return "Standardization estimator type";
}

inline double invalid_kernel_fwhm_arcsec() {
    return -99.0;
}

template <class ArrayFwhm>
double kernel_fwhm_arcsec(const std::string &kernel_type,
                          double kernel_fwhm_rad,
                          const ArrayFwhm &array_fwhm,
                          double rad_to_arcsec) {
    if (kernel_type == "fits") {
        return invalid_kernel_fwhm_arcsec();
    }
    if (kernel_fwhm_rad <= 0) {
        return (std::get<0>(array_fwhm) + std::get<1>(array_fwhm)) / 2;
    }
    return kernel_fwhm_rad * rad_to_arcsec;
}

inline bool has_nonfinite_kernel_fwhm(double fwhm_arcsec) {
    return !std::isfinite(fwhm_arcsec);
}

template <class Logger>
double kernel_fwhm_or_invalid(double fwhm_arcsec,
                              const std::string &map_name,
                              const std::string &filepath,
                              const Logger &logger) {
    if (has_nonfinite_kernel_fwhm(fwhm_arcsec)) {
        logger->warn("non-finite kernel FWHM for map {} in {}; using -99",
                     map_name, filepath);
        return invalid_kernel_fwhm_arcsec();
    }
    return fwhm_arcsec;
}

inline const char *signal_map_description() {
    return "Signal map in map units";
}

inline const char *formal_weight_map_description() {
    return "Formal mapmaker inverse variance before empirical calibration";
}

inline const char *noise_variance_map_description() {
    return "Per-pixel variance estimated from jackknife noise maps";
}

inline const char *kernel_map_description() {
    return "Mapmaking or filtering kernel image";
}

inline const char *coverage_map_description() {
    return "Effective integration time coverage map";
}

inline const char *coverage_mask_map_description() {
    return "Boolean valid-coverage support mask";
}

inline const char *legacy_pixel_snr_map_description() {
    return "Legacy alias of jackknife-calibrated pixel S/N";
}

inline const char *pixel_snr_map_description() {
    return "Pixel S/N map: signal times sqrt(empirical weight)";
}

inline const char *formal_standardized_signal_estimator_type() {
    return "formal_weight_standardized";
}

inline const char *formal_standardized_signal_map_description() {
    return "Signal times sqrt(formal mapmaker weight); not a statistical significance map";
}

inline const char *point_source_flux_map_description() {
    return "Point-source flux estimate after filter response normalization";
}

inline const char *point_source_uncertainty_map_description() {
    return "Point-source 1-sigma uncertainty from jackknife maps";
}

inline const char *point_source_snr_map_description() {
    return "Point-source S/N from flux divided by jackknife uncertainty";
}

inline const char *convolved_amplitude_map_description() {
    return "Compatibility alias of convolved map amplitude; no point-source response normalization";
}

inline const char *convolved_amplitude_uncertainty_map_description() {
    return "Jackknife 1-sigma uncertainty of unnormalized convolved map amplitude";
}

inline const char *convolved_amplitude_snr_map_description() {
    return "Convolved map amplitude divided by its jackknife uncertainty";
}
