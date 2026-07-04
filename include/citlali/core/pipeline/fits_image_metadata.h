#pragma once

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <string>

namespace citlali::pipeline {

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

template <class ImageList>
bool has_map_image_slot(const ImageList &images, Eigen::Index i,
                        Eigen::Index n_rows, Eigen::Index n_cols) {
    return i < static_cast<Eigen::Index>(images.size()) &&
           images[i].rows() == n_rows &&
           images[i].cols() == n_cols;
}

inline double map_median_error_or_zero(double median_error_variance,
                                       bool is_beammap) {
    if (is_beammap) {
        return 0.0;
    }
    if (std::isfinite(median_error_variance) &&
        median_error_variance > std::numeric_limits<double>::epsilon()) {
        return std::sqrt(median_error_variance);
    }
    return 0.0;
}

inline bool has_negative_map_median_error(double median_error_variance,
                                          bool is_beammap) {
    return !is_beammap && std::isfinite(median_error_variance) &&
           median_error_variance < 0.0;
}

template <class Hdu>
void add_image_unit_keys(Hdu &hdu, const std::string &unit) {
    hdu.addKey("UNIT", unit, "Unit of map");
    hdu.addKey("BUNIT", unit, "Physical unit of image values");
}

template <class Hdu>
void add_image_description_key(Hdu &hdu, const std::string &description) {
    hdu.addKey("DESCRIP", description, "Image product description");
}

template <class Hdu>
void add_image_type_key(Hdu &hdu, const std::string &type,
                        const std::string &comment) {
    hdu.addKey("TYPE", type, comment);
}

template <class Hdu>
void add_image_type_description_keys(Hdu &hdu, const std::string &type,
                                     const std::string &type_comment,
                                     const std::string &description) {
    add_image_type_key(hdu, type, type_comment);
    add_image_description_key(hdu, description);
}

template <class Hdu>
void add_image_unit_type_description_keys(Hdu &hdu, const std::string &unit,
                                          const std::string &type,
                                          const std::string &type_comment,
                                          const std::string &description) {
    add_image_unit_keys(hdu, unit);
    add_image_type_description_keys(hdu, type, type_comment, description);
}

template <class Hdu>
void add_image_median_error_key(Hdu &hdu, double median_error,
                                const std::string &unit) {
    hdu.addKey("MEDERR", median_error, "Median Error (" + unit + ")");
}

template <class Hdu>
void add_image_weight_threshold_key(Hdu &hdu, double weight_threshold) {
    hdu.addKey("WTTHRESH", weight_threshold, "Weight threshold");
}

template <class Hdu>
void add_empirical_weight_scale_key(Hdu &hdu, double scale) {
    hdu.addKey("EMP_SCALE", scale, "Empirical weight scale");
}

template <class Hdu>
void add_weight_variance_median_key(Hdu &hdu, double median_ratio) {
    hdu.addKey("WVARMED", median_ratio,
               "Median formal weight times jackknife variance");
}

template <class Hdu>
void add_point_source_response_norm_key(Hdu &hdu, double response_norm) {
    hdu.addKey("RESPNORM", response_norm,
               "Point-source response normalization applied");
}

template <class Hdu>
void add_kernel_fwhm_key(Hdu &hdu, double fwhm_arcsec) {
    hdu.addKey("FWHM", fwhm_arcsec, "Kernel fwhm (arcsec)");
}

template <class Hdu>
void add_noise_image_summary_keys(Hdu &hdu, const std::string &unit,
                                  double median_rms) {
    hdu.addKey("UNIT", unit, "Unit of map");
    hdu.addKey("MEDRMS", median_rms, "Median RMS of noise maps");
}

template <class Hdu>
void add_image_unit_description_keys(Hdu &hdu, const std::string &unit,
                                     const std::string &description) {
    add_image_unit_keys(hdu, unit);
    add_image_description_key(hdu, description);
}

}  // namespace citlali::pipeline
