#pragma once

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <string>
#include <tuple>

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

inline const char *weight_calibration_type_comment() {
    return "Weight calibration type";
}

inline const char *kernel_type_comment() {
    return "Kernel type";
}

inline const char *snr_estimator_type_comment() {
    return "S/N estimator type";
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
    return "Legacy pixel S/N: signal times sqrt(weight)";
}

inline const char *pixel_snr_map_description() {
    return "Pixel S/N map: signal times sqrt(empirical weight)";
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

inline std::string legacy_pixel_snr_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "sig2noise_" + map_name + stokes_suffix;
}

inline std::string pixel_snr_map_hdu_name(const std::string &map_name,
                                          const std::string &stokes_suffix) {
    return "sig2noise_pixel_" + map_name + stokes_suffix;
}

inline std::string point_source_flux_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "point_source_flux_" + map_name + stokes_suffix;
}

inline std::string point_source_uncertainty_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "point_source_uncertainty_" + map_name + stokes_suffix;
}

inline std::string point_source_snr_map_hdu_name(
    const std::string &map_name, const std::string &stokes_suffix) {
    return "sig2noise_point_source_" + map_name + stokes_suffix;
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

template <class ArrayFreqMap, class Arrays>
double map_wcs_frequency(ArrayFreqMap &array_freq_map, const Arrays &arrays,
                         Eigen::Index array_index) {
    return array_freq_map[arrays[array_index]];
}

template <class ImageList>
bool has_map_image_slot(const ImageList &images, Eigen::Index i,
                        Eigen::Index n_rows, Eigen::Index n_cols) {
    return i < static_cast<Eigen::Index>(images.size()) &&
           images[i].rows() == n_rows &&
           images[i].cols() == n_cols;
}

template <class FitsIo, class FitsIoContainer>
bool is_filtered_map_output(const FitsIo &fits_io,
                            const FitsIoContainer &filtered_fits_io,
                            const FitsIoContainer &filtered_coadd_fits_io) {
    return fits_io == &filtered_fits_io || fits_io == &filtered_coadd_fits_io;
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

template <class MedianRms>
double map_median_rms_or_zero(const MedianRms &median_rms, Eigen::Index i) {
    if (i < static_cast<Eigen::Index>(median_rms.size()) &&
        std::isfinite(median_rms(i))) {
        return median_rms(i);
    }
    return 0.0;
}

template <class MedianRms>
bool has_nonfinite_map_median_rms(const MedianRms &median_rms,
                                  Eigen::Index i) {
    return i < static_cast<Eigen::Index>(median_rms.size()) &&
           !std::isfinite(median_rms(i));
}

inline bool has_nonfinite_weight_threshold(double weight_threshold) {
    return !std::isfinite(weight_threshold);
}

inline double weight_threshold_or_zero(double weight_threshold) {
    return has_nonfinite_weight_threshold(weight_threshold) ? 0.0
                                                            : weight_threshold;
}

template <class Matrix>
Eigen::MatrixXd coverage_mask_from_weight(const Matrix &weight,
                                          double weight_threshold) {
    Eigen::MatrixXd ones;
    Eigen::MatrixXd zeros;
    ones.setOnes(weight.rows(), weight.cols());
    zeros.setZero(weight.rows(), weight.cols());
    return (weight.array() < weight_threshold).select(zeros, ones);
}

template <class Matrix>
Eigen::MatrixXd pixel_snr_from_signal_weight(const Matrix &signal,
                                             const Matrix &weight) {
    return signal.array() * weight.array().sqrt();
}

template <class ImageList, class Matrix>
Eigen::MatrixXd pixel_snr_image_or_fallback(const ImageList &pixel_snr_images,
                                            Eigen::Index i,
                                            Eigen::Index n_rows,
                                            Eigen::Index n_cols,
                                            const Matrix &signal,
                                            const Matrix &weight) {
    if (has_map_image_slot(pixel_snr_images, i, n_rows, n_cols)) {
        return pixel_snr_images[i];
    }
    return pixel_snr_from_signal_weight(signal, weight);
}

template <class NoiseList, class FitsIo>
bool should_write_noise_maps(const NoiseList &noise,
                             const FitsIo &noise_fits_io) {
    return !noise.empty() && !noise_fits_io->empty();
}

template <class FitsIo>
bool has_noise_fits_slot(const FitsIo &noise_fits_io,
                         Eigen::Index map_index) {
    return map_index >= 0 &&
           map_index < static_cast<Eigen::Index>(noise_fits_io->size());
}

template <class NoiseList>
bool has_noise_map_slot(const NoiseList &noise, Eigen::Index i) {
    return i >= 0 && i < static_cast<Eigen::Index>(noise.size());
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

template <class Hdu>
void add_signal_map_metadata(Hdu &hdu, const std::string &signal_unit) {
    add_image_unit_description_keys(hdu, signal_unit,
                                    signal_map_description());
}

template <class Hdu>
void add_weight_map_metadata(Hdu &hdu, const std::string &weight_unit,
                             bool empirical_weight_calibration) {
    add_image_unit_type_description_keys(
        hdu, weight_unit,
        weight_calibration_type(empirical_weight_calibration),
        weight_calibration_type_comment(),
        weight_map_description(empirical_weight_calibration));
}

template <class Hdu>
void add_formal_weight_map_metadata(Hdu &hdu,
                                    const std::string &weight_unit) {
    add_image_unit_type_description_keys(
        hdu, weight_unit, formal_weight_calibration_type(),
        weight_calibration_type_comment(), formal_weight_map_description());
}

template <class Hdu>
void add_noise_variance_map_metadata(Hdu &hdu,
                                     const std::string &variance_unit) {
    add_image_unit_description_keys(hdu, variance_unit,
                                    noise_variance_map_description());
}

template <class Hdu>
void add_kernel_map_metadata(Hdu &hdu, const std::string &signal_unit) {
    add_image_unit_description_keys(hdu, signal_unit,
                                    kernel_map_description());
}

template <class Hdu>
void add_coverage_map_metadata(Hdu &hdu) {
    add_image_unit_description_keys(hdu, coverage_time_unit(),
                                    coverage_map_description());
}

template <class Hdu>
void add_coverage_mask_map_metadata(Hdu &hdu) {
    add_image_unit_description_keys(hdu, not_applicable_image_unit(),
                                    coverage_mask_map_description());
}

template <class Hdu>
void add_legacy_pixel_snr_map_metadata(Hdu &hdu) {
    add_image_unit_type_description_keys(
        hdu, not_applicable_image_unit(), pixel_snr_estimator_type(),
        snr_estimator_type_comment(), legacy_pixel_snr_map_description());
}

template <class Hdu>
void add_pixel_snr_map_metadata(Hdu &hdu) {
    add_image_unit_type_description_keys(
        hdu, not_applicable_image_unit(), pixel_snr_estimator_type(),
        snr_estimator_type_comment(), pixel_snr_map_description());
}

template <class Hdu>
void add_point_source_flux_map_metadata(Hdu &hdu,
                                        const std::string &signal_unit) {
    add_image_unit_description_keys(hdu, signal_unit,
                                    point_source_flux_map_description());
}

template <class Hdu>
void add_point_source_uncertainty_map_metadata(
    Hdu &hdu, const std::string &signal_unit) {
    add_image_unit_description_keys(
        hdu, signal_unit, point_source_uncertainty_map_description());
}

template <class Hdu>
void add_point_source_snr_map_metadata(Hdu &hdu) {
    add_image_unit_type_description_keys(
        hdu, not_applicable_image_unit(), point_source_snr_estimator_type(),
        snr_estimator_type_comment(), point_source_snr_map_description());
}

}  // namespace citlali::pipeline
