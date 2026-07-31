#pragma once

#include <citlali/core/pipeline/fits_image_metadata.h>

#include <Eigen/Core>

#include <cmath>
#include <string>

namespace citlali::pipeline {

template <class FitsEntry, class MapBuffer, class Wcs, class Logger>
void add_primary_map_image_hdus(
    FitsEntry &fits_entry, MapBuffer &mb, Eigen::Index i,
    const std::string &map_name, const std::string &stokes_suffix,
    const Wcs &wcs, double source_epoch, bool empirical_weight_calibration,
    bool empirical_noise_products_expected, bool is_beammap,
    const Logger &logger) {
    add_map_hdu_with_wcs(
        fits_entry, signal_map_hdu_name(map_name, stokes_suffix),
        mb->signal[i], wcs, source_epoch);
    add_signal_map_metadata(*fits_entry.hdus.back(), mb->sig_unit);

    add_map_hdu_with_wcs(
        fits_entry, weight_map_hdu_name(map_name, stokes_suffix),
        mb->weight[i], wcs, source_epoch);
    const std::string weight_unit = map_weight_unit(mb->sig_unit);
    add_weight_map_metadata(
        *fits_entry.hdus.back(), weight_unit, empirical_weight_calibration);
    if (empirical_weight_calibration) {
        add_empirical_variance_estimator_keys(
            *fits_entry.hdus.back(),
            static_cast<long long>(mb->n_noise));
    }
    if (empirical_weight_calibration &&
        i < mb->noise_weight_scale.size()) {
        add_empirical_weight_scale_key(
            *fits_entry.hdus.back(), mb->noise_weight_scale(i));
    }
    if (empirical_weight_calibration &&
        i < mb->noise_weight_median_ratio.size()) {
        add_weight_variance_median_key(
            *fits_entry.hdus.back(), mb->noise_weight_median_ratio(i));
    }

    const double median_err_value = mb->median_err(i);
    const double median_err = map_median_error_or_zero_logged(
        median_err_value, is_beammap, map_name, fits_entry.filepath, logger);
    add_image_median_error_key(
        *fits_entry.hdus.back(), median_err, mb->sig_unit);

    if (empirical_noise_products_expected &&
        (!has_map_image_slot(mb->weight_formal, i, mb->n_rows, mb->n_cols) ||
         !has_map_image_slot(mb->noise_variance, i, mb->n_rows, mb->n_cols))) {
        fail_required_output(
            logger,
            fmt::format(
                "empirical noise products were requested but map index {} lacks formal-weight or noise-variance data",
                static_cast<long long>(i)));
    }

    if (empirical_noise_products_expected) {
        add_map_hdu_with_wcs(
            fits_entry, formal_weight_map_hdu_name(map_name, stokes_suffix),
            mb->weight_formal[i], wcs, source_epoch);
        add_formal_weight_map_metadata(*fits_entry.hdus.back(), weight_unit);

        add_map_hdu_with_wcs(
            fits_entry, noise_variance_map_hdu_name(map_name, stokes_suffix),
            mb->noise_variance[i], wcs, source_epoch);
        const std::string variance_unit = map_variance_unit(mb->sig_unit);
        add_noise_variance_map_metadata(
            *fits_entry.hdus.back(), variance_unit);
        add_empirical_variance_estimator_keys(
            *fits_entry.hdus.back(),
            static_cast<long long>(mb->n_noise));
        if (i < mb->median_rms.size() && std::isfinite(mb->median_rms(i))) {
            add_image_median_rms_key(
                *fits_entry.hdus.back(), mb->median_rms(i), mb->sig_unit);
        }
    }
}

template <class FitsEntry, class MapBuffer, class Kernel, class ArrayFwhm,
          class Wcs, class Logger>
void add_kernel_map_image_hdu(
    FitsEntry &fits_entry, MapBuffer &mb, Eigen::Index i,
    const std::string &map_name, const std::string &stokes_suffix,
    const Kernel &kernel, const ArrayFwhm &array_fwhm, const Wcs &wcs,
    double source_epoch, double rad_to_arcsec, const Logger &logger) {
    fits_entry.add_hdu(kernel_map_hdu_name(map_name, stokes_suffix),
                       mb->kernel[i]);
    add_image_type_key(
        *fits_entry.hdus.back(), kernel.type, kernel_type_comment());

    double fwhm = kernel_fwhm_arcsec(
        kernel.type, kernel.fwhm_rad, array_fwhm, rad_to_arcsec);
    fwhm = kernel_fwhm_or_invalid(
        fwhm, map_name, fits_entry.filepath, logger);
    add_kernel_fwhm_key(*fits_entry.hdus.back(), fwhm);
    fits_entry.add_wcs(fits_entry.hdus.back(), wcs, source_epoch);
    add_kernel_map_metadata(*fits_entry.hdus.back(), mb->sig_unit);
}

template <class FitsEntry, class MapBuffer, class Wcs, class Logger>
void add_coverage_support_image_hdus(
    FitsEntry &fits_entry, MapBuffer &mb, Eigen::Index i,
    const std::string &map_name, const std::string &stokes_suffix,
    const Wcs &wcs, double source_epoch, bool is_filtered_output,
    bool empirical_noise_products_expected,
    bool point_source_response_normalized, const Logger &logger) {
    if (mb->coverage.empty()) {
        return;
    }

    add_map_hdu_with_wcs(
        fits_entry, coverage_map_hdu_name(map_name, stokes_suffix),
        mb->coverage[i], wcs, source_epoch);
    add_coverage_map_metadata(*fits_entry.hdus.back());

    auto cov_region = mb->calc_cov_region(i);
    auto weight_threshold = std::get<0>(cov_region);
    weight_threshold = weight_threshold_or_zero_logged(
        weight_threshold, map_name, fits_entry.filepath, logger);
    Eigen::MatrixXd coverage_bool =
        coverage_mask_from_weight(mb->weight[i], weight_threshold);

    add_map_hdu_with_wcs(
        fits_entry, coverage_mask_map_hdu_name(map_name, stokes_suffix),
        coverage_bool, wcs, source_epoch);
    add_coverage_mask_map_metadata(*fits_entry.hdus.back());
    add_image_weight_threshold_key(*fits_entry.hdus.back(), weight_threshold);

    const bool empirical_snr_available = has_map_image_slot(
        mb->sig2noise_pixel, i, mb->n_rows, mb->n_cols);
    if (empirical_noise_products_expected && !empirical_snr_available) {
        fail_required_output(
            logger,
            fmt::format(
                "empirical noise products were requested but map index {} lacks pixel S/N data",
                static_cast<long long>(i)));
    }
    if (empirical_noise_products_expected) {
        Eigen::MatrixXd &sig2noise = mb->sig2noise_pixel[i];
        add_map_hdu_with_wcs(
            fits_entry, legacy_pixel_snr_map_hdu_name(map_name, stokes_suffix),
            sig2noise, wcs, source_epoch);
        add_legacy_pixel_snr_map_metadata(*fits_entry.hdus.back());
        add_empirical_variance_estimator_keys(
            *fits_entry.hdus.back(),
            static_cast<long long>(mb->n_noise));

        add_map_hdu_with_wcs(
            fits_entry, pixel_snr_map_hdu_name(map_name, stokes_suffix),
            sig2noise, wcs, source_epoch);
        add_pixel_snr_map_metadata(*fits_entry.hdus.back());
        add_empirical_variance_estimator_keys(
            *fits_entry.hdus.back(),
            static_cast<long long>(mb->n_noise));
    }
    else {
        Eigen::MatrixXd formal_standardized_signal =
            standardized_signal_from_weight(mb->signal[i], mb->weight[i]);
        add_map_hdu_with_wcs(
            fits_entry,
            formal_standardized_signal_map_hdu_name(
                map_name, stokes_suffix),
            formal_standardized_signal, wcs, source_epoch);
        add_formal_standardized_signal_map_metadata(
            *fits_entry.hdus.back());
    }

    const bool point_source_products_available =
        has_map_image_slot(
            mb->point_source_uncertainty, i, mb->n_rows, mb->n_cols) &&
        has_map_image_slot(
            mb->sig2noise_point_source, i, mb->n_rows, mb->n_cols);
    if (is_filtered_output && empirical_noise_products_expected &&
        !point_source_products_available) {
        fail_required_output(
            logger,
            fmt::format(
                "empirical noise products were requested for filtered map index {} but point-source uncertainty or S/N data are absent",
                static_cast<long long>(i)));
    }
    if (is_filtered_output && empirical_noise_products_expected) {
        add_map_hdu_with_wcs(
            fits_entry, point_source_flux_map_hdu_name(
                map_name, stokes_suffix),
            mb->signal[i], wcs, source_epoch);
        add_point_source_flux_map_metadata(
            *fits_entry.hdus.back(), mb->sig_unit,
            point_source_response_normalized);
        if (point_source_response_normalized) {
            add_point_source_response_norm_key(*fits_entry.hdus.back(), 1.0);
        }

        add_map_hdu_with_wcs(
            fits_entry, point_source_uncertainty_map_hdu_name(
                map_name, stokes_suffix),
            mb->point_source_uncertainty[i], wcs, source_epoch);
        add_point_source_uncertainty_map_metadata(
            *fits_entry.hdus.back(), mb->sig_unit,
            point_source_response_normalized);
        add_empirical_variance_estimator_keys(
            *fits_entry.hdus.back(),
            static_cast<long long>(mb->n_noise));

        add_map_hdu_with_wcs(
            fits_entry, point_source_snr_map_hdu_name(
                map_name, stokes_suffix),
            mb->sig2noise_point_source[i], wcs, source_epoch);
        add_point_source_snr_map_metadata(
            *fits_entry.hdus.back(), point_source_response_normalized);
        add_empirical_variance_estimator_keys(
            *fits_entry.hdus.back(),
            static_cast<long long>(mb->n_noise));
    }
}

template <class FitsEntry, class MapBuffer, class Wcs>
void add_noise_realization_image_hdus(
    FitsEntry &fits_entry, MapBuffer &mb, Eigen::Index i,
    const std::string &map_name, const std::string &stokes_suffix,
    const Wcs &wcs, double source_epoch, double median_rms) {
    for (Eigen::Index n = 0; n < mb->n_noise; ++n) {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
            noise_matrix(
                mb->noise[i].data() + n * mb->n_rows * mb->n_cols,
                mb->n_rows, mb->n_cols);

        add_map_hdu_with_wcs(
            fits_entry, noise_signal_map_hdu_name(map_name, n, stokes_suffix),
            noise_matrix, wcs, source_epoch);
        add_noise_image_summary_keys(
            *fits_entry.hdus.back(), mb->sig_unit, median_rms);
    }
}

}  // namespace citlali::pipeline
