#pragma once

#include <citlali/core/pipeline/fits_image_metadata.h>

#include <Eigen/Core>

#include <string>

namespace citlali::pipeline {

template <class FitsEntry, class MapBuffer, class Wcs, class Logger>
void add_primary_map_image_hdus(
    FitsEntry &fits_entry, MapBuffer &mb, Eigen::Index i,
    const std::string &map_name, const std::string &stokes_suffix,
    const Wcs &wcs, double source_epoch, bool run_noise_products,
    bool run_noise, bool apply_empirical_noise_weights, bool is_beammap,
    const Logger &logger) {
    add_map_hdu_with_wcs(
        fits_entry, signal_map_hdu_name(map_name, stokes_suffix),
        mb->signal[i], wcs, source_epoch);
    add_signal_map_metadata(*fits_entry.hdus.back(), mb->sig_unit);

    add_map_hdu_with_wcs(
        fits_entry, weight_map_hdu_name(map_name, stokes_suffix),
        mb->weight[i], wcs, source_epoch);
    const std::string weight_unit = map_weight_unit(mb->sig_unit);
    const bool empirical_weight_calibration =
        empirical_weight_calibration_enabled(
            run_noise_products, run_noise, apply_empirical_noise_weights);
    add_weight_map_metadata(
        *fits_entry.hdus.back(), weight_unit, empirical_weight_calibration);
    if (i < mb->noise_weight_scale.size()) {
        add_empirical_weight_scale_key(
            *fits_entry.hdus.back(), mb->noise_weight_scale(i));
    }
    if (i < mb->noise_weight_median_ratio.size()) {
        add_weight_variance_median_key(
            *fits_entry.hdus.back(), mb->noise_weight_median_ratio(i));
    }

    const double median_err_value = mb->median_err(i);
    const double median_err = map_median_error_or_zero_logged(
        median_err_value, is_beammap, map_name, fits_entry.filepath, logger);
    add_image_median_error_key(
        *fits_entry.hdus.back(), median_err, mb->sig_unit);

    if (has_map_image_slot(mb->weight_formal, i, mb->n_rows, mb->n_cols)) {
        add_map_hdu_with_wcs(
            fits_entry, formal_weight_map_hdu_name(map_name, stokes_suffix),
            mb->weight_formal[i], wcs, source_epoch);
        add_formal_weight_map_metadata(*fits_entry.hdus.back(), weight_unit);
    }

    if (has_map_image_slot(mb->noise_variance, i, mb->n_rows, mb->n_cols)) {
        add_map_hdu_with_wcs(
            fits_entry, noise_variance_map_hdu_name(map_name, stokes_suffix),
            mb->noise_variance[i], wcs, source_epoch);
        const std::string variance_unit = map_variance_unit(mb->sig_unit);
        add_noise_variance_map_metadata(
            *fits_entry.hdus.back(), variance_unit);
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

}  // namespace citlali::pipeline
