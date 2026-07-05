#pragma once

// Included by phdu_observation_metadata.h inside namespace citlali::pipeline.

template <class Index, class Logger>
void require_phdu_output_slots(Index i, Index n_files, Index n_arrays,
                               const Logger &logger) {
    if (i < 0 || i >= n_files) {
        logger->error("add_phdu index out of range: i={} fits_io_size={}",
                      static_cast<long long>(i),
                      static_cast<long long>(n_files));
        std::exit(EXIT_FAILURE);
    }
    if (i >= n_arrays) {
        logger->error(
            "add_phdu array index out of range: i={} calib.arrays.size={}",
            static_cast<long long>(i), static_cast<long long>(n_arrays));
        std::exit(EXIT_FAILURE);
    }
}

inline std::string phdu_write_error_message(
    const std::string &array_name, const std::string &filepath,
    const std::string &message) {
    return fmt::format(
        "failed to add PHDU/header for array '{}' (file={}): {}",
        array_name, filepath, message);
}

template <class ArrayFwhm>
double mean_beam_fwhm_arcsec(const ArrayFwhm &array_fwhm) {
    return (std::get<0>(array_fwhm) + std::get<1>(array_fwhm)) / 2;
}

struct PhduUnitConversionFactors {
    double mean_fwhm_arcsec = 0.0;
    double beam_area_sr = 0.0;
    double mjy_beam_to_jy_pixel = 0.0;
};

inline double gaussian_beam_area_sr(double fwhm_arcsec,
                                    double fwhm_to_std,
                                    double arcsec_to_rad,
                                    double pi_value) {
    return 2. * pi_value *
           std::pow(fwhm_arcsec * fwhm_to_std * arcsec_to_rad, 2);
}

inline double mjy_beam_to_jy_pixel_factor(double beam_area_sr,
                                          double pixel_size_rad) {
    return 1e-3 / beam_area_sr * std::pow(pixel_size_rad, 2);
}

template <class ArrayFwhm>
PhduUnitConversionFactors phdu_unit_conversion_factors(
    const ArrayFwhm &array_fwhm, double pixel_size_rad, double fwhm_to_std,
    double arcsec_to_rad, double pi_value) {
    PhduUnitConversionFactors factors;
    factors.mean_fwhm_arcsec = mean_beam_fwhm_arcsec(array_fwhm);
    factors.beam_area_sr = gaussian_beam_area_sr(
        factors.mean_fwhm_arcsec, fwhm_to_std, arcsec_to_rad, pi_value);
    factors.mjy_beam_to_jy_pixel =
        mjy_beam_to_jy_pixel_factor(factors.beam_area_sr, pixel_size_rad);
    return factors;
}

