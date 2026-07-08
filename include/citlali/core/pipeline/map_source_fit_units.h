#pragma once

// Included by map_source_finding.h inside namespace citlali::pipeline.

inline double source_fit_initial_fwhm_pixels(
    double array_fwhm_arcsec, double arcsec_to_rad, double pixel_size_rad) {
    return array_fwhm_arcsec * arcsec_to_rad / pixel_size_rad;
}

template <class ArrayFwhm, class ArrayIndex>
double source_fit_initial_fwhm_for_array(
    ArrayFwhm &array_fwhm_arcsec, ArrayIndex array_index,
    double arcsec_to_rad, double pixel_size_rad) {
    return source_fit_initial_fwhm_pixels(
        array_fwhm_arcsec[array_index], arcsec_to_rad, pixel_size_rad);
}

inline double source_fit_pixel_to_arcsec(double rad_to_arcsec,
                                         double pixel_size_rad) {
    return rad_to_arcsec * pixel_size_rad;
}

inline double source_fit_fwhm_to_arcsec(double rad_to_arcsec,
                                        double std_to_fwhm,
                                        double pixel_size_rad) {
    return rad_to_arcsec * std_to_fwhm * pixel_size_rad;
}

inline SourceFitUnitScale source_fit_unit_scale(double rad_to_arcsec,
                                                double std_to_fwhm,
                                                double pixel_size_rad) {
    return {
        source_fit_pixel_to_arcsec(rad_to_arcsec, pixel_size_rad),
        source_fit_fwhm_to_arcsec(rad_to_arcsec, std_to_fwhm,
                                  pixel_size_rad)};
}

inline SourceFitUnitScale source_fit_unit_scale(
    const SourceFitUnitConstants &constants, double pixel_size_rad) {
    return source_fit_unit_scale(
        constants.rad_to_arcsec, constants.std_to_fwhm, pixel_size_rad);
}

inline SourceFitUnitConstants source_fit_unit_constants(
    double rad_to_arcsec, double std_to_fwhm, double arcsec_to_rad,
    double rad_to_deg, double deg_to_rad, double arcsec_to_deg) {
    return {
        rad_to_arcsec,
        std_to_fwhm,
        arcsec_to_rad,
        rad_to_deg,
        deg_to_rad,
        arcsec_to_deg};
}

template <class Params, class PErrors>
void rescale_source_fit_pixel_units(Params &params, PErrors &perrors,
                                    Eigen::Index n_rows, Eigen::Index n_cols,
                                    double pixel_to_arcsec,
                                    double source_fwhm_to_arcsec) {
    params(1) = pixel_to_arcsec * (params(1) - (n_cols - 1) / 2.0);
    params(2) = pixel_to_arcsec * (params(2) - (n_rows - 1) / 2.0);
    params(3) = source_fwhm_to_arcsec * params(3);
    params(4) = source_fwhm_to_arcsec * params(4);

    perrors(1) = pixel_to_arcsec * perrors(1);
    perrors(2) = pixel_to_arcsec * perrors(2);
    perrors(3) = source_fwhm_to_arcsec * perrors(3);
    perrors(4) = source_fwhm_to_arcsec * perrors(4);
}

inline bool source_fit_uses_radec_projection(
    citlali::config::MapPixelAxes pixel_axes) {
    return citlali::config::is_radec_map_pixel_axes(pixel_axes);
}

template <class Params, class PErrors>
void rescale_source_fit_radec_errors(Params &params, PErrors &perrors,
                                     double ra_deg, double dec_deg,
                                     double arcsec_to_deg) {
    params(1) = ra_deg;
    params(2) = dec_deg;

    perrors(1) = perrors(1) * arcsec_to_deg;
    perrors(2) = perrors(2) * arcsec_to_deg;
}

template <class Params, class PErrors, class Wcs, class TangentToAbs>
void rescale_source_fit_result(
    Params &params, PErrors &perrors, Eigen::Index n_rows,
    Eigen::Index n_cols, double pixel_size_rad,
    citlali::config::MapPixelAxes pixel_axes, const Wcs &wcs,
    const SourceFitUnitConstants &constants,
    const TangentToAbs &tangent_to_abs) {
    const auto unit_scale =
        source_fit_unit_scale(constants, pixel_size_rad);
    rescale_source_fit_pixel_units(
        params, perrors, n_rows, n_cols, unit_scale.pixel_to_arcsec,
        unit_scale.source_fwhm_to_arcsec);

    if (!source_fit_uses_radec_projection(pixel_axes)) {
        return;
    }

    Eigen::VectorXd lat(1), lon(1);
    lat << params(2) * constants.arcsec_to_rad;
    lon << params(1) * constants.arcsec_to_rad;

    auto [adec, ara] = tangent_to_abs(
        lat, lon, wcs.crval[0] * constants.deg_to_rad,
        wcs.crval[1] * constants.deg_to_rad);

    rescale_source_fit_radec_errors(
        params, perrors, ara(0) * constants.rad_to_deg,
        adec(0) * constants.rad_to_deg, constants.arcsec_to_deg);
}
