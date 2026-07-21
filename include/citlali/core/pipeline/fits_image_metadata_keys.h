#pragma once

// Included by fits_image_metadata.h inside namespace citlali::pipeline.

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
void add_image_median_rms_key(Hdu &hdu, double median_rms,
                              const std::string &unit) {
    hdu.addKey("MEDRMS", median_rms,
               "Median jackknife-map RMS (" + unit + ")");
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
void add_formal_standardized_signal_map_metadata(Hdu &hdu) {
    add_image_unit_type_description_keys(
        hdu, not_applicable_image_unit(),
        formal_standardized_signal_estimator_type(),
        standardized_signal_estimator_type_comment(),
        formal_standardized_signal_map_description());
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
