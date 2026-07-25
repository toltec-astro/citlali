#pragma once

// Included by phdu_reduction_config.h inside namespace citlali::pipeline.

template <class FitsEntry, class Logger>
void add_phdu_fruit_loops_config(FitsEntry &fits_entry,
                                 const std::string &array_name,
                                 const Logger &logger,
                                 const citlali::config::TimestreamFruitLoopsConfig &config,
                                 const citlali::config::PointingConfig &pointing_config,
                                 citlali::config::ReductionType reduction_type,
                                 double flux_limit,
                                 const std::string &signal_unit) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    hdu.addKey("CONFIG.FRUITLOOPS", config.enabled, "Fruit loops");
    hdu.addKey("CONFIG.FRUITLOOPS.PATH", config.path,
               "Fruit loops path");
    hdu.addKey("CONFIG.FRUITLOOPS.RESTART_PATH", config.restart_path,
               "Exact fruit-loops restart reduction path");
    hdu.addKey("CONFIG.FRUITLOOPS.TYPE", config.type,
               "Fruit loops type");
    const auto source_center_mode =
        reduction_type == citlali::config::ReductionType::pointing
            ? std::string{citlali::config::to_string(
                  pointing_config.fruitloops_center_mode)}
            : std::string{citlali::config::to_string(
                  config.source_center_mode)};
    hdu.addKey("CONFIG.FRUITLOOPS.SRCMODE",
               source_center_mode,
               "Fruit loops source center mode");
    add_double_key("CONFIG.FRUITLOOPS.HDRMAXR",
                   pointing_config.header_max_radius_arcsec,
                   "Fruit loops header center max radius");
    hdu.addKey("CONFIG.FRUITLOOPS.HDRCOV",
               pointing_config.header_require_coverage,
               "Require coverage at header center");
    add_double_key("CONFIG.FRUITLOOPS.S2N",
                   config.sig2noise_limit,
                   "Fruit loops S/N");
    add_double_key("CONFIG.FRUITLOOPS.PEAKFRAC",
                   config.peak_fraction_limit,
                   "Fruit loops peak fraction");
    add_double_key("CONFIG.FRUITLOOPS.LOCALSNR",
                   config.local_snr_floor,
                   "Fruit loops local sigma S/N floor");
    add_double_key("CONFIG.FRUITLOOPS.LOCALSIG_INNER",
                   config.local_sigma_inner_radius_arcsec,
                   "Fruit loops local sigma inner annulus");
    add_double_key("CONFIG.FRUITLOOPS.LOCALSIG_OUTER",
                   config.local_sigma_outer_radius_arcsec,
                   "Fruit loops local sigma outer annulus");
    add_double_key("CONFIG.FRUITLOOPS.LOCALSIG_EDGE",
                   config.local_sigma_edge_guard_arcsec,
                   "Fruit loops local sigma edge guard");
    hdu.addKey("CONFIG.FRUITLOOPS.LOCALSIG_MINPIX",
               config.local_sigma_min_pixels,
               "Fruit loops local sigma minimum pixels");
    add_double_key("CONFIG.FRUITLOOPS.ADAPT_SUPPORT_RAD",
                   config.adaptive_support_radius_arcsec,
                   "Fruit loops adaptive support radius");
    add_double_key("CONFIG.FRUITLOOPS.ADAPT_SUPPORT_FWHM",
                   config.adaptive_support_radius_fwhm,
                   "Fruit loops adaptive support FWHM factor");
    hdu.addKey("CONFIG.FRUITLOOPS.WFB",
               config.weight_feedback.enabled,
               "Fruit loops weight feedback");
    hdu.addKey("CONFIG.FRUITLOOPS.WFBREF",
               std::string{citlali::config::to_string(
                   config.weight_feedback.reference)},
               "Fruit loops weight feedback reference");
    add_double_key("CONFIG.FRUITLOOPS.WFBLOW",
                   config.weight_feedback.low_relative_weight,
                   "Fruit loops weight feedback low relative weight");
    add_double_key("CONFIG.FRUITLOOPS.WFBHIGH",
                   config.weight_feedback.high_relative_weight,
                   "Fruit loops weight feedback high relative weight");
    hdu.addKey("CONFIG.FRUITLOOPS.INJECT",
               config.injected_source_test.enabled,
               "Diagnostic injected-source transfer test");
    hdu.addKey("CONFIG.FRUITLOOPS.INJITER",
               config.injected_source_test.start_iteration,
               "Injected-source first zero-based iteration");
    double injected_amplitude = 0.0;
    const int array_position =
        array_name == "a1100" ? 0 :
        array_name == "a1400" ? 1 :
        array_name == "a2000" ? 2 : -1;
    if (array_position >= 0 &&
        static_cast<std::size_t>(array_position) <
            config.injected_source_test.array_amplitude_mjy_beam.size()) {
        injected_amplitude =
            config.injected_source_test.array_amplitude_mjy_beam[
                static_cast<std::size_t>(array_position)];
    }
    add_double_key("CONFIG.FRUITLOOPS.INJAMP",
                   injected_amplitude,
                   "Injected source amplitude (mJy/beam)");
    add_double_key("CONFIG.FRUITLOOPS.FLUX", flux_limit,
                   "Fruit loops flux (" + signal_unit + ")");
    hdu.addKey("CONFIG.FRUITLOOPS.MAXITER", config.max_iters,
               "Fruit loops iterations");
}

template <class FitsEntry, class Logger>
void add_phdu_pointing_config(
    FitsEntry &fits_entry, const std::string &array_name,
    const Logger &logger,
    const citlali::config::PointingConfig &pointing_config) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    hdu.addKey("CONFIG.POINTING.STRATEGY",
               std::string(citlali::config::to_string(
                   pointing_config.source_strategy)),
               "Pointing source strategy");
    hdu.addKey("CONFIG.POINTING.FITGAUSS", pointing_config.fit_gaussian,
               "Pointing Gaussian fit enabled");
    hdu.addKey("CONFIG.POINTING.SRCMODE",
               std::string(citlali::config::to_string(
                   pointing_config.fruitloops_center_mode)),
               "Pointing fruit loops source mode");
    add_double_key("CONFIG.POINTING.HDRMAXR",
                   pointing_config.header_max_radius_arcsec,
                   "Pointing header center max radius");
    hdu.addKey("CONFIG.POINTING.HDRCOV",
               pointing_config.header_require_coverage,
               "Pointing header coverage guard");
}

template <class FitsEntry, class Logger>
void add_phdu_pointing_config_if_needed(
    FitsEntry &fits_entry, const std::string &array_name,
    const Logger &logger, citlali::config::ReductionType reduction_type,
    const citlali::config::PointingConfig &pointing_config) {
    if (!citlali::config::is_pointing_reduction_type(reduction_type)) {
        return;
    }
    add_phdu_pointing_config(
        fits_entry, array_name, logger, pointing_config);
}
