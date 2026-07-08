#pragma once

// Included by phdu_reduction_config.h inside namespace citlali::pipeline.

template <class FitsEntry, class PtcProc, class Logger>
void add_phdu_fruit_loops_config(FitsEntry &fits_entry,
                                 const std::string &array_name,
                                 const Logger &logger,
                                 const PtcProc &ptcproc,
                                 double flux_limit,
                                 const std::string &signal_unit) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    hdu.addKey("CONFIG.FRUITLOOPS", ptcproc.run_fruit_loops, "Fruit loops");
    hdu.addKey("CONFIG.FRUITLOOPS.PATH", ptcproc.fruit_loops_path,
               "Fruit loops path");
    hdu.addKey("CONFIG.FRUITLOOPS.TYPE", ptcproc.fruit_loops_type,
               "Fruit loops type");
    hdu.addKey("CONFIG.FRUITLOOPS.SRCMODE",
               ptcproc.fruit_loops_source_center_mode,
               "Fruit loops source center mode");
    add_double_key("CONFIG.FRUITLOOPS.HDRMAXR",
                   ptcproc.fruit_loops_header_center_max_radius_arcsec,
                   "Fruit loops header center max radius");
    hdu.addKey("CONFIG.FRUITLOOPS.HDRCOV",
               ptcproc.fruit_loops_header_center_require_coverage,
               "Require coverage at header center");
    add_double_key("CONFIG.FRUITLOOPS.S2N",
                   ptcproc.fruit_loops_sig2noise,
                   "Fruit loops S/N");
    add_double_key("CONFIG.FRUITLOOPS.PEAKFRAC",
                   ptcproc.fruit_loops_peak_fraction_limit,
                   "Fruit loops peak fraction");
    add_double_key("CONFIG.FRUITLOOPS.LOCALSNR",
                   ptcproc.fruit_loops_local_snr_floor,
                   "Fruit loops local sigma S/N floor");
    add_double_key("CONFIG.FRUITLOOPS.LOCALSIG_INNER",
                   ptcproc.fruit_loops_local_sigma_inner_radius_arcsec,
                   "Fruit loops local sigma inner annulus");
    add_double_key("CONFIG.FRUITLOOPS.LOCALSIG_OUTER",
                   ptcproc.fruit_loops_local_sigma_outer_radius_arcsec,
                   "Fruit loops local sigma outer annulus");
    add_double_key("CONFIG.FRUITLOOPS.LOCALSIG_EDGE",
                   ptcproc.fruit_loops_local_sigma_edge_guard_arcsec,
                   "Fruit loops local sigma edge guard");
    hdu.addKey("CONFIG.FRUITLOOPS.LOCALSIG_MINPIX",
               ptcproc.fruit_loops_local_sigma_min_pixels,
               "Fruit loops local sigma minimum pixels");
    add_double_key("CONFIG.FRUITLOOPS.ADAPT_SUPPORT_RAD",
                   ptcproc.fruit_loops_adaptive_support_radius_arcsec,
                   "Fruit loops adaptive support radius");
    add_double_key("CONFIG.FRUITLOOPS.ADAPT_SUPPORT_FWHM",
                   ptcproc.fruit_loops_adaptive_support_radius_fwhm,
                   "Fruit loops adaptive support FWHM factor");
    hdu.addKey("CONFIG.FRUITLOOPS.WFB",
               ptcproc.fruit_loops_weight_feedback_enabled,
               "Fruit loops weight feedback");
    hdu.addKey("CONFIG.FRUITLOOPS.WFBREF",
               ptcproc.fruit_loops_weight_feedback_reference,
               "Fruit loops weight feedback reference");
    add_double_key("CONFIG.FRUITLOOPS.WFBLOW",
                   ptcproc.fruit_loops_weight_feedback_low_relative_weight,
                   "Fruit loops weight feedback low relative weight");
    add_double_key("CONFIG.FRUITLOOPS.WFBHIGH",
                   ptcproc.fruit_loops_weight_feedback_high_relative_weight,
                   "Fruit loops weight feedback high relative weight");
    add_double_key("CONFIG.FRUITLOOPS.FLUX", flux_limit,
                   "Fruit loops flux (" + signal_unit + ")");
    hdu.addKey("CONFIG.FRUITLOOPS.MAXITER", ptcproc.fruit_loops_iters,
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
