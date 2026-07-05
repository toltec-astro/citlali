#pragma once

#include <string>

namespace citlali::engine_detail {

template <class PtcProc, class Logger>
void log_pointing_config(
    const std::string &pointing_source_strategy,
    bool pointing_fit_gaussian_enabled,
    const std::string &pointing_fruitloops_center_mode,
    double pointing_header_center_max_radius_arcsec,
    bool pointing_header_center_require_coverage, const PtcProc &ptcproc,
    const Logger &logger) {
    logger->info(
        "pointing source strategy: mode={} fit_gaussian={} fruitloops_center_mode={} "
        "header_max_radius_arcsec={} header_require_coverage={}",
        pointing_source_strategy, pointing_fit_gaussian_enabled,
        pointing_fruitloops_center_mode,
        pointing_header_center_max_radius_arcsec,
        pointing_header_center_require_coverage);

    if (!ptcproc.run_fruit_loops) {
        logger->warn(
            "pointing source strategy is configured but timestream.fruit_loops.enabled=false");
    }
    else if (ptcproc.fruit_loops_iters < 2) {
        logger->warn(
            "pointing source-aware fruit loops uses previous maps; max_iters={} will not run a measurement iteration",
            ptcproc.fruit_loops_iters);
    }

    if (pointing_source_strategy == "psf_preserve" &&
        pointing_fit_gaussian_enabled) {
        logger->warn(
            "pointing.source_strategy.mode=psf_preserve with fit_gaussian=true; "
            "Gaussian fits remain diagnostics only and do not constrain fruit loops");
    }
    if (pointing_source_strategy == "psf_preserve" &&
        pointing_fruitloops_center_mode == "peak") {
        logger->warn(
            "pointing.source_strategy.mode=psf_preserve with fruitloops_center_mode=peak; "
            "messy out-of-focus maps may bias the fruit loops source support");
    }
    if (!pointing_fit_gaussian_enabled &&
        (pointing_fruitloops_center_mode == "header" ||
         pointing_fruitloops_center_mode == "auto")) {
        logger->warn(
            "pointing Gaussian fitting is disabled; later fruit loops iterations will not "
            "get new valid POINTING header centers from this run");
    }
}

}  // namespace citlali::engine_detail
