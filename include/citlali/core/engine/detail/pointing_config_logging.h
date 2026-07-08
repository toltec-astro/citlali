#pragma once

#include <string>

#include <citlali/core/config/pointing_config.h>

namespace citlali::engine_detail {

template <class PtcProc, class Logger>
void log_pointing_config(
    const citlali::config::PointingConfig &pointing_config, const PtcProc &ptcproc,
    const Logger &logger) {
    const auto pointing_source_strategy =
        citlali::config::to_string(pointing_config.source_strategy);
    const auto pointing_fruitloops_center_mode =
        citlali::config::to_string(pointing_config.fruitloops_center_mode);
    logger->info(
        "pointing source strategy: mode={} fit_gaussian={} fruitloops_center_mode={} "
        "header_max_radius_arcsec={} header_require_coverage={}",
        pointing_source_strategy, pointing_config.fit_gaussian,
        pointing_fruitloops_center_mode,
        pointing_config.header_max_radius_arcsec,
        pointing_config.header_require_coverage);

    if (!ptcproc.run_fruit_loops) {
        logger->warn(
            "pointing source strategy is configured but timestream.fruit_loops.enabled=false");
    }
    else if (ptcproc.fruit_loops_iters < 2) {
        logger->warn(
            "pointing source-aware fruit loops uses previous maps; max_iters={} will not run a measurement iteration",
            ptcproc.fruit_loops_iters);
    }

    if (pointing_config.source_strategy ==
            citlali::config::PointingSourceStrategy::psf_preserve &&
        pointing_config.fit_gaussian) {
        logger->warn(
            "pointing.source_strategy.mode=psf_preserve with fit_gaussian=true; "
            "Gaussian fits remain diagnostics only and do not constrain fruit loops");
    }
    if (pointing_config.source_strategy ==
            citlali::config::PointingSourceStrategy::psf_preserve &&
        pointing_config.fruitloops_center_mode ==
            citlali::config::FruitLoopsCenterMode::peak) {
        logger->warn(
            "pointing.source_strategy.mode=psf_preserve with fruitloops_center_mode=peak; "
            "messy out-of-focus maps may bias the fruit loops source support");
    }
    if (!pointing_config.fit_gaussian &&
        (pointing_config.fruitloops_center_mode ==
             citlali::config::FruitLoopsCenterMode::header ||
         pointing_config.fruitloops_center_mode ==
             citlali::config::FruitLoopsCenterMode::automatic)) {
        logger->warn(
            "pointing Gaussian fitting is disabled; later fruit loops iterations will not "
            "get new valid POINTING header centers from this run");
    }
}

}  // namespace citlali::engine_detail
