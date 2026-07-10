#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/runtime_policy.h>
#include <citlali/core/pipeline/beammap_config_fitting_flagging.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <citlali/core/engine/detail/beammap_fit_quality_flagging_impl.h>
#include <citlali/core/engine/detail/beammap_sensitivity_flagging_impl.h>
#include <citlali/core/engine/detail/beammap_position_prior_flagging_impl.h>
#include <citlali/core/engine/detail/beammap_flux_conversion_impl.h>

void Beammap::mark_beammap_detector_flagged(
    Eigen::Index detector_index,
    AptFlags flag,
    std::atomic<int> &n_flagged_dets) {
    if (calib.apt["flag"](detector_index)==0) {
        n_flagged_dets++;
        calib.apt["flag"](detector_index) = 1;
    }
    flag2(detector_index) |= flag;
}

void Beammap::set_apt_flags() {
    // setup bitwise flags
    flag2.resize(calib.n_dets);
    flag2.setConstant(AptFlags::Good);

    // track number of flagged detectors
    std::atomic<int> n_flagged_dets{0};
    const auto &flagging_config =
        citlali::pipeline::beammap_config(*this).flagging;
    const auto flag_limits =
        citlali::pipeline::make_beammap_array_flagging_limits(
            toltec_io.array_name_map, flagging_config);
    const double lower_sens_factor = flagging_config.sens_factors[0];
    const double upper_sens_factor = flagging_config.sens_factors[1];
    const auto runtime_parallel_policy =
        citlali::pipeline::runtime_parallel_policy_name(*this);

    flag_beammap_fit_quality_outliers(
        flag_limits, runtime_parallel_policy, n_flagged_dets);

    auto nw_median_sens = beammap_network_median_sensitivities();
    flag_beammap_sensitivity_outliers(
        nw_median_sens, lower_sens_factor, upper_sens_factor,
        runtime_parallel_policy, n_flagged_dets);

    auto array_position_medians = beammap_array_position_medians();
    flag_beammap_position_outliers(
        flag_limits, array_position_medians,
        runtime_parallel_policy, n_flagged_dets);

    flag_beammap_prior_distance_outliers(
        flagging_config.max_prior_d2, array_position_medians,
        runtime_parallel_policy, n_flagged_dets);

    // print number of flagged detectors
    logger->info("{} detectors were flagged", n_flagged_dets.load());

    // Derive the calibration amplitude from an empirical array template where
    // possible.  The Gaussian fit amplitude remains in amp for morphology/QC.
    calc_empirical_template_calibration();

    calculate_beammap_flux_conversion_factors(runtime_parallel_policy);
}
