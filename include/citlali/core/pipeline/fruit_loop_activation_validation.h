#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>

#include <algorithm>
#include <cmath>

namespace citlali::pipeline {

inline bool fruit_loop_flux_gate_active(
    const citlali::config::TimestreamFruitLoopsConfig &config) {
    return std::any_of(
        config.array_flux_limit.begin(), config.array_flux_limit.end(),
        [](double limit) {
            return std::isfinite(limit) && std::abs(limit) > 0.0;
        });
}

inline bool fruit_loop_adaptive_gate_active(
    const citlali::config::TimestreamFruitLoopsConfig &config) {
    return config.peak_fraction_limit > 0.0 || config.local_snr_floor > 0.0;
}

inline bool fruit_loop_snr_gate_active(
    const citlali::config::TimestreamFruitLoopsConfig &config) {
    return std::isfinite(config.sig2noise_limit) &&
           std::abs(config.sig2noise_limit) > 0.0;
}

inline citlali::config::ValidationReport validate_fruit_loop_activation(
    const citlali::config::TimestreamFruitLoopsConfig &fruit_loops,
    const citlali::config::NoiseConfig &effective_noise,
    citlali::config::ReductionType reduction_type) {
    citlali::config::ValidationReport report;
    if (!fruit_loops.enabled ||
        citlali::config::is_beammap_reduction_type(reduction_type)) {
        return report;
    }

    const citlali::config::ConfigPath path{
        "timestream", "fruit_loops"};
    const bool snr_gate = fruit_loop_snr_gate_active(fruit_loops);
    const bool flux_gate = fruit_loop_flux_gate_active(fruit_loops);
    const bool adaptive_gate = fruit_loop_adaptive_gate_active(fruit_loops);

    if (fruit_loops.max_iters < 2) {
        report.add_error(
            citlali::config::append_config_path(path, {"max_iters"}),
            "enabled fruit loops require at least two iterations so a previous map can be fed back");
    }
    if (!snr_gate && !flux_gate && !adaptive_gate) {
        report.add_error(
            path,
            "enabled fruit loops have no active selection gate; configure a nonzero sig2noise_limit, array_flux_limit, peak_fraction_limit, or local_snr_floor");
    }
    if (snr_gate &&
        (!effective_noise.enabled || effective_noise.n_noise_maps <= 0 ||
         !effective_noise.products_enabled)) {
        report.add_error(
            citlali::config::append_config_path(path, {"sig2noise_limit"}),
            "fruit-loop S/N selection requires enabled noise maps and empirical noise products with a positive realization count");
    }
    return report;
}

}  // namespace citlali::pipeline
