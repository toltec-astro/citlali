#pragma once

#include <citlali/core/pipeline/flxscale_correction_logging.h>
#include <citlali/core/pipeline/flxscale_correction_metadata.h>

#include <Eigen/Core>

#include <cmath>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace citlali::pipeline {

inline constexpr std::string_view observation_flxscale_correction_state_key =
    "__citlali_applied_observation_flxscale_correction";

template <class Engine>
bool has_apt_flxscale_column(const Engine &engine) {
    return engine.calib.apt.count("flxscale") != 0;
}

template <class Engine, class = void>
struct has_observation_flxscale_correction_state : std::false_type {};

template <class Engine>
struct has_observation_flxscale_correction_state<
    Engine,
    std::void_t<decltype(
        std::declval<Engine &>().calib.flux_conversion_factor)>>
    : std::true_type {};

template <class Engine>
bool record_observation_flxscale_correction(Engine &engine, double factor) {
    auto &state = engine.calib.flux_conversion_factor;
    const auto &source_flxscale = engine.calib.apt.at("flxscale");
    if (state.size() != source_flxscale.size() || state.size() <= 0) {
        return false;
    }
    auto &summary = engine.calib.mean_flux_conversion_factor;
    const std::string key{observation_flxscale_correction_state_key};
    if (summary.count(key) != 0) {
        return false;
    }
    const Eigen::VectorXd composed = state.array() * factor;
    if (!(state.array().isFinite() && (state.array() > 0.0)).all() ||
        !(composed.array().isFinite() && (composed.array() > 0.0)).all()) {
        return false;
    }
    state = composed;
    summary[key] = factor;
    return true;
}

template <class Engine, class RawObs, class Logger>
bool validate_flxscale_correction(const Engine &engine, const RawObs &rawobs,
                                  const Logger &logger) {
    const auto *flxscale_corr = flxscale_correction_metadata(rawobs);
    if (!has_flxscale_correction(flxscale_corr)) {
        return true;
    }

    const double factor = flxscale_correction_factor(*flxscale_corr);
    if (!is_valid_flxscale_correction_factor(factor)) {
        log_invalid_flxscale_correction_factor(factor, rawobs, logger);
        return false;
    }
    if (!has_apt_flxscale_column(engine)) {
        log_missing_flxscale_column(rawobs, logger);
        return false;
    }
    return true;
}

template <class Engine, class RawObs, class Logger>
bool apply_flxscale_correction(Engine &engine, const RawObs &rawobs,
                               const Logger &logger) {
    if (!validate_flxscale_correction(engine, rawobs, logger)) {
        return false;
    }
    const auto *flxscale_corr = flxscale_correction_metadata(rawobs);
    if (!has_flxscale_correction(flxscale_corr)) {
        return true;
    }
    const double factor = flxscale_correction_factor(*flxscale_corr);

    if constexpr (has_observation_flxscale_correction_state<Engine>::value) {
        if (engine.calib.mean_flux_conversion_factor.count(
                std::string{observation_flxscale_correction_state_key}) != 0) {
            logger->error(
                "observation flxscale correction was already applied for "
                "observation {}",
                rawobs.name());
            return false;
        }
        if (!record_observation_flxscale_correction(engine, factor)) {
            logger->error(
                "observation flxscale correction state is unavailable, "
                "already applied, or composes to a non-finite/non-positive "
                "factor for observation {}",
                rawobs.name());
            return false;
        }
    }
    else {
        logger->error(
            "observation flxscale correction requires observation-owned "
            "applied state for observation {}; source APT remains unchanged",
            rawobs.name());
        return false;
    }
    log_applied_flxscale_correction(factor, rawobs, logger);
    return true;
}

}  // namespace citlali::pipeline
