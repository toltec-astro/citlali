#pragma once

#include <citlali/core/config/calibration_config.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace citlali::pipeline {

enum class AstrometryApplicationMode {
    constant,
    observation_span_linear,
    explicit_mjd_linear,
};

inline constexpr std::string_view to_string(
    AstrometryApplicationMode mode) {
    switch (mode) {
        case AstrometryApplicationMode::constant:
            return "constant";
        case AstrometryApplicationMode::observation_span_linear:
            return "observation-span-linear";
        case AstrometryApplicationMode::explicit_mjd_linear:
            return "explicit-mjd-linear";
    }
    return "unknown";
}

struct AstrometryEffectiveResolution {
    AstrometryApplicationMode application_mode =
        AstrometryApplicationMode::constant;
    bool explicit_mjd_support = false;
};

struct AstrometryRealizedState {
    std::size_t installation_count = 0;
    std::size_t application_count = 0;
    std::size_t telescope_sample_count = 0;
};

struct AstrometryObservationPlan {
    std::size_t observation_index = 0;
    int obsnum = 0;
    citlali::config::AstrometryConfig requested;
    citlali::config::AstrometryConfig effective;
    AstrometryEffectiveResolution resolution;
    AstrometryRealizedState realized;
};

struct AstrometryExecutionPlan {
    bool initialized = false;
    bool reduction_completed = false;
    std::size_t expected_observation_count = 0;
    std::optional<std::size_t> active_observation_index;
    std::vector<AstrometryObservationPlan> observations;

    void reset(std::size_t observation_count) {
        initialized = true;
        reduction_completed = false;
        expected_observation_count = observation_count;
        active_observation_index.reset();
        observations.clear();
        observations.reserve(observation_count);
    }
};

inline bool astrometry_config_equal(
    const citlali::config::AstrometryConfig &lhs,
    const citlali::config::AstrometryConfig &rhs) {
    const auto &left = lhs.pointing_offsets;
    const auto &right = rhs.pointing_offsets;
    return left.enabled == right.enabled &&
           left.az_arcsec == right.az_arcsec &&
           left.alt_arcsec == right.alt_arcsec &&
           left.modified_julian_date == right.modified_julian_date;
}

inline AstrometryEffectiveResolution resolve_astrometry_application(
    const citlali::config::AstrometryConfig &request) {
    const auto &offsets = request.pointing_offsets;
    if (offsets.az_arcsec.size() == 1) {
        return {AstrometryApplicationMode::constant, false};
    }
    if (offsets.az_arcsec.size() != 2) {
        throw std::invalid_argument(
            "astrometry application requires one or two offset values");
    }
    const bool explicit_mjd =
        offsets.modified_julian_date.size() == 2 &&
        offsets.modified_julian_date[0] > 0.0 &&
        offsets.modified_julian_date[1] > 0.0;
    return {
        explicit_mjd
            ? AstrometryApplicationMode::explicit_mjd_linear
            : AstrometryApplicationMode::observation_span_linear,
        explicit_mjd,
    };
}

inline AstrometryObservationPlan &record_astrometry_request(
    AstrometryExecutionPlan &plan, std::size_t observation_index,
    int obsnum, const citlali::config::AstrometryConfig &request) {
    if (!plan.initialized) {
        throw std::logic_error("astrometry plan is not initialized");
    }
    if (plan.reduction_completed) {
        throw std::logic_error("astrometry plan is already completed");
    }
    if (observation_index >= plan.expected_observation_count) {
        throw std::logic_error("astrometry observation index is out of range");
    }
    if (observation_index > plan.observations.size()) {
        throw std::logic_error(
            "astrometry observations must be registered in order");
    }
    if (observation_index == plan.observations.size()) {
        plan.observations.push_back(AstrometryObservationPlan{
            observation_index,
            obsnum,
            request,
            request,
            resolve_astrometry_application(request),
            {},
        });
    }
    auto &observation = plan.observations[observation_index];
    if (observation.obsnum != obsnum ||
        !astrometry_config_equal(observation.requested, request)) {
        throw std::logic_error(
            "repeated astrometry observation request differs from initial request");
    }
    plan.active_observation_index = observation_index;
    return observation;
}

inline void record_astrometry_installed(AstrometryExecutionPlan &plan) {
    if (plan.reduction_completed) {
        throw std::logic_error("astrometry plan is already completed");
    }
    if (!plan.active_observation_index) {
        throw std::logic_error("astrometry observation is not active");
    }
    ++plan.observations[*plan.active_observation_index]
          .realized.installation_count;
}

inline void record_astrometry_applied(
    AstrometryExecutionPlan &plan, std::size_t telescope_sample_count) {
    if (plan.reduction_completed) {
        throw std::logic_error("astrometry plan is already completed");
    }
    if (!plan.active_observation_index) {
        throw std::logic_error("astrometry observation is not active");
    }
    if (telescope_sample_count == 0) {
        throw std::logic_error(
            "astrometry cannot be applied to an empty telescope timestream");
    }
    auto &realized = plan.observations[*plan.active_observation_index].realized;
    if (realized.application_count > 0 &&
        realized.telescope_sample_count != telescope_sample_count) {
        throw std::logic_error(
            "astrometry telescope sample count changed for one observation");
    }
    ++realized.application_count;
    realized.telescope_sample_count = telescope_sample_count;
}

inline void record_astrometry_reduction_completed(
    AstrometryExecutionPlan &plan) {
    if (plan.reduction_completed) {
        throw std::logic_error("astrometry plan is already completed");
    }
    if (!plan.initialized ||
        plan.observations.size() != plan.expected_observation_count) {
        throw std::logic_error(
            "astrometry plan does not contain every observation");
    }
    for (const auto &observation : plan.observations) {
        if (observation.realized.installation_count == 0 ||
            observation.realized.application_count == 0 ||
            observation.realized.installation_count !=
                observation.realized.application_count ||
            observation.realized.telescope_sample_count == 0) {
            throw std::logic_error(
                "astrometry observation lifecycle is incomplete");
        }
    }
    plan.active_observation_index.reset();
    plan.reduction_completed = true;
}

}  // namespace citlali::pipeline
