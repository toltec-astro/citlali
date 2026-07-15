#pragma once

#include <citlali/core/config/timestream_config.h>

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace citlali::pipeline {

inline constexpr std::string_view kids_external_config_schema{
    "citlali-kidscpp-bridge-v1"};
inline constexpr bool kids_solver_extra_output_effective = false;
inline constexpr std::array<citlali::config::TodType, 4>
    supported_kids_tod_types{{
        citlali::config::TodType::xs,
        citlali::config::TodType::rs,
        citlali::config::TodType::is,
        citlali::config::TodType::qs,
    }};

struct KidsExternalFitterConfigIdentity {
    std::string modelspec;
    std::string weight_window_type;
    double weight_window_fwhm_hz = 0.0;
};

struct KidsExternalSolverConfigIdentity {
    std::string fit_report_directory;
    std::string parallel_policy;
    bool extra_output = false;
};

struct KidsExternalConfigIdentity {
    KidsExternalFitterConfigIdentity fitter;
    KidsExternalSolverConfigIdentity solver;
};

struct KidsExternalRequestedConfigIdentity {
    KidsExternalConfigIdentity values;
    bool solver_extra_output_present = false;
};

struct KidsExternalEffectiveConfigIdentity {
    KidsExternalConfigIdentity values;
    bool solver_extra_output_forced_disabled = false;
};

struct KidsExternalConfigPlan {
    bool initialized = false;
    std::string data_schema;
    std::string dependency_version;
    citlali::config::TodType selected_tod_type =
        citlali::config::TodType::xs;
    KidsExternalRequestedConfigIdentity requested;
    KidsExternalEffectiveConfigIdentity effective;
};

inline bool is_supported_kids_tod_type(citlali::config::TodType type) {
    for (const auto supported : supported_kids_tod_types) {
        if (supported == type) {
            return true;
        }
    }
    return false;
}

template <class Config>
KidsExternalRequestedConfigIdentity read_kids_external_config_identity(
    Config &root_config) {
    auto config = root_config.get_config("kids");
    KidsExternalRequestedConfigIdentity requested;
    requested.values.fitter.modelspec =
        config.get_str(std::tuple{"fitter", "modelspec"});
    requested.values.fitter.weight_window_type =
        config.get_str(std::tuple{"fitter", "weight_window", "type"});
    requested.values.fitter.weight_window_fwhm_hz =
        config.template get_typed<double>(
            std::tuple{"fitter", "weight_window", "fwhm_Hz"});
    requested.values.solver.fit_report_directory =
        config.get_str(std::tuple{"solver", "fitreportdir"});
    requested.values.solver.parallel_policy =
        config.get_str(std::tuple{"solver", "parallel_policy"});
    const auto extra_output_key = std::tuple{"solver", "extra_output"};
    requested.solver_extra_output_present = config.has(extra_output_key);
    if (requested.solver_extra_output_present) {
        requested.values.solver.extra_output =
            config.template get_typed<bool>(extra_output_key);
    }
    return requested;
}

template <class Config>
KidsExternalConfigPlan make_kids_external_config_plan(
    Config &root_config, citlali::config::TodType selected_tod_type,
    std::string data_schema, std::string dependency_version) {
    KidsExternalConfigPlan plan;
    plan.initialized = true;
    plan.data_schema = std::move(data_schema);
    plan.dependency_version = std::move(dependency_version);
    plan.selected_tod_type = selected_tod_type;
    plan.requested = read_kids_external_config_identity(root_config);
    plan.effective.values = plan.requested.values;
    plan.effective.values.solver.extra_output =
        kids_solver_extra_output_effective;
    plan.effective.solver_extra_output_forced_disabled =
        plan.requested.values.solver.extra_output &&
        !kids_solver_extra_output_effective;
    return plan;
}

inline void require_valid_kids_external_config_plan(
    const KidsExternalConfigPlan &plan) {
    if (!plan.initialized) {
        throw std::logic_error("KIDs external config plan is not initialized");
    }
    if (plan.data_schema.empty()) {
        throw std::logic_error("KIDs external data schema is empty");
    }
    if (plan.dependency_version.empty()) {
        throw std::logic_error("Kidscpp dependency version is empty");
    }
    if (!is_supported_kids_tod_type(plan.selected_tod_type)) {
        throw std::logic_error("unsupported KIDs TOD type");
    }
}

}  // namespace citlali::pipeline
