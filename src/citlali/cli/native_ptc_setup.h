#pragma once

#include "oof_development_scope.h"
#include <citlali/core/cli/successor_development_execution.h>
#include <citlali/core/pipeline/timestream_ptc_numerics.h>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace citlali::cli {

// Application-owned setup. YAML is consumed here; the scientific PTC owner
// receives only the prepared solver request. Array identity comes from the
// verified APT relation, never a network-number guess.
struct NativePtcSetup {
    int array;
    int configured_rank;
    int requested_rank; // zero: use the bound effective configuration
    pipeline::PtcSolverRequest solver;
};

inline std::string ptc_method_name(pipeline::PtcMethod method) {
    switch (method) {
    case pipeline::PtcMethod::observed_als: return "observed-als";
    case pipeline::PtcMethod::pairwise_covariance: return "pairwise-covariance";
    case pipeline::PtcMethod::masked_covariance_count: return "masked-covariance-count-v1";
    }
    throw std::invalid_argument("successor.PTC_method_unknown");
}

class NativePtcConfiguration {
public:
    NativePtcConfiguration(const YAML::Node &effective, DevelopmentPtcRequest requested)
        : effective_(YAML::Clone(effective)), requested_(std::move(requested)) {
        check(requested_.rank >= 0, "successor.PTC_rank_must_be_positive");
        check(requested_.method == "observed-als" || requested_.method == "pairwise-covariance" ||
              requested_.method == "masked-covariance-count-v1", "successor.PTC_method_unknown");
        const auto cleaning = effective_["timestream"]["processed_time_chunk"]["clean"];
        check(cleaning && cleaning["enabled"].as<bool>() &&
              cleaning["grouping"].as<std::vector<std::string>>() == std::vector<std::string>{"nw"},
              "successor.PTC_grouping_unavailable: explicit network cleaning required; no array/common-grid substitution");
        const auto pca = cleaning["standard_pca"];
        check(pca["enabled"].as<bool>() && pca["stddev_limit"].as<double>() == 0 &&
              cleaning["mask_radius_arcsec"].as<double>() == 0,
              "successor.PTC_selection_unavailable: adaptive rank or source mask not connected");
        for (const auto *name : {"adaptive_selector", "corr_grouping", "marchenko_pastur", "null_model"})
            check(!cleaning[name] || !cleaning[name]["enabled"] || !cleaning[name]["enabled"].as<bool>(),
                  "successor.PTC_selection_unavailable: alternate cleaning selector requested");
        check(!cleaning["tau"] || cleaning["tau"].as<double>() == 0, "successor.PTC_tau_selection_unavailable");
    }

    NativePtcSetup resolve(const YAML::Node &cfg, int array) const {
        const std::array<std::string, 3> arrays{"a1100", "a1400", "a2000"};
        check(array >= 0 && array < 3, "successor.PTC_array_binding_invalid");
        const auto ranks = effective_["timestream"]["processed_time_chunk"]["clean"]
            ["standard_pca"]["n_eig_to_cut"][arrays.at(array)].as<std::vector<int>>();
        // Preserve the existing contract: an explicit configured rank is
        // required even when the application request supplies an override.
        check(ranks.size() == 1 && ranks.front() > 0, "successor.PTC_explicit_rank_unavailable");
        pipeline::PtcSolverRequest solver;
        solver.method = requested_.method == "observed-als" ? pipeline::PtcMethod::observed_als :
            requested_.method == "masked-covariance-count-v1" ? pipeline::PtcMethod::masked_covariance_count :
            pipeline::PtcMethod::pairwise_covariance;
        solver.rank = requested_.rank ? requested_.rank : ranks.front();
        if (cfg["point_transfer_runtime_profile"]) {
            check(cfg["point_transfer_runtime_profile"].as<std::string>() == "point-transfer-runtime-2026-09-29-v1" &&
                  cfg["observation"].as<int>() == 152420 && cfg["network"].as<int>() >= 1 &&
                  cfg["network"].as<int>() <= 4 && array == 0 && solver.rank == 10 &&
                  solver.method == pipeline::PtcMethod::observed_als, "PTC transfer runtime profile outside approved scope");
            solver.iteration_limit = 500;
        }
        if (detail::oof_request(cfg)) {
            check(detail::oof_network_scope(cfg, cfg["observation"].as<int>(), 0, 1, cfg["network"].as<int>()) &&
                  array == detail::oof_array(cfg) && solver.rank == 3 && solver.method == pipeline::PtcMethod::observed_als,
                  "OOF cold-M0 PTC requires declared rank3 observed ALS control");
            solver.iteration_limit = 500;
        }
        if (solver.method == pipeline::PtcMethod::masked_covariance_count)
            check(cfg["observation"].as<int>() == 152430 &&
                  (cfg["network"].as<int>() == 11 || cfg["network"].as<int>() == 12) &&
                  (solver.rank == 10 || solver.rank == 15), "experimental masked covariance scope");
        return {array, ranks.front(), requested_.rank, solver};
    }

private:
    static void check(bool condition, const char *message) {
        if (!condition) throw std::invalid_argument(message);
    }
    const YAML::Node effective_;
    const DevelopmentPtcRequest requested_;
};
} // namespace citlali::cli
