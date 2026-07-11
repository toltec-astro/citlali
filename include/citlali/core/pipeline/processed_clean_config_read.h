#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cctype>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline std::string normalize_processed_clean_group(std::string group) {
    std::transform(
        group.begin(), group.end(), group.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    if (group == "network") {
        return "nw";
    }
    return group;
}

inline bool is_supported_processed_clean_group(std::string_view group) {
    return group == "all" || group == "array" || group == "nw" ||
           group == "detector" || group == "fg" || group == "corr_nw";
}

template <class Config, class Diagnostics, class ArrayNameMap, class Logger>
void read_processed_clean_core_config(
    Config &config, citlali::config::ProcessedTimeChunkCleanConfig &typed,
    Diagnostics &diagnostics, const ArrayNameMap &array_name_map,
    const Logger &logger) {
    bool enabled = typed.enabled;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "processed_time_chunk", "clean", "enabled"},
        enabled, typed.enabled, diagnostics);
    if (!typed.enabled) {
        typed.active = citlali::config::ProcessedTimeChunkCleanerMode::none;
        return;
    }

    auto grouping = typed.grouping;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "processed_time_chunk", "clean", "grouping"},
        grouping, typed.grouping, diagnostics);

    double mask_radius_arcsec = typed.mask_radius_arcsec;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "processed_time_chunk", "clean",
                   "mask_radius_arcsec"},
        mask_radius_arcsec, typed.mask_radius_arcsec, diagnostics);
    double tau = typed.tau;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "processed_time_chunk", "clean", "tau"},
        tau, typed.tau, diagnostics);

    const auto standard_key = std::tuple{
        "timestream", "processed_time_chunk", "clean", "standard_pca",
        "enabled"};
    const bool have_standard_block =
        config.template has_typed<bool>(standard_key);
    bool standard_enabled = typed.standard_pca.enabled;
    read_optional_mirrored_config_value(
        config, standard_key, standard_enabled, typed.standard_pca.enabled,
        diagnostics);

    auto read_mode_enabled = [&](const auto &key, bool &target) {
        bool value = target;
        read_optional_mirrored_config_value(
            config, key, value, target, diagnostics);
    };
    read_mode_enabled(
        std::tuple{"timestream", "processed_time_chunk", "clean",
                   "null_model", "enabled"},
        typed.null_model.enabled);
    read_mode_enabled(
        std::tuple{"timestream", "processed_time_chunk", "clean",
                   "marchenko_pastur", "enabled"},
        typed.marchenko_pastur.enabled);
    read_mode_enabled(
        std::tuple{"timestream", "processed_time_chunk", "clean",
                   "adaptive_selector", "enabled"},
        typed.adaptive_selector.enabled);
    if (!have_standard_block) {
        typed.standard_pca.enabled =
            !(typed.null_model.enabled || typed.marchenko_pastur.enabled ||
              typed.adaptive_selector.enabled);
    }

    auto read_mode_grouping = [&](const char *mode,
                                  std::vector<std::string> &target) {
        const auto key = std::tuple{
            "timestream", "processed_time_chunk", "clean", mode,
            "grouping"};
        if (!config.template has_typed<std::vector<std::string>>(key)) {
            return;
        }
        target.clear();
        std::unordered_set<std::string> seen;
        for (const auto &raw_group :
             config.template get_typed<std::vector<std::string>>(key)) {
            auto group = normalize_processed_clean_group(raw_group);
            if (!is_supported_processed_clean_group(group)) {
                logger->warn(
                    "clean.{}.grouping contains unsupported entry '{}'; ignoring",
                    mode, raw_group);
                continue;
            }
            if (seen.insert(group).second) {
                target.push_back(std::move(group));
            }
        }
    };

    typed.standard_pca.n_eig_to_cut.clear();
    for (const auto &[array_id, array_name] : array_name_map) {
        (void)array_id;
        std::vector<Eigen::Index> values;
        const auto standard_eigen_key = std::tuple{
            "timestream", "processed_time_chunk", "clean", "standard_pca",
            "n_eig_to_cut", array_name};
        const auto legacy_eigen_key = std::tuple{
            "timestream", "processed_time_chunk", "clean", "n_eig_to_cut",
            array_name};
        if (config.template has_typed<std::vector<Eigen::Index>>(
                standard_eigen_key)) {
            values = config.template get_typed<std::vector<Eigen::Index>>(
                standard_eigen_key);
        } else if (config.template has_typed<std::vector<Eigen::Index>>(
                       legacy_eigen_key)) {
            values = config.template get_typed<std::vector<Eigen::Index>>(
                legacy_eigen_key);
        }
        if (values.empty()) {
            logger->warn(
                "clean.n_eig_to_cut.{} is empty; defaulting to 0 for all {} grouping pass(es)",
                array_name, typed.grouping.size());
            values.assign(typed.grouping.size(), 0);
        } else if (values.size() < typed.grouping.size()) {
            logger->warn(
                "clean.n_eig_to_cut.{} has {} value(s) but clean.grouping has {} pass(es); padding with last value {}",
                array_name, values.size(), typed.grouping.size(),
                values.back());
            values.resize(typed.grouping.size(), values.back());
        }
        auto &typed_values = typed.standard_pca.n_eig_to_cut[array_name];
        typed_values.reserve(values.size());
        for (const auto value : values) {
            typed_values.push_back(static_cast<int>(value));
        }
    }

    double stddev_limit = typed.standard_pca.stddev_limit;
    const auto standard_stddev_key = std::tuple{
        "timestream", "processed_time_chunk", "clean", "standard_pca",
        "stddev_limit"};
    const auto legacy_stddev_key = std::tuple{
        "timestream", "processed_time_chunk", "clean", "stddev_limit"};
    if (config.template has_typed<double>(standard_stddev_key)) {
        read_mirrored_config_value(
            config, standard_stddev_key, stddev_limit,
            typed.standard_pca.stddev_limit, diagnostics);
    } else if (config.template has_typed<double>(legacy_stddev_key)) {
        read_mirrored_config_value(
            config, legacy_stddev_key, stddev_limit,
            typed.standard_pca.stddev_limit, diagnostics);
    }

    int n_calc = typed.standard_pca.n_calc;
    const auto standard_n_calc_key = std::tuple{
        "timestream", "processed_time_chunk", "clean", "standard_pca",
        "n_calc"};
    const auto legacy_n_calc_key = std::tuple{
        "timestream", "processed_time_chunk", "clean", "n_calc"};
    if (config.template has_typed<int>(standard_n_calc_key)) {
        read_mirrored_config_value(
            config, standard_n_calc_key, n_calc,
            typed.standard_pca.n_calc, diagnostics, {}, {0});
    } else if (config.template has_typed<int>(legacy_n_calc_key)) {
        read_mirrored_config_value(
            config, legacy_n_calc_key, n_calc,
            typed.standard_pca.n_calc, diagnostics, {}, {0});
    }

    auto &corr = typed.corr_grouping;
    bool corr_enabled = corr.enabled;
    read_optional_mirrored_config_value(
        config,
        std::tuple{"timestream", "processed_time_chunk", "clean",
                   "corr_grouping", "enabled"},
        corr_enabled, corr.enabled, diagnostics);
    if (corr.enabled) {
        std::string metric{citlali::config::to_string(corr.metric)};
        read_optional_parsed_mirrored_config_value(
            config,
            std::tuple{"timestream", "processed_time_chunk", "clean",
                       "corr_grouping", "metric"},
            metric, corr.metric,
            citlali::config::parse_processed_corr_grouping_metric,
            diagnostics, {"abs", "signed"});
        auto read_corr_double = [&](const char *name, double &target,
                                    double minimum, double maximum) {
            double value = target;
            read_optional_mirrored_config_value(
                config,
                std::tuple{"timestream", "processed_time_chunk", "clean",
                           "corr_grouping", name},
                value, target, diagnostics, {}, {minimum}, {maximum});
        };
        auto read_corr_int = [&](const char *name, int &target,
                                 int minimum) {
            int value = target;
            read_optional_mirrored_config_value(
                config,
                std::tuple{"timestream", "processed_time_chunk", "clean",
                           "corr_grouping", name},
                value, target, diagnostics, {}, {minimum});
        };
        read_corr_double("corr_min", corr.corr_min, 0.0, 1.0);
        read_corr_int("min_overlap", corr.min_overlap, 1);
        read_corr_double("min_good_frac", corr.min_good_frac, 0.0, 1.0);
        read_corr_int("min_group_size", corr.min_group_size, 2);
        read_corr_int("max_samples", corr.max_samples, 0);
        bool clean_residual = corr.clean_residual;
        read_optional_mirrored_config_value(
            config,
            std::tuple{"timestream", "processed_time_chunk", "clean",
                       "corr_grouping", "clean_residual"},
            clean_residual, corr.clean_residual, diagnostics);
    }

    if (typed.null_model.enabled) {
        auto &null_model = typed.null_model;
        auto read_null_int = [&](const char *name, int &target,
                                 int minimum) {
            int value = target;
            read_optional_mirrored_config_value(
                config,
                std::tuple{"timestream", "processed_time_chunk", "clean",
                           "null_model", name},
                value, target, diagnostics, {}, {minimum});
        };
        auto read_null_double = [&](const char *name, double &target,
                                    double minimum, double maximum) {
            double value = target;
            read_optional_mirrored_config_value(
                config,
                std::tuple{"timestream", "processed_time_chunk", "clean",
                           "null_model", name},
                value, target, diagnostics, {}, {minimum}, {maximum});
        };
        read_null_int("n_surrogates", null_model.n_surrogates, 4);
        read_null_double("quantile", null_model.quantile, 0.5, 0.999999);
        read_null_double(
            "min_good_frac", null_model.min_good_frac, 0.0, 1.0);
        read_null_int("max_modes", null_model.max_modes, 0);
        read_null_int("max_samples", null_model.max_samples, 0);
        read_null_int("seed", null_model.seed, 0);
        read_mode_grouping("null_model", null_model.grouping);
    }

    if (typed.marchenko_pastur.enabled) {
        auto &mp = typed.marchenko_pastur;
        auto read_mp_int = [&](const char *name, int &target, int minimum) {
            int value = target;
            read_optional_mirrored_config_value(
                config,
                std::tuple{"timestream", "processed_time_chunk", "clean",
                           "marchenko_pastur", name},
                value, target, diagnostics, {}, {minimum});
        };
        auto read_mp_double = [&](const char *name, double &target,
                                  double minimum,
                                  std::vector<double> maximum = {}) {
            double value = target;
            read_optional_mirrored_config_value(
                config,
                std::tuple{"timestream", "processed_time_chunk", "clean",
                           "marchenko_pastur", name},
                value, target, diagnostics, {}, {minimum},
                std::move(maximum));
        };
        read_mp_double("min_good_frac", mp.min_good_frac, 0.0, {1.0});
        read_mp_int("max_modes", mp.max_modes, 0);
        read_mp_int("max_samples", mp.max_samples, 0);
        read_mp_double("band_low_Hz", mp.band_low_Hz, 0.0);
        read_mp_double("band_high_Hz", mp.band_high_Hz, 0.0);
        double clip_z = mp.clip_z;
        read_optional_mirrored_config_value(
            config,
            std::tuple{"timestream", "processed_time_chunk", "clean",
                       "marchenko_pastur", "clip_z"},
            clip_z, mp.clip_z, diagnostics);
        read_mp_double("bulk_keep_frac", mp.bulk_keep_frac, 0.1, {1.0});
        read_mp_int("q_grid_size", mp.q_grid_size, 8);
        read_mode_grouping("marchenko_pastur", mp.grouping);
    }

    if (typed.adaptive_selector.enabled) {
        auto &adaptive = typed.adaptive_selector;
        auto read_adaptive_int = [&](const char *name, int &target,
                                     int minimum) {
            int value = target;
            read_optional_mirrored_config_value(
                config,
                std::tuple{"timestream", "processed_time_chunk", "clean",
                           "adaptive_selector", name},
                value, target, diagnostics, {}, {minimum});
        };
        auto read_adaptive_double = [&](const char *name, double &target,
                                        std::vector<double> minimum = {}) {
            double value = target;
            read_optional_mirrored_config_value(
                config,
                std::tuple{"timestream", "processed_time_chunk", "clean",
                           "adaptive_selector", name},
                value, target, diagnostics, {}, std::move(minimum));
        };
        double min_good_frac = adaptive.min_good_frac;
        read_optional_mirrored_config_value(
            config,
            std::tuple{"timestream", "processed_time_chunk", "clean",
                       "adaptive_selector", "min_good_frac"},
            min_good_frac, adaptive.min_good_frac, diagnostics, {}, {0.0},
            {1.0});
        read_adaptive_int("max_det", adaptive.max_det, 0);
        read_adaptive_int("max_samples", adaptive.max_samples, 0);
        read_adaptive_int("max_pairs", adaptive.max_pairs, 0);
        read_adaptive_int("seed", adaptive.seed, 0);
        read_adaptive_double("clip_z", adaptive.clip_z);
        read_adaptive_double("low_weight", adaptive.low_weight, {0.0});
        read_adaptive_double("tail_weight", adaptive.tail_weight, {0.0});
        read_adaptive_double(
            "topmode_weight", adaptive.topmode_weight, {0.0});
        read_adaptive_double("reg_weight", adaptive.reg_weight, {0.0});
        bool log_candidates = adaptive.log_candidates;
        read_optional_mirrored_config_value(
            config,
            std::tuple{"timestream", "processed_time_chunk", "clean",
                       "adaptive_selector", "log_candidates"},
            log_candidates, adaptive.log_candidates, diagnostics);

        const auto offsets_key = std::tuple{
            "timestream", "processed_time_chunk", "clean",
            "adaptive_selector", "candidate_offsets"};
        if (config.template has_typed<std::vector<int>>(offsets_key)) {
            const auto offsets =
                config.template get_typed<std::vector<int>>(offsets_key);
            if (!offsets.empty()) {
                adaptive.candidate_offsets = offsets;
            }
        }
        auto read_band = [&](const char *name,
                             std::array<double, 2> &target) {
            const auto key = std::tuple{
                "timestream", "processed_time_chunk", "clean",
                "adaptive_selector", name};
            if (!config.template has_typed<std::vector<double>>(key)) {
                return;
            }
            const auto band =
                config.template get_typed<std::vector<double>>(key);
            if (band.size() == 2 && band[0] >= 0.0 && band[1] > band[0]) {
                target = {band[0], band[1]};
            } else {
                logger->warn(
                    "clean.adaptive_selector.{} must be [fmin, fmax] with 0<=fmin<fmax",
                    name);
            }
        };
        read_band("low_band_Hz", adaptive.low_band_Hz);
        read_band("mid_band_Hz", adaptive.mid_band_Hz);
        read_mode_grouping("adaptive_selector", adaptive.grouping);
    }

    typed.active = citlali::config::ProcessedTimeChunkCleanerMode::none;
    if (typed.standard_pca.enabled) {
        typed.active =
            citlali::config::ProcessedTimeChunkCleanerMode::standard_pca;
    } else if (typed.null_model.enabled) {
        typed.active =
            citlali::config::ProcessedTimeChunkCleanerMode::null_model;
    } else if (typed.marchenko_pastur.enabled) {
        typed.active =
            citlali::config::ProcessedTimeChunkCleanerMode::marchenko_pastur;
    } else if (typed.adaptive_selector.enabled) {
        typed.active =
            citlali::config::ProcessedTimeChunkCleanerMode::adaptive_selector;
    }
}

}  // namespace citlali::pipeline
