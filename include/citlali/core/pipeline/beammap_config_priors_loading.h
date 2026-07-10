#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

using BeammapPriorsConfigValues = citlali::config::BeammapPriorsConfig;

inline void set_beammap_priors_iteration_defaults(
    BeammapPriorsConfigValues &priors) {
    priors.max_d2_iter0 = priors.max_d2;
    priors.max_d2_after_iter0 = priors.max_d2;
    priors.score_lambda_iter0 = priors.score_lambda;
    priors.score_lambda_after_iter0 = priors.score_lambda;
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_priors_core_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    BeammapPriorsConfigValues &priors) {
    read_optional_beammap_config_value(
        config, priors.enabled, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "enabled"});
    read_optional_beammap_config_value(
        config, priors.filepath, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "filepath"});
    read_optional_beammap_config_value(
        config, priors.candidate_top_n, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "candidate_top_n"}, {}, {1});
    read_optional_beammap_config_value(
        config, priors.min_snr, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "min_snr"});
    read_optional_beammap_config_value(
        config, priors.max_d2, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "max_d2"}, {}, {0.0});
    read_optional_beammap_config_value(
        config, priors.score_lambda, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "score_lambda"}, {}, {0.0});
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_priors_iteration_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    BeammapPriorsConfigValues &priors) {
    read_optional_beammap_config_value(
        config, priors.max_d2_iter0, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "max_d2_iter0"}, {}, {0.0});
    read_optional_beammap_config_value(
        config, priors.max_d2_after_iter0, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "max_d2_after_iter0"}, {}, {0.0});
    read_optional_beammap_config_value(
        config, priors.score_lambda_iter0, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "score_lambda_iter0"}, {}, {0.0});
    read_optional_beammap_config_value(
        config, priors.score_lambda_after_iter0, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "score_lambda_after_iter0"}, {},
        {0.0});
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_priors_behavior_config(Config &config,
                                         MissingKeys &missing_keys,
                                         InvalidKeys &invalid_keys,
                                         BeammapPriorsConfigValues &priors) {
    read_optional_beammap_config_value(
        config, priors.fallback_blind, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "fallback_blind"});
    read_optional_beammap_config_value(
        config, priors.align_after_iter0, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "align_after_iter0"});
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_priors_alignment_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    BeammapPriorsConfigValues &priors) {
    std::string alignment_scope_name =
        std::string(citlali::config::to_string(priors.alignment_scope));
    read_optional_beammap_config_value(
        config, alignment_scope_name, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "alignment_scope"},
        {"array", "common"});
    if (auto parsed = citlali::config::parse_beammap_prior_alignment_scope(
            alignment_scope_name)) {
        priors.alignment_scope = *parsed;
    }

    std::string alignment_support_name =
        std::string(citlali::config::to_string(
            priors.alignment_common_support));
    read_optional_beammap_config_value(
        config, alignment_support_name, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "alignment_common_support"},
        {"all", "overlap_box"});
    if (auto parsed = citlali::config::parse_beammap_prior_alignment_support(
            alignment_support_name)) {
        priors.alignment_common_support = *parsed;
    }

    read_optional_beammap_config_value(
        config, priors.alignment_common_support_quantile,
        missing_keys, invalid_keys,
        std::tuple{"beammap", "priors",
                   "alignment_common_support_quantile"},
        {}, {0.0}, {0.45});
    read_optional_beammap_config_value(
        config, priors.alignment_min_matches, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "alignment_min_matches"}, {}, {3});
    read_optional_beammap_config_value(
        config, priors.alignment_max_d2, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "alignment_max_d2"}, {}, {0.0});
    read_optional_beammap_config_value(
        config, priors.alignment_fit_rotation, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "alignment_fit_rotation"});
    read_optional_beammap_config_value(
        config, priors.alignment_max_rotation_deg, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "alignment_max_rotation_deg"}, {},
        {0.0});
}

template <class Config, class MissingKeys, class InvalidKeys, class Logger>
BeammapPriorsConfigValues read_beammap_priors_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    const Logger &logger) {
    BeammapPriorsConfigValues priors;
    read_beammap_priors_core_config(
        config, missing_keys, invalid_keys, priors);
    set_beammap_priors_iteration_defaults(priors);
    read_beammap_priors_iteration_config(
        config, missing_keys, invalid_keys, priors);
    read_beammap_priors_behavior_config(
        config, missing_keys, invalid_keys, priors);
    read_beammap_priors_alignment_config(
        config, missing_keys, invalid_keys, priors);
    disable_missing_beammap_priors(priors.enabled, priors.filepath, logger);
    return priors;
}

template <class Config, class Diagnostics, class Logger>
BeammapPriorsConfigValues read_beammap_priors_config(
    Config &config, Diagnostics &diagnostics, const Logger &logger) {
    return read_beammap_priors_config(
        config, diagnostics.missing_keys, diagnostics.invalid_keys, logger);
}
