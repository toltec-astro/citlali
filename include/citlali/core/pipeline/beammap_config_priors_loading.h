#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

struct BeammapPriorsConfigValues {
    bool enabled = false;
    std::string filepath = "null";
    int candidate_top_n = 64;
    double min_snr = 0.0;
    double max_d2 = 25.0;
    double max_d2_iter0 = 25.0;
    double max_d2_after_iter0 = 25.0;
    double score_lambda = 2.0;
    double score_lambda_iter0 = 2.0;
    double score_lambda_after_iter0 = 2.0;
    bool fallback_blind = true;
    bool align_after_iter0 = true;
    std::string alignment_scope = "array";
    std::string alignment_common_support = "all";
    double alignment_common_support_quantile = 0.02;
    int alignment_min_matches = 30;
    double alignment_max_d2 = 25.0;
    bool alignment_fit_rotation = true;
    double alignment_max_rotation_deg = 8.0;
};

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
    read_optional_beammap_config_value(
        config, priors.alignment_scope, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "alignment_scope"},
        {"array", "common"});
    read_optional_beammap_config_value(
        config, priors.alignment_common_support, missing_keys, invalid_keys,
        std::tuple{"beammap", "priors", "alignment_common_support"},
        {"all", "overlap_box"});
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

template <class BeammapControls>
void sync_beammap_priors_controls(
    BeammapControls &controls, const BeammapPriorsConfigValues &priors) {
    controls.beammap_priors_enabled = priors.enabled;
    controls.beammap_priors_filepath = priors.filepath;
    controls.beammap_priors_candidate_top_n = priors.candidate_top_n;
    controls.beammap_priors_min_snr = priors.min_snr;
    controls.beammap_priors_max_d2 = priors.max_d2;
    controls.beammap_priors_max_d2_iter0 = priors.max_d2_iter0;
    controls.beammap_priors_max_d2_after_iter0 = priors.max_d2_after_iter0;
    controls.beammap_priors_score_lambda = priors.score_lambda;
    controls.beammap_priors_score_lambda_iter0 = priors.score_lambda_iter0;
    controls.beammap_priors_score_lambda_after_iter0 =
        priors.score_lambda_after_iter0;
    controls.beammap_priors_fallback_blind = priors.fallback_blind;
    controls.beammap_priors_align_after_iter0 = priors.align_after_iter0;
    controls.beammap_priors_alignment_scope = priors.alignment_scope;
    controls.beammap_priors_alignment_common_support =
        priors.alignment_common_support;
    controls.beammap_priors_alignment_common_support_quantile =
        priors.alignment_common_support_quantile;
    controls.beammap_priors_alignment_min_matches =
        priors.alignment_min_matches;
    controls.beammap_priors_alignment_max_d2 = priors.alignment_max_d2;
    controls.beammap_priors_alignment_fit_rotation =
        priors.alignment_fit_rotation;
    controls.beammap_priors_alignment_max_rotation_deg =
        priors.alignment_max_rotation_deg;
}
