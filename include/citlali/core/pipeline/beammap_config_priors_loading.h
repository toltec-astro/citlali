#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

inline void initialize_beammap_priors_defaults(
    bool &enabled, std::string &filepath, int &candidate_top_n,
    double &min_snr, double &max_d2, double &max_d2_iter0,
    double &max_d2_after_iter0, double &score_lambda,
    double &score_lambda_iter0, double &score_lambda_after_iter0,
    bool &fallback_blind, bool &align_after_iter0,
    std::string &alignment_scope, std::string &alignment_common_support,
    double &alignment_common_support_quantile, int &alignment_min_matches,
    double &alignment_max_d2, bool &alignment_fit_rotation,
    double &alignment_max_rotation_deg) {
    enabled = false;
    filepath = "null";
    candidate_top_n = 64;
    min_snr = 0.0;
    max_d2 = 25.0;
    max_d2_iter0 = 25.0;
    max_d2_after_iter0 = 25.0;
    score_lambda = 2.0;
    score_lambda_iter0 = 2.0;
    score_lambda_after_iter0 = 2.0;
    fallback_blind = true;
    align_after_iter0 = true;
    alignment_scope = "array";
    alignment_common_support = "all";
    alignment_common_support_quantile = 0.02;
    alignment_min_matches = 30;
    alignment_max_d2 = 25.0;
    alignment_fit_rotation = true;
    alignment_max_rotation_deg = 8.0;
}

inline void set_beammap_priors_iteration_defaults(
    double max_d2, double &max_d2_iter0, double &max_d2_after_iter0,
    double score_lambda, double &score_lambda_iter0,
    double &score_lambda_after_iter0) {
    max_d2_iter0 = max_d2;
    max_d2_after_iter0 = max_d2;
    score_lambda_iter0 = score_lambda;
    score_lambda_after_iter0 = score_lambda;
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_priors_core_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    bool &enabled, std::string &filepath, int &candidate_top_n,
    double &min_snr, double &max_d2, double &score_lambda) {
    if (config.template has_typed<bool>(
            std::tuple{"beammap", "priors", "enabled"})) {
        ::get_config_value(config, enabled, missing_keys, invalid_keys,
                           std::tuple{"beammap", "priors", "enabled"});
    }
    if (config.template has_typed<std::string>(
            std::tuple{"beammap", "priors", "filepath"})) {
        ::get_config_value(config, filepath, missing_keys, invalid_keys,
                           std::tuple{"beammap", "priors", "filepath"});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "priors", "candidate_top_n"})) {
        ::get_config_value(
            config, candidate_top_n, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "candidate_top_n"}, {}, {1});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "priors", "min_snr"})) {
        ::get_config_value(config, min_snr, missing_keys, invalid_keys,
                           std::tuple{"beammap", "priors", "min_snr"});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "priors", "max_d2"})) {
        ::get_config_value(config, max_d2, missing_keys, invalid_keys,
                           std::tuple{"beammap", "priors", "max_d2"}, {},
                           {0.0});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "priors", "score_lambda"})) {
        ::get_config_value(
            config, score_lambda, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "score_lambda"}, {}, {0.0});
    }
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_priors_iteration_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    double &max_d2_iter0, double &max_d2_after_iter0,
    double &score_lambda_iter0, double &score_lambda_after_iter0) {
    if (config.template has_typed<double>(
            std::tuple{"beammap", "priors", "max_d2_iter0"})) {
        ::get_config_value(
            config, max_d2_iter0, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "max_d2_iter0"}, {}, {0.0});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "priors", "max_d2_after_iter0"})) {
        ::get_config_value(
            config, max_d2_after_iter0, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "max_d2_after_iter0"}, {},
            {0.0});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "priors", "score_lambda_iter0"})) {
        ::get_config_value(
            config, score_lambda_iter0, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "score_lambda_iter0"}, {},
            {0.0});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "priors", "score_lambda_after_iter0"})) {
        ::get_config_value(
            config, score_lambda_after_iter0, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "score_lambda_after_iter0"}, {},
            {0.0});
    }
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_priors_behavior_config(Config &config,
                                         MissingKeys &missing_keys,
                                         InvalidKeys &invalid_keys,
                                         bool &fallback_blind,
                                         bool &align_after_iter0) {
    if (config.template has_typed<bool>(
            std::tuple{"beammap", "priors", "fallback_blind"})) {
        ::get_config_value(
            config, fallback_blind, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "fallback_blind"});
    }
    if (config.template has_typed<bool>(
            std::tuple{"beammap", "priors", "align_after_iter0"})) {
        ::get_config_value(
            config, align_after_iter0, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "align_after_iter0"});
    }
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_priors_alignment_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    std::string &scope, std::string &common_support,
    double &common_support_quantile, int &min_matches, double &max_d2,
    bool &fit_rotation, double &max_rotation_deg) {
    if (config.template has_typed<std::string>(
            std::tuple{"beammap", "priors", "alignment_scope"})) {
        ::get_config_value(
            config, scope, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "alignment_scope"},
            {"array", "common"});
    }
    if (config.template has_typed<std::string>(
            std::tuple{"beammap", "priors", "alignment_common_support"})) {
        ::get_config_value(
            config, common_support, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "alignment_common_support"},
            {"all", "overlap_box"});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "priors",
                       "alignment_common_support_quantile"})) {
        ::get_config_value(
            config, common_support_quantile, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors",
                       "alignment_common_support_quantile"},
            {}, {0.0}, {0.45});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "priors", "alignment_min_matches"})) {
        ::get_config_value(
            config, min_matches, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "alignment_min_matches"}, {},
            {3});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "priors", "alignment_max_d2"})) {
        ::get_config_value(
            config, max_d2, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "alignment_max_d2"}, {}, {0.0});
    }
    if (config.template has_typed<bool>(
            std::tuple{"beammap", "priors", "alignment_fit_rotation"})) {
        ::get_config_value(
            config, fit_rotation, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "alignment_fit_rotation"});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "priors", "alignment_max_rotation_deg"})) {
        ::get_config_value(
            config, max_rotation_deg, missing_keys, invalid_keys,
            std::tuple{"beammap", "priors", "alignment_max_rotation_deg"},
            {}, {0.0});
    }
}

