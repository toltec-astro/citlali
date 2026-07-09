#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

inline std::vector<int> normalized_beammap_split_flag_values(
    std::vector<int> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

template <class Logger>
void disable_missing_beammap_priors(bool &enabled,
                                    const std::string &filepath,
                                    const Logger &logger) {
    if (!enabled || citlali::config::has_config_value(filepath)) {
        return;
    }
    logger->warn(
        "beammap.priors.enabled=true but beammap.priors.filepath is null; disabling priors");
    enabled = false;
}

template <class Config, class Logger>
void read_beammap_split_flag_values(Config &config,
                                    std::vector<int> &flag_values,
                                    const Logger &logger) {
    const auto key = std::tuple{"beammap", "split_fits_by_flag",
                                "flag_values"};
    if (!config.template has_typed<std::vector<int>>(key)) {
        return;
    }
    auto values = config.template get_typed<std::vector<int>>(key);
    if (values.empty()) {
        logger->warn(
            "beammap.split_fits_by_flag.flag_values is empty; using defaults [0, 1]");
        return;
    }
    flag_values = normalized_beammap_split_flag_values(std::move(values));
}

template <class Config, class MissingKeys, class InvalidKeys, class Logger>
citlali::config::BeammapSplitFitsByFlagConfig
read_beammap_split_fits_config(Config &config, MissingKeys &missing_keys,
                               InvalidKeys &invalid_keys,
                               const Logger &logger) {
    citlali::config::BeammapSplitFitsByFlagConfig values;
    read_optional_beammap_config_value(
        config, values.enabled, missing_keys, invalid_keys,
        std::tuple{"beammap", "split_fits_by_flag", "enabled"});
    read_beammap_split_flag_values(config, values.flag_values, logger);
    return values;
}

struct BeammapSensitivityConfigValues {
    std::vector<double> sens_factors;
    std::vector<double> sens_psd_limits_hz;
};

template <class Config, class InvalidKeys>
BeammapSensitivityConfigValues read_beammap_sensitivity_config(
    Config &config, InvalidKeys &invalid_keys) {
    BeammapSensitivityConfigValues values;
    values.sens_factors = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "sens_factors"}, 2, invalid_keys);
    values.sens_psd_limits_hz = beammap_fixed_double_vector(
        config, {"beammap", "sens_psd_limits_Hz"}, 2, invalid_keys);
    return values;
}
