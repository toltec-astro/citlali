#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

template <class Config>
void read_beammap_split_flag_values(Config &config,
                                    std::vector<int> &flag_values) {
    const auto key = std::tuple{"beammap", "split_fits_by_flag",
                                "flag_values"};
    if (!config.template has_typed<std::vector<int>>(key)) {
        return;
    }
    flag_values = config.template get_typed<std::vector<int>>(key);
}

template <class Config, class MissingKeys, class InvalidKeys>
citlali::config::BeammapSplitFitsByFlagConfig
read_beammap_split_fits_config(Config &config, MissingKeys &missing_keys,
                               InvalidKeys &invalid_keys) {
    citlali::config::BeammapSplitFitsByFlagConfig values;
    read_optional_beammap_config_value(
        config, values.enabled, missing_keys, invalid_keys,
        std::tuple{"beammap", "split_fits_by_flag", "enabled"});
    read_beammap_split_flag_values(config, values.flag_values);
    return values;
}

template <class Config, class Diagnostics>
citlali::config::BeammapSplitFitsByFlagConfig
read_beammap_split_fits_config(
    Config &config, Diagnostics &diagnostics) {
    return read_beammap_split_fits_config(
        config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
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
