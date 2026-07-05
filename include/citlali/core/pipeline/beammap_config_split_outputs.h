#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

inline std::vector<int> normalized_beammap_split_flag_values(
    std::vector<int> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

template <class Logger>
void normalize_beammap_phase_strategy(int iter_max, int &locator_iter,
                                      int &measurement_start_iter,
                                      const Logger &logger) {
    if (locator_iter != 0) {
        logger->warn(
            "beammap.phase_strategy.locator_iter={} requested, but the locator pass must be iter 0; using 0",
            locator_iter);
        locator_iter = 0;
    }
    if (measurement_start_iter <= locator_iter) {
        logger->warn(
            "beammap.phase_strategy.measurement_start_iter={} must be after locator_iter={}; using {}",
            measurement_start_iter, locator_iter, locator_iter + 1);
        measurement_start_iter = locator_iter + 1;
    }
    if (iter_max <= measurement_start_iter) {
        logger->warn(
            "beammap.iter_max={} will not run a measurement pass with measurement_start_iter={}",
            iter_max, measurement_start_iter);
    }
}

template <class Logger>
void disable_missing_beammap_priors(bool &enabled,
                                    const std::string &filepath,
                                    const Logger &logger) {
    if (!enabled || filepath != "null") {
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
void read_beammap_split_fits_config(Config &config, MissingKeys &missing_keys,
                                    InvalidKeys &invalid_keys,
                                    bool &enabled,
                                    std::vector<int> &flag_values,
                                    const Logger &logger) {
    enabled = false;
    flag_values = {0, 1};
    if (config.template has_typed<bool>(
            std::tuple{"beammap", "split_fits_by_flag", "enabled"})) {
        ::get_config_value(
            config, enabled, missing_keys, invalid_keys,
            std::tuple{"beammap", "split_fits_by_flag", "enabled"});
    }
    read_beammap_split_flag_values(config, flag_values, logger);
}

template <class Config, class InvalidKeys, class SensPsdLimits>
void read_beammap_sensitivity_config(
    Config &config, InvalidKeys &invalid_keys, double &lower_sens_factor,
    double &upper_sens_factor, SensPsdLimits &sens_psd_limits_Hz,
    std::vector<double> &sens_factors_vec,
    std::vector<double> &sens_psd_limits_Hz_vec) {
    sens_factors_vec = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "sens_factors"}, 2, invalid_keys);
    lower_sens_factor = sens_factors_vec[0];
    upper_sens_factor = sens_factors_vec[1];

    sens_psd_limits_Hz.resize(2);
    sens_psd_limits_Hz_vec = beammap_fixed_double_vector(
        config, {"beammap", "sens_psd_limits_Hz"}, 2, invalid_keys);
    sens_psd_limits_Hz = Eigen::Map<Eigen::VectorXd>(
        sens_psd_limits_Hz_vec.data(), sens_psd_limits_Hz_vec.size());
}

