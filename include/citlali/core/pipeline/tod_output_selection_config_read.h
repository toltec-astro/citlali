#pragma once

// Included by tod_output_selection.h inside namespace citlali::pipeline.

template <class Config, class Key, class Logger>
void parse_tod_output_indices_config(
    Config &config, const Key &indices_key, bool output_enabled,
    const std::string &config_path, bool &select_enabled,
    std::vector<Eigen::Index> &chunks_out, const Logger &logger) {
    select_enabled = false;
    chunks_out.clear();

    if (!output_enabled || !config.has(indices_key)) {
        return;
    }

    if (config.template has_typed<std::string>(indices_key)) {
        const auto indices_value =
            config.template get_typed<std::string>(indices_key);
        if (indices_value == "all") {
            return;
        }
        logger->error(
            "{} must be \"all\" or a non-empty list of 1-based positive integers. Found \"{}\"",
            config_path, indices_value);
        std::exit(EXIT_FAILURE);
    }

    if (config.template has_typed<std::vector<int>>(indices_key)) {
        const auto chunks = config.template get_typed<std::vector<int>>(indices_key);
        if (chunks.empty()) {
            logger->error(
                "{} must be \"all\" or a non-empty list of 1-based positive integers",
                config_path);
            std::exit(EXIT_FAILURE);
        }
        select_enabled = true;
        for (const auto chunk_index : chunks) {
            if (chunk_index <= 0) {
                logger->error("{} must be 1-based positive integers. Found {}",
                              config_path, chunk_index);
                std::exit(EXIT_FAILURE);
            }
            chunks_out.push_back(static_cast<Eigen::Index>(chunk_index));
        }
        return;
    }

    logger->error("{} must be \"all\" or a list of 1-based positive integers",
                  config_path);
    std::exit(EXIT_FAILURE);
}

template <class Config, class Logger>
void parse_tod_output_indices_configs(
    Config &config, bool raw_time_chunk_enabled,
    bool processed_time_chunk_enabled, bool &raw_select_enabled,
    std::vector<Eigen::Index> &raw_chunks,
    bool &processed_select_enabled,
    std::vector<Eigen::Index> &processed_chunks, const Logger &logger) {
    parse_tod_output_indices_config(
        config, std::tuple{"timestream", "raw_time_chunk", "output",
                           "indices"},
        raw_time_chunk_enabled, "timestream.raw_time_chunk.output.indices",
        raw_select_enabled, raw_chunks, logger);
    parse_tod_output_indices_config(
        config, std::tuple{"timestream", "processed_time_chunk", "output",
                           "indices"},
        processed_time_chunk_enabled,
        "timestream.processed_time_chunk.output.indices",
        processed_select_enabled, processed_chunks, logger);
}

template <class Config, class Key, class Logger>
void read_tod_selection_count_config(
    Config &config, const Key &key, const std::string &config_path,
    int &value, const Logger &logger) {
    if (!config.template has_typed<int>(key)) {
        return;
    }
    value = config.template get_typed<int>(key);
    if (value < 0) {
        logger->error("{} must be non-negative. Found {}", config_path, value);
        std::exit(EXIT_FAILURE);
    }
}

template <class Logger>
void validate_tod_selection_mode_counts(
    const std::string &mode, int n_uniform, int n_source_dense,
    const std::string &mode_path, const std::string &n_uniform_path,
    const std::string &n_source_dense_path, const Logger &logger) {
    if (mode != "uniform_plus_source_crossing" ||
        n_uniform + n_source_dense > 0) {
        return;
    }
    logger->error("{} selects uniform_plus_source_crossing but {} + {} is zero",
                  mode_path, n_uniform_path, n_source_dense_path);
    std::exit(EXIT_FAILURE);
}

template <class Config, class ModeKey, class UniformKey, class SourceDenseKey,
          class MissingKeys, class InvalidKeys, class Logger>
void read_tod_selection_mode_config(
    Config &config, const ModeKey &mode_key, const UniformKey &n_uniform_key,
    const SourceDenseKey &n_source_dense_key, bool output_enabled,
    const std::string &mode_path, const std::string &n_uniform_path,
    const std::string &n_source_dense_path, std::string &mode,
    int &n_uniform, int &n_source_dense, MissingKeys &missing_keys,
    InvalidKeys &invalid_keys, const Logger &logger) {
    mode = "indices";
    n_uniform = 10;
    n_source_dense = 10;
    if (!output_enabled) {
        return;
    }
    if (config.has(mode_key)) {
        ::get_config_value(config, mode, missing_keys, invalid_keys, mode_key,
                           {"indices", "all",
                            "uniform_plus_source_crossing"});
    }
    read_tod_selection_count_config(
        config, n_uniform_key, n_uniform_path, n_uniform, logger);
    read_tod_selection_count_config(
        config, n_source_dense_key, n_source_dense_path, n_source_dense,
        logger);
    validate_tod_selection_mode_counts(
        mode, n_uniform, n_source_dense, mode_path, n_uniform_path,
        n_source_dense_path, logger);
}

