#pragma once

#include <string>
#include <tula/config/core.h>
#include <tula/config/yamlconfig.h>
#include <utility>
#include <vector>

namespace citlali::cli {

template <class RuntimeConfig, class Config, class Logger, class LoadConfig,
          class MergeConfig>
Config load_config_files(const RuntimeConfig &runtime_config,
                         std::vector<std::string> &config_filepaths,
                         const Logger &logger, LoadConfig &&load_config,
                         MergeConfig &&merge_config) {
    Config config;
    const auto node_config_files = runtime_config.get_node("config_file");
    for (const auto &node : node_config_files) {
        auto filepath = node.template as<std::string>();
        config_filepaths.push_back(filepath);
        logger->info("load config from file {}", filepath);
        config = merge_config(std::move(config), load_config(filepath));
    }
    return config;
}

template <class RuntimeConfig, class Logger>
tula::config::YamlConfig load_merged_yaml_config_files(
    const RuntimeConfig &runtime_config,
    std::vector<std::string> &config_filepaths, const Logger &logger) {
    return load_config_files<RuntimeConfig, tula::config::YamlConfig>(
        runtime_config, config_filepaths, logger,
        [](const std::string &filepath) {
            return tula::config::YamlConfig::from_filepath(filepath);
        },
        [](tula::config::YamlConfig lhs,
           const tula::config::YamlConfig &rhs) {
            return tula::config::merge(lhs, rhs);
        });
}

}  // namespace citlali::cli
