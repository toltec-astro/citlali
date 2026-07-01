#pragma once

#include <string>
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

}  // namespace citlali::cli
