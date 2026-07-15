#pragma once

#include <citlali/core/pipeline/config_leaf_schema_generated.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace citlali::pipeline {

inline bool is_known_low_level_config_node(std::string_view path) {
    return std::binary_search(
        known_low_level_config_node_paths.begin(),
        known_low_level_config_node_paths.end(), path);
}

inline bool is_externally_owned_low_level_config_root(
    std::string_view path) {
    return std::binary_search(
        externally_owned_low_level_config_roots.begin(),
        externally_owned_low_level_config_roots.end(), path);
}

inline void collect_unknown_low_level_config_nodes(
    const YAML::Node &node, const std::string &normalized_path,
    const std::vector<std::string> &display_path,
    std::vector<std::vector<std::string>> &unknown_paths) {
    if (!normalized_path.empty() &&
        !is_known_low_level_config_node(normalized_path)) {
        unknown_paths.push_back(display_path);
        return;
    }
    if (is_externally_owned_low_level_config_root(normalized_path)) {
        return;
    }

    if (node.IsMap()) {
        for (const auto &entry : node) {
            if (!entry.first.IsScalar()) {
                unknown_paths.push_back(display_path);
                continue;
            }
            const auto key = entry.first.Scalar();
            auto child_display_path = display_path;
            child_display_path.push_back(key);
            const auto child_normalized_path = normalized_path.empty()
                ? key
                : normalized_path + "." + key;
            collect_unknown_low_level_config_nodes(
                entry.second, child_normalized_path,
                child_display_path, unknown_paths);
        }
        return;
    }

    if (node.IsSequence()) {
        const auto child_normalized_path = normalized_path + "[]";
        for (std::size_t index = 0; index < node.size(); ++index) {
            auto child_display_path = display_path;
            child_display_path.push_back(std::to_string(index));
            collect_unknown_low_level_config_nodes(
                node[index], child_normalized_path,
                child_display_path, unknown_paths);
        }
    }
}

template <class Config, class Diagnostics>
bool validate_low_level_config_schema(
    const Config &config, Diagnostics &diagnostics) {
    const auto invalid_before = diagnostics.invalid_key_paths().size();
    collect_unknown_low_level_config_nodes(
        config.get_node(), "", {}, diagnostics.invalid_key_paths());
    return diagnostics.invalid_key_paths().size() == invalid_before;
}

}  // namespace citlali::pipeline
