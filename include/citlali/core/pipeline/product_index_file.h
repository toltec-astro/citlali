#pragma once

#include <citlali/core/utils/utils.h>
#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali_config/gitversion.h>
#include <kidscpp_config/gitversion.h>
#include <tula_config/gitversion.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace citlali::pipeline {

inline YAML::Node make_product_index_metadata_node() {
    YAML::Node node;
    node["description"].push_back("citlali data products");
    node["date"].push_back(engine_utils::current_date_time());
    node["citlali_version"].push_back(CITLALI_GIT_VERSION);
    node["kids_version"].push_back(KIDSCPP_GIT_VERSION);
    node["tula_version"].push_back(TULA_GIT_VERSION);
    return node;
}

inline std::set<std::filesystem::path> sorted_directory_entries(
    const std::filesystem::path &filepath) {
    std::set<std::filesystem::path> sorted_by_name;
    for (auto &entry : std::filesystem::directory_iterator(filepath)) {
        sorted_by_name.insert(entry);
    }
    return sorted_by_name;
}

inline std::string index_entry_name(const std::filesystem::path &path) {
    const std::string path_string{path.generic_string()};
    return path_string.substr(path_string.find_last_of("/") + 1);
}

inline void write_product_index_file(const std::filesystem::path &filepath);

inline YAML::Node make_product_index_node(const std::filesystem::path &filepath) {
    auto node = make_product_index_metadata_node();

    for (const auto &entry : sorted_directory_entries(filepath)) {
        if (std::filesystem::is_directory(entry)) {
            write_product_index_file(entry);
        }
        node["files/dirs"].push_back(index_entry_name(entry));
    }

    return node;
}

inline void write_product_index_file(const std::filesystem::path &filepath) {
    write_yaml_file_atomic(filepath / "index.yaml", make_product_index_node(filepath));
}

inline void write_final_product_index_file(
    const std::filesystem::path &root,
    const std::vector<std::filesystem::path> &required_products) {
    for (const auto &path : required_products) {
        if (path.empty() || !std::filesystem::is_regular_file(path)) {
            throw std::logic_error(
                "cannot publish final product index with a missing required product");
        }
    }
    write_product_index_file(root);
}

}  // namespace citlali::pipeline
