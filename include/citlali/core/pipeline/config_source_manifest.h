#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/output_config_copy.h>
#include <citlali/core/utils/sha256.h>

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace citlali::pipeline {

inline constexpr std::string_view config_source_manifest_schema{
    "citlali-config-source-manifest-v1"};
inline constexpr std::string_view config_source_manifest_filename{
    "config_source_manifest.yaml"};
inline constexpr std::string_view merged_config_snapshot_filename{
    "citlali_merged_config.yaml"};

struct ConfigSourceManifestEntry {
    std::size_t precedence = 0;
    std::string source_path;
    std::string copied_filename;
    std::uintmax_t size_bytes = 0;
    std::string sha256;
};

inline std::filesystem::path config_source_manifest_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / config_source_manifest_filename;
}

inline std::filesystem::path merged_config_snapshot_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / merged_config_snapshot_filename;
}

inline std::vector<ConfigSourceManifestEntry> config_source_manifest_entries(
    const std::filesystem::path &reduction_dir,
    const std::vector<std::string> &config_filepaths) {
    std::vector<ConfigSourceManifestEntry> entries;
    entries.reserve(config_filepaths.size());
    for (std::size_t index = 0; index < config_filepaths.size(); ++index) {
        const auto copied_filename =
            config_copy_filename(config_filepaths, index);
        const auto copied_path = reduction_dir / copied_filename;
        entries.push_back(ConfigSourceManifestEntry{
            index,
            config_filepaths[index],
            copied_filename,
            std::filesystem::file_size(copied_path),
            citlali::utils::sha256_file(copied_path),
        });
    }
    return entries;
}

inline YAML::Node config_source_manifest_node(
    const std::vector<ConfigSourceManifestEntry> &entries,
    std::uintmax_t merged_size_bytes, const std::string &merged_sha256) {
    YAML::Node root;
    root["schema_version"] = std::string{config_source_manifest_schema};
    root["merge_authority"] = "citlali_cli";
    root["merge_semantics"] = "ordered_later_sources_override";
    root["upstream"]["authority"] = "tolteca";
    root["upstream"]["ordered_sources_provided"] = false;
    for (const auto &entry : entries) {
        YAML::Node item;
        item["precedence"] = entry.precedence;
        item["role"] = "citlali_cli_config";
        item["source_path"] = entry.source_path;
        item["copied_filename"] = entry.copied_filename;
        item["size_bytes"] = entry.size_bytes;
        item["sha256"] = entry.sha256;
        root["sources"].push_back(item);
    }
    root["merged"]["snapshot_filename"] =
        std::string{merged_config_snapshot_filename};
    root["merged"]["serialization"] = "yaml_cpp_dump";
    root["merged"]["size_bytes"] = merged_size_bytes;
    root["merged"]["sha256"] = merged_sha256;
    return root;
}

inline void write_config_source_manifest(
    const std::filesystem::path &reduction_dir,
    const std::vector<std::string> &config_filepaths,
    const std::string &merged_config_yaml) {
    if (config_filepaths.empty()) {
        throw std::logic_error(
            "config source manifest requires at least one input file");
    }
    const auto merged_node = YAML::Load(merged_config_yaml);
    const auto merged_path = merged_config_snapshot_path(reduction_dir);
    write_yaml_file_atomic(merged_path, merged_node);
    const auto entries =
        config_source_manifest_entries(reduction_dir, config_filepaths);
    write_yaml_file_atomic(
        config_source_manifest_path(reduction_dir),
        config_source_manifest_node(
            entries, std::filesystem::file_size(merged_path),
            citlali::utils::sha256_file(merged_path)));
}

}  // namespace citlali::pipeline
