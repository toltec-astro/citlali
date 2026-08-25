#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/utils/sha256.h>
#include <citlali/core/utils/utils.h>
#include <citlali_config/gitversion.h>
#include <kidscpp_config/gitversion.h>
#include <tula_config/gitversion.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::pipeline {

inline YAML::Node make_product_index_metadata_node(
    const std::string &publication_date = {}) {
    YAML::Node node;
    node["schema_version"] = "citlali-product-index-v2";
    node["description"].push_back("citlali data products");
    node["date"].push_back(
        publication_date.empty() ? engine_utils::current_date_time()
                                 : publication_date);
    node["citlali_version"].push_back(CITLALI_GIT_VERSION);
    node["citlali_revision"].push_back(CITLALI_GIT_REVISION);
    node["citlali_build_timestamp"].push_back(CITLALI_BUILD_TIMESTAMP);
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

inline bool is_mutable_operational_artifact(
    const std::filesystem::path &path) {
    const auto name = path.filename().string();
    return name == "citlali_profile.ecsv" ||
        name == "citlali.log" || name == "citlali.log.gz";
}

inline bool is_opaque_product_bundle_directory(
    const std::filesystem::path &filepath);

inline YAML::Node make_product_index_node(
    const std::filesystem::path &filepath,
    const std::string &publication_date = {}) {
    auto node = make_product_index_metadata_node(publication_date);

    for (const auto &entry : sorted_directory_entries(filepath)) {
        if (entry.filename() == "index.yaml") {
            continue;
        }
        if (std::filesystem::is_directory(entry)) {
            // Child indices are published by write_product_index_file after
            // their complete candidate inventories are known.
        }
        const auto name = index_entry_name(entry);
        node["files/dirs"].push_back(name);
        YAML::Node product;
        product["name"] = name;
        if (std::filesystem::is_regular_file(entry)) {
            if (is_mutable_operational_artifact(entry)) {
                product["kind"] = "operational_mutable_file";
                product["checksum_policy"] =
                    "excluded_live_at_canonical_publication";
            }
            else {
                product["kind"] = "file";
                product["size_bytes"] =
                    std::filesystem::file_size(entry);
                product["sha256"] =
                    citlali::utils::sha256_file(entry);
            }
        }
        else if (std::filesystem::is_directory(entry)) {
            product["kind"] = is_opaque_product_bundle_directory(entry)
                ? "opaque_product_bundle" : "directory";
            const auto child_index = entry / "index.yaml";
            if (std::filesystem::is_regular_file(child_index)) {
                product["index_sha256"] =
                    citlali::utils::sha256_file(child_index);
            }
            const auto manifest = entry / "manifest.ecsv";
            if (std::filesystem::is_regular_file(manifest)) {
                product["root_manifest_sha256"] =
                    citlali::utils::sha256_file(manifest);
            }
            const auto receipt = entry / "manifest.ecsv.sha256";
            if (std::filesystem::is_regular_file(receipt)) {
                product["root_receipt_sha256"] =
                    citlali::utils::sha256_file(receipt);
            }
        }
        else {
            throw std::logic_error(
                "product index encountered an unsupported filesystem entry");
        }
        node["products"].push_back(product);
    }

    return node;
}

inline std::string existing_product_index_publication_date(
    const std::filesystem::path &filepath) {
    const auto index = filepath / "index.yaml";
    if (!std::filesystem::is_regular_file(index)) {
        return {};
    }
    const auto existing = YAML::LoadFile(index.string());
    const auto dates = existing["date"];
    if (!dates || !dates.IsSequence() || dates.size() != 1 ||
        !dates[0].IsScalar()) {
        throw std::logic_error(
            "existing product index has an invalid publication date");
    }
    return dates[0].as<std::string>();
}

inline bool is_opaque_product_bundle_directory(
    const std::filesystem::path &filepath) {
    return filepath.extension() == ".apt-v2";
}

inline void write_product_index_file(const std::filesystem::path &filepath) {
    // Canonical compact-v2 APT bundles are immutable, receipt-complete
    // product namespaces. Their verifier rejects every member not declared by
    // manifest.ecsv, so the generic reduction index may list the bundle at
    // its parent but must never recurse into or write inside it.
    if (is_opaque_product_bundle_directory(filepath)) return;
    for (const auto &entry : sorted_directory_entries(filepath)) {
        if (std::filesystem::is_directory(entry) &&
            !is_opaque_product_bundle_directory(entry)) {
            write_product_index_file(entry);
        }
    }
    const auto publication_date =
        existing_product_index_publication_date(filepath);
    write_yaml_file_atomic(
        filepath / "index.yaml",
        make_product_index_node(filepath, publication_date));
}

inline void write_final_product_index_file(
    const std::filesystem::path &root,
    const std::vector<std::filesystem::path> &required_products) {
    for (const auto &product : required_products) {
        if (product.empty() || !std::filesystem::is_regular_file(product)) {
            throw std::logic_error(
                "cannot publish product index with a missing required product");
        }
    }
    write_product_index_file(root);
}

}  // namespace citlali::pipeline
