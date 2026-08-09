#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/utils/sha256.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <string>

namespace citlali::pipeline {

inline constexpr const char *timestream_output_provenance_schema_version =
    "citlali-timestream-output-provenance-v1";
inline constexpr const char *timestream_output_provenance_filename =
    "timestream_output_provenance.yaml";

inline YAML::Node tod_stream_output_requested_node(
    const citlali::config::TodStreamOutputConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["mode"] = std::string(citlali::config::to_string(config.mode));
    node["outer_context_samples"] = config.outer_context_samples;
    node["chunk_select_enabled"] = config.chunk_select_enabled;
    for (const auto chunk : config.chunks_1based) {
        node["chunks_1based"].push_back(chunk);
    }
    node["selection_mode"] =
        std::string(citlali::config::to_string(config.selection_mode));
    node["selection_n_uniform"] = config.selection_n_uniform;
    node["selection_n_source_dense"] = config.selection_n_source_dense;
    return node;
}

inline YAML::Node selected_tod_chunks_node(
    const Eigen::VectorXI &scan_to_output) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (Eigen::Index scan = 0; scan < scan_to_output.size(); ++scan) {
        if (scan_to_output(scan) >= 0) {
            node.push_back(scan + 1);
        }
    }
    return node;
}

inline YAML::Node scan_to_output_node(const Eigen::VectorXI &scan_to_output) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (Eigen::Index scan = 0; scan < scan_to_output.size(); ++scan) {
        node.push_back(scan_to_output(scan));
    }
    return node;
}

template <class Engine>
YAML::Node timestream_output_provenance_node(const Engine &engine) {
    const auto &config = timestream_config(engine);
    YAML::Node root;
    root["schema_version"] = timestream_output_provenance_schema_version;
    root["requested"]["timestream_enabled"] = config.enabled;
    root["requested"]["output"]["raw_time_chunk_enabled"] =
        config.output.raw_time_chunk_enabled;
    root["requested"]["output"]["processed_time_chunk_enabled"] =
        config.output.processed_time_chunk_enabled;
    root["requested"]["output"]["raw_time_chunk"] =
        tod_stream_output_requested_node(config.output.raw_time_chunk);
    root["requested"]["output"]["processed_time_chunk"] =
        tod_stream_output_requested_node(config.output.processed_time_chunk);
    root["requested"]["output"]["subdir_name"] =
        config.output.subdir_name;
    root["requested"]["output"]["write_eigenvalues"] =
        config.output.write_eigenvalues;
    root["requested"]["chunking"]["mode"] = config.chunking.mode;
    root["requested"]["chunking"]["value"] = config.chunking.value;
    root["requested"]["chunking"]["force"] = config.chunking.force;

    root["effective"]["output_type"] =
        std::string(citlali::config::to_string(config.output.type));
    root["effective"]["raw_time_chunk"]["enabled"] =
        raw_tod_output_enabled(engine);
    root["effective"]["raw_time_chunk"]["mode"] =
        std::string(citlali::config::to_string(
            config.output.raw_time_chunk.mode));
    root["effective"]["raw_time_chunk"]["selected_chunks_1based"] =
        selected_tod_chunks_node(engine.tod_outputs.rtc_scan_to_output_scan);
    root["effective"]["processed_time_chunk"]["enabled"] =
        processed_tod_output_enabled(engine);
    root["effective"]["processed_time_chunk"]["mode"] =
        std::string(citlali::config::to_string(
            config.output.processed_time_chunk.mode));
    root["effective"]["processed_time_chunk"]["selected_chunks_1based"] =
        selected_tod_chunks_node(engine.tod_outputs.ptc_scan_to_output_scan);
    root["effective"]["chunking"]["mode"] = config.chunking.mode;
    root["effective"]["chunking"]["value"] = config.chunking.value;
    root["effective"]["chunking"]["force"] = config.chunking.force;

    root["realized"]["n_scans"] = engine.telescope.scan_indices.cols();
    root["realized"]["raw_time_chunk"]["n_output_scans"] =
        engine.tod_outputs.n_rtc_output_scans;
    root["realized"]["raw_time_chunk"]["scan_to_output"] =
        scan_to_output_node(engine.tod_outputs.rtc_scan_to_output_scan);
    root["realized"]["processed_time_chunk"]["n_output_scans"] =
        engine.tod_outputs.n_ptc_output_scans;
    root["realized"]["processed_time_chunk"]["scan_to_output"] =
        scan_to_output_node(engine.tod_outputs.ptc_scan_to_output_scan);
    root["realized"]["files"] = YAML::Node(YAML::NodeType::Map);
    for (const auto &[stream, filepath] : engine.output_paths.tod_filename) {
        root["realized"]["files"][stream] = filepath;
    }
    if constexpr (has_raw_timestream_plan_v<Engine>) {
        const auto &raw = raw_timestream_plan(engine).realized;
        root["realized"]["rtc"]["bundle_identity"] =
            raw.rtc_bundle_identity;
        root["realized"]["rtc"]["bundle_complete"] =
            raw.rtc_bundle_complete;
        root["realized"]["rtc"]["physical_event_semantics"] =
            "unavailable";
        root["realized"]["rtc"]["products"] =
            YAML::Node(YAML::NodeType::Sequence);
        for (const auto &product : raw.rtc_products) {
            YAML::Node value;
            value["product_identity"] = product.product_identity;
            value["stage_identity"] = product.stage_identity;
            value["parent_identity"] = product.parent_identity;
            value["process_identity"] = product.process_identity;
            value["completion_identity"] = product.completion_identity;
            value["assigned_grid_identity"] =
                product.assigned_grid_identity;
            value["physical_event_semantics"] =
                product.physical_event_semantics;
            value["product_kind"] = product.product_kind;
            value["filepath"] = product.filepath;
            value["scan_id"] = product.scan_id;
            value["output_row"] = product.output_row;
            value["mini_output"] = product.mini_output;
            value["outer_output"] = product.outer_output;
            value["simulated"] = product.simulated;
            value["complete"] = product.complete;
            root["realized"]["rtc"]["products"].push_back(value);
        }
        root["realized"]["processed_time_chunk"]
            ["rtc_parent_bundle_identity"] = raw.rtc_bundle_identity;
        root["realized"]["processed_time_chunk"]
            ["rtc_downstream_acceptance"] = "not_authorized";
        if (processed_tod_output_enabled(engine) &&
            raw.rtc_bundle_complete) {
            std::string preimage =
                raw.rtc_bundle_identity + "|stream=processed_time_chunk|mode=" +
                std::string{citlali::config::to_string(
                    config.output.processed_time_chunk.mode)};
            const auto filepath = engine.output_paths.tod_filename.find("ptc");
            if (filepath != engine.output_paths.tod_filename.end()) {
                preimage += "|file=" +
                    std::filesystem::path(filepath->second)
                        .filename().string();
            }
            for (Eigen::Index scan = 0;
                 scan < engine.tod_outputs.ptc_scan_to_output_scan.size();
                 ++scan) {
                preimage += "|scan=" + std::to_string(scan) +
                    ":row=" + std::to_string(
                        engine.tod_outputs.ptc_scan_to_output_scan(scan));
            }
            root["realized"]["processed_time_chunk"]
                ["rtc_stage_identity"] =
                "rtc-processed-output-stage:sha256:" +
                citlali::utils::sha256(preimage);
            root["realized"]["processed_time_chunk"]
                ["rtc_process_identity"] =
                "rtc-processed-output-process:sha256:" +
                citlali::utils::sha256(
                    raw.rtc_bundle_identity +
                    "|consumer=existing_ptc_output|acceptance=not_authorized");
            root["realized"]["processed_time_chunk"]
                ["physical_event_semantics"] = "unavailable";
        }
    }
    return root;
}

inline std::filesystem::path timestream_output_provenance_path(
    const std::filesystem::path &observation_dir) {
    return observation_dir / timestream_output_provenance_filename;
}

template <class Engine>
void write_timestream_output_provenance_file(const Engine &engine) {
    const auto output_path = timestream_output_provenance_path(
        engine.output_paths.obsnum_dir_name);
    write_yaml_file_atomic(output_path,
                           timestream_output_provenance_node(engine));
}

}  // namespace citlali::pipeline
