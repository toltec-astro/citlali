#pragma once

// Explicit, bounded diagnostic tracing for selected native detector samples.
// This is not canonical provenance and is never emitted by default. See
// ADR 0013.

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/native_cohort_product_provenance_v3.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::pipeline {

inline constexpr std::string_view native_cohort_debug_trace_schema_v1 =
    "citlali-native-cohort-debug-trace-v1";
inline constexpr std::size_t native_cohort_debug_trace_max_records_v1 =
    4096;
inline constexpr std::size_t native_cohort_debug_trace_max_selectors_v1 =
    256;

struct NativeCohortDebugTraceRequestV1 {
    bool enabled = false;
    std::size_t max_records = 0;
    std::optional<std::int64_t> scan_index;
    std::set<TimestreamNetworkId> networks;
    std::set<TimestreamDetectorColumn> detector_columns;
    std::optional<TimestreamNativeRow> first_native_row;
    std::optional<TimestreamNativeRow> past_last_native_row;

    void validate() const {
        if (!enabled) {
            if (max_records != 0 || scan_index || !networks.empty() ||
                !detector_columns.empty() || first_native_row ||
                past_last_native_row) {
                throw std::invalid_argument(
                    "disabled native debug trace carries a selection");
            }
            return;
        }
        if (max_records == 0 ||
            max_records > native_cohort_debug_trace_max_records_v1 ||
            networks.size() > native_cohort_debug_trace_max_selectors_v1 ||
            detector_columns.size() >
                native_cohort_debug_trace_max_selectors_v1 ||
            (scan_index && *scan_index < 0) ||
            (first_native_row.has_value() !=
             past_last_native_row.has_value()) ||
            (first_native_row &&
             *first_native_row >= *past_last_native_row) ||
            (!scan_index && networks.empty() && detector_columns.empty() &&
             !first_native_row)) {
            throw std::invalid_argument(
                "native debug trace requires a bounded explicit selection");
        }
        if (std::any_of(networks.begin(), networks.end(),
                        [](auto network) { return network < 0; }) ||
            std::any_of(detector_columns.begin(), detector_columns.end(),
                        [](auto detector) { return detector < 0; })) {
            throw std::invalid_argument(
                "native debug trace selectors must be nonnegative");
        }
    }
};

struct NativeCohortDebugTraceRecordV1 {
    std::int64_t scan_index = -1;
    TimestreamNetworkId network = 0;
    TimestreamNativeRow native_row = 0;
    TimestreamDetectorColumn detector_column = -1;
    NativeDetectorFlagBits delivered_flag_bits = 0;
    NativeDetectorFlagBits operation_exclusion_bits = 0;
    std::optional<std::int64_t> apt_flag;
    TimestreamNativeRevision input_revision = 0;
    TimestreamNativeRevision output_revision = 0;
    std::string action;
};

struct NativeCohortDebugTraceV1 {
    NativeCohortDebugTraceRequestV1 request;
    std::size_t matching_record_count = 0;
    bool truncated = false;
    std::vector<NativeCohortDebugTraceRecordV1> records;
};

inline NativeCohortDebugTraceV1 make_native_cohort_debug_trace_v1(
    const NativeMeasuredDetectorLedger &ledger,
    const NativePtcPreparedOperation &prepared,
    const NativeCohortDebugTraceRequestV1 &request) {
    request.validate();
    NativeCohortDebugTraceV1 trace;
    trace.request = request;
    if (!request.enabled ||
        (request.scan_index &&
         *request.scan_index != prepared.operation().scan_index)) {
        return trace;
    }
    for (const auto &group : prepared.groups()) {
        for (Eigen::Index row = 0; row < group.slot_count(); ++row) {
            for (Eigen::Index local = 0;
                 local < group.detector_count(); ++local) {
                const auto &cell = group.cell(row, local);
                if (!cell.identity) continue;
                const auto detector = group.detector_columns().at(
                    static_cast<std::size_t>(local));
                if (!request.networks.empty() &&
                    !request.networks.contains(
                        cell.identity->network_id())) continue;
                if (!request.detector_columns.empty() &&
                    !request.detector_columns.contains(detector)) continue;
                if (request.first_native_row &&
                    (cell.identity->native_row() <
                         *request.first_native_row ||
                     cell.identity->native_row() >=
                         *request.past_last_native_row)) continue;
                ++trace.matching_record_count;
                if (trace.records.size() >= request.max_records) {
                    trace.truncated = true;
                    continue;
                }
                const auto current = ledger.record(
                    {cell.identity->key(), detector});
                const bool invalid = cell.state ==
                    CoincidenceCellState::mapped_invalid;
                trace.records.push_back({
                    prepared.operation().scan_index,
                    cell.identity->network_id(),
                    cell.identity->native_row(), detector,
                    cell.delivered_flag_bits,
                    cell.operation_exclusion_bits, cell.apt_flag,
                    cell.expected_revision, current.revision,
                    invalid ? "preserved_pca_invalid"
                            : group.role() == NativePtcGroupRole::pass_through
                                ? "preserved_pass_through"
                                : "replaced_by_pca_result"});
            }
        }
    }
    return trace;
}

inline YAML::Node native_cohort_debug_trace_node_v1(
    const NativeCohortDebugTraceV1 &trace) {
    trace.request.validate();
    if (trace.records.size() > trace.request.max_records ||
        trace.matching_record_count < trace.records.size() ||
        trace.truncated !=
            (trace.matching_record_count > trace.records.size())) {
        throw std::logic_error(
            "native debug trace violates its hard record bound");
    }
    YAML::Node node;
    node["schema_version"] =
        std::string{native_cohort_debug_trace_schema_v1};
    node["artifact_class"] = "diagnostic_not_canonical";
    node["retention_required"] = false;
    node["max_records"] = trace.request.max_records;
    if (trace.request.scan_index) {
        node["selection"]["scan_index"] =
            *trace.request.scan_index;
    }
    for (const auto network : trace.request.networks) {
        node["selection"]["networks"].push_back(network);
    }
    for (const auto detector : trace.request.detector_columns) {
        node["selection"]["detector_columns"].push_back(detector);
    }
    if (trace.request.first_native_row) {
        node["selection"]["native_row_interval"]["first"] =
            *trace.request.first_native_row;
        node["selection"]["native_row_interval"]["past_last"] =
            *trace.request.past_last_native_row;
    }
    node["matching_record_count"] = trace.matching_record_count;
    node["captured_record_count"] = trace.records.size();
    node["truncated"] = trace.truncated;
    for (const auto &record : trace.records) {
        YAML::Node item;
        item["scan_index"] = record.scan_index;
        item["network"] = record.network;
        item["native_row"] = record.native_row;
        item["detector_column"] = record.detector_column;
        item["delivered_flag_bits"] = record.delivered_flag_bits;
        item["operation_exclusion_bits"] =
            record.operation_exclusion_bits;
        item["apt_flag"]["available"] = record.apt_flag.has_value();
        if (record.apt_flag) item["apt_flag"]["value"] = *record.apt_flag;
        item["input_revision"] = record.input_revision;
        item["output_revision"] = record.output_revision;
        item["action"] = record.action;
        node["records"].push_back(item);
    }
    return node;
}

inline void write_native_cohort_debug_trace_file_v1(
    const std::filesystem::path &path,
    const NativeCohortDebugTraceV1 &trace) {
    if (!trace.request.enabled) {
        throw std::logic_error(
            "disabled native debug trace cannot be published");
    }
    write_yaml_file_atomic(path, native_cohort_debug_trace_node_v1(trace));
}

}  // namespace citlali::pipeline
