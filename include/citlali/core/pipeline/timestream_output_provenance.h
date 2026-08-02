#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/observation_exposure_time.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/timestream_scan_context.h>

#include <tula/eigen.h>

#include <citlali/core/pipeline/sci_align_scan_contract.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *timestream_output_provenance_schema_version =
    "citlali-timestream-output-provenance-v2";
inline constexpr const char *timestream_output_provenance_filename =
    "timestream_output_provenance.yaml";

enum class TimestreamOutputProvenanceStage {
    observation_setup_plan,
    observation_execution_completed,
};

inline bool timestream_output_execution_completed(
    TimestreamOutputProvenanceStage stage) {
    return stage ==
           TimestreamOutputProvenanceStage::observation_execution_completed;
}

inline const char *timestream_output_evidence_stage_name(
    TimestreamOutputProvenanceStage stage) {
    return timestream_output_execution_completed(stage)
               ? "observation_execution_completed"
               : "observation_setup_plan";
}

template <class T, class = void>
struct has_timestream_output_provenance_state : std::false_type {};

template <class T>
struct has_timestream_output_provenance_state<
    T, std::void_t<
           decltype(std::declval<const T &>()
                        .tod_outputs.rtc_scan_to_output_scan),
           decltype(std::declval<const T &>()
                        .tod_outputs.ptc_scan_to_output_scan),
           decltype(std::declval<const T &>().telescope.scan_indices),
           decltype(std::declval<const T &>()
                        .output_paths.obsnum_dir_name)>> : std::true_type {};

template <class T>
inline constexpr bool has_timestream_output_provenance_state_v =
    has_timestream_output_provenance_state<T>::value;

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

inline YAML::Node selected_tod_output_windows_node(
    const sci_align::ScanWindowPlan &plan,
    const Eigen::VectorXI &scan_to_output,
    Eigen::Index n_output_scans, bool context_outer) {
    sci_align::validate_scan_window_plan(plan);
    const auto admitted_scan_count = sci_align::checked_scan_container_size(
        plan.compatibility_to_stable_id.size(),
        "admitted ALIGN scan count");
    if (n_output_scans < 0 ||
        n_output_scans > scan_to_output.size() ||
        scan_to_output.size() != admitted_scan_count) {
        throw std::logic_error(
            "TOD output selection does not match the admitted ALIGN scan plan");
    }

    std::vector<unsigned char> output_rows_seen(
        static_cast<std::size_t>(n_output_scans), 0);
    Eigen::Index selected_count = 0;
    YAML::Node node(YAML::NodeType::Sequence);
    for (Eigen::Index compatibility_ordinal = 0;
         compatibility_ordinal < scan_to_output.size();
         ++compatibility_ordinal) {
        const Eigen::Index output_row =
            scan_to_output(compatibility_ordinal);
        if (output_row < -1 || output_row >= n_output_scans) {
            throw std::logic_error(
                "TOD output selection contains an invalid output row");
        }
        if (output_row < 0) {
            continue;
        }
        auto &seen = output_rows_seen.at(
            static_cast<std::size_t>(output_row));
        if (seen != 0) {
            throw std::logic_error(
                "TOD output selection is not a bijection onto output rows");
        }
        seen = 1;
        ++selected_count;

        const Eigen::Index stable_id =
            plan.compatibility_to_stable_id.at(
                static_cast<std::size_t>(compatibility_ordinal));
        const auto &record = plan.records.at(
            static_cast<std::size_t>(stable_id));
        if (!record.legacy_processing_admitted ||
            record.compatibility_ordinal != compatibility_ordinal) {
            throw std::logic_error(
                "TOD output selection references a nonadmitted ALIGN record");
        }
        const auto &interval = context_outer
            ? sci_align::compatibility_context_window(record)
            : sci_align::compatibility_science_window(record);
        if (!interval.valid() || interval.empty()) {
            throw std::logic_error(
                "TOD output selection references an empty output interval");
        }

        YAML::Node value;
        value["stable_processing_record_id"] = stable_id;
        value["compatibility_ordinal"] = compatibility_ordinal;
        value["output_row"] = output_row;
        value["output_interval"]["start"] = interval.start;
        value["output_interval"]["stop"] = interval.stop;
        value["interval_convention"] = "half_open_start_stop";
        value["interval_authority"] =
            context_outer ? "context_outer" : "science_inner";
        node.push_back(value);
    }
    if (selected_count != n_output_scans ||
        std::any_of(output_rows_seen.begin(), output_rows_seen.end(),
                    [](unsigned char value) { return value == 0; })) {
        throw std::logic_error(
            "TOD output selection is not a complete bijection onto output rows");
    }
    return node;
}

template <class Telescope, class = void>
struct has_sci_align_scan_plan : std::false_type {};

template <class Telescope>
struct has_sci_align_scan_plan<
    Telescope,
    std::void_t<decltype(std::declval<const Telescope &>().scan_plan)>>
    : std::true_type {};

inline YAML::Node sci_align_scan_plan_node(
    const sci_align::ScanWindowPlan &plan) {
    sci_align::validate_scan_window_plan(plan);
    YAML::Node node;
    node["identity"] = "zero_based_stable_processing_record_id";
    node["interval_convention"] = "half_open_start_stop";
    node["physical_identity"] =
        "zero_based_physical_window_id_when_authority_available";
    node["policy"] = plan.policy;
    node["requested_value"] = plan.requested_value;
    node["effective_duration_sec"] = plan.effective_duration_sec;
    node["observation_sample_count"] = plan.observation_sample_count;
    node["physical_identity_count"] = plan.physical_records.size();
    node["identity_count"] = plan.records.size();
    node["compatibility_admitted_count"] =
        plan.compatibility_to_stable_id.size();
    node["physical_records"] = YAML::Node{YAML::NodeType::Sequence};
    for (const auto stable_id : plan.compatibility_to_stable_id) {
        node["compatibility_ordinal_to_stable_id"].push_back(stable_id);
    }
    for (const auto &physical : plan.physical_records) {
        YAML::Node value;
        value["stable_id"] = physical.stable_id;
        value["interval"]["start"] = physical.interval.start;
        value["interval"]["stop"] = physical.interval.stop;
        value["authority"] = physical.authority;
        node["physical_records"].push_back(value);
    }
    for (const auto &record : plan.records) {
        YAML::Node value;
        value["stable_id"] = record.stable_id;
        value["status"] = std::string{sci_align::to_string(record.status)};
        value["identity_authority"] = record.identity_authority;
        if (record.physical_id.has_value()) {
            value["physical_id"] = *record.physical_id;
        }
        else {
            value["physical_id"] = YAML::Node{YAML::NodeType::Null};
        }
        value["processing"]["start"] = record.processing.start;
        value["processing"]["stop"] = record.processing.stop;
        value["science"]["start"] = record.science.start;
        value["science"]["stop"] = record.science.stop;
        value["context"]["start"] = record.context.start;
        value["context"]["stop"] = record.context.stop;
        value["legacy_processing_admitted"] =
            record.legacy_processing_admitted;
        value["compatibility_ordinal"] = record.compatibility_ordinal;
        if (record.legacy_processing_admitted) {
            const auto &compatibility_science =
                sci_align::compatibility_science_window(record);
            const auto &compatibility_context =
                sci_align::compatibility_context_window(record);
            value["compatibility_science"]["start"] =
                compatibility_science.start;
            value["compatibility_science"]["stop"] =
                compatibility_science.stop;
            value["compatibility_context"]["start"] =
                compatibility_context.start;
            value["compatibility_context"]["stop"] =
                compatibility_context.stop;
        }
        node["records"].push_back(value);
    }
    return node;
}

inline const char *alignment_term_availability_name(
    AlignmentTermAvailability value) {
    switch (value) {
        case AlignmentTermAvailability::available:
            return "available";
        case AlignmentTermAvailability::available_conditional:
            return "available_conditional";
        case AlignmentTermAvailability::unavailable_input:
            return "unavailable_input";
        case AlignmentTermAvailability::not_applicable:
            return "not_applicable";
        case AlignmentTermAvailability::not_persisted_standard:
            return "not_persisted_standard";
    }
    throw std::logic_error("invalid ALIGN availability state");
}

inline bool alignment_hwpr_summary_is_empty(
    const AlignmentHwprSummary &hwpr) {
    return !hwpr.observation_resolved && !hwpr.producer_input_present &&
           !hwpr.aligned_angle_available && !hwpr.intensity_eligible &&
           !hwpr.polarization_eligible && hwpr.policy.empty() &&
           hwpr.availability_reason.empty() &&
           hwpr.physical_timestamp_semantics.empty() &&
           hwpr.demodulation_semantics.empty();
}

inline void validate_bounded_nonpolarimetric_hwpr_summary(
    const AlignmentHwprSummary &hwpr, Eigen::Index hwpr_start_index,
    Eigen::Index hwpr_end_index) {
    const auto expected = bounded_nonpolarimetric_hwpr_summary(
        hwpr.producer_input_present);
    if (!hwpr.observation_resolved || hwpr.aligned_angle_available ||
        !hwpr.intensity_eligible || hwpr.polarization_eligible ||
        hwpr.policy != expected.policy ||
        hwpr.availability_reason != expected.availability_reason ||
        hwpr.physical_timestamp_semantics !=
            expected.physical_timestamp_semantics ||
        hwpr.demodulation_semantics != expected.demodulation_semantics ||
        hwpr_start_index != -1 || hwpr_end_index != -1) {
        throw std::logic_error(
            "initialized ALIGN state has an invalid bounded optional-HWPR contract");
    }
}

inline bool alignment_processing_support_is_empty(
    const AlignmentProcessingSupportSummary &support) {
    return !support.observation_resolved && support.signal_domain.empty() &&
           support.synthesized_processing_occurrence_count == 0 &&
           support.unavailable_processing_occurrence_count == 0 &&
           support.guarded_original_processing_occurrence_count == 0 &&
           support.full_network_unusable_original_occurrence_count == 0;
}

inline std::uint64_t validate_alignment_processing_runs(
    const std::vector<AlignmentIndexRun> &runs,
    const Eigen::VectorXi &interface_mask, Eigen::Index context_start,
    Eigen::Index context_stop, bool expect_missing,
    const char *run_identity) {
    std::uint64_t count = 0;
    Eigen::Index previous_stop = context_start;
    bool first = true;
    for (const auto &run : runs) {
        if (run.start < context_start || run.stop > context_stop ||
            run.stop <= run.start ||
            (!first && run.start <= previous_stop)) {
            throw std::logic_error(
                std::string{"ALIGN planned "} + run_identity +
                " runs are not compact, ordered half-open intervals");
        }
        for (Eigen::Index sample = run.start; sample < run.stop; ++sample) {
            const bool is_missing = interface_mask(sample) == 0;
            if (is_missing != expect_missing) {
                throw std::logic_error(
                    std::string{"ALIGN planned "} + run_identity +
                    " run conflicts with original acquisition support");
            }
        }
        if (expect_missing &&
            ((run.start > context_start &&
              interface_mask(run.start - 1) == 0) ||
             (run.stop < context_stop && interface_mask(run.stop) == 0))) {
            throw std::logic_error(
                std::string{"ALIGN planned "} + run_identity +
                " run is not maximal within its processing context");
        }
        count = checked_alignment_count_add(
            count, static_cast<std::uint64_t>(run.stop - run.start),
            "ALIGN planned processing run count");
        previous_stop = run.stop;
        first = false;
    }
    return count;
}

inline bool alignment_processing_runs_overlap(
    const std::vector<AlignmentIndexRun> &left,
    const std::vector<AlignmentIndexRun> &right) {
    std::size_t left_index = 0;
    std::size_t right_index = 0;
    while (left_index < left.size() && right_index < right.size()) {
        const auto &left_run = left[left_index];
        const auto &right_run = right[right_index];
        if (left_run.stop <= right_run.start) {
            ++left_index;
        }
        else if (right_run.stop <= left_run.start) {
            ++right_index;
        }
        else {
            return true;
        }
    }
    return false;
}

inline bool alignment_missing_run_has_native_brackets(
    const AlignmentIndexRun &run, const Eigen::VectorXi &interface_mask) {
    Eigen::Index observation_start = run.start;
    while (observation_start > 0 &&
           interface_mask(observation_start - 1) == 0) {
        --observation_start;
    }
    Eigen::Index observation_stop = run.stop;
    while (observation_stop < interface_mask.size() &&
           interface_mask(observation_stop) == 0) {
        ++observation_stop;
    }
    return observation_start > 0 && observation_stop < interface_mask.size();
}

inline std::pair<double, double>
alignment_exception_linear_source_weights(
    const AlignmentExceptionRun &exception, Eigen::Index target_slot) {
    if (exception.action != "bounded_continuity_candidate" ||
        target_slot < exception.start || target_slot >= exception.stop ||
        exception.left_source_slot < 0 ||
        exception.right_source_slot <= exception.left_source_slot ||
        !(exception.left_source_slot < target_slot &&
          target_slot < exception.right_source_slot)) {
        throw std::logic_error(
            "ALIGN continuity weights require a target inside a bounded exception");
    }
    const auto denominator = static_cast<long double>(
        exception.right_source_slot - exception.left_source_slot);
    const auto right_weight =
        static_cast<long double>(target_slot - exception.left_source_slot) /
        denominator;
    return {static_cast<double>(1.0L - right_weight),
            static_cast<double>(right_weight)};
}

inline const AlignmentExceptionRun &
require_alignment_continuity_source_exception(
    const TimestreamAlignmentState &alignment,
    const AlignmentChunkDisposition &disposition,
    const AlignmentIndexRun &planned_run) {
    const AlignmentExceptionRun *match = nullptr;
    for (const auto &exception : alignment.exceptions) {
        if (exception.interface_id != disposition.interface_id ||
            exception.field_id != "detector_acquisition" ||
            exception.action != "bounded_continuity_candidate" ||
            exception.start > planned_run.start ||
            exception.stop < planned_run.stop) {
            continue;
        }
        if (match != nullptr) {
            throw std::logic_error(
                "ALIGN planned continuity run has ambiguous exception identity");
        }
        match = &exception;
    }
    if (match == nullptr) {
        throw std::logic_error(
            "ALIGN planned continuity run has no compact source-endpoint identity");
    }
    return *match;
}

inline void validate_alignment_processing_support(
    const TimestreamAlignmentState &alignment,
    const sci_align::ScanWindowPlan *scan_plan = nullptr) {
    const auto &processing = alignment.processing_support;
    if (!processing.observation_resolved) {
        if (!alignment.chunk_dispositions.empty() ||
            !alignment_processing_support_is_empty(processing)) {
            throw std::logic_error(
                "unresolved ALIGN processing support retains planned state");
        }
        if (scan_plan != nullptr &&
            !scan_plan->compatibility_to_stable_id.empty()) {
            throw std::logic_error(
                "admitted ALIGN scans have no observation-resolved processing plan");
        }
        return;
    }

    if (processing.signal_domain.empty() ||
        !(citlali::config::is_xs_tod_type(processing.signal_domain) ||
          citlali::config::is_rs_tod_type(processing.signal_domain) ||
          citlali::config::is_is_tod_type(processing.signal_domain) ||
          citlali::config::is_qs_tod_type(processing.signal_domain))) {
        throw std::logic_error(
            "observation-resolved ALIGN processing plan has no valid signal domain");
    }

    if (scan_plan == nullptr) {
        throw std::logic_error(
            "observation-resolved ALIGN processing support requires its stable scan plan");
    }
    sci_align::validate_scan_window_plan(*scan_plan);
    if (alignment.masks.size() != alignment.interfaces.size()) {
        throw std::logic_error(
            "ALIGN processing support lacks one mask per detector interface");
    }
    const auto maximum_disposition_count = checked_alignment_size_product(
        scan_plan->compatibility_to_stable_id.size(),
        alignment.interfaces.size(), "ALIGN chunk disposition count");
    if (alignment.chunk_dispositions.size() > maximum_disposition_count) {
        throw std::logic_error(
            "ALIGN sparse processing plan exceeds admitted scan/interface cardinality");
    }
    if (!std::is_sorted(alignment.chunk_dispositions.begin(),
                        alignment.chunk_dispositions.end(),
                        alignment_chunk_disposition_key_less)) {
        throw std::logic_error(
            "ALIGN sparse chunk dispositions are not in deterministic key order");
    }
    const auto scan_record_count = sci_align::checked_scan_container_size(
        scan_plan->records.size(), "ALIGN stable scan record count");

    std::map<std::string, std::size_t> interface_indices;
    std::set<Eigen::Index> interface_roach_indices;
    for (std::size_t index = 0; index < alignment.interfaces.size(); ++index) {
        const auto &interface = alignment.interfaces[index];
        if (interface.interface_id.empty() || interface.roach_index < 0 ||
            !interface_indices.emplace(interface.interface_id, index).second ||
            !interface_roach_indices.insert(interface.roach_index).second) {
            throw std::logic_error(
                "ALIGN processing support has invalid detector interface identities");
        }
        if (alignment.masks[index].size() != alignment.common_time.size()) {
            throw std::logic_error(
                "ALIGN processing support mask does not cover the common axis");
        }
    }

    std::set<std::pair<Eigen::Index, std::string>> disposition_keys;
    std::uint64_t synthesized_count = 0;
    std::uint64_t unavailable_count = 0;
    std::uint64_t guarded_count = 0;
    std::uint64_t full_network_unusable_original_count = 0;
    std::uint64_t unique_science_guarded_original_count = 0;
    std::uint64_t unique_gap_policy_eligible_original_count = 0;
    std::vector<std::vector<AlignmentIndexRun>> unique_synthesized_support(
        alignment.masks.size());

    for (const auto &disposition : alignment.chunk_dispositions) {
        const auto interface_it =
            interface_indices.find(disposition.interface_id);
        if (disposition.stable_scan_id < 0 ||
            disposition.compatibility_ordinal < 0 ||
            interface_it == interface_indices.end() ||
            disposition.roach_index !=
                alignment.interfaces[interface_it->second].roach_index ||
            !disposition_keys
                 .emplace(disposition.compatibility_ordinal,
                          disposition.interface_id)
                 .second ||
            disposition.context_start < 0 ||
            disposition.context_stop <= disposition.context_start ||
            disposition.context_stop > alignment.common_time.size()) {
            throw std::logic_error(
                "ALIGN chunk disposition has invalid identity or support bounds");
        }
        if (!alignment_chunk_disposition_is_exceptional(disposition)) {
            throw std::logic_error(
                "ALIGN sparse processing plan persists a spurious ordinary disposition");
        }

        const auto &mask = alignment.masks[interface_it->second];
        if (disposition.stable_scan_id >= scan_record_count ||
            static_cast<std::uint64_t>(
                disposition.compatibility_ordinal) >=
                static_cast<std::uint64_t>(
                    scan_plan->compatibility_to_stable_id.size())) {
            throw std::logic_error(
                "ALIGN chunk disposition names an unknown admitted scan");
        }
        const auto &scan_record = scan_plan->records.at(
            static_cast<std::size_t>(disposition.stable_scan_id));
        const auto &compatibility_science =
            sci_align::compatibility_science_window(scan_record);
        const auto &compatibility_context =
            sci_align::compatibility_context_window(scan_record);
        if (!scan_record.legacy_processing_admitted ||
            scan_record.compatibility_ordinal !=
                disposition.compatibility_ordinal ||
            scan_plan->compatibility_to_stable_id.at(
                static_cast<std::size_t>(
                    disposition.compatibility_ordinal)) !=
                disposition.stable_scan_id ||
            compatibility_context.start != disposition.context_start ||
            compatibility_context.stop != disposition.context_stop) {
            throw std::logic_error(
                "ALIGN chunk disposition conflicts with its stable scan support");
        }

        Eigen::Index actual_context_missing = 0;
        for (Eigen::Index sample = disposition.context_start;
             sample < disposition.context_stop; ++sample) {
            if (mask(sample) == 0) {
                ++actual_context_missing;
            }
        }

        Eigen::Index actual_science_missing = 0;
        Eigen::Index actual_science_longest_missing = 0;
        Eigen::Index current_missing = 0;
        for (Eigen::Index sample = compatibility_science.start;
             sample < compatibility_science.stop; ++sample) {
            if (mask(sample) == 0) {
                ++actual_science_missing;
                ++current_missing;
                actual_science_longest_missing = std::max(
                    actual_science_longest_missing, current_missing);
            }
            else {
                current_missing = 0;
            }
        }
        const Eigen::Index context_size =
            disposition.context_stop - disposition.context_start;
        const Eigen::Index science_size = compatibility_science.size();
        const bool expected_unusable =
            static_cast<long double>(actual_science_missing) * 4.0L >
                static_cast<long double>(science_size) ||
            static_cast<long double>(actual_science_longest_missing) * 4.0L >
                static_cast<long double>(science_size);
        const bool expected_continuity =
            citlali::config::is_xs_tod_type(processing.signal_domain);
        if (disposition.cumulative_missing_count !=
                actual_science_missing ||
            disposition.longest_missing_run_count !=
                actual_science_longest_missing ||
            disposition.full_network_unusable != expected_unusable ||
            disposition.continuity_surrogate_permitted !=
                expected_continuity) {
            throw std::logic_error(
                "ALIGN chunk disposition conflicts with its support/action policy");
        }

        const auto synthesized = validate_alignment_processing_runs(
            disposition.synthesized_missing_runs, mask,
            disposition.context_start, disposition.context_stop, true,
            "continuity-surrogate");
        const auto unavailable = validate_alignment_processing_runs(
            disposition.unavailable_missing_runs, mask,
            disposition.context_start, disposition.context_stop, true,
            "unavailable-missing");
        const auto guarded = validate_alignment_processing_runs(
            disposition.processing_guard_runs, mask,
            disposition.context_start, disposition.context_stop, false,
            "guarded-original");
        const auto partitioned_missing = checked_alignment_count_add(
            synthesized, unavailable,
            "ALIGN planned missing-support partition");
        if (alignment_processing_runs_overlap(
                disposition.synthesized_missing_runs,
                disposition.unavailable_missing_runs) ||
            partitioned_missing !=
                static_cast<std::uint64_t>(actual_context_missing) ||
            (disposition.full_network_unusable &&
             (synthesized != 0 || guarded != 0)) ||
            (!disposition.continuity_surrogate_permitted &&
             synthesized != 0)) {
            throw std::logic_error(
                "ALIGN planned processing actions do not partition chunk support");
        }
        for (const auto &run : disposition.synthesized_missing_runs) {
            if (!alignment_missing_run_has_native_brackets(run, mask)) {
                throw std::logic_error(
                    "ALIGN continuity surrogate lacks native bracketing support");
            }
            (void)require_alignment_continuity_source_exception(
                alignment, disposition, run);
            add_to_compact_alignment_run_union(
                unique_synthesized_support[interface_it->second], run);
        }
        if (disposition.continuity_surrogate_permitted &&
            !disposition.full_network_unusable) {
            for (const auto &run : disposition.unavailable_missing_runs) {
                if (alignment_missing_run_has_native_brackets(run, mask)) {
                    throw std::logic_error(
                        "bounded ALIGN continuity support was marked unavailable");
                }
            }
        }

        synthesized_count = checked_alignment_count_add(
            synthesized_count, synthesized,
            "ALIGN synthesized processing occurrence count");
        unavailable_count = checked_alignment_count_add(
            unavailable_count, unavailable,
            "ALIGN unavailable processing occurrence count");
        guarded_count = checked_alignment_count_add(
            guarded_count, guarded,
            "ALIGN guarded processing occurrence count");
        if (disposition.full_network_unusable) {
            full_network_unusable_original_count =
                checked_alignment_count_add(
                    full_network_unusable_original_count,
                    static_cast<std::uint64_t>(
                        context_size - actual_context_missing),
                    "ALIGN unusable original occurrence count");
        }
        else {
            std::uint64_t science_original_count = 0;
            for (Eigen::Index sample = compatibility_science.start;
                 sample < compatibility_science.stop; ++sample) {
                if (mask(sample) != 0) {
                    ++science_original_count;
                }
            }
            std::uint64_t science_guarded_count = 0;
            for (const auto &run : disposition.processing_guard_runs) {
                const Eigen::Index intersection_start = std::max(
                    run.start, compatibility_science.start);
                const Eigen::Index intersection_stop = std::min(
                    run.stop, compatibility_science.stop);
                if (intersection_stop > intersection_start) {
                    science_guarded_count = checked_alignment_count_add(
                        science_guarded_count,
                        static_cast<std::uint64_t>(
                            intersection_stop - intersection_start),
                        "ALIGN science-window guarded count");
                }
            }
            if (science_guarded_count > science_original_count) {
                throw std::logic_error(
                    "ALIGN science-window guarded support exceeds original support");
            }
            unique_science_guarded_original_count =
                checked_alignment_count_add(
                    unique_science_guarded_original_count,
                    science_guarded_count,
                    "ALIGN unique science-window guarded count");
            unique_gap_policy_eligible_original_count =
                checked_alignment_count_add(
                    unique_gap_policy_eligible_original_count,
                    science_original_count - science_guarded_count,
                    "ALIGN unique science-eligible count");
        }
    }

    const auto admitted_scan_count = sci_align::checked_scan_container_size(
        scan_plan->compatibility_to_stable_id.size(),
        "admitted ALIGN scan count");
    for (Eigen::Index ordinal = 0;
         ordinal < admitted_scan_count;
         ++ordinal) {
        const auto stable_id = scan_plan->compatibility_to_stable_id.at(
            static_cast<std::size_t>(ordinal));
        const auto &record = scan_plan->records.at(
            static_cast<std::size_t>(stable_id));
        const auto &compatibility_science =
            sci_align::compatibility_science_window(record);
        const auto &compatibility_context =
            sci_align::compatibility_context_window(record);
        if (record.compatibility_ordinal != ordinal) {
            throw std::logic_error(
                "ALIGN admitted scan has an invalid compatibility ordinal");
        }
        for (std::size_t interface_index = 0;
             interface_index < alignment.interfaces.size();
             ++interface_index) {
            const auto &interface = alignment.interfaces[interface_index];
            if (disposition_keys.count(
                    {ordinal, interface.interface_id}) != 0) {
                continue;
            }
            const auto &mask = alignment.masks[interface_index];
            if (compatibility_context.stop > mask.size()) {
                throw std::logic_error(
                    "ALIGN admitted scan context exceeds detector support");
            }
            for (Eigen::Index sample = compatibility_context.start;
                 sample < compatibility_context.stop; ++sample) {
                if (mask(sample) != 1) {
                    throw std::logic_error(
                        "ALIGN sparse processing plan omits a nonordinary scan/interface disposition");
                }
            }
            unique_gap_policy_eligible_original_count =
                checked_alignment_count_add(
                    unique_gap_policy_eligible_original_count,
                    static_cast<std::uint64_t>(
                        compatibility_science.size()),
                    "ALIGN implicit ordinary science-eligible count");
        }
    }

    if (processing.synthesized_processing_occurrence_count !=
            synthesized_count ||
        processing.unavailable_processing_occurrence_count !=
            unavailable_count ||
        processing.guarded_original_processing_occurrence_count !=
            guarded_count ||
        processing.full_network_unusable_original_occurrence_count !=
            full_network_unusable_original_count) {
        throw std::logic_error(
            "ALIGN processing-support summary conflicts with planned runs");
    }
    if (alignment.support.guarded_original_count !=
            unique_science_guarded_original_count ||
        alignment.support.gap_policy_eligible_original_count !=
            unique_gap_policy_eligible_original_count) {
        throw std::logic_error(
            "ALIGN unique science-window support conflicts with planned dispositions");
    }
    std::uint64_t unique_synthesized_count = 0;
    for (const auto &support : unique_synthesized_support) {
        unique_synthesized_count = checked_alignment_count_add(
            unique_synthesized_count,
            compact_alignment_run_union_cardinality(support),
            "ALIGN unique synthesized support count");
    }
    const auto interface_slot_capacity =
        checked_alignment_interface_slot_capacity(
            alignment.support.nominal_slot_count,
            alignment.interfaces.size());
    if (alignment.support.acquired_original_count >
            interface_slot_capacity ||
        unique_synthesized_count >
            interface_slot_capacity -
                alignment.support.acquired_original_count ||
        alignment.support.synthesized_count !=
            unique_synthesized_count ||
        alignment.support.unavailable_count !=
            interface_slot_capacity -
                alignment.support.acquired_original_count -
                unique_synthesized_count) {
        throw std::logic_error(
            "ALIGN unique synthesized/unavailable support conflicts with planned dispositions");
    }
}

inline void validate_compact_alignment_provenance(
    const TimestreamAlignmentState &alignment,
    const sci_align::ScanWindowPlan *scan_plan = nullptr) {
    if (!alignment.grid.initialized) {
        if (alignment.common_time.size() != 0 || !alignment.masks.empty() ||
            !alignment.network_masks.empty() ||
            !alignment.network_times.empty() ||
            alignment.governing_compatibility_axis.initialized ||
            !alignment.interfaces.empty() || !alignment.exceptions.empty() ||
            !alignment.chunk_dispositions.empty() ||
            !alignment_hwpr_summary_is_empty(alignment.hwpr) ||
            !alignment_processing_support_is_empty(
                alignment.processing_support) ||
            alignment.support.nominal_slot_count != 0 ||
            alignment.support.acquired_original_count != 0 ||
            alignment.support.timing_coordinate_valid_original_count != 0 ||
            alignment.support.synthesized_count != 0 ||
            alignment.support.unavailable_count != 0 ||
            alignment.support.guarded_original_count != 0 ||
            alignment.support.gap_policy_eligible_original_count != 0 ||
            alignment.support.nominal_span_sec != 0.0 ||
            alignment.support
                    .acquired_original_cadence_weighted_support_sec != 0.0 ||
            !alignment.field_registry_version.empty()) {
            throw std::logic_error(
                "uninitialized alignment retains compact realized state");
        }
        return;
    }

    const auto exposure = aligned_observation_exposure_summary(alignment);
    const auto &grid = alignment.grid;
    const auto &support = alignment.support;
    validate_governing_compatibility_assigned_times(alignment);
    if (!std::isfinite(grid.phase_sec) ||
        !std::isfinite(grid.exclusive_half_cell_sec) ||
        grid.exclusive_half_cell_sec <= 0.0 ||
        grid.exclusive_half_cell_sec != grid.cadence_sec / 2.0 ||
        grid.assignment_operator.empty() || grid.phase_semantics.empty() ||
        grid.physical_timestamp_semantics.empty()) {
        throw std::logic_error(
            "initialized alignment has an incomplete grid contract");
    }
    if (alignment.field_registry_version.empty()) {
        throw std::logic_error(
            "initialized alignment has no field-registry identity");
    }
    validate_bounded_nonpolarimetric_hwpr_summary(
        alignment.hwpr, alignment.hwpr_start_index,
        alignment.hwpr_end_index);
    const auto &telescope = alignment.telescope;
    const bool telescope_target_partition_is_exact =
        telescope.exact_target_count <= support.nominal_slot_count &&
        telescope.interpolated_target_count ==
            support.nominal_slot_count - telescope.exact_target_count;
    if (!telescope.initialized || telescope.interface_id != "lmt" ||
        telescope.coordinate_identity.empty() || telescope.unit != "s" ||
        telescope.epoch_event_precision_authority != "unavailable" ||
        telescope.support_rule.empty() || telescope.native_row_count <= 0 ||
        !std::isfinite(telescope.native_first_coordinate_sec) ||
        !std::isfinite(telescope.native_last_coordinate_sec) ||
        telescope.native_last_coordinate_sec <
            telescope.native_first_coordinate_sec ||
        !telescope_target_partition_is_exact ||
        !std::isfinite(telescope.minimum_used_bracket_span_sec) ||
        !std::isfinite(telescope.maximum_used_bracket_span_sec) ||
        telescope.minimum_used_bracket_span_sec < 0.0 ||
        telescope.maximum_used_bracket_span_sec <
            telescope.minimum_used_bracket_span_sec ||
        (telescope.interpolated_target_count != 0 &&
         !(telescope.minimum_used_bracket_span_sec > 0.0))) {
        throw std::logic_error(
            "initialized alignment has an incomplete telescope support contract");
    }
    if (static_cast<std::uint64_t>(alignment.common_time.size()) !=
        support.nominal_slot_count) {
        throw std::logic_error(
            "alignment common-time cardinality conflicts with compact support");
    }
    if (!std::isfinite(support.nominal_span_sec) ||
        support.nominal_span_sec != exposure.nominal_support_span_sec ||
        !std::isfinite(
            support.acquired_original_cadence_weighted_support_sec) ||
        support.acquired_original_cadence_weighted_support_sec !=
            static_cast<double>(support.acquired_original_count) *
                grid.cadence_sec) {
        throw std::logic_error(
            "alignment exposure summary conflicts with support counts");
    }

    std::set<std::string> interface_ids;
    std::set<Eigen::Index> roach_indices;
    std::map<std::string, std::size_t> compact_interface_indices;
    if (alignment.masks.size() != alignment.interfaces.size()) {
        throw std::logic_error(
            "alignment masks do not match compact interface summaries");
    }
    for (std::size_t index = 0; index < alignment.interfaces.size(); ++index) {
        const auto &interface = alignment.interfaces[index];
        const auto &mask = alignment.masks[index];
        const auto acquired_mask_count = count_binary_alignment_mask(
            mask, "alignment interface support mask");
        if (interface.interface_id.empty() ||
            !interface_ids.insert(interface.interface_id).second ||
            interface.roach_index < 0 ||
            !roach_indices.insert(interface.roach_index).second ||
            interface.native_row_count <= 0 ||
            interface.accepted_row_count < 0 ||
            interface.accepted_row_count > interface.native_row_count ||
            interface.first_global_slot < grid.first_global_slot ||
            interface.last_global_slot > grid.last_global_slot ||
            interface.last_global_slot < interface.first_global_slot ||
            interface.leading_unavailable_count < 0 ||
            interface.trailing_unavailable_count < 0 ||
            !std::isfinite(interface.minimum_residual_sec) ||
            !std::isfinite(interface.maximum_residual_sec) ||
            !std::isfinite(interface.maximum_absolute_residual_sec) ||
            interface.minimum_residual_sec >
                interface.maximum_residual_sec ||
            interface.maximum_absolute_residual_sec < 0.0 ||
            interface.maximum_absolute_residual_sec >=
                grid.exclusive_half_cell_sec ||
            mask.size() != alignment.common_time.size() ||
            acquired_mask_count !=
                static_cast<std::uint64_t>(interface.native_row_count)) {
            throw std::logic_error(
                "alignment interface summary is incomplete or inconsistent");
        }
        compact_interface_indices.emplace(interface.interface_id, index);
    }

    std::map<std::pair<std::string, std::string>, Eigen::Index>
        previous_exception_stops;
    std::map<std::string, std::uint64_t> detector_exception_counts;
    for (const auto &exception : alignment.exceptions) {
        if (exception.interface_id.empty() || exception.field_id.empty() ||
            exception.start < 0 ||
            exception.stop <= exception.start ||
            exception.stop > static_cast<Eigen::Index>(
                support.nominal_slot_count) ||
            exception.origin.empty() || exception.validity.empty() ||
            exception.action.empty() || exception.reason.empty()) {
            throw std::logic_error(
                "alignment exception run is incomplete or out of support");
        }
        const auto exception_key =
            std::make_pair(exception.interface_id, exception.field_id);
        const auto previous_it =
            previous_exception_stops.find(exception_key);
        if (previous_it != previous_exception_stops.end() &&
            exception.start <= previous_it->second) {
            throw std::logic_error(
                "alignment exception runs overlap or are not compactly ordered");
        }
        previous_exception_stops[exception_key] = exception.stop;
        const bool is_continuity_candidate =
            exception.action == "bounded_continuity_candidate";
        const bool source_slots_unavailable =
            exception.left_source_slot == -1 &&
            exception.right_source_slot == -1;
        if (is_continuity_candidate) {
            if (exception.field_id != "detector_acquisition" ||
                exception.start == 0 ||
                exception.stop >= static_cast<Eigen::Index>(
                    support.nominal_slot_count) ||
                exception.left_source_slot != exception.start - 1 ||
                exception.right_source_slot != exception.stop) {
                throw std::logic_error(
                    "bounded ALIGN exception lacks exact observation-wide source slots");
            }
            (void)alignment_exception_linear_source_weights(
                exception, exception.start);
            (void)alignment_exception_linear_source_weights(
                exception, exception.stop - 1);
        }
        else if (!source_slots_unavailable) {
            throw std::logic_error(
                "non-continuity ALIGN exception retains source-slot identities");
        }

        if (exception.field_id == "detector_acquisition") {
            const auto interface_it =
                compact_interface_indices.find(exception.interface_id);
            if (interface_it == compact_interface_indices.end()) {
                throw std::logic_error(
                    "detector exception names an unknown interface");
            }
            const auto &mask = alignment.masks[interface_it->second];
            if ((exception.start > 0 && mask(exception.start - 1) == 0) ||
                (exception.stop < mask.size() &&
                 mask(exception.stop) == 0)) {
                throw std::logic_error(
                    "detector exception is not a maximal observation-wide run");
            }
            for (Eigen::Index slot = exception.start;
                 slot < exception.stop; ++slot) {
                if (mask(slot) != 0) {
                    throw std::logic_error(
                        "detector exception conflicts with acquisition support");
                }
            }
            if (is_continuity_candidate &&
                (mask(exception.left_source_slot) != 1 ||
                 mask(exception.right_source_slot) != 1)) {
                throw std::logic_error(
                    "bounded ALIGN exception source slots are not acquired originals");
            }
            detector_exception_counts[exception.interface_id] =
                checked_alignment_count_add(
                    detector_exception_counts[exception.interface_id],
                    static_cast<std::uint64_t>(
                        exception.stop - exception.start),
                    "ALIGN detector exception count");
        }
    }
    for (std::size_t index = 0; index < alignment.interfaces.size(); ++index) {
        const auto &interface = alignment.interfaces[index];
        const auto &mask = alignment.masks[index];
        const auto acquired_mask_count = count_binary_alignment_mask(
            mask, "alignment interface support mask");
        const auto expected_missing =
            static_cast<std::uint64_t>(mask.size()) -
            acquired_mask_count;
        if (detector_exception_counts[interface.interface_id] !=
            expected_missing) {
            throw std::logic_error(
                "detector exception catalog does not cover acquisition gaps exactly");
        }
    }

    validate_alignment_processing_support(alignment, scan_plan);

    (void)alignment_term_availability_name(alignment.availability.mapping);
    (void)alignment_term_availability_name(
        alignment.availability.conditional_response);
    (void)alignment_term_availability_name(
        alignment.availability.input_covariance);
    (void)alignment_term_availability_name(
        alignment.availability.timing_covariance);
    (void)alignment_term_availability_name(
        alignment.availability.interpolation_model_covariance);
    (void)alignment_term_availability_name(
        alignment.availability.policy_selection_covariance);
}

inline YAML::Node compact_alignment_provenance_node(
    const TimestreamAlignmentState &alignment,
    const sci_align::ScanWindowPlan *scan_plan = nullptr,
    TimestreamOutputProvenanceStage evidence_stage =
        TimestreamOutputProvenanceStage::observation_setup_plan) {
    validate_compact_alignment_provenance(alignment, scan_plan);
    YAML::Node node;
    node["initialized"] = alignment.grid.initialized;
    if (!alignment.grid.initialized) {
        node["availability"]["alignment"] = "not_realized";
        return node;
    }

    const auto exposure = aligned_observation_exposure_summary(alignment);
    const auto &grid = alignment.grid;
    const auto &support = alignment.support;
    node["representation"] =
        "compact_generative_grid_plus_exception_runs_v1";
    node["dense_mapping_persisted"] = false;
    node["field_registry_version"] = alignment.field_registry_version;
    node["grid"]["phase_sec"] = grid.phase_sec;
    node["grid"]["cadence_sec"] = grid.cadence_sec;
    node["grid"]["exclusive_half_cell_sec"] =
        grid.exclusive_half_cell_sec;
    node["grid"]["first_global_slot"] = grid.first_global_slot;
    node["grid"]["last_global_slot"] = grid.last_global_slot;
    node["grid"]["assignment_operator"] = grid.assignment_operator;
    node["grid"]["phase_semantics"] = grid.phase_semantics;
    node["grid"]["physical_timestamp_semantics"] =
        grid.physical_timestamp_semantics;

    const auto &compatibility = alignment.governing_compatibility_axis;
    if (compatibility.initialized) {
        node["governing_compatibility_axis"]["availability"] =
            "available";
        node["governing_compatibility_axis"]["source_application_sha"] =
            compatibility.source_application_sha;
        node["governing_compatibility_axis"]
            ["assigned_time_constructor"] =
                compatibility.assigned_time_constructor;
        node["governing_compatibility_axis"]["raw_overlap_end_sec"] =
            compatibility.raw_overlap_end_sec;
        node["governing_compatibility_axis"]["global_start"] =
            compatibility.first_global_slot;
        node["governing_compatibility_axis"]["global_stop"] =
            compatibility.stop_global_slot;
        node["governing_compatibility_axis"]["union_local_start"] =
            compatibility.union_local_start;
        node["governing_compatibility_axis"]["union_local_stop"] =
            compatibility.union_local_stop;
        node["governing_compatibility_axis"]["sample_count"] =
            governing_compatibility_sample_count(compatibility);
        node["governing_compatibility_axis"]["interval_convention"] =
            "half_open_start_stop";
        node["governing_compatibility_axis"]["consumer_scope"] =
            compatibility.consumer_scope;
        node["governing_compatibility_axis"]["union_axis_role"] =
            "support_and_alignment_diagnostics_not_legacy_science_extent";
    }
    else {
        node["governing_compatibility_axis"]["availability"] =
            "not_applicable_simulation_native_full_axis";
        node["governing_compatibility_axis"]["sample_count"] =
            governing_consumer_sample_count(alignment);
        node["governing_compatibility_axis"]["consumer_scope"] =
            "native_simulation_full_axis";
        node["governing_compatibility_axis"]["assigned_time_constructor"] =
            "native_telescope_values_preserved_common_grid_generative_only";
    }
    node["governing_compatibility_axis"]["dense_axis_persisted"] = false;

    const auto &hwpr = alignment.hwpr;
    node["hwpr"]["policy"] = hwpr.policy;
    node["hwpr"]["observation_resolved"] =
        hwpr.observation_resolved;
    node["hwpr"]["producer_input_present"] =
        hwpr.producer_input_present;
    node["hwpr"]["aligned_angle_available"] =
        hwpr.aligned_angle_available;
    node["hwpr"]["intensity_eligible"] = hwpr.intensity_eligible;
    node["hwpr"]["polarization_eligible"] =
        hwpr.polarization_eligible;
    node["hwpr"]["availability_reason"] =
        hwpr.availability_reason;
    node["hwpr"]["physical_timestamp_semantics"] =
        hwpr.physical_timestamp_semantics;
    node["hwpr"]["demodulation_semantics"] =
        hwpr.demodulation_semantics;
    node["hwpr"]["dense_angle_mapping_persisted"] = false;

    const auto &telescope = alignment.telescope;
    node["telescope"]["interface_id"] = telescope.interface_id;
    node["telescope"]["native_coordinate_identity"] =
        telescope.coordinate_identity;
    node["telescope"]["unit"] = telescope.unit;
    node["telescope"]["epoch_event_precision_authority"] =
        telescope.epoch_event_precision_authority;
    node["telescope"]["support_rule"] = telescope.support_rule;
    node["telescope"]["general_numeric_runtime_bracket_limit_available"] =
        false;
    node["telescope"]["native_row_count"] = telescope.native_row_count;
    node["telescope"]["native_first_coordinate_sec"] =
        telescope.native_first_coordinate_sec;
    node["telescope"]["native_last_coordinate_sec"] =
        telescope.native_last_coordinate_sec;
    node["telescope"]["exact_target_count"] =
        telescope.exact_target_count;
    node["telescope"]["interpolated_target_count"] =
        telescope.interpolated_target_count;
    node["telescope"]["minimum_used_bracket_span_sec"] =
        telescope.minimum_used_bracket_span_sec;
    node["telescope"]["maximum_used_bracket_span_sec"] =
        telescope.maximum_used_bracket_span_sec;
    node["telescope"]["native_tel_utc_available"] =
        telescope.native_tel_utc_available;
    node["telescope"]["native_pps_time_available"] =
        telescope.native_pps_time_available;

    node["support"]["count_scope"] =
        "common_axis_once_and_interface_slot_memberships_separately";
    node["support"]["cadence_weighted_support_scope"] =
        "union_of_acquired_original_detector_interface_timing_slots_counted_once";
    node["support"]["physical_detector_integration_exposure_available"] =
        false;
    node["support"]["physical_detector_integration_exposure_reason"] =
        "unavailable_no_producer_start_end_or_integration_centroid_authority";
    node["support"]["nominal_common_axis_slot_count"] =
        support.nominal_slot_count;
    node["support"]["interface_slot_capacity_count"] =
        checked_alignment_interface_slot_capacity(
            support.nominal_slot_count, alignment.interfaces.size());
    node["support"]["acquired_original_interface_slot_count"] =
        support.acquired_original_count;
    node["support"]["timing_coordinate_valid_original_interface_slot_count"] =
        support.timing_coordinate_valid_original_count;
    node["support"]["detector_signal_validity_available"] = false;
    node["support"]["detector_signal_validity_reason"] =
        "unavailable_no_detector_signal_validity_mask_at_alignment_setup";
    node["support"]["synthesized_interface_slot_count"] =
        support.synthesized_count;
    node["support"]["synthesized_interface_slot_count_scope"] =
        alignment.processing_support.observation_resolved
            ? "unique_detector_interface_slots_with_at_least_one_admitted_bounded_continuity_action"
            : "pre_scan_detector_assignment_no_synthesis_action_resolved";
    node["support"]["unavailable_interface_slot_count"] =
        support.unavailable_count;
    node["support"]["unavailable_interface_slot_count_scope"] =
        alignment.processing_support.observation_resolved
            ? "unique_detector_interface_slots_without_an_admitted_bounded_continuity_action"
            : "pre_scan_detector_assignment";
    node["support"]["guarded_original_interface_slot_count"] =
        support.guarded_original_count;
    node["support"]["guarded_original_count_scope"] =
        alignment.processing_support.observation_resolved
            ? "unique_original_interface_slots_within_admitted_science_windows"
            : "pre_scan_disposition_not_observation_resolved";
    node["support"]["gap_policy_eligible_original_interface_slot_count"] =
        support.gap_policy_eligible_original_count;
    node["support"]["gap_policy_eligible_count_scope"] =
        alignment.processing_support.observation_resolved
            ? "unique_unguarded_original_interface_slots_within_admitted_science_windows"
            : "not_observation_resolved";
    node["support"]["final_science_eligibility_available"] = false;
    node["support"]["final_science_eligibility_reason"] =
        "downstream_consumer_owned_after_signal_and_pointing_validity";
    node["support"]["nominal_support_span_sec"] =
        exposure.nominal_support_span_sec;
    node["support"]["acquired_original_observation_union_slot_count"] =
        exposure.acquired_original_observation_union_slot_count;
    node["support"]
        ["acquired_original_observation_cadence_weighted_support_sec"] =
        exposure.acquired_original_observation_cadence_weighted_support_sec;

    node["interfaces"] = YAML::Node(YAML::NodeType::Sequence);
    for (const auto &interface : alignment.interfaces) {
        YAML::Node value;
        value["interface_id"] = interface.interface_id;
        value["roach_index"] = interface.roach_index;
        value["native_row_count"] = interface.native_row_count;
        value["accepted_row_count"] = interface.accepted_row_count;
        value["minimum_residual_sec"] = interface.minimum_residual_sec;
        value["maximum_residual_sec"] = interface.maximum_residual_sec;
        value["maximum_absolute_residual_sec"] =
            interface.maximum_absolute_residual_sec;
        value["first_global_slot"] = interface.first_global_slot;
        value["last_global_slot"] = interface.last_global_slot;
        value["leading_unavailable_count"] =
            interface.leading_unavailable_count;
        value["trailing_unavailable_count"] =
            interface.trailing_unavailable_count;
        node["interfaces"].push_back(value);
    }

    node["exception_runs"] = YAML::Node(YAML::NodeType::Sequence);
    node["exception_run_contract"]["source_slot_identity"] =
        "zero_based_observation_common_axis_slot";
    node["exception_run_contract"]["continuity_action_stage"] =
        "candidate_only_chunk_plan_controls_permission";
    node["exception_run_contract"]["continuity_weight_rule"]
        ["operator"] = "linear_slot_coordinate_weights_v1";
    node["exception_run_contract"]["continuity_weight_rule"]
        ["coordinate_basis"] = "observation_common_axis_slot_coordinates";
    node["exception_run_contract"]["continuity_weight_rule"]
        ["target_domain"] = "exception_start_inclusive_stop_exclusive";
    node["exception_run_contract"]["continuity_weight_rule"]
        ["left_source_weight"] =
        "(right_source_slot-target_slot)/(right_source_slot-left_source_slot)";
    node["exception_run_contract"]["continuity_weight_rule"]
        ["right_source_weight"] =
        "(target_slot-left_source_slot)/(right_source_slot-left_source_slot)";
    node["exception_run_contract"]["continuity_weight_rule"]
        ["normalization"] = "left_source_weight_plus_right_source_weight_equals_one";
    node["exception_run_contract"]["continuity_weight_rule"]
        ["dense_source_weights_persisted"] = false;
    for (const auto &exception : alignment.exceptions) {
        YAML::Node value;
        value["interface_id"] = exception.interface_id;
        value["field_id"] = exception.field_id;
        value["start"] = exception.start;
        value["stop"] = exception.stop;
        value["interval_convention"] = "half_open_start_stop";
        value["origin"] = exception.origin;
        value["validity"] = exception.validity;
        value["action"] = exception.action;
        value["reason"] = exception.reason;
        value["source_slot_identity"] =
            "zero_based_observation_common_axis_slot";
        value["source_slots_available"] =
            exception.action == "bounded_continuity_candidate";
        value["left_source_slot"] = exception.left_source_slot;
        value["right_source_slot"] = exception.right_source_slot;
        node["exception_runs"].push_back(value);
    }

    const auto &processing = alignment.processing_support;
    const bool execution_realized =
        processing.observation_resolved &&
        timestream_output_execution_completed(evidence_stage);
    auto processing_node = node["processing_support_plan"];
    processing_node["observation_resolved"] =
        processing.observation_resolved;
    processing_node["evidence_stage"] =
        !processing.observation_resolved
            ? "not_observation_resolved"
            : execution_realized
                  ? "observation_execution_completed_compact_result"
                  : "observation_resolved_planned_processing";
    processing_node["execution_realized"] = execution_realized;
    processing_node["realization_semantics"] =
        execution_realized
            ? "required_processing_and_outputs_completed_compact_plan_result"
            : "plan_only_no_execution_outcome_claim";
    processing_node["interval_convention"] = "half_open_start_stop";
    processing_node["signal_domain"] = processing.signal_domain;
    processing_node["count_scope"] =
        "planned_occurrences_across_admitted_scan_contexts";
    processing_node["gap_admission_contract"]["support_reference"] =
        "sci_align_scan_plan.records[stable_scan_id].compatibility_science";
    processing_node["gap_admission_contract"]["window_relationship"] =
        "compatibility_science_is_a_half_open_subset_of_compatibility_context";
    processing_node["gap_admission_contract"]
                   ["cumulative_missing_count_scope"] =
        "stable_record_science_window_only";
    processing_node["gap_admission_contract"]
                   ["longest_missing_run_count_scope"] =
        "stable_record_science_window_only";
    processing_node["gap_admission_contract"]["unusable_rule"] =
        "four_times_cumulative_or_longest_missing_strictly_exceeds_science_window_size";
    processing_node["gap_admission_contract"]["exact_quarter"] =
        "admitted";
    processing_node["planned_action_support_reference"] =
        "chunk_dispositions[].context_expanded_support";
    processing_node["continuity_source_contract"] =
        "each_planned_continuity_run_is_a_subrange_of_one_bounded_exception_run";
    processing_node["chunk_disposition_encoding"]["representation"] =
        "sparse_exceptions_v1";
    processing_node["chunk_disposition_encoding"]["key_order"] =
        "compatibility_ordinal_then_roach_index";
    processing_node["chunk_disposition_encoding"]["persisted_rows"] =
        "nondefault_scan_interface_dispositions_only";
    auto absent_default = processing_node["chunk_disposition_encoding"]
                                         ["absent_default"];
    absent_default["support"] =
        "all_acquired_original_zero_detector_gap";
    absent_default["cumulative_missing_count"] = 0;
    absent_default["longest_missing_run_count"] = 0;
    absent_default["gap_policy_eligible_original_within_science"] = true;
    absent_default["full_network_unusable"] = false;
    absent_default["continuity_surrogate_permitted"] =
        "signal_domain_is_xs";
    absent_default["planned_actions"] = "none";
    processing_node["planned_occurrence_counts"]
                   ["continuity_surrogate_missing"] =
        processing.synthesized_processing_occurrence_count;
    processing_node["planned_occurrence_counts"]
                   ["unavailable_missing"] =
        processing.unavailable_processing_occurrence_count;
    processing_node["planned_occurrence_counts"]
                   ["guarded_original"] =
        processing.guarded_original_processing_occurrence_count;
    processing_node["planned_occurrence_counts"]
                   ["full_network_unusable_original"] =
        processing.full_network_unusable_original_occurrence_count;
    processing_node["chunk_dispositions"] =
        YAML::Node(YAML::NodeType::Sequence);
    for (const auto &disposition : alignment.chunk_dispositions) {
        YAML::Node value;
        value["stable_scan_id"] = disposition.stable_scan_id;
        value["compatibility_ordinal"] =
            disposition.compatibility_ordinal;
        value["interface_id"] = disposition.interface_id;
        value["roach_index"] = disposition.roach_index;
        value["context"]["start"] = disposition.context_start;
        value["context"]["stop"] = disposition.context_stop;
        value["cumulative_missing_count"] =
            disposition.cumulative_missing_count;
        value["longest_missing_run_count"] =
            disposition.longest_missing_run_count;
        value["full_network_unusable"] =
            disposition.full_network_unusable;
        value["continuity_surrogate_permitted"] =
            disposition.continuity_surrogate_permitted;
        const auto append_runs = [](YAML::Node runs_node,
                                    const std::vector<AlignmentIndexRun> &runs) {
            for (const auto &run : runs) {
                YAML::Node run_node;
                run_node["start"] = run.start;
                run_node["stop"] = run.stop;
                runs_node.push_back(run_node);
            }
        };
        value["planned_actions"]["continuity_surrogate_missing"]
             ["action"] = "bounded_continuity_surrogate";
        value["planned_actions"]["continuity_surrogate_missing"]
             ["runs"] = YAML::Node(YAML::NodeType::Sequence);
        append_runs(
            value["planned_actions"]["continuity_surrogate_missing"]
                 ["runs"],
            disposition.synthesized_missing_runs);
        value["planned_actions"]["unavailable_missing"]["action"] =
            "remain_unavailable";
        value["planned_actions"]["unavailable_missing"]["runs"] =
            YAML::Node(YAML::NodeType::Sequence);
        append_runs(
            value["planned_actions"]["unavailable_missing"]["runs"],
            disposition.unavailable_missing_runs);
        value["planned_actions"]["guarded_original"]["action"] =
            "guard_original_processing_sample";
        value["planned_actions"]["guarded_original"]["runs"] =
            YAML::Node(YAML::NodeType::Sequence);
        append_runs(
            value["planned_actions"]["guarded_original"]["runs"],
            disposition.processing_guard_runs);
        processing_node["chunk_dispositions"].push_back(value);
    }

    node["availability"]["mapping"] = alignment_term_availability_name(
        alignment.availability.mapping);
    node["availability"]["conditional_response"] =
        alignment_term_availability_name(
            alignment.availability.conditional_response);
    node["availability"]["input_covariance"] =
        alignment_term_availability_name(
            alignment.availability.input_covariance);
    node["availability"]["timing_covariance"] =
        alignment_term_availability_name(
            alignment.availability.timing_covariance);
    node["availability"]["interpolation_model_covariance"] =
        alignment_term_availability_name(
            alignment.availability.interpolation_model_covariance);
    node["availability"]["policy_selection_covariance"] =
        alignment_term_availability_name(
            alignment.availability.policy_selection_covariance);
    return node;
}

template <class Engine>
YAML::Node timestream_output_provenance_node(
    const Engine &engine,
    TimestreamOutputProvenanceStage evidence_stage =
        TimestreamOutputProvenanceStage::observation_setup_plan) {
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

    root["realized"]["evidence_stage"] =
        timestream_output_evidence_stage_name(evidence_stage);
    root["realized"]["execution_completed"] =
        timestream_output_execution_completed(evidence_stage);
    root["realized"]["n_scans"] = engine.telescope.scan_indices.cols();
    if constexpr (has_sci_align_scan_plan<
                      std::decay_t<decltype(engine.telescope)>>::value) {
        root["realized"]["sci_align_scan_plan"] =
            sci_align_scan_plan_node(engine.telescope.scan_plan);
    }
    if constexpr (has_timestream_alignment_state_v<Engine>) {
        if constexpr (has_sci_align_scan_plan<
                          std::decay_t<decltype(engine.telescope)>>::value) {
            root["realized"]["sci_align_alignment"] =
                compact_alignment_provenance_node(
                    engine.alignment, &engine.telescope.scan_plan,
                    evidence_stage);
        }
        else {
            root["realized"]["sci_align_alignment"] =
                compact_alignment_provenance_node(
                    engine.alignment, nullptr, evidence_stage);
        }
    }
    root["realized"]["raw_time_chunk"]["n_output_scans"] =
        engine.tod_outputs.n_rtc_output_scans;
    root["realized"]["raw_time_chunk"]["scan_to_output"] =
        scan_to_output_node(engine.tod_outputs.rtc_scan_to_output_scan);
    root["realized"]["processed_time_chunk"]["n_output_scans"] =
        engine.tod_outputs.n_ptc_output_scans;
    root["realized"]["processed_time_chunk"]["scan_to_output"] =
        scan_to_output_node(engine.tod_outputs.ptc_scan_to_output_scan);
    if constexpr (has_sci_align_scan_plan<
                      std::decay_t<decltype(engine.telescope)>>::value) {
        root["realized"]["raw_time_chunk"]["selected_output_windows"] =
            selected_tod_output_windows_node(
                engine.telescope.scan_plan,
                engine.tod_outputs.rtc_scan_to_output_scan,
                engine.tod_outputs.n_rtc_output_scans,
                citlali::config::is_outer_tod_stream_output_mode(
                    config.output.raw_time_chunk.mode));
        root["realized"]["processed_time_chunk"]
            ["selected_output_windows"] =
                selected_tod_output_windows_node(
                    engine.telescope.scan_plan,
                    engine.tod_outputs.ptc_scan_to_output_scan,
                    engine.tod_outputs.n_ptc_output_scans, false);
    }
    root["realized"]["files"] = YAML::Node(YAML::NodeType::Map);
    for (const auto &[stream, filepath] : engine.output_paths.tod_filename) {
        root["realized"]["files"][stream] = filepath;
    }
    return root;
}

inline std::filesystem::path timestream_output_provenance_path(
    const std::filesystem::path &observation_dir) {
    return observation_dir / timestream_output_provenance_filename;
}

template <class Engine>
void write_timestream_output_provenance_file(
    const Engine &engine,
    TimestreamOutputProvenanceStage evidence_stage =
        TimestreamOutputProvenanceStage::observation_setup_plan) {
    const auto output_path = timestream_output_provenance_path(
        engine.output_paths.obsnum_dir_name);
    write_yaml_file_atomic(output_path,
                           timestream_output_provenance_node(
                               engine, evidence_stage));
}

template <class Engine>
std::optional<std::filesystem::path>
publish_completed_timestream_output_provenance(
    const Engine &engine) {
    if constexpr (has_timestream_output_provenance_state_v<Engine>) {
        write_timestream_output_provenance_file(
            engine,
            TimestreamOutputProvenanceStage::observation_execution_completed);
        return timestream_output_provenance_path(
            engine.output_paths.obsnum_dir_name);
    }
    return std::nullopt;
}

}  // namespace citlali::pipeline
