#pragma once

#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/raw_timestream_provenance.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_output_context.h>
#include <citlali/core/pipeline/timestream_output_provenance.h>
#include <citlali/core/utils/sha256.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

inline std::size_t raw_realized_count(Eigen::Index count,
                                      const char *field) {
    if (count < 0) {
        throw std::logic_error(std::string(field) + " cannot be negative");
    }
    return static_cast<std::size_t>(count);
}

inline std::size_t raw_required_timestream_write_count(
    const TimestreamOutputExpectations &expectations) {
    const std::array<Eigen::Index, 4> counts{
        expectations.rtc, expectations.ptc,
        expectations.rtcdiag, expectations.ptcdiag};
    std::size_t total = 0;
    for (const auto count : counts) {
        const auto value =
            raw_realized_count(count, "required timestream write count");
        if (value > std::numeric_limits<std::size_t>::max() - total) {
            throw std::overflow_error(
                "required timestream write count overflow");
        }
        total += value;
    }
    return total;
}

inline void complete_raw_timestream_observation(
    RawTimestreamExecutionPlan &plan, std::size_t completed_scan_count,
    std::size_t required_timestream_write_count) {
    if (!plan.initialized) {
        throw std::logic_error(
            "cannot complete uninitialized raw timestream plan");
    }
    if (!plan.observation.has_value()) {
        throw std::logic_error(
            "cannot complete raw timestream plan before observation begins");
    }
    plan.realized.completed_scan_count = completed_scan_count;
    plan.realized.required_timestream_write_count =
        required_timestream_write_count;
    plan.realized.execution_completed = true;
}

template <class Snapshot>
void synchronize_raw_rtc_realized_state(
    RawTimestreamExecutionPlan &plan, const Snapshot &snapshot,
    std::size_t completed_scan_count,
    std::size_t expected_rtc_product_count) {
    if (snapshot.observation_scope.empty() && completed_scan_count != 0) {
        throw std::logic_error("RTC observation identity is unavailable");
    }
    std::set<std::string> stage_identities;
    std::set<Eigen::Index> inner_standard_scans;
    std::map<std::string, std::string> stage_parents;
    std::size_t influenced_sample_count = 0;
    std::vector<RawRtcStageRealization> stages;
    stages.reserve(snapshot.stages.size());
    for (const auto &source : snapshot.stages) {
        if (source.stage_identity.empty() || source.parent_identity.empty() ||
            source.process_identity.empty() ||
            source.assigned_grid_identity.empty() ||
            source.physical_event_semantics != "unavailable" ||
            source.assigned_time_semantics !=
                "compatibility_state_only" ||
            source.source_mask_timing_accuracy != "unavailable" ||
            source.representative_assigned_time_hex.empty() ||
            source.assigned_time_values_digest.empty() ||
            !stage_identities.insert(source.stage_identity).second) {
            throw std::logic_error(
                "RTC stage identity bundle is incomplete or duplicate");
        }
        if (source.stage_view == "inner" &&
            source.process_label == "standard") {
            inner_standard_scans.insert(source.scan_id);
            influenced_sample_count += source.influenced_sample_count;
        }
        stage_parents[source.stage_identity] = source.parent_identity;
        RawRtcStageRealization stage;
        stage.stage_identity = source.stage_identity;
        stage.parent_identity = source.parent_identity;
        stage.process_identity = source.process_identity;
        stage.observation_scope = source.observation_scope;
        stage.process_label = source.process_label;
        stage.stage_view = source.stage_view;
        stage.assigned_grid_identity = source.assigned_grid_identity;
        stage.physical_event_semantics = source.physical_event_semantics;
        stage.assigned_time_semantics = source.assigned_time_semantics;
        stage.lattice_label = source.lattice_label;
        stage.phase_label = source.phase_label;
        stage.representative_assigned_time_rule =
            source.representative_assigned_time_rule;
        stage.representative_assigned_time_hex =
            source.representative_assigned_time_hex;
        stage.assigned_time_values_digest =
            source.assigned_time_values_digest;
        stage.edge_rule = source.edge_rule;
        stage.influence_support_policy =
            source.influence_support_policy;
        stage.operator_ordering = source.operator_ordering;
        stage.fir_normalization = source.fir_normalization;
        stage.downsample_normalization =
            source.downsample_normalization;
        stage.detector_ordering = source.detector_ordering;
        stage.source_mask_identity = source.source_mask_identity;
        stage.source_mask_frame = source.source_mask_frame;
        stage.source_mask_admission = source.source_mask_admission;
        stage.source_mask_reason = source.source_mask_reason;
        stage.source_mask_timing_accuracy =
            source.source_mask_timing_accuracy;
        stage.fir_state_reset = source.fir_state_reset;
        stage.notch_state_reset = source.notch_state_reset;
        stage.iir_highpass_state_reset = source.iir_highpass_state_reset;
        stage.notch_section_layout = source.notch_section_layout;
        stage.scan_id = source.scan_id;
        stage.absolute_assigned_start = source.absolute_assigned_start;
        stage.input_sample_count = source.input_sample_count;
        stage.output_sample_count = source.output_sample_count;
        stage.detector_count = source.detector_count;
        stage.inner_start = source.inner_start;
        stage.inner_sample_count = source.inner_sample_count;
        stage.filter_guard_samples = source.filter_guard_samples;
        stage.filter_context_samples = source.filter_context_samples;
        stage.native_sample_rate_hz = source.native_sample_rate_hz;
        stage.effective_sample_rate_hz = source.effective_sample_rate_hz;
        stage.downsample_factor = source.downsample_factor;
        stage.simulated = source.simulated;
        stage.source_mask_admitted = source.source_mask_admitted;
        stage.complete_response_available =
            source.complete_response_available;
        stage.signal_stage_bits = source.signal_stage_bits;
        stage.response_stage_bits = source.response_stage_bits;
        stage.response_unavailable_cause_bits =
            source.response_unavailable_cause_bits;
        stage.influenced_sample_count = source.influenced_sample_count;
        for (const auto &source_interval : source.influence_intervals) {
            stage.influence_intervals.push_back(
                {source_interval.detector,
                 source_interval.first_assigned_sample,
                 source_interval.last_assigned_sample,
                 static_cast<std::uint32_t>(source_interval.causes)});
        }
        stage.fir_coefficients_hex = source.fir_coefficients_hex;
        stage.notch_a_coefficients_hex =
            source.notch_a_coefficients_hex;
        stage.notch_b_coefficients_hex =
            source.notch_b_coefficients_hex;
        stage.iir_highpass_alpha_hex = source.iir_highpass_alpha_hex;
        stage.iir_highpass_order = source.iir_highpass_order;
        stage.notch_zero_phase = source.notch_zero_phase;
        stage.iir_highpass_zero_phase = source.iir_highpass_zero_phase;
        stages.push_back(std::move(stage));
    }
    if (inner_standard_scans.size() != completed_scan_count) {
        throw std::logic_error(
            "RTC stage bundle does not cover every completed standard scan");
    }
    for (std::size_t scan = 0; scan < completed_scan_count; ++scan) {
        if (inner_standard_scans.count(
                static_cast<Eigen::Index>(scan)) != 1) {
            throw std::logic_error(
                "RTC stage bundle has a stale standard scan identity");
        }
    }

    if (snapshot.products.size() != expected_rtc_product_count) {
        throw std::logic_error(
            "RTC product completion count does not match expectations");
    }
    std::set<std::string> completion_identities;
    std::vector<RawRtcProductRealization> products;
    products.reserve(snapshot.products.size());
    for (const auto &source : snapshot.products) {
        if (!source.complete || source.product_identity.empty() ||
            source.completion_identity.empty() ||
            source.physical_event_semantics != "unavailable" ||
            stage_identities.count(source.stage_identity) != 1 ||
            stage_identities.count(source.parent_identity) != 1 ||
            stage_parents.at(source.stage_identity) !=
                source.parent_identity ||
            !completion_identities.insert(
                 source.completion_identity).second) {
            throw std::logic_error(
                "RTC product completion bundle is incomplete, stale, or duplicate");
        }
        RawRtcProductRealization product;
        product.product_identity = source.product_identity;
        product.stage_identity = source.stage_identity;
        product.parent_identity = source.parent_identity;
        product.process_identity = source.process_identity;
        product.completion_identity = source.completion_identity;
        product.assigned_grid_identity = source.assigned_grid_identity;
        product.physical_event_semantics =
            source.physical_event_semantics;
        product.product_kind = source.product_kind;
        product.filepath = source.filepath;
        product.scan_id = source.scan_id;
        product.output_row = source.output_row;
        product.mini_output = source.mini_output;
        product.outer_output = source.outer_output;
        product.simulated = source.simulated;
        product.complete = source.complete;
        products.push_back(std::move(product));
    }

    std::ostringstream bundle;
    bundle << "SCI-RTC-001-bundle-v1|observation="
           << snapshot.observation_scope;
    for (const auto &stage : stages) {
        bundle << "|stage=" << stage.stage_identity;
    }
    for (const auto &product : products) {
        bundle << "|completion=" << product.completion_identity;
    }
    plan.realized.rtc_observation_scope = snapshot.observation_scope;
    plan.realized.rtc_bundle_identity =
        "sha256:" + citlali::utils::sha256(bundle.str());
    plan.realized.rtc_stages = std::move(stages);
    plan.realized.rtc_products = std::move(products);
    plan.realized.flagged_sample_count = influenced_sample_count;
    plan.realized.rtc_bundle_complete = true;
}

template <bool IsBeammap, class Engine>
TimestreamOutputExpectations raw_observation_output_expectations(
    const Engine &engine) {
    if (!timestream_processing_enabled(engine)) {
        return {};
    }
    if constexpr (IsBeammap) {
        const auto flags =
            beammap_timestream_output_flags(engine, true);
        return beammap_timestream_output_expectations(engine, flags);
    }
    else {
        const auto flags = standard_timestream_output_flags(engine);
        return standard_timestream_output_expectations(engine, flags);
    }
}

template <bool IsBeammap, class Engine>
std::optional<std::filesystem::path>
publish_completed_raw_timestream_provenance(Engine &engine) {
    if constexpr (has_raw_timestream_plan_v<Engine>) {
        auto &plan = raw_timestream_plan(engine);
        const auto expectations =
            raw_observation_output_expectations<IsBeammap>(engine);
        const Eigen::Index scan_count =
            timestream_processing_enabled(engine)
                ? engine.telescope.scan_indices.cols()
                : 0;
        if constexpr (requires {
                          engine.rtcproc
                              .snapshot_phase_independent_state();
                      }) {
            auto &rtc_processor = engine.rtcproc;
            const auto rtc_snapshot =
                rtc_processor.snapshot_phase_independent_state();
            const std::size_t expected_rtc_products =
                raw_realized_count(expectations.rtc,
                                   "required RTC write count") +
                raw_realized_count(
                    expectations.rtcdiag,
                    "required RTC diagnostic write count");
            synchronize_raw_rtc_realized_state(
                plan, rtc_snapshot,
                raw_realized_count(scan_count, "completed scan count"),
                expected_rtc_products);
        }
        complete_raw_timestream_observation(
            plan, raw_realized_count(scan_count, "completed scan count"),
            raw_required_timestream_write_count(expectations));
        const auto path = raw_timestream_provenance_path(
            engine.output_paths.obsnum_dir_name);
        write_raw_timestream_provenance_file(
            engine.output_paths.obsnum_dir_name, plan);
        write_timestream_output_provenance_file(engine);
        return path;
    }
    return std::nullopt;
}

}  // namespace citlali::pipeline
