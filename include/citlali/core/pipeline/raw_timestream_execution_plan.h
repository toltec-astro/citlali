#pragma once

#include <citlali/core/config/interface_sync_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/raw_timestream_resolution.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::pipeline {

struct RawRtcContractState {
    std::string assigned_grid_authority =
        "ALIGN-ASSIGNED-TIME-COMPAT-001";
    std::string assigned_time_semantics =
        "compatibility_state_only";
    std::string physical_event_semantics = "unavailable";
    std::string lattice_label = "existing_assigned_sample_lattice";
    std::string phase_label = "assigned_index_phase_zero";
    std::string representative_assigned_time_rule =
        "phase_zero_first_cell_compatibility_value";
    std::string edge_rule =
        "loaded_outer_context_then_exact_inner_crop";
    std::string influence_support_policy = "typed_compact_intervals";
    std::string operator_ordering =
        "calibration>extinction>kernel>despike>pre_notch>fir>"
        "configured_notch>iir_highpass>edge_guard>detector_notch>"
        "inner_crop>phase_zero_downsample>post_notch>event_masks>"
        "altaz_projection";
    std::string fir_normalization =
        "exact_realized_coefficients_no_additional_normalization";
    std::string downsample_normalization =
        "arithmetic_mean_one_over_factor";
    std::string timing_sensitive_mask_accuracy = "unavailable";
    std::string detector_ordering = "apt_uid_column_order";
    bool scientific_eligibility_required = true;
    bool complete_response_or_unavailable_required = true;
    bool source_mask_fail_closed_required = true;
};

struct RawRtcInfluenceInterval {
    Eigen::Index detector = -1;
    Eigen::Index first_assigned_sample = 0;
    Eigen::Index last_assigned_sample = -1;
    std::uint32_t cause_bits = 0;
};

struct RawRtcStageRealization {
    std::string stage_identity;
    std::string parent_identity;
    std::string process_identity;
    std::string observation_scope;
    std::string process_label;
    std::string stage_view;
    std::string assigned_grid_identity;
    std::string physical_event_semantics = "unavailable";
    std::string assigned_time_semantics = "compatibility_state_only";
    std::string lattice_label;
    std::string phase_label;
    std::string representative_assigned_time_rule;
    std::string representative_assigned_time_hex;
    std::string assigned_time_values_digest;
    std::string edge_rule;
    std::string influence_support_policy;
    std::string operator_ordering;
    std::string fir_normalization;
    std::string downsample_normalization;
    std::string detector_ordering;
    std::string source_mask_identity;
    std::string source_mask_frame;
    std::string source_mask_admission;
    std::string source_mask_reason;
    std::string source_mask_timing_accuracy = "unavailable";
    std::string fir_state_reset;
    std::string notch_state_reset;
    std::string iir_highpass_state_reset;
    std::string notch_section_layout;
    Eigen::Index scan_id = -1;
    Eigen::Index absolute_assigned_start = 0;
    Eigen::Index input_sample_count = 0;
    Eigen::Index output_sample_count = 0;
    Eigen::Index detector_count = 0;
    Eigen::Index inner_start = 0;
    Eigen::Index inner_sample_count = 0;
    Eigen::Index filter_guard_samples = 0;
    Eigen::Index filter_context_samples = 0;
    double native_sample_rate_hz = 0.0;
    double effective_sample_rate_hz = 0.0;
    int downsample_factor = 1;
    bool simulated = false;
    bool source_mask_admitted = false;
    bool complete_response_available = false;
    std::uint32_t signal_stage_bits = 0;
    std::uint32_t response_stage_bits = 0;
    std::uint32_t response_unavailable_cause_bits = 0;
    std::size_t influenced_sample_count = 0;
    std::vector<RawRtcInfluenceInterval> influence_intervals;
    std::vector<std::string> fir_coefficients_hex;
    std::vector<std::string> notch_a_coefficients_hex;
    std::vector<std::string> notch_b_coefficients_hex;
    std::string iir_highpass_alpha_hex;
    int iir_highpass_order = 0;
    bool notch_zero_phase = false;
    bool iir_highpass_zero_phase = false;
};

struct RawRtcProductRealization {
    std::string product_identity;
    std::string stage_identity;
    std::string parent_identity;
    std::string process_identity;
    std::string completion_identity;
    std::string assigned_grid_identity;
    std::string physical_event_semantics = "unavailable";
    std::string product_kind;
    std::string filepath;
    Eigen::Index scan_id = -1;
    Eigen::Index output_row = -1;
    bool mini_output = false;
    bool outer_output = false;
    bool simulated = false;
    bool complete = false;
};

struct RawTimestreamObservationState {
    RawRtcContractState rtc_contract;
    std::optional<double> native_sample_rate_hz;
    std::optional<double> effective_sample_rate_hz;
    std::optional<int> downsample_factor;
    std::optional<int> filter_edge_guard_samples;
    std::optional<int> filter_outer_context_samples;
    bool filter_edge_guard_parity_deferred = false;
    std::optional<bool> source_protection_active;
    std::optional<bool> extinction_active;
    std::optional<std::string> extinction_model;
};

struct RawTimestreamRealizedState {
    bool execution_completed = false;
    std::optional<std::size_t> completed_scan_count;
    std::optional<std::size_t> flagged_sample_count;
    std::optional<std::size_t> dynamic_notch_count;
    std::optional<std::size_t> required_timestream_write_count;
    std::string rtc_observation_scope;
    std::string rtc_bundle_identity;
    bool rtc_bundle_complete = false;
    std::vector<RawRtcStageRealization> rtc_stages;
    std::vector<RawRtcProductRealization> rtc_products;
};

struct RawTimestreamExecutionPlan {
    bool initialized = false;
    citlali::config::RawTimeChunkConfig requested;
    citlali::config::RawTimeChunkConfig effective;
    citlali::config::InterfaceSyncOffsetConfig interface_sync_requested;
    citlali::config::InterfaceSyncOffsetConfig interface_sync_effective;
    RawTimestreamEffectiveResolutions effective_resolutions;
    RawRtcContractState requested_rtc_contract;
    RawRtcContractState effective_rtc_contract;
    std::optional<RawTimestreamObservationState> observation;
    RawTimestreamRealizedState realized;

    void reset_from_request(
        const citlali::config::RawTimeChunkConfig &request,
        const citlali::config::InterfaceSyncOffsetConfig
            &interface_sync_request = {}) {
        initialized = true;
        requested = request;
        effective = request;
        interface_sync_requested = interface_sync_request;
        interface_sync_effective = interface_sync_request;
        effective_resolutions =
            resolve_raw_timestream_effective_request(request);
        requested_rtc_contract = RawRtcContractState{};
        effective_rtc_contract = requested_rtc_contract;
        observation.reset();
        realized = {};
    }

    RawTimestreamObservationState &begin_observation() {
        if (!initialized) {
            throw std::logic_error(
                "raw timestream plan is not initialized");
        }
        observation.emplace();
        observation->rtc_contract = effective_rtc_contract;
        realized = {};
        return *observation;
    }
};

}  // namespace citlali::pipeline
