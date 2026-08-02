#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace citlali::pipeline {

enum class AlignmentTermAvailability {
    available,
    available_conditional,
    unavailable_input,
    not_applicable,
    not_persisted_standard,
};

struct AlignmentGridState {
    bool initialized = false;
    double phase_sec = 0.0;
    double cadence_sec = 0.0;
    double exclusive_half_cell_sec = 0.0;
    std::int64_t first_global_slot = 0;
    std::int64_t last_global_slot = -1;
    std::string assignment_operator = "floor_q_plus_half_v1";
    std::string phase_semantics =
        "latest_first_valid_detector_timestamp_compatibility_lattice";
    std::string physical_timestamp_semantics = "unavailable";
};

inline constexpr const char *
    governing_compatibility_assigned_time_constructor =
        "eigen_vectorxd_linspaced_9aae_gap_v1";

// Compact identity for the exact governing-application sample axis retained
// inside the larger detector-support union.  The interval is authoritative in
// global detector-slot coordinates and repeated in union-local coordinates so
// consumers never have to infer it from floating timestamps.
struct AlignmentGoverningCompatibilityAxis {
    bool initialized = false;
    double raw_overlap_end_sec = 0.0;
    std::int64_t first_global_slot = 0;
    std::int64_t stop_global_slot = 0;
    Eigen::Index union_local_start = 0;
    Eigen::Index union_local_stop = 0;
    std::string assigned_time_constructor =
        governing_compatibility_assigned_time_constructor;
    std::string source_application_sha =
        "9aae0e669384c5c0c0dda93debc194d6b8dac787";
    std::string consumer_scope =
        "legacy_outputs_and_legacy_science_consumers";
};

struct AlignmentInterfaceSummary {
    std::string interface_id;
    Eigen::Index roach_index = -1;
    Eigen::Index native_row_count = 0;
    Eigen::Index accepted_row_count = 0;
    double minimum_residual_sec = 0.0;
    double maximum_residual_sec = 0.0;
    double maximum_absolute_residual_sec = 0.0;
    std::int64_t first_global_slot = 0;
    std::int64_t last_global_slot = -1;
    std::int64_t leading_unavailable_count = 0;
    std::int64_t trailing_unavailable_count = 0;
};

struct AlignmentTelescopeSummary {
    bool initialized = false;
    std::string interface_id = "lmt";
    std::string coordinate_identity =
        "Data.TelescopeBackend.TelTime";
    std::string unit = "s";
    std::string epoch_event_precision_authority = "unavailable";
    std::string support_rule =
        "adjacent_finite_native_bracket_no_extrapolation_producer_gap_authority_unavailable_no_general_numeric_runtime_ceiling";
    Eigen::Index native_row_count = 0;
    double native_first_coordinate_sec = 0.0;
    double native_last_coordinate_sec = 0.0;
    std::uint64_t exact_target_count = 0;
    std::uint64_t interpolated_target_count = 0;
    double minimum_used_bracket_span_sec = 0.0;
    double maximum_used_bracket_span_sec = 0.0;
    bool native_tel_utc_available = false;
    bool native_pps_time_available = false;
};

// Observation-scoped status for the optional HWPR interface in the bounded
// nonpolarimetric SCI-ALIGN-001 profile.  Input presence is intentionally
// distinct from aligned-angle availability: a selected HWPR input is not
// loaded, interpreted, or demodulated by this profile.
struct AlignmentHwprSummary {
    bool observation_resolved = false;
    bool producer_input_present = false;
    bool aligned_angle_available = false;
    bool intensity_eligible = false;
    bool polarization_eligible = false;
    std::string policy;
    std::string availability_reason;
    std::string physical_timestamp_semantics;
    std::string demodulation_semantics;
};

inline AlignmentHwprSummary bounded_nonpolarimetric_hwpr_summary(
    bool producer_input_present) {
    AlignmentHwprSummary result;
    result.observation_resolved = true;
    result.producer_input_present = producer_input_present;
    result.aligned_angle_available = false;
    result.intensity_eligible = true;
    result.polarization_eligible = false;
    result.policy = "bounded_nonpolarimetric_optional_hwpr_v1";
    result.availability_reason = producer_input_present
        ? "producer_input_present_not_loaded_or_aligned"
        : "producer_input_absent_optional_nonfatal";
    result.physical_timestamp_semantics =
        "unavailable_no_producer_integration_event_authority";
    result.demodulation_semantics =
        "unavailable_not_authorized_by_bounded_profile";
    return result;
}

template <class RawObs, class = void>
struct has_hwpr_input_accessor : std::false_type {};

template <class RawObs>
struct has_hwpr_input_accessor<
    RawObs,
    std::void_t<decltype(std::declval<const RawObs &>().hwpdata()
                             .has_value())>> : std::true_type {};

template <class RawObs>
bool observation_hwpr_input_present(const RawObs &rawobs) {
    if constexpr (has_hwpr_input_accessor<RawObs>::value) {
        return rawobs.hwpdata().has_value();
    }
    else {
        return false;
    }
}

struct AlignmentExceptionRun {
    std::string interface_id;
    std::string field_id;
    Eigen::Index start = 0;
    Eigen::Index stop = 0;
    std::string origin;
    std::string validity;
    std::string action;
    std::string reason;
    Eigen::Index left_source_slot = -1;
    Eigen::Index right_source_slot = -1;
};

struct AlignmentIndexRun {
    Eigen::Index start = 0;
    Eigen::Index stop = 0;
};

struct AlignmentChunkDisposition {
    Eigen::Index stable_scan_id = -1;
    Eigen::Index compatibility_ordinal = -1;
    std::string interface_id;
    Eigen::Index roach_index = -1;
    Eigen::Index context_start = 0;
    Eigen::Index context_stop = 0;
    // These two counts and full_network_unusable are admission facts scoped
    // to the stable scan record's science window.  Planned action runs remain
    // scoped to the separately recorded, possibly expanded context window.
    Eigen::Index cumulative_missing_count = 0;
    Eigen::Index longest_missing_run_count = 0;
    bool full_network_unusable = false;
    bool continuity_surrogate_permitted = false;
    std::vector<AlignmentIndexRun> synthesized_missing_runs;
    std::vector<AlignmentIndexRun> unavailable_missing_runs;
    std::vector<AlignmentIndexRun> processing_guard_runs;
};

struct AlignmentProcessingSupportSummary {
    bool observation_resolved = false;
    std::string signal_domain;
    std::uint64_t synthesized_processing_occurrence_count = 0;
    std::uint64_t unavailable_processing_occurrence_count = 0;
    std::uint64_t guarded_original_processing_occurrence_count = 0;
    std::uint64_t full_network_unusable_original_occurrence_count = 0;
};

struct AlignmentSupportSummary {
    std::uint64_t nominal_slot_count = 0;
    std::uint64_t acquired_original_count = 0;
    // This is validity of the admitted timing coordinate/slot identity only.
    // Detector-signal validity remains unavailable at ALIGN setup.
    std::uint64_t timing_coordinate_valid_original_count = 0;
    std::uint64_t synthesized_count = 0;
    std::uint64_t unavailable_count = 0;
    std::uint64_t guarded_original_count = 0;
    // This is eligibility under ALIGN's gap-action policy only. Final science
    // eligibility remains owned by consumers after signal/pointing validity.
    std::uint64_t gap_policy_eligible_original_count = 0;
    double nominal_span_sec = 0.0;
    double acquired_original_cadence_weighted_support_sec = 0.0;
};

struct AlignmentAvailabilityManifest {
    AlignmentTermAvailability mapping =
        AlignmentTermAvailability::available_conditional;
    AlignmentTermAvailability conditional_response =
        AlignmentTermAvailability::available_conditional;
    AlignmentTermAvailability input_covariance =
        AlignmentTermAvailability::unavailable_input;
    AlignmentTermAvailability timing_covariance =
        AlignmentTermAvailability::unavailable_input;
    AlignmentTermAvailability interpolation_model_covariance =
        AlignmentTermAvailability::unavailable_input;
    AlignmentTermAvailability policy_selection_covariance =
        AlignmentTermAvailability::not_persisted_standard;
};

struct TimestreamAlignmentState {
    Eigen::VectorXd common_time;
    std::vector<Eigen::VectorXi> masks;
    std::map<Eigen::Index, Eigen::VectorXi> network_masks;
    std::vector<Eigen::VectorXd> network_times;
    std::map<std::string, int> gaps;
    std::vector<Eigen::Index> start_indices;
    std::vector<Eigen::Index> end_indices;
    Eigen::Index hwpr_start_index = -1;
    Eigen::Index hwpr_end_index = -1;
    AlignmentGridState grid;
    AlignmentGoverningCompatibilityAxis governing_compatibility_axis;
    std::vector<AlignmentInterfaceSummary> interfaces;
    AlignmentTelescopeSummary telescope;
    AlignmentHwprSummary hwpr;
    std::vector<AlignmentExceptionRun> exceptions;
    std::vector<AlignmentChunkDisposition> chunk_dispositions;
    AlignmentSupportSummary support;
    AlignmentProcessingSupportSummary processing_support;
    AlignmentAvailabilityManifest availability;
    std::string field_registry_version;
};

inline Eigen::Index checked_union_sample_count(
    const AlignmentGridState &grid) {
    if (grid.last_global_slot < grid.first_global_slot) {
        throw std::logic_error("detector union slot interval is empty");
    }
    const auto distance =
        static_cast<std::uint64_t>(grid.last_global_slot) -
        static_cast<std::uint64_t>(grid.first_global_slot);
    if (distance == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "detector union sample count exceeds uint64 range");
    }
    const auto count = distance + 1;
    if (count > static_cast<std::uint64_t>(
                    std::numeric_limits<Eigen::Index>::max())) {
        throw std::overflow_error(
            "detector union sample count exceeds Eigen index range");
    }
    return static_cast<Eigen::Index>(count);
}

inline std::int64_t checked_add_nonnegative_slot_offset(
    std::int64_t base, Eigen::Index offset) {
    if (offset < 0 ||
        static_cast<std::uint64_t>(offset) >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()) ||
        base > std::numeric_limits<std::int64_t>::max() -
                   static_cast<std::int64_t>(offset)) {
        throw std::overflow_error(
            "detector slot plus local offset exceeds int64 range");
    }
    return base + static_cast<std::int64_t>(offset);
}

inline std::uint64_t checked_alignment_count_add(
    std::uint64_t left, std::uint64_t right, const char *label) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(std::string{label} +
                                  " exceeds uint64 range");
    }
    return left + right;
}

inline std::uint64_t checked_alignment_interface_slot_capacity(
    std::uint64_t nominal_slot_count, std::size_t interface_count) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (interface_count > static_cast<std::size_t>(
                                  std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                "ALIGN interface count exceeds uint64 range");
        }
    }
    const auto interface_count64 =
        static_cast<std::uint64_t>(interface_count);
    if (interface_count64 != 0 &&
        nominal_slot_count >
            std::numeric_limits<std::uint64_t>::max() /
                interface_count64) {
        throw std::overflow_error(
            "ALIGN interface-slot capacity exceeds uint64 range");
    }
    return nominal_slot_count * interface_count64;
}

inline std::size_t checked_alignment_size_product(
    std::size_t left, std::size_t right, const char *label) {
    if (right != 0 &&
        left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(std::string{label} +
                                  " exceeds size_t range");
    }
    return left * right;
}

inline std::uint64_t count_binary_alignment_mask(
    const Eigen::VectorXi &mask, const char *label) {
    std::uint64_t count = 0;
    for (Eigen::Index index = 0; index < mask.size(); ++index) {
        if (mask(index) != 0 && mask(index) != 1) {
            throw std::logic_error(std::string{label} +
                                   " is not binary");
        }
        count += mask(index) != 0 ? 1U : 0U;
    }
    return count;
}

inline Eigen::Index governing_compatibility_sample_count(
    const AlignmentGoverningCompatibilityAxis &axis) {
    if (!axis.initialized || axis.union_local_start < 0 ||
        axis.union_local_stop <= axis.union_local_start) {
        throw std::logic_error(
            "governing compatibility axis is unavailable or empty");
    }
    return axis.union_local_stop - axis.union_local_start;
}

inline bool uses_native_simulation_full_axis(
    const TimestreamAlignmentState &state) {
    return state.grid.initialized &&
           !state.governing_compatibility_axis.initialized &&
           state.grid.phase_semantics ==
               "simulator_native_telescope_coordinate_exact";
}

inline void validate_native_simulation_full_axis_structure(
    const TimestreamAlignmentState &state) {
    Eigen::Index union_count = 0;
    try {
        union_count = checked_union_sample_count(state.grid);
    }
    catch (const std::exception &) {
        throw std::logic_error(
            "simulated native full axis conflicts with compact grid");
    }
    if (!uses_native_simulation_full_axis(state) ||
        !std::isfinite(state.grid.phase_sec) ||
        !std::isfinite(state.grid.cadence_sec) ||
        !(state.grid.cadence_sec > 0.0) ||
        state.grid.first_global_slot != 0 || union_count <= 0 ||
        state.common_time.size() != union_count ||
        !std::isfinite(state.common_time(0)) ||
        state.common_time(0) != state.grid.phase_sec) {
        throw std::logic_error(
            "simulated native full axis conflicts with compact grid");
    }
    for (Eigen::Index local = 1; local < state.common_time.size(); ++local) {
        const double previous = state.common_time(local - 1);
        const double current = state.common_time(local);
        const double realized = current - previous;
        const double representation_bound =
            std::abs(std::nextafter(
                         previous,
                         std::numeric_limits<double>::infinity()) -
                     previous) +
            std::abs(std::nextafter(
                         current,
                         std::numeric_limits<double>::infinity()) -
                     current) +
            std::abs(std::nextafter(
                         realized,
                         std::numeric_limits<double>::infinity()) -
                     realized) +
            std::abs(std::nextafter(
                         state.grid.cadence_sec,
                         std::numeric_limits<double>::infinity()) -
                     state.grid.cadence_sec);
        if (!std::isfinite(current) || !(current > previous) ||
            !std::isfinite(realized) ||
            std::abs(realized - state.grid.cadence_sec) >
                representation_bound) {
            throw std::logic_error(
                "simulated native full axis conflicts with compact grid");
        }
    }
}

inline AlignmentGoverningCompatibilityAxis
make_governing_gap_compatibility_axis(const AlignmentGridState &grid,
                                      double raw_overlap_end_sec) {
    if (!grid.initialized || !std::isfinite(grid.phase_sec) ||
        !std::isfinite(grid.cadence_sec) || !(grid.cadence_sec > 0.0) ||
        !std::isfinite(raw_overlap_end_sec) ||
        raw_overlap_end_sec < grid.phase_sec) {
        throw std::logic_error(
            "invalid grid or raw overlap for governing compatibility axis");
    }

    // This intentionally preserves the governing 9aae gap-grid count
    // operation: positive floating quotient, narrowing to int, then +1.
    const double legacy_quotient =
        (raw_overlap_end_sec - grid.phase_sec) / grid.cadence_sec;
    if (!std::isfinite(legacy_quotient) || legacy_quotient < 0.0 ||
        legacy_quotient >
            static_cast<double>(std::numeric_limits<int>::max() - 1)) {
        throw std::overflow_error(
            "governing compatibility sample count exceeds legacy int range");
    }
    const int legacy_count = static_cast<int>(legacy_quotient) + 1;
    if (legacy_count <= 0 || grid.first_global_slot > 0 ||
        grid.first_global_slot == std::numeric_limits<std::int64_t>::min()) {
        throw std::logic_error(
            "detector union does not contain governing global slot zero");
    }

    const auto stop_global_slot = static_cast<std::int64_t>(legacy_count);
    if (grid.last_global_slot < stop_global_slot - 1) {
        throw std::logic_error(
            "detector union does not contain governing compatibility support");
    }
    const auto local_start64 = -grid.first_global_slot;
    if (local_start64 < 0 ||
        local_start64 >
            std::numeric_limits<std::int64_t>::max() - stop_global_slot) {
        throw std::overflow_error(
            "governing compatibility interval exceeds local index range");
    }
    const auto local_stop64 = local_start64 + stop_global_slot;
    if (local_stop64 <= local_start64 ||
        static_cast<std::uint64_t>(local_stop64) >
            static_cast<std::uint64_t>(
                std::numeric_limits<Eigen::Index>::max())) {
        throw std::overflow_error(
            "governing compatibility interval exceeds local index range");
    }

    AlignmentGoverningCompatibilityAxis result;
    result.initialized = true;
    result.raw_overlap_end_sec = raw_overlap_end_sec;
    result.first_global_slot = 0;
    result.stop_global_slot = stop_global_slot;
    result.union_local_start = static_cast<Eigen::Index>(local_start64);
    result.union_local_stop = static_cast<Eigen::Index>(local_stop64);
    return result;
}

inline void validate_governing_compatibility_axis_structure(
    const TimestreamAlignmentState &state) {
    const auto &grid = state.grid;
    const auto &axis = state.governing_compatibility_axis;
    if (!grid.initialized || !axis.initialized ||
        !std::isfinite(grid.phase_sec) ||
        !std::isfinite(grid.cadence_sec) || !(grid.cadence_sec > 0.0) ||
        !std::isfinite(axis.raw_overlap_end_sec) ||
        axis.first_global_slot != 0 ||
        axis.stop_global_slot <= axis.first_global_slot ||
        axis.union_local_start < 0 ||
        axis.union_local_stop <= axis.union_local_start ||
        axis.assigned_time_constructor !=
            governing_compatibility_assigned_time_constructor ||
        axis.source_application_sha !=
            "9aae0e669384c5c0c0dda93debc194d6b8dac787" ||
        axis.consumer_scope !=
            "legacy_outputs_and_legacy_science_consumers") {
        throw std::logic_error(
            "governing compatibility axis has an incomplete identity");
    }

    const auto count64 =
        static_cast<std::uint64_t>(axis.stop_global_slot) -
        static_cast<std::uint64_t>(axis.first_global_slot);
    if (count64 > static_cast<std::uint64_t>(
                      std::numeric_limits<Eigen::Index>::max()) ||
        axis.union_local_stop - axis.union_local_start !=
            static_cast<Eigen::Index>(count64) ||
        checked_add_nonnegative_slot_offset(
            grid.first_global_slot, axis.union_local_start) !=
            axis.first_global_slot ||
        checked_add_nonnegative_slot_offset(
            grid.first_global_slot, axis.union_local_stop) !=
            axis.stop_global_slot ||
        axis.stop_global_slot - 1 > grid.last_global_slot) {
        throw std::logic_error(
            "governing compatibility axis conflicts with detector union slots");
    }

    const auto union_count = checked_union_sample_count(grid);
    if (axis.union_local_stop > union_count) {
        throw std::logic_error(
            "governing compatibility interval lies outside detector union");
    }

    const double legacy_quotient =
        (axis.raw_overlap_end_sec - grid.phase_sec) / grid.cadence_sec;
    if (!std::isfinite(legacy_quotient) || legacy_quotient < 0.0 ||
        legacy_quotient >
            static_cast<double>(std::numeric_limits<int>::max() - 1) ||
        static_cast<std::int64_t>(static_cast<int>(legacy_quotient)) + 1 !=
            static_cast<std::int64_t>(count64)) {
        throw std::logic_error(
            "governing compatibility support conflicts with raw overlap end");
    }
}

inline auto governing_compatibility_assigned_time_expression(
    const TimestreamAlignmentState &state) {
    validate_governing_compatibility_axis_structure(state);
    const auto count = governing_compatibility_sample_count(
        state.governing_compatibility_axis);
    const double high =
        state.grid.phase_sec +
        state.grid.cadence_sec * static_cast<double>(count - 1);
    return Eigen::VectorXd::LinSpaced(
        count, state.grid.phase_sec, high);
}

inline void validate_governing_compatibility_assigned_times(
    const Eigen::VectorXd &common_time,
    const TimestreamAlignmentState &state) {
    if (uses_native_simulation_full_axis(state)) {
        validate_native_simulation_full_axis_structure(state);
        if (common_time.size() != state.common_time.size()) {
            throw std::logic_error(
                "simulated common-time candidate has invalid cardinality");
        }
        for (Eigen::Index local = 0; local < common_time.size();
             ++local) {
            if (!std::isfinite(common_time(local)) ||
                common_time(local) != state.common_time(local) ||
                (local > 0 &&
                 common_time(local) <= common_time(local - 1))) {
                throw std::logic_error(
                    "simulated common time conflicts with its exact native coordinate identity");
            }
        }
        return;
    }
    validate_governing_compatibility_axis_structure(state);
    const auto union_count = checked_union_sample_count(state.grid);
    if (common_time.size() != union_count) {
        throw std::logic_error(
            "common time does not cover the complete detector union");
    }

    const auto &axis = state.governing_compatibility_axis;
    const auto expected_compatibility =
        governing_compatibility_assigned_time_expression(state);
    for (Eigen::Index local = 0; local < common_time.size(); ++local) {
        double expected = 0.0;
        if (local >= axis.union_local_start &&
            local < axis.union_local_stop) {
            expected = expected_compatibility(
                local - axis.union_local_start);
        }
        else {
            const auto global_slot = checked_add_nonnegative_slot_offset(
                state.grid.first_global_slot, local);
            expected = state.grid.phase_sec +
                       static_cast<double>(global_slot) *
                           state.grid.cadence_sec;
        }
        if (!std::isfinite(common_time(local)) ||
            common_time(local) != expected ||
            (local > 0 &&
             common_time(local) <= common_time(local - 1))) {
            throw std::logic_error(
                "common time conflicts with union/compatibility constructors");
        }
    }
}

inline void validate_governing_compatibility_assigned_times(
    const TimestreamAlignmentState &state) {
    validate_governing_compatibility_assigned_times(
        state.common_time, state);
}

inline Eigen::Index governing_consumer_local_start(
    const TimestreamAlignmentState &state) {
    if (state.governing_compatibility_axis.initialized) {
        validate_governing_compatibility_axis_structure(state);
        return state.governing_compatibility_axis.union_local_start;
    }
    if (uses_native_simulation_full_axis(state)) {
        validate_native_simulation_full_axis_structure(state);
        return 0;
    }
    throw std::logic_error(
        "governing consumer support is unavailable");
}

inline Eigen::Index governing_consumer_local_stop(
    const TimestreamAlignmentState &state) {
    if (state.governing_compatibility_axis.initialized) {
        validate_governing_compatibility_axis_structure(state);
        return state.governing_compatibility_axis.union_local_stop;
    }
    if (uses_native_simulation_full_axis(state)) {
        validate_native_simulation_full_axis_structure(state);
        return state.common_time.size();
    }
    throw std::logic_error(
        "governing consumer support is unavailable");
}

inline Eigen::Index governing_consumer_sample_count(
    const TimestreamAlignmentState &state) {
    return governing_consumer_local_stop(state) -
           governing_consumer_local_start(state);
}

inline void install_governing_compatibility_assigned_times(
    Eigen::VectorXd &common_time,
    const TimestreamAlignmentState &state) {
    validate_governing_compatibility_axis_structure(state);
    const auto union_count = checked_union_sample_count(state.grid);
    if (common_time.size() != union_count) {
        throw std::logic_error(
            "cannot install compatibility times outside the detector union");
    }
    const auto &axis = state.governing_compatibility_axis;
    common_time.segment(
        axis.union_local_start,
        governing_compatibility_sample_count(axis)) =
            governing_compatibility_assigned_time_expression(state);
    validate_governing_compatibility_assigned_times(common_time, state);
}

inline void install_governing_compatibility_assigned_times(
    TimestreamAlignmentState &state) {
    install_governing_compatibility_assigned_times(
        state.common_time, state);
}

template <class Series>
void validate_governing_compatibility_series(
    const Series &values, const TimestreamAlignmentState &state,
    const char *label) {
    (void)governing_consumer_sample_count(state);
    if (values.size() != state.common_time.size()) {
        throw std::logic_error(
            std::string{label} +
            " does not cover the complete detector union axis");
    }
}

inline auto governing_compatibility_segment(
    Eigen::VectorXd &values, const TimestreamAlignmentState &state) {
    validate_governing_compatibility_series(
        values, state, "governing compatibility series");
    return values.segment(
        governing_consumer_local_start(state),
        governing_consumer_sample_count(state));
}

inline auto governing_compatibility_segment(
    const Eigen::VectorXd &values, const TimestreamAlignmentState &state) {
    validate_governing_compatibility_series(
        values, state, "governing compatibility series");
    return values.segment(
        governing_consumer_local_start(state),
        governing_consumer_sample_count(state));
}

inline double governing_compatibility_mean(
    const Eigen::VectorXd &values, const TimestreamAlignmentState &state) {
    return governing_compatibility_segment(values, state).mean();
}

template <class Series>
auto governing_compatibility_start_value(
    const Series &values, const TimestreamAlignmentState &state) {
    validate_governing_compatibility_series(
        values, state, "governing compatibility series");
    return values(governing_consumer_local_start(state));
}

template <class Series>
auto governing_compatibility_stop_value(
    const Series &values, const TimestreamAlignmentState &state) {
    validate_governing_compatibility_series(
        values, state, "governing compatibility series");
    return values(governing_consumer_local_stop(state) - 1);
}

template <class Engine, class = void>
struct has_governing_compatibility_axis_state : std::false_type {};

template <class Engine>
struct has_governing_compatibility_axis_state<
    Engine,
    std::void_t<decltype(std::declval<const Engine &>()
                             .alignment.governing_compatibility_axis)>>
    : std::true_type {};

template <class Engine>
Eigen::Index governing_compatibility_sample_count_or(
    const Engine &engine, Eigen::Index fallback_count) {
    if constexpr (has_governing_compatibility_axis_state<Engine>::value) {
        if (engine.alignment.grid.initialized) {
            return governing_consumer_sample_count(engine.alignment);
        }
    }
    return fallback_count;
}

inline void reset_alignment_observation_state(
    TimestreamAlignmentState &state) {
    state = TimestreamAlignmentState{};
}

inline void clear_alignment_windows(TimestreamAlignmentState &state) {
    state.start_indices.clear();
    state.end_indices.clear();
    state.hwpr_start_index = -1;
    state.hwpr_end_index = -1;
}

inline void clear_gap_alignment_state(TimestreamAlignmentState &state) {
    reset_alignment_observation_state(state);
}

}  // namespace citlali::pipeline
