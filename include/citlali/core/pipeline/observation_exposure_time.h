#pragma once

#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace citlali::pipeline {

struct ObservationExposureSummary {
    bool alignment_initialized = false;
    std::uint64_t nominal_common_axis_slot_count = 0;
    std::uint64_t acquired_original_interface_slot_count = 0;
    std::uint64_t timing_coordinate_valid_original_interface_slot_count = 0;
    std::uint64_t synthesized_interface_slot_count = 0;
    std::uint64_t unavailable_interface_slot_count = 0;
    std::uint64_t acquired_original_observation_union_slot_count = 0;
    double cadence_sec = 0.0;
    double nominal_support_span_sec = 0.0;
    double acquired_original_observation_cadence_weighted_support_sec = 0.0;
};

template <class Engine, class = void>
struct has_timestream_alignment_state : std::false_type {};

template <class Engine>
struct has_timestream_alignment_state<
    Engine,
    std::void_t<
        decltype(std::declval<const Engine &>().alignment.grid.initialized),
        decltype(std::declval<const Engine &>().alignment.interfaces),
        decltype(std::declval<const Engine &>().alignment.masks),
        decltype(std::declval<const Engine &>().alignment.support)>>
    : std::true_type {};

template <class Engine>
inline constexpr bool has_timestream_alignment_state_v =
    has_timestream_alignment_state<Engine>::value;

inline ObservationExposureSummary aligned_observation_exposure_summary(
    const TimestreamAlignmentState &alignment) {
    if (!alignment.grid.initialized) {
        return {};
    }

    const auto &grid = alignment.grid;
    const auto &support = alignment.support;
    if (!std::isfinite(grid.cadence_sec) || grid.cadence_sec <= 0.0 ||
        support.nominal_slot_count == 0 || alignment.interfaces.empty()) {
        throw std::logic_error(
            "initialized alignment has no finite positive support grid");
    }
    const auto slot_span = static_cast<std::uint64_t>(
        checked_union_sample_count(grid));
    if (slot_span != support.nominal_slot_count) {
        throw std::logic_error(
            "alignment support count conflicts with its global slot interval");
    }
    if (alignment.masks.size() != alignment.interfaces.size()) {
        throw std::logic_error(
            "alignment support masks do not match interface identities");
    }

    if (support.nominal_slot_count > static_cast<std::uint64_t>(
            std::numeric_limits<Eigen::Index>::max())) {
        throw std::overflow_error(
            "alignment common-grid support exceeds index range");
    }
    const auto nominal_size =
        static_cast<Eigen::Index>(support.nominal_slot_count);
    std::uint64_t acquired_original_count = 0;
    for (const auto &mask : alignment.masks) {
        if (mask.size() != nominal_size) {
            throw std::logic_error(
                "alignment support mask does not cover the common grid");
        }
        acquired_original_count = checked_alignment_count_add(
            acquired_original_count,
            count_binary_alignment_mask(
                mask, "alignment support mask"),
            "alignment acquired-original count");
    }
    if (acquired_original_count != support.acquired_original_count) {
        throw std::logic_error(
            "alignment acquired-original count conflicts with support masks");
    }
    if (support.timing_coordinate_valid_original_count >
            support.acquired_original_count ||
        support.gap_policy_eligible_original_count >
            support.timing_coordinate_valid_original_count ||
        support.guarded_original_count > support.acquired_original_count) {
        throw std::logic_error(
            "alignment validity/support counts are inconsistent");
    }

    const auto interface_slot_capacity =
        checked_alignment_interface_slot_capacity(
            support.nominal_slot_count, alignment.interfaces.size());
    if (support.acquired_original_count > interface_slot_capacity ||
        support.synthesized_count >
            interface_slot_capacity - support.acquired_original_count ||
        support.unavailable_count !=
            interface_slot_capacity - support.acquired_original_count -
                support.synthesized_count) {
        throw std::logic_error(
            "alignment origin counts do not partition interface-slot support");
    }

    // The timing-coordinate-valid cohort is exactly the acquired-slot cohort.
    // Detector-signal validity has no authority here and is not inferred.
    if (support.timing_coordinate_valid_original_count !=
        support.acquired_original_count) {
        throw std::logic_error(
            "timing-coordinate validity conflicts with acquired support");
    }

    ObservationExposureSummary result;
    result.alignment_initialized = true;
    result.nominal_common_axis_slot_count = support.nominal_slot_count;
    result.acquired_original_interface_slot_count =
        support.acquired_original_count;
    result.timing_coordinate_valid_original_interface_slot_count =
        support.timing_coordinate_valid_original_count;
    result.synthesized_interface_slot_count = support.synthesized_count;
    result.unavailable_interface_slot_count = support.unavailable_count;
    for (Eigen::Index slot = 0; slot < nominal_size; ++slot) {
        for (const auto &mask : alignment.masks) {
            if (mask(slot) != 0) {
                ++result.acquired_original_observation_union_slot_count;
                break;
            }
        }
    }
    result.cadence_sec = grid.cadence_sec;
    result.nominal_support_span_sec =
        static_cast<double>(support.nominal_slot_count) * grid.cadence_sec;
    result.acquired_original_observation_cadence_weighted_support_sec =
        static_cast<double>(
            result.acquired_original_observation_union_slot_count) *
        grid.cadence_sec;
    return result;
}

template <class Engine>
auto observation_start_time(Engine &engine) {
    if constexpr (has_timestream_alignment_state_v<Engine>) {
        if (engine.alignment.grid.initialized) {
            return governing_compatibility_start_value(
                engine.telescope.tel_data["TelTime"], engine.alignment);
        }
    }
    return engine.telescope.tel_data["TelTime"](0);
}

template <class Engine>
auto observation_stop_time(Engine &engine) {
    if constexpr (has_timestream_alignment_state_v<Engine>) {
        if (engine.alignment.grid.initialized) {
            return governing_compatibility_stop_value(
                engine.telescope.tel_data["TelTime"], engine.alignment);
        }
    }
    return engine.telescope.tel_data["TelTime"](
        engine.telescope.tel_data["TelTime"].size() - 1);
}

template <class Engine>
double calculate_observation_exposure_time(Engine &engine) {
    // Preserve the governing EXPTIME/coadd compatibility identity. The
    // compact ALIGN support/exposure accounting is emitted separately in the
    // timestream provenance and does not redefine this legacy product.
    return static_cast<double>(observation_stop_time(engine) -
                               observation_start_time(engine));
}

template <class Engine>
bool should_accumulate_coadd_exposure_time(const Engine &engine) {
    return coadd_outputs_enabled(engine);
}

template <class Engine>
void accumulate_coadd_exposure_time(Engine &engine) {
    engine.cmb.exposure_time =
        engine.cmb.exposure_time + engine.omb.exposure_time;
}

template <class Engine>
void update_observation_exposure_time(Engine &engine) {
    engine.omb.exposure_time = calculate_observation_exposure_time(engine);
    if (should_accumulate_coadd_exposure_time(engine)) {
        accumulate_coadd_exposure_time(engine);
    }
}

template <class Engine>
void update_reduction_observation_exposure_time(Engine &engine) {
    update_observation_exposure_time(engine);
}

}  // namespace citlali::pipeline
