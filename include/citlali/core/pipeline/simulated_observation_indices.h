#pragma once

#include <citlali/core/config/interface_sync_config.h>
#include <citlali/core/pipeline/interface_sync_config_adapter.h>
#include <citlali/core/pipeline/rawobs_data_items.h>
#include <citlali/core/pipeline/sci_align_contract.h>
#include <citlali/core/pipeline/sci_align_field_registry.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

namespace simulation_alignment_detail {

template <class Engine, class = void>
struct has_interface_sync_state : std::false_type {};

template <class Engine>
struct has_interface_sync_state<
    Engine,
    std::void_t<decltype(std::declval<Engine &>().interface_sync)>>
    : std::true_type {};

template <class Engine, class = void>
struct has_interface_sync_request : std::false_type {};

template <class Engine>
struct has_interface_sync_request<
    Engine,
    std::void_t<decltype(
        std::declval<const Engine &>().typed_config.interface_sync)>>
    : std::true_type {};

template <class Engine, class = void>
struct has_raw_timestream_observation_state : std::false_type {};

template <class Engine>
struct has_raw_timestream_observation_state<
    Engine,
    std::void_t<decltype(
        std::declval<Engine &>().raw_timestream_plan.observation)>>
    : std::true_type {};

inline double floating_coordinate_step(double value) {
    return std::abs(
        std::nextafter(value, std::numeric_limits<double>::infinity()) -
        value);
}

inline bool cadence_matches_representation(double previous, double current,
                                           double cadence) {
    const double realized = current - previous;
    const double representation_bound =
        floating_coordinate_step(previous) +
        floating_coordinate_step(current) +
        floating_coordinate_step(realized) +
        floating_coordinate_step(cadence);
    return std::isfinite(realized) &&
           std::abs(realized - cadence) <= representation_bound;
}

template <class Engine>
InterfaceSyncState prepare_interface_sync_state(
    const Engine &engine, const std::vector<std::string> &interface_ids) {
    InterfaceSyncState state;
    if constexpr (!has_interface_sync_state<Engine>::value) {
        return state;
    }
    else {
        state = engine.interface_sync;
        if (state.lifecycle.empty()) {
            if constexpr (has_interface_sync_request<Engine>::value) {
                adapt_interface_sync_config_one_way(
                    engine.typed_config.interface_sync, state);
            }
            else {
                adapt_interface_sync_config_one_way(
                    citlali::config::InterfaceSyncOffsetConfig{}, state);
            }
        }
        begin_interface_sync_observation(state);

        std::set<std::string> selected(interface_ids.begin(),
                                       interface_ids.end());
        for (const auto &interface_id : interface_ids) {
            // A simulation supplies an index-coordinate compatibility axis,
            // not authority for relating distinct clocks.  Zero is therefore
            // the only realizable offset.
            (void)realize_interface_offset(state, interface_id, false);
        }
        for (auto &record : state.lifecycle) {
            if (record.interface_id.rfind("toltec", 0) == 0 &&
                selected.find(record.interface_id) == selected.end() &&
                record.effective_sec != 0.0) {
                record.availability =
                    OffsetAvailability::unavailable_authority;
                throw std::runtime_error(
                    "nonzero offset was requested for absent simulated interface " +
                    record.interface_id);
            }
        }
        auto &hwpr = require_interface_offset_record(state, "hwpr");
        if (hwpr.effective_sec != 0.0) {
            hwpr.availability = OffsetAvailability::unavailable_authority;
            throw std::runtime_error(
                "nonzero HWPR offset is unavailable for a non-HWPR simulation");
        }
        (void)realize_interface_offset(state, "lmt", false);
        return state;
    }
}

template <class Engine>
void reset_interface_sync_state(Engine &engine) {
    if constexpr (has_interface_sync_state<Engine>::value) {
        if (engine.interface_sync.lifecycle.empty()) {
            if constexpr (has_interface_sync_request<Engine>::value) {
                adapt_interface_sync_config_one_way(
                    engine.typed_config.interface_sync,
                    engine.interface_sync);
            }
            else {
                adapt_interface_sync_config_one_way(
                    citlali::config::InterfaceSyncOffsetConfig{},
                    engine.interface_sync);
            }
        }
        begin_interface_sync_observation(engine.interface_sync);
        if constexpr (has_raw_timestream_observation_state<Engine>::value) {
            if (engine.raw_timestream_plan.observation.has_value()) {
                engine.raw_timestream_plan.observation->interface_offsets =
                    engine.interface_sync.lifecycle;
            }
        }
    }
}

template <class Engine>
void reject_unavailable_simulation_offsets(
    Engine &engine, const std::vector<std::string> &interface_ids) {
    if constexpr (has_interface_sync_state<Engine>::value) {
        const std::set<std::string> selected(interface_ids.begin(),
                                             interface_ids.end());
        for (auto &record : engine.interface_sync.lifecycle) {
            const bool detector =
                record.interface_id.rfind("toltec", 0) == 0;
            const bool selected_detector =
                detector && selected.find(record.interface_id) !=
                                selected.end();
            const bool absent_detector = detector && !selected_detector;
            const bool unavailable_clock_relation =
                record.interface_id == "hwpr" ||
                record.interface_id == "lmt";
            if (record.effective_sec == 0.0 ||
                !(selected_detector || absent_detector ||
                  unavailable_clock_relation)) {
                continue;
            }
            record.availability =
                OffsetAvailability::unavailable_authority;
            if constexpr (
                has_raw_timestream_observation_state<Engine>::value) {
                if (engine.raw_timestream_plan.observation.has_value()) {
                    engine.raw_timestream_plan.observation
                        ->interface_offsets =
                        engine.interface_sync.lifecycle;
                }
            }
            throw std::runtime_error(
                "nonzero interface offset lacks simulation clock/epoch authority for " +
                record.interface_id);
        }
    }
}

template <class Engine>
void publish_interface_sync_state(Engine &engine, InterfaceSyncState state) {
    if constexpr (has_interface_sync_state<Engine>::value) {
        engine.interface_sync = std::move(state);
        if constexpr (has_raw_timestream_observation_state<Engine>::value) {
            if (engine.raw_timestream_plan.observation.has_value()) {
                engine.raw_timestream_plan.observation->interface_offsets =
                    engine.interface_sync.lifecycle;
            }
        }
    }
}

}  // namespace simulation_alignment_detail

template <class Engine, class RawObs>
void reset_simulated_observation_indices(Engine &engine,
                                         const RawObs &rawobs) {
    // Cross-observation state is never retained after entering a new
    // simulated-observation boundary, including when later validation fails.
    reset_alignment_observation_state(engine.alignment);
    simulation_alignment_detail::reset_interface_sync_state(engine);

    if (engine.calib.run_hwpr) {
        throw std::runtime_error(
            "enabled HWPR alignment is unavailable for simulated observations in the bounded SCI-ALIGN-001 profile");
    }

    const auto tel_time_it = engine.telescope.tel_data.find("TelTime");
    if (tel_time_it == engine.telescope.tel_data.end() ||
        tel_time_it->second.size() == 0) {
        throw std::runtime_error(
            "simulated observation requires a nonempty telescope TelTime axis");
    }
    const double sample_frequency_hz = engine.telescope.fsmp;
    sci_align::require_finite_positive(sample_frequency_hz,
                                       "simulated telescope fsmp");
    const double cadence_sec = 1.0 / sample_frequency_hz;
    sci_align::require_finite_positive(cadence_sec,
                                       "simulated telescope cadence");

    const Eigen::Index native_count =
        static_cast<Eigen::Index>(tel_time_it->second.size());
    if (native_count <= 0 ||
        static_cast<std::uint64_t>(native_count - 1) >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(
            "simulated telescope axis exceeds the ALIGN index range");
    }

    Eigen::VectorXd native_time(native_count);
    for (Eigen::Index row = 0; row < native_count; ++row) {
        native_time(row) = tel_time_it->second(row);
        if (!std::isfinite(native_time(row))) {
            throw std::runtime_error(
                "simulated telescope TelTime contains a non-finite coordinate");
        }
        if (row > 0) {
            if (!(native_time(row) > native_time(row - 1))) {
                throw std::runtime_error(
                    "simulated telescope TelTime must be strictly increasing");
            }
            if (!simulation_alignment_detail::
                    cadence_matches_representation(
                        native_time(row - 1), native_time(row),
                        cadence_sec)) {
                throw std::runtime_error(
                    "simulated telescope TelTime is not uniform at the declared fsmp");
            }
        }
    }

    const auto kids_data = rawobs.kidsdata();
    if (kids_data.empty()) {
        throw std::runtime_error(
            "simulated observation has no detector interfaces");
    }
    std::vector<std::string> interface_ids;
    interface_ids.reserve(kids_data.size());
    std::set<std::string> supplied_ids;
    for (const auto &data_item_ref : kids_data) {
        const auto &data_item =
            detail::unwrap_reference_wrapper(data_item_ref);
        const std::string interface_id = data_item.interface();
        (void)sci_align::parse_toltec_interface_identity(interface_id);
        if (!supplied_ids.insert(interface_id).second) {
            throw std::runtime_error(
                "duplicate simulated detector interface " + interface_id);
        }
        interface_ids.push_back(interface_id);
    }
    simulation_alignment_detail::reject_unavailable_simulation_offsets(
        engine, interface_ids);

    const auto slot_count = static_cast<std::uint64_t>(native_count);
    const auto interface_slot_count =
        checked_alignment_interface_slot_capacity(
            slot_count, interface_ids.size());

    TimestreamAlignmentState state;
    state.grid.initialized = true;
    state.grid.phase_sec = native_time(0);
    state.grid.cadence_sec = cadence_sec;
    state.grid.exclusive_half_cell_sec = cadence_sec / 2.0;
    state.grid.first_global_slot = 0;
    state.grid.last_global_slot =
        static_cast<std::int64_t>(native_count - 1);
    state.grid.assignment_operator = "floor_q_plus_half_v1";
    state.grid.phase_semantics =
        "simulator_native_telescope_coordinate_exact";
    state.grid.physical_timestamp_semantics =
        "unavailable_no_integration_event_authority";

    // A simulation supplies one already co-registered native coordinate.
    // Preserve its exact floating representation rather than moving an
    // accepted row onto a formula-generated coordinate by one ULP.
    state.common_time = native_time;

    state.masks.reserve(interface_ids.size());
    state.network_times.reserve(interface_ids.size());
    state.interfaces.reserve(interface_ids.size());
    state.start_indices.reserve(interface_ids.size());
    state.end_indices.reserve(interface_ids.size());
    for (const auto &interface_id : interface_ids) {
        const auto roach_index = static_cast<Eigen::Index>(
            sci_align::parse_toltec_interface_identity(interface_id));
        Eigen::VectorXi mask = Eigen::VectorXi::Ones(native_count);
        state.network_masks.emplace(roach_index, mask);
        state.masks.push_back(std::move(mask));
        state.network_times.push_back(state.common_time);
        state.start_indices.push_back(0);
        state.end_indices.push_back(native_count - 1);

        AlignmentInterfaceSummary summary;
        summary.interface_id = interface_id;
        summary.roach_index = roach_index;
        summary.native_row_count = native_count;
        summary.accepted_row_count = native_count;
        summary.first_global_slot = 0;
        summary.last_global_slot =
            static_cast<std::int64_t>(native_count - 1);
        state.interfaces.push_back(std::move(summary));
    }

    state.telescope.initialized = true;
    state.telescope.coordinate_identity =
        "Data.TelescopeBackend.TelTime";
    state.telescope.epoch_event_precision_authority = "unavailable";
    state.telescope.support_rule =
        "exact_simulator_index_co_registration_no_physical_timestamp_semantics";
    state.telescope.native_row_count = native_count;
    state.telescope.native_first_coordinate_sec = native_time(0);
    state.telescope.native_last_coordinate_sec =
        native_time(native_count - 1);
    state.telescope.exact_target_count = slot_count;
    state.telescope.interpolated_target_count = 0;
    state.telescope.minimum_used_bracket_span_sec = 0.0;
    state.telescope.maximum_used_bracket_span_sec = 0.0;
    state.telescope.native_tel_utc_available =
        engine.telescope.tel_data.find("TelUTC") !=
        engine.telescope.tel_data.end();
    state.telescope.native_pps_time_available =
        engine.telescope.tel_data.find("PpsTime") !=
        engine.telescope.tel_data.end();
    state.hwpr = bounded_nonpolarimetric_hwpr_summary(
        observation_hwpr_input_present(rawobs));

    state.support.nominal_slot_count = slot_count;
    state.support.acquired_original_count = interface_slot_count;
    state.support.timing_coordinate_valid_original_count =
        interface_slot_count;
    state.support.synthesized_count = 0;
    state.support.unavailable_count = 0;
    state.support.guarded_original_count = 0;
    // Gap-policy eligibility is observation-resolved only after the
    // scan/chunk plan and its dispositions exist. Detector-signal validity
    // and final science eligibility remain downstream/unavailable here.
    state.support.gap_policy_eligible_original_count = 0;
    state.support.nominal_span_sec =
        static_cast<double>(slot_count) * cadence_sec;
    state.support.acquired_original_cadence_weighted_support_sec =
        static_cast<double>(interface_slot_count) * cadence_sec;
    if (!std::isfinite(state.support.nominal_span_sec) ||
        !std::isfinite(
            state.support.acquired_original_cadence_weighted_support_sec)) {
        throw std::overflow_error(
            "simulated alignment support duration overflows");
    }
    state.availability.mapping = AlignmentTermAvailability::available;
    state.availability.conditional_response =
        AlignmentTermAvailability::not_applicable;
    state.field_registry_version =
        std::string{sci_align::active_field_registry_version};

    auto interface_sync =
        simulation_alignment_detail::prepare_interface_sync_state(
            engine, interface_ids);

    // Publish only after every identity, coordinate, and lifecycle gate has
    // succeeded. Native simulation TelTime/TelUTC and the internal assigned
    // coordinate retain the same exact representation.
    engine.alignment = std::move(state);
    simulation_alignment_detail::publish_interface_sync_state(
        engine, std::move(interface_sync));
}

}  // namespace citlali::pipeline
