#pragma once

#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_alignment_helpers.h>
#include <citlali/core/pipeline/timestream_native_pointing.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

template <class Engine, class = void>
struct has_native_pointing_capability : std::false_type {};

template <class Engine>
struct has_native_pointing_capability<
    Engine,
    std::void_t<
        decltype(std::declval<Engine &>()
                     .alignment.native_consumer_plan),
        decltype(std::declval<Engine &>()
                     .alignment.raw_telescope_trajectory),
        decltype(std::declval<Engine &>()
                     .alignment.native_pointing_plan),
        decltype(std::declval<Engine &>().pointing_offsets),
        decltype(std::declval<decltype(
                     std::declval<Engine &>().telescope.tel_data) &>() =
                 std::declval<NativeTelescopeData>())>>
    : std::true_type {};

template <class Engine>
inline constexpr bool has_native_pointing_capability_v =
    has_native_pointing_capability<Engine>::value;

template <class Engine>
std::shared_ptr<const NativePointingPlan>
build_native_pointing_plan_candidate(Engine &engine) {
    const auto &alignment_plan = engine.alignment.native_consumer_plan;
    const auto &raw_telescope =
        engine.alignment.raw_telescope_trajectory;
    if (!alignment_plan || !raw_telescope) {
        throw std::logic_error(
            "network-native pointing requires exact alignment and raw telescope handles");
    }

    const auto common_time_it = engine.telescope.tel_data.find("TelTime");
    if (common_time_it == engine.telescope.tel_data.end()) {
        throw std::logic_error(
            "network-native pointing requires aligned common TelTime only to preserve pointing-offset support policy");
    }
    const auto offset_model = make_native_pointing_offset_model(
        engine.pointing_offsets, common_time_it->second);

    std::vector<NativeNetworkPointing> native_networks;
    native_networks.reserve(alignment_plan->networks().size());
    for (const auto network_id :
         alignment_plan->participant_network_ids()) {
        const auto &alignment = alignment_plan->network(network_id);
        std::optional<TimestreamNativeRow> first_mapped_row;
        std::optional<TimestreamNativeRow> past_last_mapped_row;
        for (std::size_t slot = 0; slot < alignment_plan->slot_count();
             ++slot) {
            const auto &association =
                alignment_plan->association(network_id, slot);
            if (!association.mapped()) {
                continue;
            }
            if (!first_mapped_row.has_value() ||
                association.native_row < *first_mapped_row) {
                first_mapped_row = association.native_row;
            }
            const auto past_row = association.native_row + 1;
            if (!past_last_mapped_row.has_value() ||
                past_row > *past_last_mapped_row) {
                past_last_mapped_row = past_row;
            }
        }
        if (!first_mapped_row.has_value() ||
            !past_last_mapped_row.has_value() ||
            *first_mapped_row >= *past_last_mapped_row) {
            throw std::logic_error(
                "network-native pointing requires at least one mapped delivered row per participant");
        }

        const Eigen::Index first_local = static_cast<Eigen::Index>(
            *first_mapped_row - alignment.first_native_row());
        const Eigen::Index row_count = static_cast<Eigen::Index>(
            *past_last_mapped_row - *first_mapped_row);
        Eigen::VectorXd target_reconstructed_times =
            alignment.reconstructed_times_unix_sec().segment(
                first_local, row_count);
        auto evaluated_telescope =
            evaluate_raw_telescope_trajectory_at(
                *raw_telescope, target_reconstructed_times);

        // Reuse the production Telescope tangent-plane mathematics on an
        // isolated value copy.  The common-row Telescope remains untouched by
        // this network-specific evaluation.
        auto isolated_telescope = engine.telescope;
        isolated_telescope.tel_data = std::move(evaluated_telescope);
        isolated_telescope.calc_tan_pointing();

        auto evaluated_offsets =
            offset_model.evaluate_at(target_reconstructed_times);
        native_networks.emplace_back(
            network_id, *first_mapped_row,
            std::move(target_reconstructed_times),
            std::move(isolated_telescope.tel_data),
            std::move(evaluated_offsets));
    }

    return std::make_shared<const NativePointingPlan>(
        alignment_plan, raw_telescope, std::move(native_networks));
}

template <class Engine, class Logger>
void calculate_tangent_plane_pointing(Engine &engine, const Logger &logger) {
    logger->info("calculating tangent plane pointing");
    engine.telescope.calc_tan_pointing();
}

template <class TodProc, class Logger>
void interpolate_pointing_offsets(TodProc &todproc, const Logger &logger) {
    logger->info("calculating pointing offsets");
    todproc.interp_pointing();
    auto &engine = todproc.engine();
    if constexpr (has_astrometry_plan_v<decltype(engine)>) {
        record_astrometry_applied(
            astrometry_plan(engine),
            static_cast<std::size_t>(
                engine.telescope.tel_data["TelTime"].size()));
    }
}

template <class TodProc, class Logger>
void calculate_telescope_pointing(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    if constexpr (has_native_pointing_capability_v<
                      std::remove_reference_t<decltype(engine)>>) {
        const bool has_native_alignment =
            static_cast<bool>(engine.alignment.native_consumer_plan);
        const bool has_raw_telescope =
            static_cast<bool>(engine.alignment.raw_telescope_trajectory);
        if (has_native_alignment != has_raw_telescope) {
            throw std::logic_error(
                "network-native pointing state is incomplete before evaluation");
        }

        std::shared_ptr<const NativePointingPlan> candidate_native_pointing;
        if (has_native_alignment) {
            candidate_native_pointing =
                build_native_pointing_plan_candidate(engine);
        }

        calculate_tangent_plane_pointing(engine, logger);
        interpolate_pointing_offsets(todproc, logger);

        // A null pair is the explicit legacy/simulated-observation state.
        // When a native alignment exists, no common-time compatibility
        // fallback is permitted: all native evaluation above must succeed
        // before publication.
        engine.alignment.native_pointing_plan =
            std::move(candidate_native_pointing);
    } else {
        calculate_tangent_plane_pointing(engine, logger);
        interpolate_pointing_offsets(todproc, logger);
    }
}

}  // namespace citlali::pipeline
