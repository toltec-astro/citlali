#pragma once

#include <citlali/core/pipeline/native_observation_carriers.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_native_pointing.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

template <class Engine>
inline constexpr bool native_pointing_publication_capable_v =
    requires(Engine &engine, NativeTelescopeData data) {
        engine.alignment.native_alignment_plan;
        engine.alignment.raw_telescope_trajectory;
        engine.alignment.native_pointing_plan;
        engine.alignment.native_carriers;
        engine.pointing_offsets.arcsec;
        engine.telescope.tel_data = std::move(data);
    };

template <class Engine>
std::shared_ptr<const NativePointingPlan>
build_native_pointing_plan_candidate(Engine &engine) {
    const auto &alignment = engine.alignment.native_alignment_plan;
    const auto &raw = engine.alignment.raw_telescope_trajectory;
    if (!alignment || !raw) {
        throw std::logic_error(
            "native pointing requires exact alignment and raw telescope handles");
    }
    const auto common = engine.telescope.tel_data.find("TelTime");
    if (common == engine.telescope.tel_data.end() ||
        common->second.size() < 2) {
        throw std::logic_error(
            "native pointing requires common telescope support bounds");
    }
    Eigen::VectorXd support(2);
    support << common->second(0),
        common->second(common->second.size() - 1);
    NativePointingOffsetModel offset_model{
        engine.pointing_offsets.arcsec, std::move(support)};

    std::vector<NativeNetworkPointing> networks;
    networks.reserve(alignment->networks().size());
    for (const auto &network : alignment->networks()) {
        Eigen::VectorXd times =
            network.reconstructed_times_unix_sec();
        auto telescope = evaluate_raw_telescope_trajectory_at(*raw, times);
        auto isolated_telescope = engine.telescope;
        isolated_telescope.tel_data = std::move(telescope);
        isolated_telescope.calc_tan_pointing();
        networks.emplace_back(
            network.network_id(), network.first_native_row(),
            std::move(times), std::move(isolated_telescope.tel_data),
            offset_model.evaluate_at(
                network.reconstructed_times_unix_sec()));
    }
    return std::make_shared<const NativePointingPlan>(
        alignment, raw, std::move(networks));
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

    if constexpr (native_pointing_publication_capable_v<
                      std::remove_reference_t<decltype(engine)>>) {
        const bool has_alignment =
            static_cast<bool>(engine.alignment.native_alignment_plan);
        const bool has_raw =
            static_cast<bool>(engine.alignment.raw_telescope_trajectory);
        if (has_alignment != has_raw) {
            throw std::logic_error(
                "native pointing carrier inputs are partial");
        }
        std::shared_ptr<const NativePointingPlan> native_pointing;
        std::shared_ptr<const NativeObservationCarriers> native_carriers;
        if (has_alignment) {
            native_pointing = build_native_pointing_plan_candidate(engine);
            native_carriers =
                std::make_shared<const NativeObservationCarriers>(
                    engine.alignment.native_alignment_plan->scope(),
                    engine.alignment.native_alignment_plan,
                    native_pointing);
        }

        calculate_tangent_plane_pointing(engine, logger);
        interpolate_pointing_offsets(todproc, logger);
        engine.alignment.native_pointing_plan =
            std::move(native_pointing);
        engine.alignment.native_carriers = std::move(native_carriers);
    }
    else {
        calculate_tangent_plane_pointing(engine, logger);
        interpolate_pointing_offsets(todproc, logger);
    }
}

}  // namespace citlali::pipeline
