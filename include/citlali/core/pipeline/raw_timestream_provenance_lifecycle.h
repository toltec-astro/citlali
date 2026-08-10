#pragma once

#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/raw_timestream_provenance.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_output_context.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <optional>
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
void complete_raw_timestream_observation_if_available(Engine &engine) {
    if constexpr (has_raw_timestream_plan_v<Engine>) {
        auto &plan = raw_timestream_plan(engine);
        if (plan.realized.execution_completed) {
            return;
        }
        const auto expectations =
            raw_observation_output_expectations<IsBeammap>(engine);
        const Eigen::Index scan_count =
            timestream_processing_enabled(engine)
                ? engine.telescope.scan_indices.cols()
                : 0;
        complete_raw_timestream_observation(
            plan, raw_realized_count(scan_count, "completed scan count"),
            raw_required_timestream_write_count(expectations));
    }
}

template <bool IsBeammap, class Engine>
std::optional<std::filesystem::path>
publish_completed_raw_timestream_provenance(Engine &engine) {
    if constexpr (has_raw_timestream_plan_v<Engine>) {
        auto &plan = raw_timestream_plan(engine);
        complete_raw_timestream_observation_if_available<IsBeammap>(engine);
        const auto path = raw_timestream_provenance_path(
            engine.output_paths.obsnum_dir_name);
        write_raw_timestream_provenance_file(
            engine.output_paths.obsnum_dir_name, plan);
        return path;
    }
    return std::nullopt;
}

}  // namespace citlali::pipeline
