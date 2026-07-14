#pragma once

#include <citlali/core/pipeline/kids_metadata.h>
#include <citlali/core/pipeline/reduction_observation_access.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace citlali::pipeline {

template <class RawObs, class RawObsKidsMeta>
struct ReductionObservationContext {
    const RawObs &rawobs;
    RawObsKidsMeta rawobs_kids_meta;
    bool has_multiple_inputs;
    std::size_t observation_index;
};

template <class RawObs>
std::string reduction_observation_load_error(
    const RawObs &rawobs, std::size_t observation_index,
    const std::exception &error) {
    std::ostringstream message;
    message << "failed to load metadata for observation index "
            << observation_index << " name='" << rawobs.name()
            << "' telescope_file='" << rawobs.teldata().filepath()
            << "': " << error.what();
    return message.str();
}

template <class KidsProc, class IOCoordinator, class Logger>
auto make_reduction_observation_context(
    KidsProc &kidsproc, const IOCoordinator &co,
    std::size_t observation_index, const Logger &logger) {
    const auto &rawobs = reduction_observation_input_at(co, observation_index);
    auto rawobs_kids_meta = [&]() {
        try {
            return load_reduction_observation_kids_meta(
                kidsproc, rawobs, logger);
        } catch (const std::exception &error) {
            throw std::runtime_error(reduction_observation_load_error(
                rawobs, observation_index, error));
        }
    }();

    using rawobs_t =
        std::remove_cv_t<std::remove_reference_t<decltype(rawobs)>>;
    using rawobs_kids_meta_t = decltype(rawobs_kids_meta);

    return ReductionObservationContext<rawobs_t, rawobs_kids_meta_t>{
        rawobs, std::move(rawobs_kids_meta),
        has_multiple_reduction_observations(co), observation_index};
}

}  // namespace citlali::pipeline
