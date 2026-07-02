#pragma once

#include <citlali/core/pipeline/kids_metadata.h>
#include <citlali/core/pipeline/reduction_observation_access.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace citlali::pipeline {

template <class RawObs, class RawObsKidsMeta, class DateObs>
struct ReductionObservationContext {
    const RawObs &rawobs;
    RawObsKidsMeta rawobs_kids_meta;
    bool has_multiple_inputs;
    std::size_t observation_index;
    DateObs date_obs;
};

template <class KidsProc, class IOCoordinator, class DateObs, class Logger>
auto make_reduction_observation_context(
    KidsProc &kidsproc, const IOCoordinator &co,
    std::size_t observation_index, DateObs &&date_obs,
    const Logger &logger) {
    const auto &rawobs = reduction_observation_input_at(co, observation_index);
    auto rawobs_kids_meta =
        load_reduction_observation_kids_meta(kidsproc, rawobs, logger);

    using rawobs_t =
        std::remove_cv_t<std::remove_reference_t<decltype(rawobs)>>;
    using rawobs_kids_meta_t = decltype(rawobs_kids_meta);
    using date_obs_t = std::decay_t<DateObs>;

    return ReductionObservationContext<
        rawobs_t, rawobs_kids_meta_t, date_obs_t>{
        rawobs, std::move(rawobs_kids_meta),
        has_multiple_reduction_observations(co), observation_index,
        std::forward<DateObs>(date_obs)};
}

}  // namespace citlali::pipeline
