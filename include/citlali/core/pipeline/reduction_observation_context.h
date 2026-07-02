#pragma once

#include <cstddef>

namespace citlali::pipeline {

template <class RawObs, class RawObsKidsMeta, class DateObs>
struct ReductionObservationContext {
    const RawObs &rawobs;
    RawObsKidsMeta rawobs_kids_meta;
    bool has_multiple_inputs;
    std::size_t observation_index;
    DateObs date_obs;
};

}  // namespace citlali::pipeline
