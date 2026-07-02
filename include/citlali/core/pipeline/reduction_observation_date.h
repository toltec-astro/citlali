#pragma once

#include <utility>

namespace citlali::pipeline {

template <class DateObsFactory, class Engine>
auto make_reduction_observation_date_obs(DateObsFactory &&date_obs_factory,
                                         Engine &engine) {
    return std::forward<DateObsFactory>(date_obs_factory)(engine);
}

}  // namespace citlali::pipeline
