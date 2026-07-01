#pragma once

#include <citlali/core/pipeline/observation_date.h>
#include <citlali/core/utils/utils.h>

namespace citlali::cli {

template <class Engine>
auto date_obs_from_engine_telescope_time(Engine &engine) {
    return citlali::pipeline::date_obs_from_telescope_time(
        engine, [](double unix_time) {
            return engine_utils::unix_to_utc(unix_time);
        });
}

}  // namespace citlali::cli
