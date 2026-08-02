#pragma once

#include <citlali/core/pipeline/timestream_alignment_state.h>

#include <type_traits>
#include <utility>

namespace citlali::pipeline {

template <class Engine, class DateObs>
void append_observation_date(Engine &engine, DateObs &&date_obs) {
    engine.observation_dates.date_obs.push_back(
        std::forward<DateObs>(date_obs));
}

template <class Engine, class DateObs>
void append_reduction_observation_date(Engine &engine, DateObs &&date_obs) {
    append_observation_date(engine, std::forward<DateObs>(date_obs));
}

template <class Engine, class ConvertUnixToUtc>
auto date_obs_from_telescope_time(Engine &engine,
                                  ConvertUnixToUtc &&convert_unix_to_utc) {
    const auto &tel_time = engine.telescope.tel_data["TelTime"];
    if constexpr (has_governing_compatibility_axis_state<Engine>::value) {
        if (engine.alignment.grid.initialized) {
            return convert_unix_to_utc(
                governing_compatibility_start_value(
                    tel_time, engine.alignment));
        }
    }
    return convert_unix_to_utc(tel_time(0));
}

}  // namespace citlali::pipeline
