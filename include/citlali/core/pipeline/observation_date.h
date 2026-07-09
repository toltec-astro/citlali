#pragma once

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
    return convert_unix_to_utc(engine.telescope.tel_data["TelTime"](0));
}

}  // namespace citlali::pipeline
