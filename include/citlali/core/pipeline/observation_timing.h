#pragma once

#include <utility>

namespace citlali::pipeline {

template <class Engine>
void update_observation_exposure_time(Engine &engine) {
    auto t0 = engine.telescope.tel_data["TelTime"](0);
    auto tn = engine.telescope.tel_data["TelTime"](
        engine.telescope.tel_data["TelTime"].size() - 1);

    engine.omb.exposure_time = tn - t0;
    if (engine.run_coadd) {
        engine.cmb.exposure_time =
            engine.cmb.exposure_time + engine.omb.exposure_time;
    }
}

template <class Engine, class DateObs>
void append_observation_date(Engine &engine, DateObs &&date_obs) {
    engine.date_obs.push_back(std::forward<DateObs>(date_obs));
}

template <class Engine, class ConvertUnixToUtc>
auto date_obs_from_telescope_time(Engine &engine,
                                  ConvertUnixToUtc &&convert_unix_to_utc) {
    return convert_unix_to_utc(engine.telescope.tel_data["TelTime"](0));
}

}  // namespace citlali::pipeline
