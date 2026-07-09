#pragma once

#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct ObservationDateState {
    std::vector<std::string> date_obs;
};

template <class DateObs>
void append_observation_date(ObservationDateState &state, DateObs &&date_obs) {
    state.date_obs.push_back(std::forward<DateObs>(date_obs));
}

inline void clear_observation_dates(ObservationDateState &state) {
    state.date_obs.clear();
}

}  // namespace citlali::pipeline
