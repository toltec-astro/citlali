#pragma once

#include <map>
#include <string>

#include <Eigen/Core>

namespace citlali::pipeline {

struct PointingOffsetState {
    std::map<std::string, Eigen::VectorXd> arcsec;
    Eigen::ArrayXd modified_julian_date;
};

inline void clear_pointing_offsets(PointingOffsetState &state) {
    state.arcsec.clear();
    state.modified_julian_date.setZero(2);
}

}  // namespace citlali::pipeline
