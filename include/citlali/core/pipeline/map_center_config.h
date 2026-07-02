#pragma once

namespace citlali::pipeline {

inline double degrees_to_radians(double degrees) {
    constexpr double deg_to_rad = 0.017453292519943295769;
    return degrees * deg_to_rad;
}

template <class Engine>
bool has_map_center_override(const Engine &engine) {
    return engine.omb.crval_config[0] != 0 &&
           engine.omb.crval_config[1] != 0;
}

template <class Engine>
double map_center_ra_degrees(const Engine &engine) {
    return engine.omb.crval_config[0];
}

template <class Engine>
double map_center_dec_degrees(const Engine &engine) {
    return engine.omb.crval_config[1];
}

}  // namespace citlali::pipeline
