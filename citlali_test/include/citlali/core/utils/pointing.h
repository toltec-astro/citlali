#pragma once

#include <map>

#include <citlali/core/utils/constants.h>

namespace engine_utils {

using det_pointing_t = std::tuple<Eigen::VectorXd, Eigen::VectorXd>;

template <typename TD>
det_pointing_t get_det_pointing(TD &tel_meta_data, const double azoff, const double eloff,
                                const std::string map_type,
                                std::map<std::string,double> &pointing_offsets) {

    // rows, cols pointing vectors
    Eigen::VectorXd lat, lon;

    // rotate altaz offsets by elevation angle
    auto rot_azoff = cos(tel_meta_data["TelElDes"].array())*azoff
            - sin(tel_meta_data["TelElDes"].array())*eloff + pointing_offsets["az"];
    auto rot_altoff = cos(tel_meta_data["TelElDes"].array())*eloff
            + sin(tel_meta_data["TelElDes"].array())*azoff + pointing_offsets["alt"];

    // icrs map
    if (std::strcmp("icrs", map_type.c_str()) == 0) {

        auto pa2 = tel_meta_data["ParAng"].array() - pi;
        
        lon = (-rot_azoff*cos(pa2) - rot_altoff*sin(pa2))*ASEC_TO_RAD
            + tel_meta_data["TelLonPhys"].array();
        
        lat = (-rot_azoff*sin(pa2) + rot_altoff*cos(pa2))*ASEC_TO_RAD
            + tel_meta_data["TelLatPhys"].array();
    }

    // altaz map
    else if (std::strcmp("altaz", map_type.c_str()) == 0) {
        lat = (rot_altoff*ASEC_TO_RAD) + tel_meta_data["TelLatPhys"].array();
        lon = (rot_azoff*ASEC_TO_RAD) + tel_meta_data["TelLonPhys"].array();
    }

    return det_pointing_t {lat,lon};
}

} // namespace
