#pragma once

#include <map>

#include <citlali/core/utils/constants.h>

namespace engine_utils {

template <typename tel_data_t>
auto calc_det_pointing(tel_data_t &tel_data, const double az_off, const double el_off,
                                const std::string pixel_axes,
                                std::map<std::string,double> &pointing_offsets) {

    // rows, cols pointing vectors
    Eigen::VectorXd lat, lon;

    // rotate altaz offsets by elevation angle
    auto rot_az_off = cos(tel_data["TelElAct"].array())*az_off
                     - sin(tel_data["TelElAct"].array())*el_off + pointing_offsets["az"];
    auto rot_altoff = cos(tel_data["TelElAct"].array())*el_off
                      + sin(tel_data["TelElAct"].array())*az_off + pointing_offsets["alt"];

    // icrs map
    if (std::strcmp("icrs", pixel_axes.c_str()) == 0) {

        auto par_ang = -tel_data["ParAng"].array();

        lat = (-rot_az_off*sin(par_ang) + rot_altoff*cos(par_ang))*ASEC_TO_RAD
              + tel_data["lat_phys"].array();

        lon = (-rot_az_off*cos(par_ang) - rot_altoff*sin(par_ang))*ASEC_TO_RAD
              + tel_data["lon_phys"].array();
    }

    // altaz map
    else if (std::strcmp("altaz", pixel_axes.c_str()) == 0) {
        lat = (rot_altoff*ASEC_TO_RAD) + tel_data["lat_phys"].array();
        lon = (rot_az_off*ASEC_TO_RAD) + tel_data["lon_phys"].array();
    }

    return std::tuple<Eigen::VectorXd, Eigen::VectorXd>{lat,lon};
}

} // namespace engine_utils
