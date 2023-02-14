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

    // rotate altaz offsets by elevation angle and add pointing offsets
    Eigen::VectorXd rot_az_off = cos(tel_data["TelElAct"].array())*az_off
                     - sin(tel_data["TelElAct"].array())*el_off + pointing_offsets["az"];
    Eigen::VectorXd rot_alt_off = cos(tel_data["TelElAct"].array())*el_off
                      + sin(tel_data["TelElAct"].array())*az_off + pointing_offsets["alt"];

    // icrs map
    if (std::strcmp("icrs", pixel_axes.c_str()) == 0) {
        // get parallactic angle
        Eigen::ArrayXd par_ang = -tel_data["ActParAng"].array();

        // dec
        lat = (-rot_az_off.array()*sin(par_ang) + rot_alt_off.array()*cos(par_ang))*ASEC_TO_RAD
              + tel_data["lat_phys"].array();
        // ra
        lon = (-rot_az_off.array()*cos(par_ang) - rot_alt_off.array()*sin(par_ang))*ASEC_TO_RAD
              + tel_data["lon_phys"].array();
    }

    // altaz map
    else if (std::strcmp("altaz", pixel_axes.c_str()) == 0) {
        // alt
        lat = (rot_alt_off.array()*ASEC_TO_RAD) + tel_data["lat_phys"].array();
        // az
        lon = (rot_az_off.array()*ASEC_TO_RAD) + tel_data["lon_phys"].array();
    }

    return std::tuple<Eigen::VectorXd, Eigen::VectorXd>{lat,lon};
}

template <typename Derived>
auto phys_to_abs(Eigen::DenseBase<Derived>& lat, Eigen::DenseBase<Derived>& lon, const double cra, const double cdec) {

    // number of samples
    Eigen::Index n_pts = lat.size();

    // lat/lon = dec/ra = y/x (map axes)
    Eigen::VectorXd abs_lat(n_pts), abs_lon(n_pts);
    for (Eigen::Index i=0; i<n_pts; i++) {
        double rho = sqrt(pow(lat(i),2) + pow(lon(i),2));
        double c = atan(rho);
        if (c == 0.) {
            abs_lat(i) = lat(i);
            abs_lon(i) = lon(i);
        }
        else {
            double ccwhn0 = cos(c);
            double scwhn0 = sin(c);
            double ccdec = cos(cdec);
            double scdec = sin(cdec);
            double a1, a2;
            a1 = ccwhn0*scdec + lat(i)*scwhn0*ccdec/rho;
            abs_lat(i) = asin(a1);
            a2 = lon(i)*scwhn0/(rho*ccdec*ccwhn0 - lat(i)*scdec*scwhn0);
            abs_lon(i) = cra + atan(a2);
        }
    }
    return std::tuple<Eigen::VectorXd, Eigen::VectorXd>{abs_lat,abs_lon};
}

} // namespace engine_utils
