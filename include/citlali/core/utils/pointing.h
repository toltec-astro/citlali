#pragma once

#include <map>

#include <citlali/core/utils/constants.h>

namespace engine_utils {

// get a single detector's pointing
template <typename tel_data_t, typename pointing_offset_t>
auto calc_det_pointing(tel_data_t &tel_data, double az_off, double el_off,
                       const std::string pixel_axes, pointing_offset_t &pointing_offsets,
                       const std::string map_grouping, bool apply_det_offsets = false) {

    // if making per detector maps, set offsets to zero
    if (map_grouping=="detector" && !apply_det_offsets) {
        az_off = 0;
        el_off = 0;
    }

    // rows, cols pointing vectors
    Eigen::VectorXd lat, lon;

    // elevation for rotation
    auto elev = tel_data["TelElAct"].array();

    // rotate altaz offsets by elevation angle and add pointing offsets
    Eigen::VectorXd rot_az_off = cos(elev)*az_off
                                 - sin(elev)*el_off + pointing_offsets["az"].array();
    Eigen::VectorXd rot_alt_off = cos(elev)*el_off
                                  + sin(elev)*az_off + pointing_offsets["alt"].array();

    // radec map
    if (pixel_axes=="radec") {
        // get parallactic angle
        auto& par_ang = tel_data["ActParAng"];

        // dec
        lat = (rot_az_off.array()*sin(par_ang.array()) + rot_alt_off.array()*cos(par_ang.array()))*ASEC_TO_RAD
              + tel_data["dec_phys"].array();
        // ra
        // scale RA offsets with absolute declination (TelDec), not tangent-plane dec_phys.
        Eigen::ArrayXd cos_dec;
        auto teldec_it = tel_data.find("TelDec");
        if (teldec_it != tel_data.end() &&
            teldec_it->second.size() == tel_data["ra_phys"].size()) {
            cos_dec = teldec_it->second.array().cos();
        }
        else {
            cos_dec = tel_data["dec_phys"].array().cos();
        }
        lon = ((-rot_az_off.array()*cos(par_ang.array()) + rot_alt_off.array()*sin(par_ang.array()))*cos_dec)*ASEC_TO_RAD
              + tel_data["ra_phys"].array();
    }

    // altaz map
    else if (pixel_axes=="altaz") {
        // alt
        lat = (rot_alt_off.array()*ASEC_TO_RAD) + tel_data["alt_phys"].array();
        // az
        lon = (rot_az_off.array()*ASEC_TO_RAD) + tel_data["az_phys"].array();
    }

    else if (pixel_axes=="galactic") {
        // get parallactic angle
        auto ang = tel_data["ActParAng"] + tel_data["ActGalAng"];

        // b
        lat = (rot_az_off.array()*sin(ang.array()) + rot_alt_off.array()*cos(ang.array()))*ASEC_TO_RAD
              + tel_data["b_phys"].array();
        // l
        lon = (-rot_az_off.array()*cos(ang.array()) + rot_alt_off.array()*sin(ang.array()))*ASEC_TO_RAD
              + tel_data["l_phys"].array();
    }

    return std::tuple<Eigen::VectorXd, Eigen::VectorXd>{lat,lon};
}

template <typename Derived>
auto calc_par_ang_from_coords(const double lat, const double lon, Eigen::DenseBase<Derived> &az, Eigen::DenseBase<Derived> &alt,
                              Eigen::DenseBase<Derived> &ra, Eigen::DenseBase<Derived> &dec) {

    auto cosha = (sin(alt.derived().array()) - sin(dec.derived().array())* sin(lat)) /
                 (cos(dec.derived().array())* cos(lat));

    auto sinha = (-sin(az.derived().array())* cos(alt.derived().array())/ cos(dec.derived().array()));

    Eigen::VectorXd par_ang(alt.size());

    for (Eigen::Index i=0; i<alt.size(); ++i) {
        par_ang(i) = atan2(sinha(i), (tan(lat)* cos(dec(i)) - sin(dec(i)) * cosha(i)));
    }

    return par_ang;
}

template <typename Derived>
auto tangent_to_abs(Eigen::DenseBase<Derived>& lat, Eigen::DenseBase<Derived>& lon, const double cra, const double cdec) {

    // number of samples
    Eigen::Index n_pts = lat.size();

    // lat/lon = dec/ra = y/x (map axes)
    Eigen::VectorXd abs_lat(n_pts), abs_lon(n_pts);
    for (Eigen::Index i=0; i<n_pts; ++i) {
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

// function to calculate the gnomonic projection for vectors
template <typename Derived>
void gnomonic_projection(const Eigen::DenseBase<Derived> &l, const Eigen::DenseBase<Derived> &b,
                         double l0, double b0, Eigen::DenseBase<Derived> &x, Eigen::DenseBase<Derived> &y) {

    // precompute cosines and sines
    Eigen::VectorXd cos_b = b.derived().array().cos();
    Eigen::VectorXd sin_b = b.derived().array().sin();
    double cos_b0 = std::cos(b0);
    double sin_b0 = std::sin(b0);

    // calculate angular distance c
    Eigen::VectorXd cos_c = sin_b.array() * sin_b0 + cos_b.array() * cos_b0 * (l.derived().array() - l0).cos();

    // avoid division by zero or near zero
    for (int i = 0; i < cos_c.size(); ++i) {
        if (std::abs(cos_c(i)) < std::numeric_limits<double>::epsilon()) {
            x(i) = 0;
            y(i) = 0;
        }
        else {
            x(i) = cos_b(i) * std::sin(l(i) - l0) / cos_c(i);
            y(i) = (cos_b0 * sin_b(i) - sin_b0 * cos_b(i) * std::cos(l(i) - l0)) / cos_c(i);
        }
    }
}

// function to convert equatorial coordinates (RA, Dec) to galactic coordinates (l, b)
static const void equatorial_to_galactic(const double ra, const double dec, double& l, double& b) {
    // Constants (all angles in radians)
    double ra_NGP = 192.859508*DEG_TO_RAD;   // Right Ascension of North Galactic Pole
    double dec_NGP = 27.128336*DEG_TO_RAD;   // Declination of North Galactic Pole
    double l_NCP = 122.931919*DEG_TO_RAD;    // Longitude of North Celestial Pole in Galactic coordinates

    // calculate b, the Galactic latitude
    double sin_b = std::sin(dec_NGP) * std::sin(dec) + std::cos(dec_NGP) * std::cos(dec) * std::cos(ra - ra_NGP);
    b = std::asin(sin_b);  // inverse sine to get the latitude

    // calculate l, the Galactic longitude
    double sin_l_ncp_minus_l = std::cos(dec) * std::sin(ra - ra_NGP) / std::cos(b);
    double cos_l_ncp_minus_l = (std::cos(dec_NGP) * std::sin(dec) - std::sin(dec_NGP) * std::cos(dec) * std::cos(ra - ra_NGP)) / std::cos(b);
    double l_ncp_minus_l = std::atan2(sin_l_ncp_minus_l, cos_l_ncp_minus_l);
    l = l_NCP - l_ncp_minus_l;

    // normalize l to be within the range [0, 2*pi)
    l = std::fmod(l, 2 * M_PI);
}


} // namespace engine_utils
