# pragma once

auto calc_pointing(const double xd, const double yd, const std::map<std::string, Eigen::VectorXd>& tel_data,
                   const std::string pixel_axes) {

    // reference to elevation
    const auto& e = tel_data.at("TelElAct");

    // Precompute sin and cos of elevation for efficiency
    const auto cos_e = cos(e.array());
    const auto sin_e = sin(e.array());

    // Rotate offsets to sample elevation
    Eigen::VectorXd xd_rot = (cos_e * xd - sin_e * yd) + tel_data.at("pointing_offset_az_arcsec").array();
    Eigen::VectorXd yd_rot = (cos_e * yd + sin_e * xd) + tel_data.at("pointing_offset_alt_arcsec").array();

    Eigen::VectorXd x, y;

    if (pixel_axes == "radec") {
        // Rotate around -pa
        const auto& pa = tel_data.at("ActParAng");

        // ra
        x = (-xd_rot.array() * cos(pa.array()) + yd_rot.array() * sin(pa.array())) * ASEC_TO_RAD
            + tel_data.at("ra_tan").array();
        // dec
        y = (xd_rot.array() * sin(pa.array()) + yd_rot.array() * cos(pa.array())) * ASEC_TO_RAD
            + tel_data.at("dec_tan").array();

    } else if (pixel_axes == "altaz") {
        // azimuth
        x = xd_rot.array() * ASEC_TO_RAD + tel_data.at("az_tan").array();
        // altitude
        y = yd_rot.array() * ASEC_TO_RAD + tel_data.at("alt_tan").array();

    } else if (pixel_axes == "galactic") {
        // rotate around -(pa + ga)
        const auto& pa = tel_data.at("ActParAng");
        const auto& ga = tel_data.at("ActGalAng");
        const auto theta = pa.array() + ga.array();

        // galactic longitude (l)
        x = (-xd_rot.array() * cos(theta) + yd_rot.array() * sin(theta)) * ASEC_TO_RAD
            + tel_data.at("l_tan").array();
        // galactic latitude (b)
        y = (xd_rot.array() * sin(theta) + yd_rot.array() * cos(theta)) * ASEC_TO_RAD
            + tel_data.at("b_tan").array();

    } else {
        // handle unsupported coordinate systems
        throw std::runtime_error("Unsupported pixel_axes coordinate system: " + pixel_axes);
    }

    return std::make_pair(x, y);
}
