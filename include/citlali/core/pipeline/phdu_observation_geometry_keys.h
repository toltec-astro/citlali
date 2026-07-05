#pragma once

// Included by phdu_observation_metadata.h inside namespace citlali::pipeline.

template <class FitsEntry, class Logger>
void add_phdu_map_geometry_keys(
    FitsEntry &fits_entry, const std::string &array_name,
    const Logger &logger, double exposure_time,
    const std::string &pixel_axes, double source_ra, double source_dec,
    double mean_el_deg, double mean_az_deg, double mean_pa_deg) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    add_double_key("EXPTIME", exposure_time, "Exposure time (sec)");
    hdu.addKey("RADESYS", pixel_axes, "Coord Reference Frame");
    add_double_key("SRC_RA", source_ra, "Source RA (radians)");
    add_double_key("SRC_DEC", source_dec, "Source Dec (radians)");
    add_double_key("TAN_RA", source_ra, "Map Tangent Point RA (radians)");
    add_double_key("TAN_DEC", source_dec,
                   "Map Tangent Point Dec (radians)");
    add_double_key("MEAN_EL", mean_el_deg, "Mean Elevation (deg)");
    add_double_key("MEAN_AZ", mean_az_deg, "Mean Azimuth (deg)");
    add_double_key("MEAN_PA", mean_pa_deg,
                   "Mean Parallactic angle (deg)");
}

template <class FitsEntry, class ArrayFwhm, class Logger>
void add_phdu_beam_geometry_keys(
    FitsEntry &fits_entry, const std::string &array_name,
    const Logger &logger, const ArrayFwhm &array_fwhm,
    double position_angle_rad, double rad_to_deg,
    double pa_quadrature_offset_rad) {
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    if (std::get<0>(array_fwhm) >= std::get<1>(array_fwhm)) {
        add_double_key("BMAJ", std::get<0>(array_fwhm), "beammaj (arcsec)");
        add_double_key("BMIN", std::get<1>(array_fwhm), "beammin (arcsec)");
        add_double_key("BPA", position_angle_rad*rad_to_deg, "beampa (deg)");
    }
    else {
        add_double_key("BMAJ", std::get<1>(array_fwhm), "beammaj (arcsec)");
        add_double_key("BMIN", std::get<0>(array_fwhm), "beammin (arcsec)");
        add_double_key("BPA",
                       (position_angle_rad + pa_quadrature_offset_rad)*
                           rad_to_deg,
                       "beampa (deg)");
    }
}

