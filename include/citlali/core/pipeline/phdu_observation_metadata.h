#pragma once

#include <cmath>
#include <string>
#include <tuple>

#include <citlali/core/pipeline/phdu_telescope_values.h>

namespace citlali::pipeline {

template <class ArrayFwhm>
double mean_beam_fwhm_arcsec(const ArrayFwhm &array_fwhm) {
    return (std::get<0>(array_fwhm) + std::get<1>(array_fwhm)) / 2;
}

inline double gaussian_beam_area_sr(double fwhm_arcsec,
                                    double fwhm_to_std,
                                    double arcsec_to_rad,
                                    double pi_value) {
    return 2. * pi_value *
           std::pow(fwhm_arcsec * fwhm_to_std * arcsec_to_rad, 2);
}

inline double mjy_beam_to_jy_pixel_factor(double beam_area_sr,
                                          double pixel_size_rad) {
    return 1e-3 / beam_area_sr * std::pow(pixel_size_rad, 2);
}

template <class ArrayNameMap, class ArrayId>
std::string phdu_array_name(ArrayNameMap &array_name_map,
                            const ArrayId &array_id) {
    return array_name_map[array_id];
}

template <class Arrays, class Index>
auto phdu_array_id(const Arrays &arrays, Index i) {
    return arrays(i);
}

template <class FitsEntry, class Obsnums>
void add_phdu_obsnum_keys(FitsEntry &fits_entry, const Obsnums &obsnums) {
    auto &hdu = fits_entry.pfits->pHDU();
    for (decltype(obsnums.size()) j=0; j<obsnums.size(); ++j) {
        hdu.addKey("OBSNUM" + std::to_string(j), obsnums.at(j),
                   "Observation Number " + std::to_string(j));
    }
}

template <class FitsEntry, class Obsnums, class DateObs>
void add_phdu_date_obs_keys(FitsEntry &fits_entry, const Obsnums &obsnums,
                            const DateObs &date_obs) {
    auto &hdu = fits_entry.pfits->pHDU();
    if (obsnums.size() == 1) {
        hdu.addKey("DATEOBS0", date_obs.back(),
                   "Date and time of observation 0");
    }
    else {
        for (decltype(obsnums.size()) j=0; j<obsnums.size(); ++j) {
            hdu.addKey("DATEOBS" + std::to_string(j), date_obs[j],
                       "Date and time of observation " + std::to_string(j));
        }
    }
}

template <class FitsEntry>
void add_phdu_pipeline_identity_keys(
    FitsEntry &fits_entry, const std::string &source_name, bool run_hwpr,
    const std::string &array_name, const std::string &citlali_version,
    const std::string &kids_version, const std::string &tula_version,
    const std::string &project_id, const std::string &reduction_goal,
    const std::string &obs_goal, const std::string &tod_type,
    const std::string &map_grouping, const std::string &map_method) {
    auto &hdu = fits_entry.pfits->pHDU();
    hdu.addKey("SOURCE", source_name, "Source name");
    hdu.addKey("INSTRUME", "TolTEC", "Instrument");
    hdu.addKey("HWPR", run_hwpr, "HWPR installed");
    hdu.addKey("TELESCOP", "LMT", "Telescope");
    hdu.addKey("WAV", array_name, "Wavelength");
    hdu.addKey("PIPELINE", "CITLALI", "Redu pipeline");
    hdu.addKey("VERSION", citlali_version, "CITLALI_GIT_VERSION");
    hdu.addKey("KIDS", kids_version, "KIDSCPP_GIT_VERSION");
    hdu.addKey("TULA", tula_version, "TULA_GIT_VERSION");
    hdu.addKey("PROJID", project_id, "Project ID");
    hdu.addKey("GOAL", reduction_goal, "Reduction type");
    hdu.addKey("OBSGOAL", obs_goal, "Obs goal");
    hdu.addKey("TYPE", tod_type, "TOD Type");
    hdu.addKey("GROUPING", map_grouping, "Map grouping");
    hdu.addKey("METHOD", map_method, "Map method");
}

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

template <class FitsEntry>
void add_phdu_auxiliary_scalar_keys(FitsEntry &fits_entry,
                                    const std::string &signal_unit,
                                    double sample_rate_hz,
                                    int fruit_loop_iter) {
    auto &hdu = fits_entry.pfits->pHDU();
    hdu.addKey("BUNIT", signal_unit, "bunit");
    hdu.addKey("SAMPRATE", sample_rate_hz, "sample rate (Hz)");
    hdu.addKey("FRUITLOOPS_ITER", fruit_loop_iter,
               "Current fruit loops iteration");
}

template <class FitsEntry>
void add_phdu_apt_key(FitsEntry &fits_entry, const std::string &apt_name) {
    fits_entry.pfits->pHDU().addKey("APT", apt_name, "APT table used");
}

template <class FitsEntry, class ShapeValues, class Logger>
void add_phdu_jinc_shape_keys(FitsEntry &fits_entry,
                              const std::string &array_name,
                              const Logger &logger, double r_max,
                              const ShapeValues &shape_values) {
    add_phdu_double_key(fits_entry, array_name, logger, "JINC_R", r_max,
                        "Jinc filter R_max");
    add_phdu_double_key(fits_entry, array_name, logger, "JINC_A",
                        shape_values[0], "Jinc filter param a");
    add_phdu_double_key(fits_entry, array_name, logger, "JINC_B",
                        shape_values[1], "Jinc filter param b");
    add_phdu_double_key(fits_entry, array_name, logger, "JINC_C",
                        shape_values[2], "Jinc filter param c");
}

template <class FitsEntry, class HeaderValues, class Logger>
void add_phdu_telescope_header_keys(FitsEntry &fits_entry,
                                    const std::string &array_name,
                                    const Logger &logger,
                                    const HeaderValues &tel_header) {
    for (auto const& [key, val] : tel_header) {
        if (val.size() < 1 || !std::isfinite(val(0))) {
            logger->warn("skipping tel_header '{}' due to empty/non-finite value",
                         key);
            continue;
        }
        logger->debug("adding {}: {}", key, val);
        add_phdu_double_key(fits_entry, array_name, logger, key, val(0), key);
    }
}

}  // namespace citlali::pipeline
