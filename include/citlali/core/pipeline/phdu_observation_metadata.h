#pragma once

#include <string>

namespace citlali::pipeline {

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

}  // namespace citlali::pipeline
