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

}  // namespace citlali::pipeline
