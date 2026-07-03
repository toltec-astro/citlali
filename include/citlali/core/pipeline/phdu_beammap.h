#pragma once

#include <string>

#include <Eigen/Core>

#include <citlali/core/pipeline/phdu_telescope_values.h>

namespace citlali::pipeline {

struct BeammapReferenceHeaderValues {
    int det_index = -99;
    double x_t = -99.0;
    double y_t = -99.0;
};

template <class Calib>
BeammapReferenceHeaderValues beammap_reference_header_values(
    Calib &calib, Eigen::Index fallback_reference_det) {
    BeammapReferenceHeaderValues values;
    values.det_index = static_cast<int>(fallback_reference_det);

    if (calib.apt_meta["reference_det"]) {
        values.det_index = calib.apt_meta["reference_det"].template as<int>();
    }
    if (calib.apt_meta["reference_x_t"]) {
        values.x_t = calib.apt_meta["reference_x_t"].template as<double>();
    }
    else if (values.det_index >= 0 &&
             values.det_index < calib.apt["x_t"].size()) {
        values.x_t = calib.apt["x_t"](values.det_index);
    }
    if (calib.apt_meta["reference_y_t"]) {
        values.y_t = calib.apt_meta["reference_y_t"].template as<double>();
    }
    else if (values.det_index >= 0 &&
             values.det_index < calib.apt["y_t"].size()) {
        values.y_t = calib.apt["y_t"](values.det_index);
    }

    return values;
}

template <class FitsEntry, class Logger>
void add_phdu_beammap_source_flux(FitsEntry &fits_entry,
                                  const std::string &array_name,
                                  const Logger &logger,
                                  double flux_mjy_beam,
                                  double flux_mjy_sr) {
    add_phdu_double_key(fits_entry, array_name, logger,
                        "HEADER.SOURCE.FLUX_MJYPERBEAM", flux_mjy_beam,
                        "Source flux (mJy/beam)");
    add_phdu_double_key(fits_entry, array_name, logger,
                        "HEADER.SOURCE.FLUX_MJYPERSR", flux_mjy_sr,
                        "Source flux (MJy/sr)");
}

template <class FitsEntry, class Logger>
void add_phdu_beammap_tuning(FitsEntry &fits_entry,
                             const std::string &array_name,
                             const Logger &logger,
                             double iter_tolerance,
                             double convergence_radius_arcsec,
                             int iter_max,
                             bool phase_split_enabled,
                             int locator_iter,
                             int measurement_start_iter,
                             bool is_derotated) {
    auto &hdu = fits_entry.pfits->pHDU();
    add_phdu_double_key(fits_entry, array_name, logger,
                        "BEAMMAP.ITER_TOLERANCE", iter_tolerance,
                        "Beammap iteration tolerance");
    add_phdu_double_key(fits_entry, array_name, logger,
                        "BEAMMAP.CONVERGENCE_RADIUS_ARCSEC",
                        convergence_radius_arcsec,
                        "Beammap convergence aperture radius (arcsec)");
    hdu.addKey("BEAMMAP.ITER_MAX", iter_max, "Beammap max iterations");
    hdu.addKey("BEAMMAP.PHASE_SPLIT_ENABLED", phase_split_enabled,
               "Beammap locator/measurement phases enabled");
    hdu.addKey("BEAMMAP.LOCATOR_ITER", locator_iter,
               "Beammap locator iteration");
    hdu.addKey("BEAMMAP.MEASUREMENT_START_ITER", measurement_start_iter,
               "Beammap first measurement iteration");
    hdu.addKey("BEAMMAP.IS_DEROTATED", is_derotated, "Beammap derotated");
}

}  // namespace citlali::pipeline
