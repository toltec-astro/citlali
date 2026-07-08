#pragma once

#include <string>

#include <Eigen/Core>

#include <citlali/core/config/beammap_config.h>
#include <citlali/core/config/runtime_config.h>
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
                             const citlali::config::BeammapIterationConfig
                                 &iteration_config,
                             const citlali::config::BeammapPhaseStrategyConfig
                                 &phase_config,
                             const citlali::config::BeammapReferenceConfig
                                 &reference_config) {
    auto &hdu = fits_entry.pfits->pHDU();
    add_phdu_double_key(fits_entry, array_name, logger,
                        "BEAMMAP.ITER_TOLERANCE", iteration_config.tolerance,
                        "Beammap iteration tolerance");
    add_phdu_double_key(fits_entry, array_name, logger,
                        "BEAMMAP.CONVERGENCE_RADIUS_ARCSEC",
                        iteration_config.convergence_radius_arcsec,
                        "Beammap convergence aperture radius (arcsec)");
    hdu.addKey("BEAMMAP.ITER_MAX", iteration_config.max_iterations,
               "Beammap max iterations");
    hdu.addKey("BEAMMAP.PHASE_SPLIT_ENABLED", phase_config.enabled,
               "Beammap locator/measurement phases enabled");
    hdu.addKey("BEAMMAP.LOCATOR_ITER", phase_config.locator_iter,
               "Beammap locator iteration");
    hdu.addKey("BEAMMAP.MEASUREMENT_START_ITER",
               phase_config.measurement_start_iter,
               "Beammap first measurement iteration");
    hdu.addKey("BEAMMAP.IS_DEROTATED", reference_config.derotate,
               "Beammap derotated");
}

template <class FitsEntry, class ReferenceValues, class Logger>
void add_phdu_beammap_reference(FitsEntry &fits_entry,
                                const std::string &array_name,
                                const Logger &logger,
                                bool subtract_reference,
                                const ReferenceValues &reference_values) {
    auto &hdu = fits_entry.pfits->pHDU();
    if (subtract_reference) {
        hdu.addKey("BEAMMAP.REF_DET_INDEX", reference_values.det_index,
                   "Beammap Reference det (rotation center)");
        add_phdu_double_key(fits_entry, array_name, logger,
                            "BEAMMAP.REF_X_T", reference_values.x_t,
                            "Az rotation center (arcsec)");
        add_phdu_double_key(fits_entry, array_name, logger,
                            "BEAMMAP.REF_Y_T", reference_values.y_t,
                            "Alt rotation center (arcsec)");
    }
    else {
        hdu.addKey("BEAMMAP.REF_DET_INDEX", -99,
                   "Beammap Reference det (rotation center)");
        hdu.addKey("BEAMMAP.REF_X_T", "N/A",
                   "Az rotation center (arcsec)");
        hdu.addKey("BEAMMAP.REF_Y_T", "N/A",
                   "Alt rotation center (arcsec)");
    }
}

template <class FitsEntry, class Logger, class FluxMap, class Calib>
void add_phdu_beammap_keys_if_needed(
    FitsEntry &fits_entry, const std::string &array_name,
    const Logger &logger, const std::string &redu_type,
    FluxMap &flux_mjy_beam, FluxMap &flux_mjy_sr,
    const citlali::config::BeammapIterationConfig &iteration_config,
    const citlali::config::BeammapPhaseStrategyConfig &phase_config,
    const citlali::config::BeammapReferenceConfig &reference_config,
    Calib &calib) {
    if (!citlali::config::is_beammap_reduction_type(redu_type)) {
        return;
    }

    add_phdu_beammap_source_flux(
        fits_entry, array_name, logger, flux_mjy_beam[array_name],
        flux_mjy_sr[array_name]);

    add_phdu_beammap_tuning(
        fits_entry, array_name, logger, iteration_config, phase_config,
        reference_config);

    BeammapReferenceHeaderValues reference_values;
    if (reference_config.subtract_reference_detector) {
        reference_values =
            beammap_reference_header_values(
                calib, static_cast<Eigen::Index>(
                    reference_config.reference_detector));
    }
    add_phdu_beammap_reference(
        fits_entry, array_name, logger,
        reference_config.subtract_reference_detector, reference_values);
}

}  // namespace citlali::pipeline
