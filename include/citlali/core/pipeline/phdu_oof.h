#pragma once

#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Core>

#include <citlali/core/pipeline/phdu_telescope_values.h>

namespace citlali::pipeline {

template <class MapBuffer, class Logger>
double phdu_oof_rms(const MapBuffer &mb, Eigen::Index map_index,
                    const std::string &redu_type,
                    const std::string &array_name,
                    const std::string &filepath, const Logger &logger) {
    double rms = 0.0;

    if (redu_type != "beammap" && std::isfinite(mb->median_err(map_index)) &&
        mb->median_err(map_index) > std::numeric_limits<double>::epsilon()) {
        rms = std::pow(mb->median_err(map_index), 0.5);
    }
    else if (redu_type != "beammap" &&
             std::isfinite(mb->median_err(map_index)) &&
             mb->median_err(map_index) < 0.0) {
        logger->warn("negative median_err for PHDU {} in {}; using OOF_RMS=0",
                     array_name, filepath);
    }

    return rms;
}

template <class FitsEntry, class Logger>
void add_phdu_oof_keys(FitsEntry &fits_entry,
                       const std::string &array_name,
                       const Logger &logger,
                       double rms,
                       const std::string &signal_unit,
                       double wavelength_m,
                       int instrument_id,
                       double m2x_microns,
                       double m2y_microns,
                       double m2z_microns) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    add_double_key("OOF_RMS", rms, "rms of map background (" + signal_unit +")");
    add_double_key("OOF_W", wavelength_m, "wavelength (m)");
    hdu.addKey("OOF_ID", instrument_id, "instrument id");
    add_double_key("OOF_T", 3.0, "taper (dB)");
    add_double_key("OOF_M2X", m2x_microns, "oof m2x (microns)");
    add_double_key("OOF_M2Y", m2y_microns, "oof m2y (microns)");
    add_double_key("OOF_M2Z", m2z_microns, "oof m2z (microns)");
    add_double_key("OOF_RO", 25., "outer diameter of the antenna (m)");
    add_double_key("OOF_RI", 1.65, "inner diameter of the antenna (m)");
}

}  // namespace citlali::pipeline
