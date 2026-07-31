#pragma once

// Beammap FITS map product header helpers.

#include <citlali/core/engine/detail/beammap_apt_keys.h>

#include <Eigen/Core>

#include <cmath>
#include <string>

namespace beammap_map_product_headers {

template <class HduPtr>
bool add_detector_header_value(HduPtr hdu, const std::string &key,
                               double value, const std::string &comment) {
    if (!std::isfinite(value)) {
        // Missing diagnostics remain NaN in the authoritative APT table.
        // Omission is portable across the older CCfits used on Unity.
        return false;
    }
    hdu->addKey(key, value, comment);
    return true;
}

template <class HduPtr, class Calib, class Flag2Vector>
void add_detector_header_keys(HduPtr hdu,
                              Calib &calib,
                              const Flag2Vector &flag2,
                              Eigen::Index detector_index) {
    for (auto const &key: calib.apt_header_keys) {
        const std::string fits_key = "BEAMMAP." + key;
        const std::string comment =
            key + " (" + calib.apt_header_units[key] + ")";
        if (!beammap_apt_keys::is_flag2(key)) {
            add_detector_header_value(
                hdu, fits_key, calib.apt[key](detector_index), comment);
        }
        else {
            add_detector_header_value(
                hdu, fits_key, flag2(detector_index), comment);
        }
    }
}

} // namespace beammap_map_product_headers
