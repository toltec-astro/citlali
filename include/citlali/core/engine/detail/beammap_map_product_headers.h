#pragma once

// Beammap FITS map product header helpers.

namespace beammap_map_product_headers {

template <class HduPtr, class Calib, class Flag2Vector>
void add_detector_header_keys(HduPtr hdu,
                              Calib &calib,
                              const Flag2Vector &flag2,
                              Eigen::Index detector_index) {
    for (auto const &key: calib.apt_header_keys) {
        if (key != "flag2") {
            try {
                hdu->addKey("BEAMMAP." + key, calib.apt[key](detector_index),
                            key + " (" + calib.apt_header_units[key] + ")");
            }
            catch (...) {
                hdu->addKey("BEAMMAP." + key, 0.0,
                            key + " (" + calib.apt_header_units[key] + ")");
            }
        }
        else {
            hdu->addKey("BEAMMAP." + key, flag2(detector_index),
                        key + " (" + calib.apt_header_units[key] + ")");
        }
    }
}

} // namespace beammap_map_product_headers
