#pragma once

// Beammap APT table output helpers.

#include <citlali/core/engine/detail/beammap_apt_keys.h>

#include <Eigen/Core>

namespace beammap_apt_table_output_helpers {

template <class Calib, class Flag2>
Eigen::MatrixXd apt_table(Calib &calib,
                          const Flag2 &flag2) {
    Eigen::MatrixXd table(calib.n_dets, calib.apt_header_keys.size());

    Eigen::Index col = 0;
    for (const auto &key : calib.apt_header_keys) {
        if (!beammap_apt_keys::is_flag2(key)) {
            table.col(col) = calib.apt[key];
        }
        else {
            table.col(col) = flag2.template cast<double>();
        }
        ++col;
    }
    return table;
}

} // namespace beammap_apt_table_output_helpers
