#pragma once

#include <Eigen/Core>

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

}  // namespace citlali::pipeline
