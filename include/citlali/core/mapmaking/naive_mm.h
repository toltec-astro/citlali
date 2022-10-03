#pragma once

#include <citlali/core/utils/pointing.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

namespace mapmaking {

template<class map_buffer_t, typename Derived, typename apt_t, typename pointing_offset_t>
void populate_maps_naive(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                         map_buffer_t &omb, Eigen::DenseBase<Derived> &map_indices, Eigen::DenseBase<Derived> &det_indices,
                         std::string &pixel_axes, std::string &redu_type, apt_t &apt,
                         pointing_offset_t &pointing_offsets_arcsec, double d_fsmp) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    for (Eigen::Index i=0; i<n_dets; i++) {
        double az_off = 0;
        double el_off = 0;

        if (redu_type!="beammap") {
            auto det_index = det_indices(i);
            az_off = apt["x_t"](det_index);
            el_off = apt["y_t"](det_index);
        }

        Eigen::Index map_index = map_indices(i);

        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off,
                                                          pixel_axes, pointing_offsets_arcsec);

        // get map buffer row and col indices for lat and lon vectors
        Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows)/2.;
        Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols)/2.;

        for (Eigen::Index j=0; j<n_pts; j++) {
            if (in.flags.data(j,i)) {
                Eigen::Index omb_ir = omb_irow(j);
                Eigen::Index omb_ic = omb_icol(j);
                if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic <omb.n_cols)) {

                    auto signal = in.scans.data(j,i)*in.weights.data(i);
                    omb.signal[map_index](omb_ir, omb_ic) += signal;

                    omb.weight[map_index](omb_ir, omb_ic) += in.weights.data(i);

                    if (!omb.kernel.empty()) {
                        auto kernel = in.kernel.data(j,i)*in.weights.data(i);
                        omb.kernel[map_index](omb_ir, omb_ic) += kernel;
                    }

                    omb.coverage[map_index](omb_ir, omb_ic) += 1./d_fsmp;
                }
            }
        }
    }

}
} // namespace mapmaking
