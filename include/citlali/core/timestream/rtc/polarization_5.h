# pragma once

#pragma once

#include <Eigen/Core>
#include <string>

#include <citlali/core/timestream/timestream.h>

namespace timestream {

class Polarization {
public:
    // hold outputs
    using indices_t = std::tuple<Eigen::VectorXI, Eigen::VectorXI, Eigen::VectorXI, Eigen::VectorXI>;

    // stokes parameters (either I or I, Q, and U)
    std::map<int,std::string> stokes_params;

    // loc or fg
    std::string grouping;

    template<TCDataKind td_kind, class calib_type>
    indices_t calc_angle(TCData<td_kind, Eigen::MatrixXd> &in, calib_type &calib) {

        // number of data points
        Eigen::Index n_pts = in.scans.data.rows();

        // vectors of array, nw, and det indices
        Eigen::VectorXI array_indices, nw_indices, det_indices, fg_indices;

        array_indices = calib.apt["array"].template cast<Eigen::Index> ();
        nw_indices = calib.apt["nw"].template cast<Eigen::Index> ();
        det_indices = Eigen::VectorXI::LinSpaced(in.scans.data.cols(),0,in.scans.data.cols()-1);
        fg_indices = calib.apt["fg"].template cast<Eigen::Index> ();

        // rotation angle at array center
        in.angle.data = in.tel_data.data["ActParAng"].array() + in.tel_data.data["TelElAct"].array() +
                         in.pointing_offsets_arcsec.data["alt"].array()*ASEC_TO_RAD;

        // set as chunk as demodulated
        in.status.demodulated = true;

        return indices_t(array_indices, nw_indices, det_indices, fg_indices);
    }
};
} // namespace timestream
