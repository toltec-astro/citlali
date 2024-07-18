# pragma once

#pragma once

#include <Eigen/Core>
#include <string>

#include <citlali/core/timestream/timestream.h>

namespace timestream {

class Polarization {
public:
    // stokes parameters (either I or I, Q, and U)
    std::map<int,std::string> stokes_params;

    // loc or fg
    std::string grouping;

    template<TCDataKind td_kind, class calib_type>
    void calc_angle(TCData<td_kind, Eigen::MatrixXd> &in, calib_type &calib) {

        // number of data points
        Eigen::Index n_pts = in.scans.data.rows();

        // rotation angle at array center
        in.angle.data = in.tel_data.data["ActParAng"].array() + in.tel_data.data["TelElAct"].array() +
                         in.pointing_offsets_arcsec.data["alt"].array()*ASEC_TO_RAD;

        // set as chunk as demodulated
        in.status.demodulated = true;
    }
};
} // namespace timestream
