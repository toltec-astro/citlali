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

    // toltec array mounting angle
    std::map<int, double> install_ang = {
        {-1,-1},
        {0,pi/2},
        {1,-pi/2},
        {2,-pi/2},
        };

    // toltec detector orientation angles
    std::map<int, double> fgs = {
        {-1,-1},
        {0,0},
        {1,pi/4},
        {2,pi/2},
        {3,3*pi/4}
    };

    template<TCDataKind td_kind, class calib_type>
    indices_t calc_angle(TCData<td_kind, Eigen::MatrixXd> &in, calib_type &calib, bool sim_obs) {

        // number of data points
        Eigen::Index n_pts = in.scans.data.rows();

        // vectors of array, nw, and det indices
        Eigen::VectorXI array_indices, nw_indices, det_indices, fg_indices;

            array_indices = calib.apt["array"].template cast<Eigen::Index> ();
            nw_indices = calib.apt["nw"].template cast<Eigen::Index> ();
            det_indices = Eigen::VectorXI::LinSpaced(in.scans.data.cols(),0,in.scans.data.cols()-1);
            fg_indices = calib.apt["fg"].template cast<Eigen::Index> ();


            // rotation angle at array center
            auto rot_angle = in.tel_data.data["ActParAng"].array() + in.tel_data.data["TelElAct"].array() +
                             in.pointing_offsets_arcsec.data["alt"].array()*ASEC_TO_RAD;

            in.angle.data.resize(n_pts,calib.n_dets);

            for (Eigen::Index i=0; i<calib.n_dets; ++i) {
                if (fg_indices(i) != -1) {
                    auto angle = rot_angle + fgs[fg_indices(i)] + install_ang[array_indices(i)];
                    if (calib.run_hwpr) {
                        in.angle.data.col(i) = 2*in.hwpr_angle.data.array() - angle;
                    }
                    else {
                        in.angle.data.col(i) = angle;
                    }
                }
                else {
                    in.angle.data.col(i).setConstant(0);
                }
            }

            // set as chunk as demodulated
            in.status.demodulated = true;

        return indices_t(array_indices, nw_indices, det_indices, fg_indices);
    }
};

} // namespace timestream
