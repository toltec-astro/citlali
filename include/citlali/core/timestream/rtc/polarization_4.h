#pragma once

#include <Eigen/Core>
#include <string>

#include <citlali/core/timestream/timestream.h>

namespace timestream {

class Polarization {
public:
    using indices_t = std::tuple<Eigen::VectorXI, Eigen::VectorXI, Eigen::VectorXI, Eigen::VectorXI>;

    // stokes parameters
    std::map<int,std::string> stokes_params;

    // toltec array mounting angle
    std::map<int, double> install_ang = {
        {0,pi/2},
        {1,-pi/2},
        {2,-pi/2},
        };

    // toltec detector orientation angles
    std::map<int, double> fgs = {
        {0,0},
        {1,pi/4},
        {2,pi/2},
        {3,3*pi/4}
    };

    template<TCDataKind td_kind, class calib_type>
    indices_t demodulate_timestream(TCData<td_kind, Eigen::MatrixXd> &in,
                                    TCData<td_kind, Eigen::MatrixXd> &out,
                                    std::string stokes_param, std::string redu_type,
                                    calib_type &calib, bool sim_obs) {

        // vectors of array, nw, and det indices
        Eigen::VectorXI array_indices, nw_indices, det_indices, fg_indices;

        // copy input rtcdata
        out = in;

        // set up array indices
        array_indices = calib.apt["array"].template cast<Eigen::Index> ();
        // set up nw indices
        nw_indices = calib.apt["nw"].template cast<Eigen::Index> ();
        // set up detector indices
        det_indices = Eigen::VectorXI::LinSpaced(out.scans.data.cols(),0,out.scans.data.cols()-1);
        // set up fg indices
        fg_indices = calib.apt["fg"].template cast<Eigen::Index> ();

        // resize index vectors
        array_indices.resize(n_dets);
        nw_indices.resize(n_dets);
        det_indices.resize(n_dets);
        fg_indices.resize(n_dets);

        // resize angle matrix
        out.angle.data.resize(n_pts, n_dets);

        // copy input matrices
        out.tel_data.data = in.tel_data.data;
        out.scan_indices.data = in.scan_indices.data;
        out.index.data = in.index.data;

        // copy pointing offsets
        out.pointing_offsets_arcsec.data = in.pointing_offsets_arcsec.data;

        // now loop through polarized detectors
        for (Eigen::Index i=0; i<n_dets; i++) {
            // detector angle = installation angle + det orientation
            auto ang = install_ang[array_indices(i)];
            auto rot_angle = in.tel_data.data["ActParAng"].array() + in.tel_data.data["TelElAct"].array() + ang;

            if (calib.run_hwpr==false) {
                out.angle.data.col(i) = 2*rot_ang;
            }
            else {
                out.angle.data.col(i) = 4*in.hwp_angle.data.array() - 2*rot_angle;
            }
        }

        // set as chunk as demodulated
        out.demodulated = true;

        return indices_t(array_indices, nw_indices, det_indices, fg_indices);
    }
};

} // namespace timestream
