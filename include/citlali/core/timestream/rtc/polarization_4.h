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

        Eigen::Index n_pts = in.scans.data.rows();

        // vectors of array, nw, and det indices
        Eigen::VectorXI array_indices, nw_indices, det_indices, fg_indices;

        // copy input rtcdata
        out = in;

        if (stokes_params.size()==1) {
            array_indices = calib.apt["array"].template cast<Eigen::Index> ();
            nw_indices = calib.apt["nw"].template cast<Eigen::Index> ();
            det_indices = Eigen::VectorXI::LinSpaced(out.scans.data.cols(),0,out.scans.data.cols()-1);
            fg_indices = calib.apt["fg"].template cast<Eigen::Index> ();
        }

        else {
            Eigen::Index n_dets;
            if (!sim_obs) {
                n_dets = (calib.apt["fg"].array()!=-1).count();
            }
            else {
                n_dets = calib.n_dets;
            }

            // resize index vectors
            array_indices.resize(n_dets);
            nw_indices.resize(n_dets);
            det_indices.resize(n_dets);
            fg_indices.resize(n_dets);

            // resize output scans
            out.scans.data.resize(n_pts, n_dets);

            // loop through all detectors
            if (!sim_obs) {
                Eigen::Index k = 0;
                for (Eigen::Index i=0; i<calib.n_dets; i++) {
                    // if matched, add to out scans
                    if (calib.apt["fg"](i)!=-1) {
                        out.scans.data.col(k) = in.scans.data.col(i);

                        array_indices(k) = calib.apt["array"](i);
                        nw_indices(k) = calib.apt["nw"](i);
                        det_indices(k) = i;
                        fg_indices(k) = calib.apt["fg"](i);
                        k++;
                    }
                }
            }
            else {
                array_indices = calib.apt["array"].template cast<Eigen::Index> ();
                nw_indices = calib.apt["nw"].template cast<Eigen::Index> ();
                det_indices = Eigen::VectorXI::LinSpaced(n_dets,0,n_dets-1);
                fg_indices = calib.apt["fg"].template cast<Eigen::Index> ();
            }

            // resize angle matrix
            out.angle.data.resize(n_pts, n_dets);

            // rotation angle
            auto rot_angle = in.tel_data.data["ActParAng"].array() + in.tel_data.data["TelElAct"].array() +
                             in.pointing_offsets_arcsec.data["alt"].array()*ASEC_TO_RAD;

            // now loop through polarized detectors
            for (Eigen::Index i=0; i<n_dets; i++) {
                // detector angle = installation angle + det orientation + rotation angle
                auto angle = rot_angle + install_ang[array_indices(i)] + fgs[fg_indices(i)];

                // if there is no hwpr
                if (calib.run_hwpr==false) {
                    out.angle.data.col(i) = angle;
                }
                else {
                    out.angle.data.col(i) = 2*in.hwpr_angle.data.array() - angle;
                }
            }

            // set as chunk as demodulated
            out.status.demodulated = true;
        }

        return indices_t(array_indices, nw_indices, det_indices, fg_indices);
    }
};

} // namespace timestream