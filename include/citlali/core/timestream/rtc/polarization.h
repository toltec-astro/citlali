#pragma once

#include <Eigen/Core>
#include <string>

#include <citlali/core/timestream/timestream.h>

namespace timestream {

class Polarization {
public:
    using indices_t = std::tuple<Eigen::VectorXI, Eigen::VectorXI, Eigen::VectorXI>;

    std::map<int,std::string> stokes_params = {
        {0,"I"},
        {1,"Q"},
        {2,"U"}
    };

    template<TCDataKind td_kind, class calib_type>
    indices_t demodulate_timestream(TCData<td_kind, Eigen::MatrixXd> &in,
                                    TCData<td_kind, Eigen::MatrixXd> &out,
                                    std::string stokes_param, std::string redu_type,
                                    calib_type &calib, bool sim_obs) {
        // vectors of map, array, nw, and det indices
        Eigen::VectorXI map_indices, array_indices, nw_indices, det_indices;

        // copy rtcdata
        out = in;

        // stokes I
        if (stokes_param == "I") {
            array_indices = calib.apt["array"].template cast<Eigen::Index> ();
            nw_indices = calib.apt["nw"].template cast<Eigen::Index> ();
            det_indices = Eigen::VectorXI::LinSpaced(out.scans.data.cols(),0,out.scans.data.cols()-1);
        }

        else {
            // number of samples
            Eigen::Index n_pts = in.scans.data.rows();
            // number of detectors in both q and u in the detector frame
            Eigen::Index n_dets;
            if (!sim_obs) {
                Eigen::Index n_dets = (calib.apt["loc"].array()!=-1).count()/2;
            }
            else {
                Eigen::Index n_dets = (calib.apt["fg"].array() == 0).count() + (calib.apt["fg"].array() == 1).count();
            }
            SPDLOG_INFO("pol ndets {}", n_dets);

            // q and u in the detector frame
            Eigen::MatrixXd polarized_scans_det(n_pts, n_dets);

            // frequency group
            Eigen::VectorXd fg(n_dets);
            // resize index vectors
            array_indices.resize(n_dets);
            nw_indices.resize(n_dets);
            det_indices.resize(n_dets);

            // resize output scans
            out.scans.data.resize(n_pts, n_dets);

            // copy input matrices
            out.tel_data.data = in.tel_data.data;
            out.scan_indices.data = in.scan_indices.data;
            out.index.data = in.index.data;

            out.pointing_offsets_arcsec.data = in.pointing_offsets_arcsec.data;

            out.fcf = in.fcf;

            // loop through all detectors
            if (!sim_obs) {
                Eigen::Index k = 0;
                for (Eigen::Index i=0; i<calib.n_dets; i++) {
                    if (calib.apt["loc"](i)!=-1 && calib.apt["ori"](i)==0) {
                        for (Eigen::Index j=0; j<calib.n_dets; j++) {
                            if (calib.apt["loc"](i)==calib.apt["loc"](j) && calib.apt["ori"](j)!=0) {
                                polarized_scans_det.col(k) = in.scans.data.col(j) - in.scans.data.col(i);
                                fg(k) = calib.apt["fg"](i);

                                array_indices(k) = calib.apt["array"](i);
                                nw_indices(k) = calib.apt["nw"](i);
                                det_indices(k) = i;
                                k++;
                            }
                        }
                    }
                }
            }
            else {
                Eigen::Index j=0;
                for (Eigen::Index i=0; i<calib.n_dets-1; i=i+2) {
                    polarized_scans_det.col(j) = in.scans.data.col(i) - in.scans.data.col(i+1);
                    fg(j) = calib.apt["fg"](i);

                    array_indices(j) = calib.apt["array"](i);
                    nw_indices(j) = calib.apt["nw"](i);
                    det_indices(j) = i;

                    j++;
                }
            }

            Eigen::VectorXd q(n_pts), u(n_pts);

            // now loop through polarized detector pairs
            for (Eigen::Index i=0; i<n_dets; i++) {
                // rotate q and u by parallactic angle and elevation for q pairs
                if (fg(i)==0 || fg(i)==2) {
                    q = cos(2*(in.tel_data.data["ActParAng"].array() + in.tel_data.data["TelElAct"].array()))*polarized_scans_det.col(i).array();
                    u = sin(2*(in.tel_data.data["ActParAng"].array() + in.tel_data.data["TelElAct"].array()))*polarized_scans_det.col(i).array();
                }

                // rotate q and u by parallactic angle and elevation for u pairs
                else if (fg(i)==1 || fg(i)==3) {
                    q = -sin(2*(in.tel_data.data["ActParAng"].array() + in.tel_data.data["TelElAct"].array()))*polarized_scans_det.col(i).array();
                    u = cos(2*(in.tel_data.data["ActParAng"].array() + in.tel_data.data["TelElAct"].array()))*polarized_scans_det.col(i).array();
                }

                // check if hwp is requested
                if (calib.run_hwp) {
                    if (stokes_param == "Q") {
                        out.scans.data.col(i) = q.array()*cos(4*in.hwp_angle.data.array()) + u.array()*sin(4*in.hwp_angle.data.array());
                    }
                    else if (stokes_param == "U") {
                        out.scans.data.col(i) = (q.array()*sin(4*in.hwp_angle.data.array()) - u.array()*cos(4*in.hwp_angle.data.array()));
                    }
                }

                else {
                    if (stokes_param == "Q") {
                        out.scans.data.col(i) = q;
                    }
                    else if (stokes_param == "U") {
                        // not applying mirror flip
                        out.scans.data.col(i) = u;
                    }
                }
            }

            // set as chunk as demodulated
            out.demodulated = true;
        }

        return indices_t(array_indices, nw_indices, det_indices);
    }
};

} // namespace timestream
