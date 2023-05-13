#pragma once

#include <Eigen/Core>
#include <string>

#include <citlali/core/timestream/timestream.h>

namespace timestream {

class Polarization {
public:
    using indices_t = std::tuple<Eigen::VectorXI, Eigen::VectorXI, Eigen::VectorXI>;

    // stokes parameters
    std::map<int,std::string> stokes_params = {
        {0,"I"},
        {1,"Q"},
        {2,"U"}
    };

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
        // vectors of map, array, nw, and det indices
        Eigen::VectorXI map_indices, array_indices, nw_indices, det_indices;

        // copy rtcdata
        out = in;

        array_indices = calib.apt["array"].template cast<Eigen::Index> ();
        nw_indices = calib.apt["nw"].template cast<Eigen::Index> ();
        det_indices = Eigen::VectorXI::LinSpaced(out.scans.data.cols(),0,out.scans.data.cols()-1);

        // only run demodulation if not stokes I
        if (stokes_param != "I") {
            // number of samples
            Eigen::Index n_pts = in.scans.data.rows();

            Eigen::Index n_dets;
            if (!sim_obs) {
                n_dets = (calib.apt["loc"].array()!=-1).count();
            }
            else {
                n_dets = calib.n_dets;
            }

            // q and u in the detector frame
            Eigen::MatrixXd polarized_scans(n_pts, n_dets);

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
                    // if matched, add to polarized array
                    if (calib.apt["loc"](i)!=-1) {
                        polarized_scans.col(k) = in.scans.data.col(i);

                        fg(k) = calib.apt["fg"](i);

                        array_indices(k) = calib.apt["array"](i);
                        nw_indices(k) = calib.apt["nw"](i);
                        det_indices(k) = i;
                        k++;
                    }
                }
            }

            else {
                polarized_scans = in.scans.data;

                fg = calib.apt["fg"];

                array_indices = calib.apt["array"].template cast<Eigen::Index> ();
                nw_indices = calib.apt["nw"].template cast<Eigen::Index> ();
                det_indices = Eigen::VectorXI::LinSpaced(out.scans.data.cols(),0,out.scans.data.cols()-1);
            }

            // now loop through polarized detectors
            for (Eigen::Index i=0; i<n_dets; i++) {
                // detector angle = installation angle + det orientation
                auto ang = install_ang[array_indices(i)] + fgs[fg(i)];

                // rotate q and u by parallactic, elevation, and detector angle
                if (!calib.run_hwp) {
                    if (stokes_param == "Q") {
                        out.scans.data.col(i) = cos(2*(in.tel_data.data["ActParAng"].array() +
                                                       in.tel_data.data["TelElAct"].array() +
                                                       ang))*polarized_scans.col(i).array();
                    }
                    else if (stokes_param == "U") {
                        out.scans.data.col(i) = sin(2*(in.tel_data.data["ActParAng"].array() +
                                                       in.tel_data.data["TelElAct"].array() +
                                                       ang))*polarized_scans.col(i).array();
                    }
                }

                // check if hwp is requested
                else {
                    if (stokes_param == "Q") {
                        out.scans.data.col(i) = cos(4*in.hwp_angle.data.array() - 2*(in.tel_data.data["ActParAng"].array() +
                                                                                         in.tel_data.data["TelElAct"].array() +
                                                                                         ang))*polarized_scans.col(i).array();
                    }
                    else if (stokes_param == "U") {
                        out.scans.data.col(i) = sin(4*in.hwp_angle.data.array() - 2*(in.tel_data.data["ActParAng"].array() +
                                                                                         in.tel_data.data["TelElAct"].array() +
                                                                                         ang))*polarized_scans.col(i).array();
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
