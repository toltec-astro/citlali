#pragma once

#include <Eigen/Core>

#include <string>
#include <map>

#include <tula/logging.h>

#include <citlali/core/utils/constants.h>
#include <citlali/core/timestream/timestream.h>

namespace timestream {

class Polarization {
public:

    using indices_tuple_t = std::tuple<Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>, Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>>;

    // stokes params
    std::map<std::string, int> stokes_params {
        {"I",0},
        {"Q",1},
        {"U",2},
    };

    template <typename DerivedA, typename DerivedB>
    void derotate_detector(Eigen::DenseBase<DerivedA> &q0, Eigen::DenseBase<DerivedB> &u0, Eigen::Index det,
                           TCData<TCDataKind::RTC, Eigen::MatrixXd> &out, Eigen::Index nsamples, Eigen::Index ndet,
                           Eigen::Index di, double y_t, bool run_hwp) {

        // current detector's elevation
        auto lat = y_t*RAD_ASEC + out.tel_meta_data.data["TelElDes"].array();

        // rotate by elevation and flip
        /*auto qs1 = q0.derived().array()*cos(-2*out.tel_meta_data.data["TelElDes"].array()) -
                   u0.derived().array()*sin(-2*out.tel_meta_data.data["TelElDes"].array());

        auto us1 = -q0.derived().array()*sin(-2*out.tel_meta_data.data["TelElDes"].array()) -
                   u0.derived().array()*cos(-2*out.tel_meta_data.data["TelElDes"].array());

        */

        // rotate by detector elevation and flip
        auto qs1 = q0.derived().array()*cos(-2*lat.array()) - u0.derived().array()*sin(-2*lat.array());
        auto us1 = -q0.derived().array()*sin(-2*lat.array()) - u0.derived().array()*cos(-2*lat.array());

        if (run_hwp) {
            // rotate by hwp signal
            auto qs = qs1.array()*cos(4*out.hwp.data.array()) + us1.array()*sin(4*out.hwp.data.array());
            auto us = qs1.array()*sin(4*out.hwp.data.array()) - us1.array()*cos(4*out.hwp.data.array());

            out.scans.data.col(det) = qs;
            out.scans.data.col(det+ndet) = us;
        }

        else {
            out.scans.data.col(det) = qs1;
            out.scans.data.col(det+ndet) = us1;
        }
    }

    template <class Engine>
    indices_tuple_t create_rtc(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, TCData<TCDataKind::RTC, Eigen::MatrixXd> &out, std::string sp,
                                   Engine engine) {

        // copy scan and telescope metadata
        out.scan_indices.data = in.scan_indices.data;
        out.index.data = in.index.data;
        out.tel_meta_data.data = in.tel_meta_data.data;

        Eigen::Index nsamples = in.scans.data.rows();

        // vectors for map and detector indices
        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> map_index_vector, det_index_vector;

        // ndetectors for stokes directions
        Eigen::Index ndet;

        if (sp == "I") {
            SPDLOG_INFO("creating I timestream");
            map_index_vector = engine->calib_data["array"].template cast<Eigen::Index> ();
            det_index_vector.resize(engine->ndet);

            out = in;

            for (Eigen::Index i=0;i<engine->ndet;i++) {
                if (engine->run_polarization) {
                    if (map_index_vector(i) == 1) {
                        map_index_vector(i) = 3;
                    }
                    else if (map_index_vector(i) == 2) {
                        map_index_vector(i) = 6;
                    }
                }
                det_index_vector(i) = i;
            }
        }

        else {
            Eigen::MatrixXd data;
            Eigen::Index ori;
            auto pa2 = in.tel_meta_data.data["ParAng"].array() - pi;

            if (sp == "Q") {
                SPDLOG_INFO("creating Q timestream");
                ori = 0;
                ndet = (engine->calib_data["fg"].array() == ori).count();
            }

            else if (sp == "U") {
                SPDLOG_INFO("creating U timestream");
                ori = 1;
                ndet = (engine->calib_data["fg"].array() == ori).count();
            }

            data.resize(nsamples, ndet);

            map_index_vector.resize(2*ndet);
            det_index_vector.resize(2*ndet);

            Eigen::Index j = 0;
            for (Eigen::Index i=0;i<in.scans.data.cols();i++) {
                if (engine->calib_data["fg"](i) == ori) {
                    if (sp == "Q") {
                        data.col(j) = in.scans.data.col(i+1) - in.scans.data.col(i);
                    }
                    else if (sp == "U") {
                        data.col(j) = in.scans.data.col(i) - in.scans.data.col(i+1);
                    }
                    map_index_vector(j) = engine->calib_data["array"](i);
                    map_index_vector(j+ndet) = engine->calib_data["array"](i);
                    det_index_vector(j) = i;
                    det_index_vector(j + ndet) = i;

                    j++;
                }
            }

            // resize scans and flags
            out.scans.data.resize(nsamples,2*ndet);
            out.flags.data.setOnes(nsamples,2*ndet);

            // get hwp data if enabled
            if (engine->run_hwp) {
                out.hwp.data = in.hwp.data;
            }

            // loop through detectors and derotate
            for (Eigen::Index i=0; i<ndet; i++) {
                // detector index
                Eigen::Index di = det_index_vector(i);
                // rotate by PA
                if (sp == "Q") {
                    auto qs0 = cos(2*pa2.array())*data.col(i).array();
                    auto us0 = sin(2*pa2.array())*data.col(i).array();

                    derotate_detector(qs0, us0, i, out, nsamples, ndet, di, engine->calib_data["y_t"](di), engine->run_hwp);
                }

                else if (sp == "U") {
                    auto qs0 = -sin(2*pa2.array())*data.col(i).array();
                    auto us0 = cos(2*pa2.array())*data.col(i).array();

                    derotate_detector(qs0, us0, i, out, nsamples, ndet, di, engine->calib_data["y_t"](di), engine->run_hwp);
                }
            }

            for (Eigen::Index i=0; i<map_index_vector.size(); i++) {
                if (map_index_vector(i) == 0) {
                    if (i < ndet) {
                        map_index_vector(i) = 1;
                    }
                    else {
                        map_index_vector(i) = 2;
                    }
                }
                else if (map_index_vector(i) == 1) {
                    if (i < ndet) {
                        map_index_vector(i) = 4;
                    }
                    else {
                        map_index_vector(i) = 5;
                    }
                }
                else if (map_index_vector(i) == 2) {
                    if (i < ndet) {
                        map_index_vector(i) = 7;
                    }
                    else {
                        map_index_vector(i) = 8;
                    }
                }
            }
        }

        return indices_tuple_t(map_index_vector, det_index_vector);
    }
};

} // namespace timestream
