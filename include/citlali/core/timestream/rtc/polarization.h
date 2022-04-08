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

    // stokes params
    std::map<std::string, int> stokes_params {
        {"I",0},
        {"Q",1},
        {"U",2},
    };
    template <class Engine>
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> create_rtc(TCData<TCDataKind::RTC, Eigen::MatrixXd> &,
                                                            TCData<TCDataKind::RTC, Eigen::MatrixXd> &,
                                                            std::string, Engine);
};

template <class Engine>
std::tuple<Eigen::VectorXd, Eigen::VectorXd> Polarization::create_rtc(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in,
                                                                      TCData<TCDataKind::RTC, Eigen::MatrixXd> &out,
                                                                      std::string sp, Engine engine) {
    // generate rtc
    Eigen::Index nsamples = in.scans.data.rows();

    Eigen::VectorXd map_index_vector, det_index_vector;

    Eigen::Index ndet;

    if (sp == "I") {
        SPDLOG_INFO("creating I timestream");
        map_index_vector = engine->calib_data["array"];
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

        SPDLOG_INFO("map_index_vector {}", map_index_vector);
        SPDLOG_INFO("det_index_vector {}", det_index_vector);
    }

    else if (sp == "Q") {
        SPDLOG_INFO("creating Q timestream");
        ndet = (engine->calib_data["fg"].array() == 0).count();
        //engine->ndet = 2*ndet;
        Eigen::MatrixXd qr, kqr;
        qr.setZero(in.scans.data.rows(), ndet);
        kqr.setZero(in.scans.data.rows(), ndet);

        map_index_vector.resize(2*ndet);
        det_index_vector.resize(2*ndet);

        SPDLOG_INFO("ndet {}",ndet);

        Eigen::Index j = 0;
        for (Eigen::Index i=0;i<in.scans.data.cols();i++) {
            if (engine->calib_data["fg"](i) == 0) {
                qr.col(j) = in.scans.data.col(i+1) - in.scans.data.col(i);
                map_index_vector(j) = engine->calib_data["array"](i);
                map_index_vector(j+ndet) = engine->calib_data["array"](i);
                det_index_vector(j) = i;
                det_index_vector(j + ndet) = i;

                j++;
            }
        }

        auto pa2 = in.tel_meta_data.data["ParAng"].array() - pi;

        out.scans.data.resize(nsamples,2*ndet);

        for (Eigen::Index i=0;i<ndet;i++) {

            Eigen::Index di = det_index_vector(i);

            // don't use pointing here as that will rotate by azoff/eloff
            Eigen::VectorXd lat = (engine->calib_data["x_t"](di)*RAD_ASEC) + in.tel_meta_data.data["TelElDes"].array();

            // rotate by PA
            auto qs0 = cos(-2*pa2.array())*qr.col(i).array();
            auto us0 = sin(-2*pa2.array())*qr.col(i).array();

            // rotate by elevation and flip
            auto qs1 = qs0.array()*cos(2*in.tel_meta_data.data["TelElDes"].array()) -us0.array()*sin(2*in.tel_meta_data.data["TelElDes"].array());
            auto us1 = -qs0.array()*sin(2*in.tel_meta_data.data["TelElDes"].array()) - us0.array()*cos(2*in.tel_meta_data.data["TelElDes"].array());

            // rotate by elevation and flip
            //auto qs1 = qs0.array()*cos(2*lat.array()) - us0.array()*sin(2*lat.array());
            //auto us1 = - qs0.array()*sin(2*lat.array()) - us0.array()*cos(2*lat.array());

            if (engine->run_hwp) {
                // rotate by hwp signal
                auto qs = qs1.array()*cos(4*in.hwp.data.array()) + us1.array()*sin(4*in.hwp.data.array());
                auto us = qs1.array()*sin(4*in.hwp.data.array()) - us1.array()*cos(4*in.hwp.data.array());

                out.scans.data.col(i) = qs;
                out.scans.data.col(i+ndet) = us;
            }

            else {
                out.scans.data.col(i) = qs1;
                out.scans.data.col(i+ndet) = us1;
            }
        }

        out.flags.data.setOnes(out.scans.data.rows(),out.scans.data.cols());

        //map_index_vector.head(ndet).array() +=1;
        //map_index_vector.tail(ndet).array() +=2;

        SPDLOG_INFO("map_index_vector {}", map_index_vector);
        SPDLOG_INFO("det_index_vector {}", det_index_vector);

        bool done = false;
        for (Eigen::Index i=0;i<map_index_vector.size();i++) {
            if (map_index_vector(i)==0 && done==false) {
                if (i < ndet) {
                    map_index_vector(i) = 1;
                    done = true;
                }
                else {
                    map_index_vector(i) = 2;
                    done = true;
                }
            }
            if (map_index_vector(i)==1 && done==false) {
                if (i < ndet) {
                    map_index_vector(i) = 4;
                    done = true;
                }
                else {
                    map_index_vector(i) = 5;
                    done = true;
                }
            }
            if (map_index_vector(i)==2 && done==false) {
                if (i < ndet) {
                    map_index_vector(i) = 7;
                    done = true;
                }
                else {
                    map_index_vector(i) = 8;
                    done = true;
                }
            }
            done = false;

        }

        SPDLOG_INFO("map_index_vector {}", map_index_vector);
        SPDLOG_INFO("det_index_vector {}", det_index_vector);

    }

    else if (sp == "U") {
        SPDLOG_INFO("creating U timestream");
        ndet = (engine->calib_data["fg"].array() == 1).count();
        //engine->ndet = 2*ndet;
        Eigen::MatrixXd ur, kur;

        ur.setZero(in.scans.data.rows(), ndet);
        kur.setZero(in.scans.data.rows(), ndet);

        map_index_vector.resize(2*ndet);
        det_index_vector.resize(2*ndet);

        SPDLOG_INFO("ndet {}",ndet);

        Eigen::Index j = 0;
        for (Eigen::Index i=0;i<in.scans.data.cols();i++) {
            if (engine->calib_data["fg"][i] == 1) {
                ur.col(j) = in.scans.data.col(i) - in.scans.data.col(i+1);
                map_index_vector(j) = engine->calib_data["array"](i);
                map_index_vector(j+ndet) = engine->calib_data["array"](i);
                det_index_vector(j) = i;
                det_index_vector(j + ndet) = i;

                j++;
            }
        }

        auto pa2 = in.tel_meta_data.data["ParAng"].array() - pi;

        out.scans.data.resize(nsamples,2*ndet);
        //in.kernel_scans.data.resize(nsamples,2*ndet);

        for (Eigen::Index i=0;i<ndet;i++) {

            Eigen::Index di = det_index_vector(i);

            // don't use pointing here as that will rotate by azoff/eloff
            Eigen::VectorXd lat = (engine->calib_data["x_t"](di)*RAD_ASEC) + in.tel_meta_data.data["TelElDes"].array();

            // rotate by PA
            auto qs0 = -sin(-2*pa2.array())*ur.col(i).array();
            auto us0 = cos(-2*pa2.array())*ur.col(i).array();

            // rotate by elevation and flip
            auto qs1 = qs0.array()*cos(2*in.tel_meta_data.data["TelElDes"].array()) - us0.array()*sin(2*in.tel_meta_data.data["TelElDes"].array());
            auto us1 = -qs0.array()*sin(2*in.tel_meta_data.data["TelElDes"].array()) - us0.array()*cos(2*in.tel_meta_data.data["TelElDes"].array());

            // rotate by elevation and flip
            //auto qs1 = qs0.array()*cos(2*lat.array()) - us0.array()*sin(2*lat.array());
            //auto us1 = - qs0.array()*sin(2*lat.array()) - us0.array()*cos(2*lat.array());

            if (engine->run_hwp) {
                // rotate by hwp signal
                auto qs = qs1.array()*cos(4*in.hwp.data.array()) + us1.array()*sin(4*in.hwp.data.array());
                auto us = qs1.array()*sin(4*in.hwp.data.array()) - us1.array()*cos(4*in.hwp.data.array());

                out.scans.data.col(i) = qs;
                out.scans.data.col(i+ndet) = us;
            }

            else {
                out.scans.data.col(i) = qs1;
                out.scans.data.col(i+ndet) = us1;
            }
        }

        out.flags.data.setOnes(out.scans.data.rows(),out.scans.data.cols());

        SPDLOG_INFO("map_index_vector {}", map_index_vector);
        SPDLOG_INFO("det_index_vector {}", det_index_vector);

        bool done = false;
        for (Eigen::Index i=0;i<map_index_vector.size();i++) {
            if (map_index_vector(i)==0 && done==false) {
                if (i < ndet) {
                    map_index_vector(i) = 1;
                    done = true;
                }
                else {
                    map_index_vector(i) = 2;
                    done = true;
                }
            }
            if (map_index_vector(i)==1 && done==false) {
                if (i < ndet) {
                    map_index_vector(i) = 4;
                    done = true;
                }
                else {
                    map_index_vector(i) = 5;
                    done = true;
                }
            }
            if (map_index_vector(i)==2 && done==false) {
                if (i < ndet) {
                    map_index_vector(i) = 7;
                    done = true;
                }
                else {
                    map_index_vector(i) = 8;
                    done = true;
                }
            }
            done = false;
        }

        SPDLOG_INFO("map_index_vector {}", map_index_vector);
        SPDLOG_INFO("det_index_vector {}", det_index_vector);

    }

    out.scan_indices.data = in.scan_indices.data;
    out.index.data = in.index.data;
    out.tel_meta_data.data = in.tel_meta_data.data;

    return std::tuple<Eigen::VectorXd, Eigen::VectorXd>(map_index_vector, det_index_vector);

}

} // namespace timestream
