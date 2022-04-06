#pragma once

#include <Eigen/Core>

#include <string>
#include <map>

#include <tula/logging.h>

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
    void create_rtc(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string, Engine);
};

template <class Engine>
void Polarization::create_rtc(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string sp, Engine engine) {
    // generate rtc

    if (sp == "Q") {
        int ndet = (engine->calib_data["fg"].array() == 0).count();
        Eigen::MatrixXd qr(in.scans.data.rows(), ndet);
        Eigen::MatrixXd kqr(in.scans.data.rows(), ndet);

        SPDLOG_INFO("ndet {}",ndet);

        Eigen::Index j = 0;
        for (Eigen::Index i=0;i<in.scans.data.cols();i++) {
            if (engine->calib_data["fg"][i] == 0) {
                qr.col(j) = in.scans.data.col(i+1) - in.scans.data.col(i);
                kqr.col(j) = in.kernel_scans.data.col(i+1) - in.kernel_scans.data.col(i);

                j++;
            }

        }

    }

    else if (sp == "U") {
        int ndet = (engine->calib_data["fg"].array() == 1).count();
        Eigen::MatrixXd ur(in.scans.data.rows(), ndet);
        Eigen::MatrixXd kur(in.scans.data.rows(), ndet);

        SPDLOG_INFO("ndet {}",ndet);

        Eigen::Index j = 0;
        for (Eigen::Index i=0;i<in.scans.data.cols();i++) {
            if (engine->calib_data["fg"][i] == 1) {
                ur.col(j) = in.scans.data.col(i) - in.scans.data.col(i+1);
                kur.col(j) = in.kernel_scans.data.col(i) - in.kernel_scans.data.col(i+1);

                j++;
            }

        }

    }

}

} // namespace timestream
