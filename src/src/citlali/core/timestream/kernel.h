#pragma once

#include <math.h>

namespace timestream {

template<typename OT, typename FT>
void makeKernel(TCData<LaliDataKind::RTC, Eigen::MatrixXd> &in, OT &offsets, FT &fwhms, config::YamlConfig config) {

    Eigen::Index ndetectors = in.scans.data.cols();
    Eigen::VectorXd dist, lat, lon;
    in.kernelscans.data.resize(in.scans.data.rows(), in.scans.data.cols());

    auto maptype = config.get_str(std::tuple{"map","type"});

    // Loop through each detector
    for (Eigen::Index det = 0; det < ndetectors; det++) {

        if (std::strcmp("RaDec", maptype.c_str()) == 0) {
            timestream_utils::getDetectorPointing<timestream_utils::RaDec>(lat, lon,
                                                                           in.telLat.data,
                                                                           in.telLon.data,
                                                                           in.telElDes.data,
                                                                           in.ParAng.data,
                                                                           offsets["azOffset"](det),
                                                                           offsets["elOffset"](det),
                                                                           config);
        }

        else if (std::strcmp("AzEl", maptype.c_str()) == 0) {
            timestream_utils::getDetectorPointing<timestream_utils::AzEl>(lat, lon,
                                                                           in.telLat.data,
                                                                           in.telLon.data,
                                                                           in.telElDes.data,
                                                                           in.ParAng.data,
                                                                           offsets["azOffset"](det),
                                                                           offsets["elOffset"](det),
                                                                           config);
        }


        dist = (lat.array().pow(2) + lon.array().pow(2)).sqrt();

        // Replace with beammap values
        auto beamSigAz = fwhms["a_fwhm"](det);
        auto beamSigEl = fwhms["b_fwhm"](det);
        //SPDLOG_INFO("beamSigAz {} beamSigEl {}", beamSigAz, beamSigEl);

        double sigma = (beamSigAz + beamSigEl) / 2. / 3600. / 360. * 2.0*pi;

        sigma = sigma/(2*sqrt(2*log(2)));
        in.kernelscans.data.col(det) = (dist.array() <= 3. * sigma)
                                           .select(exp(-0.5 * (dist.array() / sigma).pow(2)), 0);
    }
}

} // namespace
