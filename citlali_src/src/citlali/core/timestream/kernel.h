#pragma once

namespace timestream {

template<typename OT>
void makeKernel(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in,
                OT &offsets,
                lali::YamlConfig config)
{
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

        auto beamSigAz = 5.0;
        auto beamSigEl = 5.0;

        double sigma = (beamSigAz + beamSigEl) / 2. / 3600. / 360. * 2.0*pi;
        in.kernelscans.data.col(det) = (dist.array() <= 3. * sigma)
                                           .select(exp(-0.5 * (dist.array() / sigma).pow(2)), 0);
    }
}

} // namespace
