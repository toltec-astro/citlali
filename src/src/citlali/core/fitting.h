#pragma once

#include "timestream/timestream_utils.h"

namespace timestream {

template<typename OT, typename Derived>
void addGaussian(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in, OT &offsets, Eigen::DenseBase<Derived> &fitParams,
                 lali::YamlConfig config) {

    auto n_detectors = in.scans.data.cols();
    auto npts = in.scans.data.rows();
    auto maptype = config.get_str(std::tuple{"map","type"});

    Eigen::VectorXd lat, lon;
    using timestream_utils::getDetectorPointing;
    using timestream_utils::RaDec;
    using timestream_utils::AzEl;

    // Loop through each detector
    for (Eigen::Index det = 0; det < n_detectors; det++) {

        if (std::strcmp("RaDec", maptype.c_str()) == 0) {
            getDetectorPointing<RaDec>(lat, lon, in.telLat.data, in.telLon.data, in.telElDes.data,
                                        in.ParAng.data, offsets["azOffset"](det), offsets["elOffset"](det), config);
        }

        else if (std::strcmp("AzEl", maptype.c_str()) == 0) {
            getDetectorPointing<AzEl>(lat, lon, in.telLat.data, in.telLon.data, in.telElDes.data,
                                      in.ParAng.data, offsets["azOffset"](det), offsets["elOffset"](det), config);
        }

        double amplitude = fitParams(0,det);
        double offset_lat = fitParams(1,det);
        double offset_lon = fitParams(2,det);
        double sigma_lat = fitParams(3,det);
        double sigma_lon = fitParams(4,det);

        auto toAdd = amplitude*exp(-1.*(pow(lat.array() - offset_lat, 2) / (2.*pow(sigma_lat,2))
                                        + pow(lon.array() - offset_lon, 2) / (2.*pow(sigma_lon,2))));

        in.scans.data.col(det) = in.scans.data.col(det).array() + toAdd;

        //for (Eigen::Index j=0; j<npts; j++) {
          //  double toAdd = amplitude*exp(-1.*(pow(lat(j) - offset_lat, 2) / (2.*pow(sigma_lat,2))
            //                                  + pow(lon(j) - offset_lon, 2) / (2.*pow(sigma_lon,2))));

            //in.scans.data(j,det) = in.scans.data(j,det) + toAdd;
        //}
    }
}

} // namespace
