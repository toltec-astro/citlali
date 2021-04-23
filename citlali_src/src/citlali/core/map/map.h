#pragma once

#include "map_utils.h"

// TcData is the data structure of which RTCData and PTCData are a part
using timestream::PTCProc;
using timestream::RTCProc;
using timestream::TCData;

// Selects the type of TCData
using timestream::LaliDataKind;

namespace  mapmaking {

class MapStruct: public mapmaking::MapUtils {

public:
  // Number of rows and cols for map
  Eigen::Index nrows, ncols, npixels, map_count;
  // Pixel size (radians) and Master Grid coordinates
  double masterGrid0, masterGrid1;
  // Physical coordinates for rows and cols (radians)
  Eigen::VectorXd rcphys, ccphys;

  // Map types
  std::vector<Eigen::MatrixXd> signal, weight, kernel, intMap;

  template<typename TD, typename OT>
  void allocateMaps(TD&, OT&, lali::YamlConfig);

  template <typename OT>
  void mapPopulate(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &, OT &,
                   YamlConfig, std::vector<std::tuple<int,int>> &);

  void mapNormalize();
};

// Resizes maps to nrows x ncols
template<typename TD, typename OT>
void MapStruct::allocateMaps(TD &telMetaData, OT &offsets, lali::YamlConfig config)
{
    npixels = nrows * ncols;

    for(Eigen::Index i = 0; i < map_count; i++) {
        signal.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
        weight.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
        kernel.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
        intMap.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
    }
}

// For a given scan find the row and col indices and add those the
// corresponding values into the map matricies
template <typename OT>
void MapStruct::mapPopulate(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in,
                            OT &offsets, YamlConfig config,
                            std::vector<std::tuple<int,int>> &ai) {

  SPDLOG_INFO("Populating map pixels for scan {}...", in.index.data);

  Eigen::Index npts = in.scans.data.rows();
  Eigen::Index ndetectors = in.scans.data.cols();

  auto maptype = config.get_str(std::tuple{"map","type"});
  pixelsize = config.get_typed<double>(std::tuple{"map","pixelsize"})*RAD_ASEC;

  for (Eigen::Index mc = 0; mc < map_count; mc++) {
      // Loop through each detector
      for (Eigen::Index det = std::get<0>(ai.at(mc)); det < std::get<1>(ai.at(mc)); det++) {
        Eigen::VectorXd lat, lon;

        // Get pointing for each detector using that scans's telescope pointing only
        if (std::strcmp("RaDec", maptype.c_str()) == 0) {
            getDetectorPointing<RaDec>(lat, lon, in.telLat.data, in.telLon.data,
                                       in.telElDes.data, in.ParAng.data,
                                       offsets["azOffset"](det),
                                       offsets["elOffset"](det), config);
        }

        else if (std::strcmp("AzEl", maptype.c_str()) == 0) {
            getDetectorPointing<AzEl>(lat, lon, in.telLat.data, in.telLon.data,
                                       in.telElDes.data, in.ParAng.data,
                                       offsets["azOffset"](det),
                                       offsets["elOffset"](det), config);
        }

        // Get row and col indices for lat and lon vectors
        Eigen::VectorXd irow = lat.array() / pixelsize + (nrows + 1.) / 2.;
        Eigen::VectorXd icol = lon.array() / pixelsize + (ncols + 1.) / 2.;

        // Loop through points in scan
        for (Eigen::Index s = 0; s < npts; s++) {

          Eigen::Index ir = irow(s);
          Eigen::Index ic = icol(s);

          // Exclude flagged data
          if (in.flags.data(s, det)) {
            /*Weight Map*/
            weight.at(mc)(ir,ic) += in.weights.data(det);

            /*Signal Map*/
            auto sig = in.scans.data(s, det);// * in.weights.data(det);
            signal.at(mc)(ir,ic) += sig;

            /*Kernel Map*/
            auto ker = in.kernelscans.data(s, det) * in.weights.data(det);
            kernel.at(mc)(ir,ic) += ker;

            /*Int Map*/
            intMap.at(mc)(ir,ic) += 1;

            /*Noise Maps*/
            // fix bug with boost random libraries
          }
        }
      }
  }
}

//Normallize the map pixels by weight map.  Needs the
// map to be completely populated to be run.
void MapStruct::mapNormalize() {
    double pixelWeight = 0;

    for (Eigen::Index mc = 0; mc < map_count; mc++) {
        signal.at(mc) = (weight.at(mc).array() == 0).select(0, signal.at(mc).array() / weight.at(mc).array());
        kernel.at(mc) = (weight.at(mc).array() == 0).select(0, kernel.at(mc).array() / weight.at(mc).array());
    }

    // Can parallelize so leave in for loops.

    /*for (Eigen::Index irow = 0; irow < nrows; irow++) {
    for (Eigen::Index icol = 0; icol < ncols; icol++) {
      pixelWeight = weight(irow, icol);
      if (pixelWeight != 0.) {
        signal(irow, icol) = -(signal(irow, icol)) / pixelWeight;
      } else {
        signal(irow, icol) = 0;
      }
    }
  }*/
}

} // namespace mapmaking
