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
  Eigen::Index nrows, ncols, npixels;
  // Pixel size (radians) and Master Grid coordinates
  double masterGrid0, masterGrid1;
  // Physical coordinates for rows and cols (radians)
  Eigen::VectorXd rcphys, ccphys;

  // Map types
  Eigen::MatrixXd signal, weight, kernel, intMap;

  template<typename TD, typename OT>
  void allocateMaps(TD&, OT&, std::shared_ptr<lali::YamlConfig>);

  template <typename OT>
  void mapPopulate(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &, OT &,
                   std::shared_ptr<YamlConfig>);

  void mapNormalize();
};

// Resizes maps to nrows x ncols
template<typename TD, typename OT>
void MapStruct::allocateMaps(TD &telMetaData, OT &offsets, std::shared_ptr<lali::YamlConfig> config)
{
    auto [nr, nc, rcp, ccp] = setRowsCols<Individual>(*this, telMetaData, offsets, config);

    nrows = nr;
    ncols = nc;
    rcphys = rcp;
    ccphys = ccp;

    npixels = nrows * ncols;

    signal = Eigen::MatrixXd::Zero(nrows, ncols);
    weight = Eigen::MatrixXd::Zero(nrows, ncols);
    kernel = Eigen::MatrixXd::Zero(nrows, ncols);
    intMap = Eigen::MatrixXd::Zero(nrows, ncols);
}

// For a given scan find the row and col indices and add those the
// corresponding values into the map matricies
template <typename OT>
void MapStruct::mapPopulate(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in,
                            OT &offsets, std::shared_ptr<YamlConfig> config) {

  SPDLOG_INFO("Populating map pixels for scan {}...", in.index.data);

  Eigen::Index npts = in.scans.data.rows();
  Eigen::Index ndetectors = in.scans.data.cols();

  // Loop through each detector
  for (Eigen::Index det = 0; det < ndetectors; det++) {
    Eigen::VectorXd lat, lon;

    // Get pointing for each detector using that scans's telescope pointing only
    getDetectorPointing<RaDec>(lat, lon, in.telLat.data, in.telLon.data,
                               in.telElDes.data, in.ParAng.data,
                               offsets["azOffset"](det),
                               offsets["elOffset"](det), config);

    // Get row and col indices for lat and lon vectors
    Eigen::VectorXd irow = lat.array() / pixelsize + (nrows + 1.) / 2.;
    Eigen::VectorXd icol = lon.array() / pixelsize + (ncols + 1.) / 2.;

    // Loop through points in scan
    for (Eigen::Index s = 0; s < npts; s++) {
      // Exclude flagged data
      if (in.flags.data(s, det)) {
        /*Weight Map*/
        weight(irow(s), icol(s)) += in.weights.data(det);

        /*Signal Map*/
        auto sig = in.scans.data(s, det) * in.weights.data(det);
        signal(irow(s), icol(s)) += sig;

        /*Kernel Map*/
        auto ker = in.kernelscans.data(s, det) * in.weights.data(det);
        kernel(irow(s), icol(s)) += ker;

        /*Noise Maps*/
      }
    }
  }
}

//Normallize the map pixels by weight map.  Needs the
// map to be completely populated to be run.
void MapStruct::mapNormalize() {
    double pixelWeight = 0;

    signal = (weight.array() == 0).select(0, -signal.array() / weight.array());
    kernel = (weight.array() == 0).select(0, -kernel.array() / weight.array());

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
