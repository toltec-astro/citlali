#pragma once

/* This is a place holder class for coaddition until we get file input
and MPI implementation done.*/

#include "map_utils.h"

// TcData is the data structure of which RTCData and PTCData are a part
using timestream::PTCProc;
using timestream::RTCProc;
using timestream::TCData;

// Selects the type of TCData
using timestream::LaliDataKind;

namespace mapmaking {

class CoaddedMapStruct : public MapUtils
{
public:
    // Number of rows and cols for map
    Eigen::Index nrows, ncols, npixels;

    // Number of noise maps
    Eigen::Index nNoiseMaps;

    // Pixel size (radians) and Master Grid coordinates
    double masterGrid0, masterGrid1;
    // Physical coordinates for rows and cols (radians)
    Eigen::VectorXd rcphys, ccphys;

    // Unfiltered Map types
    Eigen::MatrixXd signal, weight, kernel, intMap;
    Eigen::Tensor<double, 3> noiseMaps;

    // Filtered Map types
    Eigen::MatrixXd filteredsignal, filteredweight, filteredkernel, filteredintMap;
    Eigen::Tensor<double, 3> filterednoiseMaps;

    template<class MS, typename TD, typename OT>
    void allocateMaps(MS &, TD &, OT &, std::shared_ptr<lali::YamlConfig>);

    template<typename OT>
    void mapPopulate(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &,
                     OT &,
                     std::shared_ptr<YamlConfig>);

    void mapNormalize();
};

// Resizes maps to nrows x ncols
template<class MS, typename TD, typename OT>
void CoaddedMapStruct::allocateMaps(MS &Maps,
                                    TD &telMetaData,
                                    OT &offsets,
                                    std::shared_ptr<lali::YamlConfig> config)
{
    auto [nr, nc, rcp, ccp] = setRowsCols<Coadded>(Maps, telMetaData, offsets, config);

    nrows = nr;
    ncols = nc;
    rcphys = rcp;
    ccphys = ccp;

    SPDLOG_INFO("nrows {}", nrows);
    SPDLOG_INFO("ncols {}", ncols);

    npixels = nrows * ncols;

    signal = Eigen::MatrixXd::Zero(nrows, ncols);
    weight = Eigen::MatrixXd::Zero(nrows, ncols);
    kernel = Eigen::MatrixXd::Zero(nrows, ncols);
    intMap = Eigen::MatrixXd::Zero(nrows, ncols);
}

} //namespace
