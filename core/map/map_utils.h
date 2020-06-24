#pragma once

/*
This header file includes various helper functions for use during mapmaking.
*/

using lali::YamlConfig;

namespace mapmaking {

class MapUtils {

public:

    enum PointingType {
        RaDec = 0,
        AzEl = 1
    };

    enum MapClassType {
        Individual = 0,
        Coadded = 1
    };

    double pixelsize;

    template <PointingType maptype, typename DerivedA, typename DerivedB>
    void getDetectorPointing(Eigen::DenseBase<DerivedA> &, Eigen::DenseBase<DerivedA> &,
                             Eigen::DenseBase<DerivedB> &, Eigen::DenseBase<DerivedB> &,
                             Eigen::DenseBase<DerivedB> &, Eigen::DenseBase<DerivedB> &,
                             const double, const double, std::shared_ptr<lali::YamlConfig>);

    template <MapClassType mapclasstype, class MS, typename TD, typename OT>
    static auto setRowsCols(MS&, TD&, OT&, std::shared_ptr<lali::YamlConfig>);

    template <MapClassType mapclasstype, class MS, typename TD, typename OT>
    auto getMapMaxMin(MS&, TD&, OT&, std::shared_ptr<lali::YamlConfig>);

};

template <PointingType pointingtype, typename DerivedA, typename DerivedB>
void MapUtils::getDetectorPointing(Eigen::DenseBase<DerivedA> &lat, Eigen::DenseBase<DerivedA> &lon,
                                   Eigen::DenseBase<DerivedB> &telLat, Eigen::DenseBase<DerivedB> &telLon,
                                   Eigen::DenseBase<DerivedB> &TelElDes, Eigen::DenseBase<DerivedB> &ParAng,
                                   const double azOffset, const double elOffset, std::shared_ptr<lali::YamlConfig> config){

  // RaDec map
  if constexpr (pointingtype == RaDec) {

    auto azOfftmp = cos(TelElDes.derived().array()) * azOffset -
                    sin(TelElDes.derived().array()) * elOffset +
                    config->get_typed<double>("bsOffset_0");
    auto elOfftmp = cos(TelElDes.derived().array()) * elOffset +
                    sin(TelElDes.derived().array()) * azOffset +
                    config->get_typed<double>("bsOffset_1");
    auto pa2 = ParAng.derived().array() - pi;

    auto ratmp = -azOfftmp * cos(pa2) - elOfftmp * sin(pa2);
    auto dectmp = -azOfftmp * sin(pa2) + elOfftmp * cos(pa2);

    lat = ratmp * RAD_ASEC + telLat.derived().array();
    lon = dectmp * RAD_ASEC + telLon.derived().array();

  }

  // Az/El map
  else if constexpr (pointingtype == AzEl) {
  }
}



template <MapClassType mapclasstype, class MS, typename TD, typename OT>
auto getMapMaxMin(MS &maps, TD &telMetaData, OT &offsets, std::shared_ptr<lali::YamlConfig> config){

    Eigen::MatrixXd mapDims = Eigen::MatrixXd::Zero(2,2);

    if constexpr (mapclasstype == Individual) {
        Eigen::VectorXd lat, lon;

        //Get max and min lat and lon values out of all detectors.  Maybe parallelize?
        for (Eigen::Index det=0;det<offsets["azOffset"].size();det++) {
            getDetectorPointing<RaDec>(lat, lon, telMetaData["TelRaPhys"], telMetaData["TelDecPhys"],
                                                       telMetaData["TelElDes"], telMetaData["ParAng"],
                                                       offsets["azOffset"](det), offsets["elOffset"](det), config);

            if(lat.minCoeff() < mapDims(0,0)){
                mapDims(0,0) = lat.minCoeff();
            }
            if(lat.maxCoeff() > mapDims(1,0)){
                mapDims(1,0) = lat.maxCoeff();
            }
            if(lon.minCoeff() < mapDims(0,1)){
                mapDims(0,1) = lon.minCoeff();
            }
            if(lon.maxCoeff() > mapDims(1,1)){
                mapDims(1,1) = lon.maxCoeff();
            }
        }
    }

    else if constexpr (mapclasstype == Coadded) {
        if(maps.rcphys.minCoeff() < mapDims(0,0)){
            mapDims(0,0) = maps.rcphys.minCoeff();
        }
        if(maps.rcphys.maxCoeff() > mapDims(1,0)){
            mapDims(1,0) = maps.rcphys.maxCoeff();
        }
        if(maps.ccphys.minCoeff() < mapDims(0,1)){
            mapDims(0,1) = maps.ccphys.minCoeff();
        }
        if(maps.ccphys.maxCoeff() > mapDims(1,1)){
            mapDims(1,1) = maps.ccphys.maxCoeff();
        }
    }

    return mapDims;
}

template<MapClassType mapclasstype, class MS, typename TD, typename OT>
auto MapUtils::setRowsCols(MS &maps, TD &telMetaData, OT &offsets,
                           std::shared_ptr<lali::YamlConfig> config)
{

    // Calculate the max and min dimensions of the map.  Uses telData and offsets
    // for individual maps and the individual map rcphys and ccphys for coadded
    // maps
    auto mapDims = getMapMaxMin<mapclasstype>(maps, telMetaData, offsets, config);

    // Set the pixelsize
    maps.pixelsize = config->get_typed<double>("pixelsize")*RAD_ASEC;
    SPDLOG_INFO("pixelsize {} (arcseconds)/{} (radians)", this->pixelsize/RAD_ASEC, this->pixelsize);

    // Find the maximum pixel value in the lat dimension
    int latminpix = ceil(abs(mapDims(0,0)/this->pixelsize));
    int latmaxpix = ceil(abs(mapDims(1,0)/this->pixelsize));
    latmaxpix = std::max(latminpix, latmaxpix);
    // Set nrows
    this->nrows = 2 * latmaxpix + 4;

    // Find the maximum pixel value in the lon dimension
    int lonminpix = ceil(abs(mapDims(0,1)/this->pixelsize));
    int lonmaxpix = ceil(abs(mapDims(1,1)/this->pixelsize));
    lonmaxpix = std::max(lonminpix, lonmaxpix);
    // Set ncols
    this->ncols = 2 * lonmaxpix + 4;

    SPDLOG_INFO("nrows {}", maps.nrows);
    SPDLOG_INFO("ncols {}", maps.ncols);

    this->rcphys = (Eigen::VectorXd::LinSpaced(this->nrows, 0, this->nrows - 1).array() -
            (this->nrows + 1.) / 2.) *
           this->pixelsize;
    this->ccphys = (Eigen::VectorXd::LinSpaced(this->ncols, 0, this->ncols - 1).array() -
            (this->ncols + 1.) / 2.) *
           this->pixelsize;
}

} //namespace
