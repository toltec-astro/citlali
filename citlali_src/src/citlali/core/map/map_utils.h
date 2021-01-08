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

    template <PointingType, typename DerivedA, typename DerivedB>
    void getDetectorPointing(Eigen::DenseBase<DerivedA> &, Eigen::DenseBase<DerivedA> &,
                             Eigen::DenseBase<DerivedB> &, Eigen::DenseBase<DerivedB> &,
                             Eigen::DenseBase<DerivedB> &, Eigen::DenseBase<DerivedB> &,
                             const double, const double, std::shared_ptr<lali::YamlConfig>);

    template <MapClassType, class MS, typename TD, typename OT>
    auto getMapMaxMin(MS&, TD&, OT&, std::shared_ptr<lali::YamlConfig>);

    template <MapClassType, class MS, typename TD, typename OT>
    auto setRowsCols(MS&, TD&, OT&, std::shared_ptr<lali::YamlConfig>);


};

template <MapUtils::PointingType pointingtype, typename DerivedA, typename DerivedB>
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


template <MapUtils::MapClassType mapclasstype, class MS, typename TD, typename OT>
auto MapUtils::getMapMaxMin(MS &maps, TD &telMetaData, OT &offsets, std::shared_ptr<lali::YamlConfig> config){

    // Using the offsets, find the map max and min values and calculate
    // nrows and ncols
    Eigen::MatrixXd mapDims = Eigen::MatrixXd::Zero(2,2);

    if constexpr (mapclasstype == MapUtils::Individual) {
        Eigen::VectorXd lat, lon;

        //Get max and min lat and lon values out of all detectors.  Maybe parallelize?
        for (Eigen::Index det=0;det<offsets["azOffset"].size();det++) {
            getDetectorPointing<MapUtils::RaDec>(lat, lon, telMetaData["TelRaPhys"], telMetaData["TelDecPhys"],
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

    else if constexpr (mapclasstype == MapUtils::Coadded) {
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

    return std::move(mapDims);
}

template<MapUtils::MapClassType mapclasstype, class MS, typename TD, typename OT>
auto MapUtils::setRowsCols(MS &maps, TD &telMetaData, OT &offsets,
                           std::shared_ptr<lali::YamlConfig> config)
{

    auto mapDims = getMapMaxMin<mapclasstype>(maps, telMetaData, offsets, config);

    // Set the pixelsize
    pixelsize = config->get_typed<double>("pixelsize")*RAD_ASEC;
    SPDLOG_INFO("pixelsize {} (arcseconds)/{} (radians)", pixelsize/RAD_ASEC, pixelsize);

    // Find the maximum pixel value in the lat dimension
    int latminpix = ceil(abs(mapDims(0,0)/pixelsize));
    int latmaxpix = ceil(abs(mapDims(1,0)/pixelsize));
    latmaxpix = std::max(latminpix, latmaxpix);
    // Set nrows
    int nr = 2 * latmaxpix + 4;

    // Find the maximum pixel value in the lon dimension
    int lonminpix = ceil(abs(mapDims(0,1)/pixelsize));
    int lonmaxpix = ceil(abs(mapDims(1,1)/pixelsize));
    lonmaxpix = std::max(lonminpix, lonmaxpix);
    // Set ncols
    int nc = 2 * lonmaxpix + 4;

    SPDLOG_INFO("nrows {}", nr);
    SPDLOG_INFO("ncols {}", nc);

    auto rcp = (Eigen::VectorXd::LinSpaced(nr, 0, nr - 1).array() -
            (nr + 1.) / 2.) *
           pixelsize;
     auto ccp = (Eigen::VectorXd::LinSpaced(nc, 0, nc - 1).array() -
            (nc + 1.) / 2.) *
           pixelsize;

     return std::tuple<int, int, Eigen::VectorXd, Eigen::VectorXd>(nr, nc, std::move(rcp), std::move(ccp));
}

} //namespace