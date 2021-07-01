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
                             const double, const double, lali::YamlConfig);

    template <MapClassType, typename TD, typename OT>
    auto getMapMaxMin(TD&, OT&, lali::YamlConfig);

    template <MapClassType, typename TD, typename OT>
    auto getRowsCols(TD&, OT&, lali::YamlConfig);


};

template <MapUtils::PointingType pointingtype, typename DerivedA, typename DerivedB>
void MapUtils::getDetectorPointing(Eigen::DenseBase<DerivedA> &lat, Eigen::DenseBase<DerivedA> &lon,
                                   Eigen::DenseBase<DerivedB> &telLat, Eigen::DenseBase<DerivedB> &telLon,
                                   Eigen::DenseBase<DerivedB> &TelElDes, Eigen::DenseBase<DerivedB> &ParAng,
                                   const double azOffset, const double elOffset, lali::YamlConfig config){

  // RaDec map
  if constexpr (pointingtype == RaDec) {

    auto azOfftmp = cos(TelElDes.derived().array()) * azOffset -
                    sin(TelElDes.derived().array()) * elOffset;// +
                    //config.get_typed<double>("bsOffset_0");
    auto elOfftmp = cos(TelElDes.derived().array()) * elOffset +
                    sin(TelElDes.derived().array()) * azOffset;// +
                    //config.get_typed<double>("bsOffset_1");
    auto pa2 = ParAng.derived().array() - pi;

    auto ratmp = -azOfftmp * cos(pa2) - elOfftmp * sin(pa2);
    auto dectmp = -azOfftmp * sin(pa2) + elOfftmp * cos(pa2);

    lat = dectmp * RAD_ASEC + telLat.derived().array();
    lon = ratmp * RAD_ASEC + telLon.derived().array();

  }

  // Az/El map
  else if constexpr (pointingtype == AzEl) {
      auto azOfftmp = cos(TelElDes.derived().array()) * azOffset -
                      sin(TelElDes.derived().array()) * elOffset;// +
                      //config.get_typed<double>("bsOffset_0");
      auto elOfftmp = cos(TelElDes.derived().array()) * elOffset +
                      sin(TelElDes.derived().array()) * azOffset;// +
                      //config.get_typed<double>("bsOffset_1");

      lat = elOfftmp * RAD_ASEC + telLat.derived().array();
      lon = azOfftmp * RAD_ASEC + telLon.derived().array();
  }
}


template <MapUtils::MapClassType mapclasstype, typename TD, typename OT>
auto MapUtils::getMapMaxMin(TD &telMetaData, OT &offsets, lali::YamlConfig config){

    // Using the offsets, find the map max and min values and calculate
    // nrows and ncols
    Eigen::MatrixXd mapDims = Eigen::MatrixXd::Zero(2,2);

    auto maptype = config.get_str(std::tuple{"map","type"});

    if constexpr (mapclasstype == MapUtils::Individual) {
        Eigen::VectorXd lat, lon;
        Eigen::MatrixXd lat_lim(offsets["azOffset"].size(), 2);
        Eigen::MatrixXd lon_lim(offsets["elOffset"].size(), 2);

        std::vector<int> dets(offsets["azOffset"].size());
        std::iota (std::begin(dets), std::end(dets), 0);

        std::vector<int> w(dets.size());

        auto ex_name = config.get_str(std::tuple{"runtime","policy"});
        auto nThreads = config.get_typed<int>(std::tuple{"runtime","ncores"});

        //Get max and min lat and lon values out of all detectors.  Maybe parallelize?
        // for (Eigen::Index det=0;det<offsets["azOffset"].size();det++) {
        grppi::map(grppiex::dyn_ex(ex_name), begin(dets), end(dets), begin(w), [&](int det) {
            if (std::strcmp("RaDec", maptype.c_str()) == 0) {
                getDetectorPointing<MapUtils::RaDec>(lat, lon, telMetaData["TelRaPhys"], telMetaData["TelDecPhys"],
                                                           telMetaData["TelElDes"], telMetaData["ParAng"],
                                                           offsets["azOffset"](det), offsets["elOffset"](det), config);
            }

            else if (std::strcmp("AzEl", maptype.c_str()) == 0) {
                getDetectorPointing<MapUtils::AzEl>(lat, lon, telMetaData["TelAzPhys"], telMetaData["TelElPhys"],
                                                           telMetaData["TelElDes"], telMetaData["ParAng"],
                                                           offsets["azOffset"](det), offsets["elOffset"](det), config);
            }

            lat_lim(det,0) = lat.minCoeff();
            lat_lim(det,1) = lat.maxCoeff();
            lon_lim(det,0) = lon.minCoeff();
            lon_lim(det,1) = lon.maxCoeff();

            return 0;

        });

        for (Eigen::Index det=0;det<offsets["azOffset"].size();det++){

            if(lat_lim(det,0) < mapDims(0,0)){
                mapDims(0,0) = lat_lim(det,0);
            }
            if(lat_lim(det,1) > mapDims(1,0)){
                mapDims(1,0) = lat_lim(det,1);
            }
            if(lon_lim(det,0) < mapDims(0,1)){
                mapDims(0,1) = lon_lim(det,0);
            }
            if(lon_lim(det,1) > mapDims(1,1)){
                mapDims(1,1) = lon_lim(det,1);
            }
        }
    }

    /*else if constexpr (mapclasstype == MapUtils::Coadded) {
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
    }*/

    return std::move(mapDims);
}

template<MapUtils::MapClassType mapclasstype, typename TD, typename OT>
auto MapUtils::getRowsCols(TD &telMetaData, OT &offsets,
                           lali::YamlConfig config)
{

    auto mapDims = getMapMaxMin<mapclasstype>(telMetaData, offsets, config);

    // Set the pixelsize
    pixelsize = config.get_typed<double>(std::tuple{"map","pixelsize"})*RAD_ASEC;
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
