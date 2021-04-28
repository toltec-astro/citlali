#pragma once

#include <CCfits/CCfits>
#include <chrono>
#include <utils/nc.h>

namespace lali {

enum DataType {
    AzTEC_Testing = 0,
    AzTEC_as_TolTEC_Testing = 1,
    AzTEC = 2,
    TolTEC = 3,
    MUSCAT = 4
};

enum ProjectID {
    LMT = 0,
    Commissioning = 1,
    Engineering = 2,
    Simu = 3,
};

enum ObsType {
    Beammap = 0,
    Pointing = 1,
    Science = 2,
};

class Result{
public:

    template <class engineType>
    auto writeMapsToNetCDF(engineType &, const std::string);

    template <class engineType>
    auto writeMapsToFITS(engineType, const std::string, std::string, int mc);

    template <DataType datatype, ProjectID projectid, ObsType obstype, class engineType>
    auto composeFilename(engineType engine) {

        std::string filename;

        if constexpr (datatype == TolTEC) {
            filename = filename + "toltec_";
        }

        if constexpr (projectid == Simu) {
            filename = filename + "simu_";
        }

        if constexpr (obstype == Science) {
            // filename = filename + "simu_";
        }

        return filename;
    }

    template <typename MC, typename MT>
    auto setupNetCDFVars(MC &Maps, MT &map, std::string varname,
                         netCDF::NcFile &fo, std::vector<netCDF::NcDim> dims) {
        for (Eigen::Index mc = 0; mc < Maps.map_count; mc++) {
            auto var = varname + std::to_string(mc);
            netCDF::NcVar mapVar = fo.addVar(var, netCDF::ncDouble, dims);
            Eigen::MatrixXd rowMajorMap
                = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                    map.at(mc).data(), map.at(mc).rows(), map.at(mc).cols());
            rowMajorMap.transposeInPlace();
            mapVar.putVar(rowMajorMap.data());
        }
    }

    template <class engineType, typename mapType>
    auto setupHDU(engineType engine, mapType map, std::unique_ptr<CCfits::FITS> &pFits, std::string map_name) {
        std::vector naxes{map.cols(), map.rows()};
        auto hdu = pFits->addImage(map_name, DOUBLE_IMG, naxes);

        std::valarray<double> tmp(map.data(), map.size());
        hdu->write(1, tmp.size(), tmp);

        // add wcs to the img hdu
        hdu->addKey("CTYPE1", "RA---TAN", "");
        hdu->addKey("CTYPE2", "DEC--TAN", "");
        // center pixel. note the order of axis. crpix1 is x which is col
        // note the extra 1 in the crpix, because fits is 1-based
        hdu->addKey("CRPIX1", map.cols() / 2. + 1, "");
        hdu->addKey("CRPIX2", map.rows() / 2. + 1, "");

        // coords of the ref pixel in degrees
        double CRVAL1 = engine->telMD.srcCenter["centerRa"](0)/DEG_TO_RAD;
        double CRVAL2 = engine->telMD.srcCenter["centerDec"](0)/DEG_TO_RAD;

        hdu->addKey("CUNIT1", "deg", "");
        hdu->addKey("CUNIT2", "deg", "");
        hdu->addKey("CRVAL1", CRVAL1, "");
        hdu->addKey("CRVAL2", CRVAL2, "");

        auto pixelsize = engine->config. template get_typed<double>(std::tuple{"map","pixelsize"});

        // CD matrix. We assume pxiel schale of 1arcsec, and no rota
        //double pixsize_arcsec = 1.;
        hdu->addKey("CD1_1", -pixelsize/3600., "");
        hdu->addKey("CD1_2", 0, "");
        hdu->addKey("CD2_1", 0, "");
        hdu->addKey("CD2_2", pixelsize/3600., "");
    }

};

template <class engineType>
auto Result::writeMapsToNetCDF(engineType &engine, const std::string filepath){

    int NC_ERR;
    try {
        //Create NetCDF file
        netCDF::NcFile fo(filepath, netCDF::NcFile::replace);

        auto grouping = engine->config.get_str(std::tuple{"map","grouping"});

        //Create netCDF dimensions
        netCDF::NcDim nrows = fo.addDim("nrows", engine->Maps.nrows);
        netCDF::NcDim ncols = fo.addDim("ncols", engine->Maps.ncols);

        std::vector<netCDF::NcDim> dims;
        dims.push_back(nrows);
        dims.push_back(ncols);

        setupNetCDFVars(engine->Maps, engine->Maps.signal, "signal", fo, dims);
        setupNetCDFVars(engine->Maps, engine->Maps.weight, "weight", fo, dims);

        if (std::strcmp("beammap", grouping.c_str()) == 1) {
            setupNetCDFVars(engine->Maps, engine->Maps.kernel, "kernel", fo, dims);
            setupNetCDFVars(engine->Maps, engine->Maps.intMap, "intMap", fo, dims);
        }


        fo.close();

    } catch (netCDF::exceptions::NcException &e) {
        e.what();
        return NC_ERR;
    }

    return 0;
}


template <class engineType>
auto Result::writeMapsToFITS(engineType engine, const std::string filepath, std::string filename, int mc) {

    CCfits::FITS::setVerboseMode("True");

    auto grouping = engine->config.get_str(std::tuple{"map","grouping"});

    //for (Eigen::Index mc = 0; mc < engine->array_index.size(); mc++) {
        std::unique_ptr<CCfits::FITS> pFits(nullptr);

        if (mc == 0) {
            filename = filename + "a1100";
        }

        else if(mc == 1) {
            filename = filename + "a1400";
        }

        else if (mc == 2) {
            filename = filename + "a2000";
        }


        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << engine->obsid;
        std::string obsid = ss.str();

        if (std::strcmp("array_name", grouping.c_str()) == 0) {
            filename = filename+ "_science_" + obsid + "_";
        }

        else if (std::strcmp("beammap", grouping.c_str()) == 0) {
            filename = filename+ "_beammap_" + obsid + "_";
        }
        std::stringstream ss2;

        const auto p1 = std::chrono::system_clock::now();
        ss2 <<  std::chrono::duration_cast<std::chrono::hours>(
                           p1.time_since_epoch()).count();

        std::string unix_time = ss2.str();
        filename = filename + unix_time + ".fits";

        try
        {
            // create the fits object with empty primary hdu
            // we'll add images later
            pFits.reset( new CCfits::FITS(filepath + filename, CCfits::Write) );
        }

        catch (CCfits::FITS::CantCreate)
        {
            // ... or not, as the case may be.
            SPDLOG_ERROR("unable to create file {}", filepath + filename);
        }

        pFits->pHDU().addKey("OBJECT", "citlali_reduction", "");
        pFits->pHDU().writeDate();

        if (std::strcmp("array_name", grouping.c_str()) == 0) {
            setupHDU(engine,engine->Maps.signal[mc], pFits, "signal");
            setupHDU(engine,engine->Maps.weight[mc], pFits, "weight");
            setupHDU(engine,engine->Maps.kernel[mc], pFits, "kernel");
            setupHDU(engine,engine->Maps.intMap[mc], pFits, "intMap");
        }

        else if (std::strcmp("beammap", grouping.c_str()) == 0) {
            for (Eigen::Index det = std::get<0>(engine->det_index.at(det)); det < std::get<1>(engine->det_index.at(det)); det++) {
                setupHDU(engine,engine->Maps.signal[det], pFits, "sig_"+std::to_string(det));
                setupHDU(engine,engine->Maps.weight[det], pFits, "wt_"+std::to_string(det));
            }
        }

    //}
}


}
