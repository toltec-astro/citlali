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
    MUSCAT = 4,
    apt = 5,
};

enum ProjectID {
    LMT = 0,
    Commissioning = 1,
    Engineering = 2,
    Simu = 3,
    none = 4,
};

enum ObsType {
    Beammap = 0,
    Pointing = 1,
    Science = 2,
};

class Result{
public:

    template <class engineType>
    auto writeMapsToNetCDF(engineType, const std::string, std::string);

    template <class engineType>
    auto writeMapsToFITS(engineType, const std::string, std::string, int mc, std::vector<std::tuple<int,int>> &);

    template <DataType datatype, ProjectID projectid, ObsType obstype, class engineType>
    auto composeFilename(engineType engine, int mi) {

        std::string filename;

        if constexpr (datatype == TolTEC) {
            filename = filename + "toltec_";
        }

        else if constexpr (datatype == apt) {
            filename = filename + "apt_";
        }

        if constexpr (projectid == Simu) {
            filename = filename + "simu_";
        }

        if (mi == 0) {
            filename = filename + "a1100_";
        }

        else if(mi == 1) {
            filename = filename + "a1400_";
        }

        else if (mi == 2) {
            filename = filename + "a2000_";
        }

        if constexpr (obstype == Science) {
            filename = filename + "science_";
        }

        if constexpr (obstype == Beammap) {
            filename = filename + "beammap_";
        }

        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << engine->obsid;
        std::string obsid = ss.str();

        filename = filename + obsid + "_";

        std::stringstream ss2;

        const auto p1 = std::chrono::system_clock::now();
        ss2 <<  std::chrono::duration_cast<std::chrono::seconds>(
                           p1.time_since_epoch()).count();

        std::string unix_time = ss2.str();
        filename = filename + unix_time;

        return filename;
    }

    template <typename MC, typename MT>
    auto setupNetCDFVars(MC &Maps, MT &map, std::string varname,
                         netCDF::NcFile &fo, std::vector<netCDF::NcDim> dims) {
        for (Eigen::Index mc = 0; mc < Maps.map_count; mc++) {
            auto var = varname + std::to_string(mc);
            netCDF::NcVar mapVar = fo.addVar(var, netCDF::ncDouble, dims);
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rowMajorMap
                = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                    map.at(mc).data(), map.at(mc).rows(), map.at(mc).cols());
            mapVar.putVar(rowMajorMap.data());
        }
    }

    template <class engineType, typename mapType>
    auto setupHDU(engineType engine, mapType map, std::unique_ptr<CCfits::FITS> &pFits, std::string map_name, Eigen::Index map_num) {

        auto grouping = engine->config.get_str(std::tuple{"map","grouping"});
        std::vector naxes{map.cols(), map.rows()};
        auto hdu = pFits->addImage(map_name, DOUBLE_IMG, naxes);
        std::valarray<double> tmp(map.size());

        int k = 0;
        for (int i=0; i<map.rows(); i++){
            for (int j=0; j<map.cols(); j++) {
                tmp[k] = map(i, map.cols() - j - 1);
                k++;
            }
        }

        // Write map to hdu
        long fpixel = 1;
        hdu->write(fpixel, tmp.size(), tmp);

        // add wcs to the img hdu
        hdu->addKey("CTYPE1", "RA---TAN", "");
        hdu->addKey("CTYPE2", "DEC--TAN", "");
        // center pixel. note the order of axis. crpix1 is x which is col
        // note the extra 1 in the crpix, because fits is 1-based

        int refpixC2 = map.cols()/2;
        int refpixC1 = map.rows()/2;

        refpixC1 = map.rows()-1-refpixC1+1;
        refpixC2 += 1;

        hdu->addKey("CRPIX1", refpixC1, "");
        hdu->addKey("CRPIX2", refpixC2, "");

        // coords of the ref pixel in degrees
        double CRVAL1 = engine->telMD.srcCenter["centerRa"](0)*180./pi;
        double CRVAL2 = engine->telMD.srcCenter["centerDec"](0)*180./pi;

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

        if (std::strcmp("beammap", grouping.c_str()) == 0) {
            hdu->addKey("S/N", engine->fittedParams(0,map_num)/RAD_ASEC, "Fitted S/N");
            hdu->addKey("off_y", engine->fittedParams(1,map_num)/RAD_ASEC, "Fitted Offset y (arcsec)");
            hdu->addKey("off_x", engine->fittedParams(2,map_num)/RAD_ASEC, "Fitted Offset x (arcsec)");
            hdu->addKey("fwhm_y", engine->fittedParams(3,map_num)/RAD_ASEC, "Fitted FWHM y (arcsec)");
            hdu->addKey("fwhm_x", engine->fittedParams(4,map_num)/RAD_ASEC, "Fitted FWHM x (arcsec)");
        }
    }

};

template <class engineType>
auto Result::writeMapsToNetCDF(engineType engine, const std::string filepath, std::string filename){
    int NC_ERR;
    try {

        auto grouping = engine->config.get_str(std::tuple{"map","grouping"});

        /*std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << engine->obsid;
        std::string obsid = ss.str();

        filename = filename + obsid + "_";*/


        /*if (std::strcmp("array_name", grouping.c_str()) == 0) {
            filename = filename+ "_science_" + obsid + "_";
        }

        else if (std::strcmp("beammap", grouping.c_str()) == 0) {
            filename = filename+ "_beammap_" + obsid + "_";
        }*/
        /*std::stringstream ss2;

        const auto p1 = std::chrono::system_clock::now();
        ss2 <<  std::chrono::duration_cast<std::chrono::hours>(
                           p1.time_since_epoch()).count();

        std::string unix_time = ss2.str();
        filename = filename + unix_time + ".nc";
        */

        //Create NetCDF file
        netCDF::NcFile fo(filepath + filename + ".nc", netCDF::NcFile::replace);

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
auto Result::writeMapsToFITS(engineType engine, const std::string filepath, std::string filename, int mi,
                             std::vector<std::tuple<int,int>> &di) {

    CCfits::FITS::setVerboseMode("True");

    auto grouping = engine->config.get_str(std::tuple{"map","grouping"});

    //for (Eigen::Index mc = 0; mc < engine->array_index.size(); mc++) {
        std::unique_ptr<CCfits::FITS> pFits(nullptr);

        /*std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << engine->obsid;
        std::string obsid = ss.str();

        filename = filename + obsid + "_";*/

        /*if (std::strcmp("array_name", grouping.c_str()) == 0) {
            filename = filename+ "_science_" + obsid + "_";
        }

        else if (std::strcmp("beammap", grouping.c_str()) == 0) {
            filename = filename+ "_beammap_" + obsid + "_";
        }*/
        /*std::stringstream ss2;

        const auto p1 = std::chrono::system_clock::now();
        ss2 <<  std::chrono::duration_cast<std::chrono::hours>(
                           p1.time_since_epoch()).count();

        std::string unix_time = ss2.str();
        filename = filename + unix_time + ".fits";*/

        try
        {
            // create the fits object with empty primary hdu
            // we'll add images later
            pFits.reset( new CCfits::FITS(filepath + filename + ".fits", CCfits::Write) );
        }

        catch (CCfits::FITS::CantCreate)
        {
            // ... or not, as the case may be.
            SPDLOG_ERROR("unable to create file {}", filepath + filename);
        }

        pFits->pHDU().addKey("OBJECT", "citlali_reduction", "");
        pFits->pHDU().writeDate();

        if (std::strcmp("array_name", grouping.c_str()) == 0) {
            setupHDU(engine,engine->Maps.signal[mi], pFits, "signal", mi);
            setupHDU(engine,engine->Maps.weight[mi], pFits, "weight", mi);
            setupHDU(engine,engine->Maps.kernel[mi], pFits, "kernel", mi);
            setupHDU(engine,engine->Maps.intMap[mi], pFits, "intMap", mi);
            setupHDU(engine,engine->Maps.signal[mi].array()*engine->Maps.weight[mi].array().sqrt(), pFits, "snrMap", mi);
        }

        else if (std::strcmp("beammap", grouping.c_str()) == 0) {
            for (Eigen::Index mc = std::get<0>(engine->array_index.at(mi)); mc < std::get<1>(engine->array_index.at(mi)); mc++) {
                    setupHDU(engine,engine->Maps.signal[mc], pFits, "sig_"+std::to_string(mc), mc);
                    setupHDU(engine,engine->Maps.weight[mc], pFits, "wt_"+std::to_string(mc), mc);
            }
        }
}
} //namespace
