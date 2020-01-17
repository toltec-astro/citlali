#pragma once

#include <Eigen/Core>
#include <netcdf>
#include "../map/mapmaking.h"

// Arcseconds in 360 degrees
#define ASEC_CIRC 1296000.0
// rad per arcsecond
#define RAD_ASEC (2.0*pi / ASEC_CIRC)

// Return this in event of a problem.
static const int NC_ERR = 2;

using namespace mapmaking;

namespace aztec {

using aztec::BeammapData;

struct mapResult {

    //Cleaning Iteration
    int iter = 0;

    MapStruct mapstruct;
    //Parameter matrices
    Eigen::MatrixXd pp, pp0;
    Eigen::Matrix<bool,Eigen::Dynamic,1> done;

    int toNcFile(const std::string &filepath, BeammapData &bd) {
        using namespace netCDF;
        using namespace netCDF::exceptions;

        try{
            //Create NetCDF file
            NcFile fo(filepath, NcFile::replace);

            //Create netCDF dimensions
            NcDim nrows = fo.addDim("nrows", this->mapstruct.nrows);
            NcDim ncols = fo.addDim("ncols", this->mapstruct.ncols);
            NcDim npixels = fo.addDim("npixels", this->mapstruct.npixels);
            NcDim nnoise = fo.addDim("NNoiseMapsPerObs", this->mapstruct.NNoiseMapsPerObs);

            //NcDim ndet_dim = fo.addDim("ndet", ndet);
            NcDim psize_dim = fo.addDim("pixelsize",1);

            std::vector<NcDim> dims;
            dims.push_back(nrows);
            dims.push_back(ncols);

            NcVar rowcoordphys_var = fo.addVar("rowCoordsPhys", ncDouble, nrows);
            NcVar colcoordphys_var = fo.addVar("colCoordsPhys", ncDouble, ncols);
            fo.putAtt("pixelSize",ncDouble,this->mapstruct.pixelsize);

            rowcoordphys_var.putVar(this->mapstruct.rowcoordphys.data());
            colcoordphys_var.putVar(this->mapstruct.colcoordphys.data());

            //SPDLOG_INFO("Writing Detector {} map to ncfile", i);
            auto signalmapvar = "signal";
            auto weightmapvar = "weight";
            auto kernelmapvar = "kernel";

            NcVar signalmapdata = fo.addVar(signalmapvar, ncDouble, dims);
            NcVar weightmapdata = fo.addVar(weightmapvar, ncDouble, dims);
            NcVar kernelmapdata = fo.addVar(kernelmapvar, ncDouble, dims);

            Eigen::MatrixXd signalmatrix = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> (this->mapstruct.signal.data(),this->mapstruct.signal.dimension(0),this->mapstruct.signal.dimension(1));
            Eigen::MatrixXd weightmatrix = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> (this->mapstruct.wtt.data(),this->mapstruct.wtt.dimension(0),this->mapstruct.wtt.dimension(1));
            Eigen::MatrixXd kernelmatrix = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> (this->mapstruct.kernel.data(),this->mapstruct.kernel.dimension(0),this->mapstruct.kernel.dimension(1));

            signalmatrix.transposeInPlace();
            kernelmatrix.transposeInPlace();
            weightmatrix.transposeInPlace();

            weightmapdata.putVar(weightmatrix.data());
            signalmapdata.putVar(signalmatrix.data());
            kernelmapdata.putVar(kernelmatrix.data());

            for(int i = 0; i<this->mapstruct.NNoiseMapsPerObs;i++){
                auto noisemapvar = "noise" + std::to_string(i);
                Eigen::Tensor<double, 2> noisetensor = this->mapstruct.noisemaps.chip(i, 0);
                Eigen::MatrixXd noisematrix = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> (noisetensor.data(),noisetensor.dimension(0),noisetensor.dimension(1));

                noisematrix.transposeInPlace();
                NcVar noisemapdata = fo.addVar(noisemapvar,ncDouble,dims);
                noisemapdata.putVar(noisematrix.data());
            }

            NcDim ts = fo.addDim("ts", bd.telescope_data["TelElDes"].size());

            NcVar TelElDes = fo.addVar("TelElDes", ncDouble, ts);
            TelElDes.putVar(bd.telescope_data["TelElDes"].data());

            NcVar TelAzPhys = fo.addVar("TelAzPhys", ncDouble, ts);
            TelAzPhys.putVar(bd.telescope_data["TelAzPhys"].data());

            NcVar TelElPhys = fo.addVar("TelElPhys", ncDouble, ts);
            TelElPhys.putVar(bd.telescope_data["TelElPhys"].data());

            fo.close();
        }
        catch(NcException& e)
        {e.what();
            return NC_ERR;
        }

        return 0;
    }

};
}
