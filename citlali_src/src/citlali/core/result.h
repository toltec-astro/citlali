#pragma once

using namespace netCDF;
using namespace netCDF::exceptions;

namespace lali {

class Result{
public:
    template<typename MC>
    void output(YamlConfig config, MC &Maps)
    {
        std::string filePath = config.get_str(std::tuple{"runtime","output_filepath"});
        writeMapsToNetCDF(config, Maps, filePath);
    }

    template<typename MC, typename S>
    void output(YamlConfig config, MC &Maps, S filePath)
    {
        writeMapsToNetCDF(config, Maps, filePath);
    }

    template <typename MC>
    auto writeMapsToNetCDF(YamlConfig, MC &Maps, std::string);
};

template <typename MC>
auto Result::writeMapsToNetCDF(YamlConfig config, MC &Maps, std::string filePath){

    int NC_ERR;
    try {
        //Create NetCDF file
        NcFile fo(filePath, NcFile::replace);
        //NcFile fo("/Users/mmccrackan/toltec/temp/citlali_simu_test.nc", NcFile::replace);

        //Create netCDF dimensions
        NcDim nrows = fo.addDim("nrows", Maps.nrows);
        NcDim ncols = fo.addDim("ncols", Maps.ncols);

        std::vector<NcDim> dims;
        dims.push_back(nrows);
        dims.push_back(ncols);

        for (Eigen::Index mc = 0; mc < Maps.map_count; mc++) {
            auto signalmapvar = "signal" + std::to_string(mc);
            auto weightmapvar = "weight" + std::to_string(mc);
            auto kernelmapvar = "kernel" + std::to_string(mc);
            auto intmapvar = "intMap" + std::to_string(mc);

            NcVar signalmapdata = fo.addVar(signalmapvar, ncDouble, dims);
            NcVar weightmapdata = fo.addVar(weightmapvar, ncDouble, dims);
            NcVar kernelmapdata = fo.addVar(kernelmapvar, ncDouble, dims);
            NcVar intmapdata = fo.addVar(intmapvar, ncDouble, dims);

            Eigen::MatrixXd signalmatrix
                = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                    Maps.signal.at(mc).data(), Maps.nrows, Maps.ncols);

            Eigen::MatrixXd weightmatrix
                = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                    Maps.weight.at(mc).data(), Maps.nrows, Maps.ncols);

            Eigen::MatrixXd kernelmatrix
                = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                    Maps.kernel.at(mc).data(), Maps.nrows, Maps.ncols);

            Eigen::MatrixXd intmapmatrix
                = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                    Maps.intMap.at(mc).data(), Maps.nrows, Maps.ncols);

            signalmatrix.transposeInPlace();
            weightmatrix.transposeInPlace();
            kernelmatrix.transposeInPlace();
            intmapmatrix.transposeInPlace();

            signalmapdata.putVar(signalmatrix.data());
            weightmapdata.putVar(weightmatrix.data());
            kernelmapdata.putVar(kernelmatrix.data());
            intmapdata.putVar(intmapmatrix.data());

            }

        fo.close();

    } catch (NcException& e) {
        e.what();
        return NC_ERR;
    }

    return 0;

}


}
