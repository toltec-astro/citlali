#pragma once

using namespace netCDF;
using namespace netCDF::exceptions;

namespace lali {

class Result{
public:
    template<typename MC>
    void output(std::shared_ptr<YamlConfig> config, MC &Maps)
    {
        std::string filePath = config->get_typed<std::string>("output_filepath");
        writeMapsToNetCDF(config, Maps, filePath);
    }

    template<typename MC, typename S>
    void output(std::shared_ptr<YamlConfig> config, MC &Maps, S filePath)
    {
        writeMapsToNetCDF(config, Maps, filePath);
    }

    template <typename MC>
    auto writeMapsToNetCDF(std::shared_ptr<YamlConfig>, MC &Maps, std::string);
};

template <typename MC>
auto Result::writeMapsToNetCDF(std::shared_ptr<YamlConfig> config, MC &Maps, std::string filePath){

    int NC_ERR;
    try {
        //Create NetCDF file
        NcFile fo(filePath, NcFile::replace);

        //Create netCDF dimensions
        NcDim nrows = fo.addDim("nrows", Maps.nrows);
        NcDim ncols = fo.addDim("ncols", Maps.ncols);

        std::vector<NcDim> dims;
        dims.push_back(nrows);
        dims.push_back(ncols);

        auto signalmapvar = "signal";
        auto weightmapvar = "weight";
        auto kernelmapvar = "kernel";

        NcVar signalmapdata = fo.addVar(signalmapvar, ncDouble, dims);
        NcVar weightmapdata = fo.addVar(weightmapvar, ncDouble, dims);
        NcVar kernelmapdata = fo.addVar(kernelmapvar, ncDouble, dims);

        Eigen::MatrixXd signalmatrix
            = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                Maps.signal.data(), Maps.nrows, Maps.ncols);

        Eigen::MatrixXd weightmatrix
            = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                Maps.weight.data(), Maps.nrows, Maps.ncols);

        Eigen::MatrixXd kernelmatrix
            = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                Maps.kernel.data(), Maps.nrows, Maps.ncols);

        signalmatrix.transposeInPlace();
        weightmatrix.transposeInPlace();
        kernelmatrix.transposeInPlace();

        signalmapdata.putVar(signalmatrix.data());
        weightmapdata.putVar(weightmatrix.data());
        kernelmapdata.putVar(kernelmatrix.data());

        fo.close();

    } catch (NcException& e) {
        e.what();
        return NC_ERR;
    }

    return 0;

}


}
