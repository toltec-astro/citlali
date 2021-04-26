#pragma once

//#include <CCfits/CCfits>


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

    template <typename MC, typename MT>
    void setupVars(MC &Maps, MT &map, std::string varname, NcFile &fo, std::vector<NcDim> dims) {

        for (Eigen::Index mc = 0; mc < Maps.map_count; mc++) {
            auto var = varname + std::to_string(mc);
            NcVar mapVar = fo.addVar(var, ncDouble, dims);
            Eigen::MatrixXd rowMajorMap
                = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                    map.at(mc).data(), map.at(mc).rows(), map.at(mc).cols());
            rowMajorMap.transposeInPlace();
            mapVar.putVar(rowMajorMap.data());
        }
    }

};

template <typename MC>
auto Result::writeMapsToNetCDF(YamlConfig config, MC &Maps, std::string filePath){

    int NC_ERR;
    try {
        //Create NetCDF file
        NcFile fo(filePath, NcFile::replace);

        auto grouping = config.get_str(std::tuple{"map","grouping"});

        //Create netCDF dimensions
        NcDim nrows = fo.addDim("nrows", Maps.nrows);
        NcDim ncols = fo.addDim("ncols", Maps.ncols);

        std::vector<NcDim> dims;
        dims.push_back(nrows);
        dims.push_back(ncols);

        setupVars(Maps, Maps.signal, "signal", fo, dims);
        setupVars(Maps, Maps.weight, "weight", fo, dims);

        if (std::strcmp("beammap", grouping.c_str()) == 1) {
            setupVars(Maps, Maps.kernel, "kernel", fo, dims);
            setupVars(Maps, Maps.intMap, "intMap", fo, dims);
        }


        fo.close();

    } catch (NcException& e) {
        e.what();
        return NC_ERR;
    }

    return 0;

}


}
