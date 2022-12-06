#pragma once

#include <string>
#include <map>

#include <Eigen/Core>

namespace engine_utils {

class toltecIO {
public:
    // data type
    enum DataType {
        toltec = 0,
        apt = 1,
        ppt = 2,
    };

    // product type
    enum ProdType {
        map = 0,
        noise = 1,
        psd = 2,
        hist = 3,
        timestream = 4,
    };

    // raw or filtered
    enum FilterType {
        raw = 0,
        filtered = 1,
    };

    std::map<Eigen::Index, std::string> array_name_map = {
        {0,"a1100"},
        {1,"a1400"},
        {2,"a2000"}
    };

    // expected fwhms
    std::map<Eigen::Index, double> array_fwhm_arcsec = {
        {0, 5.0},
        {1, 6.3},
        {2, 9.5}
    };

    template <toltecIO::DataType data_t, toltecIO::ProdType prod_t, toltecIO::FilterType filter_t>
    std::string create_filename(const std::string filepath, const std::string, std::string, std::string, const bool);
};

template <toltecIO::DataType data_t, toltecIO::ProdType prod_t, toltecIO::FilterType filter_t>
std::string toltecIO::create_filename(const std::string filepath, const std::string redu_type,
                                      std::string array_name, std::string obsnum, const bool simu_obs) {

    std::string filename = filepath;

    /* data type */
    if constexpr (data_t == toltec) {
        filename  = filename + "toltec";
    }

    else if constexpr (data_t == apt) {
        filename  = filename + "apt";
    }

    else if constexpr (data_t == ppt) {
        filename  = filename + "ppt";
    }

    /* real data or simulation */
    if (!simu_obs) {
        filename = filename + "_commissioning";
    }

    else {
        filename = filename + "_simu";
    }

    if (!array_name.empty()) {
        filename = filename + "_" + array_name;
    }

    if (!redu_type.empty()) {
        filename = filename + "_" + redu_type;
    }

    if (!obsnum.empty()) {
        filename = filename + "_" + obsnum;
    }

    /* prod type */
    if constexpr (prod_t == noise) {
        filename = filename + "_noise";
    }

    if constexpr (prod_t == psd) {
        filename = filename + "_psd";
    }

    if constexpr (prod_t == hist) {
        filename = filename + "_hist";
    }

    if constexpr (prod_t == timestream) {
        filename = filename + "_timestream";
    }

    if constexpr (filter_t == filtered) {
        filename = filename + "_filtered";
    }


    return filename;
}

} // namespace engine utils
