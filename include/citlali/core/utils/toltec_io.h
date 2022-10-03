#pragma once

#include <string>
#include <map>

#include <Eigen/Core>

namespace engine_utils {

class toltecIO {
public:
    enum DataType {
        toltec = 0,
        apt = 1,
        ppt = 2,
    };

    enum ProdType {
        map = 0,
        psd = 1,
        hist = 2,
        timestream = 3,
    };

    enum FilterType {
        raw = 0,
        filtered = 1,
    };

    std::map<Eigen::Index, std::string> array_name_map = {
        {0,"a1100"},
        {1,"a1400"},
        {2,"a2000"}
    };

    std::map<Eigen::Index, double> array_fwhm_arcsec = {
        {0, 5.0},
        {1, 6.3},
        {2, 9.5}
    };

    template <toltecIO::DataType data_t, toltecIO::ProdType prod_t>
    std::string create_filename(const std::string filepath, const std::string, std::string, std::string, const bool);
};

template <toltecIO::DataType data_t, toltecIO::ProdType prod_t>
std::string toltecIO::create_filename(const std::string filepath, const std::string redu_type,
                                      std::string array_name, std::string obsnum, const bool simu_obs) {

    std::string filename = filepath;

    /* data type */
    if constexpr (data_t == toltec) {
        filename  = filename + "toltec_";
    }

    else if constexpr (data_t == apt) {
        filename  = filename + "apt_";
    }

    else if constexpr (data_t == ppt) {
        filename  = filename + "ppt_";
    }

    /* real data or simulation */
    if (!simu_obs) {
        filename = filename + "commissioning_";
    }

    else {
        filename = filename + "simu_";
    }

    if (!array_name.empty()) {
        filename = filename + array_name + "_";
    }

    filename = filename + redu_type;

    filename = filename + "_" + obsnum;

    /* prod type */
    if constexpr (prod_t == map) {

    }

    if constexpr (prod_t == psd) {
        filename = filename + "psd";
    }

    if constexpr (prod_t == hist) {
        filename = filename + "hist";
    }

    if constexpr (prod_t == timestream) {
        filename = filename + "timestream";
    }

    SPDLOG_INFO("filename {}", filename);
    return filename;
}

} // namespace engine utils
