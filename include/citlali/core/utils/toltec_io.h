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
        source = 3
    };

    // product type
    enum ProdType {
        map = 0,
        noise = 1,
        psd = 2,
        hist = 3,
        rtc_timestream = 4,
        ptc_timestream = 5,
        stats = 6
    };

    // raw or filtered
    enum FilterType {
        raw = 0,
        filtered = 1,
    };

    std::map<Eigen::Index, Eigen::Index> nw_to_array_map = {
        {0, 0},
        {1, 0},
        {2, 0},
        {3, 0},
        {4, 0},
        {5, 0},
        {6, 0},
        {7, 1},
        {8, 1},
        {9, 1},
        {10, 1},
        {11, 2},
        {12, 2},
    };

    // map from array index to array name
    std::map<Eigen::Index, std::string> array_name_map = {
        {0,"a1100"},
        {1,"a1400"},
        {2,"a2000"},
    };

    // map from array index to array wavelength
    std::map<Eigen::Index, double> array_wavelength_map = {
        {0,1.1},
        {1,1.4},
        {2,2.0},
    };

    // map from array index to array frequencies
    std::map<Eigen::Index, double> array_freq_map = {
        {0,c_m_s/(1.1/1000)},
        {1,c_m_s/(1.4/1000)},
        {2,c_m_s/(2.0/1000)},
    };

    // expected fwhms
    std::map<Eigen::Index, double> array_fwhm_arcsec = {
        {0, 5.0},
        {1, 6.3},
        {2, 9.5},
    };    

    template <toltecIO::DataType data_t, toltecIO::ProdType prod_t, toltecIO::FilterType filter_t>
    std::string create_filename(const std::string, const std::string, std::string, std::string, const bool);
};

template <toltecIO::DataType data_t, toltecIO::ProdType prod_t, toltecIO::FilterType filter_t>
std::string toltecIO::create_filename(const std::string filepath, const std::string redu_type,
                                      std::string array_name, std::string obsnum, const bool simu_obs) {
    std::string filename = filepath;

    // data type
    if constexpr (data_t == toltec) filename += "toltec";
    else if constexpr (data_t == apt) filename += "apt";
    else if constexpr (data_t == ppt) filename += "ppt";
    else if constexpr (data_t == source) filename += "source";

    // real data or simulation
    filename += simu_obs ? "_simu" : "_commissioning";

    // add array name, redu_type, and obsnum if they are not empty
    if (!array_name.empty()) filename += "_" + array_name;
    if (!redu_type.empty()) filename += "_" + redu_type;
    if (!obsnum.empty()) filename += "_" + obsnum;

    // product type
    if constexpr (prod_t == noise) filename += "_noise";
    else if constexpr (prod_t == psd) filename += "_psd";
    else if constexpr (prod_t == hist) filename += "_hist";
    else if constexpr (prod_t == rtc_timestream) filename += "_rtc_timestream";
    else if constexpr (prod_t == ptc_timestream) filename += "_ptc_timestream";
    else if constexpr (prod_t == stats) filename += "_stats";

    if constexpr (filter_t == filtered) {
        filename += "_filtered";
    }

    // filtered map or add pipeline to maps or noise maps
    if constexpr (prod_t == map || prod_t == noise) {
        filename += "_citlali";
    }

    return filename;
}

} // namespace engine utils
