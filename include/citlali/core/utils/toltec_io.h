#pragma once

#include <Eigen/Dense>

#include <chrono>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <iomanip>

#include <citlali_config/gitversion.h>
#include <citlali/core/utils/constants.h>

struct ToltecIO {
    // instrument filename
    enum DataType {
        toltec = 0,
        muscat = 1,
        apt = 2,
        ppt = 3,
        no_data_type = 4,
    };

    // project id filename
    enum ProjectID {
        lmt = 0,
        commissioning = 1,
        engineering = 2,
        simu = 3,
        no_project_id = 4,
    };

    // obs type filename
    enum ObsType {
        science = 0,
        pointing = 1,
        beammap = 2,
        wyatt = 3,
        no_obs_type = 4
    };

    // data product type
    enum ProdType {
        raw = 0,
        filtered = 1,
        noise = 2,
        hist = 3,
        psd = 4,
        no_prod_type = 5
    };

    enum ObsNum {
        obsnum_true = 0,
        obsnum_false = 1
    };

    // map between index and array name (temp)
    std::map<int, std::string> name_keys {
        {0, "a1100"},
        {1, "a1400"},
        {2, "a2000"},};

    // map between index and array name (temp)
    std::map<int, double> barea_keys {
        {0, A1100_BAREA},
        {1, A1400_BAREA},
        {2, A2000_BAREA},};

    // header keys for pointing fit ecsv
    std::vector<std::string> apt_header {
        {"amp"},
        {"x_t"},
        {"y_t"},
        {"a_fwhm"},
        {"b_fwhm"},
        {"angle"}};

    // header keys for beammap apt table
    std::vector<std::string> beammap_apt_header {
        {"array"},
        {"nw"},
        {"flxscale"},
        {"amp"},
        {"x_t"},
        {"y_t"},
        {"a_fwhm"},
        {"b_fwhm"},
        {"angle"},
        {"converge_iter"}};

    // header keys for output FITS files
    std::map<std::string, std::string> fits_header_keys {
        {"TELESCOP","LMT"},
        {"INSTRUME","TOLTEC"},
        {"CREATOR",CITLALI_GIT_VERSION},
        {"OBJECT","N/A"},
        {"OBSERVER","N/A"},
        {"PROPOSAL","N/A"},
        //{"OBS_ID","N/A"},
        //{"DATE","N/A"},
        {"DATE-OBS","N/A"},
        //{"WAV","N/A"}
};

    template <DataType data_type, ProjectID project_id,
              ObsType obs_type, ProdType prod_type, ObsNum obs_num>
    std::string setup_filepath(std::string filepath, int obsnum, int array_name) {

        // append data type to filename
        if constexpr (data_type == toltec) {
            filepath = filepath + "toltec_";
        }

        else if constexpr (data_type == apt) {
            filepath = filepath + "apt_";
        }

        else if constexpr (data_type == ppt) {
            filepath = filepath + "ppt_";
        }

        if constexpr (project_id == simu) {
            filepath = filepath + "simu_";
        }

        // append array name to filename
        if (array_name >= 0) {
            filepath = filepath + name_keys[array_name] + "_";
        }

        // append observation type to filename
        if constexpr (obs_type == science) {
            filepath = filepath + "science_";
        }

        if constexpr (obs_type == pointing) {
            filepath = filepath + "pointing_";
        }

        if constexpr (obs_type == beammap) {
            filepath = filepath + "beammap_";
        }

        // append product type
        if constexpr (prod_type == raw) {
            filepath = filepath + "raw_";
        }

        if constexpr (prod_type == filtered) {
            filepath = filepath + "filtered_";
        }

        if constexpr (prod_type == hist) {
            filepath = filepath + "hist_";
        }

        if constexpr (prod_type == psd) {
            filepath = filepath + "psd_";
        }


        if constexpr (obs_num == obsnum_true) {
            // append obsnum to filename
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(6) << obsnum;

            filepath = filepath + ss.str();
        }

        std::stringstream ss2;

        // append unix time stamp (in seconds) to filename
        const auto p1 = std::chrono::system_clock::now();
        ss2 <<  std::chrono::duration_cast<std::chrono::seconds>(
                           p1.time_since_epoch()).count();

        std::string unix_time = ss2.str();
        // filepath = filepath + unix_time;

        // return the updated filepath
        return filepath;
    }

};
