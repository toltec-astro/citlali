#pragma once

#include <Eigen/Dense>

#include <chrono>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <iomanip>

#include <tula/filesystem.h>
#include <tula/logging.h>
#include <tula/config/flatconfig.h>
#include <tula/config/yamlconfig.h>

#include <citlali_config/gitversion.h>
#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

namespace fs = std::filesystem;

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
        noise_raw = 2,
        noise_filtered = 3,
        noise_raw_psd = 4,
        noise_filtered_psd = 5,
        hist = 6,
        raw_hist = 7,
        filtered_hist = 8,
        noise_raw_hist = 9,
        noise_filtered_hist = 10,
        psd = 11,
        raw_psd = 12,
        filtered_psd = 13,
        timestream = 14,
        no_prod_type = 15
    };

    enum ObsNum {
        obsnum_true = 0,
        obsnum_false = 1
    };

    // map between index and array name (temp)
    std::map<int, std::string> name_keys {};
        //{0, "a1100"},
        //{1, "a1400"},
        //{2, "a2000"},};

    // map between index and array name (temp)
    std::map<int, std::string> polarized_name_keys {};
        /*{0, "a1100_I"},
        {1, "a1100_Q"},
        {2, "a1100_U"},
        {3, "a1400_I"},
        {4, "a1400_Q"},
        {5, "a1400_U"},
        {6, "a2000_I"},
        {7, "a2000_Q"},
        {8, "a2000_U"}
    };*/

    // map between index and beam area
    std::map<int, double> barea_keys {};

    // map between index and beam fwhm
    std::map<int, double> bfwhm_keys {};

    std::map<int, double> array_freqs {
        //{0,A1100_FREQ},
        //{1,A1400_FREQ},
        //{2,A2000_FREQ}
    };

    // header keys for pointing fit ecsv
    std::vector<std::string> apt_header {
        {"amp"},
        {"amp_err"},
        {"x_t"},
        {"x_t_err"},
        {"y_t"},
        {"y_t_err"},
        {"a_fwhm"},
        {"a_fwhm_err"},
        {"b_fwhm"},
        {"b_fwhm_err"},
        {"angle"},
        {"angle_err"},
    };

    // header keys for beammap apt table
    std::vector<std::string> beammap_apt_header {
        {"uid"},
        {"array"},
        {"nw"},
        {"fg"},
        {"pg"},
        {"ori"},
        {"responsivity"},
        {"flxscale"},
        {"sens"},
        {"derot_elev"},
        {"amp"},
        {"amp_err"},
        {"x_t"},
        {"x_t_err"},
        {"y_t"},
        {"y_t_err"},
        {"a_fwhm"},
        {"a_fwhm_err"},
        {"b_fwhm"},
        {"b_fwhm_err"},
        {"angle"},
        {"angle_err"},
        {"converge_iter"},
        {"flag"},
        {"sig2noise"}};

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

    // colnames for minkaski timestream file
    std::vector<std::string> minkaski_colnames {
        {"PIXID"},
        {"DX"},
        {"DY"},
        {"ELEV"},
        {"TIME"},
        {"FNU"},
        {"UFNU"}
    };

    std::map<std::string,std::string> toast_colnames {
        {"PIXID","PIXID"},
        {"DX","DX"},
        {"DY","DY"},
        {"ELEV","ELEV"},
        {"TIME","TIME"},
        {"FNU","FNU"},
        {"UFNU","UFNU"}
    };

    std::vector<std::string> minkaski_colform {
        {"f4.2"},
        {"f4.2"},
        {"f4.2"},
        {"f4.2"},
        {"f4.2"},
        {"f4.2"},
        {"f4.2"}
    };

    std::vector<double> nw_sizes {
        684,
        522,
        558,
        564,
        556,
        510,
        618,
        676,
        588,
        590,
        680,
        544,
        628,
        };

    std::vector<std::string> minkaski_colunits {
        {""},
        {""},
        {""},
        {""},
        {""},
        {""},
        {""}
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

        if constexpr (project_id == commissioning) {
            filepath = filepath + "commissioning_";
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
            filepath = filepath + "raw";
        }

        if constexpr (prod_type == filtered) {
            filepath = filepath + "filtered";
        }

        if constexpr (prod_type == noise_raw) {
            filepath = filepath + "noise_raw";
        }

        if constexpr (prod_type == noise_filtered) {
            filepath = filepath + "noise_filtered";
        }

        if constexpr (prod_type == noise_raw_psd) {
            filepath = filepath + "noise_raw_psd";
        }

        if constexpr (prod_type == noise_filtered_psd) {
            filepath = filepath + "noise_filtered_psd";
        }

        if constexpr (prod_type == hist) {
            filepath = filepath + "hist_";
        }

        if constexpr (prod_type == raw_hist) {
            filepath = filepath + "raw_hist";
        }

        if constexpr (prod_type == filtered_hist) {
            filepath = filepath + "filtered_hist";
        }

        if constexpr (prod_type == noise_raw_hist) {
            filepath = filepath + "noise_raw_hist";
        }

        if constexpr (prod_type == noise_filtered_hist) {
            filepath = filepath + "noise_filtered_hist";
        }

        if constexpr (prod_type == psd) {
            filepath = filepath + "psd_";
        }

        if constexpr (prod_type == raw_psd) {
            filepath = filepath + "raw_psd";
        }

        if constexpr (prod_type == filtered_psd) {
            filepath = filepath + "filtered_psd";
        }

        if constexpr (prod_type == timestream) {
            filepath = filepath + "timestream_";
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

    void setup_output_directory(std::string filepath, std::string dname) {
        if (!fs::exists(fs::status(filepath + dname))) {
            fs::create_directories(filepath + dname);
        }
        else {
            SPDLOG_WARN("directory {} already exists", filepath + dname);
        }
    }

    void make_index_file(std::string filepath) {
        std::set<fs::path> sorted_by_name;

        for (auto &entry : fs::directory_iterator(filepath))
            sorted_by_name.insert(entry);

        YAML::Node node;
        node["description"].push_back("citlali data products");
        node["date"].push_back(engine_utils::current_date_time());
        node["version"].push_back(CITLALI_GIT_VERSION);

        for (const auto & entry : sorted_by_name) {
            std::string path_string{entry.generic_string()};
            if (fs::is_directory(entry)) {
                make_index_file(path_string);
            }
            node["files"].push_back(path_string.substr(path_string.find_last_of("/") + 1));
        }
        std::ofstream fout(filepath + "/index.yaml");
        fout << node;
    }

};
