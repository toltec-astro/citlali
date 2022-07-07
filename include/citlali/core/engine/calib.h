#pragma once

#include <string>
#include <vector>
#include <map>
#include <netcdf>

#include <citlali/core/utils/ecsv_io.h>
#include <citlali/core/utils/netcdf_io.h>
#include <citlali/core/utils/constants.h>

struct ToltecCalib {
    // name and scale
    std::map<std::string, double> header_keys = {
        {"uid", 1},
        {"nw", 1},
        {"array", 1},
        {"flxscale", 1},
        {"x_t", 1},
        {"y_t", 1},
        {"a_fwhm", 1},
        {"b_fwhm", 1},
        {"fg",1},
        {"pg", 1},
        {"ori",1},
        {"responsivity",1},
        {"flag",1},
        {"sens",1},
        {"sig2noise",1}
    };

    bool run_hwp;
};


class Calib: public ToltecCalib {
public:
    using array_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;
    using det_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;

    // indices of the arrays and networks
    array_indices_t array_indices, nw_indices;
    // vector for arrays and networks
    std::map<std::string, Eigen::Index> arrays, nws;
    // indices of the detectors
    det_indices_t det_indices;

    // total number of detectors
    Eigen::Index ndet;
    // apt table columns
    std::map<std::string, Eigen::VectorXd> calib_data;

    Eigen::VectorXd responsivity;
    //Eigen::VectorXd sensitivity;

    // vector for half-wave plate
    Eigen::VectorXd hwp;

    void get_calib(const std::string &filepath) {
        // read in the apt table
        auto [table, header] = get_matrix_from_ecsv(filepath);
        SPDLOG_INFO("table {}",table);
        // set the number of detectors
        ndet = table.rows();

        // loop through the apt table header keys and populate calib_data
        for (auto const& pair: header_keys) {
            auto it = find(header.begin(), header.end(), pair.first);
            if (it != header.end()) {
                int index = it - header.begin();
                calib_data[pair.first] = table.col(index)*pair.second;
		SPDLOG_INFO("{} {}",pair.first, calib_data[pair.first]);
            }
        }

        /* TEMP */
        responsivity.setOnes(ndet);
    }

    void get_hwp(std::string &filepath) {
        using namespace netCDF;
        using namespace netCDF::exceptions;

        try {
            // get hwp file
            NcFile fo(filepath, NcFile::read, NcFile::classic);
            SPDLOG_INFO("read in hwp netCDF file {}", filepath);
            auto vars = fo.getVars();

            // check if hwp is enabled
            vars.find("Header.Hwp.RotatorEnabled")->second.getVar(&run_hwp);
            SPDLOG_INFO("Header.Hwp.RotatorEnabled {}", run_hwp);

            // get hwp signal
            Eigen::Index npts = vars.find("Data.Hwp.")->second.getDim(0).getSize();
            hwp.resize(npts);

            vars.find("Data.Hwp.")->second.getVar(hwp.data());

            fo.close();

        } catch (NcException &e) {
            SPDLOG_ERROR("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", filepath)};
        }
    }

    void check_missing() {
        for (Eigen::Index i=0; i<calib_data["uid"].size(); i++) {
            std::ostringstream os;

            os << calib_data["uid"](i);
            std::string digits = os.str();

        }
    }
};

