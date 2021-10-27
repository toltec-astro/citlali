#pragma once

#include <string>
#include <vector>
#include <map>

#include <citlali/core/utils/ecsv_io.h>
#include <citlali/core/utils/constants.h>

struct ToltecCalib {
    // name and scale
    std::map<std::string, double> header_keys = {
        {"nw", 1},
        {"array", 1},
        {"flxscale", 1},
        {"x_t", DEG_TO_ASEC},
        {"y_t", DEG_TO_ASEC},
        {"a_fwhm", DEG_TO_ASEC},
        {"b_fwhm", DEG_TO_ASEC}
    };
};


class Calib: public ToltecCalib {
public:
    using array_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;
    using det_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;

    // indices of the arrays
    array_indices_t array_indices;
    // indices of the detectors
    det_indices_t det_indices;

    // total number of detectors
    Eigen::Index ndet;
    // apt table columns
    std::map<std::string, Eigen::VectorXd> calib_data;

    Eigen::VectorXd responsivity;
    Eigen::VectorXd sensitivity;

    void get_calib(const std::string &filepath) {
        // read in the apt table
        auto [table, header] = get_matrix_from_ecsv(filepath);
        // set the number of detectors
        ndet = table.rows();

        // loop through the apt table header keys and populate calib_data
        for (auto const& pair: header_keys) {
            auto it = find(header.begin(), header.end(), pair.first);
            if (it != header.end()) {
                int index = it - header.begin();
                calib_data[pair.first] = table.col(index)*pair.second;
            }
        }

        /* TEMP */
        responsivity.setOnes(ndet);
        sensitivity.setOnes(ndet);
    }
};

