#pragma once

#include <citlali/core/config/timestream_config.h>

#include <iomanip>
#include <sstream>
#include <string>

namespace citlali::pipeline {

inline std::string fruit_loop_map_dir(const std::string &base_dir,
                                      const std::string &fruit_loops_type,
                                      const std::string &obsnum) {
    if (citlali::config::is_obsnum_raw_fruit_loops_type(fruit_loops_type)) {
        return base_dir + "/" + obsnum + "/raw/";
    }
    if (citlali::config::is_obsnum_filtered_fruit_loops_type(
            fruit_loops_type)) {
        return base_dir + "/" + obsnum + "/filtered/";
    }
    if (citlali::config::is_coadd_raw_fruit_loops_type(fruit_loops_type)) {
        return base_dir + "/coadded/raw/";
    }
    if (citlali::config::is_coadd_filtered_fruit_loops_type(
            fruit_loops_type)) {
        return base_dir + "/coadded/filtered/";
    }
    return "";
}

inline std::string previous_fruit_loop_reduction_dir_name(int redu_dir_num) {
    std::stringstream ss_redu_dir_num_i;
    ss_redu_dir_num_i << std::setfill('0') << std::setw(2) << redu_dir_num - 1;
    return "redu" + ss_redu_dir_num_i.str();
}

inline std::string previous_fruit_loop_map_dir(
    const std::string &output_dir, int redu_dir_num,
    const std::string &fruit_loops_type, const std::string &obsnum) {
    return fruit_loop_map_dir(
        output_dir + "/" + previous_fruit_loop_reduction_dir_name(redu_dir_num),
        fruit_loops_type, obsnum);
}

}  // namespace citlali::pipeline
