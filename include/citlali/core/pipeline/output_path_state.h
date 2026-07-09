#pragma once

#include <map>
#include <string>

namespace citlali::pipeline {

struct OutputPathState {
    std::string redu_dir_name;
    int redu_dir_num = 0;
    std::string obsnum_dir_name;
    std::string coadd_dir_name;
    std::map<std::string, std::string> tod_filename;
    std::string rtcdiag_filename;
    std::string ptcdiag_filename;
};

}  // namespace citlali::pipeline
