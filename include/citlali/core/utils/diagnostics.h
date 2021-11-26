#pragma once

#include <Eigen/Core>
#include <netcdf>

#include <citlali/core/utils/netcdf_io.h>
#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/timestream/timestream.h>

using namespace netCDF;
using namespace netCDF::exceptions;

namespace timestream {

class Diagnostics {
public:
    // pointer to netcdf file
    std::unique_ptr<NcFile> fo;
    void setup(std::string filepath) {
        fo.reset( new NcFile(filepath, NcFile::write, NcFile::classic));

    }

    template<class TCType>
    void write_scan(TCData<TCType, Eigen::MatrixXd> &in) {

    }

    void scan_stats();

};

void Diagnostics::scan_stats() {

}
} // namespace

