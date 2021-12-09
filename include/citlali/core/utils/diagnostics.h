#pragma once

#include <Eigen/Core>
#include <netcdf>

#include <tula/filename.h>
#include <tula/logging.h>

#include <citlali/core/utils/netcdf_io.h>
#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/timestream/timestream.h>

using namespace netCDF;
using netCDF::NcDim;
using netCDF::NcFile;
using netCDF::NcType;
using netCDF::NcVar;
using namespace netCDF::exceptions;

namespace timestream {

class Diagnostics {
public:
    std::unordered_map<std::string_view, std::string> _ = {
        {"xs", "Data.Kids.xs"},
        {"ntimes", "ntimes"},
        {"ntones", "ntones"},
    };

    // pointer to netcdf file
    std::unique_ptr<NcFile> fo;
    void setup(std::string filepath) {
        fo.reset( new NcFile(filepath, NcFile::write, NcFile::classic));
        //NcDim d_ntimes = fo->addDim(_["ntimes"]);
        //NcDim d_ntones = fo->addDim(_["ntones"], TULA_SIZET(ntones));

    }

    template<TCDataKind TCType>
    void write_scan(TCData<TCType, Eigen::MatrixXd> &in) {

    }

    template<TCDataKind TCType>
    void scan_stats(TCData<TCType, Eigen::MatrixXd> &);

};

template<TCDataKind TCType>
void Diagnostics::scan_stats(TCData<TCType, Eigen::MatrixXd> &in) {
    SPDLOG_INFO("mean {}", in.scans.data.mean());

}
} // namespace

