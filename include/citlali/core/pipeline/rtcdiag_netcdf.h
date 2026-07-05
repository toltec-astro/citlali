#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

#include <citlali/core/pipeline/reduction_config_netcdf.h>
#include <citlali/core/utils/netcdf_io.h>


namespace citlali::pipeline {

#include <citlali/core/pipeline/rtcdiag_layout_config.h>
#include <citlali/core/pipeline/rtcdiag_scan_summary.h>
#include <citlali/core/pipeline/rtcdiag_detector_outputs.h>
#include <citlali/core/pipeline/rtcdiag_network_outputs.h>
#include <citlali/core/pipeline/rtcdiag_impulsive_capture.h>
#include <citlali/core/pipeline/rtcdiag_tod_stream.h>

}  // namespace citlali::pipeline
