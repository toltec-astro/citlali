#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

#include <citlali/core/config/config_value.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/reduction_config_netcdf.h>


namespace citlali::pipeline {

#include <citlali/core/pipeline/ptcdiag_paths_dims.h>
#include <citlali/core/pipeline/ptcdiag_detector_metadata_outputs.h>
#include <citlali/core/pipeline/ptcdiag_network_blocks.h>
#include <citlali/core/pipeline/ptcdiag_file_config_network_outputs.h>
#include <citlali/core/pipeline/ptcdiag_tod_optional_diag.h>

}  // namespace citlali::pipeline
