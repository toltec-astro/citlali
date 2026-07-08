#pragma once

#include <cstddef>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

#include <citlali/core/config/config_value.h>
#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/pipeline/phdu_beammap.h>
#include <citlali/core/pipeline/phdu_observation_metadata.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/netcdf_io.h>

namespace citlali::pipeline {

#include <citlali/core/pipeline/tod_output_layout.h>
#include <citlali/core/pipeline/tod_output_identity_metadata.h>
#include <citlali/core/pipeline/tod_output_reduction_metadata.h>
#include <citlali/core/pipeline/tod_output_data_vars.h>
}  // namespace citlali::pipeline
