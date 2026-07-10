#pragma once

#include <citlali/core/config/coadd_config.h>
#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>
#include <citlali/core/pipeline/mapmaking_config_policy.h>

#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <Eigen/Core>


namespace citlali::pipeline {

#include <citlali/core/pipeline/mapmaking_config_read_core.h>
#include <citlali/core/pipeline/mapmaking_config_read_noise.h>
#include <citlali/core/pipeline/mapmaking_config_read_outputs.h>
#include <citlali/core/pipeline/mapmaking_config_read_methods.h>
#include <citlali/core/pipeline/mapmaking_config_read_noise_products.h>

}  // namespace citlali::pipeline
