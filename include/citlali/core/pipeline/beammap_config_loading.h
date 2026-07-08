#pragma once

#include <citlali/core/config/config_value.h>
#include <citlali/core/config/beammap_config.h>

#include <cstddef>
#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>


namespace citlali::pipeline {

#include <citlali/core/pipeline/beammap_config_core_loading.h>
#include <citlali/core/pipeline/beammap_config_fitting_flagging.h>
#include <citlali/core/pipeline/beammap_config_split_outputs.h>
#include <citlali/core/pipeline/beammap_config_priors_loading.h>
#include <citlali/core/pipeline/beammap_config_tod_mirror.h>

}  // namespace citlali::pipeline
