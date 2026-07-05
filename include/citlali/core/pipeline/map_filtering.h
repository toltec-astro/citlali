#pragma once

#include <cstddef>
#include <cstdlib>
#include <string>

#include <Eigen/Core>
#include <tula/logging.h>

#include <citlali/core/mapmaking/edge_guard_state.h>
#include <citlali/core/mapmaking/map.h>


namespace citlali::pipeline {

#include <citlali/core/pipeline/map_filtering_types.h>
#include <citlali/core/pipeline/map_filtering_setup_outputs.h>
#include <citlali/core/pipeline/map_filtering_template_noise.h>
#include <citlali/core/pipeline/map_filtering_output_lifecycle.h>
#include <citlali/core/pipeline/map_filtering_loop.h>

}  // namespace citlali::pipeline
