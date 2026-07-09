#pragma once

#include <citlali/core/config/calibration_config.h>
#include <citlali/core/config/timestream_config.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include <citlali/core/utils/pointing.h>

namespace citlali::pipeline {

#include <citlali/core/pipeline/tod_output_selection_chunks.h>
#include <citlali/core/pipeline/tod_output_selection_mirror.h>
#include <citlali/core/pipeline/tod_output_selection_modes.h>
#include <citlali/core/pipeline/tod_output_selection_config_read.h>
#include <citlali/core/pipeline/tod_output_selection_rows.h>
#include <citlali/core/pipeline/tod_output_selection_source.h>

}  // namespace citlali::pipeline
