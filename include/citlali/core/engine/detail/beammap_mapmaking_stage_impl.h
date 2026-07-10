#pragma once

// Beammap mapmaking stage implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/beammap_mapmaking_policy.h>
#include <citlali/core/pipeline/beammap_normalize_support_logging.h>
#include <citlali/core/pipeline/map_grouping_policy.h>
#include <citlali/core/pipeline/mapmaking_dispatch.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/stage_profile.h>

#include <sstream>

#include <citlali/core/engine/detail/beammap_map_population_impl.h>
#include <citlali/core/engine/detail/beammap_iteration_state_impl.h>
#include <citlali/core/engine/detail/beammap_mapmaking_pass_impl.h>
#include <citlali/core/engine/detail/beammap_source_aware_rtc_impl.h>
