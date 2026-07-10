#pragma once

// Beammap fit initialization implementation detail.
// Include this only after Beammap has been declared.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/engine/detail/beammap_fit_map_preparation_impl.h>
#include <citlali/core/engine/detail/beammap_fit_prior_compat_impl.h>
#include <citlali/core/engine/detail/beammap_previous_fit_init_impl.h>
#include <citlali/core/engine/detail/beammap_fit_init_selection_impl.h>
