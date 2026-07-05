#pragma once

#include <citlali/core/pipeline/mapdiag_edge_guard.h>
#include <citlali/core/pipeline/fits_image_metadata.h>
#include <citlali/core/pipeline/mapdiag_labels.h>
#include <citlali/core/pipeline/mapdiag_netcdf.h>
#include <citlali/core/pipeline/mapdiag_observation_weight.h>
#include <citlali/core/pipeline/mapdiag_stats.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>


namespace citlali::pipeline {

#include <citlali/core/pipeline/mapdiag_workspace_map_storage.h>
#include <citlali/core/pipeline/mapdiag_workspace_observation_storage.h>
#include <citlali/core/pipeline/mapdiag_workspace_label_stats.h>
#include <citlali/core/pipeline/mapdiag_workspace_outlier_collect.h>
#include <citlali/core/pipeline/mapdiag_workspace_learning_emit.h>
#include <citlali/core/pipeline/mapdiag_workspace_signal_diag.h>

}  // namespace citlali::pipeline
