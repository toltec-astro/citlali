#pragma once

#include <string>

#include <netcdf>

#include <citlali/core/pipeline/raw_iir_filter_metadata.h>
#include <citlali/core/pipeline/string_join.h>
#include <citlali/core/utils/netcdf_io.h>


namespace citlali::pipeline {

#include <citlali/core/pipeline/reduction_config_weight_runtime_netcdf.h>
#include <citlali/core/pipeline/reduction_config_cleaning_netcdf.h>
#include <citlali/core/pipeline/reduction_config_fruitloops_netcdf.h>
#include <citlali/core/pipeline/reduction_config_learning_edge_netcdf.h>

}  // namespace citlali::pipeline
