#pragma once

#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

#include <citlali/core/config/config_value.h>
#include <citlali/core/pipeline/output_netcdf_metadata.h>
#include <citlali/core/pipeline/ptcdiag_netcdf.h>


namespace citlali::pipeline {

#include <citlali/core/pipeline/stats_netcdf_paths.h>
#include <citlali/core/pipeline/stats_netcdf_layout_vars.h>
#include <citlali/core/pipeline/stats_netcdf_eigenvalues.h>
#include <citlali/core/pipeline/stats_netcdf_file_outputs.h>

}  // namespace citlali::pipeline
