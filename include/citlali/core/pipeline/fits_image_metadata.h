#pragma once

#include <Eigen/Core>
#include <fmt/format.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string>
#include <tuple>

#include <citlali/core/pipeline/required_output_failure.h>

namespace citlali::pipeline {

#include <citlali/core/pipeline/fits_image_write_slots.h>
#include <citlali/core/pipeline/fits_image_units_kernels.h>
#include <citlali/core/pipeline/fits_image_hdu_names_wcs.h>
#include <citlali/core/pipeline/fits_image_products.h>
#include <citlali/core/pipeline/fits_image_metadata_keys.h>

}  // namespace citlali::pipeline
