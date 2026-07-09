#pragma once

#include <CCfits/CCfits>

#include <citlali/core/pipeline/map_fits_output_state.h>
#include <citlali/core/pipeline/output_path_state.h>
#include <citlali/core/pipeline/tod_output_state.h>
#include <citlali/core/utils/fits_io.h>

namespace citlali::pipeline {

struct ReductionOutputState {
    OutputPathState output_paths;
    TodOutputState tod_outputs;

    using map_fits_output_handle_t =
        fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>;

    MapFitsOutputState<map_fits_output_handle_t> map_fits_outputs;
};

}  // namespace citlali::pipeline
