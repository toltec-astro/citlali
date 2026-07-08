#pragma once

#include <ostream>
#include <string>
#include <string_view>

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>


namespace citlali::pipeline {

#include <citlali/core/pipeline/summary_log_paths_identity.h>
#include <citlali/core/pipeline/summary_log_chunk_status.h>
#include <citlali/core/pipeline/summary_log_chunk_quality.h>
#include <citlali/core/pipeline/summary_log_chunk_writer.h>
#include <citlali/core/pipeline/summary_log_map_writer.h>

}  // namespace citlali::pipeline
