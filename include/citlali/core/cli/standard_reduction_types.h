#pragma once

#include <citlali/core/cli/tod_processor_selection.h>
#include <citlali/core/engine/beammap.h>
#include <citlali/core/engine/lali.h>
#include <citlali/core/engine/pointing.h>

namespace citlali::cli {

using StandardScienceTodProcessor = ::TimeOrderedDataProc<::Lali>;
using StandardPointingTodProcessor = ::TimeOrderedDataProc<::Pointing>;
using StandardBeammapTodProcessor = ::TimeOrderedDataProc<::Beammap>;
using StandardTodProcessorVariant =
    TodProcessorVariant<StandardScienceTodProcessor,
                        StandardPointingTodProcessor,
                        StandardBeammapTodProcessor>;

}  // namespace citlali::cli
