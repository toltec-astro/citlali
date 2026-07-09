#pragma once

#include <citlali_config/config.h>

#include <citlali/core/engine/calib.h>
#include <citlali/core/engine/diagnostics.h>
#include <citlali/core/engine/telescope.h>
#include <citlali/core/mapmaking/jinc_mm.h>
#include <citlali/core/mapmaking/map.h>
#include <citlali/core/mapmaking/ml_mm.h>
#include <citlali/core/mapmaking/naive_mm.h>
#if defined(CITLALI_USE_WIENER_FILTER_OMP)
#include <citlali/core/mapmaking/wiener_filter_omp.h>
#else
#include <citlali/core/mapmaking/wiener_filter.h>
#endif
#include <citlali/core/timestream/ptc/ptcproc.h>
#include <citlali/core/timestream/rtc/rtcproc.h>
#include <citlali/core/utils/fitting.h>
#include <citlali/core/utils/toltec_io.h>

struct ObservationComponents {
    engine::Calib calib;
    engine::Telescope telescope;
    engine_utils::toltecIO toltec_io;
    engine::Diagnostics diagnostics;
    engine_utils::mapFitter map_fitter;
};

struct TimestreamComponents {
    timestream::RTCProc rtcproc;
    timestream::PTCProc ptcproc;
};

struct MapmakingComponents {
    mapmaking::MapBuffer omb{"omb"}, cmb{"cmb"};
    mapmaking::NaiveMapmaker naive_mm;
    mapmaking::JincMapmaker jinc_mm;
    mapmaking::MLMapmaker ml_mm;
    mapmaking::WienerFilter wiener_filter;
};

struct ReductionComponents : public ObservationComponents,
                             public TimestreamComponents,
                             public MapmakingComponents {};
