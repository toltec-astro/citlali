#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/rawobs_adc_snap.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::get_adc_snap_from_files(const RawObs &rawobs) {
    citlali::pipeline::read_rawobs_adc_snap_data(
        rawobs, engine().diagnostics.adc_snap_data, logger);
}
