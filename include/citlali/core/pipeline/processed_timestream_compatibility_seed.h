#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/timestream_config_mirror.h>

namespace citlali::pipeline {

template <class PtcProc, class ArrayNameMap>
void seed_processed_timestream_config_from_legacy(
    citlali::config::TimestreamConfig &target, const PtcProc &ptcproc,
    const ArrayNameMap &array_name_map) {
    mirror_fruit_loops_config(target.fruit_loops, ptcproc);
    auto &processed = target.processed_time_chunk;
    mirror_processed_clean_config(
        processed.clean, ptcproc, array_name_map);
    mirror_processed_weighting_config(
        processed.weighting, processed.flagging, ptcproc);
    mirror_processed_weight_validation_config(
        processed.weighting.validation, ptcproc.weight_validation);
    mirror_processed_weight_corr_penalty_config(
        processed.weighting.corr_penalty, ptcproc.weight_corr_penalty);
    mirror_second_pass_local_config(
        processed.flagging.second_pass_local, ptcproc.second_pass_local);
}

}  // namespace citlali::pipeline
