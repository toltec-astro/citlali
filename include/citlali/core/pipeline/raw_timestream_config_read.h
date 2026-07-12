#pragma once

#include <citlali/core/pipeline/raw_filtering_config_read.h>
#include <citlali/core/pipeline/raw_flagging_config_read.h>
#include <citlali/core/pipeline/raw_line_audit_config_read.h>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
void read_raw_timestream_request_config(
    Config &config, citlali::config::RawTimeChunkConfig &raw,
    Diagnostics &diagnostics) {
    read_raw_filtering_request_config(config, raw, diagnostics);
    read_raw_flagging_and_despike_request_config(
        config, raw, diagnostics);
    read_raw_line_audit_request_config(
        config, raw.line_audit, diagnostics);
}

}  // namespace citlali::pipeline
