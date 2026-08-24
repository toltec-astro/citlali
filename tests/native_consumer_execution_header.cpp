#include <citlali/core/pipeline/native_consumer_execution.h>

static_assert(
    citlali::pipeline::native_rtc_processing_flag_bit_v2 != 0);
static_assert(
    citlali::pipeline::native_duplicate_tone_exclusion_bit_v2 != 0);
static_assert(
    citlali::pipeline::native_duplicate_tone_exclusion_bit_v2 !=
    citlali::pipeline::native_rtc_processing_flag_bit_v2);
