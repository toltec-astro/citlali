#include <citlali/core/timestream/rtc/response_manifest.h>

#include <type_traits>

static_assert(std::is_enum_v<timestream::RTCResponseState>);
static_assert(std::is_enum_v<timestream::RTCResponseStageSlot>);
static_assert(timestream::rtc_response_stage_slots.size() == 15);
