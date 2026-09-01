#include <citlali/core/pipeline/timestream_identity_route_context.h>

#include <cstdint>
#include <type_traits>

static_assert(std::is_enum_v<
              citlali::pipeline::IdentityRtcAstDependency>);
static_assert(std::is_enum_v<
              citlali::pipeline::IdentityMapAdmissionState>);
static_assert(sizeof(citlali::pipeline::IdentityRtcAstDependency) ==
              sizeof(std::uint8_t));
