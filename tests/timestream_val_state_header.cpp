#include <citlali/core/pipeline/timestream_val_state.h>

#include <type_traits>

namespace {

static_assert(sizeof(citlali::pipeline::ValGeneration) ==
              sizeof(std::uint64_t));
static_assert(
    !std::is_copy_constructible_v<citlali::pipeline::ValNativeRealization>);
static_assert(
    !std::is_move_constructible_v<citlali::pipeline::ValNativeRealization>);
static_assert(
    !std::is_default_constructible_v<citlali::pipeline::ValNativeTarget>);
static_assert(std::is_copy_constructible_v<citlali::pipeline::ValNativeTarget>);
static_assert(
    std::is_nothrow_move_constructible_v<citlali::pipeline::ValNativeTarget>);

} // namespace
