#include <citlali/core/pipeline/timestream_native_paired_readout.h>

#include <concepts>
#include <type_traits>

namespace pipeline = citlali::pipeline;

static_assert(std::movable<pipeline::NativePairedReadoutNetwork>);
static_assert(!std::copy_constructible<pipeline::NativePairedReadoutNetwork>);
static_assert(std::movable<pipeline::NativePairedReadoutObservation>);
static_assert(
    !std::copy_constructible<pipeline::NativePairedReadoutObservation>);
static_assert(sizeof(pipeline::NativeReadoutCoordinateState) ==
              sizeof(std::uint16_t));

namespace {
[[maybe_unused]] auto timestream_native_paired_readout_header_is_self_contained =
    sizeof(pipeline::NativePairedReadoutObservation);
}
