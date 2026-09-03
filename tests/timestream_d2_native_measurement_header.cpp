#include <citlali/core/pipeline/timestream_d2_native_measurement.h>

#include <concepts>
#include <cstddef>

namespace {

namespace pipeline = citlali::pipeline;

template <typename Product>
concept HasD2LocalValidity = requires(const Product &product) {
    { product.valid(0, 0, 0) } -> std::convertible_to<bool>;
};

template <typename Product>
concept HasD2LocalUsability = requires(const Product &product) {
    { product.x_usable(0, 0, 0) } -> std::convertible_to<bool>;
    { product.r_usable(0, 0, 0) } -> std::convertible_to<bool>;
};

template <typename Product>
concept HasD2LocalCauses = requires(const Product &product) {
    product.causes(0, 0, 0);
};

static_assert(!HasD2LocalValidity<pipeline::D2NativeMeasurement>);
static_assert(!HasD2LocalUsability<pipeline::D2NativeMeasurement>);
static_assert(!HasD2LocalCauses<pipeline::D2NativeMeasurement>);
static_assert(sizeof(pipeline::D2ResidualPayloadState) ==
              sizeof(std::uint8_t));

}  // namespace
