#include <citlali/core/pipeline/timestream_native_paired_readout_kids_adapter.h>

#include "timestream_successor_identity_test_support.h"

#include <gtest/gtest.h>

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

namespace {

namespace pipeline = citlali::pipeline;
namespace support = citlali::test::timestream_successor;

struct FakeCoordinateResult {
    pipeline::NativePairedReadoutMatrix data;
};

struct FakeSolverResult {
    struct {
        FakeCoordinateResult xs;
        FakeCoordinateResult rs;
    } data_out;
};

template <class Result>
concept AtomicKidsIngress = requires(
    pipeline::NativePairedReadoutNetworkIngress ingress,
    Result &&result) {
    pipeline::take_native_paired_kids_solver_result(
        std::move(ingress), std::forward<Result>(result));
};

static_assert(AtomicKidsIngress<FakeSolverResult>);
static_assert(!AtomicKidsIngress<FakeSolverResult &>);
static_assert(!AtomicKidsIngress<const FakeSolverResult>);

pipeline::NativePairedReadoutNetworkIngress ingress(
    Eigen::Index detector_count = 2) {
    const auto cells = static_cast<std::size_t>(3 * detector_count);
    return {support::occurrence_axis(0, 40, {10.0, 11.0, 12.0},
                                     {100, 101, 102}),
            support::detector_axis(0, detector_count),
            support::mapping_authority(0, "adapter"),
            support::valid_states(cells), support::valid_states(cells)};
}

TEST(native_paired_readout_kids_adapter,
     one_mutable_solver_result_transfers_both_exact_native_planes) {
    FakeSolverResult solver{
        {{support::matrix(3, 2, 1.0)},
         {support::matrix(3, 2, 101.0)}}};
    const auto *x_storage = solver.data_out.xs.data.data();
    const auto *r_storage = solver.data_out.rs.data.data();

    auto network = pipeline::take_native_paired_kids_solver_result(
        ingress(), std::move(solver));

    EXPECT_EQ(network.values(pipeline::NativeReadoutCoordinate::x).data(),
              x_storage);
    EXPECT_EQ(network.values(pipeline::NativeReadoutCoordinate::r).data(),
              r_storage);
    EXPECT_DOUBLE_EQ(network.value(
                         pipeline::NativeReadoutCoordinate::x, 42, 1),
                     22.0);
    EXPECT_DOUBLE_EQ(network.value(
                         pipeline::NativeReadoutCoordinate::r, 42, 1),
                     122.0);
    EXPECT_EQ(network.mapping_authority().producer_instance_id,
              "kids-result:adapter");
    EXPECT_EQ(network.occurrence_axis()
                  .occurrence(42)
                  .paired_xr_occurrence_key,
              20042);
}

TEST(native_paired_readout_kids_adapter,
     mismatched_coordinate_shape_fails_closed_at_canonical_admission) {
    FakeSolverResult solver{
        {{support::matrix(3, 2, 1.0)},
         {support::matrix(2, 2, 101.0)}}};

    EXPECT_THROW(
        pipeline::take_native_paired_kids_solver_result(
            ingress(), std::move(solver)),
        std::invalid_argument);
}

}  // namespace
