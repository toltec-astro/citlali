#include <citlali/core/pipeline/timestream_val_state.h>

#include "timestream_successor_identity_test_support.h"

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;
namespace support = citlali::test::timestream_successor;

std::shared_ptr<const pipeline::NativePairedReadoutObservation>
val_native_fixture(double time_offset = 0.0) {
    std::vector<pipeline::NativePairedReadoutNetwork> networks;
    networks.push_back(support::make_network(
        0, 10, {100.0 + time_offset, 101.0 + time_offset}, {20, 21},
        2, 1.0, 101.0, support::valid_states(4),
        support::valid_states(4)));
    return support::make_observation(std::move(networks), {0});
}

TEST(timestream_val_state,
     immutable_generations_bind_exact_native_and_producer_identities) {
    const auto native = val_native_fixture();
    const auto v0 = pipeline::ValSnapshot::initial(native);
    const auto first_address = v0->address(0, 10, 0);
    const auto second_address = v0->address(0, 11, 1);

    EXPECT_EQ(v0->generation(), pipeline::ValGeneration{0});
    EXPECT_EQ(v0->paired_handle(), native);
    EXPECT_TRUE(v0->contains(first_address));
    EXPECT_EQ(first_address.sample_identity(),
              native->network(0).occurrence_axis().native_identity(10));
    EXPECT_EQ(first_address.parent_readout_occurrence_key(), 10010);
    EXPECT_EQ(first_address.paired_xr_occurrence_key(), 20010);
    EXPECT_EQ(v0->detector_binding(first_address).detector_occurrence_id,
              "detector-occurrence:0:0");
    EXPECT_EQ(v0->occurrence_binding(first_address)
                  .paired_xr_occurrence_key,
              20010);

    const pipeline::ValProducerProductIdentity rtc_product{
        pipeline::ValProducer::rtc, 17};
    const pipeline::ValFindingKey first_key{
        rtc_product, first_address, pipeline::ValFactCode{1}};
    const pipeline::ValFindingKey second_key{
        rtc_product, second_address, pipeline::ValFactCode{2}};
    pipeline::ValDeltaBuilder builder{v0, rtc_product};
    // Reverse proposal order proves that freeze supplies deterministic order.
    builder.propose(second_address, pipeline::ValFactCode{2},
                    pipeline::ValFactState{20},
                    pipeline::ValFactCause{200});
    builder.propose(first_address, pipeline::ValFactCode{1},
                    pipeline::ValFactState{10},
                    pipeline::ValFactCause{100});

    EXPECT_EQ(v0->find(first_key), nullptr);
    auto delta = builder.freeze();
    ASSERT_EQ(delta.findings().size(), 2U);
    EXPECT_EQ(delta.findings().front().key(), first_key);
    EXPECT_EQ(v0->find(first_key), nullptr);

    const auto v1 = pipeline::ValSnapshot::commit(std::move(delta));
    EXPECT_EQ(v1->generation(), pipeline::ValGeneration{1});
    EXPECT_EQ(v1->paired_handle(), native);
    EXPECT_EQ(v1->parent_snapshot_handle(), v0);
    ASSERT_NE(v1->find(first_key), nullptr);
    EXPECT_EQ(v1->find(first_key)->state(), pipeline::ValFactState{10});
    ASSERT_NE(v1->find(second_key), nullptr);
    EXPECT_EQ(v1->find(second_key)->cause(), pipeline::ValFactCause{200});
    EXPECT_EQ(v0->find(first_key), nullptr);
    EXPECT_EQ(v1->memory_evidence().owned_finding_bytes,
              2U * sizeof(pipeline::ValFinding));
    EXPECT_EQ(v1->memory_evidence().referenced_parent_generation_count, 1U);
}

TEST(timestream_val_state,
     later_generation_overlays_one_fact_without_mutating_prior_snapshot) {
    const auto native = val_native_fixture();
    const auto v0 = pipeline::ValSnapshot::initial(native);
    const pipeline::ValProducerProductIdentity ast_product{
        pipeline::ValProducer::ast, 9};
    const auto address = v0->address(0, 10);
    const pipeline::ValFindingKey key{
        ast_product, address, pipeline::ValFactCode{3}};

    pipeline::ValDeltaBuilder first{v0, ast_product};
    first.propose(address, pipeline::ValFactCode{3},
                  pipeline::ValFactState{1}, pipeline::ValFactCause{1});
    const auto v1 = pipeline::ValSnapshot::commit(first.freeze());

    pipeline::ValDeltaBuilder second{v1, ast_product};
    second.propose(address, pipeline::ValFactCode{3},
                   pipeline::ValFactState{2}, pipeline::ValFactCause{2});
    const auto v2 = pipeline::ValSnapshot::commit(second.freeze());

    ASSERT_NE(v1->find(key), nullptr);
    ASSERT_NE(v2->find(key), nullptr);
    EXPECT_EQ(v1->find(key)->state(), pipeline::ValFactState{1});
    EXPECT_EQ(v2->find(key)->state(), pipeline::ValFactState{2});
    EXPECT_EQ(v2->parent_snapshot_handle(), v1);
    EXPECT_EQ(v2->memory_evidence().owned_finding_bytes,
              sizeof(pipeline::ValFinding));
}

TEST(timestream_val_state,
     staged_updates_fail_closed_on_foreign_or_duplicate_identity) {
    const auto native = val_native_fixture();
    const auto foreign_native = val_native_fixture(10.0);
    const auto identical_foreign_native = val_native_fixture();
    const auto v0 = pipeline::ValSnapshot::initial(native);
    const auto foreign_v0 = pipeline::ValSnapshot::initial(foreign_native);
    const auto identical_foreign_v0 =
        pipeline::ValSnapshot::initial(identical_foreign_native);
    const pipeline::ValProducerProductIdentity align_product{
        pipeline::ValProducer::align, 5};

    pipeline::ValDeltaBuilder foreign_builder{v0, align_product};
    EXPECT_THROW(
        foreign_builder.propose(
            foreign_v0->address(0, 10), pipeline::ValFactCode{1},
            pipeline::ValFactState{1}, pipeline::ValFactCause{1}),
        std::invalid_argument);
    EXPECT_THROW(
        foreign_builder.propose(
            identical_foreign_v0->address(0, 10),
            pipeline::ValFactCode{1}, pipeline::ValFactState{1},
            pipeline::ValFactCause{1}),
        std::invalid_argument);

    const auto address = v0->address(0, 10);
    pipeline::ValDeltaBuilder duplicate_builder{v0, align_product};
    duplicate_builder.propose(address, pipeline::ValFactCode{1},
                              pipeline::ValFactState{1},
                              pipeline::ValFactCause{1});
    duplicate_builder.propose(address, pipeline::ValFactCode{1},
                              pipeline::ValFactState{2},
                              pipeline::ValFactCause{2});
    EXPECT_THROW(duplicate_builder.freeze(), std::invalid_argument);

    pipeline::ValDeltaBuilder empty_builder{v0, align_product};
    EXPECT_THROW(
        pipeline::ValSnapshot::commit(empty_builder.freeze()),
        std::invalid_argument);
}

}  // namespace
