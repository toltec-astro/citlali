#include <citlali/core/pipeline/timestream_val_state.h>

#include "timestream_successor_identity_test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
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

using Coordinate = pipeline::NativeReadoutCoordinate;
using Role = pipeline::ValNativeProductRole;

std::shared_ptr<const pipeline::ValNativeRealization> realization(
    const std::shared_ptr<const pipeline::NativePairedReadoutObservation>
        &parent,
    Role role = Role::derived_residual, int network = 0) {
    return pipeline::ValNativeRealization::create(
        parent, {pipeline::ValProducer::rtc, 17}, 23, role, network);
}

TEST(timestream_val_state,
     native_targets_keep_coordinate_role_and_instance_distinct) {
    const auto parent = val_native_fixture();
    const auto v0 = pipeline::ValSnapshot::initial(parent);
    const auto address = v0->address(0, 10, 0);
    const auto raw = realization(parent, Role::original_input);
    const auto residual = realization(parent);
    const auto same_description = realization(parent);
    // The fact author is independent of the numerical object's producer.
    const pipeline::ValProducerProductIdentity author{
        pipeline::ValProducer::ast, 5};
    const pipeline::ValFactCode code{1};
    const std::array targets{
        v0->native_target(raw, address, Coordinate::x),
        v0->native_target(raw, address, Coordinate::r),
        v0->native_target(residual, address, Coordinate::x),
        v0->native_target(residual, address, Coordinate::r),
        v0->native_target(same_description, address, Coordinate::x)};
    pipeline::ValDeltaBuilder builder{v0, author};
    builder.propose(address, code, pipeline::ValFactState{90},
                    pipeline::ValFactCause{91});
    for (std::size_t i = 0; i < targets.size(); ++i) {
        builder.propose(
            targets[i], code,
            pipeline::ValFactState{static_cast<std::uint32_t>(i + 1)},
            pipeline::ValFactCause{static_cast<std::uint32_t>(i + 11)});
    }
    const auto v1 = pipeline::ValSnapshot::commit(builder.freeze());
    ASSERT_EQ(v1->committed_delta_findings().size(), targets.size() + 1);
    for (std::size_t i = 0; i < targets.size(); ++i) {
        const pipeline::ValFindingKey key{author, targets[i], code};
        ASSERT_NE(v1->find(key), nullptr);
        EXPECT_EQ(v1->find(key)->state().value(), i + 1);
        ASSERT_NE(key.native_target(), nullptr);
        EXPECT_EQ(key.address(), address);
        EXPECT_EQ(key.product(), author);
        EXPECT_EQ(key.native_target()->realization_handle()->producer_product(),
                  (pipeline::ValProducerProductIdentity{
                      pipeline::ValProducer::rtc, 17}));
        EXPECT_EQ(v0->find(key), nullptr);
        for (std::size_t j = 0; j < targets.size(); ++j) {
            EXPECT_EQ(targets[i] == targets[j], i == j);
        }
    }
    const pipeline::ValFindingKey unqualified{author, address, code};
    ASSERT_NE(v1->find(unqualified), nullptr);
    EXPECT_EQ(v1->find(unqualified)->state().value(), 90U);
    EXPECT_EQ(unqualified.native_target(), nullptr);
    EXPECT_EQ(residual->paired_handle(), parent);
    EXPECT_EQ(residual->realization_instance(),
              same_description->realization_instance());
    EXPECT_EQ(residual->role(), Role::derived_residual);
    EXPECT_EQ(raw->role(), Role::original_input);
}

TEST(timestream_val_state,
     native_target_binding_rejects_foreign_and_incomplete_subjects) {
    std::vector<pipeline::NativePairedReadoutNetwork> networks;
    networks.push_back(
        support::make_network(0, 10, {100.0, 101.0}, {20, 21}, 2, 1.0, 2.0));
    networks.push_back(
        support::make_network(7, 30, {200.0}, {40}, 1, 3.0, 4.0));
    const auto parent = support::make_observation(std::move(networks), {0, 7});
    const auto v0 = pipeline::ValSnapshot::initial(parent);
    const auto foreign = pipeline::ValSnapshot::initial(val_native_fixture());
    const auto descriptor = realization(parent);
    const auto address = v0->address(0, 10, 0);
    EXPECT_THROW(v0->native_target(nullptr, address, Coordinate::x),
                 std::invalid_argument);
    EXPECT_THROW(v0->native_target(realization(foreign->paired_handle()),
                                   address, Coordinate::x),
                 std::invalid_argument);
    EXPECT_THROW(v0->native_target(descriptor, foreign->address(0, 10, 0),
                                   Coordinate::x),
                 std::invalid_argument);
    EXPECT_THROW(
        v0->native_target(descriptor, v0->address(7, 30, 0), Coordinate::x),
        std::invalid_argument);
    EXPECT_THROW(
        v0->native_target(descriptor, v0->address(0, 10), Coordinate::x),
        std::invalid_argument);
    EXPECT_THROW(
        v0->native_target(descriptor, address, static_cast<Coordinate>(255)),
        std::invalid_argument);
    for (const auto row : {9, 12}) {
        EXPECT_THROW(v0->native_target(descriptor, v0->address(0, row, 0),
                                       Coordinate::x),
                     std::out_of_range);
    }
    for (const auto detector : {-1, 2}) {
        EXPECT_THROW(v0->native_target(descriptor, v0->address(0, 10, detector),
                                       Coordinate::x),
                     std::out_of_range);
    }
    EXPECT_THROW(v0->address(99, 10, 0), std::out_of_range);
    EXPECT_THROW(realization(parent, Role::derived_residual, 99),
                 std::out_of_range);
    EXPECT_THROW(realization(nullptr), std::invalid_argument);
    EXPECT_THROW(realization(parent, static_cast<Role>(255)),
                 std::invalid_argument);
    EXPECT_THROW(pipeline::ValNativeRealization::create(
                     parent, {pipeline::ValProducer::rtc, 17}, 0,
                     Role::derived_residual, 0),
                 std::invalid_argument);
    EXPECT_THROW(pipeline::ValNativeRealization::create(
                     parent, {static_cast<pipeline::ValProducer>(255), 17}, 23,
                     Role::derived_residual, 0),
                 std::invalid_argument);

    const auto foreign_target =
        foreign->native_target(realization(foreign->paired_handle()),
                               foreign->address(0, 10, 0), Coordinate::x);
    EXPECT_FALSE(v0->contains(foreign_target));
    pipeline::ValDeltaBuilder builder{v0, {pipeline::ValProducer::rtc, 17}};
    EXPECT_THROW(builder.propose(foreign_target, pipeline::ValFactCode{1},
                                 pipeline::ValFactState{1},
                                 pipeline::ValFactCause{1}),
                 std::invalid_argument);
    EXPECT_THROW(pipeline::ValSnapshot::commit(builder.freeze()),
                 std::invalid_argument);
}

TEST(
    timestream_val_state,
    native_target_key_order_is_total_and_freeze_is_proposal_order_independent) {
    const auto parent = val_native_fixture();
    const auto v0 = pipeline::ValSnapshot::initial(parent);
    const auto a = realization(parent);
    const auto b = realization(parent);
    const auto address = v0->address(0, 10, 0);
    const pipeline::ValProducerProductIdentity author{
        pipeline::ValProducer::rtc, 17};
    const pipeline::ValFactCode code{1};
    const std::array targets{
        v0->native_target(a, address, Coordinate::x),
        v0->native_target(a, address, Coordinate::r),
        v0->native_target(b, address, Coordinate::x),
        v0->native_target(a, v0->address(0, 11, 1), Coordinate::x)};
    std::vector<pipeline::ValFindingKey> keys{{author, address, code}};
    for (const auto &target : targets)
        keys.emplace_back(author, target, code);
    keys.emplace_back(author, targets[0], pipeline::ValFactCode{2});
    keys.emplace_back(
        pipeline::ValProducerProductIdentity{pipeline::ValProducer::ast, 17},
        targets[0], code);
    for (const auto &x : keys)
        for (const auto &y : keys) {
            EXPECT_EQ(!(x < y) && !(y < x), x == y);
            EXPECT_FALSE(x < y && y < x);
            for (const auto &z : keys)
                if (x < y && y < z) EXPECT_TRUE(x < z);
        }
    std::array<int, 4> order{0, 1, 2, 3};
    std::vector<pipeline::ValFinding> expected;
    do {
        pipeline::ValDeltaBuilder builder{v0, author};
        for (const int index : order)
            builder.propose(
                targets[index], code,
                pipeline::ValFactState{static_cast<std::uint32_t>(index + 1)},
                pipeline::ValFactCause{1});
        const auto delta = builder.freeze();
        const std::vector<pipeline::ValFinding> actual{delta.findings().begin(),
                                                       delta.findings().end()};
        if (expected.empty()) expected = actual;
        EXPECT_EQ(actual, expected);
    } while (std::next_permutation(order.begin(), order.end()));

    pipeline::ValDeltaBuilder duplicate{v0, author};
    duplicate.propose(targets[0], code, pipeline::ValFactState{1},
                      pipeline::ValFactCause{1});
    duplicate.propose(targets[0], code, pipeline::ValFactState{2},
                      pipeline::ValFactCause{2});
    EXPECT_THROW(duplicate.freeze(), std::invalid_argument);
}

TEST(timestream_val_state,
     qualified_overlays_preserve_prior_and_branch_snapshots_without_fallback) {
    const auto v0 = pipeline::ValSnapshot::initial(val_native_fixture());
    const auto descriptor = realization(v0->paired_handle());
    const auto address = v0->address(0, 10, 0);
    const auto x = v0->native_target(descriptor, address, Coordinate::x);
    const auto r = v0->native_target(descriptor, address, Coordinate::r);
    const pipeline::ValProducerProductIdentity author{
        pipeline::ValProducer::rtc, 17};
    const pipeline::ValFactCode code{1};
    const pipeline::ValFindingKey x_key{author, x, code};
    const pipeline::ValFindingKey r_key{author, r, code};
    pipeline::ValDeltaBuilder first{v0, author};
    first.propose(address, code, pipeline::ValFactState{10},
                  pipeline::ValFactCause{10});
    first.propose(x, code, pipeline::ValFactState{1},
                  pipeline::ValFactCause{1});
    const auto v1 = pipeline::ValSnapshot::commit(first.freeze());
    pipeline::ValDeltaBuilder sibling{v0, author};
    sibling.propose(x, code, pipeline::ValFactState{2},
                    pipeline::ValFactCause{2});
    const auto other_v1 = pipeline::ValSnapshot::commit(sibling.freeze());
    pipeline::ValDeltaBuilder next{v1, author};
    next.propose(x, code, pipeline::ValFactState{3}, pipeline::ValFactCause{3});
    const auto v2 = pipeline::ValSnapshot::commit(next.freeze());
    EXPECT_EQ(v1->generation(), other_v1->generation());
    EXPECT_NE(v1.get(), other_v1.get());
    EXPECT_EQ(v2->parent_snapshot_handle(), v1);
    ASSERT_NE(v1->find(x_key), nullptr);
    ASSERT_NE(other_v1->find(x_key), nullptr);
    ASSERT_NE(v2->find(x_key), nullptr);
    EXPECT_EQ(v1->find(x_key)->state().value(), 1U);
    EXPECT_EQ(other_v1->find(x_key)->state().value(), 2U);
    EXPECT_EQ(v2->find(x_key)->state().value(), 3U);
    EXPECT_EQ(v0->find(x_key), nullptr);
    EXPECT_EQ(v2->find(r_key), nullptr);
    EXPECT_EQ(other_v1->find({author, address, code}), nullptr);
    ASSERT_NE(v2->find({author, address, code}), nullptr);
    EXPECT_EQ(v2->find({author, address, code})->state().value(), 10U);
    EXPECT_THROW(next.propose(r, code, pipeline::ValFactState{1},
                              pipeline::ValFactCause{1}),
                 std::logic_error);
}

TEST(timestream_val_state,
     targeted_fact_storage_does_not_infer_numeric_validity_or_causes) {
    std::vector<pipeline::NativePairedReadoutNetwork> networks;
    networks.push_back(support::make_network(
        0, 10, {100.0}, {20}, 1, std::numeric_limits<double>::quiet_NaN(), 2.0,
        {pipeline::NativeReadoutCoordinateState::measured(true, false, true,
                                                          false)},
        support::valid_states(1)));
    const auto parent = support::make_observation(std::move(networks), {0});
    const auto v0 = pipeline::ValSnapshot::initial(parent);
    const auto target = v0->native_target(realization(parent),
                                          v0->address(0, 10, 0), Coordinate::x);
    const pipeline::ValProducerProductIdentity author{
        pipeline::ValProducer::rtc, 17};
    const pipeline::ValFindingKey key{author, target, pipeline::ValFactCode{1}};
    EXPECT_EQ(v0->find(key), nullptr);
    pipeline::ValDeltaBuilder builder{v0, author};
    builder.propose(target, pipeline::ValFactCode{1},
                    pipeline::ValFactState{87}, pipeline::ValFactCause{91});
    const auto v1 = pipeline::ValSnapshot::commit(builder.freeze());
    ASSERT_NE(v1->find(key), nullptr);
    EXPECT_EQ(v1->find(key)->state().value(), 87U);
    EXPECT_EQ(v1->find(key)->cause().value(), 91U);
    EXPECT_EQ(v1->find({author, target, pipeline::ValFactCode{2}}), nullptr);
}

TEST(timestream_val_state,
     realization_lifetime_is_retained_without_snapshot_cycle) {
    std::weak_ptr<const pipeline::NativePairedReadoutObservation> weak_parent;
    std::weak_ptr<const pipeline::ValNativeRealization> weak_descriptor;
    std::weak_ptr<const pipeline::ValSnapshot> weak_base;
    std::shared_ptr<const pipeline::ValSnapshot> retained;
    {
        const auto parent = val_native_fixture();
        const auto v0 = pipeline::ValSnapshot::initial(parent);
        const auto descriptor = realization(parent);
        weak_parent = parent;
        weak_descriptor = descriptor;
        weak_base = v0;
        auto target =
            v0->native_target(descriptor, v0->address(0, 10, 0), Coordinate::x);
        const auto moved = std::move(target);
        EXPECT_FALSE(v0->contains(target));
        pipeline::ValDeltaBuilder builder{v0, {pipeline::ValProducer::rtc, 17}};
        EXPECT_THROW(builder.propose(target, pipeline::ValFactCode{1},
                                     pipeline::ValFactState{1},
                                     pipeline::ValFactCause{1}),
                     std::invalid_argument);
        builder.propose(moved, pipeline::ValFactCode{1},
                        pipeline::ValFactState{1}, pipeline::ValFactCause{1});
        retained = pipeline::ValSnapshot::commit(builder.freeze());
    }
    EXPECT_FALSE(weak_parent.expired());
    EXPECT_FALSE(weak_descriptor.expired());
    EXPECT_FALSE(weak_base.expired());
    ASSERT_EQ(retained->committed_delta_findings().size(), 1U);
    const auto *target =
        retained->committed_delta_findings()[0].key().native_target();
    ASSERT_NE(target, nullptr);
    EXPECT_EQ(target->realization_handle()->paired_handle(),
              retained->paired_handle());
    retained.reset();
    EXPECT_TRUE(weak_parent.expired());
    EXPECT_TRUE(weak_descriptor.expired());
    EXPECT_TRUE(weak_base.expired());
}

TEST(timestream_val_state,
     native_target_memory_is_sparse_and_shares_one_descriptor) {
    const auto parent = val_native_fixture();
    const auto v0 = pipeline::ValSnapshot::initial(parent);
    const auto descriptor = realization(parent);
    const auto *x_data = parent->network(0).values(Coordinate::x).data();
    pipeline::ValDeltaBuilder builder{v0, {pipeline::ValProducer::rtc, 17}};
    for (int row = 10; row < 12; ++row)
        for (int detector = 0; detector < 2; ++detector) {
            for (const auto coordinate : {Coordinate::x, Coordinate::r}) {
                builder.propose(
                    v0->native_target(descriptor, v0->address(0, row, detector),
                                      coordinate),
                    pipeline::ValFactCode{1}, pipeline::ValFactState{1},
                    pipeline::ValFactCause{1});
            }
        }
    const auto v1 = pipeline::ValSnapshot::commit(builder.freeze());
    EXPECT_EQ(v0->memory_evidence().owned_finding_bytes, 0U);
    EXPECT_EQ(v0->memory_evidence().referenced_native_target_count, 0U);
    EXPECT_EQ(v1->memory_evidence().owned_finding_bytes,
              8U * sizeof(pipeline::ValFinding));
    EXPECT_EQ(v1->memory_evidence().referenced_native_target_count, 8U);
    for (const auto &finding : v1->committed_delta_findings()) {
        ASSERT_NE(finding.key().native_target(), nullptr);
        EXPECT_EQ(finding.key().native_target()->realization_handle(),
                  descriptor);
    }
    EXPECT_EQ(
        descriptor->paired_handle()->network(0).values(Coordinate::x).data(),
        x_data);
    EXPECT_LE(sizeof(pipeline::ValNativeRealization), 8U * sizeof(void *));
    EXPECT_LE(sizeof(pipeline::ValFindingKey),
              sizeof(pipeline::ValAddress) +
                  sizeof(pipeline::ValProducerProductIdentity) +
                  sizeof(pipeline::ValFactCode) + 5U * sizeof(void *));
    RecordProperty("native_realization_bytes",
                   static_cast<int>(sizeof(pipeline::ValNativeRealization)));
    RecordProperty("finding_key_bytes",
                   static_cast<int>(sizeof(pipeline::ValFindingKey)));
    RecordProperty("finding_bytes",
                   static_cast<int>(sizeof(pipeline::ValFinding)));
}

} // namespace
