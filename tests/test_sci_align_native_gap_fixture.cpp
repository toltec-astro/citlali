#include "sci_align_native_gap_fixture.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace {

namespace fixture = citlali::test_support::sci_align;
namespace pipeline = citlali::pipeline;

TEST(sci_align_native_gap_fixture,
     loads_pinned_owner_reproducible_identity_and_detector_inventory) {
    const auto loaded = fixture::load_native_gap_fixture_v1();

    EXPECT_EQ(loaded.source_sha256,
              fixture::native_gap_fixture_v1_sha256);
    EXPECT_EQ(loaded.accepted_plan_commit,
              "a3f2bf465a26048b24017ebd50876c4a2684b1b8");
    EXPECT_EQ(loaded.historical_evidence_commits,
              (std::vector<std::string>{"fd3627fc7"}));
    EXPECT_EQ(loaded.scope,
              (pipeline::NativeObservationScope{148669, 0, 2}));
    EXPECT_EQ(loaded.scan_index, 5);
    EXPECT_EQ(loaded.chunk_index, 0);
    EXPECT_DOUBLE_EQ(loaded.realized_dt_sec, 0.01);

    ASSERT_EQ(loaded.detector_columns.size(), 4U);
    EXPECT_EQ(loaded.detector_columns[0].detector_column, 0);
    EXPECT_EQ(loaded.detector_columns[0].network_id, 7);
    EXPECT_EQ(loaded.detector_columns[0].raw_channel, 1);
    EXPECT_EQ(loaded.detector_columns[0].output_uid,
              std::numeric_limits<std::int64_t>::max());
    EXPECT_EQ(loaded.detector_columns[1].output_uid, 0);
    EXPECT_EQ(loaded.detector_columns[2].output_uid,
              INT64_C(9007199254740993));
    EXPECT_EQ(loaded.detector_columns[3].output_uid, 1);
}

TEST(sci_align_native_gap_fixture,
     rejects_any_unreviewed_fixture_byte_change) {
    const auto source = fixture::native_gap_fixture_v1_path();
    const auto target = std::filesystem::temp_directory_path() /
        ("citlali-native-gap-v1-tampered-" +
         std::to_string(reinterpret_cast<std::uintptr_t>(&source)) +
         ".yaml");
    struct RemoveOnExit {
        std::filesystem::path path;
        ~RemoveOnExit() {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    } cleanup{target};

    std::filesystem::copy_file(
        source, target,
        std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream stream{target, std::ios::app};
        ASSERT_TRUE(stream.good());
        stream << "# unreviewed mutation\n";
        ASSERT_TRUE(stream.good());
    }
    EXPECT_THROW(fixture::load_native_gap_fixture_v1(target),
                 std::invalid_argument);
}

TEST(sci_align_native_gap_fixture,
     materializes_one_absent_slot_and_exact_packet_run_topology) {
    const auto loaded = fixture::load_native_gap_fixture_v1();
    const auto alignment = loaded.materialize_alignment();

    EXPECT_EQ(alignment->participant_network_ids(),
              (std::vector<pipeline::TimestreamNetworkId>{0, 7}));
    EXPECT_EQ(alignment->slot_count(), 5U);
    EXPECT_EQ(alignment->association(0, 2).native_row, 102);
    EXPECT_FALSE(alignment->association(7, 2).mapped());
    EXPECT_EQ(alignment->association(7, 2).absence_reason,
              pipeline::CoincidenceAbsenceReason::no_candidate);
    EXPECT_EQ(alignment->association(7, 3).native_row, 702);

    const auto network7_runs = pipeline::partition_native_contiguous_runs(
        alignment->network(7), 700, 704);
    ASSERT_EQ(network7_runs.size(), 2U);
    EXPECT_EQ(network7_runs[0].first_native_row, 700);
    EXPECT_EQ(network7_runs[0].past_last_native_row, 702);
    EXPECT_EQ(network7_runs[1].first_native_row, 702);
    EXPECT_EQ(network7_runs[1].past_last_native_row, 704);
    ASSERT_TRUE(network7_runs[0]
                    .boundary_after.counter_discontinuity.has_value());
    EXPECT_EQ(network7_runs[0]
                  .boundary_after.counter_discontinuity->before_counter,
              701);
    EXPECT_EQ(network7_runs[0]
                  .boundary_after.counter_discontinuity->after_counter,
              703);

    std::vector<fixture::NativeGapIntervalV1> complete;
    std::optional<std::size_t> first;
    for (std::size_t slot = 0; slot < alignment->slot_count(); ++slot) {
        const bool mapped = alignment->association(0, slot).mapped() &&
                            alignment->association(7, slot).mapped();
        if (mapped && !first.has_value()) first = slot;
        if (!mapped && first.has_value()) {
            complete.push_back({static_cast<std::int64_t>(*first),
                                static_cast<std::int64_t>(slot)});
            first.reset();
        }
    }
    if (first.has_value()) {
        complete.push_back({
            static_cast<std::int64_t>(*first),
            static_cast<std::int64_t>(alignment->slot_count())});
    }
    EXPECT_EQ(complete, loaded.expected_complete_cohort_slot_runs);
    EXPECT_EQ(complete,
              (std::vector<fixture::NativeGapIntervalV1>{{0, 2},
                                                         {3, 5}}));
}

TEST(sci_align_native_gap_fixture,
     pins_run_local_factor2_anchors_and_original_flag_oracle) {
    const auto loaded = fixture::load_native_gap_fixture_v1();
    const auto alignment = loaded.materialize_alignment();
    std::set<std::pair<std::size_t, pipeline::TimestreamNetworkId>> seen;

    for (const auto &support :
         loaded.expected_stage4_factor2_support) {
        ASSERT_TRUE(seen.emplace(support.segment_ordinal,
                                 support.network_id).second);
        const auto &network = loaded.network(support.network_id);
        ASSERT_EQ(support.past_last_common_slot -
                      support.first_common_slot,
                  2U);
        ASSERT_EQ(support.original_flag_or_by_channel.size(),
                  static_cast<std::size_t>(
                      network.original_flag_bits.cols()));

        const auto first = alignment->association(
            support.network_id, support.first_common_slot);
        ASSERT_TRUE(first.mapped());
        EXPECT_EQ(first.native_row,
                  support.selected_anchor_native_row);

        std::vector<pipeline::NativeDetectorFlagBits> flag_or(
            static_cast<std::size_t>(network.original_flag_bits.cols()),
            0U);
        for (std::size_t slot = support.first_common_slot;
             slot < support.past_last_common_slot; ++slot) {
            const auto association =
                alignment->association(support.network_id, slot);
            ASSERT_TRUE(association.mapped());
            const auto local_row = static_cast<Eigen::Index>(
                association.native_row - network.first_native_row);
            for (Eigen::Index channel = 0;
                 channel < network.original_flag_bits.cols(); ++channel) {
                flag_or[static_cast<std::size_t>(channel)] |=
                    network.original_flag_bits(local_row, channel);
            }
        }
        EXPECT_EQ(flag_or, support.original_flag_or_by_channel);
    }
    EXPECT_EQ(seen.size(), 4U);
}

}  // namespace
