#include <citlali/core/pipeline/timestream_coincidence_cohort.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace {

using citlali::pipeline::CoincidenceAbsenceReason;
using citlali::pipeline::CoincidenceCellState;
using citlali::pipeline::CoincidenceCohortBuilder;
using citlali::pipeline::FinitePcaPlaceholder;
using citlali::pipeline::NativeInvalidityProvenance;
using citlali::pipeline::NativeOperationIdentity;
using citlali::pipeline::NativeSampleIdentity;
using citlali::pipeline::NativeSampleLedger;
using citlali::pipeline::make_pca_rectangular_working_set;

std::uint64_t bits_of(double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

TEST(sci_align_partial_cohort,
     keeps_real_values_bitwise_native_and_absence_non_authoritative) {
    const NativeSampleIdentity nw0_row10{0, 10, 1000.001};
    const NativeSampleIdentity nw1_row20{1, 20, 1000.003};
    const NativeSampleIdentity nw1_row21{1, 21, 1000.011};
    constexpr double nw0_value = 1.25;
    const double nw1_invalid_value =
        std::numeric_limits<double>::quiet_NaN();
    constexpr double nw1_value = 13.5;

    NativeSampleLedger<double> ledger{{
        {nw0_row10, nw0_value},
        {nw1_row20, nw1_invalid_value},
        {nw1_row21, nw1_value},
    }};

    CoincidenceCohortBuilder builder{
        NativeOperationIdentity{4, 17}, {0, 1}, 2};
    builder.assign_mapped_valid(0, 0, nw0_row10, 0);
    builder.assign_mapped_invalid(
        1, 0, nw1_row20, 0,
        NativeInvalidityProvenance{0x20U, "native glitch flag"});
    builder.assign_absent(
        0, 1, CoincidenceAbsenceReason::outside_tolerance);
    builder.assign_mapped_valid(1, 1, nw1_row21, 0);
    auto cohort = std::move(builder).finish();

    constexpr double placeholder_value = 912345.125;
    auto working = make_pca_rectangular_working_set(
        ledger, cohort, FinitePcaPlaceholder::checked(placeholder_value));

    ASSERT_EQ(cohort.participant_network_ids(),
              (std::vector<std::int32_t>{0, 1}));
    EXPECT_EQ(cohort.slot_count(), 2U);
    EXPECT_EQ(cohort.participant_count(), 2U);

    EXPECT_EQ(bits_of(working.value(0, 0)), bits_of(nw0_value));
    EXPECT_EQ(bits_of(working.value(0, 1)),
              bits_of(placeholder_value));
    EXPECT_EQ(bits_of(working.value(1, 1)), bits_of(nw1_value));
    EXPECT_EQ(bits_of(working.value(1, 0)), bits_of(placeholder_value));

    EXPECT_FALSE(working.excluded(0, 0));
    EXPECT_TRUE(working.excluded(0, 1));
    EXPECT_TRUE(working.excluded(1, 0));
    EXPECT_FALSE(working.excluded(1, 1));
    EXPECT_NO_THROW(working.require_all_values_finite_for_pca());

    const auto &mapped_valid = cohort.cell_for_network(0, 0);
    ASSERT_TRUE(mapped_valid.identity().has_value());
    EXPECT_EQ(*mapped_valid.identity(), nw0_row10);
    EXPECT_EQ(mapped_valid.identity()->native_row(), 10);
    EXPECT_DOUBLE_EQ(
        mapped_valid.identity()->reconstructed_time_unix_sec(),
        1000.001);

    const auto &mapped_invalid = cohort.cell_for_network(0, 1);
    ASSERT_EQ(mapped_invalid.state(),
              CoincidenceCellState::mapped_invalid);
    ASSERT_TRUE(mapped_invalid.identity().has_value());
    EXPECT_EQ(*mapped_invalid.identity(), nw1_row20);
    ASSERT_TRUE(mapped_invalid.invalidity().has_value());
    EXPECT_EQ(mapped_invalid.invalidity()->original_flag_bits, 0x20U);
    EXPECT_EQ(mapped_invalid.invalidity()->reason, "native glitch flag");
    EXPECT_FALSE(mapped_invalid.absence_reason().has_value());

    const auto &absent = cohort.cell_for_network(1, 0);
    ASSERT_EQ(absent.state(), CoincidenceCellState::absent);
    EXPECT_FALSE(absent.identity().has_value());
    EXPECT_FALSE(absent.invalidity().has_value());
    ASSERT_TRUE(absent.absence_reason().has_value());
    EXPECT_EQ(*absent.absence_reason(),
              CoincidenceAbsenceReason::outside_tolerance);

    EXPECT_EQ(bits_of(ledger.at(nw0_row10.key()).measured_value),
              bits_of(nw0_value));
    EXPECT_EQ(bits_of(ledger.at(nw1_row20.key()).measured_value),
              bits_of(nw1_invalid_value));
    EXPECT_EQ(bits_of(ledger.at(nw1_row21.key()).measured_value),
              bits_of(nw1_value));
}

TEST(sci_align_partial_cohort,
     complete_alignment_selects_equal_counts_without_interpolation) {
    const std::vector<NativeSampleIdentity> identities{
        {0, 4, 20.001}, {0, 5, 20.009},
        {1, 8, 20.003}, {1, 9, 20.011}};
    const std::vector<double> values{1.125, -2.25, 3.5, -4.75};
    NativeSampleLedger<double> ledger{{
        {identities[0], values[0]}, {identities[1], values[1]},
        {identities[2], values[2]}, {identities[3], values[3]},
    }};

    CoincidenceCohortBuilder builder{
        NativeOperationIdentity{6, 2}, {0, 1}, 2};
    builder.assign_mapped_valid(0, 0, identities[0], 0);
    builder.assign_mapped_valid(1, 0, identities[2], 0);
    builder.assign_mapped_valid(0, 1, identities[1], 0);
    builder.assign_mapped_valid(1, 1, identities[3], 0);
    auto cohort = std::move(builder).finish();
    const auto working = make_pca_rectangular_working_set(
        ledger, cohort, FinitePcaPlaceholder::checked(0.0));

    std::vector<std::size_t> selected_by_participant(2, 0);
    for (std::size_t slot = 0; slot < cohort.slot_count(); ++slot) {
        for (std::size_t participant = 0;
             participant < cohort.participant_count(); ++participant) {
            ASSERT_TRUE(cohort.cell(slot, participant).is_mapped());
            ++selected_by_participant[participant];
            const auto flat = slot * cohort.participant_count() + participant;
            const auto &identity = *cohort.cell(slot, participant).identity();
            EXPECT_EQ(
                bits_of(working.value(slot, participant)),
                bits_of(ledger.at(identity.key()).measured_value));
            EXPECT_EQ(working.mapped_identities().at(flat),
                      cohort.cell(slot, participant).identity());
        }
    }
    EXPECT_EQ(selected_by_participant,
              (std::vector<std::size_t>{2, 2}));
}

TEST(sci_align_partial_cohort,
     rejects_collisions_reuse_incomplete_cells_and_identity_mismatch) {
    EXPECT_THROW(
        (CoincidenceCohortBuilder{
            NativeOperationIdentity{1, 0}, {0, 0}, 1}),
        std::invalid_argument);

    const NativeSampleIdentity nw0_row0{0, 0, 10.0};
    const NativeSampleIdentity nw1_row0{1, 0, 10.001};

    CoincidenceCohortBuilder collision{
        NativeOperationIdentity{2, 0}, {0}, 1};
    collision.assign_mapped_valid(0, 0, nw0_row0, 0);
    EXPECT_THROW(
        collision.assign_absent(
            0, 0, CoincidenceAbsenceReason::no_candidate),
        std::logic_error);

    CoincidenceCohortBuilder reused{
        NativeOperationIdentity{3, 0}, {0}, 2};
    reused.assign_mapped_valid(0, 0, nw0_row0, 0);
    EXPECT_THROW(
        reused.assign_mapped_valid(0, 1, nw0_row0, 0),
        std::logic_error);

    CoincidenceCohortBuilder mismatch{
        NativeOperationIdentity{4, 0}, {0}, 1};
    EXPECT_THROW(
        mismatch.assign_mapped_valid(0, 0, nw1_row0, 0),
        std::invalid_argument);

    CoincidenceCohortBuilder incomplete{
        NativeOperationIdentity{5, 0}, {0, 1}, 1};
    incomplete.assign_mapped_valid(0, 0, nw0_row0, 0);
    EXPECT_THROW(std::move(incomplete).finish(), std::logic_error);
}

TEST(sci_align_partial_cohort,
     checked_placeholder_and_native_identity_reject_nonfinite_time) {
    EXPECT_NO_THROW(FinitePcaPlaceholder::checked(0.0));
    EXPECT_THROW(
        FinitePcaPlaceholder::checked(
            std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
    EXPECT_THROW(
        FinitePcaPlaceholder::checked(
            std::numeric_limits<double>::infinity()),
        std::invalid_argument);
    EXPECT_THROW(
        (NativeSampleIdentity{
            0, 0, std::numeric_limits<double>::quiet_NaN()}),
        std::invalid_argument);

    const NativeSampleIdentity identity{0, 1, 10.0};
    NativeSampleLedger<double> ledger{{{identity, 2.0}}};
    CoincidenceCohortBuilder builder{
        NativeOperationIdentity{8, 0}, {0}, 1};
    builder.assign_mapped_valid(0, 0, identity, 0);
    auto cohort = std::move(builder).finish();
    auto working = make_pca_rectangular_working_set(
        ledger, cohort, FinitePcaPlaceholder::checked(0.0));
    working.mutable_value(0, 0) =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        working.require_all_values_finite_for_pca(), std::logic_error);
}

}  // namespace
