#include <citlali/core/pipeline/timestream_coincidence_cohort.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace {

using citlali::pipeline::CoincidenceAbsenceReason;
using citlali::pipeline::CoincidenceCohortBuilder;
using citlali::pipeline::FinitePcaPlaceholder;
using citlali::pipeline::NativeInvalidityProvenance;
using citlali::pipeline::NativeOperationIdentity;
using citlali::pipeline::NativeRevisionAction;
using citlali::pipeline::NativeSampleIdentity;
using citlali::pipeline::NativeSampleLedger;
using citlali::pipeline::make_pca_rectangular_working_set;
using citlali::pipeline::scatter_pca_results_transactionally;

std::uint64_t bits_of(double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

TEST(sci_align_native_scatter,
     scatters_only_real_mapped_samples_and_preserves_invalid_values) {
    const NativeSampleIdentity valid{0, 40, 2000.001};
    const NativeSampleIdentity invalid{1, 50, 2000.004};
    constexpr double valid_measured = 3.25;
    constexpr double invalid_measured = -9.5;
    NativeSampleLedger<double> ledger{{
        {valid, valid_measured},
        {invalid, invalid_measured},
    }};

    CoincidenceCohortBuilder builder{
        NativeOperationIdentity{10, 3}, {0, 1, 2}, 1};
    builder.assign_mapped_valid(0, 0, valid, 0);
    builder.assign_mapped_invalid(
        1, 0, invalid, 0,
        NativeInvalidityProvenance{0x4U, "PCA-invalid native sample"});
    builder.assign_absent(
        2, 0, CoincidenceAbsenceReason::no_candidate);
    auto cohort = std::move(builder).finish();

    constexpr double placeholder = 321000.0;
    auto result = make_pca_rectangular_working_set(
        ledger, cohort, FinitePcaPlaceholder::checked(placeholder));
    result.mutable_value(0, 0) = 4.75;
    result.mutable_value(0, 1) = -888888.0;
    result.mutable_value(0, 2) = 777777.0;

    CoincidenceCohortBuilder altered_builder{
        NativeOperationIdentity{10, 3}, {0, 1, 2}, 1};
    altered_builder.assign_mapped_valid(0, 0, valid, 0);
    altered_builder.assign_mapped_invalid(
        1, 0, invalid, 0,
        NativeInvalidityProvenance{0x8U, "different native flag"});
    altered_builder.assign_absent(
        2, 0, CoincidenceAbsenceReason::no_candidate);
    auto altered = std::move(altered_builder).finish();
    EXPECT_THROW(
        scatter_pca_results_transactionally(ledger, altered, result),
        std::logic_error);
    EXPECT_DOUBLE_EQ(ledger.at(valid.key()).current_value,
                     valid_measured);
    EXPECT_DOUBLE_EQ(ledger.at(invalid.key()).current_value,
                     invalid_measured);
    EXPECT_FALSE(ledger.last_operation().has_value());

    scatter_pca_results_transactionally(ledger, cohort, result);

    const auto &valid_record = ledger.at(valid.key());
    EXPECT_EQ(bits_of(valid_record.measured_value),
              bits_of(valid_measured));
    EXPECT_DOUBLE_EQ(valid_record.current_value, 4.75);
    EXPECT_EQ(valid_record.revision, 1U);
    ASSERT_EQ(valid_record.lineage.size(), 1U);
    EXPECT_EQ(valid_record.lineage[0].action,
              NativeRevisionAction::replaced_by_operation_result);
    ASSERT_TRUE(valid_record.lineage[0]
                    .coincidence_provenance.has_value());
    EXPECT_EQ(valid_record.lineage[0]
                  .coincidence_provenance->common_slot,
              0U);
    EXPECT_EQ(valid_record.lineage[0]
                  .coincidence_provenance->participant_index,
              0U);
    EXPECT_EQ(valid_record.lineage[0]
                  .coincidence_provenance->participant_network_id,
              0);
    EXPECT_FALSE(valid_record.lineage[0]
                     .coincidence_provenance->original_flag_bits
                     .has_value());

    const auto &invalid_record = ledger.at(invalid.key());
    EXPECT_EQ(bits_of(invalid_record.measured_value),
              bits_of(invalid_measured));
    EXPECT_EQ(bits_of(invalid_record.current_value),
              bits_of(invalid_measured));
    EXPECT_EQ(invalid_record.revision, 1U);
    ASSERT_EQ(invalid_record.lineage.size(), 1U);
    EXPECT_EQ(invalid_record.lineage[0].action,
              NativeRevisionAction::preserved_pca_invalid);
    ASSERT_TRUE(invalid_record.lineage[0]
                    .coincidence_provenance.has_value());
    ASSERT_TRUE(invalid_record.lineage[0]
                    .coincidence_provenance->original_flag_bits
                    .has_value());
    EXPECT_EQ(*invalid_record.lineage[0]
                   .coincidence_provenance->original_flag_bits,
              0x4U);
    EXPECT_EQ(invalid_record.lineage[0]
                  .coincidence_provenance->original_flag_reason,
              "PCA-invalid native sample");

    EXPECT_EQ(ledger.size(), 2U);
    ASSERT_TRUE(ledger.last_operation().has_value());
    EXPECT_EQ(ledger.last_operation()->sequence, 10U);
}

TEST(sci_align_native_scatter,
     rejects_a_stale_batch_before_mutating_any_destination) {
    const NativeSampleIdentity first{0, 1, 100.0};
    const NativeSampleIdentity second{1, 2, 100.001};
    NativeSampleLedger<double> ledger{{{first, 1.0}, {second, 2.0}}};

    ledger.apply_transaction(
        NativeOperationIdentity{1, 0},
        {NativeSampleLedger<double>::Update::replacement(first, 0, 10.0),
         NativeSampleLedger<double>::Update::replacement(second, 0, 20.0)});
    ASSERT_EQ(ledger.at(first.key()).revision, 1U);
    ASSERT_EQ(ledger.at(second.key()).revision, 1U);

    const auto before_first = ledger.at(first.key()).current_value;
    const auto before_second = ledger.at(second.key()).current_value;
    EXPECT_THROW(
        ledger.apply_transaction(
            NativeOperationIdentity{2, 0},
            {NativeSampleLedger<double>::Update::replacement(
                 first, 1, 30.0),
             NativeSampleLedger<double>::Update::replacement(
                 second, 0, 40.0)}),
        std::logic_error);

    EXPECT_DOUBLE_EQ(ledger.at(first.key()).current_value, before_first);
    EXPECT_DOUBLE_EQ(ledger.at(second.key()).current_value, before_second);
    EXPECT_EQ(ledger.at(first.key()).revision, 1U);
    EXPECT_EQ(ledger.at(second.key()).revision, 1U);
    ASSERT_TRUE(ledger.last_operation().has_value());
    EXPECT_EQ(ledger.last_operation()->sequence, 1U);
}

TEST(sci_align_native_scatter,
     rejects_cross_cohort_mapping_and_nonfinite_result_atomically) {
    const NativeSampleIdentity first{0, 1, 500.0};
    const NativeSampleIdentity second{1, 2, 500.002};
    const NativeSampleIdentity other{0, 3, 500.008};
    NativeSampleLedger<double> ledger{
        {{first, 1.0}, {second, 2.0}, {other, 3.0}}};

    CoincidenceCohortBuilder source_builder{
        NativeOperationIdentity{30, 1}, {0, 1}, 1};
    source_builder.assign_mapped_valid(0, 0, first, 0);
    source_builder.assign_mapped_valid(1, 0, second, 0);
    auto source = std::move(source_builder).finish();
    auto result = make_pca_rectangular_working_set(
        ledger, source, FinitePcaPlaceholder::checked(0.0));
    result.mutable_value(0, 0) = 10.0;
    result.mutable_value(0, 1) = 20.0;

    CoincidenceCohortBuilder other_builder{
        NativeOperationIdentity{30, 1}, {0, 1}, 1};
    other_builder.assign_mapped_valid(0, 0, other, 0);
    other_builder.assign_mapped_valid(1, 0, second, 0);
    auto other_cohort = std::move(other_builder).finish();
    EXPECT_THROW(
        scatter_pca_results_transactionally(
            ledger, other_cohort, result),
        std::logic_error);
    EXPECT_DOUBLE_EQ(ledger.at(first.key()).current_value, 1.0);
    EXPECT_DOUBLE_EQ(ledger.at(second.key()).current_value, 2.0);
    EXPECT_DOUBLE_EQ(ledger.at(other.key()).current_value, 3.0);
    EXPECT_FALSE(ledger.last_operation().has_value());

    result.mutable_value(0, 1) =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        scatter_pca_results_transactionally(ledger, source, result),
        std::logic_error);
    EXPECT_DOUBLE_EQ(ledger.at(first.key()).current_value, 1.0);
    EXPECT_DOUBLE_EQ(ledger.at(second.key()).current_value, 2.0);
    EXPECT_FALSE(ledger.last_operation().has_value());
}

TEST(sci_align_native_scatter,
     permits_explicit_reuse_only_in_a_later_operation_revision) {
    const NativeSampleIdentity sample{0, 7, 400.0};
    NativeSampleLedger<double> ledger{{{sample, 8.0}}};

    CoincidenceCohortBuilder first_builder{
        NativeOperationIdentity{3, 5}, {0}, 1};
    first_builder.assign_mapped_valid(0, 0, sample, 0);
    auto first = std::move(first_builder).finish();
    auto first_result = make_pca_rectangular_working_set(
        ledger, first, FinitePcaPlaceholder::checked(0.0));
    first_result.mutable_value(0, 0) = 9.0;
    scatter_pca_results_transactionally(ledger, first, first_result);

    EXPECT_THROW(
        scatter_pca_results_transactionally(ledger, first, first_result),
        std::logic_error);

    CoincidenceCohortBuilder second_builder{
        NativeOperationIdentity{7, 5}, {0}, 1};
    second_builder.assign_mapped_valid(0, 0, sample, 1);
    auto second = std::move(second_builder).finish();
    auto second_result = make_pca_rectangular_working_set(
        ledger, second, FinitePcaPlaceholder::checked(0.0));
    second_result.mutable_value(0, 0) = 11.0;
    scatter_pca_results_transactionally(ledger, second, second_result);

    const auto &record = ledger.at(sample.key());
    EXPECT_DOUBLE_EQ(record.measured_value, 8.0);
    EXPECT_DOUBLE_EQ(record.current_value, 11.0);
    EXPECT_EQ(record.revision, 2U);
    ASSERT_EQ(record.lineage.size(), 2U);
    EXPECT_EQ(record.lineage[0].operation.sequence, 3U);
    EXPECT_EQ(record.lineage[0].input_revision, 0U);
    EXPECT_EQ(record.lineage[0].output_revision, 1U);
    EXPECT_EQ(record.lineage[1].operation.sequence, 7U);
    EXPECT_EQ(record.lineage[1].input_revision, 1U);
    EXPECT_EQ(record.lineage[1].output_revision, 2U);
}

}  // namespace
