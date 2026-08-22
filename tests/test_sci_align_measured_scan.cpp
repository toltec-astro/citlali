#include <citlali/core/pipeline/timestream_measured_scan.h>
#include <citlali/core/utils/sha256.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;
namespace apt = pipeline::canonical_apt_v2;

std::string digest(char value) {
    return "sha256:" + std::string(64, value);
}

apt::IssuanceContext issuance(std::string occurrence,
                              std::string event) {
    return {std::move(occurrence), std::move(event), "citlali",
            "stage3-fixture-revision", digest('1'),
            "2026-08-22T15:00:00.000Z"};
}

apt::FieldRule baseline_field(std::int64_t uid, std::string name,
                              apt::ValueType datatype,
                              std::string unit) {
    return {uid, std::move(name), datatype, std::move(unit), false,
            "beammap-baseline", std::string("citlali:beammap-fit-v1"),
            apt::FieldOperation::preserve_target, std::nullopt, "reject",
            "nonidentity", "stage3 baseline fixture field"};
}

apt::FieldRule copied_field(const apt::FieldRule &baseline) {
    auto result = baseline;
    result.nullable = true;
    result.operation = apt::FieldOperation::copy_seed_or_null;
    result.missing_policy = "typed-null";
    return result;
}

std::vector<apt::FieldRule> baseline_rules() {
    auto result = apt::canonical_structural_field_rules_v2();
    result.push_back(
        baseline_field(5, "flag", apt::ValueType::int64, "N/A"));
    result.push_back(
        baseline_field(6, "fg", apt::ValueType::int64, "N/A"));
    result.push_back(
        baseline_field(7, "a_fwhm", apt::ValueType::float64, "arcsec"));
    result.push_back(
        baseline_field(8, "b_fwhm", apt::ValueType::float64, "arcsec"));
    result.push_back(
        baseline_field(9, "angle", apt::ValueType::float64, "rad"));
    return result;
}

std::vector<apt::FieldRule> matched_rules() {
    auto result = apt::canonical_structural_field_rules_v2();
    const auto baseline = baseline_rules();
    for (auto it = baseline.begin() + 5; it != baseline.end(); ++it) {
        result.push_back(copied_field(*it));
    }
    auto kmp = apt::canonical_kmp_field_rules_v2(true);
    for (std::size_t index = 0; index < kmp.size(); ++index) {
        kmp[index].field_uid = static_cast<std::int64_t>(10 + index);
        result.push_back(kmp[index]);
    }
    return result;
}

std::map<std::string, apt::Value> baseline_values(std::int64_t index) {
    return {{"flag", std::int64_t{0}},
            {"fg", index},
            {"a_fwhm", 8.0 + static_cast<double>(index)},
            {"b_fwhm", 9.0 + static_cast<double>(index)},
            {"angle", 0.1 * static_cast<double>(index)}};
}

void add_kmp_values(std::map<std::string, apt::Value> &fields,
                    std::int64_t index) {
    const double frequency = 2.0e9 + 1.0e6 * index;
    fields["kids_fr"] = frequency;
    fields["kids_f_out"] = frequency;
    fields["kids_Qr"] = 15000.0 + 100.0 * index;
    fields["kids_flag"] = std::int64_t{90 + index};
}

struct DetectorSpec {
    std::int64_t network;
    std::int64_t channel;
    std::int64_t raw_source_uid;
    std::int64_t kmp_source_uid;
    std::int64_t input_uid;
    std::int64_t output_uid;
    std::uint64_t presentation_rank;
};

const std::array<DetectorSpec, 4> detector_specs{{
    {0, 0, 20, 21, 77, 0, 1},
    {0, 1, 20, 21, 77, 1, 3},
    {7, 0, 22, 23, 78, INT64_C(9007199254740993), 2},
    {7, 1, 22, 23, 78, std::numeric_limits<std::int64_t>::max(), 0},
}};

std::shared_ptr<const pipeline::CanonicalAptDetectorRelationV2>
relation_fixture() {
    const apt::ObservationIdentity baseline_observation{148670, 0, 0};
    const apt::ObservationIdentity observation{148669, 0, 2};

    apt::AptTable baseline_table;
    baseline_table.kind = apt::BundleKind::baseline;
    baseline_table.issuance = issuance(
        "urn:citlali:stage3:baseline:occurrence",
        "urn:citlali:stage3:baseline:event");
    baseline_table.observation = baseline_observation;
    baseline_table.field_rules = baseline_rules();
    for (std::size_t index = 0; index < detector_specs.size(); ++index) {
        const auto &spec = detector_specs[index];
        baseline_table.rows.push_back({
            static_cast<std::int64_t>(10 + index), index,
            1.0e9 + 1.0e6 * static_cast<double>(index),
            apt::array_for_network(spec.network), spec.network,
            spec.channel,
            baseline_values(static_cast<std::int64_t>(index))});
    }
    const std::vector<apt::SourceRecord> baseline_sources{
        {3, apt::SourceRole::raw, digest('2'), 1024,
         baseline_observation, 0, "toltec0", 2},
        {4, apt::SourceRole::raw, digest('3'), 1024,
         baseline_observation, 7, "toltec7", 2},
    };
    auto baseline = apt::verify_bundle_payload(
        apt::prepare_baseline_bundle(baseline_table, baseline_sources)
            .payload);

    apt::AptTable output;
    output.kind = apt::BundleKind::matched;
    output.issuance = issuance(
        "urn:citlali:stage3:matched:occurrence",
        "urn:citlali:stage3:matched:event");
    output.observation = observation;
    output.field_rules = matched_rules();
    for (std::size_t index = 0; index < detector_specs.size(); ++index) {
        const auto &spec = detector_specs[index];
        auto fields = baseline_values(static_cast<std::int64_t>(index));
        add_kmp_values(fields, static_cast<std::int64_t>(index));
        output.rows.push_back({
            spec.output_uid, spec.presentation_rank,
            2.0e9 + 1.0e6 * static_cast<double>(index),
            apt::array_for_network(spec.network), spec.network,
            spec.channel, std::move(fields)});
    }

    std::vector<apt::SourceRecord> sources{
        {20, apt::SourceRole::raw, digest('4'), 2048,
         observation, 0, "toltec0", 2},
        {21, apt::SourceRole::kmp, digest('5'), 4096,
         baseline_observation, 0, "toltec0", 2},
        {22, apt::SourceRole::raw, digest('6'), 2048,
         observation, 7, "toltec7", 2},
        {23, apt::SourceRole::kmp, digest('7'), 4096,
         baseline_observation, 7, "toltec7", 2},
    };

    apt::TargetManifest target;
    target.issuance = issuance(
        "urn:citlali:stage3:target:occurrence",
        "urn:citlali:stage3:target:event");
    target.observation = observation;
    target.sources = sources;
    const std::array<std::uint64_t, 4> source_ranks{2, 0, 3, 1};
    const std::array<std::uint64_t, 4> application_ranks{1, 3, 0, 2};
    for (std::size_t index = 0; index < detector_specs.size(); ++index) {
        const auto &spec = detector_specs[index];
        std::map<std::string, apt::Value> fields;
        add_kmp_values(fields, static_cast<std::int64_t>(index));
        target.rows.push_back({
            static_cast<std::int64_t>(30 + index), spec.input_uid,
            spec.raw_source_uid, spec.kmp_source_uid, spec.channel,
            source_ranks[index], application_ranks[index],
            2.0e9 + 1.0e6 * static_cast<double>(index),
            apt::array_for_network(spec.network), spec.network,
            spec.channel, std::move(fields)});
    }

    apt::RelationTable relation;
    relation.issuance = output.issuance;
    relation.observation = observation;
    relation.target_parent = apt::target_identity(target);
    relation.target_issuance = target.issuance;
    relation.baseline_parent = baseline.identity;
    relation.matcher = {
        "urn:tolapt:stage3:matcher", digest('8'), digest('9'),
        "observation-tone-match-v2", "tolapt"};
    relation.network_evidence = {
        {40, 0, apt::NetworkEvidenceStatus::matched_capable,
         0.0, 200000.0, 15000.0},
        {41, 7, apt::NetworkEvidenceStatus::matched_capable,
         0.0, 200000.0, 15000.0},
    };
    for (std::size_t index = 0; index < detector_specs.size(); ++index) {
        const auto &spec = detector_specs[index];
        relation.rows.push_back({
            static_cast<std::int64_t>(60 + index), spec.output_uid,
            {relation.target_parent,
             static_cast<std::int64_t>(30 + index)},
            spec.input_uid, spec.raw_source_uid, spec.kmp_source_uid,
            spec.channel, source_ranks[index], application_ranks[index],
            spec.presentation_rank, apt::RelationDisposition::matched,
            static_cast<std::int64_t>(50 + index),
            apt::ScopedRowReference{
                baseline.identity, static_cast<std::int64_t>(10 + index)},
            100.0 + static_cast<double>(index), true,
            spec.network == 0 ? 40 : 41, "selected-good-seed"});
    }

    auto prepared = apt::prepare_matched_bundle(
        output, relation, sources, {}, baseline);
    auto verified = apt::verify_bundle_payload(std::move(prepared.payload));
    return std::make_shared<const pipeline::CanonicalAptDetectorRelationV2>(
        pipeline::admit_canonical_apt_detector_relation_v2(verified));
}

Eigen::VectorXd vector(std::initializer_list<double> input) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(input.size()));
    Eigen::Index index = 0;
    for (double value : input) result(index++) = value;
    return result;
}

std::shared_ptr<const pipeline::NativeObservationCarriers> carriers_fixture(
    pipeline::NativeObservationScope scope =
        pipeline::NativeObservationScope{148669, 0, 2}) {
    std::vector<pipeline::NativeNetworkAlignment> networks;
    networks.emplace_back(0, 10, vector({1.0, 2.0, 4.0}),
                          std::vector<pipeline::TimestreamPacketCounter>{
                              100, 101, 103});
    networks.emplace_back(7, 70, vector({1.0, 3.0, 4.0}),
                          std::vector<pipeline::TimestreamPacketCounter>{
                              700, 702, 703});
    std::vector<pipeline::NativeSlotAssociation> nw0(4);
    nw0[0].native_row = 10;
    nw0[1].native_row = 11;
    nw0[2].absence_reason =
        pipeline::CoincidenceAbsenceReason::no_candidate;
    nw0[3].native_row = 12;
    std::vector<pipeline::NativeSlotAssociation> nw7(4);
    nw7[0].native_row = 70;
    nw7[1].absence_reason =
        pipeline::CoincidenceAbsenceReason::no_candidate;
    nw7[2].native_row = 71;
    nw7[3].native_row = 72;
    auto alignment = std::make_shared<const pipeline::NativeAlignmentPlan>(
        scope, std::move(networks), vector({1.0, 2.0, 3.0, 4.0}),
        std::map<pipeline::TimestreamNetworkId,
                 std::vector<pipeline::NativeSlotAssociation>>{
            {0, std::move(nw0)}, {7, std::move(nw7)}});

    pipeline::NativeTelescopeData telescope;
    telescope["TelTime"] = vector({0.0, 5.0});
    telescope["TelUTC"] = vector({0.0, 5.0});
    telescope["TelAzAct"] = vector({10.0, 20.0});
    telescope["TelElAct"] = vector({30.0, 40.0});
    auto raw = std::make_shared<const pipeline::RawTelescopeTrajectory>(
        std::move(telescope));
    pipeline::NativePointingOffsetsArcsec offsets;
    offsets[citlali::config::pointing_axis_az()] = vector({0.0});
    offsets[citlali::config::pointing_axis_alt()] = vector({0.0});
    pipeline::NativePointingOffsetModel offset_model{
        std::move(offsets), vector({0.0, 5.0})};
    auto pointing = pipeline::make_native_pointing_plan(
        alignment, raw, offset_model);
    return std::make_shared<const pipeline::NativeObservationCarriers>(
        scope, alignment, pointing);
}

struct InputFixture {
    std::shared_ptr<Eigen::MatrixXd> values0 =
        std::make_shared<Eigen::MatrixXd>(3, 2);
    std::shared_ptr<Eigen::MatrixXd> values7 =
        std::make_shared<Eigen::MatrixXd>(3, 2);
    std::shared_ptr<pipeline::NativeDetectorFlagBitsMatrix> flags0 =
        std::make_shared<pipeline::NativeDetectorFlagBitsMatrix>(3, 2);
    std::shared_ptr<pipeline::NativeDetectorFlagBitsMatrix> flags7 =
        std::make_shared<pipeline::NativeDetectorFlagBitsMatrix>(3, 2);

    InputFixture() {
        *values0 << 100.0, 101.0,
                    110.0, 111.0,
                    120.0, 121.0;
        *values7 << 700.0, 701.0,
                    710.0, 711.0,
                    720.0,
                    std::numeric_limits<double>::quiet_NaN();
        flags0->setZero();
        flags7->setZero();
        (*flags0)(1, 1) = 0x20;
        (*flags7)(2, 1) = 0x80;
    }

    std::vector<pipeline::NativeMeasuredNetworkInput> inputs(
        bool reverse = false) const {
        std::vector<pipeline::NativeMeasuredNetworkInput> result;
        result.emplace_back(20, 0, "toltec0", 10, values0, flags0);
        result.emplace_back(22, 7, "toltec7", 70, values7, flags7);
        if (reverse) std::reverse(result.begin(), result.end());
        return result;
    }
};

pipeline::NativeScanChunkScope scan_scope() {
    return {pipeline::NativeObservationScope{148669, 0, 2}, 5, 0};
}

std::shared_ptr<const pipeline::NativeMeasuredDetectorScan> admitted_scan(
    const InputFixture &inputs, bool reverse = false) {
    return pipeline::NativeMeasuredDetectorScan::admit(
        scan_scope(), carriers_fixture(), relation_fixture(), 0, 4,
        inputs.inputs(reverse));
}

TEST(sci_align_measured_scan,
     joins_exact_channels_uids_flags_and_partial_cohort_cells) {
    const InputFixture inputs;
    const auto scan = admitted_scan(inputs);

    ASSERT_EQ(scan->detector_count(), 4U);
    ASSERT_EQ(scan->measured_sample_count(), 12U);
    EXPECT_EQ(scan->binding(0).network_id, 7);
    EXPECT_EQ(scan->binding(0).raw_channel, 1);
    EXPECT_EQ(scan->binding(0).output_uid,
              std::numeric_limits<std::int64_t>::max());
    EXPECT_EQ(scan->binding(1).network_id, 0);
    EXPECT_EQ(scan->binding(1).output_uid, 0);
    EXPECT_EQ(scan->binding(2).network_id, 7);
    EXPECT_EQ(scan->binding(2).output_uid, INT64_C(9007199254740993));
    EXPECT_EQ(scan->binding(3).network_id, 0);

    const auto nw7_absent = scan->cell(1, 0);
    EXPECT_EQ(nw7_absent.state(), pipeline::CoincidenceCellState::absent);
    EXPECT_FALSE(nw7_absent.mapped());
    EXPECT_EQ(nw7_absent.absence_reason(),
              pipeline::CoincidenceAbsenceReason::no_candidate);
    const auto nw0_absent = scan->cell(2, 1);
    EXPECT_EQ(nw0_absent.state(), pipeline::CoincidenceCellState::absent);

    const auto valid = scan->cell(0, 2);
    ASSERT_TRUE(valid.identity().has_value());
    EXPECT_TRUE(valid.valid());
    EXPECT_EQ(valid.identity()->network_id(), 7);
    EXPECT_EQ(valid.identity()->native_row(), 70);
    ASSERT_TRUE(valid.measured_value().has_value());
    EXPECT_DOUBLE_EQ(*valid.measured_value(), 700.0);
    EXPECT_EQ(valid.original_flag_bits(), 0U);

    const auto flagged = scan->cell(1, 3);
    EXPECT_EQ(flagged.state(),
              pipeline::CoincidenceCellState::mapped_invalid);
    EXPECT_EQ(flagged.original_flag_bits(), 0x20U);
    EXPECT_DOUBLE_EQ(*flagged.measured_value(), 111.0);
    const auto nonfinite = scan->cell(3, 0);
    EXPECT_EQ(nonfinite.state(),
              pipeline::CoincidenceCellState::mapped_invalid);
    EXPECT_TRUE(std::isnan(*nonfinite.measured_value()));
    EXPECT_EQ(nonfinite.original_flag_bits(), 0x80U);
}

TEST(sci_align_measured_scan,
     input_network_permutation_preserves_interleaved_detector_identity) {
    const InputFixture inputs;
    const auto forward = admitted_scan(inputs, false);
    const auto reverse = admitted_scan(inputs, true);
    EXPECT_EQ(forward->bindings(), reverse->bindings());
    for (std::size_t slot = 0; slot < 4; ++slot) {
        for (Eigen::Index detector = 0; detector < 4; ++detector) {
            const auto lhs = forward->cell(slot, detector);
            const auto rhs = reverse->cell(slot, detector);
            EXPECT_EQ(lhs.state(), rhs.state());
            EXPECT_EQ(lhs.identity(), rhs.identity());
            EXPECT_EQ(lhs.original_flag_bits(), rhs.original_flag_bits());
            ASSERT_EQ(lhs.measured_value().has_value(),
                      rhs.measured_value().has_value());
            if (!lhs.measured_value()) continue;
            if (std::isnan(*lhs.measured_value())) {
                EXPECT_TRUE(std::isnan(*rhs.measured_value()));
            }
            else {
                EXPECT_DOUBLE_EQ(*lhs.measured_value(),
                                 *rhs.measured_value());
            }
        }
    }
}

TEST(sci_align_measured_scan,
     retains_existing_matrix_owners_and_seeds_fresh_identity_ledger) {
    const InputFixture inputs;
    const auto scan = admitted_scan(inputs);
    EXPECT_EQ(scan->network_input(0).measured_values_handle().get(),
              inputs.values0.get());
    EXPECT_EQ(scan->network_input(7).measured_values_handle().get(),
              inputs.values7.get());
    EXPECT_EQ(scan->network_input(0).original_flag_bits_handle().get(),
              inputs.flags0.get());

    pipeline::NativeMeasuredDetectorLedger ledger{scan};
    EXPECT_EQ(ledger.size(), 12U);
    const pipeline::NativeDetectorSampleKey key{{7, 70}, 2};
    const auto record = ledger.record(key);
    EXPECT_EQ(record.identity.network_id(), 7);
    EXPECT_EQ(record.identity.native_row(), 70);
    EXPECT_EQ(record.detector_column, 2);
    EXPECT_DOUBLE_EQ(record.measured_value, 700.0);
    EXPECT_DOUBLE_EQ(record.current_value, 700.0);
    EXPECT_EQ(record.original_flag_bits, 0U);
    EXPECT_EQ(record.revision, 0U);
    EXPECT_THROW(
        ledger.record(pipeline::NativeDetectorSampleKey{{0, 10}, 2}),
        std::invalid_argument);
}

TEST(sci_align_measured_scan,
     scan_transaction_rejects_candidates_before_mutating_live_lifecycle) {
    const InputFixture inputs;
    pipeline::NativeMeasuredScanTransaction rejected{scan_scope()};

    auto missing = inputs.inputs();
    missing.pop_back();
    EXPECT_THROW(
        rejected.admit(carriers_fixture(), relation_fixture(), 0, 4,
                       std::move(missing)),
        std::invalid_argument);
    EXPECT_FALSE(rejected.active());

    auto duplicate = inputs.inputs();
    duplicate[1] = duplicate[0];
    EXPECT_THROW(
        rejected.admit(carriers_fixture(), relation_fixture(), 0, 4,
                       std::move(duplicate)),
        std::invalid_argument);
    EXPECT_FALSE(rejected.active());

    std::vector<pipeline::NativeMeasuredNetworkInput> omitted_row;
    omitted_row.emplace_back(
        20, 0, "toltec0", 10,
        std::make_shared<const Eigen::MatrixXd>(inputs.values0->topRows(2)),
        std::make_shared<const pipeline::NativeDetectorFlagBitsMatrix>(
            inputs.flags0->topRows(2)));
    omitted_row.emplace_back(22, 7, "toltec7", 70, inputs.values7,
                             inputs.flags7);
    EXPECT_THROW(
        rejected.admit(carriers_fixture(), relation_fixture(), 0, 4,
                       std::move(omitted_row)),
        std::invalid_argument);
    EXPECT_FALSE(rejected.active());

    EXPECT_THROW(
        rejected.admit(
            carriers_fixture(
                pipeline::NativeObservationScope{148670, 0, 2}),
            relation_fixture(), 0, 4,
            inputs.inputs()),
        std::invalid_argument);
    EXPECT_FALSE(rejected.active());

    pipeline::NativeMeasuredScanTransaction transaction{scan_scope()};
    transaction.admit(carriers_fixture(), relation_fixture(), 0, 4,
                      inputs.inputs());
    const auto accepted = transaction.mapping_handle();
    const auto operation = transaction.ledger().issue_operation();
    EXPECT_EQ(operation.sequence, 0U);
    EXPECT_THROW(
        transaction.admit(carriers_fixture(), relation_fixture(), 0, 4,
                          inputs.inputs()),
        std::logic_error);
    EXPECT_EQ(transaction.mapping_handle().get(), accepted.get());
    ASSERT_TRUE(transaction.ledger().last_operation().has_value());
    EXPECT_EQ(transaction.ledger().last_operation()->sequence, 0U);
}

TEST(sci_align_measured_scan,
     commit_rollback_and_retry_reset_ledger_and_operation_sequence) {
    const InputFixture inputs;
    pipeline::NativeMeasuredScanTransaction transaction{scan_scope()};
    transaction.admit(carriers_fixture(), relation_fixture(), 0, 4,
                      inputs.inputs());
    EXPECT_EQ(transaction.ledger().issue_operation().sequence, 0U);
    EXPECT_EQ(transaction.ledger().issue_operation().sequence, 1U);
    transaction.rollback();
    EXPECT_FALSE(transaction.active());
    EXPECT_THROW((void)transaction.mapping(), std::logic_error);

    transaction.admit(carriers_fixture(), relation_fixture(), 0, 4,
                      inputs.inputs(true));
    EXPECT_EQ(transaction.ledger().issue_operation().sequence, 0U);
    transaction.commit();
    EXPECT_FALSE(transaction.active());
    EXPECT_THROW((void)transaction.ledger(), std::logic_error);
}

TEST(sci_align_measured_scan,
     malformed_sources_shapes_and_slot_windows_fail_closed) {
    const InputFixture inputs;
    EXPECT_THROW(
        ((void)pipeline::NativeMeasuredNetworkInput{
            20, 0, "toltec0", 10, nullptr, inputs.flags0}),
        std::invalid_argument);
    EXPECT_THROW(
        ((void)pipeline::NativeMeasuredNetworkInput{
            20, 0, "toltec0", 10, inputs.values0,
            std::make_shared<const pipeline::NativeDetectorFlagBitsMatrix>(
                2, 2)}),
        std::invalid_argument);

    auto wrong_interface = inputs.inputs();
    wrong_interface[0] = pipeline::NativeMeasuredNetworkInput{
        20, 0, "toltec1", 10, inputs.values0, inputs.flags0};
    EXPECT_THROW(
        pipeline::NativeMeasuredDetectorScan::admit(
            scan_scope(), carriers_fixture(), relation_fixture(), 0, 4,
            std::move(wrong_interface)),
        std::invalid_argument);

    auto one_channel_values =
        std::make_shared<Eigen::MatrixXd>(inputs.values0->leftCols(1));
    auto one_channel_flags =
        std::make_shared<pipeline::NativeDetectorFlagBitsMatrix>(
            inputs.flags0->leftCols(1));
    auto wrong_channel_count = inputs.inputs();
    wrong_channel_count[0] = pipeline::NativeMeasuredNetworkInput{
        20, 0, "toltec0", 10, one_channel_values, one_channel_flags};
    EXPECT_THROW(
        pipeline::NativeMeasuredDetectorScan::admit(
            scan_scope(), carriers_fixture(), relation_fixture(), 0, 4,
            std::move(wrong_channel_count)),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::NativeMeasuredDetectorScan::admit(
            scan_scope(), carriers_fixture(), relation_fixture(), 0, 5,
            inputs.inputs()),
        std::out_of_range);
    EXPECT_THROW(
        pipeline::NativeMeasuredDetectorScan::admit(
            scan_scope(), carriers_fixture(), relation_fixture(), 2, 2,
            inputs.inputs()),
        std::invalid_argument);
}

}  // namespace
