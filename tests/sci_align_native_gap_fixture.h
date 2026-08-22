#pragma once

#include <citlali/core/pipeline/timestream_measured_scan.h>
#include <citlali/core/utils/sha256.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef CITLALI_SCI_ALIGN_FIXTURE_DIR
#error "CITLALI_SCI_ALIGN_FIXTURE_DIR must name the SCI-ALIGN fixture directory"
#endif

namespace citlali::test_support::sci_align {

inline constexpr auto native_gap_fixture_v1_schema =
    "citlali-sci-align-native-gap-fixture-v1";
inline constexpr auto native_gap_fixture_v1_id =
    "urn:citlali:sci-align:native-gap:v1";
inline constexpr auto native_gap_fixture_v1_sha256 =
    "a4dfdfe4b45638952f57f5f258badfab84f5d6ce1d022abfefc47a9e84091701";

struct NativeGapDetectorColumnV1 {
    Eigen::Index detector_column = -1;
    pipeline::TimestreamNetworkId network_id = -1;
    std::int64_t raw_source_uid = -1;
    Eigen::Index raw_channel = -1;
    std::int64_t output_uid = -1;
};

struct NativeGapIntervalV1 {
    std::int64_t first = -1;
    std::int64_t past_last = -1;

    friend bool operator==(const NativeGapIntervalV1 &,
                           const NativeGapIntervalV1 &) = default;
};

struct NativeGapStage4SupportV1 {
    std::size_t segment_ordinal = 0;
    pipeline::TimestreamNetworkId network_id = -1;
    std::size_t first_common_slot = 0;
    std::size_t past_last_common_slot = 0;
    pipeline::TimestreamNativeRow selected_anchor_native_row = -1;
    std::vector<pipeline::NativeDetectorFlagBits>
        original_flag_or_by_channel;
};

struct NativeGapNetworkV1 {
    pipeline::TimestreamNetworkId network_id = -1;
    std::int64_t raw_source_uid = -1;
    std::string interface;
    pipeline::TimestreamNativeRow first_native_row = -1;
    Eigen::VectorXd reconstructed_times_unix_sec;
    std::vector<pipeline::TimestreamPacketCounter> packet_counters;
    Eigen::VectorXi legacy_presence_mask;
    std::vector<std::optional<pipeline::TimestreamNativeRow>>
        expected_slot_native_rows;
    Eigen::MatrixXd measured_values;
    pipeline::NativeDetectorFlagBitsMatrix original_flag_bits;
    std::vector<NativeGapIntervalV1> expected_packet_contiguous_runs;
};

struct NativeGapFixtureV1 {
    std::filesystem::path source_path;
    std::string source_sha256;
    std::string accepted_plan_commit;
    std::vector<std::string> historical_evidence_commits;
    pipeline::NativeObservationScope scope{1, 0, 0};
    std::int64_t scan_index = -1;
    std::int64_t chunk_index = -1;
    double realized_dt_sec = 0.0;
    Eigen::VectorXd common_slot_reference_times_unix_sec;
    std::vector<NativeGapDetectorColumnV1> detector_columns;
    std::vector<NativeGapNetworkV1> networks;
    std::vector<NativeGapIntervalV1> expected_complete_cohort_slot_runs;
    std::vector<NativeGapStage4SupportV1>
        expected_stage4_factor2_support;

    NativeGapNetworkV1 &network(
        pipeline::TimestreamNetworkId network_id) {
        const auto found = std::find_if(
            networks.begin(), networks.end(), [&](const auto &candidate) {
                return candidate.network_id == network_id;
            });
        if (found == networks.end()) {
            throw std::out_of_range(
                "network is absent from native-gap fixture");
        }
        return *found;
    }

    const NativeGapNetworkV1 &network(
        pipeline::TimestreamNetworkId network_id) const {
        const auto found = std::find_if(
            networks.begin(), networks.end(), [&](const auto &candidate) {
                return candidate.network_id == network_id;
            });
        if (found == networks.end()) {
            throw std::out_of_range(
                "network is absent from native-gap fixture");
        }
        return *found;
    }

    std::shared_ptr<const pipeline::NativeAlignmentPlan>
    materialize_alignment() const {
        std::vector<pipeline::NativeNetworkAlignment> alignments;
        std::map<pipeline::TimestreamNetworkId,
                 std::vector<pipeline::NativeSlotAssociation>> associations;
        for (const auto &input : networks) {
            pipeline::NativeNetworkAlignment alignment{
                input.network_id, input.first_native_row,
                input.reconstructed_times_unix_sec,
                input.packet_counters};
            auto mapped = pipeline::make_gap_native_slot_associations(
                alignment, common_slot_reference_times_unix_sec,
                input.legacy_presence_mask, realized_dt_sec);
            if (mapped.size() != input.expected_slot_native_rows.size()) {
                throw std::logic_error(
                    "fixture association cardinality does not match oracle");
            }
            for (std::size_t slot = 0; slot < mapped.size(); ++slot) {
                const auto expected = input.expected_slot_native_rows[slot];
                if (expected.has_value() != mapped[slot].mapped() ||
                    (expected.has_value() &&
                     *expected != mapped[slot].native_row)) {
                    throw std::logic_error(
                        "fixture association differs from frozen oracle");
                }
            }
            associations.emplace(input.network_id, std::move(mapped));
            alignments.push_back(std::move(alignment));
        }
        return std::make_shared<const pipeline::NativeAlignmentPlan>(
            scope, std::move(alignments),
            common_slot_reference_times_unix_sec,
            std::move(associations));
    }

    std::shared_ptr<const pipeline::NativeObservationCarriers>
    materialize_carriers() const {
        const auto alignment = materialize_alignment();
        pipeline::NativeTelescopeData telescope;
        Eigen::VectorXd support(2);
        support << common_slot_reference_times_unix_sec(0) - 1.0,
            common_slot_reference_times_unix_sec(
                common_slot_reference_times_unix_sec.size() - 1) + 1.0;
        telescope["TelTime"] = support;
        telescope["TelUTC"] = support;
        telescope["TelAzAct"] = Eigen::Vector2d{10.0, 11.0};
        telescope["TelElAct"] = Eigen::Vector2d{50.0, 51.0};
        auto raw =
            std::make_shared<const pipeline::RawTelescopeTrajectory>(
                std::move(telescope));
        pipeline::NativePointingOffsetsArcsec offsets;
        offsets[citlali::config::pointing_axis_az()] =
            Eigen::VectorXd::Zero(1);
        offsets[citlali::config::pointing_axis_alt()] =
            Eigen::VectorXd::Zero(1);
        pipeline::NativePointingOffsetModel offset_model{
            std::move(offsets), support};
        auto pointing = pipeline::make_native_pointing_plan(
            alignment, raw, offset_model);
        return std::make_shared<const pipeline::NativeObservationCarriers>(
            scope, alignment, std::move(pointing));
    }
};

namespace native_gap_detail {

namespace apt = pipeline::canonical_apt_v2;

inline std::string digest(char value) {
    return "sha256:" + std::string(64, value);
}

inline apt::IssuanceContext issuance(std::string occurrence,
                                     std::string event) {
    return {std::move(occurrence), std::move(event), "citlali",
            "native-gap-v1-test-support", digest('1'),
            "2026-08-22T17:00:00.000Z"};
}

inline apt::FieldRule baseline_flag_rule() {
    return {5, "flag", apt::ValueType::int64, "N/A", false,
            "beammap-baseline", std::string("citlali:beammap-fit-v1"),
            apt::FieldOperation::preserve_target, std::nullopt, "reject",
            "nonidentity", "native-gap baseline selection flag"};
}

inline std::vector<apt::FieldRule> baseline_rules() {
    auto result = apt::canonical_structural_field_rules_v2();
    result.push_back(baseline_flag_rule());
    return result;
}

inline std::vector<apt::FieldRule> matched_rules() {
    auto result = apt::canonical_structural_field_rules_v2();
    auto flag = baseline_flag_rule();
    flag.nullable = true;
    flag.operation = apt::FieldOperation::copy_seed_or_null;
    flag.missing_policy = "typed-null";
    result.push_back(std::move(flag));
    auto kmp = apt::canonical_kmp_field_rules_v2(true);
    for (std::size_t index = 0; index < kmp.size(); ++index) {
        kmp[index].field_uid = static_cast<std::int64_t>(6 + index);
        result.push_back(kmp[index]);
    }
    return result;
}

inline std::map<std::string, apt::Value> kmp_values(
    std::int64_t detector_column) {
    const double frequency =
        2.0e9 + 1.0e6 * static_cast<double>(detector_column);
    return {{"kids_fr", frequency},
            {"kids_f_out", frequency},
            {"kids_Qr", 15000.0 + detector_column},
            {"kids_flag", std::int64_t{70 + detector_column}}};
}

inline std::map<std::string, apt::Value> matched_values(
    std::int64_t detector_column) {
    auto result = kmp_values(detector_column);
    result["flag"] = std::int64_t{0};
    return result;
}

inline std::shared_ptr<const pipeline::CanonicalAptDetectorRelationV2>
materialize_relation(const NativeGapFixtureV1 &fixture) {
    const apt::ObservationIdentity observation{
        fixture.scope.observation, fixture.scope.subobservation,
        fixture.scope.scan};
    const apt::ObservationIdentity baseline_observation{
        fixture.scope.observation + 1, 0, 0};

    apt::AptTable baseline;
    baseline.kind = apt::BundleKind::baseline;
    baseline.issuance = issuance(
        "urn:citlali:native-gap:baseline:occurrence",
        "urn:citlali:native-gap:baseline:event");
    baseline.observation = baseline_observation;
    baseline.field_rules = baseline_rules();
    for (const auto &detector : fixture.detector_columns) {
        baseline.rows.push_back({
            10 + detector.detector_column,
            static_cast<std::uint64_t>(detector.detector_column),
            1.0e9 + 1.0e6 * detector.detector_column,
            apt::array_for_network(detector.network_id),
            detector.network_id, detector.raw_channel,
            {{"flag", std::int64_t{0}}}});
    }
    const std::vector<apt::SourceRecord> baseline_sources{
        {3, apt::SourceRole::raw, digest('2'), 1024,
         baseline_observation, 0, "toltec0", 2},
        {4, apt::SourceRole::raw, digest('3'), 1024,
         baseline_observation, 7, "toltec7", 2}};
    auto verified_baseline = apt::verify_bundle_payload(
        apt::prepare_baseline_bundle(baseline, baseline_sources).payload);

    apt::AptTable output;
    output.kind = apt::BundleKind::matched;
    output.issuance = issuance(
        "urn:citlali:native-gap:matched:occurrence",
        "urn:citlali:native-gap:matched:event");
    output.observation = observation;
    output.field_rules = matched_rules();
    for (const auto &detector : fixture.detector_columns) {
        output.rows.push_back({
            detector.output_uid,
            static_cast<std::uint64_t>(detector.detector_column),
            2.0e9 + 1.0e6 * detector.detector_column,
            apt::array_for_network(detector.network_id),
            detector.network_id, detector.raw_channel,
            matched_values(detector.detector_column)});
    }

    std::vector<apt::SourceRecord> sources{
        {20, apt::SourceRole::raw, digest('4'), 2048, observation,
         0, "toltec0", 2},
        {21, apt::SourceRole::kmp, digest('5'), 4096,
         baseline_observation, 0, "toltec0", 2},
        {22, apt::SourceRole::raw, digest('6'), 2048, observation,
         7, "toltec7", 2},
        {23, apt::SourceRole::kmp, digest('7'), 4096,
         baseline_observation, 7, "toltec7", 2}};

    apt::TargetManifest target;
    target.issuance = issuance(
        "urn:citlali:native-gap:target:occurrence",
        "urn:citlali:native-gap:target:event");
    target.observation = observation;
    target.sources = sources;
    for (const auto &detector : fixture.detector_columns) {
        const auto kmp_source_uid =
            detector.network_id == 0 ? INT64_C(21) : INT64_C(23);
        target.rows.push_back({
            30 + detector.detector_column,
            77 + detector.network_id,
            detector.raw_source_uid, kmp_source_uid,
            detector.raw_channel,
            static_cast<std::uint64_t>(detector.detector_column),
            static_cast<std::uint64_t>(detector.detector_column),
            2.0e9 + 1.0e6 * detector.detector_column,
            apt::array_for_network(detector.network_id),
            detector.network_id, detector.raw_channel,
            kmp_values(detector.detector_column)});
    }

    apt::RelationTable relation;
    relation.issuance = output.issuance;
    relation.observation = observation;
    relation.target_parent = apt::target_identity(target);
    relation.target_issuance = target.issuance;
    relation.baseline_parent = verified_baseline.identity;
    relation.matcher = {
        "urn:tolapt:native-gap:matcher", digest('8'), digest('9'),
        "observation-tone-match-v2", "tolapt"};
    relation.network_evidence = {
        {40, 0, apt::NetworkEvidenceStatus::matched_capable,
         0.0, 200000.0, 15000.0},
        {41, 7, apt::NetworkEvidenceStatus::matched_capable,
         0.0, 200000.0, 15000.0}};
    for (const auto &detector : fixture.detector_columns) {
        relation.rows.push_back({
            60 + detector.detector_column, detector.output_uid,
            {relation.target_parent, 30 + detector.detector_column},
            77 + detector.network_id, detector.raw_source_uid,
            detector.network_id == 0 ? INT64_C(21) : INT64_C(23),
            detector.raw_channel,
            static_cast<std::uint64_t>(detector.detector_column),
            static_cast<std::uint64_t>(detector.detector_column),
            static_cast<std::uint64_t>(detector.detector_column),
            apt::RelationDisposition::matched,
            50 + detector.detector_column,
            apt::ScopedRowReference{
                verified_baseline.identity,
                10 + detector.detector_column},
            100.0 + detector.detector_column, true,
            detector.network_id == 0 ? 40 : 41,
            "selected-good-seed"});
    }

    auto prepared = apt::prepare_matched_bundle(
        output, relation, sources, {}, verified_baseline);
    auto verified =
        apt::verify_bundle_payload(std::move(prepared.payload));
    return std::make_shared<const pipeline::CanonicalAptDetectorRelationV2>(
        pipeline::admit_canonical_apt_detector_relation_v2(verified));
}

}  // namespace native_gap_detail

inline std::shared_ptr<const pipeline::NativeMeasuredDetectorScan>
materialize_native_gap_measured_scan(const NativeGapFixtureV1 &fixture) {
    std::vector<pipeline::NativeMeasuredNetworkInput> inputs;
    inputs.reserve(fixture.networks.size());
    for (const auto &network : fixture.networks) {
        inputs.emplace_back(
            network.raw_source_uid, network.network_id, network.interface,
            network.first_native_row,
            std::make_shared<const Eigen::MatrixXd>(
                network.measured_values),
            std::make_shared<const pipeline::NativeDetectorFlagBitsMatrix>(
                network.original_flag_bits));
    }
    return pipeline::NativeMeasuredDetectorScan::admit(
        {fixture.scope, fixture.scan_index, fixture.chunk_index},
        fixture.materialize_carriers(),
        native_gap_detail::materialize_relation(fixture), 0,
        static_cast<std::size_t>(
            fixture.common_slot_reference_times_unix_sec.size()),
        std::move(inputs));
}

inline std::filesystem::path native_gap_fixture_v1_path() {
    return std::filesystem::path{CITLALI_SCI_ALIGN_FIXTURE_DIR} /
           "native_gap_v1.yaml";
}

inline void require_mapping(const YAML::Node &node,
                            const std::string &label) {
    if (!node || !node.IsMap()) {
        throw std::invalid_argument(label + " must be a mapping");
    }
}

inline void require_sequence(const YAML::Node &node,
                             const std::string &label) {
    if (!node || !node.IsSequence() || node.size() == 0) {
        throw std::invalid_argument(
            label + " must be a nonempty sequence");
    }
}

template <class Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> fixture_vector(
    const YAML::Node &node, const std::string &label) {
    require_sequence(node, label);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> result(
        static_cast<Eigen::Index>(node.size()));
    for (std::size_t index = 0; index < node.size(); ++index) {
        result(static_cast<Eigen::Index>(index)) = node[index].as<Scalar>();
    }
    return result;
}

template <class Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> fixture_matrix(
    const YAML::Node &node, const std::string &label) {
    require_sequence(node, label);
    require_sequence(node[0], label + " row");
    const auto columns = node[0].size();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(
        static_cast<Eigen::Index>(node.size()),
        static_cast<Eigen::Index>(columns));
    for (std::size_t row = 0; row < node.size(); ++row) {
        require_sequence(node[row], label + " row");
        if (node[row].size() != columns) {
            throw std::invalid_argument(
                label + " must be rectangular");
        }
        for (std::size_t column = 0; column < columns; ++column) {
            result(static_cast<Eigen::Index>(row),
                   static_cast<Eigen::Index>(column)) =
                node[row][column].as<Scalar>();
        }
    }
    return result;
}

inline NativeGapIntervalV1 fixture_interval(
    const YAML::Node &node, const std::string &first_key,
    const std::string &past_last_key) {
    require_mapping(node, "fixture interval");
    NativeGapIntervalV1 result{
        node[first_key].as<std::int64_t>(),
        node[past_last_key].as<std::int64_t>()};
    if (result.first < 0 || result.first >= result.past_last) {
        throw std::invalid_argument("fixture interval is invalid");
    }
    return result;
}

inline void validate_native_gap_fixture_v1(
    const NativeGapFixtureV1 &fixture) {
    if (fixture.accepted_plan_commit !=
            "a3f2bf465a26048b24017ebd50876c4a2684b1b8" ||
        fixture.historical_evidence_commits !=
            std::vector<std::string>{"fd3627fc7"} ||
        fixture.scan_index < 0 || fixture.chunk_index < 0 ||
        fixture.realized_dt_sec <= 0.0 || fixture.networks.size() != 2 ||
        fixture.detector_columns.size() != 4 ||
        fixture.expected_complete_cohort_slot_runs.size() != 2 ||
        fixture.expected_stage4_factor2_support.size() != 4) {
        throw std::invalid_argument(
            "native-gap fixture identity or cardinality is invalid");
    }

    std::set<pipeline::TimestreamNetworkId> network_ids;
    std::set<std::int64_t> raw_source_uids;
    for (const auto &network : fixture.networks) {
        if (!network_ids.insert(network.network_id).second ||
            !raw_source_uids.insert(network.raw_source_uid).second ||
            network.interface.empty() || network.first_native_row < 0 ||
            network.reconstructed_times_unix_sec.size() <= 0 ||
            network.reconstructed_times_unix_sec.size() !=
                static_cast<Eigen::Index>(network.packet_counters.size()) ||
            network.legacy_presence_mask.size() !=
                fixture.common_slot_reference_times_unix_sec.size() ||
            network.expected_slot_native_rows.size() !=
                static_cast<std::size_t>(
                    fixture.common_slot_reference_times_unix_sec.size()) ||
            network.measured_values.rows() !=
                network.reconstructed_times_unix_sec.size() ||
            network.original_flag_bits.rows() !=
                network.measured_values.rows() ||
            network.original_flag_bits.cols() !=
                network.measured_values.cols()) {
            throw std::invalid_argument(
                "native-gap network shape or identity is invalid");
        }
    }
    if (network_ids !=
            std::set<pipeline::TimestreamNetworkId>{0, 7} ||
        raw_source_uids != std::set<std::int64_t>{20, 22}) {
        throw std::invalid_argument(
            "native-gap network inventory is not the frozen inventory");
    }

    std::set<Eigen::Index> columns;
    std::set<std::pair<pipeline::TimestreamNetworkId,
                       Eigen::Index>> channels;
    for (const auto &detector : fixture.detector_columns) {
        const auto &network = fixture.network(detector.network_id);
        if (!columns.insert(detector.detector_column).second ||
            !channels.emplace(detector.network_id,
                              detector.raw_channel).second ||
            detector.raw_source_uid != network.raw_source_uid ||
            detector.raw_channel < 0 ||
            detector.raw_channel >= network.measured_values.cols()) {
            throw std::invalid_argument(
                "native-gap detector relation is invalid");
        }
    }
    if (columns != std::set<Eigen::Index>{0, 1, 2, 3}) {
        throw std::invalid_argument(
            "native-gap detector columns are not complete");
    }

    const auto alignment = fixture.materialize_alignment();
    std::vector<NativeGapIntervalV1> derived_complete_runs;
    std::optional<std::size_t> open_run;
    for (std::size_t slot = 0; slot < alignment->slot_count(); ++slot) {
        bool complete = true;
        for (const auto network_id : alignment->participant_network_ids()) {
            complete = complete &&
                alignment->association(network_id, slot).mapped();
        }
        if (complete && !open_run.has_value()) open_run = slot;
        if (!complete && open_run.has_value()) {
            derived_complete_runs.push_back({
                static_cast<std::int64_t>(*open_run),
                static_cast<std::int64_t>(slot)});
            open_run.reset();
        }
    }
    if (open_run.has_value()) {
        derived_complete_runs.push_back({
            static_cast<std::int64_t>(*open_run),
            static_cast<std::int64_t>(alignment->slot_count())});
    }
    if (derived_complete_runs !=
        fixture.expected_complete_cohort_slot_runs) {
        throw std::logic_error(
            "native-gap complete cohort topology differs from oracle");
    }

    for (const auto &network : fixture.networks) {
        const auto runs = pipeline::partition_native_contiguous_runs(
            alignment->network(network.network_id),
            network.first_native_row,
            network.first_native_row +
                static_cast<pipeline::TimestreamNativeRow>(
                    network.reconstructed_times_unix_sec.size()));
        if (runs.size() != network.expected_packet_contiguous_runs.size()) {
            throw std::logic_error(
                "native-gap packet run count differs from oracle");
        }
        for (std::size_t index = 0; index < runs.size(); ++index) {
            if (runs[index].first_native_row !=
                    network.expected_packet_contiguous_runs[index].first ||
                runs[index].past_last_native_row !=
                    network.expected_packet_contiguous_runs[index].past_last) {
                throw std::logic_error(
                    "native-gap packet run differs from oracle");
            }
        }
    }
}

inline NativeGapFixtureV1 load_native_gap_fixture_v1(
    const std::filesystem::path &path = native_gap_fixture_v1_path()) {
    const auto source_sha256 = citlali::utils::sha256_file(path);
    if (source_sha256 != native_gap_fixture_v1_sha256) {
        throw std::invalid_argument(
            "native-gap fixture bytes differ from frozen SHA-256");
    }
    const auto root = YAML::LoadFile(path.string());
    require_mapping(root, "native-gap fixture");
    if (root["schema_version"].as<std::string>() !=
            native_gap_fixture_v1_schema ||
        root["fixture_id"].as<std::string>() !=
            native_gap_fixture_v1_id) {
        throw std::invalid_argument(
            "native-gap fixture schema or identity is invalid");
    }

    NativeGapFixtureV1 result;
    result.source_path = path;
    result.source_sha256 = source_sha256;
    result.accepted_plan_commit =
        root["accepted_plan_commit"].as<std::string>();
    require_sequence(root["historical_evidence_commits"],
                     "historical evidence commits");
    for (const auto commit : root["historical_evidence_commits"]) {
        result.historical_evidence_commits.push_back(
            commit.as<std::string>());
    }

    const auto scope = root["scope"];
    require_mapping(scope, "native-gap scope");
    result.scope = pipeline::NativeObservationScope{
        scope["observation"].as<std::int64_t>(),
        scope["subobservation"].as<std::int64_t>(),
        scope["scan"].as<std::int64_t>()};
    const auto scan_chunk = root["scan_chunk"];
    require_mapping(scan_chunk, "native-gap scan/chunk");
    result.scan_index = scan_chunk["scan_index"].as<std::int64_t>();
    result.chunk_index = scan_chunk["chunk_index"].as<std::int64_t>();
    result.realized_dt_sec = root["realized_dt_sec"].as<double>();
    result.common_slot_reference_times_unix_sec = fixture_vector<double>(
        root["common_slot_reference_times_unix_sec"],
        "common-slot reference times");

    require_sequence(root["detector_columns"], "detector columns");
    for (const auto node : root["detector_columns"]) {
        require_mapping(node, "detector column");
        result.detector_columns.push_back({
            node["detector_column"].as<Eigen::Index>(),
            node["network_id"].as<pipeline::TimestreamNetworkId>(),
            node["raw_source_uid"].as<std::int64_t>(),
            node["raw_channel"].as<Eigen::Index>(),
            node["output_uid"].as<std::int64_t>()});
    }

    require_sequence(root["networks"], "networks");
    for (const auto node : root["networks"]) {
        require_mapping(node, "network");
        NativeGapNetworkV1 network;
        network.network_id =
            node["network_id"].as<pipeline::TimestreamNetworkId>();
        network.raw_source_uid =
            node["raw_source_uid"].as<std::int64_t>();
        network.interface = node["interface"].as<std::string>();
        network.first_native_row =
            node["first_native_row"].as<pipeline::TimestreamNativeRow>();
        network.reconstructed_times_unix_sec = fixture_vector<double>(
            node["reconstructed_times_unix_sec"],
            "network reconstructed times");
        require_sequence(node["packet_counters"], "packet counters");
        for (const auto value : node["packet_counters"]) {
            network.packet_counters.push_back(
                value.as<pipeline::TimestreamPacketCounter>());
        }
        network.legacy_presence_mask = fixture_vector<int>(
            node["legacy_presence_mask"], "legacy presence mask");
        require_sequence(node["expected_slot_native_rows"],
                         "expected slot native rows");
        for (const auto value : node["expected_slot_native_rows"]) {
            if (value.IsNull()) {
                network.expected_slot_native_rows.push_back(std::nullopt);
            }
            else {
                network.expected_slot_native_rows.push_back(
                    value.as<pipeline::TimestreamNativeRow>());
            }
        }
        network.measured_values = fixture_matrix<double>(
            node["measured_values"], "measured values");
        network.original_flag_bits =
            fixture_matrix<pipeline::NativeDetectorFlagBits>(
                node["original_flag_bits"], "original flag bits");
        require_sequence(node["expected_packet_contiguous_runs"],
                         "expected packet contiguous runs");
        for (const auto run :
             node["expected_packet_contiguous_runs"]) {
            network.expected_packet_contiguous_runs.push_back(
                fixture_interval(run, "first_native_row",
                                 "past_last_native_row"));
        }
        result.networks.push_back(std::move(network));
    }

    require_sequence(root["expected_complete_cohort_slot_runs"],
                     "expected complete cohort runs");
    for (const auto run : root["expected_complete_cohort_slot_runs"]) {
        result.expected_complete_cohort_slot_runs.push_back(
            fixture_interval(run, "first_common_slot",
                             "past_last_common_slot"));
    }

    require_sequence(root["expected_stage4_factor2_support"],
                     "expected Stage 4 support");
    for (const auto node : root["expected_stage4_factor2_support"]) {
        require_mapping(node, "expected Stage 4 support row");
        NativeGapStage4SupportV1 support;
        support.segment_ordinal = node["segment_ordinal"].as<std::size_t>();
        support.network_id =
            node["network_id"].as<pipeline::TimestreamNetworkId>();
        support.first_common_slot =
            node["first_common_slot"].as<std::size_t>();
        support.past_last_common_slot =
            node["past_last_common_slot"].as<std::size_t>();
        support.selected_anchor_native_row =
            node["selected_anchor_native_row"]
                .as<pipeline::TimestreamNativeRow>();
        require_sequence(node["original_flag_or_by_channel"],
                         "expected flag OR values");
        for (const auto value :
             node["original_flag_or_by_channel"]) {
            support.original_flag_or_by_channel.push_back(
                value.as<pipeline::NativeDetectorFlagBits>());
        }
        result.expected_stage4_factor2_support.push_back(
            std::move(support));
    }

    validate_native_gap_fixture_v1(result);
    return result;
}

}  // namespace citlali::test_support::sci_align
