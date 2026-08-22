#include <gtest/gtest.h>

#include <citlali/core/engine/calib.h>
#include <citlali/core/pipeline/canonical_apt_bundle_v2.h>
#include <citlali/core/pipeline/canonical_apt_detector_relation_v2.h>
#include <citlali/core/utils/sha256.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace apt = citlali::pipeline::canonical_apt_v2;
using citlali::pipeline::AptDetectorRelationRetention;
using citlali::pipeline::CanonicalAptDetectorRelationV2;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
            ("citlali-stage1-apt-relation-" + std::to_string(nonce));
        std::filesystem::create_directories(path);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

std::string digest(char value) {
    return "sha256:" + std::string(64, value);
}

apt::IssuanceContext issuance(std::string occurrence,
                              std::string event) {
    return {std::move(occurrence), std::move(event), "citlali",
            "stage1-fixture-revision", digest('1'),
            "2026-08-22T12:34:56.123Z"};
}

apt::FieldRule baseline_field(std::int64_t uid, std::string name,
                              apt::ValueType datatype,
                              std::string unit) {
    return {uid, std::move(name), datatype, std::move(unit), false,
            "beammap-baseline", std::string("citlali:beammap-fit-v1"),
            apt::FieldOperation::preserve_target, std::nullopt, "reject",
            "nonidentity", "stage1 baseline fixture field"};
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
    result.push_back(baseline_field(5, "flag", apt::ValueType::int64,
                                    "N/A"));
    result.push_back(baseline_field(6, "fg", apt::ValueType::int64,
                                    "N/A"));
    result.push_back(baseline_field(7, "a_fwhm",
                                    apt::ValueType::float64, "arcsec"));
    result.push_back(baseline_field(8, "b_fwhm",
                                    apt::ValueType::float64, "arcsec"));
    result.push_back(baseline_field(9, "angle",
                                    apt::ValueType::float64, "rad"));
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

std::map<std::string, apt::Value> baseline_values(
    std::int64_t channel, bool all_flagged) {
    return {
        {"flag", std::int64_t{all_flagged ? 1 : 0}},
        {"fg", channel},
        {"a_fwhm", 8.0 + static_cast<double>(channel)},
        {"b_fwhm", 9.0 + static_cast<double>(channel)},
        {"angle", 0.1 * static_cast<double>(channel)},
    };
}

void add_kmp_values(std::map<std::string, apt::Value> &fields,
                    std::int64_t channel) {
    const double frequency = 2.0e9 + 1.0e6 * channel;
    fields["kids_fr"] = frequency;
    fields["kids_f_out"] = frequency;
    fields["kids_Qr"] = 15000.0 + 100.0 * channel;
    fields["kids_flag"] = std::int64_t{90 + channel};
}

struct MatchedBundleFixture {
    apt::VerifiedBundle baseline;
    apt::AptTable output;
    apt::RelationTable relation;
    std::vector<apt::SourceRecord> sources;
    std::vector<apt::ExceptionRecord> exceptions;
    apt::PreparedBundle prepared;
    apt::VerifiedBundle verified;
};

MatchedBundleFixture make_matched_fixture(
    std::string raw_content_sha256 = digest('3'),
    std::uint64_t raw_byte_count = 2048,
    bool mixed_dispositions = true,
    bool large_output_uids = false,
    bool all_flagged = false) {
    apt::AptTable baseline_table;
    baseline_table.kind = apt::BundleKind::baseline;
    baseline_table.issuance = issuance(
        "urn:citlali:stage1:baseline:occurrence",
        "urn:citlali:stage1:baseline:event");
    baseline_table.observation = {148670, 0, 0};
    baseline_table.field_rules = baseline_rules();
    for (std::int64_t channel = 0; channel < 3; ++channel) {
        baseline_table.rows.push_back({
            10 + channel, static_cast<std::uint64_t>(channel),
            1.0e9 + 1.0e6 * channel, 0, 0, channel,
            baseline_values(channel, all_flagged)});
    }
    const std::vector<apt::SourceRecord> baseline_sources{
        {3, apt::SourceRole::raw, digest('2'), 1024,
         baseline_table.observation, 0, "toltec0", 3},
    };
    auto baseline_prepared = apt::prepare_baseline_bundle(
        baseline_table, baseline_sources);
    auto baseline = apt::verify_bundle_payload(
        std::move(baseline_prepared.payload));

    MatchedBundleFixture result;
    result.baseline = baseline;
    result.output.kind = apt::BundleKind::matched;
    result.output.issuance = issuance(
        all_flagged
            ? "urn:citlali:stage1:matched:all-flagged:occurrence"
            : "urn:citlali:stage1:matched:occurrence",
        all_flagged
            ? "urn:citlali:stage1:matched:all-flagged:event"
            : "urn:citlali:stage1:matched:event");
    result.output.observation = {148669, 0, 2};
    result.output.field_rules = matched_rules();
    const std::int64_t output_uid_base = large_output_uids
        ? INT64_C(9007199254740993)
        : INT64_C(100);
    for (std::int64_t channel = 0; channel < 3; ++channel) {
        auto fields = baseline_values(channel, all_flagged);
        const bool has_seed = !mixed_dispositions || channel == 0;
        if (!has_seed) {
            for (auto &[name, value] : fields) {
                (void)name;
                value = apt::NullValue{};
            }
        }
        add_kmp_values(fields, channel);
        const double frequency = 2.0e9 + 1.0e6 * channel;
        result.output.rows.push_back({
            output_uid_base + channel,
            static_cast<std::uint64_t>(channel), frequency, 0, 0, channel,
            std::move(fields)});
    }

    result.sources = {
        {20, apt::SourceRole::raw, std::move(raw_content_sha256),
         raw_byte_count, result.output.observation, 0, "toltec0", 3},
        {21, apt::SourceRole::kmp, digest('4'), 4096,
         baseline_table.observation, 0, "toltec0", 3},
    };
    apt::TargetManifest target;
    target.issuance = issuance(
        "urn:citlali:stage1:target:occurrence",
        "urn:citlali:stage1:target:event");
    target.observation = result.output.observation;
    target.sources = result.sources;
    const std::vector<std::uint64_t> source_ranks{2, 0, 1};
    const std::vector<std::uint64_t> application_ranks{1, 2, 0};
    for (std::int64_t channel = 0; channel < 3; ++channel) {
        std::map<std::string, apt::Value> fields;
        add_kmp_values(fields, channel);
        const double frequency = 2.0e9 + 1.0e6 * channel;
        target.rows.push_back({
            30 + channel, 77, 20, 21, channel,
            source_ranks[static_cast<std::size_t>(channel)],
            application_ranks[static_cast<std::size_t>(channel)],
            frequency, 0, 0, channel, std::move(fields)});
    }

    result.relation.issuance = result.output.issuance;
    result.relation.observation = result.output.observation;
    result.relation.target_parent = apt::target_identity(target);
    result.relation.target_issuance = target.issuance;
    result.relation.baseline_parent = baseline.identity;
    result.relation.matcher = {
        "urn:tolapt:stage1:matcher", digest('5'), digest('6'),
        "observation-tone-match-v2", "tolapt"};
    result.relation.network_evidence = {
        {40, 0, apt::NetworkEvidenceStatus::matched_capable,
         0.0, 200000.0, 15000.0},
    };
    for (std::int64_t channel = 0; channel < 3; ++channel) {
        auto disposition = apt::RelationDisposition::matched;
        std::optional<std::int64_t> pair_uid{50 + channel};
        std::optional<apt::ScopedRowReference> selected_seed{
            apt::ScopedRowReference{baseline.identity, 10 + channel}};
        std::optional<double> separation{100.0 + channel};
        std::optional<bool> good{true};
        std::string reason = "selected-good-seed";
        if (mixed_dispositions && channel == 1) {
            disposition = apt::RelationDisposition::unmatched;
            pair_uid.reset();
            selected_seed.reset();
            separation.reset();
            good.reset();
            reason = "outside-gate";
        } else if (mixed_dispositions && channel == 2) {
            disposition = apt::RelationDisposition::ambiguous;
            pair_uid.reset();
            selected_seed.reset();
            separation.reset();
            good.reset();
            reason = "exact-candidate-tie";
        }
        result.relation.rows.push_back({
            60 + channel, output_uid_base + channel,
            {result.relation.target_parent, 30 + channel}, 77, 20, 21,
            channel, source_ranks[static_cast<std::size_t>(channel)],
            application_ranks[static_cast<std::size_t>(channel)],
            static_cast<std::uint64_t>(channel), disposition, pair_uid,
            selected_seed, separation, good, 40, std::move(reason)});
    }

    if (mixed_dispositions) {
        for (std::int64_t seed_uid : {INT64_C(11), INT64_C(12)}) {
            result.exceptions.push_back({
                seed_uid - 11, apt::ExceptionKind::ambiguity_candidate,
                32, std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, std::nullopt,
                apt::ScopedRowReference{baseline.identity, seed_uid},
                1.0, true, "exact-candidate-tie", std::nullopt});
        }
    }

    result.prepared = apt::prepare_matched_bundle(
        result.output, result.relation, result.sources,
        result.exceptions, baseline);
    result.verified = apt::verify_bundle_payload(result.prepared.payload);
    return result;
}

TEST(sci_align_canonical_apt_relation_v2,
     binds_exact_verified_identities_flags_and_distinct_ranks) {
    const auto fixture = make_matched_fixture();
    const auto relation =
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            fixture.verified);

    EXPECT_EQ(relation.bundle_identity(), fixture.verified.identity);
    EXPECT_EQ(relation.observation(), fixture.output.observation);
    EXPECT_EQ(relation.target_parent(), fixture.relation.target_parent);
    EXPECT_EQ(relation.baseline_parent(), fixture.baseline.identity);
    ASSERT_EQ(relation.raw_sources().size(), std::size_t{1});
    EXPECT_EQ(relation.raw_sources().front().source_uid, 20);
    EXPECT_EQ(relation.raw_sources().front().channel_count, 3);
    ASSERT_EQ(relation.bindings().size(), std::size_t{3});

    const auto &matched = relation.binding_for_detector_column(0);
    EXPECT_EQ(matched.output_uid, 100);
    EXPECT_EQ(matched.target.local_uid, 30);
    EXPECT_EQ(matched.source_rank, 2U);
    EXPECT_EQ(matched.application_rank, 1U);
    EXPECT_EQ(matched.presentation_rank, 0U);
    ASSERT_TRUE(matched.flag.has_value());
    EXPECT_EQ(*matched.flag, 0);
    ASSERT_TRUE(matched.selected_seed.has_value());
    EXPECT_EQ(matched.selected_seed->local_uid, 10);

    const auto &unmatched = relation.binding_for_detector_column(1);
    EXPECT_EQ(unmatched.disposition, apt::RelationDisposition::unmatched);
    EXPECT_FALSE(unmatched.flag.has_value());
    EXPECT_FALSE(unmatched.selected_seed.has_value());

    const auto &ambiguous = relation.binding_for_detector_column(2);
    EXPECT_EQ(ambiguous.disposition, apt::RelationDisposition::ambiguous);
    EXPECT_FALSE(ambiguous.flag.has_value());
    EXPECT_FALSE(ambiguous.selected_seed.has_value());
    EXPECT_EQ(relation.relation_identity().schema,
              apt::relation_table_schema_v2);
}

TEST(sci_align_canonical_apt_relation_v2,
     presentation_permutation_preserves_columns_and_large_int64_identity) {
    auto fixture = make_matched_fixture(
        digest('3'), 2048, false, true);
    std::reverse(fixture.verified.apt.rows.begin(),
                 fixture.verified.apt.rows.end());
    std::reverse(fixture.verified.relation->rows.begin(),
                 fixture.verified.relation->rows.end());

    const auto relation =
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            fixture.verified);
    ASSERT_EQ(relation.bindings().size(), std::size_t{3});
    EXPECT_EQ(relation.binding_for_detector_column(0).output_uid,
              INT64_C(9007199254740993));
    EXPECT_EQ(relation.binding_for_detector_column(1).output_uid,
              INT64_C(9007199254740994));
    EXPECT_EQ(relation.binding_for_detector_column(2).output_uid,
              INT64_C(9007199254740995));
}

TEST(sci_align_canonical_apt_relation_v2,
     rejects_missing_or_substituted_flag_and_invalid_joins) {
    const auto fixture = make_matched_fixture();

    auto missing_flag = fixture.verified;
    ASSERT_TRUE(missing_flag.apt.rows.front().fields.contains("kids_flag"));
    missing_flag.apt.rows.front().fields.erase("flag");
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            missing_flag),
        apt::ContractError);

    auto flag2_substitute = fixture.verified;
    flag2_substitute.apt.rows.front().fields.erase("flag");
    flag2_substitute.apt.rows.front().fields["flag2"] = std::int64_t{0};
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            flag2_substitute),
        apt::ContractError);

    auto sample_flag_substitute = fixture.verified;
    sample_flag_substitute.apt.rows.front().fields.erase("flag");
    sample_flag_substitute.apt.rows.front().fields["sample_flag"] =
        std::int64_t{0};
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            sample_flag_substitute),
        apt::ContractError);

    auto matched_null = fixture.verified;
    matched_null.apt.rows.front().fields["flag"] = apt::NullValue{};
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            matched_null),
        apt::ContractError);

    auto unmatched_integer = fixture.verified;
    unmatched_integer.apt.rows.at(1).fields["flag"] = std::int64_t{0};
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            unmatched_integer),
        apt::ContractError);

    auto governed_value_drift = fixture.verified;
    governed_value_drift.apt.rows.front().fields["flag"] =
        std::int64_t{17};
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            governed_value_drift),
        apt::ContractError);

    auto wrong_network = fixture.verified;
    wrong_network.apt.rows.front().network = 1;
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            wrong_network),
        apt::ContractError);

    auto duplicate_output = fixture.verified;
    duplicate_output.relation->rows.at(1).output_uid =
        duplicate_output.relation->rows.front().output_uid;
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            duplicate_output),
        apt::ContractError);

    auto duplicate_relation = fixture.verified;
    duplicate_relation.relation->rows.at(1).relation_uid =
        duplicate_relation.relation->rows.front().relation_uid;
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            duplicate_relation),
        apt::ContractError);

    auto omitted_output = fixture.verified;
    omitted_output.apt.rows.pop_back();
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            omitted_output),
        apt::ContractError);

    auto wrong_channel = fixture.verified;
    wrong_channel.apt.rows.front().channel = 3;
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            wrong_channel),
        apt::ContractError);

    auto incomplete_raw_coverage = fixture.verified;
    incomplete_raw_coverage.sources.front().channel_count = 4;
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            incomplete_raw_coverage),
        apt::ContractError);

    auto stale_bundle_scope = fixture.verified;
    stale_bundle_scope.identity.occurrence =
        "urn:citlali:stage1:stale:occurrence";
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            stale_bundle_scope),
        apt::ContractError);

    auto foreign_target = fixture.verified;
    foreign_target.relation->rows.front().target.artifact.occurrence =
        "urn:citlali:stage1:foreign-target:occurrence";
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            foreign_target),
        apt::ContractError);

    auto stale_seed = fixture.verified;
    stale_seed.relation->rows.front().selected_seed->artifact.occurrence =
        "urn:citlali:stage1:stale-seed:occurrence";
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            stale_seed),
        apt::ContractError);

    auto target_join_drift = fixture.verified;
    target_join_drift.target->rows.front().input_uid = 78;
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            target_join_drift),
        apt::ContractError);

    auto missing_relation_component = fixture.verified;
    std::erase_if(
        missing_relation_component.manifest.components,
        [](const auto &component) { return component.role == "relation"; });
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            missing_relation_component),
        apt::ContractError);

    auto relation_identity_drift = fixture.verified;
    const auto relation_component = std::find_if(
        relation_identity_drift.manifest.components.begin(),
        relation_identity_drift.manifest.components.end(),
        [](const auto &component) { return component.role == "relation"; });
    ASSERT_NE(relation_component,
              relation_identity_drift.manifest.components.end());
    relation_component->semantic_sha256 = digest('9');
    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            relation_identity_drift),
        apt::ContractError);

    EXPECT_THROW(
        citlali::pipeline::admit_canonical_apt_detector_relation_v2(
            fixture.baseline),
        apt::ContractError);
}

std::filesystem::path publish_fixture(
    const apt::PreparedBundle &prepared,
    const std::filesystem::path &directory) {
    std::filesystem::create_directory(directory);
    const auto manifest = directory / apt::root_manifest_name_v2;
    (void)apt::publish_prepared_bundle(manifest, prepared);
    return manifest;
}

TEST(sci_align_canonical_apt_relation_v2,
     calib_publication_is_atomic_and_values_only_discards_relation) {
    TemporaryDirectory temporary;
    const std::string raw_bytes = "stage1-raw-observation-bytes";
    const auto raw_path = temporary.path / "toltec0.nc";
    {
        std::ofstream raw(raw_path, std::ios::binary);
        ASSERT_TRUE(raw.good());
        raw.write(raw_bytes.data(),
                  static_cast<std::streamsize>(raw_bytes.size()));
    }
    const auto raw_digest =
        "sha256:" + citlali::utils::sha256(raw_bytes);
    auto accepted_fixture = make_matched_fixture(
        raw_digest, raw_bytes.size(), false);
    const auto accepted_manifest = publish_fixture(
        accepted_fixture.prepared, temporary.path / "accepted");

    engine::Calib calib;
    std::vector<std::string> raw_files{raw_path.string()};
    std::vector<std::string> interfaces{"toltec0"};
    ASSERT_NO_THROW(calib.get_apt(
        accepted_manifest.string(), raw_files, interfaces));
    ASSERT_TRUE(calib.has_apt_detector_relation_v2());
    const auto accepted_relation = calib.apt_detector_relation_v2_handle();
    ASSERT_TRUE(accepted_relation);
    EXPECT_EQ(accepted_relation->bindings().size(), std::size_t{3});
    EXPECT_EQ(calib.n_dets, 3);
    EXPECT_EQ(calib.apt_filepath, accepted_manifest.string());
    ASSERT_TRUE(calib.apt.contains("kids_flag"));
    EXPECT_DOUBLE_EQ(calib.apt.at("kids_flag")(0), 90.0);
    EXPECT_DOUBLE_EQ(calib.apt.at("flag")(0), 0.0);

    const auto accepted_apt = calib.apt;
    const auto accepted_headers = calib.apt_header_keys;
    const auto accepted_networks = calib.nw_detector_indices;
    const auto accepted_path = calib.apt_filepath;

    const auto tampered_manifest = publish_fixture(
        accepted_fixture.prepared, temporary.path / "tampered");
    const auto relation_component = std::find_if(
        accepted_fixture.verified.manifest.components.begin(),
        accepted_fixture.verified.manifest.components.end(),
        [](const auto &component) { return component.role == "relation"; });
    ASSERT_NE(relation_component,
              accepted_fixture.verified.manifest.components.end());
    const auto tampered_component =
        tampered_manifest.parent_path() /
        relation_component->relative_path;
    std::filesystem::permissions(
        tampered_component, std::filesystem::perms::owner_write,
        std::filesystem::perm_options::add);
    {
        std::ofstream tamper(tampered_component,
                             std::ios::binary | std::ios::app);
        ASSERT_TRUE(tamper.good());
        tamper << "# tamper\n";
    }
    EXPECT_ANY_THROW(calib.get_apt(
        tampered_manifest.string(), raw_files, interfaces));
    EXPECT_EQ(calib.apt_detector_relation_v2_handle().get(),
              accepted_relation.get());
    EXPECT_EQ(calib.apt_filepath, accepted_path);

    const auto wrong_raw_path = temporary.path / "wrong-toltec0.nc";
    {
        std::ofstream wrong(wrong_raw_path, std::ios::binary);
        wrong << "wrong";
    }
    std::vector<std::string> wrong_raw_files{wrong_raw_path.string()};
    EXPECT_ANY_THROW(calib.get_apt(
        accepted_manifest.string(), wrong_raw_files, interfaces));
    EXPECT_EQ(calib.apt_detector_relation_v2_handle().get(),
              accepted_relation.get());
    EXPECT_EQ(calib.apt_filepath, accepted_path);
    EXPECT_EQ(calib.apt_header_keys, accepted_headers);
    EXPECT_EQ(calib.nw_detector_indices, accepted_networks);
    ASSERT_EQ(calib.apt.size(), accepted_apt.size());
    for (const auto &[name, values] : accepted_apt) {
        ASSERT_TRUE(calib.apt.contains(name));
        EXPECT_TRUE(calib.apt.at(name).isApprox(values, 0.0));
    }

    auto all_flagged_fixture = make_matched_fixture(
        raw_digest, raw_bytes.size(), false, false, true);
    const auto all_flagged_manifest = publish_fixture(
        all_flagged_fixture.prepared, temporary.path / "all-flagged");
    EXPECT_ANY_THROW(calib.get_apt(
        all_flagged_manifest.string(), raw_files, interfaces));
    EXPECT_EQ(calib.apt_detector_relation_v2_handle().get(),
              accepted_relation.get());
    EXPECT_EQ(calib.apt_filepath, accepted_path);
    EXPECT_DOUBLE_EQ(calib.apt.at("flag")(0), 0.0);

    ASSERT_NO_THROW(calib.get_apt(
        accepted_manifest.string(), raw_files, interfaces,
        AptDetectorRelationRetention::discard));
    EXPECT_FALSE(calib.has_apt_detector_relation_v2());
    EXPECT_THROW(calib.require_apt_detector_relation_v2(), std::logic_error);
    EXPECT_EQ(calib.n_dets, 3);
    EXPECT_DOUBLE_EQ(calib.apt.at("flag")(0), 0.0);
}

}  // namespace
