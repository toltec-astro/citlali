#include <gtest/gtest.h>

#include <citlali/core/cli/canonical_apt_contract_protocol_v2.h>
#include <citlali/core/pipeline/canonical_apt_bundle_v2.h>

#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace apt = citlali::pipeline::canonical_apt_v2;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
            ("citlali-canonical-apt-v2-" + std::to_string(nonce));
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
            "20feebc26f5ab36f3db04d05835de6ac907fd2e6", digest('1'),
            "2026-08-19T12:34:56.123Z"};
}

std::vector<apt::FieldRule> structural_rules() {
    return apt::canonical_structural_field_rules_v2();
}

apt::VerifiedBundle baseline_fixture() {
    apt::AptTable table;
    table.kind = apt::BundleKind::baseline;
    table.issuance = issuance("urn:citlali:baseline:occurrence",
                              "urn:citlali:baseline:event");
    table.observation = {148670, 0, 0};
    table.field_rules = structural_rules();
    table.field_rules.push_back(
        {5, "a_fwhm", apt::ValueType::float64, "arcsec", true,
         "producer", std::string("citlali:beammap-fit-v1"),
         apt::FieldOperation::preserve_target, std::nullopt, "nan-token",
         "nonidentity", "fitted beam width"});
    table.rows = {
        {5, 0, 1.0e9, 0, 0, 0, {{"a_fwhm", 8.5}}},
        {9, 1, 1.1e9, 0, 0, 1, {{"a_fwhm", 9.5}}},
    };
    std::vector<apt::SourceRecord> sources{
        {3, apt::SourceRole::raw, digest('2'), 1024, {148670, 0, 0},
         0, "toltec0", 2},
    };
    auto prepared = apt::prepare_baseline_bundle(table, sources);
    return apt::verify_bundle_payload(std::move(prepared.payload));
}

struct MatchedFixture {
    apt::AptTable apt;
    apt::RelationTable relation;
    std::vector<apt::SourceRecord> sources;
};

MatchedFixture matched_fixture(const apt::VerifiedBundle &baseline) {
    MatchedFixture result;
    result.apt.kind = apt::BundleKind::matched;
    result.apt.issuance = issuance("urn:citlali:matched:occurrence",
                                   "urn:citlali:matched:event");
    result.apt.observation = {148669, 0, 2};
    result.apt.field_rules = structural_rules();
    result.apt.field_rules.push_back(
        {5, "a_fwhm", apt::ValueType::float64, "arcsec", true,
         "producer", std::string("citlali:beammap-fit-v1"),
         apt::FieldOperation::copy_seed_or_null, std::nullopt, "nan-token",
         "nonidentity", "fitted beam width"});
    auto kmp = apt::canonical_kmp_field_rules_v2(false);
    for (std::size_t index = 0; index < kmp.size(); ++index) {
        kmp[index].field_uid = static_cast<std::int64_t>(6 + index);
        result.apt.field_rules.push_back(kmp[index]);
    }
    result.apt.rows = {
        {100, 0, 2.0e9, 0, 0, 0,
         {{"a_fwhm", 8.5}, {"kids_Qr", 15000.0},
          {"kids_f_out", 2.0e9}, {"kids_fr", 2.0e9}}},
        {101, 1, 2.1e9, 0, 0, 1,
         {{"a_fwhm", apt::NullValue{}}, {"kids_Qr", 16000.0},
          {"kids_f_out", 2.1e9}, {"kids_fr", 2.1e9}}},
    };
    result.sources = {
        {20, apt::SourceRole::raw, digest('3'), 2048, {148669, 0, 2},
         0, "toltec0", 2},
        {21, apt::SourceRole::kmp, digest('4'), 4096, {148670, 0, 0},
         0, "toltec0", 2},
    };
    apt::TargetManifest target;
    target.issuance = issuance("urn:citlali:target:occurrence",
                               "urn:citlali:target:event");
    target.observation = result.apt.observation;
    target.sources = result.sources;
    target.rows = {
        {10, 77, 20, 21, 0, 1, 0, 2.0e9, 0, 0, 0,
         {{"kids_Qr", 15000.0}, {"kids_f_out", 2.0e9},
          {"kids_fr", 2.0e9}}},
        {11, 77, 20, 21, 1, 0, 1, 2.1e9, 0, 0, 1,
         {{"kids_Qr", 16000.0}, {"kids_f_out", 2.1e9},
          {"kids_fr", 2.1e9}}},
    };
    result.relation.issuance = result.apt.issuance;
    result.relation.observation = result.apt.observation;
    result.relation.target_parent = apt::target_identity(target);
    result.relation.target_issuance = target.issuance;
    result.relation.baseline_parent = baseline.identity;
    result.relation.matcher = {
        "urn:tolapt:matcher:run", digest('5'), digest('6'),
        "observation-tone-match-v2", "tolapt"};
    result.relation.network_evidence = {
        {30, 0, apt::NetworkEvidenceStatus::matched_capable,
         0.0, 200000.0, 15000.0},
    };
    result.relation.rows = {
        {40, 100, {result.relation.target_parent, 10}, 77, 20, 21, 0,
         1, 0, 0, apt::RelationDisposition::matched, 50,
         apt::ScopedRowReference{baseline.identity, 5}, 0.0, true, 30,
         "selected-good-seed"},
        {41, 101, {result.relation.target_parent, 11}, 77, 20, 21, 1,
         0, 1, 1, apt::RelationDisposition::unmatched, std::nullopt,
         std::nullopt, std::nullopt, std::nullopt, 30,
         "outside-gate"},
    };
    return result;
}

TEST(canonical_apt_v2, private_lexical_vector_matches_python) {
    apt::detail::FlatComponent document;
    document.schema = "citlali-canonical-apt-v2-lexical-vector-v1";
    document.role = "vector";
    document.kind = apt::BundleKind::matched;
    document.issuance = {
        "urn:citlali:occurrence:vector", "urn:citlali:event:vector",
        "citlali", "fixture-revision", digest('1'),
        "2026-08-19T12:34:56.123Z"};
    document.observation = {148669, 0, 2};
    document.metadata = {{"alpha", "α"}, {"empty", ""}};
    using Type = apt::detail::FlatValueType;
    document.columns = {
        {"i", Type::int64, "N/A", false},
        {"u", Type::uint64, "byte", false},
        {"f", Type::float64, "Hz", false},
        {"b", Type::boolean, "N/A", false},
        {"s", Type::string, "N/A", false},
        {"n", Type::float64, "N/A", true},
    };
    document.rows = {
        {std::numeric_limits<std::int64_t>::min(),
         apt::detail::UInt64Value{
             std::numeric_limits<std::uint64_t>::max()},
         -0.0, true, std::string("café"), apt::NullValue{}},
        {std::numeric_limits<std::int64_t>::max(),
         apt::detail::UInt64Value{0},
         std::numeric_limits<double>::denorm_min(), false,
         std::string("alpha"), std::numeric_limits<double>::quiet_NaN()},
    };
    const auto serialized = apt::detail::serialize_component(document);
    EXPECT_EQ(serialized.digests.semantic_sha256,
              "sha256:439f6582412e428f98db3094d1e4dd6dcc0e7afea15727171299b40afae48db4");
    EXPECT_EQ(serialized.digests.envelope_sha256,
              "sha256:2c53758ea4674368420f2643a8c9c46a2e969beea1d5e1f52ae6878e37206876");
    EXPECT_EQ(serialized.digests.transport_sha256,
              "sha256:60dccb8e8c859fce779dcddfd46a9c022e7fb3a76bba5bde67a946336bcce8e8");
    EXPECT_EQ(serialized.digests.byte_count, std::uint64_t{1729});
}

TEST(canonical_apt_v2, target_identity_matches_python) {
    const auto same_digest = digest('1');
    apt::TargetManifest target;
    target.issuance = {"o", "e", "p", "s", same_digest,
                       "2026-08-19T00:00:00.000Z"};
    target.observation = {1, 0, 0};
    target.sources = {
        {0, apt::SourceRole::raw, same_digest, 1, {1, 0, 0}, 0,
         "toltec0", 1},
        {1, apt::SourceRole::kmp, same_digest, 1, {2, 0, 1}, 0,
         "toltec0", 1},
    };
    target.rows = {
        {0, 0, 0, 1, 0, 0, 0, 2.0, 0, 0, 0,
         {{"kids_Qr", 3.0}, {"kids_f_out", 2.0}, {"kids_fr", 1.0}}},
    };
    EXPECT_EQ(
        apt::target_identity(target),
        (apt::ComponentIdentity{
            "citlali-observation-target-manifest-v2", "o",
            "sha256:c031af9cad8683860f023e2bfb3de1fb601cdac3ad29c86e95f7717def27df56",
            "sha256:4445ecf32b33652ad50d926ebcdc8d301daf87d216746150edb303cd7c416023"}));
}

TEST(canonical_apt_v2, baseline_and_matched_bundle_roundtrip_compactly) {
    auto baseline = baseline_fixture();
    EXPECT_EQ(baseline.parser_count, 5);
    EXPECT_EQ(baseline.identity.semantic_sha256,
              "sha256:e31c4d41203a00c11e0769c0a4d7ac64be6f1ac5d595b7f39765f553de2a12dc");
    EXPECT_EQ(baseline.identity.envelope_sha256,
              "sha256:5a1f493ed89c1f5edf79a3d069218f0c1d925130740c54ba158ddd48ee691a74");
    EXPECT_EQ(baseline.manifest_digests.transport_sha256,
              "sha256:5effe44315bb7565fe5e321aa533d82bab0c9b490e4315d5b6d79d757c8c612e");
    EXPECT_EQ(baseline.total_byte_count, std::uint64_t{10184});
    auto matched = matched_fixture(baseline);
    auto prepared = apt::prepare_matched_bundle(
        matched.apt, matched.relation, matched.sources, {}, baseline);
    EXPECT_EQ(prepared.identity.semantic_sha256,
              "sha256:a315c101983189c3e0c930bd5eb8e179775be01f7f0c33526cc4d654b5853086");
    EXPECT_EQ(prepared.identity.envelope_sha256,
              "sha256:3b49d38b24b414525b9f118e481552b9fee60d4eee6cf8f93696e04b12cc9f87");
    EXPECT_EQ(prepared.total_byte_count, std::uint64_t{34194});
    auto verified = apt::verify_bundle_payload(std::move(prepared.payload));
    EXPECT_EQ(verified.parser_count, 12);
    EXPECT_EQ(verified.apt.rows.size(), std::size_t{2});
    ASSERT_TRUE(verified.target);
    EXPECT_EQ(verified.target->rows.size(), std::size_t{2});
    EXPECT_EQ(verified.seed_dispositions.size(), std::size_t{2});
    EXPECT_LT(verified.total_byte_count,
              apt::maximum_portable_bundle_bytes_v2);
}

TEST(canonical_apt_v2,
     publication_is_no_replace_receipt_last_relocatable_and_tamper_evident) {
    const auto baseline = baseline_fixture();
    const auto prepared = apt::prepare_baseline_bundle(
        baseline.apt, baseline.sources);
    TemporaryDirectory temporary;
    const auto first = temporary.path / "first";
    ASSERT_TRUE(std::filesystem::create_directory(first));
    const auto manifest = first / apt::root_manifest_name_v2;
    const auto published = apt::publish_prepared_bundle(manifest, prepared);
    EXPECT_EQ(published.receipt_path,
              first / apt::root_receipt_name_v2);
    const auto verified = apt::verify_bundle_filesystem(manifest, true);
    EXPECT_EQ(verified.identity, prepared.identity);

    const auto relocated = temporary.path / "relocated";
    std::filesystem::rename(first, relocated);
    const auto relocated_manifest = relocated / apt::root_manifest_name_v2;
    EXPECT_EQ(apt::verify_bundle_filesystem(relocated_manifest, true).identity,
              prepared.identity);
    EXPECT_THROW(apt::publish_prepared_bundle(relocated_manifest, prepared),
                 std::runtime_error);

    const auto apt_descriptor = std::find_if(
        verified.manifest.components.begin(), verified.manifest.components.end(),
        [](const auto &item) { return item.role == "apt"; });
    ASSERT_NE(apt_descriptor, verified.manifest.components.end());
    const auto tampered_path = relocated / apt_descriptor->relative_path;
    std::filesystem::permissions(
        tampered_path, std::filesystem::perms::owner_write,
        std::filesystem::perm_options::add);
    std::ofstream tamper(tampered_path,
                         std::ios::binary | std::ios::app);
    ASSERT_TRUE(tamper.good());
    tamper << "x";
    tamper.close();
    EXPECT_THROW(apt::verify_bundle_filesystem(relocated_manifest, true),
                 apt::ContractError);

    const auto failed = temporary.path / "failed";
    ASSERT_TRUE(std::filesystem::create_directory(failed));
    citlali::pipeline::canonical_artifact_publication::BundlePublicationHooks
        hooks;
    hooks.on_stage = [](auto stage, const auto &, const auto &) {
        using Stage = citlali::pipeline::canonical_artifact_publication::
            BundlePublicationStage;
        if (stage == Stage::before_receipt_publish) {
            throw std::runtime_error("injected before completion marker");
        }
    };
    EXPECT_THROW(apt::publish_prepared_bundle(
                     failed / apt::root_manifest_name_v2, prepared, hooks),
                 std::runtime_error);
    EXPECT_TRUE(std::filesystem::is_empty(failed));

    const auto staged_tamper = temporary.path / "staged-tamper";
    ASSERT_TRUE(std::filesystem::create_directory(staged_tamper));
    hooks.on_stage = [](auto stage, const auto &staging, const auto &) {
        using Stage = citlali::pipeline::canonical_artifact_publication::
            BundlePublicationStage;
        if (stage != Stage::members_staged) return;
        for (const auto &entry : std::filesystem::directory_iterator(staging)) {
            if (entry.path().filename().string().starts_with(".")) continue;
            std::ofstream stream(entry.path(), std::ios::binary | std::ios::app);
            stream << "tamper";
            return;
        }
    };
    EXPECT_THROW(apt::publish_prepared_bundle(
                     staged_tamper / apt::root_manifest_name_v2,
                     prepared, hooks),
                 std::runtime_error);
    EXPECT_TRUE(std::filesystem::is_empty(staged_tamper));

    const auto raced = temporary.path / "raced";
    ASSERT_TRUE(std::filesystem::create_directory(raced));
    const auto raced_member =
        raced / baseline.manifest.components.front().relative_path;
    hooks.on_stage = [&raced_member](auto stage, const auto &, const auto &) {
        using Stage = citlali::pipeline::canonical_artifact_publication::
            BundlePublicationStage;
        if (stage != Stage::before_members_publish) return;
        std::ofstream stream(raced_member, std::ios::binary);
        stream << "foreign-race-winner";
    };
    EXPECT_THROW(apt::publish_prepared_bundle(
                     raced / apt::root_manifest_name_v2, prepared, hooks),
                 std::runtime_error);
    EXPECT_EQ(std::filesystem::file_size(raced_member),
              std::uintmax_t{19});
    EXPECT_FALSE(std::filesystem::exists(
        raced / apt::root_receipt_name_v2));

    const auto member_failure = temporary.path / "member-failure";
    ASSERT_TRUE(std::filesystem::create_directory(member_failure));
    hooks.on_stage = [](auto stage, const auto &, const auto &) {
        using Stage = citlali::pipeline::canonical_artifact_publication::
            BundlePublicationStage;
        if (stage == Stage::member_published) {
            throw std::runtime_error("injected after member publication");
        }
    };
    EXPECT_THROW(apt::publish_prepared_bundle(
                     member_failure / apt::root_manifest_name_v2,
                     prepared, hooks),
                 std::runtime_error);
    EXPECT_TRUE(std::filesystem::is_empty(member_failure));
}

TEST(canonical_apt_v2,
     exact_ambiguity_candidates_are_compact_and_incomplete_coverage_rejects) {
    const auto baseline = baseline_fixture();
    auto fixture = matched_fixture(baseline);
    auto &record = fixture.relation.rows[1];
    record.disposition = apt::RelationDisposition::ambiguous;
    record.reason = "exact-candidate-tie";
    std::vector<apt::ExceptionRecord> exceptions{
        {0, apt::ExceptionKind::ambiguity_candidate, record.target.local_uid,
         std::nullopt, std::nullopt, std::nullopt, std::nullopt,
         std::nullopt, apt::ScopedRowReference{baseline.identity, 5}, 1.0,
         true, "exact-candidate-tie", std::nullopt},
        {1, apt::ExceptionKind::ambiguity_candidate, record.target.local_uid,
         std::nullopt, std::nullopt, std::nullopt, std::nullopt,
         std::nullopt, apt::ScopedRowReference{baseline.identity, 9}, 1.0,
         true, "exact-candidate-tie", std::nullopt},
    };
    EXPECT_NO_THROW(apt::prepare_matched_bundle(
        fixture.apt, fixture.relation, fixture.sources, exceptions, baseline));
    exceptions.pop_back();
    EXPECT_THROW(apt::prepare_matched_bundle(
                     fixture.apt, fixture.relation, fixture.sources,
                     exceptions, baseline),
                 apt::ContractError);
}

TEST(canonical_apt_v2,
     public_protocol_validates_fresh_bundle_and_rejects_bad_framing) {
    const auto baseline = baseline_fixture();
    const auto prepared = apt::prepare_baseline_bundle(
        baseline.apt, baseline.sources);
    TemporaryDirectory temporary;
    const auto bundle = temporary.path / "bundle";
    ASSERT_TRUE(std::filesystem::create_directory(bundle));
    const auto manifest = bundle / apt::root_manifest_name_v2;
    (void)apt::publish_prepared_bundle(manifest, prepared);

    namespace protocol =
        citlali::cli::canonical_apt_contract_protocol_v2;
    const auto request =
        std::string("{\"protocol\":\"") +
        std::string(protocol::protocol_v2) +
        "\",\"request_id\":\"test\",\"operation\":\"" +
        std::string(protocol::validate_bundle_operation_v2) +
        "\",\"payload\":{\"root_manifest\":\"" +
        manifest.string() + "\"}}";
    const auto success = protocol::process_request_line(
        request, protocol::production_dependencies());
    EXPECT_EQ(success.exit_code, protocol::success_exit_code);
    EXPECT_NE(success.response_json.find(prepared.identity.semantic_sha256),
              std::string::npos);
    EXPECT_NE(success.response_json.find("\"product_kind\":\"beammap-baseline\""),
              std::string::npos);

    auto duplicate = request;
    const auto payload_position = duplicate.find("\"payload\"");
    duplicate.insert(payload_position,
                     "\"request_id\":\"duplicate\",");
    EXPECT_EQ(protocol::process_request_line(
                  duplicate, protocol::production_dependencies()).exit_code,
              protocol::protocol_error_exit_code);

    const auto disabled =
        std::string("{\"protocol\":\"") +
        std::string(protocol::protocol_v2) +
        "\",\"request_id\":\"test\",\"operation\":\"" +
        std::string(protocol::issue_observation_apt_operation_v2) +
        "\",\"payload\":{}}";
    EXPECT_EQ(protocol::process_request_line(
                  disabled, protocol::production_dependencies()).exit_code,
              protocol::contract_rejection_exit_code);
}

}  // namespace
