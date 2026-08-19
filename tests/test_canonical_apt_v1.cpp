#include <gtest/gtest.h>

#include <citlali/core/engine/calib.h>
#include <citlali/core/engine/detail/beammap_apt_table_output_helpers.h>
#include <citlali/core/pipeline/canonical_apt_ecsv.h>
#include <citlali/core/pipeline/canonical_apt_observation_v1.h>
#include <citlali/core/pipeline/canonical_apt_v1.h>
#include <citlali/core/pipeline/rawobs_detector_inventory.h>
#include <citlali/core/utils/sha256.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace apt = citlali::pipeline::canonical_apt_v1;
namespace observation_apt =
    citlali::pipeline::canonical_apt_observation_v1;
namespace artifact_publication =
    citlali::pipeline::canonical_artifact_publication;
namespace apt_producer = beammap_apt_table_output_helpers;

apt::RegisteredField extension_spec(std::string_view name) {
    for (const auto &field : apt::optional_extension_fields_v1()) {
        if (field.name == name) {
            return field;
        }
    }
    throw std::logic_error("unknown focused-test extension");
}

apt::Value fixture_value(const apt::RegisteredField &field,
                         std::size_t row_index) {
    if (field.type == apt::ValueType::int64) {
        if (field.name == "flag") {
            return static_cast<std::int64_t>(row_index == 1 ? 1 : 0);
        }
        if (field.name == "flag2") {
            return static_cast<std::int64_t>(1U << row_index);
        }
        if (field.name == "cal_amp_method" ||
            field.name == "scan_band_mask_rejected") {
            return static_cast<std::int64_t>(row_index % 2);
        }
        if (field.name == "scan_band_masked_edge") {
            return static_cast<std::int64_t>(row_index);
        }
        if (field.name == "final_prior_slot_index") {
            return static_cast<std::int64_t>(row_index) - 1;
        }
        return static_cast<std::int64_t>(row_index + 1);
    }
    if (field.type == apt::ValueType::float64) {
        return 0.25 + static_cast<double>(row_index);
    }
    if (field.type == apt::ValueType::boolean) {
        return row_index % 2 == 0;
    }
    return std::string{"row-"} + std::to_string(row_index);
}

void populate_registered_values(apt::Document &document) {
    for (std::size_t row_index = 0; row_index < document.rows.size();
         ++row_index) {
        for (const auto &field : document.registered_fields) {
            document.rows[row_index].fields[field.name] =
                fixture_value(field, row_index);
        }
    }
}

apt::Document make_document(
    std::string occurrence = "occurrence:test/opaque#001") {
    apt::Document document;
    document.envelope = {
        std::move(occurrence),
        "event:test/beammap-baseline#001",
        std::string(apt::baseline_output_role_v1),
        "citlali",
        "46ad23888a40f5102cdfd50c06e49a549bdf8a20",
        "config:test/exact-v1",
        "2026-08-13T12:34:56Z",
    };
    document.context = {
        "2025-C1-COM-01",
        "Jupiter α",
        "2026-08-13T11:22:33.125Z",
        "altaz",
    };
    document.raw_manifest.observation = {152389, 0, 1};
    // Deliberately noncanonical raw-input and row presentation order. Channel
    // zero and the same tone frequency occur in two different networks.
    document.raw_manifest.inputs = {
        {7, "toltec7", 1},
        {0, "toltec0", 2},
    };
    document.registered_fields = apt::required_baseline_fields_v1();
    for (const auto name : {"flag2", "final_prior_slot_index",
                            "final_prior_d2", "cal_amp_method",
                            "template_npix"}) {
        document.registered_fields.push_back(extension_spec(name));
    }
    document.rows = {
        {apt::uid_v1_max, 2.5e9, 1, 7, 0, {}},
        {42, 1.25e9, 0, 0, 1, {}},
        {0, 2.5e9, 0, 0, 0, {}},
    };
    populate_registered_values(document);
    return document;
}

apt::FieldRegistry typed_test_registry() {
    auto registry = apt::canonical_field_registry_v1();
    registry.version = "citlali-phase-a-test-field-registry-v1";
    const auto add = [&](std::string name, apt::ValueType type,
                         bool nullable, apt::NonFinitePolicy nonfinite) {
        registry.optional_extensions.push_back(apt::registered_field_spec(
            std::move(name), type, "N/A", nullable,
            apt::FieldAuthority::copied_declared,
            "focused-test:independent-declaration-v1", nonfinite,
            registry.version, "focused typed ECSV extension"));
    };
    add("diagnostic_code", apt::ValueType::int64, true,
        apt::NonFinitePolicy::reject);
    add("diagnostic_value", apt::ValueType::float64, true,
        apt::NonFinitePolicy::canonical_token);
    add("note", apt::ValueType::string, true,
        apt::NonFinitePolicy::reject);
    add("selected", apt::ValueType::boolean, true,
        apt::NonFinitePolicy::reject);
    return registry;
}

apt::Document make_typed_document(const apt::FieldRegistry &registry) {
    auto document = make_document();
    document.field_registry = registry.version;
    for (const auto &field : registry.optional_extensions) {
        if (field.registry == registry.version) {
            document.registered_fields.push_back(field);
        }
    }
    document.rows[0].fields["diagnostic_code"] =
        std::numeric_limits<std::int64_t>::min();
    document.rows[1].fields["diagnostic_code"] =
        std::numeric_limits<std::int64_t>::max();
    document.rows[2].fields["diagnostic_code"] = apt::NullValue{};
    document.rows[0].fields["diagnostic_value"] =
        -std::numeric_limits<double>::infinity();
    document.rows[1].fields["diagnostic_value"] =
        std::numeric_limits<double>::quiet_NaN();
    document.rows[2].fields["diagnostic_value"] = -0.0;
    document.rows[0].fields["note"] =
        std::string{"first, café \"detector\""};
    document.rows[1].fields["note"] = apt::NullValue{};
    document.rows[2].fields["note"] = std::string{"third detector"};
    document.rows[0].fields["selected"] = true;
    document.rows[1].fields["selected"] = false;
    document.rows[2].fields["selected"] = apt::NullValue{};
    return document;
}

void replace_once(std::string &value, const std::string &from,
                  const std::string &to) {
    const auto offset = value.find(from);
    ASSERT_NE(offset, std::string::npos) << from;
    value.replace(offset, from.size(), to);
}

void mutate_declared_digest(std::string &bytes, std::string_view key) {
    const std::string marker =
        "#     " + std::string(key) + ": \"sha256:";
    const auto offset = bytes.find(marker);
    ASSERT_NE(offset, std::string::npos);
    const auto digit = offset + marker.size();
    bytes[digit] = bytes[digit] == '0' ? '1' : '0';
}

std::string catalog_signature(std::vector<apt::RegisteredField> fields) {
    std::sort(fields.begin(), fields.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.name < rhs.name;
              });
    std::string result;
    for (const auto &field : fields) {
        result += field.name + "|" +
            std::string(apt::value_type_token(field.type)) + "|" +
            field.unit + "|" + (field.nullable ? "nullable" : "required") +
            "|" + std::string(apt::field_authority_token(field.authority)) +
            "|" + field.authority_reference + "|" +
            std::string(apt::nonfinite_policy_token(field.nonfinite)) + "|" +
            field.registry + "\n";
    }
    return result;
}

class HostileGrouping final : public std::numpunct<char> {
protected:
    char do_thousands_sep() const override { return '_'; }
    std::string do_grouping() const override { return "\1"; }
};

std::string sha256_reference(char digit) {
    return "sha256:" + std::string(64, digit);
}

std::string target_catalog_signature(
    const std::vector<observation_apt::TypedField> &fields) {
    std::string result;
    for (const auto &field : fields) {
        result += field.name + "|" +
            std::string(apt::value_type_token(field.type)) + "|" +
            field.unit + "|" +
            (field.nullable ? "nullable" : "required") + "|" +
            field.authority + "|" + field.authority_reference + "|" +
            std::string(apt::nonfinite_policy_token(field.nonfinite)) + "|" +
            field.registry + "|" + field.description + "|" +
            (field.source_column ? *field.source_column : "<none>") + "|" +
            field.identity_role + "\n";
    }
    return result;
}

observation_apt::SourceArtifact make_observation_source(
    std::int64_t source_key, std::string role, char digest_digit,
    apt::ObservationIdentity header, std::int64_t network,
    std::int64_t channel_count) {
    return {source_key,
            std::move(role),
            "fixture/café/toltec" + std::to_string(network),
            sha256_reference(digest_digit),
            static_cast<std::uint64_t>(1000 + source_key),
            header,
            network,
            "toltec" + std::to_string(network),
            channel_count};
}

struct ObservationContractFixture {
    observation_apt::VerifiedBaselineDescriptor baseline;
    observation_apt::TargetManifest target;
    observation_apt::MatchRelation relation;
    observation_apt::MatchedOutput output;
};

ObservationContractFixture make_observation_contract_fixture(
    bool include_target_kids_flag = true) {
    auto baseline_document = make_document();
    baseline_document.registered_fields.push_back(extension_spec("kids_flag"));
    baseline_document.rows[0].fields["kids_flag"] = std::int64_t{91};
    baseline_document.rows[1].fields["kids_flag"] = std::int64_t{-5};
    baseline_document.rows[2].fields["kids_flag"] = std::int64_t{77};
    const auto baseline_serialized = apt::serialize_ecsv(baseline_document);
    const auto receipt = observation_apt::canonical_baseline_receipt_bytes(
        baseline_serialized.transport);
    auto descriptor = observation_apt::verify_baseline_descriptor(
        baseline_serialized.bytes, receipt);

    observation_apt::TargetManifest target;
    target.envelope = {
        "occurrence:tolproj/target#opaque-A",
        "event:tolproj/observation-target#A",
        "tolproj-clean-fixture-revision",
        "tolproj-config:sha256:fixture",
        "2026-08-14T01:02:03Z",
    };
    target.observation = {152390, 0, 1};
    target.inputs = {
        {70,
         7,
         "toltec7",
         1,
         make_observation_source(700, "raw", '7', target.observation, 7, 1),
         make_observation_source(701, "kmp", '8', {152300, 0, 0}, 7,
                                 1)},
        {10,
         0,
         "toltec0",
         2,
         make_observation_source(100, "raw", '1', target.observation, 0, 2),
         make_observation_source(101, "kmp", '2', {152300, 0, 0}, 0,
                                 2)},
    };
    target.registered_fields = observation_apt::canonical_target_fields_v1();
    if (!include_target_kids_flag) {
        target.registered_fields.pop_back();
    }
    target.rows = {
        {701, 70, 701, 0, 2.5e9, 2.5001e9, 1, 7, 0, {}},
        {11, 10, 101, 1, -0.0, -0.0, 0, 0, 1, {}},
        {5,
         10,
         101,
         0,
         std::numeric_limits<double>::denorm_min(),
         1.2501e9,
         0,
         0,
         0,
         {}},
    };
    for (auto &row : target.rows) {
        row.fields["kids_fr"] = row.matching_frequency_hz;
        row.fields["kids_f_out"] = row.output_tone_frequency_hz;
        if (row.row_key == 701) {
            row.fields["kids_Qr"] = 42000.0;
            if (include_target_kids_flag) {
                row.fields["kids_flag"] = std::int64_t{-7};
            }
        } else if (row.row_key == 11) {
            // No positivity policy is silently added for imported Qr.
            row.fields["kids_Qr"] = -3.0;
            if (include_target_kids_flag) {
                row.fields["kids_flag"] = std::int64_t{3};
            }
        } else {
            row.fields["kids_Qr"] =
                std::numeric_limits<double>::denorm_min();
            if (include_target_kids_flag) {
                row.fields["kids_flag"] = std::int64_t{42};
            }
        }
    }
    target.target_source_sequence = {11, 701, 5};
    target.target_application_sequence = {5, 11, 701};
    observation_apt::validate(target);

    const auto target_identity = observation_apt::artifact_identity(target);
    const auto seed_identity = observation_apt::artifact_identity(descriptor);
    observation_apt::MatchRelation relation;
    relation.envelope = {
        "occurrence:tolproj/relation#opaque-B",
        "event:tolproj/matcher-run#B",
        "tolproj-clean-fixture-revision",
        "tolproj-match-config:sha256:fixture",
        "2026-08-14T01:03:04Z",
    };
    relation.baseline_parent = observation_apt::baseline_reference(descriptor);
    relation.target_parent = target_identity;
    relation.matcher = {
        "occurrence:tolproj/matcher-policy-run#opaque",
        "tolproj-clean-fixture-revision",
        "tolproj-match-config:sha256:fixture",
        "astropy",
        "join-distance-v1",
    };
    relation.network_evidence = {
        {7, -0.0, 200000.0, 42000.0},
        {0, std::numeric_limits<double>::denorm_min(), 200000.0, 20000.0},
    };
    const auto target_ref = [&](std::int64_t key) {
        return observation_apt::row_reference(target_identity, key);
    };
    const auto seed_ref = [&](std::int64_t key) {
        return observation_apt::row_reference(seed_identity, key);
    };
    // Deliberately exercise 1:many and many:1 without making either a
    // persistent detector identity or a Citlali matcher policy.
    relation.pairs = {
        {902, target_ref(11), seed_ref(42), 3.0, true},
        {900, target_ref(5), seed_ref(0), -0.0, true},
        {901, target_ref(5), seed_ref(42),
         std::numeric_limits<double>::denorm_min(), false},
    };
    relation.target_dispositions = {
        {1000, target_ref(701),
         observation_apt::EndpointDispositionState::unmatched, {},
         "no selected seed endpoint"},
        {1002, target_ref(5),
         observation_apt::EndpointDispositionState::matched, {900, 901},
         "two realized candidate endpoints retained"},
        {1001, target_ref(11),
         observation_apt::EndpointDispositionState::matched, {902},
         "one realized endpoint"},
    };
    relation.seed_dispositions = {
        {2000, seed_ref(apt::uid_v1_max),
         observation_apt::EndpointDispositionState::unused, {},
         "seed not used"},
        {2002, seed_ref(0),
         observation_apt::EndpointDispositionState::matched, {900},
         "one target endpoint"},
        {2001, seed_ref(42),
         observation_apt::EndpointDispositionState::matched, {901, 902},
         "two target endpoints"},
    };
    relation.seed_source_sequence = {42, apt::uid_v1_max, 0};
    observation_apt::validate(relation, descriptor, target);

    observation_apt::MatchedOutput output;
    output.envelope = {
        "occurrence:tolproj/matched-output#opaque-C",
        "event:tolproj/observation-output#C",
        "tolproj-clean-fixture-revision",
        "tolproj-output-config:sha256:fixture",
        "2026-08-14T01:04:05Z",
    };
    output.baseline_parent = observation_apt::baseline_reference(descriptor);
    output.target_parent = target_identity;
    output.relation_parent =
        observation_apt::artifact_identity(relation, descriptor, target);
    output.registered_fields =
        observation_apt::canonical_output_field_contracts_v1(descriptor,
                                                              target);

    std::map<std::int64_t, const apt::Row *> baseline_rows;
    for (const auto &row : descriptor.document().rows) {
        baseline_rows.emplace(row.uid, &row);
    }
    const auto make_output_row = [&](std::int64_t uid,
                                     const observation_apt::TargetRow &source,
                                     std::vector<std::int64_t> pair_keys,
                                     std::optional<std::int64_t> source_pair,
                                     std::optional<std::int64_t> seed_uid) {
        observation_apt::MatchedOutputRow row;
        row.uid = uid;
        row.target = target_ref(source.row_key);
        row.target_input_key = source.input_key;
        row.tone_frequency_hz = source.output_tone_frequency_hz;
        row.array = source.array;
        row.network = source.network;
        row.channel = source.channel;
        row.relation_pair_keys = std::move(pair_keys);
        for (const auto &contract : output.registered_fields) {
            observation_apt::FieldTransformation transformation;
            transformation.field_name = contract.field.name;
            transformation.operation = contract.authorized_operation;
            if (contract.authorized_operation ==
                observation_apt::TransformationOperation::preserve_target) {
                const auto value = source.fields.at(contract.field.name);
                row.fields.emplace(contract.field.name, value);
                transformation.before = value;
                transformation.after = value;
                transformation.value_source = observation_apt::
                    TransformationValueSource::target_row;
                transformation.source_row = target_ref(source.row_key);
                transformation.authority_reference =
                    contract.field.authority_reference;
                transformation.provenance_reference =
                    "target-kmp-source:" +
                    std::to_string(source.kmp_source_key) + ":row:" +
                    std::to_string(source.kmp_row_index) + ":column:" +
                    *contract.field.source_column;
            } else if (seed_uid) {
                transformation.before = apt::NullValue{};
                const auto &seed = *baseline_rows.at(*seed_uid);
                const auto value = seed.fields.at(contract.field.name);
                row.fields.emplace(contract.field.name, value);
                transformation.after = value;
                transformation.value_source = observation_apt::
                    TransformationValueSource::baseline_seed_row;
                transformation.source_pair_key = source_pair;
                transformation.source_row = seed_ref(*seed_uid);
                const auto source_contract = std::find_if(
                    descriptor.document().registered_fields.begin(),
                    descriptor.document().registered_fields.end(),
                    [&](const auto &field) {
                        return field.name == contract.field.name;
                    });
                transformation.authority_reference =
                    source_contract->authority_reference;
                transformation.provenance_reference =
                    "relation-pair:" + std::to_string(*source_pair);
            } else {
                transformation.before = apt::NullValue{};
                row.fields.emplace(contract.field.name, apt::NullValue{});
                transformation.after = apt::NullValue{};
                transformation.value_source = observation_apt::
                    TransformationValueSource::canonical_null;
                transformation.authority_reference =
                    observation_apt::unmatched_missing_authority_v1;
                transformation.provenance_reference =
                    "target-unmatched:no-fabricated-seed";
            }
            row.transformations.push_back(std::move(transformation));
        }
        return row;
    };
    const auto target_by_key = [&](std::int64_t key) -> const auto & {
        return *std::find_if(target.rows.begin(), target.rows.end(),
                             [&](const auto &row) {
                                 return row.row_key == key;
                             });
    };
    output.rows = {
        make_output_row(888, target_by_key(701), {}, std::nullopt,
                        std::nullopt),
        make_output_row(4, target_by_key(5), {900, 901}, 900, 0),
        make_output_row(apt::uid_v1_max - 1, target_by_key(11), {902}, 902,
                        42),
    };
    output.output_presentation_sequence = {apt::uid_v1_max - 1, 888, 4};
    observation_apt::validate(output, descriptor, target, relation);
    return {std::move(descriptor), std::move(target), std::move(relation),
            std::move(output)};
}

TEST(canonical_apt_observation_v1,
     exact_successor_schema_and_authority_tokens_are_pinned) {
    EXPECT_EQ(observation_apt::framing_encoding_v1,
              apt::framing_encoding_v1);
    EXPECT_EQ(observation_apt::contract_authority_v1, "citlali");
    EXPECT_EQ(observation_apt::observation_value_issuer_v1, "tolproj");
    EXPECT_EQ(observation_apt::baseline_descriptor_schema_v1,
              "citlali-verified-beammap-baseline-descriptor-v1");
    EXPECT_EQ(observation_apt::target_manifest_schema_v1,
              "citlali-observation-target-manifest-v1");
    EXPECT_EQ(observation_apt::relation_schema_v1,
              "citlali-apt-match-dispositions-v1");
    EXPECT_EQ(observation_apt::matched_output_schema_v1,
              "citlali-observation-matched-apt-v1");
    EXPECT_EQ(observation_apt::target_artifact_contract_id_v1,
              "apt-prod-002-observation-target-manifest-v1");
    EXPECT_EQ(observation_apt::relation_artifact_contract_id_v1,
              "apt-prod-002-match-dispositions-v1");
    EXPECT_EQ(observation_apt::matched_output_artifact_contract_id_v1,
              "apt-prod-002-observation-matched-apt-v1");
    EXPECT_EQ(observation_apt::kmp_source_field_map_profile_v1,
              "citlali-kmp-source-field-map-v1");
    EXPECT_EQ(observation_apt::canonical_kmp_field_bindings_v1(),
              (std::vector<observation_apt::KmpFieldBinding>{
                  {"fr", "kids_fr", true},
                  {"f_out", "kids_f_out", true},
                  {"Qr", "kids_Qr", true},
                  {"flag", "kids_flag", false}}));
    ASSERT_EQ(observation_apt::canonical_target_fields_v1().size(), 4U);
    EXPECT_EQ(observation_apt::canonical_target_fields_v1()[0].name,
              "kids_fr");
    EXPECT_EQ(observation_apt::canonical_target_fields_v1()[1].name,
              "kids_f_out");
    EXPECT_EQ(observation_apt::canonical_target_fields_v1()[2].name,
              "kids_Qr");
    EXPECT_EQ(observation_apt::canonical_target_fields_v1()[3].name,
              "kids_flag");
    EXPECT_EQ(
        target_catalog_signature(
            observation_apt::canonical_target_fields_v1()),
        R"(kids_fr|float64|Hz|required|copied-declared|kids:model-params-v1|reject|citlali-observation-target-fields-v1|imported KIDs resonant frequency; finite, nonidentity|fr|nonidentity
kids_f_out|float64|Hz|required|copied-declared|kids:model-params-v1|reject|citlali-observation-target-fields-v1|imported KIDs output tone frequency; finite, nonidentity|f_out|nonidentity
kids_Qr|float64|N/A|required|copied-declared|kids:model-params-v1|reject|citlali-observation-target-fields-v1|imported KIDs resonator Qr; finite with no positivity rule, nonidentity|Qr|nonidentity
kids_flag|int64|N/A|required|copied-declared|kids:fit-report-v1|reject|citlali-observation-target-fields-v1|imported KIDs model-fit flag; exact signed integral values, nonidentity|flag|nonidentity
)"
    );
    static_assert(!std::is_default_constructible_v<
                  observation_apt::VerifiedBaselineDescriptor>);
    static_assert(!std::is_aggregate_v<
                  observation_apt::VerifiedBaselineDescriptor>);
}

TEST(canonical_apt_observation_v1,
     kmp_source_boundary_ignores_unrequested_diagnostics_and_closes_uses) {
    const std::vector<std::string> columns{
        "fr", "f_out", "Qr", "flag", "kids_fr", "kids_flag",
        "diagnostic_snr", "fit_residual_private"};
    const std::vector<observation_apt::KmpFieldUseRequest> requests{
        {"kids_fr", observation_apt::KmpFieldUseRole::matching,
         "kids:model-params-v1"},
        {"kids_fr", observation_apt::KmpFieldUseRole::output,
         "kids:model-params-v1"},
        {"kids_fr", observation_apt::KmpFieldUseRole::authority,
         "kids:model-params-v1"},
        {"kids_f_out", observation_apt::KmpFieldUseRole::application,
         "kids:model-params-v1"},
        {"kids_f_out", observation_apt::KmpFieldUseRole::output,
         "kids:model-params-v1"},
        {"kids_Qr", observation_apt::KmpFieldUseRole::matching,
         "kids:model-params-v1"},
        {"kids_Qr", observation_apt::KmpFieldUseRole::output,
         "kids:model-params-v1"},
        {"kids_flag", observation_apt::KmpFieldUseRole::output,
         "kids:fit-report-v1"},
        {"kids_flag", observation_apt::KmpFieldUseRole::authority,
         "kids:fit-report-v1"},
    };
    const auto selected = observation_apt::select_canonical_kmp_fields_v1(
        columns, requests);
    EXPECT_EQ(selected, observation_apt::canonical_target_fields_v1());

    auto more_diagnostics = columns;
    more_diagnostics.push_back("unrequested_source_note");
    EXPECT_EQ(observation_apt::select_canonical_kmp_fields_v1(
                  more_diagnostics, requests),
              selected);

    const std::vector<std::string> required_only{
        "fr", "f_out", "Qr", "kids_flag"};
    const std::vector<observation_apt::KmpFieldUseRequest>
        required_requests(requests.begin(), requests.begin() + 7);
    EXPECT_EQ(observation_apt::select_canonical_kmp_fields_v1(
                  required_only, required_requests),
              observation_apt::canonical_required_target_fields_v1());
    EXPECT_THROW(observation_apt::select_canonical_kmp_fields_v1(
                     required_only,
                     {{"kids_flag", observation_apt::KmpFieldUseRole::output,
                       "kids:fit-report-v1"}}),
                 apt::ContractError);

    const auto expect_rejected = [&](std::string field,
                                     observation_apt::KmpFieldUseRole role,
                                     std::string authority) {
        EXPECT_THROW(observation_apt::select_canonical_kmp_fields_v1(
                         columns, {{std::move(field), role,
                                    std::move(authority)}}),
                     apt::ContractError);
    };
    expect_rejected("diagnostic_snr",
                    observation_apt::KmpFieldUseRole::output,
                    "kids:model-params-v1");
    expect_rejected("kids_fr", observation_apt::KmpFieldUseRole::identity,
                    "kids:model-params-v1");
    expect_rejected("kids_f_out",
                    observation_apt::KmpFieldUseRole::matching,
                    "kids:model-params-v1");
    expect_rejected("kids_fr",
                    observation_apt::KmpFieldUseRole::transformation,
                    "kids:model-params-v1");
    expect_rejected("kids_Qr", observation_apt::KmpFieldUseRole::matching,
                    "caller:self-authorized");
}

TEST(canonical_apt_observation_v1,
     exact_successor_scalar_and_utf8_frames_are_pinned) {
    std::string frames;
    frames += observation_apt::canonical_int64_frame(
        "i64-min", std::numeric_limits<std::int64_t>::min());
    frames += observation_apt::canonical_uint64_frame(
        "u64-max", std::numeric_limits<std::uint64_t>::max());
    frames += observation_apt::canonical_binary64_frame("negative-zero",
                                                         -0.0);
    frames += observation_apt::canonical_binary64_frame(
        "denorm-min", std::numeric_limits<double>::denorm_min());
    frames += observation_apt::canonical_binary64_frame(
        "canonical-nan", std::bit_cast<double>(0xfff8000000000042ULL));
    frames += observation_apt::canonical_null_frame(
        "missing-float", apt::ValueType::float64);
    frames += apt::canonical_frame("text", "utf8", "Jupiter α");
    EXPECT_EQ(
        frames,
        "F7:i64-minT5:int64V20:-9223372036854775808;"
        "F7:u64-maxT6:uint64V20:18446744073709551615;"
        "F13:negative-zeroT15:float64-ieee754V16:8000000000000000;"
        "F10:denorm-minT15:float64-ieee754V16:0000000000000001;"
        "F13:canonical-nanT15:float64-ieee754V16:7ff8000000000000;"
        "F13:missing-floatT12:null-float64V4:null;"
        "F4:textT4:utf8V10:Jupiter α;");
    EXPECT_EQ(citlali::utils::sha256(frames),
              "a97e7c29a17da562d44108968d120c393428577f9f218154bc5147e8f32029ec");
}

TEST(canonical_apt_observation_v1,
     baseline_descriptor_is_factory_verified_and_preserves_baseline_identity) {
    const auto fixture = make_observation_contract_fixture();
    const auto identity = observation_apt::artifact_identity(fixture.baseline);
    EXPECT_EQ(identity.schema, apt::schema_version_v1);
    EXPECT_EQ(identity.occurrence,
              fixture.baseline.document().envelope.occurrence);
    EXPECT_EQ(identity.semantic_sha256,
              fixture.baseline.digests().semantic_sha256);
    EXPECT_EQ(fixture.relation.baseline_parent.artifact, identity);
    EXPECT_EQ(fixture.relation.baseline_parent.transport_sha256,
              fixture.baseline.transport().sha256);
    EXPECT_EQ(fixture.relation.baseline_parent.receipt_sha256,
              fixture.baseline.receipt_sha256());

    auto bad_bytes = std::string(fixture.baseline.baseline_bytes());
    bad_bytes[bad_bytes.find("Jupiter")] = 'X';
    EXPECT_THROW(observation_apt::verify_baseline_descriptor(
                     bad_bytes, fixture.baseline.receipt_bytes()),
                 apt::ContractError);
    auto bad_receipt = std::string(fixture.baseline.receipt_bytes());
    const auto digest = bad_receipt.find("byte_sha256=sha256:");
    ASSERT_NE(digest, std::string::npos);
    bad_receipt[digest + std::string("byte_sha256=sha256:").size()] = 'f';
    EXPECT_THROW(observation_apt::verify_baseline_descriptor(
                     fixture.baseline.baseline_bytes(), bad_receipt),
                 apt::ContractError);

    auto changed_document = fixture.baseline.document();
    changed_document.context.source_name = "Saturn β";
    const auto changed = apt::serialize_ecsv(changed_document);
    const auto changed_descriptor =
        observation_apt::verify_baseline_descriptor(
            changed.bytes,
            observation_apt::canonical_baseline_receipt_bytes(
                changed.transport));
    EXPECT_NE(observation_apt::baseline_descriptor_sha256(fixture.baseline),
              observation_apt::baseline_descriptor_sha256(
                  changed_descriptor));

    auto changed_raw = fixture.baseline.document();
    changed_raw.rows[1].channel = 0;
    changed_raw.rows[2].channel = 1;
    const auto changed_raw_bytes = apt::serialize_ecsv(changed_raw);
    const auto changed_raw_descriptor =
        observation_apt::verify_baseline_descriptor(
            changed_raw_bytes.bytes,
            observation_apt::canonical_baseline_receipt_bytes(
                changed_raw_bytes.transport));
    EXPECT_NE(observation_apt::baseline_descriptor_sha256(fixture.baseline),
              observation_apt::baseline_descriptor_sha256(
                  changed_raw_descriptor));

    auto changed_value = fixture.baseline.document();
    changed_value.rows[0].fields["amp"] = 99.0;
    const auto changed_value_bytes = apt::serialize_ecsv(changed_value);
    const auto changed_value_descriptor =
        observation_apt::verify_baseline_descriptor(
            changed_value_bytes.bytes,
            observation_apt::canonical_baseline_receipt_bytes(
                changed_value_bytes.transport));
    EXPECT_NE(observation_apt::baseline_descriptor_sha256(fixture.baseline),
              observation_apt::baseline_descriptor_sha256(
                  changed_value_descriptor));
}

TEST(canonical_apt_observation_v1,
     target_manifest_closes_sources_relation_and_explicit_orders) {
    const auto fixture = make_observation_contract_fixture();
    EXPECT_NO_THROW(observation_apt::validate(fixture.target));
    EXPECT_FALSE(fixture.target.inputs[0].kmp_source.header_observation ==
                 fixture.target.observation);
    EXPECT_EQ(std::get<double>(fixture.target.rows[1].fields.at("kids_Qr")),
              -3.0);

    auto reordered = fixture.target;
    std::reverse(reordered.inputs.begin(), reordered.inputs.end());
    std::reverse(reordered.rows.begin(), reordered.rows.end());
    std::reverse(reordered.registered_fields.begin(),
                 reordered.registered_fields.end());
    EXPECT_EQ(observation_apt::target_semantic_sha256(reordered),
              observation_apt::target_semantic_sha256(fixture.target));

    auto presentation_only = fixture.target;
    presentation_only.inputs[0].raw_source.diagnostic_locator =
        "different/nonidentity/locator";
    EXPECT_EQ(observation_apt::target_semantic_sha256(presentation_only),
              observation_apt::target_semantic_sha256(fixture.target));
    EXPECT_NE(observation_apt::target_envelope_sha256(presentation_only),
              observation_apt::target_envelope_sha256(fixture.target));
    EXPECT_THROW(observation_apt::validate(
                     fixture.relation, fixture.baseline, presentation_only),
                 apt::ContractError);

    auto changed_source_bytes = fixture.target;
    changed_source_bytes.inputs[0].kmp_source.content_sha256 =
        sha256_reference('e');
    EXPECT_NE(observation_apt::target_semantic_sha256(changed_source_bytes),
              observation_apt::target_semantic_sha256(fixture.target));

    auto changed_sequence = fixture.target;
    std::swap(changed_sequence.target_application_sequence[0],
              changed_sequence.target_application_sequence[1]);
    EXPECT_NE(observation_apt::target_semantic_sha256(changed_sequence),
              observation_apt::target_semantic_sha256(fixture.target));

    auto other_occurrence = fixture.target;
    other_occurrence.envelope.occurrence = "occurrence:target/other";
    EXPECT_EQ(observation_apt::target_semantic_sha256(other_occurrence),
              observation_apt::target_semantic_sha256(fixture.target));
    EXPECT_NE(observation_apt::target_envelope_sha256(other_occurrence),
              observation_apt::target_envelope_sha256(fixture.target));

    auto conflicting_raw = fixture.target;
    conflicting_raw.inputs[0].raw_source.network = 0;
    EXPECT_THROW(observation_apt::validate(conflicting_raw),
                 apt::ContractError);
    auto conflicting_kmp = fixture.target;
    conflicting_kmp.inputs[1].kmp_source.channel_count = 1;
    EXPECT_THROW(observation_apt::validate(conflicting_kmp),
                 apt::ContractError);
    auto duplicate_source = fixture.target;
    duplicate_source.inputs[1].kmp_source.source_key =
        duplicate_source.inputs[1].raw_source.source_key;
    EXPECT_THROW(observation_apt::validate(duplicate_source),
                 apt::ContractError);
    auto wrong_kmp_source = fixture.target;
    wrong_kmp_source.rows[0].kmp_source_key = 101;
    EXPECT_THROW(observation_apt::validate(wrong_kmp_source),
                 apt::ContractError);
    auto wrong_kmp_row = fixture.target;
    wrong_kmp_row.rows[1].kmp_row_index = 0;
    EXPECT_THROW(observation_apt::validate(wrong_kmp_row),
                 apt::ContractError);
    auto mismatched_fr_alias = fixture.target;
    mismatched_fr_alias.rows[0].matching_frequency_hz = 2.6e9;
    EXPECT_THROW(observation_apt::validate(mismatched_fr_alias),
                 apt::ContractError);
    auto mismatched_f_out_alias = fixture.target;
    mismatched_f_out_alias.rows[0].output_tone_frequency_hz = 2.6e9;
    EXPECT_THROW(observation_apt::validate(mismatched_f_out_alias),
                 apt::ContractError);
    auto incomplete_sequence = fixture.target;
    incomplete_sequence.target_source_sequence.pop_back();
    EXPECT_THROW(observation_apt::validate(incomplete_sequence),
                 apt::ContractError);
    auto rogue_field = fixture.target;
    rogue_field.registered_fields.push_back(
        {"invented_truth", apt::ValueType::int64, "N/A", false,
         apt::NonFinitePolicy::reject, "tolproj", "caller:self-authorized",
         std::string(observation_apt::target_field_registry_v1),
         "not in the Citlali registry"});
    EXPECT_THROW(observation_apt::validate(rogue_field), apt::ContractError);
    auto wrong_authority = fixture.target;
    wrong_authority.registered_fields[0].authority_reference =
        "caller:self-authorized";
    EXPECT_THROW(observation_apt::validate(wrong_authority),
                 apt::ContractError);
    auto swapped_source_column = fixture.target;
    swapped_source_column.registered_fields[0].source_column = "f_out";
    EXPECT_THROW(observation_apt::validate(swapped_source_column),
                 apt::ContractError);
    auto invented_identity_role = fixture.target;
    invented_identity_role.registered_fields[0].identity_role = "detector-id";
    EXPECT_THROW(observation_apt::validate(invented_identity_role),
                 apt::ContractError);
    auto missing_required = fixture.target;
    missing_required.registered_fields.erase(
        missing_required.registered_fields.begin() + 2);
    for (auto &row : missing_required.rows) {
        row.fields.erase("kids_Qr");
    }
    EXPECT_THROW(observation_apt::validate(missing_required),
                 apt::ContractError);
    auto null_flag = fixture.target;
    null_flag.rows[0].fields["kids_flag"] = apt::NullValue{};
    EXPECT_THROW(observation_apt::validate(null_flag), apt::ContractError);
    auto nonintegral_flag = fixture.target;
    nonintegral_flag.rows[0].fields["kids_flag"] = 3.0;
    EXPECT_THROW(observation_apt::validate(nonintegral_flag),
                 apt::ContractError);
    auto nonfinite_qr = fixture.target;
    nonfinite_qr.rows[0].fields["kids_Qr"] =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(observation_apt::validate(nonfinite_qr),
                 apt::ContractError);
}

TEST(canonical_apt_observation_v1,
     kids_flag_is_optional_per_artifact_and_nonbinary_when_present) {
    const auto with_flag = make_observation_contract_fixture(true);
    const auto without_flag = make_observation_contract_fixture(false);
    EXPECT_NO_THROW(observation_apt::validate(with_flag.target));
    EXPECT_NO_THROW(observation_apt::validate(without_flag.target));
    EXPECT_EQ(std::get<std::int64_t>(
                  with_flag.target.rows[0].fields.at("kids_flag")),
              -7);
    EXPECT_FALSE(without_flag.target.rows[0].fields.contains("kids_flag"));
    EXPECT_TRUE(std::none_of(
        without_flag.output.registered_fields.begin(),
        without_flag.output.registered_fields.end(), [](const auto &field) {
            return field.field.name == "kids_flag";
        }));
    EXPECT_NO_THROW(observation_apt::validate(
        without_flag.output, without_flag.baseline, without_flag.target,
        without_flag.relation));
}

TEST(canonical_apt_observation_v1,
     relation_has_complete_reciprocal_set_cardinality_and_local_keys) {
    const auto fixture = make_observation_contract_fixture();
    EXPECT_NO_THROW(observation_apt::validate(
        fixture.relation, fixture.baseline, fixture.target));
    EXPECT_EQ(fixture.relation.target_dispositions[1].pair_keys.size(), 2U);
    EXPECT_EQ(fixture.relation.seed_dispositions[2].pair_keys.size(), 2U);
    EXPECT_TRUE(fixture.relation.target_dispositions[0].pair_keys.empty());
    EXPECT_TRUE(fixture.relation.seed_dispositions[0].pair_keys.empty());
    EXPECT_EQ(fixture.relation.network_evidence[0].quality_factor, 42000.0);
    EXPECT_EQ(fixture.relation.network_evidence[0].quality_factor_field,
              "kids_Qr");
    EXPECT_EQ(fixture.relation.network_evidence[0]
                  .quality_factor_authority_reference,
              "kids:model-params-v1");

    auto reordered = fixture.relation;
    std::reverse(reordered.pairs.begin(), reordered.pairs.end());
    std::reverse(reordered.target_dispositions.begin(),
                 reordered.target_dispositions.end());
    std::reverse(reordered.seed_dispositions.begin(),
                 reordered.seed_dispositions.end());
    std::reverse(reordered.network_evidence.begin(),
                 reordered.network_evidence.end());
    EXPECT_EQ(observation_apt::relation_semantic_sha256(
                  reordered, fixture.baseline, fixture.target),
              observation_apt::relation_semantic_sha256(
                  fixture.relation, fixture.baseline, fixture.target));

    auto missing_target = fixture.relation;
    missing_target.target_dispositions.pop_back();
    EXPECT_THROW(observation_apt::validate(
                     missing_target, fixture.baseline, fixture.target),
                 apt::ContractError);
    auto fabricated_endpoint = fixture.relation;
    fabricated_endpoint.target_dispositions[0].state =
        observation_apt::EndpointDispositionState::matched;
    fabricated_endpoint.target_dispositions[0].pair_keys = {900};
    EXPECT_THROW(observation_apt::validate(
                     fabricated_endpoint, fixture.baseline, fixture.target),
                 apt::ContractError);
    auto row_key_collision = fixture.relation;
    row_key_collision.target_dispositions[0].disposition_key = 900;
    EXPECT_THROW(observation_apt::validate(
                     row_key_collision, fixture.baseline, fixture.target),
                 apt::ContractError);
    auto bad_seed_sequence = fixture.relation;
    bad_seed_sequence.seed_source_sequence[0] = 0;
    EXPECT_THROW(observation_apt::validate(
                     bad_seed_sequence, fixture.baseline, fixture.target),
                 apt::ContractError);
    auto reused_occurrence = fixture.relation;
    reused_occurrence.envelope.occurrence =
        fixture.target.envelope.occurrence;
    EXPECT_THROW(observation_apt::validate(
                     reused_occurrence, fixture.baseline, fixture.target),
                 apt::ContractError);
    auto unknown_mapping_domain = fixture.relation;
    unknown_mapping_domain.mapping_domain = "caller:other-matcher-domain";
    EXPECT_THROW(observation_apt::validate(
                     unknown_mapping_domain, fixture.baseline, fixture.target),
                 apt::ContractError);
    auto unknown_matching_field = fixture.relation;
    unknown_matching_field.matcher.target_frequency_field = "kids_f_out";
    EXPECT_THROW(observation_apt::validate(
                     unknown_matching_field, fixture.baseline, fixture.target),
                 apt::ContractError);
    auto wrong_quality_authority = fixture.relation;
    wrong_quality_authority.network_evidence[0]
        .quality_factor_authority_reference = "caller:self-authorized";
    EXPECT_THROW(observation_apt::validate(
                     wrong_quality_authority, fixture.baseline,
                     fixture.target),
                 apt::ContractError);
    auto nonfinite_quality = fixture.relation;
    nonfinite_quality.network_evidence[0].quality_factor =
        std::numeric_limits<double>::infinity();
    EXPECT_THROW(observation_apt::validate(
                     nonfinite_quality, fixture.baseline, fixture.target),
                 apt::ContractError);
}

TEST(canonical_apt_observation_v1,
     matched_output_registry_is_closed_and_unmatched_is_typed_missing) {
    const auto fixture = make_observation_contract_fixture();
    EXPECT_NO_THROW(observation_apt::validate(
        fixture.output, fixture.baseline, fixture.target, fixture.relation));
    ASSERT_EQ(fixture.output.registered_fields.size(),
              fixture.baseline.document().registered_fields.size() - 1U +
                  fixture.target.registered_fields.size());
    for (const auto &contract : fixture.output.registered_fields) {
        const bool target_field = std::any_of(
            fixture.target.registered_fields.begin(),
            fixture.target.registered_fields.end(), [&](const auto &field) {
                return field.name == contract.field.name;
            });
        EXPECT_EQ(contract.field.nullable, !target_field);
        EXPECT_EQ(
            contract.authorized_operation,
            target_field
                ? observation_apt::TransformationOperation::preserve_target
                : observation_apt::TransformationOperation::
                      copy_baseline_when_matched_null_when_unmatched);
        EXPECT_EQ(contract.field.source_column.has_value(), target_field);
        EXPECT_TRUE(contract.issuer_authority_reference.empty());
    }
    const auto unmatched = std::find_if(
        fixture.output.rows.begin(), fixture.output.rows.end(),
        [](const auto &row) { return row.relation_pair_keys.empty(); });
    ASSERT_NE(unmatched, fixture.output.rows.end());
    for (const auto &[name, value] : unmatched->fields) {
        const bool target_field =
            name == "kids_fr" || name == "kids_f_out" ||
            name == "kids_Qr" || name == "kids_flag";
        EXPECT_EQ(std::holds_alternative<apt::NullValue>(value),
                  !target_field)
            << name;
    }
    for (const auto &change : unmatched->transformations) {
        const bool target_field =
            change.field_name == "kids_fr" ||
            change.field_name == "kids_f_out" ||
            change.field_name == "kids_Qr" ||
            change.field_name == "kids_flag";
        EXPECT_EQ(
            change.value_source,
            target_field
                ? observation_apt::TransformationValueSource::target_row
                : observation_apt::TransformationValueSource::canonical_null);
        EXPECT_FALSE(change.source_pair_key);
        EXPECT_EQ(change.source_row.has_value(), target_field);
    }
    EXPECT_EQ(std::get<std::int64_t>(unmatched->fields.at("kids_flag")), -7);

    const auto matched = std::find_if(
        fixture.output.rows.begin(), fixture.output.rows.end(),
        [](const auto &row) { return row.target.local_key == 5; });
    ASSERT_NE(matched, fixture.output.rows.end());
    EXPECT_EQ(std::get<std::int64_t>(matched->fields.at("kids_flag")), 42);
    const auto baseline_seed = std::find_if(
        fixture.baseline.document().rows.begin(),
        fixture.baseline.document().rows.end(),
        [](const auto &row) { return row.uid == 0; });
    ASSERT_NE(baseline_seed, fixture.baseline.document().rows.end());
    EXPECT_EQ(std::get<std::int64_t>(baseline_seed->fields.at("kids_flag")),
              77);

    auto rogue_contract = fixture.output;
    auto rogue = rogue_contract.registered_fields.front();
    rogue.field.name = "caller_field";
    rogue_contract.registered_fields.push_back(std::move(rogue));
    EXPECT_THROW(observation_apt::validate(
                     rogue_contract, fixture.baseline, fixture.target,
                     fixture.relation),
                 apt::ContractError);
    auto self_authorized = fixture.output;
    self_authorized.registered_fields.front().authorized_operation =
        observation_apt::TransformationOperation::issuer_declared;
    self_authorized.registered_fields.front().issuer_authority_reference =
        "caller:self-authorized";
    EXPECT_THROW(observation_apt::validate(
                     self_authorized, fixture.baseline, fixture.target,
                     fixture.relation),
                 apt::ContractError);
    auto changed_structure = fixture.output;
    changed_structure.rows[0].network = 0;
    EXPECT_THROW(observation_apt::validate(
                     changed_structure, fixture.baseline, fixture.target,
                     fixture.relation),
                 apt::ContractError);
    auto wrong_source = fixture.output;
    wrong_source.rows[1].transformations[0].source_pair_key = 901;
    EXPECT_THROW(observation_apt::validate(
                     wrong_source, fixture.baseline, fixture.target,
                     fixture.relation),
                 apt::ContractError);
    auto seed_flag_collision = fixture.output;
    auto &collision_row = *std::find_if(
        seed_flag_collision.rows.begin(), seed_flag_collision.rows.end(),
        [](const auto &row) { return row.target.local_key == 5; });
    collision_row.fields["kids_flag"] = std::int64_t{77};
    auto &collision_change = *std::find_if(
        collision_row.transformations.begin(),
        collision_row.transformations.end(), [](const auto &change) {
            return change.field_name == "kids_flag";
        });
    collision_change.after = std::int64_t{77};
    EXPECT_THROW(observation_apt::validate(
                     seed_flag_collision, fixture.baseline, fixture.target,
                     fixture.relation),
                 apt::ContractError);
    auto wrong_target_provenance = fixture.output;
    auto &target_change = *std::find_if(
        wrong_target_provenance.rows[0].transformations.begin(),
        wrong_target_provenance.rows[0].transformations.end(),
        [](const auto &change) { return change.field_name == "kids_Qr"; });
    target_change.provenance_reference = "caller:unbound-target-value";
    EXPECT_THROW(observation_apt::validate(
                     wrong_target_provenance, fixture.baseline,
                     fixture.target, fixture.relation),
                 apt::ContractError);
    auto reused_occurrence = fixture.output;
    reused_occurrence.envelope.occurrence =
        fixture.relation.envelope.occurrence;
    EXPECT_THROW(observation_apt::validate(
                     reused_occurrence, fixture.baseline, fixture.target,
                     fixture.relation),
                 apt::ContractError);
}

TEST(canonical_apt_observation_v1,
     complete_successor_semantic_and_envelope_vectors_are_fixed) {
    const auto fixture = make_observation_contract_fixture();
    const auto target = observation_apt::compute_digests(fixture.target);
    const auto relation = observation_apt::compute_digests(
        fixture.relation, fixture.baseline, fixture.target);
    const auto output = observation_apt::compute_digests(
        fixture.output, fixture.baseline, fixture.target, fixture.relation);
    const auto baseline =
        observation_apt::baseline_reference(fixture.baseline);
    EXPECT_EQ(baseline.profile, "citlali-beammap-baseline-apt-v1");
    EXPECT_EQ(baseline.artifact.semantic_sha256,
              "sha256:8ac14aca51f660b015e6427483e05968d1443a33d812a28ef46ed027261f0a37");
    EXPECT_EQ(baseline.artifact.envelope_sha256,
              "sha256:f44e40ae8604b85ea82f783212eb785561fbbd6b478ab1311f406bc63a1d2838");
    EXPECT_EQ(baseline.transport_sha256,
              "sha256:b4cfecf45c611ba6378bd7b88d78978b8004aa0ee8db499367499c75db05f34b");
    EXPECT_EQ(baseline.byte_count, 19327U);
    EXPECT_EQ(baseline.receipt_sha256,
              "sha256:536f689f3325e5a1d298a69bba277ef686971c97152034a1bb1bc861d1acbe30");
    EXPECT_EQ(baseline.receipt_byte_count, 287U);
    EXPECT_EQ(observation_apt::baseline_descriptor_sha256(fixture.baseline),
              "sha256:b801161d65dfea02b3c579ac5766154900b82e92de6945537caae2691d2707af");
    EXPECT_EQ(target.semantic_sha256,
              "sha256:8ad86d382b31eed82deab3118bbd5efe1fc5ce41389eac561ad2aef7e24cb30b");
    EXPECT_EQ(target.envelope_sha256,
              "sha256:3dca742ac86f93666762e33557ab91b4d061be178b936d17d379077233bd6fc5");
    EXPECT_EQ(relation.semantic_sha256,
              "sha256:7555c3f35ef57db23d32ef833d635c06cd06690ca1543cb635272328f29c93a4");
    EXPECT_EQ(relation.envelope_sha256,
              "sha256:25cd94197f41b3ec6132adfde525eeb1baae9b2dbeae7d47af38966817f5e8dc");
    EXPECT_EQ(output.semantic_sha256,
              "sha256:cac3fabbb34907013b7558c5db855c3c861e370bb05ff0ff15051dd9f4e44dba");
    EXPECT_EQ(output.envelope_sha256,
              "sha256:96fe37adc1b743dbcd7d907bb0f63b4859ff44102cc1be8556914f7978212dce");

    auto reordered = fixture.output;
    std::reverse(reordered.rows.begin(), reordered.rows.end());
    std::reverse(reordered.registered_fields.begin(),
                 reordered.registered_fields.end());
    for (auto &row : reordered.rows) {
        std::reverse(row.transformations.begin(), row.transformations.end());
    }
    EXPECT_EQ(observation_apt::matched_output_semantic_sha256(
                  reordered, fixture.baseline, fixture.target,
                  fixture.relation),
              output.semantic_sha256);
    auto other_occurrence = fixture.output;
    other_occurrence.envelope.occurrence = "occurrence:output/other";
    EXPECT_EQ(observation_apt::matched_output_semantic_sha256(
                  other_occurrence, fixture.baseline, fixture.target,
                  fixture.relation),
              output.semantic_sha256);
    EXPECT_NE(observation_apt::matched_output_envelope_sha256(
                  other_occurrence, fixture.baseline, fixture.target,
                  fixture.relation),
              output.envelope_sha256);
}

TEST(canonical_apt_observation_v1,
     matched_observation_ecsv_embeds_and_revalidates_complete_logical_bundle) {
    const auto fixture = make_observation_contract_fixture();
    const auto serialized =
        observation_apt::serialize_matched_observation_ecsv(
            fixture.output, fixture.baseline, fixture.target,
            fixture.relation);
    EXPECT_TRUE(serialized.bytes.starts_with("# %ECSV 1.0\n"));
    const auto expected_digests = observation_apt::compute_digests(
        fixture.output, fixture.baseline, fixture.target, fixture.relation);
    EXPECT_EQ(serialized.digests.semantic_sha256,
              expected_digests.semantic_sha256);
    EXPECT_EQ(serialized.digests.envelope_sha256,
              expected_digests.envelope_sha256);
    EXPECT_EQ(serialized.transport.scope,
              observation_apt::matched_output_byte_transport_scope_v1);
    EXPECT_EQ(serialized.transport.sha256,
              "sha256:a4016feb82b2d7b007ea6ae3dbbfbbf18022f25f467e50bb0fd324552bff6ded");
    EXPECT_EQ(serialized.transport.byte_count, 125302U);
    const auto receipt_binding = artifact_publication::make_receipt_binding(
        std::string(artifact_publication::receipt_schema_v1),
        std::string(observation_apt::matched_output_byte_transport_scope_v1),
        serialized.digests.envelope_sha256, serialized.bytes);
    const auto receipt_bytes =
        artifact_publication::canonical_receipt_bytes(receipt_binding);
    EXPECT_EQ(receipt_bytes.size(), 298U);
    EXPECT_EQ("sha256:" + citlali::utils::sha256(receipt_bytes),
              "sha256:fa48cca9fc8218712ac0be2e3e86bd9ed2dbd3877d2af436677c880c9a90e1e8");

    const auto parsed =
        observation_apt::parse_matched_observation_ecsv_with_receipt(
            serialized.bytes, receipt_bytes, fixture.baseline);
    const auto parsed_target = observation_apt::compute_digests(parsed.target);
    const auto fixture_target =
        observation_apt::compute_digests(fixture.target);
    EXPECT_EQ(parsed_target.semantic_sha256, fixture_target.semantic_sha256);
    EXPECT_EQ(parsed_target.envelope_sha256, fixture_target.envelope_sha256);
    const auto parsed_relation = observation_apt::compute_digests(
        parsed.relation, fixture.baseline, parsed.target);
    const auto fixture_relation = observation_apt::compute_digests(
        fixture.relation, fixture.baseline, fixture.target);
    EXPECT_EQ(parsed_relation.semantic_sha256,
              fixture_relation.semantic_sha256);
    EXPECT_EQ(parsed_relation.envelope_sha256,
              fixture_relation.envelope_sha256);
    const auto parsed_output = observation_apt::compute_digests(
        parsed.output, fixture.baseline, parsed.target, parsed.relation);
    EXPECT_EQ(parsed_output.semantic_sha256,
              serialized.digests.semantic_sha256);
    EXPECT_EQ(parsed_output.envelope_sha256,
              serialized.digests.envelope_sha256);
    EXPECT_EQ(observation_apt::serialize_matched_observation_ecsv(
                  parsed.output, fixture.baseline, parsed.target,
                  parsed.relation)
                  .bytes,
              serialized.bytes);
}

TEST(canonical_apt_observation_v1,
     matched_observation_builder_and_wire_fail_closed_without_matcher_policy) {
    const auto fixture = make_observation_contract_fixture();
    std::map<std::int64_t, std::vector<std::int64_t>> pairs_for_target;
    for (const auto &disposition : fixture.relation.target_dispositions) {
        pairs_for_target.emplace(disposition.endpoint.local_key,
                                 disposition.pair_keys);
    }
    std::vector<observation_apt::MatchedOutputFieldSource> selections;
    for (const auto &target : fixture.target.rows) {
        for (const auto &contract : fixture.output.registered_fields) {
            if (contract.authorized_operation !=
                observation_apt::TransformationOperation::
                    copy_baseline_when_matched_null_when_unmatched) {
                continue;
            }
            const auto &pair_keys = pairs_for_target.at(target.row_key);
            selections.push_back(
                {target.row_key, contract.field.name,
                 pair_keys.empty()
                     ? std::optional<std::int64_t>{}
                     : std::optional<std::int64_t>{pair_keys.front()}});
        }
    }
    const auto built = observation_apt::make_matched_observation_output_v1(
        fixture.output.envelope, fixture.baseline, fixture.target,
        fixture.relation, selections);
    ASSERT_EQ(built.rows.size(), fixture.target.rows.size());
    for (std::size_t index = 0; index < built.rows.size(); ++index) {
        EXPECT_EQ(built.rows[index].uid, static_cast<std::int64_t>(index));
    }
    EXPECT_EQ(built.output_presentation_sequence,
              (std::vector<std::int64_t>{0, 1, 2}));
    std::map<std::int64_t, const observation_apt::MatchedOutputRow *> built_rows;
    std::map<std::int64_t, const observation_apt::MatchedOutputRow *> fixture_rows;
    for (const auto &row : built.rows) {
        built_rows.emplace(row.target.local_key, &row);
    }
    for (const auto &row : fixture.output.rows) {
        fixture_rows.emplace(row.target.local_key, &row);
    }
    for (const auto &[key, row] : built_rows) {
        const auto *expected = fixture_rows.at(key);
        EXPECT_EQ(row->target_input_key, expected->target_input_key);
        EXPECT_EQ(observation_apt::canonical_binary64_payload(
                      row->tone_frequency_hz),
                  observation_apt::canonical_binary64_payload(
                      expected->tone_frequency_hz));
        EXPECT_EQ(row->array, expected->array);
        EXPECT_EQ(row->network, expected->network);
        EXPECT_EQ(row->channel, expected->channel);
        EXPECT_EQ(row->relation_pair_keys, expected->relation_pair_keys);
        EXPECT_EQ(row->fields, expected->fields);
    }

    auto reordered_target = fixture.target;
    auto reordered_relation = fixture.relation;
    auto reordered_output = fixture.output;
    std::reverse(reordered_target.inputs.begin(), reordered_target.inputs.end());
    std::reverse(reordered_target.rows.begin(), reordered_target.rows.end());
    std::reverse(reordered_target.registered_fields.begin(),
                 reordered_target.registered_fields.end());
    std::reverse(reordered_relation.network_evidence.begin(),
                 reordered_relation.network_evidence.end());
    std::reverse(reordered_relation.pairs.begin(), reordered_relation.pairs.end());
    std::reverse(reordered_relation.target_dispositions.begin(),
                 reordered_relation.target_dispositions.end());
    std::reverse(reordered_relation.seed_dispositions.begin(),
                 reordered_relation.seed_dispositions.end());
    reordered_relation.target_parent =
        observation_apt::artifact_identity(reordered_target);
    reordered_output.target_parent = reordered_relation.target_parent;
    reordered_output.relation_parent = observation_apt::artifact_identity(
        reordered_relation, fixture.baseline, reordered_target);
    std::reverse(reordered_output.registered_fields.begin(),
                 reordered_output.registered_fields.end());
    std::reverse(reordered_output.rows.begin(), reordered_output.rows.end());
    for (auto &row : reordered_output.rows) {
        row.target = observation_apt::row_reference(
            reordered_output.target_parent, row.target.local_key);
        std::reverse(row.transformations.begin(), row.transformations.end());
    }
    const auto canonical = observation_apt::serialize_matched_observation_ecsv(
        fixture.output, fixture.baseline, fixture.target, fixture.relation);
    EXPECT_EQ(observation_apt::serialize_matched_observation_ecsv(
                  reordered_output, fixture.baseline, reordered_target,
                  reordered_relation)
                  .bytes,
              canonical.bytes);

    auto tampered = canonical.bytes;
    replace_once(tampered, "fixture/café/toltec7",
                 "fixture/café/toltecX");
    EXPECT_THROW(observation_apt::parse_matched_observation_ecsv(
                     tampered, fixture.baseline),
                 apt::ContractError);
    tampered = canonical.bytes;
    replace_once(tampered, "\"[900,901]\"", "\"[901,900]\"");
    EXPECT_THROW(observation_apt::parse_matched_observation_ecsv(
                     tampered, fixture.baseline),
                 apt::ContractError);
    auto missing_selection = selections;
    missing_selection.pop_back();
    EXPECT_THROW(observation_apt::make_matched_observation_output_v1(
                     fixture.output.envelope, fixture.baseline,
                     fixture.target, fixture.relation, missing_selection),
                 apt::ContractError);
}

TEST(canonical_apt_v1, exact_labelled_type_length_digest_vectors) {
    const auto integer_frame = apt::canonical_frame(
        "uid", "int64", "9007199254740991");
    EXPECT_EQ(integer_frame,
              "F3:uidT5:int64V16:9007199254740991;");
    EXPECT_EQ(citlali::utils::sha256(integer_frame),
              "5e86d924a3acd47ae21e8fcb5c21bb40da9f37a9a16755dcdb6cf112c166250b");

    std::string float_frames;
    float_frames += apt::canonical_frame(
        "one", "float64-ieee754", apt::canonical_float64_payload(1.0));
    float_frames += apt::canonical_frame(
        "negative_zero", "float64-ieee754",
        apt::canonical_float64_payload(-0.0));
    float_frames += apt::canonical_frame(
        "denorm_min", "float64-ieee754",
        apt::canonical_float64_payload(
            std::numeric_limits<double>::denorm_min()));
    float_frames += apt::canonical_frame(
        "positive_infinity", "float64-ieee754",
        apt::canonical_float64_payload(
            std::numeric_limits<double>::infinity()));
    float_frames += apt::canonical_frame(
        "quiet_nan", "float64-ieee754",
        apt::canonical_float64_payload(
            std::numeric_limits<double>::quiet_NaN()));
    EXPECT_EQ(
        float_frames,
        "F3:oneT15:float64-ieee754V16:3ff0000000000000;"
        "F13:negative_zeroT15:float64-ieee754V16:8000000000000000;"
        "F10:denorm_minT15:float64-ieee754V16:0000000000000001;"
        "F17:positive_infinityT15:float64-ieee754V4:+inf;"
        "F9:quiet_nanT15:float64-ieee754V3:nan;");
    EXPECT_EQ(citlali::utils::sha256(float_frames),
              "4a566f76572a46c00bd06d035851e1cd80dbfbe640f90d1643bfc38197732ded");
    EXPECT_EQ(apt::canonical_float64_payload(
                  -std::numeric_limits<double>::infinity()),
              "-inf");
    const auto negative_nan =
        std::bit_cast<double>(std::uint64_t{0xfff8000000000001ULL});
    EXPECT_EQ(apt::canonical_float64_payload(negative_nan), "nan");

    const auto previous_locale = std::locale();
    std::locale::global(std::locale(previous_locale, new HostileGrouping));
    const auto locale_independent_payload =
        apt::canonical_float64_payload(1.0);
    std::locale::global(previous_locale);
    EXPECT_EQ(locale_independent_payload, "3ff0000000000000");

    const auto null_frame =
        apt::canonical_frame("missing", "null-int64", "null");
    EXPECT_EQ(null_frame, "F7:missingT10:null-int64V4:null;");
    EXPECT_EQ(citlali::utils::sha256(null_frame),
              "667dbdc49e83c7d94a4d5ea215328fbd76440a4a80e89cef2b984b7c19c0c872");
}

TEST(canonical_apt_v1, exact_baseline_and_extension_catalogs_are_pinned) {
    EXPECT_EQ(catalog_signature(apt::required_baseline_fields_v1()), R"(a_fwhm|float64|arcsec|nullable|producer|citlali:beammap-fit-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
a_fwhm_err|float64|arcsec|nullable|producer|citlali:beammap-fit-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
amp|float64|xs|nullable|producer|citlali:beammap-fit-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
amp_err|float64|xs|nullable|producer|citlali:beammap-fit-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
angle|float64|rad|nullable|producer|citlali:beammap-fit-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
angle_err|float64|rad|nullable|producer|citlali:beammap-fit-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
b_fwhm|float64|arcsec|nullable|producer|citlali:beammap-fit-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
b_fwhm_err|float64|arcsec|nullable|producer|citlali:beammap-fit-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
converge_iter|int64|N/A|required|producer|citlali:beammap-fit-v1|reject|citlali-canonical-apt-baseline-fields-v1
derot_elev|float64|rad|nullable|producer|citlali:beammap-geometry-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
fg|int64|N/A|nullable|unavailable|authority-unresolved-v1|reject|citlali-canonical-apt-baseline-fields-v1
flag|int64|N/A|required|producer|citlali:beammap-quality-v1|reject|citlali-canonical-apt-baseline-fields-v1
flxscale|float64|mJy/beam/xs|nullable|producer|citlali:beammap-calibration-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
loc|int64|N/A|nullable|unavailable|authority-unresolved-v1|reject|citlali-canonical-apt-baseline-fields-v1
ori|int64|N/A|nullable|unavailable|authority-unresolved-v1|reject|citlali-canonical-apt-baseline-fields-v1
pg|int64|N/A|nullable|unavailable|authority-unresolved-v1|reject|citlali-canonical-apt-baseline-fields-v1
responsivity|float64|N/A|nullable|unavailable|authority-unresolved-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
sens|float64|mJy/beam x s^0.5|nullable|producer|citlali:beammap-calibration-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
sig2noise|float64|N/A|nullable|producer|citlali:beammap-fit-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
x_t|float64|arcsec|nullable|producer|citlali:beammap-geometry-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
x_t_derot|float64|arcsec|nullable|producer|citlali:beammap-geometry-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
x_t_err|float64|arcsec|nullable|producer|citlali:beammap-fit-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
x_t_raw|float64|arcsec|nullable|producer|citlali:beammap-geometry-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
y_t|float64|arcsec|nullable|producer|citlali:beammap-geometry-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
y_t_derot|float64|arcsec|nullable|producer|citlali:beammap-geometry-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
y_t_err|float64|arcsec|nullable|producer|citlali:beammap-fit-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
y_t_raw|float64|arcsec|nullable|producer|citlali:beammap-geometry-v1|nan-token|citlali-canonical-apt-baseline-fields-v1
)");

    EXPECT_EQ(catalog_signature(apt::optional_extension_fields_v1()), R"(cal_amp|float64|xs|nullable|producer|citlali:beammap-empirical-calibration-v1|nan-token|citlali-canonical-apt-extension-registry-v1
cal_amp_method|int64|N/A|required|producer|citlali:beammap-empirical-calibration-v1|reject|citlali-canonical-apt-extension-registry-v1
cal_amp_over_fit_amp|float64|N/A|nullable|producer|citlali:beammap-empirical-calibration-v1|nan-token|citlali-canonical-apt-extension-registry-v1
final_prior_d2|float64|N/A|nullable|producer|citlali:beammap-soft-prior-v1|nan-token|citlali-canonical-apt-extension-registry-v1
final_prior_slot_index|int64|N/A|nullable|producer|citlali:beammap-soft-prior-v1|reject|citlali-canonical-apt-extension-registry-v1
flag2|int64|N/A|required|producer|citlali:beammap-quality-v1|reject|citlali-canonical-apt-extension-registry-v1
kids_flag|int64|N/A|required|copied-declared|kids:fit-report-v1|reject|citlali-canonical-apt-extension-registry-v1
map_peak_amp|float64|xs|nullable|producer|citlali:beammap-empirical-calibration-v1|nan-token|citlali-canonical-apt-extension-registry-v1
map_peak_amp_over_fit_amp|float64|N/A|nullable|producer|citlali:beammap-empirical-calibration-v1|nan-token|citlali-canonical-apt-extension-registry-v1
rfi_masked_samples|int64|samples|required|producer|citlali:beammap-mask-diagnostics-v1|reject|citlali-canonical-apt-extension-registry-v1
rfi_masked_scans|int64|scans|required|producer|citlali:beammap-mask-diagnostics-v1|reject|citlali-canonical-apt-extension-registry-v1
scan_band_mask_rejected|int64|N/A|required|producer|citlali:beammap-mask-diagnostics-v1|reject|citlali-canonical-apt-extension-registry-v1
scan_band_masked_edge|int64|N/A|required|producer|citlali:beammap-mask-diagnostics-v1|reject|citlali-canonical-apt-extension-registry-v1
scan_band_masked_rows|int64|rows|required|producer|citlali:beammap-mask-diagnostics-v1|reject|citlali-canonical-apt-extension-registry-v1
scan_band_masked_samples|int64|samples|required|producer|citlali:beammap-mask-diagnostics-v1|reject|citlali-canonical-apt-extension-registry-v1
template_amp|float64|xs|nullable|producer|citlali:beammap-empirical-calibration-v1|nan-token|citlali-canonical-apt-extension-registry-v1
template_amp_over_fit_amp|float64|N/A|nullable|producer|citlali:beammap-empirical-calibration-v1|nan-token|citlali-canonical-apt-extension-registry-v1
template_npix|int64|pix|required|producer|citlali:beammap-empirical-calibration-v1|reject|citlali-canonical-apt-extension-registry-v1
template_offset|float64|xs|nullable|producer|citlali:beammap-empirical-calibration-v1|nan-token|citlali-canonical-apt-extension-registry-v1
template_resid_rms|float64|xs|nullable|producer|citlali:beammap-empirical-calibration-v1|nan-token|citlali-canonical-apt-extension-registry-v1
)");
}

TEST(canonical_apt_v1,
     complete_semantic_envelope_and_transport_vectors_are_fixed) {
    auto document = make_document();
    document.rows.front().fields["final_prior_d2"] =
        std::numeric_limits<double>::quiet_NaN();
    const auto serialized = apt::serialize_ecsv(document);
    EXPECT_EQ(serialized.digests.semantic_sha256,
              "sha256:a7911ac3b08ffdb9f3c6aaab36c33bb5abb47fac2bb729c2d09d79e68228f6db");
    EXPECT_EQ(serialized.digests.envelope_sha256,
              "sha256:cb1e83e3f1a236f51ae80d8ab3f4f79106f3dde20153699513d15c883673b67b");
    EXPECT_EQ(serialized.transport.sha256,
              "sha256:4adc27eac0f9934b885b916a9d2b537ea70f37d6a14fe6e1db891c6c51dee9be");
    EXPECT_EQ(serialized.transport.byte_count, 18759U);
}

TEST(canonical_apt_v1, uid_is_exact_sparse_artifact_local_int64) {
    auto document = make_document();
    EXPECT_NO_THROW(apt::validate(document));
    EXPECT_EQ(document.rows[0].uid, apt::uid_v1_max);
    EXPECT_EQ(document.rows[1].uid, 42);
    EXPECT_EQ(document.rows[2].uid, 0);

    document.rows[0].uid = document.rows[1].uid;
    EXPECT_THROW(apt::validate(document), apt::ContractError);

    document = make_document();
    document.rows[2].uid = -1;
    EXPECT_THROW(apt::validate(document), apt::ContractError);

    document = make_document();
    document.rows[0].uid = apt::uid_v1_max + 1;
    EXPECT_THROW(apt::validate(document), apt::ContractError);
}

TEST(canonical_apt_v1, occurrence_is_opaque_and_distinct_from_content) {
    auto first = make_document("opaque:not-a-uuid/A");
    auto second = make_document("opaque:not-a-uuid/B");
    EXPECT_EQ(apt::semantic_sha256(first), apt::semantic_sha256(second));
    EXPECT_NE(apt::envelope_sha256(first), apt::envelope_sha256(second));

    second = first;
    second.envelope.event_reference = "event:test/second";
    EXPECT_EQ(apt::semantic_sha256(first), apt::semantic_sha256(second));
    EXPECT_NE(apt::envelope_sha256(first), apt::envelope_sha256(second));

    second = first;
    second.context.source_name = "Saturn";
    EXPECT_NE(apt::semantic_sha256(first), apt::semantic_sha256(second));

    first.envelope.occurrence.clear();
    EXPECT_THROW(apt::validate(first), apt::ContractError);
}

TEST(canonical_apt_v1, semantic_digest_has_explicit_order_treatment) {
    const auto first = make_document();
    auto second = first;
    std::reverse(second.rows.begin(), second.rows.end());
    std::reverse(second.raw_manifest.inputs.begin(),
                 second.raw_manifest.inputs.end());
    std::reverse(second.registered_fields.begin(),
                 second.registered_fields.end());

    EXPECT_EQ(apt::semantic_preimage(first), apt::semantic_preimage(second));
    EXPECT_EQ(apt::semantic_sha256(first), apt::semantic_sha256(second));
    EXPECT_EQ(apt::envelope_sha256(first), apt::envelope_sha256(second));

    const auto first_ecsv = apt::serialize_ecsv(first);
    const auto second_ecsv = apt::serialize_ecsv(second);
    EXPECT_NE(first_ecsv.bytes, second_ecsv.bytes);
    EXPECT_NE(first_ecsv.transport.sha256, second_ecsv.transport.sha256);
}

TEST(canonical_apt_v1, raw_manifest_proves_complete_channel_bijection) {
    const auto document = make_document();
    // Channel zero and the exact same tone are legitimate in two networks.
    EXPECT_EQ(document.rows[0].channel, document.rows[2].channel);
    EXPECT_EQ(document.rows[0].tone_frequency_hz,
              document.rows[2].tone_frequency_hz);
    EXPECT_NO_THROW(apt::validate(document));

    auto omitted_raw_input = document;
    omitted_raw_input.raw_manifest.inputs.pop_back();
    EXPECT_THROW(apt::validate(omitted_raw_input), apt::ContractError);

    auto count_omits_channel = document;
    count_omits_channel.raw_manifest.inputs[1].channel_count = 1;
    EXPECT_THROW(apt::validate(count_omits_channel), apt::ContractError);

    auto count_invents_channel = document;
    count_invents_channel.raw_manifest.inputs[1].channel_count = 3;
    EXPECT_THROW(apt::validate(count_invents_channel), apt::ContractError);

    auto zero_count = document;
    zero_count.raw_manifest.inputs[0].channel_count = 0;
    EXPECT_THROW(apt::validate(zero_count), apt::ContractError);

    auto split_network = document;
    split_network.raw_manifest.inputs.push_back({7, "toltec7", 1});
    EXPECT_THROW(apt::validate(split_network), apt::ContractError);

    auto wrong_interface = document;
    wrong_interface.raw_manifest.inputs[0].interface_name = "toltec0";
    EXPECT_THROW(apt::validate(wrong_interface), apt::ContractError);

    auto out_of_range_channel = document;
    out_of_range_channel.rows[1].channel = 2;
    EXPECT_THROW(apt::validate(out_of_range_channel), apt::ContractError);

    auto duplicate_relation = document;
    duplicate_relation.rows[1].channel = 0;
    EXPECT_THROW(apt::validate(duplicate_relation), apt::ContractError);

    auto undeclared_network = document;
    undeclared_network.rows[1].network = 1;
    EXPECT_THROW(apt::validate(undeclared_network), apt::ContractError);

    auto wrong_array = document;
    wrong_array.rows[0].array = 0;
    EXPECT_THROW(apt::validate(wrong_array), apt::ContractError);
}

TEST(canonical_apt_v1, field_surface_is_closed_and_registry_authoritative) {
    const auto document = make_document();
    EXPECT_NO_THROW(apt::validate(document));

    auto reordered = document;
    std::reverse(reordered.registered_fields.begin(),
                 reordered.registered_fields.end());
    EXPECT_EQ(apt::semantic_sha256(document),
              apt::semantic_sha256(reordered));
    EXPECT_EQ(apt::serialize_ecsv(document).bytes,
              apt::serialize_ecsv(reordered).bytes);

    auto missing_baseline = document;
    missing_baseline.registered_fields.erase(
        std::find_if(missing_baseline.registered_fields.begin(),
                     missing_baseline.registered_fields.end(),
                     [](const auto &field) { return field.name == "flag"; }));
    for (auto &row : missing_baseline.rows) {
        row.fields.erase("flag");
    }
    EXPECT_THROW(apt::validate(missing_baseline), apt::ContractError);

    auto rogue = document;
    const auto rogue_field = apt::registered_field_spec(
        "runtime_surprise", apt::ValueType::int64, "N/A", false,
        apt::FieldAuthority::producer, "untrusted", apt::NonFinitePolicy::reject,
        "invented-v99", "self-registered field");
    rogue.registered_fields.push_back(rogue_field);
    for (auto &row : rogue.rows) {
        row.fields[rogue_field.name] = std::int64_t{1};
    }
    EXPECT_THROW(apt::validate(rogue), apt::ContractError);

    auto protected_collision = document;
    auto collision = rogue_field;
    collision.name = "occurrence";
    protected_collision.registered_fields.push_back(collision);
    for (auto &row : protected_collision.rows) {
        row.fields[collision.name] = std::int64_t{1};
    }
    EXPECT_THROW(apt::validate(protected_collision), apt::ContractError);

    auto core_collision = document;
    collision.name = "uid";
    core_collision.registered_fields.push_back(collision);
    for (auto &row : core_collision.rows) {
        row.fields[collision.name] = std::int64_t{1};
    }
    EXPECT_THROW(apt::validate(core_collision), apt::ContractError);

    auto context_collision = document;
    collision.name = "project_id";
    context_collision.registered_fields.push_back(collision);
    for (auto &row : context_collision.rows) {
        row.fields[collision.name] = std::int64_t{1};
    }
    EXPECT_THROW(apt::validate(context_collision), apt::ContractError);

    auto wrong_unit = document;
    auto field = std::find_if(wrong_unit.registered_fields.begin(),
                              wrong_unit.registered_fields.end(),
                              [](const auto &item) {
                                  return item.name == "angle";
                              });
    field->unit = "deg";
    EXPECT_THROW(apt::validate(wrong_unit), apt::ContractError);

    auto duplicate = document;
    duplicate.registered_fields.push_back(duplicate.registered_fields.front());
    EXPECT_THROW(apt::validate(duplicate), apt::ContractError);
}

TEST(canonical_apt_v1, injected_registry_can_only_strictly_extend_v1) {
    auto weakened = apt::canonical_field_registry_v1();
    weakened.required_baseline_fields.erase(std::find_if(
        weakened.required_baseline_fields.begin(),
        weakened.required_baseline_fields.end(),
        [](const auto &field) { return field.name == "flag"; }));
    auto weakened_document = make_document();
    weakened_document.registered_fields.erase(std::find_if(
        weakened_document.registered_fields.begin(),
        weakened_document.registered_fields.end(),
        [](const auto &field) { return field.name == "flag"; }));
    for (auto &row : weakened_document.rows) {
        row.fields.erase("flag");
    }
    EXPECT_THROW(apt::validate(weakened_document, weakened),
                 apt::ContractError);

    auto redefined = apt::canonical_field_registry_v1();
    redefined.required_baseline_fields.front().unit = "invented-unit";
    EXPECT_THROW(apt::validate(make_document(), redefined),
                 apt::ContractError);

    auto alias = apt::canonical_field_registry_v1();
    alias.version = "empty-alias-registry-v1";
    auto alias_document = make_document();
    alias_document.field_registry = alias.version;
    EXPECT_THROW(apt::validate(alias_document, alias), apt::ContractError);

    auto wrong_owner = typed_test_registry();
    wrong_owner.optional_extensions.back().registry = "self-declared-v99";
    auto wrong_owner_document = make_typed_document(wrong_owner);
    EXPECT_THROW(apt::validate(wrong_owner_document, wrong_owner),
                 apt::ContractError);
}

TEST(canonical_apt_v1,
     design_and_polarization_values_preserve_unavailable_nonidentity_state) {
    auto document = make_document();
    for (const auto name : {"fg", "pg", "ori", "loc"}) {
        const auto field = std::find_if(
            document.registered_fields.begin(),
            document.registered_fields.end(),
            [&](const auto &item) { return item.name == name; });
        ASSERT_NE(field, document.registered_fields.end());
        EXPECT_EQ(field->authority, apt::FieldAuthority::unavailable);
        EXPECT_TRUE(field->nullable);
        EXPECT_TRUE(std::holds_alternative<std::int64_t>(
            document.rows.front().fields.at(name)));
    }
    EXPECT_NO_THROW(apt::validate(document));

    auto invented_authority = document;
    auto fg = std::find_if(invented_authority.registered_fields.begin(),
                           invented_authority.registered_fields.end(),
                           [](const auto &item) { return item.name == "fg"; });
    fg->authority = apt::FieldAuthority::producer;
    fg->authority_reference = "citlali-invented";
    EXPECT_THROW(apt::validate(invented_authority), apt::ContractError);

    document.rows.front().fields["fg"] = apt::NullValue{};
    document.rows.front().fields["pg"] = apt::NullValue{};
    document.rows.front().fields["ori"] = apt::NullValue{};
    document.rows.front().fields["loc"] = apt::NullValue{};
    EXPECT_NO_THROW(apt::validate(document));
}

TEST(canonical_apt_v1, typed_null_and_nonfinite_policies_are_exact) {
    auto document = make_document();
    document.rows[0].fields["final_prior_d2"] =
        std::numeric_limits<double>::quiet_NaN();
    document.rows[0].fields["amp_err"] = apt::NullValue{};
    EXPECT_NO_THROW(apt::validate(document));

    document.rows[0].fields["final_prior_d2"] =
        std::numeric_limits<double>::infinity();
    EXPECT_THROW(apt::validate(document), apt::ContractError);

    document.rows[0].fields["final_prior_d2"] =
        -std::numeric_limits<double>::infinity();
    EXPECT_THROW(apt::validate(document), apt::ContractError);

    document = make_document();
    document.rows[0].fields["flag"] = apt::NullValue{};
    EXPECT_THROW(apt::validate(document), apt::ContractError);

    document = make_document();
    document.rows[0].tone_frequency_hz =
        std::numeric_limits<double>::infinity();
    EXPECT_THROW(apt::validate(document), apt::ContractError);
}

TEST(canonical_apt_v1, typed_ecsv_roundtrip_recomputes_embedded_digests) {
    auto document = make_document();
    document.rows.front().fields["final_prior_d2"] =
        std::numeric_limits<double>::quiet_NaN();
    const auto serialized = apt::serialize_ecsv(document);

    EXPECT_NE(serialized.bytes.find("# %ECSV 1.0\n"), std::string::npos);
    EXPECT_NE(serialized.bytes.find("#     profile: \"citlali-beammap-baseline-apt-v1\"\n"),
              std::string::npos);
    EXPECT_NE(serialized.bytes.find("#         channel_count: 2\n"),
              std::string::npos);
    EXPECT_NE(serialized.bytes.find("uid,tone_freq,array,nw,kids_tone"),
              std::string::npos);
    EXPECT_NE(serialized.bytes.find(
                  "#     string_cell: \"quoted-utf8-single-line-v1\"\n"),
              std::string::npos);

    const auto parsed = apt::parse_ecsv(serialized.bytes);
    EXPECT_EQ(parsed.declared_digests.semantic_sha256,
              serialized.digests.semantic_sha256);
    EXPECT_EQ(parsed.declared_digests.envelope_sha256,
              serialized.digests.envelope_sha256);
    EXPECT_EQ(parsed.computed_transport.sha256, serialized.transport.sha256);
    ASSERT_EQ(parsed.document.rows.size(), 3U);
    EXPECT_EQ(parsed.document.rows[0].uid, apt::uid_v1_max);
    EXPECT_EQ(parsed.document.rows[1].uid, 42);
    EXPECT_EQ(parsed.document.rows[2].uid, 0);
    EXPECT_TRUE(std::isnan(std::get<double>(
        parsed.document.rows.front().fields.at("final_prior_d2"))));
    EXPECT_EQ(apt::serialize_ecsv(parsed.document).bytes, serialized.bytes);
}

TEST(canonical_apt_v1, injected_registry_roundtrips_every_ecsv_scalar_type) {
    const auto registry = typed_test_registry();
    const auto document = make_typed_document(registry);
    const auto serialized = apt::serialize_ecsv(document, registry);
    const auto parsed = apt::parse_ecsv(serialized.bytes, registry);

    EXPECT_EQ(std::get<std::int64_t>(
                  parsed.document.rows[0].fields.at("diagnostic_code")),
              std::numeric_limits<std::int64_t>::min());
    EXPECT_EQ(std::get<std::int64_t>(
                  parsed.document.rows[1].fields.at("diagnostic_code")),
              std::numeric_limits<std::int64_t>::max());
    EXPECT_TRUE(std::holds_alternative<apt::NullValue>(
        parsed.document.rows[2].fields.at("diagnostic_code")));
    EXPECT_TRUE(std::isinf(std::get<double>(
        parsed.document.rows[0].fields.at("diagnostic_value"))));
    EXPECT_TRUE(std::isnan(std::get<double>(
        parsed.document.rows[1].fields.at("diagnostic_value"))));
    EXPECT_EQ(std::get<std::string>(parsed.document.rows[0].fields.at("note")),
              "first, café \"detector\"");
    EXPECT_TRUE(std::get<bool>(parsed.document.rows[0].fields.at("selected")));
    EXPECT_TRUE(std::holds_alternative<apt::NullValue>(
        parsed.document.rows[2].fields.at("selected")));
    EXPECT_EQ(apt::serialize_ecsv(parsed.document, registry).bytes,
              serialized.bytes);

    // The artifact's declaration is not enough: the default producer registry
    // must reject the same otherwise-well-formed extension document.
    EXPECT_THROW(apt::validate(document), apt::ContractError);
}

TEST(canonical_apt_v1, ecsv_rejects_key_schema_digest_and_relation_tampering) {
    const auto serialized = apt::serialize_ecsv(make_document());

    auto missing_uid = serialized.bytes;
    replace_once(missing_uid, "\n9007199254740991,", "\n,");
    EXPECT_THROW(apt::parse_ecsv(missing_uid), apt::ContractError);

    auto duplicate_uid = serialized.bytes;
    replace_once(duplicate_uid, "\n9007199254740991,", "\n42,");
    EXPECT_THROW(apt::parse_ecsv(duplicate_uid), apt::ContractError);

    auto nonintegral_uid = serialized.bytes;
    replace_once(nonintegral_uid, "\n9007199254740991,",
                 "\n9007199254740991.0,");
    EXPECT_THROW(apt::parse_ecsv(nonintegral_uid), apt::ContractError);

    auto out_of_range_uid = serialized.bytes;
    replace_once(out_of_range_uid, "\n9007199254740991,",
                 "\n9007199254740992,");
    EXPECT_THROW(apt::parse_ecsv(out_of_range_uid), apt::ContractError);

    auto wrong_uid_type = serialized.bytes;
    replace_once(wrong_uid_type,
                 "# - name: \"uid\"\n#   datatype: \"int64\"\n",
                 "# - name: \"uid\"\n#   datatype: \"float64\"\n");
    EXPECT_THROW(apt::parse_ecsv(wrong_uid_type), apt::ContractError);

    auto stale_semantic = serialized.bytes;
    mutate_declared_digest(stale_semantic, "semantic_sha256");
    EXPECT_THROW(apt::parse_ecsv(stale_semantic), apt::ContractError);

    auto stale_envelope = serialized.bytes;
    mutate_declared_digest(stale_envelope, "envelope_sha256");
    EXPECT_THROW(apt::parse_ecsv(stale_envelope), apt::ContractError);

    auto stale_event = serialized.bytes;
    replace_once(stale_event,
                 "#     event_reference: \"event:test/beammap-baseline#001\"\n",
                 "#     event_reference: \"event:test/beammap-baseline#002\"\n");
    EXPECT_THROW(apt::parse_ecsv(stale_event), apt::ContractError);

    auto forged_relation = serialized.bytes;
    replace_once(forged_relation, "#         channel_count: 2\n",
                 "#         channel_count: 3\n");
    EXPECT_THROW(apt::parse_ecsv(forged_relation), apt::ContractError);

    auto invalid_observation = serialized.bytes;
    replace_once(invalid_observation, "#       observation: 152389\n",
                 "#       observation: -1\n");
    EXPECT_THROW(apt::parse_ecsv(invalid_observation), apt::ContractError);

    auto wrong_profile = serialized.bytes;
    replace_once(wrong_profile,
                 "#     profile: \"citlali-beammap-baseline-apt-v1\"\n",
                 "#     profile: \"invented-v1\"\n");
    EXPECT_THROW(apt::parse_ecsv(wrong_profile), apt::ContractError);
}

TEST(canonical_apt_v1, canonical_text_is_strict_utf8_without_normalization) {
    const auto registry = typed_test_registry();
    auto document = make_typed_document(registry);
    EXPECT_NO_THROW(apt::parse_ecsv(apt::serialize_ecsv(document, registry).bytes,
                                    registry));

    const auto rejects_string = [&](std::string invalid) {
        auto candidate = document;
        candidate.rows[0].fields["note"] = std::move(invalid);
        EXPECT_THROW(apt::validate(candidate, registry), apt::ContractError);
    };
    rejects_string(std::string(1, static_cast<char>(0x80)));
    rejects_string(std::string{"\xc0\xaf", 2});
    rejects_string(std::string{"\xed\xa0\x80", 3});
    rejects_string(std::string{"\xf4\x90\x80\x80", 4});
    rejects_string(std::string{"a\0b", 3});
    rejects_string(std::string{"\xc2\x85", 2});
    rejects_string(std::string{"\xe2\x80\xa8", 3});
    rejects_string(std::string{"\xef\xb7\x90", 3});
    rejects_string(std::string{"\xef\xbf\xbe", 3});
    rejects_string(std::string{"\xef\xbf\xbf", 3});
    rejects_string(std::string{"\xf0\x9f\xbf\xbe", 4});
    rejects_string(std::string{"\xf4\x8f\xbf\xbf", 4});

    auto invalid_metadata = document;
    invalid_metadata.context.source_name = std::string{"\xef\xb7\x90", 3};
    EXPECT_THROW(apt::validate(invalid_metadata, registry),
                 apt::ContractError);

    auto invalid_interface = document;
    invalid_interface.raw_manifest.inputs[0].interface_name.push_back(
        static_cast<char>(0x80));
    EXPECT_THROW(apt::validate(invalid_interface, registry),
                 apt::ContractError);

    auto bytes = apt::serialize_ecsv(document, registry).bytes;
    bytes[bytes.find("Jupiter")] = static_cast<char>(0x80);
    EXPECT_THROW(apt::parse_ecsv(bytes, registry), apt::ContractError);
}

TEST(canonical_apt_v1, envelope_and_context_are_typed_fail_closed_metadata) {
    auto document = make_document();
    document.envelope.event_time_utc = "2026-02-29T00:00:00Z";
    EXPECT_THROW(apt::validate(document), apt::ContractError);

    document = make_document();
    document.context.observation_time_utc = "2024-02-29T00:00:00Z";
    EXPECT_NO_THROW(apt::validate(document));

    document.context.coordinate_frame = "icrs";
    EXPECT_THROW(apt::validate(document), apt::ContractError);

    document = make_document();
    document.envelope.output_role = "science-apt";
    EXPECT_THROW(apt::validate(document), apt::ContractError);
}

TEST(canonical_apt_v1, byte_transport_hash_is_separate_and_envelope_bound) {
    const auto serialized = apt::serialize_ecsv(make_document());
    EXPECT_NO_THROW(apt::parse_ecsv_with_transport(
        serialized.bytes, serialized.transport));

    auto wrong_scope = serialized.transport;
    wrong_scope.scope = "sha256";
    EXPECT_THROW(apt::parse_ecsv_with_transport(serialized.bytes, wrong_scope),
                 apt::ContractError);

    auto wrong_hash = serialized.transport;
    wrong_hash.sha256 =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    EXPECT_THROW(apt::parse_ecsv_with_transport(serialized.bytes, wrong_hash),
                 apt::ContractError);

    auto wrong_count = serialized.transport;
    ++wrong_count.byte_count;
    EXPECT_THROW(apt::parse_ecsv_with_transport(serialized.bytes, wrong_count),
                 apt::ContractError);

    const std::string other_envelope =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const auto receipt_for_other =
        apt::make_byte_transport_hash(serialized.bytes, other_envelope);
    // A low-level receipt is internally consistent with B, but the high-level
    // validator binds it to envelope A recomputed from the ECSV and rejects it.
    EXPECT_NO_THROW(apt::validate_byte_transport(
        serialized.bytes, other_envelope, receipt_for_other));
    EXPECT_THROW(apt::parse_ecsv_with_transport(serialized.bytes,
                                                receipt_for_other),
                 apt::ContractError);

    EXPECT_EQ(apt::make_byte_transport_hash("abc", other_envelope).sha256,
              "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

citlali::pipeline::RawObsDetectorInventory make_phase_b_inventory() {
    citlali::pipeline::RawObsDetectorInventory inventory;
    inventory.observation = {152389, 0, 1};
    inventory.inputs = {
        {{0, "toltec0", 2}, 0, {1.25e9, 1.5e9},
         "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
         1024},
        {{7, "toltec7", 1}, 1, {1.25e9},
         "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
         2048},
    };
    inventory.n_dets = 3;
    inventory.dets = {2, 1};
    inventory.nws = {0, 7};
    inventory.arrays = {0, 1};
    return inventory;
}

struct PhaseBLegacyFixture {
    engine::Calib calib;
    Eigen::VectorXi flag2;
};

struct PhaseBBoundFitReport {
    std::int64_t network = 0;
    std::int64_t observation = 0;
    std::string source;
    std::vector<std::string> header;
    Eigen::MatrixXd model;
};

std::vector<PhaseBBoundFitReport> make_phase_b_bound_fit_reports() {
    std::vector<PhaseBBoundFitReport> reports(2);
    reports[0].network = 0;
    reports[0].observation = 152389;
    reports[0].source = "fit-report-toltec0.ecsv";
    reports[0].header = {"flag", "fg"};
    reports[0].model.resize(2, 2);
    reports[0].model << 3.0, 2.0, -7.0, 3.0;
    reports[1].network = 7;
    reports[1].observation = 152389;
    reports[1].source = "fit-report-toltec7.ecsv";
    reports[1].header = {"flag", "fg"};
    reports[1].model.resize(1, 2);
    reports[1].model << 42.0, 4.0;
    return reports;
}

PhaseBLegacyFixture make_phase_b_legacy_fixture() {
    PhaseBLegacyFixture fixture;
    const auto inventory = make_phase_b_inventory();
    citlali::pipeline::populate_internal_apt_from_detector_inventory(
        fixture.calib, inventory);

    fixture.calib.apt["tone_freq"] =
        (Eigen::Vector3d{} << 1.25e9, 1.5e9, 1.25e9).finished();
    fixture.calib.apt["kids_tone"] =
        (Eigen::Vector3d{} << 0.0, 1.0, 0.0).finished();
    fixture.calib.apt_header_keys.push_back("kids_tone");
    fixture.calib.apt_header_units["kids_tone"] = "N/A";

    for (const auto name : {"final_prior_slot_index", "final_prior_d2",
                            "cal_amp_method", "template_npix"}) {
        fixture.calib.apt[name].resize(3);
        fixture.calib.apt_header_keys.push_back(name);
        fixture.calib.apt_header_units[name] = extension_spec(name).unit;
    }
    fixture.calib.apt["final_prior_slot_index"] << -1.0, 4.0, 8.0;
    fixture.calib.apt["final_prior_d2"] <<
        std::numeric_limits<double>::quiet_NaN(), 1.25, 2.5;
    fixture.calib.apt["cal_amp_method"] << 0.0, 1.0, 0.0;
    fixture.calib.apt["template_npix"] << 0.0, 11.0, 17.0;
    fixture.calib.apt_header_keys.push_back("flag2");
    fixture.calib.apt_header_units["flag2"] = "N/A";
    fixture.flag2.resize(3);
    fixture.flag2 << 0, 1, 128;
    return fixture;
}

apt_producer::CanonicalAptDocumentContext phase_b_context(
    std::string occurrence = "occurrence:injected/opaque#phase-b") {
    apt_producer::CanonicalAptDocumentContext context;
    context.occurrence = std::move(occurrence);
    context.event_reference = "event:injected/beammap#phase-b";
    context.software_revision =
        "46ad23888a40f5102cdfd50c06e49a549bdf8a20";
    context.configuration_reference = "runtime-config:sha256:fixture";
    context.event_time_utc = "2026-08-13T20:01:02Z";
    context.project_id = "2025-C1-COM-01";
    context.source_name = "Jupiter";
    context.observation_time_utc = "2026-08-13T19:01:02Z";
    context.coordinate_frame = "altaz";
    return context;
}

class PhaseBTemporaryDirectory {
public:
    PhaseBTemporaryDirectory() {
        std::random_device entropy;
        path = std::filesystem::temp_directory_path() /
            ("citlali-canonical-apt-phase-b-" +
             std::to_string(entropy()) + "-" + std::to_string(entropy()));
        std::filesystem::create_directory(path);
    }
    ~PhaseBTemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    std::filesystem::path path;
};

std::string read_phase_b_file(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

void write_phase_b_file(const std::filesystem::path &path,
                        std::string_view bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    ASSERT_TRUE(stream);
}

void write_phase_b_valid_raw_netcdf(const std::filesystem::path &path) {
    netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
    const auto time = file.addDim("time", 2);
    const auto sweep = file.addDim("sweep", 2);
    const auto tone = file.addDim("tone", 2);
    const std::vector<netCDF::NcDim> is_dims{time, tone};
    const std::vector<netCDF::NcDim> tone_dims{sweep, tone};
    const double samples[4]{0.0, 0.0, 0.0, 0.0};
    file.addVar("Data.Toltec.Is", netCDF::ncDouble, is_dims)
        .putVar(samples);
    for (const auto [name, value] :
         {std::pair{"Header.Toltec.ObsNum", 152389},
          std::pair{"Header.Toltec.SubObsNum", 0},
          std::pair{"Header.Toltec.ScanNum", 1},
          std::pair{"Header.Toltec.RoachIndex", 0}}) {
        file.addVar(name, netCDF::ncInt).putVar(&value);
    }
    const double tone_offsets[4]{-10.0, 20.0,
                                 std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::infinity()};
    file.addVar("Header.Toltec.ToneFreq", netCDF::ncDouble, tone_dims)
        .putVar(tone_offsets);
    const double lo = 1.0e9;
    file.addVar("Header.Toltec.LoCenterFreq", netCDF::ncDouble).putVar(&lo);
}

TEST(canonical_apt_v1_phase_b,
     retained_raw_inventory_expands_exact_uid_channel_bijection) {
    auto fixture = make_phase_b_legacy_fixture();
    const auto &state = fixture.calib.canonical_apt_producer;
    ASSERT_TRUE(state.raw_inventory_ready);
    EXPECT_EQ(state.raw_manifest.observation, apt::ObservationIdentity({152389, 0, 1}));
    ASSERT_EQ(state.raw_manifest.inputs.size(), 2U);
    EXPECT_EQ(state.raw_manifest.inputs[0], apt::RawInput({0, "toltec0", 2}));
    EXPECT_EQ(state.raw_manifest.inputs[1], apt::RawInput({7, "toltec7", 1}));
    ASSERT_EQ(state.rows.size(), 3U);
    EXPECT_EQ(state.rows[0], engine::CanonicalAptRawRowBinding({0, 0, 0, 0, 1.25e9}));
    EXPECT_EQ(state.rows[1], engine::CanonicalAptRawRowBinding({1, 0, 1, 0, 1.5e9}));
    EXPECT_EQ(state.rows[2], engine::CanonicalAptRawRowBinding({2, 7, 0, 1, 1.25e9}));

    const auto document = apt_producer::make_canonical_document(
        fixture.calib, fixture.flag2, phase_b_context());
    EXPECT_NO_THROW(apt::validate(document));
    ASSERT_EQ(document.rows.size(), 3U);
    EXPECT_EQ(document.rows[0].channel, 0);
    EXPECT_EQ(document.rows[2].channel, 0);
    EXPECT_EQ(document.rows[0].tone_frequency_hz,
              document.rows[2].tone_frequency_hz);

    const auto prepared_v2 = apt_producer::prepare_canonical_baseline_v2(
        document, fixture.calib.canonical_apt_producer);
    const auto verified_v2 =
        citlali::pipeline::canonical_apt_v2::verify_bundle_payload(
            prepared_v2.payload);
    EXPECT_EQ(verified_v2.manifest.kind,
              citlali::pipeline::canonical_apt_v2::BundleKind::baseline);
    EXPECT_EQ(verified_v2.apt.rows.size(), document.rows.size());
    EXPECT_EQ(verified_v2.sources.size(),
              fixture.calib.canonical_apt_producer.sources_v2.size());

    auto wrong_frame = document;
    wrong_frame.context.coordinate_frame = "icrs";
    EXPECT_THROW(apt_producer::prepare_canonical_baseline_v2(
                     wrong_frame, fixture.calib.canonical_apt_producer),
                 apt::ContractError);
}

TEST(canonical_apt_v1_phase_b,
     repeated_observation_population_resets_only_derived_header_state) {
    auto fixture = make_phase_b_legacy_fixture();
    ASSERT_NE(std::find(fixture.calib.apt_header_keys.begin(),
                        fixture.calib.apt_header_keys.end(), "kids_tone"),
              fixture.calib.apt_header_keys.end());
    ASSERT_NE(std::find(fixture.calib.apt_header_keys.begin(),
                        fixture.calib.apt_header_keys.end(), "flag2"),
              fixture.calib.apt_header_keys.end());

    citlali::pipeline::populate_internal_apt_from_detector_inventory(
        fixture.calib, make_phase_b_inventory());
    EXPECT_EQ(fixture.calib.apt_header_keys.size(), 31U);
    EXPECT_EQ(std::count(fixture.calib.apt_header_keys.begin(),
                         fixture.calib.apt_header_keys.end(), "uid"),
              1);
    EXPECT_EQ(std::count(fixture.calib.apt_header_keys.begin(),
                         fixture.calib.apt_header_keys.end(), "kids_tone"),
              0);
    EXPECT_EQ(std::count(fixture.calib.apt_header_keys.begin(),
                         fixture.calib.apt_header_keys.end(), "flag2"),
              0);
    EXPECT_TRUE(fixture.calib.canonical_apt_producer.raw_inventory_ready);
    EXPECT_EQ(fixture.calib.canonical_apt_producer.rows.size(), 3U);
}

TEST(canonical_apt_v1_phase_b,
     raw_inventory_and_observation_sources_fail_closed_on_mismatch) {
    auto duplicate = make_phase_b_inventory();
    duplicate.inputs[1].manifest = {0, "toltec0", 1};
    duplicate.nws[1] = 0;
    duplicate.arrays[1] = 0;
    EXPECT_THROW(citlali::pipeline::validate_rawobs_detector_inventory(duplicate),
                 std::runtime_error);

    auto wrong_interface = make_phase_b_inventory();
    wrong_interface.inputs[0].manifest.interface_name = "toltec00";
    EXPECT_THROW(citlali::pipeline::validate_rawobs_detector_inventory(
                     wrong_interface),
                 std::runtime_error);

    auto wrong_tone_count = make_phase_b_inventory();
    wrong_tone_count.inputs[0].tone_frequencies_hz.pop_back();
    EXPECT_THROW(citlali::pipeline::validate_rawobs_detector_inventory(
                     wrong_tone_count),
                 std::runtime_error);

    auto nonfinite_tone = make_phase_b_inventory();
    nonfinite_tone.inputs[0].tone_frequencies_hz[0] =
        std::numeric_limits<double>::infinity();
    EXPECT_THROW(citlali::pipeline::validate_rawobs_detector_inventory(
                     nonfinite_tone),
                 std::runtime_error);

    const std::vector<citlali::pipeline::RawObsKidsIdentity> kids{
        {0, 152389}, {7, 152389}};
    EXPECT_EQ(citlali::pipeline::validate_rawobs_observation_identity(
                  make_phase_b_inventory(), kids, {152389, 0, 1}, 152389),
              apt::ObservationIdentity({152389, 0, 1}));
    auto mismatch = kids;
    mismatch[1].observation = 152390;
    EXPECT_THROW(citlali::pipeline::validate_rawobs_observation_identity(
                     make_phase_b_inventory(), mismatch, {152389, 0, 1},
                     152389),
                 std::runtime_error);
    EXPECT_THROW(citlali::pipeline::validate_rawobs_observation_identity(
                     make_phase_b_inventory(), kids, {152390, 0, 1}, 152389),
                 std::runtime_error);

    std::map<std::string, Eigen::VectorXd> real_headers;
    real_headers["Header.Dcs.ObsNum"] = Eigen::VectorXd::Constant(1, 152389);
    real_headers["Header.Dcs.SubObsNum"] = Eigen::VectorXd::Constant(1, 0);
    real_headers["Header.Dcs.ScanNum"] = Eigen::VectorXd::Constant(1, 1);
    EXPECT_EQ(apt_producer::telescope_observation_identity(real_headers,
                                                            false),
              apt::ObservationIdentity({152389, 0, 1}));
    EXPECT_THROW(apt_producer::telescope_observation_identity(real_headers,
                                                               true),
                 apt::ContractError);

    std::map<std::string, Eigen::VectorXd> simulation_headers;
    simulation_headers["Header.TelescopeBackend.ObsNum"] =
        Eigen::VectorXd::Constant(1, 152389);
    simulation_headers["Header.TelescopeBackend.SubObsNum"] =
        Eigen::VectorXd::Constant(1, 0);
    simulation_headers["Header.TelescopeBackend.ScanNum"] =
        Eigen::VectorXd::Constant(1, 1);
    const auto simulation_identity =
        apt_producer::telescope_observation_identity(simulation_headers, true);
    EXPECT_EQ(simulation_identity, apt::ObservationIdentity({152389, 0, 1}));
    const auto simulation_fixture = make_phase_b_legacy_fixture();
    EXPECT_EQ(citlali::pipeline::validate_rawobs_observation_identity(
                  simulation_fixture.calib.canonical_apt_producer.raw_manifest,
                  simulation_identity, 152389),
              apt::ObservationIdentity({152389, 0, 1}));
    simulation_headers["Header.TelescopeBackend.ScanNum"](0) = 2;
    EXPECT_THROW(citlali::pipeline::validate_rawobs_observation_identity(
                     simulation_fixture.calib.canonical_apt_producer.raw_manifest,
                     apt_producer::telescope_observation_identity(
                         simulation_headers, true),
                     152389),
                 std::runtime_error);
}

TEST(canonical_apt_v1_phase_b,
     raw_netcdf_authorities_require_exact_safe_types_and_shapes) {
    PhaseBTemporaryDirectory directory;
    const auto valid_path = directory.path / "valid.nc";
    write_phase_b_valid_raw_netcdf(valid_path);
    {
        netCDF::NcFile file(valid_path.string(), netCDF::NcFile::read);
        EXPECT_EQ(citlali::pipeline::detector_count_from_rawobs_file(file), 2);
        EXPECT_EQ(citlali::pipeline::rawobs_observation_identity(file),
                  apt::ObservationIdentity({152389, 0, 1}));
        EXPECT_EQ(citlali::pipeline::rawobs_exact_integer(
                      file, "Header.Toltec.RoachIndex"),
                  0);
        EXPECT_EQ(citlali::pipeline::rawobs_first_sweep_tone_frequencies(
                      file, 2),
                  (std::vector<double>{1.0e9 - 10.0, 1.0e9 + 20.0}));
    }

    const auto fractional = directory.path / "fractional.nc";
    {
        netCDF::NcFile file(fractional.string(), netCDF::NcFile::replace);
        const double value = 1.5;
        file.addVar("fractional", netCDF::ncDouble).putVar(&value);
    }
    {
        netCDF::NcFile file(fractional.string(), netCDF::NcFile::read);
        EXPECT_THROW(citlali::pipeline::rawobs_exact_integer(file,
                                                              "fractional"),
                     std::runtime_error);
    }

    const auto nonscalar = directory.path / "nonscalar.nc";
    {
        netCDF::NcFile file(nonscalar.string(), netCDF::NcFile::replace);
        const auto two = file.addDim("two", 2);
        const int values[2]{1, 2};
        file.addVar("integer_vector", netCDF::ncInt, two).putVar(values);
        const double los[2]{1.0, 2.0};
        file.addVar("Header.Toltec.LoCenterFreq", netCDF::ncDouble, two)
            .putVar(los);
    }
    {
        netCDF::NcFile file(nonscalar.string(), netCDF::NcFile::read);
        EXPECT_THROW(citlali::pipeline::rawobs_exact_integer(
                         file, "integer_vector"),
                     std::runtime_error);
        EXPECT_THROW(citlali::pipeline::rawobs_finite_scalar(
                         file, "Header.Toltec.LoCenterFreq"),
                     std::runtime_error);
    }

    const auto ranks = directory.path / "ranks.nc";
    {
        netCDF::NcFile file(ranks.string(), netCDF::NcFile::replace);
        const auto one = file.addDim("one", 1);
        const auto two = file.addDim("two", 2);
        const std::vector<netCDF::NcDim> rank_three{one, two, one};
        const double values[2]{0.0, 0.0};
        file.addVar("Data.Toltec.Is", netCDF::ncDouble, rank_three)
            .putVar(values);
        file.addVar("Header.Toltec.ToneFreq", netCDF::ncDouble,
                    rank_three)
            .putVar(values);
        const double lo = 1.0;
        file.addVar("Header.Toltec.LoCenterFreq", netCDF::ncDouble)
            .putVar(&lo);
    }
    {
        netCDF::NcFile file(ranks.string(), netCDF::NcFile::read);
        EXPECT_THROW(citlali::pipeline::detector_count_from_rawobs_file(file),
                     std::runtime_error);
        EXPECT_THROW(citlali::pipeline::rawobs_first_sweep_tone_frequencies(
                         file, 2),
                     std::runtime_error);
    }

    const auto integer_tone = directory.path / "integer-tone.nc";
    {
        netCDF::NcFile file(integer_tone.string(), netCDF::NcFile::replace);
        const auto one = file.addDim("one", 1);
        const auto two = file.addDim("two", 2);
        const std::vector<netCDF::NcDim> dims{one, two};
        const int values[2]{1, 2};
        file.addVar("Header.Toltec.ToneFreq", netCDF::ncInt, dims)
            .putVar(values);
        const double lo = 1.0;
        file.addVar("Header.Toltec.LoCenterFreq", netCDF::ncDouble)
            .putVar(&lo);
    }
    {
        netCDF::NcFile file(integer_tone.string(), netCDF::NcFile::read);
        EXPECT_THROW(citlali::pipeline::rawobs_first_sweep_tone_frequencies(
                         file, 2),
                     std::runtime_error);
    }
}

TEST(canonical_apt_v1_phase_b,
     adapter_rejects_manifest_drift_unknown_fields_and_lossy_integers) {
    const auto expect_adapter_failure = [](auto mutate) {
        auto fixture = make_phase_b_legacy_fixture();
        mutate(fixture);
        EXPECT_THROW(apt_producer::make_canonical_document(
                         fixture.calib, fixture.flag2, phase_b_context()),
                     apt::ContractError);
    };
    expect_adapter_failure([](auto &fixture) {
        fixture.calib.apt["uid"](1) = 0.0;
    });
    expect_adapter_failure([](auto &fixture) {
        fixture.calib.apt["uid"](1) = 1.5;
    });
    expect_adapter_failure([](auto &fixture) {
        fixture.calib.apt["nw"](2) = 0.0;
    });
    expect_adapter_failure([](auto &fixture) {
        fixture.calib.apt["kids_tone"](1) = 0.0;
    });
    expect_adapter_failure([](auto &fixture) {
        fixture.calib.apt["tone_freq"](0) =
            std::nextafter(fixture.calib.apt["tone_freq"](0),
                           std::numeric_limits<double>::infinity());
    });
    expect_adapter_failure([](auto &fixture) {
        fixture.calib.apt_header_keys.push_back("runtime_surprise");
        fixture.calib.apt["runtime_surprise"].setOnes(3);
    });
    expect_adapter_failure([](auto &fixture) {
        fixture.calib.apt_header_keys.push_back("occurrence");
        fixture.calib.apt["occurrence"].setOnes(3);
    });
    EXPECT_THROW(apt_producer::exact_legacy_int64(0x1p63, "boundary"),
                 apt::ContractError);
    EXPECT_EQ(apt_producer::exact_legacy_int64(-0x1p63, "boundary"),
              std::numeric_limits<std::int64_t>::min());

    auto fixture = make_phase_b_legacy_fixture();
    auto reports = make_phase_b_bound_fit_reports();
    for (auto &report : reports) {
        report.header = {"fg"};
        report.model = report.model.rightCols(1).eval();
    }
    EXPECT_NO_THROW(apt_producer::preflight_atomic_kids_fit_reports(
        reports, fixture.calib.canonical_apt_producer.raw_manifest));
    auto unknown_reports = reports;
    for (auto &report : unknown_reports) {
        report.header = {"runtime_surprise"};
    }
    EXPECT_NO_THROW(apt_producer::preflight_atomic_kids_fit_reports(
        unknown_reports,
        fixture.calib.canonical_apt_producer.raw_manifest));
    auto literal_canonical_reports = reports;
    for (auto &report : literal_canonical_reports) {
        report.header = {"kids_flag"};
    }
    EXPECT_THROW(apt_producer::preflight_atomic_kids_fit_reports(
                     literal_canonical_reports,
                     fixture.calib.canonical_apt_producer.raw_manifest),
                 apt::ContractError);

    const auto headers_before_failure = fixture.calib.apt_header_keys;
    const auto apt_before_failure = fixture.calib.apt;
    EXPECT_NO_THROW(apt_producer::apply_atomic_kids_fit_report_overlay(
        fixture.calib, unknown_reports,
        fixture.calib.canonical_apt_producer.raw_manifest));
    EXPECT_EQ(fixture.calib.apt_header_keys, headers_before_failure);
    ASSERT_EQ(fixture.calib.apt.size(), apt_before_failure.size());
    for (const auto &[name, before] : apt_before_failure) {
        const auto &after = fixture.calib.apt.at(name);
        ASSERT_EQ(after.size(), before.size()) << name;
        for (Eigen::Index index = 0; index < before.size(); ++index) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(after(index)),
                      std::bit_cast<std::uint64_t>(before(index)))
                << name;
        }
    }

    EXPECT_NO_THROW(apt_producer::apply_atomic_kids_fit_report_overlay(
        fixture.calib, reports,
        fixture.calib.canonical_apt_producer.raw_manifest));
    EXPECT_EQ(fixture.calib.apt.at("fg"),
              (Eigen::Vector3d{} << 2.0, 3.0, 4.0).finished());
    EXPECT_EQ(std::count(fixture.calib.apt_header_keys.begin(),
                         fixture.calib.apt_header_keys.end(), "fg"),
              1);
    const auto overlay_document = apt_producer::make_canonical_document(
        fixture.calib, fixture.flag2, phase_b_context());
    EXPECT_EQ(std::get<std::int64_t>(
                  overlay_document.rows[0].fields.at("fg")),
              2);

    fixture.calib.n_dets = 2;
    EXPECT_THROW(apt_producer::apply_atomic_kids_fit_report_overlay(
                     fixture.calib, reports,
                     fixture.calib.canonical_apt_producer.raw_manifest),
                 apt::ContractError);
    fixture.calib.n_dets = 4;
    EXPECT_THROW(apt_producer::apply_atomic_kids_fit_report_overlay(
                     fixture.calib, reports,
                     fixture.calib.canonical_apt_producer.raw_manifest),
                 apt::ContractError);
}

TEST(canonical_apt_v1_phase_b,
     atomic_fit_report_binding_preserves_nonbinary_kids_flag) {
    const auto expect_apt_bits_equal = [](const auto &expected,
                                          const auto &actual) {
        ASSERT_EQ(actual.size(), expected.size());
        for (const auto &[name, before] : expected) {
            const auto &after = actual.at(name);
            ASSERT_EQ(after.size(), before.size()) << name;
            for (Eigen::Index index = 0; index < before.size(); ++index) {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(after(index)),
                          std::bit_cast<std::uint64_t>(before(index)))
                    << name << " row " << index;
            }
        }
    };
    const auto expect_failure_without_mutation = [&](auto mutate) {
        auto fixture = make_phase_b_legacy_fixture();
        auto reports = make_phase_b_bound_fit_reports();
        mutate(reports);
        const auto apt_before = fixture.calib.apt;
        const auto headers_before = fixture.calib.apt_header_keys;
        const auto units_before = fixture.calib.apt_header_units;
        const auto meta_before = YAML::Dump(fixture.calib.apt_meta);
        EXPECT_THROW(apt_producer::apply_atomic_kids_fit_report_overlay(
                         fixture.calib, reports,
                         fixture.calib.canonical_apt_producer.raw_manifest),
                     apt::ContractError);
        EXPECT_EQ(fixture.calib.apt_header_keys, headers_before);
        EXPECT_EQ(fixture.calib.apt_header_units, units_before);
        EXPECT_EQ(YAML::Dump(fixture.calib.apt_meta), meta_before);
        expect_apt_bits_equal(apt_before, fixture.calib.apt);
    };

    auto no_report_fixture = make_phase_b_legacy_fixture();
    const auto no_report_document = apt_producer::make_canonical_document(
        no_report_fixture.calib, no_report_fixture.flag2, phase_b_context());
    EXPECT_EQ(std::count_if(no_report_document.registered_fields.begin(),
                            no_report_document.registered_fields.end(),
                            [](const auto &field) {
                                return field.name == "kids_flag";
                            }),
              0);

    // Reordered/renamed per-input headers are tolerated when they resolve to the
    // same positional contract and unknown diagnostics are ignored.
    // Reordered/renamed per-input headers fail closed.
    expect_failure_without_mutation([](auto &reports) {
        reports[1].header = {"fg", "flag"};
    });
    expect_failure_without_mutation([](auto &reports) {
        reports[1].header = {"flag", "pg"};
    });
    // Matrix/report order remains bound to the exact raw network and obsid.
    expect_failure_without_mutation([](auto &reports) {
        std::swap(reports[0], reports[1]);
    });
    expect_failure_without_mutation([](auto &reports) {
        reports[1].network = 0;
    });
    expect_failure_without_mutation([](auto &reports) {
        reports[1].observation = 152390;
    });
    expect_failure_without_mutation([](auto &reports) {
        reports[1].source.clear();
    });
    expect_failure_without_mutation([](auto &reports) {
        reports.pop_back();
    });
    expect_failure_without_mutation([](auto &reports) {
        reports.push_back(reports.back());
    });
    expect_failure_without_mutation([](auto &reports) {
        reports[1].header.clear();
        reports[1].model.resize(1, 0);
    });
    expect_failure_without_mutation([](auto &reports) {
        reports[1].model.resize(1, 1);
    });
    expect_failure_without_mutation([](auto &reports) {
        reports[0].model.conservativeResize(1, 2);
    });
    auto fixture_for_tolerance = make_phase_b_legacy_fixture();
    auto reports_for_tolerance = make_phase_b_bound_fit_reports();
    reports_for_tolerance[0].model(0, 0) = 1.5;
    EXPECT_NO_THROW(apt_producer::apply_atomic_kids_fit_report_overlay(
        fixture_for_tolerance.calib, reports_for_tolerance,
        fixture_for_tolerance.calib.canonical_apt_producer.raw_manifest));
    EXPECT_FALSE(std::find(fixture_for_tolerance.calib.apt_header_keys.begin(),
                          fixture_for_tolerance.calib.apt_header_keys.end(),
                          "kids_flag") !=
                 fixture_for_tolerance.calib.apt_header_keys.end());

    fixture_for_tolerance = make_phase_b_legacy_fixture();
    reports_for_tolerance = make_phase_b_bound_fit_reports();
    reports_for_tolerance[0].model(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_NO_THROW(apt_producer::apply_atomic_kids_fit_report_overlay(
        fixture_for_tolerance.calib, reports_for_tolerance,
        fixture_for_tolerance.calib.canonical_apt_producer.raw_manifest));
    EXPECT_FALSE(std::find(fixture_for_tolerance.calib.apt_header_keys.begin(),
                          fixture_for_tolerance.calib.apt_header_keys.end(),
                          "kids_flag") !=
                 fixture_for_tolerance.calib.apt_header_keys.end());
    expect_failure_without_mutation([](auto &reports) {
        reports[0].header = {"flag", "kids_flag"};
        reports[1].header = {"flag", "kids_flag"};
    });
    expect_failure_without_mutation([](auto &reports) {
        reports[0].header = {"flag", "flag2"};
        reports[1].header = {"flag", "flag2"};
    });

    auto fixture = make_phase_b_legacy_fixture();
    const auto baseline_flag = fixture.calib.apt.at("flag");
    const auto baseline_flag2 = fixture.flag2;
    const auto uid_before = fixture.calib.apt.at("uid");
    const auto network_before = fixture.calib.apt.at("nw");
    const auto tone_before = fixture.calib.apt.at("tone_freq");
    auto reports = make_phase_b_bound_fit_reports();
    ASSERT_NO_THROW(apt_producer::apply_atomic_kids_fit_report_overlay(
        fixture.calib, reports,
        fixture.calib.canonical_apt_producer.raw_manifest));
    EXPECT_EQ(fixture.calib.apt.at("kids_flag"),
              (Eigen::Vector3d{} << 3.0, -7.0, 42.0).finished());
    EXPECT_EQ(fixture.calib.apt.at("fg"),
              (Eigen::Vector3d{} << 2.0, 3.0, 4.0).finished());
    EXPECT_EQ(fixture.calib.apt.at("flag"), baseline_flag);
    EXPECT_EQ(fixture.flag2, baseline_flag2);
    EXPECT_EQ(fixture.calib.apt.at("uid"), uid_before);
    EXPECT_EQ(fixture.calib.apt.at("nw"), network_before);
    EXPECT_EQ(fixture.calib.apt.at("tone_freq"), tone_before);
    EXPECT_EQ(std::count(fixture.calib.apt_header_keys.begin(),
                         fixture.calib.apt_header_keys.end(), "kids_flag"),
              1);

    const auto document = apt_producer::make_canonical_document(
        fixture.calib, fixture.flag2, phase_b_context());
    ASSERT_EQ(document.rows.size(), 3U);
    EXPECT_EQ(std::get<std::int64_t>(
                  document.rows[0].fields.at("kids_flag")),
              3);
    EXPECT_EQ(std::get<std::int64_t>(
                  document.rows[1].fields.at("kids_flag")),
              -7);
    EXPECT_EQ(std::get<std::int64_t>(
                  document.rows[2].fields.at("kids_flag")),
              42);
    EXPECT_EQ(std::get<std::int64_t>(document.rows[1].fields.at("flag")),
              0);
    EXPECT_EQ(std::get<std::int64_t>(document.rows[1].fields.at("flag2")),
              1);
    const auto serialized = apt::serialize_ecsv(document);
    const auto parsed = apt::parse_ecsv_with_transport(serialized.bytes,
                                                       serialized.transport);
    EXPECT_EQ(std::get<std::int64_t>(
                  parsed.document.rows[1].fields.at("kids_flag")),
              -7);
}

TEST(canonical_apt_v1_phase_b,
     atomic_fit_report_tolerates_legacy_kids_flag_and_unknown_columns) {
    auto fixture = make_phase_b_legacy_fixture();
    auto reports = make_phase_b_bound_fit_reports();
    for (auto &report : reports) {
        report.header = {"flag", "f_in"};
        const auto rows = report.model.rows();
        report.model.conservativeResize(rows, 2);
        for (Eigen::Index index = 0; index < rows; ++index) {
            report.model(index, 0) = 1.5 + static_cast<double>(index);
            report.model(index, 1) = 7.25 + static_cast<double>(index);
        }
    }

    ASSERT_FALSE(std::find(fixture.calib.apt_header_keys.begin(),
                          fixture.calib.apt_header_keys.end(),
                          "kids_flag") !=
                 fixture.calib.apt_header_keys.end());
    ASSERT_FALSE(std::find(fixture.calib.apt_header_keys.begin(),
                          fixture.calib.apt_header_keys.end(),
                          "f_in") !=
                 fixture.calib.apt_header_keys.end());

    EXPECT_NO_THROW(apt_producer::apply_atomic_kids_fit_report_overlay(
        fixture.calib, reports,
        fixture.calib.canonical_apt_producer.raw_manifest));
    EXPECT_FALSE(std::find(fixture.calib.apt_header_keys.begin(),
                          fixture.calib.apt_header_keys.end(),
                          "kids_flag") !=
                 fixture.calib.apt_header_keys.end());
    EXPECT_FALSE(std::find(fixture.calib.apt_header_keys.begin(),
                          fixture.calib.apt_header_keys.end(),
                          "f_in") !=
                 fixture.calib.apt_header_keys.end());
    EXPECT_NO_THROW(apt_producer::make_canonical_document(
        fixture.calib, fixture.flag2, phase_b_context()));
}

TEST(canonical_apt_v1_phase_b,
     occurrence_is_injected_once_and_science_state_is_not_mutated) {
    auto fixture = make_phase_b_legacy_fixture();
    int calls = 0;
    fixture.calib.canonical_apt_producer.issuance_factory = [&] {
        ++calls;
        return engine::CanonicalAptIssuance{
            "opaque occurrence injected verbatim",
            "opaque event injected verbatim"};
    };
    const auto apt_before = fixture.calib.apt;
    const auto headers_before = fixture.calib.apt_header_keys;
    const auto flag2_before = fixture.flag2;

    auto context = phase_b_context();
    apt_producer::inject_issuance_context(
        context, fixture.calib.canonical_apt_producer);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(context.occurrence, "opaque occurrence injected verbatim");
    const auto document = apt_producer::make_canonical_document(
        fixture.calib, fixture.flag2, context);
    EXPECT_EQ(document.envelope.occurrence,
              "opaque occurrence injected verbatim");
    EXPECT_EQ(fixture.calib.apt_header_keys, headers_before);
    EXPECT_EQ(fixture.flag2, flag2_before);
    ASSERT_EQ(fixture.calib.apt.size(), apt_before.size());
    for (const auto &[name, before] : apt_before) {
        const auto &after = fixture.calib.apt.at(name);
        ASSERT_EQ(after.size(), before.size()) << name;
        for (Eigen::Index index = 0; index < before.size(); ++index) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(before(index)),
                      std::bit_cast<std::uint64_t>(after(index))) << name;
        }
    }

    const auto semantic = apt::semantic_sha256(document);
    const auto envelope = apt::envelope_sha256(document);
    auto other = document;
    other.envelope.occurrence = "second opaque occurrence";
    EXPECT_EQ(apt::semantic_sha256(other), semantic);
    EXPECT_NE(apt::envelope_sha256(other), envelope);

    fixture.calib.canonical_apt_producer.issuance_factory = [] {
        return engine::CanonicalAptIssuance{"", "nonempty event"};
    };
    EXPECT_THROW(apt_producer::inject_issuance_context(
                     context, fixture.calib.canonical_apt_producer),
                 apt::ContractError);
    fixture.calib.canonical_apt_producer.issuance_factory = {};
    EXPECT_THROW(apt_producer::inject_issuance_context(
                     context, fixture.calib.canonical_apt_producer),
                 apt::ContractError);
    calls = 0;
    fixture.calib.canonical_apt_producer.issuance_factory = [&] {
        ++calls;
        return engine::CanonicalAptIssuance{"opaque\noccurrence",
                                             "nonempty event"};
    };
    EXPECT_THROW(apt_producer::inject_issuance_context(
                     context, fixture.calib.canonical_apt_producer),
                 apt::ContractError);
    EXPECT_EQ(calls, 1);

    const auto entropy_a = engine::make_canonical_apt_entropy_issuance();
    const auto entropy_b = engine::make_canonical_apt_entropy_issuance();
    EXPECT_TRUE(entropy_a.occurrence.starts_with("apt-occurrence:entropy/"));
    EXPECT_TRUE(entropy_a.event_reference.starts_with("apt-event:entropy/"));
    EXPECT_NE(entropy_a.occurrence, entropy_b.occurrence);
    EXPECT_NE(entropy_a.event_reference, entropy_b.event_reference);

    EXPECT_EQ(apt_producer::utc_timestamp_from_unix_seconds(0.75),
              "1970-01-01T00:00:00Z");
    EXPECT_THROW(apt_producer::utc_timestamp_from_unix_seconds(
                     std::numeric_limits<double>::infinity()),
                 apt::ContractError);
    EXPECT_THROW(apt_producer::utc_timestamp_from_unix_seconds(
                     std::ldexp(
                         1.0,
                         std::numeric_limits<std::time_t>::digits)),
                 apt::ContractError);
    if constexpr (std::numeric_limits<std::time_t>::digits >= 40) {
        EXPECT_EQ(apt_producer::utc_timestamp_from_unix_seconds(
                      253402300799.0),
                  "9999-12-31T23:59:59Z");
        EXPECT_THROW(apt_producer::utc_timestamp_from_unix_seconds(
                         253402300800.0),
                     apt::ContractError);
    }
}

TEST(canonical_apt_v1_phase_b,
     typed_adapter_roundtrips_exact_values_and_normalizes_only_artifact_nan) {
    auto fixture = make_phase_b_legacy_fixture();
    fixture.calib.apt["amp"](0) = -0.0;
    fixture.calib.apt["amp"](1) =
        std::numeric_limits<double>::denorm_min();
    const auto document = apt_producer::make_canonical_document(
        fixture.calib, fixture.flag2, phase_b_context());
    const auto serialized = apt::serialize_ecsv(document);
    const auto parsed = apt::parse_ecsv_with_transport(
        serialized.bytes, serialized.transport);
    ASSERT_EQ(parsed.document.rows.size(), 3U);
    std::set<std::string> expected_fields;
    for (const auto &name : fixture.calib.apt_header_keys) {
        if (name != "uid" && name != "tone_freq" && name != "array" &&
            name != "nw" && name != "kids_tone") {
            expected_fields.insert(name);
        }
    }
    std::set<std::string> declared_fields;
    for (const auto &field : parsed.document.registered_fields) {
        declared_fields.insert(field.name);
    }
    EXPECT_EQ(declared_fields, expected_fields);
    ASSERT_EQ(parsed.document.rows.size(),
              fixture.calib.canonical_apt_producer.rows.size());
    for (std::size_t row_index = 0;
         row_index < parsed.document.rows.size(); ++row_index) {
        const auto &row = parsed.document.rows[row_index];
        const auto &binding =
            fixture.calib.canonical_apt_producer.rows[row_index];
        EXPECT_EQ(row.uid, binding.uid);
        EXPECT_EQ(row.network, binding.network);
        EXPECT_EQ(row.channel, binding.channel);
        EXPECT_EQ(row.array, binding.array);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(row.tone_frequency_hz),
                  std::bit_cast<std::uint64_t>(
                      fixture.calib.apt.at("tone_freq")(
                          static_cast<Eigen::Index>(row_index))));
        std::set<std::string> actual_row_fields;
        for (const auto &[name, value] : row.fields) {
            actual_row_fields.insert(name);
            if (name == "flag2") {
                EXPECT_EQ(std::get<std::int64_t>(value),
                          fixture.flag2(static_cast<Eigen::Index>(row_index)));
                continue;
            }
            const auto contract =
                apt_producer::canonical_registered_field(name);
            ASSERT_TRUE(contract.has_value()) << name;
            const auto source = fixture.calib.apt.at(name)(
                static_cast<Eigen::Index>(row_index));
            if (contract->type == apt::ValueType::int64) {
                EXPECT_EQ(std::get<std::int64_t>(value),
                          apt_producer::exact_legacy_int64(source, name))
                    << name;
            } else {
                const auto artifact = std::get<double>(value);
                if (std::isnan(source)) {
                    EXPECT_TRUE(std::isnan(artifact)) << name;
                } else {
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(artifact),
                              std::bit_cast<std::uint64_t>(source))
                        << name;
                }
            }
        }
        EXPECT_EQ(actual_row_fields, expected_fields);
    }
    EXPECT_EQ(std::bit_cast<std::uint64_t>(std::get<double>(
                  parsed.document.rows[0].fields.at("amp"))),
              std::bit_cast<std::uint64_t>(-0.0));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(std::get<double>(
                  parsed.document.rows[1].fields.at("amp"))),
              std::bit_cast<std::uint64_t>(
                  std::numeric_limits<double>::denorm_min()));
    EXPECT_TRUE(std::isnan(std::get<double>(
        parsed.document.rows[0].fields.at("final_prior_d2"))));
    EXPECT_EQ(std::get<std::int64_t>(
                  parsed.document.rows[2].fields.at("flag2")),
              128);
    for (const auto name : {"fg", "pg", "ori", "loc"}) {
        EXPECT_EQ(std::get<std::int64_t>(
                      parsed.document.rows[0].fields.at(name)),
                  1);
    }
}

TEST(canonical_apt_v1_phase_b,
     receipt_is_exact_envelope_bound_and_published_last) {
    EXPECT_EQ(apt_producer::PublicationStage::artifact_staged,
              apt_producer::PublicationStage::ecsv_staged);
    EXPECT_EQ(apt_producer::PublicationStage::artifact_validated,
              apt_producer::PublicationStage::ecsv_validated);
    EXPECT_EQ(apt_producer::PublicationStage::before_artifact_publish,
              apt_producer::PublicationStage::before_ecsv_publish);
    EXPECT_EQ(apt_producer::PublicationStage::artifact_published,
              apt_producer::PublicationStage::ecsv_published);
    EXPECT_EQ(static_cast<int>(apt_producer::PublicationStage::ecsv_staged),
              0);
    EXPECT_EQ(static_cast<int>(
                  apt_producer::PublicationStage::ecsv_validated),
              1);
    EXPECT_EQ(static_cast<int>(
                  apt_producer::PublicationStage::receipt_staged),
              2);
    EXPECT_EQ(static_cast<int>(
                  apt_producer::PublicationStage::receipt_validated),
              3);
    EXPECT_EQ(static_cast<int>(
                  apt_producer::PublicationStage::before_ecsv_publish),
              4);
    EXPECT_EQ(static_cast<int>(
                  apt_producer::PublicationStage::ecsv_published),
              5);
    EXPECT_EQ(static_cast<int>(
                  apt_producer::PublicationStage::before_receipt_publish),
              6);

    PhaseBTemporaryDirectory directory;
    const auto output = directory.path / "toltec_beammap_152389.ecsv";
    auto fixture = make_phase_b_legacy_fixture();
    const auto document = apt_producer::make_canonical_document(
        fixture.calib, fixture.flag2, phase_b_context());
    std::vector<apt_producer::PublicationStage> stages;
    apt_producer::CanonicalAptPublicationHooks hooks;
    const auto receipt_path =
        std::filesystem::path(output.string() + ".sha256");
    hooks.on_stage = [&](auto stage, const auto &, const auto &) {
        EXPECT_FALSE(std::filesystem::exists(receipt_path));
        if (stage < apt_producer::PublicationStage::ecsv_published) {
            EXPECT_FALSE(std::filesystem::exists(output));
        } else {
            EXPECT_TRUE(std::filesystem::exists(output));
        }
        stages.push_back(stage);
    };
    const auto result = apt_producer::publish_canonical_apt(
        document, output, hooks);
    EXPECT_EQ(result.ecsv_path, output);
    EXPECT_EQ(result.receipt_path,
              std::filesystem::path(output.string() + ".sha256"));
    EXPECT_TRUE(std::filesystem::exists(result.ecsv_path));
    EXPECT_TRUE(std::filesystem::exists(result.receipt_path));
    ASSERT_FALSE(stages.empty());
    EXPECT_EQ(stages.back(),
              apt_producer::PublicationStage::before_receipt_publish);
    EXPECT_EQ(stages,
              (std::vector<apt_producer::PublicationStage>{
                  apt_producer::PublicationStage::ecsv_staged,
                  apt_producer::PublicationStage::ecsv_validated,
                  apt_producer::PublicationStage::receipt_staged,
                  apt_producer::PublicationStage::receipt_validated,
                  apt_producer::PublicationStage::before_ecsv_publish,
                  apt_producer::PublicationStage::ecsv_published,
                  apt_producer::PublicationStage::before_receipt_publish}));

    const auto bytes = read_phase_b_file(result.ecsv_path);
    const auto receipt = apt_producer::parse_canonical_apt_receipt(
        read_phase_b_file(result.receipt_path));
    EXPECT_EQ(receipt.scope, apt::byte_transport_scope_v1);
    EXPECT_EQ(receipt.envelope_sha256, result.digests.envelope_sha256);
    EXPECT_EQ(receipt.sha256, result.transport.sha256);
    EXPECT_EQ(receipt.byte_count, bytes.size());
    EXPECT_NO_THROW(apt_producer::validate_published_canonical_apt(
        result.ecsv_path, result.receipt_path));
    const auto receipt_bytes = read_phase_b_file(result.receipt_path);
    EXPECT_EQ(std::count(receipt_bytes.begin(), receipt_bytes.end(), '\n'), 5);
    for (const auto &entry :
         std::filesystem::directory_iterator(directory.path)) {
        EXPECT_EQ(entry.path().filename().string().find(".stage-"),
                  std::string::npos);
    }
}

TEST(canonical_apt_v1_phase_b,
     no_overwrite_races_tampering_and_failure_cleanup_are_fail_closed) {
    const auto run_failure = [](auto configure) {
        PhaseBTemporaryDirectory directory;
        const auto output = directory.path / "beammap.ecsv";
        auto fixture = make_phase_b_legacy_fixture();
        const auto document = apt_producer::make_canonical_document(
            fixture.calib, fixture.flag2, phase_b_context());
        apt_producer::CanonicalAptPublicationHooks hooks;
        configure(output, hooks);
        EXPECT_THROW(apt_producer::publish_canonical_apt(document, output,
                                                         hooks),
                     std::runtime_error);
        EXPECT_FALSE(std::filesystem::exists(
            std::filesystem::path(output.string() + ".sha256")));
        if (std::filesystem::exists(output)) {
            EXPECT_EQ(read_phase_b_file(output), "raced-sentinel");
        }
        for (const auto &entry :
             std::filesystem::directory_iterator(directory.path)) {
            EXPECT_FALSE(entry.path().filename().string().find(".stage-") !=
                         std::string::npos);
        }
    };

    run_failure([](const auto &, auto &hooks) {
        hooks.on_stage = [](auto stage, const auto &staged_ecsv,
                            const auto &) {
            if (stage == apt_producer::PublicationStage::ecsv_staged) {
                auto bytes = read_phase_b_file(staged_ecsv);
                mutate_declared_digest(bytes, "semantic_sha256");
                write_phase_b_file(staged_ecsv, bytes);
            }
        };
    });
    run_failure([](const auto &, auto &hooks) {
        hooks.on_stage = [](auto stage, const auto &,
                            const auto &staged_receipt) {
            if (stage ==
                apt_producer::PublicationStage::before_ecsv_publish) {
                std::filesystem::remove(staged_receipt);
                write_phase_b_file(staged_receipt,
                                   "replacement-receipt-stage-entry");
            }
        };
    });
    run_failure([](const auto &, auto &hooks) {
        hooks.on_stage = [](auto stage, const auto &staged_ecsv,
                            const auto &) {
            if (stage == apt_producer::PublicationStage::ecsv_staged) {
                std::filesystem::permissions(
                    staged_ecsv.parent_path(),
                    std::filesystem::perms::none,
                    std::filesystem::perm_options::replace);
                throw std::runtime_error(
                    "injected failure with inaccessible staging directory");
            }
        };
    });
    run_failure([](const auto &, auto &hooks) {
        hooks.on_stage = [](auto stage, const auto &staged_ecsv,
                            const auto &) {
            if (stage == apt_producer::PublicationStage::ecsv_published) {
                const auto owner =
                    staged_ecsv.parent_path() / ".owner-artifact";
                std::error_code error;
                if (!std::filesystem::remove(owner, error) || error) {
                    throw std::runtime_error(
                        "failed to remove artifact owner proof in failpoint");
                }
                throw std::runtime_error(
                    "injected failure after artifact owner unlink");
            }
        };
    });
    run_failure([](const auto &, auto &hooks) {
        hooks.on_stage = [](auto stage, const auto &staged_ecsv,
                            const auto &) {
            if (stage ==
                apt_producer::PublicationStage::before_ecsv_publish) {
                std::filesystem::remove(staged_ecsv);
                write_phase_b_file(staged_ecsv, "replacement-stage-entry");
            }
        };
    });
    run_failure([](const auto &output, auto &hooks) {
        hooks.on_stage = [output](auto stage, const auto &, const auto &) {
            if (stage == apt_producer::PublicationStage::before_ecsv_publish) {
                write_phase_b_file(output, "raced-sentinel");
            }
        };
    });
    run_failure([](const auto &, auto &hooks) {
        hooks.on_stage = [](auto stage, const auto &, const auto &) {
            if (stage == apt_producer::PublicationStage::ecsv_published) {
                throw std::runtime_error("injected failure before receipt");
            }
        };
    });
    run_failure([](const auto &, auto &hooks) {
        hooks.on_stage = [](auto stage, const auto &staged_ecsv,
                            const auto &) {
            if (stage == apt_producer::PublicationStage::ecsv_published) {
                std::error_code error;
                std::filesystem::remove(staged_ecsv, error);
                if (error) {
                    throw std::runtime_error(
                        "failed to remove staged source in failpoint");
                }
                throw std::runtime_error(
                    "injected failure after staged source unlink");
            }
        };
    });

    run_failure([](const auto &, auto &hooks) {
        hooks.on_stage = [](auto stage, const auto &staged_ecsv,
                            const auto &) {
            if (stage == apt_producer::PublicationStage::ecsv_validated) {
                write_phase_b_file(staged_ecsv, "transport-tamper");
            }
        };
    });
    run_failure([](const auto &, auto &hooks) {
        hooks.on_stage = [](auto stage, const auto &, const auto &receipt) {
            if (stage == apt_producer::PublicationStage::receipt_staged) {
                write_phase_b_file(receipt, "invalid-receipt\n");
            }
        };
    });
    run_failure([](const auto &, auto &hooks) {
        hooks.on_stage = [](auto stage, const auto &staged_ecsv,
                            const auto &) {
            if (stage ==
                apt_producer::PublicationStage::before_receipt_publish) {
                std::filesystem::permissions(
                    staged_ecsv,
                    std::filesystem::perms::owner_all,
                    std::filesystem::perm_options::add);
                write_phase_b_file(staged_ecsv,
                                   "tampered-before-receipt-publication");
            }
        };
    });
    run_failure([](const auto &, auto &hooks) {
        hooks.on_stage = [](auto stage, const auto &,
                            const auto &staged_receipt) {
            if (stage ==
                apt_producer::PublicationStage::before_receipt_publish) {
                std::filesystem::permissions(
                    staged_receipt,
                    std::filesystem::perms::owner_all,
                    std::filesystem::perm_options::add);
                write_phase_b_file(staged_receipt,
                                   "tampered-receipt-before-publication\n");
            }
        };
    });

    for (const auto failing_stage : {
             apt_producer::PublicationStage::ecsv_staged,
             apt_producer::PublicationStage::ecsv_validated,
             apt_producer::PublicationStage::receipt_staged,
             apt_producer::PublicationStage::receipt_validated,
             apt_producer::PublicationStage::before_ecsv_publish,
             apt_producer::PublicationStage::ecsv_published,
             apt_producer::PublicationStage::before_receipt_publish}) {
        run_failure([failing_stage](const auto &, auto &hooks) {
            hooks.on_stage = [failing_stage](auto stage, const auto &,
                                             const auto &) {
                if (stage == failing_stage) {
                    throw std::runtime_error("injected publication failure");
                }
            };
        });
    }

    for (const auto [artifact_exists, receipt_exists] :
         {std::pair{true, false}, std::pair{false, true},
          std::pair{true, true}}) {
        PhaseBTemporaryDirectory preexisting;
        const auto output = preexisting.path / "beammap.ecsv";
        const auto receipt =
            std::filesystem::path(output.string() + ".sha256");
        if (artifact_exists) {
            write_phase_b_file(output, "preexisting-artifact-sentinel");
        }
        if (receipt_exists) {
            write_phase_b_file(receipt, "preexisting-receipt-sentinel");
        }
        auto fixture = make_phase_b_legacy_fixture();
        const auto document = apt_producer::make_canonical_document(
            fixture.calib, fixture.flag2, phase_b_context());
        EXPECT_THROW(apt_producer::publish_canonical_apt(document, output),
                     std::runtime_error);
        if (artifact_exists) {
            EXPECT_EQ(read_phase_b_file(output),
                      "preexisting-artifact-sentinel");
        } else {
            EXPECT_FALSE(std::filesystem::exists(output));
        }
        if (receipt_exists) {
            EXPECT_EQ(read_phase_b_file(receipt),
                      "preexisting-receipt-sentinel");
        } else {
            EXPECT_FALSE(std::filesystem::exists(receipt));
        }
    }

    PhaseBTemporaryDirectory receipt_race;
    const auto raced_output = receipt_race.path / "beammap.ecsv";
    const auto raced_receipt =
        std::filesystem::path(raced_output.string() + ".sha256");
    auto fixture = make_phase_b_legacy_fixture();
    const auto document = apt_producer::make_canonical_document(
        fixture.calib, fixture.flag2, phase_b_context());
    apt_producer::CanonicalAptPublicationHooks hooks;
    hooks.on_stage = [&](auto stage, const auto &, const auto &) {
        if (stage == apt_producer::PublicationStage::before_receipt_publish) {
            write_phase_b_file(raced_receipt, "receipt-race-sentinel");
        }
    };
    EXPECT_THROW(
        apt_producer::publish_canonical_apt(document, raced_output, hooks),
        std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(raced_output));
    EXPECT_EQ(read_phase_b_file(raced_receipt), "receipt-race-sentinel");
}

}  // namespace
