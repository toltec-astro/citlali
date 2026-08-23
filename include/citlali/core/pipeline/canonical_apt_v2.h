#pragma once

#include <citlali/core/pipeline/canonical_apt_v1.h>
#include <citlali/core/utils/sha256.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline::canonical_apt_v2 {

namespace v1 = citlali::pipeline::canonical_apt_v1;

inline constexpr std::string_view contract_authority_v2 =
    "citlali-canonical-apt-contract-v2";
inline constexpr std::string_view baseline_bundle_schema_v2 =
    "citlali-canonical-baseline-apt-bundle-v2";
inline constexpr std::string_view matched_bundle_schema_v2 =
    "citlali-canonical-observation-apt-bundle-v2";
inline constexpr std::string_view baseline_apt_schema_v2 =
    "citlali-canonical-baseline-apt-v2";
inline constexpr std::string_view matched_apt_schema_v2 =
    "citlali-canonical-matched-apt-v2";
inline constexpr std::string_view target_manifest_schema_v2 =
    "citlali-observation-target-manifest-v2";
inline constexpr std::string_view field_table_schema_v2 =
    "citlali-canonical-apt-fields-v2";
inline constexpr std::string_view source_table_schema_v2 =
    "citlali-canonical-apt-sources-v2";
inline constexpr std::string_view relation_table_schema_v2 =
    "citlali-canonical-apt-relation-v2";
inline constexpr std::string_view exception_table_schema_v2 =
    "citlali-canonical-apt-exceptions-v2";
inline constexpr std::string_view manifest_schema_v2 =
    "citlali-canonical-apt-root-manifest-v2";
inline constexpr std::string_view receipt_schema_v2 =
    "citlali-canonical-apt-root-receipt-v2";
inline constexpr std::string_view semantic_scope_v2 =
    "citlali-canonical-apt-component-semantic-sha256-v2";
inline constexpr std::string_view envelope_scope_v2 =
    "citlali-canonical-apt-component-envelope-sha256-v2";
inline constexpr std::string_view transport_scope_v2 =
    "citlali-canonical-apt-component-byte-transport-sha256-v2";
inline constexpr std::string_view bundle_transport_scope_v2 =
    "citlali-canonical-apt-root-manifest-byte-sha256-v2";
inline constexpr std::string_view baseline_bundle_semantic_scope_v2 =
    "citlali-canonical-baseline-apt-bundle-semantic-sha256-v2";
inline constexpr std::string_view baseline_bundle_envelope_scope_v2 =
    "citlali-canonical-baseline-apt-bundle-envelope-sha256-v2";
inline constexpr std::string_view matched_bundle_semantic_scope_v2 =
    "citlali-canonical-observation-apt-bundle-semantic-sha256-v2";
inline constexpr std::string_view matched_bundle_envelope_scope_v2 =
    "citlali-canonical-observation-apt-bundle-envelope-sha256-v2";
inline constexpr std::string_view target_semantic_scope_v2 =
    "citlali-observation-target-manifest-semantic-sha256-v2";
inline constexpr std::string_view target_envelope_scope_v2 =
    "citlali-observation-target-manifest-envelope-sha256-v2";
inline constexpr std::uint64_t maximum_portable_bundle_bytes_v2 =
    20ULL * 1024ULL * 1024ULL;

using ContractError = v1::ContractError;
using NullValue = v1::NullValue;
using Value = v1::Value;
using ValueType = v1::ValueType;
using NonFinitePolicy = v1::NonFinitePolicy;
using ObservationIdentity = v1::ObservationIdentity;

// Matched-v2 output uses one typed-null representation for every missing
// copied baseline value. Baseline-v2 may still carry legacy nan-token values,
// so normalize those values at the copy boundary instead of allowing the
// baseline representation to leak into the matched artifact.
inline Value copied_seed_value_or_null(const Value &value) {
    if (std::holds_alternative<NullValue>(value)) {
        return NullValue{};
    }
    if (const auto number = std::get_if<double>(&value);
        number != nullptr && std::isnan(*number)) {
        return NullValue{};
    }
    return value;
}

enum class BundleKind { baseline, matched };

inline std::string_view bundle_kind_token(BundleKind kind) {
    return kind == BundleKind::baseline ? "baseline" : "matched";
}

inline BundleKind parse_bundle_kind(std::string_view token) {
    if (token == "baseline") return BundleKind::baseline;
    if (token == "matched") return BundleKind::matched;
    throw ContractError("unsupported canonical APT v2 bundle kind: " +
                        std::string(token));
}

inline std::string_view product_kind_token(BundleKind kind) {
    return kind == BundleKind::baseline ? "beammap-baseline"
                                        : "observation-matched";
}

inline BundleKind parse_product_kind(std::string_view token) {
    if (token == "beammap-baseline") return BundleKind::baseline;
    if (token == "observation-matched") return BundleKind::matched;
    throw ContractError("unsupported canonical APT v2 product kind: " +
                        std::string(token));
}

enum class SourceRole { raw, kmp };

inline std::string_view source_role_token(SourceRole role) {
    return role == SourceRole::raw ? "raw" : "kmp";
}

inline SourceRole parse_source_role(std::string_view token) {
    if (token == "raw") return SourceRole::raw;
    if (token == "kmp") return SourceRole::kmp;
    throw ContractError("unsupported canonical APT v2 source role: " +
                        std::string(token));
}

enum class FieldOperation {
    preserve_structural,
    preserve_target,
    copy_seed_or_null,
    derive_declared,
    override_declared,
};

inline std::string_view field_operation_token(FieldOperation operation) {
    switch (operation) {
    case FieldOperation::preserve_structural:
        return "preserve-structural";
    case FieldOperation::preserve_target:
        return "preserve-target";
    case FieldOperation::copy_seed_or_null:
        return "copy-seed-or-null";
    case FieldOperation::derive_declared:
        return "derive-declared";
    case FieldOperation::override_declared:
        return "override-declared";
    }
    throw ContractError("unsupported canonical APT v2 field operation");
}

inline FieldOperation parse_field_operation(std::string_view token) {
    if (token == "preserve-structural") {
        return FieldOperation::preserve_structural;
    }
    if (token == "preserve-target") {
        return FieldOperation::preserve_target;
    }
    if (token == "copy-seed-or-null") {
        return FieldOperation::copy_seed_or_null;
    }
    if (token == "derive-declared") {
        return FieldOperation::derive_declared;
    }
    if (token == "override-declared") {
        return FieldOperation::override_declared;
    }
    throw ContractError("unsupported canonical APT v2 field operation: " +
                        std::string(token));
}

enum class RelationDisposition { matched, unmatched, ambiguous };

inline std::string_view relation_disposition_token(
    RelationDisposition disposition) {
    switch (disposition) {
    case RelationDisposition::matched: return "matched";
    case RelationDisposition::unmatched: return "unmatched";
    case RelationDisposition::ambiguous: return "ambiguous";
    }
    throw ContractError("unsupported canonical APT v2 relation disposition");
}

inline RelationDisposition parse_relation_disposition(
    std::string_view token) {
    if (token == "matched") return RelationDisposition::matched;
    if (token == "unmatched") return RelationDisposition::unmatched;
    if (token == "ambiguous") return RelationDisposition::ambiguous;
    throw ContractError("unsupported canonical APT v2 relation disposition: " +
                        std::string(token));
}

enum class ExceptionKind {
    field_deviation,
    ambiguity_candidate,
    seed_disposition,
};

inline std::string_view exception_kind_token(ExceptionKind kind) {
    switch (kind) {
    case ExceptionKind::field_deviation: return "field-deviation";
    case ExceptionKind::ambiguity_candidate: return "ambiguity-candidate";
    case ExceptionKind::seed_disposition:
        return "seed-disposition";
    }
    throw ContractError("unsupported canonical APT v2 exception kind");
}

inline ExceptionKind parse_exception_kind(std::string_view token) {
    if (token == "field-deviation") return ExceptionKind::field_deviation;
    if (token == "ambiguity-candidate") {
        return ExceptionKind::ambiguity_candidate;
    }
    if (token == "seed-disposition") {
        return ExceptionKind::seed_disposition;
    }
    throw ContractError("unsupported canonical APT v2 exception kind: " +
                        std::string(token));
}

struct IssuanceContext {
    std::string occurrence;
    std::string event_reference;
    std::string producer;
    std::string software_revision;
    std::string configuration_reference;
    std::string event_time_utc;

    friend bool operator==(const IssuanceContext &,
                           const IssuanceContext &) = default;
};

struct ComponentIdentity {
    std::string schema;
    std::string occurrence;
    std::string semantic_sha256;
    std::string envelope_sha256;

    friend bool operator==(const ComponentIdentity &,
                           const ComponentIdentity &) = default;
};

struct ScopedRowReference {
    ComponentIdentity artifact;
    std::int64_t local_uid = 0;

    friend bool operator==(const ScopedRowReference &,
                           const ScopedRowReference &) = default;
};

struct SourceRecord {
    std::int64_t source_uid = 0;
    SourceRole role = SourceRole::raw;
    std::string content_sha256;
    std::uint64_t byte_count = 0;
    ObservationIdentity header_observation;
    std::int64_t network = 0;
    std::string interface_name;
    std::int64_t channel_count = 0;

    friend bool operator==(const SourceRecord &, const SourceRecord &) =
        default;
};

struct FieldRule {
    std::int64_t field_uid = 0;
    std::string name;
    ValueType datatype = ValueType::float64;
    std::string unit;
    bool nullable = false;
    std::string authority;
    std::optional<std::string> authority_reference;
    FieldOperation operation = FieldOperation::preserve_target;
    std::optional<std::string> source_field;
    std::string missing_policy;
    std::string identity_role{"nonidentity"};
    std::string description;

    friend bool operator==(const FieldRule &, const FieldRule &) = default;
};

struct TargetRow {
    std::int64_t uid = 0;
    std::int64_t input_uid = 0;
    std::int64_t raw_source_uid = 0;
    std::int64_t kmp_source_uid = 0;
    std::int64_t kmp_row_index = 0;
    std::uint64_t source_rank = 0;
    std::uint64_t application_rank = 0;
    double tone_frequency_hz = 0.0;
    std::int64_t array = 0;
    std::int64_t network = 0;
    std::int64_t channel = 0;
    std::map<std::string, Value> fields;

    friend bool operator==(const TargetRow &, const TargetRow &) = default;
};

struct TargetManifest {
    IssuanceContext issuance;
    ObservationIdentity observation;
    std::vector<SourceRecord> sources;
    std::vector<TargetRow> rows;

    friend bool operator==(const TargetManifest &,
                           const TargetManifest &) = default;
};

struct AptRow {
    std::int64_t uid = 0;
    std::uint64_t presentation_rank = 0;
    double tone_frequency_hz = 0.0;
    std::int64_t array = 0;
    std::int64_t network = 0;
    std::int64_t channel = 0;
    std::map<std::string, Value> fields;

    friend bool operator==(const AptRow &, const AptRow &) = default;
};

struct AptTable {
    BundleKind kind = BundleKind::baseline;
    IssuanceContext issuance;
    ObservationIdentity observation;
    std::vector<FieldRule> field_rules;
    std::vector<AptRow> rows;

    friend bool operator==(const AptTable &, const AptTable &) = default;
};

struct MatcherEvidence {
    std::string matcher_run_occurrence;
    std::string implementation_sha256;
    std::string configuration_sha256;
    std::string method;
    std::string backend;

    friend bool operator==(const MatcherEvidence &,
                           const MatcherEvidence &) = default;
};

enum class NetworkEvidenceStatus {
    matched_capable,
    missing_baseline_network,
    no_good_seed,
};

inline std::string_view network_evidence_status_token(
    NetworkEvidenceStatus status) {
    switch (status) {
    case NetworkEvidenceStatus::matched_capable: return "matched-capable";
    case NetworkEvidenceStatus::missing_baseline_network:
        return "missing-baseline-network";
    case NetworkEvidenceStatus::no_good_seed: return "no-good-seed";
    }
    throw ContractError("unsupported canonical APT v2 network status");
}

inline NetworkEvidenceStatus parse_network_evidence_status(
    std::string_view token) {
    if (token == "matched-capable") {
        return NetworkEvidenceStatus::matched_capable;
    }
    if (token == "missing-baseline-network") {
        return NetworkEvidenceStatus::missing_baseline_network;
    }
    if (token == "no-good-seed") {
        return NetworkEvidenceStatus::no_good_seed;
    }
    throw ContractError("unsupported canonical APT v2 network status: " +
                        std::string(token));
}

struct NetworkEvidence {
    std::int64_t evidence_uid = 0;
    std::int64_t network = 0;
    NetworkEvidenceStatus status = NetworkEvidenceStatus::matched_capable;
    std::optional<double> frequency_shift_hz;
    std::optional<double> gate_hz;
    std::optional<double> quality_factor;

    friend bool operator==(const NetworkEvidence &,
                           const NetworkEvidence &) = default;
};

struct RelationRecord {
    std::int64_t relation_uid = 0;
    std::int64_t output_uid = 0;
    ScopedRowReference target;
    std::int64_t target_input_uid = 0;
    std::int64_t target_raw_source_uid = 0;
    std::int64_t target_kmp_source_uid = 0;
    std::int64_t target_kmp_row_index = 0;
    std::uint64_t source_rank = 0;
    std::uint64_t application_rank = 0;
    std::uint64_t presentation_rank = 0;
    RelationDisposition disposition = RelationDisposition::unmatched;
    std::optional<std::int64_t> selected_pair_uid;
    std::optional<ScopedRowReference> selected_seed;
    std::optional<double> separation_hz;
    std::optional<bool> is_good_match;
    std::int64_t network_evidence_uid = 0;
    std::string reason;

    friend bool operator==(const RelationRecord &,
                           const RelationRecord &) = default;
};

struct RelationTable {
    IssuanceContext issuance;
    ObservationIdentity observation;
    ComponentIdentity target_parent;
    IssuanceContext target_issuance;
    ComponentIdentity baseline_parent;
    MatcherEvidence matcher;
    std::vector<NetworkEvidence> network_evidence;
    std::vector<RelationRecord> rows;

    friend bool operator==(const RelationTable &,
                           const RelationTable &) = default;
};

struct ExceptionRecord {
    std::int64_t exception_uid = 0;
    ExceptionKind kind = ExceptionKind::field_deviation;
    std::optional<std::int64_t> target_uid;
    std::optional<std::string> field_name;
    std::optional<FieldOperation> operation;
    std::optional<ValueType> value_type;
    std::optional<Value> before;
    std::optional<Value> after;
    std::optional<ScopedRowReference> seed;
    std::optional<double> separation_hz;
    std::optional<bool> is_good_match;
    std::string reason;
    std::optional<std::string> authority_reference;

    friend bool operator==(const ExceptionRecord &,
                           const ExceptionRecord &) = default;
};

struct ComponentDescriptor {
    std::string role;
    std::string relative_path;
    std::string schema;
    std::string semantic_sha256;
    std::string envelope_sha256;
    std::string transport_sha256;
    std::uint64_t byte_count = 0;
    std::uint64_t row_count = 0;

    friend bool operator==(const ComponentDescriptor &,
                           const ComponentDescriptor &) = default;
};

struct BundleManifest {
    std::string schema{manifest_schema_v2};
    BundleKind kind = BundleKind::baseline;
    std::string profile;
    std::string issuance_class{"fresh"};
    IssuanceContext issuance;
    ObservationIdentity observation;
    std::optional<ComponentIdentity> baseline_parent;
    std::optional<ComponentIdentity> target_parent;
    std::string target_manifest_sha256;
    std::string relation_sha256;
    std::string field_rules_sha256;
    std::string exceptions_sha256;
    std::vector<ComponentDescriptor> components;

    friend bool operator==(const BundleManifest &,
                           const BundleManifest &) = default;
};

struct ComponentDigests {
    std::string semantic_sha256;
    std::string envelope_sha256;
    std::string transport_sha256;
    std::uint64_t byte_count = 0;
};

struct SerializedComponent {
    std::string role;
    std::string schema;
    std::string bytes;
    ComponentDigests digests;
    std::uint64_t row_count = 0;
};

inline bool is_sha256_reference(std::string_view value) {
    return v1::is_sha256_reference(value);
}

inline void require_text(std::string_view value, std::string_view label) {
    if (!v1::detail::canonical_text(value, false)) {
        throw ContractError("canonical APT v2 " + std::string(label) +
                            " is empty or invalid UTF-8 text");
    }
}

inline void require_sha256(std::string_view value, std::string_view label) {
    if (!is_sha256_reference(value)) {
        throw ContractError("canonical APT v2 " + std::string(label) +
                            " is not a SHA-256 reference");
    }
}

inline void validate(const IssuanceContext &issuance) {
    require_text(issuance.occurrence, "occurrence");
    require_text(issuance.event_reference, "event reference");
    require_text(issuance.producer, "producer");
    require_text(issuance.software_revision, "software revision");
    require_text(issuance.configuration_reference, "configuration reference");
    if (!v1::detail::is_utc_timestamp(issuance.event_time_utc)) {
        throw ContractError("canonical APT v2 event time is not exact UTC");
    }
}

inline void validate(const ComponentIdentity &identity) {
    require_text(identity.schema, "artifact schema");
    require_text(identity.occurrence, "artifact occurrence");
    require_sha256(identity.semantic_sha256, "artifact semantic digest");
    require_sha256(identity.envelope_sha256, "artifact envelope digest");
}

inline void validate(const ScopedRowReference &reference) {
    validate(reference.artifact);
    if (reference.local_uid < 0) {
        throw ContractError("canonical APT v2 local UID is negative");
    }
}

inline void validate_observation(const ObservationIdentity &observation) {
    if (observation.observation < 0 || observation.subobservation < 0 ||
        observation.scan < 0) {
        throw ContractError("canonical APT v2 observation tuple is negative");
    }
}

inline void validate(const SourceRecord &source) {
    if (source.source_uid < 0 || source.network < 0 || source.network > 12 ||
        source.channel_count <= 0 || source.byte_count == 0) {
        throw ContractError("canonical APT v2 source identity/count is invalid");
    }
    require_sha256(source.content_sha256, "source digest");
    validate_observation(source.header_observation);
    const auto expected = "toltec" + std::to_string(source.network);
    if (source.interface_name != expected) {
        throw ContractError("canonical APT v2 source interface/network mismatch");
    }
}

inline void validate(const FieldRule &field) {
    if (field.field_uid < 0) {
        throw ContractError("canonical APT v2 field UID is negative");
    }
    require_text(field.name, "field name");
    require_text(field.unit, "field unit");
    require_text(field.authority, "field value authority");
    if (field.authority_reference) {
        require_text(*field.authority_reference, "field authority reference");
    }
    if (field.source_field) {
        require_text(*field.source_field, "field source field");
    }
    require_text(field.missing_policy, "field missing policy");
    require_text(field.description, "field description");
    if (field.identity_role != "artifact-local" &&
        field.identity_role != "nonidentity") {
        throw ContractError("canonical APT v2 field identity role is invalid");
    }
}

inline std::string envelope_preimage(std::string_view semantic_sha256,
                                     const IssuanceContext &issuance,
                                     std::string_view scope);

inline void validate_rank_permutation(const std::vector<std::uint64_t> &ranks,
                                      std::string_view label);

inline std::int64_t array_for_network(std::int64_t network) {
    if (network >= 0 && network <= 6) return 0;
    if (network >= 7 && network <= 10) return 1;
    if (network >= 11 && network <= 12) return 2;
    throw ContractError("canonical APT v2 network is outside {0..12}");
}

inline void validate(const TargetManifest &target) {
    validate(target.issuance);
    validate_observation(target.observation);
    if (target.rows.empty() || target.sources.empty()) {
        throw ContractError("canonical APT v2 target manifest is empty");
    }
    std::map<std::int64_t, const SourceRecord *> sources;
    std::map<std::pair<SourceRole, std::int64_t>, const SourceRecord *>
        sources_by_role_network;
    for (const auto &source : target.sources) {
        validate(source);
        if ((source.role == SourceRole::raw &&
             source.header_observation != target.observation) ||
            !sources.emplace(source.source_uid, &source).second ||
            !sources_by_role_network
                 .emplace(std::pair{source.role, source.network}, &source)
                 .second) {
            throw ContractError(
                "canonical APT v2 target source inventory is inconsistent");
        }
    }
    for (const auto &[key, source] : sources_by_role_network) {
        if (key.first != SourceRole::raw) continue;
        const auto kmp = sources_by_role_network.find(
            {SourceRole::kmp, key.second});
        if (kmp == sources_by_role_network.end() ||
            kmp->second->interface_name != source->interface_name ||
            kmp->second->channel_count != source->channel_count) {
            throw ContractError(
                "canonical APT v2 target raw/KMP source pairing is invalid");
        }
    }
    std::set<std::int64_t> uids;
    std::map<std::int64_t, std::int64_t> input_uid_by_network;
    std::map<std::int64_t, std::int64_t> network_by_input_uid;
    std::set<std::pair<std::int64_t, std::int64_t>> raw_channels;
    std::vector<std::uint64_t> source_ranks;
    std::vector<std::uint64_t> application_ranks;
    std::optional<bool> has_flag;
    for (const auto &row : target.rows) {
        const auto raw = sources.find(row.raw_source_uid);
        const auto kmp = sources.find(row.kmp_source_uid);
        if (row.uid < 0 || row.input_uid < 0 || row.kmp_row_index < 0 ||
            row.channel < 0 || !std::isfinite(row.tone_frequency_hz) ||
            row.array != array_for_network(row.network) ||
            !uids.insert(row.uid).second ||
            !raw_channels.emplace(row.network, row.channel).second ||
            raw == sources.end() || kmp == sources.end() ||
            raw->second->role != SourceRole::raw ||
            kmp->second->role != SourceRole::kmp ||
            raw->second->network != row.network ||
            kmp->second->network != row.network ||
            row.channel >= raw->second->channel_count ||
            row.kmp_row_index != row.channel ||
            row.kmp_row_index >= kmp->second->channel_count) {
            throw ContractError(
                "canonical APT v2 target row/source relation is invalid");
        }
        const auto [network_input, inserted_network] =
            input_uid_by_network.emplace(row.network, row.input_uid);
        const auto [input_network, inserted_input] =
            network_by_input_uid.emplace(row.input_uid, row.network);
        if ((!inserted_network && network_input->second != row.input_uid) ||
            (!inserted_input && input_network->second != row.network)) {
            throw ContractError(
                "canonical APT v2 target input UID is not network scoped");
        }
        const bool row_has_flag = row.fields.contains("kids_flag");
        if (!has_flag) has_flag = row_has_flag;
        if (*has_flag != row_has_flag ||
            row.fields.size() != (row_has_flag ? 4U : 3U)) {
            throw ContractError(
                "canonical APT v2 target KMP field set is not closed");
        }
        for (const auto name : {"kids_fr", "kids_f_out", "kids_Qr"}) {
            const auto value = row.fields.find(name);
            if (value == row.fields.end() ||
                !std::holds_alternative<double>(value->second) ||
                !std::isfinite(std::get<double>(value->second))) {
                throw ContractError(
                    "canonical APT v2 target KMP float is absent or invalid");
            }
        }
        if (row_has_flag &&
            !std::holds_alternative<std::int64_t>(row.fields.at("kids_flag"))) {
            throw ContractError(
                "canonical APT v2 target kids_flag is not exact int64");
        }
        if (std::bit_cast<std::uint64_t>(row.tone_frequency_hz) !=
            std::bit_cast<std::uint64_t>(
                std::get<double>(row.fields.at("kids_f_out")))) {
            throw ContractError(
                "canonical APT v2 target tone frequency disagrees with kids_f_out");
        }
        source_ranks.push_back(row.source_rank);
        application_ranks.push_back(row.application_rank);
    }
    validate_rank_permutation(source_ranks, "target source ranks");
    validate_rank_permutation(application_ranks, "target application ranks");
    std::uint64_t expected_rows = 0;
    for (const auto &[key, source] : sources_by_role_network) {
        if (key.first == SourceRole::raw) {
            expected_rows += static_cast<std::uint64_t>(source->channel_count);
        }
    }
    if (expected_rows != target.rows.size()) {
        throw ContractError(
            "canonical APT v2 target raw sources do not cover every row");
    }
}

inline bool value_matches(const Value &value, const FieldRule &field) {
    if (std::holds_alternative<NullValue>(value)) return field.nullable;
    if (!v1::detail::value_matches_type(value, field.datatype)) return false;
    if (const auto number = std::get_if<double>(&value)) {
        if (std::isinf(*number) ||
            (std::isnan(*number) && field.missing_policy != "nan-token")) {
            return false;
        }
    }
    return true;
}

inline void validate_rank_permutation(const std::vector<std::uint64_t> &ranks,
                                      std::string_view label) {
    auto ordered = ranks;
    std::sort(ordered.begin(), ordered.end());
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        if (ordered[index] != index) {
            throw ContractError("canonical APT v2 " + std::string(label) +
                                " is not a complete permutation");
        }
    }
}

inline void validate(const AptTable &table) {
    validate(table.issuance);
    validate_observation(table.observation);
    if (table.rows.empty() || table.field_rules.empty()) {
        throw ContractError("canonical APT v2 APT table is empty");
    }
    const std::set<std::string> structural_names{
        "uid", "tone_freq", "array", "nw", "kids_tone"};
    std::map<std::string, const FieldRule *> fields;
    std::map<std::string, const FieldRule *> dynamic_fields;
    std::set<std::int64_t> field_uids;
    for (const auto &field : table.field_rules) {
        validate(field);
        if (!fields.emplace(field.name, &field).second ||
            !field_uids.insert(field.field_uid).second) {
            throw ContractError("canonical APT v2 field rule is duplicate");
        }
        if (!structural_names.contains(field.name)) {
            dynamic_fields.emplace(field.name, &field);
        } else if (field.operation != FieldOperation::preserve_structural ||
                   field.identity_role !=
                       (field.name == "uid" ? "artifact-local" : "nonidentity")) {
            throw ContractError(
                "canonical APT v2 structural field rule is invalid");
        }
    }
    if (!std::all_of(structural_names.begin(), structural_names.end(),
                     [&](const auto &name) { return fields.contains(name); })) {
        throw ContractError("canonical APT v2 structural field rules are incomplete");
    }
    const auto require_structural = [&](std::string_view name,
                                        ValueType datatype,
                                        std::string_view unit) {
        const auto &field = *fields.at(std::string(name));
        if (field.datatype != datatype || field.unit != unit ||
            field.nullable || field.source_field ||
            field.missing_policy != "reject") {
            throw ContractError(
                "canonical APT v2 structural field declaration is not exact");
        }
    };
    require_structural("uid", ValueType::int64, "N/A");
    require_structural("tone_freq", ValueType::float64, "Hz");
    require_structural("array", ValueType::int64, "N/A");
    require_structural("nw", ValueType::int64, "N/A");
    require_structural("kids_tone", ValueType::int64, "N/A");
    std::set<std::int64_t> row_uids;
    std::vector<std::uint64_t> presentation_ranks;
    for (const auto &row : table.rows) {
        if (row.uid < 0 || row.network < 0 || row.network > 12 ||
            row.channel < 0 ||
            !std::isfinite(row.tone_frequency_hz) ||
            !row_uids.insert(row.uid).second ||
            row.fields.size() != dynamic_fields.size()) {
            throw ContractError("canonical APT v2 row structure is invalid");
        }
        for (const auto &[name, field] : dynamic_fields) {
            const auto value = row.fields.find(name);
            if (value == row.fields.end() || !value_matches(value->second, *field)) {
                throw ContractError("canonical APT v2 row field is missing or untyped: " + name);
            }
        }
        presentation_ranks.push_back(row.presentation_rank);
    }
    validate_rank_permutation(presentation_ranks, "APT presentation ranks");
}

inline void validate(const RelationTable &table) {
    validate(table.issuance);
    validate_observation(table.observation);
    validate(table.target_parent);
    validate(table.target_issuance);
    validate(table.baseline_parent);
    const auto expected_target_envelope = "sha256:" + citlali::utils::sha256(
        envelope_preimage(table.target_parent.semantic_sha256,
                          table.target_issuance, target_envelope_scope_v2));
    if (table.target_parent.schema != target_manifest_schema_v2 ||
        table.target_parent.occurrence != table.target_issuance.occurrence ||
        table.target_parent.envelope_sha256 != expected_target_envelope ||
        table.baseline_parent.schema != baseline_bundle_schema_v2) {
        throw ContractError(
            "canonical APT v2 relation parent identity is invalid");
    }
    require_text(table.matcher.matcher_run_occurrence, "matcher occurrence");
    require_sha256(table.matcher.implementation_sha256,
                   "matcher implementation digest");
    require_sha256(table.matcher.configuration_sha256,
                   "matcher configuration digest");
    require_text(table.matcher.method, "matcher method");
    require_text(table.matcher.backend, "matcher backend");
    std::set<std::int64_t> evidence_uids;
    std::set<std::int64_t> networks;
    for (const auto &evidence : table.network_evidence) {
        const bool has_values = evidence.frequency_shift_hz.has_value() &&
            evidence.gate_hz.has_value() &&
            evidence.quality_factor.has_value();
        if (evidence.evidence_uid < 0 || evidence.network < 0 ||
            evidence.network > 12 ||
            ((evidence.status == NetworkEvidenceStatus::matched_capable) !=
             has_values) ||
            (has_values &&
             (!std::isfinite(*evidence.frequency_shift_hz) ||
              !std::isfinite(*evidence.gate_hz) ||
              *evidence.gate_hz < 0.0 ||
              !std::isfinite(*evidence.quality_factor))) ||
            !evidence_uids.insert(evidence.evidence_uid).second ||
            !networks.insert(evidence.network).second) {
            throw ContractError("canonical APT v2 network evidence is invalid");
        }
    }
    std::set<std::int64_t> relation_uids;
    std::set<std::int64_t> output_uids;
    std::set<ScopedRowReference, bool (*)(const ScopedRowReference &,
                                         const ScopedRowReference &)>
        targets([](const auto &lhs, const auto &rhs) {
            return std::tie(lhs.artifact.schema, lhs.artifact.occurrence,
                            lhs.artifact.semantic_sha256,
                            lhs.artifact.envelope_sha256, lhs.local_uid) <
                std::tie(rhs.artifact.schema, rhs.artifact.occurrence,
                         rhs.artifact.semantic_sha256,
                         rhs.artifact.envelope_sha256, rhs.local_uid);
        });
    std::set<std::tuple<std::string, std::string, std::int64_t>>
        selected_seeds;
    std::set<std::int64_t> selected_pair_uids;
    std::vector<std::uint64_t> source_ranks, application_ranks,
        presentation_ranks;
    for (const auto &row : table.rows) {
        validate(row.target);
        if (row.relation_uid < 0 || row.output_uid < 0 ||
            row.target_input_uid < 0 || row.target_raw_source_uid < 0 ||
            row.target_kmp_source_uid < 0 || row.target_kmp_row_index < 0 ||
            !relation_uids.insert(row.relation_uid).second ||
            !output_uids.insert(row.output_uid).second ||
            !targets.insert(row.target).second ||
            !evidence_uids.contains(row.network_evidence_uid)) {
            throw ContractError("canonical APT v2 relation row is duplicate or invalid");
        }
        if (row.target.artifact != table.target_parent) {
            throw ContractError(
                "canonical APT v2 relation target parent is foreign");
        }
        require_text(row.reason, "relation reason");
        const bool complete_match = row.selected_pair_uid.has_value() &&
            row.selected_seed.has_value() && row.separation_hz.has_value() &&
            row.is_good_match.has_value();
        if ((row.disposition == RelationDisposition::matched) != complete_match ||
            (row.disposition != RelationDisposition::matched &&
             (row.selected_pair_uid || row.selected_seed || row.separation_hz ||
              row.is_good_match))) {
            throw ContractError("canonical APT v2 relation selection/null matrix is invalid");
        }
        if (row.selected_pair_uid &&
            (*row.selected_pair_uid < 0 ||
             !selected_pair_uids.insert(*row.selected_pair_uid).second)) {
            throw ContractError(
                "canonical APT v2 selected pair UID is negative or duplicate");
        }
        if (row.selected_seed) {
            validate(*row.selected_seed);
            if (row.selected_seed->artifact != table.baseline_parent) {
                throw ContractError(
                    "canonical APT v2 selected seed parent is foreign");
            }
            const auto key = std::make_tuple(
                row.selected_seed->artifact.semantic_sha256,
                row.selected_seed->artifact.occurrence,
                row.selected_seed->local_uid);
            if (!selected_seeds.insert(key).second) {
                throw ContractError(
                    "canonical APT v2 selected seed is not unique");
            }
        }
        if (row.separation_hz && !std::isfinite(*row.separation_hz)) {
            throw ContractError("canonical APT v2 separation is nonfinite");
        }
        source_ranks.push_back(row.source_rank);
        application_ranks.push_back(row.application_rank);
        presentation_ranks.push_back(row.presentation_rank);
    }
    validate_rank_permutation(source_ranks, "target source ranks");
    validate_rank_permutation(application_ranks, "target application ranks");
    validate_rank_permutation(presentation_ranks, "output presentation ranks");
}

inline void validate(const ExceptionRecord &exception) {
    if (exception.exception_uid < 0) {
        throw ContractError("canonical APT v2 exception UID is negative");
    }
    require_text(exception.reason, "exception reason");
    if (exception.authority_reference) {
        require_text(*exception.authority_reference,
                     "exception authority reference");
    }
    if (exception.target_uid && *exception.target_uid < 0) {
        throw ContractError("canonical APT v2 exception target UID is negative");
    }
    const bool has_field_values = exception.target_uid && exception.field_name &&
        exception.operation && exception.value_type && exception.before &&
        exception.after &&
        exception.authority_reference;
    const bool has_candidate = exception.target_uid && exception.seed &&
        exception.separation_hz && exception.is_good_match;
    switch (exception.kind) {
    case ExceptionKind::field_deviation:
        if (!has_field_values || exception.seed ||
            exception.separation_hz || exception.is_good_match) {
            throw ContractError(
                "canonical APT v2 field exception null matrix is invalid");
        }
        if (*exception.operation != FieldOperation::derive_declared &&
            *exception.operation != FieldOperation::override_declared) {
            throw ContractError(
                "canonical APT v2 field exception operation is unauthorized");
        }
        require_text(*exception.field_name, "exception field name");
        if (!v1::detail::value_matches_type(*exception.before,
                                            *exception.value_type) ||
            !v1::detail::value_matches_type(*exception.after,
                                            *exception.value_type)) {
            throw ContractError("canonical APT v2 exception values are untyped");
        }
        break;
    case ExceptionKind::ambiguity_candidate:
        if (!has_candidate || exception.field_name || exception.operation ||
            exception.value_type || exception.before || exception.after ||
            exception.authority_reference) {
            throw ContractError(
                "canonical APT v2 ambiguity exception null matrix is invalid");
        }
        validate(*exception.seed);
        if (!std::isfinite(*exception.separation_hz)) {
            throw ContractError(
                "canonical APT v2 ambiguity separation is nonfinite");
        }
        break;
    case ExceptionKind::seed_disposition:
        if (!exception.seed || exception.target_uid || exception.field_name ||
            exception.operation ||
            exception.value_type || exception.before || exception.after ||
            exception.separation_hz ||
            exception.is_good_match || exception.authority_reference) {
            throw ContractError(
                "canonical APT v2 seed exception null matrix is invalid");
        }
        validate(*exception.seed);
        break;
    }
}

inline bool valid_component_basename(std::string_view name,
                                     std::string_view transport_sha256,
                                     std::string_view role) {
    constexpr std::string_view prefix{"sha256:"};
    if (!is_sha256_reference(transport_sha256) || name.find('/') != name.npos ||
        name.find('\\') != name.npos || name == "." || name == ".." ||
        name.starts_with('.') || name.find("..") != name.npos) {
        return false;
    }
    const auto hex = transport_sha256.substr(prefix.size());
    const auto expected = role == "baseline-receipt"
        ? "sha256-" + std::string(hex) + ".baseline-receipt.txt"
        : "sha256-" + std::string(hex) + "." + std::string(role) + ".ecsv";
    return name == expected;
}

inline void validate(const ComponentDescriptor &component) {
    require_text(component.role, "component role");
    require_text(component.schema, "component schema");
    require_sha256(component.semantic_sha256, "component semantic digest");
    require_sha256(component.envelope_sha256, "component envelope digest");
    require_sha256(component.transport_sha256, "component transport digest");
    if (component.byte_count == 0 ||
        !valid_component_basename(component.relative_path,
                                  component.transport_sha256,
                                  component.role)) {
        throw ContractError("canonical APT v2 component path/count is invalid");
    }
}

inline std::set<std::string> required_roles(BundleKind kind) {
    if (kind == BundleKind::baseline) {
        return {"apt", "fields", "sources"};
    }
    return {"apt", "relation", "fields", "sources", "exceptions",
            "baseline-apt", "baseline-fields", "baseline-sources",
            "baseline-manifest", "baseline-receipt"};
}

inline void validate(const BundleManifest &manifest) {
    if (manifest.schema != manifest_schema_v2) {
        throw ContractError("canonical APT v2 manifest schema is invalid");
    }
    validate(manifest.issuance);
    validate_observation(manifest.observation);
    require_text(manifest.profile, "bundle profile");
    if (manifest.issuance_class != "fresh" &&
        manifest.issuance_class != "migration-only") {
        throw ContractError("canonical APT v2 issuance class is invalid");
    }
    if (manifest.kind == BundleKind::baseline &&
        (manifest.baseline_parent || manifest.target_parent)) {
        throw ContractError("baseline bundle cannot claim baseline/target parents");
    }
    if (manifest.kind == BundleKind::matched &&
        (!manifest.baseline_parent || !manifest.target_parent)) {
        throw ContractError("matched bundle requires baseline and target parents");
    }
    if (manifest.baseline_parent) validate(*manifest.baseline_parent);
    if (manifest.target_parent) validate(*manifest.target_parent);
    for (const auto *digest : {&manifest.target_manifest_sha256,
                               &manifest.relation_sha256,
                               &manifest.field_rules_sha256,
                               &manifest.exceptions_sha256}) {
        if (!digest->empty()) require_sha256(*digest, "logical component digest");
    }
    const bool has_all_logical = !manifest.target_manifest_sha256.empty() &&
        !manifest.relation_sha256.empty() &&
        !manifest.field_rules_sha256.empty() &&
        !manifest.exceptions_sha256.empty();
    const bool has_any_logical = !manifest.target_manifest_sha256.empty() ||
        !manifest.relation_sha256.empty() ||
        !manifest.field_rules_sha256.empty() ||
        !manifest.exceptions_sha256.empty();
    if ((manifest.kind == BundleKind::matched && !has_all_logical) ||
        (manifest.kind == BundleKind::baseline && has_any_logical)) {
        throw ContractError(
            "canonical APT v2 logical manifest digests are incomplete");
    }
    std::set<std::string> roles;
    std::set<std::string> paths;
    std::uint64_t total = 0;
    for (const auto &component : manifest.components) {
        validate(component);
        if (!roles.insert(component.role).second ||
            !paths.insert(component.relative_path).second ||
            total > maximum_portable_bundle_bytes_v2 - component.byte_count) {
            throw ContractError("canonical APT v2 component role/path/size is invalid");
        }
        total += component.byte_count;
    }
    if (roles != required_roles(manifest.kind)) {
        throw ContractError("canonical APT v2 manifest component inventory is incomplete");
    }
}

inline std::vector<FieldRule> canonical_kmp_field_rules_v2(
    bool include_kids_flag = true) {
    std::vector<FieldRule> fields{
        {0, "kids_fr", ValueType::float64, "Hz", false,
         "copied-declared", std::string("kids:model-params-v1"),
         FieldOperation::preserve_target, std::string("fr"), "reject",
         "nonidentity", "imported KIDs resonant frequency"},
        {1, "kids_f_out", ValueType::float64, "Hz", false,
         "copied-declared", std::string("kids:model-params-v1"),
         FieldOperation::preserve_target, std::string("f_out"), "reject",
         "nonidentity", "imported KIDs output tone frequency"},
        {2, "kids_Qr", ValueType::float64, "N/A", false,
         "copied-declared", std::string("kids:model-params-v1"),
         FieldOperation::preserve_target, std::string("Qr"), "reject",
         "nonidentity", "imported KIDs resonator Qr"},
    };
    if (include_kids_flag) {
        fields.push_back(
            {3, "kids_flag", ValueType::int64, "N/A", false,
             "copied-declared", std::string("kids:fit-report-v1"),
             FieldOperation::preserve_target, std::string("flag"), "reject",
             "nonidentity", "imported KIDs model-fit flag"});
    }
    return fields;
}

inline std::vector<FieldRule> canonical_structural_field_rules_v2() {
    return {
        {0, "uid", ValueType::int64, "N/A", false,
         "canonical-issuer", std::nullopt,
         FieldOperation::preserve_structural, std::nullopt, "reject",
         "artifact-local", "artifact-local output row key"},
        {1, "tone_freq", ValueType::float64, "Hz", false,
         "raw-readout", std::nullopt,
         FieldOperation::preserve_structural, std::nullopt, "reject",
         "nonidentity", "readout tone frequency"},
        {2, "array", ValueType::int64, "N/A", false,
         "network-map", std::nullopt,
         FieldOperation::preserve_structural, std::nullopt, "reject",
         "nonidentity", "TolTEC array enum"},
        {3, "nw", ValueType::int64, "N/A", false,
         "raw-manifest", std::nullopt,
         FieldOperation::preserve_structural, std::nullopt, "reject",
         "nonidentity", "raw network"},
        {4, "kids_tone", ValueType::int64, "N/A", false,
         "raw-manifest", std::nullopt,
         FieldOperation::preserve_structural, std::nullopt, "reject",
         "nonidentity", "raw channel"},
    };
}

inline bool is_authorized_kmp_field(std::string_view name) {
    return name == "kids_fr" || name == "kids_f_out" || name == "kids_Qr" ||
        name == "kids_flag";
}

inline std::string canonical_frame(std::string_view label,
                                   std::string_view type,
                                   std::string_view payload) {
    return v1::canonical_frame(label, type, payload);
}

inline std::string canonical_binary64(double value) {
    return v1::canonical_float64_payload(value);
}

inline void add_text(std::string &result, std::string_view label,
                     std::string_view value) {
    result += canonical_frame(label, "utf8", value);
}

inline void add_int64(std::string &result, std::string_view label,
                      std::int64_t value) {
    result += canonical_frame(label, "int64", std::to_string(value));
}

inline void add_uint64(std::string &result, std::string_view label,
                       std::uint64_t value) {
    result += canonical_frame(label, "uint64", std::to_string(value));
}

inline void add_bool(std::string &result, std::string_view label, bool value) {
    result += canonical_frame(label, "bool", value ? "true" : "false");
}

inline void add_float64(std::string &result, std::string_view label,
                        double value) {
    result += canonical_frame(label, "float64-ieee754",
                              canonical_binary64(value));
}

inline void add_identity(std::string &result, std::string_view prefix,
                         const ComponentIdentity &identity) {
    add_text(result, std::string(prefix) + ".schema", identity.schema);
    add_text(result, std::string(prefix) + ".occurrence", identity.occurrence);
    add_text(result, std::string(prefix) + ".semantic",
             identity.semantic_sha256);
    add_text(result, std::string(prefix) + ".envelope",
             identity.envelope_sha256);
}

inline std::string envelope_preimage(
    std::string_view semantic_sha256, const IssuanceContext &issuance,
    std::string_view scope = envelope_scope_v2) {
    std::string result;
    add_text(result, "scope", scope);
    add_text(result, "semantic", semantic_sha256);
    add_text(result, "occurrence", issuance.occurrence);
    add_text(result, "event", issuance.event_reference);
    add_text(result, "producer", issuance.producer);
    add_text(result, "software", issuance.software_revision);
    add_text(result, "configuration", issuance.configuration_reference);
    add_text(result, "event-time", issuance.event_time_utc);
    return result;
}

inline ComponentDigests make_component_digests(
    std::string_view semantic_preimage, const IssuanceContext &issuance,
    std::string_view bytes = {},
    std::string_view envelope_scope = envelope_scope_v2) {
    ComponentDigests result;
    result.semantic_sha256 =
        "sha256:" + citlali::utils::sha256(semantic_preimage);
    result.envelope_sha256 = "sha256:" + citlali::utils::sha256(
        envelope_preimage(result.semantic_sha256, issuance, envelope_scope));
    if (!bytes.empty()) {
        result.transport_sha256 =
            "sha256:" + citlali::utils::sha256(bytes);
        result.byte_count = static_cast<std::uint64_t>(bytes.size());
    }
    return result;
}

inline std::string target_semantic_preimage(TargetManifest target) {
    validate(target);
    std::sort(target.sources.begin(), target.sources.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.source_uid < rhs.source_uid;
              });
    std::sort(target.rows.begin(), target.rows.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.uid < rhs.uid;
              });
    std::string result;
    add_text(result, "scope", target_semantic_scope_v2);
    add_text(result, "schema", target_manifest_schema_v2);
    add_int64(result, "observation.obsnum", target.observation.observation);
    add_int64(result, "observation.subobsnum",
              target.observation.subobservation);
    add_int64(result, "observation.scannum", target.observation.scan);
    add_uint64(result, "source.count", target.sources.size());
    for (std::size_t index = 0; index < target.sources.size(); ++index) {
        const auto prefix = "source." + std::to_string(index);
        const auto &source = target.sources[index];
        add_int64(result, prefix + ".uid", source.source_uid);
        add_text(result, prefix + ".role", source_role_token(source.role));
        add_text(result, prefix + ".content", source.content_sha256);
        add_uint64(result, prefix + ".bytes", source.byte_count);
        add_int64(result, prefix + ".observation",
                  source.header_observation.observation);
        add_int64(result, prefix + ".subobservation",
                  source.header_observation.subobservation);
        add_int64(result, prefix + ".scan",
                  source.header_observation.scan);
        add_int64(result, prefix + ".network", source.network);
        add_text(result, prefix + ".interface", source.interface_name);
        add_int64(result, prefix + ".channels", source.channel_count);
    }
    add_uint64(result, "row.count", target.rows.size());
    for (std::size_t index = 0; index < target.rows.size(); ++index) {
        const auto prefix = "row." + std::to_string(index);
        const auto &row = target.rows[index];
        add_int64(result, prefix + ".uid", row.uid);
        add_int64(result, prefix + ".input", row.input_uid);
        add_int64(result, prefix + ".raw-source", row.raw_source_uid);
        add_int64(result, prefix + ".kmp-source", row.kmp_source_uid);
        add_int64(result, prefix + ".kmp-row", row.kmp_row_index);
        add_uint64(result, prefix + ".source-rank", row.source_rank);
        add_uint64(result, prefix + ".application-rank",
                   row.application_rank);
        add_float64(result, prefix + ".tone-frequency",
                    row.tone_frequency_hz);
        add_int64(result, prefix + ".array", row.array);
        add_int64(result, prefix + ".network", row.network);
        add_int64(result, prefix + ".channel", row.channel);
        add_uint64(result, prefix + ".field.count", row.fields.size());
        for (const auto &[name, value] : row.fields) {
            add_text(result, prefix + ".field.name", name);
            const auto type = name == "kids_flag" ? ValueType::int64
                                                   : ValueType::float64;
            v1::detail::add_value(result, prefix + ".field.value", value,
                                  type);
        }
    }
    return result;
}

inline ComponentIdentity target_identity(const TargetManifest &target) {
    const auto digests = make_component_digests(
        target_semantic_preimage(target), target.issuance, {},
        target_envelope_scope_v2);
    return {std::string(target_manifest_schema_v2), target.issuance.occurrence,
            digests.semantic_sha256, digests.envelope_sha256};
}

inline std::string content_addressed_basename(
    std::string_view transport_sha256, std::string_view role) {
    require_sha256(transport_sha256, "transport digest");
    require_text(role, "component role");
    return role == "baseline-receipt"
        ? "sha256-" + std::string(transport_sha256.substr(7)) +
              ".baseline-receipt.txt"
        : "sha256-" + std::string(transport_sha256.substr(7)) + "." +
              std::string(role) + ".ecsv";
}

}  // namespace citlali::pipeline::canonical_apt_v2
