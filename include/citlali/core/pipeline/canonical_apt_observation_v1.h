#pragma once

#include <citlali/core/pipeline/canonical_apt_ecsv.h>
#include <citlali/core/pipeline/canonical_artifact_publication.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace citlali::pipeline::canonical_apt_observation_v1 {

namespace baseline = canonical_apt_v1;

inline constexpr std::string_view framing_encoding_v1 =
    baseline::framing_encoding_v1;
inline constexpr std::string_view contract_authority_v1 = "citlali";
inline constexpr std::string_view baseline_value_issuer_v1 = "citlali";
inline constexpr std::string_view observation_value_issuer_v1 = "tolproj";

inline constexpr std::string_view baseline_descriptor_schema_v1 =
    "citlali-verified-beammap-baseline-descriptor-v1";
inline constexpr std::string_view target_manifest_schema_v1 =
    "citlali-observation-target-manifest-v1";
inline constexpr std::string_view relation_schema_v1 =
    "citlali-apt-match-dispositions-v1";
inline constexpr std::string_view matched_output_schema_v1 =
    "citlali-observation-matched-apt-v1";

inline constexpr std::string_view baseline_descriptor_scope_v1 =
    "citlali-verified-beammap-baseline-descriptor-sha256-v1";
inline constexpr std::string_view target_semantic_scope_v1 =
    "citlali-observation-target-manifest-semantic-sha256-v1";
inline constexpr std::string_view target_envelope_scope_v1 =
    "citlali-observation-target-manifest-envelope-sha256-v1";
inline constexpr std::string_view relation_semantic_scope_v1 =
    "citlali-apt-match-dispositions-semantic-sha256-v1";
inline constexpr std::string_view relation_envelope_scope_v1 =
    "citlali-apt-match-dispositions-envelope-sha256-v1";
inline constexpr std::string_view matched_output_semantic_scope_v1 =
    "citlali-observation-matched-apt-semantic-sha256-v1";
inline constexpr std::string_view matched_output_envelope_scope_v1 =
    "citlali-observation-matched-apt-envelope-sha256-v1";
inline constexpr std::string_view matched_output_byte_transport_scope_v1 =
    "citlali-observation-matched-apt-byte-transport-sha256-v1";
inline constexpr std::string_view matched_output_ecsv_metadata_root_v1 =
    "canonical_apt_observation_v1";

inline constexpr std::string_view target_artifact_contract_id_v1 =
    "apt-prod-002-observation-target-manifest-v1";
inline constexpr std::string_view relation_artifact_contract_id_v1 =
    "apt-prod-002-match-dispositions-v1";
inline constexpr std::string_view matched_output_artifact_contract_id_v1 =
    "apt-prod-002-observation-matched-apt-v1";
inline constexpr std::string_view target_field_registry_v1 =
    "citlali-observation-target-fields-v1";
inline constexpr std::string_view kmp_source_field_map_profile_v1 =
    "citlali-kmp-source-field-map-v1";
inline constexpr std::string_view kmp_model_params_authority_reference_v1 =
    "kids:model-params-v1";
inline constexpr std::string_view kids_fit_report_authority_reference_v1 =
    "kids:fit-report-v1";
inline constexpr std::string_view matched_output_field_registry_v1 =
    "citlali-observation-matched-output-fields-v1";
inline constexpr std::string_view transformation_registry_v1 =
    "citlali-observation-apt-field-transformations-v1";
inline constexpr std::string_view unmatched_missing_authority_v1 =
    "citlali:typed-missing-unmatched-v1";
inline constexpr std::string_view mapping_domain_v1 =
    "tolproj-observation-tone-to-beammap-seed-v1";
inline constexpr std::string_view baseline_receipt_schema_v1 =
    "citlali-canonical-apt-publication-receipt-v1";

struct IssuanceEnvelope {
    // All references are opaque issuer-provided values. Nothing in this
    // contract parses them or derives them from content, paths, clocks, or
    // local row keys.
    std::string occurrence;
    std::string event_reference;
    std::string software_revision;
    std::string configuration_reference;
    std::string event_time_utc;

    friend bool operator==(const IssuanceEnvelope &,
                           const IssuanceEnvelope &) = default;
};

struct ArtifactIdentity {
    std::string schema;
    std::string occurrence;
    std::string semantic_sha256;
    std::string envelope_sha256;

    friend bool operator==(const ArtifactIdentity &,
                           const ArtifactIdentity &) = default;
};

struct VerifiedBaselineReference {
    ArtifactIdentity artifact;
    std::string profile;
    std::string descriptor_sha256;
    std::string transport_scope;
    std::string transport_sha256;
    std::uint64_t byte_count = 0;
    std::string receipt_sha256;
    std::uint64_t receipt_byte_count = 0;

    friend bool operator==(const VerifiedBaselineReference &,
                           const VerifiedBaselineReference &) = default;
};

struct RowReference {
    std::string artifact_schema;
    std::string occurrence;
    std::string envelope_sha256;
    std::int64_t local_key = 0;

    friend bool operator==(const RowReference &,
                           const RowReference &) = default;
};

class VerifiedBaselineDescriptor;

VerifiedBaselineDescriptor verify_baseline_descriptor(
    std::string_view baseline_bytes, std::string_view receipt_bytes);

class VerifiedBaselineDescriptor {
public:
    VerifiedBaselineDescriptor(const VerifiedBaselineDescriptor &) = default;
    VerifiedBaselineDescriptor(VerifiedBaselineDescriptor &&) noexcept =
        default;
    VerifiedBaselineDescriptor &operator=(
        const VerifiedBaselineDescriptor &) = default;
    VerifiedBaselineDescriptor &operator=(
        VerifiedBaselineDescriptor &&) noexcept = default;

    const std::string &schema() const noexcept { return schema_; }
    const std::string &contract_authority() const noexcept {
        return contract_authority_;
    }
    const std::string &baseline_value_issuer() const noexcept {
        return baseline_value_issuer_;
    }
    const baseline::Document &document() const noexcept { return document_; }
    const baseline::Digests &digests() const noexcept { return digests_; }
    const baseline::ByteTransportHash &transport() const noexcept {
        return transport_;
    }
    const std::string &receipt_sha256() const noexcept {
        return receipt_sha256_;
    }
    std::uint64_t receipt_byte_count() const noexcept {
        return receipt_byte_count_;
    }
    std::string_view baseline_bytes() const noexcept { return baseline_bytes_; }
    std::string_view receipt_bytes() const noexcept { return receipt_bytes_; }

private:
    VerifiedBaselineDescriptor(std::string baseline_bytes,
                               std::string receipt_bytes,
                               baseline::Document document,
                               baseline::Digests digests,
                               baseline::ByteTransportHash transport)
        : document_(std::move(document)), digests_(std::move(digests)),
          transport_(std::move(transport)),
          receipt_sha256_("sha256:" +
                          citlali::utils::sha256(receipt_bytes)),
          receipt_byte_count_(receipt_bytes.size()),
          baseline_bytes_(std::move(baseline_bytes)),
          receipt_bytes_(std::move(receipt_bytes)) {}

    std::string schema_{baseline_descriptor_schema_v1};
    std::string contract_authority_{contract_authority_v1};
    std::string baseline_value_issuer_{baseline_value_issuer_v1};
    baseline::Document document_;
    baseline::Digests digests_;
    baseline::ByteTransportHash transport_;
    std::string receipt_sha256_;
    std::uint64_t receipt_byte_count_ = 0;
    std::string baseline_bytes_;
    std::string receipt_bytes_;

    friend VerifiedBaselineDescriptor verify_baseline_descriptor(
        std::string_view, std::string_view);
};

struct TypedField {
    std::string name;
    baseline::ValueType type = baseline::ValueType::float64;
    std::string unit;
    bool nullable = false;
    baseline::NonFinitePolicy nonfinite =
        baseline::NonFinitePolicy::reject;
    std::string authority;
    std::string authority_reference;
    std::string registry;
    std::string description;
    std::optional<std::string> source_column;
    std::string identity_role{"nonidentity"};

    friend bool operator==(const TypedField &, const TypedField &) = default;
};

struct KmpFieldBinding {
    std::string source_column;
    std::string canonical_field;
    bool required = false;

    friend bool operator==(const KmpFieldBinding &,
                           const KmpFieldBinding &) = default;
};

enum class KmpFieldUseRole {
    identity,
    matching,
    application,
    transformation,
    output,
    authority,
};

struct KmpFieldUseRequest {
    std::string field_name;
    KmpFieldUseRole role = KmpFieldUseRole::identity;
    std::string authority_reference;

    friend bool operator==(const KmpFieldUseRequest &,
                           const KmpFieldUseRequest &) = default;
};

struct SourceArtifact {
    std::int64_t source_key = 0;
    std::string role;
    // A locator is diagnostic context: excluded from semantic content
    // identity, but deterministically bound by the target envelope identity.
    // Content hash, byte count, bound header, and raw relation are semantic.
    std::string diagnostic_locator;
    std::string content_sha256;
    std::uint64_t byte_count = 0;
    baseline::ObservationIdentity header_observation;
    std::int64_t network = 0;
    std::string interface_name;
    std::int64_t channel_count = 0;

    friend bool operator==(const SourceArtifact &,
                           const SourceArtifact &) = default;
};

struct TargetInput {
    std::int64_t input_key = 0;
    std::int64_t network = 0;
    std::string interface_name;
    std::int64_t channel_count = 0;
    SourceArtifact raw_source;
    SourceArtifact kmp_source;

    friend bool operator==(const TargetInput &, const TargetInput &) = default;
};

struct TargetRow {
    std::int64_t row_key = 0;
    std::int64_t input_key = 0;
    std::int64_t kmp_source_key = 0;
    std::int64_t kmp_row_index = 0;
    double matching_frequency_hz = 0.0;
    double output_tone_frequency_hz = 0.0;
    std::int64_t array = 0;
    std::int64_t network = 0;
    std::int64_t channel = 0;
    std::map<std::string, baseline::Value> fields;

    friend bool operator==(const TargetRow &, const TargetRow &) = default;
};

struct TargetManifest {
    std::string schema{target_manifest_schema_v1};
    std::string contract_authority{contract_authority_v1};
    std::string observation_value_issuer{observation_value_issuer_v1};
    IssuanceEnvelope envelope;
    baseline::ObservationIdentity observation;
    std::vector<TargetInput> inputs;
    std::vector<TypedField> registered_fields;
    std::vector<TargetRow> rows;
    std::vector<std::int64_t> target_source_sequence;
    std::vector<std::int64_t> target_application_sequence;
};

struct MatcherEvidence {
    std::string matcher_run_occurrence;
    std::string implementation_revision;
    std::string configuration_reference;
    std::string method;
    std::string backend;
    std::string target_frequency_field{"kids_fr"};
    std::string target_quality_factor_field{"kids_Qr"};
};

struct NetworkMatchEvidence {
    std::int64_t network = 0;
    double frequency_shift_hz = 0.0;
    double gate_hz = 0.0;
    double quality_factor = 0.0;
    std::string quality_factor_field{"kids_Qr"};
    std::string quality_factor_authority_reference{
        kmp_model_params_authority_reference_v1};
};

struct MatchPair {
    std::int64_t pair_key = 0;
    RowReference target;
    RowReference seed;
    double separation_hz = 0.0;
    bool is_good_match = false;
};

enum class EndpointDispositionState {
    matched,
    unmatched,
    unused,
};

struct EndpointDisposition {
    // Opaque relation-artifact-local record key. It exists independently of
    // pair membership, including for unmatched and unused endpoints.
    std::int64_t disposition_key = 0;
    RowReference endpoint;
    EndpointDispositionState state = EndpointDispositionState::unmatched;
    // Pair references are sets encoded in ascending order. This deliberately
    // supports 1:0, 1:1, 1:many, and many:1 relations without making Citlali
    // own a matcher-cardinality policy.
    std::vector<std::int64_t> pair_keys;
    std::string reason;
};

struct MatchRelation {
    std::string schema{relation_schema_v1};
    std::string contract_authority{contract_authority_v1};
    std::string observation_value_issuer{observation_value_issuer_v1};
    std::string mapping_domain{mapping_domain_v1};
    IssuanceEnvelope envelope;
    VerifiedBaselineReference baseline_parent;
    ArtifactIdentity target_parent;
    MatcherEvidence matcher;
    std::vector<NetworkMatchEvidence> network_evidence;
    std::vector<MatchPair> pairs;
    std::vector<EndpointDisposition> target_dispositions;
    std::vector<EndpointDisposition> seed_dispositions;
    std::vector<std::int64_t> seed_source_sequence;
};

enum class TransformationOperation {
    preserve_target,
    copy_baseline_when_matched_preserve_target_when_unmatched,
    copy_baseline_when_matched_null_when_unmatched,
    issuer_declared,
};

enum class TransformationValueSource {
    target_row,
    baseline_seed_row,
    observation_value_issuer,
    canonical_null,
};

struct OutputFieldContract {
    TypedField field;
    TransformationOperation authorized_operation =
        TransformationOperation::preserve_target;
    // Required and exact only for issuer-declared transformations.
    std::string issuer_authority_reference;

    friend bool operator==(const OutputFieldContract &,
                           const OutputFieldContract &) = default;
};

struct FieldTransformation {
    std::string field_name;
    TransformationOperation operation =
        TransformationOperation::preserve_target;
    baseline::Value before;
    baseline::Value after;
    TransformationValueSource value_source =
        TransformationValueSource::target_row;
    std::optional<std::int64_t> source_pair_key;
    std::optional<RowReference> source_row;
    std::string authority_reference;
    std::string provenance_reference;
};

struct MatchedOutputRow {
    std::int64_t uid = 0;
    RowReference target;
    std::int64_t target_input_key = 0;
    double tone_frequency_hz = 0.0;
    std::int64_t array = 0;
    std::int64_t network = 0;
    std::int64_t channel = 0;
    std::vector<std::int64_t> relation_pair_keys;
    std::map<std::string, baseline::Value> fields;
    std::vector<FieldTransformation> transformations;
};

struct MatchedOutput {
    std::string schema{matched_output_schema_v1};
    std::string contract_authority{contract_authority_v1};
    std::string observation_value_issuer{observation_value_issuer_v1};
    std::string transformation_registry{transformation_registry_v1};
    IssuanceEnvelope envelope;
    VerifiedBaselineReference baseline_parent;
    ArtifactIdentity target_parent;
    ArtifactIdentity relation_parent;
    std::vector<OutputFieldContract> registered_fields;
    std::vector<MatchedOutputRow> rows;
    std::vector<std::int64_t> output_presentation_sequence;
};

struct MatchedOutputFieldSource {
    std::int64_t target_row_key = 0;
    std::string field_name;
    // Required for a matched baseline-derived field and absent for an
    // unmatched target. It is a relation-artifact-local edge key, not detector
    // identity and not a matcher choice made by Citlali.
    std::optional<std::int64_t> relation_pair_key;

    friend bool operator==(const MatchedOutputFieldSource &,
                           const MatchedOutputFieldSource &) = default;
};

struct SerializedMatchedObservationEcsv {
    std::string bytes;
    baseline::Digests digests;
    baseline::ByteTransportHash transport;
};

struct ParsedMatchedObservationEcsv {
    TargetManifest target;
    MatchRelation relation;
    MatchedOutput output;
    baseline::Digests declared_digests;
    baseline::ByteTransportHash computed_transport;
};

inline std::string_view endpoint_disposition_token(
    EndpointDispositionState state) {
    switch (state) {
    case EndpointDispositionState::matched:
        return "matched";
    case EndpointDispositionState::unmatched:
        return "unmatched";
    case EndpointDispositionState::unused:
        return "unused";
    }
    throw baseline::ContractError("unsupported endpoint disposition state");
}

inline std::string_view transformation_operation_token(
    TransformationOperation operation) {
    switch (operation) {
    case TransformationOperation::preserve_target:
        return "preserve-target";
    case TransformationOperation::
        copy_baseline_when_matched_preserve_target_when_unmatched:
        return "copy-baseline-when-matched-preserve-target-when-unmatched";
    case TransformationOperation::
        copy_baseline_when_matched_null_when_unmatched:
        return "copy-baseline-when-matched-null-when-unmatched";
    case TransformationOperation::issuer_declared:
        return "issuer-declared";
    }
    throw baseline::ContractError("unsupported field transformation operation");
}

inline std::string_view transformation_value_source_token(
    TransformationValueSource source) {
    switch (source) {
    case TransformationValueSource::target_row:
        return "target-row";
    case TransformationValueSource::baseline_seed_row:
        return "baseline-seed-row";
    case TransformationValueSource::observation_value_issuer:
        return "observation-value-issuer";
    case TransformationValueSource::canonical_null:
        return "canonical-null";
    }
    throw baseline::ContractError("unsupported transformation value source");
}

inline std::string_view kmp_field_use_role_token(KmpFieldUseRole role) {
    switch (role) {
    case KmpFieldUseRole::identity:
        return "identity";
    case KmpFieldUseRole::matching:
        return "matching";
    case KmpFieldUseRole::application:
        return "application";
    case KmpFieldUseRole::transformation:
        return "transformation";
    case KmpFieldUseRole::output:
        return "output";
    case KmpFieldUseRole::authority:
        return "authority";
    }
    throw baseline::ContractError("unsupported KMP field-use role");
}

inline std::string canonical_binary64_payload(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);
    const auto bits = std::isnan(value)
        ? std::uint64_t{0x7ff8000000000000ULL}
        : std::bit_cast<std::uint64_t>(value);
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::hex << std::nouppercase << std::setfill('0')
           << std::setw(16) << bits;
    return stream.str();
}

inline std::string_view canonical_value_type_token(
    baseline::ValueType type) {
    if (type == baseline::ValueType::float64) {
        return "float64-ieee754";
    }
    return baseline::value_type_token(type);
}

inline std::string canonical_int64_frame(std::string_view label,
                                         std::int64_t value) {
    return baseline::canonical_frame(label, "int64", std::to_string(value));
}

inline std::string canonical_uint64_frame(std::string_view label,
                                          std::uint64_t value) {
    return baseline::canonical_frame(label, "uint64", std::to_string(value));
}

inline std::string canonical_binary64_frame(std::string_view label,
                                            double value) {
    return baseline::canonical_frame(label, "float64-ieee754",
                                     canonical_binary64_payload(value));
}

inline std::string canonical_null_frame(std::string_view label,
                                        baseline::ValueType type) {
    return baseline::canonical_frame(
        label, "null-" + std::string(baseline::value_type_token(type)),
        "null");
}

inline const std::vector<TypedField> &canonical_target_fields_v1() {
    // This is the complete Citlali-owned v1 KMP value catalog. The first
    // three fields are mandatory. kids_flag is optional at artifact scope,
    // but is exact signed int64 and nonnullable whenever declared. Unknown
    // KMP columns remain covered by the selected source byte digest and are
    // not promoted into this semantic field registry.
    static const std::vector<TypedField> fields{
        {"kids_fr", baseline::ValueType::float64, "Hz", false,
         baseline::NonFinitePolicy::reject, "copied-declared",
         std::string(kmp_model_params_authority_reference_v1),
         std::string(target_field_registry_v1),
         "imported KIDs resonant frequency; finite, nonidentity", "fr"},
        {"kids_f_out", baseline::ValueType::float64, "Hz", false,
         baseline::NonFinitePolicy::reject, "copied-declared",
         std::string(kmp_model_params_authority_reference_v1),
         std::string(target_field_registry_v1),
         "imported KIDs output tone frequency; finite, nonidentity",
         "f_out"},
        {"kids_Qr", baseline::ValueType::float64, "N/A", false,
         baseline::NonFinitePolicy::reject, "copied-declared",
         std::string(kmp_model_params_authority_reference_v1),
         std::string(target_field_registry_v1),
         "imported KIDs resonator Qr; finite with no positivity rule, nonidentity",
         "Qr"},
        {"kids_flag", baseline::ValueType::int64, "N/A", false,
         baseline::NonFinitePolicy::reject, "copied-declared",
         std::string(kids_fit_report_authority_reference_v1),
         std::string(target_field_registry_v1),
         "imported KIDs model-fit flag; exact signed integral values, nonidentity",
         "flag"},
    };
    return fields;
}

inline const std::vector<TypedField> &canonical_required_target_fields_v1() {
    static const std::vector<TypedField> fields{
        canonical_target_fields_v1().begin(),
        canonical_target_fields_v1().begin() + 3};
    return fields;
}

inline const std::vector<KmpFieldBinding> &
canonical_kmp_field_bindings_v1() {
    static const std::vector<KmpFieldBinding> bindings{
        {"fr", "kids_fr", true},
        {"f_out", "kids_f_out", true},
        {"Qr", "kids_Qr", true},
        {"flag", "kids_flag", false},
    };
    return bindings;
}

inline bool canonical_kmp_field_use_allowed_v1(std::string_view field_name,
                                                KmpFieldUseRole role) {
    if (role == KmpFieldUseRole::matching) {
        return field_name == "kids_fr" || field_name == "kids_Qr";
    }
    if (role == KmpFieldUseRole::application) {
        return field_name == "kids_f_out";
    }
    if (role == KmpFieldUseRole::output) {
        return field_name == "kids_fr" || field_name == "kids_f_out" ||
            field_name == "kids_Qr" || field_name == "kids_flag";
    }
    if (role == KmpFieldUseRole::authority) {
        return field_name == "kids_fr" || field_name == "kids_f_out" ||
            field_name == "kids_Qr" || field_name == "kids_flag";
    }
    // No KMP field is identity, and source-boundary callers may not use a
    // role outside the exact matrix to promote another column.
    return false;
}

inline std::vector<TypedField> select_canonical_kmp_fields_v1(
    const std::vector<std::string> &available_source_columns,
    const std::vector<KmpFieldUseRequest> &requested_uses) {
    std::set<std::string> available;
    for (const auto &name : available_source_columns) {
        if (name.empty() || !baseline::detail::canonical_text(name, true) ||
            !available.insert(name).second) {
            throw baseline::ContractError(
                "KMP source column names must be nonempty, canonical, and unique");
        }
    }

    std::map<std::string, TypedField> catalog;
    for (const auto &field : canonical_target_fields_v1()) {
        catalog.emplace(field.name, field);
    }
    std::map<std::string, KmpFieldBinding> bindings;
    for (const auto &binding : canonical_kmp_field_bindings_v1()) {
        bindings.emplace(binding.canonical_field, binding);
        if (binding.required && !available.contains(binding.source_column)) {
            throw baseline::ContractError(
                "KMP source is missing a required exact raw column");
        }
    }
    std::set<std::pair<std::string, KmpFieldUseRole>> seen_requests;
    for (const auto &request : requested_uses) {
        const auto found = catalog.find(request.field_name);
        const auto binding = bindings.find(request.field_name);
        if (found == catalog.end() || binding == bindings.end() ||
            !available.contains(binding->second.source_column) ||
            request.authority_reference !=
                found->second.authority_reference ||
            !canonical_kmp_field_use_allowed_v1(request.field_name,
                                                request.role) ||
            !seen_requests.emplace(request.field_name, request.role).second) {
            throw baseline::ContractError(
                "KMP requested field/use/authority is outside the closed Citlali v1 catalog");
        }
    }

    std::vector<TypedField> selected;
    for (const auto &field : canonical_target_fields_v1()) {
        const auto binding = bindings.find(field.name);
        if (binding->second.required ||
            available.contains(binding->second.source_column)) {
            selected.push_back(field);
        }
    }
    return selected;
}

inline std::vector<OutputFieldContract>
canonical_output_field_contracts_v1(
    const VerifiedBaselineDescriptor &descriptor,
    const TargetManifest &target);

namespace detail {

inline void add_frame(std::string &preimage, std::string label,
                      std::string_view type, std::string payload) {
    preimage += baseline::canonical_frame(label, type, payload);
}

inline void add_string(std::string &preimage, std::string label,
                       std::string_view value) {
    add_frame(preimage, std::move(label), "utf8", std::string(value));
}

inline void add_int64(std::string &preimage, std::string label,
                      std::int64_t value) {
    add_frame(preimage, std::move(label), "int64", std::to_string(value));
}

inline void add_uint64(std::string &preimage, std::string label,
                       std::uint64_t value) {
    add_frame(preimage, std::move(label), "uint64", std::to_string(value));
}

inline void add_bool(std::string &preimage, std::string label, bool value) {
    add_frame(preimage, std::move(label), "bool", value ? "true" : "false");
}

inline void add_float64(std::string &preimage, std::string label,
                        double value) {
    add_frame(preimage, std::move(label), "float64-ieee754",
              canonical_binary64_payload(value));
}

inline void add_value(std::string &preimage, std::string label,
                      const baseline::Value &value,
                      baseline::ValueType declared_type) {
    std::visit(
        [&](const auto &typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, baseline::NullValue>) {
                add_frame(preimage, std::move(label),
                          "null-" + std::string(
                              baseline::value_type_token(declared_type)),
                          "null");
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                add_int64(preimage, std::move(label), typed);
            } else if constexpr (std::is_same_v<T, double>) {
                add_float64(preimage, std::move(label), typed);
            } else if constexpr (std::is_same_v<T, bool>) {
                add_bool(preimage, std::move(label), typed);
            } else if constexpr (std::is_same_v<T, std::string>) {
                add_string(preimage, std::move(label), typed);
            }
        },
        value);
}

inline void require_local_key(std::int64_t key, std::string_view label) {
    if (key < 0 || key > baseline::uid_v1_max) {
        throw baseline::ContractError(std::string(label) +
                                      " is outside [0, 2^53-1]");
    }
}

inline void require_observation(const baseline::ObservationIdentity &value,
                                std::string_view label) {
    if (value.observation < 0 || value.subobservation < 0 || value.scan < 0) {
        throw baseline::ContractError(std::string(label) +
                                      " must be nonnegative");
    }
}

inline void require_envelope(const IssuanceEnvelope &envelope) {
    baseline::detail::require_text("opaque successor occurrence",
                                   envelope.occurrence);
    baseline::detail::require_text("successor event reference",
                                   envelope.event_reference);
    baseline::detail::require_text("successor software revision",
                                   envelope.software_revision);
    baseline::detail::require_text("successor configuration reference",
                                   envelope.configuration_reference);
    baseline::detail::require_text("successor event UTC",
                                   envelope.event_time_utc);
    if (!baseline::detail::is_utc_timestamp(envelope.event_time_utc)) {
        throw baseline::ContractError(
            "successor occurrence envelope event time is not exact UTC");
    }
}

inline void require_artifact_identity(const ArtifactIdentity &identity) {
    baseline::detail::require_text("artifact identity schema",
                                   identity.schema);
    baseline::detail::require_text("artifact identity occurrence",
                                   identity.occurrence);
    if (!baseline::is_sha256_reference(identity.semantic_sha256) ||
        !baseline::is_sha256_reference(identity.envelope_sha256)) {
        throw baseline::ContractError(
            "artifact identity requires exact semantic/envelope SHA-256 references");
    }
}

inline void require_row_reference(const RowReference &reference) {
    baseline::detail::require_text("row-reference artifact schema",
                                   reference.artifact_schema);
    baseline::detail::require_text("row-reference occurrence",
                                   reference.occurrence);
    if (!baseline::is_sha256_reference(reference.envelope_sha256)) {
        throw baseline::ContractError(
            "row reference requires exact artifact envelope SHA-256");
    }
    require_local_key(reference.local_key, "row-reference local key");
}

inline bool protected_field_name(std::string_view name) {
    if (baseline::detail::protected_contract_name(name)) {
        return true;
    }
    constexpr std::array<std::string_view, 34> protected_names{
        "row_key", "input_key", "source_key", "target_input_key",
        "kmp_source_key", "kmp_row_index",
        "matching_frequency_hz", "output_tone_frequency_hz",
        "content_sha256", "byte_count", "raw_source", "kmp_source",
        "target_source_sequence", "target_application_sequence",
        "seed_source_sequence", "output_presentation_sequence",
        "pair_key", "pair_keys", "relation_pair_keys", "target",
        "seed", "baseline_parent", "target_parent", "relation_parent",
        "contract_authority", "observation_value_issuer",
        "mapping_domain", "transformation_registry", "before", "after",
        "operation", "value_source", "source_pair_key", "disposition_key"};
    return std::find(protected_names.begin(), protected_names.end(), name) !=
        protected_names.end();
}

inline bool value_matches(const baseline::Value &value,
                          const TypedField &field) {
    if (std::holds_alternative<baseline::NullValue>(value)) {
        return field.nullable;
    }
    if (!baseline::detail::value_matches_type(value, field.type)) {
        return false;
    }
    if (field.type == baseline::ValueType::float64) {
        const auto typed = std::get<double>(value);
        if ((!std::isfinite(typed) &&
             field.nonfinite == baseline::NonFinitePolicy::reject) ||
            (std::isinf(typed) &&
             field.nonfinite == baseline::NonFinitePolicy::nan_token)) {
            return false;
        }
    }
    if (field.type == baseline::ValueType::string) {
        const auto &typed = std::get<std::string>(value);
        if (typed.empty() || !baseline::detail::canonical_text(typed, true)) {
            return false;
        }
    }
    return true;
}

inline bool values_equal(const baseline::Value &lhs,
                         const baseline::Value &rhs,
                         baseline::ValueType type) {
    if (lhs.index() != rhs.index()) {
        return false;
    }
    if (std::holds_alternative<double>(lhs)) {
        return canonical_binary64_payload(std::get<double>(lhs)) ==
            canonical_binary64_payload(std::get<double>(rhs));
    }
    std::string left;
    std::string right;
    add_value(left, "value", lhs, type);
    add_value(right, "value", rhs, type);
    return left == right;
}

inline void require_typed_field(const TypedField &field,
                                std::string_view registry) {
    if (!baseline::detail::valid_registered_name(field.name) ||
        protected_field_name(field.name)) {
        throw baseline::ContractError(
            "successor field name is invalid or structural/protected: " +
            field.name);
    }
    baseline::detail::require_text("successor field unit", field.unit);
    baseline::detail::require_text("successor field authority",
                                   field.authority);
    baseline::detail::require_text("successor field authority reference",
                                   field.authority_reference);
    baseline::detail::require_text("successor field registry",
                                   field.registry);
    baseline::detail::require_text("successor field description",
                                   field.description);
    if (field.source_column &&
        (field.source_column->empty() ||
         !baseline::detail::canonical_text(*field.source_column, true))) {
        throw baseline::ContractError(
            "successor field source column is not canonical text");
    }
    if (field.identity_role != "nonidentity") {
        throw baseline::ContractError(
            "successor v1 field identity role must be exact nonidentity");
    }
    if (field.registry != registry ||
        (field.type != baseline::ValueType::float64 &&
         field.nonfinite != baseline::NonFinitePolicy::reject)) {
        throw baseline::ContractError(
            "successor field has the wrong registry or nonfinite policy");
    }
}

inline std::map<std::string, TypedField> field_map(
    const std::vector<TypedField> &fields, std::string_view registry) {
    std::map<std::string, TypedField> result;
    for (const auto &field : fields) {
        require_typed_field(field, registry);
        if (!result.emplace(field.name, field).second) {
            throw baseline::ContractError("duplicate successor field: " +
                                          field.name);
        }
    }
    return result;
}

inline void require_exact_fields(
    const std::map<std::string, baseline::Value> &values,
    const std::map<std::string, TypedField> &fields) {
    if (values.size() != fields.size()) {
        throw baseline::ContractError(
            "successor row does not contain exactly its registered fields");
    }
    for (const auto &[name, field] : fields) {
        const auto found = values.find(name);
        if (found == values.end() || !value_matches(found->second, field)) {
            throw baseline::ContractError(
                "successor row field is absent or violates its exact type: " +
                name);
        }
    }
}

inline void require_sorted_unique_pair_keys(
    const std::vector<std::int64_t> &keys, std::string_view label) {
    if (!std::is_sorted(keys.begin(), keys.end()) ||
        std::adjacent_find(keys.begin(), keys.end()) != keys.end()) {
        throw baseline::ContractError(std::string(label) +
                                      " must be a sorted unique set");
    }
    for (const auto key : keys) {
        require_local_key(key, label);
    }
}

inline void require_permutation(const std::vector<std::int64_t> &sequence,
                                const std::set<std::int64_t> &keys,
                                std::string_view label) {
    if (sequence.size() != keys.size() ||
        std::set<std::int64_t>(sequence.begin(), sequence.end()) != keys) {
        throw baseline::ContractError(std::string(label) +
                                      " is not a complete permutation");
    }
}

inline void add_observation(std::string &preimage, std::string prefix,
                            const baseline::ObservationIdentity &value) {
    add_int64(preimage, prefix + ".observation", value.observation);
    add_int64(preimage, prefix + ".subobservation", value.subobservation);
    add_int64(preimage, prefix + ".scan", value.scan);
}

inline void add_artifact_identity(std::string &preimage, std::string prefix,
                                  const ArtifactIdentity &value) {
    add_string(preimage, prefix + ".schema", value.schema);
    add_string(preimage, prefix + ".occurrence", value.occurrence);
    add_string(preimage, prefix + ".semantic-sha256",
               value.semantic_sha256);
    add_string(preimage, prefix + ".envelope-sha256",
               value.envelope_sha256);
}

inline void add_baseline_reference(
    std::string &preimage, std::string prefix,
    const VerifiedBaselineReference &value) {
    add_artifact_identity(preimage, prefix + ".artifact", value.artifact);
    add_string(preimage, prefix + ".profile", value.profile);
    add_string(preimage, prefix + ".descriptor-sha256",
               value.descriptor_sha256);
    add_string(preimage, prefix + ".transport-scope",
               value.transport_scope);
    add_string(preimage, prefix + ".transport-sha256",
               value.transport_sha256);
    add_uint64(preimage, prefix + ".byte-count", value.byte_count);
    add_string(preimage, prefix + ".receipt-sha256",
               value.receipt_sha256);
    add_uint64(preimage, prefix + ".receipt-byte-count",
               value.receipt_byte_count);
}

inline void add_row_reference(std::string &preimage, std::string prefix,
                              const RowReference &value) {
    add_string(preimage, prefix + ".artifact-schema",
               value.artifact_schema);
    add_string(preimage, prefix + ".occurrence", value.occurrence);
    add_string(preimage, prefix + ".envelope-sha256",
               value.envelope_sha256);
    add_int64(preimage, prefix + ".local-key", value.local_key);
}

inline void add_envelope(std::string &preimage, std::string_view schema,
                         std::string_view scope,
                         std::string_view semantic_sha256,
                         const IssuanceEnvelope &envelope) {
    add_string(preimage, "encoding", framing_encoding_v1);
    add_string(preimage, "scope", scope);
    add_string(preimage, "schema", schema);
    add_string(preimage, "contract-authority", contract_authority_v1);
    add_string(preimage, "canonical-issuer", contract_authority_v1);
    add_string(preimage, "observation-value-issuer",
               observation_value_issuer_v1);
    add_string(preimage, "semantic-sha256", semantic_sha256);
    add_string(preimage, "occurrence", envelope.occurrence);
    add_string(preimage, "event-reference", envelope.event_reference);
    add_string(preimage, "software-revision", envelope.software_revision);
    add_string(preimage, "configuration-reference",
               envelope.configuration_reference);
    add_string(preimage, "event-time-utc", envelope.event_time_utc);
}

inline void add_typed_field(std::string &preimage, std::string prefix,
                            const TypedField &field) {
    add_string(preimage, prefix + ".name", field.name);
    add_string(preimage, prefix + ".type",
               canonical_value_type_token(field.type));
    add_string(preimage, prefix + ".unit", field.unit);
    add_bool(preimage, prefix + ".nullable", field.nullable);
    add_string(preimage, prefix + ".nonfinite",
               baseline::nonfinite_policy_token(field.nonfinite));
    add_string(preimage, prefix + ".authority", field.authority);
    add_string(preimage, prefix + ".authority-reference",
               field.authority_reference);
    add_string(preimage, prefix + ".registry", field.registry);
    add_string(preimage, prefix + ".description", field.description);
    add_bool(preimage, prefix + ".has-source-column",
             field.source_column.has_value());
    if (field.source_column) {
        add_string(preimage, prefix + ".source-column",
                   *field.source_column);
    }
    add_string(preimage, prefix + ".identity-role", field.identity_role);
}

inline RowReference make_row_reference(const ArtifactIdentity &artifact,
                                       std::int64_t local_key) {
    return {artifact.schema, artifact.occurrence, artifact.envelope_sha256,
            local_key};
}

}  // namespace detail

inline std::string canonical_baseline_receipt_bytes(
    const baseline::ByteTransportHash &transport) {
    if (transport.scope != baseline::byte_transport_scope_v1 ||
        !baseline::is_sha256_reference(transport.envelope_sha256) ||
        !baseline::is_sha256_reference(transport.sha256)) {
        throw baseline::ContractError(
            "baseline receipt has an invalid transport scope or digest");
    }
    return std::string(baseline_receipt_schema_v1) + "\nscope=" +
        transport.scope + "\nenvelope_sha256=" + transport.envelope_sha256 +
        "\nbyte_sha256=" + transport.sha256 + "\nbyte_count=" +
        std::to_string(transport.byte_count) + "\n";
}

inline baseline::ByteTransportHash parse_canonical_baseline_receipt(
    std::string_view bytes) {
    if (bytes.empty() || bytes.back() != '\n' ||
        bytes.find('\r') != std::string_view::npos) {
        throw baseline::ContractError(
            "baseline receipt must be exact LF-terminated ASCII text");
    }
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    while (start < bytes.size()) {
        const auto end = bytes.find('\n', start);
        lines.push_back(bytes.substr(start, end - start));
        start = end + 1;
    }
    if (lines.size() != 5 || lines[0] != baseline_receipt_schema_v1) {
        throw baseline::ContractError(
            "baseline receipt schema or line count mismatch");
    }
    const auto value = [](std::string_view line, std::string_view prefix) {
        if (!line.starts_with(prefix) || line.size() == prefix.size()) {
            throw baseline::ContractError(
                "baseline receipt field is absent or misordered");
        }
        return line.substr(prefix.size());
    };
    baseline::ByteTransportHash result;
    result.scope = value(lines[1], "scope=");
    result.envelope_sha256 = value(lines[2], "envelope_sha256=");
    result.sha256 = value(lines[3], "byte_sha256=");
    const auto count = value(lines[4], "byte_count=");
    const auto [end, error] = std::from_chars(
        count.data(), count.data() + count.size(), result.byte_count);
    if (error != std::errc{} || end != count.data() + count.size() ||
        result.scope != baseline::byte_transport_scope_v1 ||
        !baseline::is_sha256_reference(result.envelope_sha256) ||
        !baseline::is_sha256_reference(result.sha256) ||
        canonical_baseline_receipt_bytes(result) != bytes) {
        throw baseline::ContractError(
            "baseline receipt is not exact canonical v1 text");
    }
    return result;
}

inline VerifiedBaselineDescriptor verify_baseline_descriptor(
    std::string_view baseline_bytes, std::string_view receipt_bytes) {
    const auto receipt = parse_canonical_baseline_receipt(receipt_bytes);
    const auto parsed =
        baseline::parse_ecsv_with_transport(baseline_bytes, receipt);
    if (parsed.declared_digests.envelope_sha256 !=
        receipt.envelope_sha256) {
        throw baseline::ContractError(
            "baseline receipt does not bind the parsed immutable envelope");
    }
    return VerifiedBaselineDescriptor{
        std::string(baseline_bytes), std::string(receipt_bytes),
        parsed.document, parsed.declared_digests, parsed.computed_transport};
}

inline std::string baseline_descriptor_preimage(
    const VerifiedBaselineDescriptor &descriptor) {
    // Reconstruct from the exact immutable bytes on every trust-boundary use;
    // no caller assertion or mutable aggregate can manufacture verification.
    const auto reverified = verify_baseline_descriptor(
        descriptor.baseline_bytes(), descriptor.receipt_bytes());
    (void)reverified;
    const auto &document = descriptor.document();
    const auto &digests = descriptor.digests();
    const auto &transport = descriptor.transport();
    std::string preimage;
    detail::add_string(preimage, "encoding", framing_encoding_v1);
    detail::add_string(preimage, "scope", baseline_descriptor_scope_v1);
    detail::add_string(preimage, "schema", descriptor.schema());
    detail::add_string(preimage, "contract-authority",
                       descriptor.contract_authority());
    detail::add_string(preimage, "baseline-value-issuer",
                       descriptor.baseline_value_issuer());
    detail::add_string(preimage, "baseline-schema",
                       baseline::schema_version_v1);
    detail::add_string(preimage, "baseline-profile",
                       document.profile);
    detail::add_string(preimage, "baseline-occurrence",
                       document.envelope.occurrence);
    detail::add_string(preimage, "baseline-semantic-sha256",
                       digests.semantic_sha256);
    detail::add_string(preimage, "baseline-envelope-sha256",
                       digests.envelope_sha256);
    detail::add_string(preimage, "baseline-transport-scope",
                       transport.scope);
    detail::add_string(preimage, "baseline-transport-sha256",
                       transport.sha256);
    detail::add_uint64(preimage, "baseline-byte-count",
                       transport.byte_count);
    detail::add_string(preimage, "receipt-sha256",
                       descriptor.receipt_sha256());
    detail::add_uint64(preimage, "receipt-byte-count",
                       descriptor.receipt_byte_count());
    return preimage;
}

inline std::string baseline_descriptor_sha256(
    const VerifiedBaselineDescriptor &descriptor) {
    return "sha256:" +
        citlali::utils::sha256(baseline_descriptor_preimage(descriptor));
}

inline ArtifactIdentity artifact_identity(
    const VerifiedBaselineDescriptor &descriptor) {
    (void)baseline_descriptor_preimage(descriptor);
    return {std::string(baseline::schema_version_v1),
            descriptor.document().envelope.occurrence,
            descriptor.digests().semantic_sha256,
            descriptor.digests().envelope_sha256};
}

inline VerifiedBaselineReference baseline_reference(
    const VerifiedBaselineDescriptor &descriptor) {
    return {artifact_identity(descriptor),
            descriptor.document().profile,
            baseline_descriptor_sha256(descriptor),
            descriptor.transport().scope,
            descriptor.transport().sha256,
            descriptor.transport().byte_count,
            descriptor.receipt_sha256(),
            descriptor.receipt_byte_count()};
}

inline void validate(const TargetManifest &document) {
    if (document.schema != target_manifest_schema_v1 ||
        document.contract_authority != contract_authority_v1 ||
        document.observation_value_issuer != observation_value_issuer_v1) {
        throw baseline::ContractError(
            "target manifest schema or authority/issuer mismatch");
    }
    detail::require_envelope(document.envelope);
    detail::require_observation(document.observation, "target observation");
    if (document.inputs.empty() || document.rows.empty()) {
        throw baseline::ContractError(
            "target manifest requires at least one input and row");
    }
    const auto fields = detail::field_map(document.registered_fields,
                                          target_field_registry_v1);
    const auto allowed_fields = detail::field_map(
        canonical_target_fields_v1(), target_field_registry_v1);
    const auto required_fields = detail::field_map(
        canonical_required_target_fields_v1(), target_field_registry_v1);
    if (fields.size() < required_fields.size() ||
        fields.size() > allowed_fields.size()) {
        throw baseline::ContractError(
            "target manifest field declarations do not match the immutable Citlali v1 registry");
    }
    for (const auto &[name, field] : fields) {
        const auto expected = allowed_fields.find(name);
        if (expected == allowed_fields.end() || expected->second != field) {
            throw baseline::ContractError(
                "target manifest field declarations do not match the immutable Citlali v1 registry");
        }
    }
    for (const auto &[name, field] : required_fields) {
        (void)field;
        if (!fields.contains(name)) {
            throw baseline::ContractError(
                "target manifest is missing a required KMP value field");
        }
    }
    std::map<std::int64_t, TargetInput> inputs;
    std::set<std::int64_t> networks;
    std::set<std::string> interfaces;
    std::set<std::int64_t> source_keys;
    std::uint64_t expected_rows = 0;
    for (const auto &input : document.inputs) {
        detail::require_local_key(input.input_key, "target input key");
        baseline::detail::require_text("target interface",
                                       input.interface_name);
        if (input.network < 0 || input.network > 12 ||
            input.channel_count <= 0 ||
            input.channel_count > baseline::uid_v1_max + 1 ||
            !baseline::detail::is_canonical_toltec_interface(
                input.interface_name, input.network) ||
            !inputs.emplace(input.input_key, input).second ||
            !networks.insert(input.network).second ||
            !interfaces.insert(input.interface_name).second) {
            throw baseline::ContractError(
                "target inputs require unique local keys and canonical network/interfaces");
        }
        const auto validate_source = [&](const SourceArtifact &source,
                                         std::string_view role) {
            detail::require_local_key(source.source_key,
                                      "target source-artifact key");
            baseline::detail::require_text("target source role", source.role);
            baseline::detail::require_text("target diagnostic locator",
                                           source.diagnostic_locator);
            baseline::detail::require_text("target source interface",
                                           source.interface_name);
            if (source.role != role ||
                !source_keys.insert(source.source_key).second ||
                !baseline::is_sha256_reference(source.content_sha256) ||
                source.byte_count == 0 || source.network != input.network ||
                source.interface_name != input.interface_name ||
                source.channel_count != input.channel_count) {
                throw baseline::ContractError(
                    "target source artifact role/key/content or network/interface/channel binding is invalid");
            }
            detail::require_observation(source.header_observation,
                                        "target source header observation");
        };
        validate_source(input.raw_source, "raw");
        validate_source(input.kmp_source, "kmp");
        if (!(input.raw_source.header_observation == document.observation)) {
            throw baseline::ContractError(
                "raw source header does not bind the target observation");
        }
        const auto count = static_cast<std::uint64_t>(input.channel_count);
        if (expected_rows >
            static_cast<std::uint64_t>(baseline::uid_v1_max) + 1U - count) {
            throw baseline::ContractError(
                "target manifest channel inventory exceeds v1 capacity");
        }
        expected_rows += count;
    }
    if (expected_rows != document.rows.size()) {
        throw baseline::ContractError(
            "target input channel counts do not cover every target row");
    }

    std::set<std::int64_t> row_keys;
    std::set<std::pair<std::int64_t, std::int64_t>> raw_relations;
    std::map<std::int64_t, std::uint64_t> rows_per_input;
    for (const auto &row : document.rows) {
        detail::require_local_key(row.row_key, "target row key");
        if (!row_keys.insert(row.row_key).second) {
            throw baseline::ContractError("duplicate target-local row key");
        }
        const auto input = inputs.find(row.input_key);
        if (input == inputs.end() || row.network != input->second.network ||
            row.kmp_source_key != input->second.kmp_source.source_key ||
            row.kmp_row_index != row.channel ||
            row.channel < 0 || row.channel >= input->second.channel_count ||
            row.array !=
                baseline::detail::expected_array_for_network(row.network) ||
            !std::isfinite(row.matching_frequency_hz) ||
            !std::isfinite(row.output_tone_frequency_hz) ||
            !raw_relations.emplace(row.network, row.channel).second) {
            throw baseline::ContractError(
                "target row raw relation/frequency is invalid or duplicate");
        }
        ++rows_per_input[row.input_key];
        detail::require_exact_fields(row.fields, fields);
        const auto &kids_fr = row.fields.at("kids_fr");
        const auto &kids_f_out = row.fields.at("kids_f_out");
        if (canonical_binary64_payload(std::get<double>(kids_fr)) !=
                canonical_binary64_payload(row.matching_frequency_hz) ||
            canonical_binary64_payload(std::get<double>(kids_f_out)) !=
                canonical_binary64_payload(
                    row.output_tone_frequency_hz)) {
            throw baseline::ContractError(
                "target structural frequency aliases differ from their authoritative KMP values");
        }
    }
    for (const auto &[key, input] : inputs) {
        if (rows_per_input[key] !=
            static_cast<std::uint64_t>(input.channel_count)) {
            throw baseline::ContractError(
                "target row/raw/KMP relation is not a complete bijection");
        }
    }
    detail::require_permutation(document.target_source_sequence, row_keys,
                                "target source sequence");
    detail::require_permutation(document.target_application_sequence,
                                row_keys, "target application sequence");
}

inline std::vector<OutputFieldContract>
canonical_output_field_contracts_v1(
    const VerifiedBaselineDescriptor &descriptor,
    const TargetManifest &target) {
    validate(target);
    (void)baseline_descriptor_preimage(descriptor);

    std::set<std::string> reserved_target_names;
    for (const auto &field : canonical_target_fields_v1()) {
        reserved_target_names.insert(field.name);
    }

    std::vector<OutputFieldContract> result;
    result.reserve(target.registered_fields.size() +
                   descriptor.document().registered_fields.size());
    for (const auto &source : target.registered_fields) {
        result.push_back({
            {source.name,
             source.type,
             source.unit,
             source.nullable,
             source.nonfinite,
             source.authority,
             source.authority_reference,
             std::string(matched_output_field_registry_v1),
             source.description,
             source.source_column},
            TransformationOperation::preserve_target,
            {}});
    }
    for (const auto &source : descriptor.document().registered_fields) {
        if (reserved_target_names.contains(source.name)) {
            // In particular, a baseline kids_flag is seed-local fit evidence.
            // The output name is reserved for the target KMP value and can
            // never be sourced from or collide with that baseline value.
            continue;
        }
        // Successor nullability is deliberately widened only for the derived
        // observation output so unmatched rows carry typed missing rather
        // than legacy 0/1 fillers. The immutable baseline contract is not
        // altered and matched values retain their exact source bits.
        result.push_back({
            {source.name,
             source.type,
             source.unit,
             true,
             source.nonfinite,
             std::string(baseline::field_authority_token(source.authority)),
             source.authority_reference,
             std::string(matched_output_field_registry_v1),
             source.description,
             std::nullopt},
            TransformationOperation::
                copy_baseline_when_matched_null_when_unmatched,
            {}});
    }
    std::sort(result.begin(), result.end(), [](const auto &lhs,
                                               const auto &rhs) {
        return lhs.field.name < rhs.field.name;
    });
    return result;
}

inline std::string target_semantic_preimage(const TargetManifest &document) {
    validate(document);
    std::string preimage;
    detail::add_string(preimage, "encoding", framing_encoding_v1);
    detail::add_string(preimage, "scope", target_semantic_scope_v1);
    detail::add_string(preimage, "schema", document.schema);
    detail::add_string(preimage, "contract-authority",
                       document.contract_authority);
    detail::add_string(preimage, "observation-value-issuer",
                       document.observation_value_issuer);
    detail::add_observation(preimage, "observation", document.observation);
    detail::add_string(preimage, "kmp-source-field-map.profile",
                       kmp_source_field_map_profile_v1);
    const auto &bindings = canonical_kmp_field_bindings_v1();
    detail::add_uint64(preimage, "kmp-source-field-map.count",
                       bindings.size());
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        const auto prefix =
            "kmp-source-field-map." + std::to_string(index);
        detail::add_string(preimage, prefix + ".source-column",
                           bindings[index].source_column);
        detail::add_string(preimage, prefix + ".canonical-field",
                           bindings[index].canonical_field);
        detail::add_bool(preimage, prefix + ".required",
                         bindings[index].required);
    }

    auto fields = document.registered_fields;
    std::sort(fields.begin(), fields.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.name < rhs.name;
              });
    detail::add_uint64(preimage, "field.count", fields.size());
    for (std::size_t index = 0; index < fields.size(); ++index) {
        detail::add_typed_field(preimage,
                                "field." + std::to_string(index),
                                fields[index]);
    }

    auto inputs = document.inputs;
    std::sort(inputs.begin(), inputs.end(), [](const auto &lhs,
                                               const auto &rhs) {
        return lhs.input_key < rhs.input_key;
    });
    detail::add_uint64(preimage, "input.count", inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto prefix = "input." + std::to_string(index);
        const auto &input = inputs[index];
        detail::add_int64(preimage, prefix + ".input-key", input.input_key);
        detail::add_int64(preimage, prefix + ".network", input.network);
        detail::add_string(preimage, prefix + ".interface",
                           input.interface_name);
        detail::add_int64(preimage, prefix + ".channel-count",
                          input.channel_count);
        const auto add_source = [&](std::string role,
                                    const SourceArtifact &source) {
            const auto source_prefix = prefix + "." + role;
            detail::add_int64(preimage, source_prefix + ".source-key",
                              source.source_key);
            detail::add_string(preimage, source_prefix + ".role",
                               source.role);
            detail::add_string(preimage, source_prefix + ".content-sha256",
                               source.content_sha256);
            detail::add_uint64(preimage, source_prefix + ".byte-count",
                               source.byte_count);
            detail::add_observation(preimage, source_prefix + ".header",
                                    source.header_observation);
            detail::add_int64(preimage, source_prefix + ".network",
                              source.network);
            detail::add_string(preimage, source_prefix + ".interface",
                               source.interface_name);
            detail::add_int64(preimage, source_prefix + ".channel-count",
                              source.channel_count);
        };
        add_source("raw", input.raw_source);
        add_source("kmp", input.kmp_source);
    }

    auto rows = document.rows;
    std::sort(rows.begin(), rows.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.row_key < rhs.row_key;
    });
    detail::add_uint64(preimage, "row.count", rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto prefix = "row." + std::to_string(index);
        const auto &row = rows[index];
        detail::add_int64(preimage, prefix + ".row-key", row.row_key);
        detail::add_int64(preimage, prefix + ".input-key", row.input_key);
        detail::add_int64(preimage, prefix + ".kmp-source-key",
                          row.kmp_source_key);
        detail::add_int64(preimage, prefix + ".kmp-row-index",
                          row.kmp_row_index);
        detail::add_float64(preimage, prefix + ".matching-frequency-hz",
                            row.matching_frequency_hz);
        detail::add_float64(preimage, prefix + ".output-tone-frequency-hz",
                            row.output_tone_frequency_hz);
        detail::add_int64(preimage, prefix + ".array", row.array);
        detail::add_int64(preimage, prefix + ".network", row.network);
        detail::add_int64(preimage, prefix + ".channel", row.channel);
        for (const auto &field : fields) {
            detail::add_value(preimage, prefix + ".field." + field.name,
                              row.fields.at(field.name), field.type);
        }
    }
    detail::add_uint64(preimage, "target-source-sequence.count",
                       document.target_source_sequence.size());
    for (std::size_t index = 0;
         index < document.target_source_sequence.size(); ++index) {
        detail::add_int64(preimage,
                          "target-source-sequence." + std::to_string(index),
                          document.target_source_sequence[index]);
    }
    detail::add_uint64(preimage, "target-application-sequence.count",
                       document.target_application_sequence.size());
    for (std::size_t index = 0;
         index < document.target_application_sequence.size(); ++index) {
        detail::add_int64(
            preimage,
            "target-application-sequence." + std::to_string(index),
            document.target_application_sequence[index]);
    }
    return preimage;
}

inline std::string target_semantic_sha256(const TargetManifest &document) {
    return "sha256:" +
        citlali::utils::sha256(target_semantic_preimage(document));
}

inline std::string target_envelope_preimage(const TargetManifest &document) {
    validate(document);
    std::string preimage;
    detail::add_envelope(preimage, document.schema, target_envelope_scope_v1,
                         target_semantic_sha256(document), document.envelope);
    std::vector<SourceArtifact> sources;
    sources.reserve(document.inputs.size() * 2U);
    for (const auto &input : document.inputs) {
        sources.push_back(input.raw_source);
        sources.push_back(input.kmp_source);
    }
    std::sort(sources.begin(), sources.end(), [](const auto &lhs,
                                                 const auto &rhs) {
        return lhs.source_key < rhs.source_key;
    });
    detail::add_uint64(preimage, "source-locator.count", sources.size());
    for (std::size_t index = 0; index < sources.size(); ++index) {
        const auto prefix = "source-locator." + std::to_string(index);
        detail::add_int64(preimage, prefix + ".source-key",
                          sources[index].source_key);
        detail::add_string(preimage, prefix + ".role", sources[index].role);
        detail::add_string(preimage, prefix + ".diagnostic-locator",
                           sources[index].diagnostic_locator);
    }
    return preimage;
}

inline std::string target_envelope_sha256(const TargetManifest &document) {
    return "sha256:" +
        citlali::utils::sha256(target_envelope_preimage(document));
}

inline baseline::Digests compute_digests(const TargetManifest &document) {
    return {target_semantic_sha256(document),
            target_envelope_sha256(document)};
}

inline ArtifactIdentity artifact_identity(const TargetManifest &document) {
    const auto digests = compute_digests(document);
    return {document.schema, document.envelope.occurrence,
            digests.semantic_sha256, digests.envelope_sha256};
}

inline RowReference row_reference(const ArtifactIdentity &artifact,
                                  std::int64_t local_key) {
    detail::require_artifact_identity(artifact);
    detail::require_local_key(local_key, "row-reference local key");
    return detail::make_row_reference(artifact, local_key);
}

inline void validate(const MatchRelation &document,
                     const VerifiedBaselineDescriptor &baseline_descriptor,
                     const TargetManifest &target) {
    validate(target);
    (void)baseline_descriptor_preimage(baseline_descriptor);
    if (document.schema != relation_schema_v1 ||
        document.contract_authority != contract_authority_v1 ||
        document.observation_value_issuer != observation_value_issuer_v1 ||
        document.mapping_domain != mapping_domain_v1 ||
        document.baseline_parent != baseline_reference(baseline_descriptor) ||
        document.target_parent != artifact_identity(target)) {
        throw baseline::ContractError(
            "relation schema, authority, mapping domain, or parent is invalid");
    }
    detail::require_envelope(document.envelope);
    if (target.envelope.occurrence ==
            baseline_descriptor.document().envelope.occurrence ||
        document.envelope.occurrence == target.envelope.occurrence ||
        document.envelope.occurrence ==
            baseline_descriptor.document().envelope.occurrence) {
        throw baseline::ContractError(
            "target, relation, and immutable baseline occurrences must be distinct");
    }
    baseline::detail::require_text("matcher-run occurrence",
                                   document.matcher.matcher_run_occurrence);
    baseline::detail::require_text("matcher implementation revision",
                                   document.matcher.implementation_revision);
    baseline::detail::require_text("matcher configuration reference",
                                   document.matcher.configuration_reference);
    baseline::detail::require_text("matcher method", document.matcher.method);
    baseline::detail::require_text("matcher backend",
                                   document.matcher.backend);
    if (document.matcher.target_frequency_field != "kids_fr" ||
        document.matcher.target_quality_factor_field != "kids_Qr") {
        throw baseline::ContractError(
            "matcher field references are outside the closed Citlali v1 KMP use registry");
    }

    std::set<std::int64_t> target_keys;
    for (const auto &row : target.rows) {
        target_keys.insert(row.row_key);
    }
    std::set<std::int64_t> seed_keys;
    for (const auto &row : baseline_descriptor.document().rows) {
        seed_keys.insert(row.uid);
    }
    detail::require_permutation(document.seed_source_sequence, seed_keys,
                                "seed source sequence");

    const auto target_identity = artifact_identity(target);
    const auto seed_identity = artifact_identity(baseline_descriptor);
    std::map<std::int64_t, MatchPair> pairs;
    std::map<std::int64_t, std::set<std::int64_t>> target_pairs;
    std::map<std::int64_t, std::set<std::int64_t>> seed_pairs;
    std::set<std::pair<std::int64_t, std::int64_t>> endpoints;
    std::set<std::int64_t> relation_row_keys;
    for (const auto &pair : document.pairs) {
        detail::require_local_key(pair.pair_key, "match-pair local key");
        detail::require_row_reference(pair.target);
        detail::require_row_reference(pair.seed);
        if (!relation_row_keys.insert(pair.pair_key).second ||
            !pairs.emplace(pair.pair_key, pair).second ||
            pair.target !=
                detail::make_row_reference(target_identity,
                                           pair.target.local_key) ||
            pair.seed !=
                detail::make_row_reference(seed_identity,
                                           pair.seed.local_key) ||
            !target_keys.contains(pair.target.local_key) ||
            !seed_keys.contains(pair.seed.local_key) ||
            !std::isfinite(pair.separation_hz) || pair.separation_hz < 0.0 ||
            !endpoints.emplace(pair.target.local_key, pair.seed.local_key)
                 .second) {
            throw baseline::ContractError(
                "match pair key/endpoints/separation is invalid or duplicate");
        }
        target_pairs[pair.target.local_key].insert(pair.pair_key);
        seed_pairs[pair.seed.local_key].insert(pair.pair_key);
    }

    std::set<std::int64_t> target_networks;
    for (const auto &row : target.rows) {
        target_networks.insert(row.network);
    }
    std::set<std::int64_t> evidence_networks;
    for (const auto &evidence : document.network_evidence) {
        if (!target_networks.contains(evidence.network) ||
            !evidence_networks.insert(evidence.network).second ||
            !std::isfinite(evidence.frequency_shift_hz) ||
            !std::isfinite(evidence.gate_hz) || evidence.gate_hz < 0.0 ||
            !std::isfinite(evidence.quality_factor) ||
            evidence.quality_factor_field != "kids_Qr" ||
            evidence.quality_factor_authority_reference !=
                kmp_model_params_authority_reference_v1) {
            throw baseline::ContractError(
                "relation network shift/gate/quality-factor evidence is invalid or incomplete");
        }
    }
    if (evidence_networks != target_networks) {
        throw baseline::ContractError(
            "relation requires shift/gate evidence for every target network");
    }

    const auto validate_dispositions = [&]<class KeySet, class PairMap>(
                                           const auto &dispositions,
                                           const KeySet &expected_keys,
                                           const PairMap &expected_pairs,
                                           const ArtifactIdentity &identity,
                                           bool target_side) {
        std::set<std::int64_t> seen;
        for (const auto &disposition : dispositions) {
            detail::require_local_key(disposition.disposition_key,
                                      "disposition local row key");
            detail::require_row_reference(disposition.endpoint);
            baseline::detail::require_text("endpoint disposition reason",
                                           disposition.reason);
            const auto key = disposition.endpoint.local_key;
            if (!relation_row_keys.insert(disposition.disposition_key).second ||
                !expected_keys.contains(key) || !seen.insert(key).second ||
                disposition.endpoint !=
                    detail::make_row_reference(identity, key)) {
                throw baseline::ContractError(
                    "relation disposition endpoint is foreign or duplicate");
            }
            detail::require_sorted_unique_pair_keys(disposition.pair_keys,
                                                    "disposition pair keys");
            const auto found = expected_pairs.find(key);
            const std::set<std::int64_t> actual(
                disposition.pair_keys.begin(), disposition.pair_keys.end());
            const std::set<std::int64_t> expected =
                found == expected_pairs.end()
                ? std::set<std::int64_t>{}
                : found->second;
            if (actual != expected ||
                (!actual.empty() &&
                 disposition.state != EndpointDispositionState::matched) ||
                (actual.empty() && target_side &&
                 disposition.state != EndpointDispositionState::unmatched) ||
                (actual.empty() && !target_side &&
                 disposition.state != EndpointDispositionState::unused)) {
                throw baseline::ContractError(
                    "relation disposition is not reciprocal or complete");
            }
        }
        if (seen != expected_keys) {
            throw baseline::ContractError(
                "relation dispositions do not cover every endpoint");
        }
    };
    validate_dispositions(document.target_dispositions, target_keys,
                          target_pairs, target_identity, true);
    validate_dispositions(document.seed_dispositions, seed_keys, seed_pairs,
                          seed_identity, false);
}

inline std::string relation_semantic_preimage(
    const MatchRelation &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target) {
    validate(document, baseline_descriptor, target);
    std::string preimage;
    detail::add_string(preimage, "encoding", framing_encoding_v1);
    detail::add_string(preimage, "scope", relation_semantic_scope_v1);
    detail::add_string(preimage, "schema", document.schema);
    detail::add_string(preimage, "contract-authority",
                       document.contract_authority);
    detail::add_string(preimage, "observation-value-issuer",
                       document.observation_value_issuer);
    detail::add_string(preimage, "mapping-domain", document.mapping_domain);
    detail::add_baseline_reference(preimage, "baseline-parent",
                                   document.baseline_parent);
    detail::add_artifact_identity(preimage, "target-parent",
                                  document.target_parent);
    detail::add_string(preimage, "matcher.run-occurrence",
                       document.matcher.matcher_run_occurrence);
    detail::add_string(preimage, "matcher.implementation-revision",
                       document.matcher.implementation_revision);
    detail::add_string(preimage, "matcher.configuration-reference",
                       document.matcher.configuration_reference);
    detail::add_string(preimage, "matcher.target-frequency-field",
                       document.matcher.target_frequency_field);
    detail::add_string(preimage, "matcher.target-quality-factor-field",
                       document.matcher.target_quality_factor_field);
    detail::add_string(preimage, "matcher.method", document.matcher.method);
    detail::add_string(preimage, "matcher.backend", document.matcher.backend);

    auto evidence = document.network_evidence;
    std::sort(evidence.begin(), evidence.end(), [](const auto &lhs,
                                                   const auto &rhs) {
        return lhs.network < rhs.network;
    });
    detail::add_uint64(preimage, "network-evidence.count", evidence.size());
    for (std::size_t index = 0; index < evidence.size(); ++index) {
        const auto prefix = "network-evidence." + std::to_string(index);
        detail::add_int64(preimage, prefix + ".network",
                          evidence[index].network);
        detail::add_float64(preimage, prefix + ".frequency-shift-hz",
                            evidence[index].frequency_shift_hz);
        detail::add_float64(preimage, prefix + ".gate-hz",
                            evidence[index].gate_hz);
        detail::add_float64(preimage, prefix + ".quality-factor",
                            evidence[index].quality_factor);
        detail::add_string(preimage, prefix + ".quality-factor-field",
                           evidence[index].quality_factor_field);
        detail::add_string(
            preimage, prefix + ".quality-factor-authority-reference",
            evidence[index].quality_factor_authority_reference);
    }

    auto pairs = document.pairs;
    std::sort(pairs.begin(), pairs.end(), [](const auto &lhs,
                                             const auto &rhs) {
        return lhs.pair_key < rhs.pair_key;
    });
    detail::add_uint64(preimage, "pair.count", pairs.size());
    for (std::size_t index = 0; index < pairs.size(); ++index) {
        const auto prefix = "pair." + std::to_string(index);
        detail::add_int64(preimage, prefix + ".pair-key",
                          pairs[index].pair_key);
        detail::add_row_reference(preimage, prefix + ".target",
                                  pairs[index].target);
        detail::add_row_reference(preimage, prefix + ".seed",
                                  pairs[index].seed);
        detail::add_float64(preimage, prefix + ".separation-hz",
                            pairs[index].separation_hz);
        detail::add_bool(preimage, prefix + ".is-good-match",
                         pairs[index].is_good_match);
    }
    const auto add_dispositions = [&](std::string label,
                                      auto dispositions) {
        std::sort(dispositions.begin(), dispositions.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.disposition_key < rhs.disposition_key;
                  });
        detail::add_uint64(preimage, label + ".count", dispositions.size());
        for (std::size_t index = 0; index < dispositions.size(); ++index) {
            const auto prefix = label + "." + std::to_string(index);
            const auto &disposition = dispositions[index];
            detail::add_int64(preimage, prefix + ".disposition-key",
                              disposition.disposition_key);
            detail::add_row_reference(preimage, prefix + ".endpoint",
                                      disposition.endpoint);
            detail::add_string(preimage, prefix + ".state",
                               endpoint_disposition_token(
                                   disposition.state));
            detail::add_uint64(preimage, prefix + ".pair-key.count",
                               disposition.pair_keys.size());
            for (std::size_t pair_index = 0;
                 pair_index < disposition.pair_keys.size(); ++pair_index) {
                detail::add_int64(
                    preimage,
                    prefix + ".pair-key." + std::to_string(pair_index),
                    disposition.pair_keys[pair_index]);
            }
            detail::add_string(preimage, prefix + ".reason",
                               disposition.reason);
        }
    };
    add_dispositions("target-disposition", document.target_dispositions);
    add_dispositions("seed-disposition", document.seed_dispositions);
    detail::add_uint64(preimage, "seed-source-sequence.count",
                       document.seed_source_sequence.size());
    for (std::size_t index = 0; index < document.seed_source_sequence.size();
         ++index) {
        detail::add_int64(preimage,
                          "seed-source-sequence." + std::to_string(index),
                          document.seed_source_sequence[index]);
    }
    return preimage;
}

inline std::string relation_semantic_sha256(
    const MatchRelation &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target) {
    return "sha256:" + citlali::utils::sha256(
        relation_semantic_preimage(document, baseline_descriptor, target));
}

inline std::string relation_envelope_preimage(
    const MatchRelation &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target) {
    validate(document, baseline_descriptor, target);
    std::string preimage;
    detail::add_envelope(
        preimage, document.schema, relation_envelope_scope_v1,
        relation_semantic_sha256(document, baseline_descriptor, target),
        document.envelope);
    return preimage;
}

inline std::string relation_envelope_sha256(
    const MatchRelation &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target) {
    return "sha256:" + citlali::utils::sha256(
        relation_envelope_preimage(document, baseline_descriptor, target));
}

inline baseline::Digests compute_digests(
    const MatchRelation &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target) {
    return {relation_semantic_sha256(document, baseline_descriptor, target),
            relation_envelope_sha256(document, baseline_descriptor, target)};
}

inline ArtifactIdentity artifact_identity(
    const MatchRelation &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target) {
    const auto digests = compute_digests(document, baseline_descriptor, target);
    return {document.schema, document.envelope.occurrence,
            digests.semantic_sha256, digests.envelope_sha256};
}

inline void validate(
    const MatchedOutput &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target, const MatchRelation &relation) {
    validate(relation, baseline_descriptor, target);
    if (document.schema != matched_output_schema_v1 ||
        document.contract_authority != contract_authority_v1 ||
        document.observation_value_issuer != observation_value_issuer_v1 ||
        document.transformation_registry != transformation_registry_v1 ||
        document.baseline_parent != baseline_reference(baseline_descriptor) ||
        document.target_parent != artifact_identity(target) ||
        document.relation_parent !=
            artifact_identity(relation, baseline_descriptor, target)) {
        throw baseline::ContractError(
            "matched output schema, authority, registry, or parent is invalid");
    }
    detail::require_envelope(document.envelope);
    if (document.envelope.occurrence ==
            baseline_descriptor.document().envelope.occurrence ||
        document.envelope.occurrence == target.envelope.occurrence ||
        document.envelope.occurrence == relation.envelope.occurrence) {
        throw baseline::ContractError(
            "matched-output occurrence must be distinct from every parent occurrence");
    }

    std::map<std::string, OutputFieldContract> output_fields;
    for (const auto &contract : document.registered_fields) {
        detail::require_typed_field(contract.field,
                                    matched_output_field_registry_v1);
        if (!output_fields.emplace(contract.field.name, contract).second) {
            throw baseline::ContractError(
                "duplicate matched-output field contract");
        }
    }
    std::map<std::string, OutputFieldContract> expected_output_fields;
    for (const auto &contract :
         canonical_output_field_contracts_v1(baseline_descriptor, target)) {
        expected_output_fields.emplace(contract.field.name, contract);
    }
    if (output_fields != expected_output_fields) {
        throw baseline::ContractError(
            "matched-output declarations or operations do not match the immutable Citlali v1 registry");
    }

    std::map<std::string, baseline::RegisteredField> seed_fields;
    for (const auto &field :
         baseline_descriptor.document().registered_fields) {
        seed_fields.emplace(field.name, field);
    }
    std::map<std::int64_t, const TargetRow *> target_rows;
    for (const auto &row : target.rows) {
        target_rows.emplace(row.row_key, &row);
    }
    std::map<std::int64_t, const baseline::Row *> seed_rows;
    for (const auto &row : baseline_descriptor.document().rows) {
        seed_rows.emplace(row.uid, &row);
    }
    std::map<std::int64_t, MatchPair> pairs;
    for (const auto &pair : relation.pairs) {
        pairs.emplace(pair.pair_key, pair);
    }
    std::map<std::int64_t, EndpointDisposition> target_dispositions;
    for (const auto &disposition : relation.target_dispositions) {
        target_dispositions.emplace(disposition.endpoint.local_key,
                                    disposition);
    }

    const auto target_identity = artifact_identity(target);
    std::set<std::int64_t> output_keys;
    std::set<std::int64_t> covered_targets;
    for (const auto &row : document.rows) {
        detail::require_local_key(row.uid, "matched-output local uid");
        detail::require_row_reference(row.target);
        if (!output_keys.insert(row.uid).second ||
            row.target !=
                detail::make_row_reference(target_identity,
                                           row.target.local_key) ||
            !covered_targets.insert(row.target.local_key).second) {
            throw baseline::ContractError(
                "matched-output local uid or target reference is duplicate/foreign");
        }
        const auto target_row_it = target_rows.find(row.target.local_key);
        if (target_row_it == target_rows.end()) {
            throw baseline::ContractError(
                "matched-output row references an absent target");
        }
        const auto &target_row = *target_row_it->second;
        if (row.target_input_key != target_row.input_key ||
            row.array != target_row.array || row.network != target_row.network ||
            row.channel != target_row.channel ||
            canonical_binary64_payload(row.tone_frequency_hz) !=
                canonical_binary64_payload(
                    target_row.output_tone_frequency_hz)) {
            throw baseline::ContractError(
                "matched-output structural/raw values differ from the target");
        }
        const auto disposition =
            target_dispositions.find(row.target.local_key);
        if (disposition == target_dispositions.end() ||
            row.relation_pair_keys != disposition->second.pair_keys) {
            throw baseline::ContractError(
                "matched-output relation pair set is incomplete or reordered");
        }

        if (row.fields.size() != output_fields.size() ||
            row.transformations.size() != output_fields.size()) {
            throw baseline::ContractError(
                "matched-output row lacks exact fields/transformations");
        }
        std::map<std::string, FieldTransformation> transformations;
        for (const auto &transformation : row.transformations) {
            if (!transformations
                     .emplace(transformation.field_name, transformation)
                     .second) {
                throw baseline::ContractError(
                    "duplicate matched-output field transformation");
            }
        }
        for (const auto &[name, contract] : output_fields) {
            const auto output_value = row.fields.find(name);
            const auto transformation = transformations.find(name);
            if (output_value == row.fields.end() ||
                transformation == transformations.end() ||
                !detail::value_matches(output_value->second, contract.field) ||
                transformation->second.operation !=
                    contract.authorized_operation ||
                !detail::values_equal(transformation->second.after,
                                      output_value->second,
                                      contract.field.type)) {
                throw baseline::ContractError(
                    "matched-output field transformation is untyped or unauthorized: " +
                    name);
            }
            const auto &change = transformation->second;
            baseline::detail::require_text("transformation authority",
                                           change.authority_reference);
            baseline::detail::require_text("transformation provenance",
                                           change.provenance_reference);
            const bool matched = !row.relation_pair_keys.empty();
            if (contract.authorized_operation ==
                TransformationOperation::preserve_target) {
                const auto target_value = target_row.fields.find(name);
                if (target_value == target_row.fields.end() ||
                    !detail::values_equal(change.before,
                                          target_value->second,
                                          contract.field.type) ||
                    !detail::values_equal(change.after,
                                          target_value->second,
                                          contract.field.type) ||
                    change.value_source !=
                        TransformationValueSource::target_row ||
                    change.source_pair_key || !change.source_row ||
                    *change.source_row != row.target ||
                    change.authority_reference !=
                        contract.field.authority_reference ||
                    change.provenance_reference !=
                        "target-kmp-source:" +
                            std::to_string(target_row.kmp_source_key) +
                            ":row:" +
                            std::to_string(target_row.kmp_row_index) +
                            ":column:" +
                            *contract.field.source_column) {
                    throw baseline::ContractError(
                        "preserve-target transformation does not retain its exact target KMP value/source/authority");
                }
                continue;
            }
            if (contract.authorized_operation !=
                    TransformationOperation::
                        copy_baseline_when_matched_null_when_unmatched ||
                !std::holds_alternative<baseline::NullValue>(change.before)) {
                throw baseline::ContractError(
                    "matched-output v1 admits no caller-selected transformation operation");
            }
            if (matched) {
                if (change.value_source !=
                        TransformationValueSource::baseline_seed_row ||
                    !change.source_pair_key || !change.source_row ||
                    !std::binary_search(row.relation_pair_keys.begin(),
                                        row.relation_pair_keys.end(),
                                        *change.source_pair_key)) {
                    throw baseline::ContractError(
                        "baseline transformation lacks an exact relation pair source");
                }
                const auto pair = pairs.find(*change.source_pair_key);
                if (pair == pairs.end() || pair->second.target != row.target ||
                    pair->second.seed != *change.source_row) {
                    throw baseline::ContractError(
                        "baseline transformation pair/source endpoint mismatch");
                }
                const auto seed_row = seed_rows.find(
                    pair->second.seed.local_key);
                const auto seed_field = seed_fields.find(name);
                if (seed_row == seed_rows.end() ||
                    seed_field == seed_fields.end() ||
                    seed_field->second.type != contract.field.type ||
                    seed_field->second.unit != contract.field.unit) {
                    throw baseline::ContractError(
                        "baseline transformation field has no compatible source");
                }
                const auto seed_value = seed_row->second->fields.find(name);
                if (seed_value == seed_row->second->fields.end() ||
                    !detail::values_equal(change.after, seed_value->second,
                                          contract.field.type) ||
                    change.authority_reference !=
                        seed_field->second.authority_reference ||
                    change.provenance_reference !=
                        "relation-pair:" +
                            std::to_string(*change.source_pair_key)) {
                    throw baseline::ContractError(
                        "baseline transformation does not preserve exact source value/authority");
                }
            } else if (change.value_source !=
                           TransformationValueSource::canonical_null ||
                       change.source_pair_key || change.source_row ||
                       !std::holds_alternative<baseline::NullValue>(
                           change.after) ||
                       change.authority_reference !=
                           unmatched_missing_authority_v1 ||
                       change.provenance_reference !=
                           "target-unmatched:no-fabricated-seed") {
                throw baseline::ContractError(
                    "unmatched transformation is not the registered typed missing value");
            }
        }
    }
    if (covered_targets.size() != target.rows.size()) {
        throw baseline::ContractError(
            "matched output does not contain exactly one row per target");
    }
    detail::require_permutation(document.output_presentation_sequence,
                                output_keys,
                                "matched-output presentation sequence");
}

inline MatchedOutput make_matched_observation_output_v1(
    IssuanceEnvelope envelope,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target, const MatchRelation &relation,
    const std::vector<MatchedOutputFieldSource> &field_sources) {
    validate(relation, baseline_descriptor, target);

    MatchedOutput result;
    result.envelope = std::move(envelope);
    result.baseline_parent = baseline_reference(baseline_descriptor);
    result.target_parent = artifact_identity(target);
    result.relation_parent =
        artifact_identity(relation, baseline_descriptor, target);
    result.registered_fields =
        canonical_output_field_contracts_v1(baseline_descriptor, target);

    std::map<std::int64_t, const TargetRow *> targets;
    for (const auto &row : target.rows) {
        targets.emplace(row.row_key, &row);
    }
    std::map<std::int64_t, const baseline::Row *> seeds;
    for (const auto &row : baseline_descriptor.document().rows) {
        seeds.emplace(row.uid, &row);
    }
    std::map<std::string, baseline::RegisteredField> seed_fields;
    for (const auto &field : baseline_descriptor.document().registered_fields) {
        seed_fields.emplace(field.name, field);
    }
    std::map<std::int64_t, const MatchPair *> pairs;
    for (const auto &pair : relation.pairs) {
        pairs.emplace(pair.pair_key, &pair);
    }
    std::map<std::int64_t, const EndpointDisposition *> dispositions;
    for (const auto &disposition : relation.target_dispositions) {
        dispositions.emplace(disposition.endpoint.local_key, &disposition);
    }

    std::set<std::string> baseline_output_fields;
    for (const auto &contract : result.registered_fields) {
        if (contract.authorized_operation ==
            TransformationOperation::
                copy_baseline_when_matched_null_when_unmatched) {
            baseline_output_fields.insert(contract.field.name);
        }
    }
    std::map<std::pair<std::int64_t, std::string>,
             std::optional<std::int64_t>> selections;
    for (const auto &selection : field_sources) {
        detail::require_local_key(selection.target_row_key,
                                  "output source target key");
        if (!targets.contains(selection.target_row_key) ||
            !baseline_output_fields.contains(selection.field_name) ||
            !selections
                 .emplace(std::make_pair(selection.target_row_key,
                                         selection.field_name),
                          selection.relation_pair_key)
                 .second) {
            throw baseline::ContractError(
                "matched-output field-source selections are unknown or duplicate");
        }
        if (selection.relation_pair_key) {
            detail::require_local_key(*selection.relation_pair_key,
                                      "output source relation pair key");
        }
    }
    const auto expected_selection_count =
        targets.size() * baseline_output_fields.size();
    if (selections.size() != expected_selection_count) {
        throw baseline::ContractError(
            "matched-output field-source selections are incomplete");
    }

    std::map<std::int64_t, std::int64_t> output_uid_for_target;
    std::int64_t next_uid = 0;
    for (const auto &[target_key, target_row] : targets) {
        const auto disposition = dispositions.find(target_key);
        if (disposition == dispositions.end()) {
            throw baseline::ContractError(
                "matched-output builder cannot find target disposition");
        }
        MatchedOutputRow row;
        row.uid = next_uid++;
        output_uid_for_target.emplace(target_key, row.uid);
        row.target = detail::make_row_reference(result.target_parent, target_key);
        row.target_input_key = target_row->input_key;
        row.tone_frequency_hz = target_row->output_tone_frequency_hz;
        row.array = target_row->array;
        row.network = target_row->network;
        row.channel = target_row->channel;
        row.relation_pair_keys = disposition->second->pair_keys;

        for (const auto &contract : result.registered_fields) {
            FieldTransformation transformation;
            transformation.field_name = contract.field.name;
            transformation.operation = contract.authorized_operation;
            if (contract.authorized_operation ==
                TransformationOperation::preserve_target) {
                const auto value = target_row->fields.at(contract.field.name);
                row.fields.emplace(contract.field.name, value);
                transformation.before = value;
                transformation.after = value;
                transformation.value_source =
                    TransformationValueSource::target_row;
                transformation.source_row = row.target;
                transformation.authority_reference =
                    contract.field.authority_reference;
                transformation.provenance_reference =
                    "target-kmp-source:" +
                    std::to_string(target_row->kmp_source_key) + ":row:" +
                    std::to_string(target_row->kmp_row_index) + ":column:" +
                    *contract.field.source_column;
                row.transformations.push_back(std::move(transformation));
                continue;
            }

            transformation.before = baseline::NullValue{};
            const auto selected = selections.find(
                {target_key, contract.field.name});
            const bool matched = !row.relation_pair_keys.empty();
            if (matched != selected->second.has_value()) {
                throw baseline::ContractError(
                    "matched-output field-source selection does not match its target disposition");
            }
            if (!matched) {
                row.fields.emplace(contract.field.name, baseline::NullValue{});
                transformation.after = baseline::NullValue{};
                transformation.value_source =
                    TransformationValueSource::canonical_null;
                transformation.authority_reference =
                    unmatched_missing_authority_v1;
                transformation.provenance_reference =
                    "target-unmatched:no-fabricated-seed";
                row.transformations.push_back(std::move(transformation));
                continue;
            }

            const auto pair = pairs.find(*selected->second);
            if (pair == pairs.end() || pair->second->target != row.target ||
                !std::binary_search(row.relation_pair_keys.begin(),
                                    row.relation_pair_keys.end(),
                                    *selected->second)) {
                throw baseline::ContractError(
                    "matched-output field-source selection is not an exact target relation edge");
            }
            const auto seed = seeds.find(pair->second->seed.local_key);
            const auto seed_field = seed_fields.find(contract.field.name);
            if (seed == seeds.end() || seed_field == seed_fields.end()) {
                throw baseline::ContractError(
                    "matched-output field-source selection has no baseline value");
            }
            const auto value = seed->second->fields.at(contract.field.name);
            row.fields.emplace(contract.field.name, value);
            transformation.after = value;
            transformation.value_source =
                TransformationValueSource::baseline_seed_row;
            transformation.source_pair_key = *selected->second;
            transformation.source_row = pair->second->seed;
            transformation.authority_reference =
                seed_field->second.authority_reference;
            transformation.provenance_reference =
                "relation-pair:" + std::to_string(*selected->second);
            row.transformations.push_back(std::move(transformation));
        }
        result.rows.push_back(std::move(row));
    }

    result.output_presentation_sequence.reserve(
        target.target_application_sequence.size());
    for (const auto target_key : target.target_application_sequence) {
        result.output_presentation_sequence.push_back(
            output_uid_for_target.at(target_key));
    }
    validate(result, baseline_descriptor, target, relation);
    return result;
}

inline std::string matched_output_semantic_preimage(
    const MatchedOutput &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target, const MatchRelation &relation) {
    validate(document, baseline_descriptor, target, relation);
    std::string preimage;
    detail::add_string(preimage, "encoding", framing_encoding_v1);
    detail::add_string(preimage, "scope", matched_output_semantic_scope_v1);
    detail::add_string(preimage, "schema", document.schema);
    detail::add_string(preimage, "contract-authority",
                       document.contract_authority);
    detail::add_string(preimage, "observation-value-issuer",
                       document.observation_value_issuer);
    detail::add_string(preimage, "transformation-registry",
                       document.transformation_registry);
    detail::add_baseline_reference(preimage, "baseline-parent",
                                   document.baseline_parent);
    detail::add_artifact_identity(preimage, "target-parent",
                                  document.target_parent);
    detail::add_artifact_identity(preimage, "relation-parent",
                                  document.relation_parent);

    auto fields = document.registered_fields;
    std::sort(fields.begin(), fields.end(), [](const auto &lhs,
                                               const auto &rhs) {
        return lhs.field.name < rhs.field.name;
    });
    detail::add_uint64(preimage, "field.count", fields.size());
    for (std::size_t index = 0; index < fields.size(); ++index) {
        const auto prefix = "field." + std::to_string(index);
        detail::add_typed_field(preimage, prefix, fields[index].field);
        detail::add_string(preimage, prefix + ".authorized-operation",
                           transformation_operation_token(
                               fields[index].authorized_operation));
        detail::add_string(preimage, prefix + ".issuer-authority-reference",
                           fields[index].issuer_authority_reference);
    }

    auto rows = document.rows;
    std::sort(rows.begin(), rows.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.uid < rhs.uid;
    });
    detail::add_uint64(preimage, "row.count", rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto prefix = "row." + std::to_string(index);
        const auto &row = rows[index];
        detail::add_int64(preimage, prefix + ".uid", row.uid);
        detail::add_row_reference(preimage, prefix + ".target", row.target);
        detail::add_int64(preimage, prefix + ".target-input-key",
                          row.target_input_key);
        detail::add_float64(preimage, prefix + ".tone-frequency-hz",
                            row.tone_frequency_hz);
        detail::add_int64(preimage, prefix + ".array", row.array);
        detail::add_int64(preimage, prefix + ".network", row.network);
        detail::add_int64(preimage, prefix + ".channel", row.channel);
        detail::add_uint64(preimage, prefix + ".relation-pair-key.count",
                           row.relation_pair_keys.size());
        for (std::size_t pair_index = 0;
             pair_index < row.relation_pair_keys.size(); ++pair_index) {
            detail::add_int64(
                preimage,
                prefix + ".relation-pair-key." +
                    std::to_string(pair_index),
                row.relation_pair_keys[pair_index]);
        }
        for (const auto &field : fields) {
            detail::add_value(preimage,
                              prefix + ".field." + field.field.name,
                              row.fields.at(field.field.name),
                              field.field.type);
        }
        auto transformations = row.transformations;
        std::sort(transformations.begin(), transformations.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.field_name < rhs.field_name;
                  });
        detail::add_uint64(preimage, prefix + ".transformation.count",
                           transformations.size());
        for (std::size_t transform_index = 0;
             transform_index < transformations.size(); ++transform_index) {
            const auto transform_prefix = prefix + ".transformation." +
                std::to_string(transform_index);
            const auto &transformation = transformations[transform_index];
            const auto field = std::find_if(
                fields.begin(), fields.end(), [&](const auto &candidate) {
                    return candidate.field.name == transformation.field_name;
                });
            detail::add_string(preimage, transform_prefix + ".field-name",
                               transformation.field_name);
            detail::add_string(preimage, transform_prefix + ".operation",
                               transformation_operation_token(
                                   transformation.operation));
            detail::add_value(preimage, transform_prefix + ".before",
                              transformation.before, field->field.type);
            detail::add_value(preimage, transform_prefix + ".after",
                              transformation.after, field->field.type);
            detail::add_string(preimage, transform_prefix + ".value-source",
                               transformation_value_source_token(
                                   transformation.value_source));
            detail::add_bool(preimage,
                             transform_prefix + ".has-source-pair-key",
                             transformation.source_pair_key.has_value());
            if (transformation.source_pair_key) {
                detail::add_int64(preimage,
                                  transform_prefix + ".source-pair-key",
                                  *transformation.source_pair_key);
            }
            detail::add_bool(preimage, transform_prefix + ".has-source-row",
                             transformation.source_row.has_value());
            if (transformation.source_row) {
                detail::add_row_reference(preimage,
                                          transform_prefix + ".source-row",
                                          *transformation.source_row);
            }
            detail::add_string(preimage,
                               transform_prefix + ".authority-reference",
                               transformation.authority_reference);
            detail::add_string(preimage,
                               transform_prefix + ".provenance-reference",
                               transformation.provenance_reference);
        }
    }
    detail::add_uint64(preimage, "output-presentation-sequence.count",
                       document.output_presentation_sequence.size());
    for (std::size_t index = 0;
         index < document.output_presentation_sequence.size(); ++index) {
        detail::add_int64(
            preimage,
            "output-presentation-sequence." + std::to_string(index),
            document.output_presentation_sequence[index]);
    }
    return preimage;
}

inline std::string matched_output_semantic_sha256(
    const MatchedOutput &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target, const MatchRelation &relation) {
    return "sha256:" + citlali::utils::sha256(matched_output_semantic_preimage(
        document, baseline_descriptor, target, relation));
}

inline std::string matched_output_envelope_preimage(
    const MatchedOutput &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target, const MatchRelation &relation) {
    validate(document, baseline_descriptor, target, relation);
    std::string preimage;
    detail::add_envelope(
        preimage, document.schema, matched_output_envelope_scope_v1,
        matched_output_semantic_sha256(document, baseline_descriptor, target,
                                       relation),
        document.envelope);
    return preimage;
}

inline std::string matched_output_envelope_sha256(
    const MatchedOutput &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target, const MatchRelation &relation) {
    return "sha256:" + citlali::utils::sha256(matched_output_envelope_preimage(
        document, baseline_descriptor, target, relation));
}

inline baseline::Digests compute_digests(
    const MatchedOutput &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target, const MatchRelation &relation) {
    return {matched_output_semantic_sha256(document, baseline_descriptor,
                                           target, relation),
            matched_output_envelope_sha256(document, baseline_descriptor,
                                           target, relation)};
}

inline ArtifactIdentity artifact_identity(
    const MatchedOutput &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target, const MatchRelation &relation) {
    const auto digests =
        compute_digests(document, baseline_descriptor, target, relation);
    return {document.schema, document.envelope.occurrence,
            digests.semantic_sha256, digests.envelope_sha256};
}

namespace observation_ecsv_detail {

using ColumnDeclaration = baseline::ecsv_detail::ColumnDeclaration;

inline std::vector<OutputFieldContract> sorted_output_fields(
    const MatchedOutput &document) {
    auto fields = document.registered_fields;
    std::sort(fields.begin(), fields.end(), [](const auto &lhs,
                                               const auto &rhs) {
        return lhs.field.name < rhs.field.name;
    });
    return fields;
}

inline std::vector<ColumnDeclaration> expected_columns(
    const MatchedOutput &document) {
    std::vector<ColumnDeclaration> columns{
        {"uid", "int64", "N/A",
         "exact nonnegative output-artifact-local row key; never persistent detector identity"},
        {"target_row_key", "int64", "N/A",
         "exact target-parent artifact-local row reference"},
        {"target_input_key", "int64", "N/A",
         "exact target-parent input binding reference"},
        {"tone_freq", "float64", "Hz",
         "exact target kids_f_out application value; not identity"},
        {"array", "int64", "N/A",
         "canonical TolTEC array enum; not row identity"},
        {"nw", "int64", "N/A", "target raw-manifest network key"},
        {"kids_tone", "int64", "N/A",
         "zero-based target raw channel key within network"},
        {"relation_pair_keys", "string", "N/A",
         "complete sorted relation-local pair-key set using bracketed-int64-set-v1"},
    };
    for (const auto &contract : sorted_output_fields(document)) {
        columns.push_back(
            {contract.field.name,
             std::string(baseline::value_type_token(contract.field.type)),
             contract.field.unit, contract.field.description});
    }
    return columns;
}

inline std::string pair_key_set_cell(
    const std::vector<std::int64_t> &keys) {
    detail::require_sorted_unique_pair_keys(keys,
                                            "ECSV relation pair-key set");
    std::string result{"["};
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (index != 0) {
            result += ',';
        }
        result += std::to_string(keys[index]);
    }
    result += ']';
    return result;
}

inline std::vector<std::int64_t> parse_pair_key_set_cell(
    std::string_view value) {
    const std::string original(value);
    if (value.size() < 2 || value.front() != '[' || value.back() != ']') {
        throw baseline::ContractError(
            "matched observation ECSV pair-key set has invalid framing");
    }
    std::vector<std::int64_t> result;
    value.remove_prefix(1);
    value.remove_suffix(1);
    while (!value.empty()) {
        const auto separator = value.find(',');
        const auto item = value.substr(0, separator);
        result.push_back(baseline::ecsv_detail::parse_int64(
            item, "relation_pair_keys"));
        if (separator == std::string_view::npos) {
            value = {};
        } else {
            value.remove_prefix(separator + 1);
        }
    }
    detail::require_sorted_unique_pair_keys(result,
                                            "ECSV relation pair-key set");
    if (pair_key_set_cell(result) != original) {
        throw baseline::ContractError(
            "matched observation ECSV pair-key set is noncanonical");
    }
    return result;
}

inline double parse_binary64_payload(std::string_view token,
                                     std::string_view label) {
    if (token == "nan") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (token == "+inf") {
        return std::numeric_limits<double>::infinity();
    }
    if (token == "-inf") {
        return -std::numeric_limits<double>::infinity();
    }
    if (token.size() != 16) {
        throw baseline::ContractError(
            "invalid exact binary64 metadata token: " +
            std::string(label));
    }
    std::uint64_t bits = 0;
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(),
                                        bits, 16);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != token.data() + token.size() ||
        canonical_binary64_payload(std::bit_cast<double>(bits)) != token) {
        throw baseline::ContractError(
            "noncanonical exact binary64 metadata token: " +
            std::string(label));
    }
    return std::bit_cast<double>(bits);
}

inline std::string value_token(const baseline::Value &value,
                               baseline::ValueType type) {
    return std::visit(
        [&](const auto &typed) -> std::string {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, baseline::NullValue>) {
                return "null";
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                return std::to_string(typed);
            } else if constexpr (std::is_same_v<T, double>) {
                return canonical_binary64_payload(typed);
            } else if constexpr (std::is_same_v<T, bool>) {
                return typed ? "true" : "false";
            } else {
                if (type != baseline::ValueType::string) {
                    throw baseline::ContractError(
                        "matched observation metadata value type mismatch");
                }
                return typed;
            }
        },
        value);
}

inline baseline::Value parse_value_token(std::string_view token,
                                         baseline::ValueType type,
                                         bool is_null,
                                         std::string_view label) {
    if (is_null) {
        if (token != "null") {
            throw baseline::ContractError(
                "matched observation null metadata token mismatch");
        }
        return baseline::NullValue{};
    }
    switch (type) {
    case baseline::ValueType::int64:
        return baseline::ecsv_detail::parse_int64(token, label);
    case baseline::ValueType::float64:
        return parse_binary64_payload(token, label);
    case baseline::ValueType::boolean:
        if (token == "true") {
            return true;
        }
        if (token == "false") {
            return false;
        }
        throw baseline::ContractError(
            "invalid matched observation Boolean metadata token");
    case baseline::ValueType::string:
        baseline::detail::require_text(label, std::string(token));
        return std::string(token);
    }
    throw baseline::ContractError(
        "unsupported matched observation metadata value type");
}

inline std::uint64_t parse_uint64(std::string_view value,
                                  std::string_view label) {
    if (value.empty()) {
        throw baseline::ContractError("missing uint64 metadata value: " +
                                      std::string(label));
    }
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(),
                                        result, 10);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != value.data() + value.size() ||
        std::to_string(result) != value) {
        throw baseline::ContractError("invalid canonical uint64 metadata: " +
                                      std::string(label));
    }
    return result;
}

class MetadataReader {
public:
    MetadataReader(const std::vector<std::string_view> &lines,
                   std::size_t index)
        : lines_(lines), index_(index) {}

    void expect(std::string_view exact) {
        if (index_ >= lines_.size() || lines_[index_] != exact) {
            throw baseline::ContractError(
                "matched observation ECSV metadata is absent, reordered, or noncanonical");
        }
        ++index_;
    }

    std::string quoted(std::string_view prefix) {
        const auto value = suffix(prefix);
        return baseline::ecsv_detail::yaml_unquote(value);
    }

    std::int64_t int64(std::string_view prefix) {
        return baseline::ecsv_detail::parse_int64(suffix(prefix), prefix);
    }

    std::uint64_t uint64(std::string_view prefix) {
        return parse_uint64(suffix(prefix), prefix);
    }

    bool boolean(std::string_view prefix) {
        return baseline::ecsv_detail::parse_metadata_bool(suffix(prefix));
    }

    double float64(std::string_view prefix) {
        return parse_binary64_payload(quoted(prefix), prefix);
    }

    std::size_t index() const noexcept { return index_; }

private:
    std::string_view suffix(std::string_view prefix) {
        if (index_ >= lines_.size() || !lines_[index_].starts_with(prefix)) {
            throw baseline::ContractError(
                "matched observation ECSV metadata field is absent or reordered");
        }
        const auto result = lines_[index_].substr(prefix.size());
        ++index_;
        return result;
    }

    const std::vector<std::string_view> &lines_;
    std::size_t index_ = 0;
};

inline void emit_quoted(std::ostringstream &stream, std::string_view prefix,
                        std::string_view value) {
    stream << prefix << baseline::ecsv_detail::yaml_quote(value) << '\n';
}

inline void emit_int64(std::ostringstream &stream, std::string_view prefix,
                       std::int64_t value) {
    stream << prefix << value << '\n';
}

inline void emit_uint64(std::ostringstream &stream, std::string_view prefix,
                        std::uint64_t value) {
    stream << prefix << value << '\n';
}

inline void emit_bool(std::ostringstream &stream, std::string_view prefix,
                      bool value) {
    stream << prefix << (value ? "true" : "false") << '\n';
}

inline void emit_float64(std::ostringstream &stream, std::string_view prefix,
                         double value) {
    emit_quoted(stream, prefix, canonical_binary64_payload(value));
}

inline void emit_envelope(std::ostringstream &stream,
                          std::string_view indent,
                          const IssuanceEnvelope &value) {
    emit_quoted(stream, std::string(indent) + "occurrence: ",
                value.occurrence);
    emit_quoted(stream, std::string(indent) + "event_reference: ",
                value.event_reference);
    emit_quoted(stream, std::string(indent) + "software_revision: ",
                value.software_revision);
    emit_quoted(stream, std::string(indent) + "configuration_reference: ",
                value.configuration_reference);
    emit_quoted(stream, std::string(indent) + "event_time_utc: ",
                value.event_time_utc);
}

inline IssuanceEnvelope parse_envelope(MetadataReader &reader,
                                       std::string_view indent) {
    IssuanceEnvelope result;
    result.occurrence = reader.quoted(std::string(indent) + "occurrence: ");
    result.event_reference =
        reader.quoted(std::string(indent) + "event_reference: ");
    result.software_revision =
        reader.quoted(std::string(indent) + "software_revision: ");
    result.configuration_reference =
        reader.quoted(std::string(indent) + "configuration_reference: ");
    result.event_time_utc =
        reader.quoted(std::string(indent) + "event_time_utc: ");
    return result;
}

inline void emit_observation(std::ostringstream &stream,
                             std::string_view indent,
                             const baseline::ObservationIdentity &value) {
    emit_int64(stream, std::string(indent) + "observation: ",
               value.observation);
    emit_int64(stream, std::string(indent) + "subobservation: ",
               value.subobservation);
    emit_int64(stream, std::string(indent) + "scan: ", value.scan);
}

inline baseline::ObservationIdentity parse_observation(
    MetadataReader &reader, std::string_view indent) {
    return {reader.int64(std::string(indent) + "observation: "),
            reader.int64(std::string(indent) + "subobservation: "),
            reader.int64(std::string(indent) + "scan: ")};
}

inline void emit_artifact_identity(std::ostringstream &stream,
                                   std::string_view indent,
                                   const ArtifactIdentity &value) {
    emit_quoted(stream, std::string(indent) + "schema: ", value.schema);
    emit_quoted(stream, std::string(indent) + "occurrence: ",
                value.occurrence);
    emit_quoted(stream, std::string(indent) + "semantic_sha256: ",
                value.semantic_sha256);
    emit_quoted(stream, std::string(indent) + "envelope_sha256: ",
                value.envelope_sha256);
}

inline ArtifactIdentity parse_artifact_identity(MetadataReader &reader,
                                                std::string_view indent) {
    return {reader.quoted(std::string(indent) + "schema: "),
            reader.quoted(std::string(indent) + "occurrence: "),
            reader.quoted(std::string(indent) + "semantic_sha256: "),
            reader.quoted(std::string(indent) + "envelope_sha256: ")};
}

inline void emit_baseline_reference(
    std::ostringstream &stream, std::string_view indent,
    const VerifiedBaselineReference &value) {
    stream << indent << "artifact:\n";
    emit_artifact_identity(stream, std::string(indent) + "  ",
                           value.artifact);
    emit_quoted(stream, std::string(indent) + "profile: ", value.profile);
    emit_quoted(stream, std::string(indent) + "descriptor_sha256: ",
                value.descriptor_sha256);
    emit_quoted(stream, std::string(indent) + "transport_scope: ",
                value.transport_scope);
    emit_quoted(stream, std::string(indent) + "transport_sha256: ",
                value.transport_sha256);
    emit_uint64(stream, std::string(indent) + "byte_count: ",
                value.byte_count);
    emit_quoted(stream, std::string(indent) + "receipt_sha256: ",
                value.receipt_sha256);
    emit_uint64(stream, std::string(indent) + "receipt_byte_count: ",
                value.receipt_byte_count);
}

inline VerifiedBaselineReference parse_baseline_reference(
    MetadataReader &reader, std::string_view indent) {
    reader.expect(std::string(indent) + "artifact:");
    VerifiedBaselineReference result;
    result.artifact =
        parse_artifact_identity(reader, std::string(indent) + "  ");
    result.profile = reader.quoted(std::string(indent) + "profile: ");
    result.descriptor_sha256 =
        reader.quoted(std::string(indent) + "descriptor_sha256: ");
    result.transport_scope =
        reader.quoted(std::string(indent) + "transport_scope: ");
    result.transport_sha256 =
        reader.quoted(std::string(indent) + "transport_sha256: ");
    result.byte_count = reader.uint64(std::string(indent) + "byte_count: ");
    result.receipt_sha256 =
        reader.quoted(std::string(indent) + "receipt_sha256: ");
    result.receipt_byte_count =
        reader.uint64(std::string(indent) + "receipt_byte_count: ");
    return result;
}

inline void emit_row_reference(std::ostringstream &stream,
                               std::string_view indent,
                               const RowReference &value) {
    emit_quoted(stream, std::string(indent) + "artifact_schema: ",
                value.artifact_schema);
    emit_quoted(stream, std::string(indent) + "occurrence: ",
                value.occurrence);
    emit_quoted(stream, std::string(indent) + "envelope_sha256: ",
                value.envelope_sha256);
    emit_int64(stream, std::string(indent) + "local_key: ", value.local_key);
}

inline RowReference parse_row_reference(MetadataReader &reader,
                                        std::string_view indent) {
    return {reader.quoted(std::string(indent) + "artifact_schema: "),
            reader.quoted(std::string(indent) + "occurrence: "),
            reader.quoted(std::string(indent) + "envelope_sha256: "),
            reader.int64(std::string(indent) + "local_key: ")};
}

inline void emit_typed_field(std::ostringstream &stream,
                             std::string_view first_indent,
                             std::string_view indent,
                             const TypedField &field) {
    emit_quoted(stream, std::string(first_indent) + "- name: ", field.name);
    emit_quoted(stream, std::string(indent) + "datatype: ",
                baseline::value_type_token(field.type));
    emit_quoted(stream, std::string(indent) + "unit: ", field.unit);
    emit_bool(stream, std::string(indent) + "nullable: ", field.nullable);
    emit_quoted(stream, std::string(indent) + "nonfinite: ",
                baseline::nonfinite_policy_token(field.nonfinite));
    emit_quoted(stream, std::string(indent) + "authority: ",
                field.authority);
    emit_quoted(stream, std::string(indent) + "authority_reference: ",
                field.authority_reference);
    emit_quoted(stream, std::string(indent) + "registry: ", field.registry);
    emit_quoted(stream, std::string(indent) + "description: ",
                field.description);
    emit_bool(stream, std::string(indent) + "has_source_column: ",
              field.source_column.has_value());
    if (field.source_column) {
        emit_quoted(stream, std::string(indent) + "source_column: ",
                    *field.source_column);
    }
    emit_quoted(stream, std::string(indent) + "identity_role: ",
                field.identity_role);
}

inline TypedField parse_typed_field(MetadataReader &reader,
                                    std::string_view first_indent,
                                    std::string_view indent) {
    TypedField field;
    field.name = reader.quoted(std::string(first_indent) + "- name: ");
    field.type = baseline::parse_value_type_token(
        reader.quoted(std::string(indent) + "datatype: "));
    field.unit = reader.quoted(std::string(indent) + "unit: ");
    field.nullable = reader.boolean(std::string(indent) + "nullable: ");
    field.nonfinite = baseline::parse_nonfinite_policy_token(
        reader.quoted(std::string(indent) + "nonfinite: "));
    field.authority = reader.quoted(std::string(indent) + "authority: ");
    field.authority_reference =
        reader.quoted(std::string(indent) + "authority_reference: ");
    field.registry = reader.quoted(std::string(indent) + "registry: ");
    field.description =
        reader.quoted(std::string(indent) + "description: ");
    const bool has_source =
        reader.boolean(std::string(indent) + "has_source_column: ");
    if (has_source) {
        field.source_column =
            reader.quoted(std::string(indent) + "source_column: ");
    }
    field.identity_role =
        reader.quoted(std::string(indent) + "identity_role: ");
    return field;
}

inline void emit_source_artifact(std::ostringstream &stream,
                                 std::string_view heading_indent,
                                 std::string_view value_indent,
                                 std::string_view name,
                                 const SourceArtifact &source) {
    stream << heading_indent << name << ":\n";
    emit_int64(stream, std::string(value_indent) + "source_key: ",
               source.source_key);
    emit_quoted(stream, std::string(value_indent) + "role: ", source.role);
    emit_quoted(stream, std::string(value_indent) + "diagnostic_locator: ",
                source.diagnostic_locator);
    emit_quoted(stream, std::string(value_indent) + "content_sha256: ",
                source.content_sha256);
    emit_uint64(stream, std::string(value_indent) + "byte_count: ",
                source.byte_count);
    stream << value_indent << "header_observation:\n";
    emit_observation(stream, std::string(value_indent) + "  ",
                     source.header_observation);
    emit_int64(stream, std::string(value_indent) + "network: ",
               source.network);
    emit_quoted(stream, std::string(value_indent) + "interface: ",
                source.interface_name);
    emit_int64(stream, std::string(value_indent) + "channel_count: ",
               source.channel_count);
}

inline SourceArtifact parse_source_artifact(
    MetadataReader &reader, std::string_view heading_indent,
    std::string_view value_indent, std::string_view name) {
    reader.expect(std::string(heading_indent) + std::string(name) + ":");
    SourceArtifact source;
    source.source_key =
        reader.int64(std::string(value_indent) + "source_key: ");
    source.role = reader.quoted(std::string(value_indent) + "role: ");
    source.diagnostic_locator =
        reader.quoted(std::string(value_indent) + "diagnostic_locator: ");
    source.content_sha256 =
        reader.quoted(std::string(value_indent) + "content_sha256: ");
    source.byte_count =
        reader.uint64(std::string(value_indent) + "byte_count: ");
    reader.expect(std::string(value_indent) + "header_observation:");
    source.header_observation =
        parse_observation(reader, std::string(value_indent) + "  ");
    source.network = reader.int64(std::string(value_indent) + "network: ");
    source.interface_name =
        reader.quoted(std::string(value_indent) + "interface: ");
    source.channel_count =
        reader.int64(std::string(value_indent) + "channel_count: ");
    return source;
}

inline void emit_target(std::ostringstream &stream,
                        const TargetManifest &document) {
    validate(document);
    stream << "#     embedded_target:\n";
    emit_quoted(stream, "#       schema: ", document.schema);
    emit_quoted(stream, "#       contract_authority: ",
                document.contract_authority);
    emit_quoted(stream, "#       observation_value_issuer: ",
                document.observation_value_issuer);
    stream << "#       envelope:\n";
    emit_envelope(stream, "#         ", document.envelope);
    stream << "#       observation:\n";
    emit_observation(stream, "#         ", document.observation);

    auto fields = document.registered_fields;
    std::sort(fields.begin(), fields.end(), [](const auto &lhs,
                                               const auto &rhs) {
        return lhs.name < rhs.name;
    });
    emit_uint64(stream, "#       registered_field_count: ", fields.size());
    stream << "#       registered_fields:\n";
    for (const auto &field : fields) {
        emit_typed_field(stream, "#         ", "#           ", field);
    }

    auto inputs = document.inputs;
    std::sort(inputs.begin(), inputs.end(), [](const auto &lhs,
                                               const auto &rhs) {
        return lhs.input_key < rhs.input_key;
    });
    emit_uint64(stream, "#       input_count: ", inputs.size());
    stream << "#       inputs:\n";
    for (const auto &input : inputs) {
        emit_int64(stream, "#         - input_key: ", input.input_key);
        emit_int64(stream, "#           network: ", input.network);
        emit_quoted(stream, "#           interface: ", input.interface_name);
        emit_int64(stream, "#           channel_count: ",
                   input.channel_count);
        emit_source_artifact(stream, "#           ", "#             ",
                             "raw_source", input.raw_source);
        emit_source_artifact(stream, "#           ", "#             ",
                             "kmp_source", input.kmp_source);
    }

    auto rows = document.rows;
    std::sort(rows.begin(), rows.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.row_key < rhs.row_key;
    });
    emit_uint64(stream, "#       row_count: ", rows.size());
    stream << "#       rows:\n";
    for (const auto &row : rows) {
        emit_int64(stream, "#         - row_key: ", row.row_key);
        emit_int64(stream, "#           input_key: ", row.input_key);
        emit_int64(stream, "#           kmp_source_key: ",
                   row.kmp_source_key);
        emit_int64(stream, "#           kmp_row_index: ",
                   row.kmp_row_index);
        emit_float64(stream, "#           matching_frequency_hz: ",
                     row.matching_frequency_hz);
        emit_float64(stream, "#           output_tone_frequency_hz: ",
                     row.output_tone_frequency_hz);
        emit_int64(stream, "#           array: ", row.array);
        emit_int64(stream, "#           network: ", row.network);
        emit_int64(stream, "#           channel: ", row.channel);
        stream << "#           fields:\n";
        for (const auto &field : fields) {
            emit_quoted(stream, "#             " + field.name + ": ",
                        value_token(row.fields.at(field.name), field.type));
        }
    }

    emit_uint64(stream, "#       target_source_sequence_count: ",
                document.target_source_sequence.size());
    stream << "#       target_source_sequence:\n";
    for (const auto key : document.target_source_sequence) {
        emit_int64(stream, "#         - ", key);
    }
    emit_uint64(stream, "#       target_application_sequence_count: ",
                document.target_application_sequence.size());
    stream << "#       target_application_sequence:\n";
    for (const auto key : document.target_application_sequence) {
        emit_int64(stream, "#         - ", key);
    }
}

inline TargetManifest parse_target(MetadataReader &reader) {
    reader.expect("#     embedded_target:");
    TargetManifest document;
    document.schema = reader.quoted("#       schema: ");
    document.contract_authority =
        reader.quoted("#       contract_authority: ");
    document.observation_value_issuer =
        reader.quoted("#       observation_value_issuer: ");
    reader.expect("#       envelope:");
    document.envelope = parse_envelope(reader, "#         ");
    reader.expect("#       observation:");
    document.observation = parse_observation(reader, "#         ");

    const auto field_count = reader.uint64("#       registered_field_count: ");
    reader.expect("#       registered_fields:");
    for (std::uint64_t index = 0; index < field_count; ++index) {
        document.registered_fields.push_back(
            parse_typed_field(reader, "#         ", "#           "));
    }
    std::map<std::string, TypedField> fields;
    for (const auto &field : document.registered_fields) {
        if (!fields.emplace(field.name, field).second) {
            throw baseline::ContractError(
                "embedded target has duplicate typed fields");
        }
    }

    const auto input_count = reader.uint64("#       input_count: ");
    reader.expect("#       inputs:");
    for (std::uint64_t index = 0; index < input_count; ++index) {
        TargetInput input;
        input.input_key = reader.int64("#         - input_key: ");
        input.network = reader.int64("#           network: ");
        input.interface_name = reader.quoted("#           interface: ");
        input.channel_count = reader.int64("#           channel_count: ");
        input.raw_source = parse_source_artifact(
            reader, "#           ", "#             ", "raw_source");
        input.kmp_source = parse_source_artifact(
            reader, "#           ", "#             ", "kmp_source");
        document.inputs.push_back(std::move(input));
    }

    const auto row_count = reader.uint64("#       row_count: ");
    reader.expect("#       rows:");
    for (std::uint64_t index = 0; index < row_count; ++index) {
        TargetRow row;
        row.row_key = reader.int64("#         - row_key: ");
        row.input_key = reader.int64("#           input_key: ");
        row.kmp_source_key = reader.int64("#           kmp_source_key: ");
        row.kmp_row_index = reader.int64("#           kmp_row_index: ");
        row.matching_frequency_hz =
            reader.float64("#           matching_frequency_hz: ");
        row.output_tone_frequency_hz =
            reader.float64("#           output_tone_frequency_hz: ");
        row.array = reader.int64("#           array: ");
        row.network = reader.int64("#           network: ");
        row.channel = reader.int64("#           channel: ");
        reader.expect("#           fields:");
        for (const auto &[name, field] : fields) {
            const auto token =
                reader.quoted("#             " + name + ": ");
            row.fields.emplace(name, parse_value_token(
                                         token, field.type, false, name));
        }
        document.rows.push_back(std::move(row));
    }

    const auto source_count =
        reader.uint64("#       target_source_sequence_count: ");
    reader.expect("#       target_source_sequence:");
    for (std::uint64_t index = 0; index < source_count; ++index) {
        document.target_source_sequence.push_back(reader.int64("#         - "));
    }
    const auto application_count =
        reader.uint64("#       target_application_sequence_count: ");
    reader.expect("#       target_application_sequence:");
    for (std::uint64_t index = 0; index < application_count; ++index) {
        document.target_application_sequence.push_back(
            reader.int64("#         - "));
    }
    validate(document);
    return document;
}

inline EndpointDispositionState parse_endpoint_disposition_token(
    std::string_view token) {
    if (token == "matched") {
        return EndpointDispositionState::matched;
    }
    if (token == "unmatched") {
        return EndpointDispositionState::unmatched;
    }
    if (token == "unused") {
        return EndpointDispositionState::unused;
    }
    throw baseline::ContractError(
        "unsupported embedded relation endpoint disposition token");
}

inline TransformationOperation parse_transformation_operation_token(
    std::string_view token) {
    if (token == "preserve-target") {
        return TransformationOperation::preserve_target;
    }
    if (token ==
        "copy-baseline-when-matched-preserve-target-when-unmatched") {
        return TransformationOperation::
            copy_baseline_when_matched_preserve_target_when_unmatched;
    }
    if (token == "copy-baseline-when-matched-null-when-unmatched") {
        return TransformationOperation::
            copy_baseline_when_matched_null_when_unmatched;
    }
    if (token == "issuer-declared") {
        return TransformationOperation::issuer_declared;
    }
    throw baseline::ContractError(
        "unsupported matched-output transformation operation token");
}

inline TransformationValueSource parse_transformation_value_source_token(
    std::string_view token) {
    if (token == "target-row") {
        return TransformationValueSource::target_row;
    }
    if (token == "baseline-seed-row") {
        return TransformationValueSource::baseline_seed_row;
    }
    if (token == "observation-value-issuer") {
        return TransformationValueSource::observation_value_issuer;
    }
    if (token == "canonical-null") {
        return TransformationValueSource::canonical_null;
    }
    throw baseline::ContractError(
        "unsupported matched-output transformation value-source token");
}

inline void emit_relation(std::ostringstream &stream,
                          const MatchRelation &document,
                          const VerifiedBaselineDescriptor &descriptor,
                          const TargetManifest &target) {
    validate(document, descriptor, target);
    stream << "#     embedded_relation:\n";
    emit_quoted(stream, "#       schema: ", document.schema);
    emit_quoted(stream, "#       contract_authority: ",
                document.contract_authority);
    emit_quoted(stream, "#       observation_value_issuer: ",
                document.observation_value_issuer);
    emit_quoted(stream, "#       mapping_domain: ", document.mapping_domain);
    stream << "#       envelope:\n";
    emit_envelope(stream, "#         ", document.envelope);
    stream << "#       baseline_parent:\n";
    emit_baseline_reference(stream, "#         ", document.baseline_parent);
    stream << "#       target_parent:\n";
    emit_artifact_identity(stream, "#         ", document.target_parent);
    stream << "#       matcher:\n";
    emit_quoted(stream, "#         matcher_run_occurrence: ",
                document.matcher.matcher_run_occurrence);
    emit_quoted(stream, "#         implementation_revision: ",
                document.matcher.implementation_revision);
    emit_quoted(stream, "#         configuration_reference: ",
                document.matcher.configuration_reference);
    emit_quoted(stream, "#         method: ", document.matcher.method);
    emit_quoted(stream, "#         backend: ", document.matcher.backend);
    emit_quoted(stream, "#         target_frequency_field: ",
                document.matcher.target_frequency_field);
    emit_quoted(stream, "#         target_quality_factor_field: ",
                document.matcher.target_quality_factor_field);

    auto evidence = document.network_evidence;
    std::sort(evidence.begin(), evidence.end(), [](const auto &lhs,
                                                   const auto &rhs) {
        return lhs.network < rhs.network;
    });
    emit_uint64(stream, "#       network_evidence_count: ", evidence.size());
    stream << "#       network_evidence:\n";
    for (const auto &item : evidence) {
        emit_int64(stream, "#         - network: ", item.network);
        emit_float64(stream, "#           frequency_shift_hz: ",
                     item.frequency_shift_hz);
        emit_float64(stream, "#           gate_hz: ", item.gate_hz);
        emit_float64(stream, "#           quality_factor: ",
                     item.quality_factor);
        emit_quoted(stream, "#           quality_factor_field: ",
                    item.quality_factor_field);
        emit_quoted(stream,
                    "#           quality_factor_authority_reference: ",
                    item.quality_factor_authority_reference);
    }

    auto pairs = document.pairs;
    std::sort(pairs.begin(), pairs.end(), [](const auto &lhs,
                                             const auto &rhs) {
        return lhs.pair_key < rhs.pair_key;
    });
    emit_uint64(stream, "#       pair_count: ", pairs.size());
    stream << "#       pairs:\n";
    for (const auto &pair : pairs) {
        emit_int64(stream, "#         - pair_key: ", pair.pair_key);
        stream << "#           target:\n";
        emit_row_reference(stream, "#             ", pair.target);
        stream << "#           seed:\n";
        emit_row_reference(stream, "#             ", pair.seed);
        emit_float64(stream, "#           separation_hz: ",
                     pair.separation_hz);
        emit_bool(stream, "#           is_good_match: ",
                  pair.is_good_match);
    }

    const auto emit_dispositions = [&](std::string_view name,
                                       auto dispositions) {
        std::sort(dispositions.begin(), dispositions.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.disposition_key < rhs.disposition_key;
                  });
        emit_uint64(stream, "#       " + std::string(name) + "_count: ",
                    dispositions.size());
        stream << "#       " << name << ":\n";
        for (const auto &disposition : dispositions) {
            emit_int64(stream, "#         - disposition_key: ",
                       disposition.disposition_key);
            stream << "#           endpoint:\n";
            emit_row_reference(stream, "#             ",
                               disposition.endpoint);
            emit_quoted(stream, "#           state: ",
                        endpoint_disposition_token(disposition.state));
            emit_quoted(stream, "#           pair_keys: ",
                        pair_key_set_cell(disposition.pair_keys));
            emit_quoted(stream, "#           reason: ", disposition.reason);
        }
    };
    emit_dispositions("target_dispositions", document.target_dispositions);
    emit_dispositions("seed_dispositions", document.seed_dispositions);
    emit_uint64(stream, "#       seed_source_sequence_count: ",
                document.seed_source_sequence.size());
    stream << "#       seed_source_sequence:\n";
    for (const auto key : document.seed_source_sequence) {
        emit_int64(stream, "#         - ", key);
    }
}

inline MatchRelation parse_relation(
    MetadataReader &reader, const VerifiedBaselineDescriptor &descriptor,
    const TargetManifest &target) {
    reader.expect("#     embedded_relation:");
    MatchRelation document;
    document.schema = reader.quoted("#       schema: ");
    document.contract_authority =
        reader.quoted("#       contract_authority: ");
    document.observation_value_issuer =
        reader.quoted("#       observation_value_issuer: ");
    document.mapping_domain = reader.quoted("#       mapping_domain: ");
    reader.expect("#       envelope:");
    document.envelope = parse_envelope(reader, "#         ");
    reader.expect("#       baseline_parent:");
    document.baseline_parent =
        parse_baseline_reference(reader, "#         ");
    reader.expect("#       target_parent:");
    document.target_parent =
        parse_artifact_identity(reader, "#         ");
    reader.expect("#       matcher:");
    document.matcher.matcher_run_occurrence =
        reader.quoted("#         matcher_run_occurrence: ");
    document.matcher.implementation_revision =
        reader.quoted("#         implementation_revision: ");
    document.matcher.configuration_reference =
        reader.quoted("#         configuration_reference: ");
    document.matcher.method = reader.quoted("#         method: ");
    document.matcher.backend = reader.quoted("#         backend: ");
    document.matcher.target_frequency_field =
        reader.quoted("#         target_frequency_field: ");
    document.matcher.target_quality_factor_field =
        reader.quoted("#         target_quality_factor_field: ");

    const auto evidence_count =
        reader.uint64("#       network_evidence_count: ");
    reader.expect("#       network_evidence:");
    for (std::uint64_t index = 0; index < evidence_count; ++index) {
        NetworkMatchEvidence evidence;
        evidence.network = reader.int64("#         - network: ");
        evidence.frequency_shift_hz =
            reader.float64("#           frequency_shift_hz: ");
        evidence.gate_hz = reader.float64("#           gate_hz: ");
        evidence.quality_factor =
            reader.float64("#           quality_factor: ");
        evidence.quality_factor_field =
            reader.quoted("#           quality_factor_field: ");
        evidence.quality_factor_authority_reference = reader.quoted(
            "#           quality_factor_authority_reference: ");
        document.network_evidence.push_back(std::move(evidence));
    }

    const auto pair_count = reader.uint64("#       pair_count: ");
    reader.expect("#       pairs:");
    for (std::uint64_t index = 0; index < pair_count; ++index) {
        MatchPair pair;
        pair.pair_key = reader.int64("#         - pair_key: ");
        reader.expect("#           target:");
        pair.target = parse_row_reference(reader, "#             ");
        reader.expect("#           seed:");
        pair.seed = parse_row_reference(reader, "#             ");
        pair.separation_hz =
            reader.float64("#           separation_hz: ");
        pair.is_good_match =
            reader.boolean("#           is_good_match: ");
        document.pairs.push_back(std::move(pair));
    }

    const auto parse_dispositions = [&](std::string_view name) {
        std::vector<EndpointDisposition> result;
        const auto count = reader.uint64(
            "#       " + std::string(name) + "_count: ");
        reader.expect("#       " + std::string(name) + ":");
        for (std::uint64_t index = 0; index < count; ++index) {
            EndpointDisposition disposition;
            disposition.disposition_key =
                reader.int64("#         - disposition_key: ");
            reader.expect("#           endpoint:");
            disposition.endpoint =
                parse_row_reference(reader, "#             ");
            disposition.state = parse_endpoint_disposition_token(
                reader.quoted("#           state: "));
            disposition.pair_keys = parse_pair_key_set_cell(
                reader.quoted("#           pair_keys: "));
            disposition.reason = reader.quoted("#           reason: ");
            result.push_back(std::move(disposition));
        }
        return result;
    };
    document.target_dispositions =
        parse_dispositions("target_dispositions");
    document.seed_dispositions = parse_dispositions("seed_dispositions");
    const auto source_count =
        reader.uint64("#       seed_source_sequence_count: ");
    reader.expect("#       seed_source_sequence:");
    for (std::uint64_t index = 0; index < source_count; ++index) {
        document.seed_source_sequence.push_back(reader.int64("#         - "));
    }
    validate(document, descriptor, target);
    return document;
}

inline void emit_output_metadata(std::ostringstream &stream,
                                 const MatchedOutput &document) {
    stream << "#     output:\n";
    emit_quoted(stream, "#       schema: ", document.schema);
    emit_quoted(stream, "#       contract_authority: ",
                document.contract_authority);
    emit_quoted(stream, "#       observation_value_issuer: ",
                document.observation_value_issuer);
    emit_quoted(stream, "#       transformation_registry: ",
                document.transformation_registry);
    stream << "#       envelope:\n";
    emit_envelope(stream, "#         ", document.envelope);
    stream << "#       baseline_parent:\n";
    emit_baseline_reference(stream, "#         ", document.baseline_parent);
    stream << "#       target_parent:\n";
    emit_artifact_identity(stream, "#         ", document.target_parent);
    stream << "#       relation_parent:\n";
    emit_artifact_identity(stream, "#         ", document.relation_parent);

    const auto fields = sorted_output_fields(document);
    emit_uint64(stream, "#       registered_field_count: ", fields.size());
    stream << "#       registered_fields:\n";
    for (const auto &contract : fields) {
        emit_typed_field(stream, "#         ", "#           ",
                         contract.field);
        emit_quoted(stream, "#           authorized_operation: ",
                    transformation_operation_token(
                        contract.authorized_operation));
        emit_quoted(stream, "#           issuer_authority_reference: ",
                    contract.issuer_authority_reference);
    }

    emit_uint64(stream, "#       output_presentation_sequence_count: ",
                document.output_presentation_sequence.size());
    stream << "#       output_presentation_sequence:\n";
    for (const auto uid : document.output_presentation_sequence) {
        emit_int64(stream, "#         - ", uid);
    }

    auto rows = document.rows;
    std::sort(rows.begin(), rows.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.uid < rhs.uid;
    });
    emit_uint64(stream, "#       transformation_row_count: ", rows.size());
    stream << "#       transformations:\n";
    std::map<std::string, OutputFieldContract> field_map;
    for (const auto &contract : fields) {
        field_map.emplace(contract.field.name, contract);
    }
    for (const auto &row : rows) {
        emit_int64(stream, "#         - uid: ", row.uid);
        auto transformations = row.transformations;
        std::sort(transformations.begin(), transformations.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.field_name < rhs.field_name;
                  });
        emit_uint64(stream, "#           transformation_count: ",
                    transformations.size());
        stream << "#           fields:\n";
        for (const auto &change : transformations) {
            const auto field = field_map.find(change.field_name);
            if (field == field_map.end()) {
                throw baseline::ContractError(
                    "matched-output transformation has no field declaration");
            }
            emit_quoted(stream, "#             - field_name: ",
                        change.field_name);
            emit_quoted(stream, "#               operation: ",
                        transformation_operation_token(change.operation));
            emit_bool(stream, "#               before_is_null: ",
                      std::holds_alternative<baseline::NullValue>(
                          change.before));
            emit_quoted(stream, "#               before: ",
                        value_token(change.before, field->second.field.type));
            emit_bool(stream, "#               after_is_null: ",
                      std::holds_alternative<baseline::NullValue>(
                          change.after));
            emit_quoted(stream, "#               after: ",
                        value_token(change.after, field->second.field.type));
            emit_quoted(stream, "#               value_source: ",
                        transformation_value_source_token(
                            change.value_source));
            emit_bool(stream, "#               has_source_pair_key: ",
                      change.source_pair_key.has_value());
            if (change.source_pair_key) {
                emit_int64(stream, "#               source_pair_key: ",
                           *change.source_pair_key);
            }
            emit_bool(stream, "#               has_source_row: ",
                      change.source_row.has_value());
            if (change.source_row) {
                stream << "#               source_row:\n";
                emit_row_reference(stream, "#                 ",
                                   *change.source_row);
            }
            emit_quoted(stream, "#               authority_reference: ",
                        change.authority_reference);
            emit_quoted(stream, "#               provenance_reference: ",
                        change.provenance_reference);
        }
    }
}

struct ParsedOutputMetadata {
    MatchedOutput output;
    std::map<std::int64_t, std::vector<FieldTransformation>> transformations;
};

inline ParsedOutputMetadata parse_output_metadata(MetadataReader &reader) {
    reader.expect("#     output:");
    ParsedOutputMetadata parsed;
    auto &document = parsed.output;
    document.schema = reader.quoted("#       schema: ");
    document.contract_authority =
        reader.quoted("#       contract_authority: ");
    document.observation_value_issuer =
        reader.quoted("#       observation_value_issuer: ");
    document.transformation_registry =
        reader.quoted("#       transformation_registry: ");
    reader.expect("#       envelope:");
    document.envelope = parse_envelope(reader, "#         ");
    reader.expect("#       baseline_parent:");
    document.baseline_parent =
        parse_baseline_reference(reader, "#         ");
    reader.expect("#       target_parent:");
    document.target_parent =
        parse_artifact_identity(reader, "#         ");
    reader.expect("#       relation_parent:");
    document.relation_parent =
        parse_artifact_identity(reader, "#         ");

    const auto field_count =
        reader.uint64("#       registered_field_count: ");
    reader.expect("#       registered_fields:");
    std::map<std::string, OutputFieldContract> fields;
    for (std::uint64_t index = 0; index < field_count; ++index) {
        OutputFieldContract contract;
        contract.field =
            parse_typed_field(reader, "#         ", "#           ");
        contract.authorized_operation = parse_transformation_operation_token(
            reader.quoted("#           authorized_operation: "));
        contract.issuer_authority_reference =
            reader.quoted("#           issuer_authority_reference: ");
        if (!fields.emplace(contract.field.name, contract).second) {
            throw baseline::ContractError(
                "matched-output ECSV has duplicate field declarations");
        }
        document.registered_fields.push_back(std::move(contract));
    }

    const auto sequence_count =
        reader.uint64("#       output_presentation_sequence_count: ");
    reader.expect("#       output_presentation_sequence:");
    for (std::uint64_t index = 0; index < sequence_count; ++index) {
        document.output_presentation_sequence.push_back(
            reader.int64("#         - "));
    }

    const auto row_count =
        reader.uint64("#       transformation_row_count: ");
    reader.expect("#       transformations:");
    for (std::uint64_t row_index = 0; row_index < row_count; ++row_index) {
        const auto uid = reader.int64("#         - uid: ");
        const auto transformation_count =
            reader.uint64("#           transformation_count: ");
        reader.expect("#           fields:");
        std::vector<FieldTransformation> transformations;
        for (std::uint64_t index = 0; index < transformation_count; ++index) {
            FieldTransformation change;
            change.field_name =
                reader.quoted("#             - field_name: ");
            const auto field = fields.find(change.field_name);
            if (field == fields.end()) {
                throw baseline::ContractError(
                    "matched-output transformation names an undeclared field");
            }
            change.operation = parse_transformation_operation_token(
                reader.quoted("#               operation: "));
            const bool before_null =
                reader.boolean("#               before_is_null: ");
            change.before = parse_value_token(
                reader.quoted("#               before: "),
                field->second.field.type, before_null, change.field_name);
            const bool after_null =
                reader.boolean("#               after_is_null: ");
            change.after = parse_value_token(
                reader.quoted("#               after: "),
                field->second.field.type, after_null, change.field_name);
            change.value_source = parse_transformation_value_source_token(
                reader.quoted("#               value_source: "));
            const bool has_pair =
                reader.boolean("#               has_source_pair_key: ");
            if (has_pair) {
                change.source_pair_key =
                    reader.int64("#               source_pair_key: ");
            }
            const bool has_row =
                reader.boolean("#               has_source_row: ");
            if (has_row) {
                reader.expect("#               source_row:");
                change.source_row =
                    parse_row_reference(reader, "#                 ");
            }
            change.authority_reference =
                reader.quoted("#               authority_reference: ");
            change.provenance_reference =
                reader.quoted("#               provenance_reference: ");
            transformations.push_back(std::move(change));
        }
        if (!parsed.transformations.emplace(uid, std::move(transformations))
                 .second) {
            throw baseline::ContractError(
                "matched-output ECSV has duplicate transformation row uid");
        }
    }
    return parsed;
}

inline baseline::ByteTransportHash make_output_transport(
    std::string_view bytes, std::string_view envelope_sha256) {
    const auto binding = canonical_artifact_publication::make_receipt_binding(
        std::string(canonical_artifact_publication::receipt_schema_v1),
        std::string(matched_output_byte_transport_scope_v1),
        std::string(envelope_sha256), bytes);
    return {binding.scope, binding.envelope_sha256, binding.byte_sha256,
            binding.byte_count};
}

inline void validate_declared_columns(
    const std::vector<ColumnDeclaration> &declared,
    const std::vector<ColumnDeclaration> &expected) {
    if (declared.size() != expected.size()) {
        throw baseline::ContractError(
            "matched observation ECSV declared column count mismatch");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (declared[index].name != expected[index].name ||
            declared[index].datatype != expected[index].datatype ||
            declared[index].unit !=
                (baseline::ecsv_detail::physical_unit(expected[index].unit)
                     ? expected[index].unit
                     : std::string{}) ||
            declared[index].description != expected[index].description) {
            throw baseline::ContractError(
                "matched observation ECSV column order/type/unit/description mismatch");
        }
    }
}

inline baseline::Value parse_csv_value(
    const baseline::ecsv_detail::CsvCell &cell,
    const TypedField &field) {
    if (cell.value.empty() && !cell.quoted) {
        return baseline::NullValue{};
    }
    switch (field.type) {
    case baseline::ValueType::int64:
        if (cell.quoted) {
            throw baseline::ContractError(
                "matched observation exact int64 cell must be unquoted: " +
                field.name);
        }
        return baseline::ecsv_detail::parse_int64(cell.value, field.name);
    case baseline::ValueType::float64:
        if (cell.quoted) {
            throw baseline::ContractError(
                "matched observation float64 cell must be unquoted: " +
                field.name);
        }
        return baseline::ecsv_detail::parse_float64(cell.value, field.name);
    case baseline::ValueType::boolean:
        if (!cell.quoted && cell.value == "True") {
            return true;
        }
        if (!cell.quoted && cell.value == "False") {
            return false;
        }
        throw baseline::ContractError(
            "matched observation Boolean cell is noncanonical: " +
            field.name);
    case baseline::ValueType::string:
        if (!cell.quoted || cell.value.empty()) {
            throw baseline::ContractError(
                "matched observation string cell must be nonempty and quoted: " +
                field.name);
        }
        return cell.value;
    }
    throw baseline::ContractError(
        "unsupported matched observation ECSV value type");
}

}  // namespace observation_ecsv_detail

inline SerializedMatchedObservationEcsv serialize_matched_observation_ecsv(
    const MatchedOutput &document,
    const VerifiedBaselineDescriptor &baseline_descriptor,
    const TargetManifest &target, const MatchRelation &relation) {
    namespace wire = observation_ecsv_detail;
    validate(document, baseline_descriptor, target, relation);
    const auto digests =
        compute_digests(document, baseline_descriptor, target, relation);
    const auto target_digests = compute_digests(target);
    const auto relation_digests =
        compute_digests(relation, baseline_descriptor, target);
    const auto columns = wire::expected_columns(document);
    const auto fields = wire::sorted_output_fields(document);

    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "# %ECSV 1.0\n";
    stream << "# ---\n";
    stream << "# datatype:\n";
    for (const auto &column : columns) {
        baseline::ecsv_detail::emit_column(stream, column);
    }
    stream << "# meta:\n";
    stream << "#   " << matched_output_ecsv_metadata_root_v1 << ":\n";
    wire::emit_quoted(stream, "#     schema_version: ",
                      matched_output_schema_v1);
    wire::emit_quoted(stream, "#     artifact_contract_id: ",
                      matched_output_artifact_contract_id_v1);
    wire::emit_quoted(stream, "#     contract_authority: ",
                      contract_authority_v1);
    wire::emit_quoted(stream, "#     observation_value_issuer: ",
                      observation_value_issuer_v1);
    wire::emit_quoted(stream, "#     field_registry: ",
                      matched_output_field_registry_v1);
    wire::emit_quoted(stream, "#     transformation_registry: ",
                      transformation_registry_v1);
    wire::emit_quoted(stream, "#     framing_encoding: ",
                      framing_encoding_v1);
    wire::emit_quoted(stream, "#     semantic_scope: ",
                      matched_output_semantic_scope_v1);
    wire::emit_quoted(stream, "#     semantic_sha256: ",
                      digests.semantic_sha256);
    wire::emit_quoted(stream, "#     envelope_scope: ",
                      matched_output_envelope_scope_v1);
    wire::emit_quoted(stream, "#     envelope_sha256: ",
                      digests.envelope_sha256);
    wire::emit_quoted(stream, "#     byte_transport_scope: ",
                      matched_output_byte_transport_scope_v1);
    wire::emit_quoted(stream, "#     target_semantic_scope: ",
                      target_semantic_scope_v1);
    wire::emit_quoted(stream, "#     target_semantic_sha256: ",
                      target_digests.semantic_sha256);
    wire::emit_quoted(stream, "#     target_envelope_scope: ",
                      target_envelope_scope_v1);
    wire::emit_quoted(stream, "#     target_envelope_sha256: ",
                      target_digests.envelope_sha256);
    wire::emit_quoted(stream, "#     relation_semantic_scope: ",
                      relation_semantic_scope_v1);
    wire::emit_quoted(stream, "#     relation_semantic_sha256: ",
                      relation_digests.semantic_sha256);
    wire::emit_quoted(stream, "#     relation_envelope_scope: ",
                      relation_envelope_scope_v1);
    wire::emit_quoted(stream, "#     relation_envelope_sha256: ",
                      relation_digests.envelope_sha256);
    wire::emit_output_metadata(stream, document);
    wire::emit_target(stream, target);
    wire::emit_relation(stream, relation, baseline_descriptor, target);
    stream << "#     null_cell: \"unquoted-empty-v1\"\n";
    stream << "#     string_cell: \"quoted-utf8-single-line-v1\"\n";
    stream << "#     pair_key_set_cell: \"quoted-bracketed-int64-set-v1\"\n";
    stream << "#     metadata_float64: \"quoted-ieee754-bits-v1\"\n";
    stream << "# delimiter: \",\"\n";
    stream << "# schema: \"astropy-2.0\"\n";

    for (std::size_t index = 0; index < columns.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << columns[index].name;
    }
    stream << '\n';
    auto rows = document.rows;
    std::sort(rows.begin(), rows.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.uid < rhs.uid;
    });
    for (const auto &row : rows) {
        stream << row.uid << ',' << row.target.local_key << ','
               << row.target_input_key << ','
               << baseline::ecsv_detail::format_float64(
                      row.tone_frequency_hz)
               << ',' << row.array << ',' << row.network << ','
               << row.channel << ','
               << baseline::ecsv_detail::csv_quote(
                      wire::pair_key_set_cell(row.relation_pair_keys));
        for (const auto &contract : fields) {
            stream << ',' << baseline::ecsv_detail::value_to_csv(
                                  row.fields.at(contract.field.name));
        }
        stream << '\n';
    }

    SerializedMatchedObservationEcsv result;
    result.bytes = stream.str();
    result.digests = digests;
    result.transport =
        wire::make_output_transport(result.bytes, digests.envelope_sha256);
    return result;
}

inline ParsedMatchedObservationEcsv parse_matched_observation_ecsv(
    std::string_view bytes,
    const VerifiedBaselineDescriptor &baseline_descriptor) {
    namespace wire = observation_ecsv_detail;
    if (bytes.empty() || bytes.back() != '\n') {
        throw baseline::ContractError(
            "matched observation ECSV requires nonempty LF-terminated bytes");
    }
    if (bytes.find('\r') != std::string_view::npos) {
        throw baseline::ContractError(
            "matched observation ECSV rejects CR/CRLF bytes");
    }
    if (!baseline::detail::valid_utf8(bytes)) {
        throw baseline::ContractError(
            "matched observation ECSV is not valid UTF-8");
    }
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    while (start < bytes.size()) {
        const auto end = bytes.find('\n', start);
        lines.push_back(bytes.substr(start, end - start));
        start = end + 1;
    }
    if (lines.size() < 4 || lines[0] != "# %ECSV 1.0" ||
        lines[1] != "# ---" || lines[2] != "# datatype:") {
        throw baseline::ContractError(
            "matched observation ECSV requires exact ECSV 1.0 framing");
    }

    std::vector<wire::ColumnDeclaration> declared_columns;
    std::size_t index = 3;
    while (index < lines.size() && lines[index] != "# meta:") {
        if (!lines[index].starts_with("# - name: ")) {
            throw baseline::ContractError(
                "matched observation ECSV datatype declarations are reordered");
        }
        wire::ColumnDeclaration column;
        column.name = baseline::ecsv_detail::yaml_unquote(
            lines[index++].substr(std::string_view("# - name: ").size()));
        if (index >= lines.size() ||
            !lines[index].starts_with("#   datatype: ")) {
            throw baseline::ContractError(
                "matched observation ECSV datatype declaration is incomplete");
        }
        column.datatype = baseline::ecsv_detail::yaml_unquote(
            lines[index++].substr(
                std::string_view("#   datatype: ").size()));
        if (index < lines.size() && lines[index].starts_with("#   unit: ")) {
            column.unit = baseline::ecsv_detail::yaml_unquote(
                lines[index++].substr(std::string_view("#   unit: ").size()));
        }
        if (index >= lines.size() ||
            !lines[index].starts_with("#   description: ")) {
            throw baseline::ContractError(
                "matched observation ECSV column description is absent");
        }
        column.description = baseline::ecsv_detail::yaml_unquote(
            lines[index++].substr(
                std::string_view("#   description: ").size()));
        declared_columns.push_back(std::move(column));
    }
    if (index >= lines.size() || lines[index] != "# meta:") {
        throw baseline::ContractError(
            "matched observation ECSV metadata root is absent");
    }
    wire::MetadataReader reader(lines, index + 1);
    reader.expect("#   " + std::string(matched_output_ecsv_metadata_root_v1) +
                  ":");
    if (reader.quoted("#     schema_version: ") !=
            matched_output_schema_v1 ||
        reader.quoted("#     artifact_contract_id: ") !=
            matched_output_artifact_contract_id_v1 ||
        reader.quoted("#     contract_authority: ") !=
            contract_authority_v1 ||
        reader.quoted("#     observation_value_issuer: ") !=
            observation_value_issuer_v1 ||
        reader.quoted("#     field_registry: ") !=
            matched_output_field_registry_v1 ||
        reader.quoted("#     transformation_registry: ") !=
            transformation_registry_v1 ||
        reader.quoted("#     framing_encoding: ") != framing_encoding_v1 ||
        reader.quoted("#     semantic_scope: ") !=
            matched_output_semantic_scope_v1) {
        throw baseline::ContractError(
            "matched observation ECSV authority/schema/registry metadata mismatch");
    }
    baseline::Digests declared;
    declared.semantic_sha256 = reader.quoted("#     semantic_sha256: ");
    if (reader.quoted("#     envelope_scope: ") !=
        matched_output_envelope_scope_v1) {
        throw baseline::ContractError(
            "matched observation ECSV envelope scope mismatch");
    }
    declared.envelope_sha256 = reader.quoted("#     envelope_sha256: ");
    if (reader.quoted("#     byte_transport_scope: ") !=
        matched_output_byte_transport_scope_v1) {
        throw baseline::ContractError(
            "matched observation ECSV byte-transport scope mismatch");
    }
    if (reader.quoted("#     target_semantic_scope: ") !=
        target_semantic_scope_v1) {
        throw baseline::ContractError(
            "embedded target semantic scope mismatch");
    }
    const auto target_semantic =
        reader.quoted("#     target_semantic_sha256: ");
    if (reader.quoted("#     target_envelope_scope: ") !=
        target_envelope_scope_v1) {
        throw baseline::ContractError(
            "embedded target envelope scope mismatch");
    }
    const auto target_envelope =
        reader.quoted("#     target_envelope_sha256: ");
    if (reader.quoted("#     relation_semantic_scope: ") !=
        relation_semantic_scope_v1) {
        throw baseline::ContractError(
            "embedded relation semantic scope mismatch");
    }
    const auto relation_semantic =
        reader.quoted("#     relation_semantic_sha256: ");
    if (reader.quoted("#     relation_envelope_scope: ") !=
        relation_envelope_scope_v1) {
        throw baseline::ContractError(
            "embedded relation envelope scope mismatch");
    }
    const auto relation_envelope =
        reader.quoted("#     relation_envelope_sha256: ");

    auto output_metadata = wire::parse_output_metadata(reader);
    auto target = wire::parse_target(reader);
    const auto target_digests = compute_digests(target);
    if (target_digests.semantic_sha256 != target_semantic ||
        target_digests.envelope_sha256 != target_envelope) {
        throw baseline::ContractError(
            "embedded target digest declaration mismatch");
    }
    auto relation =
        wire::parse_relation(reader, baseline_descriptor, target);
    const auto relation_digests =
        compute_digests(relation, baseline_descriptor, target);
    if (relation_digests.semantic_sha256 != relation_semantic ||
        relation_digests.envelope_sha256 != relation_envelope) {
        throw baseline::ContractError(
            "embedded relation digest declaration mismatch");
    }
    reader.expect("#     null_cell: \"unquoted-empty-v1\"");
    reader.expect("#     string_cell: \"quoted-utf8-single-line-v1\"");
    reader.expect(
        "#     pair_key_set_cell: \"quoted-bracketed-int64-set-v1\"");
    reader.expect(
        "#     metadata_float64: \"quoted-ieee754-bits-v1\"");
    reader.expect("# delimiter: \",\"");
    reader.expect("# schema: \"astropy-2.0\"");
    const auto csv_header_index = reader.index();
    if (csv_header_index >= lines.size()) {
        throw baseline::ContractError(
            "matched observation ECSV CSV header is absent");
    }

    auto &document = output_metadata.output;
    wire::validate_declared_columns(declared_columns,
                                    wire::expected_columns(document));
    const auto expected = wire::expected_columns(document);
    const auto header_cells =
        baseline::ecsv_detail::parse_csv_line(lines[csv_header_index]);
    if (header_cells.size() != expected.size()) {
        throw baseline::ContractError(
            "matched observation ECSV CSV header cardinality mismatch");
    }
    for (std::size_t cell_index = 0; cell_index < expected.size();
         ++cell_index) {
        if (header_cells[cell_index].quoted ||
            header_cells[cell_index].value != expected[cell_index].name) {
            throw baseline::ContractError(
                "matched observation ECSV CSV header is noncanonical");
        }
    }

    const auto fields = wire::sorted_output_fields(document);
    for (std::size_t line_index = csv_header_index + 1;
         line_index < lines.size(); ++line_index) {
        if (lines[line_index].empty()) {
            throw baseline::ContractError(
                "matched observation ECSV contains a blank data row");
        }
        const auto cells =
            baseline::ecsv_detail::parse_csv_line(lines[line_index]);
        if (cells.size() != expected.size()) {
            throw baseline::ContractError(
                "matched observation ECSV data row cardinality mismatch");
        }
        for (std::size_t scalar = 0; scalar < 7; ++scalar) {
            if (cells[scalar].quoted) {
                throw baseline::ContractError(
                    "matched observation ECSV structural cell is quoted");
            }
        }
        if (!cells[7].quoted) {
            throw baseline::ContractError(
                "matched observation ECSV pair set must be quoted");
        }
        MatchedOutputRow row;
        row.uid = baseline::ecsv_detail::parse_int64(cells[0].value, "uid");
        row.target = detail::make_row_reference(
            document.target_parent,
            baseline::ecsv_detail::parse_int64(cells[1].value,
                                               "target_row_key"));
        row.target_input_key = baseline::ecsv_detail::parse_int64(
            cells[2].value, "target_input_key");
        row.tone_frequency_hz = baseline::ecsv_detail::parse_float64(
            cells[3].value, "tone_freq");
        row.array =
            baseline::ecsv_detail::parse_int64(cells[4].value, "array");
        row.network =
            baseline::ecsv_detail::parse_int64(cells[5].value, "nw");
        row.channel = baseline::ecsv_detail::parse_int64(
            cells[6].value, "kids_tone");
        row.relation_pair_keys =
            wire::parse_pair_key_set_cell(cells[7].value);
        for (std::size_t field_index = 0; field_index < fields.size();
             ++field_index) {
            row.fields.emplace(
                fields[field_index].field.name,
                wire::parse_csv_value(cells[8 + field_index],
                                      fields[field_index].field));
        }
        const auto transformations =
            output_metadata.transformations.find(row.uid);
        if (transformations == output_metadata.transformations.end()) {
            throw baseline::ContractError(
                "matched observation ECSV row lacks transformation evidence");
        }
        row.transformations = transformations->second;
        document.rows.push_back(std::move(row));
    }
    if (document.rows.size() != output_metadata.transformations.size()) {
        throw baseline::ContractError(
            "matched observation ECSV has stale or extra transformation rows");
    }

    validate(document, baseline_descriptor, target, relation);
    const auto computed =
        compute_digests(document, baseline_descriptor, target, relation);
    if (!baseline::is_sha256_reference(declared.semantic_sha256) ||
        !baseline::is_sha256_reference(declared.envelope_sha256) ||
        computed.semantic_sha256 != declared.semantic_sha256 ||
        computed.envelope_sha256 != declared.envelope_sha256) {
        throw baseline::ContractError(
            "matched observation ECSV semantic/envelope digest mismatch");
    }
    const auto canonical = serialize_matched_observation_ecsv(
        document, baseline_descriptor, target, relation);
    if (canonical.bytes != bytes) {
        throw baseline::ContractError(
            "matched observation ECSV bytes are not canonical v1 serialization");
    }
    return {std::move(target), std::move(relation), std::move(document),
            std::move(declared), std::move(canonical.transport)};
}

inline ParsedMatchedObservationEcsv
parse_matched_observation_ecsv_with_receipt(
    std::string_view bytes, std::string_view receipt_bytes,
    const VerifiedBaselineDescriptor &baseline_descriptor) {
    namespace publication = canonical_artifact_publication;
    const auto receipt = publication::parse_canonical_receipt(
        receipt_bytes, publication::receipt_schema_v1,
        matched_output_byte_transport_scope_v1);
    publication::validate_receipt_binding(bytes, receipt);
    auto parsed =
        parse_matched_observation_ecsv(bytes, baseline_descriptor);
    if (receipt.envelope_sha256 !=
        parsed.declared_digests.envelope_sha256) {
        throw baseline::ContractError(
            "matched observation receipt binds a foreign envelope");
    }
    return parsed;
}

}  // namespace citlali::pipeline::canonical_apt_observation_v1
