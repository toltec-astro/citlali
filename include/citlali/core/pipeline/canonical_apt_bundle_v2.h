#pragma once

#include <citlali/core/pipeline/canonical_apt_v2_ecsv.h>
#include <citlali/core/pipeline/canonical_artifact_publication.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline::canonical_apt_v2 {

namespace publication = canonical_artifact_publication;

inline constexpr std::string_view root_manifest_name_v2 = "manifest.ecsv";
inline constexpr std::string_view root_receipt_name_v2 =
    "manifest.ecsv.sha256";

struct BundlePayload {
    std::string root_manifest_bytes;
    std::string root_receipt_bytes;
    std::map<std::string, std::string> component_bytes_by_role;
};

struct SeedDisposition {
    ScopedRowReference seed;
    std::string disposition;
    std::optional<ScopedRowReference> target;
    std::optional<std::int64_t> pair_uid;

    friend bool operator==(const SeedDisposition &,
                           const SeedDisposition &) = default;
};

struct VerifiedBaselineSnapshot {
    BundleManifest manifest;
    ComponentDigests manifest_digests;
    publication::ReceiptBinding receipt;
    AptTable apt;
    std::vector<FieldRule> fields;
    std::vector<SourceRecord> sources;
};

struct VerifiedBundle {
    BundleManifest manifest;
    ComponentIdentity identity;
    ComponentDigests manifest_digests;
    publication::ReceiptBinding receipt;
    AptTable apt;
    std::vector<FieldRule> fields;
    std::vector<SourceRecord> sources;
    std::optional<RelationTable> relation;
    std::vector<ExceptionRecord> exceptions;
    std::optional<TargetManifest> target;
    std::optional<VerifiedBaselineSnapshot> baseline_snapshot;
    std::vector<SeedDisposition> seed_dispositions;
    BundlePayload payload;
    std::uint64_t total_byte_count = 0;
    std::uint64_t parser_count = 0;
    std::filesystem::path manifest_path;
    std::filesystem::path receipt_path;
};

struct PreparedBundle {
    BundlePayload payload;
    ComponentIdentity identity;
    std::uint64_t total_byte_count = 0;
};

namespace bundle_detail {

inline ComponentIdentity bundle_identity(
    const BundleManifest &manifest, const ComponentDigests &digests) {
    return {
        manifest.kind == BundleKind::baseline
            ? std::string(baseline_bundle_schema_v2)
            : std::string(matched_bundle_schema_v2),
        manifest.issuance.occurrence, digests.semantic_sha256,
        digests.envelope_sha256};
}

inline std::string expected_schema_for_role(std::string_view role,
                                            BundleKind outer_kind) {
    if (role == "apt") {
        return outer_kind == BundleKind::baseline
            ? std::string(baseline_apt_schema_v2)
            : std::string(matched_apt_schema_v2);
    }
    if (role == "fields" || role == "baseline-fields") {
        return std::string(field_table_schema_v2);
    }
    if (role == "sources" || role == "baseline-sources") {
        return std::string(source_table_schema_v2);
    }
    if (role == "relation") return std::string(relation_table_schema_v2);
    if (role == "exceptions") return std::string(exception_table_schema_v2);
    if (role == "baseline-apt") {
        return std::string(baseline_apt_schema_v2);
    }
    if (role == "baseline-manifest") {
        return std::string(manifest_schema_v2);
    }
    if (role == "baseline-receipt") {
        return std::string(receipt_schema_v2);
    }
    throw ContractError("canonical APT v2 component role is unsupported: " +
                        std::string(role));
}

inline std::map<std::string, ComponentDescriptor> descriptor_map(
    const BundleManifest &manifest) {
    std::map<std::string, ComponentDescriptor> result;
    for (const auto &component : manifest.components) {
        if (!result.emplace(component.role, component).second) {
            throw ContractError("canonical APT v2 component role is duplicate");
        }
    }
    return result;
}

inline void verify_transport_descriptor(
    const ComponentDescriptor &descriptor, std::string_view bytes,
    BundleKind outer_kind) {
    validate(descriptor);
    if (descriptor.schema !=
            expected_schema_for_role(descriptor.role, outer_kind) ||
        descriptor.byte_count != bytes.size() ||
        descriptor.transport_sha256 !=
            "sha256:" + citlali::utils::sha256(bytes)) {
        throw ContractError(
            "canonical APT v2 component transport descriptor disagrees");
    }
}

template <class Document>
inline void verify_parsed_descriptor(
    const ComponentDescriptor &descriptor,
    const VerifiedComponent<Document> &verified) {
    if (descriptor.schema != verified.schema ||
        descriptor.semantic_sha256 != verified.digests.semantic_sha256 ||
        descriptor.envelope_sha256 != verified.digests.envelope_sha256 ||
        descriptor.transport_sha256 != verified.digests.transport_sha256 ||
        descriptor.byte_count != verified.digests.byte_count ||
        descriptor.row_count != verified.row_count) {
        throw ContractError(
            "canonical APT v2 parsed component descriptor disagrees");
    }
}

template <class Document>
inline void require_root_context(
    const VerifiedComponent<Document> &verified,
    const BundleManifest &manifest) {
    if (verified.kind != manifest.kind ||
        verified.issuance != manifest.issuance ||
        verified.observation != manifest.observation) {
        throw ContractError(
            "canonical APT v2 component root context disagrees");
    }
}

inline bool exact_value_equal(const Value &lhs, const Value &rhs,
                              ValueType type) {
    if (lhs.index() != rhs.index()) return false;
    if (std::holds_alternative<double>(lhs)) {
        return canonical_binary64(std::get<double>(lhs)) ==
            canonical_binary64(std::get<double>(rhs));
    }
    std::string left;
    std::string right;
    v1::detail::add_value(left, "value", lhs, type);
    v1::detail::add_value(right, "value", rhs, type);
    return left == right;
}

inline ComponentDescriptor make_descriptor(
    std::string role, const SerializedComponent &component) {
    ComponentDescriptor result{
        std::move(role), {}, component.schema,
        component.digests.semantic_sha256,
        component.digests.envelope_sha256,
        component.digests.transport_sha256,
        component.digests.byte_count, component.row_count};
    result.relative_path = content_addressed_basename(
        result.transport_sha256, result.role);
    validate(result);
    return result;
}

inline ComponentDescriptor snapshot_descriptor(
    std::string role, const ComponentDescriptor &original) {
    ComponentDescriptor result = original;
    result.role = std::move(role);
    result.relative_path = content_addressed_basename(
        result.transport_sha256, result.role);
    validate(result);
    return result;
}

inline ComponentDescriptor snapshot_receipt_descriptor(
    std::string_view receipt_bytes, const ComponentDigests &manifest_digests) {
    ComponentDescriptor result{
        "baseline-receipt", {}, std::string(receipt_schema_v2),
        manifest_digests.semantic_sha256, manifest_digests.envelope_sha256,
        "sha256:" + citlali::utils::sha256(receipt_bytes),
        static_cast<std::uint64_t>(receipt_bytes.size()), 5};
    result.relative_path = content_addressed_basename(
        result.transport_sha256, result.role);
    validate(result);
    return result;
}

inline std::map<std::int64_t, const SourceRecord *> source_map(
    const std::vector<SourceRecord> &sources) {
    std::map<std::int64_t, const SourceRecord *> result;
    for (const auto &source : sources) {
        validate(source);
        if (!result.emplace(source.source_uid, &source).second) {
            throw ContractError("canonical APT v2 source UID is duplicate");
        }
    }
    return result;
}

inline void validate_baseline_inventory(
    const AptTable &apt, const std::vector<SourceRecord> &sources) {
    std::map<std::int64_t, const SourceRecord *> raw_by_network;
    std::uint64_t expected_rows = 0;
    for (const auto &source : sources) {
        validate(source);
        if (source.role != SourceRole::raw ||
            source.header_observation != apt.observation ||
            !raw_by_network.emplace(source.network, &source).second) {
            throw ContractError(
                "canonical APT v2 baseline source inventory is invalid");
        }
        expected_rows += static_cast<std::uint64_t>(source.channel_count);
    }
    std::set<std::pair<std::int64_t, std::int64_t>> channels;
    for (const auto &row : apt.rows) {
        const auto source = raw_by_network.find(row.network);
        if (source == raw_by_network.end() || row.channel < 0 ||
            row.channel >= source->second->channel_count ||
            row.array != array_for_network(row.network) ||
            !channels.emplace(row.network, row.channel).second) {
            throw ContractError(
                "canonical APT v2 baseline detector/source relation is invalid");
        }
    }
    if (expected_rows != apt.rows.size()) {
        throw ContractError(
            "canonical APT v2 baseline raw inventory is incomplete");
    }
}

inline TargetManifest reconstruct_target(
    const AptTable &apt, const RelationTable &relation,
    const std::vector<SourceRecord> &sources) {
    std::map<std::int64_t, const AptRow *> rows;
    for (const auto &row : apt.rows) rows.emplace(row.uid, &row);
    TargetManifest target;
    target.issuance = relation.target_issuance;
    target.observation = relation.observation;
    target.sources = sources;
    for (const auto &record : relation.rows) {
        const auto output = rows.find(record.output_uid);
        if (output == rows.end()) {
            throw ContractError(
                "canonical APT v2 relation output row is absent");
        }
        TargetRow row;
        row.uid = record.target.local_uid;
        row.input_uid = record.target_input_uid;
        row.raw_source_uid = record.target_raw_source_uid;
        row.kmp_source_uid = record.target_kmp_source_uid;
        row.kmp_row_index = record.target_kmp_row_index;
        row.source_rank = record.source_rank;
        row.application_rank = record.application_rank;
        row.tone_frequency_hz = output->second->tone_frequency_hz;
        row.array = output->second->array;
        row.network = output->second->network;
        row.channel = output->second->channel;
        for (const auto name :
             {"kids_fr", "kids_f_out", "kids_Qr", "kids_flag"}) {
            const auto found = output->second->fields.find(name);
            if (found != output->second->fields.end()) {
                row.fields.emplace(name, found->second);
            }
        }
        target.rows.push_back(std::move(row));
    }
    validate(target);
    if (target_identity(target) != relation.target_parent) {
        throw ContractError(
            "canonical APT v2 reconstructed target identity disagrees");
    }
    return target;
}

inline void require_authorized_kmp_rule(const FieldRule &field) {
    const bool is_flag = field.name == "kids_flag";
    const bool is_qr = field.name == "kids_Qr";
    const std::string expected_unit =
        (field.name == "kids_fr" || field.name == "kids_f_out") ? "Hz"
                                                                  : "N/A";
    const std::string expected_source = field.name == "kids_fr" ? "fr"
        : field.name == "kids_f_out"                            ? "f_out"
        : is_qr                                                  ? "Qr"
                                                                 : "flag";
    const std::string expected_authority = is_flag
        ? "kids:fit-report-v1"
        : "kids:model-params-v1";
    if (!is_authorized_kmp_field(field.name) ||
        field.datatype != (is_flag ? ValueType::int64
                                   : ValueType::float64) ||
        field.unit != expected_unit || field.nullable ||
        field.authority != "copied-declared" ||
        field.authority_reference != expected_authority ||
        field.operation != FieldOperation::preserve_target ||
        field.source_field != expected_source ||
        field.missing_policy != "reject" ||
        field.identity_role != "nonidentity") {
        throw ContractError(
            "canonical APT v2 KMP field rule is not exact");
    }
}

inline std::vector<SeedDisposition> validate_matched_semantics(
    const AptTable &apt, const std::vector<FieldRule> &fields,
    const std::vector<SourceRecord> &sources, const RelationTable &relation,
    const std::vector<ExceptionRecord> &exceptions,
    const AptTable &baseline_apt,
    const std::vector<FieldRule> &baseline_fields) {
    if (apt.kind != BundleKind::matched ||
        apt.observation != relation.observation ||
        apt.rows.size() != relation.rows.size() ||
        apt.field_rules != fields) {
        throw ContractError(
            "canonical APT v2 matched component cardinality disagrees");
    }
    std::map<std::string, const FieldRule *> output_rules;
    std::map<std::string, const FieldRule *> baseline_rules;
    for (const auto &field : fields) output_rules.emplace(field.name, &field);
    for (const auto &field : baseline_fields) {
        baseline_rules.emplace(field.name, &field);
    }
    for (const auto name : {"kids_fr", "kids_f_out", "kids_Qr"}) {
        if (!output_rules.contains(name)) {
            throw ContractError(
                "canonical APT v2 required KMP rule is absent");
        }
    }
    for (const auto &[name, field] : output_rules) {
        if (is_authorized_kmp_field(name)) {
            require_authorized_kmp_rule(*field);
            continue;
        }
        if (name == "uid" || name == "tone_freq" || name == "array" ||
            name == "nw" || name == "kids_tone") {
            continue;
        }
        if (name.starts_with("kids_") || !baseline_rules.contains(name)) {
            throw ContractError(
                "canonical APT v2 output field is not baseline/KMP governed");
        }
        const auto &baseline = *baseline_rules.at(name);
        if (field->datatype != baseline.datatype ||
            field->unit != baseline.unit || !field->nullable ||
            field->authority != baseline.authority ||
            field->authority_reference != baseline.authority_reference ||
            field->identity_role != baseline.identity_role ||
            field->operation != FieldOperation::copy_seed_or_null) {
            throw ContractError(
                "canonical APT v2 copied baseline field rule changed");
        }
    }
    for (const auto &[name, field] : baseline_rules) {
        (void)field;
        if (name == "uid" || name == "tone_freq" || name == "array" ||
            name == "nw" || name == "kids_tone" ||
            is_authorized_kmp_field(name)) {
            continue;
        }
        if (!output_rules.contains(name)) {
            throw ContractError(
                "canonical APT v2 baseline field was dropped");
        }
    }

    const auto sources_by_uid = source_map(sources);
    std::map<std::int64_t, const NetworkEvidence *> evidence_by_uid;
    for (const auto &evidence : relation.network_evidence) {
        evidence_by_uid.emplace(evidence.evidence_uid, &evidence);
    }
    std::map<std::int64_t, const AptRow *> output_by_uid;
    for (const auto &row : apt.rows) output_by_uid.emplace(row.uid, &row);
    std::map<std::int64_t, const AptRow *> seed_by_uid;
    for (const auto &row : baseline_apt.rows) seed_by_uid.emplace(row.uid, &row);
    std::map<std::int64_t, const RelationRecord *> relation_by_target;
    for (const auto &record : relation.rows) {
        relation_by_target.emplace(record.target.local_uid, &record);
    }

    std::map<std::pair<std::int64_t, std::string>,
             const ExceptionRecord *> field_exceptions;
    std::map<std::int64_t, std::vector<const ExceptionRecord *>>
        ambiguity_exceptions;
    std::set<std::int64_t> seed_exception_uids;
    for (const auto &exception : exceptions) {
        validate(exception);
        if (exception.kind == ExceptionKind::field_deviation) {
            throw ContractError(
                "canonical APT v2 has no activated field-deviation authority");
        } else if (exception.kind == ExceptionKind::ambiguity_candidate) {
            if (!relation_by_target.contains(*exception.target_uid) ||
                exception.seed->artifact != relation.baseline_parent ||
                !seed_by_uid.contains(exception.seed->local_uid)) {
                throw ContractError(
                    "canonical APT v2 ambiguity candidate is foreign");
            }
            ambiguity_exceptions[*exception.target_uid].push_back(&exception);
        } else {
            if (exception.seed->artifact != relation.baseline_parent ||
                !seed_by_uid.contains(exception.seed->local_uid) ||
                !seed_exception_uids.insert(exception.seed->local_uid).second) {
                throw ContractError(
                    "canonical APT v2 seed disposition exception is foreign");
            }
        }
    }

    std::map<std::int64_t, std::size_t> ambiguity_seed_edge_count;
    for (const auto &[target_uid, candidates] : ambiguity_exceptions) {
        (void)target_uid;
        for (const auto *candidate : candidates) {
            ++ambiguity_seed_edge_count[candidate->seed->local_uid];
        }
    }

    std::map<std::int64_t, std::pair<ScopedRowReference, std::int64_t>>
        selected_seed_to_target_pair;
    std::set<std::pair<std::int64_t, std::int64_t>> target_channels;
    for (const auto &record : relation.rows) {
        const auto output = output_by_uid.find(record.output_uid);
        const auto raw = sources_by_uid.find(record.target_raw_source_uid);
        const auto kmp = sources_by_uid.find(record.target_kmp_source_uid);
        const auto evidence = evidence_by_uid.find(record.network_evidence_uid);
        if (output == output_by_uid.end() || raw == sources_by_uid.end() ||
            kmp == sources_by_uid.end() || evidence == evidence_by_uid.end() ||
            raw->second->role != SourceRole::raw ||
            kmp->second->role != SourceRole::kmp ||
            raw->second->network != output->second->network ||
            kmp->second->network != output->second->network ||
            evidence->second->network != output->second->network ||
            record.presentation_rank != output->second->presentation_rank ||
            output->second->array != array_for_network(output->second->network) ||
            !target_channels
                 .emplace(output->second->network, output->second->channel)
                 .second) {
            throw ContractError(
                "canonical APT v2 matched row/source/evidence join is invalid");
        }
        if (record.disposition != RelationDisposition::unmatched &&
            evidence->second->status != NetworkEvidenceStatus::matched_capable) {
            throw ContractError(
                "canonical APT v2 selected/ambiguous relation lacks match-capable evidence");
        }
        const auto candidates = ambiguity_exceptions.find(
            record.target.local_uid);
        const auto candidate_count = candidates == ambiguity_exceptions.end()
            ? 0U
            : candidates->second.size();
        bool contention_edge = false;
        if (candidates != ambiguity_exceptions.end()) {
            contention_edge = std::any_of(
                candidates->second.begin(), candidates->second.end(),
                [&](const auto *candidate) {
                    return ambiguity_seed_edge_count.at(
                               candidate->seed->local_uid) >= 2U;
                });
        }
        if ((record.disposition == RelationDisposition::ambiguous &&
             (candidate_count == 0U ||
              (candidate_count < 2U && !contention_edge))) ||
            (record.disposition != RelationDisposition::ambiguous &&
             candidate_count != 0U)) {
            throw ContractError(
                "canonical APT v2 ambiguity candidate coverage is invalid");
        }
        if (candidates != ambiguity_exceptions.end()) {
            std::set<std::int64_t> candidate_seeds;
            for (const auto *candidate : candidates->second) {
                if (!candidate_seeds.insert(candidate->seed->local_uid).second) {
                    throw ContractError(
                        "canonical APT v2 ambiguity candidate is duplicate");
                }
            }
        }
        const AptRow *selected_seed = nullptr;
        if (record.selected_seed) {
            const auto seed = seed_by_uid.find(record.selected_seed->local_uid);
            if (seed == seed_by_uid.end() ||
                seed->second->network != output->second->network) {
                throw ContractError(
                    "canonical APT v2 selected seed is absent or cross-network");
            }
            selected_seed = seed->second;
            selected_seed_to_target_pair.emplace(
                seed->first,
                std::pair{record.target, *record.selected_pair_uid});
        }
        for (const auto &[name, rule] : output_rules) {
            if (name == "uid" || name == "tone_freq" || name == "array" ||
                name == "nw" || name == "kids_tone" ||
                is_authorized_kmp_field(name)) {
                continue;
            }
            const Value expected = selected_seed
                ? selected_seed->fields.at(name)
                : Value{NullValue{}};
            const auto &actual = output->second->fields.at(name);
            const auto exception = field_exceptions.find(
                {record.target.local_uid, name});
            if (exception == field_exceptions.end()) {
                if (!exact_value_equal(actual, expected, rule->datatype)) {
                    throw ContractError(
                        "canonical APT v2 ordinary field reconstruction disagrees");
                }
            } else if (!exact_value_equal(*exception->second->before, expected,
                                          rule->datatype) ||
                       !exact_value_equal(*exception->second->after, actual,
                                          rule->datatype) ||
                       exception->second->value_type != rule->datatype) {
                throw ContractError(
                    "canonical APT v2 field exception reconstruction disagrees");
            }
        }
    }

    std::vector<SeedDisposition> dispositions;
    std::vector<const AptRow *> ordered_seeds;
    ordered_seeds.reserve(baseline_apt.rows.size());
    for (const auto &row : baseline_apt.rows) ordered_seeds.push_back(&row);
    std::sort(ordered_seeds.begin(), ordered_seeds.end(),
              [](const auto *lhs, const auto *rhs) {
                  return lhs->uid < rhs->uid;
              });
    for (const auto *seed : ordered_seeds) {
        SeedDisposition disposition{
            {relation.baseline_parent, seed->uid}, "unused", std::nullopt,
            std::nullopt};
        const auto selected = selected_seed_to_target_pair.find(seed->uid);
        if (selected != selected_seed_to_target_pair.end()) {
            disposition.disposition = "matched";
            disposition.target = selected->second.first;
            disposition.pair_uid = selected->second.second;
        }
        dispositions.push_back(std::move(disposition));
    }
    return dispositions;
}

}  // namespace bundle_detail

inline VerifiedBundle verify_bundle_payload(
    BundlePayload payload,
    std::optional<publication::ReceiptBinding> parsed_receipt = std::nullopt,
    std::optional<VerifiedComponent<BundleManifest>> parsed_root =
        std::nullopt) {
    VerifiedBundle result;
    result.payload = payload;
    result.parser_count = 0;

    result.receipt = parsed_receipt
        ? std::move(*parsed_receipt)
        : publication::parse_canonical_receipt(
              payload.root_receipt_bytes, receipt_schema_v2,
              bundle_transport_scope_v2);
    ++result.parser_count;
    publication::validate_receipt_binding(payload.root_manifest_bytes,
                                          result.receipt);
    auto root = parsed_root
        ? std::move(*parsed_root)
        : verify_manifest_component(payload.root_manifest_bytes);
    ++result.parser_count;
    if (root.digests.transport_sha256 !=
            "sha256:" +
                citlali::utils::sha256(payload.root_manifest_bytes) ||
        root.digests.byte_count != payload.root_manifest_bytes.size()) {
        throw ContractError(
            "canonical APT v2 preparsed root does not match supplied bytes");
    }
    result.manifest = root.document;
    result.manifest_digests = root.digests;
    result.identity = bundle_detail::bundle_identity(result.manifest,
                                                     root.digests);
    if (result.receipt.envelope_sha256 != root.digests.envelope_sha256) {
        throw ContractError(
            "canonical APT v2 root receipt envelope disagrees");
    }
    const auto descriptors = bundle_detail::descriptor_map(result.manifest);
    std::set<std::string> supplied_roles;
    for (const auto &[role, bytes] : payload.component_bytes_by_role) {
        (void)bytes;
        supplied_roles.insert(role);
    }
    if (supplied_roles != required_roles(result.manifest.kind)) {
        throw ContractError(
            "canonical APT v2 supplied component inventory is not exact");
    }

    result.total_byte_count =
        static_cast<std::uint64_t>(payload.root_manifest_bytes.size()) +
        static_cast<std::uint64_t>(payload.root_receipt_bytes.size());
    for (const auto &[role, descriptor] : descriptors) {
        const auto found = payload.component_bytes_by_role.find(role);
        if (found == payload.component_bytes_by_role.end()) {
            throw ContractError("canonical APT v2 component bytes are absent");
        }
        bundle_detail::verify_transport_descriptor(
            descriptor, found->second, result.manifest.kind);
        if (result.total_byte_count >
            maximum_portable_bundle_bytes_v2 - descriptor.byte_count) {
            throw ContractError(
                "canonical APT v2 complete portable bundle exceeds 20 MiB");
        }
        result.total_byte_count += descriptor.byte_count;
    }
    const auto &bytes = payload.component_bytes_by_role;

    if (result.manifest.kind == BundleKind::baseline) {
        if (result.manifest.profile != "citlali-beammap-baseline-apt-v2") {
            throw ContractError(
                "canonical APT v2 baseline profile is not exact");
        }
        auto fields = verify_fields_component(bytes.at("fields"));
        ++result.parser_count;
        auto sources = verify_sources_component(bytes.at("sources"));
        ++result.parser_count;
        auto apt = verify_apt_component(bytes.at("apt"), fields.document);
        ++result.parser_count;
        bundle_detail::verify_parsed_descriptor(descriptors.at("fields"),
                                                fields);
        bundle_detail::verify_parsed_descriptor(descriptors.at("sources"),
                                                sources);
        bundle_detail::verify_parsed_descriptor(descriptors.at("apt"), apt);
        bundle_detail::require_root_context(fields, result.manifest);
        bundle_detail::require_root_context(sources, result.manifest);
        bundle_detail::require_root_context(apt, result.manifest);
        if (apt.document.kind != BundleKind::baseline ||
            apt.document.field_rules != fields.document) {
            throw ContractError(
                "canonical APT v2 baseline field catalog disagrees");
        }
        bundle_detail::validate_baseline_inventory(apt.document,
                                                   sources.document);
        result.fields = std::move(fields.document);
        result.sources = std::move(sources.document);
        result.apt = std::move(apt.document);
        return result;
    }

    if (result.manifest.profile != "citlali-observation-matched-apt-v2") {
        throw ContractError(
            "canonical APT v2 matched profile is not exact");
    }
    auto baseline_manifest =
        verify_manifest_component(bytes.at("baseline-manifest"));
    ++result.parser_count;
    bundle_detail::verify_parsed_descriptor(
        descriptors.at("baseline-manifest"), baseline_manifest);
    if (baseline_manifest.document.kind != BundleKind::baseline ||
        baseline_manifest.document.profile !=
            "citlali-beammap-baseline-apt-v2") {
        throw ContractError(
            "canonical APT v2 embedded baseline manifest is not baseline v2");
    }
    const auto baseline_identity = bundle_detail::bundle_identity(
        baseline_manifest.document, baseline_manifest.digests);
    if (!result.manifest.baseline_parent ||
        *result.manifest.baseline_parent != baseline_identity) {
        throw ContractError(
            "canonical APT v2 embedded baseline parent disagrees");
    }
    auto baseline_receipt = publication::parse_canonical_receipt(
        bytes.at("baseline-receipt"), receipt_schema_v2,
        bundle_transport_scope_v2);
    ++result.parser_count;
    publication::validate_receipt_binding(bytes.at("baseline-manifest"),
                                          baseline_receipt);
    if (baseline_receipt.envelope_sha256 !=
            baseline_manifest.digests.envelope_sha256 ||
        descriptors.at("baseline-receipt").semantic_sha256 !=
            baseline_manifest.digests.semantic_sha256 ||
        descriptors.at("baseline-receipt").envelope_sha256 !=
            baseline_manifest.digests.envelope_sha256 ||
        descriptors.at("baseline-receipt").row_count != 5) {
        throw ContractError(
            "canonical APT v2 embedded baseline receipt disagrees");
    }

    auto baseline_fields =
        verify_fields_component(bytes.at("baseline-fields"));
    ++result.parser_count;
    auto baseline_sources =
        verify_sources_component(bytes.at("baseline-sources"));
    ++result.parser_count;
    auto baseline_apt = verify_apt_component(
        bytes.at("baseline-apt"), baseline_fields.document);
    ++result.parser_count;
    bundle_detail::verify_parsed_descriptor(
        descriptors.at("baseline-fields"), baseline_fields);
    bundle_detail::verify_parsed_descriptor(
        descriptors.at("baseline-sources"), baseline_sources);
    bundle_detail::verify_parsed_descriptor(
        descriptors.at("baseline-apt"), baseline_apt);
    const auto inner_descriptors =
        bundle_detail::descriptor_map(baseline_manifest.document);
    const auto require_snapshot_match = [&](std::string_view outer_role,
                                            std::string_view inner_role) {
        const auto &outer = descriptors.at(std::string(outer_role));
        const auto &inner = inner_descriptors.at(std::string(inner_role));
        if (outer.schema != inner.schema ||
            outer.semantic_sha256 != inner.semantic_sha256 ||
            outer.envelope_sha256 != inner.envelope_sha256 ||
            outer.transport_sha256 != inner.transport_sha256 ||
            outer.byte_count != inner.byte_count ||
            outer.row_count != inner.row_count) {
            throw ContractError(
                "canonical APT v2 baseline snapshot component was swapped");
        }
    };
    require_snapshot_match("baseline-apt", "apt");
    require_snapshot_match("baseline-fields", "fields");
    require_snapshot_match("baseline-sources", "sources");
    if (baseline_fields.kind != BundleKind::baseline ||
        baseline_sources.kind != BundleKind::baseline ||
        baseline_apt.kind != BundleKind::baseline ||
        baseline_fields.issuance != baseline_manifest.document.issuance ||
        baseline_sources.issuance != baseline_manifest.document.issuance ||
        baseline_apt.issuance != baseline_manifest.document.issuance ||
        baseline_fields.observation != baseline_manifest.document.observation ||
        baseline_sources.observation != baseline_manifest.document.observation ||
        baseline_apt.observation != baseline_manifest.document.observation ||
        baseline_apt.document.field_rules != baseline_fields.document) {
        throw ContractError(
            "canonical APT v2 embedded baseline context disagrees");
    }
    bundle_detail::validate_baseline_inventory(baseline_apt.document,
                                               baseline_sources.document);

    auto fields = verify_fields_component(bytes.at("fields"));
    ++result.parser_count;
    auto sources = verify_sources_component(bytes.at("sources"));
    ++result.parser_count;
    auto apt = verify_apt_component(bytes.at("apt"), fields.document);
    ++result.parser_count;
    auto relation =
        verify_relation_component(bytes.at("relation"), result.manifest.observation);
    ++result.parser_count;
    auto exceptions = verify_exceptions_component(
        bytes.at("exceptions"), baseline_identity,
        result.manifest.observation);
    ++result.parser_count;
    bundle_detail::verify_parsed_descriptor(descriptors.at("fields"), fields);
    bundle_detail::verify_parsed_descriptor(descriptors.at("sources"), sources);
    bundle_detail::verify_parsed_descriptor(descriptors.at("apt"), apt);
    bundle_detail::verify_parsed_descriptor(descriptors.at("relation"),
                                            relation);
    bundle_detail::verify_parsed_descriptor(descriptors.at("exceptions"),
                                            exceptions);
    bundle_detail::require_root_context(fields, result.manifest);
    bundle_detail::require_root_context(sources, result.manifest);
    bundle_detail::require_root_context(apt, result.manifest);
    bundle_detail::require_root_context(relation, result.manifest);
    bundle_detail::require_root_context(exceptions, result.manifest);
    if (!result.manifest.target_parent ||
        relation.document.baseline_parent != baseline_identity ||
        relation.document.target_parent != *result.manifest.target_parent ||
        result.manifest.target_manifest_sha256 !=
            result.manifest.target_parent->semantic_sha256 ||
        result.manifest.relation_sha256 !=
            relation.digests.semantic_sha256 ||
        result.manifest.field_rules_sha256 != fields.digests.semantic_sha256 ||
        result.manifest.exceptions_sha256 !=
            exceptions.digests.semantic_sha256) {
        throw ContractError(
            "canonical APT v2 matched logical component identity disagrees");
    }
    auto target = bundle_detail::reconstruct_target(
        apt.document, relation.document, sources.document);
    auto seed_dispositions = bundle_detail::validate_matched_semantics(
        apt.document, fields.document, sources.document, relation.document,
        exceptions.document, baseline_apt.document, baseline_fields.document);

    result.fields = std::move(fields.document);
    result.sources = std::move(sources.document);
    result.apt = std::move(apt.document);
    result.relation = std::move(relation.document);
    result.exceptions = std::move(exceptions.document);
    result.target = std::move(target);
    result.seed_dispositions = std::move(seed_dispositions);
    result.baseline_snapshot = VerifiedBaselineSnapshot{
        std::move(baseline_manifest.document),
        std::move(baseline_manifest.digests), std::move(baseline_receipt),
        std::move(baseline_apt.document), std::move(baseline_fields.document),
        std::move(baseline_sources.document)};
    return result;
}

inline PreparedBundle prepare_baseline_bundle(
    AptTable apt, std::vector<SourceRecord> sources,
    std::string issuance_class = "fresh") {
    if (apt.kind != BundleKind::baseline) {
        throw ContractError(
            "canonical APT v2 baseline preparation received matched APT");
    }
    const auto issuance = apt.issuance;
    const auto observation = apt.observation;
    const auto fields_component = serialize_fields_component(
        apt.issuance, BundleKind::baseline, apt.observation,
        apt.field_rules);
    const auto sources_component = serialize_sources_component(
        apt.issuance, BundleKind::baseline, apt.observation,
        std::move(sources));
    const auto apt_component = serialize_apt_component(std::move(apt));

    BundleManifest manifest;
    manifest.kind = BundleKind::baseline;
    manifest.profile = "citlali-beammap-baseline-apt-v2";
    manifest.issuance_class = std::move(issuance_class);
    manifest.issuance = issuance;
    manifest.observation = observation;
    manifest.components = {
        bundle_detail::make_descriptor("apt", apt_component),
        bundle_detail::make_descriptor("fields", fields_component),
        bundle_detail::make_descriptor("sources", sources_component)};
    const auto manifest_component = serialize_manifest_component(manifest);
    const auto receipt = publication::make_receipt_binding(
        std::string(receipt_schema_v2), std::string(bundle_transport_scope_v2),
        manifest_component.digests.envelope_sha256,
        manifest_component.bytes);
    BundlePayload payload;
    payload.root_manifest_bytes = manifest_component.bytes;
    payload.root_receipt_bytes = publication::canonical_receipt_bytes(receipt);
    payload.component_bytes_by_role = {
        {"apt", apt_component.bytes},
        {"fields", fields_component.bytes},
        {"sources", sources_component.bytes}};
    const auto verified = verify_bundle_payload(payload);
    return {std::move(payload), verified.identity,
            verified.total_byte_count};
}

inline PreparedBundle prepare_matched_bundle(
    AptTable apt, RelationTable relation,
    std::vector<SourceRecord> sources,
    std::vector<ExceptionRecord> exceptions,
    const VerifiedBundle &baseline,
    std::string issuance_class = "fresh") {
    if (apt.kind != BundleKind::matched ||
        baseline.manifest.kind != BundleKind::baseline ||
        !baseline.payload.component_bytes_by_role.contains("apt") ||
        !baseline.payload.component_bytes_by_role.contains("fields") ||
        !baseline.payload.component_bytes_by_role.contains("sources")) {
        throw ContractError(
            "canonical APT v2 matched preparation lacks verified baseline");
    }
    const auto fields_component = serialize_fields_component(
        apt.issuance, BundleKind::matched, apt.observation, apt.field_rules);
    const auto sources_component = serialize_sources_component(
        apt.issuance, BundleKind::matched, apt.observation,
        std::move(sources));
    const auto apt_component = serialize_apt_component(apt);
    const auto target_parent = relation.target_parent;
    const auto relation_component =
        serialize_relation_component(std::move(relation));
    const auto exceptions_component = serialize_exceptions_component(
        apt.issuance, apt.observation, baseline.identity,
        std::move(exceptions));

    const auto baseline_descriptors =
        bundle_detail::descriptor_map(baseline.manifest);
    ComponentDescriptor baseline_manifest_descriptor{
        "baseline-manifest", {}, std::string(manifest_schema_v2),
        baseline.manifest_digests.semantic_sha256,
        baseline.manifest_digests.envelope_sha256,
        "sha256:" + citlali::utils::sha256(
            baseline.payload.root_manifest_bytes),
        static_cast<std::uint64_t>(
            baseline.payload.root_manifest_bytes.size()),
        static_cast<std::uint64_t>(baseline.manifest.components.size())};
    baseline_manifest_descriptor.relative_path = content_addressed_basename(
        baseline_manifest_descriptor.transport_sha256,
        baseline_manifest_descriptor.role);
    auto baseline_receipt_descriptor =
        bundle_detail::snapshot_receipt_descriptor(
            baseline.payload.root_receipt_bytes,
            baseline.manifest_digests);

    BundleManifest manifest;
    manifest.kind = BundleKind::matched;
    manifest.profile = "citlali-observation-matched-apt-v2";
    manifest.issuance_class = std::move(issuance_class);
    manifest.issuance = apt.issuance;
    manifest.observation = apt.observation;
    manifest.baseline_parent = baseline.identity;
    manifest.target_parent = target_parent;
    manifest.target_manifest_sha256 = target_parent.semantic_sha256;
    manifest.relation_sha256 = relation_component.digests.semantic_sha256;
    manifest.field_rules_sha256 = fields_component.digests.semantic_sha256;
    manifest.exceptions_sha256 =
        exceptions_component.digests.semantic_sha256;
    manifest.components = {
        bundle_detail::make_descriptor("apt", apt_component),
        bundle_detail::make_descriptor("relation", relation_component),
        bundle_detail::make_descriptor("fields", fields_component),
        bundle_detail::make_descriptor("sources", sources_component),
        bundle_detail::make_descriptor("exceptions", exceptions_component),
        bundle_detail::snapshot_descriptor(
            "baseline-apt", baseline_descriptors.at("apt")),
        bundle_detail::snapshot_descriptor(
            "baseline-fields", baseline_descriptors.at("fields")),
        bundle_detail::snapshot_descriptor(
            "baseline-sources", baseline_descriptors.at("sources")),
        std::move(baseline_manifest_descriptor),
        std::move(baseline_receipt_descriptor)};
    const auto manifest_component = serialize_manifest_component(manifest);
    const auto receipt = publication::make_receipt_binding(
        std::string(receipt_schema_v2), std::string(bundle_transport_scope_v2),
        manifest_component.digests.envelope_sha256,
        manifest_component.bytes);
    BundlePayload payload;
    payload.root_manifest_bytes = manifest_component.bytes;
    payload.root_receipt_bytes = publication::canonical_receipt_bytes(receipt);
    payload.component_bytes_by_role = {
        {"apt", apt_component.bytes},
        {"relation", relation_component.bytes},
        {"fields", fields_component.bytes},
        {"sources", sources_component.bytes},
        {"exceptions", exceptions_component.bytes},
        {"baseline-apt", baseline.payload.component_bytes_by_role.at("apt")},
        {"baseline-fields",
         baseline.payload.component_bytes_by_role.at("fields")},
        {"baseline-sources",
         baseline.payload.component_bytes_by_role.at("sources")},
        {"baseline-manifest", baseline.payload.root_manifest_bytes},
        {"baseline-receipt", baseline.payload.root_receipt_bytes}};
    const auto verified = verify_bundle_payload(payload);
    return {std::move(payload), verified.identity,
            verified.total_byte_count};
}

inline VerifiedBundle admit_fresh_bundle(BundlePayload payload) {
    auto verified = verify_bundle_payload(std::move(payload));
    if (verified.manifest.issuance_class != "fresh" ||
        (verified.baseline_snapshot &&
         verified.baseline_snapshot->manifest.issuance_class != "fresh")) {
        throw ContractError(
            "canonical APT v2 guardian rejects migration-only issuance");
    }
    return verified;
}

inline publication::BundlePublicationResult publish_prepared_bundle(
    const std::filesystem::path &root_manifest_path,
    const PreparedBundle &prepared,
    const publication::BundlePublicationHooks &hooks = {}) {
    if (!root_manifest_path.is_absolute() ||
        root_manifest_path.filename() != root_manifest_name_v2) {
        throw ContractError(
            "canonical APT v2 publication root must be absolute manifest.ecsv");
    }
    const auto directory = root_manifest_path.parent_path();
    if (!std::filesystem::is_directory(directory) ||
        std::filesystem::directory_iterator(directory) !=
            std::filesystem::directory_iterator{}) {
        throw ContractError(
            "canonical APT v2 publication requires an existing empty bundle directory");
    }
    const auto verified = verify_bundle_payload(prepared.payload);
    if (verified.identity != prepared.identity ||
        verified.total_byte_count != prepared.total_byte_count) {
        throw ContractError(
            "canonical APT v2 prepared publication identity disagrees");
    }
    auto descriptors = verified.manifest.components;
    std::sort(descriptors.begin(), descriptors.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.role < rhs.role;
              });
    publication::BundlePublicationPlan plan;
    std::map<std::string, std::string> role_by_filename;
    for (const auto &descriptor : descriptors) {
        const auto member = directory / descriptor.relative_path;
        plan.members.emplace_back(
            member,
            prepared.payload.component_bytes_by_role.at(descriptor.role));
        role_by_filename.emplace(descriptor.relative_path, descriptor.role);
    }
    // The root manifest is the last noncompletion member.
    plan.members.emplace_back(root_manifest_path,
                              prepared.payload.root_manifest_bytes);
    plan.receipt_path = directory / root_receipt_name_v2;
    plan.receipt_bytes = prepared.payload.root_receipt_bytes;
    plan.validate = [role_by_filename = std::move(role_by_filename)](
        const std::vector<std::pair<std::filesystem::path, std::string>> &members,
        std::string_view receipt_bytes) {
        BundlePayload payload;
        payload.root_receipt_bytes = std::string(receipt_bytes);
        for (const auto &[path, bytes] : members) {
            if (path.filename() == root_manifest_name_v2) {
                if (!payload.root_manifest_bytes.empty()) {
                    throw ContractError(
                        "canonical APT v2 publication manifest is duplicate");
                }
                payload.root_manifest_bytes = bytes;
                continue;
            }
            const auto role = role_by_filename.find(path.filename().string());
            if (role == role_by_filename.end() ||
                !payload.component_bytes_by_role
                     .emplace(role->second, bytes)
                     .second) {
                throw ContractError(
                    "canonical APT v2 publication member is unknown/duplicate");
            }
        }
        (void)verify_bundle_payload(std::move(payload));
    };
    return publication::publish_canonical_bundle(plan, hooks);
}

namespace bundle_detail {

inline std::string read_regular_file_once(
    const std::filesystem::path &path, std::uint64_t maximum_bytes) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || status.type() != std::filesystem::file_type::regular) {
        throw ContractError(
            "canonical APT v2 member is missing, symlinked, or nonregular: " +
            path.string());
    }
    const auto size = std::filesystem::file_size(path, error);
    if (error || size > maximum_bytes) {
        throw ContractError(
            "canonical APT v2 member exceeds its read bound: " +
            path.string());
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw ContractError("canonical APT v2 member cannot be opened: " +
                            path.string());
    }
    std::string bytes{std::istreambuf_iterator<char>(stream),
                      std::istreambuf_iterator<char>()};
    if (stream.bad() || bytes.size() != size) {
        throw ContractError("canonical APT v2 member changed while reading: " +
                            path.string());
    }
    return bytes;
}

inline void require_no_aliases(
    const std::vector<std::filesystem::path> &paths) {
    for (std::size_t left = 0; left < paths.size(); ++left) {
        for (std::size_t right = left + 1; right < paths.size(); ++right) {
            std::error_code error;
            if (std::filesystem::equivalent(paths[left], paths[right], error) &&
                !error) {
                throw ContractError(
                    "canonical APT v2 member roles alias one inode");
            }
            if (error) {
                throw ContractError(
                    "canonical APT v2 member alias check failed");
            }
        }
    }
}

}  // namespace bundle_detail

inline VerifiedBundle verify_bundle_filesystem(
    const std::filesystem::path &root_manifest_path,
    bool require_fresh = false) {
    if (!root_manifest_path.is_absolute() ||
        root_manifest_path.filename() != root_manifest_name_v2 ||
        root_manifest_path.parent_path().empty()) {
        throw ContractError(
            "canonical APT v2 root manifest locator must be absolute manifest.ecsv");
    }
    const auto directory = root_manifest_path.parent_path();
    const auto root_receipt_path = directory / root_receipt_name_v2;
    BundlePayload payload;

    // Completion is checked first. The small hard bound prevents an
    // untrusted completion marker from becoming an allocation authority.
    payload.root_receipt_bytes = bundle_detail::read_regular_file_once(
        root_receipt_path, 4096);
    auto parsed_receipt = publication::parse_canonical_receipt(
        payload.root_receipt_bytes, receipt_schema_v2,
        bundle_transport_scope_v2);
    if (parsed_receipt.byte_count > maximum_portable_bundle_bytes_v2) {
        throw ContractError(
            "canonical APT v2 root manifest receipt count exceeds bound");
    }
    payload.root_manifest_bytes = bundle_detail::read_regular_file_once(
        root_manifest_path, parsed_receipt.byte_count);
    publication::validate_receipt_binding(payload.root_manifest_bytes,
                                          parsed_receipt);
    auto parsed_root =
        verify_manifest_component(payload.root_manifest_bytes);
    const auto descriptors =
        bundle_detail::descriptor_map(parsed_root.document);

    std::set<std::filesystem::path> expected_paths{
        root_manifest_path.lexically_normal(),
        root_receipt_path.lexically_normal()};
    std::vector<std::filesystem::path> member_paths{
        root_manifest_path, root_receipt_path};
    std::uint64_t admitted_bytes =
        static_cast<std::uint64_t>(payload.root_receipt_bytes.size()) +
        static_cast<std::uint64_t>(payload.root_manifest_bytes.size());
    if (admitted_bytes > maximum_portable_bundle_bytes_v2) {
        throw ContractError(
            "canonical APT v2 root files exceed the complete bundle bound");
    }
    for (const auto &[role, descriptor] : descriptors) {
        const std::filesystem::path relative{descriptor.relative_path};
        if (relative.is_absolute() || relative.has_parent_path() ||
            relative.filename() != relative) {
            throw ContractError(
                "canonical APT v2 component locator is not one basename");
        }
        const auto member = directory / relative;
        if (descriptor.byte_count >
            maximum_portable_bundle_bytes_v2 - admitted_bytes) {
            throw ContractError(
                "canonical APT v2 manifest declares an oversized complete bundle");
        }
        admitted_bytes += descriptor.byte_count;
        expected_paths.insert(member.lexically_normal());
        member_paths.push_back(member);
        payload.component_bytes_by_role.emplace(
            role, bundle_detail::read_regular_file_once(
                      member, descriptor.byte_count));
    }
    bundle_detail::require_no_aliases(member_paths);
    std::error_code directory_error;
    for (std::filesystem::directory_iterator iterator(directory,
                                                       directory_error),
         end;
         !directory_error && iterator != end; iterator.increment(directory_error)) {
        if (!expected_paths.contains(iterator->path().lexically_normal())) {
            throw ContractError(
                "canonical APT v2 bundle directory contains an extra member");
        }
    }
    if (directory_error) {
        throw ContractError(
            "canonical APT v2 bundle directory enumeration failed");
    }
    auto verified = verify_bundle_payload(
        std::move(payload), std::move(parsed_receipt),
        std::move(parsed_root));
    verified.manifest_path = root_manifest_path;
    verified.receipt_path = root_receipt_path;
    if (require_fresh &&
        (verified.manifest.issuance_class != "fresh" ||
         (verified.baseline_snapshot &&
          verified.baseline_snapshot->manifest.issuance_class != "fresh"))) {
        throw ContractError(
            "canonical APT v2 guardian rejects migration-only issuance");
    }
    return verified;
}

}  // namespace citlali::pipeline::canonical_apt_v2
