#pragma once

#include <citlali/core/pipeline/canonical_apt_bundle_v2.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline {

enum class AptDetectorRelationRetention {
    retain,
    discard,
};

struct CanonicalAptRawSourceBindingV2 {
    std::int64_t source_uid = 0;
    std::int64_t network = 0;
    std::string interface_name;
    std::int64_t channel_count = 0;
    std::string content_sha256;
    std::uint64_t byte_count = 0;
    canonical_apt_v2::ObservationIdentity header_observation;

    friend bool operator==(const CanonicalAptRawSourceBindingV2 &,
                           const CanonicalAptRawSourceBindingV2 &) = default;
};

struct CanonicalAptDetectorBindingV2 {
    std::size_t detector_column = 0;
    std::int64_t relation_uid = 0;
    std::int64_t output_uid = 0;
    canonical_apt_v2::ScopedRowReference target;
    std::int64_t target_input_uid = 0;
    std::int64_t raw_source_uid = 0;
    std::int64_t network = 0;
    std::int64_t channel = 0;
    std::uint64_t source_rank = 0;
    std::uint64_t application_rank = 0;
    std::uint64_t presentation_rank = 0;
    canonical_apt_v2::RelationDisposition disposition =
        canonical_apt_v2::RelationDisposition::unmatched;
    std::optional<canonical_apt_v2::ScopedRowReference> selected_seed;
    // This is exactly the baseline-governed field named "flag". A missing
    // value is retained only for a verified unmatched/ambiguous output row.
    std::optional<std::int64_t> flag;

    friend bool operator==(const CanonicalAptDetectorBindingV2 &,
                           const CanonicalAptDetectorBindingV2 &) = default;
};

class CanonicalAptDetectorRelationV2 {
public:
    const canonical_apt_v2::ComponentIdentity &bundle_identity() const
        noexcept {
        return bundle_identity_;
    }

    const canonical_apt_v2::ComponentIdentity &relation_identity() const
        noexcept {
        return relation_identity_;
    }

    const canonical_apt_v2::ObservationIdentity &observation() const
        noexcept {
        return observation_;
    }

    const canonical_apt_v2::ComponentIdentity &target_parent() const
        noexcept {
        return target_parent_;
    }

    const canonical_apt_v2::ComponentIdentity &baseline_parent() const
        noexcept {
        return baseline_parent_;
    }

    const std::vector<CanonicalAptRawSourceBindingV2> &raw_sources() const
        noexcept {
        return raw_sources_;
    }

    const std::vector<CanonicalAptDetectorBindingV2> &bindings() const
        noexcept {
        return bindings_;
    }

    const CanonicalAptDetectorBindingV2 &binding_for_detector_column(
        std::size_t detector_column) const {
        return bindings_.at(detector_column);
    }

private:
    friend CanonicalAptDetectorRelationV2
    admit_canonical_apt_detector_relation_v2(
        const canonical_apt_v2::VerifiedBundle &);

    CanonicalAptDetectorRelationV2(
        canonical_apt_v2::ComponentIdentity bundle_identity,
        canonical_apt_v2::ComponentIdentity relation_identity,
        canonical_apt_v2::ObservationIdentity observation,
        canonical_apt_v2::ComponentIdentity target_parent,
        canonical_apt_v2::ComponentIdentity baseline_parent,
        std::vector<CanonicalAptRawSourceBindingV2> raw_sources,
        std::vector<CanonicalAptDetectorBindingV2> bindings)
        : bundle_identity_{std::move(bundle_identity)},
          relation_identity_{std::move(relation_identity)},
          observation_{observation}, target_parent_{std::move(target_parent)},
          baseline_parent_{std::move(baseline_parent)},
          raw_sources_{std::move(raw_sources)},
          bindings_{std::move(bindings)} {}

    canonical_apt_v2::ComponentIdentity bundle_identity_;
    canonical_apt_v2::ComponentIdentity relation_identity_;
    canonical_apt_v2::ObservationIdentity observation_;
    canonical_apt_v2::ComponentIdentity target_parent_;
    canonical_apt_v2::ComponentIdentity baseline_parent_;
    std::vector<CanonicalAptRawSourceBindingV2> raw_sources_;
    std::vector<CanonicalAptDetectorBindingV2> bindings_;
};

inline CanonicalAptDetectorRelationV2
admit_canonical_apt_detector_relation_v2(
    const canonical_apt_v2::VerifiedBundle &bundle) {
    namespace apt = canonical_apt_v2;

    if (bundle.manifest.kind != apt::BundleKind::matched ||
        bundle.apt.kind != apt::BundleKind::matched ||
        bundle.identity.schema != apt::matched_bundle_schema_v2 ||
        !bundle.relation.has_value() ||
        bundle.apt.observation != bundle.relation->observation ||
        bundle.manifest.observation != bundle.apt.observation) {
        throw apt::ContractError(
            "typed detector relation requires one verified matched-v2 bundle");
    }

    apt::validate(bundle.manifest);
    apt::validate(bundle.identity);
    apt::validate(bundle.apt);
    apt::validate(*bundle.relation);
    for (const auto &source : bundle.sources) apt::validate(source);
    if (!bundle.target.has_value() ||
        !bundle.baseline_snapshot.has_value()) {
        throw apt::ContractError(
            "matched-v2 detector relation lacks verified target/baseline context");
    }
    apt::validate(*bundle.target);
    if (bundle.apt.field_rules != bundle.fields ||
        bundle.apt.issuance != bundle.manifest.issuance ||
        bundle.relation->issuance != bundle.manifest.issuance ||
        bundle.target->observation != bundle.apt.observation ||
        bundle.target->issuance != bundle.relation->target_issuance ||
        bundle.target->sources != bundle.sources ||
        apt::target_identity(*bundle.target) !=
            bundle.relation->target_parent ||
        !bundle.manifest.target_parent.has_value() ||
        *bundle.manifest.target_parent != bundle.relation->target_parent ||
        !bundle.manifest.baseline_parent.has_value() ||
        *bundle.manifest.baseline_parent != bundle.relation->baseline_parent ||
        bundle.baseline_snapshot->manifest.kind !=
            apt::BundleKind::baseline ||
        bundle.baseline_snapshot->apt.kind != apt::BundleKind::baseline ||
        bundle.baseline_snapshot->apt.field_rules !=
            bundle.baseline_snapshot->fields ||
        bundle.relation->baseline_parent.occurrence !=
            bundle.baseline_snapshot->manifest.issuance.occurrence ||
        bundle.relation->baseline_parent.semantic_sha256 !=
            bundle.baseline_snapshot->manifest_digests.semantic_sha256 ||
        bundle.relation->baseline_parent.envelope_sha256 !=
            bundle.baseline_snapshot->manifest_digests.envelope_sha256 ||
        bundle.identity.occurrence != bundle.manifest.issuance.occurrence ||
        bundle.identity.semantic_sha256 !=
            bundle.manifest_digests.semantic_sha256 ||
        bundle.identity.envelope_sha256 !=
            bundle.manifest_digests.envelope_sha256) {
        throw apt::ContractError(
            "matched-v2 detector relation verified contexts disagree");
    }

    const apt::ComponentDescriptor *relation_descriptor = nullptr;
    for (const auto &descriptor : bundle.manifest.components) {
        if (descriptor.role != "relation") continue;
        if (relation_descriptor != nullptr) {
            throw apt::ContractError(
                "matched-v2 bundle repeats its relation component");
        }
        relation_descriptor = &descriptor;
    }
    if (relation_descriptor == nullptr ||
        relation_descriptor->schema != apt::relation_table_schema_v2 ||
        relation_descriptor->semantic_sha256 !=
            bundle.manifest.relation_sha256 ||
        relation_descriptor->row_count != bundle.relation->rows.size()) {
        throw apt::ContractError(
            "matched-v2 bundle omits its exact relation component");
    }
    apt::ComponentIdentity relation_identity{
        relation_descriptor->schema, bundle.manifest.issuance.occurrence,
        relation_descriptor->semantic_sha256,
        relation_descriptor->envelope_sha256};
    apt::validate(relation_identity);

    const apt::FieldRule *flag_rule = nullptr;
    for (const auto &rule : bundle.fields) {
        if (rule.name != "flag") continue;
        if (flag_rule != nullptr) {
            throw apt::ContractError(
                "matched-v2 detector relation repeats baseline flag rule");
        }
        flag_rule = &rule;
    }
    const apt::FieldRule *baseline_flag_rule = nullptr;
    for (const auto &rule : bundle.baseline_snapshot->fields) {
        if (rule.name == "flag") baseline_flag_rule = &rule;
    }
    if (flag_rule == nullptr || baseline_flag_rule == nullptr ||
        flag_rule->datatype != apt::ValueType::int64 ||
        !flag_rule->nullable ||
        flag_rule->operation != apt::FieldOperation::copy_seed_or_null ||
        flag_rule->missing_policy != "typed-null" ||
        baseline_flag_rule->datatype != apt::ValueType::int64 ||
        baseline_flag_rule->nullable ||
        flag_rule->unit != baseline_flag_rule->unit ||
        flag_rule->authority != baseline_flag_rule->authority ||
        flag_rule->authority_reference !=
            baseline_flag_rule->authority_reference ||
        flag_rule->identity_role != baseline_flag_rule->identity_role) {
        throw apt::ContractError(
            "matched-v2 detector relation requires nullable copied baseline flag");
    }

    std::map<std::int64_t, const apt::SourceRecord *> raw_by_uid;
    std::map<std::int64_t, const apt::SourceRecord *> raw_by_network;
    std::vector<CanonicalAptRawSourceBindingV2> raw_sources;
    for (const auto &source : bundle.sources) {
        if (source.role != apt::SourceRole::raw) continue;
        if (!raw_by_uid.emplace(source.source_uid, &source).second ||
            !raw_by_network.emplace(source.network, &source).second) {
            throw apt::ContractError(
                "matched-v2 detector relation repeats a raw source");
        }
        raw_sources.push_back({
            source.source_uid, source.network, source.interface_name,
            source.channel_count, source.content_sha256, source.byte_count,
            source.header_observation});
    }
    std::sort(raw_sources.begin(), raw_sources.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.network < rhs.network;
              });
    if (raw_sources.empty()) {
        throw apt::ContractError(
            "matched-v2 detector relation has no raw source inventory");
    }

    std::map<std::int64_t, const apt::TargetRow *> target_by_uid;
    for (const auto &target : bundle.target->rows) {
        if (!target_by_uid.emplace(target.uid, &target).second) {
            throw apt::ContractError(
                "matched-v2 detector relation repeats a target row");
        }
    }
    std::map<std::int64_t, const apt::AptRow *> seed_by_uid;
    for (const auto &seed : bundle.baseline_snapshot->apt.rows) {
        if (!seed_by_uid.emplace(seed.uid, &seed).second) {
            throw apt::ContractError(
                "matched-v2 detector relation repeats a baseline seed row");
        }
    }

    std::map<std::int64_t, const apt::RelationRecord *> relation_by_output;
    std::set<std::int64_t> relation_uids;
    for (const auto &record : bundle.relation->rows) {
        if (!relation_by_output.emplace(record.output_uid, &record).second ||
            !relation_uids.insert(record.relation_uid).second) {
            throw apt::ContractError(
                "matched-v2 detector relation repeats an output or relation UID");
        }
    }
    if (relation_by_output.size() != bundle.apt.rows.size()) {
        throw apt::ContractError(
            "matched-v2 APT and relation cardinalities differ");
    }

    std::vector<const apt::AptRow *> rows;
    rows.reserve(bundle.apt.rows.size());
    for (const auto &row : bundle.apt.rows) rows.push_back(&row);
    std::sort(rows.begin(), rows.end(), [](const auto *lhs, const auto *rhs) {
        return lhs->presentation_rank < rhs->presentation_rank;
    });

    using ScopedTargetKey = std::tuple<
        std::string, std::string, std::string, std::string, std::int64_t>;
    std::set<ScopedTargetKey> targets;
    std::set<std::pair<std::int64_t, std::int64_t>> raw_channels;
    std::map<std::int64_t, std::int64_t> detector_count_by_network;
    std::vector<CanonicalAptDetectorBindingV2> bindings;
    bindings.reserve(rows.size());
    for (std::size_t detector_column = 0;
         detector_column < rows.size(); ++detector_column) {
        const auto &row = *rows[detector_column];
        if (row.presentation_rank != detector_column) {
            throw apt::ContractError(
                "matched-v2 detector presentation rank is not a complete column permutation");
        }
        const auto relation = relation_by_output.find(row.uid);
        if (relation == relation_by_output.end()) {
            throw apt::ContractError(
                "matched-v2 detector row lacks its relation record");
        }
        const auto &record = *relation->second;
        const auto raw = raw_by_uid.find(record.target_raw_source_uid);
        const auto target = target_by_uid.find(record.target.local_uid);
        if (raw == raw_by_uid.end() ||
            target == target_by_uid.end() ||
            raw->second->network != row.network || row.channel < 0 ||
            row.channel >= raw->second->channel_count ||
            record.presentation_rank != row.presentation_rank ||
            record.target.artifact != bundle.relation->target_parent ||
            target->second->input_uid != record.target_input_uid ||
            target->second->raw_source_uid !=
                record.target_raw_source_uid ||
            target->second->kmp_source_uid !=
                record.target_kmp_source_uid ||
            target->second->kmp_row_index !=
                record.target_kmp_row_index ||
            target->second->source_rank != record.source_rank ||
            target->second->application_rank != record.application_rank ||
            std::bit_cast<std::uint64_t>(
                target->second->tone_frequency_hz) !=
                std::bit_cast<std::uint64_t>(row.tone_frequency_hz) ||
            target->second->array != row.array ||
            target->second->network != row.network ||
            target->second->channel != row.channel ||
            !targets.emplace(
                 record.target.artifact.schema,
                 record.target.artifact.occurrence,
                 record.target.artifact.semantic_sha256,
                 record.target.artifact.envelope_sha256,
                 record.target.local_uid).second ||
            !raw_channels.emplace(row.network, row.channel).second) {
            throw apt::ContractError(
                "matched-v2 detector row/relation/raw-source join is invalid");
        }

        const auto flag_value = row.fields.find("flag");
        if (flag_value == row.fields.end()) {
            throw apt::ContractError(
                "matched-v2 detector row omits baseline flag");
        }
        std::optional<std::int64_t> flag;
        if (const auto integer =
                std::get_if<std::int64_t>(&flag_value->second)) {
            if (record.disposition != apt::RelationDisposition::matched) {
                throw apt::ContractError(
                    "unmatched/ambiguous detector row carries a baseline flag");
            }
            flag = *integer;
            if (!record.selected_seed.has_value() ||
                record.selected_seed->artifact !=
                    bundle.relation->baseline_parent) {
                throw apt::ContractError(
                    "matched detector row lacks its scoped baseline seed");
            }
            const auto seed =
                seed_by_uid.find(record.selected_seed->local_uid);
            if (seed == seed_by_uid.end()) {
                throw apt::ContractError(
                    "matched detector row refers to an absent baseline seed");
            }
            const auto seed_flag = seed->second->fields.find("flag");
            if (seed_flag == seed->second->fields.end() ||
                !std::holds_alternative<std::int64_t>(
                    seed_flag->second) ||
                std::get<std::int64_t>(seed_flag->second) != *integer) {
                throw apt::ContractError(
                    "matched detector flag disagrees with its baseline seed");
            }
        } else if (std::holds_alternative<apt::NullValue>(
                       flag_value->second)) {
            if (record.disposition == apt::RelationDisposition::matched) {
                throw apt::ContractError(
                    "matched detector row has a missing baseline flag");
            }
        } else {
            throw apt::ContractError(
                "baseline detector flag is neither exact int64 nor authorized typed missing");
        }

        bindings.push_back({
            detector_column, record.relation_uid, row.uid, record.target,
            record.target_input_uid, record.target_raw_source_uid,
            row.network, row.channel, record.source_rank,
            record.application_rank, record.presentation_rank,
            record.disposition, record.selected_seed, flag});
        ++detector_count_by_network[row.network];
    }

    for (const auto &[network, source] : raw_by_network) {
        if (detector_count_by_network[network] != source->channel_count) {
            throw apt::ContractError(
                "matched-v2 detector relation does not cover every raw channel");
        }
    }

    return CanonicalAptDetectorRelationV2{
        bundle.identity, std::move(relation_identity), bundle.apt.observation,
        bundle.relation->target_parent, bundle.relation->baseline_parent,
        std::move(raw_sources), std::move(bindings)};
}

}  // namespace citlali::pipeline
