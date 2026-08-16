#pragma once

#include <citlali/core/pipeline/canonical_apt_observation_v1.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace citlali::pipeline {

namespace apt_observation = canonical_apt_observation_v1;
namespace canonical_apt = canonical_apt_v1;

inline constexpr std::string_view apt_producer_raw_manifest_scope_v1 =
    "citlali-apt-producer-raw-manifest-scope-v1";
inline constexpr std::string_view apt_producer_raw_manifest_digest_scope_v1 =
    "citlali-apt-producer-raw-manifest-scope-sha256-v1";

class AptDetectorRelationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class AptDetectorScopeKind {
    published_artifact,
    producer_raw_manifest,
};

enum class PublishedAptKind {
    canonical_baseline,
    matched_observation,
};

struct AptDetectorColumnAddress {
    std::size_t detector_column = 0;
    std::int64_t network = 0;
    std::int64_t kids_tone = 0;

    friend bool operator==(const AptDetectorColumnAddress &,
                           const AptDetectorColumnAddress &) = default;
};

struct AptDetectorBinding {
    std::size_t detector_column = 0;
    std::int64_t uid = 0;
    std::int64_t network = 0;
    std::int64_t kids_tone = 0;
    // A missing flag is permitted only for a contract-validated unmatched
    // observation row. It is never filled or interpreted as a detector flag.
    std::optional<std::int64_t> flag;

    friend bool operator==(const AptDetectorBinding &,
                           const AptDetectorBinding &) = default;
};

// Minimal immutable target/raw identity retained by a matched published APT
// consumer.  It deliberately omits diagnostic locators, KMP inputs and
// parent APT contents.  The target artifact identity binds those producer
// inputs semantically; the fields below are the exact raw occurrence facts
// needed to join the admitted APT to the raw observation at runtime.
struct MatchedAptRawInputIdentity {
    std::int64_t input_key = 0;
    std::int64_t network = 0;
    std::string interface_name;
    std::int64_t channel_count = 0;
    std::int64_t raw_source_key = 0;
    std::string raw_content_sha256;
    std::uint64_t raw_byte_count = 0;
    canonical_apt::ObservationIdentity raw_header_observation;

    friend bool operator==(const MatchedAptRawInputIdentity &,
                           const MatchedAptRawInputIdentity &) = default;
};

struct MatchedAptTargetScope {
    apt_observation::ArtifactIdentity target_artifact;
    canonical_apt::ObservationIdentity observation;
    // Canonical target identity orders inputs by target-local input_key.
    std::vector<MatchedAptRawInputIdentity> ordered_inputs;

    friend bool operator==(const MatchedAptTargetScope &,
                           const MatchedAptTargetScope &) = default;
};

struct PublishedAptScope {
    PublishedAptKind kind = PublishedAptKind::canonical_baseline;
    apt_observation::ArtifactIdentity artifact;
    canonical_apt::ByteTransportHash transport;
    std::string receipt_sha256;
    std::uint64_t receipt_byte_count = 0;
    apt_observation::VerifiedBaselineReference baseline_parent;
    bool parent_content_revalidated = false;
    std::optional<apt_observation::ArtifactIdentity> target_parent;
    std::optional<apt_observation::ArtifactIdentity> relation_parent;
    // Present exactly for a matched-observation artifact.  A canonical
    // baseline has no target/raw child scope.
    std::optional<MatchedAptTargetScope> matched_target;
};

inline bool operator==(const PublishedAptScope &lhs,
                       const PublishedAptScope &rhs) {
    return lhs.kind == rhs.kind && lhs.artifact == rhs.artifact &&
        lhs.transport.scope == rhs.transport.scope &&
        lhs.transport.envelope_sha256 == rhs.transport.envelope_sha256 &&
        lhs.transport.sha256 == rhs.transport.sha256 &&
        lhs.transport.byte_count == rhs.transport.byte_count &&
        lhs.receipt_sha256 == rhs.receipt_sha256 &&
        lhs.receipt_byte_count == rhs.receipt_byte_count &&
        lhs.baseline_parent == rhs.baseline_parent &&
        lhs.parent_content_revalidated == rhs.parent_content_revalidated &&
        lhs.target_parent == rhs.target_parent &&
        lhs.relation_parent == rhs.relation_parent &&
        lhs.matched_target == rhs.matched_target;
}

struct ProducerRawManifestScope {
    std::string schema{apt_producer_raw_manifest_scope_v1};
    std::string producer_scope_reference;
    canonical_apt::ObservationIdentity observation;
    std::vector<canonical_apt::RawInput> inputs;
    std::string manifest_scope_sha256;

    bool requires_published_artifact_join() const noexcept { return true; }

    friend bool operator==(const ProducerRawManifestScope &,
                           const ProducerRawManifestScope &) = default;
};

struct PublishedAptDetectorIdentity {
    apt_observation::RowReference row;

    friend bool operator==(const PublishedAptDetectorIdentity &,
                           const PublishedAptDetectorIdentity &) = default;
};

struct ProducerAptDetectorIdentity {
    std::string schema{apt_producer_raw_manifest_scope_v1};
    std::string producer_scope_reference;
    std::string manifest_scope_sha256;
    std::int64_t local_uid = 0;

    friend bool operator==(const ProducerAptDetectorIdentity &,
                           const ProducerAptDetectorIdentity &) = default;
};

using AptDetectorIdentity =
    std::variant<PublishedAptDetectorIdentity,
                 ProducerAptDetectorIdentity>;

// Detector identity is occurrence/scope plus the typed artifact-local key.
// Column, network, and channel are a checked binding witness, never identity.
struct AptDetectorBindingReference {
    AptDetectorIdentity detector_identity;
    std::size_t detector_column = 0;
    std::int64_t network = 0;
    std::int64_t kids_tone = 0;

    friend bool operator==(const AptDetectorBindingReference &,
                           const AptDetectorBindingReference &) = default;
};

namespace apt_detector_relation_detail {

using NetworkChannel = std::pair<std::int64_t, std::int64_t>;

inline void require_text(std::string_view value, std::string_view label) {
    if (value.empty() || value.find('\n') != std::string_view::npos ||
        value.find('\r') != std::string_view::npos ||
        value.find('\0') != std::string_view::npos) {
        throw AptDetectorRelationError(std::string(label) + " is empty");
    }
}

inline void require_uid(std::int64_t uid) {
    if (uid < 0 || uid > canonical_apt::uid_v1_max) {
        throw AptDetectorRelationError(
            "APT detector uid is outside exact v1 range [0, 2^53-1]");
    }
}

inline std::vector<AptDetectorColumnAddress> normalize_layout(
    std::vector<AptDetectorColumnAddress> layout) {
    if (layout.empty()) {
        throw AptDetectorRelationError(
            "APT detector layout must contain at least one detector");
    }
    std::sort(layout.begin(), layout.end(), [](const auto &lhs,
                                               const auto &rhs) {
        return lhs.detector_column < rhs.detector_column;
    });
    std::set<NetworkChannel> relations;
    for (std::size_t column = 0; column < layout.size(); ++column) {
        const auto &address = layout[column];
        if (address.detector_column != column || address.network < 0 ||
            address.kids_tone < 0 ||
            !relations.emplace(address.network, address.kids_tone).second) {
            throw AptDetectorRelationError(
                "APT detector layout must be a complete injective [0,N) mapping");
        }
    }
    return layout;
}

inline std::vector<AptDetectorBinding> normalize_bindings(
    std::vector<AptDetectorBinding> bindings) {
    if (bindings.empty()) {
        throw AptDetectorRelationError(
            "APT detector relation must contain at least one binding");
    }
    std::sort(bindings.begin(), bindings.end(), [](const auto &lhs,
                                                   const auto &rhs) {
        return lhs.detector_column < rhs.detector_column;
    });
    std::set<std::int64_t> uids;
    std::set<NetworkChannel> relations;
    for (std::size_t column = 0; column < bindings.size(); ++column) {
        const auto &binding = bindings[column];
        require_uid(binding.uid);
        if (binding.detector_column != column || binding.network < 0 ||
            binding.kids_tone < 0 || !uids.insert(binding.uid).second ||
            !relations.emplace(binding.network, binding.kids_tone).second) {
            throw AptDetectorRelationError(
                "APT detector bindings must be complete and injective by column, uid, and network/channel");
        }
    }
    return bindings;
}

inline MatchedAptTargetScope matched_target_scope(
    const apt_observation::TargetManifest &target) {
    // Both observation admission routes have already validated this target,
    // but repeat the public contract validation here so this retained runtime
    // carrier can never be constructed from an unchecked producer object.
    apt_observation::validate(target);
    std::vector<MatchedAptRawInputIdentity> inputs;
    inputs.reserve(target.inputs.size());
    for (const auto &input : target.inputs) {
        inputs.push_back(MatchedAptRawInputIdentity{
            input.input_key,
            input.network,
            input.interface_name,
            input.channel_count,
            input.raw_source.source_key,
            input.raw_source.content_sha256,
            input.raw_source.byte_count,
            input.raw_source.header_observation});
    }
    std::sort(inputs.begin(), inputs.end(), [](const auto &lhs,
                                               const auto &rhs) {
        return lhs.input_key < rhs.input_key;
    });
    return MatchedAptTargetScope{
        apt_observation::artifact_identity(target), target.observation,
        std::move(inputs)};
}

inline std::optional<std::int64_t> flag_from_value(
    const canonical_apt::Value &value, bool nullable) {
    if (const auto *flag = std::get_if<std::int64_t>(&value)) {
        return *flag;
    }
    if (nullable && std::holds_alternative<canonical_apt::NullValue>(value)) {
        return std::nullopt;
    }
    throw AptDetectorRelationError(
        "canonical APT flag is neither exact int64 nor an authorized typed missing value");
}

template <typename Row>
inline std::vector<AptDetectorBinding> join_rows_to_layout(
    const std::vector<Row> &rows,
    std::vector<AptDetectorColumnAddress> layout,
    bool nullable_flag) {
    layout = normalize_layout(std::move(layout));
    if (rows.size() != layout.size()) {
        throw AptDetectorRelationError(
            "APT artifact and detector layout cardinalities differ");
    }
    std::map<NetworkChannel, const Row *> by_relation;
    for (const auto &row : rows) {
        require_uid(row.uid);
        if (!by_relation.emplace(NetworkChannel{row.network, row.channel},
                                 &row).second) {
            throw AptDetectorRelationError(
                "APT artifact repeats a network/channel detector relation");
        }
    }
    std::vector<AptDetectorBinding> bindings;
    bindings.reserve(layout.size());
    for (const auto &address : layout) {
        const auto row = by_relation.find(
            NetworkChannel{address.network, address.kids_tone});
        if (row == by_relation.end()) {
            throw AptDetectorRelationError(
                "APT detector layout references a detector absent from the verified artifact");
        }
        const auto flag = row->second->fields.find("flag");
        if (flag == row->second->fields.end()) {
            throw AptDetectorRelationError(
                "verified APT artifact omits its required flag field");
        }
        bindings.push_back({address.detector_column, row->second->uid,
                            address.network, address.kids_tone,
                            flag_from_value(flag->second, nullable_flag)});
    }
    return normalize_bindings(std::move(bindings));
}

inline std::vector<canonical_apt::RawInput> validate_manifest(
    const canonical_apt::RawManifest &manifest,
    std::size_t binding_count) {
    if (manifest.observation.observation < 0 ||
        manifest.observation.subobservation < 0 ||
        manifest.observation.scan < 0 || manifest.inputs.empty()) {
        throw AptDetectorRelationError(
            "producer raw manifest requires a nonnegative observation and at least one input");
    }
    auto inputs = manifest.inputs;
    std::sort(inputs.begin(), inputs.end(), [](const auto &lhs,
                                               const auto &rhs) {
        return lhs.network < rhs.network;
    });
    std::set<std::int64_t> networks;
    std::set<std::string> interfaces;
    std::uint64_t total = 0;
    for (const auto &input : inputs) {
        const auto expected_interface =
            "toltec" + std::to_string(input.network);
        if (input.network < 0 || input.network > 12 ||
            input.channel_count <= 0 ||
            input.channel_count > canonical_apt::uid_v1_max + 1 ||
            input.interface_name != expected_interface ||
            !networks.insert(input.network).second ||
            !interfaces.insert(input.interface_name).second) {
            throw AptDetectorRelationError(
                "producer raw manifest requires unique canonical TolTEC inputs with positive channel counts");
        }
        const auto count = static_cast<std::uint64_t>(input.channel_count);
        if (total > static_cast<std::uint64_t>(
                        canonical_apt::uid_v1_max) + 1U - count) {
            throw AptDetectorRelationError(
                "producer raw manifest exceeds the canonical v1 identity capacity");
        }
        total += count;
    }
    if (total != binding_count) {
        throw AptDetectorRelationError(
            "producer raw manifest and detector binding cardinalities differ");
    }
    return inputs;
}

inline std::string manifest_scope_preimage(
    std::string_view producer_scope_reference,
    const canonical_apt::ObservationIdentity &observation,
    const std::vector<canonical_apt::RawInput> &inputs) {
    std::string result;
    const auto add = [&](std::string_view label, std::string_view value) {
        result.append(label);
        result.push_back('=');
        result.append(std::to_string(value.size()));
        result.push_back(':');
        result.append(value);
        result.push_back('\n');
    };
    add("scope", apt_producer_raw_manifest_digest_scope_v1);
    add("producer-reference", producer_scope_reference);
    add("observation", std::to_string(observation.observation));
    add("subobservation", std::to_string(observation.subobservation));
    add("scan", std::to_string(observation.scan));
    add("input-count", std::to_string(inputs.size()));
    for (const auto &input : inputs) {
        add("network", std::to_string(input.network));
        add("interface", input.interface_name);
        add("channel-count", std::to_string(input.channel_count));
    }
    return result;
}

inline void require_manifest_coverage(
    const std::vector<canonical_apt::RawInput> &inputs,
    const std::vector<AptDetectorBinding> &bindings) {
    std::map<std::int64_t, std::int64_t> counts;
    for (const auto &input : inputs) {
        counts.emplace(input.network, input.channel_count);
    }
    std::set<NetworkChannel> covered;
    for (const auto &binding : bindings) {
        const auto count = counts.find(binding.network);
        if (count == counts.end() || binding.kids_tone < 0 ||
            binding.kids_tone >= count->second ||
            !binding.flag.has_value() ||
            (*binding.flag != 0 && *binding.flag != 1) ||
            !covered.emplace(binding.network, binding.kids_tone).second) {
            throw AptDetectorRelationError(
                "producer detector binding references a missing, duplicate, or out-of-range raw channel");
        }
    }
    for (const auto &input : inputs) {
        for (std::int64_t channel = 0; channel < input.channel_count;
             ++channel) {
            if (!covered.contains({input.network, channel})) {
                throw AptDetectorRelationError(
                    "producer detector relation does not cover every declared raw channel");
            }
        }
    }
}

}  // namespace apt_detector_relation_detail

class AptDetectorRelation {
public:
    AptDetectorScopeKind scope_kind() const noexcept {
        return std::holds_alternative<PublishedAptScope>(scope_)
            ? AptDetectorScopeKind::published_artifact
            : AptDetectorScopeKind::producer_raw_manifest;
    }

    const PublishedAptScope &published_scope() const {
        const auto *scope = std::get_if<PublishedAptScope>(&scope_);
        if (scope == nullptr) {
            throw AptDetectorRelationError(
                "producer/raw-manifest detector scope is not a published artifact occurrence");
        }
        return *scope;
    }

    const MatchedAptTargetScope &matched_target_scope() const {
        const auto &scope = published_scope();
        if (scope.kind != PublishedAptKind::matched_observation ||
            !scope.matched_target.has_value()) {
            throw AptDetectorRelationError(
                "APT detector relation is not bound to a matched target/raw scope");
        }
        return *scope.matched_target;
    }

    const ProducerRawManifestScope &producer_scope() const {
        const auto *scope = std::get_if<ProducerRawManifestScope>(&scope_);
        if (scope == nullptr) {
            throw AptDetectorRelationError(
                "published detector scope is not a producer/raw-manifest scope");
        }
        return *scope;
    }

    bool requires_published_artifact_join() const noexcept {
        return scope_kind() == AptDetectorScopeKind::producer_raw_manifest;
    }

    const std::vector<AptDetectorBinding> &bindings() const noexcept {
        return bindings_;
    }

    const AptDetectorBinding &binding_for_column(
        std::size_t detector_column) const {
        if (detector_column >= bindings_.size()) {
            throw AptDetectorRelationError(
                "APT detector column is outside the immutable relation");
        }
        return bindings_[detector_column];
    }

    AptDetectorIdentity identity_for_column(
        std::size_t detector_column) const {
        const auto &binding = binding_for_column(detector_column);
        if (const auto *scope = std::get_if<PublishedAptScope>(&scope_)) {
            return PublishedAptDetectorIdentity{
                {scope->artifact.schema, scope->artifact.occurrence,
                 scope->artifact.envelope_sha256, binding.uid}};
        }
        const auto &scope = std::get<ProducerRawManifestScope>(scope_);
        return ProducerAptDetectorIdentity{
            scope.schema, scope.producer_scope_reference,
            scope.manifest_scope_sha256, binding.uid};
    }

    AptDetectorBindingReference binding_reference_for_column(
        std::size_t detector_column) const {
        const auto &binding = binding_for_column(detector_column);
        return {identity_for_column(detector_column),
                binding.detector_column, binding.network,
                binding.kids_tone};
    }

    const AptDetectorBinding &require_binding(
        const AptDetectorBindingReference &reference) const {
        if (reference !=
            binding_reference_for_column(reference.detector_column)) {
            throw AptDetectorRelationError(
                "stale or cross-scope APT detector binding reference");
        }
        return binding_for_column(reference.detector_column);
    }

    bool same_scope(const AptDetectorRelation &other) const noexcept {
        return scope_ == other.scope_;
    }

private:
    using Scope =
        std::variant<PublishedAptScope, ProducerRawManifestScope>;

    AptDetectorRelation(Scope scope,
                        std::vector<AptDetectorBinding> bindings)
        : scope_(std::move(scope)), bindings_(std::move(bindings)) {}

    Scope scope_;
    std::vector<AptDetectorBinding> bindings_;

    friend AptDetectorRelation admit_published_baseline_apt_relation(
        std::string_view, std::string_view,
        std::vector<AptDetectorColumnAddress>);
    friend AptDetectorRelation admit_published_observation_apt_relation(
        std::string_view, std::string_view,
        const apt_observation::VerifiedBaselineDescriptor &,
        std::vector<AptDetectorColumnAddress>);
    friend AptDetectorRelation admit_published_observation_apt_relation(
        std::string_view, std::string_view,
        std::vector<AptDetectorColumnAddress>);
    friend AptDetectorRelation admit_producer_raw_manifest_relation(
        std::string, canonical_apt::RawManifest,
        std::vector<AptDetectorBinding>);
};

inline AptDetectorRelation admit_published_baseline_apt_relation(
    std::string_view bytes, std::string_view receipt_bytes,
    std::vector<AptDetectorColumnAddress> layout) {
    const auto descriptor =
        apt_observation::verify_baseline_descriptor(bytes, receipt_bytes);
    auto bindings = apt_detector_relation_detail::join_rows_to_layout(
        descriptor.document().rows, std::move(layout), false);
    PublishedAptScope scope;
    scope.kind = PublishedAptKind::canonical_baseline;
    scope.artifact = apt_observation::artifact_identity(descriptor);
    scope.transport = descriptor.transport();
    scope.receipt_sha256 = descriptor.receipt_sha256();
    scope.receipt_byte_count = descriptor.receipt_byte_count();
    scope.baseline_parent = apt_observation::baseline_reference(descriptor);
    scope.parent_content_revalidated = true;
    return AptDetectorRelation{std::move(scope), std::move(bindings)};
}

inline AptDetectorRelation admit_published_observation_apt_relation(
    std::string_view bytes, std::string_view receipt_bytes,
    const apt_observation::VerifiedBaselineDescriptor &baseline,
    std::vector<AptDetectorColumnAddress> layout) {
    const auto parsed =
        apt_observation::parse_matched_observation_ecsv_with_receipt(
            bytes, receipt_bytes, baseline);
    auto bindings = apt_detector_relation_detail::join_rows_to_layout(
        parsed.output.rows, std::move(layout), true);
    PublishedAptScope scope;
    scope.kind = PublishedAptKind::matched_observation;
    scope.artifact = apt_observation::artifact_identity(
        parsed.output, baseline, parsed.target, parsed.relation);
    scope.transport = parsed.computed_transport;
    scope.receipt_sha256 =
        "sha256:" + citlali::utils::sha256(receipt_bytes);
    scope.receipt_byte_count = receipt_bytes.size();
    scope.baseline_parent = parsed.output.baseline_parent;
    scope.parent_content_revalidated = parsed.parent_content_revalidated;
    scope.target_parent = parsed.output.target_parent;
    scope.relation_parent = parsed.output.relation_parent;
    scope.matched_target =
        apt_detector_relation_detail::matched_target_scope(parsed.target);
    return AptDetectorRelation{std::move(scope), std::move(bindings)};
}

inline AptDetectorRelation admit_published_observation_apt_relation(
    std::string_view bytes, std::string_view receipt_bytes,
    std::vector<AptDetectorColumnAddress> layout) {
    const auto parsed =
        apt_observation::parse_issued_matched_observation_ecsv_with_receipt(
            bytes, receipt_bytes);
    auto bindings = apt_detector_relation_detail::join_rows_to_layout(
        parsed.output.rows, std::move(layout), true);
    PublishedAptScope scope;
    scope.kind = PublishedAptKind::matched_observation;
    scope.artifact = apt_observation::producer_issued_artifact_identity(
        parsed.output, parsed.target, parsed.relation);
    scope.transport = parsed.computed_transport;
    scope.receipt_sha256 =
        "sha256:" + citlali::utils::sha256(receipt_bytes);
    scope.receipt_byte_count = receipt_bytes.size();
    scope.baseline_parent = parsed.output.baseline_parent;
    scope.parent_content_revalidated = parsed.parent_content_revalidated;
    scope.target_parent = parsed.output.target_parent;
    scope.relation_parent = parsed.output.relation_parent;
    scope.matched_target =
        apt_detector_relation_detail::matched_target_scope(parsed.target);
    return AptDetectorRelation{std::move(scope), std::move(bindings)};
}

inline AptDetectorRelation admit_producer_raw_manifest_relation(
    std::string producer_scope_reference,
    canonical_apt::RawManifest manifest,
    std::vector<AptDetectorBinding> bindings) {
    apt_detector_relation_detail::require_text(
        producer_scope_reference, "producer raw-manifest scope reference");
    bindings =
        apt_detector_relation_detail::normalize_bindings(std::move(bindings));
    auto inputs = apt_detector_relation_detail::validate_manifest(
        manifest, bindings.size());
    apt_detector_relation_detail::require_manifest_coverage(inputs, bindings);
    ProducerRawManifestScope scope;
    scope.producer_scope_reference = std::move(producer_scope_reference);
    scope.observation = manifest.observation;
    scope.inputs = std::move(inputs);
    scope.manifest_scope_sha256 =
        "sha256:" + citlali::utils::sha256(
            apt_detector_relation_detail::manifest_scope_preimage(
                scope.producer_scope_reference, scope.observation,
                scope.inputs));
    return AptDetectorRelation{std::move(scope), std::move(bindings)};
}

}  // namespace citlali::pipeline
