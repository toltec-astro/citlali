#pragma once

#include <citlali/core/pipeline/timestream_native_paired_readout.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace citlali::pipeline {

// VAL 0.1 stores producer-owned facts. It does not interpret them, combine
// them into a score, design an operation, or admit a downstream consumer.
enum class ValProducer : std::uint8_t {
    align,
    ast,
    rtc,
    cal,
    ptc,
};

struct ValGeneration {
    std::uint64_t value = 0;

    friend bool operator==(const ValGeneration &,
                           const ValGeneration &) = default;
};

// Product-instance values are meaningful only in the named producer's typed
// contract. VAL preserves the binding but assigns no cross-producer meaning.
class ValProducerProductIdentity {
public:
    ValProducerProductIdentity(ValProducer producer,
                               std::uint64_t product_instance)
        : producer_{producer}, product_instance_{product_instance} {
        if (product_instance == 0) {
            throw std::invalid_argument(
                "VAL producer product identity must be nonzero");
        }
    }

    ValProducer producer() const noexcept { return producer_; }
    std::uint64_t product_instance() const noexcept {
        return product_instance_;
    }

    friend bool operator==(const ValProducerProductIdentity &,
                           const ValProducerProductIdentity &) = default;
    friend bool operator<(const ValProducerProductIdentity &lhs,
                          const ValProducerProductIdentity &rhs) noexcept {
        if (lhs.producer_ != rhs.producer_) {
            return static_cast<std::uint8_t>(lhs.producer_) <
                   static_cast<std::uint8_t>(rhs.producer_);
        }
        return lhs.product_instance_ < rhs.product_instance_;
    }

private:
    ValProducer producer_;
    std::uint64_t product_instance_;
};

// These codes remain local to the producer contract named by a finding's
// product identity. Keeping them opaque to VAL prevents the container from
// becoming an unapproved scientific registry or inference engine.
class ValFactCode {
public:
    explicit ValFactCode(std::uint32_t value) : value_{value} {
        if (value == 0) {
            throw std::invalid_argument("VAL fact code must be nonzero");
        }
    }
    std::uint32_t value() const noexcept { return value_; }
    friend bool operator==(const ValFactCode &,
                           const ValFactCode &) = default;
    friend bool operator<(const ValFactCode &lhs,
                          const ValFactCode &rhs) noexcept {
        return lhs.value_ < rhs.value_;
    }

private:
    std::uint32_t value_;
};

class ValFactState {
public:
    explicit ValFactState(std::uint32_t value) : value_{value} {
        if (value == 0) {
            throw std::invalid_argument("VAL fact state must be nonzero");
        }
    }
    std::uint32_t value() const noexcept { return value_; }
    friend bool operator==(const ValFactState &,
                           const ValFactState &) = default;

private:
    std::uint32_t value_;
};

class ValFactCause {
public:
    explicit ValFactCause(std::uint32_t value) : value_{value} {
        if (value == 0) {
            throw std::invalid_argument("VAL fact cause must be nonzero");
        }
    }
    std::uint32_t value() const noexcept { return value_; }
    friend bool operator==(const ValFactCause &,
                           const ValFactCause &) = default;

private:
    std::uint32_t value_;
};

class ValSnapshot;

// An address is compact because its exact immutable Paired-D1 handle is owned
// once by the snapshot. Network/native-row plus the occurrence keys and an
// optional detector column resolve the exact sample, occurrence, network, and
// detector identities without copying identity strings into every finding.
class ValAddress {
public:
    const NativeSampleIdentity &sample_identity() const noexcept {
        return sample_identity_;
    }
    std::int64_t parent_readout_occurrence_key() const noexcept {
        return parent_readout_occurrence_key_;
    }
    std::int64_t paired_xr_occurrence_key() const noexcept {
        return paired_xr_occurrence_key_;
    }
    bool detector_bound() const noexcept { return detector_index_ >= 0; }
    std::optional<Eigen::Index> detector_index() const noexcept {
        return detector_bound() ? std::optional<Eigen::Index>{detector_index_}
                                : std::nullopt;
    }

    friend bool operator==(const ValAddress &, const ValAddress &) = default;
    friend bool operator<(const ValAddress &lhs,
                          const ValAddress &rhs) noexcept {
        const auto &lhs_key = lhs.sample_identity_.key();
        const auto &rhs_key = rhs.sample_identity_.key();
        if (lhs_key < rhs_key) return true;
        if (rhs_key < lhs_key) return false;
        if (lhs.sample_identity_.reconstructed_time_unix_sec() !=
            rhs.sample_identity_.reconstructed_time_unix_sec()) {
            return lhs.sample_identity_.reconstructed_time_unix_sec() <
                   rhs.sample_identity_.reconstructed_time_unix_sec();
        }
        if (lhs.parent_readout_occurrence_key_ !=
            rhs.parent_readout_occurrence_key_) {
            return lhs.parent_readout_occurrence_key_ <
                   rhs.parent_readout_occurrence_key_;
        }
        if (lhs.paired_xr_occurrence_key_ !=
            rhs.paired_xr_occurrence_key_) {
            return lhs.paired_xr_occurrence_key_ <
                   rhs.paired_xr_occurrence_key_;
        }
        if (lhs.detector_index_ != rhs.detector_index_) {
            return lhs.detector_index_ < rhs.detector_index_;
        }
        return std::less<const NativePairedReadoutObservation *>{}(
            lhs.paired_product_identity_,
            rhs.paired_product_identity_);
    }

private:
    friend class ValSnapshot;

    ValAddress(NativeSampleIdentity sample_identity,
               std::int64_t parent_readout_occurrence_key,
               std::int64_t paired_xr_occurrence_key,
               Eigen::Index detector_index,
               const NativePairedReadoutObservation *paired_product_identity)
        : sample_identity_{std::move(sample_identity)},
          parent_readout_occurrence_key_{parent_readout_occurrence_key},
          paired_xr_occurrence_key_{paired_xr_occurrence_key},
          detector_index_{detector_index},
          paired_product_identity_{paired_product_identity} {}

    NativeSampleIdentity sample_identity_;
    std::int64_t parent_readout_occurrence_key_;
    std::int64_t paired_xr_occurrence_key_;
    Eigen::Index detector_index_;
    const NativePairedReadoutObservation *paired_product_identity_;
};

enum class ValNativeProductRole : std::uint8_t {
    original_input,
    derived_residual,
};

// A producer creates one immutable identity per native-network realization.
// Its full occurrence/detector support is the exact Paired-D1 network. This
// descriptor owns no payload, snapshot, policy, or per-cell identity text.
// Retaining its handle establishes scoped in-memory identity; equal producer
// numbers in independently created descriptors do not alias. This is not a
// persistent identity or permission to compute/consume either coordinate.
class ValNativeRealization {
public:
    ValNativeRealization(const ValNativeRealization &) = delete;
    ValNativeRealization &operator=(const ValNativeRealization &) = delete;

    static std::shared_ptr<const ValNativeRealization>
    create(std::shared_ptr<const NativePairedReadoutObservation> paired,
           ValProducerProductIdentity product,
           std::uint64_t realization_instance, ValNativeProductRole role,
           TimestreamNetworkId network_id) {
        if (!paired || realization_instance == 0 ||
            (role != ValNativeProductRole::original_input &&
             role != ValNativeProductRole::derived_residual)) {
            throw std::invalid_argument(
                "VAL native realization requires parent, identity and role");
        }
        switch (product.producer()) {
        case ValProducer::align:
        case ValProducer::ast:
        case ValProducer::rtc:
        case ValProducer::cal:
        case ValProducer::ptc:
            break;
        default:
            throw std::invalid_argument(
                "VAL native realization requires a named producer");
        }
        (void)paired->network(network_id);
        return std::shared_ptr<const ValNativeRealization>(
            new ValNativeRealization{std::move(paired), product,
                                     realization_instance, role, network_id});
    }

    const std::shared_ptr<const NativePairedReadoutObservation> &
    paired_handle() const noexcept {
        return paired_;
    }
    const ValProducerProductIdentity &producer_product() const noexcept {
        return product_;
    }
    std::uint64_t realization_instance() const noexcept {
        return realization_instance_;
    }
    ValNativeProductRole role() const noexcept { return role_; }
    TimestreamNetworkId network_id() const noexcept { return network_id_; }

private:
    ValNativeRealization(
        std::shared_ptr<const NativePairedReadoutObservation> paired,
        ValProducerProductIdentity product, std::uint64_t realization_instance,
        ValNativeProductRole role, TimestreamNetworkId network_id)
        : paired_{std::move(paired)}, product_{product},
          realization_instance_{realization_instance}, role_{role},
          network_id_{network_id} {}

    std::shared_ptr<const NativePairedReadoutObservation> paired_;
    ValProducerProductIdentity product_;
    std::uint64_t realization_instance_;
    ValNativeProductRole role_;
    TimestreamNetworkId network_id_;
};

// One coordinate of one explicitly detector-bound native occurrence. The
// producer owns coordinate meaning/units and facts; VAL only preserves their
// exact subject. A moved-from target is not admissible to a delta.
class ValNativeTarget {
public:
    const ValAddress &address() const noexcept { return address_; }
    NativeReadoutCoordinate coordinate() const noexcept { return coordinate_; }
    const std::shared_ptr<const ValNativeRealization> &
    realization_handle() const noexcept {
        return realization_;
    }

    friend bool operator==(const ValNativeTarget &,
                           const ValNativeTarget &) = default;
    friend bool operator<(const ValNativeTarget &lhs,
                          const ValNativeTarget &rhs) noexcept {
        if (lhs.address_ < rhs.address_) return true;
        if (rhs.address_ < lhs.address_) return false;
        if (lhs.coordinate_ != rhs.coordinate_) {
            return static_cast<std::uint8_t>(lhs.coordinate_) <
                   static_cast<std::uint8_t>(rhs.coordinate_);
        }
        // A strict total order for retained instances, not serialized pointer
        // order or an assertion that fresh allocations are the same subject.
        return std::less<const ValNativeRealization *>{}(
            lhs.realization_.get(), rhs.realization_.get());
    }

private:
    friend class ValSnapshot;

    ValNativeTarget(std::shared_ptr<const ValNativeRealization> realization,
                    ValAddress address, NativeReadoutCoordinate coordinate)
        : address_{std::move(address)}, realization_{std::move(realization)},
          coordinate_{coordinate} {}

    ValAddress address_;
    std::shared_ptr<const ValNativeRealization> realization_;
    NativeReadoutCoordinate coordinate_;
};

class RtcOutputGrid;

// An RTC output occurrence is a distinct VAL subject, even when its selected
// representative has the same time/value as an original native occurrence.
// Only the immutable RTC grid can construct this relation. VAL preserves it;
// it does not derive a grid, pointing or scientific-use policy from the address.
class ValRtcOutputTarget {
public:
    const auto &grid_handle() const noexcept { return grid_; }
    const auto &input_snapshot_handle() const noexcept { return input_snapshot_; }
    ValAddress address() const;
    std::size_t slot() const noexcept { return slot_; }
    NativeReadoutCoordinate coordinate() const noexcept { return coordinate_; }
    friend bool operator==(const ValRtcOutputTarget &, const ValRtcOutputTarget &) = default;
    friend bool operator<(const ValRtcOutputTarget &a, const ValRtcOutputTarget &b) noexcept {
        if (a.grid_.get() != b.grid_.get())
            return std::less<const RtcOutputGrid *>{}(a.grid_.get(), b.grid_.get());
        return std::tie(a.network_, a.detector_, a.slot_, a.row_, a.coordinate_) <
               std::tie(b.network_, b.detector_, b.slot_, b.row_, b.coordinate_);
    }
private:
    friend class RtcOutputGrid;
    ValRtcOutputTarget(std::shared_ptr<const RtcOutputGrid> grid,
                       std::shared_ptr<const ValSnapshot> input_snapshot,
                       ValAddress representative, std::size_t slot,
                       NativeReadoutCoordinate coordinate)
        : grid_{std::move(grid)}, input_snapshot_{std::move(input_snapshot)},
          row_{representative.sample_identity().native_row()}, slot_{slot},
          network_{representative.sample_identity().network_id()},
          detector_{static_cast<std::uint32_t>(*representative.detector_index())},
          coordinate_{coordinate} {}
    std::shared_ptr<const RtcOutputGrid> grid_;
    std::shared_ptr<const ValSnapshot> input_snapshot_;
    // Compact indices are interpreted only through these exact immutable
    // parents. This avoids enlarging every existing native VAL finding.
    TimestreamNativeRow row_;
    std::size_t slot_;
    TimestreamNetworkId network_;
    std::uint32_t detector_;
    NativeReadoutCoordinate coordinate_;
};

class ValFindingKey {
public:
    ValFindingKey(ValProducerProductIdentity product, ValAddress address,
                  ValFactCode fact)
        : product_{product}, subject_{std::move(address)}, fact_{fact} {}
    ValFindingKey(ValProducerProductIdentity product, ValNativeTarget target,
                  ValFactCode fact)
        : product_{product}, subject_{std::move(target)}, fact_{fact} {}
    ValFindingKey(ValProducerProductIdentity product, ValRtcOutputTarget target,
                  ValFactCode fact)
        : product_{product}, subject_{std::move(target)}, fact_{fact} {}

    const ValProducerProductIdentity &product() const noexcept {
        return product_;
    }
    ValAddress address() const {
        if (const auto *target = native_target()) return target->address();
        if (const auto *target = rtc_output_target()) return target->address();
        return std::get<ValAddress>(subject_);
    }
    const ValNativeTarget *native_target() const noexcept {
        return std::get_if<ValNativeTarget>(&subject_);
    }
    const ValRtcOutputTarget *rtc_output_target() const noexcept {
        return std::get_if<ValRtcOutputTarget>(&subject_);
    }
    ValFactCode fact() const noexcept { return fact_; }

    friend bool operator==(const ValFindingKey &,
                           const ValFindingKey &) = default;
    friend bool operator<(const ValFindingKey &lhs,
                          const ValFindingKey &rhs) noexcept {
        if (lhs.product_ < rhs.product_) return true;
        if (rhs.product_ < lhs.product_) return false;
        if (lhs.subject_ < rhs.subject_) return true;
        if (rhs.subject_ < lhs.subject_) return false;
        return lhs.fact_ < rhs.fact_;
    }

private:
    ValProducerProductIdentity product_;
    // Unqualified and coordinate-qualified facts are disjoint domains. There
    // is no implicit pair-wide meaning, coordinate broadcast or fallback.
    std::variant<ValAddress, ValNativeTarget, ValRtcOutputTarget> subject_;
    ValFactCode fact_;
};

class ValFinding {
public:
    const ValFindingKey &key() const noexcept { return key_; }
    ValFactState state() const noexcept { return state_; }
    ValFactCause cause() const noexcept { return cause_; }

    friend bool operator==(const ValFinding &,
                           const ValFinding &) = default;
    friend bool operator<(const ValFinding &lhs,
                          const ValFinding &rhs) noexcept {
        return lhs.key_ < rhs.key_;
    }

private:
    friend class ValDeltaBuilder;

    ValFinding(ValFindingKey key, ValFactState state, ValFactCause cause)
        : key_{std::move(key)}, state_{state}, cause_{cause} {}

    ValFindingKey key_;
    ValFactState state_;
    ValFactCause cause_;
};

class ValDelta {
public:
    ValDelta(const ValDelta &) = delete;
    ValDelta &operator=(const ValDelta &) = delete;
    ValDelta(ValDelta &&) noexcept = default;
    ValDelta &operator=(ValDelta &&) noexcept = default;

    const std::shared_ptr<const ValSnapshot> &base_snapshot_handle()
        const noexcept {
        return base_snapshot_;
    }
    const ValProducerProductIdentity &producer_product() const noexcept {
        return producer_product_;
    }
    std::span<const ValFinding> findings() const noexcept {
        return findings_;
    }

private:
    friend class ValDeltaBuilder;
    friend class ValSnapshot;

    ValDelta(std::shared_ptr<const ValSnapshot> base_snapshot,
             ValProducerProductIdentity producer_product,
             std::vector<ValFinding> findings)
        : base_snapshot_{std::move(base_snapshot)},
          producer_product_{producer_product},
          findings_{std::move(findings)} {}

    std::shared_ptr<const ValSnapshot> base_snapshot_;
    ValProducerProductIdentity producer_product_;
    std::vector<ValFinding> findings_;
};

struct ValSnapshotMemoryEvidence {
    std::size_t owned_finding_bytes = 0;
    std::size_t referenced_paired_product_count = 0;
    std::size_t referenced_parent_generation_count = 0;
    // Counts handle references in this delta, not unique descriptor objects.
    std::size_t referenced_native_target_count = 0;
    std::size_t referenced_rtc_output_target_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return owned_finding_bytes;
    }
};

// A snapshot is immutable after construction. A new generation owns only its
// committed delta and references its exact prior generation, so stages do not
// duplicate a large state container merely to claim parallel membership.
class ValSnapshot {
public:
    static std::shared_ptr<const ValSnapshot> initial(
        std::shared_ptr<const NativePairedReadoutObservation> paired) {
        if (!paired) {
            throw std::invalid_argument(
                "initial VAL snapshot requires Paired-D1");
        }
        return std::shared_ptr<const ValSnapshot>(
            new ValSnapshot{std::move(paired)});
    }

    static std::shared_ptr<const ValSnapshot> commit(ValDelta delta) {
        if (!delta.base_snapshot_ || delta.findings_.empty()) {
            throw std::invalid_argument(
                "VAL commit requires a base snapshot and findings");
        }
        if (delta.base_snapshot_->generation_.value ==
            std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("VAL generation would overflow");
        }
        return std::shared_ptr<const ValSnapshot>(new ValSnapshot{
            std::move(delta.base_snapshot_),
            std::move(delta.findings_)});
    }

    ValGeneration generation() const noexcept { return generation_; }
    const std::shared_ptr<const NativePairedReadoutObservation> &
    paired_handle() const noexcept {
        return paired_;
    }
    const NativeObservationScope &scope() const noexcept {
        return paired_->scope();
    }
    const std::shared_ptr<const ValSnapshot> &parent_snapshot_handle()
        const noexcept {
        return parent_;
    }
    std::span<const ValFinding> committed_delta_findings() const noexcept {
        return findings_;
    }

    ValAddress address(TimestreamNetworkId network_id,
                       TimestreamNativeRow native_row,
                       std::optional<Eigen::Index> detector_index =
                           std::nullopt) const {
        const auto &network = paired_->network(network_id);
        const auto &axis = network.occurrence_axis();
        const auto &occurrence = axis.occurrence(native_row);
        Eigen::Index compact_detector_index = -1;
        if (detector_index) {
            (void)network.detector(*detector_index);
            compact_detector_index = *detector_index;
        }
        return ValAddress{
            axis.native_identity(native_row),
            occurrence.parent_readout_occurrence_key,
            occurrence.paired_xr_occurrence_key,
            compact_detector_index,
            paired_.get()};
    }

    bool contains(const ValAddress &address) const noexcept {
        try {
            const auto expected = this->address(
                address.sample_identity().network_id(),
                address.sample_identity().native_row(),
                address.detector_index());
            return expected == address;
        } catch (const std::exception &) {
            return false;
        }
    }

    ValNativeTarget
    native_target(std::shared_ptr<const ValNativeRealization> realization,
                  ValAddress address,
                  NativeReadoutCoordinate coordinate) const {
        ValNativeTarget target{std::move(realization), std::move(address),
                               coordinate};
        if (!contains(target)) {
            throw std::invalid_argument(
                "VAL native target requires exact parent, network, detector "
                "and coordinate");
        }
        return target;
    }

    bool contains(const ValNativeTarget &target) const noexcept {
        const auto &realization = target.realization_handle();
        return realization &&
               realization->paired_handle().get() == paired_.get() &&
               realization->network_id() ==
                   target.address().sample_identity().network_id() &&
               target.address().detector_bound() &&
               (target.coordinate() == NativeReadoutCoordinate::x ||
                target.coordinate() == NativeReadoutCoordinate::r) &&
               contains(target.address());
    }

    bool contains(const ValRtcOutputTarget &target) const noexcept {
        if (!target.grid_handle() || !target.input_snapshot_handle() ||
            !target.address().detector_bound() || !contains(target.address()) ||
            (target.coordinate() != NativeReadoutCoordinate::x &&
             target.coordinate() != NativeReadoutCoordinate::r)) return false;
        // A later immutable generation may attach facts to the same output;
        // an unrelated generation with identical native values may not.
        for (auto current = this; current; current = current->parent_.get())
            if (current == target.input_snapshot_handle().get()) return true;
        return false;
    }

    const NativeReadoutDetectorBinding &detector_binding(
        const ValAddress &address) const {
        if (!contains(address) || !address.detector_index()) {
            throw std::invalid_argument(
                "VAL address has no exact detector binding");
        }
        return paired_->network(address.sample_identity().network_id())
            .detector(*address.detector_index());
    }

    const NativePairedReadoutOccurrenceBinding &occurrence_binding(
        const ValAddress &address) const {
        if (!contains(address)) {
            throw std::invalid_argument(
                "VAL address differs from the bound Paired-D1 product");
        }
        return paired_->network(address.sample_identity().network_id())
            .occurrence_axis().occurrence(
                address.sample_identity().native_row());
    }

    const ValFinding *find(const ValFindingKey &key) const noexcept {
        const auto found = std::lower_bound(
            findings_.begin(), findings_.end(), key,
            [](const ValFinding &candidate,
               const ValFindingKey &requested) {
                return candidate.key() < requested;
            });
        if (found != findings_.end() && found->key() == key) {
            return &*found;
        }
        return parent_ ? parent_->find(key) : nullptr;
    }

    ValSnapshotMemoryEvidence memory_evidence() const noexcept {
        return {findings_.size() * sizeof(ValFinding), 1, parent_ ? 1U : 0U,
                static_cast<std::size_t>(std::count_if(
                    findings_.begin(), findings_.end(),
                    [](const ValFinding &finding) {
                        return finding.key().native_target() != nullptr;
                    })), static_cast<std::size_t>(std::count_if(
                    findings_.begin(), findings_.end(), [](const ValFinding &finding) {
                        return finding.key().rtc_output_target() != nullptr;
                    }))};
    }

private:
    explicit ValSnapshot(
        std::shared_ptr<const NativePairedReadoutObservation> paired)
        : generation_{0}, paired_{std::move(paired)} {}

    ValSnapshot(std::shared_ptr<const ValSnapshot> parent,
                std::vector<ValFinding> findings)
        : generation_{parent->generation_.value + 1},
          paired_{parent->paired_}, parent_{std::move(parent)},
          findings_{std::move(findings)} {}

    ValGeneration generation_;
    std::shared_ptr<const NativePairedReadoutObservation> paired_;
    std::shared_ptr<const ValSnapshot> parent_;
    std::vector<ValFinding> findings_;
};

inline ValAddress ValRtcOutputTarget::address() const {
    if (!grid_ || !input_snapshot_)
        throw std::invalid_argument("VAL RTC output target has no retained parent");
    return input_snapshot_->address(network_, row_, detector_);
}

// The builder is the only mutable VAL object. It is producer-scoped, local to
// one phase, and cannot change the immutable base snapshot. Freeze sorts by
// exact key and rejects an ambiguous duplicate before commit.
class ValDeltaBuilder {
public:
    ValDeltaBuilder(std::shared_ptr<const ValSnapshot> base_snapshot,
                    ValProducerProductIdentity producer_product)
        : base_snapshot_{std::move(base_snapshot)},
          producer_product_{producer_product} {
        if (!base_snapshot_) {
            throw std::invalid_argument(
                "VAL delta builder requires a base snapshot");
        }
    }

    ValDeltaBuilder &propose(ValAddress address, ValFactCode fact,
                             ValFactState state, ValFactCause cause) {
        return propose_key(
            ValFindingKey{producer_product_, std::move(address), fact}, state,
            cause);
    }

    ValDeltaBuilder &propose(ValNativeTarget target, ValFactCode fact,
                             ValFactState state, ValFactCause cause) {
        return propose_key(
            ValFindingKey{producer_product_, std::move(target), fact}, state,
            cause);
    }
    ValDeltaBuilder &propose(ValRtcOutputTarget target, ValFactCode fact,
                             ValFactState state, ValFactCause cause) {
        return propose_key(
            ValFindingKey{producer_product_, std::move(target), fact}, state, cause);
    }

    ValDelta freeze() {
        if (frozen_) {
            throw std::logic_error("VAL delta builder is already frozen");
        }
        std::sort(findings_.begin(), findings_.end());
        if (std::adjacent_find(
                findings_.begin(), findings_.end(),
                [](const ValFinding &lhs, const ValFinding &rhs) {
                    return lhs.key() == rhs.key();
                }) != findings_.end()) {
            throw std::invalid_argument(
                "VAL delta repeats one producer finding key");
        }
        frozen_ = true;
        return ValDelta{base_snapshot_, producer_product_,
                        std::move(findings_)};
    }

private:
    ValDeltaBuilder &propose_key(ValFindingKey key, ValFactState state,
                                 ValFactCause cause) {
        if (frozen_) {
            throw std::logic_error("VAL delta builder is already frozen");
        }
        if (!base_snapshot_->contains(key.address())) {
            throw std::invalid_argument(
                "VAL finding address differs from the base snapshot");
        }
        if (const auto *target = key.native_target();
            target && !base_snapshot_->contains(*target)) {
            throw std::invalid_argument(
                "VAL finding target differs from the base snapshot");
        }
        if (const auto *target = key.rtc_output_target();
            target && !base_snapshot_->contains(*target)) {
            throw std::invalid_argument("VAL RTC output target differs from the base snapshot lineage");
        }
        findings_.push_back(ValFinding{std::move(key), state, cause});
        return *this;
    }

    std::shared_ptr<const ValSnapshot> base_snapshot_;
    ValProducerProductIdentity producer_product_;
    std::vector<ValFinding> findings_;
    bool frozen_ = false;
};

}  // namespace citlali::pipeline
