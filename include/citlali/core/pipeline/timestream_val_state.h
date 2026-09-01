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
#include <utility>
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

class ValFindingKey {
public:
    ValFindingKey(ValProducerProductIdentity product,
                  ValAddress address, ValFactCode fact)
        : product_{product}, address_{std::move(address)}, fact_{fact} {}

    const ValProducerProductIdentity &product() const noexcept {
        return product_;
    }
    const ValAddress &address() const noexcept { return address_; }
    ValFactCode fact() const noexcept { return fact_; }

    friend bool operator==(const ValFindingKey &,
                           const ValFindingKey &) = default;
    friend bool operator<(const ValFindingKey &lhs,
                          const ValFindingKey &rhs) noexcept {
        if (lhs.product_ < rhs.product_) return true;
        if (rhs.product_ < lhs.product_) return false;
        if (lhs.address_ < rhs.address_) return true;
        if (rhs.address_ < lhs.address_) return false;
        return lhs.fact_ < rhs.fact_;
    }

private:
    ValProducerProductIdentity product_;
    ValAddress address_;
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
        return {findings_.size() * sizeof(ValFinding), 1,
                parent_ ? 1U : 0U};
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
        if (frozen_) {
            throw std::logic_error("VAL delta builder is already frozen");
        }
        if (!base_snapshot_->contains(address)) {
            throw std::invalid_argument(
                "VAL finding address differs from the base snapshot");
        }
        findings_.push_back(ValFinding{
            ValFindingKey{producer_product_, std::move(address), fact},
            state, cause});
        return *this;
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
    std::shared_ptr<const ValSnapshot> base_snapshot_;
    ValProducerProductIdentity producer_product_;
    std::vector<ValFinding> findings_;
    bool frozen_ = false;
};

}  // namespace citlali::pipeline
