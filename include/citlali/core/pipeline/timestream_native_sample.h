#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

using TimestreamNetworkId = std::int32_t;
using TimestreamNativeRow = std::int64_t;
using TimestreamNativeRevision = std::uint64_t;

struct NativeSampleKey {
    TimestreamNetworkId network_id = -1;
    TimestreamNativeRow native_row = -1;

    friend bool operator==(const NativeSampleKey &lhs,
                           const NativeSampleKey &rhs) noexcept {
        return lhs.network_id == rhs.network_id &&
               lhs.native_row == rhs.native_row;
    }

    friend bool operator<(const NativeSampleKey &lhs,
                          const NativeSampleKey &rhs) noexcept {
        if (lhs.network_id != rhs.network_id) {
            return lhs.network_id < rhs.network_id;
        }
        return lhs.native_row < rhs.native_row;
    }
};

// This identity names a delivered row and its reconstructed timestamp.  It
// deliberately makes no claim about which physical detector integration
// event that delivered timestamp represents.
class NativeSampleIdentity {
public:
    NativeSampleIdentity(TimestreamNetworkId network_id,
                         TimestreamNativeRow native_row,
                         double reconstructed_time_unix_sec)
        : key_{network_id, native_row},
          reconstructed_time_unix_sec_{reconstructed_time_unix_sec} {
        if (network_id < 0) {
            throw std::invalid_argument(
                "native sample network identity must be nonnegative");
        }
        if (native_row < 0) {
            throw std::invalid_argument(
                "native sample row identity must be nonnegative");
        }
        if (!std::isfinite(reconstructed_time_unix_sec)) {
            throw std::invalid_argument(
                "native reconstructed timestamp must be finite");
        }
    }

    const NativeSampleKey &key() const noexcept { return key_; }
    TimestreamNetworkId network_id() const noexcept {
        return key_.network_id;
    }
    TimestreamNativeRow native_row() const noexcept {
        return key_.native_row;
    }
    double reconstructed_time_unix_sec() const noexcept {
        return reconstructed_time_unix_sec_;
    }

    friend bool operator==(const NativeSampleIdentity &lhs,
                           const NativeSampleIdentity &rhs) noexcept {
        return lhs.key_ == rhs.key_ &&
               lhs.reconstructed_time_unix_sec_ ==
                   rhs.reconstructed_time_unix_sec_;
    }

private:
    NativeSampleKey key_;
    double reconstructed_time_unix_sec_;
};

struct NativeOperationIdentity {
    std::uint64_t sequence = 0;
    std::int64_t scan_index = -1;

    NativeOperationIdentity(std::uint64_t sequence_,
                            std::int64_t scan_index_)
        : sequence{sequence_}, scan_index{scan_index_} {
        if (scan_index < 0) {
            throw std::invalid_argument(
                "native operation scan index must be nonnegative");
        }
    }

    friend bool operator==(const NativeOperationIdentity &lhs,
                           const NativeOperationIdentity &rhs) noexcept {
        return lhs.sequence == rhs.sequence &&
               lhs.scan_index == rhs.scan_index;
    }
};

// Identifies where a real native sample participated in one relational
// coincidence cohort.  Common-slot position is provenance only; it carries
// no physical-time authority.
struct NativeCoincidenceProvenance {
    std::size_t common_slot = 0;
    std::size_t participant_index = 0;
    TimestreamNetworkId participant_network_id = -1;
    std::optional<std::uint64_t> original_flag_bits;
    std::string original_flag_reason;
};

enum class NativeRevisionAction {
    replaced_by_operation_result,
    preserved_pca_invalid,
};

struct NativeRevisionRecord {
    NativeOperationIdentity operation;
    TimestreamNativeRevision input_revision = 0;
    TimestreamNativeRevision output_revision = 0;
    NativeRevisionAction action =
        NativeRevisionAction::replaced_by_operation_result;
    std::optional<NativeCoincidenceProvenance>
        coincidence_provenance;
};

template <class Value>
class NativeSampleLedger {
public:
    struct Seed {
        NativeSampleIdentity identity;
        Value measured_value;
    };

    struct Record {
        NativeSampleIdentity identity;
        Value measured_value;
        Value current_value;
        TimestreamNativeRevision revision = 0;
        std::vector<NativeRevisionRecord> lineage;
    };

    class Update {
    public:
        static Update replacement(NativeSampleIdentity identity,
                                  TimestreamNativeRevision expected_revision,
                                  Value value,
                                  std::optional<NativeCoincidenceProvenance>
                                      coincidence_provenance =
                                          std::nullopt) {
            return Update{
                std::move(identity), expected_revision,
                NativeRevisionAction::replaced_by_operation_result,
                std::optional<Value>{std::move(value)},
                std::move(coincidence_provenance)};
        }

        static Update preserve_invalid(
            NativeSampleIdentity identity,
            TimestreamNativeRevision expected_revision,
            NativeCoincidenceProvenance coincidence_provenance) {
            return Update{
                std::move(identity), expected_revision,
                NativeRevisionAction::preserved_pca_invalid, std::nullopt,
                std::optional<NativeCoincidenceProvenance>{
                    std::move(coincidence_provenance)}};
        }

        const NativeSampleIdentity &identity() const noexcept {
            return identity_;
        }
        TimestreamNativeRevision expected_revision() const noexcept {
            return expected_revision_;
        }
        NativeRevisionAction action() const noexcept { return action_; }
        const std::optional<Value> &replacement_value() const noexcept {
            return replacement_value_;
        }
        const std::optional<NativeCoincidenceProvenance> &
        coincidence_provenance() const noexcept {
            return coincidence_provenance_;
        }

    private:
        Update(NativeSampleIdentity identity,
               TimestreamNativeRevision expected_revision,
               NativeRevisionAction action,
               std::optional<Value> replacement_value,
               std::optional<NativeCoincidenceProvenance>
                   coincidence_provenance)
            : identity_{std::move(identity)},
              expected_revision_{expected_revision}, action_{action},
              replacement_value_{std::move(replacement_value)},
              coincidence_provenance_{
                  std::move(coincidence_provenance)} {}

        NativeSampleIdentity identity_;
        TimestreamNativeRevision expected_revision_;
        NativeRevisionAction action_;
        std::optional<Value> replacement_value_;
        std::optional<NativeCoincidenceProvenance>
            coincidence_provenance_;
    };

    explicit NativeSampleLedger(std::vector<Seed> seeds) {
        for (auto &seed : seeds) {
            const auto key = seed.identity.key();
            Record record{
                seed.identity, seed.measured_value,
                std::move(seed.measured_value), 0, {}};
            const auto inserted = records_.emplace(key, std::move(record));
            if (!inserted.second) {
                throw std::invalid_argument(
                    "duplicate native sample identity in ledger seed");
            }
        }
    }

    bool contains(const NativeSampleKey &key) const noexcept {
        return records_.find(key) != records_.end();
    }

    const Record &at(const NativeSampleKey &key) const {
        const auto it = records_.find(key);
        if (it == records_.end()) {
            throw std::out_of_range("native sample identity is not in ledger");
        }
        return it->second;
    }

    std::size_t size() const noexcept { return records_.size(); }

    const std::optional<NativeOperationIdentity> &last_operation() const
        noexcept {
        return last_operation_;
    }

    // The entire batch is validated and applied to a copy before the live
    // ledger is swapped.  Any stale identity, duplicate destination, copy
    // failure, or lifecycle error therefore leaves every native sample and
    // the operation lineage unchanged.
    void apply_transaction(const NativeOperationIdentity &operation,
                           const std::vector<Update> &updates) {
        if (last_operation_.has_value() &&
            operation.sequence <= last_operation_->sequence) {
            throw std::logic_error(
                "native operation sequence must increase monotonically");
        }

        std::set<NativeSampleKey> destinations;
        for (const auto &update : updates) {
            if (!destinations.insert(update.identity().key()).second) {
                throw std::logic_error(
                    "native scatter contains a duplicate destination");
            }
            const auto it = records_.find(update.identity().key());
            if (it == records_.end()) {
                throw std::logic_error(
                    "native scatter destination is not present");
            }
            if (!(it->second.identity == update.identity())) {
                throw std::logic_error(
                    "native scatter timestamp or identity changed");
            }
            if (it->second.revision != update.expected_revision()) {
                throw std::logic_error(
                    "native scatter expected revision is stale");
            }
            if (it->second.revision ==
                std::numeric_limits<TimestreamNativeRevision>::max()) {
                throw std::overflow_error(
                    "native sample revision would overflow");
            }
            const bool replacement_expected =
                update.action() ==
                NativeRevisionAction::replaced_by_operation_result;
            if (replacement_expected !=
                update.replacement_value().has_value()) {
                throw std::logic_error(
                    "native scatter action and replacement disagree");
            }
            if (update.coincidence_provenance().has_value()) {
                const auto &provenance =
                    *update.coincidence_provenance();
                if (provenance.participant_network_id !=
                    update.identity().network_id()) {
                    throw std::logic_error(
                        "native scatter participant network changed");
                }
                const bool has_invalidity =
                    provenance.original_flag_bits.value_or(0) != 0 ||
                    !provenance.original_flag_reason.empty();
                if (replacement_expected && has_invalidity) {
                    throw std::logic_error(
                        "PCA-valid scatter cannot carry invalidity");
                }
                if (!replacement_expected && !has_invalidity) {
                    throw std::logic_error(
                        "PCA-invalid scatter requires original flags");
                }
            }
            else if (!replacement_expected) {
                throw std::logic_error(
                    "PCA-invalid scatter requires coincidence provenance");
            }
        }

        auto candidate = records_;
        for (const auto &update : updates) {
            auto &record = candidate.at(update.identity().key());
            const auto input_revision = record.revision;
            const auto output_revision = input_revision + 1;
            if (update.replacement_value().has_value()) {
                record.current_value = *update.replacement_value();
            }
            record.revision = output_revision;
            record.lineage.push_back(NativeRevisionRecord{
                operation, input_revision, output_revision,
                update.action(), update.coincidence_provenance()});
        }

        records_.swap(candidate);
        last_operation_ = operation;
    }

private:
    std::map<NativeSampleKey, Record> records_;
    std::optional<NativeOperationIdentity> last_operation_;
};

}  // namespace citlali::pipeline
