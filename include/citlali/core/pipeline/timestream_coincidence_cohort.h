#pragma once

#include <citlali/core/pipeline/timestream_native_sample.h>

#include <algorithm>
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

enum class CoincidenceCellState {
    mapped_valid,
    mapped_invalid,
    absent,
};

enum class CoincidenceAbsenceReason {
    no_candidate,
    outside_tolerance,
    outside_native_support,
    participant_unavailable,
};

struct NativeInvalidityProvenance {
    std::uint64_t original_flag_bits = 0;
    std::string reason;

    NativeInvalidityProvenance(std::uint64_t original_flag_bits_,
                               std::string reason_)
        : original_flag_bits{original_flag_bits_},
          reason{std::move(reason_)} {
        if (original_flag_bits == 0 && reason.empty()) {
            throw std::invalid_argument(
                "mapped-invalid provenance requires a flag or reason");
        }
    }

    friend bool operator==(const NativeInvalidityProvenance &lhs,
                           const NativeInvalidityProvenance &rhs) {
        return lhs.original_flag_bits == rhs.original_flag_bits &&
               lhs.reason == rhs.reason;
    }
};

class CoincidenceCohortCell {
public:
    static CoincidenceCohortCell mapped_valid(
        NativeSampleIdentity identity,
        TimestreamNativeRevision expected_revision) {
        return CoincidenceCohortCell{
            CoincidenceCellState::mapped_valid, std::move(identity),
            expected_revision, std::nullopt, std::nullopt};
    }

    static CoincidenceCohortCell mapped_invalid(
        NativeSampleIdentity identity,
        TimestreamNativeRevision expected_revision,
        NativeInvalidityProvenance invalidity) {
        return CoincidenceCohortCell{
            CoincidenceCellState::mapped_invalid, std::move(identity),
            expected_revision, std::move(invalidity), std::nullopt};
    }

    static CoincidenceCohortCell absent(
        CoincidenceAbsenceReason reason) {
        return CoincidenceCohortCell{
            CoincidenceCellState::absent, std::nullopt, 0, std::nullopt,
            reason};
    }

    CoincidenceCellState state() const noexcept { return state_; }
    bool is_mapped() const noexcept {
        return state_ != CoincidenceCellState::absent;
    }
    bool pca_valid() const noexcept {
        return state_ == CoincidenceCellState::mapped_valid;
    }
    const std::optional<NativeSampleIdentity> &identity() const noexcept {
        return identity_;
    }
    TimestreamNativeRevision expected_revision() const noexcept {
        return expected_revision_;
    }
    const std::optional<NativeInvalidityProvenance> &invalidity() const
        noexcept {
        return invalidity_;
    }
    const std::optional<CoincidenceAbsenceReason> &absence_reason() const
        noexcept {
        return absence_reason_;
    }

private:
    CoincidenceCohortCell(
        CoincidenceCellState state,
        std::optional<NativeSampleIdentity> identity,
        TimestreamNativeRevision expected_revision,
        std::optional<NativeInvalidityProvenance> invalidity,
        std::optional<CoincidenceAbsenceReason> absence_reason)
        : state_{state}, identity_{std::move(identity)},
          expected_revision_{expected_revision},
          invalidity_{std::move(invalidity)},
          absence_reason_{absence_reason} {}

    CoincidenceCellState state_;
    std::optional<NativeSampleIdentity> identity_;
    TimestreamNativeRevision expected_revision_;
    std::optional<NativeInvalidityProvenance> invalidity_;
    std::optional<CoincidenceAbsenceReason> absence_reason_;
};

class CoincidenceCohort {
public:
    const NativeOperationIdentity &operation() const noexcept {
        return operation_;
    }
    std::size_t slot_count() const noexcept { return slot_count_; }
    std::size_t participant_count() const noexcept {
        return participant_network_ids_.size();
    }
    const std::vector<TimestreamNetworkId> &participant_network_ids() const
        noexcept {
        return participant_network_ids_;
    }

    const CoincidenceCohortCell &cell(
        std::size_t slot, std::size_t participant_index) const {
        return cells_.at(flat_index(slot, participant_index));
    }

    const CoincidenceCohortCell &cell_for_network(
        std::size_t slot, TimestreamNetworkId network_id) const {
        const auto it = participant_index_by_network_.find(network_id);
        if (it == participant_index_by_network_.end()) {
            throw std::out_of_range(
                "network is not a frozen cohort participant");
        }
        return cell(slot, it->second);
    }

private:
    friend class CoincidenceCohortBuilder;

    CoincidenceCohort(
        NativeOperationIdentity operation,
        std::vector<TimestreamNetworkId> participant_network_ids,
        std::map<TimestreamNetworkId, std::size_t>
            participant_index_by_network,
        std::size_t slot_count,
        std::vector<CoincidenceCohortCell> cells)
        : operation_{operation},
          participant_network_ids_{std::move(participant_network_ids)},
          participant_index_by_network_{
              std::move(participant_index_by_network)},
          slot_count_{slot_count}, cells_{std::move(cells)} {}

    std::size_t flat_index(std::size_t slot,
                           std::size_t participant_index) const {
        if (slot >= slot_count_ ||
            participant_index >= participant_network_ids_.size()) {
            throw std::out_of_range("cohort cell index is out of range");
        }
        return slot * participant_network_ids_.size() + participant_index;
    }

    NativeOperationIdentity operation_;
    std::vector<TimestreamNetworkId> participant_network_ids_;
    std::map<TimestreamNetworkId, std::size_t>
        participant_index_by_network_;
    std::size_t slot_count_;
    std::vector<CoincidenceCohortCell> cells_;
};

class CoincidenceCohortBuilder {
public:
    CoincidenceCohortBuilder(
        NativeOperationIdentity operation,
        std::vector<TimestreamNetworkId> participant_network_ids,
        std::size_t slot_count)
        : operation_{operation},
          participant_network_ids_{std::move(participant_network_ids)},
          slot_count_{slot_count} {
        if (participant_network_ids_.empty()) {
            throw std::invalid_argument(
                "cohort participant set must not be empty");
        }
        if (slot_count_ == 0) {
            throw std::invalid_argument(
                "cohort must contain at least one relational slot");
        }
        for (std::size_t index = 0;
             index < participant_network_ids_.size(); ++index) {
            const auto network_id = participant_network_ids_[index];
            if (network_id < 0) {
                throw std::invalid_argument(
                    "cohort network identity must be nonnegative");
            }
            if (!participant_index_by_network_
                     .emplace(network_id, index)
                     .second) {
                throw std::invalid_argument(
                    "cohort participant network is duplicated");
            }
        }
        if (slot_count_ >
            std::numeric_limits<std::size_t>::max() /
                participant_network_ids_.size()) {
            throw std::length_error(
                "cohort rectangular cardinality would overflow");
        }
        cells_.resize(slot_count_ * participant_network_ids_.size());
    }

    void assign_mapped_valid(
        TimestreamNetworkId network_id, std::size_t slot,
        NativeSampleIdentity identity,
        TimestreamNativeRevision expected_revision) {
        assign_mapped(
            network_id, slot,
            CoincidenceCohortCell::mapped_valid(
                std::move(identity), expected_revision));
    }

    void assign_mapped_invalid(
        TimestreamNetworkId network_id, std::size_t slot,
        NativeSampleIdentity identity,
        TimestreamNativeRevision expected_revision,
        NativeInvalidityProvenance invalidity) {
        assign_mapped(
            network_id, slot,
            CoincidenceCohortCell::mapped_invalid(
                std::move(identity), expected_revision,
                std::move(invalidity)));
    }

    void assign_absent(TimestreamNetworkId network_id, std::size_t slot,
                       CoincidenceAbsenceReason reason) {
        auto &target = target_cell(network_id, slot);
        require_unassigned(target);
        target.emplace(CoincidenceCohortCell::absent(reason));
    }

    CoincidenceCohort finish() && {
        std::vector<CoincidenceCohortCell> completed;
        completed.reserve(cells_.size());
        for (auto &cell : cells_) {
            if (!cell.has_value()) {
                throw std::logic_error(
                    "every cohort participant/slot must be explicit");
            }
            completed.push_back(std::move(*cell));
        }
        return CoincidenceCohort{
            operation_, std::move(participant_network_ids_),
            std::move(participant_index_by_network_), slot_count_,
            std::move(completed)};
    }

private:
    std::optional<CoincidenceCohortCell> &target_cell(
        TimestreamNetworkId network_id, std::size_t slot) {
        const auto it = participant_index_by_network_.find(network_id);
        if (it == participant_index_by_network_.end()) {
            throw std::out_of_range(
                "network is not a frozen cohort participant");
        }
        if (slot >= slot_count_) {
            throw std::out_of_range("cohort slot is out of range");
        }
        return cells_.at(slot * participant_network_ids_.size() +
                         it->second);
    }

    static void require_unassigned(
        const std::optional<CoincidenceCohortCell> &target) {
        if (target.has_value()) {
            throw std::logic_error(
                "cohort participant/slot collision is not permitted");
        }
    }

    void assign_mapped(TimestreamNetworkId network_id, std::size_t slot,
                       CoincidenceCohortCell cell) {
        auto &target = target_cell(network_id, slot);
        require_unassigned(target);
        if (!cell.identity().has_value() ||
            cell.identity()->network_id() != network_id) {
            throw std::invalid_argument(
                "mapped native identity does not match participant network");
        }
        if (!mapped_native_rows_.insert(cell.identity()->key()).second) {
            throw std::logic_error(
                "native row reuse within one cohort is not permitted");
        }
        target.emplace(std::move(cell));
    }

    NativeOperationIdentity operation_;
    std::vector<TimestreamNetworkId> participant_network_ids_;
    std::map<TimestreamNetworkId, std::size_t>
        participant_index_by_network_;
    std::size_t slot_count_;
    std::vector<std::optional<CoincidenceCohortCell>> cells_;
    std::set<NativeSampleKey> mapped_native_rows_;
};

// This value exists only inside the excluded-cell PCA working buffer.  It is
// not a detector sample, must never be scattered, and must be finite because
// the existing ordinary PCA masks by multiplication.
class FinitePcaPlaceholder {
public:
    static FinitePcaPlaceholder checked(double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "PCA exclusion placeholder must be finite");
        }
        return FinitePcaPlaceholder{value};
    }

    double value() const noexcept { return value_; }

private:
    explicit FinitePcaPlaceholder(double value) : value_{value} {}
    double value_;
};

class PcaRectangularWorkingSet {
public:
    // Foundational scalar row-cohort buffer.  A production consumer must
    // expand each selected network row into its existing detector columns
    // and detector-level sample flags without changing this native-row map.
    // That detector-column bridge is intentionally outside Phase A.
    std::size_t slot_count() const noexcept { return slot_count_; }
    std::size_t participant_count() const noexcept {
        return participant_count_;
    }
    const NativeOperationIdentity &operation() const noexcept {
        return operation_;
    }
    const std::vector<double> &values() const noexcept { return values_; }
    double &mutable_value(std::size_t slot, std::size_t participant) {
        return values_.at(flat_index(slot, participant));
    }
    double *mutable_values_data_for_pca() noexcept {
        return values_.data();
    }
    void require_all_values_finite_for_pca() const {
        if (!std::all_of(
                values_.begin(), values_.end(),
                [](double value) { return std::isfinite(value); })) {
            throw std::logic_error(
                "PCA rectangular working buffer must remain finite");
        }
    }
    const std::vector<std::uint8_t> &exclusion_flags() const noexcept {
        return exclusion_flags_;
    }
    const std::vector<CoincidenceCellState> &provenance_states() const
        noexcept {
        return provenance_states_;
    }
    const std::vector<TimestreamNetworkId> &participant_network_ids() const
        noexcept {
        return participant_network_ids_;
    }
    const std::vector<std::optional<NativeSampleIdentity>> &
    mapped_identities() const noexcept {
        return mapped_identities_;
    }
    const std::vector<TimestreamNativeRevision> &expected_revisions() const
        noexcept {
        return expected_revisions_;
    }
    const std::vector<std::optional<NativeInvalidityProvenance>> &
    invalidity_provenance() const noexcept {
        return invalidity_provenance_;
    }
    const std::vector<std::optional<CoincidenceAbsenceReason>> &
    absence_reasons() const noexcept {
        return absence_reasons_;
    }

    double value(std::size_t slot, std::size_t participant) const {
        return values_.at(flat_index(slot, participant));
    }
    bool excluded(std::size_t slot, std::size_t participant) const {
        return exclusion_flags_.at(flat_index(slot, participant)) != 0;
    }

private:
    friend PcaRectangularWorkingSet make_pca_rectangular_working_set(
        const NativeSampleLedger<double> &, const CoincidenceCohort &,
        FinitePcaPlaceholder);

    PcaRectangularWorkingSet(
        NativeOperationIdentity operation, std::size_t slot_count,
        std::vector<TimestreamNetworkId> participant_network_ids,
        std::vector<double> values,
        std::vector<std::uint8_t> exclusion_flags,
        std::vector<CoincidenceCellState> provenance_states,
        std::vector<std::optional<NativeSampleIdentity>> mapped_identities,
        std::vector<TimestreamNativeRevision> expected_revisions,
        std::vector<std::optional<NativeInvalidityProvenance>>
            invalidity_provenance,
        std::vector<std::optional<CoincidenceAbsenceReason>>
            absence_reasons)
        : operation_{operation}, slot_count_{slot_count},
          participant_count_{participant_network_ids.size()},
          participant_network_ids_{std::move(participant_network_ids)},
          values_{std::move(values)},
          exclusion_flags_{std::move(exclusion_flags)},
          provenance_states_{std::move(provenance_states)},
          mapped_identities_{std::move(mapped_identities)},
          expected_revisions_{std::move(expected_revisions)},
          invalidity_provenance_{std::move(invalidity_provenance)},
          absence_reasons_{std::move(absence_reasons)} {}

    std::size_t flat_index(std::size_t slot,
                           std::size_t participant) const {
        if (slot >= slot_count_ || participant >= participant_count_) {
            throw std::out_of_range(
                "PCA rectangular working-set index is out of range");
        }
        return slot * participant_count_ + participant;
    }

    NativeOperationIdentity operation_;
    std::size_t slot_count_;
    std::size_t participant_count_;
    std::vector<TimestreamNetworkId> participant_network_ids_;
    std::vector<double> values_;
    std::vector<std::uint8_t> exclusion_flags_;
    std::vector<CoincidenceCellState> provenance_states_;
    std::vector<std::optional<NativeSampleIdentity>> mapped_identities_;
    std::vector<TimestreamNativeRevision> expected_revisions_;
    std::vector<std::optional<NativeInvalidityProvenance>>
        invalidity_provenance_;
    std::vector<std::optional<CoincidenceAbsenceReason>> absence_reasons_;
};

inline PcaRectangularWorkingSet make_pca_rectangular_working_set(
    const NativeSampleLedger<double> &ledger,
    const CoincidenceCohort &cohort,
    FinitePcaPlaceholder excluded_placeholder) {
    std::vector<double> values;
    std::vector<std::uint8_t> exclusion_flags;
    std::vector<CoincidenceCellState> provenance_states;
    std::vector<std::optional<NativeSampleIdentity>> mapped_identities;
    std::vector<TimestreamNativeRevision> expected_revisions;
    std::vector<std::optional<NativeInvalidityProvenance>>
        invalidity_provenance;
    std::vector<std::optional<CoincidenceAbsenceReason>> absence_reasons;
    const auto size = cohort.slot_count() * cohort.participant_count();
    values.reserve(size);
    exclusion_flags.reserve(size);
    provenance_states.reserve(size);
    mapped_identities.reserve(size);
    expected_revisions.reserve(size);
    invalidity_provenance.reserve(size);
    absence_reasons.reserve(size);

    for (std::size_t slot = 0; slot < cohort.slot_count(); ++slot) {
        for (std::size_t participant = 0;
             participant < cohort.participant_count(); ++participant) {
            const auto &cell = cohort.cell(slot, participant);
            provenance_states.push_back(cell.state());
            mapped_identities.push_back(cell.identity());
            expected_revisions.push_back(cell.expected_revision());
            invalidity_provenance.push_back(cell.invalidity());
            absence_reasons.push_back(cell.absence_reason());
            exclusion_flags.push_back(cell.pca_valid() ? 0U : 1U);
            if (!cell.is_mapped()) {
                values.push_back(excluded_placeholder.value());
                continue;
            }
            const auto &identity = *cell.identity();
            const auto &record = ledger.at(identity.key());
            if (!(record.identity == identity)) {
                throw std::logic_error(
                    "cohort native timestamp does not match ledger");
            }
            if (record.revision != cell.expected_revision()) {
                throw std::logic_error(
                    "cohort native revision is stale during gather");
            }
            // A mapped-invalid sample retains authoritative identity and
            // value in the native ledger, but its excluded rectangular cell
            // carries only the private finite placeholder.  This keeps the
            // PCA buffer free of nonfinite or otherwise unusable native data
            // without conflating mapped-invalid provenance with absence.
            if (!cell.pca_valid()) {
                values.push_back(excluded_placeholder.value());
                continue;
            }
            if (!std::isfinite(record.current_value)) {
                throw std::logic_error(
                    "PCA-valid native sample value must be finite");
            }
            values.push_back(record.current_value);
        }
    }

    return PcaRectangularWorkingSet{
        cohort.operation(), cohort.slot_count(),
        cohort.participant_network_ids(),
        std::move(values), std::move(exclusion_flags),
        std::move(provenance_states), std::move(mapped_identities),
        std::move(expected_revisions), std::move(invalidity_provenance),
        std::move(absence_reasons)};
}

inline void scatter_pca_results_transactionally(
    NativeSampleLedger<double> &ledger,
    const CoincidenceCohort &cohort,
    const PcaRectangularWorkingSet &result) {
    const auto expected_size =
        cohort.slot_count() * cohort.participant_count();
    if (!(result.operation() == cohort.operation()) ||
        result.slot_count() != cohort.slot_count() ||
        result.participant_count() != cohort.participant_count() ||
        result.participant_network_ids() !=
            cohort.participant_network_ids() ||
        result.values().size() != expected_size ||
        result.exclusion_flags().size() != expected_size ||
        result.provenance_states().size() != expected_size ||
        result.mapped_identities().size() != expected_size ||
        result.expected_revisions().size() != expected_size ||
        result.invalidity_provenance().size() != expected_size ||
        result.absence_reasons().size() != expected_size) {
        throw std::logic_error(
            "PCA result identity or rectangular shape changed");
    }

    std::vector<NativeSampleLedger<double>::Update> updates;
    updates.reserve(cohort.slot_count() * cohort.participant_count());
    for (std::size_t slot = 0; slot < cohort.slot_count(); ++slot) {
        for (std::size_t participant = 0;
             participant < cohort.participant_count(); ++participant) {
            const auto flat =
                slot * cohort.participant_count() + participant;
            const auto &cell = cohort.cell(slot, participant);
            const auto expected_exclusion = cell.pca_valid() ? 0U : 1U;
            if (result.provenance_states().at(flat) != cell.state() ||
                result.exclusion_flags().at(flat) != expected_exclusion ||
                !(result.mapped_identities().at(flat) ==
                  cell.identity()) ||
                result.expected_revisions().at(flat) !=
                    cell.expected_revision() ||
                !(result.invalidity_provenance().at(flat) ==
                  cell.invalidity()) ||
                result.absence_reasons().at(flat) !=
                    cell.absence_reason()) {
                throw std::logic_error(
                    "PCA result native mapping or exclusion state changed");
            }
            if (!cell.is_mapped()) {
                continue;
            }
            if (cell.state() == CoincidenceCellState::mapped_invalid) {
                const auto &invalidity = *cell.invalidity();
                updates.push_back(
                    NativeSampleLedger<double>::Update::preserve_invalid(
                        *cell.identity(), cell.expected_revision(),
                        NativeCoincidenceProvenance{
                            slot, participant,
                            cohort.participant_network_ids().at(participant),
                            invalidity.original_flag_bits,
                            invalidity.reason}));
                continue;
            }
            const double value = result.values().at(flat);
            if (!std::isfinite(value)) {
                throw std::logic_error(
                    "valid PCA result must be finite before native scatter");
            }
            updates.push_back(
                NativeSampleLedger<double>::Update::replacement(
                    *cell.identity(), cell.expected_revision(), value,
                    NativeCoincidenceProvenance{
                        slot, participant,
                        cohort.participant_network_ids().at(participant),
                        std::nullopt, {}}));
        }
    }
    ledger.apply_transaction(cohort.operation(), updates);
}

enum class PcaCompatibilityHazard : std::uint8_t {
    null_model = 1U << 0U,
    adaptive_selector = 1U << 1U,
    band_limited_marchenko_pastur = 1U << 2U,
};

struct PcaCompatibilityInputs {
    bool null_model_active_for_operation = false;
    bool adaptive_selector_active_for_operation = false;
    bool marchenko_pastur_active_for_operation = false;
    bool marchenko_pastur_band_requested = false;
};

class PcaCompatibilityClassification {
public:
    bool compatible() const noexcept { return hazards_ == 0; }

    bool has(PcaCompatibilityHazard hazard) const noexcept {
        return (hazards_ & static_cast<std::uint8_t>(hazard)) != 0;
    }

private:
    friend PcaCompatibilityClassification classify_pca_compatibility(
        const PcaRectangularWorkingSet &,
        const PcaCompatibilityInputs &);
    explicit PcaCompatibilityClassification(std::uint8_t hazards)
        : hazards_{hazards} {}
    std::uint8_t hazards_ = 0;
};

inline PcaCompatibilityClassification classify_pca_compatibility(
    const PcaRectangularWorkingSet &working_set,
    const PcaCompatibilityInputs &inputs) {
    const bool has_excluded_cells = std::any_of(
        working_set.exclusion_flags().begin(),
        working_set.exclusion_flags().end(),
        [](std::uint8_t excluded) { return excluded != 0; });
    std::uint8_t hazards = 0;
    if (has_excluded_cells &&
        inputs.null_model_active_for_operation) {
        hazards |=
            static_cast<std::uint8_t>(PcaCompatibilityHazard::null_model);
    }
    if (has_excluded_cells &&
        inputs.adaptive_selector_active_for_operation) {
        hazards |= static_cast<std::uint8_t>(
            PcaCompatibilityHazard::adaptive_selector);
    }
    if (has_excluded_cells &&
        inputs.marchenko_pastur_active_for_operation &&
        inputs.marchenko_pastur_band_requested) {
        hazards |= static_cast<std::uint8_t>(
            PcaCompatibilityHazard::band_limited_marchenko_pastur);
    }
    return PcaCompatibilityClassification{hazards};
}

inline void require_pca_compatibility(
    const PcaCompatibilityClassification &classification) {
    if (!classification.compatible()) {
        throw std::logic_error(
            "excluded cohort cells are incompatible with the requested "
            "optional PCA mode; no fallback is selected");
    }
}

}  // namespace citlali::pipeline
