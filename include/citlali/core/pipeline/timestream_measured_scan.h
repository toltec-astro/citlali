#pragma once

#include <citlali/core/pipeline/canonical_apt_detector_relation_v2.h>
#include <citlali/core/pipeline/native_observation_carriers.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline {

using TimestreamDetectorColumn = Eigen::Index;
using NativeDetectorFlagBits = std::uint64_t;
using NativeDetectorFlagBitsMatrix =
    Eigen::Matrix<NativeDetectorFlagBits, Eigen::Dynamic, Eigen::Dynamic>;

struct NativeScanChunkScope {
    NativeObservationScope observation_scope;
    std::int64_t scan_index = -1;
    std::int64_t chunk_index = -1;

    NativeScanChunkScope(NativeObservationScope observation_scope_,
                         std::int64_t scan_index_,
                         std::int64_t chunk_index_)
        : observation_scope{observation_scope_}, scan_index{scan_index_},
          chunk_index{chunk_index_} {
        if (scan_index < 0 || chunk_index < 0) {
            throw std::invalid_argument(
                "native measured scan/chunk indices must be nonnegative");
        }
    }

    friend bool operator==(const NativeScanChunkScope &,
                           const NativeScanChunkScope &) = default;
};

struct NativeDetectorSampleKey {
    NativeSampleKey native_sample;
    TimestreamDetectorColumn detector_column = -1;

    friend bool operator==(const NativeDetectorSampleKey &,
                           const NativeDetectorSampleKey &) = default;
    friend bool operator<(const NativeDetectorSampleKey &lhs,
                          const NativeDetectorSampleKey &rhs) noexcept {
        if (lhs.native_sample < rhs.native_sample) return true;
        if (rhs.native_sample < lhs.native_sample) return false;
        return lhs.detector_column < rhs.detector_column;
    }
};

// One delivered raw KIDs matrix for this scan/chunk. The immutable mapping
// retains shared handles to the existing value and flag owners; it does not
// copy either O(rows x channels) matrix.
class NativeMeasuredNetworkInput {
public:
    NativeMeasuredNetworkInput(
        std::int64_t raw_source_uid,
        TimestreamNetworkId network_id,
        std::string interface_name,
        TimestreamNativeRow first_native_row,
        std::shared_ptr<const Eigen::MatrixXd> measured_values,
        std::shared_ptr<const NativeDetectorFlagBitsMatrix>
            original_flag_bits)
        : raw_source_uid_{raw_source_uid}, network_id_{network_id},
          interface_name_{std::move(interface_name)},
          first_native_row_{first_native_row},
          measured_values_{std::move(measured_values)},
          original_flag_bits_{std::move(original_flag_bits)} {
        if (raw_source_uid_ < 0 || network_id_ < 0 ||
            interface_name_.empty() || first_native_row_ < 0 ||
            !measured_values_ || !original_flag_bits_) {
            throw std::invalid_argument(
                "native measured network input identity is incomplete");
        }
        if (measured_values_->rows() <= 0 ||
            measured_values_->cols() <= 0 ||
            original_flag_bits_->rows() != measured_values_->rows() ||
            original_flag_bits_->cols() != measured_values_->cols()) {
            throw std::invalid_argument(
                "native measured values and original flags require equal nonempty shape");
        }
        if (measured_values_->rows() >
            std::numeric_limits<TimestreamNativeRow>::max() -
                first_native_row_) {
            throw std::length_error(
                "native measured network row interval would overflow");
        }
    }

    std::int64_t raw_source_uid() const noexcept { return raw_source_uid_; }
    TimestreamNetworkId network_id() const noexcept { return network_id_; }
    const std::string &interface_name() const noexcept {
        return interface_name_;
    }
    TimestreamNativeRow first_native_row() const noexcept {
        return first_native_row_;
    }
    TimestreamNativeRow past_last_native_row() const noexcept {
        return first_native_row_ +
            static_cast<TimestreamNativeRow>(measured_values_->rows());
    }
    Eigen::Index row_count() const noexcept {
        return measured_values_->rows();
    }
    Eigen::Index channel_count() const noexcept {
        return measured_values_->cols();
    }
    const std::shared_ptr<const Eigen::MatrixXd> &measured_values_handle()
        const noexcept {
        return measured_values_;
    }
    const std::shared_ptr<const NativeDetectorFlagBitsMatrix> &
    original_flag_bits_handle() const noexcept {
        return original_flag_bits_;
    }

    Eigen::Index local_row(TimestreamNativeRow native_row) const {
        if (native_row < first_native_row_ ||
            native_row >= past_last_native_row()) {
            throw std::out_of_range(
                "native row is outside its measured network input");
        }
        return static_cast<Eigen::Index>(native_row - first_native_row_);
    }

private:
    std::int64_t raw_source_uid_;
    TimestreamNetworkId network_id_;
    std::string interface_name_;
    TimestreamNativeRow first_native_row_;
    std::shared_ptr<const Eigen::MatrixXd> measured_values_;
    std::shared_ptr<const NativeDetectorFlagBitsMatrix>
        original_flag_bits_;
};

struct NativeMeasuredDetectorBinding {
    TimestreamDetectorColumn detector_column = -1;
    std::int64_t output_uid = 0;
    std::int64_t array = 0;
    TimestreamNetworkId network_id = -1;
    std::int64_t raw_source_uid = 0;
    Eigen::Index raw_channel = -1;
    canonical_apt_v2::RelationDisposition disposition =
        canonical_apt_v2::RelationDisposition::unmatched;
    std::optional<std::int64_t> apt_flag;

    friend bool operator==(const NativeMeasuredDetectorBinding &,
                           const NativeMeasuredDetectorBinding &) = default;
};

class NativeMeasuredDetectorCell {
public:
    static NativeMeasuredDetectorCell absent(
        std::size_t common_slot,
        TimestreamDetectorColumn detector_column,
        std::int64_t output_uid,
        CoincidenceAbsenceReason reason) {
        return NativeMeasuredDetectorCell{
            CoincidenceCellState::absent, common_slot, detector_column,
            output_uid, std::nullopt, std::nullopt, 0, {}, reason};
    }

    static NativeMeasuredDetectorCell measured(
        std::size_t common_slot,
        TimestreamDetectorColumn detector_column,
        std::int64_t output_uid,
        NativeSampleIdentity identity,
        double measured_value,
        NativeDetectorFlagBits original_flag_bits) {
        const bool finite = std::isfinite(measured_value);
        const bool valid = finite && original_flag_bits == 0;
        std::string reason;
        if (original_flag_bits != 0) {
            reason = "delivered detector flag bits";
        }
        if (!finite) {
            if (!reason.empty()) reason += "; ";
            reason += "nonfinite delivered detector value";
        }
        return NativeMeasuredDetectorCell{
            valid ? CoincidenceCellState::mapped_valid
                  : CoincidenceCellState::mapped_invalid,
            common_slot, detector_column, output_uid,
            std::move(identity), measured_value, original_flag_bits,
            std::move(reason), std::nullopt};
    }

    CoincidenceCellState state() const noexcept { return state_; }
    bool mapped() const noexcept {
        return state_ != CoincidenceCellState::absent;
    }
    bool valid() const noexcept {
        return state_ == CoincidenceCellState::mapped_valid;
    }
    std::size_t common_slot() const noexcept { return common_slot_; }
    TimestreamDetectorColumn detector_column() const noexcept {
        return detector_column_;
    }
    std::int64_t output_uid() const noexcept { return output_uid_; }
    const std::optional<NativeSampleIdentity> &identity() const noexcept {
        return identity_;
    }
    const std::optional<double> &measured_value() const noexcept {
        return measured_value_;
    }
    NativeDetectorFlagBits original_flag_bits() const noexcept {
        return original_flag_bits_;
    }
    const std::string &invalidity_reason() const noexcept {
        return invalidity_reason_;
    }
    const std::optional<CoincidenceAbsenceReason> &absence_reason() const
        noexcept {
        return absence_reason_;
    }

private:
    NativeMeasuredDetectorCell(
        CoincidenceCellState state,
        std::size_t common_slot,
        TimestreamDetectorColumn detector_column,
        std::int64_t output_uid,
        std::optional<NativeSampleIdentity> identity,
        std::optional<double> measured_value,
        NativeDetectorFlagBits original_flag_bits,
        std::string invalidity_reason,
        std::optional<CoincidenceAbsenceReason> absence_reason)
        : state_{state}, common_slot_{common_slot},
          detector_column_{detector_column}, output_uid_{output_uid},
          identity_{std::move(identity)}, measured_value_{measured_value},
          original_flag_bits_{original_flag_bits},
          invalidity_reason_{std::move(invalidity_reason)},
          absence_reason_{absence_reason} {}

    CoincidenceCellState state_;
    std::size_t common_slot_;
    TimestreamDetectorColumn detector_column_;
    std::int64_t output_uid_;
    std::optional<NativeSampleIdentity> identity_;
    std::optional<double> measured_value_;
    NativeDetectorFlagBits original_flag_bits_;
    std::string invalidity_reason_;
    std::optional<CoincidenceAbsenceReason> absence_reason_;
};

// Immutable join of one verified compact-v2 detector relation, one exact
// observation ALIGN/pointing carrier pair, and one scan/chunk's delivered raw
// matrices. Common slots remain relational coordinates only.
class NativeMeasuredDetectorScan {
public:
    static std::shared_ptr<const NativeMeasuredDetectorScan> admit(
        NativeScanChunkScope scope,
        std::shared_ptr<const NativeObservationCarriers> carriers,
        std::shared_ptr<const CanonicalAptDetectorRelationV2> relation,
        std::size_t first_common_slot,
        std::size_t past_last_common_slot,
        std::vector<NativeMeasuredNetworkInput> network_inputs) {
        if (!carriers || !relation || network_inputs.empty() ||
            first_common_slot >= past_last_common_slot) {
            throw std::invalid_argument(
                "native measured scan admission is incomplete");
        }
        if (!(carriers->scope() == scope.observation_scope)) {
            throw std::invalid_argument(
                "native measured scan carriers are stale or foreign");
        }
        const auto &observation = relation->observation();
        if (observation.observation !=
                scope.observation_scope.observation ||
            observation.subobservation !=
                scope.observation_scope.subobservation ||
            observation.scan != scope.observation_scope.scan) {
            throw std::invalid_argument(
                "native measured scan relation is stale or foreign");
        }
        const auto &alignment =
            *carriers->alignment_handle();
        if (past_last_common_slot > alignment.slot_count()) {
            throw std::out_of_range(
                "native measured scan common-slot interval is out of range");
        }

        std::sort(network_inputs.begin(), network_inputs.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.network_id() < rhs.network_id();
                  });
        if (network_inputs.size() != relation->raw_sources().size() ||
            network_inputs.size() != alignment.networks().size()) {
            throw std::invalid_argument(
                "native measured scan network inventories differ");
        }

        std::map<TimestreamNetworkId, std::size_t> input_index_by_network;
        std::map<TimestreamNetworkId, std::size_t> raw_index_by_network;
        std::set<std::int64_t> source_uids;
        for (std::size_t index = 0; index < relation->raw_sources().size();
             ++index) {
            const auto &raw = relation->raw_sources()[index];
            if (raw.network < 0 ||
                raw.network >
                    std::numeric_limits<TimestreamNetworkId>::max() ||
                !raw_index_by_network
                     .emplace(static_cast<TimestreamNetworkId>(raw.network),
                              index)
                     .second ||
                !source_uids.insert(raw.source_uid).second) {
                throw std::invalid_argument(
                    "native measured scan raw-source inventory is invalid");
            }
        }

        std::size_t sample_count = 0;
        std::map<TimestreamNetworkId, std::size_t>
            sample_offset_by_network;
        for (std::size_t index = 0; index < network_inputs.size(); ++index) {
            const auto &input = network_inputs[index];
            if (!input_index_by_network
                     .emplace(input.network_id(), index)
                     .second) {
                throw std::invalid_argument(
                    "native measured scan repeats a network input");
            }
            const auto raw_it = raw_index_by_network.find(input.network_id());
            if (raw_it == raw_index_by_network.end()) {
                throw std::invalid_argument(
                    "native measured scan network is absent from the verified relation");
            }
            const auto &raw = relation->raw_sources().at(raw_it->second);
            if (input.raw_source_uid() != raw.source_uid ||
                input.interface_name() != raw.interface_name ||
                input.channel_count() != raw.channel_count ||
                !(raw.header_observation == observation)) {
                throw std::invalid_argument(
                    "native measured matrix disagrees with its verified raw source");
            }
            const auto &network = alignment.network(input.network_id());
            if (input.first_native_row() < network.first_native_row() ||
                input.past_last_native_row() >
                    network.past_last_native_row()) {
                throw std::invalid_argument(
                    "native measured matrix row interval is outside ALIGN authority");
            }
            for (std::size_t slot = first_common_slot;
                 slot < past_last_common_slot; ++slot) {
                const auto &association =
                    alignment.association(input.network_id(), slot);
                if (association.mapped() &&
                    (association.native_row < input.first_native_row() ||
                     association.native_row >=
                         input.past_last_native_row())) {
                    throw std::invalid_argument(
                        "native measured matrix omits an admitted relational cell");
                }
            }
            const auto rows = static_cast<std::size_t>(input.row_count());
            const auto channels =
                static_cast<std::size_t>(input.channel_count());
            if (channels != 0 &&
                rows > (std::numeric_limits<std::size_t>::max() -
                        sample_count) /
                    channels) {
                throw std::length_error(
                    "native measured detector cardinality would overflow");
            }
            sample_offset_by_network.emplace(input.network_id(),
                                             sample_count);
            sample_count += rows * channels;
        }
        if (alignment.participant_network_ids().size() !=
            input_index_by_network.size()) {
            throw std::invalid_argument(
                "native measured ALIGN participant inventory differs");
        }
        for (const auto network_id : alignment.participant_network_ids()) {
            if (!input_index_by_network.contains(network_id)) {
                throw std::invalid_argument(
                    "native measured ALIGN participant lacks raw data");
            }
        }

        std::vector<NativeMeasuredDetectorBinding> bindings;
        bindings.reserve(relation->bindings().size());
        std::set<std::pair<TimestreamNetworkId, Eigen::Index>> raw_channels;
        for (std::size_t index = 0; index < relation->bindings().size();
             ++index) {
            const auto &binding = relation->bindings()[index];
            if (index > static_cast<std::size_t>(
                            std::numeric_limits<
                                TimestreamDetectorColumn>::max()) ||
                binding.detector_column != index || binding.network < 0 ||
                binding.network >
                    std::numeric_limits<TimestreamNetworkId>::max() ||
                binding.channel < 0 ||
                binding.channel >
                    std::numeric_limits<Eigen::Index>::max()) {
                throw std::invalid_argument(
                    "native measured detector binding identity is invalid");
            }
            const auto network_id =
                static_cast<TimestreamNetworkId>(binding.network);
            const auto input_it = input_index_by_network.find(network_id);
            if (input_it == input_index_by_network.end()) {
                throw std::invalid_argument(
                    "native measured detector network lacks a matrix");
            }
            const auto &input = network_inputs.at(input_it->second);
            const auto channel = static_cast<Eigen::Index>(binding.channel);
            if (binding.raw_source_uid != input.raw_source_uid() ||
                channel >= input.channel_count() ||
                !raw_channels.emplace(network_id, channel).second) {
                throw std::invalid_argument(
                    "native measured raw channel-to-detector join is invalid");
            }
            bindings.push_back(NativeMeasuredDetectorBinding{
                static_cast<TimestreamDetectorColumn>(index),
                binding.output_uid, binding.array, network_id,
                binding.raw_source_uid,
                channel, binding.disposition, binding.flag});
        }
        if (bindings.empty() || raw_channels.size() != bindings.size()) {
            throw std::invalid_argument(
                "native measured detector binding inventory is empty or incomplete");
        }
        for (const auto &input : network_inputs) {
            for (Eigen::Index channel = 0; channel < input.channel_count();
                 ++channel) {
                if (!raw_channels.contains({input.network_id(), channel})) {
                    throw std::invalid_argument(
                        "native measured matrix channel lacks one detector column");
                }
            }
        }

        return std::shared_ptr<const NativeMeasuredDetectorScan>(
            new NativeMeasuredDetectorScan{
                std::move(scope), std::move(carriers),
                std::move(relation), first_common_slot,
                past_last_common_slot, std::move(network_inputs),
                std::move(input_index_by_network), std::move(bindings),
                std::move(sample_offset_by_network), sample_count});
    }

    const NativeScanChunkScope &scope() const noexcept { return scope_; }
    const std::shared_ptr<const NativeObservationCarriers> &carriers_handle()
        const noexcept {
        return carriers_;
    }
    const std::shared_ptr<const CanonicalAptDetectorRelationV2> &
    relation_handle() const noexcept {
        return relation_;
    }
    std::size_t first_common_slot() const noexcept {
        return first_common_slot_;
    }
    std::size_t past_last_common_slot() const noexcept {
        return past_last_common_slot_;
    }
    std::size_t relational_slot_count() const noexcept {
        return past_last_common_slot_ - first_common_slot_;
    }
    std::size_t detector_count() const noexcept { return bindings_.size(); }
    std::size_t measured_sample_count() const noexcept {
        return measured_sample_count_;
    }
    const std::vector<NativeMeasuredDetectorBinding> &bindings() const
        noexcept {
        return bindings_;
    }
    const NativeMeasuredDetectorBinding &binding(
        TimestreamDetectorColumn detector_column) const {
        return bindings_.at(checked_detector_index(detector_column));
    }
    const NativeMeasuredNetworkInput &network_input(
        TimestreamNetworkId network_id) const {
        const auto found = input_index_by_network_.find(network_id);
        if (found == input_index_by_network_.end()) {
            throw std::out_of_range(
                "network is absent from the native measured scan");
        }
        return network_inputs_.at(found->second);
    }

    NativeMeasuredDetectorCell cell(
        std::size_t common_slot,
        TimestreamDetectorColumn detector_column) const {
        require_common_slot(common_slot);
        const auto &detector = binding(detector_column);
        const auto &association = carriers_->alignment_handle()->association(
            detector.network_id, common_slot);
        if (!association.mapped()) {
            return NativeMeasuredDetectorCell::absent(
                common_slot, detector_column, detector.output_uid,
                association.absence_reason);
        }
        const auto &input = network_input(detector.network_id);
        const auto local_row = input.local_row(association.native_row);
        const double value =
            (*input.measured_values_handle())(local_row,
                                              detector.raw_channel);
        const auto flags =
            (*input.original_flag_bits_handle())(local_row,
                                                  detector.raw_channel);
        return NativeMeasuredDetectorCell::measured(
            common_slot, detector_column, detector.output_uid,
            carriers_->alignment_handle()
                ->network(detector.network_id)
                .identity(association.native_row),
            value, flags);
    }

    double measured_value(const NativeDetectorSampleKey &key) const {
        const auto [input, local_row, channel] = locate(key);
        return (*input->measured_values_handle())(local_row, channel);
    }
    NativeDetectorFlagBits original_flag_bits(
        const NativeDetectorSampleKey &key) const {
        const auto [input, local_row, channel] = locate(key);
        return (*input->original_flag_bits_handle())(local_row, channel);
    }
    NativeSampleIdentity sample_identity(
        const NativeDetectorSampleKey &key) const {
        const auto &detector = binding(key.detector_column);
        if (key.native_sample.network_id != detector.network_id) {
            throw std::invalid_argument(
                "native detector sample key changes its network binding");
        }
        return carriers_->alignment_handle()
            ->network(detector.network_id)
            .identity(key.native_sample.native_row);
    }
    std::size_t dense_sample_index(
        const NativeDetectorSampleKey &key) const {
        const auto [input, local_row, channel] = locate(key);
        const auto base = sample_offset_by_network_.at(input->network_id());
        return base + static_cast<std::size_t>(local_row) *
            static_cast<std::size_t>(input->channel_count()) +
            static_cast<std::size_t>(channel);
    }

private:
    using LocatedSample =
        std::tuple<const NativeMeasuredNetworkInput *, Eigen::Index,
                   Eigen::Index>;

    NativeMeasuredDetectorScan(
        NativeScanChunkScope scope,
        std::shared_ptr<const NativeObservationCarriers> carriers,
        std::shared_ptr<const CanonicalAptDetectorRelationV2> relation,
        std::size_t first_common_slot,
        std::size_t past_last_common_slot,
        std::vector<NativeMeasuredNetworkInput> network_inputs,
        std::map<TimestreamNetworkId, std::size_t> input_index_by_network,
        std::vector<NativeMeasuredDetectorBinding> bindings,
        std::map<TimestreamNetworkId, std::size_t>
            sample_offset_by_network,
        std::size_t measured_sample_count)
        : scope_{std::move(scope)}, carriers_{std::move(carriers)},
          relation_{std::move(relation)},
          first_common_slot_{first_common_slot},
          past_last_common_slot_{past_last_common_slot},
          network_inputs_{std::move(network_inputs)},
          input_index_by_network_{std::move(input_index_by_network)},
          bindings_{std::move(bindings)},
          sample_offset_by_network_{std::move(sample_offset_by_network)},
          measured_sample_count_{measured_sample_count} {}

    std::size_t checked_detector_index(
        TimestreamDetectorColumn detector_column) const {
        if (detector_column < 0 ||
            static_cast<std::size_t>(detector_column) >= bindings_.size()) {
            throw std::out_of_range(
                "detector column is outside the native measured scan");
        }
        return static_cast<std::size_t>(detector_column);
    }
    void require_common_slot(std::size_t common_slot) const {
        if (common_slot < first_common_slot_ ||
            common_slot >= past_last_common_slot_) {
            throw std::out_of_range(
                "common slot is outside the native measured scan");
        }
    }
    LocatedSample locate(const NativeDetectorSampleKey &key) const {
        const auto &detector = binding(key.detector_column);
        if (key.native_sample.network_id != detector.network_id) {
            throw std::invalid_argument(
                "native detector sample key changes its network binding");
        }
        const auto &input = network_input(detector.network_id);
        const auto local_row = input.local_row(key.native_sample.native_row);
        return {&input, local_row, detector.raw_channel};
    }

    NativeScanChunkScope scope_;
    std::shared_ptr<const NativeObservationCarriers> carriers_;
    std::shared_ptr<const CanonicalAptDetectorRelationV2> relation_;
    std::size_t first_common_slot_;
    std::size_t past_last_common_slot_;
    std::vector<NativeMeasuredNetworkInput> network_inputs_;
    std::map<TimestreamNetworkId, std::size_t> input_index_by_network_;
    std::vector<NativeMeasuredDetectorBinding> bindings_;
    std::map<TimestreamNetworkId, std::size_t>
        sample_offset_by_network_;
    std::size_t measured_sample_count_;
};

class NativeMeasuredDetectorLedger {
public:
    enum class RevisionAction {
        replaced_by_pca_result,
        preserved_pca_invalid,
        preserved_pass_through,
    };

    class Update {
    public:
        static Update replacement(
            NativeSampleIdentity identity,
            TimestreamDetectorColumn detector_column,
            TimestreamNativeRevision expected_revision,
            double value) {
            return Update{std::move(identity), detector_column,
                          expected_revision,
                          RevisionAction::replaced_by_pca_result, value};
        }

        static Update preserve_invalid(
            NativeSampleIdentity identity,
            TimestreamDetectorColumn detector_column,
            TimestreamNativeRevision expected_revision,
            double preserved_value) {
            return Update{std::move(identity), detector_column,
                          expected_revision,
                          RevisionAction::preserved_pca_invalid,
                          preserved_value};
        }

        static Update preserve_pass_through(
            NativeSampleIdentity identity,
            TimestreamDetectorColumn detector_column,
            TimestreamNativeRevision expected_revision,
            double preserved_value) {
            return Update{std::move(identity), detector_column,
                          expected_revision,
                          RevisionAction::preserved_pass_through,
                          preserved_value};
        }

        const NativeSampleIdentity &identity() const noexcept {
            return identity_;
        }
        TimestreamDetectorColumn detector_column() const noexcept {
            return detector_column_;
        }
        TimestreamNativeRevision expected_revision() const noexcept {
            return expected_revision_;
        }
        RevisionAction action() const noexcept { return action_; }
        const std::optional<double> &replacement_value() const noexcept {
            return replacement_value_;
        }

    private:
        Update(NativeSampleIdentity identity,
               TimestreamDetectorColumn detector_column,
               TimestreamNativeRevision expected_revision,
               RevisionAction action,
               std::optional<double> replacement_value)
            : identity_{std::move(identity)},
              detector_column_{detector_column},
              expected_revision_{expected_revision}, action_{action},
              replacement_value_{replacement_value} {}

        NativeSampleIdentity identity_;
        TimestreamDetectorColumn detector_column_;
        TimestreamNativeRevision expected_revision_;
        RevisionAction action_;
        std::optional<double> replacement_value_;
    };

    struct RecordView {
        NativeSampleIdentity identity;
        TimestreamDetectorColumn detector_column = -1;
        double measured_value = 0.0;
        double current_value = 0.0;
        NativeDetectorFlagBits original_flag_bits = 0;
        TimestreamNativeRevision revision = 0;
    };

    explicit NativeMeasuredDetectorLedger(
        std::shared_ptr<const NativeMeasuredDetectorScan> mapping)
        : mapping_{std::move(mapping)} {
        if (!mapping_) {
            throw std::invalid_argument(
                "native measured detector ledger requires its scan mapping");
        }
        revisions_.assign(mapping_->measured_sample_count(), 0);
    }

    const std::shared_ptr<const NativeMeasuredDetectorScan> &mapping_handle()
        const noexcept {
        return mapping_;
    }
    std::size_t size() const noexcept { return revisions_.size(); }
    RecordView record(const NativeDetectorSampleKey &key) const {
        const auto index = mapping_->dense_sample_index(key);
        const double measured = mapping_->measured_value(key);
        const auto current = current_values_.find(index);
        return RecordView{
            mapping_->sample_identity(key), key.detector_column, measured,
            current == current_values_.end() ? measured : current->second,
            mapping_->original_flag_bits(key),
            revisions_.at(index)};
    }
    const std::optional<NativeOperationIdentity> &last_operation() const
        noexcept {
        return last_operation_;
    }
    const std::optional<NativeOperationIdentity> &last_committed_operation()
        const noexcept {
        return last_committed_operation_;
    }
    NativeOperationIdentity next_operation() const {
        if (operation_sequence_exhausted_) {
            throw std::overflow_error(
                "native measured operation sequence is exhausted");
        }
        return NativeOperationIdentity{
            next_operation_sequence_, mapping_->scope().scan_index};
    }
    NativeOperationIdentity issue_operation() {
        const auto operation = next_operation();
        last_operation_ = operation;
        if (next_operation_sequence_ ==
            std::numeric_limits<std::uint64_t>::max()) {
            operation_sequence_exhausted_ = true;
        }
        else {
            ++next_operation_sequence_;
        }
        return operation;
    }

    // Validate and stage the complete affected detector set before swapping
    // either current values or revisions. A rejected batch can therefore be
    // corrected and retried with the same issued operation identity.
    void apply_transaction(
        const NativeOperationIdentity &operation,
        const std::vector<Update> &updates) {
        if (!last_operation_.has_value() ||
            !(*last_operation_ == operation)) {
            throw std::logic_error(
                "native measured scatter operation was not issued by this ledger");
        }
        if (last_committed_operation_.has_value() &&
            operation.sequence <= last_committed_operation_->sequence) {
            throw std::logic_error(
                "native measured scatter operation is stale or already committed");
        }

        std::set<NativeDetectorSampleKey> destinations;
        struct LocatedUpdate {
            std::size_t index = 0;
            const Update *update = nullptr;
        };
        std::vector<LocatedUpdate> located;
        located.reserve(updates.size());
        for (const auto &update : updates) {
            const NativeDetectorSampleKey key{
                update.identity().key(), update.detector_column()};
            if (!destinations.insert(key).second) {
                throw std::logic_error(
                    "native measured scatter repeats a detector destination");
            }
            const auto record_before = record(key);
            if (!(record_before.identity == update.identity())) {
                throw std::logic_error(
                    "native measured scatter identity or timestamp changed");
            }
            if (record_before.revision != update.expected_revision()) {
                throw std::logic_error(
                    "native measured scatter expected revision is stale");
            }
            if (record_before.revision ==
                std::numeric_limits<TimestreamNativeRevision>::max()) {
                throw std::overflow_error(
                    "native measured detector revision would overflow");
            }
            if (!update.replacement_value().has_value()) {
                throw std::logic_error(
                    "native measured scatter update has no projected value");
            }
            if (!std::isfinite(*update.replacement_value())) {
                throw std::logic_error(
                    "native measured projected value must be finite");
            }
            located.push_back(
                {mapping_->dense_sample_index(key), &update});
        }

        auto candidate_values = current_values_;
        auto candidate_revisions = revisions_;
        for (const auto &candidate : located) {
            candidate_values[candidate.index] =
                *candidate.update->replacement_value();
            ++candidate_revisions.at(candidate.index);
        }
        current_values_.swap(candidate_values);
        revisions_.swap(candidate_revisions);
        last_committed_operation_ = operation;
    }

private:
    std::shared_ptr<const NativeMeasuredDetectorScan> mapping_;
    std::vector<TimestreamNativeRevision> revisions_;
    std::map<std::size_t, double> current_values_;
    std::uint64_t next_operation_sequence_ = 0;
    bool operation_sequence_exhausted_ = false;
    std::optional<NativeOperationIdentity> last_operation_;
    std::optional<NativeOperationIdentity> last_committed_operation_;
};

// One mutable scan/chunk owner. Admission builds the immutable mapping and
// fresh ledger completely before publication; commit, rollback, or boundary
// exit destroys both and their operation sequence.
class NativeMeasuredScanTransaction {
public:
    explicit NativeMeasuredScanTransaction(NativeScanChunkScope scope)
        : scope_{std::move(scope)} {}

    void admit(
        std::shared_ptr<const NativeObservationCarriers> carriers,
        std::shared_ptr<const CanonicalAptDetectorRelationV2> relation,
        std::size_t first_common_slot,
        std::size_t past_last_common_slot,
        std::vector<NativeMeasuredNetworkInput> network_inputs) {
        if (current_) {
            throw std::logic_error(
                "native measured scan transaction requires commit or rollback before readmission");
        }
        auto mapping = NativeMeasuredDetectorScan::admit(
            scope_, std::move(carriers), std::move(relation),
            first_common_slot, past_last_common_slot,
            std::move(network_inputs));
        auto candidate = std::make_unique<State>(
            State{mapping, NativeMeasuredDetectorLedger{mapping}});
        current_.swap(candidate);
    }

    bool active() const noexcept { return current_ != nullptr; }
    const NativeScanChunkScope &scope() const noexcept { return scope_; }
    const std::shared_ptr<const NativeMeasuredDetectorScan> &mapping_handle()
        const {
        return require_state().mapping;
    }
    const NativeMeasuredDetectorScan &mapping() const {
        return *mapping_handle();
    }
    NativeMeasuredDetectorLedger &ledger() {
        return require_state().ledger;
    }
    const NativeMeasuredDetectorLedger &ledger() const {
        return require_state().ledger;
    }
    void commit() noexcept { current_.reset(); }
    void rollback() noexcept { current_.reset(); }

private:
    struct State {
        std::shared_ptr<const NativeMeasuredDetectorScan> mapping;
        NativeMeasuredDetectorLedger ledger;
    };

    State &require_state() {
        if (!current_) {
            throw std::logic_error(
                "native measured scan transaction is not active");
        }
        return *current_;
    }
    const State &require_state() const {
        if (!current_) {
            throw std::logic_error(
                "native measured scan transaction is not active");
        }
        return *current_;
    }

    NativeScanChunkScope scope_;
    std::unique_ptr<State> current_;
};

}  // namespace citlali::pipeline
