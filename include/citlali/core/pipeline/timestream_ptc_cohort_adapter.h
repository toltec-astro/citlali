#pragma once

#include <citlali/core/pipeline/timestream_rtc_run_adapter.h>

#include <Eigen/Core>

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

using NativePtcExclusionMatrix =
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;

enum class NativePtcGroupRole {
    pca_clean,
    pass_through,
};

struct NativePtcRowBinding {
    std::size_t segment_ordinal = 0;
    Eigen::Index segment_output_row = -1;
    std::vector<std::size_t> exact_common_slots;
    std::vector<std::pair<TimestreamNetworkId, NativeSampleIdentity>>
        network_anchors;
};

struct NativePtcRtcCohortRow {
    std::size_t segment_ordinal = 0;
    Eigen::Index segment_output_row = -1;
    std::vector<std::size_t> exact_common_slots;
};

struct NativePtcRtcCohortSegment {
    std::size_t segment_ordinal = 0;
    std::vector<NativePtcRtcCohortRow> rows;
    std::vector<std::pair<TimestreamNetworkId,
                          const NativeRtcRunResult *>> network_runs;

    const NativeRtcRunResult &run(TimestreamNetworkId network_id) const {
        const auto found = std::find_if(
            network_runs.begin(), network_runs.end(),
            [network_id](const auto &candidate) {
                return candidate.first == network_id;
            });
        if (found == network_runs.end() || found->second == nullptr) {
            throw std::logic_error(
                "native PTC segment lacks its RTC network run");
        }
        return *found->second;
    }
};

namespace detail {

inline std::string normalize_native_ptc_grouping(std::string grouping) {
    std::transform(
        grouping.begin(), grouping.end(), grouping.begin(),
        [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    if (grouping == "network") grouping = "nw";
    return grouping;
}

inline bool native_ptc_grouping_is_supported(
    const std::string &grouping) {
    return grouping == "all" || grouping == "nw" ||
           grouping == "array" || grouping == "detector" ||
           grouping == "corr_nw";
}

inline bool native_ptc_matrix_bits_equal(
    const Eigen::MatrixXd &lhs, const Eigen::MatrixXd &rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) return false;
    for (Eigen::Index row = 0; row < lhs.rows(); ++row) {
        for (Eigen::Index column = 0; column < lhs.cols(); ++column) {
            if (std::bit_cast<std::uint64_t>(lhs(row, column)) !=
                std::bit_cast<std::uint64_t>(rhs(row, column))) {
                return false;
            }
        }
    }
    return true;
}

inline std::vector<NativePtcRtcCohortSegment>
make_native_ptc_rtc_cohort_segments(
    const NativeMeasuredDetectorScan &scan,
    const NativeRtcDispatchResult &rtc) {
    if (rtc.downsample_factor <= 0 || rtc.runs.empty()) {
        throw std::invalid_argument(
            "native PTC requires a nonempty admitted RTC dispatch");
    }
    const auto &participant_networks = scan.carriers_handle()
                                           ->alignment_handle()
                                           ->participant_network_ids();
    if (participant_networks.empty()) {
        throw std::logic_error(
            "native PTC RTC dispatch has no participant networks");
    }
    std::size_t selected_first =
        std::numeric_limits<std::size_t>::max();
    std::size_t selected_past = 0;
    for (const auto &run : rtc.runs) {
        selected_first = std::min(
            selected_first, run.input.selected_first_common_slot);
        selected_past = std::max(
            selected_past, run.input.selected_past_last_common_slot);
    }
    const auto expected_inputs = prepare_native_rtc_runs(
        scan, {rtc.downsample_factor, false,
               selected_first, selected_past});
    if (expected_inputs.size() != rtc.runs.size()) {
        throw std::logic_error(
            "native PTC RTC dispatch inventory differs from the admitted scan");
    }
    std::map<std::pair<std::size_t, TimestreamNetworkId>,
             const NativeRtcRunInput *> expected_by_run;
    for (const auto &expected : expected_inputs) {
        expected_by_run.emplace(
            std::make_pair(expected.segment_ordinal,
                           expected.run.network_id),
            &expected);
    }
    std::map<std::size_t,
             std::map<TimestreamNetworkId, const NativeRtcRunResult *>>
        runs_by_segment;
    for (const auto &run : rtc.runs) {
        const auto expected_it = expected_by_run.find(
            {run.input.segment_ordinal, run.input.run.network_id});
        if (expected_it == expected_by_run.end()) {
            throw std::logic_error(
                "native PTC RTC dispatch contains a foreign run");
        }
        const auto &expected = *expected_it->second;
        const auto &actual_run = run.input.run;
        const auto &expected_run = expected.run;
        if (run.input.run.network_id < 0 ||
            run.input.detector_columns.empty() ||
            run.selected_values.rows() !=
                static_cast<Eigen::Index>(run.support.size()) ||
            run.ored_flag_bits.rows() != run.selected_values.rows() ||
            run.ored_flag_bits.cols() != run.selected_values.cols() ||
            run.selected_values.cols() !=
                static_cast<Eigen::Index>(
                    run.input.detector_columns.size()) ||
            !run.selected_values.array().isFinite().all()) {
            throw std::logic_error(
                "native PTC RTC run identity or shape is invalid");
        }
        if (run.input.first_common_slot !=
                expected.first_common_slot ||
            run.input.past_last_common_slot !=
                expected.past_last_common_slot ||
            run.input.selected_first_common_slot !=
                expected.selected_first_common_slot ||
            run.input.selected_past_last_common_slot !=
                expected.selected_past_last_common_slot ||
            actual_run.network_id != expected_run.network_id ||
            actual_run.first_native_row !=
                expected_run.first_native_row ||
            actual_run.past_last_native_row !=
                expected_run.past_last_native_row ||
            !(actual_run.boundary_before ==
              expected_run.boundary_before) ||
            !(actual_run.boundary_after == expected_run.boundary_after) ||
            run.input.common_slots != expected.common_slots ||
            run.input.detector_columns != expected.detector_columns ||
            !native_ptc_matrix_bits_equal(
                run.input.measured_values,
                expected.measured_values) ||
            run.input.input_flag_bits.rows() !=
                expected.input_flag_bits.rows() ||
            run.input.input_flag_bits.cols() !=
                expected.input_flag_bits.cols() ||
            !(run.input.input_flag_bits.array() ==
              expected.input_flag_bits.array()).all()) {
            throw std::logic_error(
                "native PTC RTC run differs from the admitted scan");
        }
        if (!runs_by_segment[run.input.segment_ordinal]
                 .emplace(run.input.run.network_id, &run)
                 .second) {
            throw std::logic_error(
                "native PTC RTC segment repeats a network run");
        }
    }
    if (runs_by_segment.empty() ||
        runs_by_segment.begin()->first != 0) {
        throw std::logic_error(
            "native PTC RTC segment ordinals must begin at zero");
    }

    std::vector<NativePtcRtcCohortSegment> result;
    result.reserve(runs_by_segment.size());
    std::set<std::size_t> used_common_slots;
    std::size_t expected_segment = 0;
    for (const auto &[segment_ordinal, network_runs] : runs_by_segment) {
        if (segment_ordinal != expected_segment++ ||
            network_runs.size() != participant_networks.size()) {
            throw std::logic_error(
                "native PTC RTC segment inventory is incomplete");
        }
        for (const auto network_id : participant_networks) {
            if (!network_runs.contains(network_id)) {
                throw std::logic_error(
                    "native PTC RTC segment omits a participant network");
            }
        }

        const auto &reference = *network_runs.at(
            participant_networks.front());
        const auto output_rows = reference.selected_values.rows();
        const bool has_kernel =
            reference.selected_kernel_values.has_value();
        if (output_rows <= 0) {
            throw std::logic_error(
                "native PTC RTC segment has no output rows");
        }
        std::vector<NativePtcRtcCohortRow> rows;
        rows.reserve(static_cast<std::size_t>(output_rows));
        for (Eigen::Index row = 0; row < output_rows; ++row) {
            const auto &reference_support = reference.support.at(
                static_cast<std::size_t>(row));
            if (reference_support.exact_common_slots.empty()) {
                throw std::logic_error(
                    "native PTC RTC output has empty exact support");
            }
            for (const auto slot : reference_support.exact_common_slots) {
                if (!used_common_slots.insert(slot).second) {
                    throw std::logic_error(
                        "native PTC RTC cohorts bridge or reuse a common slot");
                }
            }
            rows.push_back(NativePtcRtcCohortRow{
                segment_ordinal, row,
                reference_support.exact_common_slots});
        }

        std::vector<bool> detector_seen(scan.detector_count(), false);
        for (const auto network_id : participant_networks) {
            const auto &run = *network_runs.at(network_id);
            if (run.input.segment_ordinal != segment_ordinal ||
                run.input.run.network_id != network_id ||
                run.input.first_common_slot !=
                    reference.input.first_common_slot ||
                run.input.past_last_common_slot !=
                    reference.input.past_last_common_slot ||
                run.input.selected_first_common_slot !=
                    reference.input.selected_first_common_slot ||
                run.input.selected_past_last_common_slot !=
                    reference.input.selected_past_last_common_slot ||
                run.selected_values.rows() != output_rows ||
                run.support.size() != reference.support.size() ||
                run.selected_kernel_values.has_value() != has_kernel ||
                (has_kernel &&
                 (run.selected_kernel_values->rows() != output_rows ||
                  run.selected_kernel_values->cols() !=
                      run.selected_values.cols() ||
                  !run.selected_kernel_values->array().isFinite().all()))) {
                throw std::logic_error(
                    "native PTC RTC participant runs disagree on their cohort");
            }
            for (Eigen::Index row = 0; row < output_rows; ++row) {
                const auto &support = run.support.at(
                    static_cast<std::size_t>(row));
                const auto first = run.input.selected_row_offset() +
                    static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(rtc.downsample_factor);
                const auto past = std::min(
                    first + static_cast<std::size_t>(rtc.downsample_factor),
                    run.input.selected_row_offset() +
                        run.input.selected_row_count());
                std::vector<std::size_t> expected_common_slots(
                    run.input.common_slots.begin() +
                        static_cast<std::ptrdiff_t>(first),
                    run.input.common_slots.begin() +
                        static_cast<std::ptrdiff_t>(past));
                std::vector<NativeSampleIdentity> expected_native_support;
                expected_native_support.reserve(past - first);
                const auto &alignment = *scan.carriers_handle()
                                             ->alignment_handle();
                for (const auto slot : expected_common_slots) {
                    const auto native_row =
                        alignment.association(network_id, slot).native_row;
                    expected_native_support.push_back(
                        alignment.network(network_id).identity(native_row));
                }
                if (support.segment_ordinal != segment_ordinal ||
                    support.run_output_row != row ||
                    support.factor != rtc.downsample_factor ||
                    support.detector_columns !=
                        run.input.detector_columns ||
                    support.ored_flag_support.size() !=
                        run.input.detector_columns.size() ||
                    support.final_short_support !=
                        (past - first < static_cast<std::size_t>(
                                            rtc.downsample_factor)) ||
                    support.exact_common_slots !=
                        expected_common_slots ||
                    support.exact_native_support !=
                        expected_native_support ||
                    support.selected_anchor.network_id() != network_id ||
                    support.selected_anchor !=
                        expected_native_support.front()) {
                    throw std::logic_error(
                        "native PTC RTC output support is unequal or foreign");
                }
                for (Eigen::Index local = 0;
                     local < run.ored_flag_bits.cols(); ++local) {
                    if (support.ored_flag_support.at(
                            static_cast<std::size_t>(local)) !=
                        run.ored_flag_bits(row, local)) {
                        throw std::logic_error(
                            "native PTC RTC flag support is inconsistent");
                    }
                }
            }
            for (std::size_t local = 0;
                 local < run.input.detector_columns.size(); ++local) {
                const auto detector_column =
                    run.input.detector_columns[local];
                if (detector_column < 0 ||
                    static_cast<std::size_t>(detector_column) >=
                        scan.detector_count() ||
                    detector_seen.at(
                        static_cast<std::size_t>(detector_column)) ||
                    scan.binding(detector_column).network_id != network_id) {
                    throw std::logic_error(
                        "native PTC RTC detector partition is invalid");
                }
                detector_seen[static_cast<std::size_t>(detector_column)] =
                    true;
            }
        }
        if (!std::all_of(detector_seen.begin(), detector_seen.end(),
                         [](bool value) { return value; })) {
            throw std::logic_error(
                "native PTC RTC detector partition is incomplete");
        }
        std::vector<std::pair<TimestreamNetworkId,
                              const NativeRtcRunResult *>> segment_runs;
        segment_runs.reserve(participant_networks.size());
        for (const auto network_id : participant_networks) {
            segment_runs.emplace_back(
                network_id, network_runs.at(network_id));
        }
        result.push_back(NativePtcRtcCohortSegment{
            segment_ordinal, std::move(rows), std::move(segment_runs)});
    }
    return result;
}

}  // namespace detail

class NativePtcGroupWorkingSet {
public:
    const NativeOperationIdentity &operation() const noexcept {
        return operation_;
    }
    std::size_t segment_ordinal() const noexcept {
        return segment_ordinal_;
    }
    const std::string &effective_grouping() const noexcept {
        return effective_grouping_;
    }
    std::int64_t group_key() const noexcept { return group_key_; }
    std::size_t subgroup_index() const noexcept {
        return subgroup_index_;
    }
    NativePtcGroupRole role() const noexcept { return role_; }
    const std::vector<TimestreamDetectorColumn> &detector_columns() const
        noexcept {
        return detector_columns_;
    }
    const std::vector<std::int64_t> &detector_uids() const noexcept {
        return detector_uids_;
    }
    const std::vector<std::optional<std::int64_t>> &detector_apt_flags()
        const noexcept {
        return detector_apt_flags_;
    }
    const Eigen::MatrixXd &values() const noexcept { return values_; }
    const std::optional<Eigen::MatrixXd> &kernel_values() const noexcept {
        return kernel_values_;
    }
    const NativePtcExclusionMatrix &exclusion_flags() const noexcept {
        return exclusion_flags_;
    }
    const NativePtcExclusionMatrix &initial_exclusion_flags() const noexcept {
        return initial_exclusion_flags_;
    }
    const Eigen::VectorXi &apt_exclusion_flags() const noexcept {
        return apt_exclusion_flags_;
    }
    const NativeDetectorFlagBitsMatrix &delivered_flag_bits() const noexcept {
        return delivered_flag_bits_;
    }
    NativeDetectorFlagBits operation_exclusion_bits(
        Eigen::Index detector) const {
        if (detector < 0 || detector >= detector_count()) {
            throw std::out_of_range(
                "native PTC detector exclusion is out of range");
        }
        return detector_operation_exclusion_bits_.at(
            static_cast<std::size_t>(detector));
    }
    const std::vector<NativePtcRowBinding> &row_bindings() const noexcept {
        return row_bindings_;
    }
    Eigen::Index slot_count() const noexcept { return values_.rows(); }
    Eigen::Index detector_count() const noexcept {
        return values_.cols();
    }
    const NativePtcRowBinding &row_binding(Eigen::Index row) const {
        if (row < 0 || row >= slot_count()) {
            throw std::out_of_range(
                "native PTC working row is out of range");
        }
        return row_bindings_.at(static_cast<std::size_t>(row));
    }
    const NativeSampleIdentity &identity(Eigen::Index row,
                                         Eigen::Index detector) const {
        if (row < 0 || detector < 0 || row >= slot_count() ||
            detector >= detector_count()) {
            throw std::out_of_range(
                "native PTC working cell is out of range");
        }
        const auto network = detector_network_ids_.at(
            static_cast<std::size_t>(detector));
        const auto &anchors = row_binding(row).network_anchors;
        const auto found = std::find_if(
            anchors.begin(), anchors.end(),
            [network](const auto &candidate) {
                return candidate.first == network;
            });
        if (found == anchors.end()) {
            throw std::logic_error(
                "native PTC row lacks its detector-network anchor");
        }
        return found->second;
    }
    CoincidenceCellState initial_state(Eigen::Index row,
                                       Eigen::Index detector) const {
        return initial_exclusion_flags_(row, detector)
            ? CoincidenceCellState::mapped_invalid
            : CoincidenceCellState::mapped_valid;
    }
    double input_value(Eigen::Index row, Eigen::Index detector) const {
        return input_values_(row, detector);
    }
    std::string invalidity_reason(Eigen::Index row,
                                  Eigen::Index detector) const {
        const auto delivered = delivered_flag_bits_(row, detector);
        const auto operation = operation_exclusion_bits(detector);
        const auto &apt = detector_apt_flags_.at(
            static_cast<std::size_t>(detector));
        std::string result;
        if (initial_state(row, detector) ==
            CoincidenceCellState::mapped_invalid) {
            if (delivered != 0) result = "RTC-delivered detector flag support";
            if (operation != 0) {
                if (!result.empty()) result += "; ";
                result += "actual PTC operation exclusion bits";
            }
            if (!apt.has_value() || *apt != 0) {
                if (!result.empty()) result += "; ";
                result += apt.has_value() ? "typed APT detector exclusion"
                                          : "typed APT detector flag missing";
            }
        }
        return result;
    }

    NativePtcGroupWorkingSet(
        NativeOperationIdentity operation, std::size_t segment_ordinal,
        std::string effective_grouping, std::int64_t group_key,
        std::size_t subgroup_index, NativePtcGroupRole role,
        std::vector<TimestreamDetectorColumn> detector_columns,
        std::vector<std::int64_t> detector_uids,
        std::vector<TimestreamNetworkId> detector_network_ids,
        std::vector<std::optional<std::int64_t>> detector_apt_flags,
        Eigen::MatrixXd values,
        Eigen::MatrixXd input_values,
        NativePtcExclusionMatrix exclusion_flags,
        NativeDetectorFlagBitsMatrix delivered_flag_bits,
        std::vector<NativeDetectorFlagBits>
            detector_operation_exclusion_bits,
        Eigen::VectorXi apt_exclusion_flags,
        std::vector<NativePtcRowBinding> row_bindings,
        std::optional<Eigen::MatrixXd> kernel_values = {})
        : operation_{operation}, segment_ordinal_{segment_ordinal},
          effective_grouping_{std::move(effective_grouping)},
          group_key_{group_key}, subgroup_index_{subgroup_index},
          role_{role}, detector_columns_{std::move(detector_columns)},
          detector_uids_{std::move(detector_uids)},
          detector_network_ids_{std::move(detector_network_ids)},
          detector_apt_flags_{std::move(detector_apt_flags)},
          values_{std::move(values)},
          input_values_{std::move(input_values)},
          initial_exclusion_flags_{exclusion_flags},
          exclusion_flags_{std::move(exclusion_flags)},
          delivered_flag_bits_{std::move(delivered_flag_bits)},
          detector_operation_exclusion_bits_{
              std::move(detector_operation_exclusion_bits)},
          apt_exclusion_flags_{std::move(apt_exclusion_flags)},
          row_bindings_{std::move(row_bindings)},
          kernel_values_{std::move(kernel_values)} {}

    void commit_processed_values(
        Eigen::MatrixXd values,
        NativePtcExclusionMatrix exclusion_flags) {
        if (values.rows() != slot_count() ||
            values.cols() != detector_count() ||
            !values.array().isFinite().all() ||
            exclusion_flags.rows() != slot_count() ||
            exclusion_flags.cols() != detector_count() ||
            (exclusion_flags_.array() &&
             !exclusion_flags.array()).any()) {
            throw std::logic_error(
                "native PTC committed numerical state is incomplete or removes an exclusion");
        }
        for (Eigen::Index row = 0; row < slot_count(); ++row) {
            for (Eigen::Index detector = 0;
                 detector < detector_count(); ++detector) {
                if (exclusion_flags(row, detector) ||
                    role_ == NativePtcGroupRole::pass_through) {
                    values(row, detector) = input_values_(row, detector);
                }
            }
        }
        values_ = std::move(values);
        exclusion_flags_ = std::move(exclusion_flags);
    }

private:
    NativeOperationIdentity operation_;
    std::size_t segment_ordinal_ = 0;
    std::string effective_grouping_;
    std::int64_t group_key_ = -1;
    std::size_t subgroup_index_ = 0;
    NativePtcGroupRole role_ = NativePtcGroupRole::pca_clean;
    std::vector<TimestreamDetectorColumn> detector_columns_;
    std::vector<std::int64_t> detector_uids_;
    std::vector<TimestreamNetworkId> detector_network_ids_;
    std::vector<std::optional<std::int64_t>> detector_apt_flags_;
    Eigen::MatrixXd values_;
    Eigen::MatrixXd input_values_;
    NativePtcExclusionMatrix initial_exclusion_flags_;
    NativePtcExclusionMatrix exclusion_flags_;
    NativeDetectorFlagBitsMatrix delivered_flag_bits_;
    std::vector<NativeDetectorFlagBits>
        detector_operation_exclusion_bits_;
    Eigen::VectorXi apt_exclusion_flags_;
    std::vector<NativePtcRowBinding> row_bindings_;
    std::optional<Eigen::MatrixXd> kernel_values_;
};

struct NativePtcCohortRequest {
    std::string grouping;
    FinitePcaPlaceholder excluded_placeholder;
    std::map<TimestreamDetectorColumn, NativeDetectorFlagBits>
        operation_exclusion_bits;
    PcaCompatibilityInputs optional_modes;
    bool corr_grouping_enabled = false;
    bool requires_second_pass_window = false;
};

class NativePtcCorrGroupingBody {
public:
    using Groups = std::vector<std::vector<Eigen::Index>>;
    using Function =
        std::function<Groups(const NativePtcGroupWorkingSet &)>;

    NativePtcCorrGroupingBody() = default;
    NativePtcCorrGroupingBody(Function function)
        : function_{std::move(function)} {}

    explicit operator bool() const noexcept {
        return static_cast<bool>(function_);
    }
    Groups operator()(const NativePtcGroupWorkingSet &group) const {
        if (!function_) {
            throw std::logic_error(
                "native corr_nw grouping body is not configured");
        }
        return function_(group);
    }

private:
    Function function_;
};

class NativePtcPreparedOperation {
public:
    const NativeOperationIdentity &operation() const noexcept {
        return operation_;
    }
    const std::string &requested_grouping() const noexcept {
        return requested_grouping_;
    }
    const std::string &effective_grouping() const noexcept {
        return effective_grouping_;
    }
    std::size_t detector_count() const noexcept {
        return detector_count_;
    }
    std::size_t segment_count() const noexcept { return segment_count_; }
    const std::vector<NativePtcGroupWorkingSet> &groups() const noexcept {
        return groups_;
    }
    std::vector<NativePtcGroupWorkingSet> &groups_for_commit() noexcept {
        return groups_;
    }
    const std::shared_ptr<const NativeMeasuredDetectorScan> &mapping_handle()
        const noexcept {
        return mapping_;
    }

    NativePtcPreparedOperation(
        std::shared_ptr<const NativeMeasuredDetectorScan> mapping,
        NativeOperationIdentity operation, std::string requested_grouping,
        std::string effective_grouping, std::size_t detector_count,
        std::size_t segment_count,
        std::vector<NativePtcGroupWorkingSet> groups)
        : mapping_{std::move(mapping)}, operation_{operation},
          requested_grouping_{std::move(requested_grouping)},
          effective_grouping_{std::move(effective_grouping)},
          detector_count_{detector_count}, segment_count_{segment_count},
          groups_{std::move(groups)} {
        if (!mapping_) {
            throw std::invalid_argument(
                "native PTC prepared operation requires its scan mapping");
        }
    }

private:
    std::shared_ptr<const NativeMeasuredDetectorScan> mapping_;
    NativeOperationIdentity operation_;
    std::string requested_grouping_;
    std::string effective_grouping_;
    std::size_t detector_count_ = 0;
    std::size_t segment_count_ = 0;
    std::vector<NativePtcGroupWorkingSet> groups_;
};

inline NativePtcGroupWorkingSet make_native_ptc_group(
    const NativeMeasuredDetectorScan &scan,
    const NativePtcRtcCohortSegment &segment,
    NativeOperationIdentity operation, std::string effective_grouping,
    std::int64_t group_key, std::size_t subgroup_index,
    NativePtcGroupRole role,
    std::vector<TimestreamDetectorColumn> detector_columns,
    const std::map<TimestreamDetectorColumn, NativeDetectorFlagBits> &
        operation_exclusion_bits,
    FinitePcaPlaceholder excluded_placeholder) {
    if (segment.rows.empty() || detector_columns.empty()) {
        throw std::invalid_argument(
            "native PTC group requires cohort rows and detector columns");
    }
    const auto rows = static_cast<Eigen::Index>(segment.rows.size());
    const auto columns =
        static_cast<Eigen::Index>(detector_columns.size());
    Eigen::MatrixXd values(rows, columns);
    Eigen::MatrixXd input_values(rows, columns);
    const bool has_kernel = segment.network_runs.front()
        .second->selected_kernel_values.has_value();
    std::optional<Eigen::MatrixXd> kernel_values;
    if (has_kernel) kernel_values.emplace(rows, columns);
    NativePtcExclusionMatrix exclusions(rows, columns);
    exclusions.setConstant(true);
    NativeDetectorFlagBitsMatrix delivered_flag_bits(rows, columns);
    delivered_flag_bits.setZero();
    std::vector<NativeDetectorFlagBits> detector_operation_bits;
    detector_operation_bits.reserve(detector_columns.size());
    Eigen::VectorXi apt_exclusions(columns);
    std::vector<std::int64_t> detector_uids;
    detector_uids.reserve(detector_columns.size());
    std::vector<TimestreamNetworkId> detector_network_ids;
    detector_network_ids.reserve(detector_columns.size());
    std::vector<std::optional<std::int64_t>> detector_apt_flags;
    detector_apt_flags.reserve(detector_columns.size());
    std::vector<NativePtcRowBinding> row_bindings;
    row_bindings.reserve(static_cast<std::size_t>(rows));
    std::set<TimestreamDetectorColumn> seen_columns;

    for (Eigen::Index local = 0; local < columns; ++local) {
        const auto detector_column = detector_columns.at(
            static_cast<std::size_t>(local));
        if (!seen_columns.insert(detector_column).second) {
            throw std::logic_error(
                "native PTC group repeats a detector column");
        }
        const auto &binding = scan.binding(detector_column);
        detector_uids.push_back(binding.output_uid);
        detector_network_ids.push_back(binding.network_id);
        detector_apt_flags.push_back(binding.apt_flag);
        const auto operation = operation_exclusion_bits.find(
            detector_column);
        detector_operation_bits.push_back(
            operation == operation_exclusion_bits.end()
                ? NativeDetectorFlagBits{0}
                : operation->second);
        apt_exclusions(local) =
            binding.apt_flag.has_value() && *binding.apt_flag == 0 ? 0 : 1;
    }

    for (Eigen::Index row = 0; row < rows; ++row) {
        const auto &cohort_row = segment.rows.at(
            static_cast<std::size_t>(row));
        NativePtcRowBinding row_binding{
            segment.segment_ordinal, cohort_row.segment_output_row,
            cohort_row.exact_common_slots, {}};
        for (Eigen::Index local = 0; local < columns; ++local) {
            const auto detector_column = detector_columns.at(
                static_cast<std::size_t>(local));
            const auto &binding = scan.binding(detector_column);
            const auto &run = segment.run(binding.network_id);
            const auto detector_it = std::find(
                run.input.detector_columns.begin(),
                run.input.detector_columns.end(), detector_column);
            if (detector_it == run.input.detector_columns.end()) {
                throw std::logic_error(
                    "native PTC RTC run lacks its detector column");
            }
            const auto run_local = static_cast<Eigen::Index>(
                std::distance(
                    run.input.detector_columns.begin(), detector_it));
            const auto &support = run.support.at(
                static_cast<std::size_t>(row));
            const auto &sample_identity = support.selected_anchor;
            const auto sample_value = run.selected_values(row, run_local);
            const auto sample_delivered_flag_bits =
                run.ored_flag_bits(row, run_local);
            if (run.selected_kernel_values.has_value() != has_kernel) {
                throw std::logic_error(
                    "native PTC RTC kernel inventory is inconsistent");
            }
            if (kernel_values) {
                (*kernel_values)(row, local) =
                    (*run.selected_kernel_values)(row, run_local);
            }
            const NativeDetectorSampleKey key{
                sample_identity.key(), detector_column};
            const auto operation_bits = detector_operation_bits.at(
                static_cast<std::size_t>(local));
            if (!(scan.sample_identity(key) == sample_identity)) {
                throw std::logic_error(
                    "native PTC RTC anchor differs from the admitted scan");
            }
            const auto anchor = std::find_if(
                row_binding.network_anchors.begin(),
                row_binding.network_anchors.end(),
                [&](const auto &candidate) {
                    return candidate.first == binding.network_id;
                });
            if (anchor == row_binding.network_anchors.end()) {
                row_binding.network_anchors.emplace_back(
                    binding.network_id, sample_identity);
            }
            else if (!(anchor->second == sample_identity)) {
                throw std::logic_error(
                    "native PTC row has unequal detector-network anchors");
            }
            const bool valid = sample_delivered_flag_bits == 0 &&
                operation_bits == 0 && binding.apt_flag.has_value() &&
                *binding.apt_flag == 0;
            input_values(row, local) = sample_value;
            delivered_flag_bits(row, local) =
                sample_delivered_flag_bits;
            if (!valid) {
                values(row, local) = excluded_placeholder.value();
            }
            else {
                exclusions(row, local) = false;
                values(row, local) = sample_value;
            }
        }
        std::sort(
            row_binding.network_anchors.begin(),
            row_binding.network_anchors.end(),
            [](const auto &lhs, const auto &rhs) {
                return lhs.first < rhs.first;
            });
        row_bindings.push_back(std::move(row_binding));
    }
    if (!values.array().isFinite().all()) {
        throw std::logic_error(
            "native PTC working values must be finite");
    }
    return NativePtcGroupWorkingSet{
        operation, segment.segment_ordinal,
        std::move(effective_grouping), group_key, subgroup_index, role,
        std::move(detector_columns), std::move(detector_uids),
        std::move(detector_network_ids), std::move(detector_apt_flags),
        std::move(values), std::move(input_values),
        std::move(exclusions), std::move(delivered_flag_bits),
        std::move(detector_operation_bits), std::move(apt_exclusions),
        std::move(row_bindings),
        std::move(kernel_values)};
}

inline NativePtcPreparedOperation prepare_native_ptc_cohorts(
    NativeMeasuredDetectorLedger &ledger,
    const NativeRtcDispatchResult &rtc,
    const NativePtcCohortRequest &request,
    const NativePtcCorrGroupingBody &corr_grouping_body = {}) {
    const auto &scan = *ledger.mapping_handle();
    const auto requested =
        detail::normalize_native_ptc_grouping(request.grouping);
    if (!detail::native_ptc_grouping_is_supported(requested)) {
        throw std::invalid_argument(
            "native PTC grouping lacks exact typed membership");
    }
    const auto effective =
        requested == "corr_nw" && !request.corr_grouping_enabled
            ? std::string{"nw"}
            : requested;
    if (request.requires_second_pass_window && effective != "nw") {
        throw std::logic_error(
            "native PTC second-pass window requires one complete network cohort");
    }
    const auto segments =
        detail::make_native_ptc_rtc_cohort_segments(scan, rtc);

    bool has_excluded_cells = !request.operation_exclusion_bits.empty();
    for (const auto &[detector, bits] :
         request.operation_exclusion_bits) {
        if (bits == 0 || detector < 0 ||
            static_cast<std::size_t>(detector) >=
                scan.detector_count()) {
            throw std::invalid_argument(
                "native PTC operation exclusion is zero or foreign");
        }
    }
    for (std::size_t detector = 0;
         detector < scan.detector_count(); ++detector) {
        const auto &binding = scan.binding(
            static_cast<TimestreamDetectorColumn>(detector));
        if (!binding.apt_flag.has_value() || *binding.apt_flag != 0) {
            has_excluded_cells = true;
        }
    }
    for (const auto &run : rtc.runs) {
        if ((run.ored_flag_bits.array() != 0).any()) {
            has_excluded_cells = true;
        }
    }
    require_pca_compatibility(classify_pca_compatibility(
        has_excluded_cells, request.optional_modes));

    const auto operation = ledger.next_operation();
    std::map<std::int64_t, std::vector<TimestreamDetectorColumn>>
        ordinary_groups;
    for (std::size_t detector = 0;
         detector < scan.detector_count(); ++detector) {
        const auto column =
            static_cast<TimestreamDetectorColumn>(detector);
        const auto &binding = scan.binding(column);
        std::int64_t key = 0;
        if (effective == "nw" || effective == "corr_nw") {
            key = binding.network_id;
        }
        else if (effective == "array") {
            key = binding.array;
        }
        else if (effective == "detector") {
            key = static_cast<std::int64_t>(column);
        }
        ordinary_groups[key].push_back(column);
    }

    std::vector<NativePtcGroupWorkingSet> groups;
    for (const auto &segment : segments) {
        if (effective == "corr_nw") {
            if (!corr_grouping_body) {
                throw std::logic_error(
                    "enabled native corr_nw grouping requires its established grouping body");
            }
            for (const auto &[network, columns] : ordinary_groups) {
                const auto base = make_native_ptc_group(
                    scan, segment, operation, effective, network,
                    0, NativePtcGroupRole::pca_clean, columns,
                    request.operation_exclusion_bits,
                    request.excluded_placeholder);
                const auto memberships = corr_grouping_body(base);
                std::vector<bool> grouped(columns.size(), false);
                std::size_t subgroup_index = 0;
                for (const auto &membership : memberships) {
                    if (membership.size() < 2) {
                        throw std::logic_error(
                            "native corr_nw clean subgroup requires at least two detectors");
                    }
                    std::vector<TimestreamDetectorColumn> subgroup;
                    subgroup.reserve(membership.size());
                    for (const auto local : membership) {
                        if (local < 0 ||
                            static_cast<std::size_t>(local) >=
                                columns.size() ||
                            grouped.at(static_cast<std::size_t>(local))) {
                            throw std::logic_error(
                                "native corr_nw membership is foreign or duplicated");
                        }
                        grouped.at(static_cast<std::size_t>(local)) = true;
                        subgroup.push_back(
                            columns.at(static_cast<std::size_t>(local)));
                    }
                    groups.push_back(make_native_ptc_group(
                        scan, segment, operation, effective,
                        network, subgroup_index++,
                        NativePtcGroupRole::pca_clean,
                        std::move(subgroup),
                        request.operation_exclusion_bits,
                        request.excluded_placeholder));
                }
                std::vector<TimestreamDetectorColumn> pass_through;
                for (std::size_t local = 0; local < columns.size();
                     ++local) {
                    if (!grouped[local]) {
                        pass_through.push_back(columns[local]);
                    }
                }
                if (!pass_through.empty()) {
                    groups.push_back(make_native_ptc_group(
                        scan, segment, operation, effective,
                        network, subgroup_index,
                        NativePtcGroupRole::pass_through,
                        std::move(pass_through),
                        request.operation_exclusion_bits,
                        request.excluded_placeholder));
                }
            }
        }
        else {
            for (const auto &[key, columns] : ordinary_groups) {
                groups.push_back(make_native_ptc_group(
                    scan, segment, operation, effective, key, 0,
                    NativePtcGroupRole::pca_clean, columns,
                    request.operation_exclusion_bits,
                    request.excluded_placeholder));
            }
        }
    }

    std::vector<std::vector<bool>> seen(
        segments.size(), std::vector<bool>(scan.detector_count(), false));
    for (const auto &group : groups) {
        if (group.segment_ordinal() >= seen.size()) {
            throw std::logic_error(
                "native PTC group has a foreign segment ordinal");
        }
        for (const auto column : group.detector_columns()) {
            if (column < 0 ||
                static_cast<std::size_t>(column) >=
                    scan.detector_count() ||
                seen[group.segment_ordinal()].at(
                    static_cast<std::size_t>(column))) {
                throw std::logic_error(
                    "native PTC segment detector partition is not injective");
            }
            seen[group.segment_ordinal()][
                static_cast<std::size_t>(column)] = true;
        }
    }
    for (const auto &segment_seen : seen) {
        if (!std::all_of(segment_seen.begin(), segment_seen.end(),
                         [](bool value) { return value; })) {
            throw std::logic_error(
                "native PTC segment detector partition is incomplete");
        }
    }

    const auto issued = ledger.issue_operation();
    if (!(issued == operation)) {
        throw std::logic_error(
            "native PTC operation sequence changed during preparation");
    }
    return NativePtcPreparedOperation{
        ledger.mapping_handle(), operation, requested, effective,
        scan.detector_count(),
        segments.size(), std::move(groups)};
}

struct NativePtcNumericalResult {
    Eigen::MatrixXd values;
    std::optional<Eigen::MatrixXd> kernel_values;
    std::optional<NativePtcExclusionMatrix> preclean_exclusion_flags;
    std::optional<NativePtcExclusionMatrix> exclusion_flags;
};

class NativePtcProcessedGroup {
public:
    explicit NativePtcProcessedGroup(
        const NativePtcGroupWorkingSet &source,
        NativePtcNumericalResult result)
        : segment_ordinal_{source.segment_ordinal()},
          effective_grouping_{source.effective_grouping()},
          group_key_{source.group_key()},
          subgroup_index_{source.subgroup_index()}, role_{source.role()},
          detector_columns_{source.detector_columns()},
          values_{std::move(result.values)},
          kernel_values_{std::move(result.kernel_values)},
          preclean_exclusion_flags_{
              std::move(result.preclean_exclusion_flags)},
          exclusion_flags_{std::move(result.exclusion_flags)} {}

    explicit NativePtcProcessedGroup(
        const NativePtcGroupWorkingSet &source, Eigen::MatrixXd values)
        : NativePtcProcessedGroup(
              source, NativePtcNumericalResult{
                          std::move(values), source.kernel_values(),
                          source.exclusion_flags(),
                          source.exclusion_flags()}) {}

    std::size_t segment_ordinal() const noexcept {
        return segment_ordinal_;
    }
    const std::string &effective_grouping() const noexcept {
        return effective_grouping_;
    }
    std::int64_t group_key() const noexcept { return group_key_; }
    std::size_t subgroup_index() const noexcept {
        return subgroup_index_;
    }
    NativePtcGroupRole role() const noexcept { return role_; }
    const std::vector<TimestreamDetectorColumn> &detector_columns() const
        noexcept {
        return detector_columns_;
    }
    const Eigen::MatrixXd &values() const noexcept { return values_; }
    const std::optional<Eigen::MatrixXd> &kernel_values() const noexcept {
        return kernel_values_;
    }
    const std::optional<NativePtcExclusionMatrix> &exclusion_flags() const
        noexcept {
        return exclusion_flags_;
    }
    const std::optional<NativePtcExclusionMatrix> &preclean_exclusion_flags()
        const noexcept {
        return preclean_exclusion_flags_;
    }
    Eigen::MatrixXd &mutable_values_for_retry() noexcept { return values_; }
    Eigen::MatrixXd take_values_for_commit() noexcept {
        return std::move(values_);
    }
    NativePtcExclusionMatrix take_exclusion_flags_for_commit() {
        if (!exclusion_flags_) {
            throw std::logic_error(
                "native PTC processed group lacks final exclusions");
        }
        return std::move(*exclusion_flags_);
    }

private:
    std::size_t segment_ordinal_ = 0;
    std::string effective_grouping_;
    std::int64_t group_key_ = -1;
    std::size_t subgroup_index_ = 0;
    NativePtcGroupRole role_ = NativePtcGroupRole::pca_clean;
    std::vector<TimestreamDetectorColumn> detector_columns_;
    Eigen::MatrixXd values_;
    std::optional<Eigen::MatrixXd> kernel_values_;
    std::optional<NativePtcExclusionMatrix> preclean_exclusion_flags_;
    std::optional<NativePtcExclusionMatrix> exclusion_flags_;
};

class NativePtcProcessedOperation {
public:
    const NativeOperationIdentity &operation() const noexcept {
        return operation_;
    }
    const std::vector<NativePtcProcessedGroup> &groups() const noexcept {
        return groups_;
    }
    NativePtcProcessedGroup &mutable_group_for_retry(std::size_t index) {
        return groups_.at(index);
    }

    NativePtcProcessedOperation(
        NativeOperationIdentity operation,
        std::vector<NativePtcProcessedGroup> groups)
        : operation_{operation}, groups_{std::move(groups)} {}

private:
    NativeOperationIdentity operation_;
    std::vector<NativePtcProcessedGroup> groups_;
};

template <class NumericalBody>
NativePtcProcessedOperation run_native_ptc_groups(
    const NativePtcPreparedOperation &prepared,
    NumericalBody &&numerical_body,
    bool invoke_pass_through = false) {
    std::vector<NativePtcProcessedGroup> processed;
    processed.reserve(prepared.groups().size());
    for (const auto &group : prepared.groups()) {
        NativePtcNumericalResult result{
            group.values(), group.kernel_values(), group.exclusion_flags(),
            group.exclusion_flags()};
        if (group.role() == NativePtcGroupRole::pca_clean ||
            invoke_pass_through) {
            auto invoked =
                std::invoke(numerical_body, std::as_const(group));
            if constexpr (std::is_same_v<
                              std::decay_t<decltype(invoked)>,
                              NativePtcNumericalResult>) {
                result = std::move(invoked);
            }
            else {
                result.values = std::move(invoked);
            }
        }
        if (result.values.rows() != group.slot_count() ||
            result.values.cols() != group.detector_count() ||
            (result.kernel_values &&
             (result.kernel_values->rows() != group.slot_count() ||
              result.kernel_values->cols() != group.detector_count() ||
              !result.kernel_values->array().isFinite().all())) ||
            result.kernel_values.has_value() !=
                group.kernel_values().has_value() ||
            !result.preclean_exclusion_flags ||
            result.preclean_exclusion_flags->rows() != group.slot_count() ||
            result.preclean_exclusion_flags->cols() !=
                group.detector_count() ||
            (group.exclusion_flags().array() &&
             !result.preclean_exclusion_flags->array()).any() ||
            !result.exclusion_flags ||
            result.exclusion_flags->rows() != group.slot_count() ||
            result.exclusion_flags->cols() != group.detector_count() ||
            (result.preclean_exclusion_flags->array() &&
             !result.exclusion_flags->array()).any()) {
            throw std::logic_error(
                "native PTC numerical result has invalid signal/kernel/flag support");
        }
        processed.emplace_back(group, std::move(result));
    }
    return NativePtcProcessedOperation{
        prepared.operation(), std::move(processed)};
}

inline void scatter_native_ptc_results_transactionally(
    NativeMeasuredDetectorLedger &ledger,
    NativePtcPreparedOperation &prepared,
    NativePtcProcessedOperation &processed) {
    if (!(processed.operation() == prepared.operation()) ||
        processed.groups().size() != prepared.groups().size()) {
        throw std::logic_error(
            "native PTC processed operation identity or group count changed");
    }

    for (std::size_t group_index = 0;
         group_index < prepared.groups().size(); ++group_index) {
        const auto &source = prepared.groups()[group_index];
        const auto &result = processed.groups()[group_index];
        if (result.segment_ordinal() != source.segment_ordinal() ||
            result.effective_grouping() != source.effective_grouping() ||
            result.group_key() != source.group_key() ||
            result.subgroup_index() != source.subgroup_index() ||
            result.role() != source.role() ||
            result.detector_columns() != source.detector_columns() ||
            result.values().rows() != source.slot_count() ||
            result.values().cols() != source.detector_count()) {
            throw std::logic_error(
                "native PTC processed group identity or shape changed");
        }
        if (!result.values().array().isFinite().all()) {
            throw std::logic_error(
                "native PTC processed group contains nonfinite values");
        }
    }

    // All identities, shapes, flags, and finite values are validated before
    // any working group is changed. The commit then replaces dense numerical
    // state as a whole and records one operation identity; no per-cell update
    // batch or revision history is retained.
    for (std::size_t group_index = 0;
         group_index < prepared.groups().size(); ++group_index) {
        auto &source = prepared.groups_for_commit()[group_index];
        auto &result = processed.mutable_group_for_retry(group_index);
        source.commit_processed_values(
            result.take_values_for_commit(),
            result.take_exclusion_flags_for_commit());
    }
    ledger.commit_operation(prepared.operation());
}

}  // namespace citlali::pipeline
