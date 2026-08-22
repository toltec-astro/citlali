#pragma once

#include <citlali/core/pipeline/timestream_measured_scan.h>
#include <citlali/core/timestream/rtc/downsample.h>

#include <Eigen/Core>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace citlali::pipeline {

// Stage 4 admits only numerical bodies whose complete temporal support is
// contained by one packet-contiguous native run. A caller that needs samples
// from either adjacent run must fail before any body invocation.
struct NativeRtcDispatchRequest {
    int downsample_factor = 1;
    bool requires_cross_run_window = false;
};

struct NativeRtcRunInput {
    std::size_t segment_ordinal = 0;
    std::size_t first_common_slot = 0;
    std::size_t past_last_common_slot = 0;
    NativeContiguousRun run;
    std::vector<std::size_t> common_slots;
    std::vector<TimestreamDetectorColumn> detector_columns;
    Eigen::MatrixXd measured_values;
    NativeDetectorFlagBitsMatrix input_flag_bits;
};

// The established RTC body receives one owned run-local matrix. It may change
// values and add flag bits, but it cannot change row, detector, or native
// identity. Original delivered bits must remain present. The recorded OR is
// the exact actual flag support entering the established RTC downsampler.
struct NativeRtcProcessedRun {
    Eigen::MatrixXd values;
    NativeDetectorFlagBitsMatrix flag_bits;
};

struct NativeRtcOutputRowSupport {
    std::size_t segment_ordinal = 0;
    Eigen::Index run_output_row = -1;
    int factor = 1;
    NativeSampleIdentity selected_anchor;
    std::vector<std::size_t> exact_common_slots;
    std::vector<NativeSampleIdentity> exact_native_support;
    bool final_short_support = false;
    std::vector<TimestreamDetectorColumn> detector_columns;
    std::vector<NativeDetectorFlagBits> ored_flag_support;
};

struct NativeRtcRunResult {
    NativeRtcRunInput input;
    Eigen::MatrixXd selected_values;
    NativeDetectorFlagBitsMatrix ored_flag_bits;
    std::vector<NativeRtcOutputRowSupport> support;
};

struct NativeRtcDispatchResult {
    int downsample_factor = 1;
    std::vector<NativeRtcRunResult> runs;

    std::size_t output_row_count() const noexcept {
        std::size_t result = 0;
        for (const auto &run : runs) result += run.support.size();
        return result;
    }
};

namespace detail {

inline bool native_rtc_slot_is_complete(
    const NativeMeasuredDetectorScan &scan, std::size_t common_slot) {
    const auto &alignment = *scan.carriers_handle()->alignment_handle();
    for (const auto network_id : alignment.participant_network_ids()) {
        if (!alignment.association(network_id, common_slot).mapped()) {
            return false;
        }
    }
    return true;
}

inline bool native_rtc_slots_share_one_run(
    const NativeMeasuredDetectorScan &scan, std::size_t before,
    std::size_t after) {
    const auto &alignment = *scan.carriers_handle()->alignment_handle();
    for (const auto network_id : alignment.participant_network_ids()) {
        const auto before_row =
            alignment.association(network_id, before).native_row;
        const auto after_row =
            alignment.association(network_id, after).native_row;
        if (before_row == std::numeric_limits<TimestreamNativeRow>::max() ||
            after_row != before_row + 1 ||
            alignment.network(network_id)
                .discontinuity_between(before_row, after_row)
                .has_value()) {
            return false;
        }
    }
    return true;
}

inline NativeRunBoundary native_rtc_boundary_before(
    const NativeNetworkAlignment &network,
    TimestreamNativeRow first_native_row, bool scan_boundary) {
    NativeRunBoundary result;
    result.scan_boundary = scan_boundary;
    result.stream_boundary =
        first_native_row == network.first_native_row();
    if (first_native_row > network.first_native_row()) {
        result.counter_discontinuity = network.discontinuity_between(
            first_native_row - 1, first_native_row);
    }
    return result;
}

inline NativeRunBoundary native_rtc_boundary_after(
    const NativeNetworkAlignment &network,
    TimestreamNativeRow past_last_native_row, bool scan_boundary) {
    NativeRunBoundary result;
    result.scan_boundary = scan_boundary;
    result.stream_boundary =
        past_last_native_row == network.past_last_native_row();
    if (past_last_native_row < network.past_last_native_row()) {
        result.counter_discontinuity = network.discontinuity_between(
            past_last_native_row - 1, past_last_native_row);
    }
    return result;
}

inline std::vector<TimestreamDetectorColumn>
native_rtc_detector_partition(const NativeMeasuredDetectorScan &scan,
                              TimestreamNetworkId network_id) {
    std::vector<TimestreamDetectorColumn> result;
    for (const auto &binding : scan.bindings()) {
        if (binding.network_id == network_id) {
            result.push_back(binding.detector_column);
        }
    }
    if (result.empty()) {
        throw std::logic_error(
            "native RTC participant has no detector partition");
    }
    return result;
}

inline NativeRtcRunInput native_rtc_gather_run(
    const NativeMeasuredDetectorScan &scan, std::size_t segment_ordinal,
    std::size_t first_common_slot, std::size_t past_last_common_slot,
    TimestreamNetworkId network_id) {
    const auto &alignment = *scan.carriers_handle()->alignment_handle();
    const auto &network = alignment.network(network_id);
    const auto detector_columns =
        native_rtc_detector_partition(scan, network_id);
    const auto row_count = past_last_common_slot - first_common_slot;
    if (row_count == 0 ||
        row_count > static_cast<std::size_t>(
                        std::numeric_limits<Eigen::Index>::max()) ||
        detector_columns.size() > static_cast<std::size_t>(
                                      std::numeric_limits<Eigen::Index>::max())) {
        throw std::length_error(
            "native RTC run shape is empty or unrepresentable");
    }

    const auto first_native_row =
        alignment.association(network_id, first_common_slot).native_row;
    const auto past_last_native_row =
        alignment.association(network_id, past_last_common_slot - 1)
            .native_row + 1;
    if (past_last_native_row - first_native_row !=
        static_cast<TimestreamNativeRow>(row_count)) {
        throw std::logic_error(
            "native RTC run associations are not row-contiguous");
    }

    std::vector<std::size_t> common_slots;
    common_slots.reserve(row_count);
    Eigen::MatrixXd measured_values(
        static_cast<Eigen::Index>(row_count),
        static_cast<Eigen::Index>(detector_columns.size()));
    NativeDetectorFlagBitsMatrix input_flag_bits(
        measured_values.rows(), measured_values.cols());
    for (std::size_t offset = 0; offset < row_count; ++offset) {
        const auto slot = first_common_slot + offset;
        const auto native_row =
            alignment.association(network_id, slot).native_row;
        if (native_row != first_native_row +
                              static_cast<TimestreamNativeRow>(offset)) {
            throw std::logic_error(
                "native RTC gathered rows are not ordered and complete");
        }
        common_slots.push_back(slot);
        for (std::size_t detector = 0;
             detector < detector_columns.size(); ++detector) {
            const NativeDetectorSampleKey key{
                {network_id, native_row}, detector_columns[detector]};
            const auto row = static_cast<Eigen::Index>(offset);
            const auto column = static_cast<Eigen::Index>(detector);
            measured_values(row, column) = scan.measured_value(key);
            input_flag_bits(row, column) =
                scan.original_flag_bits(key);
        }
    }

    return NativeRtcRunInput{
        segment_ordinal,
        first_common_slot,
        past_last_common_slot,
        NativeContiguousRun{
            network_id, first_native_row, past_last_native_row,
            native_rtc_boundary_before(
                network, first_native_row,
                first_common_slot == scan.first_common_slot()),
            native_rtc_boundary_after(
                network, past_last_native_row,
                past_last_common_slot == scan.past_last_common_slot())},
        std::move(common_slots), detector_columns,
        std::move(measured_values), std::move(input_flag_bits)};
}

inline NativeDetectorFlagBitsMatrix native_rtc_or_flag_support(
    const NativeDetectorFlagBitsMatrix &input, int factor) {
    const auto output_rows =
        (input.rows() + static_cast<Eigen::Index>(factor) - 1) /
        static_cast<Eigen::Index>(factor);
    NativeDetectorFlagBitsMatrix result(output_rows, input.cols());
    result.setZero();
    for (Eigen::Index output_row = 0; output_row < output_rows;
         ++output_row) {
        const auto first = output_row * factor;
        const auto past = std::min<Eigen::Index>(
            first + factor, input.rows());
        for (Eigen::Index row = first; row < past; ++row) {
            for (Eigen::Index detector = 0; detector < input.cols();
                 ++detector) {
                result(output_row, detector) |= input(row, detector);
            }
        }
    }
    return result;
}

inline std::vector<NativeRtcOutputRowSupport> native_rtc_support(
    const NativeMeasuredDetectorScan &scan, const NativeRtcRunInput &input,
    const NativeDetectorFlagBitsMatrix &actual_flag_bits, int factor) {
    const auto &network = scan.carriers_handle()->alignment_handle()->network(
        input.run.network_id);
    const auto output_rows =
        (input.measured_values.rows() +
         static_cast<Eigen::Index>(factor) - 1) /
        static_cast<Eigen::Index>(factor);
    std::vector<NativeRtcOutputRowSupport> result;
    result.reserve(static_cast<std::size_t>(output_rows));
    for (Eigen::Index output_row = 0; output_row < output_rows;
         ++output_row) {
        const auto first = output_row * factor;
        const auto past = std::min<Eigen::Index>(
            first + factor, input.measured_values.rows());
        std::vector<std::size_t> common_slots;
        std::vector<NativeSampleIdentity> native_support;
        std::vector<NativeDetectorFlagBits> ored_flags(
            input.detector_columns.size(), 0);
        common_slots.reserve(static_cast<std::size_t>(past - first));
        native_support.reserve(static_cast<std::size_t>(past - first));
        for (Eigen::Index row = first; row < past; ++row) {
            common_slots.push_back(
                input.common_slots.at(static_cast<std::size_t>(row)));
            const auto native_row = input.run.first_native_row + row;
            native_support.push_back(network.identity(native_row));
            for (Eigen::Index detector = 0;
                 detector < actual_flag_bits.cols(); ++detector) {
                ored_flags.at(static_cast<std::size_t>(detector)) |=
                    actual_flag_bits(row, detector);
            }
        }
        result.push_back(NativeRtcOutputRowSupport{
            input.segment_ordinal, output_row, factor,
            native_support.front(), std::move(common_slots),
            std::move(native_support), past - first < factor,
            input.detector_columns, std::move(ored_flags)});
    }
    return result;
}

}  // namespace detail

// Build the complete candidate before invoking a numerical body. This keeps
// malformed, absent, nonfinite, or cross-run requests atomic with respect to
// the scan mapping, ledger, and body invocation count.
inline std::vector<NativeRtcRunInput> prepare_native_rtc_runs(
    const NativeMeasuredDetectorScan &scan,
    const NativeRtcDispatchRequest &request) {
    if (request.downsample_factor <= 0) {
        throw std::invalid_argument(
            "native RTC downsample factor must be positive");
    }
    if (request.requires_cross_run_window) {
        throw std::logic_error(
            "native RTC operation requires a forbidden cross-run window");
    }

    std::vector<std::pair<std::size_t, std::size_t>> segments;
    std::optional<std::size_t> open_segment;
    for (std::size_t slot = scan.first_common_slot();
         slot < scan.past_last_common_slot(); ++slot) {
        if (!detail::native_rtc_slot_is_complete(scan, slot)) {
            if (open_segment.has_value()) {
                segments.emplace_back(*open_segment, slot);
                open_segment.reset();
            }
            continue;
        }
        if (!open_segment.has_value()) {
            open_segment = slot;
            continue;
        }
        if (!detail::native_rtc_slots_share_one_run(scan, slot - 1,
                                                    slot)) {
            segments.emplace_back(*open_segment, slot);
            open_segment = slot;
        }
    }
    if (open_segment.has_value()) {
        segments.emplace_back(*open_segment,
                              scan.past_last_common_slot());
    }

    const auto &network_ids = scan.carriers_handle()
                                  ->alignment_handle()
                                  ->participant_network_ids();
    std::vector<NativeRtcRunInput> result;
    result.reserve(segments.size() * network_ids.size());
    for (std::size_t segment_ordinal = 0;
         segment_ordinal < segments.size(); ++segment_ordinal) {
        const auto [first, past_last] = segments[segment_ordinal];
        for (const auto network_id : network_ids) {
            auto run = detail::native_rtc_gather_run(
                scan, segment_ordinal, first, past_last, network_id);
            if (!run.measured_values.array().isFinite().all()) {
                throw std::logic_error(
                    "native RTC run contains a nonfinite measured value");
            }
            result.push_back(std::move(run));
        }
    }
    return result;
}

template <class NumericalBody>
NativeRtcDispatchResult dispatch_native_rtc_runs(
    const NativeMeasuredDetectorScan &scan,
    const NativeRtcDispatchRequest &request,
    NumericalBody &&numerical_body) {
    auto inputs = prepare_native_rtc_runs(scan, request);
    NativeRtcDispatchResult result;
    result.downsample_factor = request.downsample_factor;
    result.runs.reserve(inputs.size());

    for (auto &input : inputs) {
        auto processed = std::invoke(numerical_body,
                                     std::as_const(input));
        if (processed.values.rows() != input.measured_values.rows() ||
            processed.values.cols() != input.measured_values.cols() ||
            processed.flag_bits.rows() != input.input_flag_bits.rows() ||
            processed.flag_bits.cols() != input.input_flag_bits.cols()) {
            throw std::logic_error(
                "native RTC numerical body changed its run shape");
        }
        if (!processed.values.array().isFinite().all()) {
            throw std::logic_error(
                "native RTC numerical body returned nonfinite values");
        }
        for (Eigen::Index row = 0; row < processed.flag_bits.rows(); ++row) {
            for (Eigen::Index detector = 0;
                 detector < processed.flag_bits.cols(); ++detector) {
                const auto original = input.input_flag_bits(row, detector);
                if ((processed.flag_bits(row, detector) & original) !=
                    original) {
                    throw std::logic_error(
                        "native RTC numerical body removed delivered flag bits");
                }
            }
        }

        timestream::Downsampler downsampler;
        downsampler.factor = request.downsample_factor;
        Eigen::MatrixXd selected_values;
        downsampler.downsample(processed.values, selected_values);
        auto ored_flags = detail::native_rtc_or_flag_support(
            processed.flag_bits, request.downsample_factor);
        auto support = detail::native_rtc_support(
            scan, input, processed.flag_bits, request.downsample_factor);
        if (selected_values.rows() !=
                static_cast<Eigen::Index>(support.size()) ||
            ored_flags.rows() !=
                static_cast<Eigen::Index>(support.size()) ||
            selected_values.cols() !=
                static_cast<Eigen::Index>(input.detector_columns.size()) ||
            ored_flags.cols() != selected_values.cols()) {
            throw std::logic_error(
                "native RTC downsample result and support disagree");
        }
        for (Eigen::Index row = 0; row < ored_flags.rows(); ++row) {
            const auto &row_support =
                support.at(static_cast<std::size_t>(row));
            for (Eigen::Index detector = 0;
                 detector < ored_flags.cols(); ++detector) {
                const auto anchor = static_cast<Eigen::Index>(
                    row_support.selected_anchor.native_row() -
                    input.run.first_native_row);
                if (std::bit_cast<std::uint64_t>(
                        selected_values(row, detector)) !=
                    std::bit_cast<std::uint64_t>(
                        processed.values(anchor, detector))) {
                    throw std::logic_error(
                        "native RTC stride anchor differs from the established downsampler");
                }
                if (ored_flags(row, detector) !=
                    row_support.ored_flag_support.at(
                        static_cast<std::size_t>(detector))) {
                    throw std::logic_error(
                        "native RTC bitwise flag support is inconsistent");
                }
            }
        }
        result.runs.push_back(NativeRtcRunResult{
            std::move(input), std::move(selected_values),
            std::move(ored_flags), std::move(support)});
    }
    return result;
}

}  // namespace citlali::pipeline
