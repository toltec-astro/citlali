#pragma once

// Implementation detail included by kidsproc.h.

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace citlali::pipeline::native_ingress_v2_detail {

inline std::int64_t parse_network(std::string_view interface_name) {
    constexpr std::string_view prefix = "toltec";
    if (!interface_name.starts_with(prefix) ||
        interface_name.size() == prefix.size()) {
        throw std::runtime_error(
            "native ingress requires an exact toltecN interface");
    }
    std::int64_t result = -1;
    const auto digits = interface_name.substr(prefix.size());
    const auto [end, error] = std::from_chars(
        digits.data(), digits.data() + digits.size(), result);
    if (error != std::errc{} || end != digits.data() + digits.size() ||
        result < 0 || interface_name !=
            std::string{prefix} + std::to_string(result)) {
        throw std::runtime_error(
            "native ingress requires an exact toltecN interface");
    }
    return result;
}

inline int checked_slice_index(TimestreamNativeRow row) {
    if (row < 0 || row > std::numeric_limits<int>::max()) {
        throw std::overflow_error(
            "native ingress row is outside the KIDs slice range");
    }
    return static_cast<int>(row);
}

}  // namespace citlali::pipeline::native_ingress_v2_detail

template <typename Derived>
auto KidsDataProc::populate_rtc_from_rawobs(const RawObs &rawobs, const Eigen::Index scan,
                                            Eigen::DenseBase<Derived> &scan_indices,
                                            std::vector<Eigen::Index> &start_indices,
                                            std::vector<Eigen::Index> &end_indices,
                                            const int n_pts, const int n_det,
                                            citlali::config::TodType data_type) {
    // resize data
    Eigen::MatrixXd data(n_pts, n_det);

    Eigen::Index i = 0;
    for (const auto &data_item : rawobs.kidsdata()) {
        auto slice = tula::container_utils::Slice<int>{scan_indices(2,scan) + start_indices[i],
                                                       scan_indices(3,scan) + 1 + start_indices[i],
                                                       std::nullopt};
        auto rts = load_data_item(data_item, slice);
        auto result = this->solver()(rts, Solver::Config{});

        Eigen::Index n_cols = 0;
        citlali::pipeline::visit_kids_tod_channel(
            result, data_type, [&](const auto &channel) {
                Eigen::Index n_rows = channel.rows();
                n_cols = channel.cols();
                data.block(0, i, n_rows, n_cols) = channel;
            });

        // increment columns
        i += n_cols;
    }

    citlali::pipeline::require_finite_kids_input(
        data, "direct RTC KIDs input");

    return data;
}

inline auto KidsDataProc::make_native_measured_scan(
    const RawObs &rawobs,
    citlali::pipeline::NativeScanChunkScope scope,
    std::shared_ptr<const citlali::pipeline::NativeObservationCarriers>
        carriers,
    std::shared_ptr<const
        citlali::pipeline::CanonicalAptDetectorRelationV2> relation,
    std::size_t first_common_slot, std::size_t past_last_common_slot,
    citlali::config::TodType data_type)
    -> std::shared_ptr<const
        citlali::pipeline::NativeMeasuredDetectorScan> {
    namespace pipeline = citlali::pipeline;
    namespace detail = pipeline::native_ingress_v2_detail;
    if (!carriers || !relation || first_common_slot >= past_last_common_slot ||
        past_last_common_slot > carriers->alignment_handle()->slot_count()) {
        throw std::invalid_argument(
            "native ingress relation, carriers, or scan slots are incomplete");
    }

    std::map<pipeline::TimestreamNetworkId, const RawObs::DataItem *>
        data_by_network;
    for (const auto &data_item_ref : rawobs.kidsdata()) {
        const RawObs::DataItem &data_item = data_item_ref.get();
        const auto network = detail::parse_network(data_item.interface());
        if (!data_by_network.emplace(network, &data_item).second) {
            throw std::invalid_argument(
                "native ingress repeats a raw network");
        }
    }

    std::vector<pipeline::NativeMeasuredNetworkInput> inputs;
    const auto &alignment = *carriers->alignment_handle();
    inputs.reserve(alignment.participant_network_ids().size());
    for (const auto network_id : alignment.participant_network_ids()) {
        const auto raw = std::find_if(
            relation->raw_sources().begin(), relation->raw_sources().end(),
            [&](const auto &source) { return source.network == network_id; });
        const auto item = data_by_network.find(network_id);
        if (raw == relation->raw_sources().end() ||
            item == data_by_network.end() ||
            raw->interface_name != item->second->interface()) {
            throw std::logic_error(
                "native ingress raw manifest and delivered input disagree");
        }
        std::optional<pipeline::TimestreamNativeRow> first;
        std::optional<pipeline::TimestreamNativeRow> past;
        for (std::size_t slot = first_common_slot;
             slot < past_last_common_slot; ++slot) {
            const auto &association = alignment.association(network_id, slot);
            if (!association.mapped()) continue;
            first = first ? std::min(*first, association.native_row)
                          : association.native_row;
            const auto next = association.native_row + 1;
            past = past ? std::max(*past, next) : next;
        }
        if (!first || !past || *first >= *past) {
            throw std::logic_error(
                "native ingress scan has no delivered participant rows");
        }

        const auto slice = tula::container_utils::Slice<int>{
            detail::checked_slice_index(*first),
            detail::checked_slice_index(*past), std::nullopt};
        auto raw_timestream = load_data_item(*item->second, slice);
        auto solver_result = this->solver()(raw_timestream, Solver::Config{});
        Eigen::MatrixXd solved;
        bool selected = false;
        pipeline::visit_kids_tod_channel(
            solver_result, data_type, [&](const auto &channel) {
                if (selected) {
                    throw std::logic_error(
                        "native ingress selected its TOD channel twice");
                }
                selected = true;
                solved = channel;
            });
        if (!selected || solved.rows() != *past - *first ||
            solved.cols() != raw->channel_count) {
            throw std::runtime_error(
                "native KIDs solver result has foreign row/channel shape");
        }
        pipeline::require_finite_kids_input(
            solved, "native measured RTC KIDs input");
        auto values = std::make_shared<const Eigen::MatrixXd>(
            std::move(solved));
        auto flags = std::make_shared<const
            pipeline::NativeDetectorFlagBitsMatrix>(
                pipeline::NativeDetectorFlagBitsMatrix::Zero(
                    values->rows(), values->cols()));
        inputs.emplace_back(
            raw->source_uid, network_id, raw->interface_name, *first,
            std::move(values), std::move(flags));
    }
    return pipeline::NativeMeasuredDetectorScan::admit(
        std::move(scope), std::move(carriers), std::move(relation),
        first_common_slot, past_last_common_slot, std::move(inputs));
}
