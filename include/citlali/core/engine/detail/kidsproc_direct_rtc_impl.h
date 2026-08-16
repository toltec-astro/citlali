#pragma once

// Implementation detail included by kidsproc.h.

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string_view>

namespace citlali::pipeline::native_ingress_detail {

inline std::int64_t parse_exact_toltec_network(
    std::string_view interface_name) {
    constexpr std::string_view prefix = "toltec";
    if (!interface_name.starts_with(prefix) ||
        interface_name.size() == prefix.size()) {
        throw std::runtime_error(
            "native ingress requires an exact toltecN interface");
    }
    const auto digits = interface_name.substr(prefix.size());
    std::int64_t network = -1;
    const auto [end, error] = std::from_chars(
        digits.data(), digits.data() + digits.size(), network);
    if (error != std::errc{} || end != digits.data() + digits.size() ||
        network < 0 ||
        interface_name != std::string(prefix) + std::to_string(network)) {
        throw std::runtime_error(
            "native ingress requires an exact toltecN interface");
    }
    return network;
}

struct RawDetectorInput {
    const RawObs::DataItem *data_item = nullptr;
    std::int64_t network = -1;
    std::size_t channel_count = 0;
};

inline RawDetectorInput inspect_raw_detector_input(
    const RawObs::DataItem &data_item) {
    const auto declared_network =
        parse_exact_toltec_network(data_item.interface());
    try {
        netCDF::NcFile file(data_item.filepath(), netCDF::NcFile::read);
        const auto roach = file.getVar("Header.Toltec.RoachIndex");
        if (roach.isNull() || !roach.getDims().empty() ||
            roach.getType().getTypeClass() != netCDF::NcType::nc_INT) {
            throw std::runtime_error(
                "native ingress requires scalar int Header.Toltec.RoachIndex");
        }
        int roach_index = -1;
        roach.getVar(&roach_index);
        if (roach_index != declared_network) {
            throw std::runtime_error(
                "native ingress raw RoachIndex disagrees with its interface");
        }
        const auto detector_data = file.getVar("Data.Toltec.Is");
        if (detector_data.isNull()) {
            throw std::runtime_error(
                "native ingress requires two-dimensional Data.Toltec.Is");
        }
        const auto dimensions = detector_data.getDims();
        if (dimensions.size() != 2 || dimensions[1].getSize() == 0 ||
            dimensions[1].getSize() >
                static_cast<std::size_t>(
                    canonical_apt_v1::uid_v1_max + 1)) {
            throw std::runtime_error(
                "native ingress requires a representable nonempty detector axis");
        }
        return RawDetectorInput{
            &data_item, declared_network, dimensions[1].getSize()};
    }
    catch (const netCDF::exceptions::NcException &error) {
        throw std::runtime_error(fmt::format(
            "native ingress could not inspect raw detector input {}: {}",
            data_item.filepath(), error.what()));
    }
}

inline int checked_native_slice_index(TimestreamNativeRow value,
                                      std::string_view label) {
    if (value < 0 ||
        value > static_cast<TimestreamNativeRow>(
                    std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            std::string(label) + " is outside the KIDs slice index range");
    }
    return static_cast<int>(value);
}

inline Eigen::Index checked_eigen_index(std::size_t value,
                                        std::string_view label) {
    if (value > static_cast<std::size_t>(
                    std::numeric_limits<Eigen::Index>::max())) {
        throw std::overflow_error(
            std::string(label) + " is outside the Eigen index range");
    }
    return static_cast<Eigen::Index>(value);
}

}  // namespace citlali::pipeline::native_ingress_detail

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

inline auto KidsDataProc::solve_native_detector_scan(
    const RawObs &rawobs,
    const citlali::pipeline::NativeAlignmentPlan &alignment,
    const citlali::pipeline::AptDetectorRelation &relation,
    const std::vector<citlali::pipeline::NativeCompleteCohortRun> &runs,
    citlali::config::TodType data_type)
    -> citlali::pipeline::NativeMeasuredDetectorScan {
    using namespace citlali::pipeline;
    namespace detail = citlali::pipeline::native_ingress_detail;

    if (runs.empty()) {
        throw std::invalid_argument(
            "native detector ingress requires at least one complete run");
    }
    if (relation.bindings().empty() ||
        relation.bindings().size() >
            static_cast<std::size_t>(
                std::numeric_limits<Eigen::Index>::max())) {
        throw std::invalid_argument(
            "native detector ingress requires a representable typed relation");
    }

    std::map<std::int64_t, detail::RawDetectorInput> raw_by_network;
    for (const auto &data_item_ref : rawobs.kidsdata()) {
        const RawObs::DataItem &data_item = data_item_ref.get();
        auto input = detail::inspect_raw_detector_input(data_item);
        if (!raw_by_network.emplace(input.network, input).second) {
            throw std::invalid_argument(
                "native detector ingress received a duplicate raw network");
        }
    }

    std::map<std::int64_t, std::vector<const AptDetectorBinding *>>
        bindings_by_network;
    std::vector<AptDetectorBindingReference> binding_references;
    binding_references.reserve(relation.bindings().size());
    for (std::size_t column = 0; column < relation.bindings().size();
         ++column) {
        const auto &binding = relation.binding_for_column(column);
        if (!binding.flag.has_value() ||
            (*binding.flag != 0 && *binding.flag != 1)) {
            throw std::invalid_argument(
                "native detector ingress requires a complete typed artifact flag relation");
        }
        auto reference = relation.binding_reference_for_column(column);
        (void)relation.require_binding(reference);
        binding_references.push_back(std::move(reference));
        bindings_by_network[binding.network].push_back(&binding);
    }

    const auto &participant_networks = alignment.participant_network_ids();
    if (participant_networks.empty() ||
        participant_networks.size() != raw_by_network.size() ||
        participant_networks.size() != bindings_by_network.size()) {
        throw std::invalid_argument(
            "native alignment, raw inputs, and typed detector relation have different network sets");
    }
    std::set<std::int64_t> expected_networks;
    for (const auto network_id : participant_networks) {
        if (!expected_networks.insert(network_id).second ||
            raw_by_network.find(network_id) == raw_by_network.end() ||
            bindings_by_network.find(network_id) ==
                bindings_by_network.end()) {
            throw std::invalid_argument(
                "native alignment network is absent or duplicated in raw/typed detector inputs");
        }
    }
    for (const auto &[network, input] : raw_by_network) {
        if (!expected_networks.contains(network)) {
            throw std::invalid_argument(
                "native raw detector input has no alignment participant");
        }
        auto &bindings = bindings_by_network.at(network);
        std::sort(bindings.begin(), bindings.end(), [](const auto *lhs,
                                                       const auto *rhs) {
            return lhs->kids_tone < rhs->kids_tone;
        });
        if (bindings.size() != input.channel_count) {
            throw std::invalid_argument(
                "native raw detector count disagrees with the typed network relation");
        }
        const auto first_column = bindings.front()->detector_column;
        for (std::size_t channel = 0; channel < bindings.size(); ++channel) {
            if (bindings[channel]->kids_tone !=
                    static_cast<std::int64_t>(channel) ||
                bindings[channel]->detector_column !=
                    first_column + channel) {
                throw std::invalid_argument(
                    "native typed network relation is not a contiguous explicit channel/column mapping");
            }
        }
    }

    std::size_t output_row_count = 0;
    const auto operation = runs.front().selection.cohort().operation();
    for (std::size_t run_index = 0; run_index < runs.size(); ++run_index) {
        const auto &run = runs[run_index];
        require_complete_native_cohort(run.selection);
        const auto &cohort = run.selection.cohort();
        if (run.run_ordinal != run_index ||
            !(cohort.operation() == operation) ||
            run.first_common_slot >= run.past_last_common_slot ||
            run.past_last_common_slot - run.first_common_slot !=
                cohort.slot_count() ||
            run.participant_runs.size() != participant_networks.size() ||
            cohort.participant_network_ids() != participant_networks) {
            throw std::invalid_argument(
                "native detector run metadata disagrees with its complete cohort");
        }
        if (output_row_count >
            static_cast<std::size_t>(
                std::numeric_limits<Eigen::Index>::max()) -
                cohort.slot_count()) {
            throw std::overflow_error(
                "native detector output row count is not representable");
        }
        output_row_count += cohort.slot_count();
        for (std::size_t participant = 0;
             participant < participant_networks.size(); ++participant) {
            const auto &participant_run = run.participant_runs[participant];
            if (participant_run.network_id !=
                    participant_networks[participant] ||
                participant_run.row_count() !=
                    static_cast<TimestreamNativeRow>(cohort.slot_count())) {
                throw std::invalid_argument(
                    "native detector participant run has wrong identity or cardinality");
            }
            const auto &network = alignment.network(
                participant_run.network_id);
            for (std::size_t slot = 0; slot < cohort.slot_count(); ++slot) {
                const auto &cell = cohort.cell(slot, participant);
                const auto expected_row =
                    participant_run.first_native_row +
                    static_cast<TimestreamNativeRow>(slot);
                const auto expected_identity = network.identity(expected_row);
                if (!cell.identity().has_value() ||
                    !(*cell.identity() == expected_identity) ||
                    cell.expected_revision() != 0) {
                    throw std::invalid_argument(
                        "native detector cohort cell changed its delivered row, timestamp, or initial revision");
                }
            }
        }
    }

    const Eigen::Index n_rows =
        detail::checked_eigen_index(output_row_count, "native output rows");
    const Eigen::Index n_dets = detail::checked_eigen_index(
        relation.bindings().size(), "native detector columns");
    NativeMeasuredDetectorScan candidate;
    candidate.measured_values.resize(n_rows, n_dets);
    candidate.delivered_flag_bits =
        NativeDetectorFlagBitsMatrix::Zero(n_rows, n_dets);
    candidate.detector_binding_references = std::move(binding_references);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> assigned =
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>::Zero(
            n_rows, n_dets);

    Eigen::Index first_output_row = 0;
    for (const auto &run : runs) {
        const Eigen::Index run_rows = detail::checked_eigen_index(
            run.selection.cohort().slot_count(), "native run rows");
        for (std::size_t participant = 0;
             participant < participant_networks.size(); ++participant) {
            const auto network_id = participant_networks[participant];
            const auto &participant_run = run.participant_runs[participant];
            const auto &input = raw_by_network.at(network_id);
            const auto &bindings = bindings_by_network.at(network_id);

            const auto slice = tula::container_utils::Slice<int>{
                detail::checked_native_slice_index(
                    participant_run.first_native_row,
                    "native slice first row"),
                detail::checked_native_slice_index(
                    participant_run.past_last_native_row,
                    "native slice past-last row"),
                std::nullopt};
            auto raw_timestream = load_data_item(*input.data_item, slice);
            auto solver_result = this->solver()(
                raw_timestream, Solver::Config{});

            Eigen::MatrixXd solved;
            bool selected_channel = false;
            citlali::pipeline::visit_kids_tod_channel(
                solver_result, data_type, [&](const auto &channel) {
                    if (selected_channel) {
                        throw std::logic_error(
                            "native KIDs solver selected its TOD channel more than once");
                    }
                    selected_channel = true;
                    solved = channel;
                });
            if (!selected_channel || solved.rows() != run_rows ||
                solved.cols() !=
                    static_cast<Eigen::Index>(input.channel_count)) {
                throw std::runtime_error(
                    "native KIDs solver result does not match its exact delivered run/channel shape");
            }
            citlali::pipeline::require_finite_kids_input(
                solved, "measured native RTC KIDs input");

            const Eigen::Index first_detector_column =
                static_cast<Eigen::Index>(
                    bindings.front()->detector_column);
            Eigen::MatrixXd block_values(run_rows, solved.cols());
            for (Eigen::Index channel = 0; channel < solved.cols();
                 ++channel) {
                const auto &binding =
                    *bindings.at(static_cast<std::size_t>(channel));
                const Eigen::Index detector_column =
                    static_cast<Eigen::Index>(binding.detector_column);
                block_values.col(channel) =
                    solved.col(static_cast<Eigen::Index>(binding.kids_tone));
                candidate.measured_values
                    .block(first_output_row, detector_column, run_rows, 1) =
                    block_values.col(channel);
                assigned.block(
                    first_output_row, detector_column, run_rows, 1)
                    .setOnes();
            }
            auto block_flags = NativeDetectorFlagBitsMatrix::Zero(
                run_rows, solved.cols());
            candidate.measured_blocks.emplace_back(
                alignment.network(network_id),
                participant_run.first_native_row, first_detector_column,
                std::move(block_values), std::move(block_flags));
        }
        first_output_row += run_rows;
    }

    if (first_output_row != n_rows || !(assigned.array() == 1).all()) {
        throw std::logic_error(
            "native detector ingress did not assign every admitted measured cell exactly once");
    }
    citlali::pipeline::require_finite_kids_input(
        candidate.measured_values,
        "complete measured native RTC KIDs input");
    return candidate;
}
