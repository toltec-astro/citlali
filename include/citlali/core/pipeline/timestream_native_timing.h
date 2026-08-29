#pragma once

#include <citlali/core/pipeline/timestream_native_identity.h>

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace citlali::pipeline {

using TimestreamPacketCounter = std::int64_t;

// Observation identity is carried independently of the APT implementation so
// native timing remains a distinct authority. The observation owner binds it
// to the verified compact-v2 relation before publication.
struct NativeObservationScope {
    std::int64_t observation = 0;
    std::int64_t subobservation = 0;
    std::int64_t scan = 0;

    explicit NativeObservationScope(std::int64_t observation_,
                                    std::int64_t subobservation_,
                                    std::int64_t scan_)
        : observation{observation_}, subobservation{subobservation_},
          scan{scan_} {
        if (observation <= 0 || subobservation < 0 || scan < 0) {
            throw std::invalid_argument(
                "native observation scope is invalid");
        }
    }

    friend bool operator==(const NativeObservationScope &,
                           const NativeObservationScope &) = default;
};

// This transition records delivered packet-counter provenance only. It does
// not infer a duration, a rollover policy, or a missing detector sample.
struct NativeCounterDiscontinuity {
    TimestreamNativeRow before_native_row = -1;
    TimestreamNativeRow after_native_row = -1;
    TimestreamPacketCounter before_counter = 0;
    TimestreamPacketCounter after_counter = 0;

    friend bool operator==(const NativeCounterDiscontinuity &,
                           const NativeCounterDiscontinuity &) = default;
};

struct NativeRunBoundary {
    bool stream_boundary = false;
    bool scan_boundary = false;
    std::optional<NativeCounterDiscontinuity> counter_discontinuity;

    friend bool operator==(const NativeRunBoundary &,
                           const NativeRunBoundary &) = default;
};

struct NativeContiguousRun {
    TimestreamNetworkId network_id = -1;
    TimestreamNativeRow first_native_row = -1;
    TimestreamNativeRow past_last_native_row = -1;
    NativeRunBoundary boundary_before;
    NativeRunBoundary boundary_after;

    TimestreamNativeRow row_count() const noexcept {
        return past_last_native_row - first_native_row;
    }
};

inline bool packet_counters_are_contiguous(
    TimestreamPacketCounter before,
    TimestreamPacketCounter after) noexcept {
    return before != std::numeric_limits<TimestreamPacketCounter>::max() &&
           after == before + 1;
}

template <class TimeMatrix>
std::vector<TimestreamPacketCounter>
packet_counters_from_timestream_matrix(const TimeMatrix &timestamps) {
    if (timestamps.rows() <= 0 || timestamps.cols() <= 3) {
        throw std::invalid_argument(
            "native packet-counter source requires rows and column 3");
    }

    std::vector<TimestreamPacketCounter> result;
    result.reserve(static_cast<std::size_t>(timestamps.rows()));
    for (Eigen::Index row = 0; row < timestamps.rows(); ++row) {
        const long double value =
            static_cast<long double>(timestamps(row, 3));
        if (!std::isfinite(value) || std::floor(value) != value ||
            value < static_cast<long double>(
                        std::numeric_limits<TimestreamPacketCounter>::min()) ||
            value > static_cast<long double>(
                        std::numeric_limits<TimestreamPacketCounter>::max())) {
            throw std::invalid_argument(
                "native packet counter must be a finite representable integer");
        }
        result.push_back(static_cast<TimestreamPacketCounter>(value));
    }
    return result;
}

inline Eigen::VectorXd network_time_from_timestream_matrix(
    const Eigen::MatrixXd &ts_double, double fpga_freq,
    double interface_sync_offset) {
    auto sec = ts_double.col(0);
    auto nsec = ts_double.col(5);
    auto pps = ts_double.col(1);
    auto msec = ts_double.col(2) / fpga_freq;
    auto pps_msec = ts_double.col(4) / fpga_freq;

    double start_time_dbl = sec[0] + nsec[0] * 1e-9;
    const int start_time = int(start_time_dbl - 0.5);
    start_time_dbl = start_time;

    Eigen::VectorXd dt = msec - pps_msec;
    dt = (dt.array() < 0).select(
        msec.array() - pps_msec.array() +
            (std::pow(2.0, 32) - 1) / fpga_freq,
        msec - pps_msec);

    return start_time_dbl + pps.array() + dt.array() +
           interface_sync_offset;
}

class NativeNetworkAlignment {
public:
    NativeNetworkAlignment(
        TimestreamNetworkId network_id,
        TimestreamNativeRow first_native_row,
        Eigen::VectorXd reconstructed_times_unix_sec,
        std::vector<TimestreamPacketCounter> packet_counters)
        : network_id_{network_id}, first_native_row_{first_native_row},
          reconstructed_times_unix_sec_{
              std::move(reconstructed_times_unix_sec)},
          packet_counters_{std::move(packet_counters)} {
        if (network_id_ < 0 || first_native_row_ < 0) {
            throw std::invalid_argument(
                "native network alignment requires nonnegative identity");
        }
        if (reconstructed_times_unix_sec_.size() <= 0 ||
            static_cast<std::size_t>(
                reconstructed_times_unix_sec_.size()) !=
                packet_counters_.size()) {
            throw std::invalid_argument(
                "native times and counters require equal nonzero cardinality");
        }
        if (reconstructed_times_unix_sec_.size() >
            std::numeric_limits<TimestreamNativeRow>::max() -
                first_native_row_) {
            throw std::length_error("native row interval would overflow");
        }
        for (Eigen::Index row = 0;
             row < reconstructed_times_unix_sec_.size(); ++row) {
            const double value = reconstructed_times_unix_sec_(row);
            if (!std::isfinite(value) ||
                (row > 0 &&
                 !(value > reconstructed_times_unix_sec_(row - 1)))) {
                throw std::invalid_argument(
                    "native reconstructed times must be finite and strictly increasing");
            }
        }
    }

    TimestreamNetworkId network_id() const noexcept { return network_id_; }
    TimestreamNativeRow first_native_row() const noexcept {
        return first_native_row_;
    }
    TimestreamNativeRow past_last_native_row() const noexcept {
        return first_native_row_ + static_cast<TimestreamNativeRow>(
            reconstructed_times_unix_sec_.size());
    }
    Eigen::Index row_count() const noexcept {
        return reconstructed_times_unix_sec_.size();
    }
    const Eigen::VectorXd &reconstructed_times_unix_sec() const noexcept {
        return reconstructed_times_unix_sec_;
    }
    const std::vector<TimestreamPacketCounter> &packet_counters() const
        noexcept {
        return packet_counters_;
    }

    NativeSampleIdentity identity(TimestreamNativeRow native_row) const {
        const auto offset = checked_offset(native_row);
        return NativeSampleIdentity{
            network_id_, native_row,
            reconstructed_times_unix_sec_(offset)};
    }

    TimestreamPacketCounter packet_counter(
        TimestreamNativeRow native_row) const {
        return packet_counters_.at(
            static_cast<std::size_t>(checked_offset(native_row)));
    }

    std::optional<NativeCounterDiscontinuity> discontinuity_between(
        TimestreamNativeRow before_native_row,
        TimestreamNativeRow after_native_row) const {
        if (before_native_row ==
                std::numeric_limits<TimestreamNativeRow>::max() ||
            after_native_row != before_native_row + 1) {
            throw std::invalid_argument(
                "native discontinuity query requires adjacent delivered rows");
        }
        const auto before = packet_counter(before_native_row);
        const auto after = packet_counter(after_native_row);
        if (packet_counters_are_contiguous(before, after)) {
            return std::nullopt;
        }
        return NativeCounterDiscontinuity{
            before_native_row, after_native_row, before, after};
    }

private:
    Eigen::Index checked_offset(TimestreamNativeRow native_row) const {
        if (native_row < first_native_row_ ||
            native_row >= past_last_native_row()) {
            throw std::out_of_range(
                "native row is outside the delivered network alignment");
        }
        return static_cast<Eigen::Index>(native_row - first_native_row_);
    }

    TimestreamNetworkId network_id_;
    TimestreamNativeRow first_native_row_;
    Eigen::VectorXd reconstructed_times_unix_sec_;
    std::vector<TimestreamPacketCounter> packet_counters_;
};

template <class TimeMatrix>
NativeNetworkAlignment make_native_network_alignment(
    TimestreamNetworkId network_id,
    TimestreamNativeRow first_native_row,
    const TimeMatrix &timestamps,
    double fpga_frequency_hz,
    double interface_sync_offset_sec) {
    if (timestamps.rows() <= 0 || timestamps.cols() <= 5 ||
        !std::isfinite(fpga_frequency_hz) || fpga_frequency_hz <= 0.0 ||
        !std::isfinite(interface_sync_offset_sec)) {
        throw std::invalid_argument(
            "native timestamp reconstruction inputs are invalid");
    }
    for (Eigen::Index row = 0; row < timestamps.rows(); ++row) {
        for (Eigen::Index column : {Eigen::Index{0}, Eigen::Index{1},
                                    Eigen::Index{2}, Eigen::Index{4},
                                    Eigen::Index{5}}) {
            if (!std::isfinite(
                    static_cast<long double>(timestamps(row, column)))) {
                throw std::invalid_argument(
                    "native timestamp reconstruction source must be finite");
            }
        }
    }
    Eigen::MatrixXd source = timestamps.template cast<double>();
    auto reconstructed = network_time_from_timestream_matrix(
        source, fpga_frequency_hz, interface_sync_offset_sec);
    auto counters = packet_counters_from_timestream_matrix(timestamps);
    return NativeNetworkAlignment{
        network_id, first_native_row, std::move(reconstructed),
        std::move(counters)};
}

inline std::vector<NativeContiguousRun> partition_native_contiguous_runs(
    const NativeNetworkAlignment &network,
    TimestreamNativeRow first_native_row,
    TimestreamNativeRow past_last_native_row) {
    if (first_native_row < network.first_native_row() ||
        past_last_native_row > network.past_last_native_row() ||
        first_native_row >= past_last_native_row) {
        throw std::invalid_argument(
            "native scan row window must be a nonempty delivered interval");
    }

    auto boundary_before = [&](TimestreamNativeRow row) {
        NativeRunBoundary result;
        result.scan_boundary = row == first_native_row;
        result.stream_boundary = row == network.first_native_row();
        if (row > network.first_native_row()) {
            result.counter_discontinuity =
                network.discontinuity_between(row - 1, row);
        }
        return result;
    };
    auto boundary_after = [&](TimestreamNativeRow row) {
        NativeRunBoundary result;
        result.scan_boundary = row == past_last_native_row;
        result.stream_boundary = row == network.past_last_native_row();
        if (row < network.past_last_native_row()) {
            result.counter_discontinuity =
                network.discontinuity_between(row - 1, row);
        }
        return result;
    };

    std::vector<NativeContiguousRun> runs;
    TimestreamNativeRow run_begin = first_native_row;
    for (TimestreamNativeRow row = first_native_row + 1;
         row < past_last_native_row; ++row) {
        const auto discontinuity =
            network.discontinuity_between(row - 1, row);
        if (!discontinuity.has_value()) continue;
        auto before = boundary_before(run_begin);
        auto after = boundary_after(row);
        after.counter_discontinuity = discontinuity;
        runs.push_back(NativeContiguousRun{
            network.network_id(), run_begin, row,
            std::move(before), std::move(after)});
        run_begin = row;
    }
    auto before = boundary_before(run_begin);
    if (run_begin > first_native_row) {
        before.counter_discontinuity =
            network.discontinuity_between(run_begin - 1, run_begin);
    }
    runs.push_back(NativeContiguousRun{
        network.network_id(), run_begin, past_last_native_row,
        std::move(before), boundary_after(past_last_native_row)});
    return runs;
}

}  // namespace citlali::pipeline
