#pragma once

#include <citlali/core/pipeline/sci_align_contract.h>
#include <citlali/core/pipeline/timestream_scan_context.h>

#include <Eigen/Core>

#include <limits>
#include <optional>
#include <stdexcept>

namespace citlali::pipeline {

inline void require_consistent_gap_execution_grid(
    const AlignmentGridState &grid,
    double telescope_sample_frequency_hz) {
    if (!grid.initialized) {
        throw std::invalid_argument(
            "gap execution requires an initialized realized alignment grid");
    }
    sci_align::require_finite_positive(
        grid.cadence_sec, "realized alignment cadence");
    sci_align::require_finite_positive(
        grid.exclusive_half_cell_sec,
        "realized alignment exclusive half-cell");
    if (!sci_align::machine_equal(
            grid.exclusive_half_cell_sec, grid.cadence_sec / 2.0)) {
        throw std::invalid_argument(
            "realized alignment half-cell conflicts with its cadence");
    }
    sci_align::require_finite_positive(
        telescope_sample_frequency_hz,
        "telescope detector sample frequency");
    if (!sci_align::machine_equal(
            1.0 / telescope_sample_frequency_hz,
            grid.cadence_sec)) {
        throw std::invalid_argument(
            "telescope detector sample frequency conflicts with the realized alignment cadence");
    }
}

class ScanCursor {
public:
    explicit ScanCursor(Eigen::Index scan_count) : scan_count_(scan_count) {}

    std::optional<Eigen::Index> next() noexcept {
        if (next_scan_ >= scan_count_) {
            return std::nullopt;
        }
        return next_scan_++;
    }

private:
    Eigen::Index scan_count_ = 0;
    Eigen::Index next_scan_ = 0;
};

template <class RtcData, class Telescope>
Eigen::Index initialize_rtc_scan(
    RtcData &rtcdata, const Telescope &telescope, Eigen::Index scan) {
    if (scan < 0 || scan >= telescope.scan_indices.cols() ||
        telescope.scan_indices.rows() < 4) {
        throw std::runtime_error("RTC scan identity is outside the scan plan");
    }
    rtcdata.scan_indices.data = telescope.scan_indices.col(scan);
    rtcdata.index.data = scan;
    const Eigen::Index start = rtcdata.scan_indices.data(2);
    const Eigen::Index stop = rtcdata.scan_indices.data(3);
    if (start < 0 || stop < start ||
        stop == std::numeric_limits<Eigen::Index>::max()) {
        throw std::runtime_error("RTC scan context window is invalid");
    }
    return stop - start + 1;
}

template <class RtcData, class MapBuffer, class Calib,
          class RandomDistribution, class RandomEngine>
void populate_noise_map_signs(
    RtcData &rtcdata, const MapBuffer &omb, const Calib &calib,
    bool enabled, RandomDistribution &rands, RandomEngine &eng) {
    if (!enabled) {
        return;
    }
    if (omb.n_noise <= 0 || calib.n_dets < 0 ||
        (omb.randomize_dets && calib.n_dets != 0 &&
         omb.n_noise >
             std::numeric_limits<Eigen::Index>::max() /
                 calib.n_dets)) {
        throw std::overflow_error(
            "noise-map sign dimensions are invalid or exceed the Eigen index range");
    }

    if (omb.randomize_dets) {
        rtcdata.noise.data =
            Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                omb.n_noise, calib.n_dets)
                .unaryExpr([&](int) { return 2 * rands(eng) - 1; });
    }
    else {
        rtcdata.noise.data =
            Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(omb.n_noise)
                .unaryExpr([&](int) { return 2 * rands(eng) - 1; });
    }
}

template <class RtcData, class KidsProc, class RawObs, class Telescope,
          class Alignment, class TimestreamType>
void populate_rtc_scan_samples(
    RtcData &rtcdata, KidsProc &kidsproc, RawObs &rawobs, Eigen::Index scan,
    Telescope &telescope, Alignment &alignment,
    bool interp_over_gaps, Eigen::Index scan_length, Eigen::Index n_dets,
    TimestreamType timestream_type) {
    if (!interp_over_gaps) {
        rtcdata.scans.data = kidsproc.populate_rtc_from_rawobs(
            rawobs, scan, telescope.scan_indices, alignment.start_indices,
            alignment.end_indices,
            scan_length, n_dets, timestream_type);
        return;
    }

    require_consistent_gap_execution_grid(
        alignment.grid, telescope.fsmp);
    const double realized_cadence = alignment.grid.cadence_sec;
    const double gap_tolerance =
        alignment.grid.exclusive_half_cell_sec;
    auto scan_rawobs = kidsproc.load_rawobs_gaps(
        rawobs, scan, telescope.scan_indices, alignment.start_indices,
        alignment.common_time, alignment.network_times, gap_tolerance);
    const auto synthesize_gaps = alignment_gap_synthesis_permissions(
        alignment, scan);
    rtcdata.scans.data = kidsproc.populate_rtc_gaps(
        scan_rawobs, alignment.common_time, alignment.network_times,
        alignment.masks, synthesize_gaps, scan, realized_cadence,
        gap_tolerance,
        telescope.scan_indices, scan_length, n_dets, timestream_type);
    decltype(scan_rawobs)().swap(scan_rawobs);
}

}  // namespace citlali::pipeline
