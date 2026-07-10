#pragma once

#include <Eigen/Core>

#include <optional>

namespace citlali::pipeline {

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
    rtcdata.scan_indices.data = telescope.scan_indices.col(scan);
    rtcdata.index.data = scan;
    return rtcdata.scan_indices.data(3) - rtcdata.scan_indices.data(2) + 1;
}

template <class RtcData, class MapBuffer, class Calib,
          class RandomDistribution, class RandomEngine>
void populate_noise_map_signs(
    RtcData &rtcdata, const MapBuffer &omb, const Calib &calib,
    bool enabled, RandomDistribution &rands, RandomEngine &eng) {
    if (!enabled) {
        return;
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
          class StartIndices, class EndIndices, class TCommon, class NwTimes,
          class Masks, class TimestreamType>
void populate_rtc_scan_samples(
    RtcData &rtcdata, KidsProc &kidsproc, RawObs &rawobs, Eigen::Index scan,
    Telescope &telescope, StartIndices &start_indices, EndIndices &end_indices,
    TCommon &t_common, NwTimes &nw_times, Masks &masks,
    bool interp_over_gaps, int scan_length, int n_dets,
    TimestreamType timestream_type) {
    if (!interp_over_gaps) {
        rtcdata.scans.data = kidsproc.populate_rtc_from_rawobs(
            rawobs, scan, telescope.scan_indices, start_indices, end_indices,
            scan_length, n_dets, timestream_type);
        return;
    }

    const double gap_tolerance = 1 / (2 * telescope.fsmp);
    auto scan_rawobs = kidsproc.load_rawobs_gaps(
        rawobs, scan, telescope.scan_indices, start_indices, t_common,
        nw_times, gap_tolerance);
    rtcdata.scans.data = kidsproc.populate_rtc_gaps(
        scan_rawobs, t_common, nw_times, masks, scan, gap_tolerance,
        telescope.scan_indices, scan_length, n_dets, timestream_type);
    decltype(scan_rawobs)().swap(scan_rawobs);
}

}  // namespace citlali::pipeline
