#pragma once

// Implementation detail included by kidsproc.h.

#include <citlali/core/engine/detail/kidsproc_gap_cardinality.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace citlali::engine_detail {

template <class TimeVector>
std::pair<Eigen::Index, Eigen::Index>
detector_source_slice_for_target_window(
    const TimeVector &time, double target_start, double target_stop,
    double tolerance, std::size_t stream_index, Eigen::Index scan) {
    if (time.size() == 0 || !std::isfinite(target_start) ||
        !std::isfinite(target_stop) || target_stop < target_start ||
        !std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::runtime_error(fmt::format(
            "invalid detector/source window for stream {} scan {}",
            stream_index, scan));
    }
    // Observation setup has already validated the complete detector-time
    // vector.  Scan generation must not rescan it once per stream and chunk.
    // Retain one native row on each side so observation-wide gaps crossing a
    // processing boundary keep their true bounding samples.
    const auto *begin = time.data();
    const auto *end = begin + time.size();
    const auto first_it = std::lower_bound(
        begin, end, target_start - tolerance);
    Eigen::Index first_at_or_after = first_it - begin;
    Eigen::Index source_start = first_at_or_after;
    if (source_start >= time.size()) {
        source_start = time.size() - 1;
    }
    else if (source_start > 0) {
        // Retain one left endpoint so a gap crossing the processing-window
        // boundary is classified from observation-wide support.
        --source_start;
    }

    const auto after_it = std::upper_bound(
        begin + source_start, end, target_stop + tolerance);
    const Eigen::Index first_after = after_it - begin;
    const Eigen::Index source_stop =
        first_after < time.size() ? first_after : time.size() - 1;
    if (source_stop < source_start) {
        throw std::logic_error("detector source slice is reversed");
    }
    return {source_start, source_stop};
}

}  // namespace citlali::engine_detail

template <typename DerivedA, typename DerivedB, typename DerivedC>
auto KidsDataProc::load_rawobs_gaps(const RawObs &rawobs, const Eigen::Index scan,
                                    Eigen::DenseBase<DerivedA>& scan_indices,
                                    std::vector<Eigen::Index>& /*legacy_start_indices*/,
                                    Eigen::DenseBase<DerivedB>& t_common,
                                    std::vector<DerivedC>& times,
                                    const double tol) {

    std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> result;

    if (scan < 0 || scan >= scan_indices.cols() ||
        scan_indices.rows() < 4) {
        throw std::runtime_error(
            "gap scan identity is outside the scan-index matrix");
    }
    if (scan_indices(2, scan) < 0 || scan_indices(3, scan) >= t_common.size() ||
        scan_indices(3, scan) < scan_indices(2, scan)) {
        throw std::runtime_error(fmt::format(
            "invalid gap scan time window for scan {}: start={} end={} t_common.size={}",
            scan, scan_indices(2, scan), scan_indices(3, scan), t_common.size()));
    }
    double t0 = t_common(scan_indices(2, scan));
    double t1 = t_common(scan_indices(3, scan));

    const auto kids_data = rawobs.kidsdata();
    // The gap path slices from authoritative detector-time vectors. Legacy
    // direct-path overlap offsets are neither consumed nor populated by the
    // union-lattice alignment, so they are not a gap-loader cardinality.
    citlali::engine_detail::require_gap_stream_cardinality(
        kids_data.size(), times.size());
    std::size_t stream_index = 0;
    for (const auto &data_item : kids_data) {
        if (stream_index >= times.size()) {
            throw std::runtime_error(
                "rawobs KIDs stream count exceeds time-vector count");
        }
        auto [i_start, i_end] =
            citlali::engine_detail::detector_source_slice_for_target_window(
                times[stream_index], t0, t1, tol,
                stream_index, scan);

        // get slice of data for current scan
        auto slice = tula::container_utils::Slice<int>{
            citlali::engine_detail::checked_kids_slice_index(
                i_start, "gap RTC slice start"),
            citlali::engine_detail::checked_kids_slice_index(
                i_end + 1, "gap RTC slice stop"),
            std::nullopt};
        result.push_back(load_data_item(data_item, slice));

        ++stream_index;
    }

    return result;
}

template <typename LoadedType, typename DerivedA, typename DerivedB,
          typename DerivedC, typename GapPermissions, typename DerivedD>
auto KidsDataProc::populate_rtc_gaps(LoadedType &loaded, Eigen::DenseBase<DerivedA>& t_common,
                                     std::vector<DerivedB>& times,
                                     std::vector<DerivedC>& masks,
                                     const GapPermissions& synthesize_gaps,
                                     const Eigen::Index scan,
                                     const double cadence,
                                     const double tol,
                                     Eigen::DenseBase<DerivedD>& scan_indices,
                                     const Eigen::Index n_pts,
                                     const Eigen::Index n_det,
                                     citlali::config::TodType data_type) {
    citlali::engine_detail::require_kids_matrix_dimensions(n_pts, n_det);
    if (loaded.size() != times.size() || loaded.size() != masks.size() ||
        loaded.size() != synthesize_gaps.size()) {
        throw std::runtime_error(
            "loaded KIDs, time, mask, and gap-plan cardinalities differ");
    }

    if (scan < 0 || scan >= scan_indices.cols() ||
        scan_indices.rows() < 4) {
        throw std::runtime_error(
            "gap scan identity is outside the scan-index matrix");
    }
    if (scan_indices(2, scan) < 0 || scan_indices(3, scan) >= t_common.size() ||
        scan_indices(3, scan) < scan_indices(2, scan)) {
        throw std::runtime_error(fmt::format(
            "invalid gap scan time window for scan {}: start={} end={} t_common.size={}",
            scan, scan_indices(2, scan), scan_indices(3, scan), t_common.size()));
    }
    if (scan_indices(3, scan) ==
            std::numeric_limits<Eigen::Index>::max() ||
        scan_indices(3, scan) - scan_indices(2, scan) + 1 != n_pts) {
        throw std::runtime_error(
            "gap RTC scan length conflicts with its admitted context window");
    }
    double t0 = t_common(scan_indices(2, scan));
    double t1 = t_common(scan_indices(3, scan));

    Eigen::MatrixXd data = Eigen::MatrixXd::Zero(n_pts, n_det);

    Eigen::Index i = 0, j = 0;
    // loop through raw timestream objects
    for (std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>::
         iterator it = loaded.begin(); it != loaded.end(); ++it) {
        // run the solver
        auto result = this->solver()(*it, Solver::Config{});
        Eigen::Index n_cols = 0;
        Eigen::MatrixXd block;
        citlali::pipeline::visit_kids_tod_channel(
            result, data_type, [&](const auto &channel) {
                n_cols = channel.cols();
                block = channel;
            });

        if (j >= static_cast<Eigen::Index>(times.size()) || j >= static_cast<Eigen::Index>(masks.size())) {
            throw std::runtime_error("loaded KIDs stream count exceeds time or mask vector count");
        }
        auto [i_start, i_end] =
            citlali::engine_detail::detector_source_slice_for_target_window(
                times[j], t0, t1, tol, j, scan);

        block = engine_utils::interp_data_with_observation_resolved_admission(
            t_common.segment(scan_indices(2, scan), n_pts),
            masks[j].segment(scan_indices(2, scan), n_pts),
            times[j].segment(i_start, i_end - i_start + 1), block,
            synthesize_gaps[static_cast<std::size_t>(j)] != 0,
            cadence, tol);

        if (block.rows() != n_pts || n_cols < 0 || i > n_det ||
            n_cols > n_det - i) {
            throw std::runtime_error(
                "gap-aligned solver shape exceeds configured matrix cardinality");
        }
        data.block(0, i, n_pts, n_cols) = block;
        // increment columns
        i += n_cols;
        j++;
    }

    if (i != n_det || j != static_cast<Eigen::Index>(times.size())) {
        throw std::runtime_error(
            "gap-aligned solver output does not match configured detector cardinality");
    }

    citlali::pipeline::require_finite_kids_input(
        data, "gap-aligned RTC KIDs input");

    return data;
}
