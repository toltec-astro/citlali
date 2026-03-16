#pragma once

#include <boost/random.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <Eigen/Core>

#include <tula/logging.h>

#include <tula/algorithm/mlinterp/mlinterp.hpp>
#include <citlali/core/utils/utils.h>

namespace timestream {

struct DespikeDetectorDiagSummary {
    int raw_exceed_count = 0;
    int local_raw_candidate_count = 0;
    int local_exceed_count = 0;
    int local_raw_reject_count = 0;
    int delta_spike_count = 0;
    int local_delta_candidate_count = 0;
    int local_delta_exceed_count = 0;
    int local_delta_reject_count = 0;
    double added_flagged_frac = std::numeric_limits<double>::quiet_NaN();
    int added_region_count = 0;
    double added_region_len_median = std::numeric_limits<double>::quiet_NaN();
    int added_region_len_max = 0;
    double max_raw_abs_z = std::numeric_limits<double>::quiet_NaN();
    double max_local_abs_z = std::numeric_limits<double>::quiet_NaN();
    double max_delta_abs_z = std::numeric_limits<double>::quiet_NaN();
    double max_local_delta_abs_z = std::numeric_limits<double>::quiet_NaN();
};

class Despiker {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // spike sigma, time constant, sample rate
    double min_spike_sigma, time_constant_sec, window_size;
    double fsmp;

    struct LocalResidualOptions {
        struct CompactRawGateOptions {
            bool enabled = true;
            double window_sec = 0.18;
            double half_peak_frac = 0.5;
            double max_width_sec = 0.18;
            double max_step_shift_z = 3.0;
        };
        struct CompactDeltaGateOptions {
            bool enabled = true;
            double window_sec = 0.12;
            double half_peak_frac = 0.5;
            double max_width_sec = 0.10;
            double max_step_shift_z = 3.0;
        };
        bool enabled = false;
        double window_sec = 0.25;
        double sigma_scale = 0.75;
        double delta_sigma_scale = 0.75;
        CompactRawGateOptions compact_raw_gate;
        CompactDeltaGateOptions compact_delta_gate;
    };
    LocalResidualOptions local_residual;

    // for window size
    bool run_filter = false;

    // size of region to merge flags (samples)
    int size = 10;
    // maximum window length (seconds) for sigma estimation
    double max_window_sec = 10.0;

    // standard deviation limit
    double sigest_lim = 0;//1e-8;

    // grouping for replacing spikes
    std::string grouping;

    // use all detectors in replacing flagged scans
    bool use_all_det = false;

    // scan-level detector summaries from the most recent despike call
    std::vector<DespikeDetectorDiagSummary> last_detector_diag;

    // the main despiking routine
    template <typename DerivedA, typename DerivedB, typename apt_t>
    void despike(Eigen::DenseBase<DerivedA> &, Eigen::DenseBase<DerivedB> &, apt_t &);

    // replace flagged data with interpolation
    template<typename DerivedA, typename DerivedB, typename apt_t>
    void replace_spikes(Eigen::DenseBase<DerivedA> &, Eigen::DenseBase<DerivedB>&, apt_t &,
                        Eigen::Index);

private:
    // this function loops through the delta timestream values and finds the
    // spikes setting the corresponding flags to zero and calculating n_spikes
    template <typename DerivedA, typename DerivedB, typename DerivedC>
    void spike_finder(Eigen::DenseBase<DerivedA> &flags,
                      Eigen::DenseBase<DerivedB> &delta,
                      Eigen::DenseBase<DerivedC> &diff, int &n_spikes,
                      double &cutoff) {

        Eigen::Index n_pts = flags.size();

        // set flag matrix to zero if scan delta is above the cutoff
        flags.segment(1,n_pts - 2) =
            (diff.segment(1,n_pts - 2).array() > cutoff).select(1, flags.segment(1,n_pts - 2));

        // set corresponding delta values to zero
        delta.segment(1,n_pts - 2) =
            (diff.segment(1,n_pts - 2).array() > cutoff).select(0, delta.segment(1,n_pts - 2));

        // update difference vector and cutoff
        diff = abs(delta.derived().array() - delta.mean());
        cutoff = min_spike_sigma * engine_utils::calc_std_dev(delta);
    }

    template <typename Derived>
    auto make_window(Eigen::DenseBase<Derived> &spike_loc, int n_spikes,
                     Eigen::Index n_pts) {
        // window first index, last index, and size
        int win_index_0 = 0;
        int win_index_1 = static_cast<int>(n_pts);
        int win_size = 0;
        const int n_pts_i = static_cast<int>(n_pts);
        const int fsmp_i = std::max(0, static_cast<int>(std::lround(fsmp)));

        // find largest unpadded interval between spikes/edges for fallback
        auto set_fallback_window = [&]() {
            int best_0 = 0;
            int best_1 = n_pts_i;
            int best_len = -1;
            int prev_spike = -1;
            for (int i = 0; i < n_spikes; ++i) {
                int s = std::clamp(spike_loc(i), 0, n_pts_i - 1);
                int cand_0 = prev_spike + 1;
                int cand_1 = s;
                int cand_len = cand_1 - cand_0;
                if (cand_len > best_len) {
                    best_len = cand_len;
                    best_0 = cand_0;
                    best_1 = cand_1;
                }
                prev_spike = s;
            }
            int cand_0 = prev_spike + 1;
            int cand_1 = n_pts_i;
            int cand_len = cand_1 - cand_0;
            if (cand_len > best_len) {
                best_0 = cand_0;
                best_1 = cand_1;
            }
            win_index_0 = std::clamp(best_0, 0, n_pts_i);
            win_index_1 = std::clamp(best_1, 0, n_pts_i);
        };

        // find biggest spike-free window
        // first deal with the harder case of multiple spikes
        // remember that there are n_spikes + 1 possible windows
        // only do if more than one spike
        if (n_spikes > 1) {
            Eigen::Matrix<int, Eigen::Dynamic, 1> delta_spike_loc(n_spikes + 1);
            // first element is the distance to first spike
            delta_spike_loc(0) = spike_loc(0);
            // last element is the distance to last spike
            delta_spike_loc(n_spikes) = n_pts - spike_loc(n_spikes - 1);
            // populate the delta spike location vector
            delta_spike_loc.segment(1,n_spikes - 1) =
                spike_loc.tail(n_spikes - 1) - spike_loc.head(n_spikes - 1);

            // get the maximum and index of the delta spike locations
            Eigen::Index mx_window_index;
            delta_spike_loc.maxCoeff(&mx_window_index);

            logger->trace("delta_spike_loc {} fsmp {} mx_window_index {} n_spikes {} n_pts {}",
                         delta_spike_loc, fsmp, mx_window_index, n_spikes, n_pts);
            if (mx_window_index == 0) {
                logger->trace("spike_loc(0) {}", spike_loc(0));
            }
            else if (mx_window_index == n_spikes) {
                logger->trace("spike_loc(n_spikes-1) {}", spike_loc(n_spikes - 1));
            }
            else {
                logger->trace("spike_loc(mx_window_index) {} spike_loc(mx_window_index-1) {}",
                             spike_loc(mx_window_index), spike_loc(mx_window_index - 1));
            }

            // set the starting and ending indices for the window
            if (mx_window_index == 0) {
                win_index_0 = 0;
                win_index_1 = spike_loc(0);// - fsmp;
            }
            else {
                if (mx_window_index == n_spikes) {
                    win_index_0 = spike_loc(n_spikes - 1);
                    win_index_1 = n_pts;
                }
                // leave a 2 second region after the spike beginning the
                // window and a 1 second region before the spike ending the window
                else {
                    win_index_0 = spike_loc(mx_window_index - 1);// + 2 * fsmp;
                    win_index_1 = spike_loc(mx_window_index);// - fsmp;
                }
            }
        }
        else {
            if (n_pts - spike_loc(0) > spike_loc(0)) {
                win_index_0 = spike_loc(0) + 2 * fsmp_i;
                win_index_1 = n_pts - 1;
            }
            else {
                win_index_0 = 0;
                win_index_1 = spike_loc(0) - fsmp_i;
            }
        }

        // guard indices before use
        win_index_0 = std::clamp(win_index_0, 0, n_pts_i);
        win_index_1 = std::clamp(win_index_1, 0, n_pts_i);
        if (win_index_1 - win_index_0 <= 1) {
            logger->warn("despike make_window produced degenerate window; using fallback");
            set_fallback_window();
        }

        logger->trace("win_index_0 {}", win_index_0);
        logger->trace("win_index_1 {}", win_index_1);

        // limit the maximum window size
        if (max_window_sec > 0) {
            int max_window_samples = static_cast<int>(max_window_sec * fsmp);
            if (max_window_samples > 0 && (win_index_1 - win_index_0 - 1) > max_window_samples) {
                win_index_1 = win_index_0 + max_window_samples + 1;
            }
        }

        win_size = win_index_1 - win_index_0 - 1;

        if (win_size <= 0) {
            logger->warn("despike make_window still invalid (scan size {}, spike_loc {}); using full-scan fallback",
                         n_pts, spike_loc.derived());
            win_index_0 = 0;
            win_index_1 = n_pts_i;
            win_size = std::max(1, n_pts_i - 1);
        }

        return std::tuple<int, int, int>(win_index_0, win_index_1, win_size);
    }
};

template <typename DerivedA, typename DerivedB, typename apt_t>
void Despiker::despike(Eigen::DenseBase<DerivedA> &scans,
                       Eigen::DenseBase<DerivedB> &flags,
                       apt_t &apt) {
    Eigen::Index n_pts = scans.rows();
    Eigen::Index n_dets = scans.cols();

    auto robust_center_scale = [&](const Eigen::VectorXd &x,
                                   const Eigen::Matrix<bool, Eigen::Dynamic, 1> &flag_mask) {
        std::vector<double> vals;
        vals.reserve(static_cast<std::size_t>(x.size()));
        for (Eigen::Index i = 0; i < x.size(); ++i) {
            if (!flag_mask(i) && std::isfinite(x(i))) {
                vals.push_back(x(i));
            }
        }
        if (vals.size() < 8) {
            vals.clear();
            vals.reserve(static_cast<std::size_t>(x.size()));
            for (Eigen::Index i = 0; i < x.size(); ++i) {
                if (std::isfinite(x(i))) {
                    vals.push_back(x(i));
                }
            }
        }
        if (vals.size() < 8) {
            return std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN());
        }
        Eigen::Map<const Eigen::VectorXd> vals_map(vals.data(), static_cast<Eigen::Index>(vals.size()));
        const double med = tula::alg::median(vals_map);
        Eigen::VectorXd abs_dev = (vals_map.array() - med).abs();
        double sigma = 1.4826 * tula::alg::median(abs_dev);
        if (sigma < sigest_lim) {
            sigma = sigest_lim;
        }
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            return std::make_pair(med, std::numeric_limits<double>::quiet_NaN());
        }
        return std::make_pair(med, sigma);
    };

    auto shape_gate_local_raw = [&](const Eigen::VectorXd &resid,
                                    const Eigen::VectorXd &abs_z,
                                    const Eigen::Matrix<bool, Eigen::Dynamic, 1> &base_flags,
                                    Eigen::Index peak_sample,
                                    double resid_sigma) {
        const auto &gate = local_residual.compact_raw_gate;
        if (!gate.enabled) {
            return true;
        }
        if (!(std::isfinite(resid_sigma) && resid_sigma > 0.0) ||
            peak_sample < 0 || peak_sample >= abs_z.size()) {
            return false;
        }

        const Eigen::Index n = resid.size();
        const double peak_z = abs_z(peak_sample);
        if (!std::isfinite(peak_z) || peak_z <= 0.0) {
            return false;
        }

        const Eigen::Index gate_half_window = std::max<Eigen::Index>(
            4, static_cast<Eigen::Index>(std::llround(gate.window_sec * fsmp)));
        const Eigen::Index max_width_samples = std::max<Eigen::Index>(
            1, static_cast<Eigen::Index>(std::llround(gate.max_width_sec * fsmp)));
        const double width_thresh =
            std::max(gate.half_peak_frac * peak_z, std::min(peak_z, 1.5));
        const Eigen::Index left_bound = std::max<Eigen::Index>(0, peak_sample - gate_half_window);
        const Eigen::Index right_bound = std::min<Eigen::Index>(n - 1, peak_sample + gate_half_window);

        Eigen::Index left = peak_sample;
        while (left - 1 >= left_bound &&
               std::isfinite(abs_z(left - 1)) &&
               abs_z(left - 1) >= width_thresh) {
            --left;
        }
        Eigen::Index right = peak_sample;
        while (right + 1 <= right_bound &&
               std::isfinite(abs_z(right + 1)) &&
               abs_z(right + 1) >= width_thresh) {
            ++right;
        }
        const Eigen::Index width_samples = right - left + 1;
        if (width_samples > max_width_samples) {
            return false;
        }

        const Eigen::Index pre_lo = std::max<Eigen::Index>(0, peak_sample - gate_half_window);
        const Eigen::Index pre_hi = std::max<Eigen::Index>(pre_lo, peak_sample - 1);
        const Eigen::Index post_lo = std::min<Eigen::Index>(n, peak_sample + 2);
        const Eigen::Index post_hi = std::min<Eigen::Index>(n, peak_sample + gate_half_window + 1);
        std::vector<double> pre_vals;
        std::vector<double> post_vals;
        pre_vals.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(pre_hi - pre_lo, 0)));
        post_vals.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(post_hi - post_lo, 0)));
        for (Eigen::Index i = pre_lo; i < pre_hi; ++i) {
            if (!base_flags(i) && std::isfinite(resid(i))) {
                pre_vals.push_back(resid(i));
            }
        }
        for (Eigen::Index i = post_lo; i < post_hi; ++i) {
            if (!base_flags(i) && std::isfinite(resid(i))) {
                post_vals.push_back(resid(i));
            }
        }
        if (pre_vals.size() < 4 || post_vals.size() < 4) {
            return false;
        }
        Eigen::Map<const Eigen::VectorXd> pre_map(pre_vals.data(), static_cast<Eigen::Index>(pre_vals.size()));
        Eigen::Map<const Eigen::VectorXd> post_map(post_vals.data(), static_cast<Eigen::Index>(post_vals.size()));
        const double pre_med = tula::alg::median(pre_map);
        const double post_med = tula::alg::median(post_map);
        const double step_shift_z = std::abs(post_med - pre_med) / resid_sigma;
        return std::isfinite(step_shift_z) && step_shift_z <= gate.max_step_shift_z;
    };

    auto shape_gate_local_delta = [&](const Eigen::VectorXd &resid,
                                      const Eigen::VectorXd &delta_abs_z,
                                      const Eigen::Matrix<bool, Eigen::Dynamic, 1> &base_flags,
                                      Eigen::Index peak_edge,
                                      double resid_med,
                                      double resid_sigma) {
        const auto &gate = local_residual.compact_delta_gate;
        if (!gate.enabled) {
            return true;
        }
        if (!(std::isfinite(resid_med) && std::isfinite(resid_sigma) && resid_sigma > 0.0) ||
            peak_edge < 0 || peak_edge >= delta_abs_z.size()) {
            return false;
        }

        const Eigen::Index n = resid.size();
        const Eigen::Index peak_sample = peak_edge + 1;
        const double peak_delta_z = delta_abs_z(peak_edge);
        if (!std::isfinite(peak_delta_z) || peak_delta_z <= 0.0) {
            return false;
        }

        const Eigen::Index gate_half_window = std::max<Eigen::Index>(
            4, static_cast<Eigen::Index>(std::llround(gate.window_sec * fsmp)));
        const Eigen::Index max_width_edges = std::max<Eigen::Index>(
            1, static_cast<Eigen::Index>(std::llround(gate.max_width_sec * fsmp)));
        const double width_thresh =
            std::max(gate.half_peak_frac * peak_delta_z, std::min(peak_delta_z, 1.5));
        const Eigen::Index left_bound = std::max<Eigen::Index>(0, peak_edge - gate_half_window);
        const Eigen::Index right_bound =
            std::min<Eigen::Index>(delta_abs_z.size() - 1, peak_edge + gate_half_window);

        Eigen::Index left = peak_edge;
        while (left - 1 >= left_bound &&
               std::isfinite(delta_abs_z(left - 1)) &&
               delta_abs_z(left - 1) >= width_thresh) {
            --left;
        }
        Eigen::Index right = peak_edge;
        while (right + 1 <= right_bound &&
               std::isfinite(delta_abs_z(right + 1)) &&
               delta_abs_z(right + 1) >= width_thresh) {
            ++right;
        }
        const Eigen::Index width_edges = right - left + 1;
        if (width_edges > max_width_edges) {
            return false;
        }

        const Eigen::Index pre_lo = std::max<Eigen::Index>(0, peak_sample - gate_half_window);
        const Eigen::Index pre_hi = std::max<Eigen::Index>(pre_lo, peak_sample - 2);
        const Eigen::Index post_lo = std::min<Eigen::Index>(n, peak_sample + 2);
        const Eigen::Index post_hi = std::min<Eigen::Index>(n, peak_sample + gate_half_window + 1);
        std::vector<double> pre_vals;
        std::vector<double> post_vals;
        pre_vals.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(pre_hi - pre_lo, 0)));
        post_vals.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(post_hi - post_lo, 0)));
        for (Eigen::Index i = pre_lo; i < pre_hi; ++i) {
            if (!base_flags(i) && std::isfinite(resid(i))) {
                pre_vals.push_back(resid(i));
            }
        }
        for (Eigen::Index i = post_lo; i < post_hi; ++i) {
            if (!base_flags(i) && std::isfinite(resid(i))) {
                post_vals.push_back(resid(i));
            }
        }
        if (pre_vals.size() < 4 || post_vals.size() < 4) {
            return false;
        }
        Eigen::Map<const Eigen::VectorXd> pre_map(pre_vals.data(), static_cast<Eigen::Index>(pre_vals.size()));
        Eigen::Map<const Eigen::VectorXd> post_map(post_vals.data(), static_cast<Eigen::Index>(post_vals.size()));
        const double pre_med = tula::alg::median(pre_map);
        const double post_med = tula::alg::median(post_map);
        const double step_shift_z = std::abs(post_med - pre_med) / resid_sigma;
        return std::isfinite(step_shift_z) && step_shift_z <= gate.max_step_shift_z;
    };

    last_detector_diag.assign(static_cast<std::size_t>(n_dets), DespikeDetectorDiagSummary{});

    if (n_pts < 3) {
        logger->warn("despike skipped: too few samples in scan (n_pts={})", n_pts);
        return;
    }

    // loop through detectors
    for (Eigen::Index det=0; det<n_dets; det++) {
        auto &diag = last_detector_diag[static_cast<std::size_t>(det)];
        // only run if detector is good
        if (apt["flag"](det)==0) {
            // get detector's flags
            Eigen::Matrix<bool, Eigen::Dynamic, 1> det_flags = flags.col(det);
            Eigen::Matrix<bool, Eigen::Dynamic, 1> base_flags = det_flags;
            Eigen::Matrix<bool, Eigen::Dynamic, 1> raw_flags =
                Eigen::Matrix<bool, Eigen::Dynamic, 1>::Zero(n_pts);
            Eigen::Matrix<bool, Eigen::Dynamic, 1> local_flags =
                Eigen::Matrix<bool, Eigen::Dynamic, 1>::Zero(n_pts);

            // total number of spikes
            int n_spikes = 0;

            // also flag single-sample outliers in the raw signal (robust MAD)
            {
                Eigen::VectorXd raw = scans.col(det);
                auto [med, sigma] = robust_center_scale(raw, base_flags);
                if (std::isfinite(med) && std::isfinite(sigma) && sigma > 0.0) {
                    Eigen::VectorXd abs_dev = (raw.array() - med).abs();
                    double raw_cutoff = min_spike_sigma * sigma;
                    for (Eigen::Index i = 0; i < n_pts; ++i) {
                        if (!base_flags(i) && std::isfinite(abs_dev(i)) && abs_dev(i) > raw_cutoff) {
                            raw_flags(i) = 1;
                        }
                    }
                    diag.raw_exceed_count = static_cast<int>((raw_flags.array() == 1).count());
                    diag.max_raw_abs_z = abs_dev.maxCoeff() / sigma;
                }
            }

            if (local_residual.enabled) {
                Eigen::VectorXd raw = scans.col(det);
                auto [med, sigma] = robust_center_scale(raw, base_flags);
                if (std::isfinite(med) && std::isfinite(sigma) && sigma > 0.0) {
                    int smooth_window = static_cast<int>(std::lround(local_residual.window_sec * fsmp));
                    const int max_window = static_cast<int>((n_pts % 2 == 0) ? (n_pts - 1) : n_pts);
                    smooth_window = std::max(3, smooth_window);
                    smooth_window = std::min(smooth_window, std::max(3, max_window));
                    if ((smooth_window % 2) == 0) {
                        --smooth_window;
                    }
                    if (smooth_window >= 3 && smooth_window <= max_window) {
                        Eigen::VectorXd baseline_input = raw;
                        for (Eigen::Index i = 0; i < n_pts; ++i) {
                            if (base_flags(i) || !std::isfinite(baseline_input(i))) {
                                baseline_input(i) = med;
                            }
                        }
                        Eigen::VectorXd smooth = Eigen::VectorXd::Zero(n_pts);
                        engine_utils::smooth<engine_utils::SmoothType::edge_truncate>(
                            baseline_input, smooth, smooth_window);
                        Eigen::VectorXd resid = raw - smooth;
                        auto [resid_med, resid_sigma] = robust_center_scale(resid, base_flags);
                        if (std::isfinite(resid_med) && std::isfinite(resid_sigma) && resid_sigma > 0.0) {
                            const double local_cutoff =
                                local_residual.sigma_scale * min_spike_sigma * resid_sigma;
                            Eigen::VectorXd abs_dev = (resid.array() - resid_med).abs();
                            diag.max_local_abs_z = abs_dev.maxCoeff() / resid_sigma;
                            if (local_residual.compact_raw_gate.enabled) {
                                std::vector<Eigen::Index> candidate_samples;
                                candidate_samples.reserve(static_cast<std::size_t>(n_pts));
                                Eigen::VectorXd local_abs_z =
                                    Eigen::VectorXd::Constant(n_pts, std::numeric_limits<double>::quiet_NaN());
                                for (Eigen::Index i = 0; i < n_pts; ++i) {
                                    if (base_flags(i) || raw_flags(i) || !std::isfinite(abs_dev(i))) {
                                        continue;
                                    }
                                    local_abs_z(i) = abs_dev(i) / resid_sigma;
                                    if (abs_dev(i) > local_cutoff) {
                                        candidate_samples.push_back(i);
                                    }
                                }
                                if (!candidate_samples.empty()) {
                                    Eigen::Index cluster_start = candidate_samples.front();
                                    Eigen::Index cluster_end = candidate_samples.front();
                                    auto flush_cluster = [&](Eigen::Index lo, Eigen::Index hi) {
                                        Eigen::Index best_sample = lo;
                                        double best_z = -1.0;
                                        for (Eigen::Index sample = lo; sample <= hi; ++sample) {
                                            if (sample >= 0 && sample < local_abs_z.size() &&
                                                std::isfinite(local_abs_z(sample)) &&
                                                local_abs_z(sample) > best_z) {
                                                best_z = local_abs_z(sample);
                                                best_sample = sample;
                                            }
                                        }
                                        ++diag.local_raw_candidate_count;
                                        if (shape_gate_local_raw(
                                                resid, local_abs_z, base_flags,
                                                best_sample, resid_sigma)) {
                                            local_flags.segment(lo, hi - lo + 1).setOnes();
                                        }
                                        else {
                                            ++diag.local_raw_reject_count;
                                        }
                                    };
                                    for (std::size_t ii = 1; ii < candidate_samples.size(); ++ii) {
                                        const auto sample = candidate_samples[ii];
                                        if (sample <= cluster_end + 1) {
                                            cluster_end = sample;
                                        }
                                        else {
                                            flush_cluster(cluster_start, cluster_end);
                                            cluster_start = sample;
                                            cluster_end = sample;
                                        }
                                    }
                                    flush_cluster(cluster_start, cluster_end);
                                }
                            }
                            else {
                                for (Eigen::Index i = 0; i < n_pts; ++i) {
                                    if (!(base_flags(i) || raw_flags(i)) &&
                                        std::isfinite(abs_dev(i)) &&
                                        abs_dev(i) > local_cutoff) {
                                        local_flags(i) = 1;
                                    }
                                }
                            }
                            diag.local_exceed_count =
                                static_cast<int>((local_flags.array() == 1).count());

                            std::vector<double> local_delta_vals;
                            local_delta_vals.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(n_pts - 1, 0)));
                            for (Eigen::Index i = 0; i < n_pts - 1; ++i) {
                                if (!(base_flags(i) || base_flags(i + 1) || raw_flags(i) ||
                                      raw_flags(i + 1) || local_flags(i) || local_flags(i + 1)) &&
                                    std::isfinite(resid(i)) && std::isfinite(resid(i + 1))) {
                                    local_delta_vals.push_back(resid(i + 1) - resid(i));
                                }
                            }
                            if (local_delta_vals.size() >= 8) {
                                Eigen::Map<const Eigen::VectorXd> delta_map(
                                    local_delta_vals.data(),
                                    static_cast<Eigen::Index>(local_delta_vals.size()));
                                const double delta_med = tula::alg::median(delta_map);
                                Eigen::VectorXd delta_abs_dev = (delta_map.array() - delta_med).abs();
                                double delta_sigma = 1.4826 * tula::alg::median(delta_abs_dev);
                                if (delta_sigma < sigest_lim) {
                                    delta_sigma = sigest_lim;
                                }
                                if (std::isfinite(delta_sigma) && delta_sigma > 0.0) {
                                    const double local_delta_cutoff =
                                        local_residual.delta_sigma_scale * min_spike_sigma * delta_sigma;
                                    double max_local_delta_abs = 0.0;
                                    std::vector<Eigen::Index> candidate_edges;
                                    candidate_edges.reserve(static_cast<std::size_t>(n_pts));
                                    Eigen::VectorXd local_delta_abs_z =
                                        Eigen::VectorXd::Constant(std::max<Eigen::Index>(n_pts - 1, 0),
                                                                  std::numeric_limits<double>::quiet_NaN());
                                    for (Eigen::Index i = 0; i < n_pts - 1; ++i) {
                                        if (base_flags(i) || base_flags(i + 1) || raw_flags(i) ||
                                            raw_flags(i + 1) || !std::isfinite(resid(i)) ||
                                            !std::isfinite(resid(i + 1))) {
                                            continue;
                                        }
                                        const double abs_delta =
                                            std::abs((resid(i + 1) - resid(i)) - delta_med);
                                        max_local_delta_abs = std::max(max_local_delta_abs, abs_delta);
                                        local_delta_abs_z(i) = abs_delta / delta_sigma;
                                        if (abs_delta > local_delta_cutoff) {
                                            candidate_edges.push_back(i);
                                        }
                                    }
                                    diag.max_local_delta_abs_z = max_local_delta_abs / delta_sigma;
                                    if (!candidate_edges.empty()) {
                                        Eigen::Index cluster_start = candidate_edges.front();
                                        Eigen::Index cluster_end = candidate_edges.front();
                                        auto flush_cluster = [&](Eigen::Index lo, Eigen::Index hi) {
                                            Eigen::Index best_edge = lo;
                                            double best_z = -1.0;
                                            for (Eigen::Index edge = lo; edge <= hi; ++edge) {
                                                if (edge >= 0 && edge < local_delta_abs_z.size() &&
                                                    std::isfinite(local_delta_abs_z(edge)) &&
                                                    local_delta_abs_z(edge) > best_z) {
                                                    best_z = local_delta_abs_z(edge);
                                                    best_edge = edge;
                                                }
                                            }
                                            ++diag.local_delta_candidate_count;
                                            if (shape_gate_local_delta(
                                                    resid, local_delta_abs_z, base_flags,
                                                    best_edge, resid_med, resid_sigma)) {
                                                ++diag.local_delta_exceed_count;
                                                local_flags(best_edge) = 1;
                                                if (best_edge + 1 < n_pts) {
                                                    local_flags(best_edge + 1) = 1;
                                                }
                                            }
                                            else {
                                                ++diag.local_delta_reject_count;
                                            }
                                        };
                                        for (std::size_t ii = 1; ii < candidate_edges.size(); ++ii) {
                                            const auto edge = candidate_edges[ii];
                                            if (edge <= cluster_end + 1) {
                                                cluster_end = edge;
                                            }
                                            else {
                                                flush_cluster(cluster_start, cluster_end);
                                                cluster_start = edge;
                                                cluster_end = edge;
                                            }
                                        }
                                        flush_cluster(cluster_start, cluster_end);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // array of delta's between adjacent points
            Eigen::VectorXd delta = scans.col(det).tail(n_pts - 1) - scans.col(det).head(n_pts - 1);
            // mask deltas adjacent to pre-existing or raw-flagged samples
            for (Eigen::Index i = 0; i < n_pts - 1; ++i) {
                if (base_flags(i) == 1 || base_flags(i + 1) == 1 ||
                    raw_flags(i) == 1 || raw_flags(i + 1) == 1 ||
                    local_flags(i) == 1 || local_flags(i + 1) == 1) {
                    delta(i) = 0;
                }
            }

            // minimum amplitude of spike
            const double delta_sigma = engine_utils::calc_std_dev(delta);
            double cutoff = min_spike_sigma * delta_sigma;

            // mean subtracted delta array
            Eigen::VectorXd diff = abs(delta.array() - delta.mean());
            if (std::isfinite(delta_sigma) && delta_sigma > 0.) {
                diag.max_delta_abs_z = diff.maxCoeff() / delta_sigma;
            }

            // run the spike finder,
            spike_finder(det_flags, delta, diff, n_spikes, cutoff);

            // variable to control spike_finder while loop
            bool new_spikes_found = ((diff.segment(1,n_pts - 2).array() > cutoff).count() > 0) ? 1 : 0;

            // keep despiking recursively to remove effects on the mean from large spikes
            while (new_spikes_found) {
                // if no new spikes found, set new_found to zero to end while loop
                new_spikes_found = ((diff.segment(1,n_pts - 2).array() > cutoff).count() > 0) ? 1 : 0;

                // only run if there are spikes
                if (new_spikes_found) {
                    spike_finder(det_flags, delta, diff, n_spikes, cutoff);
                }
            }

            // count up the number of spikes
            n_spikes = (det_flags.head(n_pts - 1).array() == 1).count();

            // if there are other spikes within set number of samples after a spike, set only the
            // center value to be a spike
            for (Eigen::Index i = 0; i < n_pts - 1; ++i) {
                if (det_flags(i) == 1) {
                    if (i >= n_pts - 1) {
                        break;
                    }
                    int size_loop = std::min(size, static_cast<int>(n_pts - i - 1));
                    // check the size of the region to set un_flagged if a flag is found.
                    if ((n_pts - i - 1) < size_loop) {
                        logger->trace("rng {} {} {}", (n_pts - i - 1), size,i );
                        size_loop = n_pts - i - 1;
                    }
                    if (size_loop <= 0) {
                        break;
                    }

                    // count up the flags in the region
                    int c = (det_flags.segment(i + 1, size_loop).array() == 1).count();

                    // if flags are found
                    if (c > 0) {
                        // remove those flags from the total count
                        n_spikes -= c;
                        // set region to un_flagged (including current sample)
                        det_flags.segment(i, size_loop + 1).setZero();

                        // is this a bug?  if n_pts - i <= size/2, i + size/2 >= n_pts
                        // for now, let's limit it to i + size/2 < n_pts since the
                        // start and end of the scan are not used due to
                        // filtering
                        if ((i + size_loop/2) < n_pts) {
                            det_flags(i + size_loop/2) = 1;
                        }
                    }

                    // increment so we go to the next sample region
                    i = i + size_loop - 1;
                }
            }
            diag.delta_spike_count = n_spikes;

            // now loop through if spikes were found
            if (n_spikes > 0) {
                // recount spikes
                n_spikes = (det_flags.head(n_pts - 1).array() == 1).count();

                logger->trace("n_spikes 3 {} {}", n_spikes, det);

                // vector for spike indices
                Eigen::Matrix<int, Eigen::Dynamic, 1> spike_loc(n_spikes);
                // amplitude of spike
                Eigen::VectorXd spike_vals(n_spikes);

                // populate scan location and amplitude vectors
                int count = 0;
                for (Eigen::Index i = 0; i < n_pts - 1; ++i) {
                    if (det_flags(i) == 1) {
                        spike_loc(count) = i + 1;
                        spike_vals(count) = scans(i+1, det) - scans(i, det);
                        count++;
                    }
                }

                // sanity check spike locations before windowing/replacement
                bool bad_spike_loc = false;
                for (Eigen::Index i = 0; i < spike_loc.size(); ++i) {
                    if (spike_loc(i) < 0 || spike_loc(i) >= n_pts) {
                        bad_spike_loc = true;
                        break;
                    }
                }
                if (bad_spike_loc || count != n_spikes) {
                    logger->error("despike invalid spike_loc: det {} n_pts {} n_spikes {} count {} spike_loc {}",
                                  det, n_pts, n_spikes, count, spike_loc.transpose());
                    continue;
                }

                if (run_filter) {
                    int decay_window = static_cast<int>(window_size);
                    for (Eigen::Index i=0; i<n_spikes; ++i) {
                        if (spike_loc(i) - decay_window >= 0 &&
                            spike_loc(i) + decay_window + 1 < n_pts) {
                            det_flags
                                .segment(spike_loc(i) - decay_window, 2*decay_window + 1)
                                .setOnes();
                        }
                    }
                }
                else {
                    // get the largest window that is without spikes
                    auto [win_index_0, win_index_1, win_size] =
                        make_window(spike_loc, n_spikes, n_pts);

                    if (win_size <= 1) {
                        logger->warn("despike detector {} has insufficient window for sigma estimate (win_size={}); skipping decay-based expansion",
                                     det, win_size);
                        continue;
                    }

                    // create a sub-array with values from the largest spike-free window
                    Eigen::VectorXd sub_vals =
                        scans.col(det).segment(win_index_0, win_size);

                    // copy of the sub-array for smoothing
                    Eigen::VectorXd smoothed_sub_vals = Eigen::VectorXd::Zero(win_size);

                    // smooth the sub-array with a box-car filter
                    engine_utils::smooth<engine_utils::SmoothType::boxcar>(sub_vals, smoothed_sub_vals, size);

                    // estimate the standard deviation
                    sub_vals -= smoothed_sub_vals;
                    auto sigest = engine_utils::calc_std_dev(sub_vals);
                    if (sigest < sigest_lim) {
                        sigest = sigest_lim;
                    }
                    if (!std::isfinite(sigest) || sigest <= 0.) {
                        logger->warn("despike detector {} has non-finite/zero sigma estimate ({}); skipping decay-based expansion",
                                     det, sigest);
                        continue;
                    }

                    // calculate the decay length of all the spikes
                    Eigen::ArrayXd ratio =
                        (sigest / spike_vals.array()).abs().cwiseMax(1.e-12);
                    Eigen::VectorXd decay_length =
                        (-fsmp * time_constant_sec * ratio.log()).matrix();

                    // if a decay length is less than 6, set it to 6
                    decay_length = (decay_length.array() < 6.).select(6., decay_length);

                    // replace non-finite values with minimum allowed decay length
                    for (Eigen::Index i = 0; i < decay_length.size(); ++i) {
                        if (!std::isfinite(decay_length(i))) {
                            decay_length(i) = 6.;
                        }
                    }

                    // clip overly long decay lengths instead of aborting the reduction
                    if (max_window_sec > 0) {
                        double max_len = max_window_sec * fsmp;
                        if ((decay_length.array() > max_len).any()) {
                            logger->warn("despike detector {} has decay length longer than {} * fsmp; clipping to max",
                                         det, max_window_sec);
                            decay_length = decay_length.array().min(max_len).matrix();
                        }
                    }

                    // use the decay length to flag a region around each spike
                    for (Eigen::Index i=0; i<n_spikes; ++i) {
                        if (spike_loc(i) - decay_length(i) >= 0 &&
                            spike_loc(i) + decay_length(i) + 1 < n_pts) {
                            det_flags
                                .segment(spike_loc(i) - decay_length(i), 2*decay_length(i) + 1)
                                .setOnes();
                        }
                    }
                }

            } // end of "if (n_spikes > 0)" loop

            // apply raw-sample flags after delta-based spike handling
            if ((raw_flags.array() == 1).any()) {
                det_flags = (raw_flags.array() == 1).select(1, det_flags);
                if (run_filter) {
                    int decay_window = static_cast<int>(window_size);
                    for (Eigen::Index i = 0; i < n_pts; ++i) {
                        if (raw_flags(i) == 1) {
                            if (i - decay_window >= 0 && i + decay_window + 1 < n_pts) {
                                det_flags.segment(i - decay_window, 2*decay_window + 1).setOnes();
                            }
                        }
                    }
                }
            }

            if ((local_flags.array() == 1).any()) {
                det_flags = (local_flags.array() == 1).select(1, det_flags);
                if (run_filter) {
                    int decay_window = static_cast<int>(window_size);
                    for (Eigen::Index i = 0; i < n_pts; ++i) {
                        if (local_flags(i) == 1) {
                            if (i - decay_window >= 0 && i + decay_window + 1 < n_pts) {
                                det_flags.segment(i - decay_window, 2 * decay_window + 1).setOnes();
                            }
                        }
                    }
                }
            }

            Eigen::ArrayXi added_flags =
                (det_flags.array().template cast<int>() * (base_flags.array() == 0).template cast<int>());
            diag.added_flagged_frac =
                static_cast<double>(added_flags.sum()) / static_cast<double>(n_pts);
            {
                std::vector<double> run_lengths;
                run_lengths.reserve(static_cast<std::size_t>(n_pts));
                int max_run = 0;
                Eigen::Index i = 0;
                while (i < n_pts) {
                    if (added_flags(i) != 0) {
                        Eigen::Index j = i;
                        while (j < n_pts && added_flags(j) != 0) {
                            ++j;
                        }
                        const int run_len = static_cast<int>(j - i);
                        run_lengths.push_back(static_cast<double>(run_len));
                        max_run = std::max(max_run, run_len);
                        i = j;
                    }
                    else {
                        ++i;
                    }
                }
                diag.added_region_count = static_cast<int>(run_lengths.size());
                diag.added_region_len_max = max_run;
                if (!run_lengths.empty()) {
                    const auto mid = run_lengths.size() / 2;
                    std::nth_element(run_lengths.begin(),
                                     run_lengths.begin() + static_cast<std::ptrdiff_t>(mid),
                                     run_lengths.end());
                    double med = run_lengths[mid];
                    if ((run_lengths.size() % 2) == 0) {
                        auto max_it = std::max_element(
                            run_lengths.begin(),
                            run_lengths.begin() + static_cast<std::ptrdiff_t>(mid));
                        med = 0.5 * (med + *max_it);
                    }
                    diag.added_region_len_median = med;
                }
            }

            // preserve any pre-existing flags
            det_flags = (base_flags.array() == 1).select(1, det_flags);
            flags.col(det) = det_flags;
        } // end of apt["flag"] loop
    } // end of "for (Eigen::Index det = 0; det < n_dets; det++)" loop
}

template<typename DerivedA, typename DerivedB, typename apt_t>
void Despiker::replace_spikes(Eigen::DenseBase<DerivedA> &scans, Eigen::DenseBase<DerivedB> &flags,
                              apt_t &apt, Eigen::Index start_det) {

    // declare random number generator
    thread_local boost::random::mt19937 eng;
    thread_local bool seeded = false;
    if (!seeded) {
        std::random_device rd;
        eng.seed(static_cast<boost::random::mt19937::result_type>(rd()));
        seeded = true;
    }

    Eigen::Index n_dets = flags.cols();
    Eigen::Index n_pts = flags.rows();
    Eigen::MatrixXd scans_ref = scans;

    // figure out if there are any flag-free detectors
    Eigen::Index n_flagged = 0;

    // if spike_free(detector) == 1, it contains a spike
    // otherwise none found
    auto spike_free = flags.colwise().maxCoeff();
    n_flagged = n_dets - spike_free.template cast<int>().sum();

    logger->trace("has spikes {}", spike_free);
    logger->trace("n_flagged {}", n_flagged);

    for (Eigen::Index det = 0; det < n_dets; det++) {
        if (apt["flag"](det + start_det)==0) {
            if (spike_free(det)) {
                // keep original flag structure (do not morph regions)
                // and the first and last samples
                flags(0, det) = flags(1, det);
                flags(n_pts - 1, det) = flags(n_pts - 2, det);

                // find the start and end index for each flagged region
                std::vector<int> si_flags_vec;
                std::vector<int> ei_flags_vec;

                Eigen::Index j = 0;
                while (j < n_pts) {
                    if (flags(j, det) == 1) {
                        int jstart = j;
                        while (j < n_pts && flags(j, det) == 1) {
                            j++;
                        }
                        si_flags_vec.push_back(jstart);
                        ei_flags_vec.push_back(j - 1);
                    } else {
                        j++;
                    }
                }

                Eigen::Index n_flagged_regions = static_cast<Eigen::Index>(si_flags_vec.size());
                if (n_flagged_regions == 0) {
                    continue;
                }

                logger->trace("n_flagged_regions {}", n_flagged_regions);

                Eigen::Matrix<int, Eigen::Dynamic, 1> si_flags(n_flagged_regions);
                Eigen::Matrix<int, Eigen::Dynamic, 1> ei_flags(n_flagged_regions);
                for (Eigen::Index i = 0; i < n_flagged_regions; ++i) {
                    si_flags(i) = si_flags_vec[i];
                    ei_flags(i) = ei_flags_vec[i];
                }

                // now loop on the number of flagged regions for the fix
                Eigen::VectorXd xx(2);
                Eigen::VectorXd yy(2);
                Eigen::Matrix<Eigen::Index, 1, 1> tn_pts;
                tn_pts << 2;

                for (Eigen::Index j = 0; j < n_flagged_regions; ++j) {
                    // determine the linear baseline for flagged region
                    //but use flat level if flagged at endpoints
                    Eigen::Index n_flags = ei_flags(j) - si_flags(j) + 1;
                    Eigen::VectorXd lin_offset(n_flags);

                    if (si_flags(j) == 0 && ei_flags(j) == n_pts - 1) {
                        lin_offset.setConstant(scans_ref(0, det));
                    }
                    else if (si_flags(j) == 0) {
                        lin_offset.setConstant(scans_ref(ei_flags(j) + 1, det));
                    }

                    else if (ei_flags(j) == n_pts - 1) {
                        lin_offset.setConstant(scans_ref(si_flags(j) - 1, det));
                    }

                    else {
                        // linearly interpolate between the before and after good samples
                        xx(0) = si_flags(j) - 1;
                        xx(1) = ei_flags(j) + 1;
                        yy(0) = scans_ref(si_flags(j) - 1, det);
                        yy(1) = scans_ref(ei_flags(j) + 1, det);

                        Eigen::VectorXd xlin_offset =
                            Eigen::VectorXd::LinSpaced(n_flags, si_flags(j), si_flags(j) + n_flags - 1);

                        mlinterp::interp(tn_pts.data(), n_flags, yy.data(), lin_offset.data(), xx.data(),
                                         xlin_offset.data());
                        logger->trace("xlin_offset {}", xlin_offset);
                    }

                    logger->trace("xx {}", xx);
                    logger->trace("yy {}", yy);
                    logger->trace("lin_offset {}", lin_offset);

                    // all non-flagged detectors repeat for all detectors without spikes
                    // count up spike-free detectors and store their values
                    int det_count = 0;
                    if (use_all_det) {
                        det_count = (apt["flag"].segment(start_det,n_dets).array()==0).count();
                    }
                    else {
                        for (Eigen::Index ii=0;ii<n_dets;ii++) {
                            if (!spike_free(ii) && apt["flag"](ii + start_det)==0) {
                                det_count++;
                            }
                        }
                    }

                    logger->trace("det_count {}", det_count);
                    if (det_count == 0) {
                        continue;
                    }

                    Eigen::MatrixXd detm(n_flags, det_count);
                    detm.setConstant(-99);
                    Eigen::VectorXd res(det_count);

                    logger->trace("si {}", si_flags);
                    int c = 0;
                    for (Eigen::Index ii = 0; ii < n_dets; ii++) {
                        if ((use_all_det || !spike_free(ii)) && apt["flag"](ii + start_det)==0) {
                            detm.col(c) =
                                scans_ref.block(si_flags(j), ii, n_flags, 1);
                            res(c) = apt["responsivity"](ii + start_det);
                            c++;
                        }
                    }

                    detm.transposeInPlace();

                    logger->trace("detm {}", detm);

                    // for each of these go through and redo the offset
                    Eigen::MatrixXd lin_offset_others(det_count, n_flags);

                    // first sample in scan is flagged so offset is flat
                    // with the value of the last sample in the flagged region
                    if (si_flags(j) == 0) {
                        lin_offset_others = detm.col(0).replicate(1, n_flags);
                    }

                    // last sample in scan is flagged so offset is flat
                    // with the value of the first sample in the flagged region
                    else if (ei_flags(j) == n_pts - 1) {
                        lin_offset_others = detm.col(n_flags - 1).replicate(1, n_flags);
                    }

                    else {
                        Eigen::VectorXd tmp_vec(n_flags);
                        Eigen::VectorXd xlin_offset =
                            Eigen::VectorXd::LinSpaced(n_flags, si_flags(j), si_flags(j) + n_flags - 1);

                        xx(0) = si_flags(j) - 1;
                        xx(1) = ei_flags(j) + 1;
                        // do we need this loop?
                        for (Eigen::Index ii = 0; ii < det_count; ii++) {
                            yy(0) = detm(ii, 0);
                            yy(1) = detm(ii, n_flags - 1);

                            mlinterp::interp(tn_pts.data(), n_flags, yy.data(), tmp_vec.data(), xx.data(),
                                             xlin_offset.data());
                            lin_offset_others.row(ii) = tmp_vec;
                        }

                        logger->trace("xlin_offset {}", xlin_offset);
                    }

                    logger->trace("xx {}", xx);
                    logger->trace("yy {}", yy);
                    logger->trace("lin_offset_others {}", lin_offset_others);

                    detm.noalias() = detm - lin_offset_others;

                    logger->trace("detm {}", detm);

                    // scale det by responsivities and average to make sky model
                    Eigen::VectorXd sky_model = Eigen::VectorXd::Zero(n_flags);

                    //sky_model = sky_model.array() + (detm.array().colwise() / res.array()).rowwise().sum();
                    //sky_model /= det_count;

                    for (Eigen::Index ii=0; ii<det_count; ii++) {
                        for (Eigen::Index l=0; l<n_flags; l++) {
                            sky_model(l) += detm(ii,l)/res(ii);
                        }
                    }

                    sky_model = sky_model/det_count;

                    logger->trace("sky_model {}",sky_model);

                    Eigen::VectorXd std_dev_ff = Eigen::VectorXd::Zero(det_count);

                    for (Eigen::Index ii = 0; ii < det_count; ii++) {
                        Eigen::VectorXd tmp_vec = detm.row(ii).array() / res(ii) - sky_model.transpose().array();

                        double tmp_mean = tmp_vec.mean();

                        std_dev_ff(ii) = (tmp_vec.array() - tmp_mean).pow(2).sum();
                        std_dev_ff(ii) = (n_flags == 1.) ? std_dev_ff(ii) / n_flags
                                                        : std_dev_ff(ii) / (n_flags - 1.);

                    }

                    logger->trace("std_dev_ff {}",std_dev_ff);

                    double mean_std_dev = (std_dev_ff.array().sqrt()).sum() / det_count;

                    // add noise to the fake signal
                    //mean_std_dev *= apt["responsivity"](det + start_det); // not used

                    // boost random number generator
                    boost::random::normal_distribution<> rands{0, mean_std_dev};

                    Eigen::VectorXd error =
                        Eigen::VectorXd::Zero(n_flags).unaryExpr([&](double dummy){return rands(eng);});

                    logger->trace("error {}", error);

                    // the noiseless fake data is then the sky model plus the
                    // flagged detectors linear offset
                    Eigen::VectorXd fake =
                        (sky_model.array() + error.array()) * apt["responsivity"](det + start_det) +
                        lin_offset.array();

                    logger->trace("fake {}", fake);

                    logger->trace("mean std dev {}", mean_std_dev);

                    scans.col(det).segment(si_flags(j), n_flags) = fake;
                    flags.col(det).segment(si_flags(j), n_flags).setOnes();
                } // flagged regions
            } // if it has spikes
        } // apt flag
    } // main detector loop
}

} // namespace timestream
