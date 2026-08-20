#pragma once

#include <string>
#include <algorithm>
#include <array>
#include <chrono>
#include <utility>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <limits>
#include <random>
#include <new>
#include <stdexcept>
#include <vector>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Spectra/SymEigsSolver.h>

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/utils/utils.h>

namespace timestream {

class Cleaner {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // eigen solver backend to use
    enum EigenSolverBackend {
        EigenBackend = 0,
        SpectraBackend = 1
    };

    // standard deviation limit
    double stddev_limit;

    // number of eigenvalues to remove
    std::map<Eigen::Index,Eigen::VectorXI> n_eig_to_cut;

    // number of eigenvalues to calculate
    int n_calc = 64;

    // tolerance for strength of correlation
    double tau;

    // grouping
    std::vector<std::string> grouping;

    struct StandardPCAOptions {
        bool enabled = true;
    };

    StandardPCAOptions standard_pca;

    struct NullModelOptions {
        bool enabled = false;
        int n_surrogates = 16;
        double quantile = 0.99;
        double min_good_frac = 0.8;
        int max_modes = 64; // 0 => use all available modes
        int max_samples = 20000; // 0 => use all time samples
        std::uint32_t seed = 12345;
        // Optional list of cleaning groupings where null-model is active.
        // Empty => enabled for all configured clean.grouping passes.
        std::vector<std::string> grouping;
    };

    // brute-force null-model mode selection
    NullModelOptions null_model;

    struct MarchenkoPasturOptions {
        bool enabled = false;
        double min_good_frac = 0.8;
        int max_modes = 64; // 0 => use all available modes
        int max_samples = 20000; // 0 => use all time samples
        double band_low_Hz = 0.0; // 0 => no lower band edge
        double band_high_Hz = 0.0; // 0 => no upper band edge
        double clip_z = 12.0; // robust clip after whitening; <=0 disables
        double bulk_keep_frac = 0.8; // fraction of smallest eigenvalues used for MP bulk fit
        int q_grid_size = 64; // candidate q values for MP bulk fit
        // Optional list of cleaning groupings where MP mode selection is active.
        // Empty => enabled for all configured clean.grouping passes.
        std::vector<std::string> grouping;
    };

    // Marchenko-Pastur mode selection for adaptive PCA depth
    MarchenkoPasturOptions marchenko_pastur;

    struct AdaptiveSelectorOptions {
        bool enabled = false;
        double min_good_frac = 0.7;
        int max_det = 120; // 0 => use all eligible detectors
        int max_samples = 1024; // 0 => use all time samples
        int max_pairs = 2000; // 0 => use all possible detector pairs
        std::uint32_t seed = 12345;
        double clip_z = 50.0; // clip robust-whitened detector samples before scoring; <=0 disables
        double low_weight = 1.0;
        double tail_weight = 0.0;
        double topmode_weight = 0.1;
        double reg_weight = 0.3;
        std::array<double, 2> low_band_Hz{0.05, 0.5};
        std::array<double, 2> mid_band_Hz{0.5, 2.0};
        std::vector<int> candidate_offsets{-2, 0, 2, 4};
        std::vector<std::string> grouping;
        bool log_candidates = false;
    };

    struct AdaptiveSelectorCandidateDiag {
        Eigen::Index k = 0;
        Eigen::Index n_det_used = 0;
        Eigen::Index n_time_used = 0;
        Eigen::Index sample_step = 1;
        double valid_frac = std::numeric_limits<double>::quiet_NaN();
        double med_abs_corr = std::numeric_limits<double>::quiet_NaN();
        double cm_low_mid_ratio = std::numeric_limits<double>::quiet_NaN();
        double tail4_binom_z = std::numeric_limits<double>::quiet_NaN();
        double top_mode_frac = std::numeric_limits<double>::quiet_NaN();
        double score = std::numeric_limits<double>::quiet_NaN();
        double elapsed_msec = std::numeric_limits<double>::quiet_NaN();
    };

    struct AdaptiveSelectorResult {
        bool used = false;
        bool fallback = false;
        Eigen::Index baseline_k = 0;
        Eigen::Index chosen_k = 0;
        Eigen::Index runnerup_k = -1;
        Eigen::Index n_det_input = 0;
        Eigen::Index n_candidates = 0;
        double chosen_score = std::numeric_limits<double>::quiet_NaN();
        double runnerup_score = std::numeric_limits<double>::quiet_NaN();
        double score_margin = std::numeric_limits<double>::quiet_NaN();
        double candidate_eval_msec = std::numeric_limits<double>::quiet_NaN();
        AdaptiveSelectorCandidateDiag chosen_diag;
        std::vector<AdaptiveSelectorCandidateDiag> candidates;
        Eigen::MatrixXd chosen_cleaned_scans;
    };

    // bounded adaptive PCA selector calibrated from blank-sky oracle studies
    AdaptiveSelectorOptions adaptive_selector;

    // processed timestream sample rate for optional band-limited MP covariance
    double sample_rate_Hz = 0.0;

    struct CorrGroupingOptions {
        bool enabled = false;
        std::string metric = "abs"; // "abs" or "signed"
        double corr_min = 0.6; // threshold on corr metric for graph connectivity
        int min_overlap = 300;
        double min_good_frac = 0.8;
        int min_group_size = 10;
        int max_samples = 20000; // 0 => use all time samples
        bool clean_residual = true; // clean all small/leftover eligible dets as one residual group
    };

    struct CorrGroupingResult {
        std::vector<std::vector<Eigen::Index>> groups;
        Eigen::Index n_det_input = 0;
        Eigen::Index n_det_candidates = 0;
        Eigen::Index n_det_used = 0;
        Eigen::Index n_det_grouped = 0;
        Eigen::Index n_det_ungrouped = 0;
        Eigen::Index n_groups_raw = 0;
        Eigen::Index n_groups_final = 0;
        Eigen::Index sample_step = 1;
    };

    CorrGroupingOptions corr_grouping;

    static auto normalize_group_name(std::string group) {
        std::transform(group.begin(), group.end(), group.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (group == "network") {
            group = "nw";
        }
        return group;
    }

    static auto is_corr_nw_clean_group(std::string_view group) {
        return group == "corr_nw";
    }

    static auto is_all_clean_group(std::string_view group) {
        return group == "all";
    }

    static auto is_supported_clean_group(std::string group) {
        group = normalize_group_name(std::move(group));
        return is_all_clean_group(group) ||
               citlali::config::is_array_map_grouping(group) ||
               citlali::config::is_network_map_grouping(group) ||
               citlali::config::is_detector_map_grouping(group) ||
               citlali::config::is_frequency_group_map_grouping(group) ||
               is_corr_nw_clean_group(group);
    }

    auto null_model_enabled_for_group(const std::string &group) const {
        if (null_model.grouping.empty()) {
            return true;
        }
        const auto g = normalize_group_name(group);
        for (const auto &allowed : null_model.grouping) {
            if (normalize_group_name(allowed) == g) {
                return true;
            }
        }
        return false;
    }

    auto marchenko_pastur_enabled_for_group(const std::string &group) const {
        if (marchenko_pastur.grouping.empty()) {
            return true;
        }
        const auto g = normalize_group_name(group);
        for (const auto &allowed : marchenko_pastur.grouping) {
            if (normalize_group_name(allowed) == g) {
                return true;
            }
        }
        return false;
    }

    auto adaptive_selector_enabled_for_group(const std::string &group) const {
        if (adaptive_selector.grouping.empty()) {
            return true;
        }
        const auto g = normalize_group_name(group);
        for (const auto &allowed : adaptive_selector.grouping) {
            if (normalize_group_name(allowed) == g) {
                return true;
            }
        }
        return false;
    }

    auto adaptive_selector_candidate_cuts(const Eigen::Index baseline_k, const Eigen::Index max_k) const {
        std::vector<Eigen::Index> ks;
        ks.reserve(adaptive_selector.candidate_offsets.size() + 1);
        const auto max_keep = std::max<Eigen::Index>(0, max_k);
        for (const auto offset : adaptive_selector.candidate_offsets) {
            const auto k = std::clamp<Eigen::Index>(baseline_k + static_cast<Eigen::Index>(offset), 0, max_keep);
            ks.push_back(k);
        }
        ks.push_back(std::clamp<Eigen::Index>(baseline_k, 0, max_keep));
        std::sort(ks.begin(), ks.end());
        ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
        return ks;
    }

    auto adaptive_mode_selection_enabled() const {
        return null_model.enabled || marchenko_pastur.enabled;
    }

    auto active_cleaner_mode() const {
        using CleanerMode = citlali::config::ProcessedTimeChunkCleanerMode;
        if (adaptive_selector.enabled) {
            return CleanerMode::adaptive_selector;
        }
        if (standard_pca.enabled) {
            return CleanerMode::standard_pca;
        }
        if (null_model.enabled) {
            return CleanerMode::null_model;
        }
        if (marchenko_pastur.enabled) {
            return CleanerMode::marchenko_pastur;
        }
        return CleanerMode::none;
    }

    auto active_cleaner_label() const {
        return std::string{
            citlali::config::to_string(active_cleaner_mode())};
    }

    auto adaptive_mode_selection_max_modes() const {
        if (null_model.enabled) {
            return null_model.max_modes;
        }
        if (marchenko_pastur.enabled) {
            return marchenko_pastur.max_modes;
        }
        return 0;
    }

    [[noreturn]] void fail_cleaner(const std::string &cleaner_name, const std::string &message) const {
        logger->error("{}: {}", cleaner_name, message);
        throw std::runtime_error(cleaner_name + ": " + message);
    }

    template <typename Derived>
    auto get_stddev_index(const Eigen::DenseBase<Derived> &evals) {
        if (evals.size() < 2) {
            logger->warn("stddev cut: not enough eigenvalues ({}); skipping cut", evals.size());
            return Eigen::Index{0};
        }
        // copy eigenvalues
        Eigen::VectorXd ev = evals.derived().array().abs().log10();

        auto n_dets = evals.size();
        // mean of eigenvalues
        auto m_ev = ev.mean();
        // standard deviation of eigenvalues
        auto stddev = engine_utils::calc_std_dev(ev);

        bool keep_going = true;
        int n_keep_last = n_dets;

        // vector of eigenvalues below stddev cut
        Eigen::Matrix<bool,Eigen::Dynamic,1> good(n_dets);
        good.setOnes(n_dets);

        int iterator = 0;
        while (keep_going) {
            // count up number of eigenvalues that pass the cut
            int count = 0;
            for (Eigen::Index i=0; i<n_dets; i++) {
                if (good(i)) {
                    if (ev(i) > m_ev + stddev_limit*stddev) {
                        good(i) = false;
                    }
                    else {
                        count++;
                    }
                }
            }

            if (count <= 1) {
                logger->warn("stddev cut: only {} eigenvalue(s) remain after clipping; skipping cut", count);
                return Eigen::Index{0};
            }

            if (count >= n_keep_last) {
                keep_going = false;
            }
            else {
                // get mean for good eigen values
                m_ev = 0.;
                for (Eigen::Index i=0; i<n_dets; i++) {
                    if (good(i)) {
                        m_ev += ev(i);
                    }
                }
                // get stddev for good eigen values
                m_ev /= count;
                stddev = 0.;
                for (Eigen::Index i=0; i<n_dets; i++) {
                    if (good(i)) {
                        stddev += (ev(i) - m_ev)*(ev(i) - m_ev);
                    }
                }
                stddev = stddev/(count-1.);
                stddev = sqrt(stddev);
                n_keep_last = count;
            }
            iterator++;
        }

        if (!std::isfinite(m_ev) || !std::isfinite(stddev)) {
            logger->warn("stddev cut: non-finite stats (mean={}, stddev={}); skipping cut", m_ev, stddev);
            return Eigen::Index{0};
        }

        // stddev limit
        double limit = pow(10.,m_ev + stddev_limit*stddev);
        // index where limit occurs
        Eigen::Index limit_index = 0;

        // find index
        for (Eigen::Index i=0; i<n_dets; i++) {
            if (evals(i) <= limit){
                limit_index = i;
                break;
            }
        }

        return limit_index;
    }

    template <typename DerivedA, typename DerivedB>
    Eigen::MatrixXd calc_cov_with_mask(const Eigen::DenseBase<DerivedA> &sig, const Eigen::DenseBase<DerivedB> &good) {
        Eigen::MatrixXd det = (sig.derived().array() * good.derived().array()).matrix();
        Eigen::MatrixXd numer = det.adjoint() * det;
        Eigen::MatrixXd denom = (good.derived().adjoint() * good.derived()).array() - 1.0;
        // Return a concrete matrix. Returning the select expression here would
        // keep references to local temporaries (numer/denom) and can segfault.
        Eigen::MatrixXd cov = (denom.array() > 0.0).select(numer.array() / denom.array(), 0.0).matrix();
        return cov;
    }

    template <typename Derived>
    auto enforce_monotonic_nonincreasing(const Eigen::DenseBase<Derived> &in) {
        Eigen::VectorXd out = in.derived();
        if (out.size() < 2) {
            return out;
        }
        for (Eigen::Index i = out.size() - 2; i >= 0; --i) {
            if (out(i) < out(i + 1)) {
                out(i) = out(i + 1);
            }
            if (i == 0) {
                break;
            }
        }
        return out;
    }

    template <typename DerivedA, typename DerivedB, typename DerivedC>
    auto get_null_model_index(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &,
                              const Eigen::DenseBase<DerivedC> &);

    template <typename DerivedA, typename DerivedB, typename DerivedC>
    auto get_marchenko_pastur_index(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &,
                                    const Eigen::DenseBase<DerivedC> &);

    template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    auto select_adaptive_cut(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &,
                             const Eigen::DenseBase<DerivedC> &, const Eigen::DenseBase<DerivedD> &,
                             const Eigen::Index, const std::string &, const Eigen::Index,
                             const Eigen::Index) const;

    template <typename DerivedA, typename DerivedB, typename DerivedC>
    auto get_corr_groups(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &,
                         const Eigen::DenseBase<DerivedC> &);

    // calculate the eigenvalues from a matrix while removing flags
    template <EigenSolverBackend backend, typename DerivedA, typename DerivedB, typename DerivedC>
    auto calc_eig_values(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &, Eigen::DenseBase<DerivedC> &,
                         const Eigen::Index);

    // remove computed eigenvalues from matrix and recompute tods
    template <EigenSolverBackend backend, typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    auto remove_eig_values(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &,
                           const Eigen::DenseBase<DerivedC> &, const Eigen::DenseBase<DerivedD> &,
                           Eigen::DenseBase<DerivedA> &, const Eigen::Index, const Eigen::Index,
                           const std::string &, const Eigen::Index, const Eigen::Index);
};

namespace detail {

class DisjointSet {
public:
    explicit DisjointSet(Eigen::Index n) {
        parent.resize(static_cast<std::size_t>(n));
        rank.resize(static_cast<std::size_t>(n), 0);
        for (Eigen::Index i = 0; i < n; ++i) {
            parent[static_cast<std::size_t>(i)] = i;
        }
    }

    Eigen::Index find(Eigen::Index x) {
        auto &p = parent[static_cast<std::size_t>(x)];
        if (p != x) {
            p = find(p);
        }
        return p;
    }

    void unite(Eigen::Index a, Eigen::Index b) {
        auto ra = find(a);
        auto rb = find(b);
        if (ra == rb) {
            return;
        }
        auto &rra = rank[static_cast<std::size_t>(ra)];
        auto &rrb = rank[static_cast<std::size_t>(rb)];
        if (rra < rrb) {
            parent[static_cast<std::size_t>(ra)] = rb;
        }
        else if (rra > rrb) {
            parent[static_cast<std::size_t>(rb)] = ra;
        }
        else {
            parent[static_cast<std::size_t>(rb)] = ra;
            rra += 1;
        }
    }

private:
    std::vector<Eigen::Index> parent;
    std::vector<int> rank;
};

template <typename Derived>
double robust_center(const Eigen::DenseBase<Derived> &data) {
    if (data.size() == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    Eigen::VectorXd tmp = data.derived();
    return tula::alg::median(tmp);
}

template <typename Derived>
double robust_scale(const Eigen::DenseBase<Derived> &data, const double center) {
    if (data.size() < 2 || !std::isfinite(center)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    Eigen::VectorXd tmp = data.derived();
    Eigen::VectorXd abs_dev = (tmp.array() - center).abs().matrix();
    double sigma = 1.4826 * tula::alg::median(abs_dev);
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        sigma = engine_utils::calc_std_dev(tmp);
    }
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sigma;
}

inline double median_from_values(std::vector<double> values) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    if ((values.size() % 2) == 1) {
        return *mid;
    }
    const auto mid_lo = std::max(values.begin(), mid - 1);
    std::nth_element(values.begin(), mid_lo, values.end());
    return 0.5 * ((*mid_lo) + (*mid));
}

inline auto downsample_even_indices(Eigen::Index n, int max_keep) {
    std::vector<Eigen::Index> out;
    if (n <= 0) {
        return out;
    }
    if (max_keep <= 0 || n <= static_cast<Eigen::Index>(max_keep)) {
        out.reserve(static_cast<std::size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i) {
            out.push_back(i);
        }
        return out;
    }
    const Eigen::Index step = std::max<Eigen::Index>(
        1, static_cast<Eigen::Index>(std::ceil(static_cast<double>(n) / static_cast<double>(max_keep))));
    out.reserve(static_cast<std::size_t>(max_keep));
    for (Eigen::Index i = 0; i < n && static_cast<int>(out.size()) < max_keep; i += step) {
        out.push_back(i);
    }
    if (!out.empty() && out.back() != n - 1 && static_cast<int>(out.size()) < max_keep) {
        out.push_back(n - 1);
    }
    return out;
}

inline double quantile_sorted(const Eigen::VectorXd &sorted, const double p) {
    if (sorted.size() == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double pc = std::clamp(p, 0.0, 1.0);
    const double pos = pc * static_cast<double>(sorted.size() - 1);
    const auto i0 = static_cast<Eigen::Index>(std::floor(pos));
    const auto i1 = static_cast<Eigen::Index>(std::ceil(pos));
    if (i0 == i1) {
        return sorted(i0);
    }
    const double frac = pos - static_cast<double>(i0);
    return (1.0 - frac) * sorted(i0) + frac * sorted(i1);
}

inline std::array<double, 3> mp_quantiles(double q, int grid_n = 1024) {
    q = std::max(q, 1.0e-4);
    const double pi = std::acos(-1.0);
    const double lambda_minus = std::pow(1.0 - std::sqrt(q), 2);
    const double lambda_plus = std::pow(1.0 + std::sqrt(q), 2);
    const int n = std::max(grid_n, 64);
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, lambda_minus, lambda_plus);
    Eigen::VectorXd y = ((lambda_plus - x.array()) * (x.array() - lambda_minus)).max(0.0).sqrt().matrix();
    Eigen::VectorXd denom = (2.0 * pi * q * x.array()).max(1.0e-12).matrix();
    Eigen::VectorXd pdf = y.array() / denom.array();
    Eigen::VectorXd cdf = Eigen::VectorXd::Zero(n);
    for (int i = 1; i < n; ++i) {
        const double dx = x(i) - x(i - 1);
        cdf(i) = cdf(i - 1) + 0.5 * (pdf(i) + pdf(i - 1)) * dx;
    }
    if (cdf(n - 1) <= 0.0 || !std::isfinite(cdf(n - 1))) {
        return {lambda_minus, 0.5 * (lambda_minus + lambda_plus), lambda_plus};
    }
    cdf /= cdf(n - 1);
    auto interp = [&](double p) {
        const double pc = std::clamp(p, 0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            if (cdf(i) >= pc) {
                if (i == 0) {
                    return x(0);
                }
                const double denom_local = cdf(i) - cdf(i - 1);
                const double frac = (denom_local > 0.0) ? (pc - cdf(i - 1)) / denom_local : 0.0;
                return x(i - 1) + frac * (x(i) - x(i - 1));
            }
        }
        return x(n - 1);
    };
    return {interp(0.1), interp(0.5), interp(0.9)};
}

struct MPBulkFitResult {
    Eigen::Index k_mp = 0;
    Eigen::Index n_bulk = 0;
    double q_fit = std::numeric_limits<double>::quiet_NaN();
    double n_eff_fit = std::numeric_limits<double>::quiet_NaN();
    double sigma2_fit = std::numeric_limits<double>::quiet_NaN();
    double lambda_minus = std::numeric_limits<double>::quiet_NaN();
    double lambda_plus = std::numeric_limits<double>::quiet_NaN();
    double top_over_edge = std::numeric_limits<double>::quiet_NaN();
    double fit_err = std::numeric_limits<double>::quiet_NaN();
};

inline auto fit_mp_bulk(const Eigen::VectorXd &evals_desc, const double n_eff_time,
                        const double bulk_keep_frac, const int q_grid_size) {
    MPBulkFitResult out;
    if (evals_desc.size() < 8 || !(std::isfinite(n_eff_time) && n_eff_time >= 2.0)) {
        return out;
    }

    std::vector<double> positive_bulk;
    positive_bulk.reserve(static_cast<std::size_t>(evals_desc.size()));
    const double rel_floor = std::max(evals_desc(0) * 1.0e-10, std::numeric_limits<double>::min());
    for (Eigen::Index i = 0; i < evals_desc.size(); ++i) {
        const double v = evals_desc(i);
        if (std::isfinite(v) && v > rel_floor) {
            positive_bulk.push_back(v);
        }
    }
    if (positive_bulk.size() < 8) {
        positive_bulk.reserve(static_cast<std::size_t>(evals_desc.size()));
        positive_bulk.clear();
        for (Eigen::Index i = 0; i < evals_desc.size(); ++i) {
            const double v = evals_desc(i);
            if (std::isfinite(v) && v > 0.0) {
                positive_bulk.push_back(v);
            }
        }
    }
    if (positive_bulk.size() < 8) {
        return out;
    }

    std::sort(positive_bulk.begin(), positive_bulk.end());
    out.n_bulk = std::max<Eigen::Index>(
        6, static_cast<Eigen::Index>(std::floor(static_cast<double>(positive_bulk.size()) *
                                                std::clamp(bulk_keep_frac, 0.1, 1.0))));
    out.n_bulk = std::min<Eigen::Index>(out.n_bulk, static_cast<Eigen::Index>(positive_bulk.size()));
    Eigen::VectorXd bulk(out.n_bulk);
    for (Eigen::Index i = 0; i < out.n_bulk; ++i) {
        bulk(i) = positive_bulk[static_cast<std::size_t>(i)];
    }

    const double emp_q10 = quantile_sorted(bulk, 0.1);
    const double emp_q50 = quantile_sorted(bulk, 0.5);
    const double emp_q90 = quantile_sorted(bulk, 0.9);
    if (!(std::isfinite(emp_q10) && std::isfinite(emp_q50) && std::isfinite(emp_q90) &&
          emp_q10 > 0.0 && emp_q50 > 0.0 && emp_q90 > 0.0)) {
        return out;
    }

    const Eigen::Index n_det = evals_desc.size();
    const double q_min = std::max(static_cast<double>(n_det) / std::max<double>(n_eff_time, 1.0), 1.0e-3);
    const double q_max = std::max(4.0 * q_min, 8.0);
    const int n_q = std::max(q_grid_size, 8);
    std::vector<double> q_candidates(static_cast<std::size_t>(n_q));
    const bool use_geom = (q_max / std::max(q_min, 1.0e-6)) > 2.0;
    for (int i = 0; i < n_q; ++i) {
        const double t = (n_q == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(n_q - 1);
        if (use_geom) {
            q_candidates[static_cast<std::size_t>(i)] = q_min * std::pow(q_max / q_min, t);
        }
        else {
            q_candidates[static_cast<std::size_t>(i)] = q_min + t * (q_max - q_min);
        }
    }

    bool have_best = false;
    for (const auto q : q_candidates) {
        const auto qtls = mp_quantiles(q);
        if (!(qtls[0] > 0.0 && qtls[1] > 0.0 && qtls[2] > 0.0)) {
            continue;
        }
        const double sigma2 = emp_q50 / qtls[1];
        const double pred_q10 = sigma2 * qtls[0];
        const double pred_q90 = sigma2 * qtls[2];
        if (!(pred_q10 > 0.0 && pred_q90 > 0.0)) {
            continue;
        }
        const double err = std::pow(std::log(pred_q10 / emp_q10), 2) + std::pow(std::log(pred_q90 / emp_q90), 2);
        if (!std::isfinite(err)) {
            continue;
        }
        if (!have_best || err < out.fit_err) {
            have_best = true;
            out.q_fit = q;
            out.n_eff_fit = static_cast<double>(n_det) / q;
            out.sigma2_fit = sigma2;
            out.lambda_minus = sigma2 * std::pow(1.0 - std::sqrt(q), 2);
            out.lambda_plus = sigma2 * std::pow(1.0 + std::sqrt(q), 2);
            out.fit_err = err;
        }
    }
    if (!have_best || !(out.lambda_plus > 0.0)) {
        return out;
    }

    out.k_mp = 0;
    for (Eigen::Index i = 0; i < evals_desc.size(); ++i) {
        if (evals_desc(i) > out.lambda_plus) {
            ++out.k_mp;
        }
    }
    out.top_over_edge = evals_desc(0) / out.lambda_plus;
    return out;
}

inline double median_positive_overlap_count(const Eigen::MatrixXd &overlap, const bool use_pairs = true) {
    std::vector<double> vals;
    const Eigen::Index n = overlap.rows();
    if (n == 0 || overlap.cols() != n) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    vals.reserve(static_cast<std::size_t>(use_pairs ? (n * (n - 1)) / 2 : n));
    if (use_pairs) {
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = i + 1; j < n; ++j) {
                const double v = overlap(i, j) - 1.0;
                if (std::isfinite(v) && v > 0.0) {
                    vals.push_back(v);
                }
            }
        }
    }
    if (vals.size() < 4) {
        vals.clear();
        vals.reserve(static_cast<std::size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i) {
            const double v = overlap(i, i) - 1.0;
            if (std::isfinite(v) && v > 0.0) {
                vals.push_back(v);
            }
        }
    }
    if (vals.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(vals.begin(), vals.end());
    const auto m = vals.size() / 2;
    if ((vals.size() % 2) == 0) {
        return 0.5 * (vals[m - 1] + vals[m]);
    }
    return vals[m];
}

inline std::pair<double, double> good_fraction_stats(const Eigen::MatrixXd &good) {
    if (good.rows() <= 0 || good.cols() <= 0) {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    Eigen::VectorXd frac = good.colwise().mean().transpose();
    std::vector<double> vals;
    vals.reserve(static_cast<std::size_t>(frac.size()));
    for (Eigen::Index i = 0; i < frac.size(); ++i) {
        const double v = frac(i);
        if (std::isfinite(v)) {
            vals.push_back(v);
        }
    }
    if (vals.empty()) {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    std::sort(vals.begin(), vals.end());
    const double min_frac = vals.front();
    const double med_frac = (vals.size() % 2 == 0)
                                ? 0.5 * (vals[vals.size() / 2 - 1] + vals[vals.size() / 2])
                                : vals[vals.size() / 2];
    return {min_frac, med_frac};
}

} // namespace detail

template <typename DerivedA, typename DerivedB, typename DerivedC>
auto Cleaner::get_corr_groups(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags,
                              const Eigen::DenseBase<DerivedC> &apt_flags) {
    CorrGroupingResult result;
    result.n_det_input = scans.cols();

    const Eigen::Index n_pts_full = scans.rows();
    const Eigen::Index n_dets = scans.cols();
    if (!corr_grouping.enabled || n_pts_full < 4 || n_dets < 2) {
        return result;
    }

    try {
        Eigen::Index sample_step = 1;
        if (corr_grouping.max_samples > 0 && n_pts_full > corr_grouping.max_samples) {
            sample_step = static_cast<Eigen::Index>(
                std::ceil(static_cast<double>(n_pts_full) / static_cast<double>(corr_grouping.max_samples)));
        }
        result.sample_step = sample_step;
        const Eigen::Index n_pts = (n_pts_full + sample_step - 1) / sample_step;
        if (n_pts < 4) {
            return result;
        }

        auto is_good = [&](Eigen::Index i_sub, Eigen::Index j_det) {
            const Eigen::Index i = i_sub * sample_step;
            return !flags.derived()(i, j_det);
        };

        std::vector<Eigen::Index> keep_frac;
        keep_frac.reserve(static_cast<std::size_t>(n_dets));
        for (Eigen::Index j = 0; j < n_dets; ++j) {
            if (apt_flags.derived()(j) != 0) {
                continue;
            }
            double good_count = 0.0;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (is_good(i, j)) {
                    good_count += 1.0;
                }
            }
            const double frac = good_count / static_cast<double>(n_pts);
            if (good_count > 1.0 && frac >= corr_grouping.min_good_frac) {
                keep_frac.push_back(j);
            }
        }
        result.n_det_candidates = static_cast<Eigen::Index>(keep_frac.size());
        if (keep_frac.size() < 2) {
            return result;
        }

        const Eigen::Index n_keep_frac = static_cast<Eigen::Index>(keep_frac.size());
        Eigen::VectorXd means = Eigen::VectorXd::Zero(n_keep_frac);
        Eigen::VectorXd stds = Eigen::VectorXd::Zero(n_keep_frac);
        std::vector<Eigen::Index> keep_std;
        keep_std.reserve(static_cast<std::size_t>(n_keep_frac));
        for (Eigen::Index j = 0; j < n_keep_frac; ++j) {
            const Eigen::Index det_j = keep_frac[static_cast<std::size_t>(j)];
            double denom = 0.0;
            double sum = 0.0;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (is_good(i, det_j)) {
                    const double v = scans.derived()(i * sample_step, det_j);
                    sum += v;
                    denom += 1.0;
                }
            }
            if (denom <= 1.0) {
                continue;
            }
            const double mean = sum / denom;
            double var_num = 0.0;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (is_good(i, det_j)) {
                    const double d = scans.derived()(i * sample_step, det_j) - mean;
                    var_num += d * d;
                }
            }
            const double var_den = denom - 1.0;
            if (var_den <= 0.0) {
                continue;
            }
            const double std = std::sqrt(std::max(var_num / var_den, 0.0));
            if (std > 0.0 && std::isfinite(std)) {
                means(j) = mean;
                stds(j) = std;
                keep_std.push_back(j);
            }
        }

        if (keep_std.size() < 2) {
            return result;
        }
        const Eigen::Index n_used = static_cast<Eigen::Index>(keep_std.size());
        result.n_det_used = n_used;

        Eigen::MatrixXd sigz(n_pts, n_used);
        Eigen::MatrixXd good_used(n_pts, n_used);
        for (Eigen::Index k = 0; k < n_used; ++k) {
            const Eigen::Index j = keep_std[static_cast<std::size_t>(k)];
            const Eigen::Index det_j = keep_frac[static_cast<std::size_t>(j)];
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                const double g = is_good(i, det_j) ? 1.0 : 0.0;
                good_used(i, k) = g;
                const double v = scans.derived()(i * sample_step, det_j);
                sigz(i, k) = (v - means(j)) / stds(j);
            }
        }

        Eigen::MatrixXd cov = calc_cov_with_mask(sigz, good_used);
        Eigen::MatrixXd overlap = good_used.adjoint() * good_used;
        Eigen::VectorXd var = cov.diagonal().cwiseMax(0.0);
        Eigen::VectorXd std = var.array().sqrt();
        Eigen::MatrixXd corr = Eigen::MatrixXd::Zero(n_used, n_used);
        corr.diagonal().setOnes();
        for (Eigen::Index i = 0; i < n_used; ++i) {
            for (Eigen::Index j = i + 1; j < n_used; ++j) {
                if (overlap(i, j) < static_cast<double>(corr_grouping.min_overlap)) {
                    continue;
                }
                const double denom = std(i) * std(j);
                if (denom <= 0.0 || !std::isfinite(denom)) {
                    continue;
                }
                double v = cov(i, j) / denom;
                if (!std::isfinite(v)) {
                    v = 0.0;
                }
                v = std::clamp(v, -1.0, 1.0);
                corr(i, j) = v;
                corr(j, i) = v;
            }
        }

        detail::DisjointSet dsu(n_used);
        const bool use_abs =
            !citlali::config::is_signed_processed_corr_grouping_metric(
                corr_grouping.metric);
        const double thr = std::clamp(corr_grouping.corr_min, 0.0, 1.0);
        for (Eigen::Index i = 0; i < n_used; ++i) {
            for (Eigen::Index j = i + 1; j < n_used; ++j) {
                const double v = use_abs ? std::abs(corr(i, j)) : corr(i, j);
                if (v >= thr) {
                    dsu.unite(i, j);
                }
            }
        }

        std::unordered_map<Eigen::Index, std::vector<Eigen::Index>> root_to_group;
        root_to_group.reserve(static_cast<std::size_t>(n_used));
        for (Eigen::Index i = 0; i < n_used; ++i) {
            const auto root = dsu.find(i);
            auto det_j = keep_frac[static_cast<std::size_t>(keep_std[static_cast<std::size_t>(i)])];
            root_to_group[root].push_back(det_j);
        }

        std::vector<std::vector<Eigen::Index>> groups_raw;
        groups_raw.reserve(root_to_group.size());
        for (auto &kv : root_to_group) {
            auto &g = kv.second;
            std::sort(g.begin(), g.end());
            groups_raw.push_back(std::move(g));
        }
        result.n_groups_raw = static_cast<Eigen::Index>(groups_raw.size());

        std::vector<std::vector<Eigen::Index>> groups_final;
        std::vector<Eigen::Index> residual;
        for (auto &g : groups_raw) {
            if (static_cast<int>(g.size()) >= std::max(2, corr_grouping.min_group_size)) {
                groups_final.push_back(std::move(g));
            }
            else {
                residual.insert(residual.end(), g.begin(), g.end());
            }
        }
        if (corr_grouping.clean_residual && residual.size() >= 2) {
            std::sort(residual.begin(), residual.end());
            groups_final.push_back(std::move(residual));
        }

        std::sort(groups_final.begin(), groups_final.end(), [](const auto &a, const auto &b) {
            return a.size() > b.size();
        });

        Eigen::Index n_grouped = 0;
        for (const auto &g : groups_final) {
            n_grouped += static_cast<Eigen::Index>(g.size());
        }
        result.groups = std::move(groups_final);
        result.n_groups_final = static_cast<Eigen::Index>(result.groups.size());
        result.n_det_grouped = n_grouped;
        result.n_det_ungrouped = result.n_det_input - n_grouped;

        return result;
    }
    catch (const std::bad_alloc &) {
        logger->warn("corr_grouping: memory allocation failed; falling back to base grouping");
        return CorrGroupingResult{};
    }
    catch (const std::exception &e) {
        logger->warn("corr_grouping: exception {}; falling back to base grouping", e.what());
        return CorrGroupingResult{};
    }
}

template <typename DerivedA, typename DerivedB, typename DerivedC>
auto Cleaner::get_null_model_index(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags,
                                   const Eigen::DenseBase<DerivedC> &apt_flags) {
    if (!null_model.enabled) {
        return Eigen::Index{0};
    }
    if (null_model.n_surrogates < 4) {
        fail_cleaner("null_model", "n_surrogates=" + std::to_string(null_model.n_surrogates) + " is too small");
    }

    try {
        const Eigen::Index n_pts_full = scans.rows();
        const Eigen::Index n_dets = scans.cols();
        if (n_pts_full < 4 || n_dets < 2) {
            fail_cleaner("null_model", "insufficient data (n_pts=" + std::to_string(n_pts_full)
                                           + ", n_dets=" + std::to_string(n_dets) + ")");
        }

        // Subsample in time to cap memory and runtime.
        Eigen::Index sample_step = 1;
        if (null_model.max_samples > 0 && n_pts_full > null_model.max_samples) {
            sample_step = static_cast<Eigen::Index>(
                std::ceil(static_cast<double>(n_pts_full) / static_cast<double>(null_model.max_samples)));
        }
        const Eigen::Index n_pts = (n_pts_full + sample_step - 1) / sample_step;
        if (n_pts < 4) {
            fail_cleaner("null_model", "max_samples yields too few samples after subsampling");
        }

        auto is_good = [&](Eigen::Index i_sub, Eigen::Index j_det) {
            const Eigen::Index i = i_sub * sample_step;
            return !flags.derived()(i, j_det);
        };

        std::vector<Eigen::Index> keep_frac;
        keep_frac.reserve(static_cast<std::size_t>(n_dets));
        for (Eigen::Index j = 0; j < n_dets; ++j) {
            if (apt_flags.derived()(j) != 0) {
                continue;
            }
            double good_count = 0.0;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (is_good(i, j)) {
                    good_count += 1.0;
                }
            }
            const double frac = good_count / static_cast<double>(n_pts);
            if (good_count > 1.0 && frac >= null_model.min_good_frac) {
                keep_frac.push_back(j);
            }
        }

        if (keep_frac.size() < 2) {
            fail_cleaner("null_model", "only " + std::to_string(keep_frac.size())
                                           + " detector(s) pass min_good_frac=" + std::to_string(null_model.min_good_frac));
        }

        const Eigen::Index n_keep_frac = static_cast<Eigen::Index>(keep_frac.size());
        Eigen::VectorXd means = Eigen::VectorXd::Zero(n_keep_frac);
        Eigen::VectorXd stds = Eigen::VectorXd::Zero(n_keep_frac);
        std::vector<Eigen::Index> keep_std;
        keep_std.reserve(static_cast<std::size_t>(n_keep_frac));
        for (Eigen::Index j = 0; j < n_keep_frac; ++j) {
            const Eigen::Index det_j = keep_frac[static_cast<std::size_t>(j)];
            double denom = 0.0;
            double sum = 0.0;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (is_good(i, det_j)) {
                    const double v = scans.derived()(i * sample_step, det_j);
                    sum += v;
                    denom += 1.0;
                }
            }
            if (denom <= 1.0) {
                continue;
            }
            const double mean = sum / denom;
            double var_num = 0.0;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (is_good(i, det_j)) {
                    const double d = scans.derived()(i * sample_step, det_j) - mean;
                    var_num += d * d;
                }
            }
            const double var_den = denom - 1.0;
            if (var_den <= 0.0) {
                continue;
            }
            const double std = std::sqrt(std::max(var_num / var_den, 0.0));
            if (std > 0.0 && std::isfinite(std)) {
                means(j) = mean;
                stds(j) = std;
                keep_std.push_back(j);
            }
        }

        if (keep_std.size() < 2) {
            fail_cleaner("null_model", "only " + std::to_string(keep_std.size())
                                           + " detector(s) have finite non-zero stddev");
        }

        const Eigen::Index n_used = static_cast<Eigen::Index>(keep_std.size());
        Eigen::MatrixXd sigz(n_pts, n_used);
        Eigen::MatrixXd good_used(n_pts, n_used);
        for (Eigen::Index k = 0; k < n_used; ++k) {
            const Eigen::Index j = keep_std[static_cast<std::size_t>(k)];
            const Eigen::Index det_j = keep_frac[static_cast<std::size_t>(j)];
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                const double g = is_good(i, det_j) ? 1.0 : 0.0;
                good_used(i, k) = g;
                const double v = scans.derived()(i * sample_step, det_j);
                sigz(i, k) = (v - means(j)) / stds(j);
            }
        }

        Eigen::MatrixXd cov_obs = calc_cov_with_mask(sigz, good_used);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> obs_solver(cov_obs);
        if (obs_solver.info() != Eigen::Success) {
            fail_cleaner("null_model", "failed to compute observed eigenspectrum");
        }
        Eigen::VectorXd obs_evals = obs_solver.eigenvalues().reverse();

        Eigen::Index n_modes = obs_evals.size();
        if (null_model.max_modes > 0) {
            n_modes = std::min<Eigen::Index>(n_modes, static_cast<Eigen::Index>(null_model.max_modes));
        }
        if (n_modes < 2) {
            fail_cleaner("null_model", "not enough modes available after max_modes truncation");
        }
        obs_evals = obs_evals.head(n_modes);

        Eigen::MatrixXd null_eigs(null_model.n_surrogates, n_modes);
        Eigen::MatrixXd sur(sigz.rows(), sigz.cols());
        std::mt19937 rng(null_model.seed);
        std::uniform_int_distribution<Eigen::Index> shift_dist(0, n_pts - 1);

        for (Eigen::Index s = 0; s < null_model.n_surrogates; ++s) {
            for (Eigen::Index j = 0; j < n_used; ++j) {
                const Eigen::Index shift = shift_dist(rng);
                for (Eigen::Index i = 0; i < n_pts; ++i) {
                    Eigen::Index src = i - shift;
                    if (src < 0) {
                        src += n_pts;
                    }
                    sur(i, j) = sigz(src, j);
                }
            }

            Eigen::MatrixXd cov_s = calc_cov_with_mask(sur, good_used);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sur_solver(cov_s);
            if (sur_solver.info() != Eigen::Success) {
                fail_cleaner("null_model", "surrogate eigensolver failed at trial " + std::to_string(s));
            }
            null_eigs.row(s) = sur_solver.eigenvalues().reverse().head(n_modes);
        }

        Eigen::VectorXd null_q = Eigen::VectorXd::Zero(n_modes);
        const double q = std::clamp(null_model.quantile, 0.5, 0.999999);
        const Eigen::Index q_idx = static_cast<Eigen::Index>(
            std::floor(q * static_cast<double>(null_model.n_surrogates - 1)));
        for (Eigen::Index k = 0; k < n_modes; ++k) {
            std::vector<double> vals(static_cast<std::size_t>(null_model.n_surrogates));
            for (Eigen::Index s = 0; s < null_model.n_surrogates; ++s) {
                vals[static_cast<std::size_t>(s)] = null_eigs(s, k);
            }
            std::sort(vals.begin(), vals.end());
            null_q(k) = vals[static_cast<std::size_t>(q_idx)];
        }
        null_q = enforce_monotonic_nonincreasing(null_q);

        Eigen::Index k_null = 0;
        for (Eigen::Index k = 0; k < n_modes; ++k) {
            if (obs_evals(k) > null_q(k)) {
                ++k_null;
            }
        }
        k_null = std::min<Eigen::Index>(k_null, n_dets - 1);

        logger->debug("null_model: n_det_input={} n_det_used={} n_modes={} n_pts={} step={} k={}",
                      n_dets, n_used, n_modes, n_pts, sample_step, k_null);
        return k_null;
    }
    catch (const std::bad_alloc &) {
        fail_cleaner("null_model", "memory allocation failed");
    }
    catch (const std::exception &e) {
        fail_cleaner("null_model", e.what());
    }
}

template <typename DerivedA, typename DerivedB, typename DerivedC>
auto Cleaner::get_marchenko_pastur_index(const Eigen::DenseBase<DerivedA> &scans,
                                         const Eigen::DenseBase<DerivedB> &flags,
                                         const Eigen::DenseBase<DerivedC> &apt_flags) {
    if (!marchenko_pastur.enabled) {
        return Eigen::Index{0};
    }

    try {
        const Eigen::Index n_pts_full = scans.rows();
        const Eigen::Index n_dets = scans.cols();
        if (n_pts_full < 8 || n_dets < 6) {
            fail_cleaner("marchenko_pastur", "insufficient data (n_pts=" + std::to_string(n_pts_full)
                                                   + ", n_dets=" + std::to_string(n_dets) + ")");
        }
        if (sample_rate_Hz <= 0.0 &&
            (marchenko_pastur.band_low_Hz > 0.0 || marchenko_pastur.band_high_Hz > 0.0)) {
            fail_cleaner("marchenko_pastur", "sample_rate_Hz=" + std::to_string(sample_rate_Hz)
                                                   + " invalid for band-limited covariance");
        }
        if (marchenko_pastur.band_high_Hz > 0.0 && marchenko_pastur.band_low_Hz > marchenko_pastur.band_high_Hz) {
            fail_cleaner("marchenko_pastur", "band_low_Hz exceeds band_high_Hz");
        }

        Eigen::Index sample_step = 1;
        if (marchenko_pastur.max_samples > 0 && n_pts_full > marchenko_pastur.max_samples) {
            sample_step = static_cast<Eigen::Index>(
                std::ceil(static_cast<double>(n_pts_full) / static_cast<double>(marchenko_pastur.max_samples)));
        }
        const Eigen::Index n_pts = (n_pts_full + sample_step - 1) / sample_step;
        if (n_pts < 8) {
            fail_cleaner("marchenko_pastur", "max_samples yields too few samples after subsampling");
        }
        const double dt_sec = (sample_rate_Hz > 0.0)
                                  ? static_cast<double>(sample_step) / sample_rate_Hz
                                  : 0.0;

        auto is_good = [&](Eigen::Index i_sub, Eigen::Index j_det) {
            const Eigen::Index i = i_sub * sample_step;
            return !flags.derived()(i, j_det);
        };

        std::vector<Eigen::Index> keep_frac;
        keep_frac.reserve(static_cast<std::size_t>(n_dets));
        for (Eigen::Index j = 0; j < n_dets; ++j) {
            if (apt_flags.derived()(j) != 0) {
                continue;
            }
            double good_count = 0.0;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (is_good(i, j)) {
                    good_count += 1.0;
                }
            }
            const double frac = good_count / static_cast<double>(n_pts);
            if (good_count > 1.0 && frac >= marchenko_pastur.min_good_frac) {
                keep_frac.push_back(j);
            }
        }
        if (keep_frac.size() < 6) {
            fail_cleaner("marchenko_pastur", "only " + std::to_string(keep_frac.size())
                                                   + " detector(s) pass min_good_frac=" + std::to_string(marchenko_pastur.min_good_frac));
        }

        const Eigen::Index n_keep_frac = static_cast<Eigen::Index>(keep_frac.size());
        Eigen::VectorXd centers = Eigen::VectorXd::Zero(n_keep_frac);
        std::vector<Eigen::Index> keep_std;
        keep_std.reserve(static_cast<std::size_t>(n_keep_frac));
        for (Eigen::Index j = 0; j < n_keep_frac; ++j) {
            const Eigen::Index det_j = keep_frac[static_cast<std::size_t>(j)];
            std::vector<double> vals;
            vals.reserve(static_cast<std::size_t>(n_pts));
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (is_good(i, det_j)) {
                    vals.push_back(scans.derived()(i * sample_step, det_j));
                }
            }
            if (vals.size() < 8) {
                continue;
            }
            Eigen::VectorXd vv = Eigen::Map<Eigen::VectorXd>(vals.data(), static_cast<Eigen::Index>(vals.size()));
            const double center = detail::robust_center(vv);
            const double sigma = detail::robust_scale(vv, center);
            if (std::isfinite(center) && sigma > 0.0) {
                centers(j) = center;
                keep_std.push_back(j);
            }
        }
        if (keep_std.size() < 6) {
            fail_cleaner("marchenko_pastur", "only " + std::to_string(keep_std.size())
                                                   + " detector(s) have robust finite scale");
        }

        const Eigen::Index n_used = static_cast<Eigen::Index>(keep_std.size());
        Eigen::MatrixXd centered = Eigen::MatrixXd::Zero(n_pts, n_used);
        Eigen::MatrixXd good_used = Eigen::MatrixXd::Zero(n_pts, n_used);
        for (Eigen::Index k = 0; k < n_used; ++k) {
            const Eigen::Index j = keep_std[static_cast<std::size_t>(k)];
            const Eigen::Index det_j = keep_frac[static_cast<std::size_t>(j)];
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                const bool good = is_good(i, det_j);
                good_used(i, k) = good ? 1.0 : 0.0;
                if (good) {
                    centered(i, k) = scans.derived()(i * sample_step, det_j) - centers(j);
                }
            }
        }

        const auto [min_good_frac_final, med_good_frac_final] = detail::good_fraction_stats(good_used);
        const bool allow_banded_cov =
            (marchenko_pastur.band_low_Hz > 0.0 || marchenko_pastur.band_high_Hz > 0.0) &&
            dt_sec > 0.0 &&
            std::isfinite(min_good_frac_final) &&
            std::isfinite(med_good_frac_final) &&
            min_good_frac_final >= 0.90 &&
            med_good_frac_final >= 0.95;

        if ((marchenko_pastur.band_low_Hz > 0.0 || marchenko_pastur.band_high_Hz > 0.0) &&
            dt_sec > 0.0 && !allow_banded_cov) {
            logger->debug(
                "marchenko_pastur: skipping band-limited covariance due to gappy flags "
                "(min_good_frac={:.4f} med_good_frac={:.4f})",
                min_good_frac_final, med_good_frac_final);
        }

        if (allow_banded_cov) {
            const Eigen::Index n_freq = n_pts / 2 + 1;
            Eigen::VectorXd freqs = Eigen::VectorXd::LinSpaced(
                n_freq, 0.0, static_cast<double>(n_freq - 1) / (dt_sec * static_cast<double>(n_pts)));
            Eigen::Array<bool, Eigen::Dynamic, 1> keep = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(n_freq, true);
            if (marchenko_pastur.band_low_Hz > 0.0) {
                keep = keep && (freqs.array() >= marchenko_pastur.band_low_Hz);
            }
            if (marchenko_pastur.band_high_Hz > 0.0) {
                keep = keep && (freqs.array() <= marchenko_pastur.band_high_Hz);
            }
            if (keep.any()) {
                Eigen::FFT<double> fft;
                fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
                fft.SetFlag(Eigen::FFT<double>::Unscaled);
                for (Eigen::Index j = 0; j < n_used; ++j) {
                    Eigen::VectorXcd spec;
                    fft.fwd(spec, centered.col(j));
                    for (Eigen::Index i = 0; i < spec.size(); ++i) {
                        if (!keep(i)) {
                            spec(i) = std::complex<double>(0.0, 0.0);
                        }
                    }
                    Eigen::VectorXd filtered;
                    fft.inv(filtered, spec, n_pts);
                    centered.col(j) = filtered;
                }
            }
        }

        std::vector<Eigen::Index> keep_rescaled;
        keep_rescaled.reserve(static_cast<std::size_t>(n_used));
        Eigen::VectorXd scales = Eigen::VectorXd::Zero(n_used);
        for (Eigen::Index j = 0; j < n_used; ++j) {
            std::vector<double> vals;
            vals.reserve(static_cast<std::size_t>(n_pts));
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (good_used(i, j) > 0.5) {
                    vals.push_back(centered(i, j));
                }
            }
            if (vals.size() < 8) {
                continue;
            }
            Eigen::VectorXd vv = Eigen::Map<Eigen::VectorXd>(vals.data(), static_cast<Eigen::Index>(vals.size()));
            const double scale = detail::robust_scale(vv, 0.0);
            if (std::isfinite(scale) && scale > 0.0) {
                scales(j) = scale;
                keep_rescaled.push_back(j);
            }
        }
        if (keep_rescaled.size() < 6) {
            fail_cleaner("marchenko_pastur", "only " + std::to_string(keep_rescaled.size())
                                                   + " detector(s) remain after re-scaling");
        }

        const Eigen::Index n_final = static_cast<Eigen::Index>(keep_rescaled.size());
        Eigen::MatrixXd z = Eigen::MatrixXd::Zero(n_pts, n_final);
        Eigen::MatrixXd good_final = Eigen::MatrixXd::Zero(n_pts, n_final);
        for (Eigen::Index k = 0; k < n_final; ++k) {
            const Eigen::Index j = keep_rescaled[static_cast<std::size_t>(k)];
            z.col(k) = centered.col(j) / scales(j);
            good_final.col(k) = good_used.col(j);
        }
        if (std::isfinite(marchenko_pastur.clip_z) && marchenko_pastur.clip_z > 0.0) {
            z = z.array().min(marchenko_pastur.clip_z).max(-marchenko_pastur.clip_z).matrix();
        }

        Eigen::MatrixXd cov = calc_cov_with_mask(z, good_final);
        Eigen::MatrixXd overlap = good_final.adjoint() * good_final;
        const double n_eff_mp = detail::median_positive_overlap_count(overlap);
        if (!(std::isfinite(n_eff_mp) && n_eff_mp >= 8.0)) {
            fail_cleaner("marchenko_pastur", "failed to estimate effective sample count from flag overlap");
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
        if (solver.info() != Eigen::Success) {
            fail_cleaner("marchenko_pastur", "failed to compute eigenspectrum");
        }
        Eigen::VectorXd evals = solver.eigenvalues().reverse();
        Eigen::Index n_modes = evals.size();
        if (marchenko_pastur.max_modes > 0) {
            n_modes = std::min<Eigen::Index>(n_modes, static_cast<Eigen::Index>(marchenko_pastur.max_modes));
        }
        if (n_modes < 8) {
            fail_cleaner("marchenko_pastur", "not enough modes available after max_modes truncation");
        }
        evals = evals.head(n_modes);

        auto fit = detail::fit_mp_bulk(evals, n_eff_mp, marchenko_pastur.bulk_keep_frac, marchenko_pastur.q_grid_size);
        if (!(std::isfinite(fit.lambda_plus) && fit.lambda_plus > 0.0 &&
              std::isfinite(fit.q_fit) && std::isfinite(fit.n_eff_fit))) {
            fail_cleaner("marchenko_pastur", "failed to fit MP bulk");
        }
        Eigen::Index k_mp = std::min<Eigen::Index>(fit.k_mp, n_dets - 1);
        if (k_mp < 0) {
            k_mp = 0;
        }

        logger->debug(
            "marchenko_pastur: n_det_input={} n_det_used={} n_modes={} n_pts={} step={} "
            "n_eff_mp={:.4g} min_good_frac={:.4f} med_good_frac={:.4f} k={} q_fit={:.4g} lambda_plus={:.4g} top_over_edge={:.4g}",
            n_dets, n_final, n_modes, n_pts, sample_step, n_eff_mp, min_good_frac_final, med_good_frac_final,
            k_mp, fit.q_fit, fit.lambda_plus, fit.top_over_edge);
        return k_mp;
    }
    catch (const std::bad_alloc &) {
        fail_cleaner("marchenko_pastur", "memory allocation failed");
    }
    catch (const std::exception &e) {
        fail_cleaner("marchenko_pastur", e.what());
    }
}

template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
auto Cleaner::select_adaptive_cut(const Eigen::DenseBase<DerivedA> &scans,
                                  const Eigen::DenseBase<DerivedB> &flags,
                                  const Eigen::DenseBase<DerivedC> &apt_flags,
                                  const Eigen::DenseBase<DerivedD> &evecs,
                                  const Eigen::Index baseline_k,
                                  const std::string &group_name,
                                  const Eigen::Index group_key,
                                  const Eigen::Index arr_index) const {

    using clock_t = std::chrono::steady_clock;
    const auto t0 = clock_t::now();

    AdaptiveSelectorResult result;
    result.baseline_k = baseline_k;
    result.chosen_k = baseline_k;
    result.n_det_input = scans.cols();

    try {
        Eigen::Index n_evec_solved = 0;
        for (Eigen::Index c = 0; c < evecs.cols(); ++c) {
            if (evecs.col(c).squaredNorm() > 0.0) {
                ++n_evec_solved;
            }
        }
        const Eigen::Index max_k = std::max<Eigen::Index>(
            0, std::min<Eigen::Index>(scans.cols() - 1, n_evec_solved));
        auto candidate_ks = adaptive_selector_candidate_cuts(baseline_k, max_k);
        result.n_candidates = static_cast<Eigen::Index>(candidate_ks.size());
        if (candidate_ks.empty()) {
            result.fallback = true;
            return result;
        }

        const Eigen::Index n_pts_full = scans.rows();
        const Eigen::Index n_dets = scans.cols();
        Eigen::Index sample_step = 1;
        if (adaptive_selector.max_samples > 0 &&
            n_pts_full > static_cast<Eigen::Index>(adaptive_selector.max_samples)) {
            sample_step = static_cast<Eigen::Index>(std::ceil(
                static_cast<double>(n_pts_full) / static_cast<double>(adaptive_selector.max_samples)));
        }
        sample_step = std::max<Eigen::Index>(sample_step, 1);
        const Eigen::Index n_pts = (n_pts_full + sample_step - 1) / sample_step;
        if (n_pts < 8 || n_dets < 2) {
            result.fallback = true;
            return result;
        }

        auto finite_or_nan = [](double v) {
            return std::isfinite(v) ? v : std::numeric_limits<double>::quiet_NaN();
        };
        auto safe_ratio = [&](double num, double den) {
            if (!std::isfinite(num) || !std::isfinite(den) || den == 0.0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return num / den;
        };

        auto gaussian_tail_prob = [](double threshold) {
            return std::erfc(threshold / std::sqrt(2.0));
        };

        auto apply_cut = [&](Eigen::Index limit_index) {
            if (limit_index <= 0) {
                return scans.derived().template cast<double>().eval();
            }
            const Eigen::Index k_use = std::min<Eigen::Index>(limit_index, evecs.cols());
            Eigen::MatrixXd good = (flags.derived().template cast<double>().array() == 0.0)
                                       .template cast<double>()
                                       .matrix();
            auto evecs_cut = evecs.derived().leftCols(k_use);
            Eigen::MatrixXd proj = (scans.derived().array() * good.array()).matrix() * evecs_cut;
            Eigen::MatrixXd model = proj * evecs_cut.adjoint();
            return (scans.derived().template cast<double>() -
                    (model.array() * good.array()).matrix())
                .eval();
        };

        auto normalize_terms = [](const std::vector<double> &vals) {
            std::vector<double> norm(vals.size(), std::numeric_limits<double>::quiet_NaN());
            double vmin = std::numeric_limits<double>::infinity();
            double vmax = -std::numeric_limits<double>::infinity();
            for (const auto v : vals) {
                if (std::isfinite(v)) {
                    vmin = std::min(vmin, v);
                    vmax = std::max(vmax, v);
                }
            }
            if (!std::isfinite(vmin) || !std::isfinite(vmax)) {
                return norm;
            }
            if (!(vmax > vmin)) {
                for (std::size_t i = 0; i < vals.size(); ++i) {
                    if (std::isfinite(vals[i])) {
                        norm[i] = 0.0;
                    }
                }
                return norm;
            }
            for (std::size_t i = 0; i < vals.size(); ++i) {
                if (std::isfinite(vals[i])) {
                    norm[i] = (vals[i] - vmin) / (vmax - vmin);
                }
            }
            return norm;
        };

        std::vector<double> corr_terms;
        std::vector<double> low_terms;
        std::vector<double> tail_terms;
        std::vector<double> top_terms;
        corr_terms.reserve(candidate_ks.size());
        low_terms.reserve(candidate_ks.size());
        tail_terms.reserve(candidate_ks.size());
        top_terms.reserve(candidate_ks.size());

        double candidate_eval_ms = 0.0;

        for (const auto k_candidate : candidate_ks) {
            const auto cand_t0 = clock_t::now();
            AdaptiveSelectorCandidateDiag diag;
            diag.k = k_candidate;
            diag.sample_step = sample_step;

            Eigen::MatrixXd cleaned = apply_cut(k_candidate);

            std::vector<Eigen::Index> det_keep_frac;
            det_keep_frac.reserve(static_cast<std::size_t>(n_dets));
            for (Eigen::Index j = 0; j < n_dets; ++j) {
                if (apt_flags.derived()(j) != 0) {
                    continue;
                }
                double good_count = 0.0;
                for (Eigen::Index is = 0; is < n_pts; ++is) {
                    const Eigen::Index i = is * sample_step;
                    if (i >= n_pts_full) {
                        break;
                    }
                    if (!flags.derived()(i, j) && std::isfinite(cleaned(i, j))) {
                        good_count += 1.0;
                    }
                }
                const double frac = good_count / static_cast<double>(n_pts);
                if (good_count > 1.0 && frac >= adaptive_selector.min_good_frac) {
                    det_keep_frac.push_back(j);
                }
            }

            auto det_sample_idx = detail::downsample_even_indices(
                static_cast<Eigen::Index>(det_keep_frac.size()), adaptive_selector.max_det);
            std::vector<Eigen::Index> det_keep;
            det_keep.reserve(det_sample_idx.size());
            for (const auto idx : det_sample_idx) {
                det_keep.push_back(det_keep_frac[static_cast<std::size_t>(idx)]);
            }

            if (det_keep.size() >= 6) {
                std::vector<Eigen::Index> det_final;
                std::vector<double> centers;
                std::vector<double> scales;
                det_final.reserve(det_keep.size());
                centers.reserve(det_keep.size());
                scales.reserve(det_keep.size());

                for (const auto det : det_keep) {
                    std::vector<double> vals;
                    vals.reserve(static_cast<std::size_t>(n_pts));
                    for (Eigen::Index is = 0; is < n_pts; ++is) {
                        const Eigen::Index i = is * sample_step;
                        if (i >= n_pts_full) {
                            break;
                        }
                        if (flags.derived()(i, det)) {
                            continue;
                        }
                        const double v = cleaned(i, det);
                        if (std::isfinite(v)) {
                            vals.push_back(v);
                        }
                    }
                    if (vals.size() < 8) {
                        continue;
                    }
                    Eigen::Map<const Eigen::VectorXd> vv(vals.data(), static_cast<Eigen::Index>(vals.size()));
                    const double center = detail::robust_center(vv);
                    const double scale = detail::robust_scale(vv, center);
                    if (std::isfinite(center) && std::isfinite(scale) && scale > 0.0) {
                        det_final.push_back(det);
                        centers.push_back(center);
                        scales.push_back(scale);
                    }
                }

                if (det_final.size() >= 6) {
                    const Eigen::Index n_det_used = static_cast<Eigen::Index>(det_final.size());
                    Eigen::MatrixXd centered = Eigen::MatrixXd::Zero(n_pts, n_det_used);
                    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> valid =
                        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(n_pts, n_det_used, false);
                    for (Eigen::Index k = 0; k < n_det_used; ++k) {
                        const auto det = det_final[static_cast<std::size_t>(k)];
                        const auto center = centers[static_cast<std::size_t>(k)];
                        for (Eigen::Index is = 0; is < n_pts; ++is) {
                            const Eigen::Index i = is * sample_step;
                            if (i >= n_pts_full) {
                                break;
                            }
                            if (flags.derived()(i, det)) {
                                continue;
                            }
                            const double v = cleaned(i, det);
                            if (!std::isfinite(v)) {
                                continue;
                            }
                            centered(is, k) = v - center;
                            valid(is, k) = true;
                        }
                    }

                    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(n_pts, n_det_used);
                    for (Eigen::Index k = 0; k < n_det_used; ++k) {
                        const auto scale = scales[static_cast<std::size_t>(k)];
                        z.col(k) = centered.col(k) / scale;
                    }
                    if (std::isfinite(adaptive_selector.clip_z) && adaptive_selector.clip_z > 0.0) {
                        z = z.array()
                                .min(adaptive_selector.clip_z)
                                .max(-adaptive_selector.clip_z)
                                .matrix();
                    }

                    diag.n_det_used = n_det_used;
                    diag.n_time_used = n_pts;
                    diag.valid_frac = valid.cast<double>().mean();

                    std::vector<double> z_valid;
                    z_valid.reserve(static_cast<std::size_t>(n_pts * n_det_used));
                    for (Eigen::Index is = 0; is < n_pts; ++is) {
                        for (Eigen::Index jd = 0; jd < n_det_used; ++jd) {
                            if (valid(is, jd) && std::isfinite(z(is, jd))) {
                                z_valid.push_back(z(is, jd));
                            }
                        }
                    }

                    std::vector<double> abs_corrs;
                    const std::uint64_t n_pairs_total = static_cast<std::uint64_t>(n_det_used) *
                                                        static_cast<std::uint64_t>(n_det_used - 1) / 2ULL;
                    std::uint64_t target_pairs = n_pairs_total;
                    if (adaptive_selector.max_pairs > 0) {
                        target_pairs = std::min<std::uint64_t>(
                            n_pairs_total, static_cast<std::uint64_t>(adaptive_selector.max_pairs));
                    }

                    auto pair_corr_for = [&](Eigen::Index a, Eigen::Index b) {
                        double dot = 0.0;
                        for (Eigen::Index is = 0; is < n_pts; ++is) {
                            dot += z(is, a) * z(is, b);
                        }
                        return dot / std::max<double>(1.0, static_cast<double>(n_pts - 1));
                    };

                    if (target_pairs == n_pairs_total) {
                        abs_corrs.reserve(static_cast<std::size_t>(target_pairs));
                        for (Eigen::Index a = 0; a < n_det_used; ++a) {
                            for (Eigen::Index b = a + 1; b < n_det_used; ++b) {
                                const auto corr = pair_corr_for(a, b);
                                if (std::isfinite(corr)) {
                                    abs_corrs.push_back(std::abs(corr));
                                }
                            }
                        }
                    }
                    else if (target_pairs > 0) {
                        abs_corrs.reserve(static_cast<std::size_t>(target_pairs));
                        const std::uint64_t seed_mix =
                            static_cast<std::uint64_t>(adaptive_selector.seed) ^
                            (static_cast<std::uint64_t>(group_key + 1) * 2654435761ULL) ^
                            (static_cast<std::uint64_t>(arr_index + 1) * 2246822519ULL) ^
                            (static_cast<std::uint64_t>(k_candidate + 1) * 1315423911ULL);
                        std::mt19937 rng(static_cast<std::uint32_t>(seed_mix & 0xffffffffULL));
                        std::uniform_int_distribution<Eigen::Index> det_dist(0, n_det_used - 1);
                        std::unordered_set<std::uint64_t> seen_pairs;
                        seen_pairs.reserve(static_cast<std::size_t>(target_pairs * 2 + 1));
                        std::uint64_t tries = 0;
                        const std::uint64_t max_tries = std::max<std::uint64_t>(target_pairs * 32ULL, 1024ULL);
                        while (seen_pairs.size() < target_pairs && tries < max_tries) {
                            tries++;
                            Eigen::Index a = det_dist(rng);
                            Eigen::Index b = det_dist(rng);
                            if (a == b) {
                                continue;
                            }
                            if (a > b) {
                                std::swap(a, b);
                            }
                            const auto key = (static_cast<std::uint64_t>(a) << 32ULL) |
                                             static_cast<std::uint64_t>(b);
                            if (!seen_pairs.insert(key).second) {
                                continue;
                            }
                            const auto corr = pair_corr_for(a, b);
                            if (std::isfinite(corr)) {
                                abs_corrs.push_back(std::abs(corr));
                            }
                        }
                    }
                    diag.med_abs_corr = detail::median_from_values(std::move(abs_corrs));

                    if (n_pts >= 8) {
                        Eigen::MatrixXd cov = (z.adjoint() * z) / std::max<double>(1.0, static_cast<double>(n_pts - 1));
                        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
                        if (solver.info() == Eigen::Success) {
                            Eigen::VectorXd evals_cov = solver.eigenvalues().reverse();
                            double eval_sum = 0.0;
                            for (Eigen::Index i = 0; i < evals_cov.size(); ++i) {
                                if (std::isfinite(evals_cov(i)) && evals_cov(i) > 0.0) {
                                    eval_sum += evals_cov(i);
                                }
                            }
                            if (eval_sum > 0.0 && std::isfinite(evals_cov(0))) {
                                diag.top_mode_frac = evals_cov(0) / eval_sum;
                            }
                        }
                    }

                    if (!z_valid.empty()) {
                        const double p = gaussian_tail_prob(4.0);
                        const double n = static_cast<double>(z_valid.size());
                        double count = 0.0;
                        for (const auto v : z_valid) {
                            if (std::abs(v) > 4.0) {
                                count += 1.0;
                            }
                        }
                        const double var = n * p * std::max(1.0 - p, 0.0);
                        if (var > 0.0) {
                            diag.tail4_binom_z = (count - n * p) / std::sqrt(var);
                        }
                    }

                    if (sample_rate_Hz > 0.0 && n_pts >= 16) {
                        Eigen::VectorXd common_mode = Eigen::VectorXd::Zero(n_pts);
                        for (Eigen::Index is = 0; is < n_pts; ++is) {
                            std::vector<double> row_vals;
                            row_vals.reserve(static_cast<std::size_t>(n_det_used));
                            for (Eigen::Index jd = 0; jd < n_det_used; ++jd) {
                                row_vals.push_back(centered(is, jd));
                            }
                            common_mode(is) = detail::median_from_values(std::move(row_vals));
                        }
                        const double cm_mean = common_mode.mean();
                        Eigen::VectorXd x = common_mode.array() - cm_mean;
                        if (n_pts > 1) {
                            constexpr double two_pi = 6.283185307179586476925286766559;
                            for (Eigen::Index is = 0; is < n_pts; ++is) {
                                const double w = 0.5 * (1.0 - std::cos(
                                    two_pi * static_cast<double>(is) /
                                    static_cast<double>(n_pts - 1)));
                                x(is) *= w;
                            }
                        }

                        Eigen::FFT<double> fft;
                        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
                        fft.SetFlag(Eigen::FFT<double>::Unscaled);
                        Eigen::VectorXcd freq;
                        fft.fwd(freq, x);

                        const double fs_eff = sample_rate_Hz / static_cast<double>(sample_step);
                        std::vector<double> p_low_vals;
                        std::vector<double> p_mid_vals;
                        for (Eigen::Index fi = 1; fi < freq.size(); ++fi) {
                            const double hz = static_cast<double>(fi) * fs_eff / static_cast<double>(n_pts);
                            const double power = std::norm(freq(fi));
                            if (hz >= adaptive_selector.low_band_Hz[0] && hz < adaptive_selector.low_band_Hz[1]) {
                                p_low_vals.push_back(power);
                            }
                            if (hz >= adaptive_selector.mid_band_Hz[0] && hz < adaptive_selector.mid_band_Hz[1]) {
                                p_mid_vals.push_back(power);
                            }
                        }
                        const double p_low = detail::median_from_values(std::move(p_low_vals));
                        const double p_mid = detail::median_from_values(std::move(p_mid_vals));
                        diag.cm_low_mid_ratio = safe_ratio(p_low, p_mid);
                    }
                }
            }

            const double corr_term = std::isfinite(diag.med_abs_corr)
                ? std::max(diag.med_abs_corr, 0.0)
                : std::numeric_limits<double>::quiet_NaN();
            const double low_term = (std::isfinite(diag.cm_low_mid_ratio) && diag.cm_low_mid_ratio > 0.0)
                ? std::max(std::log2(diag.cm_low_mid_ratio), 0.0)
                : std::numeric_limits<double>::quiet_NaN();
            const double tail_term = std::isfinite(diag.tail4_binom_z)
                ? std::max(diag.tail4_binom_z, 0.0)
                : std::numeric_limits<double>::quiet_NaN();
            const double top_term = diag.top_mode_frac;
            corr_terms.push_back(corr_term);
            low_terms.push_back(low_term);
            tail_terms.push_back(tail_term);
            top_terms.push_back(top_term);

            const auto cand_t1 = clock_t::now();
            diag.elapsed_msec = std::chrono::duration<double, std::milli>(cand_t1 - cand_t0).count();
            candidate_eval_ms += diag.elapsed_msec;
            result.candidates.push_back(diag);

            if (adaptive_selector.log_candidates) {
                logger->debug(
                    "adaptive_selector candidate grouping={} key={} array={} k={} det_used={} time_used={} step={} valid_frac={:.4f} med_abs_corr={:.4g} cm_low_mid_ratio={:.4g} tail4_binom_z={:.4g} top_mode_frac={:.4f} eval_ms={:.2f}",
                    group_name, group_key, arr_index, diag.k, diag.n_det_used, diag.n_time_used,
                    diag.sample_step, finite_or_nan(diag.valid_frac), finite_or_nan(diag.med_abs_corr),
                    finite_or_nan(diag.cm_low_mid_ratio), finite_or_nan(diag.tail4_binom_z),
                    finite_or_nan(diag.top_mode_frac), finite_or_nan(diag.elapsed_msec));
            }
        }

        auto corr_norm = normalize_terms(corr_terms);
        auto low_norm = normalize_terms(low_terms);
        auto tail_norm = normalize_terms(tail_terms);
        auto top_norm = normalize_terms(top_terms);
        const auto k_step = (candidate_ks.size() >= 2)
            ? std::max<Eigen::Index>(1, candidate_ks[1] - candidate_ks[0])
            : Eigen::Index{1};

        std::vector<double> scores(result.candidates.size(), std::numeric_limits<double>::quiet_NaN());
        for (std::size_t i = 0; i < result.candidates.size(); ++i) {
            const double corr_score = std::isfinite(corr_norm[i]) ? corr_norm[i] : 1.0;
            const double low_score = std::isfinite(low_norm[i]) ? low_norm[i] : 1.0;
            const double tail_score = std::isfinite(tail_norm[i]) ? tail_norm[i] : 1.0;
            const double top_score = std::isfinite(top_norm[i]) ? top_norm[i] : 1.0;
            const double reg_score =
                static_cast<double>(std::abs(result.candidates[i].k - baseline_k)) /
                static_cast<double>(k_step);
            const double score =
                corr_score +
                adaptive_selector.low_weight * low_score +
                adaptive_selector.tail_weight * tail_score +
                adaptive_selector.topmode_weight * top_score +
                adaptive_selector.reg_weight * reg_score;
            result.candidates[i].score = score;
            scores[i] = score;
        }

        if (scores.empty()) {
            result.fallback = true;
            return result;
        }

        const auto best_it = std::min_element(scores.begin(), scores.end(), [](double a, double b) {
            if (!std::isfinite(a)) {
                return false;
            }
            if (!std::isfinite(b)) {
                return true;
            }
            return a < b;
        });
        if (best_it == scores.end() || !std::isfinite(*best_it)) {
            result.fallback = true;
            return result;
        }
        const auto best_idx = static_cast<std::size_t>(std::distance(scores.begin(), best_it));
        result.used = true;
        result.chosen_k = result.candidates[best_idx].k;
        result.chosen_score = result.candidates[best_idx].score;
        result.chosen_diag = result.candidates[best_idx];
        result.candidate_eval_msec = candidate_eval_ms;
        result.chosen_cleaned_scans = apply_cut(result.chosen_k);

        std::vector<std::pair<double, Eigen::Index>> ranked;
        ranked.reserve(result.candidates.size());
        for (const auto &diag : result.candidates) {
            ranked.emplace_back(diag.score, diag.k);
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
            if (!std::isfinite(a.first)) {
                return false;
            }
            if (!std::isfinite(b.first)) {
                return true;
            }
            if (a.first != b.first) {
                return a.first < b.first;
            }
            return a.second < b.second;
        });
        if (ranked.size() >= 2 && std::isfinite(ranked[1].first)) {
            result.runnerup_score = ranked[1].first;
            result.runnerup_k = ranked[1].second;
            result.score_margin = ranked[1].first - ranked[0].first;
        }

        const auto t1 = clock_t::now();
        const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        logger->info(
            "adaptive_selector grouping={} key={} array={} baseline_k={} chosen_k={} runnerup_k={} margin={} det_in={} det_used={} time_used={} n_candidates={} candidate_ms={} total_ms={}",
            group_name, group_key, arr_index, baseline_k, result.chosen_k, result.runnerup_k,
            finite_or_nan(result.score_margin), result.n_det_input, result.chosen_diag.n_det_used,
            result.chosen_diag.n_time_used, result.n_candidates, candidate_eval_ms, total_ms);
        return result;
    }
    catch (const std::exception &e) {
        logger->warn(
            "adaptive_selector failed for grouping={} key={} array={} baseline_k={}; "
            "falling back to configured PCA cut: {}",
            group_name, group_key, arr_index, baseline_k, e.what());
        result.fallback = true;
        return result;
    }
}

template <Cleaner::EigenSolverBackend backend, typename DerivedA, typename DerivedB, typename DerivedC>
auto Cleaner::calc_eig_values(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags,
                              Eigen::DenseBase<DerivedC> &apt_flags, const Eigen::Index group_n_eig) {

    // dimensions
    Eigen::Index n_pts = scans.rows();
    Eigen::Index n_dets = scans.cols();

    std::vector<Eigen::Index> active_cols;
    active_cols.reserve(static_cast<std::size_t>(n_dets));
    for (Eigen::Index i=0; i<n_dets; i++) {
        if (apt_flags.derived()(i) != 0) {
            continue;
        }
        Eigen::Index n_good = 0;
        for (Eigen::Index j=0; j<n_pts; j++) {
            if (!flags.derived()(j, i)) {
                n_good++;
            }
        }
        if (n_good > 1) {
            active_cols.push_back(i);
        }
    }

    const Eigen::Index n_active_dets = static_cast<Eigen::Index>(active_cols.size());
    const int n_active_dets_int = static_cast<int>(n_active_dets);
    if (n_active_dets != n_dets) {
        logger->debug("PCA covariance using {}/{} active detectors", n_active_dets, n_dets);
    }

    if (n_active_dets <= 1) {
        logger->warn("PCA covariance has only {} active detector(s); skipping eigen solve", n_active_dets);
        Eigen::VectorXd evals = Eigen::VectorXd::Zero(n_dets);
        Eigen::MatrixXd evecs = Eigen::MatrixXd::Zero(n_dets, n_dets);
        return std::tuple<Eigen::VectorXd, Eigen::MatrixXd> {evals, evecs};
    }

    Eigen::MatrixXd f_active(n_pts, n_active_dets);
    for (Eigen::Index i=0; i<n_active_dets; i++) {
        const auto src_col = active_cols[static_cast<std::size_t>(i)];
        for (Eigen::Index j=0; j<n_pts; j++) {
            f_active(j, i) = flags.derived()(j, src_col) ? 0.0 : 1.0;
        }
    }

    // container for covariance matrix
    Eigen::MatrixXd pca_cov(n_active_dets, n_active_dets);

    // number of unflagged samples
    auto denom = (f_active.adjoint() * f_active).array() - 1;

    // multiply scans by flags to remove flagged signal
    Eigen::MatrixXd det(n_pts, n_active_dets);
    for (Eigen::Index i=0; i<n_active_dets; i++) {
        const auto src_col = active_cols[static_cast<std::size_t>(i)];
        det.col(i) = (scans.derived().col(src_col).array() * f_active.col(i).array()).matrix();
    }

    // calculate the covariance matrix with safe denominator handling
    Eigen::MatrixXd numer = det.adjoint() * det;
    pca_cov = (denom.array() > 0).select(numer.array() / denom.array(), 0.0);

    /*Eigen::VectorXd avg_corrs(n_dets);
    avg_corrs.setZero();

    // remove weakly correlated detectors
    for (Eigen::Index i=0; i<n_dets; i++) {
        for (Eigen::Index j=0; j<n_dets; j++) {
            if (i!=j) {
                avg_corrs(i) += pca_cov(i,j);
            }
        }
        avg_corrs(i) /= (n_dets - 1);
    }

    double mean_corr = avg_corrs.mean();
*/

    //logger->info("average correlations {}", avg_corrs);
    //logger->info("mean global correlation {}", mean_corr);

    /*for (Eigen::Index i=0; i<n_dets; i++) {
        if (avg_corrs(i) < tau*mean_corr) {
            pca_cov.row(i).setZero();
            pca_cov.col(i).setZero();
        }
    }*/

    // eigenvalues
    Eigen::VectorXd evals;
    // eigenvectors
    Eigen::MatrixXd evecs;

    if constexpr (backend == SpectraBackend) {
        // determine how many modes must be solved for cleaning
        int n_ev = static_cast<int>(group_n_eig);
        if (adaptive_mode_selection_enabled()) {
            if (adaptive_mode_selection_max_modes() > 0) {
                n_ev = std::min<int>(adaptive_mode_selection_max_modes(), n_active_dets_int - 1);
            }
            else {
                n_ev = n_active_dets_int - 1;
            }
        }
        else if (stddev_limit > 0 && group_n_eig == 0) {
            n_ev = n_active_dets_int - 1;
        }
        else if (n_calc > 0) {
            n_ev = n_calc;
        }

        n_ev = std::min<int>(n_ev, n_active_dets_int - 1);
        if (n_ev <= 0) {
            evals = Eigen::VectorXd::Zero(n_dets);
            evecs = Eigen::MatrixXd::Zero(n_dets, n_dets);
            return std::tuple<Eigen::VectorXd, Eigen::MatrixXd> {evals, evecs};
        }

        if (stddev_limit > 0 && group_n_eig == 0 && n_calc > 0 && !adaptive_mode_selection_enabled()) {
            logger->warn("stddev_limit active but n_calc={} limits eigen spectrum; consider setting n_calc=0", n_calc);
        }
        if (adaptive_mode_selection_enabled() && n_calc > 0 && n_calc < n_ev) {
            logger->warn("{} enabled: ignoring n_calc={} for cleaning solve depth n_ev={}",
                         active_cleaner_label(), n_calc, n_ev);
        }

        // number of values to calculate
        int n_cv = std::min<int>(n_active_dets_int, std::max<int>(n_ev + 2, static_cast<int>(std::ceil(n_ev * 2.5))));
        if (n_cv <= n_ev) {
            n_cv = std::min<int>(n_active_dets_int, n_ev + 1);
        }
        if (n_cv <= n_ev) {
            throw std::runtime_error("invalid Spectra settings: n_cv <= n_ev");
        }

        // set up spectra
        Spectra::DenseSymMatProd<double> op(pca_cov);
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, n_ev, n_cv);

        eigs.init();
        // largest eigenvalues first
        int n_conv = eigs.compute(Spectra::SortRule::LargestAlge);

        // retrieve results
        evals = Eigen::VectorXd::Zero(n_dets);
        evecs = Eigen::MatrixXd::Zero(n_dets, n_dets);

        // copy the eigenvalues and eigenvectors
        if (eigs.info() == Spectra::CompInfo::Successful) {
            Eigen::VectorXd evals_sub = eigs.eigenvalues();
            Eigen::MatrixXd evecs_sub = eigs.eigenvectors();

            // enforce descending order for consistency with stddev cut
            std::vector<Eigen::Index> order(static_cast<std::size_t>(n_ev));
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](Eigen::Index a, Eigen::Index b) {
                return evals_sub(a) > evals_sub(b);
            });

            for (Eigen::Index k=0; k<n_ev; ++k) {
                evals(k) = evals_sub(order[static_cast<std::size_t>(k)]);
                for (Eigen::Index i=0; i<n_active_dets; i++) {
                    const auto dst_row = active_cols[static_cast<std::size_t>(i)];
                    evecs(dst_row, k) = evecs_sub(i, order[static_cast<std::size_t>(k)]);
                }
            }
        }
        else {
            throw std::runtime_error("spectra failed to compute eigen values");
        }
    }

    else if constexpr (backend == EigenBackend) {
        // use Eigen's eigen solver
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solution(pca_cov);

        // copy the eigenvalues and eigenvectors
        if (!solution.info()) {
            Eigen::VectorXd evals_active = solution.eigenvalues();
            Eigen::MatrixXd evecs_active = solution.eigenvectors();

            evals_active.reverseInPlace();
            evecs_active.rowwise().reverseInPlace();

            evals = Eigen::VectorXd::Zero(n_dets);
            evecs = Eigen::MatrixXd::Zero(n_dets, n_dets);
            for (Eigen::Index k=0; k<n_active_dets; k++) {
                evals(k) = evals_active(k);
                for (Eigen::Index i=0; i<n_active_dets; i++) {
                    const auto dst_row = active_cols[static_cast<std::size_t>(i)];
                    evecs(dst_row, k) = evecs_active(i, k);
                }
            }
        }
        else {
            throw std::runtime_error("eigen failed to compute eigen values");
        }
    }

    return std::tuple<Eigen::VectorXd, Eigen::MatrixXd> {evals, evecs};
}

template <Cleaner::EigenSolverBackend backend,typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
auto Cleaner::remove_eig_values(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags,
                                const Eigen::DenseBase<DerivedC> &evals, const Eigen::DenseBase<DerivedD> &evecs,
                                Eigen::DenseBase<DerivedA> &cleaned_scans, const Eigen::Index group_n_eig,
                                const Eigen::Index forced_limit_index, const std::string &group_name,
                                const Eigen::Index nw_index, const Eigen::Index arr_index) {

    // number of detectors
    Eigen::Index n_dets = scans.cols();

    // number of eigenvalues to remove
    Eigen::Index limit_index;

    if (forced_limit_index >= 0) {
        limit_index = forced_limit_index;
    }
    // if using std dev limit, calculate index
    else if (stddev_limit > 0) {
        int n_ev_available = n_dets;
        if constexpr (backend == SpectraBackend) {
            if (adaptive_mode_selection_enabled()) {
                if (adaptive_mode_selection_max_modes() > 0) {
                    n_ev_available = std::min<int>(adaptive_mode_selection_max_modes(), n_dets - 1);
                }
                else {
                    n_ev_available = n_dets - 1;
                }
            }
            else if (n_calc > 0) {
                n_ev_available = std::min<int>(n_calc, n_dets - 1);
            }
            else if (group_n_eig == 0) {
                n_ev_available = n_dets - 1;
            }
            else {
                n_ev_available = std::min<int>(group_n_eig, n_dets - 1);
            }
        }
        else if constexpr (backend == EigenBackend) {
            n_ev_available = (group_n_eig == 0) ? n_dets : std::min<int>(group_n_eig, n_dets);
        }
        // calculate index above which to remove eigenvalues
        limit_index = get_stddev_index(evals.head(n_ev_available));
    }
    // otherwise use number of eigenvalues from config
    else {
        limit_index = group_n_eig;
    }

    limit_index = std::max<Eigen::Index>(0, std::min<Eigen::Index>(limit_index, evecs.cols()));
    if constexpr (backend == SpectraBackend) {
        limit_index = std::min<Eigen::Index>(limit_index, n_dets - 1);
    }

    logger->debug("removing {} largest eigenvalue(s) grouping={} network={} array={}",
                  limit_index, group_name, nw_index, arr_index);

    // keep flagged samples out of the mode projection and subtraction.
    Eigen::MatrixXd good = (flags.derived().template cast<double>().array() == 0.0)
                               .template cast<double>()
                               .matrix();
    const auto evecs_cut = evecs.derived().leftCols(limit_index);
    Eigen::MatrixXd proj = (scans.derived().array() * good.array()).matrix() * evecs_cut;
    Eigen::MatrixXd model = proj * evecs_cut.adjoint();
    Eigen::MatrixXd cleaned = scans.derived() - (model.array() * good.array()).matrix();
    if (cleaned_scans.derived().data() == scans.derived().data()) {
        cleaned_scans.derived() = std::move(cleaned);
    }
    else {
        cleaned_scans.derived().noalias() = cleaned;
    }
    return limit_index;
}

} // namespace timestream
