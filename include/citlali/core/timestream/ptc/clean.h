#pragma once

#include <string>
#include <algorithm>
#include <utility>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <random>
#include <new>
#include <vector>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Spectra/SymEigsSolver.h>

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
        const bool use_abs = (corr_grouping.metric != "signed");
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
        logger->warn("null_model: n_surrogates={} is too small; skipping cut", null_model.n_surrogates);
        return Eigen::Index{0};
    }

    try {
        const Eigen::Index n_pts_full = scans.rows();
        const Eigen::Index n_dets = scans.cols();
        if (n_pts_full < 4 || n_dets < 2) {
            logger->warn("null_model: insufficient data (n_pts={}, n_dets={}); skipping cut", n_pts_full, n_dets);
            return Eigen::Index{0};
        }

        // Subsample in time to cap memory and runtime.
        Eigen::Index sample_step = 1;
        if (null_model.max_samples > 0 && n_pts_full > null_model.max_samples) {
            sample_step = static_cast<Eigen::Index>(
                std::ceil(static_cast<double>(n_pts_full) / static_cast<double>(null_model.max_samples)));
        }
        const Eigen::Index n_pts = (n_pts_full + sample_step - 1) / sample_step;
        if (n_pts < 4) {
            return Eigen::Index{0};
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
            logger->warn("null_model: only {} detector(s) pass min_good_frac={}; skipping cut",
                         keep_frac.size(), null_model.min_good_frac);
            return Eigen::Index{0};
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
            logger->warn("null_model: only {} detector(s) have finite non-zero stddev; skipping cut", keep_std.size());
            return Eigen::Index{0};
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
            logger->warn("null_model: failed to compute observed eigenspectrum; skipping cut");
            return Eigen::Index{0};
        }
        Eigen::VectorXd obs_evals = obs_solver.eigenvalues().reverse();

        Eigen::Index n_modes = obs_evals.size();
        if (null_model.max_modes > 0) {
            n_modes = std::min<Eigen::Index>(n_modes, static_cast<Eigen::Index>(null_model.max_modes));
        }
        if (n_modes < 2) {
            return Eigen::Index{0};
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
                logger->warn("null_model: surrogate eigensolver failed at trial {}; skipping cut", s);
                return Eigen::Index{0};
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
        logger->warn("null_model: memory allocation failed; skipping cut");
        return Eigen::Index{0};
    }
    catch (const std::exception &e) {
        logger->warn("null_model: exception {}; skipping cut", e.what());
        return Eigen::Index{0};
    }
}

template <Cleaner::EigenSolverBackend backend, typename DerivedA, typename DerivedB, typename DerivedC>
auto Cleaner::calc_eig_values(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags,
                              Eigen::DenseBase<DerivedC> &apt_flags, const Eigen::Index group_n_eig) {

    // dimensions
    Eigen::Index n_dets = scans.cols();

    // make copy of flags
    Eigen::MatrixXd f = abs(flags.derived().template cast<double> ().array() - 1);

    // zero out flagged detectors in apt table.  we need to do this because we want
    // to make maps of all detectors when in detector mode so the timestreams cannot
    // be completely flagged.
    for (Eigen::Index i=0; i<n_dets; i++) {
        if (apt_flags.derived()(i)!=0) {
            f.col(i).setZero();
        }
    }

    // container for covariance matrix
    Eigen::MatrixXd pca_cov(n_dets, n_dets);

    // number of unflagged samples
    auto denom = (f.adjoint() * f).array() - 1;

    // multiply scans by flags to remove flagged signal
    auto det = (scans.derived().array()*f.array()).matrix();

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
        if (null_model.enabled) {
            if (null_model.max_modes > 0) {
                n_ev = std::min<int>(null_model.max_modes, n_dets - 1);
            }
            else {
                n_ev = n_dets - 1;
            }
        }
        else if (stddev_limit > 0 && group_n_eig == 0) {
            n_ev = n_dets - 1;
        }
        else if (n_calc > 0) {
            n_ev = n_calc;
        }

        n_ev = std::min<int>(n_ev, n_dets - 1);
        if (n_ev <= 0) {
            evals = Eigen::VectorXd::Zero(n_dets);
            evecs = Eigen::MatrixXd::Identity(n_dets, n_dets);
            return std::tuple<Eigen::VectorXd, Eigen::MatrixXd> {evals, evecs};
        }

        if (stddev_limit > 0 && group_n_eig == 0 && n_calc > 0 && !null_model.enabled) {
            logger->warn("stddev_limit active but n_calc={} limits eigen spectrum; consider setting n_calc=0", n_calc);
        }
        if (null_model.enabled && n_calc > 0 && n_calc < n_ev) {
            logger->warn("null_model enabled: ignoring n_calc={} for cleaning solve depth n_ev={}", n_calc, n_ev);
        }

        // number of values to calculate
        int n_cv = std::min<int>(n_dets, std::max<int>(n_ev + 2, static_cast<int>(std::ceil(n_ev * 2.5))));
        if (n_cv <= n_ev) {
            n_cv = std::min<int>(n_dets, n_ev + 1);
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
                evecs.col(k) = evecs_sub.col(order[static_cast<std::size_t>(k)]);
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
            evals = solution.eigenvalues();
            evecs = solution.eigenvectors();

            evals.reverseInPlace();
            evecs.rowwise().reverseInPlace();
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
            if (null_model.enabled) {
                if (null_model.max_modes > 0) {
                    n_ev_available = std::min<int>(null_model.max_modes, n_dets - 1);
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

    // subtract out the desired eigenvectors
    Eigen::MatrixXd proj = scans.derived() * evecs.derived().leftCols(limit_index);
    Eigen::MatrixXd cleaned = scans.derived() - proj * evecs.derived().adjoint().topRows(limit_index);
    if (cleaned_scans.derived().data() == scans.derived().data()) {
        cleaned_scans.derived() = std::move(cleaned);
    }
    else {
        cleaned_scans.derived().noalias() = cleaned;
    }
}

} // namespace timestream
