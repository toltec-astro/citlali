#pragma once

#include <string>
#include <algorithm>
#include <utility>
#include <numeric>
#include <cmath>
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

    // calculate the eigenvalues from a matrix while removing flags
    template <EigenSolverBackend backend, typename DerivedA, typename DerivedB, typename DerivedC>
    auto calc_eig_values(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &, Eigen::DenseBase<DerivedC> &,
                         const Eigen::Index);

    // remove computed eigenvalues from matrix and recompute tods
    template <EigenSolverBackend backend, typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    auto remove_eig_values(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &,
                           const Eigen::DenseBase<DerivedC> &, const Eigen::DenseBase<DerivedD> &,
                           Eigen::DenseBase<DerivedA> &, const Eigen::Index);
};

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
        // if using std dev limit and n_eig_to_cut is zero, use all detectors (-1 for spectra requirement)
        int n_ev = (stddev_limit > 0 && group_n_eig==0) ? n_dets - 1: group_n_eig;

        // number of values to calculate
        int n_cv;

        if (n_calc==0) {
            n_cv = n_ev * 2.5 < n_dets?int(n_ev * 2.5):n_dets;
        }
        else {
            n_cv = n_calc * 2.5 < n_dets?int(n_calc * 2.5):n_dets;
            n_ev = n_calc;
        }

        if (stddev_limit > 0 && group_n_eig == 0 && n_calc > 0) {
            logger->warn("stddev_limit active but n_calc={} limits eigen spectrum; consider setting n_calc=0", n_calc);
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
                                Eigen::DenseBase<DerivedA> &cleaned_scans, const Eigen::Index group_n_eig) {

    // number of detectors
    Eigen::Index n_dets = scans.cols();

    // number of eigenvalues to remove
    Eigen::Index limit_index;

    // if using std dev limit, calculate index
    if (stddev_limit > 0) {
        int n_ev_available = n_dets;
        if constexpr (backend == SpectraBackend) {
            if (n_calc > 0) {
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

    logger->debug("removing {} largest eigenvalue(s)", limit_index);

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
