#pragma once

#include <string>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Spectra/SymEigsSolver.h>

#include <citlali/core/utils/utils.h>

namespace timestream {

class Cleaner {
public:
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

    // grouping
    std::vector<std::string> grouping;

    template <typename Derived>
    auto get_stddev_index(const Eigen::DenseBase<Derived> &evals) {
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
                    if (abs(ev(i) - m_ev) > abs(stddev_limit*stddev)) {
                        good(i) = false;
                    }
                    else {
                        count++;
                    }
                }
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

    // zero out flagged detectors in apt table (used for per scan beammap flags)
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

    // calculate the covariance matrix
    pca_cov.noalias() = ((det.adjoint() * det).array() / denom.array()).matrix();

    // eigenvalues
    Eigen::VectorXd evals;
    // eigenvectors
    Eigen::MatrixXd evecs;

    if constexpr (backend == SpectraBackend) {
        // number of eigenvalues to remove
        int n_ev = group_n_eig;

        // if using std dev limit and n_eig_to_cut is zero, use all detectors (-1 for spectra requirement)
        if (stddev_limit > 0 && group_n_eig==0) {
            n_ev = n_dets - 1;
        }

        // number of values to calculate
        int n_cv;

        if (n_calc==0) {
            n_cv = n_ev * 2.5 < n_dets?int(n_ev * 2.5):n_dets;
        }
        else {
            n_cv = n_calc * 2.5 < n_dets?int(n_calc * 2.5):n_dets;
            n_ev = n_calc;
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
            evals.head(n_ev) = eigs.eigenvalues();
            evecs.leftCols(n_ev) = eigs.eigenvectors();
        }
        else {
            throw std::runtime_error("spectra failed to compute eigen values");
            std::exit(EXIT_FAILURE);
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
            std::exit(EXIT_FAILURE);
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
        int n_ev;
        // if using std dev limit and n_eig_to_cut is zero, use all detectors
        if (group_n_eig == 0) {
            if constexpr (backend == SpectraBackend) {
                n_ev = n_dets - 1;
            }
            else if constexpr (backend == EigenBackend) {
                n_ev = n_dets;
            }
        }
        // if n_eig_to_cut is not zero, calc std dev for those eigs only
        else {
            n_ev = group_n_eig;
        }
        // calculate index above which to remove eigenvalues
        limit_index = get_stddev_index(evals.head(n_ev));
    }
    // otherwise use number of eigenvalues from config
    else {
        limit_index = group_n_eig;
    }

    SPDLOG_DEBUG("removing {} largest eigenvalue(s)", limit_index);

    // subtract out the desired eigenvectors
    Eigen::MatrixXd proj = scans.derived() * evecs.derived().leftCols(limit_index);
    cleaned_scans.derived().noalias() = scans.derived() - proj * evecs.derived().adjoint().topRows(limit_index);
}

} // namespace timestream
