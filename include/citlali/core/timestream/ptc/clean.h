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

    int n_eig_to_cut;
    double stddev_limit;
    std::string grouping;

    template <typename Derived>
    auto get_stddev_index(const Eigen::DenseBase<Derived> &evals) {
        // copy eigenvalues
        Eigen::VectorXd ev = evals.derived().array().abs().log10();

        SPDLOG_INFO("evals {} ev {}", evals, ev);
        SPDLOG_INFO("ev max {} ev min {}", ev.maxCoeff(), ev.minCoeff());


        auto n_dets = evals.size();
        // mean of eigenvalues
        auto m_ev = ev.mean();
        // standard deviation of eigenvalues
        auto stddev = engine_utils::calc_std_dev(ev);

        SPDLOG_INFO("m_ev {} stddev {}",m_ev, stddev);

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
            SPDLOG_INFO("count {}", count);
        }

        SPDLOG_INFO("m_ev {} stddev_limit {} stddev {}", m_ev, stddev_limit, stddev);

        double limit = m_ev + stddev_limit*stddev;
        limit = pow(10.,limit);
        Eigen::Index limit_index = 0;

        SPDLOG_INFO("limit {}", limit);

        for (Eigen::Index i=0; i<n_dets; i++) {
            if (evals(i) <= limit){
                limit_index = i;
                break;
            }
        }

        return limit_index;
    }

    template <EigenSolverBackend backend, typename DerivedA, typename DerivedB, typename DerivedC>
    auto calc_eig_values(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &, Eigen::DenseBase<DerivedC> &);

    template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    auto remove_eig_values(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &,
                                    const Eigen::DenseBase<DerivedC> &, const Eigen::DenseBase<DerivedD> &,
                                    Eigen::DenseBase<DerivedA> &);
};

template <Cleaner::EigenSolverBackend backend, typename DerivedA, typename DerivedB, typename DerivedC>
auto Cleaner::calc_eig_values(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags,
                              Eigen::DenseBase<DerivedC> &apt_flags) {
    // dimensions
    Eigen::Index n_pts = scans.rows();
    Eigen::Index n_dets = scans.cols();

    // eigenvalues
    Eigen::VectorXd evals;
    // eigenvecs
    Eigen::MatrixXd evecs;

    // normalization denominator
    //Eigen::MatrixXd denom;

    /*
    Eigen::MatrixXd det;

    // number of non zero dets
    Eigen::Index n_non_zero = 0;

    // find number of non zero detectors
    for (Eigen::Index i=0; i<n_dets; i++) {
        if ((flags.col(i).array()!=0).all()) {
            n_non_zero++;
        }
    }

    // resize det matrix
    det.setZero(n_pts, n_non_zero);

    // flag matrix for non zero detectors
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> f;
    f.resize(n_pts, n_non_zero);

    // populate det matrix with non zero detectors
    Eigen::Index j=0;
    for (Eigen::Index i=0; i<n_dets; i++) {
        if ((flags.col(i).array() !=0).all()) {
            det.col(j) = scans.col(i);
            f.col(j) = flags.col(i);
            j++;
        }
    }

    // calculate denominator
    denom = (f.template cast <double> ().adjoint() * f.template cast <double> ()).array() - 1;

    // container for covariance matrix
    Eigen::MatrixXd pca_cov(n_non_zero, n_non_zero);
    */

    // make copy of flags
    Eigen::MatrixXd f = flags.derived().template cast<double> ();

    // zero out flagged detectors in apt table
    for (Eigen::Index i=0; i<n_dets; i++) {
        if (apt_flags.derived()(i) == 0) {
            f.col(i).setZero();
        }
    }

    // container for covariance matrix
    Eigen::MatrixXd pca_cov(n_dets, n_dets);

    // number of unflagged samples
    //denom = (flags.derived().template cast <double> ().adjoint() * flags.derived().template cast <double> ()).array() - 1;
    auto denom = (f.adjoint() * f).array() - 1;

    auto det = (scans.derived().array()*f.array()).matrix();

    // calculate the covariance matrix
    pca_cov.noalias() = ((det.adjoint() * det).array() / denom.array()).matrix();

    if constexpr (backend == SpectraBackend) {
        // number of eigenvalues to remove
        int n_ev = n_eig_to_cut;

        // if using std dev limit and n_eig_to_cut is zero, use all detectors
        if (stddev_limit > 0 && n_eig_to_cut==0) {
            n_ev = n_dets - 1;
        }

        // number of values to calculate
        int n_cv = n_ev * 2.5 < n_dets?int(n_ev * 2.5):n_dets;
        //int n_cv = n_ev * 2.5 < n_dets?int(nev * 2.5):n_non_zero;

        // set up spectra
        Spectra::DenseSymMatProd<double> op(pca_cov);
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, n_ev, n_cv);

        eigs.init();
        // largest eigenvalues first
        int n_conv = eigs.compute(Spectra::SortRule::LargestAlge);

        // retrieve results
        evals = Eigen::VectorXd::Zero(n_dets);
        //evals = Eigen::VectorXd::Zero(n_non_zero);

        evecs = Eigen::MatrixXd::Zero(n_dets, n_dets);
        //evecs = Eigen::MatrixXd::Zero(n_non_zero, n_non_zero);

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

template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
auto Cleaner::remove_eig_values(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags,
                                const Eigen::DenseBase<DerivedC> &evals, const Eigen::DenseBase<DerivedD> &evecs,
                                Eigen::DenseBase<DerivedA> &cleaned_scans) {

    // number of eigenvalues to remove
    Eigen::Index limit_index;

    Eigen::Index n_dets = scans.cols();

    // if using std dev limit, calculate index
    if (stddev_limit > 0) {
        int n_ev;
        // if using std dev limit and n_eig_to_cut is zero, use all detectors
        if (n_eig_to_cut==0) {
            n_ev = n_dets - 1;
        }
        else {
            n_ev = n_eig_to_cut;
        }
        limit_index = get_stddev_index(evals.head(n_ev));
        SPDLOG_INFO("limit index {}", limit_index);
    }
    // otherwise use number of eigenvalues from config
    else {
        limit_index = n_eig_to_cut;
    }

    // subtract out the desired eigenvectors
    Eigen::MatrixXd proj = scans.derived() * evecs.derived().leftCols(limit_index);
    cleaned_scans.derived().noalias() = scans.derived() - proj * evecs.derived().adjoint().topRows(limit_index);
}

} // namespace timestream
