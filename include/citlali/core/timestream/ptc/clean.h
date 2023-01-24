#pragma once

#include <string>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Spectra/SymEigsSolver.h>

namespace timestream {

enum EigenSolverBackend {
    EigenBackend = 0,
    SpectraBackend = 1
};

class Cleaner {
public:
    int n_eig_to_cut;
    double cut_std;
    std::string grouping;

    template <EigenSolverBackend backend, typename DerivedA, typename DerivedB, typename DerivedC>
    auto calc_eig_values(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &, Eigen::DenseBase<DerivedC> &);

    template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    auto remove_eig_values(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &,
                                    const Eigen::DenseBase<DerivedC> &, const Eigen::DenseBase<DerivedD> &,
                                    Eigen::DenseBase<DerivedA> &);
};

template <EigenSolverBackend backend, typename DerivedA, typename DerivedB, typename DerivedC>
auto Cleaner::calc_eig_values(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags,
                              Eigen::DenseBase<DerivedC> &apt) {
    // dimensions
    Eigen::Index n_pts = scans.rows();
    Eigen::Index n_dets = scans.cols();

    // eigenvalues
    Eigen::VectorXd evals;
    // eigenvecs
    Eigen::MatrixXd evecs;

    // normalization denominator
    Eigen::MatrixXd denom;

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

    for (Eigen::Index i=0; i<n_dets; i++) {
        if (apt.derived()(i) == 0) {
            f.col(i).setZero();
        }
    }

    // container for covariance matrix
    Eigen::MatrixXd pca_cov(n_dets, n_dets);

    // number of unflagged samples
    //denom = (flags.derived().template cast <double> ().adjoint() * flags.derived().template cast <double> ()).array() - 1;
    denom = (f.adjoint() * f).array() - 1;

    Eigen::MatrixXd det = scans.derived().array()*f.array();

    // calculate the covariance Matrix
    //pca_cov.noalias() = ((scans.derived().adjoint() * scans.derived()).array() / denom.array()).matrix();
    pca_cov.noalias() = ((det.adjoint() * det).array() / denom.array()).matrix();

    if constexpr (backend == SpectraBackend) {
        // number of eigenvalues to remove
        int n_ev = n_eig_to_cut;

        // number of values to calculate
        int n_cv = n_ev * 2.5 < n_dets?int(n_ev * 2.5):n_dets;
        //int ncv = nev * 2.5 < n_dets?int(nev * 2.5):n_non_zero;

        // set up spectra
        Spectra::DenseSymMatProd<double> op(pca_cov);
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, n_ev, n_cv);

        eigs.init();
        // largest eigenvalues first
        int nconv = eigs.compute(Spectra::SortRule::LargestAlge);

        // retrieve results
        evals = Eigen::VectorXd::Zero(n_dets);
        //evals = Eigen::VectorXd::Zero(n_non_zero);

        // do we need to have n_dets x n_dets?
        evecs = Eigen::MatrixXd::Zero(n_dets, n_dets);
        //evecs = Eigen::MatrixXd::Zero(n_non_zero, n_non_zero);

        // copy the eigenvalues and eigenvectors
        if (eigs.info() == Spectra::CompInfo::Successful) {
            evals.head(n_ev) = eigs.eigenvalues();
            evecs.leftCols(n_ev) = eigs.eigenvectors();
            SPDLOG_INFO("eigs.eigenvectors() {}",eigs.eigenvectors());
        }

        else {
            throw std::runtime_error("spectra failed to compute eigen values");
            std::exit(EXIT_FAILURE);
        }
    }

    else if constexpr (backend == EigenBackend) {
        // use Eigen's eigen solver
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solution(pca_cov);

        SPDLOG_INFO("soultion.info() {}",solution.info());

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

    // subtract out the desired eigenvectors
    Eigen::MatrixXd proj;
    proj.noalias() = scans.derived() * evecs.derived().leftCols(n_eig_to_cut);

    cleaned_scans.derived().noalias() = scans.derived() - proj * evecs.derived().adjoint().topRows(n_eig_to_cut);
}

} // namespace timestream
