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

    template <EigenSolverBackend backend, typename DerivedA, typename DerivedB>
    auto calc_eig_values(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &);

    template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    auto remove_eig_values(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &,
                                    const Eigen::DenseBase<DerivedC> &, const Eigen::DenseBase<DerivedD> &,
                                    Eigen::DenseBase<DerivedA> &);

};

template <EigenSolverBackend backend, typename DerivedA, typename DerivedB>
auto Cleaner::calc_eig_values(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags) {
    Eigen::Index n_pts = scans.rows();
    Eigen::Index n_dets = scans.cols();

    Eigen::MatrixXd det;

    Eigen::VectorXd evals;
    Eigen::MatrixXd evecs;

    Eigen::MatrixXd denom;

    // mean of each detector
    Eigen::RowVectorXd det_means = (scans.derived().array()*flags.derived().array().template cast <double> ()).colwise().sum()/
                                   flags.derived().array().template cast <double> ().colwise().sum();

    // remove nans from completely flagged detectors
    Eigen::RowVectorXd dm = (det_means).array().isNaN().select(0,det_means);

    // subtract mean from data and copy into det matrix
    det = (scans.derived().array()*flags.derived().array().template cast <double> ()).matrix().rowwise() - dm;

    // container for covariance matrix
    Eigen::MatrixXd pca_cov(n_dets, n_dets);

    // number of unflagged samples
    denom = (flags.derived().template cast <double> ().adjoint() * flags.derived().template cast <double> ()).array() - 1;

    // calculate the covariance Matrix
    pca_cov.noalias() = ((det.adjoint() * det).array() / denom.array()).matrix();

    if constexpr (backend == SpectraBackend) {
        // number of eigenvalues to remove
        int nev = n_eig_to_cut;

        // number of values to calculate
        int ncv = nev * 2.5 < n_dets?int(nev * 2.5):n_dets;

        // set up spectra
        Spectra::DenseSymMatProd<double> op(pca_cov);
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, nev, ncv);

        eigs.init();
        // largest eigenvalues first
        int nconv = eigs.compute(Spectra::SortRule::LargestAlge);

        // retrieve results
        evals = Eigen::VectorXd::Zero(n_dets);

        // do we need to have n_dets x n_dets?
        evecs = Eigen::MatrixXd::Zero(n_dets, n_dets);

        if (eigs.info() == Spectra::CompInfo::Successful) {
            evals.head(nev) = eigs.eigenvalues();
            evecs.leftCols(nev) = eigs.eigenvectors();
        }

        else {
            throw std::runtime_error("failed to compute eigen values");

        }

    }

    return std::tuple<Eigen::VectorXd, Eigen::MatrixXd> {evals, evecs};

}

template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
auto Cleaner::remove_eig_values(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags,
                                const Eigen::DenseBase<DerivedC> &evals, const Eigen::DenseBase<DerivedD> &evecs,
                                Eigen::DenseBase<DerivedA> &cleaned_scans) {

    // mean of each detector
    Eigen::RowVectorXd det_means = (scans.derived().array()*flags.derived().array().template cast <double> ()).colwise().sum()/
                                   flags.derived().array().template cast <double> ().colwise().sum();

    // remove nans from completely flagged detectors
    Eigen::RowVectorXd dm = (det_means).array().isNaN().select(0,det_means);

    // subtract mean from data and copy into det matrix
    Eigen::MatrixXd det = (scans.derived().array()*flags.derived().array().template cast <double> ()).matrix().rowwise() - dm;

    // subtract out the desired eigenvectors
    Eigen::MatrixXd proj;
    proj.noalias() = det * evecs.leftCols(n_eig_to_cut);
    cleaned_scans.derived().noalias() = det - proj * evecs.derived().adjoint().topRows(n_eig_to_cut);

}

} // namespace timestream
