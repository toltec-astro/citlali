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

    // mean subtracted detectors
    Eigen::MatrixXd det;

    // eigenvalues
    Eigen::VectorXd evals;
    // eigenvecs
    Eigen::MatrixXd evecs;

    // normalization denominator
    Eigen::MatrixXd denom;

    // number of non zero dets
    /*Eigen::Index n_non_zero = 0;

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

    SPDLOG_INFO("det {}", det);

    // calculate detector means
    Eigen::RowVectorXd det_means = det.colwise().mean();
    // subtract detector means
    det.noalias() = det.rowwise() - det_means;

    SPDLOG_INFO("det_means {}", det_means);

    SPDLOG_INFO("det {}", det);

    // calculate denominator
    denom = (f.template cast <double> ().adjoint() * f.template cast <double> ()).array() - 1;

    SPDLOG_INFO("denom {}", denom);


    // container for covariance matrix
    Eigen::MatrixXd pca_cov(n_non_zero, n_non_zero);*/

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


    SPDLOG_INFO("pca_cov {}", pca_cov);

    if constexpr (backend == SpectraBackend) {
        // number of eigenvalues to remove
        int nev = n_eig_to_cut;

        // number of values to calculate
        int ncv = nev * 2.5 < n_dets?int(nev * 2.5):n_dets;
        //int ncv = nev * 2.5 < n_dets?int(nev * 2.5):n_non_zero;

        // set up spectra
        Spectra::DenseSymMatProd<double> op(pca_cov);
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, nev, ncv);

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
            evals.head(nev) = eigs.eigenvalues();
            evecs.leftCols(nev) = eigs.eigenvectors();
        }

        else {
            throw std::runtime_error("failed to compute eigen values");
        }

        SPDLOG_INFO("evals {}, evecs {}", evals, evecs);
        SPDLOG_INFO("max evecs {} min evecs {}", evecs.maxCoeff(), evecs.minCoeff());
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

    SPDLOG_INFO("det means 2 {}", det_means);

    // remove nans from completely flagged detectors
    Eigen::RowVectorXd dm = (det_means).array().isNaN().select(0,det_means);

    SPDLOG_INFO("dm {}", dm);


    // subtract mean from data and copy into det matrix
    Eigen::MatrixXd det = scans.derived().rowwise() - dm;

    SPDLOG_INFO("det 2 {} {} {}", det, det.maxCoeff(), det.minCoeff());


    // subtract out the desired eigenvectors
    Eigen::MatrixXd proj;
    proj.noalias() = det * evecs.derived().leftCols(n_eig_to_cut);

    SPDLOG_INFO("proj {}", proj);

    cleaned_scans.derived().noalias() = det - proj * evecs.derived().adjoint().topRows(n_eig_to_cut);
}

} // namespace timestream
