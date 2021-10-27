#pragma once

#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>

namespace timestream {

enum EigenSolverBackend {
        EigenBackend = 0,
        SpectraBackend = 1
    };

enum ScanType {
        DataType = 0,
        KernelType = 1
    };


class Cleaner{
public:
    int neig;
    double cut_std;
    std::string grouping;

    //Eigen::Index ndetectors, npts;

    //Eigen::MatrixXd det;
    Eigen::Matrix<bool,Eigen::Dynamic, Eigen::Dynamic> flg;
    Eigen::Matrix<bool,Eigen::Dynamic, Eigen::Dynamic> denom;

    //Eigen::VectorXd evals;
    //Eigen::MatrixXd evecs;

    template <EigenSolverBackend backend, typename DerivedA, typename DerivedB>
    auto calcEigs(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &);

    template <EigenSolverBackend backend, ScanType stype, typename DerivedA, typename DerivedB,
              typename DerivedC, typename DerivedD>
    void removeEigs(Eigen::DenseBase<DerivedA>&, Eigen::DenseBase<DerivedB> &,
                   Eigen::DenseBase<DerivedC>&, Eigen::DenseBase<DerivedD> &);
};

template <EigenSolverBackend backend, typename DerivedA, typename DerivedB>
auto Cleaner::calcEigs(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags){

    auto ndetectors = scans.cols();
    auto npts = scans.rows();

    Eigen::MatrixXd det;
    Eigen::VectorXd evals;
    Eigen::MatrixXd evecs;

    // mean of each detector
    Eigen::RowVectorXd det_means = scans.derived().colwise().mean();

    // subtract median from scans and copy into det matrix
    det = scans.derived().rowwise() - det_means;

    // container for Correlation Matrix
    Eigen::MatrixXd pcaCorr(ndetectors, ndetectors);

    // calculate the Correlation Matrix
    pcaCorr.noalias() = (det.derived().adjoint() * det);

    if constexpr (backend == SpectraBackend) {
        // number of eigs to cut
        int nev = neig;
        // number of values to caluclate
        int ncv = nev * 2.5 < ndetectors?int(nev * 2.5):ndetectors;
        // set up spectra
        Spectra::DenseSymMatProd<double> op(pcaCorr);
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, nev, ncv);

        eigs.init();
        // largest eigenvalues first
        int nconv = eigs.compute(Spectra::SortRule::LargestAlge);

        // Retrieve results
        evals = Eigen::VectorXd::Zero(ndetectors);

        // Do we need to have ndetectors x ndetectors?
        evecs = Eigen::MatrixXd::Zero(ndetectors, neig);

        if (eigs.info() == Spectra::CompInfo::Successful) {
            evals.head(nev) = eigs.eigenvalues();
            evecs.leftCols(nev) = eigs.eigenvectors();
        }
        else {
            throw std::runtime_error("failed to compute eigen values");
        }
    }
    return std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> {std::move(det), std::move(evals), std::move(evecs)};
}

template <EigenSolverBackend backend, ScanType stype, typename DerivedA, typename DerivedB,
          typename DerivedC, typename DerivedD>
void Cleaner::removeEigs(Eigen::DenseBase<DerivedA> &scans, Eigen::DenseBase<DerivedB> &cleanedscans,
                         Eigen::DenseBase<DerivedC> &evals, Eigen::DenseBase<DerivedD> &evecs){

    // subtract mean if scan is a kernel scan (signal scan already mean subtracted to save time)
    if constexpr (stype == KernelType) {
        Eigen::RowVectorXd det_means = scans.derived().colwise().mean();
        scans.derived().noalias() = scans.derived().rowwise() - det_means;
    }

        // subtract out the desired eigenvectors
        Eigen::MatrixXd proj;
        proj.noalias() = scans.derived() * evecs.leftCols(neig);
        cleanedscans.derived().noalias() = scans.derived() - proj * evecs.derived().adjoint().topRows(neig);
    }

} //namespace
