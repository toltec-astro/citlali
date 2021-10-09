#pragma once

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


class pcaCleaner{
public:
    pcaCleaner(int neC, double cs): neigToCut(neC), cutStd(cs){}
    int neigToCut;
    double cutStd;

    Eigen::Index ndetectors, npts;

    Eigen::MatrixXd det;
    Eigen::Matrix<bool,Eigen::Dynamic, Eigen::Dynamic> flg;
    Eigen::Matrix<bool,Eigen::Dynamic, Eigen::Dynamic> denom;

    Eigen::VectorXd evals;
    Eigen::MatrixXd evecs;

    template <EigenSolverBackend backend, typename DerivedA, typename DerivedB>
    void calcEigs(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &);

    template <EigenSolverBackend backend, ScanType stype, typename DerivedA, typename DerivedB>
    void removeEigs(Eigen::DenseBase<DerivedA>&, Eigen::DenseBase<DerivedB> &);
};

template <EigenSolverBackend backend, typename DerivedA, typename DerivedB>
void pcaCleaner::calcEigs(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &scanflags){

    ndetectors = scans.cols();
    npts = scans.rows();

    Eigen::RowVectorXd detMeans = scans.derived().colwise().mean();

    //Subtract median from scans and copy into det matrix
    det.noalias() = scans.derived().rowwise() - detMeans;

    //Container for Correlation Matrix
    Eigen::MatrixXd pcaCorr(ndetectors, ndetectors);

    // Calculate the Correlation Matrix
    pcaCorr.noalias() = (det.adjoint() * det);

    if constexpr (backend == SpectraBackend) {
        int nev = neigToCut; // ndetectors <= 100?ndetectors - 1:100;
        int ncv = nev * 2.5 < ndetectors?int(nev * 2.5):ndetectors;
        Spectra::DenseSymMatProd<double> op(pcaCorr);
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, nev, ncv);

        eigs.init();
        int nconv = eigs.compute(Spectra::SortRule::LargestAlge);  //Largest eigenvalues first

        // Retrieve results
        evals = Eigen::VectorXd::Zero(ndetectors);

        // Do we need to have ndetectors x ndetectors?
        evecs = Eigen::MatrixXd::Zero(ndetectors, neigToCut);

        if (eigs.info() == Spectra::CompInfo::Successful) {
            evals.head(nev) = eigs.eigenvalues();
            evecs.leftCols(nev) = eigs.eigenvectors();
        }
        else {
            throw std::runtime_error("failed to compute eigen values");
        }
    }
}

template <EigenSolverBackend backend, ScanType stype, typename DerivedA, typename DerivedB>
void pcaCleaner::removeEigs(Eigen::DenseBase<DerivedA> &scans, Eigen::DenseBase<DerivedB> &cleanedscans){

        //Resize output
        // cleanedscans.derived().resize(npts, ndetectors);

        //If kernelscans, subtract median
        if constexpr (stype == KernelType){
            Eigen::RowVectorXd detMeans = scans.derived().colwise().mean();
            scans.derived().noalias() = scans.derived().rowwise() - detMeans;
        }

        //Subtract out the desired eigenvectors
        Eigen::MatrixXd proj;
        proj.noalias() = scans.derived() * evecs.leftCols(neigToCut);
        cleanedscans.derived().noalias() = scans.derived() - proj * evecs.adjoint().topRows(neigToCut);

    }

} //namespace
