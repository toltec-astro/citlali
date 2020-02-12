#pragma once

#include <Spectra/SymEigsSolver.h>

namespace timestream {

using namespace std;
//using namespace prod_test;

namespace internal {

template <typename Derived>
std::tuple<double, double> sigma_clipped_stats(const Eigen::DenseBase<Derived>& values,
                                               double sigma_lower, double sigma_upper) {
    double mean, stddev;
    return {mean, stddev};
}

template <typename Derived>
Eigen::Index cutoffindex(const Eigen::DenseBase<Derived>& evals, Eigen::Index neigToCut, double cutStd) {
    if (neigToCut <= 0 && cutStd < 2.) {
        throw std::runtime_error("insufficient cut in eigen modes");
    }
    if (cutStd <= 0) {
        return neigToCut;
    }
    Eigen::VectorXd ev = evals.derived().array().abs().log10();
    auto [mev, std] = sigma_clipped_stats(ev, cutStd, cutStd);
    double cut = pow(10, mev + cutStd * std);
    // return which ever the smaller, cut values beyond this point
    return neigToCut < cut? neigToCut: cut;
}

} // namespace internal


enum EigenSolverBackend {
        EigenBackend = 0,
        SpectraBackend = 1
    };


template <EigenSolverBackend backend, typename DerivedA, typename DerivedB>
void pcaclean2(
    const Eigen::DenseBase<DerivedA> &scans,
    const Eigen::DenseBase<DerivedA> &kernelscans,
    const Eigen::DenseBase<DerivedB> &scanflags,
    Eigen::DenseBase<DerivedA> &cleanedscans,
    Eigen::DenseBase<DerivedA> &cleanedkernelscans,
    Eigen::Index neigToCut,
    double cutStd) {

        // scans, kernalscans, and scanflags are [ndata, ndetectors]
        // timestream data matrix from all detectors
        Eigen::Index ndetectors = scans.cols();
        //Eigen::Matrix<Eigen::Index,Eigen::Dynamic,1> npts = scans.rows();//scanindex.row(1) - scanindex.row(0);
        Eigen::Index npts = scans.rows();

        // containers of pca [npts, ndetctors]
        Eigen::MatrixXd det, ker, efdet, efker;
        Eigen::Matrix<bool,Eigen::Dynamic, Eigen::Dynamic> flg;
        Eigen::Matrix<bool,Eigen::Dynamic, Eigen::Dynamic> denom(ndetectors, ndetectors);
        Eigen::MatrixXd pcaCorr(ndetectors, ndetectors);

        //flg.resize(npts, ndetectors);

        Eigen::RowVectorXd scan_means = scans.derived().colwise().mean();
        Eigen::RowVectorXd kernel_means = kernelscans.derived().colwise().mean();

        det = scans.derived().rowwise() - scan_means;
        ker = kernelscans.derived().rowwise() - kernel_means;

        // return data
        cleanedscans.derived().resize(scans.rows(), scans.cols());
        cleanedkernelscans.derived().resize(kernelscans.rows(), kernelscans.cols());

        pcaCorr.noalias() = (det.adjoint() * det);//.cwiseQuotient(denom.template cast<double>());

        if constexpr (backend == SpectraBackend) {
            int nev = neigToCut;//ndetectors <= 100?ndetectors - 1:100;
            int ncv = nev * 2.5 < ndetectors?int(nev * 2.5):ndetectors;
            Spectra::DenseSymMatProd<double> op(pcaCorr);
            Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double>> eigs(&op, nev, ncv);

            eigs.init();
            int nconv = eigs.compute();  // the results will be sorted largest first
            // Retrieve results
            Eigen::VectorXd evals = Eigen::VectorXd::Zero(ndetectors);
            Eigen::MatrixXd evecs = Eigen::MatrixXd::Zero(ndetectors, ndetectors);
            if (eigs.info() == Spectra::SUCCESSFUL) {
                evals.head(nev) = eigs.eigenvalues();
                evecs.leftCols(nev) = eigs.eigenvectors();
            } else {
                throw std::runtime_error("failed to compute eigen values");
            }

            Eigen::MatrixXd proj;//(scans.rows(), scans.cols());
            Eigen::MatrixXd kproj;//(kernelscans.rows(), kernelscans.cols());

            proj.noalias() = det * evecs.leftCols(neigToCut);
            kproj.noalias() = ker.derived() * evecs.leftCols(neigToCut);
            cleanedscans.derived().noalias() = det - proj * evecs.adjoint().topRows(neigToCut);
            cleanedkernelscans.derived().noalias() = ker.derived() - kproj * evecs.adjoint().topRows(neigToCut);
        }
}
} // namespace timestream
