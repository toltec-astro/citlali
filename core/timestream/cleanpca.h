#pragma once

#include <Eigen/Dense>
#include <Spectra/SymEigsSolver.h>


//#include "densesymmatprod.h"

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
void pcaclean(
    const Eigen::DenseBase<DerivedA> &scans,
    const Eigen::DenseBase<DerivedA> &kernelscans,
    const Eigen::DenseBase<DerivedB> &scanflags,
    Eigen::DenseBase<DerivedA> &cleanedscans,
    Eigen::DenseBase<DerivedA> &cleanedkernelscans,
    Eigen::Index neigToCut,
    double cutStd
    ) {
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
    // return data
    cleanedscans.derived().resize(scans.rows(), scans.cols());
    cleanedkernelscans.derived().resize(kernelscans.rows(), kernelscans.cols());

    //SPDLOG_INFO("input data {}", logging::pprint(scans));
        // prepare containers
        det.resize(ndetectors, npts);
        ker.resize(ndetectors, npts);
        flg.resize(ndetectors, npts);
        // populate
        for (Eigen::Index i = 0; i < ndetectors; ++i) {
            det.row(i) = scans.col(i);
            ker.row(i) = kernelscans.col(i);
            flg.row(i) = scanflags.col(i);

            det.row(i) = det.row(i).array() - det.row(i).mean();
            ker.row(i) = ker.row(i).array() - ker.row(i).mean();


            //Eigen::VectorXd detsorted = det.row(i);

            //std::sort(detsorted.data(), detsorted.data() + detsorted.size());
            //int indx = ((ndetectors-1)/2 + (ndetectors)/2)/2;

            //det.row(i) = (det.row(i).array() - detsorted(indx));
            //ker.row(i) = ker.row(i).array() - ker.row(i).mean();

            // a function should be implemented to median-subtract the values
            // of det and ker inplace
            // it should possibly only include data of the center region
            // the same as in the case of macana, although it is implemented
            // to be true all the time there
        }
        // calculate denom
        // TODO figure out what this really does
        //denom = (flg * flg.adjoint()).array();
        // noalias to force eigen evaluate into pcaCorr
        //det = det.cwiseProduct(flg.template cast<double>());
        pcaCorr.noalias() = (det * det.adjoint());//.cwiseQuotient(denom.template cast<double>());
        //pcaCorr.noalias() = pcaCorr/(ndetectors-1);

        //pcaCorr.noalias() = pcaCorr.cwiseQuotient(denom.template cast<double>());
        // possibly make use of use additional corrMatrix from atmTemplate
        // pcaCorr = pcaCorr.cwiseProduct(corrMatrix)
        // compute eigen values and eigen vectors
        if constexpr (backend == EigenBackend) {
            // these are sorted in increasing order, which is different from IDL
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solution(pcaCorr);
            const auto& evals = solution.eigenvalues();
            const auto& evecs_ = solution.eigenvectors();
            Eigen::MatrixXd evecs(evecs_);
            // find out the cutoff index for evals and do the cut
            Eigen::Index cut = internal::cutoffindex(evals, neigToCut, cutStd);
            //SPDLOG_INFO("cut {} largest modes", cut);
            //SPDLOG_INFO("evals: {}", evals.tail(cut));
            evecs.rightCols(cut).setConstant(0);
            evecs.rightCols(cut).setConstant(0);
            efdet.resize(ndetectors, npts);
            efker.resize(ndetectors, npts);
            efdet.noalias() = evecs.adjoint() * det;
            efker.noalias() = evecs.adjoint() * ker;
            efdet.bottomRows(cut).setConstant(0.);
            efker.bottomRows(cut).setConstant(0.);
            // create cleaned data
            det.noalias() = evecs * efdet;
            ker.noalias() = evecs * efker;
       } else if constexpr (backend == SpectraBackend) {
            // Construct matrix operation object using the wrapper class DenseSymMatProd
            // int nev = ndetectors / 3;
            int nev = neigToCut;//ndetectors <= 100?ndetectors - 1:100;
            int ncv = nev * 2.5 < ndetectors?int(nev * 2.5):ndetectors;
            //SPDLOG_INFO("spectra eigen solver nev={} ncv={}", nev, ncv);
            Spectra::DenseSymMatProd<double> op(pcaCorr);
            Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double>> eigs(&op, nev, ncv);
            // Initialize and compute
            eigs.init();
            int nconv = eigs.compute();  // the results will be sorted largest first
            // Retrieve results
            Eigen::VectorXd evals = Eigen::VectorXd::Zero(ndetectors);
            Eigen::MatrixXd evecs = Eigen::MatrixXd::Zero(ndetectors, ndetectors);
            if(eigs.info() == Spectra::SUCCESSFUL) {
                evals.head(nev) = eigs.eigenvalues();
                evecs.leftCols(nev) = eigs.eigenvectors();
            } else {
                throw std::runtime_error("failed to compute eigen values");
            }
            efdet.resize(ndetectors, npts);
            efker.resize(ndetectors, npts);
            efdet.noalias() = evecs.adjoint() * det;
            efker.noalias() = evecs.adjoint() * ker;
            // find out the cutoff index for evals and do the cut
            Eigen::Index cut = internal::cutoffindex(evals, neigToCut, cutStd);
            if (cut > nev) {
                throw std::runtime_error("too few eigen values computed");
            }
            //SPDLOG_INFO("cut {} largest modes", cut);
            //SPDLOG_INFO("evals: {}", evals.head(cut));
            // here since we are computing the larget vectors, we first
            // construct the data from the larget modes, and then subtract
            efdet.topRows(cut).setConstant(0.);
            efker.topRows(cut).setConstant(0.);
            // create data to be cleaned and substract
            det.noalias() -= evecs * efdet;
            ker.noalias() -= evecs * efker;
        } else {
            static_assert(backend == EigenBackend, "UNKNOWN EIGEN SOLVER BACKEND");
        }
        // update cleaned data
        for (Eigen::Index i = 0; i < ndetectors; ++i) {
            // //SPDLOG_LOGGER_TRACE(logger, "write cleaned data {}", logging::pprint(det.row(i)));
            cleanedscans.col(i) = det.row(i);
            cleanedkernelscans.col(i) = ker.row(i);
        }
    //}
}

} // namespace timestream
