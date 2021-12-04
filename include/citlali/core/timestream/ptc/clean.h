#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Spectra/SymEigsSolver.h>

#include <citlali/core/utils/utils.h>

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

    template <typename Derived>
    auto get_stddev_index(Eigen::DenseBase<Derived> &);

    template <EigenSolverBackend backend, typename DerivedA, typename DerivedB>
    auto calcEigs(const Eigen::DenseBase<DerivedA> &, const Eigen::DenseBase<DerivedB> &);

    template <EigenSolverBackend backend, ScanType stype, typename DerivedA, typename DerivedB,
              typename DerivedC, typename DerivedD>
    void removeEigs(Eigen::DenseBase<DerivedA>&, Eigen::DenseBase<DerivedB> &,
                   Eigen::DenseBase<DerivedC>&, Eigen::DenseBase<DerivedD> &);
};


template <typename Derived>
auto Cleaner::get_stddev_index(Eigen::DenseBase<Derived> &evals) {

    // copy eigenvalues
    Eigen::VectorXd ev = evals.derived().array().abs().log10();

    auto ndetectors = evals.size();
    // mean of eigenvalues
    auto mev = ev.mean();
    // standard deviation of eigenvalues
    auto stddev = engine_utils::stddev(ev);

    bool keep_going = 1;
    int n_keep_last = ndetectors;

    // vector of eigenvalues below stddev cut
    Eigen::VectorXd good(ndetectors);
    good.setOnes(ndetectors);

    int iterator = 0;
    while (keep_going) {
        // count up number of eigenvalues that pass the cut
        int count = 0;
        for (Eigen::Index i=0; i<ndetectors; i++) {
            if (good(i) > 0) {
                if (abs(ev(i) - mev) > abs(cut_std*stddev)){
                    good(i) = 0;
                }
                else {
                    count++;
                }
            }
        }

        if (count >= n_keep_last) {
            keep_going = 0;
        }

        else {
            // get new mean and stddev for only the good eigenvectors
            mev = 0.;
            for (Eigen::Index i=0; i<ndetectors; i++) {
                if (good(i) > 0) {
                    mev += ev(i);
                }
            }
            mev /= count;
            stddev = 0.;
            for (Eigen::Index i=0; i<ndetectors; i++) {
                if (good(i)>0) {
                    stddev += (ev(i) - mev)*(ev(i) - mev);
                }
            }
            stddev = stddev/(count-1.);
            stddev = sqrt(stddev);
            n_keep_last = count;
        }
        iterator++;
    }

    double cut = mev + cut_std*stddev;
    cut = pow(10.,cut);
    int cut_index = 0;

    for (Eigen::Index i=0; i<ndetectors; i++) {
        if(evals(i) <= cut){
            cut_index=i;
            break;
        }
    }

    return cut_index;
}

template <EigenSolverBackend backend, typename DerivedA, typename DerivedB>
auto Cleaner::calcEigs(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags){

    auto ndetectors = scans.cols();
    auto npts = scans.rows();

    Eigen::MatrixXd det;
    Eigen::VectorXd evals;
    Eigen::MatrixXd evecs;

    //Eigen::Matrix<bool,Eigen::Dynamic, Eigen::Dynamic> flg;
    //Eigen::Matrix<bool,Eigen::Dynamic, Eigen::Dynamic> denom;

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

        if (cut_std > 0) {
            nev = ndetectors - 1;
        }

        // number of values to calculate
        int ncv = nev * 2.5 < ndetectors?int(nev * 2.5):ndetectors;

        // set up spectra
        Spectra::DenseSymMatProd<double> op(pcaCorr);
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, nev, ncv);

        eigs.init();
        // largest eigenvalues first
        int nconv = eigs.compute(Spectra::SortRule::LargestAlge);

        // retrieve results
        evals = Eigen::VectorXd::Zero(ndetectors);

        // do we need to have ndetectors x ndetectors?
        evecs = Eigen::MatrixXd::Zero(ndetectors, neig);

        if (eigs.info() == Spectra::CompInfo::Successful) {
            evals.head(nev) = eigs.eigenvalues();
            evecs.leftCols(nev) = eigs.eigenvectors();
        }
        else {
            throw std::runtime_error("failed to compute eigen values");
        }
    }

    if constexpr (backend == EigenBackend) {
        // use Eigen's eigen solver
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solution(pcaCorr);
        evals = solution.eigenvalues();
        evecs = solution.eigenvectors();
    }

    return std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> {std::move(det), std::move(evals), std::move(evecs)};
}

template <EigenSolverBackend backend, ScanType stype, typename DerivedA, typename DerivedB,
          typename DerivedC, typename DerivedD>
void Cleaner::removeEigs(Eigen::DenseBase<DerivedA> &scans, Eigen::DenseBase<DerivedB> &cleanedscans,
                         Eigen::DenseBase<DerivedC> &evals, Eigen::DenseBase<DerivedD> &evecs) {

    // subtract mean if scan is a kernel scan (signal scan already mean subtracted to save time)
    if constexpr (stype == KernelType) {
        Eigen::RowVectorXd det_means = scans.derived().colwise().mean();
        scans.derived().noalias() = scans.derived().rowwise() - det_means;
    }

    Eigen::Index cut_index = neig;
    if (cut_std > 0) {
        cut_index = get_stddev_index(evals);
    }

    // subtract out the desired eigenvectors
    Eigen::MatrixXd proj;
    proj.noalias() = scans.derived() * evecs.leftCols(cut_index);
    cleanedscans.derived().noalias() = scans.derived() - proj * evecs.derived().adjoint().topRows(cut_index);
}

} //namespace
