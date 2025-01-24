#pragma once

#include <Eigen/Eigenvalues>
#include <Spectra/SymEigsSolver.h>

// PcaClean
template <typename TCDataType>
class PcaClean : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    std::vector<std::string> groups;
    std::map<int, std::vector<int>> n_eigs;

    Instrument& toltec;
    Telescope& telescope;

    template <typename ConfigType>
    PcaClean(Instrument& toltec_ref, Telescope& telescope_ref, ConfigType& config)
        : toltec(toltec_ref), telescope(telescope_ref) {

        config.get(groups, std::tuple{"timestream","processed_time_chunk","clean","grouping"});

        for (const auto& [index, name] : toltec.array_index_to_name) {
            std::vector<int> n_eig_array;
            config.get(n_eig_array, std::tuple{"timestream","processed_time_chunk","clean", "n_eig_to_cut",name});
            n_eigs[index] = n_eig_array;
        }
    }

    void init() {}
    void process(TCDataType& tcdata) override {
        logger->info("pca clean processing");
        subtract_mean(tcdata);
        pca(tcdata);
    }

    void pca(TCDataType&);

    void subtract_mean(TCDataType& tcdata) {
        for (int det = 0; det < tcdata.n_dets(); ++det) {
            double mean = tcdata.signal.col(det).mean();
            tcdata.signal.col(det) = tcdata.signal.col(det).array() - mean;

            if (tcdata.kernel.size() > 0) {
                tcdata.kernel.array().col(det) = tcdata.kernel.array().col(det).array() - tcdata.kernel.col(det).mean();
            }
        }
    }

    // function to calculate the top k largest eigenvalues and eigenvectors
    template <typename DerivedA, typename DerivedB>
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> calculate_eigenvalues(Eigen::DenseBase<DerivedA>& matrix,
                                                                       Eigen::DenseBase<DerivedB>& flag,
                                                                       const int n_eig) {

        // flag mask
        auto mask = (flag.derived().array() == false).matrix().template cast<double>();
        auto masked_matrix = (matrix.derived().array() * mask.derived().array()).matrix();

        // compute the covariance matrix
        Eigen::MatrixXd cov_matrix = (masked_matrix.transpose() * masked_matrix).array() /
                                     ((mask.sum()) - 1);

        // calculate the n_eig largest eigenvalues and corresponding eigenvectors
        int n_calc = (2 * n_eig >= matrix.cols()) ? matrix.cols() : 2 * n_eig;

        Spectra::DenseSymMatProd<double> op(cov_matrix);
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, n_eig, n_calc);

        eigs.init();
        eigs.compute(Spectra::SortRule::LargestAlge);

        if (eigs.info() != Spectra::CompInfo::Successful) {
            throw std::runtime_error("Spectra failed to compute eigenvalues.");
        }

        // get the top k eigenvalues and corresponding eigenvectors
        Eigen::VectorXd eigenvalues = eigs.eigenvalues();
        Eigen::MatrixXd eigenvectors = eigs.eigenvectors();

        return std::make_tuple(eigenvalues, eigenvectors);
    }

    // function to remove eigenvectors
    template <typename Derived>
    void remove_eigenvalues(Eigen::DenseBase<Derived>& matrix, const Eigen::MatrixXd& eigenvectors) {
        Eigen::MatrixXd projections = matrix.derived() * eigenvectors;
        matrix -= projections * eigenvectors.transpose();
    }
};

template <typename TCDataType>
void PcaClean<TCDataType>::pca(TCDataType& tcdata) {

    // for det-based grppi maps
    std::vector<int> in, out;
    in.resize(groups.size());
    std::iota(in.begin(), in.end(), 0);
    out.resize(groups.size());

    auto exec_mode = tula::grppi_utils::dyn_ex(citlali::utils::threads::det_exec_mode,
                                               citlali::utils::threads::n_det_threads);

    // loop through cleaning groups
    grppi::map(exec_mode, in, out, [&](int i) {

        std::vector<std::pair<int, int>> indices;

        if (groups[i] == "nw") {
            indices = toltec.apt.nw_indices;
        } else if (groups[i] == "array") {
            indices = toltec.apt.array_indices;
        } else if (groups[i] == "all") {
            indices.emplace_back(0, toltec.apt.n_dets);
        }

        int j = 0;
        // lop through indices for current cleaning group
        for (const auto& [start, end]: indices) {
            int array_index;
            if (groups[i] == "nw") {
                array_index = toltec.nw_to_array[toltec.apt.nws(j)];
            } else if (groups[i] == "array") {
                array_index = toltec.apt.arrays(j);
            } else if (groups[i] == "all") {
                array_index = toltec.apt.arrays(0);
            }

            // array to store good indices
            Eigen::ArrayXi good_indices;

            int n_good = 0;

            // find number of good detectors
            for (int k = start; k <= end; ++k) {
                if (!tcdata.apt_flag(k) && (tcdata.flag.col(k).array() == false).any()) {
                    n_good++;
                }
            }

            good_indices.resize(n_good);

            // populate good indices
            int m = 0;
            for (int k = start; k <= end; ++k) {
                if (!tcdata.apt_flag(k) && (tcdata.flag.col(k).array() == false).any()) {
                    good_indices(m) = k;
                    m++;
                }
            }

            // signal matrix for current cleaning group
            auto signal = tcdata.signal(Eigen::all, good_indices);
            // flag matrix for current cleaning group
            auto flag = tcdata.flag(Eigen::all, good_indices);

            auto [eigenvalues, eigenvectors] = calculate_eigenvalues(signal, flag, n_eigs[array_index][i]);

            remove_eigenvalues(signal, eigenvectors);

            if (tcdata.kernel.size() > 0) {
                // kernel matrix for current cleaning group
                auto kernel = tcdata.kernel(Eigen::all, good_indices);
                remove_eigenvalues(kernel, eigenvectors);
            }
            j++;
        }
        return 0;
    });
}


