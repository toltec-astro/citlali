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
            auto mask = (tcdata.flag.col(det).array() == false).matrix().template cast<double>();
            int n_pts =  mask.sum();
            double mean = (tcdata.signal.col(det).array() * mask.array()).sum() / n_pts;
            tcdata.signal.col(det) = tcdata.signal.col(det).array() - mean;

            if (tcdata.kernel.size() > 0) {
                mean = (tcdata.kernel.col(det).array() * mask.array()).sum() / n_pts;
                tcdata.kernel.array().col(det) = tcdata.kernel.array().col(det).array() - mean;
            }
        }
    }

    // function to calculate the top n_eig largest eigenvalues and eigenvectors
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

    auto [in, out] = citlali::utils::threads::get_grppi_vectors(groups.size());
    auto exec_mode = citlali::utils::threads::get_chunk_remainder_exec_mode();

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
        // loop through indices for current cleaning group
        for (const auto& [start, end]: indices) {
            int array_index;
            if (groups[i] == "nw") {
                array_index = toltec.nw_to_array[toltec.apt.nws(j)];
            } else if (groups[i] == "array") {
                array_index = toltec.apt.arrays(j);
            } else if (groups[i] == "all") {
                array_index = toltec.apt.arrays(0);
            }

            // good detector indices
            auto good_indices = tcdata.get_good_indices(start, end);

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


