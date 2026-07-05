#pragma once

// Included by stats_netcdf.h inside namespace citlali::pipeline.

template <class Diagnostics, class Cleaner>
bool should_write_stats_eigenvalues(const Diagnostics &diagnostics,
                                    const Cleaner &cleaner) {
    return !diagnostics.evals.empty() && cleaner.n_calc > 0;
}

template <class EvalMap>
bool has_stats_eigenvalue_groups(const EvalMap &evals) {
    if (evals.empty()) {
        return false;
    }
    const auto first_it = evals.begin();
    return !first_it->second.empty() && !first_it->second[0].empty();
}

inline std::vector<netCDF::NcDim> add_stats_eigenvalue_dims(
    netCDF::NcFile &fo, Eigen::Index n_calc, std::size_t n_eig_groups) {
    netCDF::NcDim n_eigs_dim = fo.addDim("n_eigs", n_calc);
    netCDF::NcDim n_eig_grp_dim = fo.addDim("n_eig_grp", n_eig_groups);
    return {n_eig_grp_dim, n_eigs_dim};
}

inline std::string stats_eigenvalue_var_name(
    const std::string &grouping_name, Eigen::Index grouping_index,
    Eigen::Index chunk_index) {
    return "evals_" + grouping_name + "_" +
           std::to_string(grouping_index) + "_chunk_" +
           std::to_string(chunk_index);
}

inline std::vector<std::size_t> stats_eigenvalue_start_index() {
    return {0, 0};
}

inline std::vector<std::size_t> stats_eigenvalue_write_shape(
    Eigen::Index n_calc) {
    return {1, static_cast<std::size_t>(n_calc)};
}

inline netCDF::NcVar add_stats_eigenvalue_var(
    netCDF::NcFile &fo, const std::string &name,
    const std::vector<netCDF::NcDim> &dims) {
    return fo.addVar(name, netCDF::ncDouble, dims);
}

template <class EvalVectors>
void write_stats_eigenvalue_rows(netCDF::NcVar &eval_v,
                                 const EvalVectors &eval_vectors,
                                 Eigen::Index n_cleaner_eigenvalues,
                                 double fill_value) {
    auto start_eig_index = stats_eigenvalue_start_index();
    const auto eig_write_shape =
        stats_eigenvalue_write_shape(n_cleaner_eigenvalues);
    for (const auto &evals : eval_vectors) {
        Eigen::VectorXd padded_evals =
            ptcdiag_padded_eigenvalues(
                evals, n_cleaner_eigenvalues, fill_value);
        eval_v.putVar(start_eig_index, eig_write_shape, padded_evals.data());
        start_eig_index[0] += 1;
    }
}

template <class EvalVectors>
void add_stats_eigenvalue_group_var(
    netCDF::NcFile &fo, const std::string &name,
    const std::vector<netCDF::NcDim> &dims,
    const EvalVectors &eval_vectors,
    Eigen::Index n_cleaner_eigenvalues, double fill_value) {
    netCDF::NcVar eval_v = add_stats_eigenvalue_var(fo, name, dims);
    write_stats_eigenvalue_rows(
        eval_v, eval_vectors, n_cleaner_eigenvalues, fill_value);
}

template <class Diagnostics, class Cleaner, class Logger>
void add_stats_eigenvalue_outputs_if_needed(
    netCDF::NcFile &fo, const Diagnostics &diagnostics,
    const Cleaner &cleaner, const Logger &logger, double fill_value) {
    if (!should_write_stats_eigenvalues(diagnostics, cleaner)) {
        return;
    }

    if (!has_stats_eigenvalue_groups(diagnostics.evals)) {
        logger->warn("evals requested but empty; skipping eval/evec output");
        return;
    }

    const auto first_it = diagnostics.evals.begin();
    const Eigen::Index n_cleaner_eigenvalues = cleaner.n_calc;
    const auto &cleaner_grouping = cleaner.grouping;
    const auto n_eig_groups = first_it->second[0].size();
    const auto eval_dims =
        add_stats_eigenvalue_dims(fo, n_cleaner_eigenvalues, n_eig_groups);

    for (const auto &[chunk_index, eval_groups] : diagnostics.evals) {
        const auto n_eval_groupings = eval_groups.size();
        for (Eigen::Index i = 0;
             i < static_cast<Eigen::Index>(n_eval_groupings); ++i) {
            const auto &cleaner_grouping_name = cleaner_grouping[i];
            const auto eval_var_name =
                stats_eigenvalue_var_name(
                    cleaner_grouping_name, i, chunk_index);
            add_stats_eigenvalue_group_var(
                fo, eval_var_name, eval_dims, eval_groups[i],
                n_cleaner_eigenvalues, fill_value);
        }
    }
}

