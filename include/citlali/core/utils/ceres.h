#include <ceres/ceres.h>

namespace citlali::utils::fitting {

// generic cost function  for ceres (1D)
template <typename Model>
struct CostFunction1D {
    CostFunction1D(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& w)
        : x_(x.data(), x.size()), y_(y.data(), y.size()), weights_(w.data(), w.size()) {}

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        for (int i = 0; i < x_.size(); ++i) {
            if (weights_[i] == 0.0) {
                residuals[i] = T(0.0);  // set residual to 0 if the weight is 0
            } else {
                T predicted_y = Model::eval(T(x_[i]), params);  // evaluate the model
                residuals[i] = weights_[i] * (y_[i] - predicted_y);  // weighted residual
            }
        }
        return true;
    }

private:
    const Eigen::Map<const Eigen::VectorXd> x_;
    const Eigen::Map<const Eigen::VectorXd> y_;
    const Eigen::Map<const Eigen::VectorXd> weights_;
};

// generic cost function  for ceres (2D)
template <typename Model>
struct CostFunction2D {
    CostFunction2D(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
                   const Eigen::MatrixXd& z, const Eigen::MatrixXd& w)
        : x_(x.data(), x.size()), y_(y.data(), y.size()),
        z_(z.data(), z.size()), weights_(w.data(), w.size()) {}

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        for (int i = 0; i < x_.size(); ++i) {
            if (weights_(i) == T(0.0)) {
                residuals[i] = T(0.0);  // set residual to 0 if the weight is 0
            } else {
                T predicted_z = Model::eval(T(x_(i)), T(y_(i)), params);  // evaluate the model
                residuals[i] = T(weights_(i)) * (T(z_(i)) - predicted_z);  // weighted residual
            }
        }
        return true;
    }

private:
    const Eigen::Map<const Eigen::VectorXd> x_;
    const Eigen::Map<const Eigen::VectorXd> y_;
    const Eigen::Map<const Eigen::VectorXd> z_;
    const Eigen::Map<const Eigen::VectorXd> weights_;
};

auto set_options() {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
    options.logging_type = ceres::LoggingType::SILENT;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 1;

    return options;
}

// function to calculate uncertainties from the covariance matrix
Eigen::VectorXd calculate_uncertainties(ceres::Problem& problem, Eigen::VectorXd& params) {
    // initialize uncertainties vector
    Eigen::VectorXd uncertainties = Eigen::VectorXd::Zero(params.size());

    // setup covariance options
    ceres::Covariance::Options covariance_options;
    covariance_options.sparse_linear_algebra_library_type = ceres::SparseLinearAlgebraLibraryType::EIGEN_SPARSE;
    covariance_options.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
    covariance_options.null_space_rank = -1;

    // compute covariance matrix
    ceres::Covariance covariance(covariance_options);
    std::vector<std::pair<const double*, const double*>> covariance_blocks;
    covariance_blocks.emplace_back(params.data(), params.data());

    if (covariance.Compute(covariance_blocks, &problem)) {
        Eigen::MatrixXd covariance_matrix(params.size(), params.size());
        covariance.GetCovarianceBlock(params.data(), params.data(), covariance_matrix.data());
        uncertainties = covariance_matrix.diagonal().cwiseSqrt();
    }

    return uncertainties;
}

// fit model using ceres
template <typename Model>
std::pair<Eigen::VectorXd, Eigen::VectorXd> ceres_fit(ceres::Problem &problem,
                                                      Eigen::VectorXd& params,
                                                      const Eigen::VectorXd& lower_bounds,
                                                      const Eigen::VectorXd& upper_bounds,
                                                      const Eigen::Matrix<bool, Eigen::Dynamic, 1>& fixed_params) {
    // set parameter limits
    for (int i = 0; i < Model::nparams; ++i) {
        problem.SetParameterLowerBound(params.data(), i, lower_bounds[i]);
        problem.SetParameterUpperBound(params.data(), i, upper_bounds[i]);
    }

    // fix parameters as per the fixed_params vector
    std::vector<int> fixed_indices;
    for (int i = 0; i < Model::nparams; ++i) {
        if (fixed_params(i)) {
            fixed_indices.push_back(i);
        }
    }

    if (!fixed_indices.empty()) {
        ceres::SubsetParameterization* subset_parameterization =
            new ceres::SubsetParameterization(Model::nparams, fixed_indices);
        problem.SetParameterization(params.data(), subset_parameterization);
    }

    // set up the solver options
    ceres::Solver::Options options = set_options();

    // solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // initialize uncertainties vector
    Eigen::VectorXd uncertainties = Eigen::VectorXd::Zero(params.size());

    if (summary.IsSolutionUsable()) {
        // get uncertainties
        uncertainties = calculate_uncertainties(problem, params);
    }
    else {
        params.setZero();
    }

    // return both optimized parameters and uncertainties
    return std::make_pair(params, uncertainties);
}

// specialization for 1d fitting
template <typename Model>
std::pair<Eigen::VectorXd, Eigen::VectorXd> fit_model(const Eigen::VectorXd& x_data, const Eigen::VectorXd& y_data,
                                                      const Eigen::VectorXd& weights, const Eigen::VectorXd params_initial,
                                                      const Eigen::VectorXd& lower_bounds, const Eigen::VectorXd& upper_bounds,
                                                      const Eigen::Matrix<bool, Eigen::Dynamic, 1>& fixed_params) {

    // set up the ceres problem
    ceres::Problem problem;

    Eigen::VectorXd params(params_initial);

    // add a residual block for the entire dataset (all x_ and y_ points)
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CostFunction1D<Model>, ceres::DYNAMIC, Model::nparams>(
            new CostFunction1D<Model>(x_data, y_data, weights),
            static_cast<int>(x_data.size())),  // number of residuals is the number of data points
        nullptr,  // no loss function
        params.data()  // pass parameters
        );

    return ceres_fit<Model>(problem, params, lower_bounds, upper_bounds, fixed_params);
}

// specialization for 2d fitting
template <typename Model>
std::pair<Eigen::VectorXd, Eigen::VectorXd> fit_model(const Eigen::MatrixXd& x_data, const Eigen::MatrixXd& y_data,
                                                      const Eigen::MatrixXd& z_data, const Eigen::MatrixXd& weights,
                                                      const Eigen::VectorXd params_initial, const Eigen::VectorXd& lower_bounds,
                                                      const Eigen::VectorXd& upper_bounds,
                                                      const Eigen::Matrix<bool, Eigen::Dynamic, 1>& fixed_params) {

    // set up the ceres problem
    ceres::Problem problem;

    Eigen::VectorXd params(params_initial);

    // add a residual block for the entire dataset (all x_, y_, z_ points)
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CostFunction2D<Model>, ceres::DYNAMIC, Model::nparams>(
            new CostFunction2D<Model>(x_data, y_data, z_data, weights),
            static_cast<int>(x_data.size())),  // number of residuals is the number of data points
        nullptr,  // no loss function
        params.data()  // pass parameters
        );

    return ceres_fit<Model>(problem, params, lower_bounds, upper_bounds, fixed_params);
}
} // namespace
