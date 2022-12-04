#pragma once

#include <Eigen/Core>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/gauss_models.h>

namespace engine_utils {

class mapFitter {
public:
    enum FitMode {
        peakValue = 0,
        centerValue = 1,
        aptTable = 2,
        };

    // box around source fit
    double bounding_box_pix;

    double flux_low = 0.75;
    double flux_high = 1.5;

    double fwhm_low = 0.1;
    double fwhm_high = 2.0;

    double ang_low = -pi/2;
    double ang_high = pi/2;

    template <typename Model, typename Derived>
    auto ceres_fit(const Model &,
                   const typename Model::InputType &,
                   const typename Model::InputDataType &,
                   const typename Model::DataType &,
                   const typename Model::DataType &,
                   const Eigen::DenseBase<Derived> &);

    template <mapFitter::FitMode fit_mode, typename Derived>
    auto fit_to_gaussian(Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &,
                         double);

};

template <typename Model, typename Derived>
auto mapFitter::ceres_fit(const Model &model,
                          const typename Model::InputType &init_params,
                          const typename Model::InputDataType &xy_data,
                          const typename Model::DataType &z_data,
                          const typename Model::DataType &sigma,
                          const Eigen::DenseBase<Derived> &limits) {

    using Fitter = CeresAutoDiffFitter<Model>;
    Fitter* fitter = new Fitter(&model, z_data.size());
    Eigen::Map<const typename Model::InputDataType> _x(xy_data.data(), xy_data.rows(), xy_data.cols());
    Eigen::Map<const typename Fitter::ValueType> _y(z_data.data(), z_data.size());
    Eigen::Map<const typename Fitter::ValueType> _s(sigma.data(), sigma.size());

    fitter->xdata = &_x;
    fitter->ydata = &_y;
    fitter->sigma = &_s;

    CostFunction* cost_function =
        new AutoDiffCostFunction<Fitter, Fitter::ValuesAtCompileTime, Fitter::InputsAtCompileTime>(fitter, fitter->values());

    Eigen::VectorXd params(init_params);
    auto problem = fitter->createProblem(params.data());

    // including CauchyLoss(0.5) leads to large covariances.
    problem->AddResidualBlock(cost_function, nullptr, params.data());

    // set limits
    for (int i = 0; i < limits.rows(); ++i) {
        problem->SetParameterLowerBound(params.data(), i, limits(i,0));
        problem->SetParameterUpperBound(params.data(), i, limits(i,1));
    }

    // indices to hold constant
    /*std::vector<int> sspv;
    sspv.push_back(limits.rows()-1);
    if (sspv.size() > 0 ){
        ceres::SubsetParameterization *pcssp
                = new ceres::SubsetParameterization(limits.rows(), sspv);
        problem->SetParameterization(params.data(), pcssp);
    }*/

    Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
    // disable logging
    options.logging_type = ceres::LoggingType::SILENT;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;
    Solve(options, problem.get(), &summary);

    // for storing uncertainties
    Eigen::VectorXd uncertainty(params.size());

    // get uncertainty
    if (summary.IsSolutionUsable()) {
        // for storing covariance matrix
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> covariance_matrix;

        // set covariance options
        Covariance::Options covariance_options;
        // EIGEN_SPARSE and DENSE_SVD are the slower, but more accurate options
        covariance_options.sparse_linear_algebra_library_type = ceres::SparseLinearAlgebraLibraryType::EIGEN_SPARSE;
        covariance_options.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
        Covariance covariance(covariance_options);

        std::vector<std::pair<const double*, const double*>> covariance_blocks;
        covariance_blocks.push_back(std::make_pair(params.data(), params.data()));
        auto covariance_result = covariance.Compute(covariance_blocks, problem.get());

        // if covariance calculation suceeded
        if (covariance_result) {
            covariance_matrix.resize(params.size(),params.size());
            covariance.GetCovarianceBlock(params.data(),params.data(),covariance_matrix.data());
            // calculate uncertainty
            uncertainty = covariance_matrix.diagonal().cwiseSqrt();
        }
        else {
            uncertainty.setConstant(-99);
        }
    }
    else {
        params.setConstant(-99);
        uncertainty.setConstant(-99);
    }

    return std::tuple<Eigen::VectorXd, Eigen::VectorXd,bool>(params,uncertainty,summary.IsSolutionUsable());
}

template <mapFitter::FitMode fit_mode, typename Derived>
auto mapFitter::fit_to_gaussian(Eigen::DenseBase<Derived> &signal, Eigen::DenseBase<Derived> &weight,
                                double init_fwhm) {

    // initial parameters and limits
    Eigen::VectorXd init_params(6);
    Eigen::MatrixXd limits(6,2);

    double init_row, init_col, init_flux;

    if constexpr (fit_mode == centerValue) {
        init_row = signal.rows()/2;
        init_col = signal.cols()/2;

        init_flux = signal(static_cast<int>(init_row), static_cast<int>(init_col));
    }

    else if constexpr (fit_mode == peakValue) {
        auto sig2noise = signal.derived().array()*sqrt(weight.derived().array());
        sig2noise.maxCoeff(&init_row, &init_col);

        init_flux = signal(static_cast<int>(init_row), static_cast<int>(init_col));
    }

    init_params << init_flux, init_col, init_row, init_fwhm, init_fwhm, 0;

    double lower_row, lower_col, upper_row, upper_col;

    // ensure lower limits are not less than zero
    lower_row = std::max(init_row - bounding_box_pix, 0.0);
    lower_col = std::max(init_col - bounding_box_pix, 0.0);

    // ensure upper limits are not bigger than the map
    upper_row = std::min(init_row + bounding_box_pix, static_cast<double>(signal.rows()) - 1);
    upper_col = std::min(init_col + bounding_box_pix, static_cast<double>(signal.cols()) - 1);

    double n_rows = upper_row - lower_row + 1;
    double n_cols = upper_col - lower_col + 1;

    // set lower limits
    limits.col(0) << flux_low*init_flux, lower_col, lower_row, fwhm_low*init_fwhm,
        fwhm_low*init_fwhm, ang_low;

    // set upper limits
    limits.col(1) << flux_high*init_flux, upper_col, upper_row, fwhm_high*init_fwhm,
        fwhm_high*init_fwhm, ang_high;

    Eigen::VectorXd x,y;

    // axes coordinate vectors for meshgrid
    x = Eigen::VectorXd::LinSpaced(n_cols, lower_col, upper_col);
    y = Eigen::VectorXd::LinSpaced(n_rows, lower_row, upper_row);

    SPDLOG_INFO("x {} y {}",x,y);
    SPDLOG_INFO("limtis {}", limits);
    SPDLOG_INFO("init_params {}", init_params);

    // create gaussian 2d model
    auto g = create_model<Gaussian2D>(init_params);
    // get meshgrid
    auto xy = g.meshgrid(x, y);

    Eigen::MatrixXd sigma(weight.rows(), weight.cols());

    for (Eigen::Index i=0; i<weight.rows(); i++) {
        for (Eigen::Index j=0; j<weight.cols(); j++) {
            if (weight(i,j)!=0) {
                sigma(i,j) = 1./sqrt(weight(i,j));
            }
            else {
                sigma(i,j) = 0;
            }
        }
    }

    // copy data and sigma within bounding box region
    Eigen::MatrixXd _signal = signal.block(lower_row, lower_col, n_rows, n_cols);
    Eigen::MatrixXd _sigma = sigma.block(lower_row, lower_col, n_rows, n_cols);

    return ceres_fit(g, init_params, xy, _signal, _sigma, limits);
}

} //namespace engine_utils
