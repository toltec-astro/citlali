#pragma once

#include <Eigen/Core>

#include <tula/algorithm/ei_stats.h>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/gauss_models.h>

namespace engine_utils {

class mapFitter {
public:
    enum FitMode {
        pointing = 0,
        beammap = 1,
        };

    // number of parameters
    int n_params = 6;

    // box around source fit
    double bounding_box_pix;
    double fitting_region_pix;

    // fitting limits from config file
    Eigen::VectorXd flux_limits, fwhm_limits;

    // flux lower limit factor
    double flux_low = 0.1;
    // flux upper limit factor
    double flux_high = 2.0;

    // fwhm lower limit factor
    double fwhm_low = 0.1;
    // fwhm upper limit factor
    double fwhm_high = 2.0;

    // fit rotation angle?
    bool fit_angle;

    //lower limit on rotation angle
    double angle_low = -pi/2;
    // upper limit on rotation angle
    double angle_high = pi/2;

    template <typename Model, typename Derived>
    auto ceres_fit(const Model &,
                   const typename Model::InputType &,
                   const typename Model::InputDataType &,
                   const typename Model::DataType &,
                   const typename Model::DataType &,
                   const Eigen::DenseBase<Derived> &);

    template <mapFitter::FitMode fit_mode, typename Derived>
    auto fit_to_gaussian(Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &,
                         double, double, double);
};

template <typename Model, typename Derived>
auto mapFitter::ceres_fit(const Model &model,
                          const typename Model::InputType &init_params,
                          const typename Model::InputDataType &xy_data,
                          const typename Model::DataType &z_data,
                          const typename Model::DataType &sigma,
                          const Eigen::DenseBase<Derived> &limits) {

    // fitter
    using Fitter = CeresAutoDiffFitter<Model>;
    Fitter* fitter = new Fitter(&model, z_data.size());
    Eigen::Map<const typename Model::InputDataType> _x(xy_data.data(), xy_data.rows(), xy_data.cols());
    Eigen::Map<const typename Fitter::ValueType> _y(z_data.data(), z_data.size());
    Eigen::Map<const typename Fitter::ValueType> _s(sigma.data(), sigma.size());

    // get x, y, and sigma
    fitter->xdata = &_x;
    fitter->ydata = &_y;
    fitter->sigma = &_s;

    // define cost function
    CostFunction* cost_function =
        new AutoDiffCostFunction<Fitter, Fitter::ValuesAtCompileTime, Fitter::InputsAtCompileTime>(fitter, fitter->values());

    // parameter vector
    Eigen::VectorXd params(init_params);
    auto problem = fitter->createProblem(params.data());

    // including CauchyLoss(0.5) leads to large covariances.
    problem->AddResidualBlock(cost_function, nullptr, params.data());
    //problem->AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), params.data());

    // set limits
    for (int i=0; i<limits.rows(); ++i) {
        problem->SetParameterLowerBound(params.data(), i, limits(i,0));
        problem->SetParameterUpperBound(params.data(), i, limits(i,1));
    }

    // vector to store indices of parameters to keep constant

    if (!fit_angle) {
        std::vector<int> sspv;
        sspv.push_back(limits.rows()-1);
        // mark parameter as constant
        if (sspv.size() > 0 ){
            ceres::SubsetParameterization *pcssp
                    = new ceres::SubsetParameterization(limits.rows(), sspv);
            problem->SetParameterization(params.data(), pcssp);
        }
    }

    // apply solver options
    Solver::Options options;
    // method
    options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
    // disable logging
    options.logging_type = ceres::LoggingType::SILENT;
    // disable output
    options.minimizer_progress_to_stdout = false;
    // output info
    Solver::Summary summary;
    // run the fit
    Solve(options, problem.get(), &summary);

    // vector for storing uncertainties
    Eigen::VectorXd uncertainty(params.size());

    // get uncertainty if solution is usable
    if (summary.IsSolutionUsable()) {
        // set covariance options
        Covariance::Options covariance_options;
        // EIGEN_SPARSE and DENSE_SVD are the slower, but more accurate options
        covariance_options.sparse_linear_algebra_library_type = ceres::SparseLinearAlgebraLibraryType::EIGEN_SPARSE;
        covariance_options.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
        // gets rid of error messages related to bad fits
        covariance_options.null_space_rank = -1;
        // create covariance object with current covariance options
        Covariance covariance(covariance_options);

        // set up covariance block
        std::vector<std::pair<const double*, const double*>> covariance_blocks;
        // populate covariance block
        covariance_blocks.push_back(std::make_pair(params.data(), params.data()));
        // compute covariance
        auto covariance_result = covariance.Compute(covariance_blocks, problem.get());

        // if covariance calculation suceeded
        if (covariance_result) {
            // for storing covariance matrix
            Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> covariance_matrix;
            covariance_matrix.resize(params.size(),params.size());
            covariance.GetCovarianceBlock(params.data(),params.data(),covariance_matrix.data());
            // calculate uncertainty
            uncertainty = covariance_matrix.diagonal().cwiseSqrt();
        }
        // if covariance calculation fails, set uncertainty to zero
        else {
            uncertainty.setConstant(0);
        }
    }
    // if fit is bad, set both fit and uncertainty to zero
    else {
        params.setConstant(0);
        uncertainty.setConstant(0);
    }

    return std::tuple<Eigen::VectorXd, Eigen::VectorXd,bool>(params,uncertainty,summary.IsSolutionUsable());
}

template <mapFitter::FitMode fit_mode, typename Derived>
auto mapFitter::fit_to_gaussian(Eigen::DenseBase<Derived> &signal, Eigen::DenseBase<Derived> &weight,
                                double init_fwhm, double init_row, double init_col) {

    // initial parameters and limits
    Eigen::VectorXd init_params(n_params);
    Eigen::MatrixXd limits(n_params,2);

    // intiial position and flux
    double init_flux = 0;

    // initial gaussian standard deviation
    double init_sigma = init_fwhm*FWHM_TO_STD;

    // if no initial position is input find peak
    if (init_row<0 && init_col<0) {
        // center positions
        double center_row = (signal.rows() - 1)/2;
        double center_col = (signal.cols() - 1)/2;

        // signal-to-noise map
        auto sig2noise = signal.derived().array()*sqrt(weight.derived().array());

        Eigen::Index ir, ic;

        // find peak in the entire map
        if (fitting_region_pix <= 0) {
            sig2noise.maxCoeff(&ir, &ic);
            init_flux = signal(ir,ic);
        }
        // find peak within inner radius
        else {
            for (Eigen::Index i=0; i<sig2noise.rows(); i++) {
                for (Eigen::Index j=0; j<sig2noise.cols(); j++) {
                    auto dist = sqrt(pow(i - center_row,2) + pow(j - center_col,2));
                    if (dist < fitting_region_pix) {
                        if (sig2noise(i,j) > init_flux) {
                            init_flux = sig2noise(i,j);
                            ir = i;
                            ic = j;
                        }
                    }
                }
            }
            // initial guess for flux
            init_flux = signal(ir, ic);
        }

        init_row = ir;
        init_col = ic;
    }    
    // otherwise use the input initial position
    else {
        init_flux = signal(static_cast<int>(init_row), static_cast<int>(init_col));
    }

    // initial parameter guesses (order of positions is x,y = col,row)
    init_params << init_flux, init_col, init_row, init_sigma, init_sigma, 0;

    // limits of bounding box
    double lower_row, lower_col, upper_row, upper_col;

    // ignore bounding box if less than/equal to zero
    if (bounding_box_pix <= 0) {
        lower_row = 0;
        upper_row = 0;

        upper_row = signal.rows() - 1;
        upper_col = signal.cols() - 1;
    }
    // determine bounding box size
    else {
        // ensure lower limits of bounding box are not less than zero
        lower_row = std::max(init_row - bounding_box_pix, 0.0);
        lower_col = std::max(init_col - bounding_box_pix, 0.0);

        // ensure upper limits of bounding box are not bigger than the map
        upper_row = std::min(init_row + bounding_box_pix, static_cast<double>(signal.rows()) - 1);
        upper_col = std::min(init_col + bounding_box_pix, static_cast<double>(signal.cols()) - 1);
    }

    // size of bounding box region
    double n_rows = upper_row - lower_row + 1;
    double n_cols = upper_col - lower_col + 1;

    // set lower limits of fitting parameters
    limits.col(0) << flux_low*init_flux, lower_col, lower_row, fwhm_low*init_sigma,
        fwhm_low*init_sigma, angle_low;

    // set upper limits of fitting parameters
    limits.col(1) << flux_high*init_flux, upper_col, upper_row, fwhm_high*init_sigma,
        fwhm_high*init_sigma, angle_high;

    Eigen::VectorXd x, y;

    // axes coordinate vectors for meshgrid
    x = Eigen::VectorXd::LinSpaced(n_cols, lower_col, upper_col);
    y = Eigen::VectorXd::LinSpaced(n_rows, lower_row, upper_row);

    // create gaussian 2d model
    auto g = create_model<Gaussian2D>(init_params);
    // get meshgrid
    auto xy = g.meshgrid(x, y);

    // get map stddev
    auto map_sigma = engine_utils::calc_std_dev(signal);

    // standard deviation of signal map
    Eigen::MatrixXd sigma(weight.rows(), weight.cols());

    for (Eigen::Index i=0; i<weight.rows(); i++) {
        for (Eigen::Index j=0; j<weight.cols(); j++) {
            if (weight(i,j)!=0) {
                // use map sigma for beammaps
                if constexpr (fit_mode == FitMode::beammap) {
                    sigma(i,j) = map_sigma;
                }
                // use weights for pointing
                else if constexpr (fit_mode == FitMode::pointing) {
                    sigma(i,j) = 1./sqrt(weight(i,j));
                }
            }
            else {
                sigma(i,j) = 0;
            }
        }
    }

    // copy data and sigma within bounding box region
    Eigen::MatrixXd _signal = signal.block(lower_row, lower_col, n_rows, n_cols);
    Eigen::MatrixXd _sigma = sigma.block(lower_row, lower_col, n_rows, n_cols);

    // do the fit and return
    return ceres_fit(g, init_params, xy, _signal, _sigma, limits);
}

} //namespace engine_utils
