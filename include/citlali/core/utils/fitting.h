#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/QR>

#include <tula/algorithm/ei_stats.h>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/gauss_models.h>

namespace engine_utils {

class mapFitter {
public:
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

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

    struct FitDiagnostics {
        bool valid = false;
        Eigen::VectorXd init_params;
        Eigen::VectorXd lower_limits;
        Eigen::VectorXd upper_limits;
        Eigen::VectorXi hit_lower;
        Eigen::VectorXi hit_upper;
        Eigen::VectorXd frac_in_range;
        Eigen::Index lower_row = -1;
        Eigen::Index lower_col = -1;
        Eigen::Index upper_row = -1;
        Eigen::Index upper_col = -1;
        double map_sigma = std::numeric_limits<double>::quiet_NaN();
        Eigen::Index sigma_nonzero = 0;
    };

    template <typename Model, typename Derived>
    auto ceres_fit(const Model &,
                   const typename Model::InputType &,
                   const typename Model::InputDataType &,
                   const typename Model::DataType &,
                   const typename Model::DataType &,
                   const Eigen::DenseBase<Derived> &,
                   bool use_ceres_covariance = true);

    template <mapFitter::FitMode fit_mode, typename Derived>
    auto fit_to_gaussian(Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &,
                         double, double, double, FitDiagnostics *diag = nullptr);
};

template <typename Model, typename Derived>
auto mapFitter::ceres_fit(const Model &model,
                          const typename Model::InputType &init_params,
                          const typename Model::InputDataType &xy_data,
                          const typename Model::DataType &z_data,
                          const typename Model::DataType &sigma,
                          const Eigen::DenseBase<Derived> &limits,
                          bool use_ceres_covariance) {
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

    logger->info("ceres_fit begin: values={} params={} fit_angle={} sigma_nonzero={}/{}",
                 z_data.size(), init_params.size(), fit_angle,
                 (_s.array() > 0).count(), _s.size());

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
            logger->info("ceres_fit angle fixed via subset parameterization");
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
    logger->info("ceres_fit solve start");
    Solve(options, problem.get(), &summary);
    logger->info("ceres_fit solve done: usable={} brief={}",
                 summary.IsSolutionUsable(), summary.BriefReport());
    if (!summary.IsSolutionUsable()) {
        logger->warn("ceres_fit full report:\n{}", summary.FullReport());
    }

    // vector for storing uncertainties
    Eigen::VectorXd uncertainty(params.size());

    // get uncertainty if solution is usable
    if (summary.IsSolutionUsable()) {
        if (use_ceres_covariance) {
            // set covariance options
            Covariance::Options covariance_options;
            // Keep covariance single-threaded for deterministic diagnostics.
            covariance_options.num_threads = 1;
            // gets rid of error messages related to bad fits
            covariance_options.null_space_rank = -1;
            // create covariance object with current covariance options
            Covariance covariance(covariance_options);

            // set up covariance block
            std::vector<std::pair<const double*, const double*>> covariance_blocks;
            // populate covariance block
            covariance_blocks.push_back(std::make_pair(params.data(), params.data()));
            // compute covariance
            logger->info("ceres_fit covariance start");
            auto covariance_result = covariance.Compute(covariance_blocks, problem.get());
            logger->info("ceres_fit covariance done: success={}", covariance_result);

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
        } else {
            // Beammap fallback: linearized covariance from J^T J at the solution.
            logger->info("ceres_fit covariance disabled; using linearized uncertainty estimate");
            ceres::Problem::EvaluateOptions eval_options;
            eval_options.apply_loss_function = false;
            eval_options.num_threads = 1;
            eval_options.parameter_blocks.push_back(params.data());

            const Eigen::Index n_params = params.size();
            double cost = 0.0;
            ceres::CRSMatrix jacobian;
            const bool eval_ok = problem->Evaluate(eval_options, &cost, nullptr, nullptr, &jacobian);
            if (!eval_ok) {
                logger->warn("ceres_fit linearized uncertainty failed: problem->Evaluate returned false");
                uncertainty.setConstant(0);
            } else {
                const bool valid_structure =
                    jacobian.num_rows > 0 &&
                    jacobian.num_cols > 0 &&
                    jacobian.rows.size() == static_cast<std::size_t>(jacobian.num_rows + 1) &&
                    jacobian.cols.size() == jacobian.values.size();

                if (!valid_structure) {
                    logger->warn("ceres_fit linearized uncertainty failed: invalid CRS Jacobian structure (rows={} cols={} nnz={})",
                                 jacobian.num_rows, jacobian.num_cols, jacobian.values.size());
                    uncertainty.setConstant(0);
                } else {
                    const Eigen::Index n_var = jacobian.num_cols;
                    Eigen::MatrixXd jtj = Eigen::MatrixXd::Zero(n_var, n_var);

                    bool invalid_index = false;
                    for (int r = 0; r < jacobian.num_rows; ++r) {
                        const int begin = jacobian.rows[r];
                        const int end = jacobian.rows[r + 1];
                        if (begin < 0 || end < begin || end > static_cast<int>(jacobian.values.size())) {
                            invalid_index = true;
                            break;
                        }
                        for (int a = begin; a < end; ++a) {
                            const int c1 = jacobian.cols[a];
                            const double v1 = jacobian.values[a];
                            if (c1 < 0 || c1 >= jacobian.num_cols || !std::isfinite(v1)) {
                                invalid_index = true;
                                break;
                            }
                            for (int b = a; b < end; ++b) {
                                const int c2 = jacobian.cols[b];
                                const double v2 = jacobian.values[b];
                                if (c2 < 0 || c2 >= jacobian.num_cols || !std::isfinite(v2)) {
                                    invalid_index = true;
                                    break;
                                }
                                const double contrib = v1 * v2;
                                jtj(c1, c2) += contrib;
                                if (c1 != c2) {
                                    jtj(c2, c1) += contrib;
                                }
                            }
                            if (invalid_index) {
                                break;
                            }
                        }
                        if (invalid_index) {
                            break;
                        }
                    }

                    if (invalid_index || !jtj.array().isFinite().all()) {
                        logger->warn("ceres_fit linearized uncertainty failed: invalid Jacobian values");
                        uncertainty.setConstant(0);
                    } else {
                        const double max_diag = jtj.diagonal().cwiseAbs().maxCoeff();
                        const double reg = std::max(1e-14, 1e-10 * (std::isfinite(max_diag) ? max_diag : 1.0));
                        jtj.diagonal().array() += reg;

                        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(jtj);
                        const Eigen::MatrixXd jtj_inv = cod.pseudoInverse();
                        const Eigen::VectorXd sigma_var = jtj_inv.diagonal().cwiseAbs().cwiseSqrt();

                        uncertainty.setZero();
                        const Eigen::Index n_fill = std::min<Eigen::Index>(n_params, sigma_var.size());
                        uncertainty.head(n_fill) = sigma_var.head(n_fill);

                        const Eigen::Index n_nonzero_sigma = (_s.array() > 0.0).count();
                        const Eigen::Index dof = std::max<Eigen::Index>(1, n_nonzero_sigma - n_var);
                        const double reduced_chi2 = std::max(0.0, (2.0 * cost) / static_cast<double>(dof));
                        const double scale = std::sqrt(std::max(1e-14, reduced_chi2));
                        uncertainty.head(n_fill).array() *= scale;
                    }
                }
            }
        }
    }
    // if fit is bad, set both fit and uncertainty to zero
    else {
        params.setConstant(0);
        uncertainty.setConstant(0);
    }

    // sanitize uncertainty for downstream uses that divide by perrors.
    uncertainty = uncertainty.unaryExpr([](double v) {
        return (std::isfinite(v) && v > 0.0) ? v : 0.0;
    });

    return std::tuple<Eigen::VectorXd, Eigen::VectorXd,bool>(params,uncertainty,summary.IsSolutionUsable());
}

template <mapFitter::FitMode fit_mode, typename Derived>
auto mapFitter::fit_to_gaussian(Eigen::DenseBase<Derived> &signal, Eigen::DenseBase<Derived> &weight,
                                double init_fwhm, double init_row, double init_col, FitDiagnostics *diag) {

    if (signal.rows() <= 0 || signal.cols() <= 0 ||
        weight.rows() != signal.rows() || weight.cols() != signal.cols()) {
        Eigen::VectorXd p = Eigen::VectorXd::Zero(n_params);
        Eigen::VectorXd e = Eigen::VectorXd::Zero(n_params);
        logger->warn("fit_to_gaussian skipped due to invalid map geometry (signal={}x{}, weight={}x{})",
                     signal.rows(), signal.cols(), weight.rows(), weight.cols());
        return std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>(p, e, false);
    }

    // initial parameters and limits
    Eigen::VectorXd init_params(n_params);
    Eigen::MatrixXd limits(n_params,2);

    // intiial position and flux
    double init_flux = 0;

    // initial gaussian standard deviation
    double init_sigma = init_fwhm*FWHM_TO_STD;

    // if no initial position is input find peak
    if (init_row < 0 || init_col < 0) {
        // center positions
        double center_row = (signal.rows() - 1)/2;
        double center_col = (signal.cols() - 1)/2;

        // signal-to-noise map
        auto sig2noise = signal.derived().array()*sqrt(weight.derived().array());

        Eigen::Index ir = static_cast<Eigen::Index>(center_row);
        Eigen::Index ic = static_cast<Eigen::Index>(center_col);
        bool found_peak = false;

        // find peak in the entire map
        if (fitting_region_pix <= 0) {
            sig2noise.maxCoeff(&ir, &ic);
            init_flux = signal(ir,ic);
            found_peak = true;
        }
        // find peak within inner radius
        else {
            for (Eigen::Index i=0; i<sig2noise.rows(); ++i) {
                for (Eigen::Index j=0; j<sig2noise.cols(); ++j) {
                    auto dist = sqrt(pow(i - center_row,2) + pow(j - center_col,2));
                    if (dist < fitting_region_pix) {
                        if (sig2noise(i,j) > init_flux) {
                            init_flux = sig2noise(i,j);
                            ir = i;
                            ic = j;
                            found_peak = true;
                        }
                    }
                }
            }
            // initial guess for flux
            if (found_peak) {
                init_flux = signal(ir, ic);
            }
        }

        // fall back to global max if the inner-radius search found nothing
        if (!found_peak) {
            sig2noise.maxCoeff(&ir, &ic);
            init_flux = signal(ir, ic);
        }

        init_row = ir;
        init_col = ic;
    }    
    // otherwise use the input initial position
    else {
        init_row = std::clamp(init_row, 0.0, static_cast<double>(signal.rows() - 1));
        init_col = std::clamp(init_col, 0.0, static_cast<double>(signal.cols() - 1));
        init_flux = signal(static_cast<Eigen::Index>(init_row), static_cast<Eigen::Index>(init_col));
    }

    // initial parameter guesses (order of positions is x,y = col,row)
    init_params << init_flux, init_col, init_row, init_sigma, init_sigma, 0;

    // limits of bounding box
    Eigen::Index lower_row, lower_col, upper_row, upper_col;

    // ignore bounding box if less than/equal to zero
    if (bounding_box_pix <= 0) {
        lower_row = 0;
        lower_col = 0;
        upper_row = signal.rows() - 1;
        upper_col = signal.cols() - 1;
    }
    // determine bounding box size
    else {
        const double lower_row_d = std::max(init_row - bounding_box_pix, 0.0);
        const double lower_col_d = std::max(init_col - bounding_box_pix, 0.0);
        const double upper_row_d = std::min(init_row + bounding_box_pix, static_cast<double>(signal.rows()) - 1);
        const double upper_col_d = std::min(init_col + bounding_box_pix, static_cast<double>(signal.cols()) - 1);

        // ensure lower limits of bounding box are not less than zero
        lower_row = static_cast<Eigen::Index>(std::floor(lower_row_d));
        lower_col = static_cast<Eigen::Index>(std::floor(lower_col_d));

        // ensure upper limits of bounding box are not bigger than the map
        upper_row = static_cast<Eigen::Index>(std::ceil(upper_row_d));
        upper_col = static_cast<Eigen::Index>(std::ceil(upper_col_d));

        lower_row = std::clamp<Eigen::Index>(lower_row, 0, signal.rows() - 1);
        lower_col = std::clamp<Eigen::Index>(lower_col, 0, signal.cols() - 1);
        upper_row = std::clamp<Eigen::Index>(upper_row, 0, signal.rows() - 1);
        upper_col = std::clamp<Eigen::Index>(upper_col, 0, signal.cols() - 1);
    }

    // size of bounding box region
    Eigen::Index n_rows = upper_row - lower_row + 1;
    Eigen::Index n_cols = upper_col - lower_col + 1;
    if (n_rows <= 0 || n_cols <= 0) {
        Eigen::VectorXd p = Eigen::VectorXd::Zero(n_params);
        Eigen::VectorXd e = Eigen::VectorXd::Zero(n_params);
        logger->warn("fit_to_gaussian skipped due to empty bounding box");
        return std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>(p, e, false);
    }

    // preserve valid ordering even when init_flux is negative.
    double amp_lower = std::min(flux_low * init_flux, flux_high * init_flux);
    double amp_upper = std::max(flux_low * init_flux, flux_high * init_flux);
    if (!std::isfinite(amp_lower) || !std::isfinite(amp_upper)) {
        amp_lower = -std::abs(init_flux);
        amp_upper = std::abs(init_flux);
    }

    // set lower limits of fitting parameters
    limits.col(0) << amp_lower, static_cast<double>(lower_col), static_cast<double>(lower_row), fwhm_low*init_sigma,
        fwhm_low*init_sigma, angle_low;

    // set upper limits of fitting parameters
    limits.col(1) << amp_upper, static_cast<double>(upper_col), static_cast<double>(upper_row), fwhm_high*init_sigma,
        fwhm_high*init_sigma, angle_high;

    Eigen::VectorXd x, y;

    // axes coordinate vectors for meshgrid
    x = Eigen::VectorXd::LinSpaced(n_cols, static_cast<double>(lower_col), static_cast<double>(upper_col));
    y = Eigen::VectorXd::LinSpaced(n_rows, static_cast<double>(lower_row), static_cast<double>(upper_row));

    // create gaussian 2d model
    auto g = create_model<Gaussian2D>(init_params);
    // get meshgrid
    auto xy = g.meshgrid(x, y);

    // get map stddev
    double map_sigma = engine_utils::calc_std_dev(signal);
    if (!std::isfinite(map_sigma) || map_sigma <= 0.0) {
        map_sigma = 1.0;
    }

    // standard deviation of signal map
    Eigen::MatrixXd sigma(weight.rows(), weight.cols());

    // loop through pixels
    for (Eigen::Index i=0; i<weight.rows(); ++i) {
        for (Eigen::Index j=0; j<weight.cols(); ++j) {
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

    // avoid running Ceres on an unconstrained region.
    if ((_sigma.array() > 0.0).count() < n_params) {
        Eigen::VectorXd p = Eigen::VectorXd::Zero(n_params);
        Eigen::VectorXd e = Eigen::VectorXd::Zero(n_params);
        return std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>(p, e, false);
    }

    if (diag != nullptr) {
        diag->valid = true;
        diag->init_params = init_params;
        diag->lower_limits = limits.col(0);
        diag->upper_limits = limits.col(1);
        diag->hit_lower = Eigen::VectorXi::Zero(n_params);
        diag->hit_upper = Eigen::VectorXi::Zero(n_params);
        diag->frac_in_range = Eigen::VectorXd::Constant(n_params, std::numeric_limits<double>::quiet_NaN());
        diag->lower_row = lower_row;
        diag->lower_col = lower_col;
        diag->upper_row = upper_row;
        diag->upper_col = upper_col;
        diag->map_sigma = map_sigma;
        diag->sigma_nonzero = (_sigma.array() > 0.0).count();
    }

    // do the fit
    constexpr bool use_ceres_covariance = (fit_mode == FitMode::pointing);
    auto [fit_params, fit_uncertainty, fit_is_good] =
        ceres_fit(g, init_params, xy, _signal, _sigma, limits, use_ceres_covariance);

    if (diag != nullptr && diag->valid) {
        for (Eigen::Index i = 0; i < n_params; ++i) {
            const double p = fit_params(i);
            const double low = diag->lower_limits(i);
            const double high = diag->upper_limits(i);
            const double span = high - low;
            const double tol = std::max(1e-9, 1e-6 * std::max(1.0, std::abs(span)));
            if (std::isfinite(p) && std::isfinite(low) && std::isfinite(high)) {
                diag->hit_lower(i) = (p <= low + tol) ? 1 : 0;
                diag->hit_upper(i) = (p >= high - tol) ? 1 : 0;
                if (std::isfinite(span) && std::abs(span) > 0.0) {
                    diag->frac_in_range(i) = (p - low) / span;
                }
            }
        }
    }

    return std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>(fit_params, fit_uncertainty, fit_is_good);
}
} //namespace engine_utils
