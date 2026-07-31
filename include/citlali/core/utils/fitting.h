#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>
#include <ceres/version.h>

#include <tula/algorithm/ei_stats.h>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/gauss_models.h>
#include <citlali/core/utils/process_resource_snapshot.h>
#include <citlali/core/utils/utils.h>

namespace engine_utils {

inline double vector_median_copy(std::vector<double> values) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    if (values.size() % 2 != 0) {
        return *mid;
    }
    const auto lower = std::max_element(values.begin(), mid);
    return (*lower + *mid) / 2.0;
}

inline double vector_stddev(const std::vector<double> &values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    Eigen::Index n = 0;
    for (const double v : values) {
        if (std::isfinite(v)) {
            sum += v;
            ++n;
        }
    }
    if (n <= 0) {
        return 0.0;
    }
    const double mean = sum / static_cast<double>(n);
    double sumsq = 0.0;
    for (const double v : values) {
        if (std::isfinite(v)) {
            const double d = v - mean;
            sumsq += d * d;
        }
    }
    const double denom = static_cast<double>(n > 1 ? n - 1 : n);
    return std::sqrt(sumsq / denom);
}

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
    double beammap_fit_radius_fwhm = 0.0;

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
    auto flush_logger = [&]() {
        if (logger && logger->should_log(spdlog::level::debug)) {
            logger->flush();
        }
    };
    auto log_and_flush_warn = [&](const std::string &msg) {
        logger->warn(msg);
        if (logger) {
            logger->flush();
        }
    };
    auto matrix_stats = [](const auto &m) {
        struct Stats {
            Eigen::Index finite = 0;
            Eigen::Index positive = 0;
            double min = std::numeric_limits<double>::quiet_NaN();
            double max = std::numeric_limits<double>::quiet_NaN();
        };
        Stats stats;
        for (Eigen::Index i = 0; i < m.size(); ++i) {
            const double v = m.derived().coeff(i);
            if (std::isfinite(v)) {
                if (stats.finite == 0) {
                    stats.min = v;
                    stats.max = v;
                }
                else {
                    stats.min = std::min(stats.min, v);
                    stats.max = std::max(stats.max, v);
                }
                if (v > 0.0) {
                    ++stats.positive;
                }
                ++stats.finite;
            }
        }
        return stats;
    };

    const Eigen::Index expected_values = xy_data.rows();
    const Eigen::Index z_values = z_data.size();
    const Eigen::Index sigma_values = sigma.size();
    const bool preflight_ok =
        expected_values > 0 &&
        xy_data.cols() == Model::DimensionsAtCompileTime &&
        z_values == expected_values &&
        sigma_values == expected_values &&
        init_params.size() == model.params.size() &&
        limits.rows() == init_params.size() &&
        limits.cols() == 2;
    const auto xy_stats = matrix_stats(xy_data);
    const auto z_stats = matrix_stats(z_data);
    const auto sigma_stats = matrix_stats(sigma);

    logger->debug(
        "ceres_fit preflight: xy={}x{} z={}x{} sigma={}x{} values={} expected={} params={} model_params={} limits={}x{} "
        "ptrs xy={} z={} sigma={} init={} limits={} finite xy={}/{} z={}/{} sigma={}/{} sigma_pos={} "
        "ranges xy=[{:.6g}, {:.6g}] z=[{:.6g}, {:.6g}] sigma=[{:.6g}, {:.6g}]",
        xy_data.rows(), xy_data.cols(),
        z_data.rows(), z_data.cols(),
        sigma.rows(), sigma.cols(),
        z_values, expected_values,
        init_params.size(), model.params.size(),
        limits.rows(), limits.cols(),
        static_cast<const void*>(xy_data.data()),
        static_cast<const void*>(z_data.data()),
        static_cast<const void*>(sigma.data()),
        static_cast<const void*>(init_params.data()),
        static_cast<const void*>(limits.derived().data()),
        xy_stats.finite, xy_data.size(),
        z_stats.finite, z_data.size(),
        sigma_stats.finite, sigma.size(),
        sigma_stats.positive,
        xy_stats.min, xy_stats.max,
        z_stats.min, z_stats.max,
        sigma_stats.min, sigma_stats.max);
    flush_logger();

    if (!preflight_ok || xy_stats.finite != xy_data.size() ||
        z_stats.finite != z_data.size() || sigma_stats.finite != sigma.size() ||
        sigma_stats.positive < init_params.size()) {
        log_and_flush_warn(
            fmt::format("ceres_fit preflight failed: ok={} xy_finite={}/{} z_finite={}/{} sigma_finite={}/{} sigma_pos={} params={}",
                        preflight_ok,
                        xy_stats.finite, xy_data.size(),
                        z_stats.finite, z_data.size(),
                        sigma_stats.finite, sigma.size(),
                        sigma_stats.positive,
                        init_params.size()));
        Eigen::VectorXd p = Eigen::VectorXd::Zero(init_params.size());
        Eigen::VectorXd e = Eigen::VectorXd::Zero(init_params.size());
        return std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>(p, e, false);
    }

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

    logger->debug("ceres_fit begin: values={} params={} fit_angle={} sigma_nonzero={}/{}",
                  z_data.size(), init_params.size(), fit_angle,
                  (_s.array() > 0).count(), _s.size());
    flush_logger();

    // define cost function
    logger->debug("ceres_fit checkpoint: constructing AutoDiffCostFunction values={}", fitter->values());
    flush_logger();
    CostFunction* cost_function =
        new AutoDiffCostFunction<Fitter, Fitter::ValuesAtCompileTime, Fitter::InputsAtCompileTime>(fitter, fitter->values());
    logger->debug("ceres_fit checkpoint: AutoDiffCostFunction constructed ptr={}", static_cast<const void*>(cost_function));
    flush_logger();

    // parameter vector
    Eigen::VectorXd params(init_params);
    logger->debug("ceres_fit checkpoint: params copied size={} init=[{:.6g}, {:.6g}, {:.6g}, {:.6g}, {:.6g}, {:.6g}]",
                  params.size(),
                  params.size() > 0 ? params(0) : 0.0,
                  params.size() > 1 ? params(1) : 0.0,
                  params.size() > 2 ? params(2) : 0.0,
                  params.size() > 3 ? params(3) : 0.0,
                  params.size() > 4 ? params(4) : 0.0,
                  params.size() > 5 ? params(5) : 0.0);
    flush_logger();
    for (int i = 0; i < limits.rows(); ++i) {
        logger->debug("ceres_fit bounds preflight[{}]=[{:.6g}, {:.6g}] init={:.6g}",
                      i, limits(i, 0), limits(i, 1), params(i));
    }
    flush_logger();
    logger->debug("ceres_fit checkpoint: Problem construct start");
    flush_logger();
    ceres::Problem problem;
    logger->debug("ceres_fit checkpoint: Problem construct done ptr={}", static_cast<const void*>(&problem));
    flush_logger();

    // including CauchyLoss(0.5) leads to large covariances.
    logger->debug("ceres_fit checkpoint: AddResidualBlock start");
    flush_logger();
    problem.AddResidualBlock(cost_function, nullptr, params.data());
    logger->debug("ceres_fit checkpoint: AddResidualBlock done");
    flush_logger();
    //problem->AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), params.data());

    // set limits
    logger->debug("ceres_fit checkpoint: SetParameterBounds start rows={}", limits.rows());
    flush_logger();
    for (int i=0; i<limits.rows(); ++i) {
        logger->debug("ceres_fit bounds[{}]=[{:.6g}, {:.6g}]", i, limits(i,0), limits(i,1));
        problem.SetParameterLowerBound(params.data(), i, limits(i,0));
        problem.SetParameterUpperBound(params.data(), i, limits(i,1));
    }
    logger->debug("ceres_fit checkpoint: SetParameterBounds done");
    flush_logger();

    // vector to store indices of parameters to keep constant
    if (!fit_angle) {
        std::vector<int> sspv;
        sspv.push_back(limits.rows()-1);
        // mark parameter as constant
        if (sspv.size() > 0 ){
            logger->debug("ceres_fit checkpoint: SubsetParameterization start constant_index={}", sspv.front());
            flush_logger();
#if CERES_VERSION_MAJOR >= 2
            auto *pcssp = new ceres::SubsetManifold(limits.rows(), sspv);
#else
            auto *pcssp = new ceres::SubsetParameterization(limits.rows(), sspv);
#endif
            logger->debug("ceres_fit checkpoint: subset constraint constructed ptr={}", static_cast<const void*>(pcssp));
            flush_logger();
#if CERES_VERSION_MAJOR >= 2
            problem.SetManifold(params.data(), pcssp);
#else
            problem.SetParameterization(params.data(), pcssp);
#endif
            logger->debug("ceres_fit angle fixed via subset parameterization");
            flush_logger();
        }
    }

    logger->debug("ceres_fit checkpoint: residual pre-eval start values={}", fitter->values());
    flush_logger();
    std::vector<double> residuals(static_cast<std::size_t>(fitter->values()), 0.0);
    const bool residual_ok = (*fitter)(params.data(), residuals.data());
    Eigen::Index residual_finite = 0;
    double residual_abs_max = 0.0;
    double residual_sumsq = 0.0;
    for (const double r : residuals) {
        if (std::isfinite(r)) {
            ++residual_finite;
            residual_abs_max = std::max(residual_abs_max, std::abs(r));
            residual_sumsq += r * r;
        }
    }
    logger->debug("ceres_fit checkpoint: residual pre-eval done ok={} finite={}/{} abs_max={:.6g} rms={:.6g}",
                  residual_ok, residual_finite, residuals.size(), residual_abs_max,
                  residual_finite > 0 ? std::sqrt(residual_sumsq / static_cast<double>(residual_finite)) :
                      std::numeric_limits<double>::quiet_NaN());
    flush_logger();
    if (!residual_ok || residual_finite != static_cast<Eigen::Index>(residuals.size())) {
        log_and_flush_warn(
            fmt::format("ceres_fit skipped: invalid initial residuals ok={} finite={}/{}",
                        residual_ok, residual_finite, residuals.size()));
        Eigen::VectorXd p = Eigen::VectorXd::Zero(init_params.size());
        Eigen::VectorXd e = Eigen::VectorXd::Zero(init_params.size());
        return std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>(p, e, false);
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
    citlali::utils::log_process_resource_snapshot(
        logger, "ceres solve start");
    logger->debug("ceres_fit solve start");
    flush_logger();
    Solve(options, &problem, &summary);
    logger->debug("ceres_fit solve done: usable={} brief={}",
                  summary.IsSolutionUsable(), summary.BriefReport());
    citlali::utils::log_process_resource_snapshot(
        logger, "ceres solve done");
    flush_logger();
    if (!summary.IsSolutionUsable()) {
        logger->warn("ceres_fit failed: {}", summary.BriefReport());
        logger->debug("ceres_fit full report:\n{}", summary.FullReport());
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
            logger->debug("ceres_fit covariance start");
            auto covariance_result = covariance.Compute(covariance_blocks, &problem);
            logger->debug("ceres_fit covariance done: success={}", covariance_result);

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
            // Fallback: linearized covariance from J^T J at the solution.
            logger->debug("ceres_fit covariance disabled; using linearized uncertainty estimate");
            ceres::Problem::EvaluateOptions eval_options;
            eval_options.apply_loss_function = false;
            eval_options.num_threads = 1;
            eval_options.parameter_blocks.push_back(params.data());

            const Eigen::Index n_params = params.size();
            double cost = 0.0;
            ceres::CRSMatrix jacobian;
            const bool eval_ok = problem.Evaluate(eval_options, &cost, nullptr, nullptr, &jacobian);
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

    if (logger) {
        logger->debug("fit_to_gaussian entry: mode={} signal={}x{} weight={}x{} init_fwhm={:.6g} init_row={:.3f} init_col={:.3f} bbox_pix={:.3f} radius_pix={:.3f} fit_radius_fwhm={:.3f}",
                      fit_mode == FitMode::beammap ? "beammap" : "pointing",
                      signal.rows(), signal.cols(), weight.rows(), weight.cols(),
                      init_fwhm, init_row, init_col, bounding_box_pix,
                      fitting_region_pix, beammap_fit_radius_fwhm);
    }

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

    auto find_weighted_peak = [&](Eigen::Index row_lo, Eigen::Index row_hi,
                                  Eigen::Index col_lo, Eigen::Index col_hi,
                                  bool apply_radius, double center_row, double center_col, double radius_pix,
                                  Eigen::Index &best_row, Eigen::Index &best_col, double &best_snr) -> bool {
        best_row = -1;
        best_col = -1;
        best_snr = -std::numeric_limits<double>::infinity();

        row_lo = std::clamp<Eigen::Index>(row_lo, 0, signal.rows() - 1);
        row_hi = std::clamp<Eigen::Index>(row_hi, 0, signal.rows() - 1);
        col_lo = std::clamp<Eigen::Index>(col_lo, 0, signal.cols() - 1);
        col_hi = std::clamp<Eigen::Index>(col_hi, 0, signal.cols() - 1);
        if (row_hi < row_lo || col_hi < col_lo) {
            return false;
        }

        bool found = false;
        const double radius2 = radius_pix * radius_pix;
        for (Eigen::Index row = row_lo; row <= row_hi; ++row) {
            for (Eigen::Index col = col_lo; col <= col_hi; ++col) {
                const double s = signal(row, col);
                const double w = weight(row, col);
                if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                    continue;
                }
                if constexpr (fit_mode == FitMode::beammap) {
                    if (s <= 0.0) {
                        continue;
                    }
                }
                if (apply_radius) {
                    const double dr = static_cast<double>(row) - center_row;
                    const double dc = static_cast<double>(col) - center_col;
                    if (dr * dr + dc * dc >= radius2) {
                        continue;
                    }
                }
                const double snr = s * std::sqrt(w);
                if (!std::isfinite(snr)) {
                    continue;
                }
                if (!found || snr > best_snr) {
                    best_snr = snr;
                    best_row = row;
                    best_col = col;
                    found = true;
                }
            }
        }
        return found;
    };

    // if no initial position is input find peak
    if (init_row < 0 || init_col < 0) {
        // center positions
        double center_row = (signal.rows() - 1)/2;
        double center_col = (signal.cols() - 1)/2;
        Eigen::Index ir = -1;
        Eigen::Index ic = -1;
        double best_snr = -std::numeric_limits<double>::infinity();
        bool found_peak = false;

        // find peak in the entire map
        if (fitting_region_pix <= 0) {
            found_peak = find_weighted_peak(0, signal.rows() - 1, 0, signal.cols() - 1,
                                            false, center_row, center_col, 0.0,
                                            ir, ic, best_snr);
        }
        // find peak within inner radius
        else {
            found_peak = find_weighted_peak(0, signal.rows() - 1, 0, signal.cols() - 1,
                                            true, center_row, center_col, fitting_region_pix,
                                            ir, ic, best_snr);
        }

        // fall back to global max if the inner-radius search found nothing
        if (!found_peak) {
            found_peak = find_weighted_peak(0, signal.rows() - 1, 0, signal.cols() - 1,
                                            false, center_row, center_col, 0.0,
                                            ir, ic, best_snr);
        }

        if (!found_peak) {
            Eigen::VectorXd p = Eigen::VectorXd::Zero(n_params);
            Eigen::VectorXd e = Eigen::VectorXd::Zero(n_params);
            logger->warn("fit_to_gaussian skipped: unable to find valid weighted seed pixel");
            return std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>(p, e, false);
        }

        init_row = static_cast<double>(ir);
        init_col = static_cast<double>(ic);
        init_flux = signal(ir, ic);
    }    
    // otherwise use the input initial position
    else {
        init_row = std::clamp(init_row, 0.0, static_cast<double>(signal.rows() - 1));
        init_col = std::clamp(init_col, 0.0, static_cast<double>(signal.cols() - 1));
        Eigen::Index ir = static_cast<Eigen::Index>(std::llround(init_row));
        Eigen::Index ic = static_cast<Eigen::Index>(std::llround(init_col));
        ir = std::clamp<Eigen::Index>(ir, 0, signal.rows() - 1);
        ic = std::clamp<Eigen::Index>(ic, 0, signal.cols() - 1);

        const double seed_w = weight(ir, ic);
        const double seed_s = signal(ir, ic);
        bool seed_valid = std::isfinite(seed_w) && seed_w > 0.0 && std::isfinite(seed_s);
        if constexpr (fit_mode == FitMode::beammap) {
            seed_valid = seed_valid && seed_s > 0.0;
        }
        if (!seed_valid) {
            Eigen::Index search_row_lo = 0;
            Eigen::Index search_row_hi = signal.rows() - 1;
            Eigen::Index search_col_lo = 0;
            Eigen::Index search_col_hi = signal.cols() - 1;
            if (bounding_box_pix > 0) {
                search_row_lo = std::clamp<Eigen::Index>(
                    static_cast<Eigen::Index>(std::floor(init_row - bounding_box_pix)), 0, signal.rows() - 1);
                search_row_hi = std::clamp<Eigen::Index>(
                    static_cast<Eigen::Index>(std::ceil(init_row + bounding_box_pix)), 0, signal.rows() - 1);
                search_col_lo = std::clamp<Eigen::Index>(
                    static_cast<Eigen::Index>(std::floor(init_col - bounding_box_pix)), 0, signal.cols() - 1);
                search_col_hi = std::clamp<Eigen::Index>(
                    static_cast<Eigen::Index>(std::ceil(init_col + bounding_box_pix)), 0, signal.cols() - 1);
            }

            Eigen::Index best_row = -1;
            Eigen::Index best_col = -1;
            double best_snr = -std::numeric_limits<double>::infinity();
            bool found = find_weighted_peak(search_row_lo, search_row_hi, search_col_lo, search_col_hi,
                                            false, init_row, init_col, 0.0,
                                            best_row, best_col, best_snr);
            if (!found) {
                found = find_weighted_peak(0, signal.rows() - 1, 0, signal.cols() - 1,
                                           false, init_row, init_col, 0.0,
                                           best_row, best_col, best_snr);
            }
            if (!found) {
                Eigen::VectorXd p = Eigen::VectorXd::Zero(n_params);
                Eigen::VectorXd e = Eigen::VectorXd::Zero(n_params);
                logger->warn("fit_to_gaussian skipped: provided seed pixel is invalid and no weighted fallback was found");
                return std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>(p, e, false);
            }
            ir = best_row;
            ic = best_col;
            init_row = static_cast<double>(ir);
            init_col = static_cast<double>(ic);
        }

        init_flux = signal(ir, ic);
    }

    if constexpr (fit_mode == FitMode::beammap) {
        if (!std::isfinite(init_flux) || init_flux <= 0.0) {
            Eigen::VectorXd p = Eigen::VectorXd::Zero(n_params);
            Eigen::VectorXd e = Eigen::VectorXd::Zero(n_params);
            logger->warn("fit_to_gaussian skipped: beammap initial amplitude is non-positive ({})", init_flux);
            return std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>(p, e, false);
        }
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
    if (!std::isfinite(amp_lower) || !std::isfinite(amp_upper) ||
        std::abs(amp_upper - amp_lower) <= 1e-12) {
        const double amp_span = std::max(std::abs(init_flux), 1e-6);
        amp_lower = -amp_span;
        amp_upper = amp_span;
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

    std::vector<double> support_signal_values;
    std::vector<double> support_weight_values;
    if constexpr (fit_mode == FitMode::beammap) {
        support_signal_values.reserve(static_cast<std::size_t>(signal.size()));
        support_weight_values.reserve(static_cast<std::size_t>(weight.size()));
        for (Eigen::Index i = 0; i < signal.rows(); ++i) {
            for (Eigen::Index j = 0; j < signal.cols(); ++j) {
                const double s = signal(i, j);
                const double w = weight(i, j);
                if (std::isfinite(s) && std::isfinite(w) && w > 0.0) {
                    support_signal_values.push_back(s);
                    support_weight_values.push_back(w);
                }
            }
        }
    }

    // get map stddev
    double map_sigma = std::numeric_limits<double>::quiet_NaN();
    double support_weight_median = 1.0;
    if constexpr (fit_mode == FitMode::beammap) {
        if (support_signal_values.size() >= 2) {
            map_sigma = vector_stddev(support_signal_values);
        }
        if (!support_weight_values.empty()) {
            support_weight_median = vector_median_copy(support_weight_values);
            if (!std::isfinite(support_weight_median) ||
                support_weight_median <= std::numeric_limits<double>::epsilon()) {
                support_weight_median = 1.0;
            }
        }
    }
    else {
        map_sigma = engine_utils::calc_std_dev(signal);
    }
    if (!std::isfinite(map_sigma) || map_sigma <= 0.0) {
        map_sigma = 1.0;
    }

    // standard deviation of signal map
    Eigen::MatrixXd sigma(weight.rows(), weight.cols());

    // loop through pixels
    for (Eigen::Index i=0; i<weight.rows(); ++i) {
        for (Eigen::Index j=0; j<weight.cols(); ++j) {
            const double w = weight(i,j);
            if (std::isfinite(w) && w > 0.0) {
                // use support-scaled map sigma for beammaps
                if constexpr (fit_mode == FitMode::beammap) {
                    sigma(i,j) = map_sigma / std::sqrt(w / support_weight_median);
                }
                // use weights for pointing
                else if constexpr (fit_mode == FitMode::pointing) {
                    sigma(i,j) = 1./sqrt(w);
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

    if constexpr (fit_mode == FitMode::beammap) {
        if (beammap_fit_radius_fwhm > 0.0 && std::isfinite(beammap_fit_radius_fwhm) &&
            std::isfinite(init_fwhm) && init_fwhm > 0.0) {
            const double fit_radius_pix = beammap_fit_radius_fwhm * init_fwhm;
            const double fit_radius2 = fit_radius_pix * fit_radius_pix;
            for (Eigen::Index row = 0; row < n_rows; ++row) {
                const double map_row = static_cast<double>(lower_row + row);
                const double dr = map_row - init_row;
                for (Eigen::Index col = 0; col < n_cols; ++col) {
                    const double map_col = static_cast<double>(lower_col + col);
                    const double dc = map_col - init_col;
                    if (dr * dr + dc * dc > fit_radius2) {
                        _sigma(row, col) = 0.0;
                    }
                }
            }
        }
    }

    // avoid running Ceres on an unconstrained region.
    if ((_sigma.array() > 0.0).count() < n_params) {
        Eigen::VectorXd p = Eigen::VectorXd::Zero(n_params);
        Eigen::VectorXd e = Eigen::VectorXd::Zero(n_params);
        return std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>(p, e, false);
    }

    if (logger) {
        logger->debug("fit_to_gaussian pre-ceres: mode={} bbox=[{}:{},{}:{}] cutout={}x{} sigma_pos={} map_sigma={:.6g} support_weight_median={:.6g} init=[{:.6g}, {:.3f}, {:.3f}, {:.4g}, {:.4g}, {:.4g}] limits_amp=[{:.6g}, {:.6g}] limits_x=[{:.3f}, {:.3f}] limits_y=[{:.3f}, {:.3f}]",
                      fit_mode == FitMode::beammap ? "beammap" : "pointing",
                      lower_row, upper_row, lower_col, upper_col,
                      n_rows, n_cols, (_sigma.array() > 0.0).count(),
                      map_sigma, support_weight_median,
                      init_params(0), init_params(1), init_params(2),
                      init_params(3), init_params(4), init_params(5),
                      limits(0, 0), limits(0, 1),
                      limits(1, 0), limits(1, 1),
                      limits(2, 0), limits(2, 1));
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
    // Ceres covariance has shown heap corruption on some deployed systems, even
    // for a single pointing fit. Use the linearized uncertainty path instead.
    constexpr bool use_ceres_covariance = false;
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
