#pragma once

#include "../eigen.h"
#include "../formatter/matrix.h"
#include "../logging.h"
#include "../meta.h"
#include <Eigen/Core>
#include <ceres/ceres.h>

namespace alg {

namespace ceresfit {

using Eigen::Dynamic;
using Eigen::Index;

using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::SubsetParameterization;

/// @brief The mode of the evaluation
/// @var Residual
///     Evaluate residual.
/// @var Model
///     Evaluate model.
enum class EvalMode { Residual, Model };

/*
/// @class that wraps a fitter with model interface
template <typename Fitter>
struct AsModel {
    AsModel(Fitter* fitter_): fitter(fitter_) {}
    template<typename... Args>
    auto operator () (Args ... args) {
        return fitter->eval<EvalMode::Model>(FWD(args)...);
    }
private:
    Fitter* fitter;
};
*/

/// @brief Defines properties for parameter
template <typename Scalar = double>
struct ParamSetting {
    using OptScalar = std::optional<Scalar>;
    OptScalar value{std::nullopt};
    OptScalar fixed{std::nullopt};
    OptScalar lower_bound{std::nullopt};
    OptScalar upper_bound{std::nullopt};
    static auto getbounded(const OptScalar& lower, const OptScalar& upper) {
        ParamSetting p{};
        p.lower_bound = lower;
        p.upper_bound = upper;
        return p;
    }
    static auto getfixed(Scalar value) {
        ParamSetting p{};
        p.fixed = value;
        return p;
    }
};

template <typename Scalar = double>
using ParamSettings =
    std::unordered_map<std::string_view, ParamSetting<Scalar>>;

/// @brief Base class to use to define a model and fit with ceres.
/// @tparam _NP The number of mode parameters
/// @tparam _ND_IN The input dimension (number of independant variables) of the
/// model.
/// @tparam _ND_OUT The Output dimension of the model.
/// @tparam _Scalar The numeric type to use.
template <Index _NP = Dynamic, Index _ND_IN = Dynamic, Index _ND_OUT = Dynamic,
          typename _Scalar = double>
struct Fitter {
    constexpr static Index NP = _NP;
    constexpr static Index ND_IN = _ND_IN;
    constexpr static Index ND_OUT = _ND_OUT;
    using Scalar = _Scalar;
    /*
    using InputType = Eigen::Matrix<Scalar, NP, 1>;
    using InputDataType = Eigen::Matrix<Scalar, Dynamic, ND>;
    using InputDataBasisType = Eigen::Matrix<Scalar, Dynamic, 1>;
    using ValueType = Eigen::Matrix<Scalar, Dynamic, 1>;
    */

    using InputType = Eigen::Map<const Eigen::Matrix<Scalar, NP, 1>>;
    Fitter() = default;

    /// @brief Create autodiff cost function by providing number of residuals
    template <typename Fitter_>
    static auto set_autodiff_residual(Problem *problem, Scalar *paramblock,
                                      Fitter_ *fitter) {
        // setup residual block
        CostFunction *cost_function =
            new AutoDiffCostFunction<Fitter_, Dynamic, NP>(fitter, fitter->ny);
        // problem->AddResidualBlock(cost_function, nullptr, paramblock);
        problem->AddResidualBlock(cost_function, new CauchyLoss(2.),
                                   paramblock);
    }

    /// @brief Create ceres problem by providing parameters
    template <typename Derived>
    static auto make_problem(const Eigen::DenseBase<Derived> &params_,
                             const std::vector<ParamSetting<Scalar>> &settings) {
        auto &params =
            const_cast<Eigen::DenseBase<Derived> &>(params_).derived();
        if constexpr (eigen_utils::is_plain_v<Derived>) {
            if (auto size = params.size(); size != NP) {
                params.derived().resize(NP);
                SPDLOG_TRACE("resize params size {} to {}", size,
                             params.size());
            }
        }
        // makesure params.data() is continugous up to NP
        if (params.innerStride() != 1 || params.innerSize() < NP) {
            throw std::runtime_error(
                fmt::format("fitter requires params data of size {}"
                            " in continugous memory",
                            NP));
        }

        // default settings make use of params values
        //         if (settings.size() == 0) {
        //             settings.resize(NP);
        //             for (std::size_t i = 0; i < NP; ++i) {
        //                 settings[i].value = params.coeff(i);
        //             }
        //         }
        // ensure NP param settings if user supplied
        if (auto size = settings.size(); size != NP) {
            throw std::runtime_error(fmt::format(
                "param setting size {} mismatch params size {}", size, NP));
        }

        // setup problem
        auto problem = std::make_shared<Problem>();
        problem->AddParameterBlock(params.data(), NP);
        std::vector<int> fixed_params;
        // setup params
        for (Index i = 0; i < NP; ++i) {
            auto p = settings[i];
            if (p.lower_bound.has_value()) {
                problem->SetParameterLowerBound(params.data(), i,
                                                p.lower_bound.value());
            }
            if (p.upper_bound.has_value()) {
                problem->SetParameterUpperBound(params.data(), i,
                                                p.upper_bound.value());
            }
            if (p.fixed.has_value()) {
                params.coeffRef(i) = p.fixed.value();
                fixed_params.push_back(i);
            }
            if (p.value.has_value()) {
                params.coeffRef(i) = p.value.value();
            }
        }
        if (fixed_params.size() > 0) {
            SubsetParameterization *sp =
                new SubsetParameterization(NP, fixed_params);
            problem->SetParameterization(params.data(), sp);
        }
        SPDLOG_TRACE("params init values: {}", params);
        SPDLOG_TRACE("fixed params: {}", fixed_params);
        return std::make_pair(problem, params.data());
    }
};

template <typename Fitter_, typename DerivedA, typename DerivedB,
          typename DerivedC, typename DerivedD>
auto fit(const Eigen::DenseBase<DerivedA> &xdata_,
         const Eigen::DenseBase<DerivedB> &ydata_,
         const Eigen::DenseBase<DerivedC> &yerr_,
         Eigen::DenseBase<DerivedD> const &params_,
         const ParamSettings<typename Fitter_::Scalar> &param_settings = {}) {
    Eigen::DenseBase<DerivedD> &params =
        const_cast<Eigen::DenseBase<DerivedD> &>(params_);
    auto &xdata = xdata_.derived();
    auto &ydata = ydata_.derived();
    auto &yerr = yerr_.derived();
    // make sure the data are continuous
    if (!(eigen_utils::is_contiguous(xdata) &&
          eigen_utils::is_contiguous(ydata) &&
          eigen_utils::is_contiguous(yerr) &&
          eigen_utils::is_contiguous(params))) {
        throw std::runtime_error("fit does not work with non-continugous data");
    }
    // construct Fitter
    auto fitter = new Fitter_();
    // create settings
    std::vector<ParamSetting<typename Fitter_::Scalar>> settings(Fitter_::NP);
        for (std::size_t i = 0; i < Fitter_::NP; ++i) {
            auto name = Fitter_::param_names[i];
            // SPDLOG_TRACE("check settings for param {} {}", i, name);
            if (auto it = param_settings.find(name);
                it != param_settings.end()) {
                // SPDLOG_TRACE("found runtime settings for param {}", name);
                settings[i] = it->second;
            } else if (auto it = Fitter_::param_settings.find(name);
                       it != Fitter_::param_settings.end()) {
                // SPDLOG_TRACE("found static settings for param {}", name);
                settings[i] = it->second;
            }
            // SPDLOG_TRACE("param settings: name={} value={} fixed={} "
//                          "lower_bound={} upper_bound={}",
//                          name, settings[i].value, settings[i].fixed,
//                          settings[i].lower_bound, settings[i].upper_bound);
        }
    // create problem
    auto [problem, paramblock] =
        Fitter_::make_problem(params.derived(), settings);
    // setup inputs
    SPDLOG_TRACE("input xdata{}", xdata);
    SPDLOG_TRACE("input ydata{}", ydata);
    SPDLOG_TRACE("input yerr{}", yerr);
    // find out scalar size ratio
    constexpr auto xscalar_ratio =
        sizeof(typename std::decay_t<decltype(xdata)>::Scalar) /
        sizeof(typename Fitter_::Scalar);
    constexpr auto yscalar_ratio =
        sizeof(typename std::decay_t<decltype(ydata)>::Scalar) /
        sizeof(typename Fitter_::Scalar);
    SPDLOG_TRACE("data cast size ratio: x {}, y {}", xscalar_ratio,
                 yscalar_ratio);

    fitter->nx = xdata.size() * xscalar_ratio;
    fitter->ny = ydata.size() * yscalar_ratio;

    fitter->xdata =
        reinterpret_cast<const typename Fitter_::Scalar *>(xdata.data());
    fitter->ydata =
        reinterpret_cast<const typename Fitter_::Scalar *>(ydata.data());
    fitter->yerr =
        reinterpret_cast<const typename Fitter_::Scalar *>(yerr.data());

    // setup residuals
    Fitter_::set_autodiff_residual(problem.get(), paramblock, fitter);

    SPDLOG_TRACE("initial params {}",
                 fmt_utils::pprint(paramblock, Fitter_::NP));
    SPDLOG_TRACE("xdata{}", fmt_utils::pprint(fitter->xdata, fitter->nx));
    SPDLOG_TRACE("ydata{}", fmt_utils::pprint(fitter->ydata, fitter->ny));
    SPDLOG_TRACE("yerr{}", fmt_utils::pprint(fitter->yerr, fitter->ny));

    // do the fit
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.use_inner_iterations = true;
    // options.minimizer_progress_to_stdout = true;
    options.logging_type = ceres::SILENT;
    Solver::Summary summary;
    ceres::Solve(options, problem.get(), &summary);

    SPDLOG_TRACE("{}", summary.BriefReport());
    SPDLOG_TRACE("fitted paramblock {}",
                 fmt_utils::pprint(paramblock, Fitter_::NP));
    SPDLOG_TRACE("fitted params {}", params.derived());
    return std::make_tuple(summary.termination_type == ceres::CONVERGENCE, std::move(summary));
}

template <typename Fitter_, typename DerivedA, typename DerivedB,
          typename DerivedC>
void eval(const Eigen::DenseBase<DerivedA> &xdata_,
          const Eigen::DenseBase<DerivedB> &params_,
          Eigen::DenseBase<DerivedC> const &ydata_) {
    auto &xdata = xdata_.derived();
    auto &params = params_.derived();
    auto &ydata = const_cast<Eigen::DenseBase<DerivedC> &>(ydata_).derived();
    // make sure the data are continuous
    if (!(eigen_utils::is_contiguous(xdata) &&
          eigen_utils::is_contiguous(params))) {
        throw std::runtime_error("fit does not work with non-continugous data");
    }
    SPDLOG_TRACE("eval model wiht params {}", params);
    // construct Fitter
    auto fitter = Fitter_();
    // find out scalar size ratio
    constexpr auto xscalar_ratio =
        sizeof(typename std::decay_t<decltype(xdata)>::Scalar) /
        sizeof(typename Fitter_::Scalar);

    fitter.nx = xdata.size() * xscalar_ratio;

    fitter.xdata =
        reinterpret_cast<const typename Fitter_::Scalar *>(xdata.data());
    fitter.eval(params.data(),
                reinterpret_cast<typename Fitter_::Scalar *>(ydata.data()));
}

} // namespace ceresfit
} // namespace alg
