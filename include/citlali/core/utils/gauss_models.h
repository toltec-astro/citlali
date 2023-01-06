#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include <citlali/core/utils/constants.h>

namespace engine_utils {

template <typename Model>
typename std::enable_if<Model::DimensionsAtCompileTime == 2, Model>::type
create_model(const Eigen::VectorXd& params) {
    Model g(params);
    return g;
}

// a general functor, assuming X inputs and Y outputs
template <typename _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
struct DenseFunctor {
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };

    using Scalar = _Scalar;
    using InputType = Eigen::Matrix<Scalar,InputsAtCompileTime, 1>;
    using ValueType = Eigen::Matrix<Scalar,ValuesAtCompileTime, 1>;

    constexpr static std::string_view name = "functor";
    DenseFunctor(int inputs, int values) : m_inputs(inputs), m_values(values) {
    }
    // default
    DenseFunctor(): DenseFunctor(InputsAtCompileTime, ValuesAtCompileTime) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }

    using Self = DenseFunctor<_Scalar, NX, NY>;

    template<typename OStream>
    friend OStream &operator<<(OStream &os, const Self& f) {
        return os << f.name << "<" << static_cast<int>(Self::InputsAtCompileTime) << ", " << static_cast<int>(Self::ValuesAtCompileTime) << ">(" << f.inputs() << ", " << f.values() << ")";
    }

protected:
    int m_inputs = InputsAtCompileTime;
    int m_values = ValuesAtCompileTime;
};


// Model is a functor working on double (do we need model of other types?)
// NP -- number of input parameters
// ND -- number of demensions of input data
template <int NP=Eigen::Dynamic, int ND=Eigen::Dynamic>
struct Model: DenseFunctor<double, NP, Eigen::Dynamic> {
    enum {
        DimensionsAtCompileTime = ND
    };
    using _Base = DenseFunctor<double, NP, Eigen::Dynamic>;
    using DataType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using InputDataType = Eigen::Matrix<double, Eigen::Dynamic, ND>;
    using InputDataBasisType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    constexpr static std::string_view name = "model";
    // via known size of params
    Model(int inputs): _Base(inputs, Eigen::Dynamic), params(inputs) {
    }
    // via copy of params
    Model(const typename _Base::InputType& params): Model(static_cast<int>(params.size())) {this->params=params;}
    // via initializer list of params
    Model(std::initializer_list<double> params): Model(static_cast<int>(params.size())) {
        int i = 0;
        for (auto& p: params) {
            this->params(i) = p;
            ++i;
        }
    }
    // default
    Model(): Model(Model::InputsAtCompileTime) {}

    typename Model::InputType params;

    // eval()
    // should be defined to take (ndata, ndim) mesh and return vector of (ndata, 1)

    // meshgrid()
    // maybe be defined to covert input of (ndata_dim1, ... ndata_dimn) to (ndata, ndim) mesh

    // operator()
    // cound be defined to perform eval in nd shapes

    // using DataType = ValueType;
    // cound be defined to make it semantically clear for what data type it works with

    template <typename T=InputDataType>
    typename std::enable_if<ND == 2, T>::type meshgrid(const InputDataBasisType& x, const InputDataBasisType& y) const {
        // column major
        // [x0, y0,] [x1, y0] [x2, y0] ... [xn, y0] [x0, y1] ... [xn, yn]
        const long nx = x.size(), ny = y.size();
        InputDataType xy(nx * ny, 2);
        // map xx [ny, nx] to the first column of xy
        Eigen::Map<Eigen::MatrixXd> xx(xy.data(), ny, nx);
        // map yy [ny, nx] to the second column of xy
        Eigen::Map<Eigen::MatrixXd> yy(xy.data() + xy.rows(), ny, nx);
        // populate xx such that each row is x
        for (Eigen::Index i = 0; i < ny; ++i) {
            xx.row(i) = x.transpose();
        }
        // populate yy such that each col is y
        for (Eigen::Index j = 0; j < nx; ++j) {
            yy.col(j) = y;
        }
        return xy;
    }

    typename Model::InputType transform(const typename Model::InputType& p) const {
        return p;
    }
    typename Model::InputType inverseTransform(const typename Model::InputType& p) const {
        return p;
    }

    struct Parameter {
        std::string name = "unnammed";
        bool fixed = false;
        bool bounded = false;
        double lower = - std::numeric_limits<double>::infinity();
        double upper = std::numeric_limits<double>::infinity();
    };
};

// 3 params, 1 dim
struct Gaussian1D: Model<3, 1> {
    constexpr static std::string_view name = "gaussian1d";
    using Model<3, 1>::Model; // model constructors
    Gaussian1D(double amplitude=1., double mean=0., double stddev=1.);

    ValueType eval(const InputType& p, const InputDataType& x) const;
    // no meshgrid needed here

    // convinient methods
    ValueType operator() (const InputType& p, const InputDataType& x) const;
    ValueType operator() (const InputDataType& x) const;
    std::vector<Parameter> param_settings{
        {"amplitude"},
        {"mean"},
        {"stddev"},
    };
};

struct Gaussian2D: Model<6, 2> {
    constexpr static std::string_view name = "gaussian2d";
    using Model<6, 2>::Model;
    Gaussian2D(double amplitude=1., double xmean=0., double ymean=0., double xstddev=1., double ystddev=1., double theta=0.);
    ~Gaussian2D(){}

    // operates on meshgrid xy of shape (ny * nx, 2), return a flat vector
    ValueType eval(const InputType& p, const InputDataType& xy) const;

    // convinient methods
    // operates on x and y coords separately. return a (ny, nx) Eigen::Matrix
    DataType operator() (
            const InputType& p,
            const InputDataBasisType& x,
            const InputDataBasisType& y) const;
    DataType operator() (
            const InputDataBasisType& x,
            const InputDataBasisType& y) const;

    InputType transform(const InputType& p) const;
    InputType inverseTransform(const InputType& p) const;
    std::vector<Parameter> param_settings {
        {"amplitude"},
        {"xmean"},
        {"ymean"},
        {"xstddev"},
        {"ystddev"},
        {"theta"},
    };
};

struct SymmetricGaussian2D: Model<4, 2> {
    constexpr static std::string_view name = "symmetricgaussian2d";
    using Model<4, 2>::Model; // model constructors;
    SymmetricGaussian2D(double amplitude=1., double xmean=0., double ymean=0., double stddev=1.);
    ~SymmetricGaussian2D(){}

    // operates on meshgrid xy of shape (ny * nx, 2), return a flat vector
    ValueType eval(const InputType& p, const InputDataType& xy) const;

    // convinient methods
    // operates on x and y coords separately. return a (ny, nx) Eigen::Matrix
    DataType operator() (
            const InputType& p,
            const InputDataBasisType& x,
            const InputDataBasisType& y) const;
    DataType operator() (
            const InputDataBasisType& x,
            const InputDataBasisType& y) const;
    std::vector<Parameter> param_settings{
        {"amplitude"},
        {"xmean"},
        {"ymean"},
        {"stddev"},
    };
};

// Fitter is a functor that matches the data types of the Model.
// Fitter relies on the eval() method
template <typename _Model>
struct Fitter: _Model::_Base {
    using _Base = typename _Model::_Base;
    using Model = _Model;

    Fitter (const Model* model, int values): _Base(model->inputs(), values), m_model(model) {
    }
    Fitter (const Model* model): Fitter(model, Fitter::InputsAtCompileTime) {}

    const Model* model() const {return m_model;}

    const Eigen::Map<const typename Model::InputDataType>* xdata = nullptr;  // set via meshgrid
    const Eigen::Map<const typename Fitter::ValueType>* ydata = nullptr;
    const Eigen::Map<const typename Fitter::ValueType>* sigma = nullptr;

    //int operator()(const InputType &x, ValueType& fvec) { }
    // should be defined in derived classes

private:
    const Model* m_model = nullptr;
};

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using ceres::Covariance;

// CeresAutoDiff Fitter provides concrete method for least-square minimization using ceres
template <typename Model>
struct CeresAutoDiffFitter: Fitter<Model> {
    using _Base = Fitter<Model>;
    // the base constructors
    using _Base::_Base;

    template <typename T>
    bool operator()(const T* const p, T* r) const {
        auto cost2 = cos(p[5]) * cos(p[5]);
        auto sint2 = sin(p[5]) * sin(p[5]);
        auto sin2t = sin(2. * p[5]);
        auto xstd2 = p[3] * p[3];
        auto ystd2 = p[4] * p[4];
        auto a = - 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
        auto b = - 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
        auto c = - 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));

        for (int i=0; i < this->values(); ++i){
            if (this->sigma->coeffRef(i) != 0) {
                r[i] =  (
                        this->ydata->coeffRef(i) -
                        p[0] * exp(
                            pow(this->xdata->coeffRef(i, 0) - p[1], 2) * a +
                            (this->xdata->coeffRef(i, 0) - p[1]) * (this->xdata->coeffRef(i, 1) - p[2]) * b +
                            pow(this->xdata->coeffRef(i, 1) - p[2], 2) * c
                            )
                    ) / this->sigma->coeffRef(i);
            }
            // remove zero weights
            else {
                r[i] =  (
                        this->ydata->coeffRef(i) -
                        p[0] * exp(
                            pow(this->xdata->coeffRef(i, 0) - p[1], 2) * a +
                            (this->xdata->coeffRef(i, 0) - p[1]) * (this->xdata->coeffRef(i, 1) - p[2]) * b +
                            pow(this->xdata->coeffRef(i, 1) - p[2], 2) * c
                            )
                        ) * this->sigma->coeffRef(i);
            }
        }
        return true;
    }

    std::shared_ptr<Problem> createProblem(double* params) {
        std::shared_ptr<Problem> problem = std::make_shared<Problem>();
        problem->AddParameterBlock(params, this->model()->params.size());
        return problem;
    }
};

} // namespace engine_utils
