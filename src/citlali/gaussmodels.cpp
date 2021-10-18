#include "core/gaussfit.h"
#include <iostream>

namespace gaussfit {

Gaussian1D::Gaussian1D(double amplitude, double mean, double stddev)
    : Model<3, 1>({amplitude, mean, stddev}) {}

Gaussian1D::ValueType Gaussian1D::eval(const Gaussian1D::InputType& p, const Gaussian1D::InputDataType& x) const
{
    SPDLOG_TRACE("eval {} with params{} on input{}", *this, p, x);
    return p[0] * (-0.5 * (x.array() - p[1]).square() / p[2] / p[2]).exp();
}

Gaussian1D::ValueType Gaussian1D::operator() (const Gaussian1D::InputType& p, const Gaussian1D::InputDataType& x) const
{
    return eval(p, x);
}

Gaussian1D::ValueType Gaussian1D::operator() (const Gaussian1D::InputDataType& x) const
{
    return operator() (this->params, x);
}


Gaussian2D::Gaussian2D(double amplitude, double xmean, double ymean, double xstddev, double ystddev, double theta)
    //             0          1      2      3        4        5
    : Model<6, 2>({amplitude, xmean, ymean, xstddev, ystddev, theta}) {}

Gaussian2D::ValueType Gaussian2D::eval(const Gaussian2D::InputType& p, const Gaussian2D::InputDataType& xy) const
{
    SPDLOG_TRACE("eval {} with params{} on input{}", *this, p, xy);
    double cost2 = cos(p[5]) * cos(p[5]);
    double sint2 = sin(p[5]) * sin(p[5]);
    double sin2t = sin(2. * p[5]);
    double xstd2 = p[3] * p[3];
    double ystd2 = p[4] * p[4];
    double a = - 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
    double b = - 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
    double c = - 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));
    return p[0] * (
                    (xy.col(0).array() - p[1]).square() * a +
                    (xy.col(0).array() - p[1]) * (xy.col(1).array() - p[2]) * b +
                    (xy.col(1).array() - p[2]).square() * c
                ).exp();
}

Gaussian2D::DataType Gaussian2D::operator() (
        const Gaussian2D::InputType& p,
        const Gaussian2D::InputDataBasisType& x,
        const Gaussian2D::InputDataBasisType& y) const
{
    return eval(p, meshgrid(x, y));
}

Gaussian2D::DataType Gaussian2D::operator() (
        const Gaussian2D::InputDataBasisType& x,
        const Gaussian2D::InputDataBasisType& y) const
{
    return operator() (this->params, x, y);
}

Gaussian2D::InputType Gaussian2D::transform (
        const Gaussian2D::InputType& p) const
{
    // fix degeneracy issue by constrain the position angle to [0, pi/2]
    // pp[5] is [-inf, inf]
    Gaussian2D::InputType pp(p);
    pp[5] = tan(p[5] * 2. - PI / 2.);
    return pp;
}

Gaussian2D::InputType Gaussian2D::inverseTransform (
        const Gaussian2D::InputType& pp) const
{
    // p[5] is [0, pi/2]
    Gaussian2D::InputType p(pp);
    p[5] = (atan(pp[5]) + PI / 2.) / 2.;
    return p;
}

SymmetricGaussian2D::SymmetricGaussian2D(double amplitude, double xmean, double ymean, double stddev)
    //             0          1      2      3
    : Model<4, 2>({amplitude, xmean, ymean, stddev}) {}

SymmetricGaussian2D::ValueType SymmetricGaussian2D::eval(const SymmetricGaussian2D::InputType& p, const SymmetricGaussian2D::InputDataType& xy) const
{
    SPDLOG_TRACE("eval {} with params{} on data{}", *this, p, xy);
    double cost2 = cos(0) * cos(0);
    double sint2 = sin(0) * sin(0);
    double sin2t = sin(2. * 0);
    double xstd2 = p[3] * p[3];
    double ystd2 = p[3] * p[3];
    double a = - 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
    double b = - 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
    double c = - 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));
    return p[0] * (
                    (xy.col(0).array() - p[1]).square() * a +
                    (xy.col(0).array() - p[1]) * (xy.col(1).array() - p[2]) * b +
                    (xy.col(1).array() - p[2]).square() * c
                ).exp();
}

SymmetricGaussian2D::DataType SymmetricGaussian2D::operator() (
        const SymmetricGaussian2D::InputType& p,
        const SymmetricGaussian2D::InputDataBasisType& x,
        const SymmetricGaussian2D::InputDataBasisType& y) const
{
    return eval(p, meshgrid(x, y));
}

SymmetricGaussian2D::DataType SymmetricGaussian2D::operator() (
        const SymmetricGaussian2D::InputDataBasisType& x,
        const SymmetricGaussian2D::InputDataBasisType& y) const
{
    return operator() (this->params, x, y);
}

}  // namespace
