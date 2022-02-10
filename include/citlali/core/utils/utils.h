#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/Splines>
#include <vector>
#include <numeric>
#include <complex>

#include <tula/grppi.h>

#include <citlali/core/utils/constants.h>

namespace engine_utils {

using RowMatrixXd = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using RowMatrixXcd = Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

template<typename Derived>
auto stddev(Eigen::DenseBase<Derived> &vec) {
    double norm;
    if (vec.derived().size() == 1) {
        norm = 1;
    }
    else {
        norm = vec.derived().size() - 1;
    }
    double tmp = std::sqrt(
        (vec.derived().array() - vec.derived().mean()).square().sum()
        / norm);
    return tmp;
}

template <typename DerivedA, typename DerivedB>
auto stddev(Eigen::DenseBase<DerivedA> &scans, Eigen::DenseBase<DerivedB> &flags) {

    // count up number of unflagged samples
    double ngood = (flags.derived().array() == 1).count();

    double N;
    if (ngood == 1) {
        N = ngood;
    }
    else {
        N = ngood - 1;
    }

    // calculate the standard deviation
    double tmp = std::sqrt(((scans.derived().array() *flags.derived().template cast<double>().array()) -
                            (scans.derived().array() * flags.derived().template cast<double>().array()).sum()/
                                ngood).square().sum() / N);

    /*auto x = scans.derived().array() *flags.derived().template cast<double>().array();
    auto mean = (scans.derived().array() * flags.derived().template cast<double>().array()).sum()/ngood;
*/
    // calculate the standard deviation
    //double tmp = std::sqrt((x - mean).square().sum() / N);

    // return stddev and the number of unflagged samples
    return std::tuple<double, double>(tmp, ngood);
}

// box car smooth function
template<typename DerivedA, typename DerivedB>
void smooth(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &out, int w) {
    // ensure box-car width is odd
    if (w % 2 == 0) {
        w++;
    }

    Eigen::Index npts = in.size();

    out.head((w - 1) / 2) = in.head((w - 1) / 2);
    out.tail(npts - (w + 1) / 2. + 1) = in.tail(npts - (w + 1) / 2. + 1);

    double winv = 1. / w;
    int wm1d2 = (w - 1) / 2.;
    int wp1d2 = (w + 1) / 2.;

    for (int i = wm1d2; i <= npts - wp1d2; i++) {
        out(i) = winv * in.segment(i - wm1d2, w).sum();
    }
}

// iterate over tuple (https://stackoverflow.com/questions/26902633/how-to-iterate-over-a-stdtuple-in-c-11)
template<class F, class...Ts, std::size_t...Is>
void for_each_in_tuple(const std::tuple<Ts...> & tuple, F func, std::index_sequence<Is...>) {
    using expander = int[];
    (void)expander { 0, ((void)func(std::get<Is>(tuple)), 0)... };
}

template<class F, class...Ts>
void for_each_in_tuple(const std::tuple<Ts...> & tuple, F func){
    for_each_in_tuple(tuple, func, std::make_index_sequence<sizeof...(Ts)>());
}

template<typename Derived>
std::vector<std::tuple<double, int>> sorter(Eigen::DenseBase<Derived> &vec){
    std::vector<std::tuple<double, int>> vis;
    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(vec.size(),0,vec.size()-1);

    for(Eigen::Index i=0; i<vec.size(); i++){
        std::tuple<double, double> vec_and_val(vec(i), indices(i));
        vis.push_back(vec_and_val);
    }

    std::sort(vis.begin(), vis.end(),
              [&](const std::tuple<double, int> &a, const std::tuple<double, int> &b) -> bool{
                  return std::get<0>(a) < std::get<0>(b);
              });

    return vis;
}

template<typename Derived>
void shift_vector(Eigen::DenseBase<Derived> &in, const int n){
    int nx = in.size();
    Eigen::VectorXd vec2(nx);
    for(int i=0;i<nx;i++){
        int ti = (i+n)%nx;
        int shifti = (ti < 0) ? nx+ti : ti;
        vec2[shifti] = in(i);
    }
    for(int i=0;i<nx;i++) in(i) = vec2(i);
}

template <typename Derived>
auto shift_matrix(Eigen::DenseBase<Derived> &in, const int n1, const int n2) {
    Eigen::Index nr = in.rows();
    Eigen::Index nc = in.cols();

    Eigen::MatrixXd out;
    out.setZero(nr, nc);

    for (Eigen::Index i=0; i<nc; i++) {
        for (Eigen::Index j=0; j<nr; j++){
            Eigen::Index ti = (i+n2) % nc;
            Eigen::Index tj = (j+n1) % nr;
            Eigen::Index shifti = (ti < 0) ? nc+ti : ti;
            Eigen::Index shiftj = (tj < 0) ? nr+tj : tj;
            out(shiftj,shifti) = in(j,i);
        }
    }
    return out;
}

enum FFTdirection {
    forward = 0,
    backward = 1
};

template<FFTdirection direc, typename Derived>
Eigen::MatrixXcd fft2w(Eigen::DenseBase<Derived> &in, const Eigen::Index nrows, const Eigen::Index ncols){

    // placeholder vectors for grppi maps
    std::vector<int> ri(nrows);
    std::iota(ri.begin(), ri.end(), 0);
    std::vector<int> ro(nrows);

    std::vector<int> ci(ncols);
    std::iota(ci.begin(), ci.end(), 0);
    std::vector<int> co(ncols);

    Eigen::MatrixXcd out(nrows, ncols);

    // do the fft over the rows
    grppi::map(tula::grppi_utils::dyn_ex("omp"),ri,ro,[&](auto k){
        Eigen::FFT<double> fft;
        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        fft.SetFlag(Eigen::FFT<double>::Unscaled);

        Eigen::VectorXcd tmp_out(ncols);
        if constexpr(direc == forward) {
            fft.fwd(tmp_out, in.row(k));
        }
        else if constexpr(direc == backward){
            fft.inv(tmp_out, in.row(k));
        }
        out.row(k) = tmp_out;
        return 0;
    });

    // do the fft over the cols
    grppi::map(tula::grppi_utils::dyn_ex("omp"),ci,co,[&](auto k){
        Eigen::FFT<double> fft;
        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        fft.SetFlag(Eigen::FFT<double>::Unscaled);

        Eigen::VectorXcd tmp_out(nrows);
        if constexpr(direc == forward) {
            fft.fwd(tmp_out, out.col(k));
        }
        else if constexpr(direc == backward){
            fft.inv(tmp_out, out.col(k));
        }
        out.col(k) = tmp_out;
        return 0;
    });

    /*Eigen::FFT<double> fft;
    fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
    fft.SetFlag(Eigen::FFT<double>::Unscaled);

    for (Eigen::Index k=0; k<nrows; ++k) {
        Eigen::VectorXcd tmp_out(ncols);
        if constexpr(direc == forward){
            fft.fwd(tmp_out, in.row(k));
        }
        else if constexpr(direc == backward){
            fft.inv(tmp_out, in.row(k));
        }
        out.row(k) = tmp_out;
    }

    for (Eigen::Index k=0; k<ncols; ++k) {
        Eigen::VectorXcd tmp_out(nrows);
        if constexpr(direc == forward){
            fft.fwd(tmp_out, out.col(k));
        }
        else if constexpr(direc == backward){
            fft.inv(tmp_out, out.col(k));
        }
        out.col(k) = tmp_out;
    }*/

    return std::move(out);
}

template <typename Derived>
std::tuple<Eigen::VectorXd, Eigen::VectorXd> set_coverage_cut_ranges(Eigen::DenseBase<Derived> &in, const double weight_cut) {
    Eigen::VectorXd cut_row_range(2);
    Eigen::VectorXd cut_col_range(2);

    cut_row_range(0) = 0;
    cut_row_range(1) = in.rows() - 1;
    cut_col_range(0) = 0;
    cut_col_range(1) = in.cols() - 1;

    // find lower row bound
    bool flag = false;
    for (int i = 0; i < in.rows(); i++) {
        for (int j = 0; j < in.cols(); j++) {
            if (in(i,j) >= weight_cut) {
                cut_row_range(0) = i;
                flag = true;
                break;
            }
        }
        if (flag == true) {
            break;
        }
    }

    // find upper row bound
    flag = false;
    for (int i = in.rows() - 1; i > -1; i--) {
        for (int j = 0; j < in.cols(); j++) {
            if (in(i,j) >= weight_cut) {
                cut_row_range(1) = i;
                flag = true;
                break;
            }
        }
        if (flag == true) {
            break;
        }
    }

    // find lower column bound
    flag = false;
    for (int i = 0; i < in.cols(); i++) {
        for (int j = cut_row_range(0); j < cut_row_range(1) + 1; j++) {
            if (in(j,i) >= weight_cut) {
                cut_col_range(0) = i;
                flag = true;
                break;
            }
        }
        if (flag == true) {
            break;
        }
    }

    // find upper column bound
    flag = false;
    for (int i = in.cols() - 1; i > -1; i--) {
        for (int j = cut_row_range(0); j < cut_row_range(1) + 1; j++) {
            if (in(j,i) >= weight_cut) {
                cut_col_range(1) = i;
                flag = true;
                break;
            }
        }
        if (flag == true) {
            break;
        }
    }

    return std::make_tuple(cut_row_range,cut_col_range);
}

double selector(std::vector<double> input, int index){
    unsigned int pivot_index = rand() % input.size();
    double pivot_value = input[pivot_index];
    std::vector<double> left;
    std::vector<double> right;
    for (unsigned int x = 0; x < input.size(); x++) {
        if (x != pivot_index) {
            if (input[x] > pivot_value) {
                right.push_back(input[x]);
            }
            else {
                left.push_back(input[x]);
            }
        }
    }
    if ((int)left.size() == index) {
        return pivot_value;
    }
    else if ((int)left.size() < index) {
        return selector(right, index - left.size() - 1);
    }
    else {
        return selector(left, index);
    }
}

template <typename Derived>
double find_weight_threshold(Eigen::DenseBase<Derived> &in, const double cov) {
    int nr = in.rows();
    int nc = in.cols();
    std::vector<double> og;

    for (Eigen::Index x = 0; x < nc; x++) {
        for (Eigen::Index y = 0; y < nr; y++) {
            if (in(y,x) > 0.){
                og.push_back(in(y,x));
            }
        }
    }

    // find the point where 25% of nonzero elements have greater weight
    double covlim;
    int covlimi;
    covlimi = 0.75*og.size();

    covlim = selector(og, covlimi);

    double mval, mvali;
    mvali = floor((covlimi+og.size())/2.);

    mval = selector(og, mvali);

    // return the weight cut value
    return cov*mval;
}

auto hanning(int n1in, int n2in) {
  double n1 = (double) n1in;
  double n2 = (double) n2in;
  double a = 2.*pi/n1;

  Eigen::VectorXd index(n1in);
  for(int i=0;i<n1in;i++) index(i) = (double) i;

  double b = 2.*pi/n2;

  Eigen::VectorXd row(n1in);
  for(int i=0;i<n1in;i++) row(i) = -0.5 * cos(index(i)*a) + 0.5;
  index.resize(n2in);

  for(int i=0;i<n2in;i++) index(i) = (double) i;

  Eigen::VectorXd col(n2in);
  for(int i=0;i<n2in;i++) col(i) = -0.5 * cos(index(i)*b) + 0.5;

  Eigen::MatrixXd han(n1in,n2in);
  for(int i=0;i<n2in;i++)
    for(int j=0;j<n1in;j++)
      han(j,i) = row(j)*col(i);

  return han;
}

class SplineFunction {
public:
    double x_min;
    double x_max;
    SplineFunction(Eigen::VectorXd const &x_vec,
                 Eigen::VectorXd const &y_vec)
    : x_min(x_vec.minCoeff()),
      x_max(x_vec.maxCoeff()),
      // Spline fitting here. X values are scaled down to [0, 1] for this.
      spline_(Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(
                y_vec.transpose(),
                 // No more than cubic spline, but accept short vectors.

                std::min<int>(x_vec.rows() - 1, 3),
                scaled_values(x_vec)))
  { }

  double operator()(double x) const {
    // x values need to be scaled down in extraction as well.
    return spline_(scaled_value(x))(0);
  }

private:
  // Helpers to scale X values down to [0, 1]
  double scaled_value(double x) const {
    return (x - x_min) / (x_max - x_min);
  }

  Eigen::RowVectorXd scaled_values(Eigen::VectorXd const &x_vec) const {
    return x_vec.unaryExpr([this](double x) { return scaled_value(x); }).transpose();
  }

  // Spline of one-dimensional "points."
  Eigen::Spline<double, 1> spline_;
};

} // namespace
