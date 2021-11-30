#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/FFT>
#include <vector>
#include <numeric>
#include <complex>

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
auto stddev(Eigen::DenseBase<DerivedA> &scans,
            Eigen::DenseBase<DerivedB> &flags) {

    Eigen::Index ngood = (flags.derived().array() == 1).count();

    double norm;
    if (ngood == 1) {
        norm = 1;
    }
    else {
        norm = scans.derived().size() - 1;
    }

    //Calculate the standard deviation
    double tmp = std::sqrt(((scans.derived().array() *flags.derived().template cast<double>().array()) -
                            (scans.derived().array() * flags.derived().template cast<double>().array()).sum()/
                                ngood).square().sum() / norm);

    // Return stddev and the number of unflagged samples
    return std::tuple<double, double>(tmp, ngood);
}

// box car smooth function
template<typename DerivedA, typename DerivedB>
void smooth(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &out, int w) {
    //Ensure box-car width is odd
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

    for(int i=0; i<vec.size(); i++){
        std::tuple<double, double> vec_and_val(vec[i], indices[i]);
        vis.push_back(vec_and_val);
    }

    std::sort(vis.begin(), vis.end(),
              [&](const std::tuple<double, int>& a, const std::tuple<double, int>& b) -> bool{
                  return std::get<0>(a) < std::get<0>(b);
              });

    return vis;
}

template <typename Derived>
auto shift_matrix(Eigen::DenseBase<Derived> &in, const int n1, const int n2) {
    Eigen::Index nx = in.rows();
    Eigen::Index ny = in.cols();

    Eigen::MatrixXd out;
    out.setZero(nx, ny);

    for (Eigen::Index i=0; i<nx; i++) {
        for (Eigen::Index j=0; j<ny; j++){
            Eigen::Index ti = (i+n1) % nx;
            Eigen::Index tj = (j+n2) % ny;
            Eigen::Index shifti = (ti < 0) ? nx+ti : ti;
            Eigen::Index shiftj = (tj < 0) ? ny+tj : tj;
            out(shifti,shiftj) = in(i,j);
        }
    }
    return out;
}

enum FFTdirection {
    forward = 0,
    backward = 1
};

template<FFTdirection direc, typename Derived>
Eigen::VectorXcd fft2w(Eigen::DenseBase<Derived> &in, const int nx, const int ny){

    //Eigen::Map<RowMatrixXcd> matIn(in.derived().data(),nx,ny);

    /*std::vector<int> rowvec_in(nx);
    std::iota(rowvec_in.begin(), rowvec_in.end(), 0);
    std::vector<int> rowvec_out(nx);

    std::vector<int> colvec_in(ny);
    std::iota(colvec_in.begin(), colvec_in.end(), 0);
    std::vector<int> colvec_out(ny);*/

    RowMatrixXcd out(nx, ny);

    /*grppi::map(grppiex::dyn_ex("omp"),rowvec_in,rowvec_out,[&](auto k){
        Eigen::FFT<double> fft;
        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        fft.SetFlag(Eigen::FFT<double>::Unscaled);

        Eigen::VectorXcd tmpOut(ny);
        if constexpr(direc == forward){
            fft.fwd(tmpOut, matIn.row(k));
        }
        else{
            fft.inv(tmpOut, matIn.row(k));
        }
        matOut.row(k) = tmpOut;
        return 0;
    });

    grppi::map(grppiex::dyn_ex("omp"),colvec_in,colvec_out,[&](auto k){
        Eigen::FFT<double> fft;
        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        fft.SetFlag(Eigen::FFT<double>::Unscaled);

        Eigen::VectorXcd tmpOut(nx);
        if constexpr(direc == forward){
            fft.fwd(tmpOut, matOut.col(k));
        }
        else{
            fft.inv(tmpOut, matOut.col(k));
        }
        matOut.col(k) = tmpOut;
        return 0;
    });*/


    Eigen::FFT<double> fft;
    fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
    fft.SetFlag(Eigen::FFT<double>::Unscaled);

    for (int k=0; k<nx; ++k) {
        Eigen::VectorXcd tmp_out(ny);
        if constexpr(direc == forward){
            fft.fwd(tmp_out, in.row(k));
        }
        else{
            fft.inv(tmp_out, in.row(k));
        }
        out.row(k) = tmp_out;
    }

    for (int k=0; k<ny; ++k) {
        Eigen::VectorXcd tmp_out(nx);
        if constexpr(direc == forward){
            fft.fwd(tmp_out, out.col(k));
        }
        else{
            fft.inv(tmp_out, out.col(k));
        }
        out.col(k) = tmp_out;
    }

    //Eigen::VectorXcd out_vec(nx*ny);

    /*for(int i=0; i<nx; i++){
        for(int j=0; j<ny; j++)
            out_vec(ny*i+j) = out(i,j);
    }*/

    Eigen::Map<Eigen::VectorXcd> out_vec(out.data(),nx*ny);

    return std::move(out_vec);
}

template <typename Derived>
std::tuple<Eigen::VectorXd, Eigen::VectorXd> set_coverage_cut_ranges(Eigen::DenseBase<Derived> &in, const double weight_cut) {
    Eigen::VectorXd cut_x_range(2);
    Eigen::VectorXd cut_y_range(2);

    cut_x_range(0) = 0;
    cut_x_range(1) = in.rows() - 1;
    cut_y_range(0) = 0;
    cut_y_range(1) = in.cols() - 1;

    // find lower row bound
    bool flag = false;
    for (int i = 0; i < in.rows(); i++) {
        for (int j = 0; j < in.cols(); j++) {
            if (in(i,j) >= weight_cut) {
                cut_x_range(0) = i;
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
                cut_x_range(1) = i;
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
        for (int j = cut_x_range(0); j < cut_x_range(1) + 1; j++) {
            if (in(j,i) >= weight_cut) {
                cut_y_range(0) = i;
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
        for (int j = cut_x_range(0); j < cut_x_range(1) + 1; j++) {
            if (in(j,i) >= weight_cut) {
                cut_y_range(1) = i;
                flag = true;
                break;
            }
        }
        if (flag == true) {
            break;
        }
    }

    return std::make_tuple(cut_x_range,cut_y_range);
}

double selector(std::vector<double> input, int index){
    unsigned int pivotIndex = rand() % input.size();
    double pivotValue = input[pivotIndex];
    std::vector<double> left;
    std::vector<double> right;
    for (unsigned int x = 0; x < input.size(); x++) {
        if (x != pivotIndex) {
            if (input[x] > pivotValue) {
                right.push_back(input[x]);
            }
            else {
                left.push_back(input[x]);
            }
        }
    }
    if ((int)left.size() == index) {
        return pivotValue;
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

    for (int x = 0; x < nr; x++){
        for (int y = 0; y < nc; y++){
            if (in(x,y) > 0.){
                og.push_back(in(x,y));
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

} // namespace
