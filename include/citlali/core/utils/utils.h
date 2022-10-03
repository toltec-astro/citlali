#pragma once

#include <time.h>
#include <vector>
#include <numeric>
#include <complex>
#include <Eigen/Core>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/Splines>

#include <tula/grppi.h>


namespace engine_utils {

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

enum FFTDirection {
    forward = 0,
    inverse = 1
};

// parallelized 2d fft using eigen
template <FFTDirection direction, typename Derived>
auto fft(Eigen::DenseBase<Derived> &in, std::string exmode) {
    Eigen::Index n_rows = in.rows();
    Eigen::Index n_cols = in.cols();

    // output matrix
    Eigen::MatrixXcd out(n_rows, n_cols);

    // placeholder vectors for grppi maps
    std::vector<Eigen::Index> row_vec_in(n_rows);
    std::iota(row_vec_in.begin(), row_vec_in.end(),0);
    std::vector<Eigen::Index> row_vec_out(n_rows);

    std::vector<Eigen::Index> col_vec_in(n_cols);
    std::iota(col_vec_in.begin(), col_vec_in.end(),0);
    std::vector<Eigen::Index> col_vec_out(n_cols);

    // do the fft over the rows
    grppi::map(tula::grppi_utils::dyn_ex(exmode),row_vec_in,row_vec_out,[&](auto i) {
        Eigen::FFT<double> fft;
        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        fft.SetFlag(Eigen::FFT<double>::Unscaled);

        Eigen::VectorXcd temp_row(n_cols);

        if constexpr(direction == forward) {
            fft.fwd(temp_row, in.row(i));
        }
        else if constexpr(direction == inverse){
            fft.inv(temp_row, in.row(i));
        }
        out.row(i) = temp_row;

        return 0;});

    // do the fft over the cols
    grppi::map(tula::grppi_utils::dyn_ex(exmode),col_vec_in,col_vec_out,[&](auto i) {
        Eigen::FFT<double> fft;
        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        fft.SetFlag(Eigen::FFT<double>::Unscaled);

        Eigen::VectorXcd temp_col(n_rows);

        if constexpr(direction == forward) {
            fft.fwd(temp_col, in.col(i));
        }
        else if constexpr(direction == inverse){
            fft.inv(temp_col, in.col(i));
        }
        out.col(i) = temp_col;

        return 0;});

    return out;
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

template <typename DerivedA>
auto calc_std_dev(Eigen::DenseBase<DerivedA> &data) {

    auto n_good = data.derived().size();

    // number of samples for divisor
    double n_samples;
    if (n_good == 1) {
        n_samples = n_good;
    }
    else {
        n_samples = n_good - 1;
    }

    // calc standard deviation
    double std_dev = std::sqrt((((data.derived().array()) -
                                 (data.derived().array()).sum()/
                                     n_good).square().sum()) / n_samples);

    return std_dev;
}

template <typename DerivedA, typename DerivedB>
auto calc_std_dev(Eigen::DenseBase<DerivedA> &data, Eigen::DenseBase<DerivedB> &flag) {

    // number of unflagged samples
    auto n_good = (flag.derived().array() == 1).count();

    // number of samples for divisor
    double n_samples;
    if (n_good == 1) {
        n_samples = n_good;
    }
    else {
        n_samples = n_good - 1;
    }

    // calc standard deviation
    double std_dev = std::sqrt((((data.derived().array() *flag.derived().template cast<double>().array()) -
                            (data.derived().array() * flag.derived().template cast<double>().array()).sum()/
                                n_good).square().sum()) / n_samples);

    return std_dev;
}

auto hanning_window(Eigen::Index n_rows, Eigen::Index n_cols) {
    double a = 2.*pi/n_rows;
    double b = 2.*pi/n_cols;

    Eigen::ArrayXd index = Eigen::ArrayXd::LinSpaced(n_rows,0,n_rows+1);
    //for (Eigen::Index i=0; i<n_rows; i++) index(i) = (double) i;

    Eigen::RowVectorXd row = -0.5 * cos(index*a) + 0.5;

    //for(Eigen::Index i=0;i<n_rows;i++) row(i) = -0.5 * cos(index(i)*a) + 0.5;

    index = Eigen::ArrayXd::LinSpaced(n_cols,0,n_cols+1);

    //for (Eigen::Index i=0;i<n_cols;i++) index(i) = (double) i;

    Eigen::VectorXd col = -0.5 * cos(index*b) + 0.5;
    //for (Eigen::Index i=0;i<n_cols;i++) col(i) = -0.5 * cos(index(i)*b) + 0.5;

    Eigen::MatrixXd window(n_rows, n_cols);
    window = row.array().colwise()*col.array();
    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            window(j,i) = row(j)*col(i);
        }
    }

    return window;
}


} //namespace engine_utils
