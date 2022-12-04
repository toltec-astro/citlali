#pragma once

#include <time.h>
#include <vector>
#include <numeric>
#include <complex>
#include <Eigen/Core>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/Splines>

#include <tula/grppi.h>

#include <citlali/core/utils/constants.h>

namespace engine_utils {

// get current date/time, format is YYYY-MM-DD.HH:mm:ss
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

// direction of fft
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

    // do the fft over the rows
    grppi::map(tula::grppi_utils::dyn_ex(exmode),row_vec_in,row_vec_out,[&](auto i) {
        Eigen::FFT<double> fft;
        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        fft.SetFlag(Eigen::FFT<double>::Unscaled);

        Eigen::VectorXcd temp_row(n_cols);

        if constexpr(direction == forward) {
            fft.fwd(temp_row, out.row(i));
        }
        else if constexpr(direction == inverse){
            fft.inv(temp_row, out.row(i));
        }
        out.row(i) = temp_row;

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

    if (n_good == 0) {
        return 0.0;
    }

    else if (n_good == 1) {
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

    Eigen::ArrayXd index = Eigen::ArrayXd::LinSpaced(n_rows,0,n_rows-1);
    Eigen::RowVectorXd row = -0.5 * cos(index*a) + 0.5;

    index = Eigen::ArrayXd::LinSpaced(n_cols,0,n_cols-1);
    Eigen::VectorXd col = -0.5 * cos(index*b) + 0.5;

    Eigen::MatrixXd window(n_rows, n_cols);
    //window = row.array().colwise()*col.array();

    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            window(j,i) = row(j)*col(i);
        }
    }

    return window;
}

template <typename Derived>
auto find_weight_threshold(const Eigen::DenseBase<Derived> &weight, const double cov) {

    // find number of non-zero elements in weights
    Eigen::Index n_non_zero = (weight.derived().array() !=0).count();

    // vector to hold non-zero elements
    Eigen::VectorXd non_zero_weights(n_non_zero);

    // populate vector with non-zero elements
    Eigen::Index k = 0;
    for (Eigen::Index i=0; i<weight.rows(); i++) {
        for (Eigen::Index j=0; j<weight.rows(); j++) {
            if (weight(i,j) > 0) {
                non_zero_weights(k) = weight(i,j);
                k++;
            }
        }
    }

    // sort in ascending order
    std::sort(non_zero_weights.data(), non_zero_weights.data() + non_zero_weights.size(),
              [](double lhs, double rhs){return rhs > lhs;});

    // find index of upper 25 % of weights
    Eigen::Index cov_limit_index = 0.75*non_zero_weights.size();

    // get weight value at cov_limit_index + size/2
    double weight_val = non_zero_weights[std::floor((cov_limit_index + non_zero_weights.size())/2.)];

    // return weight value x coverage cut
    return weight_val*cov;
}

template <typename Derived>
auto set_cov_cov_ranges(const Eigen::DenseBase<Derived> &weight, const double weight_threshold) {
    // matrix to hold coverage ranges
    Eigen::MatrixXd cov_ranges(2,2);

    cov_ranges.row(0).setZero();
    cov_ranges(1,0) = weight.rows() - 1;
    cov_ranges(1,1) = weight.cols() - 1;

    // find lower row bound
    bool flag = false;
    for (Eigen::Index i = 0; i < weight.rows(); i++) {
        for (Eigen::Index j = 0; j < weight.cols(); j++) {
            if (weight(i,j) >= weight_threshold) {
                cov_ranges(0,0) = i;
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
    for (Eigen::Index i = weight.rows() - 1; i > -1; i--) {
        for (Eigen::Index j = 0; j < weight.cols(); j++) {
            if (weight(i,j) >= weight_threshold) {
                cov_ranges(1,0) = i;
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
    for (Eigen::Index i = 0; i < weight.cols(); i++) {
        for (Eigen::Index j = cov_ranges(0,0); j < cov_ranges(1,0) + 1; j++) {
            if (weight(j,i) >= weight_threshold) {
                cov_ranges(0,1) = i;
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
    for (Eigen::Index i = weight.cols() - 1; i > -1; i--) {
        for (Eigen::Index j = cov_ranges(0,0); j < cov_ranges(1,0) + 1; j++) {
            if (weight(j,i) >= weight_threshold) {
                cov_ranges(1,1) = i;
                flag = true;
                break;
            }
        }
        if (flag == true) {
            break;
        }
    }

    return cov_ranges;
}

// box car smooth function
template<typename DerivedA, typename DerivedB>
void smooth_boxcar(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &out, int w) {
    // ensure box-car width is odd
    if (w % 2 == 0) {
        w++;
    }

    Eigen::Index n_pts = in.size();

    out.head((w - 1) / 2) = in.head((w - 1) / 2);
    out.tail(n_pts - (w + 1) / 2. + 1) = in.tail(n_pts - (w + 1) / 2. + 1);

    double winv = 1. / w;
    int wm1d2 = (w - 1) / 2.;
    int wp1d2 = (w + 1) / 2.;

    for (int i = wm1d2; i <= n_pts - wp1d2; i++) {
        out(i) = winv * in.segment(i - wm1d2, w).sum();
    }
}

template <typename Derived>
void smooth_edge_truncate(Eigen::DenseBase<Derived> &in, Eigen::DenseBase<Derived> &out, int w) {
    Eigen::Index n_pts = in.size();

    // ensure w is odd
    if (w % 2 == 0) {
        w++;
    }

    double w_inv = 1./w;
    int w_mid = (w - 1)/2;

    double sum;
    for (Eigen::Index i=0; i<n_pts; i++) {
        sum=0;
        for (int j=0; j<w; j++) {
            int add_index = i + j - w_mid;

            if (add_index < 0) {
                add_index=0;
            }
            else if (add_index > n_pts-1) {
                add_index = n_pts-1;
            }
            sum += in(add_index);
        }
        out(i) = w_inv*sum;
    }
}

template <typename DerivedA, typename DerivedB>
auto calc_2D_psd(Eigen::DenseBase<DerivedA> &data, Eigen::DenseBase<DerivedB> &x, Eigen::DenseBase<DerivedB> &y,
              Eigen::Index n_rows, Eigen::Index n_cols, int smooth_window, std::string parallel_policy) {
    Eigen::VectorXd psd, psd_freq;

    double diff_rows = y(1) - y(0);
    double diff_cols = x(1) - x(0);
    double rsize = diff_rows * n_rows;
    double csize = diff_cols * n_cols;
    double diffq_rows = 1. / rsize;
    double diffq_cols = 1. / csize;

    // setup input signal data
    Eigen::MatrixXcd in(n_rows, n_cols);
    in.real() = data;
    in.imag().setZero();

    // apply hanning window
    in.real() = in.real().array()*engine_utils::hanning_window(n_rows, n_cols).array();

    // do fft
    in = engine_utils::fft<engine_utils::forward>(in, parallel_policy);
    in = in/n_rows/n_cols;

    // get power
    Eigen::MatrixXd out = diffq_rows*diffq_cols*(in.real().array().pow(2) + in.imag().array().pow(2));

    // make vectors for frequencies
    Eigen::VectorXd q_rows(n_rows), q_cols(n_cols);

    // shift q_rows
    Eigen::Index index;
    Eigen::Index shift = n_rows/2 - 1;
    for (Eigen::Index i=0; i<n_rows; i++) {
        index = i-shift;
        if (index < 0) {
            index += n_rows;
        }
        q_rows(index) = diffq_rows*(i-(n_rows/2-1));
    }

    // shift q_cols
    shift = n_cols/2 - 1;
    for (Eigen::Index i=0; i<n_cols; i++) {
        index = i-shift;
        if (index < 0) {
            index += n_cols;
        }
        q_cols(index) = diffq_cols*(i-(n_cols/2-1));
    }

    // remove first row and column of power, qr, qc
    Eigen::MatrixXd pmfq = out.block(1,1,n_rows-1,n_cols-1);

    // shift rows over by 1
    for (Eigen::Index i=0; i<n_rows-1; i++) {
        q_rows(i) = q_rows(i+1);
    }

    // shift cols over by 1
    for (Eigen::Index j=0; j<n_cols-1; j++) {
        q_cols(j) = q_cols(j+1);
    }

    // matrices of frequencies and distances
    Eigen::MatrixXd qmap(n_rows-1,n_cols-1);
    Eigen::MatrixXd qsymm(n_rows-1,n_cols-1);

    for (Eigen::Index i=1; i<n_cols; i++) {
        for(Eigen::Index j=1; j<n_rows; j++) {
            qmap(j-1,i-1) = sqrt(pow(q_rows(j),2) + pow(q_cols(i),2));
            qsymm(j-1,i-1) = q_rows(j)*q_cols(i);
        }
    }

    // find max of n_rows and n_cols and get diff_q
    Eigen::Index nn;
    double diff_q;
    if (n_rows > n_cols) {
        nn = n_rows/2 + 1;
        diff_q = diffq_rows;
    }
    else {
        nn = n_cols/2 + 1;
        diff_q = diffq_cols;
    }

    // psd frequency vector
    psd_freq = Eigen::VectorXd::LinSpaced(nn,diff_q*0.5,diff_q*(nn-1 + 0.5));

    // get psd values
    psd.setZero(nn);
    for (Eigen::Index i=0; i<nn; i++) {
        int count_s = 0;
        int count_a = 0;
        double psdarr_s = 0.;
        double psdarr_a = 0.;
        for (int j=0; j<n_cols-1; j++) {
            for (int k=0; k<n_rows-1; k++) {
                if ((int) (qmap(k,j) / diff_q) == i && qsymm(k,j) >= 0.){
                    count_s++;
                    psdarr_s += pmfq(k,j);
                }
                else {
                    count_a++;
                    psdarr_a += pmfq(k,j);
                }
            }
        }
        if (count_s != 0) {
            psdarr_s /= count_s;
        }
        if (count_a != 0) {
            psdarr_a /= count_a;
        }
        psd(i) = std::min(psdarr_s,psdarr_a);
    }

     // smooth the psd
     Eigen::VectorXd smoothed_psd(nn);
     engine_utils::smooth_edge_truncate(psd, smoothed_psd, smooth_window);
     psd = std::move(smoothed_psd);

    return std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd>(psd, psd_freq, pmfq, qmap);
}

template <typename Derived>
auto calc_hist(Eigen::DenseBase<Derived> &data, int n_bins) {
    double min_data = data.minCoeff();
    double max_data = data.maxCoeff();

    // force the histogram to be symmetric about 0
    double rg = (abs(min_data) > abs(max_data)) ? abs(min_data) : abs(max_data);

    // set up hist bins
    Eigen::VectorXd hist_bins = Eigen::VectorXd::LinSpaced(n_bins,-rg,rg);
    // set size of hist values
    Eigen::VectorXd hist;
    hist.setZero(n_bins);

    // loop through bins and count up values
    for (Eigen::Index j=0; j<n_bins-1; j++) {
        hist(j) = ((data.derived().array() >= hist_bins(j)) && (data.derived().array() < hist_bins(j+1))).count();
    }

    return std::tuple<Eigen::VectorXd, Eigen::VectorXd>(hist, hist_bins);
}

template <typename Derived>
auto shift_1D(Eigen::DenseBase<Derived> &in, std::vector<Eigen::Index> shift_indices) {

    Eigen::Index n_pts = in.size();

    Eigen::VectorXd out(n_pts);

    for (Eigen::Index i=0; i<n_pts; i++) {
        Eigen::Index ti = (i+shift_indices[0])%n_pts;
        Eigen::Index shift_index = (ti < 0) ? n_pts+ti : ti;
        out(shift_index) = in(i);
    }

    return std::move(out);
}

template <typename Derived>
auto shift_2D(Eigen::DenseBase<Derived> &in, std::vector<Eigen::Index> shift_indices) {
    Eigen::Index n_rows = in.rows();
    Eigen::Index n_cols = in.cols();

    Eigen::MatrixXd out(n_rows,n_cols);

    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            Eigen::Index ti = (i+shift_indices[1]) % n_cols;
            Eigen::Index tj = (j+shift_indices[0]) % n_rows;
            Eigen::Index shift_col = (ti < 0) ? n_cols+ti : ti;
            Eigen::Index shift_row = (tj < 0) ? n_rows+tj : tj;
            out(shift_row,shift_col) = in(shift_row,shift_col);
        }
    }

    return std::move(out);
}

} //namespace engine_utils
