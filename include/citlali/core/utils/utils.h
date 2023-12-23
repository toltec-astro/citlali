#pragma once

#include <time.h>
#include <vector>
#include <numeric>
#include <complex>
//#include <png.h>

#include <fftw3.h>

#include <Eigen/Core>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/Splines>

#include <tula/grppi.h>
#include <tula/algorithm/ei_stats.h>

#include <citlali/core/utils/constants.h>

//#include "matplotlibcpp.h"

namespace engine_utils {

static const int parseLine(char* line){
    // this assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') {
        p++;
    }
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

static const int get_phys_memory() { // note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

// planck function (W x sr-1 x m-2 x Hz−1)
static const double planck_nu(const double freq_Hz, const double T_K) {
    return 2.*h_J_s*pow(freq_Hz,3)/pow(c_m_s,2)/(exp((h_J_s*freq_Hz)/(kB_J_K*T_K)) - 1.);
}

// convert flux in mJy/beam to uK
static const double mJy_beam_to_uK(const double flux_mjy_beam, const double freq_Hz, const double fwhm) {
    // planck function at T_CMB
    auto planck = planck_nu(freq_Hz, T_cmb_K);
    // exponent term (h x nu)/(k_B x T_cmb)
    auto h_nu_k_T = (h_J_s*freq_Hz)/(kB_J_K*T_cmb_K);
    // beam area in steradians
    //auto beam_area_rad = 2.*pi*pow(fwhm*FWHM_TO_STD*ASEC_TO_RAD,2);
    // conversion from K to mJy/beam
    auto K_to_mJy_beam = planck*exp(h_nu_k_T)/(exp(h_nu_k_T)-1)*h_nu_k_T/pow(T_cmb_K,2)*1e26*1e3;//*beam_area_rad;
    // flux in mJy/beam converted to uK
    return 1e6*flux_mjy_beam/K_to_mJy_beam;
}

// get current date/time, format is YYYY-MM-DD.HH:mm:ss
static const std::string current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

// convert unixt time to utc, format is YYYY-MM-DD.HH:mm:ss
static const std::string unix_to_utc(double &t) {
    time_t     now = t;
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

template <typename DerivedA, typename DerivedB>
void utc_to_unix(Eigen::DenseBase<DerivedA> &tel_utc, Eigen::DenseBase<DerivedB> &ut_date) {
    // size of time vector
    Eigen::Index n_pts = tel_utc.size();

    time_t utc_time;

    // who would have guessed?
    auto days_in_year = 365.0;

    int year = std::floor(ut_date(0));
    int days = int(((ut_date(0)-year)*days_in_year)+1);

    auto ut_time = 180/15/pi * tel_utc.derived().array();

    Eigen::VectorXd tel_unix(n_pts);

    // loop through points
    for (Eigen::Index i=0; i<n_pts; ++i) {
        auto h =(int)ut_time(i);
        auto m = (int)((ut_time(i) - h)*60);
        auto s = (((ut_time(i) - h)*60 - m)*60);

        struct tm tm_time;
        tm_time.tm_isdst = -1;
        tm_time.tm_mon = 0;
        tm_time.tm_mday = days;
        tm_time.tm_year = year - 1900;

        tm_time.tm_sec = s;
        tm_time.tm_min = m;
        tm_time.tm_hour = h;

        // get UTC time
        utc_time = timegm(&tm_time);

        // convert UTC time to Unix timestamp
        time_t unix_time = (time_t) utc_time;
        // add Unix time to vector
        tel_unix(i) = unix_time;
    }
    // overrwite UTC time with Unix time
    tel_utc = tel_unix;
}

static long long modified_julian_date_to_unix(double jd) {
    // unix epoch in julian date
    const double UNIX_EPOCH_JD = 2440587.5;

    // difference in days and convert from modified julian to julian date
    double diff = jd - UNIX_EPOCH_JD + 2400000.5;

    // convert days to seconds
    long long unix_time = static_cast<long long>(diff * 86400.0);

    return unix_time;
}

static long long unix_to_modified_julian_date(double unix_time) {
    const double seconds_per_day = 86400.0;
    const double mjd_offset = 40587.0;
    return unix_time / seconds_per_day + mjd_offset;
}

template <typename Derived>
auto make_meshgrid(const Eigen::DenseBase<Derived> &x, const Eigen::DenseBase<Derived> &y) {
    // column major
    // [x0, y0,] [x1, y0] [x2, y0] ... [xn, y0] [x0, y1] ... [xn, yn]
    const long nx = x.size(), ny = y.size();
    Derived xy(nx * ny, 2);
    // map xx [ny, nx] to the first column of xy
    Eigen::Map<Eigen::MatrixXd> xx(xy.data(), ny, nx);
    // map yy [ny, nx] to the second column of xy
    Eigen::Map<Eigen::MatrixXd> yy(xy.data() + xy.rows(), ny, nx);
    // populate xx such that each row is x
    for (Eigen::Index i=0; i<ny; ++i) {
        xx.row(i) = x.transpose();
    }
    // populate yy such that each col is y
    for (Eigen::Index j=0; j<nx; ++j) {
        yy.col(j) = y;
    }
    return xy;
}

// direction of fft
enum FFTDirection {
    forward = 0,
    inverse = 1
};

// parallelized 2d fft using eigen
template <FFTDirection direction, typename Derived>
auto fft(Eigen::DenseBase<Derived> &in, std::string parallel_policy) {
    // get dimensions of data
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
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy),col_vec_in,col_vec_out,[&](auto i) {
        Eigen::FFT<double> fft;
        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        fft.SetFlag(Eigen::FFT<double>::Unscaled);

        Eigen::VectorXcd temp_col(n_rows);
        // forward fft
        if constexpr(direction == forward) {
            fft.fwd(temp_col, in.col(i));
        }
        // inverse fft
        else if constexpr(direction == inverse){
            fft.inv(temp_col, in.col(i));
        }
        out.col(i) = temp_col;

        return 0;});

    // do the fft over the rows
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy),row_vec_in,row_vec_out,[&](auto i) {
        Eigen::FFT<double> fft;
        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        fft.SetFlag(Eigen::FFT<double>::Unscaled);

        Eigen::VectorXcd temp_row(n_cols);
        // forward fft
        if constexpr(direction == forward) {
            fft.fwd(temp_row, out.row(i));
        }
        // inversee fft
        else if constexpr(direction == inverse){
            fft.inv(temp_row, out.row(i));
        }
        out.row(i) = temp_row;

        return 0;});

    if constexpr(direction == forward) {
        // set fft normalization
        out = out/n_rows/n_cols;
    }

    return out;
}

// function to compute the nearest power of two greater than or equal to n
static int nearest_power_of_two(int n) {
    if (n < 1) return 1;
    int power = std::ceil(std::log2(n));
    return std::pow(2, power);
}

// parallelized 2d fft using eigen
template <FFTDirection direction, typename Derived, typename fftw_plan_t>
auto fft2(Eigen::DenseBase<Derived> &in, fftw_plan_t &plan, fftw_complex* a, fftw_complex*b){
    // get dimensions of data
    Eigen::Index n_rows = in.rows();
    Eigen::Index n_cols = in.cols();

    // output matrix
    Eigen::MatrixXcd out(n_rows, n_cols);

    // copy data from input (row major?)
    for (Eigen::Index i=0; i< n_rows; ++i) {
        for (Eigen::Index j=0; j<n_cols; ++j) {
            int ii = n_cols*i + j;
            a[ii][0] = in(i,j).real();
            a[ii][1] = in(i,j).imag();
        }
    }

    fftw_execute(plan);

    // copy data to output (row major?)
    for (Eigen::Index i=0; i< n_rows; ++i) {
        for (Eigen::Index j=0; j<n_cols; ++j) {
            int ii = n_cols*i + j;
            out.real()(i,j) = b[ii][0];
            out.imag()(i,j) = b[ii][1];
        }
    }

    if constexpr(direction == forward) {
        // set fft normalization
        out = out/n_rows/n_cols;
    }
    return out;
}

// iterate over tuple (https://stackoverflow.com/questions/26902633/how-to-iterate-over-a-stdtuple-in-c-11)
template<class F, class...Ts, std::size_t...Is>
void for_each_in_tuple(const std::tuple<Ts...> & tuple, F func, std::index_sequence<Is...>) {
    using expander = int[];
    (void)expander { 0, ((void)func(std::get<Is>(tuple)), 0)... };
}

template<class F, class...Ts>
void for_each_in_tuple(const std::tuple<Ts...> & tuple, F func) {
    for_each_in_tuple(tuple, func, std::make_index_sequence<sizeof...(Ts)>());
}

template<typename Derived>
std::vector<std::tuple<double, int>> sorter(Eigen::DenseBase<Derived> &vec) {
    std::vector<std::tuple<double, int>> vis;
    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(vec.size(),0,vec.size()-1);

    for(Eigen::Index i=0; i<vec.size(); ++i) {
        std::tuple<double, double> vec_and_val(vec(i), indices(i));
        vis.push_back(vec_and_val);
    }

    std::sort(vis.begin(), vis.end(),
              [&](const std::tuple<double, int> &a, const std::tuple<double, int> &b) -> bool {
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
                                 (data.derived().array().mean())).square().sum()) / n_samples);

    return std_dev;
}

template <typename DerivedA, typename DerivedB>
auto calc_std_dev(Eigen::DenseBase<DerivedA> &data, Eigen::DenseBase<DerivedB> &flag) {

    auto f = (1 - flag.derived().template cast<double>().array());

    // number of unflagged samples
    auto n_good = (f.array() == 1).count();

    // number of samples for divisor
    double n_samples;

    // no good samples
    if (n_good == 0) {
        return 0.0;
    }
    // one good sample
    else if (n_good == 1) {
        n_samples = n_good;
    }
    // > 1 good sample
    else {
        n_samples = n_good - 1;
    }

    auto mean = (data.derived().array()*f.array()).mean();

    // calc standard deviation
    double std_dev = std::sqrt((((data.derived().array()*f.array()) - mean).square().sum()) / n_samples);

    return std_dev;
}

template <typename DerivedA>
auto calc_rms(Eigen::DenseBase<DerivedA> &data) {

    // number of unflagged samples
    auto n_good = data.size();

    // number of samples for divisor
    double n_samples;

    if (n_good == 0) {
        return 0.0;
    }

    else {
        n_samples = n_good;
    }

    // calc rms
    return std::sqrt(data.derived().array().square().mean());
}

template <typename DerivedA, typename DerivedB>
auto calc_rms(Eigen::DenseBase<DerivedA> &data, Eigen::DenseBase<DerivedB> &flag) {

    // flip flags
    auto f = (flag.derived().template cast<double>().array() - 1).abs();

    // number of unflagged samples
    auto n_good = (f.array() == 1).count();

    // number of samples for divisor
    double n_samples;

    if (n_good == 0) {
        return 0.0;
    }

    else {
        n_samples = n_good;
    }

    // calc rms
    return std::sqrt((((data.derived().array()*f.array())).square().sum()) / n_samples);
}

template <typename Derived>
double calc_mad(Eigen::DenseBase<Derived> &data) {
    Eigen::Index n_pts = data.size();
    double med = tula::alg::median(data);

    Derived abs_delt = abs(data.derived().array() - med);
    return tula::alg::median(abs_delt);
}

static auto hanning_window(Eigen::Index n_rows, Eigen::Index n_cols) {
    double a = 2.*pi/n_rows;
    double b = 2.*pi/n_cols;

    Eigen::ArrayXd index = Eigen::ArrayXd::LinSpaced(n_rows,0,n_rows-1);
    Eigen::RowVectorXd row = -0.5 * cos(index*a) + 0.5;

    index = Eigen::ArrayXd::LinSpaced(n_cols,0,n_cols-1);
    Eigen::VectorXd col = -0.5 * cos(index*b) + 0.5;

    Eigen::MatrixXd window(n_rows, n_cols);

    for (Eigen::Index i=0; i<n_cols; ++i) {
        for (Eigen::Index j=0; j<n_rows; ++j) {
            window(j,i) = row(j)*col(i);
        }
    }

    return window;
}

static double pivot_select(std::vector<double> input, int index) {
    unsigned int pivot_index = rand() % input.size();
    double pivot_value = input[pivot_index];
    std::vector<double> left;
    std::vector<double> right;
    for (unsigned int i = 0; i < input.size(); ++i) {
        if (i != pivot_index) {
            if (input[i] > pivot_value) {
                right.push_back(input[i]);
            }
            else {
                left.push_back(input[i]);
            }
        }
    }
    if ((int)left.size() == index) {
        return pivot_value;
    }
    else if ((int)left.size() < index) {
        return pivot_select(right, index - left.size() - 1);
    }
    else {
        return pivot_select(left, index);
    }
}

template <typename Derived>
double find_weight_threshold(Eigen::DenseBase<Derived> &weight, double cov) {

    // find number of non-zero elements in weights
    Eigen::Index n_non_zero = (weight.derived().array() > 0).count();

    // if all values are zero, set coverage limit to zero
    if (n_non_zero==0) {
        return 0;
    }

    // vector to hold non-zero elements
    //Eigen::VectorXd non_zero_weights;
    //non_zero_weights.setZero();

    std::vector<double> non_zero_weights_vec;

    // check if weights are all constant
    int diff = 0;
    bool constant = false;

    Eigen::Index k = 0;
    // populate vector with non-zero elements
    for (Eigen::Index i=0; i<weight.rows(); ++i) {
        for (Eigen::Index j=0; j<weight.cols(); ++j) {
            if (weight(i,j) > 0) {
                //non_zero_weights(k) = weight(i,j);
                non_zero_weights_vec.push_back(weight(i,j));
                if (k==0) {
                    diff = weight(i,j);
                }
                else if (weight(i,j)!=diff) {
                    constant = true;
                }
                k++;
            }
        }
    }

    // value of weight at coverage limit value
    double weight_val;

    // if all weight values are constant, pivot search doesn't work
    if (constant) {
        Eigen::VectorXd non_zero_weights = Eigen::Map<Eigen::VectorXd>(non_zero_weights_vec.data(),
                                                                       non_zero_weights_vec.size());
        // sort in ascending order
        std::sort(non_zero_weights.data(), non_zero_weights.data() + non_zero_weights.size(),
                  [](double lhs, double rhs){return rhs > lhs;});

        // find index of upper 25 % of weights
        Eigen::Index cov_limit_index = 0.75*non_zero_weights.size();

        // get weight value at cov_limit_index + size/2
        Eigen::Index weight_index = std::floor((cov_limit_index + non_zero_weights.size())/2.);
        weight_val = non_zero_weights(weight_index);
    }

    // sort using pivot selection
    else {
        // find index of upper 25 % of weights
        Eigen::Index cov_limit_index = 0.75*non_zero_weights_vec.size();
        // get weight value at cov_limit_index + size/2
        //weight_val = pivot_select(non_zero_weights_vec, cov_limit_index);
        // get weight index
        Eigen::Index weight_index = floor((cov_limit_index + non_zero_weights_vec.size())/2.);
        weight_val = pivot_select(non_zero_weights_vec, weight_index);
    }

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
    for (Eigen::Index i=0; i<weight.rows(); ++i) {
        for (Eigen::Index j=0; j<weight.cols(); ++j) {
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
    for (Eigen::Index i=weight.rows()-1; i>-1; i--) {
        for (Eigen::Index j = 0; j < weight.cols(); ++j) {
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
    for (Eigen::Index i=0; i<weight.cols(); ++i) {
        for (Eigen::Index j = cov_ranges(0,0); j < cov_ranges(1,0) + 1; ++j) {
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
    for (Eigen::Index i=weight.cols()-1; i>-1; i--) {
        for (Eigen::Index j = cov_ranges(0,0); j < cov_ranges(1,0) + 1; ++j) {
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

// direction of fft
enum SmoothType {
    boxcar = 0,
    edge_truncate = 1
};

template<SmoothType smooth_type, typename DerivedA, typename DerivedB>
void smooth(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &out, int w) {
    // ensure w is odd
    if (w % 2 == 0) {
        w++;
    }

    Eigen::Index n_pts = in.size();
    if constexpr (smooth_type==boxcar) {
        out.head((w - 1) / 2) = in.head((w - 1) / 2);
        out.tail(n_pts - (w + 1) / 2. + 1) = in.tail(n_pts - (w + 1) / 2. + 1);

        double winv = 1. / w;
        int wm1d2 = (w - 1) / 2.;
        int wp1d2 = (w + 1) / 2.;

        for (int i = wm1d2; i <= n_pts - wp1d2; ++i) {
            out(i) = winv * in.segment(i - wm1d2, w).sum();
        }
    }

    else if constexpr (smooth_type==edge_truncate) {
        double w_inv = 1./w;
        int w_mid = (w - 1)/2;

        double sum;
        for (Eigen::Index i=0; i<n_pts; ++i) {
            sum=0;
            for (int j=0; j<w; ++j) {
                int add_index = i + j - w_mid;

                if (add_index < 0) {
                    add_index=0;
                }
                else if (add_index > n_pts-1) {
                    add_index = n_pts - 1;
                }
                sum += in(add_index);
            }
            out(i) = w_inv*sum;
        }
    }
}

template <typename DerivedA, typename DerivedB>
auto calc_2D_psd(Eigen::DenseBase<DerivedA> &data, Eigen::DenseBase<DerivedB> &y, Eigen::DenseBase<DerivedB> &x,
                 int smooth_window, std::string parallel_policy) {

    Eigen::Index n_rows = data.rows();
    Eigen::Index n_cols = data.cols();

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

    // fftw setup
    fftw_complex *a;
    fftw_complex *b;
    fftw_plan pf;

    a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
    b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);

    pf = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_FORWARD, FFTW_ESTIMATE);

    // do fft
    //in = engine_utils::fft<engine_utils::forward>(in, parallel_policy);
    in = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

    // free fftw vectors
    fftw_free(a);
    fftw_free(b);

    // destroy fftw plan
    fftw_destroy_plan(pf);

    //in = in/n_rows/n_cols;

    // get power
    Eigen::MatrixXd out = diffq_rows*diffq_cols*(in.real().array().pow(2) + in.imag().array().pow(2));

    // make vectors for frequencies
    Eigen::VectorXd q_rows(n_rows), q_cols(n_cols);

    // shift q_rows
    Eigen::Index index;
    Eigen::Index shift = n_rows/2 - 1;
    for (Eigen::Index i=0; i<n_rows; ++i) {
        index = i-shift;
        if (index < 0) {
            index += n_rows;
        }
        q_rows(index) = diffq_rows*(i-(n_rows/2-1));
    }

    // shift q_cols
    shift = n_cols/2 - 1;
    for (Eigen::Index i=0; i<n_cols; ++i) {
        index = i-shift;
        if (index < 0) {
            index += n_cols;
        }
        q_cols(index) = diffq_cols*(i-(n_cols/2-1));
    }

    // remove first row and column of power, qr, qc
    Eigen::MatrixXd pmfq = out.block(1,1,n_rows-1,n_cols-1);

    // shift rows over by 1
    for (Eigen::Index i=0; i<n_rows-1; ++i) {
        q_rows(i) = q_rows(i+1);
    }

    // shift cols over by 1
    for (Eigen::Index j=0; j<n_cols-1; ++j) {
        q_cols(j) = q_cols(j+1);
    }

    // matrices of frequencies and distances
    Eigen::MatrixXd qmap(n_rows-1,n_cols-1);
    Eigen::MatrixXd qsymm(n_rows-1,n_cols-1);

    for (Eigen::Index i=1; i<n_cols; ++i) {
        for(Eigen::Index j=1; j<n_rows; ++j) {
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

    std::vector<Eigen::Index> psd_vec_in(nn);
    std::iota(psd_vec_in.begin(), psd_vec_in.end(),0);
    std::vector<Eigen::Index> psd_vec_out(nn);

    // do the fft over the cols (parallelized for large maps)
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy),psd_vec_in,psd_vec_out,[&](auto i) {
    //for (Eigen::Index i=0; i<nn; ++i) {
        int count_s = 0;
        int count_a = 0;
        double psdarr_s = 0.;
        double psdarr_a = 0.;
        for (int j=0; j<n_cols-1; ++j) {
            for (int k=0; k<n_rows-1; ++k) {
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
        return 0;
    });

     // smooth the psd
     Eigen::VectorXd smoothed_psd(nn);
     engine_utils::smooth<edge_truncate>(psd, smoothed_psd, smooth_window);
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
    for (Eigen::Index j=0; j<n_bins-1; ++j) {
        hist(j) = ((data.derived().array() >= hist_bins(j)) && (data.derived().array() < hist_bins(j+1))).count();
    }

    return std::tuple<Eigen::VectorXd, Eigen::VectorXd>(hist, hist_bins);
}

// shift a vector
template <typename Derived>
auto shift_1D(Eigen::DenseBase<Derived> &in, std::vector<Eigen::Index> shift_indices) {

    Eigen::Index n_pts = in.size();

    Derived out(n_pts);

    for (Eigen::Index i=0; i<n_pts; ++i) {
        Eigen::Index ti = (i+shift_indices[0])%n_pts;
        Eigen::Index shift_index = (ti < 0) ? n_pts+ti : ti;
        out(shift_index) = in(i);
    }

    return out;
}

// shift a 2D matrix
template <typename Derived>
auto shift_2D(Eigen::DenseBase<Derived> &in, std::vector<Eigen::Index> shift_indices) {
    Eigen::Index n_rows = in.rows();
    Eigen::Index n_cols = in.cols();

    Derived out(n_rows,n_cols);

    for (Eigen::Index i=0; i<n_cols; ++i) {
        Eigen::Index ti = (i+shift_indices[1]) % n_cols;
        Eigen::Index shift_col = (ti < 0) ? n_cols+ti : ti;
        for (Eigen::Index j=0; j<n_rows; ++j) {
            Eigen::Index tj = (j+shift_indices[0]) % n_rows;
            Eigen::Index shift_row = (tj < 0) ? n_rows+tj : tj;
            out(shift_row,shift_col) = in(j,i);
        }
    }
    return out;
}

/*
void abort_(const char * s, ...) {
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

void write_png_file(char* file_name, int width, int height) {

    int x, y;

    png_byte color_type;
    png_byte bit_depth;

    png_structp png_ptr;
    png_infop info_ptr;
    int number_of_passes;
    png_bytep * row_pointers;

    // create file
    FILE *fp = fopen(file_name, "wb");
    if (!fp)
        abort_("[write_png_file] File %s could not be opened for writing", file_name);

    // initialize stuff
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
        abort_("[write_png_file] png_create_write_struct failed");

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
        abort_("[write_png_file] png_create_info_struct failed");

    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during init_io");

    png_init_io(png_ptr, fp);

    // write header
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during writing header");

    png_set_IHDR(png_ptr, info_ptr, width, height,
                 bit_depth, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    // write bytes
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during writing bytes");

    png_write_image(png_ptr, row_pointers);

    // end write
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during end of write");

    png_write_end(png_ptr, NULL);

    // cleanup heap allocation
    for (y=0; y<height; y++)
        free(row_pointers[y]);
    free(row_pointers);

    fclose(fp);
}*/

class SplineFunction {
public:
    double x_min;
    double x_max;

    SplineFunction(Eigen::VectorXd const &x_vec,
                   Eigen::VectorXd const &y_vec)
        : x_min(x_vec.minCoeff()),
          x_max(x_vec.maxCoeff()),
          // spline fitting here. X values are scaled down to [0, 1] for this.
          spline_(Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(
              y_vec.transpose(),
              // no more than cubic spline, but accept short vectors.
              std::min<int>(x_vec.rows() - 1, 3),
              scaled_values(x_vec))) {}

    SplineFunction() = default;

    void interpolate(Eigen::VectorXd const &x_vec,
                     Eigen::VectorXd const &y_vec) {
        x_min = x_vec.minCoeff();
        x_max = x_vec.maxCoeff();
        // spline fitting here. X values are scaled down to [0, 1] for this.
        spline_ = Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(
            y_vec.transpose(),
            // no more than cubic spline, but accept short vectors.
            std::min<int>(x_vec.rows() - 1, 3),
            scaled_values(x_vec));
    }

    double operator()(double x) const {
        // x values need to be scaled down in extraction as well.
        return spline_(scaled_value(x))(0);
    }

private:
    // helpers to scale x values down to [0, 1]
    double scaled_value(double x) const {
        return (x - x_min) / (x_max - x_min);
    }

    Eigen::RowVectorXd scaled_values(Eigen::VectorXd const &x_vec) const {
        return x_vec.unaryExpr([this](double x) { return scaled_value(x); }).transpose();
    }

    // spline of one-dimensional points.
    Eigen::Spline<double, 1> spline_;
};

template <typename Derived>
void fix_periodic_boundary(Eigen::DenseBase<Derived>&data, double low,
                           double high, double upper_limit) {

    Eigen::Index n_pts = data.size();
    double max = data.maxCoeff();
    double min = data.minCoeff();
    if (max > high && min < low) {
        for (Eigen::Index i=0; i<n_pts; ++i) {
            if (data(i) < low) {
                data(i) += upper_limit;
            }
        }
    }
}

/*template <class map_class_t, class map_fitter_t>
void calc_gaussian_transfer_func(map_class_t &mb, map_fitter_t &map_fitter, int map_index) {
    // x and y spacing should be equal
    diff_rows = abs(mb.rows_tan_vec(1) - mb.rows_tan_vec(0));
    diff_cols = abs(mb.cols_tan_vec(1) - mb.cols_tan_vec(0));

    // collect what we need
    Eigen::MatrixXd temp_kernel = mb.kernel[map_index];

    double init_row = -99;
    double init_col = -99;

    // carry out fit to kernel to get centroid
    auto [map_params, map_perror, good_fit] =
        map_fitter.fit_to_gaussian<engine_utils::mapFitter::pointing>(mb.kernel[map_index], mb.weight[map_index],
                                                                      init_fwhm, init_row, init_col);

    // if fit failed, give up
    if (!good_fit) {
        logger->error("fit to kernel map failed. try setting a small fitting_region_arcsec value.");
        std::exit(EXIT_FAILURE);
    }

    // rescale parameters to on-sky units
    map_params(1) = mb.pixel_size_rad*(map_params(1) - (n_cols)/2);
    map_params(2) = mb.pixel_size_rad*(map_params(2) - (n_rows)/2);

    Eigen::Index shift_row = -std::round(map_params(2)/diff_rows);
    Eigen::Index shift_col = -std::round(map_params(1)/diff_cols);

    std::vector<Eigen::Index> shift_indices = {shift_row,shift_col};
    temp_kernel = engine_utils::shift_2D(temp_kernel, shift_indices);

    // calculate distance
    Eigen::MatrixXd dist(n_rows,n_cols);
    for (Eigen::Index i=0; i<n_cols; ++i) {
        for (Eigen::Index j=0; j<n_rows; ++j) {
            dist(j,i) = sqrt(pow(mb.rows_tan_vec(j),2)+pow(mb.cols_tan_vec(i),2));
        }
    }

    // pixel closet to tangent point
    Eigen::Index row_index, col_index;
    auto min_dist = dist.minCoeff(&row_index,&col_index);

    // create new bins based on diff_rows
    int n_bins = mb.rows_tan_vec(n_rows-1)/diff_rows;
    Eigen::VectorXd bin_low = Eigen::VectorXd::LinSpaced(n_bins,0,n_bins-1)*diff_rows;

    Eigen::VectorXd kernel_interp(n_bins-1);
    kernel_interp.setZero();
    Eigen::VectorXd dist_interp(n_bins-1);
    dist_interp.setZero();

    // radial averages
    for (Eigen::Index i=0; i<n_bins-1; ++i) {
        int c = 0;
        for (Eigen::Index j=0; j<n_cols; ++j) {
            for (Eigen::Index k=0; k<n_rows; ++k) {
                if (dist(k,j) >= bin_low(i) && dist(k,j) < bin_low(i+1)){
                    c++;
                    kernel_interp(i) += temp_kernel(k,j);
                    dist_interp(i) += dist(k,j);
                }
            }
        }
        kernel_interp(i) /= c;
        dist_interp(i) /= c;
    }

    // now spline interpolate to generate new template array
    filter_template.resize(n_rows,n_cols);

    // create spline function
    engine_utils::SplineFunction s(dist_interp, kernel_interp);

    // carry out the interpolation
    for (Eigen::Index i=0; i<n_cols; ++i) {
        for (Eigen::Index j=0; j<n_rows; ++j) {
            Eigen::Index tj = (j-row_index)%n_rows;
            Eigen::Index ti = (i-col_index)%n_cols;
            Eigen::Index shiftj = (tj < 0) ? n_rows+tj : tj;
            Eigen::Index shifti = (ti < 0) ? n_cols+ti : ti;

            // if within limits
            if (dist(j,i) <= s.x_max && dist(j,i) >= s.x_min) {
                filter_template(shiftj,shifti) = s(dist(j,i));
            }
            // if above x limit
            else if (dist(j,i) > s.x_max) {
                filter_template(shiftj,shifti) = kernel_interp(kernel_interp.size()-1);
            }
            // if below x limit
            else if (dist(j,i) < s.x_min) {
                filter_template(shiftj,shifti) = kernel_interp(0);
            }
        }
    }
}*/

/// @brief Return the current time.
inline auto now() { return std::chrono::high_resolution_clock::now(); }

/// @brief Return the time elapsed since given time.
inline auto elapsed_since(
    const std::chrono::time_point<std::chrono::high_resolution_clock> &t0) {
    return now() - t0;
}

/// @brief An RAII class to report the lifetime of itself.
struct scoped_timeit {
    scoped_timeit(std::string_view msg_, double *elapsed_msec_ = nullptr)
        : msg(msg_), elapsed_msec(elapsed_msec_) {
        logger->info("**timeit** {}", msg);
    }
    ~scoped_timeit() {
        auto t = elapsed_since(t0);
        constexpr auto s_to_ms = 1e3;
        if (this->elapsed_msec != nullptr) {
            *(this->elapsed_msec) =
                std::chrono::duration_cast<std::chrono::duration<double>>(t)
                    .count() *
                s_to_ms;
        }
        logger->info("**timeit** {} finished in {}", msg, t);
    }
    scoped_timeit(const scoped_timeit &) = delete;
    scoped_timeit(scoped_timeit &&) = delete;
    auto operator=(const scoped_timeit &) -> scoped_timeit & = delete;
    auto operator=(scoped_timeit &&) -> scoped_timeit & = delete;

    std::chrono::time_point<std::chrono::high_resolution_clock> t0{now()};
    std::string_view msg;
    double *elapsed_msec{nullptr};
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

};

} //namespace engine_utils
