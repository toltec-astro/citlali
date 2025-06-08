#pragma once

#include <utility>

#include <gsl/gsl_sf_bessel.h>

template<typename T>
struct is_std_vector : std::false_type {};

template<typename T, typename Alloc>
struct is_std_vector<std::vector<T, Alloc>> : std::true_type {};

template <typename Derived>
struct is_eigen_vector {
    static constexpr bool value = (Eigen::internal::traits<Derived>::RowsAtCompileTime == 1 ||
                                   Eigen::internal::traits<Derived>::ColsAtCompileTime == 1);
};

template <typename Derived>
struct is_eigen_matrix {
    static constexpr bool value = (Eigen::internal::traits<Derived>::RowsAtCompileTime != 1 &&
                                   Eigen::internal::traits<Derived>::ColsAtCompileTime != 1);
};

// function to create mappings from numbers to offsets and offsets to numbers
void create_mappings(const std::vector<std::vector<int>>& lists,
                     std::map<int, int>& number_to_offset,
                     std::map<int, int>& offset_to_number) {
    // initialize offset
    int offset = 0;

    // process each list
    for (const auto& list : lists) {
        for (int num : list) {
            number_to_offset[num] = offset;
            offset_to_number[offset] = num;
            ++offset;
        }
    }
}

// function to calculate the variance excluding flagged samples
double flagged_variance(const Eigen::VectorXd& data, const Eigen::Matrix<bool, Eigen::Dynamic, 1>& flags) {
    // create a mask where flagged samples
    Eigen::Matrix<bool, Eigen::Dynamic, 1> mask = (flags.array() == 0);

    // get the number of unflagged samples
    int count = mask.cast<int>().sum() - 1;

    // if there are no unflagged samples
    if (count == 0) {
        return 0.0;
    }

    // filter the data based on the mask (only unflagged samples)
    Eigen::VectorXd unflagged_samples = data.array() * mask.array().cast<double>();

    // calculate the mean of unflagged samples
    double mean = unflagged_samples.sum() / count;

    // calculate the variance
    double variance = (unflagged_samples.array().square().sum() / count) - (mean * mean);

    return variance;
}

// calculates arbitrary order polynomial values given coefficients and a vector x
template <typename DerivedA, typename DerivedB>
auto polynomial_model(const Eigen::DenseBase<DerivedA>& coeff, const Eigen::DenseBase<DerivedB>& x) {
    using Scalar = typename DerivedA::Scalar;
    Eigen::Matrix<typename DerivedB::Scalar, Eigen::Dynamic, 1> result = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(x.size());

    // accumulate the polynomial sum, starting from the highest degree
    for (int i = 0; i < coeff.size(); ++i) {
        int degree = coeff.size() - 1 - i;
        result += coeff(i) * x.derived().array().pow(degree).matrix();
    }

    return result;
}

template <typename Derived>
std::vector<std::tuple<typename Derived::Scalar, int>> sorter(Eigen::DenseBase<Derived> &vec) {
    std::vector<std::tuple<typename Derived::Scalar, int>> vis;

    for (int i = 0; i < vec.size(); ++i) {
        std::tuple<typename Derived::Scalar, int> vec_and_val(vec(i), i);
        vis.push_back(vec_and_val);
    }

    std::sort(vis.begin(), vis.end(),
              [&](const std::tuple<typename Derived::Scalar, int> &a,
                  const std::tuple<typename Derived::Scalar, int> &b) -> bool {
                  return std::get<0>(a) < std::get<0>(b);
              });

    return vis;
}

template <typename Derived,  typename... Args>
auto shift(const Eigen::DenseBase<Derived>& input, Args... args) {
    if constexpr (is_eigen_vector<Derived>::value) {
        int shift = std::get<0>(std::tuple<Args...>(args...));
        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, 1> output(input.size());
        Eigen::ArrayXi indices = (Eigen::ArrayXi::LinSpaced(input.size(), 0, input.size() - 1) + shift) % input.size();

        output(indices) = input;

        return output;

    } else if constexpr (is_eigen_matrix<Derived>::value) {
        int shift_row = std::get<0>(std::tuple<Args...>(args...));
        int shift_col = std::get<1>(std::tuple<Args...>(args...));

        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> output(input.rows(), input.cols());
        Eigen::ArrayXi row_indices = (Eigen::ArrayXi::LinSpaced(input.rows(), 0, input.rows() - 1) + shift_row) % input.rows();
        Eigen::ArrayXi col_indices = (Eigen::ArrayXi::LinSpaced(input.cols(), 0, input.cols() - 1) + shift_col) % input.cols();

        output(row_indices, col_indices) = input;

        return output;

    } else {
        static_assert(is_eigen_vector<Derived>::value || is_eigen_matrix<Derived>::value,
                      "Input type must be either an Eigen vector or matrix.");
    }
}

template <typename Derived>
auto meshgrid(const Eigen::DenseBase<Derived>& x, const Eigen::DenseBase<Derived>& y) {
    using Scalar = typename Derived::Scalar;
    // Get the number of x and y coordinates
    const int nx = x.size(), ny = y.size();

    // initialize the output matrix (column-major) with size [nx * ny, 2]
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> xy(nx * ny, 2);
    // map the first column of xy to xx (reshaped as [ny, nx])
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> xx(xy.data(), ny, nx);
    // map the second column of xy to yy (reshaped as [ny, nx])
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> yy(xy.data() + xy.rows(), ny, nx);

    // fill xx: each row of xx is a transposed copy of the x vector
    for (int i = 0; i < ny; ++i) {
        xx.row(i) = x.transpose();
    }

    // fill yy: each column of yy is a copy of the y vector
    for (int j = 0; j < nx; ++j) {
        yy.col(j) = y;
    }

    return xy;  // return the combined x-y meshgrid matrix
}

// find the maximum within a radius around the center of an array
std::pair<int, int> find_max_in_radius(const Eigen::MatrixXd& matrix, double radius) {
    // find the center of the matrix
    int n_rows = matrix.rows();
    int n_cols = matrix.cols();
    double center_x = (n_cols - 1) / 2.0;
    double center_y = (n_rows - 1) / 2.0;

    // initialize variables to store the max value and its coordinates
    double max_value = std::numeric_limits<double>::lowest();
    std::pair<int, int> max_coord = {-1, -1};

    // loop through the matrix to find the max value within the radius
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            double distance = std::sqrt(std::pow(i - center_y, 2) + std::pow(j - center_x, 2));
            if (distance <= radius && matrix(i, j) > max_value) {
                max_value = matrix(i, j);
                max_coord = {i, j};
            }
        }
    }

    return max_coord;
}

// define a box around an array element that is within array limits
std::tuple<int, int> find_bounding_limits(int pix_coord, int bounding_size, int array_size) {
    int half_size = bounding_size / 2;
    int lower_bound = std::max(0, pix_coord - half_size);
    int upper_bound = std::min(array_size - 1, pix_coord + half_size - (bounding_size % 2 == 0 ? 1 : 0));

    return std::make_tuple(lower_bound, upper_bound);
}

// create FIR filter based on the sampling frequency
auto create_kaiser_filter(const double data_fs_hz, const int filter_order, const double gibbs_factor,
                          const double low_cutoff_Hz, const double high_cutoff_Hz) {

    // calculate Nyquist frequency (half of the sampling frequency)
    double nyquist_frequency = data_fs_hz / 2.0;

    // normalize cutoff frequencies relative to the Nyquist frequency
    double normalized_low = low_cutoff_Hz / nyquist_frequency;
    double normalized_high = high_cutoff_Hz / nyquist_frequency;

    // adjust the filter if the high cutoff is lower than the low cutoff
    double stop_band_adjustment = (normalized_high < normalized_low) ? 1.0 : 0.0;

    // calculate the alpha parameter for the Kaiser window based on the Gibbs factor
    double alpha;
    if (gibbs_factor < 21.0) {
        alpha = 0.0;
    } else if (gibbs_factor > 50.0) {
        alpha = 0.1102 * (gibbs_factor - 8.7);
    } else {
        alpha = 0.5842 * std::pow(gibbs_factor - 21.0, 0.4) + 0.07886 * (gibbs_factor - 21.0);
    }

    // generate the arguments for the Bessel function, which are used in the Kaiser window
    Eigen::VectorXd bessel_arg = Eigen::VectorXd::LinSpaced(filter_order, 1, filter_order);
    bessel_arg = alpha * (1.0 - (bessel_arg / filter_order).cwiseAbs2().array()).sqrt();

    // normalize the Bessel function using the zeroth order modified Bessel function
    double bessel_norm_factor = gsl_sf_bessel_In(0, alpha);

    // calculate Kaiser window coefficients using the normalized Bessel function
    Eigen::VectorXd coefficients = bessel_arg.unaryExpr([bessel_norm_factor](double x) {
        return gsl_sf_bessel_In(0, x) / bessel_norm_factor;
    });

    // create a time vector to calculate the sinc function for the FIR filter
    Eigen::VectorXd time_vector = Eigen::VectorXd::LinSpaced(filter_order, 1, filter_order) * pi;

    // calculate the sinc function and apply the windowing function (Kaiser window)
    coefficients = coefficients.array() * (sin(time_vector.array() * normalized_high) -
                                           sin(time_vector.array() * normalized_low)) / time_vector.array();

    // populate the filter vector by mirroring the coefficients around the center
    Eigen::VectorXd filter(2 * filter_order + 1);
    filter.setZero();
    filter.head(filter_order) = coefficients.reverse();
    filter(filter_order) = normalized_high - normalized_low - stop_band_adjustment;
    filter.tail(filter_order) = coefficients;

    return filter;
}

// apply FIR filter
template <typename DerivedA, typename DerivedB>
void convolve_filter(Eigen::DenseBase<DerivedA> &data, Eigen::DenseBase<DerivedB> &filter) {
    int filter_order = (filter.size() - 1) / 2;

    // define the axis along which the convolution will be performed (0 = time axis)
    Eigen::array<ptrdiff_t, 1> convolution_axis{0};

    // map the Eigen matrices to Eigen::Tensor objects to leverage Eigen's tensor convolution capabilities
    Eigen::TensorMap<Eigen::Tensor<double, 2>> data_tensor(data.derived().data(), data.rows(), data.cols());
    Eigen::TensorMap<Eigen::Tensor<double, 1>> filter_tensor(filter.derived().data(), filter.size());

    // perform the convolution operation along the time axis
    Eigen::Tensor<double, 2> convolved_data(data_tensor.dimension(0) - filter_tensor.dimension(0) + 1, data.cols());
    convolved_data = data_tensor.convolve(filter_tensor, convolution_axis);

    // update the original data matrix with the filtered result
    // the first and last filter_order samples are not overwritten
    data.block(filter_order, 0, convolved_data.dimension(0), data.cols()) =
        Eigen::Map<Eigen::MatrixXd>(convolved_data.data(), convolved_data.dimension(0), convolved_data.dimension(1));
}

// check if a point is within a convex hull polygon
bool is_inside_convex_hull(const std::vector<std::pair<double, double>>& hull, double px, double py) {
    int n = hull.size();

    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        double x1 = hull[i].first, y1 = hull[i].second;
        double x2 = hull[j].first, y2 = hull[j].second;

        // check if point is inside the polygon using ray-casting algorithm
        if (((y1 > py) != (y2 > py)) && (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1)) {
            inside = !inside;
        }
    }
    return inside;
}

// find index of element nearest to some factor times the max value
template <typename Derived>
typename Derived::Scalar  threshold(Eigen::DenseBase<Derived>&data, typename Derived::Scalar value, typename Derived::Scalar lower) {
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> mask = (data > lower);
    Eigen::Array<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> filtered_data
        = data.array() * mask.cast<typename Derived::Scalar>();

    return (filtered_data - value * filtered_data.maxCoeff()).abs().minCoeff();
}

