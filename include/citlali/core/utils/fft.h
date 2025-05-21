# pragma once

#include <fftw3.h>

namespace citlali::utils::fft {

// function to pad the matrix with zeros to the specified dimensions
Eigen::MatrixXd pad_matrix(const Eigen::MatrixXd& matrix, int n_rows, int n_cols) {
    Eigen::MatrixXd padded = Eigen::MatrixXd::Zero(n_rows, n_cols);
    padded.block(0, 0, matrix.rows(), matrix.cols()) = matrix;
    return padded;
}

// create a hanning window
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> hanning_window_1d(int n_pts) {
    Eigen::Array<Scalar, Eigen::Dynamic, 1> indices = Eigen::Array<Scalar, Eigen::Dynamic, 1>::LinSpaced(n_pts, 0, n_pts - 1);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> window =
        (0.5 * (1.0 - (2.0 * pi * indices / static_cast<Scalar>(n_pts - 1)).cos())).matrix();

    return window;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> hanning_window_2d(int n_rows, int n_cols) {
    double a = 2.0 * pi / n_rows;
    double b = 2.0 * pi / n_cols;

    // generate and calculate the row and column windows in a single step
    Eigen::Array<Scalar, Eigen::Dynamic, 1> row_window = (-0.5 * (Eigen::Array<Scalar, Eigen::Dynamic, 1>::LinSpaced(n_rows, 0, n_rows - 1) * a).cos() + 0.5);
    Eigen::Array<Scalar, Eigen::Dynamic, 1> col_window = (-0.5 * (Eigen::Array<Scalar, Eigen::Dynamic, 1>::LinSpaced(n_cols, 0, n_cols - 1) * b).cos() + 0.5);

    // create a matrix where each row is a copy of the row_window
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> row_matrix = row_window.replicate(1, n_cols);

    // create a matrix where each column is a copy of the col_window
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> col_matrix = col_window.replicate(n_rows, 1);

    // multiply row_matrix and col_matrix element-wise
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> window = row_matrix.cwiseProduct(col_matrix);

    return window;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> fftfreq(int n, double d) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> freqs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::LinSpaced(n, 0, n - 1);

    int half_n = (n + 1) / 2;

    freqs.head(half_n) = freqs.head(half_n) / (n * d);
    freqs.tail(n - half_n) = (freqs.tail(n - half_n).array() - n) / (n * d);

    return freqs;
}

// function to perform 2d convolution
Eigen::MatrixXd convolve_fft_2d(const Eigen::MatrixXd& data, const Eigen::MatrixXd& kernel,
                                fftw_plan forward_plan = nullptr, fftw_plan inverse_plan = nullptr) {

    int n_rows = data.rows() + kernel.rows() - 1;
    int n_cols = data.cols() + kernel.cols() - 1;

    // pad the input matrices
    Eigen::MatrixXcd padded_data(n_rows, n_cols);
    padded_data.real() = pad_matrix(data, n_rows, n_cols);
    padded_data.imag().setZero();

    // allocate memory for FFT results using Eigen matrices
    Eigen::MatrixXcd fft_data(n_rows, n_cols);
    // store fft of multiplication of data and kernel
    Eigen::MatrixXcd fft_convolved(n_rows, n_cols);
    // store final result
    Eigen::MatrixXcd convolved(n_rows, n_cols);

    // create plans if not input
    bool destroy_forward = false, destroy_inverse = false;
    if (forward_plan == nullptr) {
        forward_plan = fftw_plan_dft_2d(n_cols, n_rows, reinterpret_cast<fftw_complex*>(padded_data.data()),
                                        reinterpret_cast<fftw_complex*>(fft_data.data()), FFTW_FORWARD, FFTW_ESTIMATE);
        destroy_forward = true;
    }
    if (inverse_plan == nullptr) {
        inverse_plan = fftw_plan_dft_2d(n_cols, n_rows, reinterpret_cast<fftw_complex*>(fft_convolved.data()),
                                        reinterpret_cast<fftw_complex*>(convolved.data()), FFTW_BACKWARD, FFTW_ESTIMATE);
        destroy_inverse = true;
    }

    // perform fft on data
    fftw_execute(forward_plan);

    // copy fft of data
    fft_convolved = fft_data;

    // reuse padded data for kernel
    padded_data.real() = pad_matrix(kernel, n_rows, n_cols);
    padded_data.imag().setZero();

    // perform fft on kernel
    fftw_execute(forward_plan);

    // perform element-wise multiplication in frequency domain
    fft_convolved = fft_convolved.cwiseProduct(fft_data);

    // perform inverse fft to get the convolved result
    fftw_execute(inverse_plan);
    convolved /= (n_rows * n_cols);

    // remove zeros and get real component
    Eigen::MatrixXd result = convolved.block(0, 0, data.rows(), data.cols()).real();

    if (destroy_forward) {
        fftw_destroy_plan(forward_plan);
    }
    if (destroy_inverse) {
        fftw_destroy_plan(inverse_plan);
    }
    return result;
}

// function to compute the 1d psd using fftw and eigen, with a hanning window applied
std::pair<Eigen::VectorXd, Eigen::VectorXd> calc_psd_1d(const Eigen::VectorXd& signal, const double fs_hz,
                                                        fftw_plan plan = nullptr) {
    // length of the input signal
    int n_pts = signal.size();

    // create a hanning window and apply it to the signal
    Eigen::VectorXd window = hanning_window_1d<double>(n_pts);
    Eigen::VectorXd windowed_signal = signal.array() * window.array();

    // fftw output will be complex (n_pts/2+1 values due to symmetry)
    int fft_size = n_pts / 2 + 1;

    // allocate memory for fftw input and output
    Eigen::VectorXcd fft_output(fft_size);  // complex output for fft
    Eigen::VectorXd psd(fft_size);  // psd

    // create fftw plan
    bool destroy_plan = false;
    if (plan == nullptr) {
        plan = fftw_plan_dft_r2c_1d(static_cast<int>(n_pts), reinterpret_cast<double*>(windowed_signal.data()),
                                    reinterpret_cast<fftw_complex*>(fft_output.data()), FFTW_ESTIMATE);
    }

    // perform the fft
    fftw_execute(plan);

    // calculate the psd
    for (int i = 0; i < fft_size; ++i) {
        double magnitude = std::norm(fft_output[i]);  // magnitude squared of complex number
        psd(i) = (2.0 / (fs_hz * n_pts)) * magnitude;  // scaling factor for psd
    }

    // normalize the first and last element (nyquist frequency)
    psd(0) /= 2.0;
    if (n_pts % 2 == 0) {
        psd(fft_size - 1) /= 2.0;
    }

    // calculate the corresponding frequencies
    Eigen::VectorXd frequencies = Eigen::VectorXd::LinSpaced(fft_size, 0, (fft_size - 1) * fs_hz / n_pts);

    // destroy the FFTW plan
    if (destroy_plan) {
        fftw_destroy_plan(plan);
    }

    // return the psd and frequencies as a pair
    return {psd, frequencies};
}

// function to compute the welch psd using fftw and eigen, with a hanning window applied
// std::pair<Eigen::VectorXd, Eigen::VectorXd> welch(const Eigen::VectorXd& signal, const double fs_hz,
//                                                   const int n_perseg, fftw_plan plan = nullptr) {

// }


// function to compute 2D PSD
auto calc_psd_2d(const Eigen::MatrixXd& data, const double dy, const double dx, fftw_plan forward_plan = nullptr) {

    int n_rows = data.rows();
    int n_cols = data.cols();

    // allocate matrix for FFT input
    Eigen::MatrixXcd fft_input(n_rows, n_cols);
    fft_input.real() = data;
    fft_input.imag().setZero();

    // allocate memory for FFT output
    Eigen::MatrixXcd fft_output(n_rows, n_cols);

    // create a FFTW plan for the 2D FFT
    bool destroy_forward = false;
    if (forward_plan == nullptr) {
        forward_plan = fftw_plan_dft_2d(n_cols, n_rows, reinterpret_cast<fftw_complex*>(fft_input.data()),
                                        reinterpret_cast<fftw_complex*>(fft_output.data()), FFTW_FORWARD, FFTW_ESTIMATE);
        destroy_forward = true;
    }

    // execute forward fft
    fftw_execute(forward_plan);

    // compute the PSD using element-wise squared magnitude, throw away 0
    Eigen::MatrixXd psd = fft_output.block(1, 1, n_rows - 1, n_cols - 1).cwiseAbs2();

    // normalize the PSD by the total number of points
    psd /= (n_rows * n_cols);

    // get frequencies, throw away 0
    Eigen::VectorXd row_freqs = fftfreq<double>(n_rows, dx).tail(n_rows - 1);
    Eigen::VectorXd col_freqs = fftfreq<double>(n_cols, dy).tail(n_cols - 1);

    Eigen::MatrixXd freqs = (row_freqs.array().square().replicate(1, n_cols - 1) +
                             col_freqs.array().square().transpose().replicate(n_rows - 1, 1)).sqrt();

    int n_bins = std::max(n_rows, n_cols) / 2 + 1;
    double d = (n_rows > n_cols) ? dx : dy;

    Eigen::VectorXd radial_psd(n_bins);
    radial_psd.setZero();

    // frequencies for radially averaged psd offset by 0.5 for bin center
    Eigen::VectorXd radial_freqs = Eigen::VectorXd::LinSpaced(n_bins, d * 0.5, d * (n_bins - 1 + 0.5));

    for (int bin = 0; bin < n_bins; ++bin) {
        int n_pts = 0;
        for (int row = 0; row < n_rows - 1; ++row) {
            for (int col = 0; col < n_cols - 1; ++col) {
                if (int(freqs(row, col) / d) == bin) {
                    radial_psd(bin) += psd(row, col);
                    n_pts++;
                }
            }
        }
        radial_psd(bin) /= n_pts;
    }

    // destroy the FFTW plan
    if (destroy_forward) {
        fftw_destroy_plan(forward_plan);
    }

    return std::make_tuple(radial_psd, radial_freqs, psd, freqs);
}
} // namespace
