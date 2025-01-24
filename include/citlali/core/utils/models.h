#pragma once

namespace citlali::utils::models {

// define the 1D Polynomial model
template <int Degree>
struct Polynomial1DModel {
    static constexpr int nparams = Degree + 1;  // degree + 1 parameters for the polynomial

    template <typename T>
    static T eval(const T& x, const T* const params) {
        T result = T(0);
        T x_pow = T(1);
        for (int j = 0; j < nparams; ++j) {
            result += params[j] * x_pow;
            x_pow *= x;
        }
        return result;
    }
};

// define the 2D Gaussian model
struct Gaussian2DModel {
    static constexpr int nparams = 6;  // A, x0, y0, sigma_x, sigma_y, theta

    template <typename T>
    static T eval(const T& x, const T& y, const T* const params) {
        // extract parameters
        const T A = params[0];  // amplitude
        const T x0 = params[1]; // center x
        const T y0 = params[2]; // center y
        const T sigma_x = params[3]; // standard deviation in x
        const T sigma_y = params[4]; // standard deviation in y
        const T theta = params[5]; // rotation angle

        const T cos_theta = ceres::cos(theta);
        const T sin_theta = ceres::sin(theta);

        const T x_rot = cos_theta * (x - x0) + sin_theta * (y - y0);
        const T y_rot = -sin_theta * (x - x0) + cos_theta * (y - y0);

        // compute Gaussian value
        return A * ceres::exp(-0.5 * ((x_rot * x_rot) / (sigma_x * sigma_x) + (y_rot * y_rot) / (sigma_y * sigma_y)));
    }
};

// define the 2D Airy pattern model
struct Airy2DModel {
    static constexpr int nparams = 4;  // A, x0, y0, radius

    template <typename T>
    static T eval(const T& x, const T& y, const T* const params) {
        // extract parameters
        const T A = params[0];  // amplitude
        const T x0 = params[1]; // center x
        const T y0 = params[2]; // center y
        const T radius = params[3];  // radius of the Airy disk

        const T r = ceres::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0));
        const T r_norm = pi * r / radius;
        const T bessel_value = ceres::BesselJ1(r_norm);

        T airy_value = T(0.0);
        if (r_norm != T(0.0)) {
            airy_value = A * (2.0 * bessel_value / r_norm) * (2.0 * bessel_value / r_norm);
        } else {
            airy_value = A;
        }

        return airy_value;
    }
};

// define the 2D Moffat model
struct Moffat2DModel {
    static constexpr int nparams = 5;  // A, x0, y0, alpha, beta

    template <typename T>
    static T eval(const T& x, const T& y, const T* const params) {
        // Extract parameters
        const T A = params[0];    // amplitude
        const T x0 = params[1];   // center x
        const T y0 = params[2];   // center y
        const T alpha = params[3]; // scale parameter
        const T beta = params[4];  // shape parameter

        const T r_squared = (x - x0) * (x - x0) + (y - y0) * (y - y0);

        return A * ceres::pow(1.0 + r_squared / (alpha * alpha), -beta);
    }
};
} // namespace
