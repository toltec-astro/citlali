# pragma once

#include <Eigen/Core>

// pi from eigen
constexpr double pi = static_cast<double>(EIGEN_PI);

// astropy.constants planck constant (J x s)
constexpr double h_J_s = 6.62607015e-34;
// astropy.constants speed of light (m/s)
constexpr double c_m_s = 299792458.0;
// astropy constants k_B (J/K)
constexpr double kB_J_K = 1.380649e-23;
// CMB Temperature (K)
constexpr double T_cmb_K = 2.7255;
// degrees to arcsecs
constexpr double DEG_TO_ASEC = 3600.0;
// arcsec to degrees
constexpr double ASEC_TO_DEG = 1 / 3600.0;
// degrees to radians
constexpr double DEG_TO_RAD = pi / 180.;
// radians to degrees
constexpr double RAD_TO_DEG = 1. / (pi / 180.);
// arcsec to radians
constexpr double ASEC_TO_RAD = pi / 180. / 3600.;
// radians to arcsec
constexpr double RAD_TO_ASEC = 1. / (pi / 180. / 3600.);
// degrees to steradians
constexpr double DEG_TO_SR = 1. / pow(180. / pi, 2.);
// steradians to degrees
constexpr double SR_TO_DEG = pow(180. / pi, 2.);
// standard deviation to fwhm
constexpr double STD_TO_FWHM = sqrt(8. * log(2.));
// fwhm to standard deviation
constexpr double FWHM_TO_STD = 1 / sqrt(8. * log(2.));
// MJy/sr to mJy/arcsec
constexpr double MJY_SR_TO_mJY_ASEC = (DEG_TO_SR) * (pow(3600.0, -2.0)) * pow(10.0, 6.0) * 1e3;
// mJy/arcsec to MJy/sr
constexpr double mJY_ASEC_to_MJY_SR = (SR_TO_DEG) * (pow(3600.0, 2.0)) * pow(10.0, -6.0) * 1e-3;

// planck function
double planck_nu(const double nu_Hz, const double T_K) {
    return 2. * h_J_s* std::pow(nu_Hz, 3) / std::pow(c_m_s, 2) / (std::exp((h_J_s * nu_Hz) / (kB_J_K * T_K)) - 1.);
}
