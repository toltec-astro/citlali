#pragma once

#include <Eigen/Core>

// pi from eigen
constexpr auto pi = static_cast<double>(EIGEN_PI);

// astropy.constants planck constant (J s)
#define h_J_s 6.62607015e-34

// astropy.constants speed of light (m/s)
#define c_m_s 299792458.0

// astropy constants k_B (J/K)
#define k_B_J_K 1.380649e-23

// CMB Temperature (K)
#define tcmb_K 2.7255

// arcseconds in 360 degrees
#define ASEC_CIRC 1296000.0

// degrees to arcsecs
#define DEG_TO_ASEC 3600.0

// rad per arcsecond
#define ASEC_TO_RAD (2.0*pi/ASEC_CIRC)

// degrees to radians
#define DEG_TO_RAD (DEG_TO_ASEC*ASEC_TO_RAD)

// standard deviation to fwhm
#define STD_TO_FWHM sqrt(8.*log(2.))

// 1.1 mm freq
#define A1100_FREQ c_m_s/(1.1/1000)

// 1.4 mm freq
#define A1400_FREQ c_m_s/(1.4/1000)

// 2.0 mm freq
#define A2000_FREQ c_m_s/(2.0/1000)

// MJy/Sr to mJy/arcsec
#define MJY_SR_TO_mJY_ASEC (1/3282.8)*(1/pow(3600.0,2.0))*pow(10.0,6.0)*1000
