# pragma once

#include <Eigen/Core>

// pi from eigen
constexpr auto pi = static_cast<double>(EIGEN_PI);

// astropy.constants planck constant (J x s)
#define h_J_s 6.62607015e-34

// astropy.constants speed of light (m/s)
#define c_m_s 299792458.0

// astropy constants k_B (J/K)
#define kB_J_K 1.380649e-23

// CMB Temperature (K)
#define T_cmb_K 2.7255

// degrees to arcsecs
#define DEG_TO_ASEC 3600.0

// arcsec to degrees
#define ASEC_TO_DEG 1/3600.0

// degrees to radians
#define DEG_TO_RAD (pi/180.)

// radians to degrees
#define RAD_TO_DEG 1./(pi/180.)

// arcsec to radians
#define ASEC_TO_RAD pi/180./3600.

// radians to arcsec
#define RAD_TO_ASEC 1./(pi/180./3600.)

// degrees to steradians
#define DEG_TO_SR 1./pow(180./pi,2.)

// steradians to degrees
#define SR_TO_DEG pow(180./pi,2.)

// standard deviation to fwhm
#define STD_TO_FWHM sqrt(8.*log(2.))

// fwhm to standard deviation
#define FWHM_TO_STD 1/sqrt(8.*log(2.))

// MJy/sr to mJy/arcsec
#define MJY_SR_TO_mJY_ASEC (1./SR_TO_DEG)/(pow(3600.0,2.0))*pow(10.0,6.0)*1000

// mJy/arcsec to MJy/sr
#define mJY_ASEC_to_MJY_SR (1./DEG_TO_SR)/(1/pow(3600.0,2.0))/pow(10.0,6.0)/1000
