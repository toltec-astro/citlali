#pragma once

#include <Eigen/Core>

// pi from eigen
constexpr auto pi = static_cast<double>(EIGEN_PI);

// arcseconds in 360 degrees
#define ASEC_CIRC 1296000.0

// rad per arcsecond
#define RAD_ASEC (2.0*pi / ASEC_CIRC)

// degrees to arcsecs
#define DEG_TO_ASEC 3600.0

// degrees to radians
#define DEG_TO_RAD (DEG_TO_ASEC*RAD_ASEC)

// standard deviation to fwhm
#define STD_TO_FWHM sqrt(8.*log(2.))

// 1.1 mm freq
#define A1100_FREQ 3*pow(10,8)/(1.1/1000)

// 1.4 mm freq
#define A1400_FREQ 3*pow(10,8)/(1.4/1000)

// 2.0 mm freq
#define A2000_FREQ 3*pow(10,8)/(2.0/1000)

// MJy/sr to mJy/arcsec
#define MJY_SR_TO_mJY_ASEC (1/3282.8)*(1/pow(3600.0,2.0))*pow(10.0,6.0)*1000
