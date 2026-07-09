#pragma once

#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>

struct ReductionControls {
    // interpolate over gaps in timestreams
    bool interp_over_gaps;
    // create reduction subdirectories
    bool use_subdir;

    // run or skip tod processing
    bool run_tod;

    // output timestreams
    bool run_tod_output;

    // controls for mapmaking
    bool run_mapmaking;
    bool run_coadd;
    bool run_noise;
    bool write_noise_realizations;
    bool run_noise_products;
    bool apply_empirical_noise_weights;
    bool run_map_filter;

    // run source finding
    bool run_source_finder;
};

struct BeammapControls {
    // source name
    std::string beammap_source_name;

    // beammap source position
    double beammap_ra_rad, beammap_dec_rad;

    // fluxes and errs
    std::map<std::string, double> beammap_fluxes_mJy_beam, beammap_err_mJy_beam;
    std::map<std::string, double> beammap_fluxes_MJy_Sr, beammap_err_MJy_Sr;

    // iteration to write out beammap PTC data; -1 means final attempted iteration
    int beammap_tod_output_iter = -1;

    // upper and lower limits of psd for sensitivity calc
    Eigen::VectorXd sens_psd_limits_Hz;

    // limits on fwhm, sig2noise, and distance from center for flagging
    std::map<std::string, double> lower_fwhm_arcsec, upper_fwhm_arcsec, lower_sig2noise,
        upper_sig2noise, max_dist_arcsec, network_robust_z;

    // limits on sensitivity for flagging
    double lower_sens_factor, upper_sens_factor;
};

struct PointingControls {
    // source-aware pointing strategy.  Gaussian fits are optional diagnostics;
    // fruit loops remains empirical and uses previous maps.
    std::string pointing_source_strategy = "standard";
    bool pointing_fit_gaussian_enabled = true;
    std::string pointing_fruitloops_center_mode = "auto";
    double pointing_header_center_max_radius_arcsec = 0.0;
    bool pointing_header_center_require_coverage = true;
};

using reduControls = ReductionControls;
using beammapControls = BeammapControls;
using pointingControls = PointingControls;
