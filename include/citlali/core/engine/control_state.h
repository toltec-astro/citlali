#pragma once

#include <map>
#include <string>

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

struct BeammapFluxState {
    // source fluxes used for beammap calibration and output metadata
    std::map<std::string, double> beammap_fluxes_mJy_beam;
    std::map<std::string, double> beammap_fluxes_MJy_Sr;
};

using reduControls = ReductionControls;
using BeammapControls = BeammapFluxState;
using beammapControls = BeammapControls;
