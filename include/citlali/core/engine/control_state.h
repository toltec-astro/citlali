#pragma once

#include <map>
#include <string>

struct BeammapFluxState {
    // source fluxes used for beammap calibration and output metadata
    std::map<std::string, double> beammap_fluxes_mJy_beam;
    std::map<std::string, double> beammap_fluxes_MJy_Sr;
};

using BeammapControls = BeammapFluxState;
using beammapControls = BeammapControls;
