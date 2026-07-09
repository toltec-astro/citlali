#pragma once

#include <map>
#include <string>

struct BeammapFluxState {
    // source fluxes used for beammap calibration and output metadata
    std::map<std::string, double> source_flux_mJy_beam;
    std::map<std::string, double> source_flux_MJy_Sr;
};
