#pragma once

#include <vector>

namespace citlali::pipeline {

struct MapdiagEdgeGuardIntRefs {
    std::vector<int> &applied;
    std::vector<int> &support_radius_pix;
    std::vector<int> &science_npix;
    std::vector<int> &support_npix;
    std::vector<int> &guardband_npix;
};

}  // namespace citlali::pipeline
