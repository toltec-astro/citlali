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

struct MapdiagEdgeGuardDoubleRefs {
    std::vector<double> &weight_thresholds;
    std::vector<double> &hits_thresholds;
    std::vector<double> &background_levels;
    std::vector<double> &science_frac;
    std::vector<double> &support_frac;
    std::vector<double> &guardband_rms_pre;
    std::vector<double> &guardband_rms_post;
    std::vector<double> &exterior_rms_pre;
    std::vector<double> &exterior_rms_post;
    std::vector<double> &exterior_max_abs_pre;
    std::vector<double> &exterior_max_abs_post;
};

}  // namespace citlali::pipeline
