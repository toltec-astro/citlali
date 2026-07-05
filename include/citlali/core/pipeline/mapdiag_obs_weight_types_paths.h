#pragma once

// Included by mapdiag_observation_weight.h inside namespace citlali::pipeline.

struct MapdiagObsWeightTotals {
    double weight;
    double core_weight;
};

struct MapdiagObsTableRefs {
    std::vector<double> &weight_sum;
    std::vector<double> &core_weight_sum;
    std::vector<int> &valid_pixels;
    std::vector<int> &core_pixels;
};

template <class Context>
bool mapdiag_is_single_observation_context(const Context &context) {
    return !context.is_coadd;
}

inline std::string mapdiag_obs_raw_dir(const std::string &redu_dir_name,
                                       const std::string &obsnum) {
    return redu_dir_name + "/" + obsnum + "/raw/";
}

