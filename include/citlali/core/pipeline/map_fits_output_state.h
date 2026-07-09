#pragma once

#include <vector>

namespace citlali::pipeline {

template <class FitsIo>
struct MapFitsOutputState {
    using map_fits_io_t = std::vector<FitsIo>;

    map_fits_io_t obs;
    map_fits_io_t obs_noise;
    map_fits_io_t filtered_obs;
    map_fits_io_t filtered_obs_noise;

    map_fits_io_t coadd;
    map_fits_io_t coadd_noise;
    map_fits_io_t filtered_coadd;
    map_fits_io_t filtered_coadd_noise;
};

}  // namespace citlali::pipeline
