#pragma once

// Beammap output target selection helpers.

#include <string>
#include <vector>

namespace beammap_output_targets {

using FitsOutput = fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>;

struct Targets {
    mapmaking::MapBuffer *mb;
    std::vector<FitsOutput> *f_io;
    std::vector<FitsOutput> *n_io;
    std::string dir_name;
};

template <mapmaking::MapType map_type, class BeammapState>
Targets targets(BeammapState &beammap) {
    if constexpr (map_type == mapmaking::RawObs) {
        return {&beammap.omb, &beammap.fits_io_vec,
                &beammap.noise_fits_io_vec,
                beammap.obsnum_dir_name + "raw/"};
    }
    else if constexpr (map_type == mapmaking::FilteredObs) {
        return {&beammap.omb, &beammap.filtered_fits_io_vec,
                &beammap.filtered_noise_fits_io_vec,
                beammap.obsnum_dir_name + "filtered/"};
    }
    else if constexpr (map_type == mapmaking::RawCoadd) {
        return {&beammap.cmb, &beammap.coadd_fits_io_vec,
                &beammap.coadd_noise_fits_io_vec,
                beammap.coadd_dir_name + "raw/"};
    }
    else if constexpr (map_type == mapmaking::FilteredCoadd) {
        return {&beammap.cmb, &beammap.filtered_coadd_fits_io_vec,
                &beammap.filtered_coadd_noise_fits_io_vec,
                beammap.coadd_dir_name + "filtered/"};
    }
    else {
        static_assert(map_type == mapmaking::RawObs ||
                      map_type == mapmaking::FilteredObs ||
                      map_type == mapmaking::RawCoadd ||
                      map_type == mapmaking::FilteredCoadd,
                      "unsupported beammap output map type");
    }
}

template <class BeammapState>
void write_raw_obs_outputs(BeammapState &beammap,
                           const Targets &output_targets) {
    beammap.write_stats();
    output_targets.mb->calc_median_err();

    if (beammap.run_tod_output && !beammap.tod_filename.empty()) {
        beammap.add_tod_header(output_targets.mb);
    }

    beammap.write_detector_table_outputs();
}

} // namespace beammap_output_targets
