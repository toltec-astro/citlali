#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_output_targets.h>

template <mapmaking::MapType map_type>
void Beammap::output(
    citlali::pipeline::StageProfileCollector &stage_profile) {
    auto output_targets =
        beammap_output_targets::targets<map_type>(*this);

    // raw obs maps
    if constexpr (map_type == mapmaking::RawObs) {
        beammap_output_targets::write_raw_obs_outputs(
            *this, output_targets);
    }

    write_beammap_map_products<map_type>(
        output_targets.mb, output_targets.f_io, output_targets.n_io,
        stage_profile, output_targets.dir_name);
}
