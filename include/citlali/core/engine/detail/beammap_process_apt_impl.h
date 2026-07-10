#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <citlali/core/engine/detail/beammap_apt_derotation_impl.h>
#include <citlali/core/engine/detail/beammap_reference_selection_impl.h>

void Beammap::process_apt() {
    const auto &reference_config =
        citlali::pipeline::beammap_config(*this).reference;

    // reference detector x and y
    double ref_det_x_t = 0;
    double ref_det_y_t = 0;

    select_beammap_reference_detector(ref_det_x_t, ref_det_y_t);
    record_beammap_reference_metadata(ref_det_x_t, ref_det_y_t);
    preserve_beammap_raw_detector_offsets();
    populate_beammap_derotation_elevation();
    apply_beammap_reference_offsets(ref_det_x_t, ref_det_y_t);
    apply_beammap_derotation(reference_config.derotate);
}
