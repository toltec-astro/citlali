#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/map_grouping_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <citlali/core/engine/detail/beammap_setup_state_impl.h>
#include <citlali/core/engine/detail/beammap_setup_metadata_impl.h>
#include <citlali/core/engine/detail/beammap_setup_diagnostics_impl.h>
#include <citlali/core/engine/detail/beammap_setup_soft_prior_impl.h>

void Beammap::setup() {
    // assign parallel policies
    map_parallel_policy = citlali::pipeline::runtime_parallel_policy_name(*this);

    // run obsnum setup
    obsnum_setup();

    setup_beammap_kids_tone_column();
    resize_beammap_state_buffers();
    populate_beammap_setup_metadata();
    init_beammap_diagnostic_apt_columns();
    init_beammap_flag_metadata();
    configure_beammap_soft_prior_setup();
}
