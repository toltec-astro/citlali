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

    if constexpr (map_type == mapmaking::RawObs) {
        if (citlali::pipeline::beammap_direction_mode_is_all(
                citlali::pipeline::beammap_config(*this).direction_mode)) {
            if (!beammap_direction_products.left_product.has_value() ||
                !beammap_direction_products.right_product.has_value()) {
                throw std::logic_error(
                    "beammap direction_mode=all output lacks finalized directional products");
            }
            write_beammap_directional_raw_product(
                *beammap_direction_products.left_product,
                beammap_direction_products.left, stage_profile);
            write_beammap_directional_raw_product(
                *beammap_direction_products.right_product,
                beammap_direction_products.right, stage_profile);
        }
    }
    else if constexpr (map_type == mapmaking::FilteredObs) {
        if (citlali::pipeline::beammap_direction_mode_is_all(
                citlali::pipeline::beammap_config(*this).direction_mode)) {
            if (!beammap_direction_products.left_product.has_value() ||
                !beammap_direction_products.right_product.has_value()) {
                throw std::logic_error(
                    "beammap direction_mode=all filtered output lacks finalized directional products");
            }
            write_beammap_directional_filtered_product(
                *beammap_direction_products.left_product,
                beammap_direction_products.left, stage_profile);
            write_beammap_directional_filtered_product(
                *beammap_direction_products.right_product,
                beammap_direction_products.right, stage_profile);
        }
    }
}
