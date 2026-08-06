#pragma once

// Engine post-processing implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/post_processing_provenance_lifecycle.h>

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_sources(map_buffer_t &mb, std::string dir_name) {
    // get filename for source table
    const std::string source_filename =
        setup_filenames<map_t, engine_utils::toltecIO::source,
                        engine_utils::toltecIO::map>(dir_name);

    const auto map_to_array_index = [&](Eigen::Index map_index) {
        return map_indices.maps_to_arrays(map_index);
    };
    const auto calc_map_std_dev = [](auto &signal) {
        return engine_utils::calc_std_dev(signal);
    };
    const auto write_source_table =
        [&](const std::string &filename, auto &source_table,
            auto source_header, auto source_meta) {
            to_ecsv_from_matrix(
                filename, source_table, source_header, source_meta);
        };
    const auto source_table_callbacks =
        citlali::pipeline::make_source_table_callbacks(
            map_to_array_index, calc_map_std_dev, write_source_table);
    citlali::pipeline::write_source_table_output(
        source_filename, *mb, map_fitter.n_params,
        citlali::pipeline::mapmaking_config(*this).pixel_axes_frame,
        telescope.source_name, engine_utils::current_date_time(),
        citlali::pipeline::latest_observation_date(observation_dates),
        calib.apt_header_description,
        source_table_callbacks);
    citlali::pipeline::record_noise_published_member(
        citlali::pipeline::noise_plan(*this), source_filename + ".ecsv",
        citlali::pipeline::NoisePublishedMemberKind::ecsv);

    if constexpr (map_t == mapmaking::FilteredObs ||
                  map_t == mapmaking::FilteredCoadd) {
        constexpr auto context =
            map_t == mapmaking::FilteredObs
                ? citlali::pipeline::PostProcessingMapContext::observation
                : citlali::pipeline::PostProcessingMapContext::coadd;
        citlali::pipeline::record_post_processing_source_table_written(
            citlali::pipeline::post_processing_plan(*this), context,
            static_cast<std::size_t>(mb->source_params.rows()));
    }
}
