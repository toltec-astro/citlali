#pragma once

// Beammap APT table output implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_apt_table_output_helpers.h>
#include <citlali/core/pipeline/observation_map_files.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/runtime_provenance_output.h>
#include <citlali/core/utils/sha256.h>

std::string Beammap::write_beammap_apt_table() {
    logger->info("writing apt table");
    auto apt_filename =
        citlali::pipeline::observation_output_filename<
            engine_utils::toltecIO::apt, engine_utils::toltecIO::map,
            engine_utils::toltecIO::raw>(
            toltec_io, output_paths.obsnum_dir_name + "raw/",
            citlali::pipeline::runtime_reduction_type(*this), "",
            observation_identity.obsnum,
            telescope.sim_obs);

    beammap_apt_table_output_helpers::CanonicalAptDocumentContext context;
    beammap_apt_table_output_helpers::inject_issuance_context(
        context, calib.canonical_apt_producer);
    context.software_revision = CITLALI_GIT_VERSION;
    const auto runtime_provenance =
        citlali::pipeline::runtime_provenance_path(
            output_paths.redu_dir_name);
    context.configuration_reference =
        "runtime-provenance:sha256:" +
        citlali::utils::sha256_file(runtime_provenance);
    context.event_time_utc =
        beammap_apt_table_output_helpers::current_utc_timestamp();
    context.project_id = telescope.project_id;
    context.source_name = telescope.source_name;
    context.observation_time_utc =
        beammap_apt_table_output_helpers::utc_timestamp_from_unix_seconds(
            beammap_apt_table_output_helpers::
                telescope_observation_unix_time(telescope.tel_data));
    context.coordinate_frame = telescope.pixel_axes;

    const auto document =
        beammap_apt_table_output_helpers::make_canonical_document(
            calib, flag2, context);
    const auto output_path =
        std::filesystem::path(apt_filename + ".ecsv");
    const auto publication =
        beammap_apt_table_output_helpers::publish_canonical_apt(
            document, output_path);

    logger->info(
        "done writing canonical apt table {} receipt={} semantic={} envelope={} transport={}",
        publication.ecsv_path.string(), publication.receipt_path.string(),
        publication.digests.semantic_sha256,
        publication.digests.envelope_sha256,
        publication.transport.sha256);
    return apt_filename;
}
