#pragma once

// Beammap APT table output implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_apt_table_output_helpers.h>
#include <citlali/core/pipeline/observation_map_files.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

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

    Eigen::MatrixXd apt_table =
        beammap_apt_table_output_helpers::apt_table(calib, flag2);

    to_ecsv_from_matrix(
        apt_filename, apt_table, calib.apt_header_keys, calib.apt_meta);

    logger->info("done writing apt table {}.ecsv", apt_filename);
    return apt_filename;
}
