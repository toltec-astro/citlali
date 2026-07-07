#pragma once

// Beammap APT table output implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_apt_table_output_helpers.h>

std::string Beammap::write_beammap_apt_table() {
    logger->info("writing apt table");
    auto apt_filename =
        toltec_io
            .create_filename<engine_utils::toltecIO::apt,
                             engine_utils::toltecIO::map,
                             engine_utils::toltecIO::raw>(
                obsnum_dir_name + "raw/", redu_type, "", obsnum,
                telescope.sim_obs);

    Eigen::MatrixXd apt_table =
        beammap_apt_table_output_helpers::apt_table(calib, flag2);

    to_ecsv_from_matrix(
        apt_filename, apt_table, calib.apt_header_keys, calib.apt_meta);

    logger->info("done writing apt table {}.ecsv", apt_filename);
    return apt_filename;
}
