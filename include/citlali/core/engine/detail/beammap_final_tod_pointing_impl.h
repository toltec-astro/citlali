#pragma once

// Beammap final TOD pointing update implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_apt_keys.h>
#include <citlali/core/pipeline/output_policy.h>

#include <vector>

void Beammap::update_beammap_final_tod_pointing(
    const std::string &map_parallel_policy,
    citlali::config::MapGrouping mapmaking_grouping) {
    if (!citlali::pipeline::tod_output_files_available(*this)) {
        return;
    }

    // vectors to hold tangent plane pointing for all ptcs (n_chunks x [n_pts x n_dets])
    std::vector<Eigen::MatrixXd> lat, lon;

    // recalculate tangent plane pointing for tod output
    for (Eigen::Index i=0; i<ptcs.size(); ++i) {
        // tangent plane pointing for each detector
        Eigen::MatrixXd ptc_lat(ptcs[i].scans.data.rows(), ptcs[i].scans.data.cols());
        Eigen::MatrixXd ptc_lon(ptcs[i].scans.data.rows(), ptcs[i].scans.data.cols());
        // loop through detectors
        grppi::map(tula::grppi_utils::dyn_ex(map_parallel_policy), det_in_vec, det_out_vec, [&](auto j) {
            // det indices
            auto det_index = j;
            double az_off = calib.apt["x_t"](det_index);
            double el_off = calib.apt["y_t"](det_index);

            // get tangent pointing
            auto [det_lat, det_lon] = engine_utils::calc_det_pointing(ptcs[i].tel_data.data, az_off,
                                                                      el_off, telescope.pixel_axes,
                                                                      ptcs[i].pointing_offsets_arcsec.data,
                                                                      mapmaking_grouping, true);
            ptc_lat.col(j) = std::move(det_lat);
            ptc_lon.col(j) = std::move(det_lon);

            return 0;
        });
        lat.push_back(std::move(ptc_lat));
        lon.push_back(std::move(ptc_lon));
    }

    logger->info("adding final apt and detector pointing to tod files");
    // loop through tod files
    for (const auto & [key, val]: output_paths.tod_filename) {
        netCDF::NcFile fo(val, netCDF::NcFile::write);
        // overwrite apt table
        for (auto const& x: calib.apt) {
            if (!beammap_apt_keys::is_flag2(x.first)) {
                // start index for apt table
                std::vector<std::size_t> start_index_apt = {0};
                // size for apt
                std::vector<std::size_t> size_apt = {1};
                netCDF::NcVar apt_v = fo.getVar("apt_" + x.first);
                if (!apt_v.isNull()) {
                    for (std::size_t i=0; i< TULA_SIZET(calib.n_dets); ++i) {
                        start_index_apt[0] = i;
                        apt_v.putVar(start_index_apt, size_apt, &calib.apt[x.first](i));
                    }
                }
            }
        }

        // detector tangent plane pointing
        netCDF::NcVar det_lat_v = fo.getVar("det_lat");
        netCDF::NcVar det_lon_v = fo.getVar("det_lon");

        // detector absolute pointing
        netCDF::NcVar det_ra_v = fo.getVar("det_ra");
        netCDF::NcVar det_dec_v = fo.getVar("det_dec");
        const bool write_tangent_pointing = !det_lat_v.isNull() && !det_lon_v.isNull();
        const bool write_abs_pointing =
            citlali::config::is_radec_map_pixel_axes(
                telescope.pixel_axes) &&
            !det_ra_v.isNull() && !det_dec_v.isNull();
        if (!write_tangent_pointing && !write_abs_pointing) {
            logger->debug("tod file {} has no detector pointing variables; skipping final detector pointing update", val);
            continue;
        }

        // start indices for data
        std::vector<std::size_t> start_index = {0, 0};
        // size for data
        std::vector<std::size_t> size = {1, TULA_SIZET(calib.n_dets)};
        std::size_t k = 0;
        // loop through ptcs
        for (Eigen::Index i=0; i<lat.size(); ++i) {
            // loop through n_pts
            for (std::size_t j=0; j < TULA_SIZET(lat[i].rows()); ++j) {
                start_index[0] = k;
                k++;
                // append detector latitudes
                Eigen::VectorXd lat_row = lat[i].row(j);

                // append detector longitudes
                Eigen::VectorXd lon_row = lon[i].row(j);
                if (write_tangent_pointing) {
                    det_lat_v.putVar(start_index, size, lat_row.data());
                    det_lon_v.putVar(start_index, size, lon_row.data());
                }

                if (write_abs_pointing) {
                    // get absolute pointing
                    auto [dec, ra] = engine_utils::tangent_to_abs(lat_row, lon_row, telescope.tel_header["Header.Source.Ra"](0),
                                                                  telescope.tel_header["Header.Source.Dec"](0));
                    // append detector ra
                    det_ra_v.putVar(start_index, size, ra.data());

                    // append detector dec
                    det_dec_v.putVar(start_index, size, dec.data());
                }
            }
        }
    }

    // empty ptcdata vector to save memory
    ptcs.clear();
}
