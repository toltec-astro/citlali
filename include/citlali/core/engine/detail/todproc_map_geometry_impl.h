#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/config/calibration_config.h>
#include <citlali/core/pipeline/map_dimension_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/runtime_policy.h>

// calculate map dimensions
template <class EngineType>
void TimeOrderedDataProc<EngineType>::calc_omb_size(std::vector<map_extent_t> &map_extents, std::vector<map_coord_t> &map_coords) {

    // reference to map buffer
    auto& omb = engine().omb;
    const auto &mapmaking_settings =
        citlali::pipeline::mapmaking_config(engine());

    // only run if manual map sizes have not been input
    if ((engine().omb.wcs.naxis[0] <= 0) || (engine().omb.wcs.naxis[1] <= 0)) {
        // matrix to store size limits
        Eigen::MatrixXd det_lat_limits(engine().calib.n_dets, 2);
        Eigen::MatrixXd det_lon_limits(engine().calib.n_dets, 2);
        det_lat_limits.setZero();
        det_lon_limits.setZero();

        // placeholder vectors for grppi maps
        std::vector<int> det_in_vec, det_out_vec;

        // placeholder vectors for grppi loop
        det_in_vec.resize(engine().calib.n_dets);
        std::iota(det_in_vec.begin(), det_in_vec.end(), 0);
        det_out_vec.resize(engine().calib.n_dets);

        // get telescope meta data for current scan
        std::map<std::string, Eigen::VectorXd> tel_data;

        // pointing offsets
        std::map<std::string, Eigen::VectorXd> pointing_offsets_arcsec;

        // loop through scans
        for (Eigen::Index i=0; i<engine().telescope.scan_indices.cols(); ++i) {
            // lower scan index
            auto si = engine().telescope.scan_indices(0,i);
            // upper scan index
            auto sl = engine().telescope.scan_indices(1,i) - engine().telescope.scan_indices(0,i) + 1;

            for (auto const& x: engine().telescope.tel_data) {
                tel_data[x.first] = engine().telescope.tel_data[x.first].segment(si,sl);
            }

            // get pointing offsets for current scan
            pointing_offsets_arcsec[citlali::config::pointing_axis_az()] =
                engine().pointing_offsets.arcsec[
                    citlali::config::pointing_axis_az()].segment(si, sl);
            pointing_offsets_arcsec[citlali::config::pointing_axis_alt()] =
                engine().pointing_offsets.arcsec[
                    citlali::config::pointing_axis_alt()].segment(si, sl);

            // don't need to find the offsets if in detector mode
            if (mapmaking_settings.grouping !=
                citlali::config::MapGrouping::detector) {
                // loop through detectors
                grppi::map(
                    tula::grppi_utils::dyn_ex(
                        citlali::pipeline::runtime_parallel_policy_name(
                            engine())),
                    det_in_vec, det_out_vec, [&](auto j) {

                    // get pointing
                    auto [lat, lon] = engine_utils::calc_det_pointing(
                        tel_data, engine().calib.apt["x_t"](j),
                        engine().calib.apt["y_t"](j),
                        engine().telescope.pixel_axes, pointing_offsets_arcsec,
                        mapmaking_settings.grouping);
                    // check for min and max
                    if (engine().calib.apt["flag"](j)==0) {
                        if (lat.minCoeff() < det_lat_limits(j,0)) {
                            det_lat_limits(j,0) = lat.minCoeff();
                        }
                        if (lat.maxCoeff() > det_lat_limits(j,1)) {
                            det_lat_limits(j,1) = lat.maxCoeff();
                        }
                        if (lon.minCoeff() < det_lon_limits(j,0)) {
                            det_lon_limits(j,0) = lon.minCoeff();
                        }
                        if (lon.maxCoeff() > det_lon_limits(j,1)) {
                            det_lon_limits(j,1) = lon.maxCoeff();
                        }
                    }
                    return 0;
                });
            }
            else {
                // calculate detector pointing for first detector only since offsets are zero
                auto [lat, lon] = engine_utils::calc_det_pointing(
                    tel_data, 0., 0., engine().telescope.pixel_axes,
                    pointing_offsets_arcsec,
                    mapmaking_settings.grouping);
                if (lat.minCoeff() < det_lat_limits(0,0)) {
                    det_lat_limits.col(0).setConstant(lat.minCoeff());
                }
                if (lat.maxCoeff() > det_lat_limits(0,1)) {
                    det_lat_limits.col(1).setConstant(lat.maxCoeff());
                }
                if (lon.minCoeff() < det_lon_limits(0,0)) {
                    det_lon_limits.col(0).setConstant(lon.minCoeff());
                }
                if (lon.maxCoeff() > det_lon_limits(0,1)) {
                    det_lon_limits.col(1).setConstant(lon.maxCoeff());
                }
            }
        }

        // get the global min and max
        double min_lat = det_lat_limits.col(0).minCoeff();
        double max_lat = det_lat_limits.col(1).maxCoeff();
        double min_lon = det_lon_limits.col(0).minCoeff();
        double max_lon = det_lon_limits.col(1).maxCoeff();

        // get n_rows and n_cols
        omb.n_rows = citlali::pipeline::symmetric_odd_pixel_count(
            min_lat, max_lat, omb.pixel_size_rad);
        omb.n_cols = citlali::pipeline::symmetric_odd_pixel_count(
            min_lon, max_lon, omb.pixel_size_rad);
    }

    else {
        // Ensure odd dimensions
        omb.n_rows =
            citlali::pipeline::odd_dimension_from_config(omb.wcs.naxis[1]);
        omb.n_cols =
            citlali::pipeline::odd_dimension_from_config(omb.wcs.naxis[0]);
    }

    Eigen::VectorXd rows_tan_vec =
        citlali::pipeline::tangent_coordinate_vector(omb.n_rows,
                                                     omb.pixel_size_rad);
    Eigen::VectorXd cols_tan_vec =
        citlali::pipeline::tangent_coordinate_vector(omb.n_cols,
                                                     omb.pixel_size_rad);


    // push back map sizes and coordinates
    map_extents.push_back({static_cast<int>(omb.n_rows), static_cast<int>(omb.n_cols)});
    map_coords.push_back({std::move(rows_tan_vec), std::move(cols_tan_vec)});
}

// determine the map dimensions of the coadded map buffer
template <class EngineType>
void TimeOrderedDataProc<EngineType>::calc_cmb_size(std::vector<map_coord_t> &map_coords) {
    auto& cmb = engine().cmb;

    const auto limits = citlali::pipeline::coordinate_limits(map_coords);

    // get dimensions and tangent coordinate vectorx
    auto [n_rows, rows_tan_vec] =
        citlali::pipeline::dimension_and_tangent_coordinates(
            limits.min_row, limits.max_row, engine().cmb.pixel_size_rad);
    auto [n_cols, cols_tan_vec] =
        citlali::pipeline::dimension_and_tangent_coordinates(
            limits.min_col, limits.max_col, engine().cmb.pixel_size_rad);

    // Set dimensions and wcs parameters
    cmb.n_rows = n_rows;
    cmb.n_cols = n_cols;
    cmb.wcs.naxis[0] = n_cols;
    cmb.wcs.naxis[1] = n_rows;
    cmb.wcs.crpix[0] = (n_cols - 1) / 2.0;
    cmb.wcs.crpix[1] = (n_rows - 1) / 2.0;
    // set tangent plane coordinate vectors
    cmb.rows_tan_vec = std::move(rows_tan_vec);
    cmb.cols_tan_vec = std::move(cols_tan_vec);
}

// allocate observation map buffer
