#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/config/calibration_config.h>
#include <citlali/core/pipeline/apt_detector_relation.h>
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
    const bool native_coordinates =
        engine().calib.has_apt_detector_relation();
    const citlali::pipeline::AptDetectorRelation *native_relation = nullptr;
    if (native_coordinates) {
        if (engine().alignment.native_consumer_plan == nullptr ||
            engine().alignment.native_pointing_plan == nullptr ||
            !engine().alignment.native_pointing_plan->bound_to(
                engine().alignment.native_consumer_plan)) {
            throw std::runtime_error(
                "native map geometry requires coherent alignment and network-native pointing plans");
        }
        native_relation =
            &engine().calib.require_apt_detector_relation();
        if (native_relation->bindings().size() !=
            static_cast<std::size_t>(engine().calib.n_dets)) {
            throw std::runtime_error(
                "native map geometry detector relation cardinality differs from calibration columns");
        }
    }

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

            std::map<citlali::pipeline::TimestreamNetworkId,
                     citlali::pipeline::NativeTelescopeData>
                native_tel_data;
            std::map<citlali::pipeline::TimestreamNetworkId,
                     citlali::pipeline::NativePointingOffsetsArcsec>
                native_pointing_offsets;
            if (native_coordinates) {
                if (si < 0 || sl <= 0 ||
                    static_cast<std::size_t>(si + sl) >
                        engine().alignment.native_consumer_plan->slot_count()) {
                    throw std::runtime_error(
                        "native map geometry scan exceeds relational common-slot bounds");
                }
                const auto complete_runs =
                    citlali::pipeline::partition_complete_native_cohort_runs(
                        *engine().alignment.native_consumer_plan,
                        citlali::pipeline::NativeOperationIdentity{
                            0, static_cast<std::int64_t>(i)},
                        static_cast<std::size_t>(si),
                        static_cast<std::size_t>(si + sl), 0);
                if (complete_runs.empty()) {
                    throw std::runtime_error(
                        "native map geometry scan has no complete measured cohort");
                }
                const auto &participant_ids = engine()
                    .alignment.native_consumer_plan
                    ->participant_network_ids();
                for (std::size_t participant = 0;
                     participant < participant_ids.size(); ++participant) {
                    const auto network_id = participant_ids[participant];
                    const auto &pointing = engine()
                        .alignment.native_pointing_plan->network(network_id);
                    Eigen::Index row_count = 0;
                    for (const auto &run : complete_runs) {
                        row_count += static_cast<Eigen::Index>(
                            run.participant_runs.at(participant).row_count());
                    }
                    if (row_count <= 0) {
                        throw std::runtime_error(
                            "native map geometry network has no measured rows");
                    }
                    auto &network_tel = native_tel_data[network_id];
                    for (const auto &[key, values] :
                         pointing.telescope_data()) {
                        (void)values;
                        network_tel[key].resize(row_count);
                    }
                    auto &network_offsets =
                        native_pointing_offsets[network_id];
                    for (const auto &[axis, values] :
                         pointing.pointing_offsets_arcsec()) {
                        (void)values;
                        network_offsets[axis].resize(row_count);
                    }
                    Eigen::Index destination = 0;
                    for (const auto &complete : complete_runs) {
                        const auto &run =
                            complete.participant_runs.at(participant);
                        const auto count =
                            static_cast<Eigen::Index>(run.row_count());
                        const auto source =
                            pointing.local_row(run.first_native_row);
                        for (Eigen::Index local = 0; local < count;
                             ++local) {
                            const auto native_row =
                                run.first_native_row + local;
                            if (!(pointing.identity(native_row) ==
                                  engine().alignment.native_consumer_plan
                                      ->network(network_id)
                                      .identity(native_row))) {
                                throw std::runtime_error(
                                    "native map geometry pointing identity is stale");
                            }
                        }
                        for (auto &[key, values] : network_tel) {
                            values.segment(destination, count) =
                                pointing.telescope_series(key).segment(
                                    source, count);
                        }
                        for (auto &[axis, values] : network_offsets) {
                            values.segment(destination, count) =
                                pointing.pointing_offset_arcsec(axis).segment(
                                    source, count);
                        }
                        destination += count;
                    }
                }
            }
            else {
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
            }

            // don't need to find the offsets if in detector mode
            if (mapmaking_settings.grouping !=
                citlali::config::MapGrouping::detector) {
                // loop through detectors
                grppi::map(
                    tula::grppi_utils::dyn_ex(
                        citlali::pipeline::runtime_parallel_policy_name(
                            engine())),
                    det_in_vec, det_out_vec, [&](auto j) {

                    auto *selected_tel = &tel_data;
                    auto *selected_offsets =
                        &pointing_offsets_arcsec;
                    if (native_coordinates) {
                        const auto &binding =
                            native_relation->binding_for_column(
                                static_cast<std::size_t>(j));
                        (void)native_relation->require_binding(
                            native_relation->binding_reference_for_column(
                                static_cast<std::size_t>(j)));
                        selected_tel = &native_tel_data.at(
                            static_cast<citlali::pipeline::TimestreamNetworkId>(
                                binding.network));
                        selected_offsets = &native_pointing_offsets.at(
                            static_cast<citlali::pipeline::TimestreamNetworkId>(
                                binding.network));
                    }

                    // get pointing
                    auto [lat, lon] = engine_utils::calc_det_pointing(
                        *selected_tel, engine().calib.apt["x_t"](j),
                        engine().calib.apt["y_t"](j),
                        engine().telescope.pixel_axes, *selected_offsets,
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
                if (native_coordinates) {
                    // Detector grouping still has zero detector offsets, but
                    // every detector must use its own network's native-time
                    // telescope evaluation.
                    for (Eigen::Index j = 0;
                         j < engine().calib.n_dets; ++j) {
                        const auto &binding =
                            native_relation->binding_for_column(
                                static_cast<std::size_t>(j));
                        (void)native_relation->require_binding(
                            native_relation->binding_reference_for_column(
                                static_cast<std::size_t>(j)));
                        const auto network_id = static_cast<
                            citlali::pipeline::TimestreamNetworkId>(
                                binding.network);
                        auto [lat, lon] = engine_utils::calc_det_pointing(
                            native_tel_data.at(network_id), 0., 0.,
                            engine().telescope.pixel_axes,
                            native_pointing_offsets.at(network_id),
                            mapmaking_settings.grouping);
                        if (engine().calib.apt["flag"](j) == 0) {
                            det_lat_limits(j, 0) = std::min(
                                det_lat_limits(j, 0), lat.minCoeff());
                            det_lat_limits(j, 1) = std::max(
                                det_lat_limits(j, 1), lat.maxCoeff());
                            det_lon_limits(j, 0) = std::min(
                                det_lon_limits(j, 0), lon.minCoeff());
                            det_lon_limits(j, 1) = std::max(
                                det_lon_limits(j, 1), lon.maxCoeff());
                        }
                    }
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
