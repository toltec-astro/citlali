#pragma once

// Implementation detail included by todproc.h.

template <class EngineType>
void TimeOrderedDataProc<EngineType>::calc_map_num() {
    // auto map grouping
    if (engine().map_grouping=="auto") {
        // array map grouping for science and pointing
        if ((engine().redu_type == "science") || (engine().redu_type == "pointing")) {
            engine().map_grouping = "array";
        }

        // detector map grouping for beammaps
        else if ((engine().redu_type == "beammap")) {
            engine().map_grouping = "detector";
        }
    }

    if (engine().map_grouping == "detector" && engine().redu_type != "beammap") {
        logger->warn("mapmaking.grouping=detector is only supported for beammap; defaulting to array for {}",
                     engine().redu_type);
        engine().map_grouping = "array";
    }

    engine().omb.map_grouping = engine().map_grouping;
    engine().cmb.map_grouping = engine().map_grouping;
    engine().rtcproc.kernel.map_grouping = engine().map_grouping;

    // overwrite map number for detectors
    if (engine().map_grouping == "detector") {
        engine().n_maps = engine().calib.n_dets;
    }

    // overwrite map number for networks
    else if (engine().map_grouping == "nw") {
        engine().n_maps = engine().calib.n_nws;
    }

    // overwrite map number for arrays
    else if (engine().map_grouping == "array") {
        engine().n_maps = engine().calib.n_arrays;
    }

    // overwrite map number for fg grouping
    else if (engine().map_grouping == "fg") {
        // there are potentially 4 fg's per array, so total number of maps is max 4 x n_arrays
        engine().n_maps = engine().calib.fg.size()*engine().calib.n_arrays;
    }

    if (engine().rtcproc.run_polarization) {
        // multiply by number of polarizations (stokes I + Q + U = 3)
        engine().n_maps = engine().n_maps*engine().rtcproc.polarization.stokes_params.size();
    }

    // mapping from index in map vector to detector array index
    // if stokes I array grouping with all arrays, this will be [0,1,2]
    // if missing array 0, this will be [1,2]
    engine().maps_to_arrays.resize(engine().n_maps);

    // mapping from index in map vector to stokes parameter index (I=0, Q=1, U=2)
    // if array grouping with all arrays this will be [0,0,0,1,1,2,2,2]
    // and maps_to_arrays will be [0,1,2,0,1,2,0,1,2]
    engine().maps_to_stokes.resize(engine().n_maps);

    // mapping from array index to index in map vectors (reverse of maps_to_arrays)
    // if stokes I array grouping with all arrays, this will also be [0,1,2]
    // if missing array 0, this will be [0,1]
    engine().arrays_to_maps.resize(engine().n_maps);

    // array to hold mapping from group to detector array index
    Eigen::VectorXI array_indices;

    // detector gropuing
    if (engine().map_grouping == "detector") {
        // only do stokes I as Q and U don't make sense for detector grouping
        // this is just a copy of the array indices from the apt
        array_indices = engine().calib.apt["array"].template cast<Eigen::Index> ();
    }

    // array grouping
    else if (engine().map_grouping == "array") {
        // if all arrays are included this will be [0,1,2]
        array_indices = engine().calib.arrays;
    }

    // network grouping
    else if (engine().map_grouping == "nw") {
        // if all nws/arrays are included this will be:
        // [0,0,0,0,0,0,0,0,1,1,1,1,2,2]
        // nws are ordered automatically when files are read in
        array_indices.resize(engine().calib.nws.size());

        // find all map from nw to arrays
        for (Eigen::Index i=0; i<engine().calib.nws.size(); ++i) {
            // get array for current nw
            array_indices(i) = engine().toltec_io.nw_to_array_map[engine().calib.nws(i)];
        }
    }

    // frequency grouping
    else if (engine().map_grouping == "fg") {
        // size of array indices is number of fg's x number of arrays
        // if all fgs are included, this will be:
        // [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
        // the order of the fgs will vary depending on the apt, but this is irrelevant
        array_indices.resize(engine().calib.fg.size()*engine().calib.n_arrays);

        // map from fg to array index
        Eigen::Index j = 0;
        // loop through arrays
        for (Eigen::Index i=0; i<engine().calib.n_arrays; ++i) {
            // append current array index to all elements within a segment of fg size
            array_indices.segment(j,engine().calib.fg.size()).setConstant(engine().calib.arrays(i));
            // increment by fg size
            j = j + engine().calib.fg.size();
        }
    }

    // copy array_indices into maps_to_arrays and maps_to_stokes for each stokes param
    Eigen::Index j = 0;
    // loop through stokes params
    for (const auto &[stokes_index,stokes_param]: engine().rtcproc.polarization.stokes_params) {
        // for each stokes param append all array indices in order
        engine().maps_to_arrays.segment(j,array_indices.size()) = array_indices;
        // for each stokes param append current stokes index
        engine().maps_to_stokes.segment(j,array_indices.size()).setConstant(stokes_index);
        // increment by array index size
        j = j + array_indices.size();
    }

    // calculate detector array index to map index
    Eigen::Index index = 0;
    // start at map index 0
    engine().arrays_to_maps(0) = index;
    for (Eigen::Index i=1; i<engine().n_maps; ++i) {
        // we move to the next map index when the array increments
        if (engine().maps_to_arrays(i) > engine().maps_to_arrays(i-1)) {
            index++;
        }
        // reset to first map index when we return the an earlier array
        else if (engine().maps_to_arrays(i) < engine().maps_to_arrays(i-1)) {
            index = 0;
        }
        engine().arrays_to_maps(i) = index;
    }
}

// calculate map dimensions
template <class EngineType>
void TimeOrderedDataProc<EngineType>::calc_omb_size(std::vector<map_extent_t> &map_extents, std::vector<map_coord_t> &map_coords) {

    // reference to map buffer
    auto& omb = engine().omb;

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
            pointing_offsets_arcsec["az"] = engine().pointing_offsets_arcsec["az"].segment(si,sl);
            pointing_offsets_arcsec["alt"] = engine().pointing_offsets_arcsec["alt"].segment(si,sl);

            // don't need to find the offsets if in detector mode
            if (engine().map_grouping!="detector") {
                // loop through detectors
                grppi::map(tula::grppi_utils::dyn_ex(engine().parallel_policy), det_in_vec, det_out_vec, [&](auto j) {

                    // get pointing
                    auto [lat, lon] = engine_utils::calc_det_pointing(tel_data, engine().calib.apt["x_t"](j), engine().calib.apt["y_t"](j),
                                                                      engine().telescope.pixel_axes, pointing_offsets_arcsec, engine().map_grouping);
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
                auto [lat, lon] = engine_utils::calc_det_pointing(tel_data, 0., 0., engine().telescope.pixel_axes,
                                                                  pointing_offsets_arcsec, engine().map_grouping);
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

        // calculate dimensions
        auto calc_map_dims = [&](double min_dim, double max_dim) {
            int min_pix = static_cast<int>(ceil(abs(min_dim / omb.pixel_size_rad)));
            int max_pix = static_cast<int>(ceil(abs(max_dim / omb.pixel_size_rad)));
            return 2 * std::max(min_pix, max_pix) + 1;
        };

        // get n_rows and n_cols
        omb.n_rows = calc_map_dims(min_lat, max_lat);
        omb.n_cols = calc_map_dims(min_lon, max_lon);
    }

    else {
        // Ensure odd dimensions
        omb.n_rows = (omb.wcs.naxis[1] % 2 == 0) ? omb.wcs.naxis[1] + 1 : omb.wcs.naxis[1];
        omb.n_cols = (omb.wcs.naxis[0] % 2 == 0) ? omb.wcs.naxis[0] + 1 : omb.wcs.naxis[0];
    }

    const double omb_row_center = (omb.n_rows - 1) / 2.0;
    const double omb_col_center = (omb.n_cols - 1) / 2.0;
    Eigen::VectorXd rows_tan_vec = Eigen::VectorXd::LinSpaced(omb.n_rows, 0, omb.n_rows - 1).array() * omb.pixel_size_rad -
                                   omb_row_center * omb.pixel_size_rad;
    Eigen::VectorXd cols_tan_vec = Eigen::VectorXd::LinSpaced(omb.n_cols, 0, omb.n_cols - 1).array() * omb.pixel_size_rad -
                                   omb_col_center * omb.pixel_size_rad;


    // push back map sizes and coordinates
    map_extents.push_back({static_cast<int>(omb.n_rows), static_cast<int>(omb.n_cols)});
    map_coords.push_back({std::move(rows_tan_vec), std::move(cols_tan_vec)});
}

// determine the map dimensions of the coadded map buffer
template <class EngineType>
void TimeOrderedDataProc<EngineType>::calc_cmb_size(std::vector<map_coord_t> &map_coords) {
    auto& cmb = engine().cmb;

    // Initialize min/max values
    double min_row = std::numeric_limits<double>::max();
    double max_row = std::numeric_limits<double>::lowest();
    double min_col = min_row;
    double max_col = max_row;

    // Find global min/max for rows and columns
    for (const auto& coord : map_coords) {
        min_row = std::min(min_row, coord.front().minCoeff());
        max_row = std::max(max_row, coord.front().maxCoeff());
        min_col = std::min(min_col, coord.back().minCoeff());
        max_col = std::max(max_col, coord.back().maxCoeff());
    }

    // calculate dimensions
    auto calc_map_dims = [&](auto min_dim, auto max_dim) {
        int min_pix = static_cast<int>(ceil(abs(min_dim / engine().cmb.pixel_size_rad)));
        int max_pix = static_cast<int>(ceil(abs(max_dim / engine().cmb.pixel_size_rad)));

        int n_dim = 2 * std::max(min_pix, max_pix) + 1;
        const double dim_center = (n_dim - 1) / 2.0;
        Eigen::VectorXd dim_vec = Eigen::VectorXd::LinSpaced(n_dim, 0, n_dim - 1)
                                          .array() * engine().cmb.pixel_size_rad - dim_center * engine().cmb.pixel_size_rad;

        return std::make_tuple(n_dim, std::move(dim_vec));
    };

    // get dimensions and tangent coordinate vectorx
    auto [n_rows, rows_tan_vec] = calc_map_dims(min_row, max_row);
    auto [n_cols, cols_tan_vec] = calc_map_dims(min_col, max_col);

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

