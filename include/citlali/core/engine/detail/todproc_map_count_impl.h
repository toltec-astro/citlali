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
    if (auto parsed_grouping =
            citlali::config::parse_map_grouping(engine().map_grouping)) {
        engine().typed_config.mapmaking.grouping = *parsed_grouping;
    }

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
