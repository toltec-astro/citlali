#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_output_dirs.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::get_apt_from_files(const RawObs &rawobs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    // total number of detectors
    Eigen::Index n_dets = 0;
    // detector, nw and array names for each network
    std::vector<Eigen::Index> dets, nws, arrays;

    // loop through input files
    for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            // load data file
            NcFile fo(data_item.filepath(), NcFile::read);

            // get the interface
            auto interface_id = std::stoi(data_item.interface().substr(6));
            // add the current file's number of dets to the total
            n_dets += fo.getVar("Data.Toltec.Is").getDim(1).getSize();

            // get the number of dets in file
            dets.push_back(fo.getVar("Data.Toltec.Is").getDim(1).getSize());
            // get the nw from interface
            nws.push_back(interface_id);
            // get the array from the interface
            arrays.push_back(engine().toltec_io.nw_to_array_map[interface_id]);

            fo.close();

        } catch (NcException &e) {
            logger->error("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", data_item.filepath())};
        }
    }

    // explicitly clear the apt
    engine().calib.apt.clear();

    // resize the apt vectors
    for (auto const& key : engine().calib.apt_header_keys) {
        if (key=="x_t" || key=="y_t") {
            engine().calib.apt[key].setZero(n_dets);
        }
        else {
            engine().calib.apt[key].setOnes(n_dets);
        }
    }

    // set all flags to good
    engine().calib.apt["flag"].setZero(n_dets);

    // add the nws and arrays to the apt table
    Eigen::Index j = 0;
    for (Eigen::Index i=0; i<nws.size(); ++i) {
        engine().calib.apt["nw"].segment(j,dets[i]).setConstant(nws[i]);
        engine().calib.apt["array"].segment(j,dets[i]).setConstant(arrays[i]);

        j = j + dets[i];
    }

    // set uids
    engine().calib.apt["uid"] = Eigen::VectorXd::LinSpaced(n_dets,0,n_dets-1);

    // setup nws, arrays, etc.
    engine().calib.setup();

    // filepath
    engine().calib.apt_filepath = "internally generated for beammap";
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::get_tone_freqs_from_files(const RawObs &rawobs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    // tone frquencies for each network
    std::map<Eigen::Index,Eigen::MatrixXd> tone_freqs;

    // loop through input files
    for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            // load data file
            NcFile fo(data_item.filepath(), NcFile::read);

            // get the interface
            auto interface_id = std::stoi(data_item.interface().substr(6));

            // dimension of tone freqs is (n_sweeps, n_tones)
            Eigen::Index n_sweeps = fo.getVar("Header.Toltec.ToneFreq").getDim(0).getSize();

            // get local oscillator frequency
            double lo_freq;
            fo.getVar("Header.Toltec.LoCenterFreq").getVar(&lo_freq);

            // get tone_freqs for interface
            tone_freqs[interface_id].resize(fo.getVar("Header.Toltec.ToneFreq").getDim(1).getSize(), n_sweeps);
            fo.getVar("Header.Toltec.ToneFreq").getVar(tone_freqs[interface_id].data());

            // add local oscillator freq
            tone_freqs[interface_id] = tone_freqs[interface_id].array() + lo_freq;

            fo.close();

        } catch (NcException &e) {
            logger->error("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", data_item.filepath())};
        }
    }

    engine().calib.apt["tone_freq"].resize(engine().calib.n_dets);

    // assign tone freqs by nw id to avoid ordering mismatches
    for (const auto& [nw, limits] : engine().calib.nw_limits) {
        auto it = tone_freqs.find(nw);
        if (it == tone_freqs.end()) {
            logger->error("missing tone freqs for nw {}", nw);
            std::exit(EXIT_FAILURE);
        }

        const auto& tf = it->second;
        Eigen::Index n_tones = tf.rows();
        Eigen::Index n_sweeps = tf.cols();

        const auto start = std::get<0>(limits);
        const auto end = std::get<1>(limits);
        const auto expected = end - start;

        if (n_sweeps < 1) {
            logger->error("no tone freq sweeps for nw {}", nw);
            std::exit(EXIT_FAILURE);
        }
        if (n_tones != expected) {
            logger->error("tone freq size mismatch for nw {} (tones={}, expected dets={})",
                          nw, n_tones, expected);
            std::exit(EXIT_FAILURE);
        }
        if (n_sweeps > 1) {
            logger->warn("tone freqs have {} sweeps for nw {}, using first sweep", n_sweeps, nw);
        }

        engine().calib.apt["tone_freq"].segment(start, expected) = tf.col(0);
    }

    if (!engine().telescope.sim_obs) {
        /* find duplicates */

        // frequency separation
        Eigen::VectorXd dfreq(engine().calib.n_dets);
        dfreq(0) = engine().calib.apt["tone_freq"](1) - engine().calib.apt["tone_freq"](0);

        // loop through tone freqs and find distance
        for (Eigen::Index i=1; i<engine().calib.apt["tone_freq"].size()-1; ++i) {
            dfreq(i) = std::min(abs(engine().calib.apt["tone_freq"](i) - engine().calib.apt["tone_freq"](i-1)),
                                abs(engine().calib.apt["tone_freq"](i+1) - engine().calib.apt["tone_freq"](i)));
        }
        // get last distance
        dfreq(dfreq.size()-1) = abs(engine().calib.apt["tone_freq"](dfreq.size()-1)-engine().calib.apt["tone_freq"](dfreq.size()-2));

        // number of nearby tones found
        int n_nearby_tones = 0;

        // store duplicates
        engine().calib.apt["duplicate_tone"].setZero(engine().calib.n_dets);

        // loop through flag columns
        for (Eigen::Index i=0; i<engine().calib.n_dets; ++i) {
            // if closer than freq separation limit and unflagged, flag it
            if (dfreq(i) < engine().rtcproc.delta_f_min_Hz) {
                engine().calib.apt["duplicate_tone"](i) = 1;
                n_nearby_tones++;
            }
        }
        logger->info("{} nearby tones found. these will be flagged.",n_nearby_tones);
    }
}

// create output directories
template <class EngineType>
void TimeOrderedDataProc<EngineType>::create_output_dir() {
    // redu subdir
    engine().redu_dir_name = "";

    // create reduction subdir
    if (engine().use_subdir) {
        engine().redu_dir_name =
            citlali::pipeline::next_reduction_subdir_path(
                engine().output_dir, engine().redu_dir_num);
        fs::create_directories(engine().redu_dir_name);
        citlali::pipeline::configure_reduction_logging_and_profile(
            engine().redu_dir_name, logger);
    }
    else {
        engine().redu_dir_name = engine().output_dir + "/";
        citlali::pipeline::configure_reduction_logging_and_profile(
            engine().redu_dir_name, logger);
    }

    // coadded subdir
    if (citlali::pipeline::coadd_outputs_enabled(engine())) {
        engine().coadd_dir_name = engine().redu_dir_name + "/coadded/";
        citlali::pipeline::create_coadd_output_dirs(
            engine().coadd_dir_name,
            citlali::pipeline::map_filter_outputs_enabled(engine()), logger);
    }
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::check_inputs(const RawObs &rawobs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    Eigen::Index n_dets = 0;

    // loop through input files
    for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            // load data file
            NcFile fo(data_item.filepath(), NcFile::read);
            // get number of dets from data and add to global value
            n_dets += fo.getVar("Data.Toltec.Is").getDim(1).getSize();

            fo.close();

        } catch (NcException &e) {
            logger->error("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", data_item.filepath())};
        }
    }

    // check if number of detectors in apt file is equal to those in files
    if (n_dets != engine().calib.n_dets) {
        logger->error("number of detectors in data files and apt file do not match");
        std::exit(EXIT_FAILURE);
    }
}

// align tod with telescope
