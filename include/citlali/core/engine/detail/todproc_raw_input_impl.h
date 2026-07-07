#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <class Logger>
void log_reduction_version_stamp(const Logger &logger) {
    logger->info("citlali version: {}", CITLALI_GIT_VERSION);
    logger->info("kids version: {}", KIDSCPP_GIT_VERSION);
    logger->info("tula version: {}", TULA_GIT_VERSION);
}

}  // namespace citlali::pipeline

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
        // redu number
        engine().redu_dir_num = 0;

        std::stringstream ss_redu_dir_num;
        // add leading zero to redu_dir_num (i.e., '00', '01',...)
        ss_redu_dir_num << std::setfill('0') << std::setw(2) << engine().redu_dir_num;

        // create redu dir name ('redu00', 'redu01',...)
        std::string redu_dir_name = "redu" + ss_redu_dir_num.str();

        // iteratively check if current subdir with current redu number exists
        while (fs::exists(fs::status(engine().output_dir + "/" + redu_dir_name))) {
            // increment redu number if subdir exists
            engine().redu_dir_num++;
            std::stringstream ss_redu_dir_num_i;
            ss_redu_dir_num_i << std::setfill('0') << std::setw(2) << engine().redu_dir_num;
            redu_dir_name = "redu" + ss_redu_dir_num_i.str();
        }

        // final redu dir name is output directory from config + /reduNN
        engine().redu_dir_name = engine().output_dir + "/" + redu_dir_name;

        // create redu dir directory
        fs::create_directories(engine().redu_dir_name);
        try {
            auto log_path = citlali::logging::enable_reduction_gzip_logs(engine().redu_dir_name);
            logger->info("reduction-local compressed log: {}", log_path);
            citlali::pipeline::log_reduction_version_stamp(logger);
        } catch (const std::exception &e) {
            logger->warn("failed to enable reduction-local compressed log in {}: {}",
                         engine().redu_dir_name, e.what());
        }
        citlali::pipeline::configure_stage_profile_output(
            engine().redu_dir_name, logger);
    }
    else {
        engine().redu_dir_name = engine().output_dir + "/";
        try {
            auto log_path = citlali::logging::enable_reduction_gzip_logs(engine().redu_dir_name);
            logger->info("reduction-local compressed log: {}", log_path);
            citlali::pipeline::log_reduction_version_stamp(logger);
        } catch (const std::exception &e) {
            logger->warn("failed to enable reduction-local compressed log in {}: {}",
                         engine().redu_dir_name, e.what());
        }
        citlali::pipeline::configure_stage_profile_output(
            engine().redu_dir_name, logger);
    }

    // coadded subdir
    if (citlali::pipeline::coadd_outputs_enabled(engine())) {
        engine().coadd_dir_name = engine().redu_dir_name + "/coadded/";
        // coadded raw subdir
        if (!fs::exists(fs::status(engine().coadd_dir_name + "raw/"))) {
            fs::create_directories(engine().coadd_dir_name + "raw/");
        }
        else {
            logger->warn("directory {} already exists", engine().coadd_dir_name + "raw/");
        }
        // if map filtering is requested
        if (citlali::pipeline::map_filter_outputs_enabled(engine())) {
            // coadded filtered subdir
            if (!fs::exists(fs::status(engine().coadd_dir_name + "filtered/"))) {
                fs::create_directories(engine().coadd_dir_name + "filtered/");
            }
            else {
                logger->warn("directory {} already exists", engine().coadd_dir_name + "filtered/");
            }
        }
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
