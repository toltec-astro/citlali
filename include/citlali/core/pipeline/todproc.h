#pragma once

namespace fs = std::filesystem;
using namespace citlali::config::options;

struct DummyEngine {
    template <typename OStream>
    friend OStream &operator<<(OStream &os, const DummyEngine &e) {
        return os << fmt::format("DummyEngine()");
    }
};

template <class EngineType>
struct TimeOrderedDataProc : ConfigMapper<TimeOrderedDataProc<EngineType>> {
    using Base = ConfigMapper<TimeOrderedDataProc<EngineType>>;
    using ConfigType = typename Base::config_t;
    using Engine = EngineType;
    using MapExtentType = std::pair<int, int>;
    using MapCoordType = std::pair<Eigen::VectorXd, Eigen::VectorXd>;

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // number of maps to make
    Eigen::Index n_maps;

    // unique std::map keys for maps
    Eigen::VectorXi map_keys;
    std::map<int, Eigen::VectorXi> unique_map_keys;

    // create vectors for map size and grouping parameters
    std::vector<MapExtentType> map_extents{};
    std::vector<MapCoordType> map_coords{};

    // obsnums
    std::vector<std::string> obsnums;

    TimeOrderedDataProc(ConfigType config) : Base{std::move(config)} {}

    // check if config file has nodes
    static auto check_config(const ConfigType &config)
        -> std::optional<std::string> {
        /**
         * @brief check if any of the main required nodes were not found
         */
        // get logger
        std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

        std::vector<std::string> missing_keys;
        logger->debug("check TOD proc config\n{}", config);
        // check for runtime config node
        if (!config.has("runtime")) {
            missing_keys.push_back("runtime");
        }
        // check for timestream config node
        if (!config.has("timestream")) {
            missing_keys.push_back("timestream");
        }
        // check for mapmaking config node
        if (!config.has("mapmaking")) {
            missing_keys.push_back("mapmaking");
        }
        // check for beammap config node
        if (!config.has("beammap")) {
            missing_keys.push_back("beammap");
        }
        // check for coadd config node
        if (!config.has("coadd")) {
            missing_keys.push_back("coadd");
        }
        // check for noise map config node
        if (!config.has("noise_maps")) {
            missing_keys.push_back("noise_maps");
        }
        // check for post processing config node
        if (!config.has("post_processing")) {
            missing_keys.push_back("post_processing");
        }
        if (missing_keys.empty()) {
            return std::nullopt;
        }
        return fmt::format("invalid or missing keys={}", missing_keys);
    }

    void set_parallelization() {
        /**
         * @brief set global number of threads, parallelization for
         * OMP and eigen.
         */
        // ensure one thread for sequential mode
        if (exec_mode == "seq") {
            n_threads = 1;
        }

        // set omp parallelization explicitly
        if (exec_mode == "omp") {
            #ifdef _OPENMP
            omp_set_num_threads(n_threads);
            #else
            exec_mode = "thr";
            logger->info("omp not found, using iso c++ threads");
            #endif
        }

        // disable eigen underlying parallelization
        Eigen::setNbThreads(1);

        // set fftw threads
        int fftw_threads = fftw_init_threads();
        fftw_plan_with_nthreads(n_threads);
    }

    void align_timestreams(const RawObs &rawobs) {
        /**
         * @brief finds a common time grid for all networks and hwpr
         * and interpolates the telescope pointing vectors onto it.
         * The latest start time and earliest end time are used assuming
         * to define the start and end of the common nw/hwpr grid
         */
        using namespace netCDF;
        using namespace netCDF::exceptions;

        // hold time vectors for each network
        std::vector<Eigen::VectorXd> nw_times;

        // loop through networks and build time vectors
        for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
            try {
                // load data file
                NcFile fo(data_item.filepath(), NcFile::read);
                auto vars = fo.getVars();

                // get roach sample rate
                double f_smp_roach;
                vars.find("Header.Toltec.SampleFreq")->second.getVar(&f_smp_roach);

                // get roach index for offsets
                int roach_index;
                vars.find("Header.Toltec.RoachIndex")->second.getVar(&roach_index);

                // get dimensions for time matrix
                Eigen::Index n_pts = vars.find("Data.Toltec.Ts")->second.getDim(0).getSize();
                Eigen::Index n_times = vars.find("Data.Toltec.Ts")->second.getDim(1).getSize();

                // get time matrix
                Eigen::MatrixXi ts(n_times, n_pts);
                vars.find("Data.Toltec.Ts")->second.getVar(ts.data());

                // transpose due to row-major order (now n_pts, n_times)
                ts.transposeInPlace();

                // get fpga frequency
                double fpga_freq;
                vars.find("Header.Toltec.FpgaFreq")->second.getVar(&fpga_freq);

                // cast to double
                Eigen::MatrixXd ts_double = ts.cast<double>();

                // extract columns with descriptive names
                Eigen::VectorXd sec = ts_double.col(0);        // ClockTime (sec)
                Eigen::VectorXd nsec = ts_double.col(5);       // ClockTimeNanoSec (nsec)
                Eigen::VectorXd pps = ts_double.col(1);        // PpsCount (pps ticks)
                Eigen::VectorXd msec = ts_double.col(2) / fpga_freq;  // ClockCount (clock ticks) to seconds
                //Eigen::VectorXd count = ts_double.col(3);      // PacketCount (packet ticks)
                Eigen::VectorXd pps_msec = ts_double.col(4) / fpga_freq; // PpsTime (clock ticks) to seconds

                // determine start time with empirical offset
                double start_time_dbl = sec[0] + nsec[0] * 1e-9;
                int start_time = int(start_time_dbl - 0.5);
                start_time_dbl = start_time;

                // calculate clock count difference (dt)
                Eigen::VectorXd dt = msec - pps_msec;

                // handle overflow due to int32, using Eigen array logic
                dt = (dt.array() < 0).select(msec.array() - pps_msec.array() + (pow(2.0, 32) - 1) / fpga_freq, msec - pps_msec);

                // build the time vector for the current network
                Eigen::VectorXd nw_time = start_time_dbl + pps.array() + dt.array()
                                          + interface_offset_map["toltec" + std::to_string(roach_index)];

                // store all time vectors
                nw_times.push_back(std::move(nw_time));

                fo.close();

            } catch (NcException &e) {
                throw std::runtime_error(fmt::format("unable to open file {}: {}", data_item.filepath(), e.what()));
            }
        }

        // get hwpr times if not ignored
        if (engine().toltec.hwpr.run_hwpr) {
            logger->debug("calculating hwpr time");
            // hwpr gets added alongside networks
            nw_times.push_back(engine().toltec.hwpr.calc_time_vector(interface_offset_map["hwpr"]));
        }

        // latest starting time of all networks
        double max_init_time = std::numeric_limits<double>::lowest();
        // earliest final time of all networks
        double min_final_time = std::numeric_limits<double>::max();

        // indices for max_init_time and min_final_time
        Eigen::Index max_init_idx = 0;
        Eigen::Index min_final_idx = 0;

        // get global max init and min final times and indices
        for (Eigen::Index i = 0; i < nw_times.size(); ++i) {
            double initial_time = nw_times[i](0);
            double final_time = nw_times[i](nw_times[i].size() - 1);

            // get latest starting network
            if (initial_time > max_init_time) {
                max_init_time = initial_time;
                max_init_idx = i;
            }
            // get earliest ending network
            if (final_time < min_final_time) {
                min_final_time = final_time;
                min_final_idx = i;
            }
        }

        // store start and end indices for each network
        engine().init_indices.setZero(nw_times.size());
        engine().final_indices.setZero(nw_times.size());

        for (Eigen::Index i = 0; i < nw_times.size(); ++i) {
            Eigen::Index nw_init_idx, nw_final_idx;

            // find index where current nw time is closest to max start time
            (nw_times[i].array() - max_init_time).abs().minCoeff(&nw_init_idx);
            // make sure it starts after the latest starting time
            while (nw_times[i](nw_init_idx) < max_init_time) {
                nw_init_idx += 1;
            }

            engine().init_indices(i) = nw_init_idx;

            // find index where current nw time is closest to min final time
            (nw_times[i].array() - min_final_time).abs().minCoeff(&nw_final_idx);
            // make sure it ends before the latest starting time
            while (nw_times[i](nw_final_idx) > min_final_time) {
                nw_final_idx -= 1;
            }

            engine().final_indices(i) = nw_final_idx;
        }

        // get shortest timestream
        Eigen::Index n_samples = (engine().final_indices.array() - engine().init_indices.array() + 1).minCoeff();

        // size of telescope data
        Eigen::Matrix<Eigen::Index,1,1> nd;
        nd << engine().telescope.data["TelTime"].size();

        // data time vector to interpolate onto
        Eigen::VectorXd xi = nw_times[max_init_idx].head(n_samples);

        // interpolate telescope data onto data timestream
        for (const auto &tel_it : engine().telescope.data) {
            // don't interpolate telescope time itself
            if (tel_it.first !="TelTime" && tel_it.first !="TelUTC") {
                // telescope vector to interpolate
                Eigen::VectorXd yd = engine().telescope.data.at(tel_it.first);
                // vector to store interpolated outputs in
                Eigen::VectorXd yi(n_samples);

                mlinterp::interp(nd.data(), n_samples, // nd, ni
                                 yd.data(), yi.data(), // yd, yi
                                 engine().telescope.data.at("TelTime").data(), xi.data()); // xd, xi

                // move back into data vector
                engine().telescope.data[tel_it.first] = std::move(yi);
            }
        }

        // replace telescope time vectors
        engine().telescope.data.at("TelTime") = xi;
        engine().telescope.data.at("TelUTC") = xi;

        // interpolate hwpr
        if (engine().toltec.hwpr.run_hwpr) {
            logger->debug("interpolating hwpr angle");
            int n_times = nw_times.size();
            nd << nw_times[n_times - 1].size();

            // vector to store interpolated outputs in
            Eigen::VectorXd yi(n_samples);

            mlinterp::interp(nd.data(), n_samples, // nd, ni
                             engine().toltec.hwpr.hwpr_theta.data(), yi.data(), // yd, yi
                             nw_times[n_times - 1].data(), xi.data()); // xd, xi

            engine().toltec.hwpr.hwpr_theta = std::move(yi);
        }
    }

    void interpolate_pointing_offsets() {
        /**
         * @brief interpolate between pointing offsets in config file
         * Citlali accepts one or two pointing offsets in the configuration file as well as
         * mjd (either two or they are ignored and the start and end times of the observation
         * are used instead.  Linear interpolation in time.
         */
        // determine the number of offsets from the config
        Eigen::Index n_offsets = engine().telescope.data.at("pointing_offset_az_arcsec").size();
        // must occur after alignment
        const auto& tel_time = engine().telescope.data.at("TelTime");
        Eigen::Index ni = tel_time.size();

        // keys for pointing offsets
        std::vector<std::string> altaz_keys = {"pointing_offset_alt_arcsec", "pointing_offset_az_arcsec"};

        for (const auto &key : altaz_keys) {
            if (n_offsets == 1) {
                // single offset value - set it constant for all times
                double offset = engine().telescope.data[key](0);
                engine().telescope.data.at(key).resize(ni);
                engine().telescope.data.at(key).setConstant(offset);
            }
            else if (n_offsets == 2) {
                // two offsets - interpolate over time
                Eigen::VectorXd xd(2);
                const auto& mjd_offsets = engine().telescope.pointing_offset_mjd;

                // use specified mjd if available, else use start/end of the observation
                if ((mjd_offsets.array() <= 0).any()) {
                    xd << tel_time(0), tel_time(ni - 1);
                } else {
                    xd << citlali::utils::timing::mjd_to_unix(mjd_offsets(0)),
                          citlali::utils::timing::mjd_to_unix(mjd_offsets(1));

                    // Ensure offsets cover the entire observation period
                    if (xd(0) > tel_time(0) || xd(1) < tel_time(ni - 1)) {
                        throw std::runtime_error("mjd range is invalid.");
                    }
                }

                // interpolate offsets onto the telescope time vector
                Eigen::VectorXd yi(ni);
                mlinterp::interp(&n_offsets, ni,
                                 engine().telescope.data.at(key).data(), yi.data(),
                                 xd.data(), tel_time.data());

                // store the interpolated pointing offsets
                engine().telescope.data[key] = yi;
            }
            else {
                throw std::runtime_error("Only one or two values for altaz offsets are supported.");
            }
        }
    }

    // stuff that needs to be run for each obs
    void setup_obs_tod(const RawObs &rawobs) {
        /**
         * @brief run all per observation setup steps
         * 1. get meta information from rawobs (n_dets, interfaces, roach_id, tone_freqs, adc_snap_data)
         * 2. get photometry config (beammap only)
         * 3. get telescope data
         * 4. get apt (load or generate from raw files)
         * 5. rescale apt flxscale
         * 6. populate apt with file tone frequency and make per nw and per array apts
         * 7. get hwpr
         * 8. align timestreams
         * 9. get astrometry config (pointing offsets)
         * 10. interpolate pointing offsets
         * 11. calculate tangent plane pointing
         * 12. calculate chunk indices
         */
        using namespace netCDF;
        using namespace netCDF::exceptions;

        // number of input files
        int n_files = rawobs.kidsdata().size();

        // network interfaces from config (may not match roach indices)
        Eigen::VectorXi interfaces(n_files);
        // roach indices for each file
        Eigen::VectorXi roach_indices(n_files);

        std::regex number_regex("(\\d+)$");

        // total number of detectors in raw files
        int raw_file_n_dets = 0;
        // detector, nw and array names for each network
        std::vector<int> raw_file_dets, raw_file_nws, raw_file_arrays;

        // adc snap data
        std::vector<Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>> adc_snap_data;

        // tone frquencies for each network
        std::map<int,Eigen::MatrixXd> tone_freqs;

        // get interfaces
        int i = 0;
        for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
            std::smatch match;
            if (std::regex_search(data_item.interface(), match, number_regex)) {
                interfaces(i) = std::stoi(match.str());
            }
            i++;
        }

        // load raw files to get required variables
        i = 0;
        for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
            try {
                // load data file
                NcFile fo(data_item.filepath(), NcFile::read);
                auto vars = fo.getVars();

                // get the number of detectors in current file
                int n_dets = vars.find("Data.Toltec.Is")->second.getDim(1).getSize();
                raw_file_n_dets += n_dets;

                // throw an exception if only one detector is found (weird kids behavior)
                if (n_dets == 1) {
                    throw std::runtime_error(fmt::format("only one detector found on {}.", data_item.filepath()));
                }

                // get the number of dets in file
                raw_file_dets.push_back(vars.find("Data.Toltec.Is")->second.getDim(1).getSize());
                // get the nw from interface
                raw_file_nws.push_back(interfaces(i));
                // get the array from the interface
                raw_file_arrays.push_back(engine().toltec.nw_to_array[interfaces(i)]);

                // get roach indices
                int roach_index;
                vars.find("Header.Toltec.RoachIndex")->second.getVar(&roach_index);
                roach_indices(i) = roach_index;

                // get LO center frequency
                double lo_center_freq;
                vars.find("Header.Toltec.LoCenterFreq")->second.getVar(&lo_center_freq);

                // dimension of tone freqs is (n_sweeps, n_tones)
                Eigen::Index n_sweeps = vars.find("Header.Toltec.ToneFreq")->second.getDim(0).getSize();

                // get tone_freqs for current interface
                tone_freqs[interfaces(i)].resize(vars.find("Header.Toltec.ToneFreq")->second.getDim(1).getSize(),n_sweeps);
                vars.find("Header.Toltec.ToneFreq")->second.getVar(tone_freqs[interfaces(i)].data());

                // add LO center freq
                tone_freqs[interfaces(i)] = tone_freqs[interfaces(i)].array() + lo_center_freq;

                // some files don't have adc snap data so don't exit by default
                auto adc_snap_data_var = fo.getVar("Header.Toltec.AdcSnapData");
                if (!adc_snap_data_var.isNull()) {
                    // dimension 0 of adc data
                    Eigen::Index adc_snap_dim = vars.find("Header.Toltec.AdcSnapData")->second.getDim(0).getSize();
                    // dimension 1 of adc data
                    Eigen::Index adc_snap_data_dim = vars.find("Header.Toltec.AdcSnapData")->second.getDim(1).getSize();

                    // matrix to hold adc data for current file
                    Eigen::Matrix<short,Eigen::Dynamic, Eigen::Dynamic> data_item_adc_snap(adc_snap_data_dim, adc_snap_dim);
                    // load adc data
                    vars.find("Header.Toltec.AdcSnapData")->second.getVar(data_item_adc_snap.data());
                    adc_snap_data.push_back(data_item_adc_snap);

                } else {
                    logger->debug("adc data not found for toltec{}", std::to_string(interfaces(i)));
                }

                i++;
                fo.close();

            } catch (NcException &e) {
                throw std::runtime_error(fmt::format("unable to open file {}: {}", data_item.filepath(), e.what()));
            }
        }

        // get photometry config values
        if (redu_type == "beammap") {
            logger->info("getting photometry config");
            get_photometry_configs(rawobs.photometry_calib_info().config());
        }

        // get telescope
        logger->info("getting telescope data");
        engine().telescope.load_telescope(rawobs.teldata().filepath());

        if (map_grouping == "uid") {
            // load apt from file
            logger->info("generating apt from file");

            // set columns to ones
            for (auto& [key, apt_column] : engine().toltec.apt.columns) {
                apt_column.data.setOnes(raw_file_n_dets);
            }

            // set amp and offsets to zero
            engine().toltec.apt["amp"].data.setZero();
            engine().toltec.apt["x_t"].data.setZero();
            engine().toltec.apt["y_t"].data.setZero();

            // set all flags to good
            engine().toltec.apt["flag"].data.setZero(raw_file_n_dets);

            // add the nws and arrays to the apt table
            int j = 0;
            for (int i = 0; i < raw_file_nws.size(); ++i) {
                engine().toltec.apt["nw"].data.segment(j, raw_file_dets[i]).setConstant(raw_file_nws[i]);
                engine().toltec.apt["array"].data.segment(j, raw_file_dets[i]).setConstant(raw_file_arrays[i]);

                j += raw_file_dets[i];
            }

            // set uids
            engine().toltec.apt["uid"].data = Eigen::VectorXd::LinSpaced(raw_file_n_dets, 0, raw_file_n_dets - 1);

            // filepath
            engine().toltec.apt.filepath = "internal";

            // init apt
            engine().toltec.apt.init();

        } else {
            // load apt from file
            logger->info("load apt");
            engine().toltec.apt.load(rawobs.array_prop_table().filepath(), interfaces);

            // check if number of detectors in apt file is equal to those in files
            // indicating mismatched apts
            if (raw_file_n_dets != engine().toltec.apt.n_dets) {
                throw std::runtime_error("Number of detectors in data files and apt file do not match");
            }

            // convert flxscale to requested units
            if (run_flux_calib) {
                engine().toltec.apt.rescale_fcf(units, engine().obs_maps.pix_size_radians);
            }
        }

        // copy tone freqs from file (temp?)
        i = 0;
        for (const auto& nw: engine().toltec.apt.nws) {
            engine().toltec.apt["tone_freq"].data.segment(i, tone_freqs[nw].size()) = tone_freqs[nw];
            i += tone_freqs[nw].size();
        }

        // flag nearby tones only if not in simu mode and not using internal apt
        if (!engine().telescope.sim_obs && map_grouping != "uid") {
            // frequency separation
            int n_dets = engine().toltec.apt.n_dets;
            Eigen::VectorXd d_freq(n_dets);

            // first separation
            d_freq(0) = engine().toltec.apt["tone_freq"].data(1) - engine().toltec.apt["tone_freq"].data(0);

            // loop through tone frequencies and find separation
            for (int i = 1; i < n_dets - 1; ++i) {
                d_freq(i) = std::min(abs(engine().toltec.apt["tone_freq"].data(i) - engine().toltec.apt["tone_freq"].data(i - 1)),
                                    abs(engine().toltec.apt["tone_freq"].data(i + 1) - engine().toltec.apt["tone_freq"].data(i)));
            }
            // get last separation
            d_freq(n_dets - 1) = abs(engine().toltec.apt["tone_freq"].data(n_dets - 1) - engine().toltec.apt["tone_freq"].data(n_dets - 2));

            // number of nearby tones found
            int n_nearby_tones = 0;

            // loop through flag columns
            for (int i = 0; i < n_dets; ++i) {
                // if closer than freq separation limit and unflagged, flag it
                if (d_freq(i) < delta_f_min_hz) {
                    engine().toltec.apt["flag"].data(i) = 1;
                    n_nearby_tones++;
                }
            }

            logger->info("{} nearby tones found. flagging.", n_nearby_tones);
        }

        // populate array classes within instrument class
        for (const auto& array: engine().toltec.apt.arrays) {
            engine().toltec.arrays[array] = ArrayContainer(array, engine().toltec.apt.filter_dets("array", array));
            engine().toltec.arrays[array].apt.init();
        }

        // populate network classes within array classes
        i = 0;
        for (const auto& nw: engine().toltec.apt.nws) {
            // get array name for current network.
            int array = engine().toltec.nw_to_array.at(nw);
            engine().toltec.arrays[array].networks[nw] = NetworkContainer(nw, engine().toltec.apt.filter_dets("nw", nw));
            engine().toltec.arrays[array].networks[nw].apt.init();
            engine().toltec.arrays[array].networks[nw].roach_id = roach_indices(i);
            if (!adc_snap_data.empty()) {
                engine().toltec.arrays[array].networks[nw].adc_snap_data = adc_snap_data[i];
            }
            i++;
        }

        // get HWPR data if polarized reduction is requested
        if (run_polarization) {
            // check if HWPR file is available and not ignored
            if (rawobs.hwpdata().has_value() && ignore_hwpr != "true") {
                const auto& hwpr_filepath = rawobs.hwpdata()->filepath();

                // if the HWPR filepath is valid, retrieve the data
                if (hwpr_filepath != "null") {
                    logger->info("getting HWPR file {}", hwpr_filepath);
                    engine().toltec.hwpr.load_hwpr(hwpr_filepath, engine().telescope.sim_obs);
                } else {
                    engine().toltec.hwpr.run_hwpr = false;
                }
            } else {
                engine().toltec.hwpr.run_hwpr = false;
            }
        }
        else {
            engine().toltec.hwpr.run_hwpr = false;
        }

        // align timestreams
        logger->info("aligning timestreams");
        align_timestreams(rawobs);

        // get astrometry config (pointing offsets and mjd)
        logger->info("getting astrometry config");
        engine().telescope.get_pointing_offsets(rawobs.astrometry_calib_info().config());

        // interpolate pointing offsets
        logger->info("interpolating pointing offsets");
        interpolate_pointing_offsets();

        // calc tangent plane pointing
        logger->info("calculating tangent plane pointing");
        engine().telescope.calc_tangent_plane_pointing();

        // calc chunk indices
        logger->info("calculating chunk indices");

        // get tod filter order for chunking
        int filter_order = 0;
        if (run_tod_filter) {
            auto result = engine().rtc_pipeline.get_component("Filter");

            if (result) {
                auto& [index, component] = result.value();
                auto filter_ptr = dynamic_cast<Filter<TCData>*>(component);
                if (filter_ptr) {
                    filter_order = filter_ptr->filter_order;
                }
            }
        }

        engine().telescope.calc_chunk_indices(engine().toltec.data_fs_hz, filter_order);
    }

    void setup_maps() {
        /**
         * @brief some basic map setup
         * 1. get map grouping
         * 2. populate basic map parameters for obs and coadd maps
         */
        // auto map grouping
        if (map_grouping == "auto") {
            // determine map grouping based on reduction type
            if (redu_type == "science" || redu_type == "pointing") {
                map_grouping = "array";
            } else if (redu_type == "beammap") {
                map_grouping = "uid";
            }
        }

        auto it = engine().toltec.apt.columns.find(map_grouping);
        if (it == engine().toltec.apt.columns.end()) {
            throw std::runtime_error(fmt::format("mapmaking grouping {} is not an apt key", map_grouping));
        }

        // populate map grouping
        engine().obs_maps.map_grouping = map_grouping;
        engine().coadd_maps.map_grouping = map_grouping;

        // optional map type controls
        engine().obs_maps.include_kernel = run_kernel;
        engine().obs_maps.include_coverage = (map_grouping != "uid");
        engine().obs_maps.include_polarization = run_polarization;

        engine().coadd_maps.include_kernel = run_kernel;
        engine().coadd_maps.include_coverage = (map_grouping != "uid");
        engine().coadd_maps.include_polarization = run_polarization;

        // set map container pixel sizes
        engine().obs_maps.pix_size_radians = pix_size_arcsec * ASEC_TO_RAD;
        engine().coadd_maps.pix_size_radians = pix_size_arcsec * ASEC_TO_RAD;

        // set map container noise maps (default 0)
        if (run_noise_maps) {
            engine().noise_maps.include_polarization = run_polarization;
            engine().noise_maps.n_noise_maps = n_noise_maps;
        }
    }

    void setup_obs_maps() {
        /**
         * @brief some basic map setup for obs maps
         * 1. get unique map keys
         * 2. get number of maps
         * determine map dimensions either by using config values or by looping through detectors and finding min and max pointing
         * store dimensions and coordinates in vectors
         */
        // find unique keys
        map_keys = find_unique_elements<Eigen::VectorXi, Eigen::VectorXi>(
            engine().toltec.apt[map_grouping].data.template cast<int>());

        auto arrays = engine().toltec.apt["array"].data.template cast<int>();
        auto group = engine().toltec.apt[map_grouping].data.template cast<int>();
        // these are the keys that belong to each array
        unique_map_keys = find_corresponding_unique_elements(arrays, group);

        // get total number of maps
        n_maps = map_keys.size();

        // use config file map dimensions
        if (config_wcs.naxis[0] != 0 && config_wcs.naxis[1] != 0) {

            int n_rows = config_wcs.naxis[1];
            int n_cols = config_wcs.naxis[0];

            n_rows = (n_rows % 2 == 0) ? n_rows + 1 : n_rows;
            n_cols = (n_cols % 2 == 0) ? n_cols + 1 : n_cols;

            Eigen::VectorXd col_coords = Eigen::VectorXd::LinSpaced(n_cols, 0, n_cols - 1)
                                                 .array() * engine().obs_maps.pix_size_radians - (n_cols / 2.0) * engine().obs_maps.pix_size_radians;
            Eigen::VectorXd row_coords = Eigen::VectorXd::LinSpaced(n_rows, 0, n_rows - 1)
                                                 .array() * engine().obs_maps.pix_size_radians - (n_rows / 2.0) * engine().obs_maps.pix_size_radians;

            map_extents.emplace_back(n_rows, n_cols);
            map_coords.emplace_back(row_coords, col_coords);
        }
        else {
            // mins and maxes for all detectors over the entire tod
            Eigen::MatrixXd x_lim, y_lim;

            x_lim.setZero(engine().toltec.apt.n_dets, 2);
            y_lim.setZero(engine().toltec.apt.n_dets, 2);

            std::map<std::string, Eigen::VectorXd> tel_data_chunk;

            // loop through time chunks and find max and min of each detector pointing
            for (int i = 0; i < engine().telescope.n_chunks; ++i) {
                // start index of current inner chunk
                auto start_index = engine().telescope.chunk_indices(0,i);
                auto chunk_size = engine().telescope.chunk_indices(1,i) - start_index + 1;

                for (auto const& [key, value]: engine().telescope.data) {
                    tel_data_chunk[key] = value.segment(start_index, chunk_size);
                }

                // don't run on all detectors if in detector mode
                int loop_range = (map_grouping == "uid") ? 1 : engine().toltec.apt.n_dets;

                for (int det = 0; det < loop_range; ++det) {
                    if (!engine().toltec.apt["flag"].data(det)) {
                        auto xy = engine().telescope.calc_pointing(engine().toltec.apt["x_t"].data(det),
                                                                   engine().toltec.apt["y_t"].data(det), tel_data_chunk);

                        // calculate the min and max coefficients for pointing
                        double x_min = xy.first.minCoeff();
                        double x_max = xy.first.maxCoeff();
                        double y_min = xy.second.minCoeff();
                        double y_max = xy.second.maxCoeff();

                        // update x and y limits for current detector
                        x_lim(det, 0) = std::min(x_lim(det, 0), x_min);
                        x_lim(det, 1) = std::max(x_lim(det, 1), x_max);
                        y_lim(det, 0) = std::min(y_lim(det, 0), y_min);
                        y_lim(det, 1) = std::max(y_lim(det, 1), y_max);
                    }
                }
            }

            // determine global x and y min and max
            double x_min = x_lim.col(0).minCoeff();
            double x_max = x_lim.col(1).maxCoeff();
            double y_min = y_lim.col(0).minCoeff();
            double y_max = y_lim.col(1).maxCoeff();

            auto [n_cols, n_rows, col_coords, row_coords] = calc_map_shape(x_min, x_max, y_min, y_max,
                                                                           engine().obs_maps.pix_size_radians);
            map_extents.emplace_back(n_rows, n_cols);
            map_coords.emplace_back(row_coords, col_coords);
        }
    }

    void allocate_coadded_maps() {
        /**
         * @brief determine the coadded map size and allocate them.  Map dimensions are found from the minimum overlapped
         * region of all individual observation maps.
         */
        // initialize min/max values
        double x_min = std::numeric_limits<double>::max();
        double x_max = std::numeric_limits<double>::lowest();
        double y_min = x_min;
        double y_max = x_max;

        // find global min/max for rows and columns
        for (const auto& coord : map_coords) {
            x_min = std::min(x_min, coord.first.minCoeff());
            x_max = std::max(x_max, coord.first.maxCoeff());
            y_min = std::min(y_min, coord.second.minCoeff());
            y_max = std::max(y_max, coord.second.maxCoeff());
        }

        auto [n_cols, n_rows, col_coords, row_coords] = calc_map_shape(x_min, x_max, y_min, y_max,
                                                                       engine().coadd_maps.pix_size_radians);

        // set up coaded map buffer
        engine().coadd_maps.n_rows = n_rows;
        engine().coadd_maps.n_cols = n_cols;
        if (redu_type != "science") {
            engine().coadd_maps.n_params = citlali::utils::models::Gaussian2DModel::nparams;
        }
        engine().coadd_maps.row_coords = row_coords;
        engine().coadd_maps.col_coords = col_coords;

        for (const auto& array :engine().toltec.apt.arrays) {
            engine().coadd_maps.init_array(array, unique_map_keys[array]);

            if (redu_type != "science") {
                engine().coadd_maps.init_fit(array, unique_map_keys[array]);
            }
        }

        // build eigen map vectors to signal, weight, kernel, and coverage maps
        engine().coadd_maps.build_vectors();
    }

    void allocate_noise_maps(DataMapsContainer &data_maps) {
        engine().noise_maps.n_rows = data_maps.n_rows;
        engine().noise_maps.n_cols = data_maps.n_cols;
        engine().noise_maps.row_coords = data_maps.row_coords;
        engine().noise_maps.col_coords = data_maps.col_coords;

        for (const auto& array :engine().toltec.apt.arrays) {
            engine().noise_maps.init_array(array, unique_map_keys[array]);
        }

        engine().noise_maps.build_vectors();
    }

    void setup_directories() {
        /**
         * @brief create the reduction subdirectories and all of their subdirectories (obsnum, coadd, raw, filtered, etc).
         */
        using namespace tula::filename_utils::fs;

        // start from config filepath
        engine().reduction_directory = output_directory_base + "/";

        // only make sub directories if requested
        if (use_subdir) {
            int reduction_number = 0;

            // start with redu00
            std::string test_subdir_name = subdir_name + fmt::format("{:02}", reduction_number);

            // while current redu directory exists, increment
            while (fs::exists(engine().reduction_directory + test_subdir_name)) {
                reduction_number++;
                test_subdir_name =  subdir_name + fmt::format("{:02}", reduction_number);
            }

            // update reduction directory
            engine().reduction_directory = engine().reduction_directory + test_subdir_name + "/";

            fs::create_directories(engine().reduction_directory);

            // make reduction directories for each obsnum
            for (const auto& obsnum : obsnums) {
                fs::create_directories(engine().reduction_directory + obsnum + "/raw/");

                // make filtered directory if requested and not coadding
                if (run_map_filter && !run_map_coadd) {
                    fs::create_directories(engine().reduction_directory + obsnum + "/filtered/");
                }
            }
            // make coadded directories
            if (run_map_coadd) {
                fs::create_directories(engine().reduction_directory + "coadded/raw/");
                if (run_map_filter) {
                    fs::create_directories(engine().reduction_directory + "coadded/filtered/");
                }
            }
        }
    }

    void make_index_file(std::string& filepath) {
        /**
         * @brief create index files that list all directories and files within output directory.
         * Called recursively.
        */
        using namespace tula::filename_utils::fs;

        // get sortedfiles and directories in filepath
        std::set<fs::path> sorted_by_name;
        for (auto &entry : fs::directory_iterator(filepath))
            sorted_by_name.insert(entry);

        // yaml node to store names
        YAML::Node node;
        // data products
        node["description"].push_back("citlali data products");
        // datetime when file is created
        node["date"].push_back(citlali::utils::timing::current_date_time());
        // citlali version
        node["citlali_version"].push_back(CITLALI_GIT_VERSION);
        // kids version
        node["kids_version"].push_back(KIDSCPP_GIT_VERSION);
        // tula version
        node["tula_version"].push_back(TULA_GIT_VERSION);

        // call make_index_file recursively if current object is directory
        for (const auto & entry : sorted_by_name) {
            std::string path_string{entry.generic_string()};
            if (fs::is_directory(entry)) {
                make_index_file(path_string);
            }
            node["files/dirs"].push_back(path_string.substr(path_string.find_last_of("/") + 1));
        }
        // output yaml index file
        std::ofstream fout(filepath + "/index.yaml");
        fout << node;
    }

    Engine& engine() { return m_engine; }

    const Engine &engine() const { return m_engine; }

    template <typename OStream>
    friend OStream &operator<<(OStream &os,
                               const TimeOrderedDataProc &todproc) {
        return os << fmt::format("TimeOrderedDataProc(engine={})",
                                 todproc.engine());
    }

private:
    Engine m_engine;

    // calculate map dimensions and coordinates
    std::tuple<int, int, Eigen::VectorXd, Eigen::VectorXd> calc_map_shape(
        double x_min, double x_max, double y_min, double y_max, double pixel_size_radians) {
        /**
         * @brief helper function for calculating map dimensions and coordinates.
        */

        // calculate the number of columns and rows based on the pixel size and coordinate range
        int n_cols = 2 * std::max(static_cast<int>(ceil(abs(x_min / pixel_size_radians))),
                                  static_cast<int>(ceil(abs(x_max / pixel_size_radians)))) + 1;

        int n_rows = 2 * std::max(static_cast<int>(ceil(abs(y_min / pixel_size_radians))),
                                  static_cast<int>(ceil(abs(y_max / pixel_size_radians)))) + 1;

        // generate row and column coordinate vectors
        Eigen::VectorXd col_coords = Eigen::VectorXd::LinSpaced(n_cols, 0, n_cols - 1)
                                             .array() * pixel_size_radians - (n_cols / 2.0) * pixel_size_radians;
        Eigen::VectorXd row_coords = Eigen::VectorXd::LinSpaced(n_rows, 0, n_rows - 1)
                                             .array() * pixel_size_radians - (n_rows / 2.0) * pixel_size_radians;

        return std::make_tuple(n_cols, n_rows, std::move(row_coords), std::move(col_coords));
    }
};
