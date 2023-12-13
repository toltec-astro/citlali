#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

#include <tula/eigen.h>

#include <citlali/core/utils/pointing.h>
#include <tula/algorithm/mlinterp/mlinterp.hpp>

#include <citlali/core/utils/fits_io.h>
#include <citlali/core/utils/netcdf_io.h>
#include <citlali/core/utils/toltec_io.h>

namespace fs = std::filesystem;

struct DummyEngine {
    template <typename OStream>
    friend OStream &operator<<(OStream &os, const DummyEngine &e) {
        return os << fmt::format("DummyEngine()");
    }
};

/**
 * @brief The time ordered data processing struct
 * This wraps around the lali config
 */

template <class EngineType>
struct TimeOrderedDataProc : ConfigMapper<TimeOrderedDataProc<EngineType>> {
    using Base = ConfigMapper<TimeOrderedDataProc<EngineType>>;
    using config_t = typename Base::config_t;
    using Engine = EngineType;
    using scanindicies_t = Eigen::MatrixXI;
    using map_extent_t = std::vector<int>;
    using map_coord_t = std::vector<Eigen::VectorXd>;
    using map_count_t = std::size_t;
    using array_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;
    using det_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    TimeOrderedDataProc(config_t config) : Base{std::move(config)} {}

    // check if config file has nodes
    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        // get logger
        std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

        std::vector<std::string> missing_keys;
        logger->info("check TOD proc config\n{}", config);
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

    // create output FITS files (does not populate)
    void create_coadded_map_files();
    // get apt from raw data files (beammapping)
    void get_apt_from_files(const RawObs &rawobs);
    // get tone frequencies from raw files
    void get_tone_freqs_from_files(const RawObs &rawobs);
    // get adc snap data from raw files
    void get_adc_snap_from_files(const RawObs &rawobs);
    // create output directories
    void create_output_dir();
    // count up detectors from input files and check for mismatch with apt
    void check_inputs(const RawObs &rawobs);
    // align networks and hwpr vectors in time
    void align_timestreams(const RawObs &rawobs);
    // updated alignment of networks and hwpr vectors in time
    void align_timestreams_2(const RawObs &rawobs);
    // interpolate pointing vectors
    void interp_pointing();
    // calculate number of maps
    void calc_map_num();
    // calculate size of omb maps
    void calc_omb_size(std::vector<map_extent_t> &, std::vector<map_coord_t> &);
    // allocate observation maps
    void allocate_omb(map_extent_t &, map_coord_t &);
    // calculate size of cmb maps
    void calc_cmb_size(std::vector<map_coord_t> &);
    // allocate coadded maps
    void allocate_cmb();
    // allocate noise maps
    template<class map_buffer_t>
    void allocate_nmb(map_buffer_t &);
    // coadd omb into cmb
    void coadd();
    // make index files
    void make_index_file(std::string);

    // TODO fix the const correctness
    Engine &engine() { return m_engine; }

    const Engine &engine() const { return m_engine; }

    template <typename OStream>
    friend OStream &operator<<(OStream &os,
                               const TimeOrderedDataProc &todproc) {
        return os << fmt::format("TimeOrderedDataProc(engine={})",
                                 todproc.engine());
    }

private:
    Engine m_engine;
};

// make apt table from raw files instead of an ecsv table
template <class EngineType>
void TimeOrderedDataProc<EngineType>::get_apt_from_files(const RawObs &rawobs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    // nw names
    std::vector<Eigen::Index> interfaces;

    // total number of detectors
    Eigen::Index n_dets = 0;
    // detector, nw and array names for each network
    std::vector<Eigen::Index> dets, nws, arrays;
    // loop through input files
    for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            // load data file
            NcFile fo(data_item.filepath(), NcFile::read);
            auto vars = fo.getVars();

            // get the interface
            interfaces.push_back(std::stoi(data_item.interface().substr(6)));

            // add the current file's number of dets to the total
            n_dets += vars.find("Data.Toltec.Is")->second.getDim(1).getSize();

            // get the number of dets in file
            dets.push_back(vars.find("Data.Toltec.Is")->second.getDim(1).getSize());
            // get the nw from interface
            nws.push_back(interfaces.back());
            // get the array from the interface
            arrays.push_back(engine().toltec_io.nw_to_array_map[interfaces.back()]);

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
        engine().calib.apt[key].setOnes(n_dets);
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

    // nw names
    std::vector<Eigen::Index> interfaces;

    // total number of detectors
    Eigen::Index n_dets = 0;
    // detector, nw and array names for each network
    std::vector<Eigen::Index> dets, nws, arrays;
    // loop through input files
    for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            // load data file
            NcFile fo(data_item.filepath(), NcFile::read);
            auto vars = fo.getVars();

            // get the interface
            interfaces.push_back(std::stoi(data_item.interface().substr(6)));

            // add the current file's number of dets to the total
            n_dets += vars.find("Data.Toltec.Is")->second.getDim(1).getSize();

            // dimension of tone freqs is (n_sweeps, n_tones)
            Eigen::Index n_sweeps = vars.find("Header.Toltec.ToneFreq")->second.getDim(0).getSize();

            // get local oscillator frequency
            double lo_freq;
            vars.find("Header.Toltec.LoCenterFreq")->second.getVar(&lo_freq);

            // get tone_freqs for interface
            tone_freqs[interfaces.back()].resize(vars.find("Header.Toltec.ToneFreq")->second.getDim(1).getSize(),n_sweeps);
            vars.find("Header.Toltec.ToneFreq")->second.getVar(tone_freqs[interfaces.back()].data());

            // add local oscillator freq
            tone_freqs[interfaces.back()] = tone_freqs[interfaces.back()].array() + lo_freq;

            fo.close();

        } catch (NcException &e) {
            logger->error("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", data_item.filepath())};
        }
    }

    engine().calib.apt["tone_freq"].resize(engine().calib.n_dets);

    // add the nws and arrays to the apt table
    Eigen::Index j = 0;
    for (Eigen::Index i=0; i<engine().calib.nws.size(); ++i) {
        engine().calib.apt["tone_freq"].segment(j,tone_freqs[interfaces[i]].size()) = tone_freqs[interfaces[i]];

        j = j + tone_freqs[interfaces[i]].size();
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

template <class EngineType>
void TimeOrderedDataProc<EngineType>::get_adc_snap_from_files(const RawObs &rawobs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    // explicitly clear adc vector
    engine().diagnostics.adc_snap_data.clear();

    // loop through input files
    for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            // load data file
            NcFile fo(data_item.filepath(), NcFile::read);
            auto vars = fo.getVars();

            // dimension 0 of adc data
            Eigen::Index adcSnapDim = vars.find("Header.Toltec.AdcSnapData")->second.getDim(0).getSize();
            // dimension 1 of adc data
            Eigen::Index adcSnapDataDim = vars.find("Header.Toltec.AdcSnapData")->second.getDim(1).getSize();

            // matrix to hold adc data for current file
            Eigen::Matrix<short,Eigen::Dynamic, Eigen::Dynamic> adcsnap(adcSnapDataDim,adcSnapDim);
            // load adc data
            vars.find("Header.Toltec.AdcSnapData")->second.getVar(adcsnap.data());
            // append to vector of adc data
            engine().diagnostics.adc_snap_data.push_back(adcsnap);

            fo.close();

        } catch (NcException &e) {
            logger->warn("{} adc data not found",data_item.filepath());
        }
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
    }
    else {
        engine().redu_dir_name = engine().output_dir + "/";
    }

    // coadded subdir
    if (engine().run_coadd) {
        engine().coadd_dir_name = engine().redu_dir_name + "/coadded/";
        // coadded raw subdir
        if (!fs::exists(fs::status(engine().coadd_dir_name + "raw/"))) {
            fs::create_directories(engine().coadd_dir_name + "raw/");
        }
        else {
            logger->warn("directory {} already exists", engine().coadd_dir_name + "raw/");
        }
        // if map filtering is requested
        if (engine().run_map_filter) {
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
            auto vars = fo.getVars();
            // get number of dets from data and add to global value
            n_dets += vars.find("Data.Toltec.Is")->second.getDim(1).getSize();

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
template <class EngineType>
void TimeOrderedDataProc<EngineType>::align_timestreams(const RawObs &rawobs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    // clear start and end indices for each observation
    engine().start_indices.clear();
    engine().end_indices.clear();

    // clear gaps
    engine().gaps.clear();

    // vector of network times
    std::vector<Eigen::VectorXd> nw_ts;
    // start and end times
    std::vector<double> nw_t0, nw_tn;

    // maximum start time
    double max_t0 = -99;

    // minimum end time
    double min_tn = std::numeric_limits<double>::max();
    // indices of max start time and min end time
    Eigen::Index max_t0_i, min_tn_i;

    // set network
    Eigen::Index nw = 0;
    // sample rate
    double fsmp = -1;

    // loop through input files
    for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            // load data file
            NcFile fo(data_item.filepath(), NcFile::read);
            auto vars = fo.getVars();

            // get roach index for offsets
            int roach_index;
            vars.find("Header.Toltec.RoachIndex")->second.getVar(&roach_index);

            // get sample rate
            double fsmp_roach;
            vars.find("Header.Toltec.SampleFreq")->second.getVar(&fsmp_roach);

            // check if sample rate is the same and exit if not
            if (fsmp!=-1 && fsmp_roach!=fsmp) {
                logger->error("mismatched sample rate in toltec{}",roach_index);
                std::exit(EXIT_FAILURE);
            }
            else {
                fsmp = fsmp_roach;
            }

            // get dimensions for time matrix
            Eigen::Index n_pts = vars.find("Data.Toltec.Ts")->second.getDim(0).getSize();
            Eigen::Index n_time = vars.find("Data.Toltec.Ts")->second.getDim(1).getSize();

            // get time matrix
            Eigen::MatrixXi ts(n_time,n_pts);
            vars.find("Data.Toltec.Ts")->second.getVar(ts.data());

            // transpose due to row-major order
            ts.transposeInPlace();

            // find gaps
            int gaps = ((ts.block(1,3,n_pts,1).array() - ts.block(0,3,n_pts-1,1).array()).array() > 1).count();

            // add gaps to engine map
            if (gaps>0) {
                engine().gaps["Toltec" + std::to_string(roach_index)] = gaps;
            }

            // get fpga frequency
            double fpga_freq;
            vars.find("Header.Toltec.FpgaFreq")->second.getVar(&fpga_freq);

            // ClockTime (sec)
            auto sec0 = ts.cast <double> ().col(0);
            // ClockTimeNanoSec (nsec)
            auto nsec0 = ts.cast <double> ().col(5);
            // PpsCount (pps ticks)
            auto pps = ts.cast <double> ().col(1);
            // ClockCount (clock ticks)
            auto msec = ts.cast <double> ().col(2)/fpga_freq;
            // PacketCount (packet ticks)
            auto count = ts.cast <double> ().col(3);
            // PpsTime (clock ticks)
            auto pps_msec = ts.cast <double> ().col(4)/fpga_freq;
            // get start time
            auto t0 = sec0 + nsec0*1e-9;

            // shift start time (offset determined empirically)
            int start_t = int(t0[0] - 0.5);
            //int start_t = int(t0[0]);

            // convert start time to double
            double start_t_dbl = start_t;
            // clock count - clock ticks
            Eigen::VectorXd dt = msec - pps_msec;
            // remove overflow due to int32
            dt = (dt.array() < 0).select(msec.array() - pps_msec.array() + (pow(2.0,32)-1)/fpga_freq,msec - pps_msec);
            // get network time and add offsets
            nw_ts.push_back(start_t_dbl + pps.array() + dt.array() +
                            engine().interface_sync_offset["toltec"+std::to_string(roach_index)]);

            // push back start time
            nw_t0.push_back(nw_ts.back()[0]);

            // push back end time
            nw_tn.push_back(nw_ts.back()[n_pts - 1]);

            // get global max start time and index
            if (nw_t0.back() > max_t0) {
                max_t0 = nw_t0.back();
                max_t0_i = nw;
            }

            // get global min end time and index
            if (nw_tn.back() < min_tn) {
                min_tn = nw_tn.back();
                min_tn_i = nw;
            }

            // increment nw
            nw++;

            fo.close();

        } catch (NcException &e) {
            logger->error("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", data_item.filepath())};
        }
    }

    // get hwpr timing
    if (engine().calib.run_hwpr) {
        /*auto sec0 = engine().calib.hwpr_ts.template cast <double> ().col(0);
        // ClockTimeNanoSec (nsec)
        auto nsec0 = engine().calib.hwpr_ts.template cast <double> ().col(5);
        // PpsCount (pps ticks)
        auto pps = engine().calib.hwpr_ts.template cast <double> ().col(1);
        // ClockCount (clock ticks)
        auto msec = engine().calib.hwpr_ts.template cast <double> ().col(2)/engine().calib.hwpr_fpga_freq;
        // PacketCount (packet ticks)
        auto count = engine().calib.hwpr_ts.template cast <double> ().col(3);
        // PpsTime (clock ticks)
        auto pps_msec = engine().calib.hwpr_ts.template cast <double> ().col(4)/engine().calib.hwpr_fpga_freq;
        // get start time
        auto t0 = sec0 + nsec0*1e-9;

        // shift start time
        int start_t = int(t0[0] - 0.5);
        //int start_t = int(t0[0]);

        // convert start time to double
        double start_t_dbl = start_t;

        Eigen::VectorXd dt = msec - pps_msec;

        // remove overflow due to int32
        dt = (dt.array() < 0).select(msec.array() - pps_msec.array() + (pow(2.0,32)-1)/engine().calib.hwpr_,msec - pps_msec);

        // get network time and add offsets
        engine().calib.hwpr_recvt = start_t_dbl + pps.array() + dt.array() + engine().interface_sync_offset["hwpr"];
        */

        // if hwpr init time is larger than max start time, replace global max start time
        Eigen::Index hwpr_ts_n_pts = engine().calib.hwpr_recvt.size();
        if (engine().calib.hwpr_recvt(0) > max_t0) {
            max_t0 = engine().calib.hwpr_recvt(0);
        }

        // if hwpr init time is smaller than min end time, replace global min end time
        if (engine().calib.hwpr_recvt(hwpr_ts_n_pts - 1) < min_tn) {
            min_tn = engine().calib.hwpr_recvt(hwpr_ts_n_pts - 1);
        }
    }

    // size of smallest data time vector
    Eigen::Index min_size = nw_ts[0].size();

    // loop through time vectors and get the smallest
    for (Eigen::Index i=0; i<nw_ts.size(); ++i) {
        // find start index that is larger than max start
        Eigen::Index si, ei;
        // find index closest to max start time
        auto s = (abs(nw_ts[i].array() - max_t0)).minCoeff(&si);

        // if closest index is smaller than max start time
        // incrememnt index until it is larger or equal
        while (nw_ts[i][si] < max_t0) {
            si++;
        }
        // pushback start index on start index vector
        engine().start_indices.push_back(si);

        // find end index that is smaller than min end
        auto e = (abs(nw_ts[i].array() - min_tn)).minCoeff(&ei);
        // if closest index is larger than min end time
        // incrememnt index until it is smaller or equal
        while (nw_ts[i][ei] > min_tn) {
            ei--;
        }
        // pushback end index on end index vector
        engine().end_indices.push_back(ei);
    }

    // get min size
    for (Eigen::Index i=0; i<nw_ts.size(); ++i) {
        // start indices
        auto si = engine().start_indices[i];
        // end indices
        auto ei = engine().end_indices[i];

        // if smallest length, update min_size
        if ((ei - si + 1) < min_size) {
            min_size = ei - si + 1;
        }
    }

    // if hwpr requested
    if (engine().calib.run_hwpr) {
        Eigen::Index si, ei;
        // find start index that is larger than max start for hwpr
        auto s = (abs(engine().calib.hwpr_recvt.array() - max_t0)).minCoeff(&si);

        // if closest index is smaller than max start time
        // incrememnt index until it is larger or equal
        while (engine().calib.hwpr_recvt(si) < max_t0) {
            si++;
        }
        // pushback start index on hwpr start index vector
        engine().hwpr_start_indices = si;

        // find end index that is smaller than min end for hwpr
        auto e = (abs(engine().calib.hwpr_recvt.array() - min_tn)).minCoeff(&ei);
        // if closest index is larger than min end time
        // incrememnt index until it is smaller or equal
        while (engine().calib.hwpr_recvt(ei) > min_tn) {
            ei--;
        }
        // pushback end index on hwpr end index vector
        engine().hwpr_end_indices = ei;

        // update min_size for all time vectors if hwpr data is shorter (data and hwpr)
        if ((ei - si + 1) < min_size) {
            min_size = ei - si + 1;
        }
    }

    // size of telescope data
    Eigen::Matrix<Eigen::Index,1,1> nd;
    nd << engine().telescope.tel_data["TelTime"].size();

    // shortest data time vector
    Eigen::VectorXd xi = nw_ts[max_t0_i].head(min_size);

    // interpolate telescope data
    for (const auto &tel_it : engine().telescope.tel_data) {
        if (tel_it.first !="TelTime") {
            // telescope vector to interpolate
            Eigen::VectorXd yd = engine().telescope.tel_data[tel_it.first];
            // vector to store interpolated outputs in
            Eigen::VectorXd yi(min_size);

            mlinterp::interp(nd.data(), min_size, // nd, ni
                             yd.data(), yi.data(), // yd, yi
                             engine().telescope.tel_data["TelTime"].data(), xi.data()); // xd, xi

            // move back into tel_data vector
            engine().telescope.tel_data[tel_it.first] = std::move(yi);
        }
    }

    // replace telescope time vectors
    engine().telescope.tel_data["TelTime"] = xi;
    engine().telescope.tel_data["TelUTC"] = xi;

    // interpolate hwpr data
    if (engine().calib.run_hwpr) {
        Eigen::VectorXd yd = engine().calib.hwpr_angle;
        // vector to store interpolated outputs in
        Eigen::VectorXd yi(min_size);
        mlinterp::interp(nd.data(), min_size, // nd, ni
                         yd.data(), yi.data(), // yd, yi
                         engine().calib.hwpr_recvt.data(), xi.data()); // xd, xi

        // move back into hwpr angle
        engine().calib.hwpr_angle = std::move(yi);
    }
}

// upgraded alignment of tod with telescope
template <class EngineType>
void TimeOrderedDataProc<EngineType>::align_timestreams_2(const RawObs &rawobs) {}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::interp_pointing() {
    // how many offsets in config file
    Eigen::Index n_offsets = engine().pointing_offsets_arcsec["az"].size();

    // keys for pointing offsets
    std::vector<std::string> altaz_keys = {"alt", "az"};

    for (const auto &key: altaz_keys) {
        // if only one value given
        if (n_offsets==1) {
            double offset = engine().pointing_offsets_arcsec[key](0);
            engine().pointing_offsets_arcsec[key].resize(engine().telescope.tel_data["TelTime"].size());
            engine().pointing_offsets_arcsec[key].setConstant(offset);
        }
        else if (n_offsets==2) {
            // size of telescope data
            Eigen::Index ni = engine().telescope.tel_data["TelTime"].size();

            // size of telescope data
            Eigen::Matrix<Eigen::Index,1,1> nd;
            nd << n_offsets;

            // vector to store interpolation
            Eigen::VectorXd yi(ni);

            // start and end times of observation
            Eigen::VectorXd xd(n_offsets);
            // use start and end of current obs if julian dates not specified
            if ((engine().pointing_offsets_modified_julian_date<=0).any()) {
                xd << engine().telescope.tel_data["TelTime"](0), engine().telescope.tel_data["TelTime"](ni-1);
            }
            // else use specified modified julian dates, convert to julian dates, and calc unix time
            else {
                xd << engine_utils::modified_julian_date_to_unix(engine().pointing_offsets_modified_julian_date(0)),
                    engine_utils::modified_julian_date_to_unix(engine().pointing_offsets_modified_julian_date(1));

                // make sure offsets are before and after the observation
                if (xd(0) > engine().telescope.tel_data["TelTime"](0) || xd(1) < engine().telescope.tel_data["TelTime"](ni-1)) {
                    logger->error("offsets are out of range");
                    std::exit(EXIT_FAILURE);
                }
            }

            // interpolate offset onto time vector
            mlinterp::interp(nd.data(), ni, // nd, ni
                             engine().pointing_offsets_arcsec[key].data(), yi.data(), // yd, yi
                             xd.data(), engine().telescope.tel_data["TelTime"].data()); // xd, xi

            // overwrite pointing offsets
            engine().pointing_offsets_arcsec[key] = yi;

        }
        else {
            logger->error("only one or two values for altaz offsets are supported");
            std::exit(EXIT_FAILURE);
        }
    }
}

// get map number
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

    Eigen::VectorXd rows_tan_vec = Eigen::VectorXd::LinSpaced(omb.n_rows, 0, omb.n_rows - 1).array() * omb.pixel_size_rad -
                                   (omb.n_rows - 1) / 2.0 * omb.pixel_size_rad;
    Eigen::VectorXd cols_tan_vec = Eigen::VectorXd::LinSpaced(omb.n_cols, 0, omb.n_cols - 1).array() * omb.pixel_size_rad -
                                   (omb.n_cols - 1) / 2.0 * omb.pixel_size_rad;


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
        Eigen::VectorXd dim_vec = Eigen::VectorXd::LinSpaced(n_dim, 0, n_dim - 1)
                                          .array() * engine().cmb.pixel_size_rad - (n_dim - 1) / 2.0 * engine().cmb.pixel_size_rad;

        return std::make_tuple(n_dim, std::move(dim_vec));
    };

    // get dimensions and tangent coordinate vectorx
    auto [n_rows, rows_tan_vec] = calc_map_dims(min_row, max_row);
    auto [n_cols, cols_tan_vec] = calc_map_dims(min_col, max_col);

    // Set dimensions and wcs parameters
    cmb.n_rows = n_rows;
    cmb.n_cols = n_cols;
    cmb.wcs.naxis[1] = n_rows;
    cmb.wcs.naxis[0] = n_cols;
    cmb.wcs.crpix[0] = (n_cols - 1) / 2.0;
    cmb.wcs.crpix[1] = (n_rows - 1) / 2.0;
    cmb.rows_tan_vec = std::move(rows_tan_vec);
    cmb.cols_tan_vec = std::move(cols_tan_vec);
}

// allocate observation map buffer
template <class EngineType>
void TimeOrderedDataProc<EngineType>::allocate_omb(map_extent_t &map_extent, map_coord_t &map_coord) {
    auto& omb = engine().omb;

    std::vector<Eigen::MatrixXd>().swap(omb.signal);
    std::vector<Eigen::MatrixXd>().swap(omb.weight);
    std::vector<Eigen::MatrixXd>().swap(omb.kernel);
    std::vector<Eigen::MatrixXd>().swap(omb.coverage);
    std::vector<Eigen::Tensor<double,3>>().swap(omb.pointing);

    // set omb dimensions and wcs parameters
    omb.n_rows = map_extent[0];
    omb.n_cols = map_extent[1];
    omb.wcs.naxis[1] = omb.n_rows;
    omb.wcs.naxis[0] = omb.n_cols;
    omb.wcs.crpix[0] = (omb.n_cols - 1) / 2.0;
    omb.wcs.crpix[1] = (omb.n_rows - 1) / 2.0;

    // allocate and initialize matrices
    Eigen::MatrixXd zero_matrix = Eigen::MatrixXd::Zero(omb.n_rows, omb.n_cols);

    for (Eigen::Index i=0; i<engine().n_maps; ++i) {
        omb.signal.push_back(zero_matrix);
        omb.weight.push_back(zero_matrix);

        if (engine().rtcproc.run_kernel) {
            omb.kernel.push_back(zero_matrix);
        }

        if (engine().map_grouping != "detector") {
            omb.coverage.push_back(zero_matrix);
        }
    }

    if (engine().rtcproc.run_polarization) {
        // allocate pointing matrix
        for (Eigen::Index i=0; i<engine().n_maps/engine().rtcproc.polarization.stokes_params.size(); ++i) {
            omb.pointing.emplace_back(omb.n_rows, omb.n_cols, 9);
            engine().omb.pointing.back().setZero();
        }
    }

    // set tangent plane coordinate vectors
    omb.rows_tan_vec = map_coord[0];
    omb.cols_tan_vec = map_coord[1];
}

// allocate the coadded map buffer
template <class EngineType>
void TimeOrderedDataProc<EngineType>::allocate_cmb() {
    auto& cmb = engine().cmb;

    // clear map vectors
    std::vector<Eigen::MatrixXd>().swap(cmb.signal);
    std::vector<Eigen::MatrixXd>().swap(cmb.weight);
    std::vector<Eigen::MatrixXd>().swap(cmb.kernel);
    std::vector<Eigen::MatrixXd>().swap(cmb.coverage);
    std::vector<Eigen::Tensor<double,3>>().swap(cmb.pointing);

    // loop through maps and allocate space
    for (Eigen::Index i=0; i<engine().n_maps; ++i) {
        cmb.signal.push_back(Eigen::MatrixXd::Zero(cmb.n_rows, cmb.n_cols));
        cmb.weight.push_back(Eigen::MatrixXd::Zero(cmb.n_rows, cmb.n_cols));

        if (engine().rtcproc.run_kernel) {
            // allocate kernel
            cmb.kernel.push_back(Eigen::MatrixXd::Zero(cmb.n_rows, cmb.n_cols));
        }

        if (engine().map_grouping!="detector") {
            // allocate coverage
            cmb.coverage.push_back(Eigen::MatrixXd::Zero(cmb.n_rows, cmb.n_cols));
        }
    }

    if (engine().rtcproc.run_polarization && engine().run_noise) {
        // allocate pointing matrix
        for (Eigen::Index i=0; i<engine().n_maps/engine().rtcproc.polarization.stokes_params.size(); ++i) {
            cmb.pointing.push_back(Eigen::Tensor<double,3>(cmb.n_rows, cmb.n_cols, 9));
            cmb.pointing.at(i).setZero();
        }
    }
}

template <class EngineType>
template <class map_buffer_t>
void TimeOrderedDataProc<EngineType>::allocate_nmb(map_buffer_t &nmb) {
    // clear noise map buffer
    std::vector<Eigen::Tensor<double,3>>().swap(nmb.noise);

    // resize noise maps (n_maps, [n_rows, n_cols, n_noise])
    for (Eigen::Index i=0; i<engine().n_maps; ++i) {
        nmb.noise.emplace_back(nmb.n_rows, nmb.n_cols, nmb.n_noise);
        nmb.noise.at(i).setZero();
    }
}

// coadd maps
template <class EngineType>
void TimeOrderedDataProc<EngineType>::coadd() {
    // calculate the offset between cmb and omb tangent plane coordinates
    int delta_row = (engine().omb.rows_tan_vec(0) - engine().cmb.rows_tan_vec(0)) / engine().cmb.pixel_size_rad;
    int delta_col = (engine().omb.cols_tan_vec(0) - engine().cmb.cols_tan_vec(0)) / engine().cmb.pixel_size_rad;

    // loop through the maps
    for (Eigen::Index i=0; i<engine().n_maps; ++i) {
        // define common block references
        auto cmb_weight_block = engine().cmb.weight.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols);
        auto cmb_signal_block = engine().cmb.signal.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols);

        // update cmb.weight with omb.weight
        cmb_weight_block += engine().omb.weight.at(i);

        // update cmb.signal with omb.signal * omb.weight
        cmb_signal_block += (engine().omb.signal.at(i).array() * engine().omb.weight.at(i).array()).matrix();

        // update cmb.kernel with omb.kernel * omb.weight
        if (engine().rtcproc.run_kernel) {
            auto cmb_kernel_block = engine().cmb.kernel.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols);
            cmb_kernel_block += (engine().omb.kernel.at(i).array() * engine().omb.weight.at(i).array()).matrix();
        }

        // update coverage
        if (!engine().cmb.coverage.empty()) {
            auto cmb_coverage_block = engine().cmb.coverage.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols);
            cmb_coverage_block += engine().omb.coverage.at(i);
        }
    }
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::create_coadded_map_files() {
    // clear fits_io vectors
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(engine().coadd_fits_io_vec);
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(engine().coadd_noise_fits_io_vec);
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(engine().filtered_coadd_fits_io_vec);
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(engine().filtered_coadd_noise_fits_io_vec);

    // loop through arrays
    for (Eigen::Index i=0; i<engine().calib.n_arrays; ++i) {
        // array index
        auto array = engine().calib.arrays[i];
        // array name
        std::string array_name = engine().toltec_io.array_name_map[array];
        // map filename
        auto filename = engine().toltec_io.template create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::map,
                                                                    engine_utils::toltecIO::raw>(engine().coadd_dir_name + "raw/",
                                                                                                 "", array_name, "",
                                                                                                 engine().telescope.sim_obs);
        // create fits_io class for current array file
        fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
        // append to fits_io vector
        engine().coadd_fits_io_vec.push_back(std::move(fits_io));

        // if noise maps requested
        if (engine().run_noise) {
            // noise map filename
            auto filename = engine().toltec_io.template create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::noise,
                                                                        engine_utils::toltecIO::raw>(engine().coadd_dir_name + "raw/",
                                                                                                     "", array_name, "",
                                                                                                     engine().telescope.sim_obs);
            // create fits_io class for current array file
            fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
            // append to fits_io vector
            engine().coadd_noise_fits_io_vec.push_back(std::move(fits_io));
        }
    }

    // if map filtering are requested
    if (engine().run_map_filter) {
        // loop through arrays
        for (Eigen::Index i=0; i<engine().calib.n_arrays; ++i) {
            // array index
            auto array = engine().calib.arrays[i];
            // array name
            std::string array_name = engine().toltec_io.array_name_map[array];
            // filtered map filename
            auto filename = engine().toltec_io.template create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::map,
                                                                        engine_utils::toltecIO::filtered>(engine().coadd_dir_name +
                                                                                                          "filtered/","", array_name,
                                                                                                          "", engine().telescope.sim_obs);
            // create fits_io class for current array file
            fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
            // append to fits_io vector
            engine().filtered_coadd_fits_io_vec.push_back(std::move(fits_io));

            // if noise maps requested
            if (engine().run_noise) {
                // filtered noise map filename
                auto filename = engine().toltec_io.template create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::noise,
                                                                            engine_utils::toltecIO::filtered>(engine().coadd_dir_name +
                                                                                                              "filtered/","", array_name,
                                                                                                              "", engine().telescope.sim_obs);
                // create fits_io class for current array file
                fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
                // append to fits_io vector
                engine().filtered_coadd_noise_fits_io_vec.push_back(std::move(fits_io));
            }
        }
    }
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::make_index_file(std::string filepath) {
    // get sortedfiles and directories in filepath
    std::set<fs::path> sorted_by_name;
    for (auto &entry : fs::directory_iterator(filepath))
        sorted_by_name.insert(entry);

    // yaml node to store names
    YAML::Node node;
    // data products
    node["description"].push_back("citlali data products");
    // datetime when file is created
    node["date"].push_back(engine_utils::current_date_time());
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
