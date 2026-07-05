#pragma once

#include <stdexcept>

#include <unsupported/Eigen/CXX11/Tensor>

#include <tula/eigen.h>

#include <citlali/core/utils/pointing.h>
#include <tula/algorithm/mlinterp/mlinterp.hpp>

#include <citlali/core/utils/fits_io.h>
#include <citlali/core/utils/compressed_log_sink.h>
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
    // updated alignment of networks and hwpr vectors in time that accounts for gaps
    void align_timestreams_gaps(const RawObs &rawobs);
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

#include <citlali/core/engine/detail/todproc_raw_input_impl.h>
#include <citlali/core/engine/detail/todproc_alignment_impl.h>
#include <citlali/core/engine/detail/todproc_pointing_impl.h>
#include <citlali/core/engine/detail/todproc_map_geometry_impl.h>
#include <citlali/core/engine/detail/todproc_allocation_impl.h>
#include <citlali/core/engine/detail/todproc_coadd_output_impl.h>
