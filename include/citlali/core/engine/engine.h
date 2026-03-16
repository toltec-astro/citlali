#pragma once

#include "sys/types.h"
#include "sys/sysinfo.h"

#include <memory>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cctype>
#include <omp.h>
#include <fstream>
#include <limits>

#include <Eigen/Core>

#include <citlali_config/config.h>
#include <citlali_config/gitversion.h>
#include <citlali_config/default_config.h>
#include <kids/core/kidsdata.h>
#include <kids/sweep/fitter.h>
#include <kids/timestream/solver.h>
#include <kids/toltec/toltec.h>
#include <kidscpp_config/gitversion.h>
#include <tula_config/gitversion.h>
#include <tula/cli.h>
#include <tula/config/core.h>
#include <tula/config/flatconfig.h>
#include <tula/config/yamlconfig.h>
#include <tula/enum.h>
#include <tula/filesystem.h>
#include <tula/formatter/container.h>
#include <tula/formatter/enum.h>
#include <tula/grppi.h>
#include <tula/logging.h>
#include <tula/switch_invoke.h>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/fits_io.h>
#include <citlali/core/utils/toltec_io.h>
#include <citlali/core/utils/gauss_models.h>
#include <citlali/core/utils/fitting.h>

#include <citlali/core/engine/config.h>
#include <citlali/core/engine/calib.h>
#include <citlali/core/engine/telescope.h>
#include <citlali/core/engine/diagnostics.h>
#include <citlali/core/timestream/timestream.h>

#include <citlali/core/timestream/rtc/polarization.h>
#include <citlali/core/timestream/rtc/kernel.h>
#include <citlali/core/timestream/rtc/despike.h>
#include <citlali/core/timestream/rtc/filter.h>
#include <citlali/core/timestream/rtc/downsample.h>
#include <citlali/core/timestream/rtc/calibrate.h>

#include <citlali/core/timestream/ptc/clean.h>
#include <citlali/core/timestream/ptc/sensitivity.h>

#include <citlali/core/timestream/rtc/rtcproc.h>
#include <citlali/core/timestream/ptc/ptcproc.h>

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/mapmaking/naive_mm.h>
#include <citlali/core/mapmaking/jinc_mm.h>
#include <citlali/core/mapmaking/ml_mm.h>
#if defined(CITLALI_USE_WIENER_FILTER_OMP)
#include <citlali/core/mapmaking/wiener_filter_omp.h>
#else
#include <citlali/core/mapmaking/wiener_filter.h>
#endif

#include <citlali/core/engine/io.h>
#include <citlali/core/engine/kidsproc.h>
#include <citlali/core/engine/todproc.h>

struct reduControls {
    // interpolate over gaps in timestreams
    bool interp_over_gaps;
    // create reduction subdirectories
    bool use_subdir;

    // run or skip tod processing
    bool run_tod;

    // output timestreams
    bool run_tod_output;

    // controls for mapmaking
    bool run_mapmaking;
    bool run_coadd;
    bool run_noise;
    bool run_map_filter;

    // run source finding
    bool run_source_finder;
};

struct reduClasses {
    // reduction classes
    engine::Calib calib;
    engine::Telescope telescope;
    engine_utils::toltecIO toltec_io;
    engine::Diagnostics diagnostics;
    engine_utils::mapFitter map_fitter;

    // rtc processing class
    timestream::RTCProc rtcproc;

    // ptc processing class
    timestream::PTCProc ptcproc;

    // map classes
    mapmaking::MapBuffer omb{"omb"}, cmb{"cmb"};
    mapmaking::NaiveMapmaker naive_mm;
    mapmaking::JincMapmaker jinc_mm;
    mapmaking::MLMapmaker ml_mm;
    mapmaking::WienerFilter wiener_filter;
};

struct beammapControls {
    // source name
    std::string beammap_source_name;

    // beammap source position
    double beammap_ra_rad, beammap_dec_rad;

    // fluxes and errs
    std::map<std::string, double> beammap_fluxes_mJy_beam, beammap_err_mJy_beam;
    std::map<std::string, double> beammap_fluxes_MJy_Sr, beammap_err_MJy_Sr;

    // maximum beammap iterations
    int beammap_iter_max;

    // beammap tolerance
    double beammap_iter_tolerance;

    // subtract reference detector
    bool beammap_subtract_reference;

    // beammap reference detector
    Eigen::Index beammap_reference_det;

    // derotate fitted detectors
    bool beammap_derotate;

    // optional robust sample-level RFI masking in detector-grouped beammaps
    bool beammap_rfi_mask_enabled = false;
    int beammap_rfi_mask_block_size_samples = 64;
    int beammap_rfi_mask_min_good_samples = 32;
    int beammap_rfi_mask_dilate_blocks = 1;
    double beammap_rfi_mask_sigma_threshold = 6.0;
    double beammap_rfi_mask_sigma_floor = 0.0;
    double beammap_rfi_mask_max_flagged_fraction = 0.35;

    // optional detector-map edge-band masking for coherent bad scan legs
    bool beammap_scan_band_mask_enabled = false;
    int beammap_scan_band_mask_edge_rows = 24;
    int beammap_scan_band_mask_min_row_pixels = 8;
    int beammap_scan_band_mask_min_contiguous_rows = 2;
    double beammap_scan_band_mask_row_median_sigma_threshold = 4.0;
    double beammap_scan_band_mask_row_sigma_ratio_threshold = 2.5;
    double beammap_scan_band_mask_max_flagged_fraction = 0.30;

    // optional beammap detector-map FITS splitting by detector quality flag
    bool beammap_split_fits_by_flag = false;
    std::vector<int> beammap_split_flag_values = {0, 1};

    // optional soft priors for beammap peak initialization
    bool beammap_priors_enabled = false;
    std::string beammap_priors_filepath = "null";
    int beammap_priors_candidate_top_n = 64;
    double beammap_priors_min_snr = 0.0;
    double beammap_priors_max_d2 = 25.0;
    double beammap_priors_score_lambda = 2.0;
    bool beammap_priors_fallback_blind = true;

    // iteration to write out ptcdata
    int beammap_tod_output_iter = 0;

    // upper and lower limits of psd for sensitivity calc
    Eigen::VectorXd sens_psd_limits_Hz;

    // limits on fwhm, sig2noise, and distance from center for flagging
    std::map<std::string, double> lower_fwhm_arcsec, upper_fwhm_arcsec, lower_sig2noise,
        upper_sig2noise, max_dist_arcsec, network_robust_z;
    double beammap_flag_max_prior_d2 = 0.0;

    // limits on sensitivity for flagging
    double lower_sens_factor, upper_sens_factor;
};

class Engine: public reduControls, public reduClasses, public beammapControls {
public:
    // type for missing/invalid keys
    using key_vec_t = std::vector<std::vector<std::string>>;

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // for timing
    Eigen::VectorXd t_common;
    std::vector<Eigen::VectorXi> masks;
    std::map<Eigen::Index, Eigen::VectorXi> nw_masks;
    std::vector<Eigen::VectorXd> nw_times;

    // date/time of each obs
    std::vector<std::string> date_obs;

    // add extra output for debugging
    bool verbose_mode;

    // time gaps
    std::map<std::string,int> gaps;

    // output directory and optional sub directory name
    std::string output_dir, redu_dir_name;

    // reduction directory number
    int redu_dir_num;

    // obsnum and coadded directory names
    std::string obsnum_dir_name, coadd_dir_name;

    // tod output file name
    std::map<std::string, std::string> tod_filename;

    // vectors to hold missing/invalid keys
    key_vec_t missing_keys, invalid_keys;

    // number of threads
    int n_threads;

    // parallel execution policy
    std::string parallel_policy;

    // number of scans completed
    int n_scans_done;

    // manual offsets for nws and hwp
    std::map<std::string,double> interface_sync_offset;

    // vectors for tod alignment offsets
    std::vector<Eigen::Index> start_indices, end_indices;

    // indices for hwpr alignment offsets
    Eigen::Index hwpr_start_indices, hwpr_end_indices;

    // xs, rs, is, qs
    std::string tod_type;

    // reduction type (science, pointing, beammap)
    std::string redu_type;

    // obsnum
    std::string obsnum;

    // write filtered maps as they complete
    bool write_filtered_maps_partial;

    // rtc or ptc types
    std::string tod_output_type, tod_output_subdir_name;
    bool run_tod_output_rtc = false;
    bool run_tod_output_ptc = false;

    // legacy shared TOD output selection (kept for backward compatibility helpers)
    bool tod_output_chunk_select_enabled = false;
    std::vector<Eigen::Index> tod_output_chunks;
    Eigen::VectorXI tod_scan_to_output_scan;
    Eigen::Index n_tod_output_scans = 0;

    // per-stream TOD output selection
    bool tod_output_chunk_select_enabled_rtc = false;
    bool tod_output_chunk_select_enabled_ptc = false;
    std::vector<Eigen::Index> tod_output_chunks_rtc;
    std::vector<Eigen::Index> tod_output_chunks_ptc;
    Eigen::VectorXI tod_scan_to_output_scan_rtc;
    Eigen::VectorXI tod_scan_to_output_scan_ptc;
    Eigen::Index n_tod_output_scans_rtc = 0;
    Eigen::Index n_tod_output_scans_ptc = 0;

    // map grouping and algorithm
    std::string map_grouping, map_method;

    // number of maps
    int n_maps;

    // mapping from index in map vector to array index
    Eigen::VectorXI maps_to_arrays, arrays_to_maps;

    // mapping from index in map vector to array index
    Eigen::VectorXI maps_to_stokes;

    // current fruit loops iteration
    int fruit_iter;

    // manual pointing offsets
    std::map<std::string, Eigen::VectorXd> pointing_offsets_arcsec;
    // modified julian dates of pointing offsets
    Eigen::ArrayXd pointing_offsets_modified_julian_date;

    // map output files
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> fits_io_vec, noise_fits_io_vec;
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> filtered_fits_io_vec, filtered_noise_fits_io_vec;

    // coadded map output files
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> coadd_fits_io_vec, coadd_noise_fits_io_vec;
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> filtered_coadd_fits_io_vec, filtered_coadd_noise_fits_io_vec;

    // per obsnum setup common to all redu types
    void obsnum_setup();

    // get RTC config options
    template<typename CT>
    void get_rtc_config(CT &);

    // get PTC config options
    template<typename CT>
    void get_ptc_config(CT &);

    // get timestream config options
    template<typename CT>
    void get_timestream_config(CT &);

    // get beammap config options
    template<typename CT>
    void get_beammap_config(CT &);

    // get mapmaking config options
    template<typename CT>
    void get_mapmaking_config(CT &);

    // get map filtering config options
    template<typename CT>
    void get_map_filter_config(CT &);

    // get all non-input config options and call other config functions
    template<typename CT>
    void get_citlali_config(CT &);

    // get source fluxes (beammap only)
    template<typename CT>
    void get_photometry_config(CT &);

    // get pointing offsets
    template<typename CT>
    void get_astrometry_config(CT &);

    // create fits files (does not populate them)
    void create_obs_map_files();

    // add FITS header values to tod files
    template <class map_buffer_t>
    void add_tod_header(map_buffer_t &);

    // create tod files (does not populate them)
    template <engine_utils::toltecIO::ProdType prod_t>
    void create_tod_files();

    // setup and query selected TOD output chunks
    void setup_tod_output_chunk_selection();
    bool should_write_tod_chunk(Eigen::Index) const;
    Eigen::Index tod_output_scan_row(Eigen::Index) const;
    Eigen::Index tod_output_scan_row(Eigen::Index, const std::string &) const;

    // output obs summary at command line
    void cli_summary();

    // write time chunk summary (verbose mode)
    template <TCDataKind tc_t>
    void write_chunk_summary(TCData<tc_t, Eigen::MatrixXd> &);

    // write map summary (verbose mode)
    template <typename map_buffer_t>
    void write_map_summary(map_buffer_t &);

    // create filenames
    template <mapmaking::MapType map_t, engine_utils::toltecIO::DataType data_t,
             engine_utils::toltecIO::ProdType prod_t>
    auto setup_filenames(std::string dir_name);

    // create variable names for maps, psds, and hists
    auto get_map_name(int);

    // add primary header to FITS files
    template <typename fits_io_type, class map_buffer_t>
    void add_phdu(fits_io_type &, map_buffer_t &, Eigen::Index);

    // add maps to FITS files and output them
    template <typename fits_io_type, class map_buffer_t>
    void write_maps(fits_io_type &, fits_io_type &, map_buffer_t &, Eigen::Index);

    // write map psds
    template <mapmaking::MapType map_t, class map_buffer_t>
    void write_psd(map_buffer_t &, std::string);

    // write map histograms
    template <mapmaking::MapType map_t, class map_buffer_t>
    void write_hist(map_buffer_t &, std::string);

    // write stats netCDF4 file
    void write_stats();

    // run the wiener filter
    template <mapmaking::MapType map_t, class map_buffer_t>
    void run_wiener_filter(map_buffer_t &);

    // find sources in the maps
    template <mapmaking::MapType map_t, class map_buffer_t>
    void find_sources(map_buffer_t &);

    // write the sources to ecsv table
    template <mapmaking::MapType map_t, class map_buffer_t>
    void write_sources(map_buffer_t &, std::string);
};

void Engine::obsnum_setup() {
    if (rtcproc.run_extinction) {
        // get atm model
        rtcproc.calibration.setup(telescope.tau_225_GHz);

        logger->info("using {} model for extinction correction",rtcproc.calibration.extinction_model);

        // check tau (may be unnecessary now)
        if (!telescope.sim_obs) {
            Eigen::VectorXd tau_el(1);
            // get mean elevation
            tau_el << telescope.tel_data["TelElAct"].mean();
            // get tau at mean elevation for each band
            auto tau_freq = rtcproc.calibration.calc_tau(tau_el, telescope.tau_225_GHz);
            // loop through and make sure average tau is not negative (implies wrong model)
            for (auto const& [key, val] : tau_freq) {
                if (val[0] < 0) {
                    logger->error("calculated mean {} tau {} < 0",toltec_io.array_name_map[key], val[0]);
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }
    else {
        rtcproc.calibration.extinction_model = "N/A";
    }

    // make sure there are matched fg's in apt if reducing in polarized mode
    if (rtcproc.run_polarization) {
        if ((calib.apt["fg"].array()==-1).all()) {
            logger->error("no matched freq groups.  cannot run in polarized mode");
            std::exit(EXIT_FAILURE);
        }
    }

    // setup kernel
    if (rtcproc.run_kernel) {
        rtcproc.kernel.setup(n_maps);
    }

    // set despiker sample rate
    rtcproc.despiker.fsmp = telescope.fsmp;
    // set processed timestream sample rate for optional adaptive PTC mode selection
    ptcproc.cleaner.sample_rate_Hz = telescope.d_fsmp;

    // if filter is requested, make it here
    if (rtcproc.run_tod_filter) {
        rtcproc.filter.make_filter(telescope.fsmp);
        if (rtcproc.run_tod_notch) {
            rtcproc.filter.make_notch_filter(telescope.fsmp);
        }
    }
    if (rtcproc.run_tod_iir_highpass) {
        const double nyquist_Hz = telescope.fsmp / 2.0;
        if (rtcproc.filter.iir_highpass_freq_Hz >= nyquist_Hz) {
            logger->error("timestream.raw_time_chunk.IIR_filter.freq_Hz ({}) must be less than Nyquist ({})",
                          rtcproc.filter.iir_highpass_freq_Hz, nyquist_Hz);
            std::exit(EXIT_FAILURE);
        }
    }

    // set map wcs crvals to source ra/dec
    if (telescope.pixel_axes == "radec") {
        omb.wcs.crval[0] = telescope.tel_header["Header.Source.Ra"](0)*RAD_TO_DEG;
        omb.wcs.crval[1] = telescope.tel_header["Header.Source.Dec"](0)*RAD_TO_DEG;

        if (run_coadd) {
            cmb.wcs.crval[0] = telescope.tel_header["Header.Source.Ra"](0)*RAD_TO_DEG;
            cmb.wcs.crval[1] = telescope.tel_header["Header.Source.Dec"](0)*RAD_TO_DEG;
        }
    }

    // set map wcs crvals to source l/b
    else if (telescope.pixel_axes == "galactic") {
        omb.wcs.crval[0] = telescope.tel_header["Header.Source.L"](0)*RAD_TO_DEG;
        omb.wcs.crval[1] = telescope.tel_header["Header.Source.B"](0)*RAD_TO_DEG;

        if (run_coadd) {
            cmb.wcs.crval[0] = telescope.tel_header["Header.Source.L"](0)*RAD_TO_DEG;
            cmb.wcs.crval[1] = telescope.tel_header["Header.Source.B"](0)*RAD_TO_DEG;
        }
    }

    // create timestream files
    if (run_tod_output) {
        setup_tod_output_chunk_selection();
        // create tod output subdirectory if requested
        if (tod_output_subdir_name!="null") {
            fs::create_directories(obsnum_dir_name + "raw/" + tod_output_subdir_name);
        }
        // make rtc tod output file
        if (tod_output_type == "rtc" || tod_output_type == "both") {
            create_tod_files<engine_utils::toltecIO::rtc_timestream>();
        }
        // make ptc tod output file
        if (tod_output_type == "ptc" || tod_output_type == "both") {
            create_tod_files<engine_utils::toltecIO::ptc_timestream>();
        }
    }
    // don't calculate any eigenvalues
    else if (!diagnostics.write_evals) {
        ptcproc.cleaner.n_calc = 0;
    }

    // output basic info for obs reduction to command line
    cli_summary();

    // set up per-det stats file values
    for (const auto &stat: diagnostics.det_stats_header) {
        diagnostics.stats[stat].setZero(calib.n_dets, telescope.scan_indices.cols());
    }
    // set up per-group stats file values
    for (const auto &stat: diagnostics.grp_stats_header) {
        diagnostics.stats[stat].setZero(calib.n_arrays, telescope.scan_indices.cols());
    }
    // clear stored eigenvalues
    std::map<Eigen::Index, std::vector<std::vector<Eigen::VectorXd>>>().swap(diagnostics.evals);
}

void Engine::setup_tod_output_chunk_selection() {
    const Eigen::Index n_scans = telescope.scan_indices.cols();
    auto setup_one = [&](const std::string &stream_name,
                         bool output_enabled,
                         bool select_enabled,
                         const std::vector<Eigen::Index> &chunks_1based,
                         Eigen::VectorXI &scan_to_output,
                         Eigen::Index &n_output_scans) {
        scan_to_output.resize(n_scans);
        scan_to_output.setConstant(-1);
        n_output_scans = 0;

        if (!output_enabled) {
            logger->info("{} TOD output disabled", stream_name);
            return;
        }

        if (!select_enabled || chunks_1based.empty()) {
            for (Eigen::Index i = 0; i < n_scans; ++i) {
                scan_to_output(i) = i;
            }
            n_output_scans = n_scans;
            logger->info("{} TOD output chunk selection disabled: writing all {} chunks",
                         stream_name, n_output_scans);
            return;
        }

        std::set<Eigen::Index> selected_chunks;
        for (const auto chunk_1based : chunks_1based) {
            if (chunk_1based < 1 || chunk_1based > n_scans) {
                logger->error("{} TOD output indices contain {} but valid scan range is [1, {}]",
                              stream_name, chunk_1based, n_scans);
                std::exit(EXIT_FAILURE);
            }
            selected_chunks.insert(chunk_1based - 1);
        }

        Eigen::Index out_index = 0;
        for (Eigen::Index i = 0; i < n_scans; ++i) {
            if (selected_chunks.count(i) > 0) {
                scan_to_output(i) = out_index;
                ++out_index;
            }
        }
        n_output_scans = out_index;
        logger->info("{} TOD output chunk selection enabled: writing {} of {} chunks",
                     stream_name, n_output_scans, n_scans);
    };

    if (!run_tod_output) {
        tod_scan_to_output_scan_rtc.resize(0);
        tod_scan_to_output_scan_ptc.resize(0);
        n_tod_output_scans_rtc = 0;
        n_tod_output_scans_ptc = 0;
    }
    else {
        setup_one("RTC", run_tod_output_rtc, tod_output_chunk_select_enabled_rtc, tod_output_chunks_rtc,
                  tod_scan_to_output_scan_rtc, n_tod_output_scans_rtc);
        setup_one("PTC", run_tod_output_ptc, tod_output_chunk_select_enabled_ptc, tod_output_chunks_ptc,
                  tod_scan_to_output_scan_ptc, n_tod_output_scans_ptc);
    }

    // keep legacy shared fields for backwards compatibility with call sites that
    // do not specify stream type explicitly.
    if (run_tod_output_rtc) {
        tod_scan_to_output_scan = tod_scan_to_output_scan_rtc;
        n_tod_output_scans = n_tod_output_scans_rtc;
    }
    else if (run_tod_output_ptc) {
        tod_scan_to_output_scan = tod_scan_to_output_scan_ptc;
        n_tod_output_scans = n_tod_output_scans_ptc;
    }
    else {
        tod_scan_to_output_scan.resize(0);
        n_tod_output_scans = 0;
    }
}

bool Engine::should_write_tod_chunk(Eigen::Index scan_index) const {
    return tod_output_scan_row(scan_index) >= 0;
}

Eigen::Index Engine::tod_output_scan_row(Eigen::Index scan_index) const {
    if (run_tod_output_rtc) {
        return tod_output_scan_row(scan_index, "rtc");
    }
    if (run_tod_output_ptc) {
        return tod_output_scan_row(scan_index, "ptc");
    }
    return -1;
}

Eigen::Index Engine::tod_output_scan_row(Eigen::Index scan_index, const std::string &stream_name) const {
    const Eigen::VectorXI *scan_to_output = nullptr;
    if (stream_name == "rtc") {
        scan_to_output = &tod_scan_to_output_scan_rtc;
    }
    else if (stream_name == "ptc") {
        scan_to_output = &tod_scan_to_output_scan_ptc;
    }
    else {
        logger->error("invalid TOD stream name '{}' for output row lookup", stream_name);
        return -1;
    }

    if (scan_index < 0 || scan_index >= scan_to_output->size()) {
        return -1;
    }
    return (*scan_to_output)(scan_index);
}

template<typename CT>
void Engine::get_rtc_config(CT &config) {
    logger->info("getting rtc config options");
    // get rtcproc config
    rtcproc.get_config(config, missing_keys, invalid_keys);

    // offset inner chunks
    if (rtcproc.run_tod_filter) {
        telescope.inner_scans_chunk = rtcproc.filter.n_terms;
    }
    // otherwise start at zero
    else {
        telescope.inner_scans_chunk = 0;
    }

    // ignore hwpr?
    get_config_value(config, calib.ignore_hwpr, missing_keys, invalid_keys,
                     std::tuple{"timestream","polarimetry", "ignore_hwpr"});
}

template<typename CT>
void Engine::get_ptc_config(CT &config) {
    logger->info("getting ptc config options");
    // get ptcproc config
    ptcproc.get_config(config, missing_keys, invalid_keys);

    // copy tod output bool for eigenvalues
    ptcproc.run_tod_output = run_tod_output;
    ptcproc.write_evals = diagnostics.write_evals;
}

template<typename CT>
void Engine::get_timestream_config(CT &config) {
    logger->info("getting timestream config options");
    // run tod processing
    get_config_value(config, run_tod, missing_keys, invalid_keys,
                     std::tuple{"timestream","enabled"});
    if (!run_tod) {
        logger->error("timestream.enabled is false. This reduction requires TOD processing; set "
                      "low_level.timestream.enabled: true in your reduce config.");
        std::exit(EXIT_FAILURE);
    }
    // tod type (xs, rs, is, qs)
    get_config_value(config, tod_type, missing_keys, invalid_keys,
                     std::tuple{"timestream","type"});

    // run rtc or ptc tod output?
    // output rtc
    get_config_value(config, run_tod_output_rtc, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","output","enabled"});
    rtcproc.tod_output_mini = false;
    if (run_tod_output_rtc && config.has(std::tuple{"timestream","raw_time_chunk","output","mode"})) {
        std::string rtc_output_mode = "full";
        get_config_value(config, rtc_output_mode, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","output","mode"}, {"full","mini"});
        rtcproc.tod_output_mini = (rtc_output_mode == "mini");
    }
    // output ptc
    get_config_value(config, run_tod_output_ptc, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","output","enabled"});
    ptcproc.tod_output_mini = false;
    if (run_tod_output_ptc && config.has(std::tuple{"timestream","processed_time_chunk","output","mode"})) {
        std::string ptc_output_mode = "full";
        get_config_value(config, ptc_output_mode, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","output","mode"}, {"full","mini"});
        ptcproc.tod_output_mini = (ptc_output_mode == "mini");
    }
    // set tod output to false by default
    run_tod_output = false;

    // check if rtc output is requested
    if (run_tod_output_rtc) {
        run_tod_output = true;
        tod_output_type = "rtc";
    }
    // if ptc output is requested
    if (run_tod_output_ptc) {
        // check if rtc output was requested
        if (run_tod_output == true) {
            tod_output_type = "both";
        }
        // else just output ptc
        else {
            run_tod_output = true;
            tod_output_type = "ptc";
        }
    }

    // tod subdirectory name
    get_config_value(config, tod_output_subdir_name, missing_keys, invalid_keys,
                     std::tuple{"timestream","output", "subdir_name"});
    // write eigenvalues to stats file
    get_config_value(config, diagnostics.write_evals, missing_keys, invalid_keys,
                     std::tuple{"timestream","output", "stats","eigenvalues"});

    // optional selection of TOD chunks to write (1-based indices) under each output block.
    // default is "all" for both rtc and ptc outputs.
    auto parse_tod_output_indices = [&](const auto &indices_key, bool output_enabled, const std::string &config_path,
                                        bool &select_enabled, std::vector<Eigen::Index> &chunks_out) {
        select_enabled = false;
        chunks_out.clear();

        if (!output_enabled || !config.has(indices_key)) {
            return;
        }

        if (config.template has_typed<std::string>(indices_key)) {
            const auto indices_value = config.template get_typed<std::string>(indices_key);
            if (indices_value == "all") {
                return;
            }
            logger->error("{} must be \"all\" or a non-empty list of 1-based positive integers. Found \"{}\"",
                          config_path, indices_value);
            std::exit(EXIT_FAILURE);
        }

        if (config.template has_typed<std::vector<int>>(indices_key)) {
            const auto chunks = config.template get_typed<std::vector<int>>(indices_key);
            if (chunks.empty()) {
                logger->error("{} must be \"all\" or a non-empty list of 1-based positive integers", config_path);
                std::exit(EXIT_FAILURE);
            }
            select_enabled = true;
            for (const auto chunk_index : chunks) {
                if (chunk_index <= 0) {
                    logger->error("{} must be 1-based positive integers. Found {}", config_path, chunk_index);
                    std::exit(EXIT_FAILURE);
                }
                chunks_out.push_back(static_cast<Eigen::Index>(chunk_index));
            }
            return;
        }

        logger->error("{} must be \"all\" or a list of 1-based positive integers", config_path);
        std::exit(EXIT_FAILURE);
    };

    bool rtc_chunk_select_enabled = false;
    bool ptc_chunk_select_enabled = false;
    std::vector<Eigen::Index> rtc_output_chunks, ptc_output_chunks;

    parse_tod_output_indices(std::tuple{"timestream","raw_time_chunk","output","indices"}, run_tod_output_rtc,
                             "timestream.raw_time_chunk.output.indices", rtc_chunk_select_enabled, rtc_output_chunks);
    parse_tod_output_indices(std::tuple{"timestream","processed_time_chunk","output","indices"}, run_tod_output_ptc,
                             "timestream.processed_time_chunk.output.indices", ptc_chunk_select_enabled, ptc_output_chunks);

    tod_output_chunk_select_enabled_rtc = rtc_chunk_select_enabled;
    tod_output_chunk_select_enabled_ptc = ptc_chunk_select_enabled;
    tod_output_chunks_rtc = std::move(rtc_output_chunks);
    tod_output_chunks_ptc = std::move(ptc_output_chunks);

    // keep legacy shared fields aligned with rtc (or ptc if rtc is disabled)
    if (run_tod_output_rtc) {
        tod_output_chunk_select_enabled = tod_output_chunk_select_enabled_rtc;
        tod_output_chunks = tod_output_chunks_rtc;
    }
    else if (run_tod_output_ptc) {
        tod_output_chunk_select_enabled = tod_output_chunk_select_enabled_ptc;
        tod_output_chunks = tod_output_chunks_ptc;
    }
    else {
        tod_output_chunk_select_enabled = false;
        tod_output_chunks.clear();
    }

    // get time chunk size
    get_config_value(config, telescope.chunk_mode, missing_keys, invalid_keys,
                     std::tuple{"timestream","chunking", "chunk_mode"});
    // get time chunk size
    get_config_value(config, telescope.chunking_value, missing_keys, invalid_keys,
                     std::tuple{"timestream","chunking", "value"});
    // force chunking?
    get_config_value(config, telescope.force_chunk, missing_keys, invalid_keys,
                     std::tuple{"timestream","chunking", "force_chunking"});

    /* get raw time chunk config */
    get_rtc_config(config);

    /* get processed time chunk config */
    get_ptc_config(config);
}

template<typename CT>
void Engine::get_mapmaking_config(CT &config) {
    logger->info("getting mapmaking config options");
    // enable mapmaking?
    get_config_value(config, run_mapmaking, missing_keys, invalid_keys,
                     std::tuple{"mapmaking","enabled"});
    // map grouping
    get_config_value(config, map_grouping, missing_keys, invalid_keys,
                     std::tuple{"mapmaking","grouping"},{"auto","array","nw","detector","fg"});

    // polarization is disabled for detector grouping
    if (rtcproc.run_polarization && ((redu_type=="beammap" && map_grouping=="auto") || map_grouping=="detector")) {
        logger->error("Detector grouping reductions do not currently support polarimetry mode");
        std::exit(EXIT_FAILURE);
    }

    // set rtcproc map_grouping
    rtcproc.kernel.map_grouping = map_grouping;
    ptcproc.active_map_grouping = map_grouping;

    // map_method
    get_config_value(config, map_method, missing_keys, invalid_keys,
                     std::tuple{"mapmaking","method"},{"naive","jinc","maximum_likelihood"});
    std::string fruit_interp_default = (map_method == "jinc") ? "jinc" : "bilinear";
    ptcproc.fruit_loops_interp_mode = fruit_interp_default;
    if (ptcproc.run_fruit_loops && ptcproc.fruit_loops_interp_mode_override != "auto") {
        ptcproc.fruit_loops_interp_mode = ptcproc.fruit_loops_interp_mode_override;
    }
    if (ptcproc.fruit_loops_interp_mode == "jinc" && map_method != "jinc") {
        logger->warn("fruit_loops.interp_mode_override='jinc' requires mapmaking.method='jinc'; using bilinear");
        ptcproc.fruit_loops_interp_mode = "bilinear";
    }
    logger->info("fruit loops interpolation mode: {} (default from mapmaking.method='{}' is {})",
                 ptcproc.fruit_loops_interp_mode, map_method, fruit_interp_default);
    logger->info("fruit loops center convention: {}",
                 ptcproc.fruit_loops_legacy_center ? "legacy n/2" : "current (n-1)/2");
    logger->info("fruit loops post-addback weight mode: {}",
                 ptcproc.fruit_loops_recompute_weights_after_addback
                     ? "recompute from add-back TOD"
                     : "keep source-subtracted");
    ptcproc.fruit_loops_jinc_r_max = 0.0;
    ptcproc.fruit_loops_jinc_subpixel_n = 1;
    ptcproc.fruit_loops_jinc_shape_params.clear();

    // map reference frame (radec, altaz, galactic)
    get_config_value(config, telescope.pixel_axes, missing_keys, invalid_keys,
                     std::tuple{"mapmaking","pixel_axes"},{"radec","altaz", "galactic"});

    // get config for omb
    logger->info("getting omb config options");
    omb.get_config(config, missing_keys, invalid_keys, telescope.pixel_axes, redu_type);

    // run coaddition?
    get_config_value(config, run_coadd, missing_keys, invalid_keys,
                     std::tuple{"coadd","enabled"});

    // re-run to get config for cmb
    if (run_coadd) {
        logger->info("getting cmb config options");
        cmb.get_config(config, missing_keys, invalid_keys, telescope.pixel_axes, redu_type);
    }

    // if flux calibration is not enabled, use tod type units (xs, rs, is, or qs)
    if (!rtcproc.run_calibrate) {
        omb.sig_unit = tod_type;
        cmb.sig_unit = tod_type;
    }

    // set parallelization for psd filter ffts (maintained with tod output/verbose mode)
    omb.parallel_policy = parallel_policy;
    cmb.parallel_policy = parallel_policy;
    jinc_mm.parallel_policy = parallel_policy;

    if (map_method=="jinc") {
        // maximum radius for jinc filter
        get_config_value(config, jinc_mm.r_max, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","jinc_filter","r_max"});
        // get jinc filter shape params
        for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
            auto jinc_shape_vec = config.template get_typed<std::vector<double>>(std::tuple{"mapmaking","jinc_filter","shape_params",arr_name});
            jinc_mm.shape_params[arr_index] = Eigen::Map<Eigen::VectorXd>(jinc_shape_vec.data(),jinc_shape_vec.size());
        }
        // optional: sub-pixel sampling for jinc kernel
        if (config.template has_typed<int>(std::tuple{"mapmaking","jinc_filter","subpixel_n"})) {
            get_config_value(config, jinc_mm.subpixel_n, missing_keys, invalid_keys,
                             std::tuple{"mapmaking","jinc_filter","subpixel_n"},{},{1});
        }
        ptcproc.fruit_loops_jinc_r_max = jinc_mm.r_max;
        ptcproc.fruit_loops_jinc_subpixel_n = jinc_mm.subpixel_n;
        ptcproc.fruit_loops_jinc_shape_params = jinc_mm.shape_params;

        if (jinc_mm.mode=="matrix") {
            // allocate jinc matrix
            jinc_mm.allocate_jinc_matrix(omb.pixel_size_rad);
        }
        else if (jinc_mm.mode=="splines") {
            // precompute jinc spline
            jinc_mm.calculate_jinc_splines();
        }
    }

    else if (map_method=="maximum_likelihood") {
        get_config_value(config, ml_mm.tolerance, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","maximum_likelihood","tolerance"});
        get_config_value(config, ml_mm.max_iterations, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","maximum_likelihood","max_iterations"});
    }

    // make noise maps?
    get_config_value(config, run_noise, missing_keys, invalid_keys,
                     std::tuple{"noise_maps","enabled"});
    if (run_noise) {
        // number of noise maps
        get_config_value(config, omb.n_noise, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","n_noise_maps"},{},{0},{});
        // randomize noise maps on detector as well as time chunk
        get_config_value(config, omb.randomize_dets, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","randomize_dets"});

        if (run_coadd) {
            // copy omb number of noise maps to cmb
            cmb.n_noise = omb.n_noise;
            // copy randomize_dets to cmb
            cmb.randomize_dets = omb.randomize_dets;
        }
    }
    // otherwise set number of noise maps to zero
    else {
        omb.n_noise = 0;
        cmb.n_noise = 0;
    }

    // set mapmaker polarization
    naive_mm.run_polarization = rtcproc.run_polarization;
    jinc_mm.run_polarization = rtcproc.run_polarization;
}

template<typename CT>
void Engine::get_beammap_config(CT &config) {
    logger->info("getting beammap config options");
    // max beammap iteration
    get_config_value(config, beammap_iter_max, missing_keys, invalid_keys,
                     std::tuple{"beammap","iter_max"});
    // beammap iteration tolerance
    get_config_value(config, beammap_iter_tolerance, missing_keys, invalid_keys,
                     std::tuple{"beammap","iter_tolerance"});
    // beammap reference detector
    get_config_value(config, beammap_reference_det, missing_keys, invalid_keys,
                     std::tuple{"beammap","reference_det"});
    // subtract reference detector?
    get_config_value(config, beammap_subtract_reference, missing_keys, invalid_keys,
                     std::tuple{"beammap","subtract_reference_det"});
    // derotate apt?
    get_config_value(config, beammap_derotate, missing_keys, invalid_keys,
                     std::tuple{"beammap","derotate"});

    // optional robust sample-level RFI masking (detector grouping)
    beammap_rfi_mask_enabled = false;
    beammap_rfi_mask_block_size_samples = 64;
    beammap_rfi_mask_min_good_samples = 32;
    beammap_rfi_mask_dilate_blocks = 1;
    beammap_rfi_mask_sigma_threshold = 6.0;
    beammap_rfi_mask_sigma_floor = 0.0;
    beammap_rfi_mask_max_flagged_fraction = 0.35;

    if (config.template has_typed<bool>(std::tuple{"beammap","rfi_mask","enabled"})) {
        get_config_value(config, beammap_rfi_mask_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","enabled"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","rfi_mask","block_size_samples"})) {
        get_config_value(config, beammap_rfi_mask_block_size_samples, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","block_size_samples"},
                         {}, {8});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","rfi_mask","min_good_samples"})) {
        get_config_value(config, beammap_rfi_mask_min_good_samples, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","min_good_samples"},
                         {}, {4});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","rfi_mask","dilate_blocks"})) {
        get_config_value(config, beammap_rfi_mask_dilate_blocks, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","dilate_blocks"},
                         {}, {0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","rfi_mask","sigma_threshold"})) {
        get_config_value(config, beammap_rfi_mask_sigma_threshold, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","sigma_threshold"},
                         {}, {1.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","rfi_mask","sigma_floor"})) {
        get_config_value(config, beammap_rfi_mask_sigma_floor, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","sigma_floor"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","rfi_mask","max_flagged_fraction"})) {
        get_config_value(config, beammap_rfi_mask_max_flagged_fraction, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","max_flagged_fraction"},
                         {}, {0.0}, {1.0});
    }

    // optional detector-map edge-band masking for coherent bad scan legs
    beammap_scan_band_mask_enabled = false;
    beammap_scan_band_mask_edge_rows = 24;
    beammap_scan_band_mask_min_row_pixels = 8;
    beammap_scan_band_mask_min_contiguous_rows = 2;
    beammap_scan_band_mask_row_median_sigma_threshold = 4.0;
    beammap_scan_band_mask_row_sigma_ratio_threshold = 2.5;
    beammap_scan_band_mask_max_flagged_fraction = 0.30;

    if (config.template has_typed<bool>(std::tuple{"beammap","scan_band_mask","enabled"})) {
        get_config_value(config, beammap_scan_band_mask_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","enabled"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","scan_band_mask","edge_rows"})) {
        get_config_value(config, beammap_scan_band_mask_edge_rows, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","edge_rows"},
                         {}, {2});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","scan_band_mask","min_row_pixels"})) {
        get_config_value(config, beammap_scan_band_mask_min_row_pixels, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","min_row_pixels"},
                         {}, {1});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","scan_band_mask","min_contiguous_rows"})) {
        get_config_value(config, beammap_scan_band_mask_min_contiguous_rows, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","min_contiguous_rows"},
                         {}, {1});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","scan_band_mask","row_median_sigma_threshold"})) {
        get_config_value(config, beammap_scan_band_mask_row_median_sigma_threshold, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","row_median_sigma_threshold"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","scan_band_mask","row_sigma_ratio_threshold"})) {
        get_config_value(config, beammap_scan_band_mask_row_sigma_ratio_threshold, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","row_sigma_ratio_threshold"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","scan_band_mask","max_flagged_fraction"})) {
        get_config_value(config, beammap_scan_band_mask_max_flagged_fraction, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","max_flagged_fraction"},
                         {}, {0.0}, {1.0});
    }

    // optional split output detector-map FITS files by detector quality flag
    beammap_split_fits_by_flag = false;
    beammap_split_flag_values = {0, 1};
    if (config.template has_typed<bool>(std::tuple{"beammap","split_fits_by_flag","enabled"})) {
        get_config_value(config, beammap_split_fits_by_flag, missing_keys, invalid_keys,
                         std::tuple{"beammap","split_fits_by_flag","enabled"});
    }
    if (config.template has_typed<std::vector<int>>(std::tuple{"beammap","split_fits_by_flag","flag_values"})) {
        auto values = config.template get_typed<std::vector<int>>(
            std::tuple{"beammap","split_fits_by_flag","flag_values"});
        if (values.empty()) {
            logger->warn("beammap.split_fits_by_flag.flag_values is empty; using defaults [0, 1]");
        }
        else {
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
            beammap_split_flag_values = std::move(values);
        }
    }

    // optional soft priors for beammap peak initialization
    beammap_priors_enabled = false;
    beammap_priors_filepath = "null";
    beammap_priors_candidate_top_n = 64;
    beammap_priors_min_snr = 0.0;
    beammap_priors_max_d2 = 25.0;
    beammap_priors_score_lambda = 2.0;
    beammap_priors_fallback_blind = true;

    if (config.template has_typed<bool>(std::tuple{"beammap","priors","enabled"})) {
        get_config_value(config, beammap_priors_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","enabled"});
    }
    if (config.template has_typed<std::string>(std::tuple{"beammap","priors","filepath"})) {
        get_config_value(config, beammap_priors_filepath, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","filepath"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","priors","candidate_top_n"})) {
        get_config_value(config, beammap_priors_candidate_top_n, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","candidate_top_n"},
                         {}, {1});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","min_snr"})) {
        get_config_value(config, beammap_priors_min_snr, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","min_snr"});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","max_d2"})) {
        get_config_value(config, beammap_priors_max_d2, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","max_d2"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","score_lambda"})) {
        get_config_value(config, beammap_priors_score_lambda, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","score_lambda"},
                         {}, {0.0});
    }
    if (config.template has_typed<bool>(std::tuple{"beammap","priors","fallback_blind"})) {
        get_config_value(config, beammap_priors_fallback_blind, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","fallback_blind"});
    }
    if (beammap_priors_enabled && beammap_priors_filepath == "null") {
        logger->warn("beammap.priors.enabled=true but beammap.priors.filepath is null; disabling priors");
        beammap_priors_enabled = false;
    }

    // lower fwhm limit
    auto lower_fwhm_arcsec_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","flagging","array_lower_fwhm_arcsec"});
    // upper fwhm limit
    auto upper_fwhm_arcsec_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","flagging","array_upper_fwhm_arcsec"});
    // lower signal-to-noise limit
    auto lower_sig2noise_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","flagging","array_lower_sig2noise"});
    // upper signal-to-noise limit
    auto upper_sig2noise_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","flagging","array_upper_sig2noise"});
    // maximum allowed distance limit
    auto max_dist_arcsec_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","flagging","array_max_dist_arcsec"});
    // per-array post-derotation network geometry cut
    auto network_robust_z_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","flagging","array_network_robust_z"});
    beammap_flag_max_prior_d2 = 0.0;
    if (config.template has_typed<double>(std::tuple{"beammap","flagging","max_prior_d2"})) {
        get_config_value(config, beammap_flag_max_prior_d2, missing_keys, invalid_keys,
                         std::tuple{"beammap","flagging","max_prior_d2"},
                         {}, {0.0});
    }

    // add params to respective array values
    Eigen::Index i = 0;
    for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
        // lower fwhm limit
        lower_fwhm_arcsec[arr_name] = lower_fwhm_arcsec_vec[i];
        // upper fwhm limit
        upper_fwhm_arcsec[arr_name] = upper_fwhm_arcsec_vec[i];
        // lower signal-to-noise limit
        lower_sig2noise[arr_name] = lower_sig2noise_vec[i];
        // upper signal-to-noise limit
        upper_sig2noise[arr_name] = upper_sig2noise_vec[i];
        // maximum allowed distance limit
        max_dist_arcsec[arr_name] = max_dist_arcsec_vec[i];
        // post-process per-network robust-z limit
        network_robust_z[arr_name] = network_robust_z_vec[i];
        i++;
    }

    // sensitivity factors
    auto sens_factors_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","flagging","sens_factors"});
    lower_sens_factor = sens_factors_vec[0];
    upper_sens_factor = sens_factors_vec[1];

    // upper and lower frequencies over which to calculate sensitivity
    sens_psd_limits_Hz.resize(2);
    // get psd limits for sens from config
    auto sens_psd_limits_Hz_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","sens_psd_limits_Hz"});
    // map sens limits back to Eigen vector
    sens_psd_limits_Hz = (Eigen::Map<Eigen::VectorXd>(sens_psd_limits_Hz_vec.data(), sens_psd_limits_Hz_vec.size()));

    // if no tolerance is specified, write out max iteration tod
    if (run_tod_output) {
        if (beammap_iter_tolerance <=0) {
            beammap_tod_output_iter = (beammap_iter_max > 0) ? (beammap_iter_max - 1) : 0;
        }
        // otherwise write out first iteration tod
        else {
            beammap_tod_output_iter = 0;
        }
    }
}

template<typename CT>
void Engine::get_map_filter_config(CT &config) {
    logger->info("getting map filtering config options");
    // get wiener filter config options
    wiener_filter.get_config(config, missing_keys, invalid_keys);

    // if in science mode, write filtered maps as they complete
    if (redu_type=="science") {
        write_filtered_maps_partial = true;
    }
    // otherwise write at end
    else {
        write_filtered_maps_partial = false;
    }
    // check if kernel is enabled
    if (wiener_filter.template_type=="kernel") {
        if (!rtcproc.run_kernel) {
            logger->error("wiener filter kernel template requires kernel");
            std::exit(EXIT_FAILURE);
        }
        // copy the map fitter
        else {
            wiener_filter.map_fitter = map_fitter;
        }
    }
    // make sure noise maps were enabled
    if (!run_noise && (!wiener_filter.run_lowpass && wiener_filter.filter_type=="wiener_filter")) {
        logger->error("wiener filter requires noise maps");
        std::exit(EXIT_FAILURE);
    }

    // set parallelization for ffts (maintained with tod output/verbose mode)
    wiener_filter.parallel_policy = parallel_policy;
}

template<typename CT>
void Engine::get_citlali_config(CT &config) {
    // interface key names
    const std::vector<std::string> interface_keys = {
        "toltec0",
        "toltec1",
        "toltec2",
        "toltec3",
        "toltec4",
        "toltec5",
        "toltec6",
        "toltec7",
        "toltec8",
        "toltec9",
        "toltec10",
        "toltec11",
        "toltec12",
        "hwpr"
    };
    // initialize all offsets explicitly to zero
    for (const auto &key : interface_keys) {
        interface_sync_offset[key] = 0.0;
    }

    //  get interface offsets
    if (config.has(std::tuple{"interface_sync_offset"})) {
        auto interface_node = config.get_node(std::tuple{"interface_sync_offset"});
        std::set<std::string> configured_keys;
        // parse each list entry by key name so YAML order does not matter
        for (Eigen::Index i=0; i<interface_node.size(); ++i) {
            bool found_key = false;
            for (const auto &key : interface_keys) {
                if (config.has(std::tuple{"interface_sync_offset", i, key})) {
                    auto offset = config.template get_typed<double>(std::tuple{"interface_sync_offset", i, key});
                    if (configured_keys.find(key) != configured_keys.end()) {
                        logger->warn("interface_sync_offset for {} specified multiple times; using last value", key);
                    }
                    interface_sync_offset[key] = offset;
                    configured_keys.insert(key);
                    found_key = true;
                }
            }
            if (!found_key) {
                logger->warn("interface_sync_offset entry {} does not contain a recognized interface key; ignoring entry", i);
            }
        }
        for (const auto &key : interface_keys) {
            if (configured_keys.find(key) == configured_keys.end()) {
                logger->warn("interface_sync_offset missing {}; using 0.0 s", key);
            }
        }
    }

    // verbose mode?
    get_config_value(config, verbose_mode, missing_keys, invalid_keys,
                     std::tuple{"runtime","verbose"});
    // output directory
    get_config_value(config, output_dir, missing_keys, invalid_keys,
                     std::tuple{"runtime","output_dir"});
    // number of threads to use
    get_config_value(config, n_threads, missing_keys, invalid_keys,
                     std::tuple{"runtime","n_threads"});
    // overall parallel policy
    get_config_value(config, parallel_policy, missing_keys, invalid_keys,
                     std::tuple{"runtime","parallel_policy"},{"seq","omp"});
    // reduction type (science, pointing, beammap)
    get_config_value(config, redu_type, missing_keys, invalid_keys,
                     std::tuple{"runtime","reduction_type"},{"science","pointing","beammap"});
    // create redu00, redu01... subdirectories
    get_config_value(config, use_subdir, missing_keys, invalid_keys,
                     std::tuple{"runtime","use_subdir"});
    // interp over gaps in align_timestream
    get_config_value(config, interp_over_gaps, missing_keys, invalid_keys,
                     std::tuple{"runtime","interp_over_gaps"});

    /* get timestream config */
    get_timestream_config(config);

    /* get mapmaking config */
    get_mapmaking_config(config);

    // run map filter?
    get_config_value(config, run_map_filter, missing_keys, invalid_keys,
                     std::tuple{"post_processing","map_filtering","enabled"});

    // run source finder?
    get_config_value(config, run_source_finder, missing_keys, invalid_keys,
                     std::tuple{"post_processing","source_finding","enabled"});

    // map fitter options if in pointing or beammap mode or if map filtering or source finding are enabled
    if (redu_type=="pointing" || redu_type=="beammap" || run_map_filter || run_source_finder) {
        // size of region around found source to fit
        get_config_value(config, map_fitter.bounding_box_pix, missing_keys, invalid_keys,
                         std::tuple{"post_processing","source_fitting","bounding_box_arcsec"},{},{0});
        // radius around center of map to find source within
        get_config_value(config, map_fitter.fitting_region_pix, missing_keys, invalid_keys,
                         std::tuple{"post_processing","source_fitting","fitting_radius_arcsec"});
        // fit 2d gaussian rotation angle
        get_config_value(config, map_fitter.fit_angle, missing_keys, invalid_keys,
                         std::tuple{"post_processing","source_fitting", "gauss_model","fit_rotation_angle"});

        // convert bounding box and fitting region to pixels
        map_fitter.bounding_box_pix = ASEC_TO_RAD*map_fitter.bounding_box_pix/omb.pixel_size_rad;
        map_fitter.fitting_region_pix = ASEC_TO_RAD*map_fitter.fitting_region_pix/omb.pixel_size_rad;

        // fitter flux and fwhm limits
        map_fitter.flux_limits.resize(2);
        map_fitter.fwhm_limits.resize(2);
        for (Eigen::Index i=0; i<map_fitter.flux_limits.size(); ++i) {
            // flux limit
            map_fitter.flux_limits(i) = config.template get_typed<double>(std::tuple{"post_processing","source_fitting",
                                                                                     "gauss_model","amp_limit_factors",i});
            // fwhm limit
            map_fitter.fwhm_limits(i) = config.template get_typed<double>(std::tuple{"post_processing","source_fitting",
                                                                                     "gauss_model","fwhm_limit_factors",i});
        }

        // flux lower factor
        if (map_fitter.flux_limits(0) > 0) {
            map_fitter.flux_low = map_fitter.flux_limits(0);
        }
        // flux lower factor
        if (map_fitter.flux_limits(1) > 0) {
            map_fitter.flux_high = map_fitter.flux_limits(1);
        }
        // fwhm lower factor
        if (map_fitter.fwhm_limits(0) > 0) {
            map_fitter.fwhm_low = map_fitter.fwhm_limits(0);
        }
        // fwhm upper factor
        if (map_fitter.fwhm_limits(1) > 0) {
            map_fitter.fwhm_high = map_fitter.fwhm_limits(1);
        }
    }

    /* get wiener filter config */
    if (run_map_filter) {
        // needs map fitter config
        get_map_filter_config(config);
    }

    // get source finder config options
    if (run_source_finder) {
        // minimum found source sigma
        get_config_value(config, omb.source_sigma, missing_keys, invalid_keys,
                         std::tuple{"post_processing","source_finding","source_sigma"});
        // window around source to exclude other sources
        get_config_value(config, omb.source_window_rad, missing_keys, invalid_keys,
                         std::tuple{"post_processing","source_finding","source_window_arcsec"});
        // search map, negative of map, or both
        get_config_value(config, omb.source_finder_mode, missing_keys, invalid_keys,
                         std::tuple{"post_processing","source_finding","mode"});

        // convert source window to radians
        omb.source_window_rad = omb.source_window_rad*ASEC_TO_RAD;

        if (run_coadd) {
            // copy omb source sigma to cmb
            cmb.source_sigma = omb.source_sigma;
            // copy omb source_window_rad to cmb
            cmb.source_window_rad = omb.source_window_rad;
            // copy omb source_finder_mode to cmb
            cmb.source_finder_mode = omb.source_finder_mode;
        }
    }

    /* get beammap config */
    if (redu_type=="beammap") {
        // needs redu_type config
        get_beammap_config(config);
    }

    // disable map related keys if map-making is disabled
    if (!run_mapmaking) {
        run_coadd = false;
        run_noise = false;
        run_map_filter = false;
        run_source_finder = false;
        // we don't need to do iterations if no maps are made
        beammap_iter_max = 1;
    }
}

template<typename CT>
void Engine::get_photometry_config(CT &config) {
    // beammap source name
    get_config_value(config, beammap_source_name, missing_keys, invalid_keys,
                     std::tuple{"beammap_source","name"});
    // beammap source ra
    get_config_value(config, beammap_ra_rad, missing_keys, invalid_keys,
                     std::tuple{"beammap_source","ra_deg"});
    // convert ra to radians
    beammap_ra_rad = beammap_ra_rad*DEG_TO_RAD;

    // beammap source dec
    get_config_value(config, beammap_dec_rad, missing_keys, invalid_keys,
                     std::tuple{"beammap_source","dec_deg"});
    // convert dec to radians
    beammap_dec_rad = beammap_dec_rad*DEG_TO_RAD;

    // number of fluxes
    Eigen::Index n_fluxes = config.get_node(std::tuple{"beammap_source","fluxes"}).size();

    // get source fluxes
    for (Eigen::Index i=0; i<n_fluxes; ++i) {
        auto array = config.get_str(std::tuple{"beammap_source","fluxes",i,"array_name"});
        // source flux in mJy/beam
        auto flux = config.template get_typed<double>(std::tuple{"beammap_source","fluxes",i,"value_mJy"});
        // source flux uncertainty in mJy/beam
        auto uncertainty_mJy = config.template get_typed<double>(std::tuple{"beammap_source","fluxes",i,"uncertainty_mJy"});

        // copy flux and uncertainty
        beammap_fluxes_mJy_beam[array] = flux;
        beammap_err_mJy_beam[array] = uncertainty_mJy;
    }
}

template<typename CT>
void Engine::get_astrometry_config(CT &config) {
    // check if config file has pointing_offsets
    if (config.has("pointing_offsets")) {
        // reset for each observation
        pointing_offsets_arcsec.clear();
        pointing_offsets_modified_julian_date.setZero(2);

        auto pointing_node = config.get_node(std::tuple{"pointing_offsets"});
        bool has_az = false;
        bool has_alt = false;
        bool has_mjd = false;
        std::vector<double> mjd_values;

        for (Eigen::Index i = 0; i < pointing_node.size(); ++i) {
            if (config.has(std::tuple{"pointing_offsets", i, "axes_name"})) {
                auto axis = config.get_str(std::tuple{"pointing_offsets", i, "axes_name"});
                std::transform(axis.begin(), axis.end(), axis.begin(),
                               [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
                if (axis == "az" || axis == "alt") {
                    auto offset = config.template get_typed<std::vector<double>>(
                        std::tuple{"pointing_offsets", i, "value_arcsec"});
                    if (offset.empty()) {
                        logger->error("pointing_offsets {} has empty value_arcsec", axis);
                        std::exit(EXIT_FAILURE);
                    }
                    if (pointing_offsets_arcsec.find(axis) != pointing_offsets_arcsec.end()) {
                        logger->warn("pointing_offsets {} specified multiple times; using last value", axis);
                    }
                    pointing_offsets_arcsec[axis] =
                        Eigen::Map<Eigen::VectorXd>(offset.data(), offset.size());
                    if (axis == "az") {
                        has_az = true;
                    }
                    else {
                        has_alt = true;
                    }
                }
                else {
                    logger->warn("unknown pointing_offsets axis_name '{}' at entry {}", axis, i);
                }
            }
            else if (config.has(std::tuple{"pointing_offsets", i, "modified_julian_date"})) {
                mjd_values = config.template get_typed<std::vector<double>>(
                    std::tuple{"pointing_offsets", i, "modified_julian_date"});
                has_mjd = true;
            }
            else {
                logger->warn("unrecognized pointing_offsets entry {}. expected axes_name/value_arcsec or modified_julian_date", i);
            }
        }

        // backward-compatible fallback for positional configs
        if (!has_az && config.has(std::tuple{"pointing_offsets", 0, "value_arcsec"})) {
            auto offset = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets", 0, "value_arcsec"});
            if (offset.empty()) {
                logger->error("pointing_offsets az has empty value_arcsec");
                std::exit(EXIT_FAILURE);
            }
            logger->warn("pointing_offsets az parsed by positional index; consider setting axes_name: az");
            pointing_offsets_arcsec["az"] = Eigen::Map<Eigen::VectorXd>(offset.data(), offset.size());
            has_az = true;
        }
        if (!has_alt && config.has(std::tuple{"pointing_offsets", 1, "value_arcsec"})) {
            auto offset = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets", 1, "value_arcsec"});
            if (offset.empty()) {
                logger->error("pointing_offsets alt has empty value_arcsec");
                std::exit(EXIT_FAILURE);
            }
            logger->warn("pointing_offsets alt parsed by positional index; consider setting axes_name: alt");
            pointing_offsets_arcsec["alt"] = Eigen::Map<Eigen::VectorXd>(offset.data(), offset.size());
            has_alt = true;
        }
        if (!has_mjd && config.has(std::tuple{"pointing_offsets", 2, "modified_julian_date"})) {
            mjd_values = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets", 2, "modified_julian_date"});
            has_mjd = true;
        }

        if (!has_az || !has_alt) {
            logger->error("pointing_offsets must include both az and alt entries");
            std::exit(EXIT_FAILURE);
        }

        const auto n_az = pointing_offsets_arcsec["az"].size();
        const auto n_alt = pointing_offsets_arcsec["alt"].size();
        if (n_az != n_alt) {
            logger->error("pointing_offsets az/alt lengths differ (az={} alt={})", n_az, n_alt);
            std::exit(EXIT_FAILURE);
        }
        if (n_az != 1 && n_az != 2) {
            logger->error("pointing_offsets supports only one or two values per axis (got {})", n_az);
            std::exit(EXIT_FAILURE);
        }

        if (has_mjd) {
            if (mjd_values.size() == 2) {
                pointing_offsets_modified_julian_date =
                    Eigen::Map<Eigen::VectorXd>(mjd_values.data(), mjd_values.size());
            }
            else if (!mjd_values.empty() &&
                     std::all_of(mjd_values.begin(), mjd_values.end(), [](double v){ return v <= 0.0; })) {
                // non-positive sentinel means "not specified"
                pointing_offsets_modified_julian_date.setZero(2);
            }
            else if (mjd_values.size() == 1 && n_az == 1) {
                logger->warn(
                    "ignoring single pointing_offsets.modified_julian_date for single pointing offset; using a constant offset across the observation");
                pointing_offsets_modified_julian_date.setZero(2);
            }
            else {
                logger->error(
                    "pointing_offsets.modified_julian_date must contain 2 values when interpolating two offsets, or non-positive sentinels");
                std::exit(EXIT_FAILURE);
            }
        }
    }
    else {
        logger->error("pointing_offsets not found in config");
        std::exit(EXIT_FAILURE);
    }
}

void Engine::create_obs_map_files() {
    // clear fits vectors for each observation
    fits_io_vec.clear();
    noise_fits_io_vec.clear();
    filtered_fits_io_vec.clear();
    filtered_noise_fits_io_vec.clear();

    // loop through arrays
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        // array index
        auto array = calib.arrays[i];
        // array name
        std::string array_name = toltec_io.array_name_map[array];
        // map filename
        auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::map,
                                                  engine_utils::toltecIO::raw>(obsnum_dir_name + "raw/", redu_type, array_name,
                                                                               obsnum, telescope.sim_obs);
        // create fits_io class for current array file
        fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
        // append to fits_io vector
        fits_io_vec.push_back(std::move(fits_io));

        // if noise maps are requested but coadding is not, populate noise fits vector
        if (!run_coadd) {
            if (run_noise) {
                // noise map filename
                auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::noise,
                                                          engine_utils::toltecIO::raw>(obsnum_dir_name + "raw/", redu_type, array_name,
                                                                                       obsnum, telescope.sim_obs);
                // create fits_io class for current array file
                fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
                // append to fits_io vector
                noise_fits_io_vec.push_back(std::move(fits_io));
            }

            // map filtering
            if (run_map_filter) {
                // filtered map filename
                auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::map,
                                                          engine_utils::toltecIO::filtered>(obsnum_dir_name + "filtered/",
                                                                                            redu_type, array_name,
                                                                                            obsnum, telescope.sim_obs);
                // create fits_io class for current array file
                fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
                // append to fits_io vector
                filtered_fits_io_vec.push_back(std::move(fits_io));

                // filtered noise maps
                if (run_noise) {
                    // filtered noise map filename
                    auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::noise,
                                                              engine_utils::toltecIO::filtered>(obsnum_dir_name + "filtered/", redu_type,
                                                                                                array_name, obsnum, telescope.sim_obs);
                    // create fits_io class for current array file
                    fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
                    // append to fits_io vector
                    filtered_noise_fits_io_vec.push_back(std::move(fits_io));
                }
            }
        }
    }
}

template <class map_buffer_t>
void Engine::add_tod_header(map_buffer_t &mb) {
    // loop through viles
    for (const auto & [fkey, fval]: tod_filename) {
        netCDF::NcFile fo(fval, netCDF::NcFile::write);

        // add unit conversions
        if (rtcproc.run_calibrate) {
            for (const auto &val: calib.arrays) {
                auto name = toltec_io.array_name_map[val];
                // conversion to uK
                auto fwhm = (std::get<0>(calib.array_fwhms[val]) + std::get<1>(calib.array_fwhms[val]))/2;
                auto mJy_beam_to_uK = engine_utils::mJy_beam_to_uK(1, toltec_io.array_freq_map[val], fwhm*ASEC_TO_RAD);

                // beam area in steradians
                auto beam_area_rad = 2.*pi*pow(fwhm*FWHM_TO_STD*ASEC_TO_RAD,2);
                // get Jy/pixel
                auto mJy_beam_to_Jy_px = 1e-3/beam_area_rad*pow(omb.pixel_size_rad,2);

                if (omb.sig_unit == "mJy/beam") {
                    // conversion to mJy/beam
                    add_netcdf_var(fo, "to_mJy_beam_"+name, 1);
                    // conversion to MJy/sr
                    add_netcdf_var(fo, "to_MJy_sr_"+name, 1/(calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC));
                    // conversion to uK
                    add_netcdf_var(fo, "to_uK_"+name, mJy_beam_to_uK);
                    // conversion to Jy/pixel
                    add_netcdf_var(fo, "to_Jy_pixel_"+name, mJy_beam_to_Jy_px);
                }
                else if (omb.sig_unit == "MJy/sr") {
                    // conversion to mJy/beam
                    add_netcdf_var(fo, "to_mJy_beam_"+name, calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC);
                    // conversion to MJy/Sr
                    add_netcdf_var(fo, "to_MJy_sr_"+name, 1);
                    // conversion to uK
                    add_netcdf_var(fo, "to_uK_"+name, calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC*mJy_beam_to_uK);
                    // conversion to Jy/pixel
                    add_netcdf_var(fo, "to_Jy_pixel_"+name, calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC*mJy_beam_to_Jy_px);
                }
                else if (omb.sig_unit == "uK") {
                    // conversion to mJy/beam
                    add_netcdf_var(fo, "to_mJy_beam_"+name, 1/mJy_beam_to_uK);
                    // conversion to MJy/sr
                    add_netcdf_var(fo, "to_MJy_sr_"+name, 1/mJy_beam_to_uK/(calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC));
                    // conversion to uK
                    add_netcdf_var(fo, "to_uK_"+name, 1);
                    // conversion to Jy/pixel
                    add_netcdf_var(fo, "to_Jy_pixel_"+name, (1/mJy_beam_to_uK)*mJy_beam_to_Jy_px);
                }
                else if (omb.sig_unit == "Jy/pixel") {
                    // conversion to mJy/beam
                    add_netcdf_var(fo, "to_mJy_beam_"+name, 1/mJy_beam_to_Jy_px);
                    // conversion to MJy/sr
                    add_netcdf_var(fo, "to_MJy_sr_"+name, (1/mJy_beam_to_Jy_px)/(calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC));
                    // conversion to uK
                    add_netcdf_var(fo, "to_uK_"+name, mJy_beam_to_uK/mJy_beam_to_Jy_px);
                    // conversion to Jy/pixel
                    add_netcdf_var(fo, "to_Jy_pixel_"+name, 1);
                }
            }
        }

        // add date and time of obs
        add_netcdf_var<std::string>(fo, "DATEOBS0", date_obs.back());

        // add source
        add_netcdf_var<std::string>(fo,"SOURCE",telescope.source_name);

        // add source flux for beammaps
        if (redu_type == "beammap") {
            for (const auto &val: calib.arrays) {
                auto name = toltec_io.array_name_map[val];
                add_netcdf_var(fo, "HEADER.SOURCE.FLUX_MJYPERBEAM_"+name, beammap_fluxes_mJy_beam[name]);
                add_netcdf_var(fo, "HEADER.SOURCE.FLUX_MJYPERSR_"+name, beammap_fluxes_MJy_Sr[name]);
            }
            add_netcdf_var(fo, "BEAMMAP.ITER_TOLERANCE", beammap_iter_tolerance);
            add_netcdf_var(fo, "BEAMMAP.ITER_MAX", beammap_iter_max);
            add_netcdf_var(fo, "BEAMMAP.IS_DEROTATED", beammap_derotate);

            // add reference detector information
            if (beammap_subtract_reference) {
                int ref_det_index = beammap_reference_det;
                if (calib.apt_meta["reference_det"]) {
                    ref_det_index = calib.apt_meta["reference_det"].as<int>();
                }
                add_netcdf_var(fo, "BEAMMAP.REF_DET_INDEX", ref_det_index);
                double ref_x_t = -99.0;
                double ref_y_t = -99.0;
                if (calib.apt_meta["reference_x_t"]) {
                    ref_x_t = calib.apt_meta["reference_x_t"].as<double>();
                }
                else if (ref_det_index >= 0 && ref_det_index < calib.apt["x_t"].size()) {
                    ref_x_t = calib.apt["x_t"](ref_det_index);
                }
                if (calib.apt_meta["reference_y_t"]) {
                    ref_y_t = calib.apt_meta["reference_y_t"].as<double>();
                }
                else if (ref_det_index >= 0 && ref_det_index < calib.apt["y_t"].size()) {
                    ref_y_t = calib.apt["y_t"](ref_det_index);
                }
                add_netcdf_var(fo, "BEAMMAP.REF_X_T", ref_x_t);
                add_netcdf_var(fo, "BEAMMAP.REF_Y_T", ref_y_t);
            }
            else {
                add_netcdf_var(fo, "BEAMMAP.REF_DET_INDEX", -99);
                add_netcdf_var(fo, "BEAMMAP.REF_X_T", -99);
                add_netcdf_var(fo, "BEAMMAP.REF_Y_T", -99);
            }
        }

        add_netcdf_var<std::string>(fo,"INSTRUME","TolTEC");
        add_netcdf_var(fo, "HWPR", calib.run_hwpr);
        add_netcdf_var<std::string>(fo, "TELESCOP", "LMT");
        add_netcdf_var<std::string>(fo, "PIPELINE", "CITLALI");
        add_netcdf_var<std::string>(fo, "VERSION", CITLALI_GIT_VERSION);
        add_netcdf_var<std::string>(fo, "KIDS", KIDSCPP_GIT_VERSION);
        add_netcdf_var<std::string>(fo, "TULA", TULA_GIT_VERSION);
        add_netcdf_var<std::string>(fo, "PROJID", telescope.project_id);
        add_netcdf_var<std::string>(fo, "GOAL", redu_type);
        add_netcdf_var<std::string>(fo, "OBSGOAL", telescope.obs_goal);
        add_netcdf_var<std::string>(fo, "TYPE", tod_type);
        add_netcdf_var<std::string>(fo, "GROUPING", map_grouping);
        add_netcdf_var<std::string>(fo, "METHOD", map_method);
        add_netcdf_var(fo, "EXPTIME", omb.exposure_time);
        add_netcdf_var<std::string>(fo, "RADESYS", telescope.pixel_axes);
        add_netcdf_var(fo, "TAN_RA", telescope.tel_header["Header.Source.Ra"][0]);
        add_netcdf_var(fo, "TAN_DEC", telescope.tel_header["Header.Source.Dec"][0]);
        add_netcdf_var(fo, "MEAN_EL", RAD_TO_DEG*telescope.tel_data["TelElAct"].mean());
        add_netcdf_var(fo, "MEAN_AZ", RAD_TO_DEG*telescope.tel_data["TelAzAct"].mean());
        add_netcdf_var(fo, "MEAN_PA", RAD_TO_DEG*telescope.tel_data["ActParAng"].mean());

        // add beamsizes
        for (const auto &arr: calib.arrays) {
            if (std::get<0>(calib.array_fwhms[arr]) >= std::get<1>(calib.array_fwhms[arr])) {
                add_netcdf_var(fo, "BMAJ_"+toltec_io.array_name_map[arr], std::get<0>(calib.array_fwhms[arr]));
                add_netcdf_var(fo, "BMIN_"+toltec_io.array_name_map[arr], std::get<1>(calib.array_fwhms[arr]));
                add_netcdf_var(fo, "BPA_"+toltec_io.array_name_map[arr], calib.array_pas[arr]*RAD_TO_DEG);
            }
            else {
                add_netcdf_var(fo, "BMAJ_"+toltec_io.array_name_map[arr], std::get<1>(calib.array_fwhms[arr]));
                add_netcdf_var(fo, "BMIN_"+toltec_io.array_name_map[arr], std::get<0>(calib.array_fwhms[arr]));
                add_netcdf_var(fo, "BPA_"+toltec_io.array_name_map[arr], (calib.array_pas[arr] + pi/2)*RAD_TO_DEG);
            }
        }

        add_netcdf_var(fo, "BUNIT", omb.sig_unit);

        // add jinc shape params
        if (map_method=="jinc") {
            add_netcdf_var(fo, "JINC_R", jinc_mm.r_max);
            for (const auto &[key,val]: toltec_io.array_name_map) {
                add_netcdf_var(fo, "JINC_A_"+val, jinc_mm.shape_params[calib.arrays(key)][0]);
                add_netcdf_var(fo, "JINC_B_"+val, jinc_mm.shape_params[calib.arrays(key)][0]);
                add_netcdf_var(fo, "JINC_C_"+val, jinc_mm.shape_params[calib.arrays(key)][0]);
            }
        }

        // add mean tau
        if (rtcproc.run_extinction) {
            Eigen::VectorXd tau_el(1);
            tau_el << telescope.tel_data["TelElAct"].mean();
            auto tau_freq = rtcproc.calibration.calc_tau(tau_el, telescope.tau_225_GHz);

            Eigen::Index i = 0;
            for (auto const& [key, val] : tau_freq) {
                add_netcdf_var(fo, "MEAN_TAU_"+toltec_io.array_name_map[calib.arrays(i)], val[0]);
                i++;
            }
        }
        else {
            for (Eigen::Index i=0; i<calib.arrays.size(); ++i) {
                add_netcdf_var(fo, "MEAN_TAU_"+toltec_io.array_name_map[calib.arrays(i)], 0.);
            }
        }

        // add sample rate
        add_netcdf_var(fo, "SAMPRATE", telescope.fsmp);

        // add apt table
        std::vector<string> apt_filename;
        std::stringstream ss(calib.apt_filepath);
        std::string item;
        char delim = '/';

        while (getline (ss, item, delim)) {
            apt_filename.push_back(item);
        }
        add_netcdf_var<std::string>(fo, "APT", apt_filename.back());

        add_netcdf_var(fo, "FRUITLOOPS_ITER", fruit_iter);

        // add control/runtime parameters
        add_netcdf_var(fo, "CONFIG.VERBOSE", verbose_mode);
        const bool run_any_tod_filter = rtcproc.run_tod_filter || rtcproc.run_tod_iir_highpass;
        add_netcdf_var(fo, "CONFIG.POLARIZED", rtcproc.run_polarization);
        add_netcdf_var(fo, "CONFIG.DESPIKED", rtcproc.run_despike);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.ENABLED", rtcproc.despiker.local_residual.enabled);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.WINDOW_SEC", rtcproc.despiker.local_residual.window_sec);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.SIGMA_SCALE", rtcproc.despiker.local_residual.sigma_scale);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_SIGMA_SCALE", rtcproc.despiker.local_residual.delta_sigma_scale);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.ENABLED", rtcproc.despiker.local_residual.compact_raw_gate.enabled);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_REL_SIGMA_SCALE",
                       rtcproc.despiker.local_residual.compact_raw_gate.candidate_rel_sigma_scale);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_SIGMA_SCALE",
                       rtcproc.despiker.local_residual.compact_raw_gate.candidate_rel_sigma_scale *
                           rtcproc.despiker.local_residual.sigma_scale);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.WINDOW_SEC", rtcproc.despiker.local_residual.compact_raw_gate.window_sec);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.HALF_PEAK_FRAC", rtcproc.despiker.local_residual.compact_raw_gate.half_peak_frac);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_WIDTH_SEC", rtcproc.despiker.local_residual.compact_raw_gate.max_width_sec);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_STEP_SHIFT_Z", rtcproc.despiker.local_residual.compact_raw_gate.max_step_shift_z);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.ENABLED", rtcproc.despiker.local_residual.compact_delta_gate.enabled);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.WINDOW_SEC", rtcproc.despiker.local_residual.compact_delta_gate.window_sec);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.HALF_PEAK_FRAC", rtcproc.despiker.local_residual.compact_delta_gate.half_peak_frac);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_WIDTH_SEC", rtcproc.despiker.local_residual.compact_delta_gate.max_width_sec);
        add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_STEP_SHIFT_Z", rtcproc.despiker.local_residual.compact_delta_gate.max_step_shift_z);
        add_netcdf_var(fo, "CONFIG.TODFILTERED", run_any_tod_filter);
        add_netcdf_var(fo, "CONFIG.TODNOTCH", rtcproc.run_tod_notch);
        add_netcdf_var(fo, "CONFIG.TODIIRHP", rtcproc.run_tod_iir_highpass);
        add_netcdf_var(fo, "CONFIG.TODIIRHP.FREQ_HZ", rtcproc.filter.iir_highpass_freq_Hz);
        add_netcdf_var(fo, "CONFIG.TODIIRHP.ORDER", rtcproc.filter.iir_highpass_order);
        add_netcdf_var(fo, "CONFIG.TODIIRHP.ZEROPHASE", rtcproc.filter.iir_highpass_zero_phase);
        add_netcdf_var(fo, "CONFIG.DOWNSAMPLED", rtcproc.run_downsample);
        add_netcdf_var(fo, "CONFIG.CALIBRATED", rtcproc.run_calibrate);
        add_netcdf_var(fo, "CONFIG.EXTINCTION", rtcproc.run_extinction);
        add_netcdf_var<std::string>(fo, "CONFIG.EXTINCTION.EXTMODEL", rtcproc.calibration.extinction_model);
        add_netcdf_var<std::string>(fo, "CONFIG.WEIGHT.TYPE", ptcproc.weighting_type);
        add_netcdf_var(fo, "CONFIG.INV_VAR.RTC.WTLOW", rtcproc.lower_inv_var_factor);
        add_netcdf_var(fo, "CONFIG.INV_VAR.RTC.WTHIGH", rtcproc.upper_inv_var_factor);
        add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.ENABLED", rtcproc.network_step_mask.enabled);
        add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.STEP_WINDOW_SEC", rtcproc.network_step_mask.step_window_sec);
        add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.STEP_SCORE_THRESH", rtcproc.network_step_mask.step_score_thresh);
        add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_GOOD_FRAC", rtcproc.network_step_mask.min_good_frac);
        add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_DET_USED", rtcproc.network_step_mask.min_det_used);
        add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_STEP_DET_FRAC", rtcproc.network_step_mask.min_step_det_frac);
        add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_ALIGNMENT_FRAC", rtcproc.network_step_mask.min_alignment_frac);
        add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.CLUSTER_TOL_SEC", rtcproc.network_step_mask.cluster_tol_sec);
        add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.HALF_WIDTH_SEC", rtcproc.network_step_mask.mask_half_width_sec);
        add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MAX_FLAGGED_FRAC", rtcproc.network_step_mask.max_flagged_fraction);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.ENABLED", rtcproc.impulsive_capture.enabled);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.MIN_GOOD_FRAC", rtcproc.impulsive_capture.min_good_frac);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.MIN_EVENT_Z", rtcproc.impulsive_capture.min_event_z);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.NEAR_EVENT_Z", rtcproc.impulsive_capture.near_event_z);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.MAX_EVENTS", rtcproc.impulsive_capture.max_events_per_network);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.HALF_WIDTH_SEC", rtcproc.impulsive_capture.snippet_half_width_sec);
        add_netcdf_var(fo, "CONFIG.INV_VAR.PTC.WTLOW", ptcproc.lower_inv_var_factor);
        add_netcdf_var(fo, "CONFIG.INV_VAR.PTC.WTHIGH", ptcproc.upper_inv_var_factor);
        add_netcdf_var(fo, "CONFIG.WEIGHT.PTC.WTLOW", ptcproc.lower_weight_factor);
        add_netcdf_var(fo, "CONFIG.WEIGHT.PTC.WTHIGH", ptcproc.upper_weight_factor);
        add_netcdf_var(fo, "CONFIG.WEIGHT.MEDWTFACTOR", ptcproc.med_weight_factor);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.ENABLED", ptcproc.weight_corr_penalty.enabled);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.MIN_GOOD_FRAC", ptcproc.weight_corr_penalty.min_good_frac);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.MIN_OVERLAP", ptcproc.weight_corr_penalty.min_overlap);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.MAX_SAMPLES", ptcproc.weight_corr_penalty.max_samples);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.MAX_PAIRS", ptcproc.weight_corr_penalty.max_pairs);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.FLOOR", ptcproc.weight_corr_penalty.floor);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.EXPONENT", ptcproc.weight_corr_penalty.exponent);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.PAIR.ENABLED", ptcproc.weight_corr_penalty.pair_corr.enabled);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.PAIR.REF", ptcproc.weight_corr_penalty.pair_corr.ref);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.PAIR.SPAN", ptcproc.weight_corr_penalty.pair_corr.span);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.PAIR.WEIGHT", ptcproc.weight_corr_penalty.pair_corr.weight);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.CM_EL.ENABLED", ptcproc.weight_corr_penalty.cm_el_corr.enabled);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.CM_EL.REF", ptcproc.weight_corr_penalty.cm_el_corr.ref);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.CM_EL.SPAN", ptcproc.weight_corr_penalty.cm_el_corr.span);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.CM_EL.WEIGHT", ptcproc.weight_corr_penalty.cm_el_corr.weight);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.ENABLED", ptcproc.weight_corr_penalty.cm_low_mid_ratio.enabled);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.REF", ptcproc.weight_corr_penalty.cm_low_mid_ratio.ref);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.SPAN", ptcproc.weight_corr_penalty.cm_low_mid_ratio.span);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.WEIGHT", ptcproc.weight_corr_penalty.cm_low_mid_ratio.weight);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMIN_HZ", ptcproc.weight_corr_penalty.cm_low_mid_ratio.low_min_Hz);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMAX_HZ", ptcproc.weight_corr_penalty.cm_low_mid_ratio.low_max_Hz);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMIN_HZ", ptcproc.weight_corr_penalty.cm_low_mid_ratio.mid_min_Hz);
        add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMAX_HZ", ptcproc.weight_corr_penalty.cm_low_mid_ratio.mid_max_Hz);
        add_netcdf_var(fo, "CONFIG.CLEANED", ptcproc.run_clean);
        add_netcdf_var<std::string>(fo, "CONFIG.CLEANED.MODESEL", ptcproc.cleaner.active_cleaner_label());
        add_netcdf_var(fo, "CONFIG.CLEANED.MP.ENABLED", ptcproc.cleaner.marchenko_pastur.enabled);
        add_netcdf_var(fo, "CONFIG.CLEANED.MP.BANDLOW_HZ", ptcproc.cleaner.marchenko_pastur.band_low_Hz);
        add_netcdf_var(fo, "CONFIG.CLEANED.MP.BANDHIGH_HZ", ptcproc.cleaner.marchenko_pastur.band_high_Hz);
        add_netcdf_var(fo, "CONFIG.CLEANED.MP.MAXMODES", ptcproc.cleaner.marchenko_pastur.max_modes);

        // loop through arrays and add number of eigenvalues removed
        for (Eigen::Index i=0; i<calib.arrays.size(); ++i) {
            if (ptcproc.run_clean) {
                add_netcdf_var(fo, "CONFIG.CLEANED.NEIG_"+toltec_io.array_name_map[calib.arrays(i)],
                                                    ptcproc.cleaner.n_eig_to_cut[calib.arrays(i)].sum());
            }
            else {
                add_netcdf_var(fo, "CONFIG.CLEANED.NEIG_"+toltec_io.array_name_map[calib.arrays(i)], 0);
            }
        }

        // out-of-focus holography parameters
        if (! telescope.sim_obs) {
            add_netcdf_var(fo, "OOF_T", 3.0);
            add_netcdf_var(fo, "OOF_M2X", telescope.tel_header["Header.M2.XReq"](0)/1000.*1e6);
            add_netcdf_var(fo, "OOF_M2Y", telescope.tel_header["Header.M2.YReq"](0)/1000.*1e6);
            add_netcdf_var(fo, "OOF_M2Z", telescope.tel_header["Header.M2.ZReq"](0)/1000.*1e6);

            add_netcdf_var(fo, "OOF_RO", 25.);
            add_netcdf_var(fo, "OOF_RI", 1.65);
            for (int i = 0; i < calib.arrays.size(); ++i) {
                double rms;

                if (redu_type != "beammap" && run_mapmaking) {
                    rms = pow(mb->median_err(i), 0.5);
                }
                else {
                    rms = 0.0;
                }
                auto name = toltec_io.array_name_map[calib.arrays(i)];
                add_netcdf_var(fo, "OOF_RMS_" + name, rms);
                add_netcdf_var(fo, "OOF_W_" + name, toltec_io.array_wavelength_map[calib.arrays(i)]/1000.);
                add_netcdf_var(fo, "OOF_ID_" + name, static_cast<int>(toltec_io.array_wavelength_map[calib.arrays(i)]*1000));
            }
        }

        // fruit loops parameters
        add_netcdf_var(fo, "CONFIG.FRUITLOOPS", ptcproc.run_fruit_loops);
        add_netcdf_var<std::string>(fo, "CONFIG.FRUITLOOPS.PATH", ptcproc.fruit_loops_path);
        add_netcdf_var(fo, "CONFIG.FRUITLOOPS.S2N", ptcproc.fruit_loops_sig2noise);
        for (Eigen::Index i=0; i<calib.arrays.size(); ++i) {
            double flux_limit = 0.0;
            if (ptcproc.run_fruit_loops) {
                if (ptcproc.fruit_loops_flux.size() == calib.arrays.size()) {
                    flux_limit = ptcproc.fruit_loops_flux(i);
                }
                else if (calib.arrays(i) < ptcproc.fruit_loops_flux.size()) {
                    flux_limit = ptcproc.fruit_loops_flux(calib.arrays(i));
                }
            }
            add_netcdf_var(fo, "CONFIG.FRUITLOOPS.FLUX_"+toltec_io.array_name_map[calib.arrays(i)], flux_limit);
        }

        add_netcdf_var(fo, "CONFIG.FRUITLOOPS.MAXITER", ptcproc.fruit_loops_iters);

        fo.close();
    }
}

template <engine_utils::toltecIO::ProdType prod_t>
void Engine::create_tod_files() {
    // name for std map
    std::string name;
    // subdirectory name
    std::string dir_name = obsnum_dir_name + "raw/";

    // if config subdirectory name is specified, add it
    if (tod_output_subdir_name != "null") {
        dir_name = dir_name + tod_output_subdir_name + "/";
    }

    // rtc tod output filename setup
    if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
        auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                  engine_utils::toltecIO::rtc_timestream,
                                                  engine_utils::toltecIO::raw>(dir_name, redu_type, "",
                                                                               obsnum, telescope.sim_obs);

        tod_filename["rtc"] = filename + ".nc";
        name = "rtc";
    }

    // ptc tod output filename setup
    else if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
        auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                  engine_utils::toltecIO::ptc_timestream,
                                                  engine_utils::toltecIO::raw>(dir_name, redu_type, "",
                                                                               obsnum, telescope.sim_obs);

        tod_filename["ptc"] = filename + ".nc";
        name = "ptc";
    }

    // create netcdf file
    netCDF::NcFile fo(tod_filename[name], netCDF::NcFile::replace);

    // add tod output type to file
    netCDF::NcDim n_tod_output_type_dim = fo.addDim("n_tod_output_type",1);
    netCDF::NcVar tod_output_type_var = fo.addVar("tod_output_type",netCDF::ncString, n_tod_output_type_dim);
    const std::vector<size_t> tod_output_type_index = {0};

    if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
        std::string tod_output_type_name = "rtc";
        tod_output_type_var.putVar(tod_output_type_index,tod_output_type_name);
    }
    else if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
        std::string tod_output_type_name = "ptc";
        tod_output_type_var.putVar(tod_output_type_index,tod_output_type_name);

        // number of eigenvalues
        netCDF::NcDim n_eigs_dim = fo.addDim("n_eigs",ptcproc.cleaner.n_calc);
    }

    // add obsnum
    netCDF::NcVar obsnum_v = fo.addVar("obsnum",netCDF::ncInt);
    obsnum_v.putAtt("units","N/A");
    int obsnum_int = std::stoi(obsnum);
    obsnum_v.putVar(&obsnum_int);

    // add source ra
    netCDF::NcVar source_ra_v = fo.addVar("SourceRa",netCDF::ncDouble);
    source_ra_v.putAtt("units","rad");
    source_ra_v.putVar(&telescope.tel_header["Header.Source.Ra"](0));

    // add source dec
    netCDF::NcVar source_dec_v = fo.addVar("SourceDec",netCDF::ncDouble);
    source_dec_v.putAtt("units","rad");
    source_dec_v.putVar(&telescope.tel_header["Header.Source.Dec"](0));

    const Eigen::Index n_tod_output_scans_for_stream =
        (prod_t == engine_utils::toltecIO::rtc_timestream) ? n_tod_output_scans_rtc : n_tod_output_scans_ptc;
    const bool tod_output_mini =
        (prod_t == engine_utils::toltecIO::rtc_timestream) ? rtcproc.tod_output_mini : ptcproc.tod_output_mini;

    netCDF::NcDim n_pts_dim = fo.addDim("n_pts");
    netCDF::NcDim n_raw_scan_indices_dim = fo.addDim("n_raw_scan_indices", telescope.scan_indices.rows());
    netCDF::NcDim n_scan_indices_dim = fo.addDim("n_scan_indices", 2);
    netCDF::NcDim n_scans_dim = fo.addDim("n_scans", n_tod_output_scans_for_stream);

    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", calib.n_dets);

    std::vector<netCDF::NcDim> dims = {n_pts_dim, n_dets_dim};
    std::vector<netCDF::NcDim> raw_scans_dims = {n_scans_dim, n_raw_scan_indices_dim};
    std::vector<netCDF::NcDim> scans_dims = {n_scans_dim, n_scan_indices_dim};

    // raw file scan indices
    netCDF::NcVar raw_scan_indices_v = fo.addVar("raw_scan_indices",netCDF::ncInt, raw_scans_dims);
    raw_scan_indices_v.putAtt("units","N/A");
    raw_scan_indices_v.putAtt("comment","indices in output timebase; outer=inner (output stores inner scans only)");
    std::vector<int> raw_scan_init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                   static_cast<std::size_t>(telescope.scan_indices.rows()), -2147483647);
    raw_scan_indices_v.putVar(raw_scan_init.data());

    // scan indices for data
    netCDF::NcVar scan_indices_v = fo.addVar("scan_indices",netCDF::ncInt, scans_dims);
    scan_indices_v.putAtt("units","N/A");
    std::vector<int> scan_init(static_cast<std::size_t>(n_tod_output_scans_for_stream) * 2, -2147483647);
    scan_indices_v.putVar(scan_init.data());

    // mapping from output scan row to original scan number (1-based)
    netCDF::NcVar output_scan_index_v = fo.addVar("output_scan_index", netCDF::ncInt, n_scans_dim);
    output_scan_index_v.putAtt("units","N/A");
    output_scan_index_v.putAtt("comment","1-based original scan index from the full observation");
    std::vector<int> output_scan_init(static_cast<std::size_t>(n_tod_output_scans_for_stream), -2147483647);
    output_scan_index_v.putVar(output_scan_init.data());

    // signal
    netCDF::NcVar signal_v;
    if (tod_output_mini) {
        signal_v = fo.addVar("signal", netCDF::ncFloat, dims);
    }
    else {
        signal_v = fo.addVar("signal", netCDF::ncDouble, dims);
    }
    signal_v.putAtt("units",omb.sig_unit);

    // chunk sizes
    std::vector<std::size_t> chunkSizes;
    // set chunk mode
    netCDF::NcVar::ChunkMode chunkMode = netCDF::NcVar::nc_CHUNKED;

    // set chunking to mean scan size and n_dets
    chunkSizes.push_back(((telescope.scan_indices.row(3) - telescope.scan_indices.row(2)).array() + 1).mean());
    chunkSizes.push_back(calib.n_dets);

    // set signal chunking
    signal_v.setChunking(chunkMode, chunkSizes);

    // flags
    netCDF::NcVar flags_v;
    if (tod_output_mini) {
        flags_v = fo.addVar("flags", netCDF::ncByte, dims);
    }
    else {
        flags_v = fo.addVar("flags", netCDF::ncDouble, dims);
    }
    flags_v.putAtt("units","N/A");
    if (tod_output_mini) {
        flags_v.putAtt("comment", "0=good,1=flagged");
    }
    flags_v.setChunking(chunkMode, chunkSizes);

    // kernel
    if (rtcproc.run_kernel && !tod_output_mini) {
        netCDF::NcVar kernel_v = fo.addVar("kernel",netCDF::ncDouble, dims);
        kernel_v.putAtt("units","N/A");
        kernel_v.setChunking(chunkMode, chunkSizes);
    }

    if (!tod_output_mini) {
        // detector lat
        netCDF::NcVar det_lat_v = fo.addVar("det_lat",netCDF::ncDouble, dims);
        det_lat_v.putAtt("units","rad");
        det_lat_v.setChunking(chunkMode, chunkSizes);

        // detector lon
        netCDF::NcVar det_lon_v = fo.addVar("det_lon",netCDF::ncDouble, dims);
        det_lon_v.putAtt("units","rad");
        det_lon_v.setChunking(chunkMode, chunkSizes);

        // calc absolute pointing if in radec frame
        if (telescope.pixel_axes == "radec") {
            // detector absolute ra
            netCDF::NcVar det_ra_v = fo.addVar("det_ra",netCDF::ncDouble, dims);
            det_ra_v.putAtt("units","rad");
            det_ra_v.setChunking(chunkMode, chunkSizes);

            // detector absolute dec
            netCDF::NcVar det_dec_v = fo.addVar("det_dec",netCDF::ncDouble, dims);
            det_dec_v.putAtt("units","rad");
            det_dec_v.setChunking(chunkMode, chunkSizes);
        }
    }

    // add apt table
    for (auto const& x: calib.apt) {
        netCDF::NcVar apt_v = fo.addVar("apt_" + x.first,netCDF::ncDouble, n_dets_dim);
        apt_v.putAtt("units",calib.apt_header_units[x.first]);
    }

    // add telescope parameters
    for (auto const& x: telescope.tel_data) {
        netCDF::NcVar tel_data_v = fo.addVar(x.first,netCDF::ncDouble, n_pts_dim);
        tel_data_v.putAtt("units","rad");
        tel_data_v.setChunking(chunkMode, chunkSizes);
    }

    // add pointing offset parameters
    for (auto const& x: pointing_offsets_arcsec) {
        logger->info("pointing_offsets_arcsec.second {} {}",x.first, x.second);
        netCDF::NcVar offsets_v = fo.addVar("pointing_offset_"+x.first,netCDF::ncDouble, n_pts_dim);
        offsets_v.putAtt("units","arcsec");
        offsets_v.setChunking(chunkMode, chunkSizes);
    }

    if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
        const int fill_int = -2147483647;
        const double fill_double = std::numeric_limits<double>::quiet_NaN();
        std::vector<netCDF::NcDim> rtc_det_dims = {n_scans_dim, n_dets_dim};

        auto add_rtc_det_double = [&](const std::string &name, const std::string &comment) {
            netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, rtc_det_dims);
            v.putAtt("units", "N/A");
            v.putAtt("comment", comment);
            std::vector<double> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                     static_cast<std::size_t>(calib.n_dets), fill_double);
            v.putVar(init.data());
        };
        auto add_rtc_det_int = [&](const std::string &name, const std::string &comment) {
            netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, rtc_det_dims);
            v.putAtt("units", "N/A");
            v.putAtt("comment", comment);
            std::vector<int> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                  static_cast<std::size_t>(calib.n_dets), fill_int);
            v.putVar(init.data());
        };

        add_rtc_det_int("rtc_despike_raw_exceed_count",
                        "per-detector count of raw-sample MAD-threshold exceedances before despike expansion");
        add_rtc_det_int("rtc_despike_local_raw_candidate_count",
                        "per-detector count of locally detrended raw candidate events considered by the compact-raw gate");
        add_rtc_det_int("rtc_despike_local_exceed_count",
                        "per-detector count of locally detrended raw-sample exceedance samples accepted after compact-raw gating");
        add_rtc_det_int("rtc_despike_local_raw_reject_count",
                        "per-detector count of locally detrended raw candidate events rejected by the compact-raw gate");
        add_rtc_det_int("rtc_despike_delta_spike_count",
                        "per-detector count of delta-domain spikes identified by the RTC despiker");
        add_rtc_det_int("rtc_despike_local_delta_candidate_count",
                        "per-detector count of locally detrended delta candidate events considered by the compact-delta gate");
        add_rtc_det_int("rtc_despike_local_delta_exceed_count",
                        "per-detector count of locally detrended delta candidate events accepted by the compact-delta gate");
        add_rtc_det_int("rtc_despike_local_delta_reject_count",
                        "per-detector count of locally detrended delta candidate events rejected by the compact-delta gate");
        add_rtc_det_double("rtc_despike_added_flagged_frac",
                           "fraction of samples newly flagged by RTC despiking, excluding pre-existing flags");
        add_rtc_det_int("rtc_despike_added_region_count",
                        "count of newly flagged contiguous sample regions added by RTC despiking");
        add_rtc_det_double("rtc_despike_added_region_len_median",
                           "median length of newly flagged contiguous sample regions added by RTC despiking");
        add_rtc_det_int("rtc_despike_added_region_len_max",
                        "maximum length of newly flagged contiguous sample regions added by RTC despiking");
        add_rtc_det_double("rtc_despike_max_raw_abs_z",
                           "maximum absolute raw-sample deviation in robust-sigma units before despiking");
        add_rtc_det_double("rtc_despike_max_local_abs_z",
                           "maximum absolute locally detrended raw-sample deviation in robust-sigma units before despiking");
        add_rtc_det_double("rtc_despike_max_delta_abs_z",
                           "maximum absolute adjacent-sample delta deviation in sigma units before despiking");
        add_rtc_det_double("rtc_despike_max_local_delta_abs_z",
                           "maximum absolute locally detrended adjacent-sample delta deviation in sigma units before despiking");
        add_rtc_det_double("rtc_final_flagged_frac",
                           "final per-detector flagged-sample fraction in the RTC product actually written");
        add_rtc_det_int("rtc_final_region_count",
                        "final count of flagged contiguous sample regions in the RTC product actually written");
        add_rtc_det_double("rtc_final_region_len_median",
                           "final median flagged-region length in the RTC product actually written");
        add_rtc_det_int("rtc_final_region_len_max",
                        "final maximum flagged-region length in the RTC product actually written");
        add_rtc_det_double("rtc_step_score",
                           "per-detector step-like pre/post window jump score on the RTC output");
        add_rtc_det_int("rtc_step_sample",
                        "sample index of the strongest per-detector RTC step-like jump; -2147483647 means unavailable");
        add_rtc_det_double("rtc_impulsive_peak_abs_z",
                           "maximum absolute per-sample deviation in robust-sigma units on the RTC output");
        add_rtc_det_int("rtc_impulsive_peak_abs_sample",
                        "sample index of the maximum absolute per-sample deviation; -2147483647 means unavailable");
        add_rtc_det_double("rtc_impulsive_peak_delta_abs_z",
                           "maximum absolute adjacent-sample delta deviation in robust-sigma units on the RTC output");
        add_rtc_det_int("rtc_impulsive_peak_delta_abs_sample",
                        "sample index of the strongest adjacent-sample delta excursion; -2147483647 means unavailable");
        add_rtc_det_int("rtc_impulsive_near_abs_count",
                        "count of RTC samples exceeding near_event_z in absolute robust-z units");
        add_rtc_det_int("rtc_impulsive_near_delta_count",
                        "count of RTC adjacent-sample delta excursions exceeding near_event_z");
        add_rtc_det_double("rtc_impulsive_event_score",
                           "per-detector impulsive event score, max of raw and delta robust-z peaks");
        add_rtc_det_int("rtc_impulsive_event_sample",
                        "sample index of the strongest per-detector impulsive event; -2147483647 means unavailable");
        add_rtc_det_int("rtc_impulsive_event_kind",
                        "0=raw-sample peak, 1=delta peak, -2147483647 means unavailable");

        netCDF::NcDim n_nws_rtcdiag_dim = fo.addDim("n_nws_rtcdiag", calib.n_nws);
        netCDF::NcVar nw_ids_v = fo.addVar("rtc_diag_network_ids", netCDF::ncInt, n_nws_rtcdiag_dim);
        nw_ids_v.putAtt("units", "N/A");
        nw_ids_v.putAtt("comment", "network IDs corresponding to n_nws_rtcdiag axis");
        std::vector<int> nw_ids(static_cast<std::size_t>(calib.n_nws), fill_int);
        for (Eigen::Index i = 0; i < calib.n_nws; ++i) {
            nw_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
        }
        nw_ids_v.putVar(nw_ids.data());

        std::vector<netCDF::NcDim> rtc_nw_dims = {n_scans_dim, n_nws_rtcdiag_dim};
        auto add_rtc_nw_double = [&](const std::string &name, const std::string &comment) {
            netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, rtc_nw_dims);
            v.putAtt("units", "N/A");
            v.putAtt("comment", comment);
            std::vector<double> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                     static_cast<std::size_t>(calib.n_nws), fill_double);
            v.putVar(init.data());
        };
        auto add_rtc_nw_int = [&](const std::string &name, const std::string &comment) {
            netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, rtc_nw_dims);
            v.putAtt("units", "N/A");
            v.putAtt("comment", comment);
            std::vector<int> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                  static_cast<std::size_t>(calib.n_nws), fill_int);
            v.putVar(init.data());
        };

        add_rtc_nw_int("rtc_network_n_det_input",
                       "input detector count in each RTC network block");
        add_rtc_nw_int("rtc_network_n_det_used",
                       "detectors with >= 80% valid samples and finite robust scale used for RTC diagnostics");
        add_rtc_nw_double("rtc_network_step_score_median",
                          "median detector step score within each RTC network block");
        add_rtc_nw_double("rtc_network_step_score_max",
                          "maximum detector step score within each RTC network block");
        add_rtc_nw_double("rtc_network_step_det_frac",
                          "fraction of diagnostic-used detectors with strong step-like score in each RTC network block");
        add_rtc_nw_double("rtc_network_step_alignment_frac",
                          "fraction of strong-step detectors aligned in the dominant step-time cluster");
        add_rtc_nw_int("rtc_network_step_dominant_sample",
                       "dominant aligned step sample within each RTC network block; -2147483647 means unavailable");
        add_rtc_nw_double("rtc_network_cm_low_mid_ratio",
                          "low-band to mid-band common-mode power ratio for each RTC network block");
        add_rtc_nw_double("rtc_network_cm_peak_freq_hz",
                          "frequency of the strongest common-mode spectral peak for each RTC network block");
        add_rtc_nw_double("rtc_network_cm_peak_prominence",
                          "prominence of the strongest common-mode spectral peak for each RTC network block");
        add_rtc_nw_int("rtc_network_step_mask_applied",
                       "1 if network_step_mask flagged a time window for this RTC network block, else 0");
        add_rtc_nw_int("rtc_network_step_mask_start_sample",
                       "inclusive starting sample of the applied network_step_mask window; -2147483647 means none");
        add_rtc_nw_int("rtc_network_step_mask_end_sample",
                       "inclusive ending sample of the applied network_step_mask window; -2147483647 means none");
        add_rtc_nw_int("rtc_network_step_mask_window_samples",
                       "number of RTC time samples in the applied network_step_mask window");
        add_rtc_nw_int("rtc_network_step_mask_n_det_masked",
                       "number of detectors included in the applied network_step_mask window");
        add_rtc_nw_int("rtc_network_step_mask_n_det_samples_flagged",
                       "number of previously good detector-samples newly flagged by network_step_mask");
        add_rtc_nw_double("rtc_network_step_mask_flagged_fraction",
                          "fraction of previously good detector-samples in the network block newly flagged by network_step_mask");

        if (rtcproc.impulsive_capture.enabled) {
            const auto n_slots = static_cast<std::size_t>(std::max<Eigen::Index>(rtcproc.impulsive_capture.max_events_per_network, 1));
            const double rtc_fsmp = rtcproc.run_downsample ? telescope.d_fsmp : telescope.fsmp;
            const auto snippet_half_width = static_cast<std::size_t>(std::max(0.0, std::round(rtcproc.impulsive_capture.snippet_half_width_sec * rtc_fsmp)));
            const auto n_snippet = 2 * snippet_half_width + 1;
            netCDF::NcDim n_rtc_impulsive_slots_dim = fo.addDim("n_rtc_impulsive_slots", n_slots);
            netCDF::NcDim n_rtc_impulsive_samples_dim = fo.addDim("n_rtc_impulsive_samples", n_snippet);

            netCDF::NcVar offset_v = fo.addVar("rtc_impulsive_snippet_offset_samples", netCDF::ncInt, n_rtc_impulsive_samples_dim);
            offset_v.putAtt("units", "samples");
            offset_v.putAtt("comment", "sample offsets relative to rtc_impulsive_slot_event_sample");
            std::vector<int> offsets(n_snippet, fill_int);
            for (std::size_t i = 0; i < n_snippet; ++i) {
                offsets[i] = static_cast<int>(i) - static_cast<int>(snippet_half_width);
            }
            offset_v.putVar(offsets.data());

            std::vector<netCDF::NcDim> rtc_impulsive_slot_dims = {n_scans_dim, n_nws_rtcdiag_dim, n_rtc_impulsive_slots_dim};
            std::vector<netCDF::NcDim> rtc_impulsive_snippet_dims = {n_scans_dim, n_nws_rtcdiag_dim, n_rtc_impulsive_slots_dim, n_rtc_impulsive_samples_dim};

            auto add_rtc_imp_slot_double = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, rtc_impulsive_slot_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<double> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                         static_cast<std::size_t>(calib.n_nws) * n_slots, fill_double);
                v.putVar(init.data());
            };
            auto add_rtc_imp_slot_int = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, rtc_impulsive_slot_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<int> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                      static_cast<std::size_t>(calib.n_nws) * n_slots, fill_int);
                v.putVar(init.data());
            };
            auto add_rtc_imp_snip_double = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, rtc_impulsive_snippet_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<double> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                         static_cast<std::size_t>(calib.n_nws) * n_slots * n_snippet, fill_double);
                v.putVar(init.data());
            };
            auto add_rtc_imp_snip_int = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, rtc_impulsive_snippet_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<int> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                      static_cast<std::size_t>(calib.n_nws) * n_slots * n_snippet, fill_int);
                v.putVar(init.data());
            };

            add_rtc_imp_slot_int("rtc_impulsive_slot_det_index",
                                 "detector index of a captured impulsive RTC event for each scan/network/slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_event_sample",
                                 "sample index of a captured impulsive RTC event; -2147483647 means unavailable");
            add_rtc_imp_slot_int("rtc_impulsive_slot_event_kind",
                                 "0=raw-sample peak, 1=delta peak, -2147483647 means unavailable");
            add_rtc_imp_slot_double("rtc_impulsive_slot_event_score",
                                    "impulsive event score for a captured scan/network detector slot");
            add_rtc_imp_slot_double("rtc_impulsive_slot_peak_abs_z",
                                    "maximum per-sample absolute robust-z for a captured scan/network detector slot");
            add_rtc_imp_slot_double("rtc_impulsive_slot_peak_delta_abs_z",
                                    "maximum adjacent-sample delta robust-z for a captured scan/network detector slot");
            add_rtc_imp_slot_double("rtc_impulsive_slot_added_flagged_frac",
                                    "fraction of samples newly flagged by RTC despiking for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_raw_exceed_count",
                                 "count of raw-sample MAD exceedances for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_raw_candidate_count",
                                 "count of locally detrended raw candidate events considered by the compact-raw gate for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_exceed_count",
                                 "count of locally detrended raw-sample exceedance samples accepted after compact-raw gating for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_raw_reject_count",
                                 "count of locally detrended raw candidate events rejected by the compact-raw gate for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_delta_spike_count",
                                 "count of delta-domain spikes for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_delta_candidate_count",
                                 "count of locally detrended delta candidate events considered by the compact-delta gate for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_delta_exceed_count",
                                 "count of locally detrended delta candidate events accepted by the compact-delta gate for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_delta_reject_count",
                                 "count of locally detrended delta candidate events rejected by the compact-delta gate for a captured detector slot");
            add_rtc_imp_snip_double("rtc_impulsive_slot_snippet_z",
                                    "standardized RTC snippet around each captured impulsive event");
            add_rtc_imp_snip_int("rtc_impulsive_slot_snippet_flag",
                                 "final RTC flag state for each sample in a captured impulsive snippet");
        }
    }

    // add weights
    if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
        std::vector<netCDF::NcDim> weight_dims = {n_scans_dim, n_dets_dim};
        netCDF::NcVar weights_v = fo.addVar("weights",netCDF::ncDouble, weight_dims);
        weights_v.putAtt("units","("+omb.sig_unit+")^-2");

        // optional diagnostics for correlation-defined network cleaning groups
        bool corr_nw_requested = false;
        for (const auto &g : ptcproc.cleaner.grouping) {
            if (g == "corr_nw") {
                corr_nw_requested = true;
                break;
            }
        }
        if (ptcproc.cleaner.corr_grouping.enabled && corr_nw_requested) {
            const int fill_value = -2147483647;
            std::vector<netCDF::NcDim> corr_det_dims = {n_scans_dim, n_dets_dim};
            netCDF::NcVar corr_group_id_v = fo.addVar("corr_nw_group_id", netCDF::ncInt, corr_det_dims);
            corr_group_id_v.putAtt("units", "N/A");
            corr_group_id_v.putAtt("comment",
                                   "corr_nw group index for each detector in each output scan; -2147483647 means not assigned");
            std::vector<int> corr_group_init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                             static_cast<std::size_t>(calib.n_dets), fill_value);
            corr_group_id_v.putVar(corr_group_init.data());

            netCDF::NcDim n_nws_corr_dim = fo.addDim("n_nws_corr", calib.n_nws);
            netCDF::NcVar corr_nw_ids_v = fo.addVar("corr_nw_network_ids", netCDF::ncInt, n_nws_corr_dim);
            corr_nw_ids_v.putAtt("units", "N/A");
            corr_nw_ids_v.putAtt("comment", "network IDs corresponding to n_nws_corr axis");
            std::vector<int> nw_ids(static_cast<std::size_t>(calib.n_nws), fill_value);
            for (Eigen::Index i = 0; i < calib.n_nws; ++i) {
                nw_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
            }
            corr_nw_ids_v.putVar(nw_ids.data());

            std::vector<netCDF::NcDim> corr_nw_dims = {n_scans_dim, n_nws_corr_dim};
            auto add_corr_nw_var = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, corr_nw_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<int> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                      static_cast<std::size_t>(calib.n_nws), fill_value);
                v.putVar(init.data());
            };
            add_corr_nw_var("corr_nw_n_groups", "number of final corr_nw cleaning groups per network");
            add_corr_nw_var("corr_nw_n_groups_raw", "number of raw connected components before min_group_size filtering");
            add_corr_nw_var("corr_nw_n_det_input", "input detector count in each network block");
            add_corr_nw_var("corr_nw_n_det_candidates", "detectors passing apt flag and min_good_frac");
            add_corr_nw_var("corr_nw_n_det_used", "candidate detectors with finite non-zero std for correlation");
            add_corr_nw_var("corr_nw_n_det_grouped", "detectors included in final cleaned corr_nw groups");
            add_corr_nw_var("corr_nw_n_det_ungrouped", "detectors excluded from final cleaned corr_nw groups");
            add_corr_nw_var("corr_nw_sample_step", "time decimation factor used for corr_nw grouping");
        }

        if (ptcproc.weight_corr_penalty.enabled) {
            const int fill_int = -2147483647;
            const double fill_double = std::numeric_limits<double>::quiet_NaN();
            netCDF::NcDim n_nws_wcorr_dim = fo.addDim("n_nws_wcorr", calib.n_nws);
            netCDF::NcVar nw_ids_v = fo.addVar("weight_corr_penalty_network_ids", netCDF::ncInt, n_nws_wcorr_dim);
            nw_ids_v.putAtt("units", "N/A");
            nw_ids_v.putAtt("comment", "network IDs corresponding to n_nws_wcorr axis");
            std::vector<int> nw_ids(static_cast<std::size_t>(calib.n_nws), fill_int);
            for (Eigen::Index i = 0; i < calib.n_nws; ++i) {
                nw_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
            }
            nw_ids_v.putVar(nw_ids.data());

            std::vector<netCDF::NcDim> wcorr_dims = {n_scans_dim, n_nws_wcorr_dim};
            auto add_wcorr_double = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, wcorr_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<double> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                         static_cast<std::size_t>(calib.n_nws), fill_double);
                v.putVar(init.data());
            };
            auto add_wcorr_int = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, wcorr_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<int> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                      static_cast<std::size_t>(calib.n_nws), fill_int);
                v.putVar(init.data());
            };

            add_wcorr_double("weight_corr_penalty_factor",
                             "multiplicative weight penalty factor applied per network in each output scan");
            add_wcorr_double("weight_corr_penalty_severity",
                             "normalized [0,1] severity used to derive weight_corr_penalty_factor");
            add_wcorr_double("weight_corr_penalty_pair_med_abs_corr",
                             "median absolute sampled detector-detector correlation per network");
            add_wcorr_double("weight_corr_penalty_cm_el_abs_corr",
                             "absolute correlation between network common mode and TelElAct");
            add_wcorr_double("weight_corr_penalty_cm_low_mid_ratio",
                             "common-mode low/mid bandpower ratio");
            add_wcorr_int("weight_corr_penalty_n_det_input",
                          "detector count in each network block");
            add_wcorr_int("weight_corr_penalty_n_det_candidates",
                          "detectors passing apt flag and min_good_frac");
            add_wcorr_int("weight_corr_penalty_n_det_used",
                          "candidate detectors with finite non-zero std");
            add_wcorr_int("weight_corr_penalty_n_det_weighted",
                          "detectors with positive map weight multiplied by penalty factor");
            add_wcorr_int("weight_corr_penalty_sample_step",
                          "time decimation factor used for penalty metrics");
        }
    }

    // add hwpr
    if (rtcproc.run_polarization) {
        if (calib.run_hwpr) {
            netCDF::NcVar hwpr_v = fo.addVar("hwpr",netCDF::ncDouble, n_pts_dim);
            hwpr_v.putAtt("units","rad");
        }
    }

    // add tel header
    netCDF::NcDim tel_header_dim = fo.addDim("tel_header_n_pts", 1);
    for (const auto &[key,val]: telescope.tel_header) {
        netCDF::NcVar tel_header_v = fo.addVar(key,netCDF::ncDouble, tel_header_dim);
        tel_header_v.putVar(&val(0));
    }

    fo.close();
}

//template <TCDataKind tc_t>
void Engine::cli_summary() {
    logger->info("reduction info");
    logger->info("obsnum: {}", obsnum);
    logger->info("map buffer rows: {}", omb.n_rows);
    logger->info("map buffer cols: {}", omb.n_cols);
    logger->info("number of maps: {}", omb.signal.size());
    logger->info("map units: {}", omb.sig_unit);
    logger->info("polarized reduction: {}", rtcproc.run_polarization);

    // total size of all maps
    double mb_size_total = 0;

    // make a rough estimate of memory usage for obs map buffer
    double omb_size = 8*omb.n_rows*omb.n_cols*(omb.signal.size() + omb.weight.size() +
                                               omb.kernel.size() + omb.coverage.size() +
                                               omb.grid_weight.size())/1e9;

    logger->info("estimated size of map buffer {} GB", omb_size);

    mb_size_total = mb_size_total + omb_size;

    // print info if coadd is requested
    if (run_coadd) {
        logger->info("coadd map buffer rows: {}", cmb.n_rows);
        logger->info("coadd map buffer cols: {}", cmb.n_cols);

        // make a rough estimate of memory usage for coadd map buffer
        double cmb_size = 8*cmb.n_rows*cmb.n_cols*(cmb.signal.size() + cmb.weight.size() +
                                                   cmb.kernel.size() + cmb.coverage.size() +
                                                   cmb.grid_weight.size())/1e9;

        logger->info("estimated size of coadd buffer {} GB", cmb_size);

        mb_size_total = mb_size_total + cmb_size;

        // output info if coadd noise maps are requested
        if (run_noise) {
            logger->info("coadd map buffer noise maps: {}", cmb.n_noise);
            // make a rough estimate of memory usage for coadd noise maps
            double nmb_size = 8*cmb.n_rows*cmb.n_cols*cmb.noise.size()*cmb.n_noise/1e9;
            logger->info("estimated size of noise buffer {} GB", nmb_size);
            mb_size_total = mb_size_total + nmb_size;
        }
    }
    else {
        // output info if obs noise maps are requested
        if (run_noise) {
            logger->info("observation map buffer noise maps: {}", omb.n_noise);
            // make a rough estimate of memory usage for obs noise maps
            double nmb_size = 8*omb.n_rows*omb.n_cols*omb.noise.size()*omb.n_noise/1e9;
            logger->info("estimated size of noise buffer {} GB", nmb_size);
            mb_size_total = mb_size_total + nmb_size;
        }
    }

    logger->info("estimated size of all maps {} GB", mb_size_total);
    logger->info("number of scans: {}",telescope.scan_indices.cols());
    if (run_tod_output) {
        if (tod_output_type == "rtc" || tod_output_type == "both") {
            logger->info("RTC TOD output scans: {}", n_tod_output_scans_rtc);
            logger->info("RTC TOD output mode: {}", rtcproc.tod_output_mini ? "mini" : "full");
        }
        if (tod_output_type == "ptc" || tod_output_type == "both") {
            logger->info("PTC TOD output scans: {}", n_tod_output_scans_ptc);
            logger->info("PTC TOD output mode: {}", ptcproc.tod_output_mini ? "mini" : "full");
        }
    }

    // test getting memory usage for fun
    /*struct sysinfo memInfo;
    long long totalPhysMem = memInfo.totalram;
    totalPhysMem *= memInfo.mem_unit;

    logger->info("total physical memory available {} GB", (totalPhysMem/1024)/1e7);*/
    logger->info("physical memory used {} GB", engine_utils::get_phys_memory()/1e7);
}

template <TCDataKind tc_t>
void Engine::write_chunk_summary(TCData<tc_t, Eigen::MatrixXd> &in) {

    logger->debug("writing summary files for chunk {}",in.index.data);

    std::string filename = "chunk_summary_" + std::to_string(in.index.data);

    // write summary log file
    std::ofstream f;
    f.open (obsnum_dir_name+"/logs/" + filename + ".log");

    f << "Summary file for scan " << in.index.data << "\n";
    f << "-Citlali version: " << CITLALI_GIT_VERSION << "\n";
    f << "-Kidscpp version: " << KIDSCPP_GIT_VERSION << "\n";
    f << "-Time of time chunk creation: " + in.creation_time + "\n";
    f << "-Time of file writing: " << engine_utils::current_date_time() << "\n";

    f << "-Reduction type: " << redu_type << "\n";
    f << "-TOD type: " << tod_type << "\n";
    f << "-TOD unit: " << omb.sig_unit << "\n";
    f << "-TOD chunk type: " << in.name << "\n";

    f << "-Calibrated: " << in.status.calibrated << "\n";
    f << "-Extinction Corrected: " << in.status.extinction_corrected << "\n";
    f << "-Demodulated: " << in.status.demodulated << "\n";
    f << "-Kernel Generated: " << in.status.kernel_generated << "\n";
    f << "-Despiked: " << in.status.despiked << "\n";
    f << "-TOD filtered: " << in.status.tod_filtered << "\n";
    f << "-TOD notch enabled: " << rtcproc.run_tod_notch << "\n";
    f << "-TOD IIR highpass enabled: " << rtcproc.run_tod_iir_highpass << "\n";
    f << "-TOD IIR highpass freq (Hz): " << rtcproc.filter.iir_highpass_freq_Hz << "\n";
    f << "-TOD IIR highpass order: " << rtcproc.filter.iir_highpass_order << "\n";
    f << "-TOD IIR highpass zero-phase: " << rtcproc.filter.iir_highpass_zero_phase << "\n";
    f << "-Downsampled: " << in.status.downsampled << "\n";
    f << "-Cleaned: " << in.status.cleaned << "\n";

    f << "-Scan length: " << in.scans.data.rows() << "\n";

    f << "-Number of detectors: " << in.scans.data.cols() << "\n";
    f << "-Number of detectors flagged in APT table: " << (calib.apt["flag"].array()!=0).count() << "\n";
    f << "-Number of detectors flagged below weight limit: " << in.n_dets_low <<"\n";
    f << "-Number of detectors flagged above weight limit: " << in.n_dets_high << "\n";
    Eigen::Index n_flagged = in.n_dets_low + in.n_dets_high + (calib.apt["flag"].array()!=0).count();
    f << "-Number of detectors flagged: " << n_flagged << " (" << 100*float(n_flagged)/float(in.scans.data.cols()) << "%)\n";

    f << "-NaNs found: " << in.scans.data.array().isNaN().count() << "\n";
    f << "-Infs found: " << in.scans.data.array().isInf().count() << "\n";
    f << "-Data min: " << in.scans.data.minCoeff() << " " << omb.sig_unit << "\n";
    f << "-Data max: " << in.scans.data.maxCoeff() << " " << omb.sig_unit << "\n";
    f << "-Data mean: " << in.scans.data.mean() << " " << omb.sig_unit << "\n";
    f << "-Data median: " << tula::alg::median(in.scans.data) << " " << omb.sig_unit << "\n";
    f << "-Data stddev: " << engine_utils::calc_std_dev(in.scans.data) << " " << omb.sig_unit << "\n";

    if (in.status.kernel_generated) {
        f << "-Kernel max: " << in.kernel.data.maxCoeff() << " " << omb.sig_unit << "\n";
    }

    f.close();
}

template <typename map_buffer_t>
void Engine::write_map_summary(map_buffer_t &mb) {

    logger->debug("writing map summary files");

    std::string filename = "map_summary";
    std::ofstream f;
    f.open (obsnum_dir_name+"/logs/" + filename + ".log");

    f << "Summary file for maps\n";
    f << "-Citlali version: " << CITLALI_GIT_VERSION << "\n";
    f << "-Kidscpp version: " << KIDSCPP_GIT_VERSION << "\n";
    f << "-Time of file writing: " << engine_utils::current_date_time() << "\n";

    f << "-Reduction type: " << redu_type << "\n";
    f << "-Map type: " << tod_type << "\n";
    f << "-Map grouping: " << map_grouping << "\n";
    f << "-Rows: " << mb.n_rows << "\n";
    f << "-Cols: " << mb.n_cols << "\n";
    f << "-Number of maps: " << n_maps << "\n";
    f << "-Signal map unit: " << mb.sig_unit << "\n";
    f << "-Weight map unit: " << "1/(" + mb.sig_unit + ")^2" << "\n";
    f << "-Kernel maps generated: " << !mb.kernel.empty() << "\n";
    f << "-Coverage maps generated: " << !mb.coverage.empty() << "\n";
    f << "-Noise maps generated: " << !mb.noise.empty() << "\n";
    f << "-Number of noise maps: " << mb.noise.size() << "\n";

    // map to count nans for all maps
    std::map<std::string,int> n_nans;
    n_nans["signal"] = 0;
    n_nans["weight"] = 0;
    n_nans["kernel"] = 0;
    n_nans["coverage"] = 0;
    n_nans["noise"] = 0;

    // maps to hold infs for all maps
    std::map<std::string,int> n_infs;
    n_infs["signal"] = 0;
    n_infs["weight"] = 0;
    n_infs["kernel"] = 0;
    n_infs["coverage"] = 0;
    n_infs["noise"] = 0;

    // loop through maps and count up nans and infs
    for (Eigen::Index i=0; i<mb.signal.size(); ++i) {
        n_nans["signal"] = n_nans["signal"] + mb.signal[i].array().isNaN().count();
        n_nans["weight"] = n_nans["weight"] + mb.weight[i].array().isNaN().count();

        // check kernel for nans if requested
        if (!mb.kernel.empty()) {
            n_nans["kernel"] = n_nans["kernel"] + mb.kernel[i].array().isNaN().count();
        }
        // check coverage map for nans if available
        if (!mb.coverage.empty()) {
            n_nans["coverage"] = n_nans["coverage"] + mb.coverage[i].array().isNaN().count();
        }

        n_infs["signal"] = n_infs["signal"] + mb.signal[i].array().isInf().count();
        n_infs["weight"] = n_infs["weight"] + mb.weight[i].array().isInf().count();

        // check kernel for infs if requested
        if (!mb.kernel.empty()) {
            n_infs["kernel"] = n_infs["kernel"] + mb.kernel[i].array().isInf().count();
        }
        // check coverage map for infs if available
        if (!mb.coverage.empty()) {
            n_infs["coverage"] = n_infs["coverage"] + mb.coverage[i].array().isInf().count();
        }

        // loop through noise maps and check for nans and infs
        if (!mb.noise.empty()) {
            const Eigen::Index n_noise_maps = mb.noise[i].dimension(2);
            for (Eigen::Index j=0; j<n_noise_maps; ++j) {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(mb.noise[i].data() + j * mb.n_rows * mb.n_cols,
                                                                                               mb.n_rows, mb.n_cols);

                n_nans["noise"] = n_nans["noise"] + noise_matrix.array().isNaN().count();
                n_infs["noise"] = n_infs["noise"] + noise_matrix.array().isInf().count();
            }
        }
    }

    for (auto const& [key, val] : n_nans) {
         f << "-Number of "+ key + " NaNs: " << val << "\n";
    }

    for (auto const& [key, val] : n_infs) {
        f << "-Number of "+ key + " Infs: " << val << "\n";
    }
}

template <mapmaking::MapType map_t, engine_utils::toltecIO::DataType data_t, engine_utils::toltecIO::ProdType prod_t>
auto Engine::setup_filenames(std::string dir_name) {

    std::string filename;

    // raw obs maps
    if constexpr (map_t == mapmaking::RawObs) {
        filename = toltec_io.create_filename<data_t, prod_t, engine_utils::toltecIO::raw>
                   (dir_name, redu_type, "", obsnum, telescope.sim_obs);
    }
    // filtered obs maps
    else if constexpr (map_t == mapmaking::FilteredObs) {
        filename = toltec_io.create_filename<data_t, prod_t, engine_utils::toltecIO::filtered>
                   (dir_name, redu_type, "", obsnum, telescope.sim_obs);
    }
    // raw coadded maps
    else if constexpr (map_t == mapmaking::RawCoadd) {
        filename = toltec_io.create_filename<data_t, prod_t, engine_utils::toltecIO::raw>
                   (dir_name, "", "", "", telescope.sim_obs);
    }
    // filtered coadded maps
    else if constexpr (map_t == mapmaking::FilteredCoadd) {
        filename = toltec_io.create_filename<data_t, prod_t, engine_utils::toltecIO::filtered>
                   (dir_name, "", "", "", telescope.sim_obs);
    }

    return filename;
}

auto Engine::get_map_name(int i) {
    // get name for extension layer
    std::string map_name = "";

    // only update name if we're not in array mode
    if (map_grouping!="array") {
        // if in nw mode
        if (map_grouping=="nw") {
            map_name = map_name + "nw_" + std::to_string(calib.nws(i)) + "_";
        }
        else if (map_grouping=="fg") {
            // find all detectors belonging to each fg
            Eigen::VectorXI array_indices(calib.fg.size()*calib.n_arrays*rtcproc.polarization.stokes_params.size());
            Eigen::Index k = 0;
            for (Eigen::Index j=0; j<calib.n_arrays; ++j) {
                for (Eigen::Index l=0; l<rtcproc.polarization.stokes_params.size(); ++l) {
                    for (Eigen::Index m=0; m<calib.fg.size(); ++m) {
                        array_indices(k) = calib.fg(m);
                        k++;
                    }
                }
            }
            // if in fg mode
            map_name = map_name + "fg_" + std::to_string(array_indices(i)) + "_";
        }
        // if in detector mode
        else if (map_grouping=="detector") {
            map_name = map_name + "det_" + std::to_string(i) + "_";
        }
    }

    return map_name;
}

template <typename fits_io_type, class map_buffer_t>
void Engine::add_phdu(fits_io_type &fits_io, map_buffer_t &mb, Eigen::Index i) {
    if (i < 0 || i >= static_cast<Eigen::Index>(fits_io->size())) {
        logger->error("add_phdu index out of range: i={} fits_io_size={}",
                      static_cast<long long>(i), static_cast<long long>(fits_io->size()));
        std::exit(EXIT_FAILURE);
    }
    if (i >= calib.arrays.size()) {
        logger->error("add_phdu array index out of range: i={} calib.arrays.size={}",
                      static_cast<long long>(i), static_cast<long long>(calib.arrays.size()));
        std::exit(EXIT_FAILURE);
    }

    // array name
    std::string name = toltec_io.array_name_map[calib.arrays(i)];

    logger->debug("adding unit conversions");

    // conversion to uK
    auto fwhm = (std::get<0>(calib.array_fwhms[calib.arrays(i)]) + std::get<1>(calib.array_fwhms[calib.arrays(i)]))/2;
    auto mJy_beam_to_uK = engine_utils::mJy_beam_to_uK(1, toltec_io.array_freq_map[calib.arrays(i)], fwhm*ASEC_TO_RAD);

    // beam area in steradians
    auto beam_area_rad = 2.*pi*pow(fwhm*FWHM_TO_STD*ASEC_TO_RAD,2);
    // get Jy/pixel
    auto mJy_beam_to_Jy_px = 1e-3/beam_area_rad*pow(mb->pixel_size_rad,2);

    auto get_tel_header_scalar = [&](const std::string &key, double fallback) {
        auto it = telescope.tel_header.find(key);
        if (it == telescope.tel_header.end() || it->second.size() < 1) {
            logger->warn("tel_header '{}' missing/empty; using fallback {}", key, fallback);
            return fallback;
        }
        const double value = it->second(0);
        if (!std::isfinite(value)) {
            logger->warn("tel_header '{}' non-finite ({}); using fallback {}", key, value, fallback);
            return fallback;
        }
        return value;
    };

    auto get_tel_data_mean = [&](const std::string &key, double fallback) {
        auto it = telescope.tel_data.find(key);
        if (it == telescope.tel_data.end() || it->second.size() < 1) {
            logger->warn("tel_data '{}' missing/empty; using fallback {}", key, fallback);
            return fallback;
        }
        const double value = it->second.mean();
        if (!std::isfinite(value)) {
            logger->warn("tel_data '{}' mean non-finite ({}); using fallback {}", key, value, fallback);
            return fallback;
        }
        return value;
    };

    // add unit conversions
    if (rtcproc.run_calibrate) {
        if (mb->sig_unit == "mJy/beam") {
            // conversion to mJy/beam
            fits_io->at(i).pfits->pHDU().addKey("to_mJy/beam", 1, "Conversion to mJy/beam");
            // conversion to MJy/sr
            fits_io->at(i).pfits->pHDU().addKey("to_MJy/sr", 1/(calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC),
                                                "Conversion to MJy/sr");
            // conversion to uK
            fits_io->at(i).pfits->pHDU().addKey("to_uK", mJy_beam_to_uK, "Conversion to uK");
            // conversion to Jy/pixel
            fits_io->at(i).pfits->pHDU().addKey("to_Jy/pixel", mJy_beam_to_Jy_px, "Conversion to Jy/pixel");
        }
        else if (mb->sig_unit == "MJy/sr") {
            // conversion to mJy/beam
            fits_io->at(i).pfits->pHDU().addKey("to_mJy/beam", calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC,
                                                "Conversion to mJy/beam");
            // conversion to MJy/Sr
            fits_io->at(i).pfits->pHDU().addKey("to_MJy/sr", 1, "Conversion to MJy/sr");
            // conversion to uK
            fits_io->at(i).pfits->pHDU().addKey("to_uK", calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC*mJy_beam_to_uK,
                                                "Conversion to uK");
            // conversion to Jy/pixel
            fits_io->at(i).pfits->pHDU().addKey("to_Jy/pixel", calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC*mJy_beam_to_Jy_px,
                                                "Conversion to Jy/pixel");
        }
        else if (mb->sig_unit == "uK") {
            // conversion to mJy/beam
            fits_io->at(i).pfits->pHDU().addKey("to_mJy/beam", 1/mJy_beam_to_uK, "Conversion to mJy/beam");
            // conversion to MJy/sr
            fits_io->at(i).pfits->pHDU().addKey("to_MJy/sr", 1/mJy_beam_to_uK/(calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC),
                                                "Conversion to MJy/sr");
            // conversion to uK
            fits_io->at(i).pfits->pHDU().addKey("to_uK", 1, "Conversion to uK");
            // conversion to Jy/pixel
            fits_io->at(i).pfits->pHDU().addKey("to_Jy/pixel", (1/mJy_beam_to_uK)*mJy_beam_to_Jy_px, "Conversion to Jy/pixel");
        }
        else if (mb->sig_unit == "Jy/pixel") {
            // conversion to mJy/beam
            fits_io->at(i).pfits->pHDU().addKey("to_mJy/beam", 1/mJy_beam_to_Jy_px, "Conversion to mJy/beam");
            // conversion to MJy/sr
            fits_io->at(i).pfits->pHDU().addKey("to_MJy/sr", (1/mJy_beam_to_Jy_px)/(calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC),
                                                "Conversion to MJy/sr");
            // conversion to uK
            fits_io->at(i).pfits->pHDU().addKey("to_uK", mJy_beam_to_uK/mJy_beam_to_Jy_px, "Conversion to uK");
            // conversion to Jy/pixel
            fits_io->at(i).pfits->pHDU().addKey("to_Jy/pixel", 1, "Conversion to Jy/pixel");
        }
    }
    // if flux calibration is disabled
    else {
        fits_io->at(i).pfits->pHDU().addKey("to_mJy/beam", "N/A", "Conversion to mJy/beam");
        fits_io->at(i).pfits->pHDU().addKey("to_MJy/sr", "N/A", "Conversion to MJy/sr");
        fits_io->at(i).pfits->pHDU().addKey("to_uK", "N/A", "Conversion to uK");
        fits_io->at(i).pfits->pHDU().addKey("to_Jy/pixel", "N/A", "Conversion to Jy/pixel");
    }

    // add source flux for beammaps
    if (redu_type == "beammap") {
        fits_io->at(i).pfits->pHDU().addKey("HEADER.SOURCE.FLUX_MJYPERBEAM", beammap_fluxes_mJy_beam[name], "Source flux (mJy/beam)");
        fits_io->at(i).pfits->pHDU().addKey("HEADER.SOURCE.FLUX_MJYPERSR", beammap_fluxes_MJy_Sr[name], "Source flux (MJy/sr)");

        fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.ITER_TOLERANCE", beammap_iter_tolerance, "Beammap iteration tolerance");
        fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.ITER_MAX", beammap_iter_max, "Beammap max iterations");
        fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.IS_DEROTATED", beammap_derotate, "Beammap derotated");
        // add reference detector information
        if (beammap_subtract_reference) {
            int ref_det_index = beammap_reference_det;
            if (calib.apt_meta["reference_det"]) {
                ref_det_index = calib.apt_meta["reference_det"].as<int>();
            }
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_DET_INDEX", ref_det_index, "Beammap Reference det (rotation center)");
            double ref_x_t = -99.0;
            double ref_y_t = -99.0;
            if (calib.apt_meta["reference_x_t"]) {
                ref_x_t = calib.apt_meta["reference_x_t"].as<double>();
            }
            else if (ref_det_index >= 0 && ref_det_index < calib.apt["x_t"].size()) {
                ref_x_t = calib.apt["x_t"](ref_det_index);
            }
            if (calib.apt_meta["reference_y_t"]) {
                ref_y_t = calib.apt_meta["reference_y_t"].as<double>();
            }
            else if (ref_det_index >= 0 && ref_det_index < calib.apt["y_t"].size()) {
                ref_y_t = calib.apt["y_t"](ref_det_index);
            }
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_X_T", ref_x_t, "Az rotation center (arcsec)");
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_Y_T", ref_y_t, "Alt rotation center (arcsec)");
        }
        else {
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_DET_INDEX", -99, "Beammap Reference det (rotation center)");
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_X_T", "N/A", "Az rotation center (arcsec)");
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_Y_T", "N/A", "Alt rotation center (arcsec)");
        }
    }

    logger->debug("adding obsnums");

    // add obsnums
    for (Eigen::Index j=0; j<mb->obsnums.size(); ++j) {
        fits_io->at(i).pfits->pHDU().addKey("OBSNUM"+std::to_string(j), mb->obsnums.at(j), "Observation Number " + std::to_string(j));
    }

    // add date and time of obs
    if (mb->obsnums.size()==1) {
        fits_io->at(i).pfits->pHDU().addKey("DATEOBS0", date_obs.back(), "Date and time of observation 0");
    }
    else {
        for (Eigen::Index j=0; j<mb->obsnums.size(); ++j) {
            fits_io->at(i).pfits->pHDU().addKey("DATEOBS"+std::to_string(j), date_obs[j], "Date and time of observation "+std::to_string(j));
        }
    }

    logger->debug("adding obs info");

    // add source
    fits_io->at(i).pfits->pHDU().addKey("SOURCE", telescope.source_name, "Source name");
    // add instrument
    fits_io->at(i).pfits->pHDU().addKey("INSTRUME", "TolTEC", "Instrument");
    // add hwpr
    fits_io->at(i).pfits->pHDU().addKey("HWPR", calib.run_hwpr, "HWPR installed");
    // add telescope
    fits_io->at(i).pfits->pHDU().addKey("TELESCOP", "LMT", "Telescope");
    // add wavelength
    fits_io->at(i).pfits->pHDU().addKey("WAV", name, "Wavelength");
    // add pipeline
    fits_io->at(i).pfits->pHDU().addKey("PIPELINE", "CITLALI", "Redu pipeline");
    // add citlali version
    fits_io->at(i).pfits->pHDU().addKey("VERSION", CITLALI_GIT_VERSION, "CITLALI_GIT_VERSION");
    // add kids version
    fits_io->at(i).pfits->pHDU().addKey("KIDS", KIDSCPP_GIT_VERSION, "KIDSCPP_GIT_VERSION");
    // add kids version
    fits_io->at(i).pfits->pHDU().addKey("TULA", TULA_GIT_VERSION, "TULA_GIT_VERSION");
    // project id
    fits_io->at(i).pfits->pHDU().addKey("PROJID", telescope.project_id, "Project ID");
    // add redu type
    fits_io->at(i).pfits->pHDU().addKey("GOAL", redu_type, "Reduction type");
    // add obs goal
    fits_io->at(i).pfits->pHDU().addKey("OBSGOAL", telescope.obs_goal, "Obs goal");
    // add tod type
    fits_io->at(i).pfits->pHDU().addKey("TYPE", tod_type, "TOD Type");
    // add map grouping
    fits_io->at(i).pfits->pHDU().addKey("GROUPING", map_grouping, "Map grouping");
    // add map grouping
    fits_io->at(i).pfits->pHDU().addKey("METHOD", map_method, "Map method");
    // add exposure time
    fits_io->at(i).pfits->pHDU().addKey("EXPTIME", mb->exposure_time, "Exposure time (sec)");
    // add pixel axes
    fits_io->at(i).pfits->pHDU().addKey("RADESYS", telescope.pixel_axes, "Coord Reference Frame");
    const double source_ra = get_tel_header_scalar("Header.Source.Ra", 0.0);
    const double source_dec = get_tel_header_scalar("Header.Source.Dec", 0.0);
    // add source ra
    fits_io->at(i).pfits->pHDU().addKey("SRC_RA", source_ra, "Source RA (radians)");
    // add source dec
    fits_io->at(i).pfits->pHDU().addKey("SRC_DEC", source_dec, "Source Dec (radians)");
    // add map tangent point ra
    fits_io->at(i).pfits->pHDU().addKey("TAN_RA", source_ra, "Map Tangent Point RA (radians)");
    //add map tangent point dec
    fits_io->at(i).pfits->pHDU().addKey("TAN_DEC", source_dec, "Map Tangent Point Dec (radians)");
    // add mean alt
    fits_io->at(i).pfits->pHDU().addKey("MEAN_EL", RAD_TO_DEG*get_tel_data_mean("TelElAct", 0.0), "Mean Elevation (deg)");
    // add mean az
    fits_io->at(i).pfits->pHDU().addKey("MEAN_AZ", RAD_TO_DEG*get_tel_data_mean("TelAzAct", 0.0), "Mean Azimuth (deg)");
    // add mean parallactic angle
    fits_io->at(i).pfits->pHDU().addKey("MEAN_PA", RAD_TO_DEG*get_tel_data_mean("ActParAng", 0.0), "Mean Parallactic angle (deg)");

    logger->debug("adding beamsizes");

    // add beamsizes
    if (std::get<0>(calib.array_fwhms[calib.arrays(i)]) >= std::get<1>(calib.array_fwhms[calib.arrays(i)])) {
        fits_io->at(i).pfits->pHDU().addKey("BMAJ", std::get<0>(calib.array_fwhms[calib.arrays(i)]), "beammaj (arcsec)");
        fits_io->at(i).pfits->pHDU().addKey("BMIN", std::get<1>(calib.array_fwhms[calib.arrays(i)]), "beammin (arcsec)");
        fits_io->at(i).pfits->pHDU().addKey("BPA", calib.array_pas[calib.arrays(i)]*RAD_TO_DEG, "beampa (deg)");
    }
    else {
        fits_io->at(i).pfits->pHDU().addKey("BMAJ", std::get<1>(calib.array_fwhms[calib.arrays(i)]), "beammaj (arcsec)");
        fits_io->at(i).pfits->pHDU().addKey("BMIN", std::get<0>(calib.array_fwhms[calib.arrays(i)]), "beammin (arcsec)");
        fits_io->at(i).pfits->pHDU().addKey("BPA", (calib.array_pas[calib.arrays(i)] + pi/2)*RAD_TO_DEG, "beampa (deg)");
    }

    fits_io->at(i).pfits->pHDU().addKey("BUNIT", mb->sig_unit, "bunit");

    // add jinc shape params
    if (map_method=="jinc") {
        logger->debug("adding jinc params");

        fits_io->at(i).pfits->pHDU().addKey("JINC_R", jinc_mm.r_max, "Jinc filter R_max");
        fits_io->at(i).pfits->pHDU().addKey("JINC_A", jinc_mm.shape_params[calib.arrays(i)][0], "Jinc filter param a");
        fits_io->at(i).pfits->pHDU().addKey("JINC_B", jinc_mm.shape_params[calib.arrays(i)][1], "Jinc filter param b");
        fits_io->at(i).pfits->pHDU().addKey("JINC_C", jinc_mm.shape_params[calib.arrays(i)][2], "Jinc filter param c");
    }

    // add mean tau
    logger->debug("adding extinction");
    double mean_tau = 0.0;
    if (rtcproc.run_extinction) {
        const auto tel_it = telescope.tel_data.find("TelElAct");
        if (tel_it != telescope.tel_data.end() && tel_it->second.size() > 0) {
            Eigen::VectorXd tau_el(1);
            tau_el << tel_it->second.mean();
            auto tau_freq = rtcproc.calibration.calc_tau(tau_el, telescope.tau_225_GHz);
            const auto array_id = calib.arrays(i);
            const auto tau_it = tau_freq.find(array_id);
            if (tau_it != tau_freq.end() && tau_it->second.size() > 0 && std::isfinite(tau_it->second(0))) {
                mean_tau = tau_it->second(0);
            }
            else {
                logger->warn("MEAN_TAU unavailable for array {} (tau_freq missing/empty); defaulting to 0", array_id);
            }
        }
        else {
            logger->warn("MEAN_TAU unavailable (TelElAct missing/empty); defaulting to 0");
        }
    }
    fits_io->at(i).pfits->pHDU().addKey("MEAN_TAU", mean_tau, "mean tau (" + name + ")");

    // add sample rate
    fits_io->at(i).pfits->pHDU().addKey("SAMPRATE", telescope.fsmp, "sample rate (Hz)");

    // add apt table to header
    if (mb->obsnums.size()==1) {
        std::string apt_name = "N/A";
        if (!calib.apt_filepath.empty()) {
            std::vector<string> apt_filename;
            std::stringstream ss(calib.apt_filepath);
            std::string item;
            char delim = '/';

            while (getline (ss, item, delim)) {
                apt_filename.push_back(item);
            }
            if (!apt_filename.empty()) {
                apt_name = apt_filename.back();
            }
            else {
                logger->warn("APT filepath '{}' parsed empty; using N/A", calib.apt_filepath);
            }
        }
        else {
            logger->warn("APT filepath empty; using N/A");
        }
        fits_io->at(i).pfits->pHDU().addKey("APT", apt_name, "APT table used");
    }

    double rms;

    if (redu_type != "beammap") {
        // estimate rms from weight maps
        rms = pow(mb->median_err(i),0.5);
    }
    else {
        rms = 0.0;
    }

    // out-of-focus holography parameters
    if (! telescope.sim_obs) {
	    logger->debug("adding oof params");
	    fits_io->at(i).pfits->pHDU().addKey("OOF_RMS", rms, "rms of map background (" + mb->sig_unit +")");
	    fits_io->at(i).pfits->pHDU().addKey("OOF_W", toltec_io.array_wavelength_map[calib.arrays(i)]/1000., "wavelength (m)");
	    fits_io->at(i).pfits->pHDU().addKey("OOF_ID", static_cast<int>(toltec_io.array_wavelength_map[calib.arrays(i)]*1000), "instrument id");
	    fits_io->at(i).pfits->pHDU().addKey("OOF_T", 3.0, "taper (dB)");
	    fits_io->at(i).pfits->pHDU().addKey("OOF_M2X", get_tel_header_scalar("Header.M2.XReq", 0.0)/1000.*1e6, "oof m2x (microns)");
	    fits_io->at(i).pfits->pHDU().addKey("OOF_M2Y", get_tel_header_scalar("Header.M2.YReq", 0.0)/1000.*1e6, "oof m2y (microns)");
	    fits_io->at(i).pfits->pHDU().addKey("OOF_M2Z", get_tel_header_scalar("Header.M2.ZReq", 0.0)/1000.*1e6, "oof m2z (microns)");

	    fits_io->at(i).pfits->pHDU().addKey("OOF_RO", 25., "outer diameter of the antenna (m)");
	    fits_io->at(i).pfits->pHDU().addKey("OOF_RI", 1.65, "inner diameter of the antenna (m)");
    }

    fits_io->at(i).pfits->pHDU().addKey("FRUITLOOPS_ITER", fruit_iter, "Current fruit loops iteration");

    // add control/runtime parameters
    logger->debug("adding config params");
    const bool run_any_tod_filter = rtcproc.run_tod_filter || rtcproc.run_tod_iir_highpass;
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.VERBOSE", verbose_mode, "Reduced in verbose mode");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.POLARIZED", rtcproc.run_polarization, "Polarized Obs");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKED", rtcproc.run_despike, "Despiked");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.ENABLED",
                                        rtcproc.despiker.local_residual.enabled,
                                        "Enable local-residual RTC despike pass");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.WINDOW_SEC",
                                        rtcproc.despiker.local_residual.window_sec,
                                        "Local-residual despike smoothing window");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.SIGMA_SCALE",
                                        rtcproc.despiker.local_residual.sigma_scale,
                                        "Local-residual despike raw threshold scale");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.DELTA_SIGMA_SCALE",
                                        rtcproc.despiker.local_residual.delta_sigma_scale,
                                        "Local-residual despike delta threshold scale");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.RAW_GATE.ENABLED",
                                        rtcproc.despiker.local_residual.compact_raw_gate.enabled,
                                        "Enable compact morphology gate for local-residual raw candidates");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_REL_SIGMA_SCALE",
                                        rtcproc.despiker.local_residual.compact_raw_gate.candidate_rel_sigma_scale,
                                        "Candidate threshold scale relative to the accepted local-residual raw threshold");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_SIGMA_SCALE",
                                        rtcproc.despiker.local_residual.compact_raw_gate.candidate_rel_sigma_scale *
                                            rtcproc.despiker.local_residual.sigma_scale,
                                        "Effective candidate threshold scale in units of min_spike_sigma for compact local-residual raw gate");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.RAW_GATE.WINDOW_SEC",
                                        rtcproc.despiker.local_residual.compact_raw_gate.window_sec,
                                        "Window used to score compactness of local-residual raw candidates");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.RAW_GATE.HALF_PEAK_FRAC",
                                        rtcproc.despiker.local_residual.compact_raw_gate.half_peak_frac,
                                        "Half-peak fraction used to measure local-residual raw candidate width");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_WIDTH_SEC",
                                        rtcproc.despiker.local_residual.compact_raw_gate.max_width_sec,
                                        "Maximum width allowed for compact local-residual raw candidates");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_STEP_SHIFT_Z",
                                        rtcproc.despiker.local_residual.compact_raw_gate.max_step_shift_z,
                                        "Maximum allowed pre/post baseline shift for compact local-residual raw candidates");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.DELTA_GATE.ENABLED",
                                        rtcproc.despiker.local_residual.compact_delta_gate.enabled,
                                        "Enable compact morphology gate for local-residual delta candidates");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.DELTA_GATE.WINDOW_SEC",
                                        rtcproc.despiker.local_residual.compact_delta_gate.window_sec,
                                        "Window used to score compactness of local-residual delta candidates");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.DELTA_GATE.HALF_PEAK_FRAC",
                                        rtcproc.despiker.local_residual.compact_delta_gate.half_peak_frac,
                                        "Half-peak fraction used to measure local-residual delta candidate width");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_WIDTH_SEC",
                                        rtcproc.despiker.local_residual.compact_delta_gate.max_width_sec,
                                        "Maximum width allowed for compact local-residual delta candidates");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_STEP_SHIFT_Z",
                                        rtcproc.despiker.local_residual.compact_delta_gate.max_step_shift_z,
                                        "Maximum allowed pre/post baseline shift for compact local-residual delta candidates");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODFILTERED", run_any_tod_filter, "TOD Filtered");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODNOTCH", rtcproc.run_tod_notch, "TOD notch enabled");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODIIRHP", rtcproc.run_tod_iir_highpass, "TOD IIR highpass enabled");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODIIRHP.FREQ_HZ", rtcproc.filter.iir_highpass_freq_Hz, "TOD IIR highpass cutoff frequency");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODIIRHP.ORDER", rtcproc.filter.iir_highpass_order, "TOD IIR highpass cascaded order");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODIIRHP.ZEROPHASE", rtcproc.filter.iir_highpass_zero_phase, "TOD IIR highpass forward-backward");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DOWNSAMPLED", rtcproc.run_downsample, "Downsampled");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CALIBRATED", rtcproc.run_calibrate, "Calibrated");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.EXTINCTION", rtcproc.run_extinction, "Extinction corrected");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.EXTINCTION.EXTMODEL", rtcproc.calibration.extinction_model, "Extinction model");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.TYPE", ptcproc.weighting_type, "Weighting scheme");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.INV_VAR.RTC.WTLOW", rtcproc.lower_inv_var_factor, "RTC lower inv var cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.INV_VAR.RTC.WTHIGH", rtcproc.upper_inv_var_factor, "RTC upper inv var cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.STEP_MASK.ENABLED",
                                        rtcproc.network_step_mask.enabled,
                                        "Enable RTC network-window step masking");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.STEP_MASK.STEP_WINDOW_SEC",
                                        rtcproc.network_step_mask.step_window_sec,
                                        "Window used for RTC step-score estimation");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.STEP_MASK.STEP_SCORE_THRESH",
                                        rtcproc.network_step_mask.step_score_thresh,
                                        "Detector step-score threshold for RTC step masking");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.STEP_MASK.MIN_GOOD_FRAC",
                                        rtcproc.network_step_mask.min_good_frac,
                                        "Minimum good-sample fraction for RTC step-mask metrics");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.STEP_MASK.MIN_DET_USED",
                                        static_cast<int>(rtcproc.network_step_mask.min_det_used),
                                        "Minimum detectors required in a network for RTC step masking");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.STEP_MASK.MIN_STEP_DET_FRAC",
                                        rtcproc.network_step_mask.min_step_det_frac,
                                        "Minimum step-like detector fraction for RTC step masking");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.STEP_MASK.MIN_ALIGNMENT_FRAC",
                                        rtcproc.network_step_mask.min_alignment_frac,
                                        "Minimum aligned-step detector fraction for RTC step masking");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.STEP_MASK.CLUSTER_TOL_SEC",
                                        rtcproc.network_step_mask.cluster_tol_sec,
                                        "Allowed timing tolerance for aligned RTC step clusters");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.STEP_MASK.HALF_WIDTH_SEC",
                                        rtcproc.network_step_mask.mask_half_width_sec,
                                        "Half-width of the applied RTC step-mask window");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.STEP_MASK.MAX_FLAGGED_FRAC",
                                        rtcproc.network_step_mask.max_flagged_fraction,
                                        "Maximum allowed newly flagged detector-sample fraction per RTC network mask");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.IMPULSIVE.ENABLED",
                                        rtcproc.impulsive_capture.enabled,
                                        "Enable RTC impulsive-event snippet capture");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.IMPULSIVE.MIN_GOOD_FRAC",
                                        rtcproc.impulsive_capture.min_good_frac,
                                        "Minimum good-sample fraction for RTC impulsive capture");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.IMPULSIVE.MIN_EVENT_Z",
                                        rtcproc.impulsive_capture.min_event_z,
                                        "Minimum event score for RTC impulsive capture");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.IMPULSIVE.NEAR_EVENT_Z",
                                        rtcproc.impulsive_capture.near_event_z,
                                        "Near-threshold z for RTC impulsive counts");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.IMPULSIVE.MAX_EVENTS",
                                        static_cast<int>(rtcproc.impulsive_capture.max_events_per_network),
                                        "Maximum captured impulsive detectors per network");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.IMPULSIVE.HALF_WIDTH_SEC",
                                        rtcproc.impulsive_capture.snippet_half_width_sec,
                                        "Half-width of captured RTC impulsive snippets");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.INV_VAR.PTC.WTLOW", ptcproc.lower_inv_var_factor, "PTC lower inv var cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.INV_VAR.PTC.WTHIGH", ptcproc.upper_inv_var_factor, "PTC upper inv var cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.PTC.WTLOW", ptcproc.lower_weight_factor, "PTC lower weight cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.PTC.WTHIGH", ptcproc.upper_weight_factor, "PTC upper weight cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.MEDWTFACTOR", ptcproc.med_weight_factor, "Median weight factor");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.ENABLED",
                                        ptcproc.weight_corr_penalty.enabled,
                                        "Enable per-network corr-based weight penalties");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.MIN_GOOD_FRAC",
                                        ptcproc.weight_corr_penalty.min_good_frac,
                                        "Minimum unflagged sample fraction per detector");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.MIN_OVERLAP",
                                        ptcproc.weight_corr_penalty.min_overlap,
                                        "Minimum overlap for pairwise corr metric");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.MAX_SAMPLES",
                                        ptcproc.weight_corr_penalty.max_samples,
                                        "Max sampled timestream points for penalty metrics");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.MAX_PAIRS",
                                        ptcproc.weight_corr_penalty.max_pairs,
                                        "Max sampled detector pairs for corr metric");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.FLOOR",
                                        ptcproc.weight_corr_penalty.floor,
                                        "Minimum per-network multiplicative weight factor");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.EXPONENT",
                                        ptcproc.weight_corr_penalty.exponent,
                                        "Exponent shaping corr penalty response");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.PAIR.ENABLED",
                                        ptcproc.weight_corr_penalty.pair_corr.enabled,
                                        "Enable pairwise corr penalty term");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.PAIR.REF",
                                        ptcproc.weight_corr_penalty.pair_corr.ref,
                                        "Pairwise corr reference value");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.PAIR.SPAN",
                                        ptcproc.weight_corr_penalty.pair_corr.span,
                                        "Pairwise corr scale span");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.PAIR.WEIGHT",
                                        ptcproc.weight_corr_penalty.pair_corr.weight,
                                        "Pairwise corr term weight");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.CM_EL.ENABLED",
                                        ptcproc.weight_corr_penalty.cm_el_corr.enabled,
                                        "Enable common-mode elevation corr penalty term");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.CM_EL.REF",
                                        ptcproc.weight_corr_penalty.cm_el_corr.ref,
                                        "Common-mode elevation corr reference");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.CM_EL.SPAN",
                                        ptcproc.weight_corr_penalty.cm_el_corr.span,
                                        "Common-mode elevation corr scale span");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.CM_EL.WEIGHT",
                                        ptcproc.weight_corr_penalty.cm_el_corr.weight,
                                        "Common-mode elevation corr term weight");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.ENABLED",
                                        ptcproc.weight_corr_penalty.cm_low_mid_ratio.enabled,
                                        "Enable common-mode low/mid ratio penalty term");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.REF",
                                        ptcproc.weight_corr_penalty.cm_low_mid_ratio.ref,
                                        "Common-mode low/mid ratio reference");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.SPAN",
                                        ptcproc.weight_corr_penalty.cm_low_mid_ratio.span,
                                        "Common-mode low/mid ratio scale span");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.WEIGHT",
                                        ptcproc.weight_corr_penalty.cm_low_mid_ratio.weight,
                                        "Common-mode low/mid ratio term weight");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMIN_HZ",
                                        ptcproc.weight_corr_penalty.cm_low_mid_ratio.low_min_Hz,
                                        "Low-band minimum frequency for low/mid ratio");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMAX_HZ",
                                        ptcproc.weight_corr_penalty.cm_low_mid_ratio.low_max_Hz,
                                        "Low-band maximum frequency for low/mid ratio");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMIN_HZ",
                                        ptcproc.weight_corr_penalty.cm_low_mid_ratio.mid_min_Hz,
                                        "Mid-band minimum frequency for low/mid ratio");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMAX_HZ",
                                        ptcproc.weight_corr_penalty.cm_low_mid_ratio.mid_max_Hz,
                                        "Mid-band maximum frequency for low/mid ratio");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED", ptcproc.run_clean, "Cleaned");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.MODESEL",
                                        ptcproc.cleaner.active_cleaner_label(),
                                        "PTC cleaner method");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.MP.ENABLED",
                                        ptcproc.cleaner.marchenko_pastur.enabled,
                                        "Marchenko-Pastur mode selection enabled");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.MP.BANDLOW_HZ",
                                        ptcproc.cleaner.marchenko_pastur.band_low_Hz,
                                        "MP covariance low-band edge (Hz)");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.MP.BANDHIGH_HZ",
                                        ptcproc.cleaner.marchenko_pastur.band_high_Hz,
                                        "MP covariance high-band edge (Hz)");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.MP.MAXMODES",
                                        ptcproc.cleaner.marchenko_pastur.max_modes,
                                        "MP max modes considered");
    if (ptcproc.run_clean) {
        fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.NEIG", ptcproc.cleaner.n_eig_to_cut[calib.arrays(i)].sum(),
                                            "Number of eigenvalues removed");
    }
    else {
        fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.NEIG", 0, "Number of eigenvalues removed");
    }

    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS", ptcproc.run_fruit_loops, "Fruit loops");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS.PATH", ptcproc.fruit_loops_path, "Fruit loops path");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS.TYPE", ptcproc.fruit_loops_type, "Fruit loops type");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS.S2N", ptcproc.fruit_loops_sig2noise, "Fruit loops S/N");
    {
        double flux_limit = 0.0;
        if (ptcproc.run_fruit_loops) {
            if (ptcproc.fruit_loops_flux.size() == calib.arrays.size()) {
                flux_limit = ptcproc.fruit_loops_flux(i);
            }
            else if (calib.arrays(i) < ptcproc.fruit_loops_flux.size()) {
                flux_limit = ptcproc.fruit_loops_flux(calib.arrays(i));
            }
        }
        fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS.FLUX", flux_limit,
                                            "Fruit loops flux (" + mb->sig_unit + ")");
    }
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS.MAXITER", ptcproc.fruit_loops_iters, "Fruit loops iterations");

    // add telescope file header information
    if (mb->obsnums.size()==1) {
        logger->debug("adding tel params");
        for (auto const& [key, val] : telescope.tel_header) {
            if (val.size() < 1 || !std::isfinite(val(0))) {
                logger->warn("skipping tel_header '{}' due to empty/non-finite value", key);
                continue;
            }
            logger->debug("adding {}: {}", key, val);
            fits_io->at(i).pfits->pHDU().addKey(key, val(0), key);
        }
    }
}

template <typename fits_io_type, class map_buffer_t>
void Engine::write_maps(fits_io_type &fits_io, fits_io_type &noise_fits_io, map_buffer_t &mb, Eigen::Index i) {
    if (i < 0 || i >= static_cast<Eigen::Index>(mb->signal.size()) ||
        i >= static_cast<Eigen::Index>(mb->weight.size())) {
        logger->error("write_maps map index out of range: i={} signal_size={} weight_size={}",
                      static_cast<long long>(i),
                      static_cast<long long>(mb->signal.size()),
                      static_cast<long long>(mb->weight.size()));
        std::exit(EXIT_FAILURE);
    }

    // get name for extension layer
    std::string map_name = get_map_name(i);

    // get the array for the given map
    Eigen::Index map_index = arrays_to_maps(i);
    // get the stokes parameter for the given map
    Eigen::Index stokes_index = maps_to_stokes(i);
    if (map_index < 0 || map_index >= static_cast<Eigen::Index>(fits_io->size())) {
        logger->error("write_maps file index out of range: map_index={} fits_io_size={} map_i={}",
                      static_cast<long long>(map_index),
                      static_cast<long long>(fits_io->size()),
                      static_cast<long long>(i));
        std::exit(EXIT_FAILURE);
    }
    if (stokes_index < 0 || stokes_index >= static_cast<Eigen::Index>(rtcproc.polarization.stokes_params.size())) {
        logger->error("write_maps stokes index out of range: stokes_index={} stokes_size={} map_i={}",
                      static_cast<long long>(stokes_index),
                      static_cast<long long>(rtcproc.polarization.stokes_params.size()),
                      static_cast<long long>(i));
        std::exit(EXIT_FAILURE);
    }
    if (maps_to_arrays(i) < 0 || maps_to_arrays(i) >= calib.arrays.size()) {
        logger->error("write_maps maps_to_arrays index out of range: maps_to_arrays(i)={} calib.arrays.size={} map_i={}",
                      static_cast<long long>(maps_to_arrays(i)),
                      static_cast<long long>(calib.arrays.size()),
                      static_cast<long long>(i));
        std::exit(EXIT_FAILURE);
    }

    double source_epoch = 2000.0;
    auto epoch_it = telescope.tel_header.find("Header.Source.Epoch");
    if (epoch_it != telescope.tel_header.end() && epoch_it->second.size() > 0 &&
        std::isfinite(epoch_it->second(0))) {
        source_epoch = epoch_it->second(0);
    }
    else {
        logger->warn("Header.Source.Epoch missing/invalid; using epoch={} for WCS", source_epoch);
    }

    // update wcs ctypes for frequency and stokes params
    mb->wcs.crval[2] = toltec_io.array_freq_map[calib.arrays[maps_to_arrays(i)]];
    mb->wcs.crval[3] = stokes_index;

    // signal map
    fits_io->at(map_index).add_hdu("signal_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->signal[i]);
    fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
    fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");

    // weight map
    fits_io->at(map_index).add_hdu("weight_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->weight[i]);
    fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
    fits_io->at(map_index).hdus.back()->addKey("UNIT", "1/("+mb->sig_unit+")^2", "Unit of map");
    if (redu_type != "beammap" && std::fabs(mb->median_err(i)) > std::numeric_limits<double>::epsilon()) {
        fits_io->at(map_index).hdus.back()->addKey("MEDERR", pow(mb->median_err(i),0.5), "Median Error ("+mb->sig_unit+")");
    }
    else {
        fits_io->at(map_index).hdus.back()->addKey("MEDERR", 0.0, "Median Error ("+mb->sig_unit+")");
    }

    // kernel map
    if (rtcproc.run_kernel) {
        fits_io->at(map_index).add_hdu("kernel_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->kernel[i]);
        fits_io->at(map_index).hdus.back()->addKey("TYPE",rtcproc.kernel.type, "Kernel type");

        // add fwhm
        double fwhm = -99;
        if (rtcproc.kernel.type!="fits") {
            if (rtcproc.kernel.fwhm_rad<=0) {
                fwhm = (std::get<0>(calib.array_fwhms[calib.arrays(i)]) + std::get<1>(calib.array_fwhms[calib.arrays(i)]))/2;
            }
            else {
                fwhm = rtcproc.kernel.fwhm_rad*RAD_TO_ASEC;
            }
        }
        fits_io->at(map_index).hdus.back()->addKey("FWHM",fwhm,"Kernel fwhm (arcsec)");
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
        fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");
    }

    // coverage map
    if (!mb->coverage.empty()) {
        fits_io->at(map_index).add_hdu("coverage_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->coverage[i]);
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
        fits_io->at(map_index).hdus.back()->addKey("UNIT", "sec", "Unit of map");
    }

    /* coverage bool and signal-to-noise maps */
    if (!mb->coverage.empty()) {
        // need these to use eigen select
        Eigen::MatrixXd ones, zeros;
        ones.setOnes(mb->weight[i].rows(), mb->weight[i].cols());
        zeros.setZero(mb->weight[i].rows(), mb->weight[i].cols());

        // get weight threshold for current map
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = mb->calc_cov_region(i);
        // if weight is less than threshold, set to zero, otherwise set to one
        Eigen::MatrixXd coverage_bool = (mb->weight[i].array() < weight_threshold).select(zeros,ones);

        // coverage bool map
        fits_io->at(map_index).add_hdu("coverage_bool_" + map_name + rtcproc.polarization.stokes_params[stokes_index], coverage_bool);
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
        fits_io->at(map_index).hdus.back()->addKey("UNIT", "N/A", "Unit of map");
        fits_io->at(map_index).hdus.back()->addKey("WTTHRESH", weight_threshold, "Weight threshold");

        // signal-to-noise map
        Eigen::MatrixXd sig2noise = mb->signal[i].array()*sqrt(mb->weight[i].array());
        fits_io->at(map_index).add_hdu("sig2noise_" + map_name + rtcproc.polarization.stokes_params[stokes_index], sig2noise);
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
        fits_io->at(map_index).hdus.back()->addKey("UNIT", "N/A", "Unit of map");
    }

    // write noise maps
    if (!mb->noise.empty()) {
        if (map_index < 0 || map_index >= static_cast<Eigen::Index>(noise_fits_io->size())) {
            logger->error("write_maps noise file index out of range: map_index={} noise_fits_io_size={} map_i={}",
                          static_cast<long long>(map_index),
                          static_cast<long long>(noise_fits_io->size()),
                          static_cast<long long>(i));
            std::exit(EXIT_FAILURE);
        }
        if (i >= static_cast<Eigen::Index>(mb->noise.size())) {
            logger->error("write_maps noise map index out of range: i={} noise_size={}",
                          static_cast<long long>(i), static_cast<long long>(mb->noise.size()));
            std::exit(EXIT_FAILURE);
        }
        for (Eigen::Index n=0; n<mb->n_noise; ++n) {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(mb->noise[i].data() + n * mb->n_rows * mb->n_cols,
                                                                                           mb->n_rows, mb->n_cols);

            noise_fits_io->at(map_index).add_hdu("signal_" + map_name + std::to_string(n) + "_" + rtcproc.polarization.stokes_params[stokes_index],
                                                 noise_matrix);
            noise_fits_io->at(map_index).add_wcs(noise_fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
            noise_fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");
            noise_fits_io->at(map_index).hdus.back()->addKey("MEDRMS", mb->median_rms[i], "Median RMS of noise maps");
        }
    }
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_psd(map_buffer_t &mb, std::string dir_name) {
    // get filename
    std::string filename = setup_filenames<map_t,engine_utils::toltecIO::toltec,engine_utils::toltecIO::psd>(dir_name);

    // create file
    netCDF::NcFile fo(filename + ".nc", netCDF::NcFile::replace);

    // loop through psd vector
    for (Eigen::Index i=0; i<mb->psds.size(); ++i) {
        // get name for extension layer
        std::string map_name = get_map_name(i);

        // get the array for the given map
        Eigen::Index map_index = arrays_to_maps(i);
        // get the stokes parameter for the given map
        Eigen::Index stokes_index = maps_to_stokes(i);

        auto array = calib.arrays[map_index];
        std::string name = toltec_io.array_name_map[array] + "_" + map_name + rtcproc.polarization.stokes_params[stokes_index];

        // add dimensions
        netCDF::NcDim psd_dim = fo.addDim(name + "_nfreq",mb->psds[i].size());
        netCDF::NcDim pds_2d_row_dim = fo.addDim(name + "_rows",mb->psd_2ds[i].rows());
        netCDF::NcDim pds_2d_col_dim = fo.addDim(name + "_cols",mb->psd_2ds[i].cols());

        std::vector<netCDF::NcDim> dims;
        dims.push_back(pds_2d_row_dim);
        dims.push_back(pds_2d_col_dim);

        // psd
        netCDF::NcVar psd_v = fo.addVar(name + "_psd",netCDF::ncDouble, psd_dim);
        psd_v.putVar(mb->psds[i].data());

        // psd freq
        netCDF::NcVar psd_freq_v = fo.addVar(name + "_psd_freq",netCDF::ncDouble, psd_dim);
        psd_freq_v.putVar(mb->psd_freqs[i].data());

        // transpose 2d psd and freq
        Eigen::MatrixXd psd_2d_transposed = mb->psd_2ds[i].transpose();
        Eigen::MatrixXd psd_2d_freq_transposed = mb->psd_2d_freqs[i].transpose();

        // 2d psd
        netCDF::NcVar psd_2d_v = fo.addVar(name + "_psd_2d",netCDF::ncDouble, dims);
        psd_2d_v.putVar(psd_2d_transposed.data());

        // 2d psd freq
        netCDF::NcVar psd_2d_freq_v = fo.addVar(name + "_psd_2d_freq",netCDF::ncDouble, dims);
        psd_2d_freq_v.putVar(psd_2d_freq_transposed.data());

        if (!mb->noise.empty()) {
            // add dimensions
            netCDF::NcDim noise_psd_dim = fo.addDim(name + "_noise_nfreq",mb->noise_psds[i].size());
            netCDF::NcDim noise_pds_2d_row_dim = fo.addDim(name + "_noise_rows",mb->noise_psd_2ds[i].rows());
            netCDF::NcDim noise_pds_2d_col_dim = fo.addDim(name + "_noise_cols",mb->noise_psd_2ds[i].cols());

            std::vector<netCDF::NcDim> noise_dims;
            noise_dims.push_back(noise_pds_2d_row_dim);
            noise_dims.push_back(noise_pds_2d_col_dim);

            // noise psd
            netCDF::NcVar noise_psd_v = fo.addVar(name + "_noise_psd",netCDF::ncDouble, noise_psd_dim);
            noise_psd_v.putVar(mb->noise_psds[i].data());

            // noise psd freq
            netCDF::NcVar noise_psd_freq_v = fo.addVar(name + "_noise_psd_freq",netCDF::ncDouble, noise_psd_dim);
            noise_psd_freq_v.putVar(mb->noise_psd_freqs[i].data());

            // transpose 2d noise psd and freq
            Eigen::MatrixXd noise_psd_2d_transposed = mb->noise_psd_2ds[i].transpose();
            Eigen::MatrixXd noise_psd_2d_freq_transposed = mb->noise_psd_2d_freqs[i].transpose();

            // 2d noise psd
            netCDF::NcVar noise_psd_2d_v = fo.addVar(name + "_noise_psd_2d",netCDF::ncDouble, noise_dims);
            noise_psd_2d_v.putVar(noise_psd_2d_transposed.data());

            // 2d noise psd freq
            netCDF::NcVar noise_psd_2d_freq_v = fo.addVar(name + "_noise_psd_2d_freq",netCDF::ncDouble, noise_dims);
            noise_psd_2d_freq_v.putVar(noise_psd_2d_freq_transposed.data());
        }
    }
    // close file
    fo.close();
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_hist(map_buffer_t &mb, std::string dir_name) {
    std::string filename = setup_filenames<map_t,engine_utils::toltecIO::toltec,engine_utils::toltecIO::hist>(dir_name);

    netCDF::NcFile fo(filename + ".nc", netCDF::NcFile::replace);
    netCDF::NcDim hist_bins_dim = fo.addDim("n_bins", mb->hist_n_bins);

    // loop through stored histograms
    for (Eigen::Index i=0; i<mb->hists.size(); ++i) {
        // string to hold name
        // get name for extension layer
        std::string map_name = get_map_name(i);

        // get the array for the given map
        Eigen::Index map_index = arrays_to_maps(i);
        // get the stokes parameter for the given map
        Eigen::Index stokes_index = maps_to_stokes(i);

        // array index
        auto array = calib.arrays[map_index];
        std::string name = toltec_io.array_name_map[array] + "_" + map_name + rtcproc.polarization.stokes_params[stokes_index];

        // histogram bins
        netCDF::NcVar hist_bins_v = fo.addVar(name + "_bins",netCDF::ncDouble, hist_bins_dim);
        hist_bins_v.putVar(mb->hist_bins[i].data());

        // histogram
        netCDF::NcVar hist_v = fo.addVar(name + "_hist",netCDF::ncDouble, hist_bins_dim);
        hist_v.putVar(mb->hists[i].data());

        if (!mb->noise.empty()) {
            // average noise histogram
            netCDF::NcVar hist_v = fo.addVar(name + "_noise_hist",netCDF::ncDouble, hist_bins_dim);
            hist_v.putVar(mb->noise_hists[i].data());
        }
    }
    // close file
    fo.close();
}

void Engine::write_stats() {
    std::string path = obsnum_dir_name + "raw/";
    // if using tod subdir, put stats file in it
    if (tod_output_subdir_name!="null") {
        if (!fs::exists(fs::status(path + tod_output_subdir_name))) {
            fs::create_directories(path + tod_output_subdir_name);
            path = path + tod_output_subdir_name + "/";
        }
    }
    // create stats filename
    auto stats_filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::stats,
                                                    engine_utils::toltecIO::raw>
                          (path, redu_type, "", obsnum, telescope.sim_obs);

    // det stats header
    std::map<std::string, std::string> det_stats_header_units {
        {"rms", omb.sig_unit},
        {"stddev",omb.sig_unit},
        {"median",omb.sig_unit},
        {"flagged_frac","N/A"},
        {"weights","1/(" + omb.sig_unit + ")^2"},
        };
    // group stats header
    std::map<std::string, std::string> grp_stats_header_units {
        {"median_weights", "1/(" + omb.sig_unit + ")^2"},
        };

    netCDF::NcFile fo(stats_filename + ".nc", netCDF::NcFile::replace);

    // add obsnum
    netCDF::NcVar obsnum_v = fo.addVar("obsnum",netCDF::ncInt);
    obsnum_v.putAtt("units","N/A");
    int obsnum_int = std::stoi(obsnum);
    obsnum_v.putVar(&obsnum_int);

    // add dimensions
    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", calib.n_dets);
    netCDF::NcDim n_arrays_dim = fo.addDim("n_arrays", calib.n_arrays);
    netCDF::NcDim n_chunks_dim = fo.addDim("n_chunks", telescope.scan_indices.cols());

    std::vector<netCDF::NcDim> dims = {n_chunks_dim, n_dets_dim};
    std::vector<netCDF::NcDim> grp_dims = {n_chunks_dim, n_arrays_dim};

    // add det stats
    for (const auto &stat: diagnostics.det_stats_header) {
        netCDF::NcVar stat_v = fo.addVar(stat,netCDF::ncDouble, dims);
        stat_v.putVar(diagnostics.stats[stat].data());
        stat_v.putAtt("units",det_stats_header_units[stat]);
    }
    // add group stats
    for (const auto &stat: diagnostics.grp_stats_header) {
        netCDF::NcVar stat_v = fo.addVar(stat,netCDF::ncDouble, grp_dims);
        stat_v.putVar(diagnostics.stats[stat].data());
        stat_v.putAtt("units",grp_stats_header_units[stat]);
    }

    // add apt table
    for (auto const& x: calib.apt) {
        netCDF::NcVar apt_v = fo.addVar("apt_" + x.first,netCDF::ncDouble, n_dets_dim);
        apt_v.putVar(x.second.data());
        apt_v.putAtt("units",calib.apt_header_units[x.first]);
    }

    // add adc
    if (!diagnostics.adc_snap_data.empty()) {
        netCDF::NcDim adc_snap_dim = fo.addDim("adcSnapDim", diagnostics.adc_snap_data[0].cols());
        netCDF::NcDim adc_snap_data_dim = fo.addDim("adcSnapDataDim", diagnostics.adc_snap_data[0].rows());
        std::vector<netCDF::NcDim> dims = {adc_snap_dim, adc_snap_data_dim};
        Eigen::Index i = 0;
        for (auto const& x: diagnostics.adc_snap_data) {
            netCDF::NcVar adc_snap_v = fo.addVar("toltec" + std::to_string(calib.nws(i)) + "_adc_snap_data",netCDF::ncDouble, dims);
            adc_snap_v.putVar(x.data());
            i++;
        }
    }

    // add eigenvalues
    if (!diagnostics.evals.empty() && ptcproc.cleaner.n_calc > 0) {
        const auto first_it = diagnostics.evals.begin();
        if (!first_it->second.empty() && !first_it->second[0].empty()) {
            netCDF::NcDim n_eigs_dim = fo.addDim("n_eigs", ptcproc.cleaner.n_calc);
            netCDF::NcDim n_eig_grp_dim = fo.addDim("n_eig_grp", first_it->second[0].size());

            std::vector<netCDF::NcDim> eval_dims = {n_eig_grp_dim, n_eigs_dim};

            // loop through chunks
            for (const auto &[key, val]: diagnostics.evals) {
                // loop through cleaner grouping
                for (Eigen::Index i=0; i<val.size(); ++i) {

                    netCDF::NcVar eval_v = fo.addVar("evals_" + ptcproc.cleaner.grouping[i] + "_" + std::to_string(i) +
                                                         "_chunk_" + std::to_string(key), netCDF::ncDouble, eval_dims);
                    std::vector<std::size_t> start_eig_index = {0, 0};
                    std::vector<std::size_t> size = {1, TULA_SIZET(ptcproc.cleaner.n_calc)};

                    // loop through eigenvalues in current group
                    for (const auto &evals: val[i]) {
                        Eigen::VectorXd tmp = Eigen::VectorXd::Constant(ptcproc.cleaner.n_calc,
                                                                        std::numeric_limits<double>::quiet_NaN());
                        const Eigen::Index n_copy = std::min<Eigen::Index>(evals.size(), ptcproc.cleaner.n_calc);
                        if (n_copy > 0) {
                            tmp.head(n_copy) = evals.head(n_copy);
                        }
                        eval_v.putVar(start_eig_index, size, tmp.data());
                        start_eig_index[0] += 1;
                    }
                }
            }
        }
        else {
            logger->warn("evals requested but empty; skipping eval/evec output");
        }
    }
    fo.close();
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::run_wiener_filter(map_buffer_t &mb) {
    // pointer to map buffer
    mapmaking::MapBuffer* pmb = &mb;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = nullptr;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = nullptr;
    // directory name
    std::string dir_name;
    // logging label
    const char* map_label = "filtered maps";

    // filtered obs maps
    if constexpr (map_t == mapmaking::FilteredObs) {
        f_io = &filtered_fits_io_vec;
        n_io = &filtered_noise_fits_io_vec;
        dir_name = obsnum_dir_name + "filtered/";
        map_label = "filtered obs maps";
    }

    // filtered coadded maps
    else if constexpr (map_t == mapmaking::FilteredCoadd) {
        f_io = &filtered_coadd_fits_io_vec;
        n_io = &filtered_coadd_noise_fits_io_vec;
        dir_name = coadd_dir_name + "filtered/";
        map_label = "filtered coadded maps";
    }

    logger->info("preparing {} FITS headers ({} files)", map_label, f_io->size());
    for (Eigen::Index i=0; i<f_io->size(); ++i) {
        // get the array for the given map
        // add primary hdu
        add_phdu(f_io, pmb, i);

        // add primary hdu to noise maps
        if (!pmb->noise.empty()) {
            add_phdu(n_io, pmb, i);
        }
    }

    // loop through maps and run wiener filter
    for (Eigen::Index i=0; i<n_maps; ++i) {
        // current array
        auto array = maps_to_arrays(i);
        // get file index
        auto map_index = arrays_to_maps(i);
        logger->info("starting {} map {}/{} (array={})",
                     map_label, i + 1, n_maps, toltec_io.array_name_map[array]);
        // init fwhm in pixels
        wiener_filter.init_fwhm = toltec_io.array_fwhm_arcsec[array]*ASEC_TO_RAD/mb.pixel_size_rad;
        // make wiener filter template
        logger->info("building Wiener template for {} map {}/{} (array={})",
                     map_label, i + 1, n_maps, toltec_io.array_name_map[array]);
        double template_fwhm_rad = 0.0;
        const auto &array_name = toltec_io.array_name_map[array];
        if (wiener_filter.template_type=="gaussian" || wiener_filter.template_type=="airy") {
            auto it = wiener_filter.template_fwhm_rad.find(array_name);
            if (it == wiener_filter.template_fwhm_rad.end()) {
                logger->error("missing Wiener template_fwhm_rad for array {}", array_name);
                std::exit(EXIT_FAILURE);
            }
            template_fwhm_rad = it->second;
        }
        wiener_filter.make_template(mb, calib.apt, template_fwhm_rad, i);
        logger->info("Wiener template ready for {} map {}/{} (array={})",
                     map_label, i + 1, n_maps, toltec_io.array_name_map[array]);
        // run the filter for the current map
        logger->info("running Wiener filter core for {} map {}/{} (array={})",
                     map_label, i + 1, n_maps, toltec_io.array_name_map[array]);
        wiener_filter.filter_maps(mb,i);
        logger->info("map filtering complete for {} map {}/{}", map_label, i + 1, n_maps);

        // filter noise maps
        if (run_noise) {
#if defined(CITLALI_USE_WIENER_FILTER_OMP)
            logger->info("filtering noise for {} map {}/{} (n_noise={})",
                         map_label, i + 1, n_maps, mb.n_noise);
            #pragma omp parallel for schedule(dynamic)
            for (Eigen::Index j=0; j<mb.n_noise; ++j) {
                wiener_filter.filter_noise_threadsafe(mb, i, j);
            }
            logger->info("noise filtering complete for {} map {}/{}",
                         map_label, i + 1, n_maps);
#else
            tula::logging::progressbar pb(
                [&](const auto &msg) { logger->info("{}", msg); }, 100,
                "filtering noise");

            for (Eigen::Index j=0; j<mb.n_noise; ++j) {
                wiener_filter.filter_noise(mb, i, j);
                pb.count(mb.n_noise, mb.n_noise / 100);
            }
            logger->info("noise filtering complete for {} map {}/{}",
                         map_label, i + 1, n_maps);
#endif

            if (wiener_filter.normalize_error) {
                logger->info("renormalizing errors for {} map {}/{}",
                             map_label, i + 1, n_maps);
                bool scaled = false;
                if (!mb.noise.empty() && mb.n_noise > 0) {
                    Eigen::MatrixXd var_map = Eigen::MatrixXd::Zero(mb.n_rows, mb.n_cols);
                    for (Eigen::Index j=0; j<mb.n_noise; ++j) {
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(
                            mb.noise[i].data() + j * mb.n_rows * mb.n_cols,
                            mb.n_rows, mb.n_cols);
                        var_map.array() += noise_matrix.array().square();
                    }
                    var_map.array() /= static_cast<double>(mb.n_noise);

                    double weight_threshold = 0.0;
                    if (mb.cov_cut > 0.0) {
                        weight_threshold = engine_utils::find_weight_threshold(mb.weight[i], mb.cov_cut);
                    }
                    if (!std::isfinite(weight_threshold) || weight_threshold < 0.0) {
                        weight_threshold = 0.0;
                    }

                    Eigen::Index n_valid = 0;
                    for (Eigen::Index r=0; r<mb.n_rows; ++r) {
                        for (Eigen::Index c=0; c<mb.n_cols; ++c) {
                            const double w = mb.weight[i](r,c);
                            const double v = var_map(r,c);
                            if (w > 0.0 && std::isfinite(w) && w >= weight_threshold &&
                                v > 0.0 && std::isfinite(v)) {
                                n_valid++;
                            }
                        }
                    }

                    if (n_valid > 0) {
                        Eigen::VectorXd ratios(n_valid);
                        Eigen::Index idx = 0;
                        for (Eigen::Index r=0; r<mb.n_rows; ++r) {
                            for (Eigen::Index c=0; c<mb.n_cols; ++c) {
                                const double w = mb.weight[i](r,c);
                                const double v = var_map(r,c);
                                if (w > 0.0 && std::isfinite(w) && w >= weight_threshold &&
                                    v > 0.0 && std::isfinite(v)) {
                                    ratios(idx) = w * v;
                                    idx++;
                                }
                            }
                        }

                        double med_ratio = tula::alg::median(ratios);
                        if (std::isfinite(med_ratio) && med_ratio > 0.0) {
                            const double scale = 1.0 / med_ratio;
                            mb.weight[i].noalias() = mb.weight[i] * scale;
                            logger->info("weight renorm (noise-based): median(w*var)={} scale={}",
                                         static_cast<float>(med_ratio), static_cast<float>(scale));
                            scaled = true;
                        }
                    }
                }

                if (!scaled) {
                    // fallback to legacy normalization
                    // get median error from weight maps
                    mb.calc_median_err();
                    // get median map rms from noise maps
                    mb.calc_median_rms();

                    // get rescaled normalization factor
                    auto noise_factor = (1./pow(mb.median_rms.array(),2.))*mb.median_err.array();
                    // re-normalize weight map
                    mb.weight[i].noalias() = mb.weight[i]*noise_factor(i);

                    logger->info("median rms {} ({})", static_cast<float>(mb.median_rms(i)), mb.sig_unit);
                }
            }
        }

        if (write_filtered_maps_partial) {
            // only write if saving all iterations or on last iteration
            // write maps immediately after filtering due to computation time
            logger->info("writing {} map {}/{} to disk", map_label, i + 1, n_maps);
            write_maps(f_io,n_io,pmb,i);

            logger->info("file has been written to:");
            logger->info("{}.fits",f_io->at(map_index).filepath);

            // explicitly destroy the fits file after we're done with it
            bool close_file = true;
            if (rtcproc.run_polarization) {
                if (rtcproc.polarization.stokes_params[maps_to_stokes(i)]!="U") {
                    close_file = false;
                }
            }
            // check if we're moving onto a new file
            if (i<n_maps-1) {
                if (arrays_to_maps(i+1) > arrays_to_maps(i) && close_file) {
                    logger->info("closing FITS handle for {}", f_io->at(map_index).filepath);
                    f_io->at(map_index).pfits->destroy();
                    logger->info("closed FITS handle for {}", f_io->at(map_index).filepath);
                }
            }
        }

        logger->info("completed {} map {}/{}", map_label, i + 1, n_maps);
    }

    if (write_filtered_maps_partial) {
        // clear fits file vectors to ensure its closed.
        logger->info("finalizing {} FITS handles", map_label);
        f_io->clear();
        n_io->clear();
        logger->info("finished finalizing {} FITS handles", map_label);
    }
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::find_sources(map_buffer_t &mb) {
    // clear all source vectors
    mb.n_sources.clear();
    mb.row_source_locs.clear();
    mb.col_source_locs.clear();
    // loop through maps
    for (Eigen::Index i=0; i<n_maps; ++i) {
        // update source vectors
        mb.n_sources.push_back(0);
        mb.row_source_locs.push_back(Eigen::VectorXi::Ones(1));
        mb.col_source_locs.push_back(Eigen::VectorXi::Ones(1));

        // default value of -99 to keep size of vectors same as map vector
        mb.row_source_locs.back()*=-99;
        mb.col_source_locs.back()*=-99;

        // run source finder
        auto sources_found = mb.find_sources(i);

        // number of sources found for current map
        if (sources_found) {
            logger->info("{} source(s) found", mb.n_sources.back());
        }
        else {
            logger->info("no sources found");
        }
    }

    // count up the total number of sources
    Eigen::Index n_sources = 0;
    for (const auto &sources: mb.n_sources) {
        n_sources += sources;
    }

    // matrix to store source parameters
    mb.source_params.setZero(n_sources,map_fitter.n_params);
    mb.source_perror.setZero(n_sources,map_fitter.n_params);

    // keep track of row in total source count
    Eigen::Index k = 0;

    // now loop through and fit the sources
    for (Eigen::Index i=0; i<n_maps; ++i) {
        // skip map if no sources found
        if (mb.n_sources[i] > 0) {
            // current array
            auto array = maps_to_arrays(i);
            // init fwhm in pixels
            auto init_fwhm = toltec_io.array_fwhm_arcsec[array]*ASEC_TO_RAD/mb.pixel_size_rad;

            // placeholder vectors for grppi map
            std::vector<int> source_in_vec, source_out_vec;

            source_in_vec.resize(mb.n_sources[i]);
            std::iota(source_in_vec.begin(), source_in_vec.end(), 0);
            source_out_vec.resize(mb.n_sources[i]);

            // loop through sources and fit them
            grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), source_in_vec, source_out_vec, [&](auto j) {
                // update source rows and cols
                double init_row = mb.row_source_locs[i](j);
                double init_col = mb.col_source_locs[i](j);

                // fit source
                auto [params, perrors, good_fit] =
                    map_fitter.fit_to_gaussian<engine_utils::mapFitter::pointing>(mb.signal[i], mb.weight[i],
                                                                                  init_fwhm, init_row, init_col);
                if (good_fit) {
                    // rescale fit params from pixel to on-sky units
                    params(1) = RAD_TO_ASEC*mb.pixel_size_rad*(params(1) - (mb.n_cols - 1)/2.0);
                    params(2) = RAD_TO_ASEC*mb.pixel_size_rad*(params(2) - (mb.n_rows - 1)/2.0);
                    params(3) = RAD_TO_ASEC*STD_TO_FWHM*mb.pixel_size_rad*(params(3));
                    params(4) = RAD_TO_ASEC*STD_TO_FWHM*mb.pixel_size_rad*(params(4));

                    // rescale fit errors from pixel to on-sky units
                    perrors(1) = RAD_TO_ASEC*mb.pixel_size_rad*(perrors(1));
                    perrors(2) = RAD_TO_ASEC*mb.pixel_size_rad*(perrors(2));
                    perrors(3) = RAD_TO_ASEC*STD_TO_FWHM*mb.pixel_size_rad*(perrors(3));
                    perrors(4) = RAD_TO_ASEC*STD_TO_FWHM*mb.pixel_size_rad*(perrors(4));

                    // if in radec calculate absolute pointing
                    if (telescope.pixel_axes=="radec") {
                        Eigen::VectorXd lat(1), lon(1);
                        lat << params(2)*ASEC_TO_RAD;
                        lon << params(1)*ASEC_TO_RAD;

                        auto [adec, ara] = engine_utils::tangent_to_abs(lat, lon, mb.wcs.crval[0]*DEG_TO_RAD, mb.wcs.crval[1]*DEG_TO_RAD);

                        params(1) = ara(0)*RAD_TO_DEG;
                        params(2) = adec(0)*RAD_TO_DEG;

                        perrors(1) = perrors(1)*ASEC_TO_DEG;
                        perrors(2) = perrors(2)*ASEC_TO_DEG;
                    }

                    // add source params and errors to table
                    mb.source_params.row(k+j) = params;
                    mb.source_perror.row(k+j) = perrors;
                }
                return 0;
            });

            // update row
            k += mb.n_sources[i];
        }
    }
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_sources(map_buffer_t &mb, std::string dir_name) {
    // get filenmame for source table
    std::string source_filename = setup_filenames<map_t,engine_utils::toltecIO::source,
                                                  engine_utils::toltecIO::map>(dir_name);

    // source header information
    std::vector<std::string> source_header = {
        "array",
        "amp",
        "amp_err",
        "x_t",
        "x_t_err",
        "y_t",
        "y_t_err",
        "a_fwhm",
        "a_fwhm_err",
        "b_fwhm",
        "b_fwhm_err",
        "angle",
        "angle_err",
        "sig2noise"
    };

    // units for fitted parameter centroids
    std::string pos_units = (telescope.pixel_axes == "radec") ? "deg" : "arcsec";

    // units for source header
    std::map<std::string,std::string> source_header_units = {
        {"array","N/A"},
        {"amp", mb->sig_unit},
        {"amp_err", mb->sig_unit},
        {"x_t", pos_units},
        {"x_t_err", pos_units},
        {"y_t", pos_units},
        {"y_t_err", pos_units},
        {"a_fwhm", "arcsec"},
        {"a_fwhm_err", "arcsec"},
        {"b_fwhm", "arcsec"},
        {"b_fwhm_err", "arcsec"},
        {"angle", "rad"},
        {"angle_err", "rad"},
        {"sig2noise", "N/A"}
    };

    // meta information for source table
    YAML::Node source_meta;

    // add obsnums
    for (Eigen::Index i=0; i<mb->obsnums.size(); ++i) {
        // add obsnum to meta data
        source_meta["obsnum" + std::to_string(i)] = mb->obsnums[i];
    }

    // add source name
    source_meta["Source"] = telescope.source_name;

    // add date of file creation
    source_meta["creation_date"] = engine_utils::current_date_time();

    // add observation date
    source_meta["date"] = date_obs.back();


    // populate source meta information
    for (const auto &[key,val]: source_header_units) {
        source_meta[key].push_back("units: " + val);
        // description from apt
        auto description = calib.apt_header_description[key];
        source_meta[key].push_back(description);
    }

    // count up the total number of sources
    Eigen::Index n_sources = 0;
    for (const auto &sources: mb->n_sources) {
        n_sources += sources;
    }

    // matrix to hold source information (floats for readability)
    Eigen::MatrixXf source_table(n_sources, 2*map_fitter.n_params + 2);

    // loop through params and add arrays
    Eigen::Index k=0;
    for (Eigen::Index i=0; i<mb->n_sources.size(); ++i) {
        if (mb->n_sources[i]!=0) {
            // calculate map standard deviation
            double map_std_dev = engine_utils::calc_std_dev(mb->signal[i]);

            for (Eigen::Index j=0; j<mb->n_sources[i]; ++j) {
                source_table(k,0) = maps_to_arrays(i);
                // set signal to noise
                source_table(k,2*map_fitter.n_params + 1) = mb->source_params(k,0)/map_std_dev;

                k++;
            }
        }
    }

    // populate source table
    Eigen::Index j = 0;
    for (Eigen::Index i=1; i<2*map_fitter.n_params; i=i+2) {
        source_table.col(i) = mb->source_params.col(j).template cast <float> ();
        source_table.col(i+1) = mb->source_perror.col(j).template cast <float> ();
        j++;
    }

    // write source table
    to_ecsv_from_matrix(source_filename, source_table, source_header, source_meta);
}
