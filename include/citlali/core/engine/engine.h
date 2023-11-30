#pragma once

#include "sys/types.h"
#include "sys/sysinfo.h"

#include <memory>
#include <string>
#include <vector>
#include <omp.h>
#include <Eigen/Core>
#include <fstream>

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

#include <citlali/core/timestream/rtc/polarization_4.h>
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
#include <citlali/core/mapmaking/wiener_filter.h>

#include <citlali/core/engine/io.h>
#include <citlali/core/engine/kidsproc.h>
#include <citlali/core/engine/todproc.h>

struct reduControls {
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
    mapmaking::ObsMapBuffer omb{"omb"}, cmb{"cmb"};
    mapmaking::NaiveMapmaker naive_mm;
    mapmaking::JincMapmaker jinc_mm;
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

    // iteration to write out ptcdata
    int beammap_tod_output_iter = 0;

    // upper and lower limits of psd for sensitivity calc
    Eigen::VectorXd sens_psd_limits_Hz;

    // limits on fwhm, sig2noise, and distance from center for flagging
    std::map<std::string, double> lower_fwhm_arcsec, upper_fwhm_arcsec, lower_sig2noise,
        upper_sig2noise, max_dist_arcsec;

    // limits on sensitivity for flagging
    double lower_sens_factor, upper_sens_factor;
};

class Engine: public reduControls, public reduClasses, public beammapControls {
public:
    // type for missing/invalid keys
    using key_vec_t = std::vector<std::vector<std::string>>;

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

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
    Eigen::Index n_scans_done;

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

    // map grouping and algorithm
    std::string map_grouping, map_method;

    // number of maps
    Eigen::Index n_maps;

    // mapping from index in map vector to array index
    Eigen::VectorXI maps_to_arrays, arrays_to_maps;

    // mapping from index in map vector to array index
    Eigen::VectorXI maps_to_stokes;

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
    void obsnum_setup(int);

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
    void add_tod_header();

    // create tod files (does not populate them)
    template <engine_utils::toltecIO::ProdType prod_t>
    void create_tod_files();

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
    void run_wiener_filter(map_buffer_t &, int);

    // find sources in the maps
    template <mapmaking::MapType map_t, class map_buffer_t>
    void find_sources(map_buffer_t &);

    // write the sources to ecsv table
    template <mapmaking::MapType map_t, class map_buffer_t>
    void write_sources(map_buffer_t &, std::string);
};

void Engine::obsnum_setup(int fruit_iter) {
    if (rtcproc.run_extinction) {
        // get atm model
        rtcproc.calibration.setup(telescope.tau_225_GHz);

        logger->info("using {} model for extinction correction",rtcproc.calibration.extinction_model);

        // check tau (may be unnecessary)
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

    // if filter is requested, make it here
    if (rtcproc.run_tod_filter) {
        rtcproc.filter.make_filter(telescope.fsmp);

        /*
        rtcproc.filter.w0s.clear();
        rtcproc.filter.qs.clear();
        rtcproc.filter.w0s.push_back(24.46);
        rtcproc.filter.qs.push_back(0.05);
        rtcproc.filter.make_notch_filter(telescope.fsmp);
        */
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

    // create output map files
    if (run_mapmaking) {
        if (ptcproc.save_all_iters || fruit_iter == ptcproc.fruit_loops_iters - 1) {
        create_obs_map_files();
        }
    }
    // create timestream files
    if (run_tod_output) {
        if (ptcproc.save_all_iters || fruit_iter == ptcproc.fruit_loops_iters - 1) {
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
    }
    // don't calculate any eigenvalues
    else if (!diagnostics.write_evals) {
        ptcproc.cleaner.n_calc = 0;
    }

    // tod output mode require sequential policy so set explicitly
    if (run_tod_output) {
        logger->warn("tod output mode require sequential policy. "
                     "parallelization will be disabled for some stages.");
        parallel_policy = "seq";
    }

    // output basic info for obs reduction to command line
    cli_summary();

    int n_dets = 0;

    // set number of dets for unpolarized timestreams
    if (!rtcproc.run_polarization) {
        n_dets = calib.apt["array"].size();
    }
    // set number of detectors for polarized timestreams
    else {
        if (!telescope.sim_obs) {
            n_dets = (calib.apt[rtcproc.polarization.grouping].array()!=-1).count();
        }
        else {
            n_dets = calib.n_dets;
        }
    }

    // set up per-det stats file values
    for (const auto &stat: diagnostics.det_stats_header) {
        diagnostics.stats[stat].setZero(n_dets, telescope.scan_indices.cols());
    }
    // set up per-group stats file values
    for (const auto &stat: diagnostics.grp_stats_header) {
        diagnostics.stats[stat].setZero(calib.n_arrays, telescope.scan_indices.cols());
    }
    // clear stored eigenvalues
    diagnostics.evals.clear();
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
}

template<typename CT>
void Engine::get_timestream_config(CT &config) {
    logger->info("getting timestream config options");
    // run tod processing
    get_config_value(config, run_tod, missing_keys, invalid_keys,
                     std::tuple{"timestream","enabled"});
    // tod type (xs, rs, is, qs)
    get_config_value(config, tod_type, missing_keys, invalid_keys,
                     std::tuple{"timestream","type"});

    // run rtc or ptc tod output?
    bool run_tod_output_rtc, run_tod_output_ptc;
    // output rtc
    get_config_value(config, run_tod_output_rtc, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","output","enabled"});
    // output ptc
    get_config_value(config, run_tod_output_ptc, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","output","enabled"});
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
    // get time chunk size
    get_config_value(config, telescope.time_chunk, missing_keys, invalid_keys,
                     std::tuple{"timestream","chunking", "length_sec"});
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

    // map_method
    get_config_value(config, map_method, missing_keys, invalid_keys,
                     std::tuple{"mapmaking","method"},{"naive","jinc"});

    // map reference frame (radec or altaz)
    get_config_value(config, telescope.pixel_axes, missing_keys, invalid_keys,
                     std::tuple{"mapmaking","pixel_axes"},{"radec","altaz"});

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

        if (jinc_mm.mode=="matrix") {
            // allocate jinc matrix
            jinc_mm.allocate_jinc_matrix(omb.pixel_size_rad);
        }
        else if (jinc_mm.mode=="splines") {
            // precompute jinc spline
            jinc_mm.calculate_jinc_splines();
        }
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

    // lower fwhm limit
    auto lower_fwhm_arcsec_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","flagging","lower_fwhm_arcsec"});
    // upper fwhm limit
    auto upper_fwhm_arcsec_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","flagging","upper_fwhm_arcsec"});
    // lower signal-to-noise limit
    auto lower_sig2noise_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","flagging","lower_sig2noise"});
    // upper signal-to-noise limit
    auto upper_sig2noise_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","flagging","upper_sig2noise"});
    // maximum allowed distance limit
    auto max_dist_arcsec_vec = config.template get_typed<std::vector<double>>(std::tuple{"beammap","flagging","max_dist_arcsec"});

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
            beammap_tod_output_iter = beammap_iter_max;
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
    if (!run_noise) {
        logger->error("wiener filter requires noise maps");
        std::exit(EXIT_FAILURE);
    }

    // set parallelization for ffts (maintained with tod output/verbose mode)
    wiener_filter.parallel_policy = parallel_policy;
}

template<typename CT>
void Engine::get_citlali_config(CT &config) {
    //  get interface offsets
    if (config.has(std::tuple{"interface_sync_offset"})) {
        auto interface_node = config.get_node(std::tuple{"interface_sync_offset"});
        // interface key names
        std::vector<std::string> interface_keys = {
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
        // loop through interfaces
        for (Eigen::Index i=0; i<interface_node.size(); ++i) {
            auto offset = config.template get_typed<double>(std::tuple{"interface_sync_offset",i, interface_keys[i]});
            interface_sync_offset[interface_keys[i]] = offset;
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
            map_fitter.flux_limits(i) = config.template get_typed<double>(std::tuple{"post_processing","source_fitting",
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
        std::vector<double> offset;
        // get az offset
        offset = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets",0,"value_arcsec"});
        pointing_offsets_arcsec["az"] = Eigen::Map<Eigen::VectorXd>(offset.data(),offset.size());
        // get alt offset
        offset = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets",1,"value_arcsec"});
        pointing_offsets_arcsec["alt"] = Eigen::Map<Eigen::VectorXd>(offset.data(),offset.size());

        // get julian date of pointing offsets
        try {
            offset = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets",2,"modified_julian_date"});
            pointing_offsets_modified_julian_date = Eigen::Map<Eigen::VectorXd>(offset.data(),offset.size());
        }
        catch (...) {
            pointing_offsets_modified_julian_date.setZero(2);
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

void Engine::add_tod_header() {
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
                add_netcdf_var(fo, "BEAMMAP.REF_DET_INDEX", beammap_reference_det);
                add_netcdf_var(fo, "BEAMMAP.REF_X_T", calib.apt["x_t"](beammap_reference_det));
                add_netcdf_var(fo, "BEAMMAP.REF_Y_T", calib.apt["y_t"](beammap_reference_det));
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

        // add control/runtime parameters
        add_netcdf_var(fo, "CONFIG.VERBOSE", verbose_mode);
        add_netcdf_var(fo, "CONFIG.POLARIZED", rtcproc.run_polarization);
        add_netcdf_var(fo, "CONFIG.DESPIKED", rtcproc.run_despike);
        add_netcdf_var(fo, "CONFIG.TODFILTERED", rtcproc.run_tod_filter);
        add_netcdf_var(fo, "CONFIG.DOWNSAMPLED", rtcproc.run_downsample);
        add_netcdf_var(fo, "CONFIG.CALIBRATED", rtcproc.run_calibrate);
        add_netcdf_var(fo, "CONFIG.EXTINCTION", rtcproc.run_extinction);
        add_netcdf_var<std::string>(fo, "CONFIG.EXTINCTION.EXTMODEL", rtcproc.calibration.extinction_model);
        add_netcdf_var<std::string>(fo, "CONFIG.WEIGHT.TYPE", ptcproc.weighting_type);
        add_netcdf_var(fo, "CONFIG.WEIGHT.RTC.WTLOW", rtcproc.lower_weight_factor);
        add_netcdf_var(fo, "CONFIG.WEIGHT.RTC.WTHIGH", rtcproc.upper_weight_factor);
        add_netcdf_var(fo, "CONFIG.WEIGHT.PTC.WTLOW", ptcproc.lower_weight_factor);
        add_netcdf_var(fo, "CONFIG.WEIGHT.PTC.WTHIGH", ptcproc.upper_weight_factor);
        add_netcdf_var(fo, "CONFIG.WEIGHT.MEDWTFACTOR", ptcproc.med_weight_factor);
        add_netcdf_var(fo, "CONFIG.CLEANED", ptcproc.run_clean);

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

        // fruit loops parameters
        add_netcdf_var(fo, "CONFIG.FRUITLOOPS", ptcproc.run_fruit_loops);
        add_netcdf_var<std::string>(fo, "CONFIG.FRUITLOOPS.PATH", ptcproc.fruit_loops_path);
        add_netcdf_var(fo, "CONFIG.FRUITLOOPS.S2N", ptcproc.fruit_loops_sig2noise);
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

    netCDF::NcDim n_pts_dim = fo.addDim("n_pts");
    netCDF::NcDim n_raw_scan_indices_dim = fo.addDim("n_raw_scan_indices", telescope.scan_indices.rows());
    netCDF::NcDim n_scan_indices_dim = fo.addDim("n_scan_indices", 2);
    netCDF::NcDim n_scans_dim = fo.addDim("n_scans", telescope.scan_indices.cols());

    Eigen::Index n_dets;

    // set number of dets for unpolarized timestreams
    if (!rtcproc.run_polarization) {
        n_dets = calib.apt["array"].size();
    }
    // set number of detectors for polarized timestreams
    else {
        if (!telescope.sim_obs) {
            n_dets = (calib.apt[rtcproc.polarization.grouping].array()!=-1).count();
        }
        else {
            n_dets = calib.n_dets;
        }
    }

    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", n_dets);

    std::vector<netCDF::NcDim> dims = {n_pts_dim, n_dets_dim};
    std::vector<netCDF::NcDim> raw_scans_dims = {n_scans_dim, n_raw_scan_indices_dim};
    std::vector<netCDF::NcDim> scans_dims = {n_scans_dim, n_scan_indices_dim};

    // raw file scan indices
    netCDF::NcVar raw_scan_indices_v = fo.addVar("raw_scan_indices",netCDF::ncInt, raw_scans_dims);
    raw_scan_indices_v.putAtt("units","N/A");
    raw_scan_indices_v.putVar(telescope.scan_indices.data());

    // scan indices for data
    netCDF::NcVar scan_indices_v = fo.addVar("scan_indices",netCDF::ncInt, scans_dims);
    scan_indices_v.putAtt("units","N/A");

    // signal
    netCDF::NcVar signal_v = fo.addVar("signal",netCDF::ncDouble, dims);
    signal_v.putAtt("units",omb.sig_unit);

    // chunk sizes
    std::vector<std::size_t> chunkSizes;
    // set chunk mode
    netCDF::NcVar::ChunkMode chunkMode = netCDF::NcVar::nc_CHUNKED;

    // set chunking to mean scan size and n_dets
    chunkSizes.push_back(((telescope.scan_indices.row(3) - telescope.scan_indices.row(2)).array() + 1).mean());
    chunkSizes.push_back(n_dets);

    // set signal chunking
    signal_v.setChunking(chunkMode, chunkSizes);

    // flags
    netCDF::NcVar flags_v = fo.addVar("flags",netCDF::ncDouble, dims);
    flags_v.putAtt("units","N/A");
    flags_v.setChunking(chunkMode, chunkSizes);

    // kernel
    if (rtcproc.run_kernel) {
        netCDF::NcVar kernel_v = fo.addVar("kernel",netCDF::ncDouble, dims);
        kernel_v.putAtt("units","N/A");
        kernel_v.setChunking(chunkMode, chunkSizes);
    }

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
        netCDF::NcVar offsets_v = fo.addVar("pointing_offset_"+x.first,netCDF::ncDouble, n_pts_dim);
        offsets_v.putAtt("units","arcsec");
        offsets_v.setChunking(chunkMode, chunkSizes);
    }

    // add weights
    if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
        std::vector<netCDF::NcDim> weight_dims = {n_scans_dim, n_dets_dim};
        netCDF::NcVar weights_v = fo.addVar("weights",netCDF::ncDouble, weight_dims);
        weights_v.putAtt("units","("+omb.sig_unit+")^-2");
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
                                               omb.kernel.size() + omb.coverage.size())/1e9;

    logger->info("estimated size of map buffer {} GB", omb_size);

    mb_size_total = mb_size_total + omb_size;

    // print info if coadd is requested
    if (run_coadd) {
        logger->info("coadd map buffer rows: {}", cmb.n_rows);
        logger->info("coadd map buffer cols: {}", cmb.n_cols);

        // make a rough estimate of memory usage for coadd map buffer
        double cmb_size = 8*cmb.n_rows*cmb.n_cols*(cmb.signal.size() + cmb.weight.size() +
                                                   cmb.kernel.size() + cmb.coverage.size())/1e9;

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
    n_nans["kernel"] = 0;
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
            for (Eigen::Index j=0; j<mb.noise.size(); ++j) {
                Eigen::Tensor<double,2> out = mb.noise[i].chip(j,2);
                auto out_matrix = Eigen::Map<Eigen::MatrixXd>(out.data(), out.dimension(0), out.dimension(1));
                n_nans["noise"] = n_nans["noise"] + out_matrix.array().isNaN().count();
                n_infs["noise"] = n_infs["noise"] + out_matrix.array().isInf().count();
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
    // array name
    std::string name = toltec_io.array_name_map[calib.arrays(i)];

    // conversion to uK
    auto fwhm = (std::get<0>(calib.array_fwhms[calib.arrays(i)]) + std::get<1>(calib.array_fwhms[calib.arrays(i)]))/2;
    auto mJy_beam_to_uK = engine_utils::mJy_beam_to_uK(1, toltec_io.array_freq_map[calib.arrays(i)], fwhm*ASEC_TO_RAD);

    // beam area in steradians
    auto beam_area_rad = 2.*pi*pow(fwhm*FWHM_TO_STD*ASEC_TO_RAD,2);
    // get Jy/pixel
    auto mJy_beam_to_Jy_px = 1e-3/beam_area_rad*pow(mb->pixel_size_rad,2);

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
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_DET_INDEX", beammap_reference_det, "Beammap Reference det (rotation center)");
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_X_T", calib.apt["x_t"](beammap_reference_det), "Az rotation center (arcsec)");
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_Y_T", calib.apt["y_t"](beammap_reference_det), "Alt rotation center (arcsec)");
        }
        else {
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_DET_INDEX", -99, "Beammap Reference det (rotation center)");
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_X_T", "N/A", "Az rotation center (arcsec)");
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_Y_T", "N/A", "Alt rotation center (arcsec)");
        }
    }

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
    // add source ra
    fits_io->at(i).pfits->pHDU().addKey("SRC_RA", telescope.tel_header["Header.Source.Ra"][0], "Source RA (radians)");
    // add source dec
    fits_io->at(i).pfits->pHDU().addKey("SRC_DEC", telescope.tel_header["Header.Source.Dec"][0], "Source Dec (radians)");
    // add map tangent point ra
    fits_io->at(i).pfits->pHDU().addKey("TAN_RA", telescope.tel_header["Header.Source.Ra"][0], "Map Tangent Point RA (radians)");
    //add map tangent point dec
    fits_io->at(i).pfits->pHDU().addKey("TAN_DEC", telescope.tel_header["Header.Source.Dec"][0], "Map Tangent Point Dec (radians)");
    // add mean alt
    fits_io->at(i).pfits->pHDU().addKey("MEAN_EL", RAD_TO_DEG*telescope.tel_data["TelElAct"].mean(), "Mean Elevation (deg)");
    // add mean az
    fits_io->at(i).pfits->pHDU().addKey("MEAN_AZ", RAD_TO_DEG*telescope.tel_data["TelAzAct"].mean(), "Mean Azimuth (deg)");
    // add mean parallactic angle
    fits_io->at(i).pfits->pHDU().addKey("MEAN_PA", RAD_TO_DEG*telescope.tel_data["ActParAng"].mean(), "Mean Parallactic angle (deg)");

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
        fits_io->at(i).pfits->pHDU().addKey("JINC_R", jinc_mm.r_max, "Jinc filter R_max");
        fits_io->at(i).pfits->pHDU().addKey("JINC_A", jinc_mm.shape_params[calib.arrays(i)][0], "Jinc filter param a");
        fits_io->at(i).pfits->pHDU().addKey("JINC_B", jinc_mm.shape_params[calib.arrays(i)][1], "Jinc filter param b");
        fits_io->at(i).pfits->pHDU().addKey("JINC_C", jinc_mm.shape_params[calib.arrays(i)][2], "Jinc filter param c");
    }

    // add mean tau
    if (rtcproc.run_extinction) {
        Eigen::VectorXd tau_el(1);
        tau_el << telescope.tel_data["TelElAct"].mean();
        auto tau_freq = rtcproc.calibration.calc_tau(tau_el, telescope.tau_225_GHz);

        fits_io->at(i).pfits->pHDU().addKey("MEAN_TAU", tau_freq[i](0), "mean tau (" + name + ")");
    }
    else {
        fits_io->at(i).pfits->pHDU().addKey("MEAN_TAU", 0., "mean tau (" + name + ")");
    }

    // add sample rate
    fits_io->at(i).pfits->pHDU().addKey("SAMPRATE", telescope.fsmp, "sample rate (Hz)");

    // add apt table to header
    if (mb->obsnums.size()==1) {
        std::vector<string> apt_filename;
        std::stringstream ss(calib.apt_filepath);
        std::string item;
        char delim = '/';

        while (getline (ss, item, delim)) {
            apt_filename.push_back(item);
        }
        fits_io->at(i).pfits->pHDU().addKey("APT", apt_filename.back(), "APT table used");
    }

    // estimate rms from weight maps
    mb->calc_mean_err();
    auto rms = 1./pow(mb->mean_err(i),-0.5);

    // out-of-focus holography parameters
    fits_io->at(i).pfits->pHDU().addKey("OOF_RMS", rms, "rms of map background (" + mb->sig_unit +")");
    fits_io->at(i).pfits->pHDU().addKey("OOF_W", toltec_io.array_wavelength_map[calib.arrays(i)]/1000., "wavelength (m)");
    fits_io->at(i).pfits->pHDU().addKey("OOF_ID", static_cast<int>(toltec_io.array_wavelength_map[calib.arrays(i)]*1000), "instrument id");
    fits_io->at(i).pfits->pHDU().addKey("OOF_T", 3.0, "taper (dB)");
    fits_io->at(i).pfits->pHDU().addKey("OOF_M2X", telescope.tel_header["Header.M2.XReq"](0)/1000.*1e6, "oof m2x (microns)");
    fits_io->at(i).pfits->pHDU().addKey("OOF_M2Y", telescope.tel_header["Header.M2.YReq"](0)/1000.*1e6, "oof m2y (microns)");
    fits_io->at(i).pfits->pHDU().addKey("OOF_M2Z", telescope.tel_header["Header.M2.ZReq"](0)/1000.*1e6, "oof m2x (microns)");

    fits_io->at(i).pfits->pHDU().addKey("OOF_RO", 25., "outer diameter of the antenna (m)");
    fits_io->at(i).pfits->pHDU().addKey("OOF_RI", 1.65, "inner diameter of the antenna (m)");

    // add control/runtime parameters
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.VERBOSE", verbose_mode, "Reduced in verbose mode");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.POLARIZED", rtcproc.run_polarization, "Polarized Obs");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKED", rtcproc.run_despike, "Despiked");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODFILTERED", rtcproc.run_tod_filter, "TOD Filtered");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DOWNSAMPLED", rtcproc.run_downsample, "Downsampled");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CALIBRATED", rtcproc.run_calibrate, "Calibrated");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.EXTINCTION", rtcproc.run_extinction, "Extinction corrected");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.EXTINCTION.EXTMODEL", rtcproc.calibration.extinction_model, "Extinction model");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.TYPE", ptcproc.weighting_type, "Weighting scheme");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.RTC.WTLOW", rtcproc.lower_weight_factor, "RTC lower weight cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.RTC.WTHIGH", rtcproc.upper_weight_factor, "RTC upper weight cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.PTC.WTLOW", ptcproc.lower_weight_factor, "PTC lower weight cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.PTC.WTHIGH", ptcproc.upper_weight_factor, "PTC upper weight cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.MEDWTFACTOR", ptcproc.med_weight_factor, "Median weight factor");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED", ptcproc.run_clean, "Cleaned");
    if (ptcproc.run_clean) {
        fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.NEIG", ptcproc.cleaner.n_eig_to_cut[calib.arrays(i)].sum(),
                                            "Number of eigenvalues removed");
    }
    else {
        fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.NEIG", 0, "Number of eigenvalues removed");
    }

    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS", ptcproc.run_fruit_loops, "Fruit loops");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS.PATH", ptcproc.fruit_loops_path, "Fruit loops path");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS.S2N", ptcproc.fruit_loops_sig2noise, "Fruit loops S/N");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS.MAXITER", ptcproc.fruit_loops_iters, "Fruit loops iterations");


    // add telescope file header information
    if (mb->obsnums.size()==1) {
        for (auto const& [key, val] : telescope.tel_header) {
            fits_io->at(i).pfits->pHDU().addKey(key, val(0), key);
        }
    }
}

template <typename fits_io_type, class map_buffer_t>
void Engine::write_maps(fits_io_type &fits_io, fits_io_type &noise_fits_io, map_buffer_t &mb, Eigen::Index i) {
    // get name for extension layer
    std::string map_name = get_map_name(i);

    // get the array for the given map
    Eigen::Index map_index = arrays_to_maps(i);
    // get the stokes parameter for the given map
    Eigen::Index stokes_index = maps_to_stokes(i);

    // update wcs ctypes for frequency and stokes params
    mb->wcs.crval[2] = toltec_io.array_freq_map[calib.arrays[maps_to_arrays(i)]];
    mb->wcs.crval[3] = stokes_index;

    // signal map
    fits_io->at(map_index).add_hdu("signal_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->signal[i]);
    fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(),mb->wcs, telescope.tel_header["Header.Source.Epoch"](0));
    fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");

    // weight map
    fits_io->at(map_index).add_hdu("weight_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->weight[i]);
    fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(),mb->wcs, telescope.tel_header["Header.Source.Epoch"](0));
    fits_io->at(map_index).hdus.back()->addKey("UNIT", "1/("+mb->sig_unit+")^2", "Unit of map");
    fits_io->at(map_index).hdus.back()->addKey("MEANERR", mb->mean_err(i), "Mean Square Error");

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
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(),mb->wcs, telescope.tel_header["Header.Source.Epoch"](0));
        fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");
    }

    // coverage map
    if (!mb->coverage.empty()) {
        fits_io->at(map_index).add_hdu("coverage_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->coverage[i]);
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(),mb->wcs, telescope.tel_header["Header.Source.Epoch"](0));
        fits_io->at(map_index).hdus.back()->addKey("UNIT", "sec", "Unit of map");
    }

    /* coverage bool and signal-to-noise maps */
    if (!mb->coverage.empty()) {
        Eigen::MatrixXd ones, zeros;
        ones.setOnes(mb->weight[i].rows(), mb->weight[i].cols());
        zeros.setZero(mb->weight[i].rows(), mb->weight[i].cols());

        // get weight threshold for current map
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = mb->calc_cov_region(i);
        // if weight is less than threshold, set to zero, otherwise set to one
        Eigen::MatrixXd coverage_bool = (mb->weight[i].array() < weight_threshold).select(zeros,ones);

        // coverage bool map
        fits_io->at(map_index).add_hdu("coverage_bool_" + map_name + rtcproc.polarization.stokes_params[stokes_index], coverage_bool);
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(),mb->wcs, telescope.tel_header["Header.Source.Epoch"](0));
        fits_io->at(map_index).hdus.back()->addKey("UNIT", "N/A", "Unit of map");

        // signal-to-noise map
        Eigen::MatrixXd sig2noise = mb->signal[i].array()*sqrt(mb->weight[i].array());
        fits_io->at(map_index).add_hdu("sig2noise_" + map_name + rtcproc.polarization.stokes_params[stokes_index], sig2noise);
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(),mb->wcs, telescope.tel_header["Header.Source.Epoch"](0));
        fits_io->at(map_index).hdus.back()->addKey("UNIT", "N/A", "Unit of map");
    }

    // write noise maps
    if (!mb->noise.empty()) {
        for (Eigen::Index n=0; n<mb->n_noise; ++n) {
            Eigen::Tensor<double,2> out = mb->noise[i].chip(n,2);
            auto out_matrix = Eigen::Map<Eigen::MatrixXd>(out.data(), out.dimension(0), out.dimension(1));

            noise_fits_io->at(map_index).add_hdu("signal_" + map_name + std::to_string(n) + "_" +
                                                 rtcproc.polarization.stokes_params[stokes_index],
                                                 out_matrix);
            noise_fits_io->at(map_index).add_wcs(noise_fits_io->at(map_index).hdus.back(),mb->wcs, telescope.tel_header["Header.Source.Epoch"](0));
            noise_fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");
            noise_fits_io->at(map_index).hdus.back()->addKey("MEANRMS", mb->mean_rms[i], "Mean RMS of noise maps");
        }
    }
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_psd(map_buffer_t &mb, std::string dir_name) {

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
    if (!diagnostics.evals.empty()) {
        netCDF::NcDim n_eigs_dim = fo.addDim("n_eigs",ptcproc.cleaner.n_calc);
        netCDF::NcDim n_eig_grp_dim = fo.addDim("n_eig_grp",diagnostics.evals[0][0].size());

        std::vector<netCDF::NcDim> eval_dims = {n_eig_grp_dim, n_eigs_dim};

        // loop through chunks
        for (const auto &[key, val]: diagnostics.evals) {
            // loop through cleaner gropuing
            for (Eigen::Index i=0; i<val.size(); ++i) {

                netCDF::NcVar eval_v = fo.addVar("evals_" + ptcproc.cleaner.grouping[i] + "_" + std::to_string(i) +
                                                     "_chunk_" + std::to_string(key), netCDF::ncDouble,eval_dims);
                std::vector<std::size_t> start_eig_index = {0, 0};
                std::vector<std::size_t> size = {1, TULA_SIZET(ptcproc.cleaner.n_calc)};

                // loop through eigenvalues in current group
                for (const auto &evals: val[i]) {
                    eval_v.putVar(start_eig_index,size,evals.data());
                    start_eig_index[0] += 1;
                }
            }
        }
    }
    fo.close();
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::run_wiener_filter(map_buffer_t &mb, int fruit_iter) {
    // pointer to map buffer
    mapmaking::ObsMapBuffer* pmb = &mb;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = NULL;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = NULL;

    // directory name
    std::string dir_name;

    // filtered obs maps
    if constexpr (map_t == mapmaking::FilteredObs) {
        f_io = &filtered_fits_io_vec;
        n_io = &filtered_noise_fits_io_vec;
        dir_name = obsnum_dir_name + "filtered/";
    }

    // filtered coadded maps
    else if constexpr (map_t == mapmaking::FilteredCoadd) {
        f_io = &filtered_coadd_fits_io_vec;
        n_io = &filtered_coadd_noise_fits_io_vec;
        dir_name = coadd_dir_name + "filtered/";
    }

    for (Eigen::Index i=0; i<f_io->size(); ++i) {
        // get the array for the given map
        // add primary hdu
        add_phdu(f_io, pmb, i);

        // add primary hdu to noise maps
        if (!pmb->noise.empty()) {
            add_phdu(n_io, pmb, i);
        }
    }

    Eigen::Index j = 0;
    // loop through maps and run wiener filter
    for (Eigen::Index i=0; i<n_maps; ++i) {
        // current array
        auto array = maps_to_arrays(i);
        // get file index
        auto map_index = arrays_to_maps(i);
        // init fwhm in pixels
        wiener_filter.init_fwhm = toltec_io.array_fwhm_arcsec[array]*ASEC_TO_RAD/omb.pixel_size_rad;
        // make wiener filter template
        wiener_filter.make_template(mb, calib.apt, wiener_filter.template_fwhm_rad[toltec_io.array_name_map[array]],i);
        // run the filter for the current map
        wiener_filter.filter_maps(mb,i);

        // filter noise maps
        if (run_noise) {
            tula::logging::progressbar pb(
                [&](const auto &msg) { logger->info("{}", msg); }, 100,
                "filtering noise");

            for (Eigen::Index j=0; j<mb.n_noise; ++j) {
                wiener_filter.filter_noise(mb, i, j);
                pb.count(mb.n_noise, mb.n_noise / 100);
            }
        }

        if (wiener_filter.normalize_error) {
            logger->info("renormalizing errors");
            // get mean error from weight maps
            mb.calc_mean_err();

            // get mean map rms from noise maps
            mb.calc_mean_rms();

            // get rescaled normalization factor
            auto noise_factor = (1./pow(mb.mean_rms.array(),2.))*mb.mean_err.array();

            // re-normalize weight map
            mb.weight[i].noalias() = mb.weight[i]*noise_factor(i);

            logger->info("mean rms {} ({})", static_cast<float>(mb.mean_rms(i)), mb.sig_unit);
        }

        if (write_filtered_maps_partial) {
            // only write if saving all iterations or on last iteration
            if (ptcproc.save_all_iters || fruit_iter == ptcproc.fruit_loops_iters - 1) {
            // write maps immediately after filtering due to computation time
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
                    f_io->at(map_index).pfits->destroy();
                }
            }
        }
        }
    }

    if (write_filtered_maps_partial) {
        // clear fits file vectors to ensure its closed.
        f_io->clear();
        n_io->clear();
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
                    params(1) = RAD_TO_ASEC*mb.pixel_size_rad*(params(1) - (mb.n_cols)/2);
                    params(2) = RAD_TO_ASEC*mb.pixel_size_rad*(params(2) - (mb.n_rows)/2);
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
            k = k + mb.n_sources[i];
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

    // add date
    source_meta["Date"] = engine_utils::current_date_time();

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
