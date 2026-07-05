#pragma once

#include "sys/types.h"
#if defined(__linux__)
#include "sys/sysinfo.h"
#endif

#include <memory>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <tuple>

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
#include <citlali/core/utils/pointing.h>

#include <citlali/core/config/beammap_config.h>
#include <citlali/core/config/calibration_config.h>
#include <citlali/core/config/coadd_config.h>
#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/config/pointing_config.h>
#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/engine/config.h>
#include <citlali/core/engine/learning.h>
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

#include <citlali/core/mapmaking/edge_guard_state.h>
#include <citlali/core/mapmaking/map.h>
#include <citlali/core/mapmaking/naive_mm.h>
#include <citlali/core/mapmaking/jinc_mm.h>
#include <citlali/core/mapmaking/ml_mm.h>
#if defined(CITLALI_USE_WIENER_FILTER_OMP)
#include <citlali/core/mapmaking/wiener_filter_omp.h>
#else
#include <citlali/core/mapmaking/wiener_filter.h>
#endif
#include <citlali/core/pipeline/fits_image_metadata.h>
#include <citlali/core/pipeline/cli_summary.h>
#include <citlali/core/pipeline/map_filename.h>
#include <citlali/core/pipeline/mapdiag_edge_guard.h>
#include <citlali/core/pipeline/map_layer_name.h>
#include <citlali/core/pipeline/map_summary_stats.h>
#include <citlali/core/pipeline/mapdiag_labels.h>
#include <citlali/core/pipeline/mapdiag_netcdf.h>
#include <citlali/core/pipeline/mapdiag_observation_weight.h>
#include <citlali/core/pipeline/mapdiag_stage.h>
#include <citlali/core/pipeline/mapdiag_stats.h>
#include <citlali/core/pipeline/output_netcdf_metadata.h>
#include <citlali/core/pipeline/phdu_beammap.h>
#include <citlali/core/pipeline/phdu_extinction.h>
#include <citlali/core/pipeline/phdu_observation_metadata.h>
#include <citlali/core/pipeline/phdu_oof.h>
#include <citlali/core/pipeline/phdu_reduction_config.h>
#include <citlali/core/pipeline/phdu_rtc_config.h>
#include <citlali/core/pipeline/phdu_telescope_values.h>
#include <citlali/core/pipeline/ptcdiag_netcdf.h>
#include <citlali/core/pipeline/reduction_config_netcdf.h>
#include <citlali/core/pipeline/rtcdiag_netcdf.h>
#include <citlali/core/pipeline/spectral_diagnostics_netcdf.h>
#include <citlali/core/pipeline/stats_netcdf.h>
#include <citlali/core/pipeline/summary_log.h>

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
    bool write_noise_realizations;
    bool run_noise_products;
    bool apply_empirical_noise_weights;
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

    // beammap convergence aperture radius
    double beammap_convergence_radius_arcsec = 10.0;

    // detector-beammap iteration phase controls
    bool beammap_phase_split_enabled = true;
    int beammap_locator_iter = 0;
    int beammap_measurement_start_iter = 1;

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

    // detector-map sample weighting policy
    std::string beammap_detector_weighting_mode = "const";

    // optional circular residual support for beammap Gaussian fits, in nominal FWHM units
    double beammap_fit_radius_fwhm = 0.0;

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
    double beammap_priors_max_d2_iter0 = 25.0;
    double beammap_priors_max_d2_after_iter0 = 25.0;
    double beammap_priors_score_lambda = 2.0;
    double beammap_priors_score_lambda_iter0 = 2.0;
    double beammap_priors_score_lambda_after_iter0 = 2.0;
    bool beammap_priors_fallback_blind = true;
    bool beammap_priors_align_after_iter0 = true;
    std::string beammap_priors_alignment_scope = "array";
    std::string beammap_priors_alignment_common_support = "all";
    double beammap_priors_alignment_common_support_quantile = 0.02;
    int beammap_priors_alignment_min_matches = 30;
    double beammap_priors_alignment_max_d2 = 25.0;
    bool beammap_priors_alignment_fit_rotation = true;
    double beammap_priors_alignment_max_rotation_deg = 8.0;

    // iteration to write out beammap PTC data; -1 means final attempted iteration
    int beammap_tod_output_iter = -1;

    // optional detector-specific PTC TOD diagnostic sidecar for beammaps
    bool beammap_detector_tod_output_enabled = false;
    std::string beammap_detector_tod_output_subdir_name = "source_crossing_tod";
    int beammap_detector_tod_output_n_uniform = 10;
    int beammap_detector_tod_output_n_source_dense = 10;

    // upper and lower limits of psd for sensitivity calc
    Eigen::VectorXd sens_psd_limits_Hz;

    // limits on fwhm, sig2noise, and distance from center for flagging
    std::map<std::string, double> lower_fwhm_arcsec, upper_fwhm_arcsec, lower_sig2noise,
        upper_sig2noise, max_dist_arcsec, network_robust_z;
    double beammap_flag_max_prior_d2 = 0.0;

    // limits on sensitivity for flagging
    double lower_sens_factor, upper_sens_factor;
};

struct pointingControls {
    // source-aware pointing strategy.  Gaussian fits are optional diagnostics;
    // fruit loops remains empirical and uses previous maps.
    std::string pointing_source_strategy = "standard";
    bool pointing_fit_gaussian_enabled = true;
    std::string pointing_fruitloops_center_mode = "auto";
    double pointing_header_center_max_radius_arcsec = 0.0;
    bool pointing_header_center_require_coverage = true;
};

class Engine: public reduControls, public reduClasses, public beammapControls, public pointingControls {
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

    // expected sky regime for map interpretation
    std::string map_regime = "unknown";

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

    // typed runtime config mirror for staged config migration
    citlali::config::RuntimeConfig typed_runtime_config;
    citlali::config::TimestreamConfig typed_timestream_config;
    citlali::config::MapmakingConfig typed_mapmaking_config;
    citlali::config::CoaddConfig typed_coadd_config;
    citlali::config::NoiseConfig typed_noise_config;
    citlali::config::PostProcessingConfig typed_post_processing_config;
    citlali::config::PointingConfig typed_pointing_config;
    citlali::config::BeammapConfig typed_beammap_config;
    citlali::config::AstrometryConfig typed_astrometry_config;

    // obsnum
    std::string obsnum;

    // write filtered maps as they complete
    bool write_filtered_maps_partial;

    // rtc or ptc types
    std::string tod_output_type, tod_output_subdir_name;
    bool run_tod_output_rtc = false;
    bool run_tod_output_ptc = false;
    std::string rtcdiag_filename;
    std::string ptcdiag_filename;

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
    std::string tod_output_selection_mode_rtc = "indices";
    std::string tod_output_selection_mode_ptc = "indices";
    int tod_output_uniform_count_rtc = 10;
    int tod_output_uniform_count_ptc = 10;
    int tod_output_source_dense_count_rtc = 10;
    int tod_output_source_dense_count_ptc = 10;
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

    // shared state learned across RTC, PTC, and mapmaking phases
    ReductionLearningState reduction_learning;

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

    // get shared reduction-learning config options
    template<typename CT>
    void get_learning_config(CT &);

    // apply masks learned in earlier iterations; behavior is gated by reduction_learning phase
    template <class rtc_t, class calib_t>
    void apply_learned_rtc_sample_masks(rtc_t &, calib_t &);
    template <class ptc_t, class calib_t>
    void apply_learned_ptc_sample_masks(ptc_t &, calib_t &);
    template <class ptc_t, class calib_t>
    void apply_learned_ptc_detector_exclusions(ptc_t &, calib_t &);
    template <class tc_t, class calib_t>
    void apply_learned_mapmaking_detector_exclusions(tc_t &, calib_t &);
    template <class tc_t, class calib_t>
    void apply_learned_detector_exclusions(tc_t &, calib_t &, const std::string &,
                                           bool, bool, bool, bool);
    template <class tc_t, class calib_t>
    void apply_learned_sample_masks(tc_t &, calib_t &, bool, const std::string &,
                                    bool, double);

    // collect passive RTC/PTC diagnostics into the shared reduction-learning state
    template <class rtc_t, class ptc_t, class calib_t>
    void collect_rtc_learning_diagnostics(rtc_t &, ptc_t &, calib_t &,
                                          const std::vector<timestream::RTCProc::RTCDetectorDiagSummary> &);
    template <class ptc_t, class calib_t>
    void collect_ptc_learning_diagnostics(ptc_t &, calib_t &,
                                          const std::vector<timestream::PTCProc::SecondPassDiagSummary> &,
                                          const std::vector<timestream::PTCProc::HighWeightDiagSummary> &);
    void write_learning_summary();

    // get beammap config options
    template<typename CT>
    void get_beammap_config(CT &);

    // get pointing config options
    template<typename CT>
    void get_pointing_config(CT &);

    // get mapmaking config options
    template<typename CT>
    void get_mapmaking_config(CT &);

    // get map filtering config options
    template<typename CT>
    void get_map_filter_config(CT &);

    // get runtime config options
    template<typename CT>
    citlali::config::RuntimeConfig get_runtime_config(CT &);

    // get all non-input config options and call other config functions
    template<typename CT>
    void get_citlali_config(CT &);

    // get source fluxes (beammap only)
    template<typename CT>
    void get_photometry_config(CT &);

    // get pointing offsets
    template<typename CT>
    void get_astrometry_config(CT &);

    // effective sample frequency after RTC downsampling
    double processed_time_chunk_fs_hz() const;

    // optional model-protected line-audit notch pass on source-subtracted PTC residuals
    template <class calib_t>
    Eigen::Index apply_model_protected_ptc_line_audit(
        TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, bool);

    // create fits files (does not populate them)
    void create_obs_map_files();

    // add FITS header values to tod files
    template <class map_buffer_t>
    void add_tod_header(map_buffer_t &);

    // create tod files (does not populate them)
    template <engine_utils::toltecIO::ProdType prod_t>
    void create_tod_files();
    void create_rtcdiag_file();
    void create_ptcdiag_file();

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

    // write compact map diagnostics sidecar
    template <mapmaking::MapType map_t, class map_buffer_t>
    void write_mapdiag(map_buffer_t &, std::string);
    void configure_map_pixel_contribution_targets(mapmaking::MapBuffer &,
                                                  const std::string &);

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

    setup_tod_output_chunk_selection();
    // create output subdirectory if requested
    if (tod_output_subdir_name != "null") {
        fs::create_directories(obsnum_dir_name + "raw/" + tod_output_subdir_name);
    }
    // create timestream files
    if (run_tod_output) {
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
    create_rtcdiag_file();
    create_ptcdiag_file();

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
    auto vector_to_string = [](const std::vector<Eigen::Index> &values) {
        std::ostringstream os;
        os << "[";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i != 0) {
                os << ", ";
            }
            os << values[i];
        }
        os << "]";
        return os.str();
    };

    auto build_uniform_plus_source_crossing_chunks =
        [&](const std::string &stream_name, int n_uniform, int n_source_dense) {
            std::set<Eigen::Index> selected_0based;
            std::vector<Eigen::Index> selected_1based;
            if (n_scans <= 0) {
                return selected_1based;
            }

            n_uniform = std::max(0, n_uniform);
            n_source_dense = std::max(0, n_source_dense);

            if (n_uniform == 1) {
                selected_0based.insert((n_scans - 1) / 2);
            }
            else if (n_uniform > 1) {
                for (int i = 0; i < n_uniform; ++i) {
                    const double frac = static_cast<double>(i) /
                                        static_cast<double>(n_uniform - 1);
                    Eigen::Index scan_index =
                        static_cast<Eigen::Index>(std::lround(frac * (n_scans - 1)));
                    scan_index = std::clamp<Eigen::Index>(scan_index, 0, n_scans - 1);
                    selected_0based.insert(scan_index);
                }
            }

            Eigen::Index source_scan = (n_scans - 1) / 2;
            double best_scan_d2 = std::numeric_limits<double>::infinity();
            try {
                auto tel_data_copy = telescope.tel_data;
                std::map<std::string, Eigen::VectorXd> pointing_offsets;
                Eigen::Index n_tel = 0;
                if (!tel_data_copy.empty()) {
                    n_tel = tel_data_copy.begin()->second.size();
                }
                auto make_offset = [&](const std::string &axis) -> Eigen::VectorXd {
                    auto it = pointing_offsets_arcsec.find(axis);
                    if (it != pointing_offsets_arcsec.end() && it->second.size() == n_tel) {
                        return it->second;
                    }
                    return Eigen::VectorXd::Zero(n_tel);
                };
                pointing_offsets["az"] = make_offset("az");
                pointing_offsets["alt"] = make_offset("alt");

                auto [lat, lon] = engine_utils::calc_det_pointing(
                    tel_data_copy, 0.0, 0.0, telescope.pixel_axes, pointing_offsets,
                    map_grouping, true);

                for (Eigen::Index scan_index = 0; scan_index < n_scans; ++scan_index) {
                    const Eigen::Index start =
                        std::max<Eigen::Index>(0, telescope.scan_indices(0, scan_index));
                    const Eigen::Index end =
                        std::min<Eigen::Index>(lat.size() - 1, telescope.scan_indices(1, scan_index));
                    if (end < start || lon.size() <= end) {
                        continue;
                    }
                    double scan_best_d2 = std::numeric_limits<double>::infinity();
                    for (Eigen::Index sample = start; sample <= end; ++sample) {
                        const double y = lat(sample);
                        const double x = lon(sample);
                        if (!std::isfinite(x) || !std::isfinite(y)) {
                            continue;
                        }
                        const double d2 = x * x + y * y;
                        if (d2 < scan_best_d2) {
                            scan_best_d2 = d2;
                        }
                    }
                    if (scan_best_d2 < best_scan_d2) {
                        best_scan_d2 = scan_best_d2;
                        source_scan = scan_index;
                    }
                }
            }
            catch (const std::exception &e) {
                logger->warn(
                    "{} TOD uniform_plus_source_crossing selection could not calculate source-crossing scan ({}); using scan {}",
                    stream_name, e.what(), source_scan + 1);
            }

            if (n_source_dense > 0) {
                Eigen::Index first_dense =
                    source_scan - static_cast<Eigen::Index>((n_source_dense - 1) / 2);
                first_dense = std::clamp<Eigen::Index>(
                    first_dense, 0, std::max<Eigen::Index>(0, n_scans - n_source_dense));
                const Eigen::Index last_dense =
                    std::min<Eigen::Index>(n_scans - 1,
                                           first_dense + static_cast<Eigen::Index>(n_source_dense) - 1);
                for (Eigen::Index scan_index = first_dense; scan_index <= last_dense; ++scan_index) {
                    selected_0based.insert(scan_index);
                }
            }

            selected_1based.reserve(selected_0based.size());
            for (const auto scan_index : selected_0based) {
                selected_1based.push_back(scan_index + 1);
            }

            logger->info(
                "{} TOD output selection mode uniform_plus_source_crossing: n_uniform={} n_source_dense={} source_scan={} source_min_distance_arcsec={:.3f} selected={}",
                stream_name,
                n_uniform,
                n_source_dense,
                source_scan + 1,
                std::isfinite(best_scan_d2) ? std::sqrt(best_scan_d2) * RAD_TO_ASEC
                                            : std::numeric_limits<double>::quiet_NaN(),
                vector_to_string(selected_1based));
            return selected_1based;
        };

    auto setup_one = [&](const std::string &stream_name,
                         bool output_enabled,
                         bool select_enabled,
                         const std::vector<Eigen::Index> &chunks_1based,
                         const std::string &selection_mode,
                         int uniform_count,
                         int source_dense_count,
                         Eigen::VectorXI &scan_to_output,
                         Eigen::Index &n_output_scans) {
        scan_to_output.resize(n_scans);
        scan_to_output.setConstant(-1);
        n_output_scans = 0;

        if (!output_enabled) {
            logger->info("{} TOD output disabled", stream_name);
            return;
        }

        std::vector<Eigen::Index> effective_chunks = chunks_1based;
        bool effective_select_enabled = select_enabled;
        if (selection_mode == "all") {
            effective_select_enabled = false;
            effective_chunks.clear();
        }
        else if (selection_mode == "uniform_plus_source_crossing") {
            effective_select_enabled = true;
            effective_chunks = build_uniform_plus_source_crossing_chunks(
                stream_name, uniform_count, source_dense_count);
            if (effective_chunks.empty()) {
                logger->error("{} TOD output selection mode uniform_plus_source_crossing selected no chunks",
                              stream_name);
                std::exit(EXIT_FAILURE);
            }
        }
        else if (selection_mode != "indices") {
            logger->error("{} TOD output selection mode '{}' is invalid", stream_name, selection_mode);
            std::exit(EXIT_FAILURE);
        }

        if (!effective_select_enabled || effective_chunks.empty()) {
            for (Eigen::Index i = 0; i < n_scans; ++i) {
                scan_to_output(i) = i;
            }
            n_output_scans = n_scans;
            logger->info("{} TOD output chunk selection disabled: writing all {} chunks",
                         stream_name, n_output_scans);
            return;
        }

        std::set<Eigen::Index> selected_chunks;
        for (const auto chunk_1based : effective_chunks) {
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
                  tod_output_selection_mode_rtc, tod_output_uniform_count_rtc, tod_output_source_dense_count_rtc,
                  tod_scan_to_output_scan_rtc, n_tod_output_scans_rtc);
        setup_one("PTC", run_tod_output_ptc, tod_output_chunk_select_enabled_ptc, tod_output_chunks_ptc,
                  tod_output_selection_mode_ptc, tod_output_uniform_count_ptc, tod_output_source_dense_count_ptc,
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
    auto &typed_despike = typed_timestream_config.raw_time_chunk.despike;
    typed_despike.enabled = rtcproc.run_despike;
    typed_despike.source_protection.enabled =
        rtcproc.despike_source_protection_config_enabled;
    typed_despike.source_protection.radius_arcsec =
        rtcproc.despiker.source_protection_radius_arcsec;
    if (rtcproc.run_despike) {
        typed_despike.min_spike_sigma = rtcproc.despiker.min_spike_sigma;
        typed_despike.time_constant_sec = rtcproc.despiker.time_constant_sec;
        typed_despike.window_size = rtcproc.despiker.window_size;
        typed_despike.legacy_enabled = rtcproc.despiker.run_legacy;

        const auto &local = rtcproc.despiker.local_residual;
        auto &typed_local = typed_despike.local_residual;
        typed_local.enabled = local.enabled;
        typed_local.window_sec = local.window_sec;
        typed_local.sigma_scale = local.sigma_scale;
        typed_local.delta_sigma_scale = local.delta_sigma_scale;
        typed_local.expand_with_filter = local.expand_with_filter;
        typed_local.event_padding_sec = local.event_padding_sec;
        typed_local.high_score_event_override = local.high_score_event_override;
        typed_local.max_added_flagged_fraction = local.max_added_flagged_fraction;
        typed_local.compact_raw_gate.enabled = local.compact_raw_gate.enabled;
        typed_local.compact_raw_gate.candidate_rel_sigma_scale =
            local.compact_raw_gate.candidate_rel_sigma_scale;
        typed_local.compact_raw_gate.window_sec = local.compact_raw_gate.window_sec;
        typed_local.compact_raw_gate.half_peak_frac =
            local.compact_raw_gate.half_peak_frac;
        typed_local.compact_raw_gate.max_width_sec =
            local.compact_raw_gate.max_width_sec;
        typed_local.compact_raw_gate.max_step_shift_z =
            local.compact_raw_gate.max_step_shift_z;
        typed_local.compact_delta_gate.enabled = local.compact_delta_gate.enabled;
        typed_local.compact_delta_gate.window_sec =
            local.compact_delta_gate.window_sec;
        typed_local.compact_delta_gate.half_peak_frac =
            local.compact_delta_gate.half_peak_frac;
        typed_local.compact_delta_gate.max_width_sec =
            local.compact_delta_gate.max_width_sec;
        typed_local.compact_delta_gate.max_step_shift_z =
            local.compact_delta_gate.max_step_shift_z;
    }

    auto &typed_raw = typed_timestream_config.raw_time_chunk;
    auto &typed_flagging = typed_raw.flagging;
    typed_flagging.delta_f_min_Hz = rtcproc.delta_f_min_Hz;
    typed_flagging.lower_tod_inv_var_factor = rtcproc.lower_inv_var_factor;
    typed_flagging.upper_tod_inv_var_factor = rtcproc.upper_inv_var_factor;

    const auto &network_step_mask = rtcproc.network_step_mask;
    auto &typed_network_step = typed_flagging.network_step_mask;
    typed_network_step.enabled = network_step_mask.enabled;
    typed_network_step.step_window_sec = network_step_mask.step_window_sec;
    typed_network_step.step_score_thresh = network_step_mask.step_score_thresh;
    typed_network_step.min_good_frac = network_step_mask.min_good_frac;
    typed_network_step.min_det_used =
        static_cast<int>(network_step_mask.min_det_used);
    typed_network_step.min_step_det_frac =
        network_step_mask.min_step_det_frac;
    typed_network_step.min_alignment_frac =
        network_step_mask.min_alignment_frac;
    typed_network_step.cluster_tol_sec = network_step_mask.cluster_tol_sec;
    typed_network_step.mask_half_width_sec =
        network_step_mask.mask_half_width_sec;
    typed_network_step.max_flagged_fraction =
        network_step_mask.max_flagged_fraction;

    const auto &impulsive_capture = rtcproc.impulsive_capture;
    auto &typed_capture = typed_flagging.impulsive_capture;
    typed_capture.enabled = impulsive_capture.enabled;
    typed_capture.min_good_frac = impulsive_capture.min_good_frac;
    typed_capture.min_event_z = impulsive_capture.min_event_z;
    typed_capture.near_event_z = impulsive_capture.near_event_z;
    typed_capture.max_events_per_network =
        static_cast<int>(impulsive_capture.max_events_per_network);
    typed_capture.snippet_pre_window_sec =
        impulsive_capture.snippet_pre_window_sec;
    typed_capture.snippet_post_window_sec =
        impulsive_capture.snippet_post_window_sec;

    const auto &impulsive_coincidence = rtcproc.impulsive_coincidence;
    auto &typed_coincidence = typed_flagging.impulsive_coincidence;
    typed_coincidence.enabled = impulsive_coincidence.enabled;
    typed_coincidence.min_good_frac = impulsive_coincidence.min_good_frac;
    typed_coincidence.event_score_thresh =
        impulsive_coincidence.event_score_thresh;
    typed_coincidence.min_det_used =
        static_cast<int>(impulsive_coincidence.min_det_used);
    typed_coincidence.min_impulsive_det_frac =
        impulsive_coincidence.min_impulsive_det_frac;
    typed_coincidence.min_alignment_frac =
        impulsive_coincidence.min_alignment_frac;
    typed_coincidence.min_networks_aligned =
        static_cast<int>(impulsive_coincidence.min_networks_aligned);
    typed_coincidence.high_score_override_thresh =
        impulsive_coincidence.high_score_override_thresh;
    typed_coincidence.high_score_min_networks_aligned =
        static_cast<int>(
            impulsive_coincidence.high_score_min_networks_aligned);
    typed_coincidence.cluster_tol_sec = impulsive_coincidence.cluster_tol_sec;
    typed_coincidence.mask_pre_window_sec =
        impulsive_coincidence.mask_pre_window_sec;
    typed_coincidence.mask_post_window_sec =
        impulsive_coincidence.mask_post_window_sec;
    typed_coincidence.max_flagged_fraction =
        impulsive_coincidence.max_flagged_fraction;

    auto &typed_kernel = typed_raw.kernel;
    typed_kernel.enabled = rtcproc.run_kernel;
    if (rtcproc.run_kernel) {
        typed_kernel.filepath = rtcproc.kernel.filepath;
        typed_kernel.type = rtcproc.kernel.type;
        typed_kernel.fwhm_arcsec = rtcproc.kernel.fwhm_rad * RAD_TO_ASEC;
        typed_kernel.image_ext_names = rtcproc.kernel.img_ext_names;
    }

    auto &typed_altaz = typed_raw.altaz_destripe;
    typed_altaz.enabled = rtcproc.altaz_destripe.enabled;
    typed_altaz.grouping = rtcproc.altaz_destripe.grouping;
    typed_altaz.fit_time_trend = rtcproc.altaz_destripe.fit_time_trend;
    typed_altaz.fit_derivs = rtcproc.altaz_destripe.fit_derivs;
    typed_altaz.min_samples =
        static_cast<int>(rtcproc.altaz_destripe.min_samples);

    const auto &line_audit = rtcproc.line_audit;
    auto &typed_line_audit = typed_raw.line_audit;
    typed_line_audit.enabled = line_audit.enabled;
    typed_line_audit.line_min_hz = line_audit.line_min_hz;
    typed_line_audit.line_max_hz = line_audit.line_max_hz;
    typed_line_audit.segment_sec = line_audit.segment_sec;
    typed_line_audit.min_segment_sec = line_audit.min_segment_sec;
    typed_line_audit.overlap_frac = line_audit.overlap_frac;
    typed_line_audit.continuum_radius_bins =
        static_cast<int>(line_audit.continuum_radius_bins);
    typed_line_audit.prominence_thresh = line_audit.prominence_thresh;
    typed_line_audit.cm_prominence_thresh = line_audit.cm_prominence_thresh;
    typed_line_audit.min_good_frac = line_audit.min_good_frac;
    typed_line_audit.min_windows = static_cast<int>(line_audit.min_windows);
    typed_line_audit.max_peaks_per_detector =
        static_cast<int>(line_audit.max_peaks_per_detector);
    typed_line_audit.max_det = static_cast<int>(line_audit.max_det);
    typed_line_audit.min_det_for_network =
        static_cast<int>(line_audit.min_det_for_network);
    typed_line_audit.cluster_tol_hz = line_audit.cluster_tol_hz;
    typed_line_audit.notch_min_detector_frac =
        line_audit.notch_min_detector_frac;
    typed_line_audit.notch_min_detectors =
        static_cast<int>(line_audit.notch_min_detectors);
    typed_line_audit.notch_min_cm_prominence =
        line_audit.notch_min_cm_prominence;
    typed_line_audit.detector_min_prominence =
        line_audit.detector_min_prominence;
    typed_line_audit.detector_min_line_power_frac =
        line_audit.detector_min_line_power_frac;
    typed_line_audit.bad_detector_max_cluster_frac =
        line_audit.bad_detector_max_cluster_frac;
    typed_line_audit.pre_filter_enabled = line_audit.pre_filter_enabled;
    typed_line_audit.post_filter_enabled = line_audit.post_filter_enabled;
    typed_line_audit.post_filter_apply_shared_notches =
        line_audit.post_filter_apply_shared_notches;
    typed_line_audit.post_filter_apply_detector_notches =
        line_audit.post_filter_apply_detector_notches;
    typed_line_audit.post_filter_apply_iterations =
        static_cast<int>(line_audit.post_filter_apply_iterations);
    typed_line_audit.post_filter_line_min_hz =
        line_audit.post_filter_line_min_hz;
    typed_line_audit.post_filter_line_max_hz =
        line_audit.post_filter_line_max_hz;
    typed_line_audit.ptc_model_protected_enabled =
        line_audit.ptc_model_protected_enabled;
    typed_line_audit.ptc_require_model_subtracted =
        line_audit.ptc_require_model_subtracted;
    typed_line_audit.ptc_apply_fixed_notches =
        line_audit.ptc_apply_fixed_notches;
    typed_line_audit.ptc_apply_shared_notches =
        line_audit.ptc_apply_shared_notches;
    typed_line_audit.ptc_apply_detector_notches =
        line_audit.ptc_apply_detector_notches;
    typed_line_audit.ptc_apply_iterations =
        static_cast<int>(line_audit.ptc_apply_iterations);
    typed_line_audit.ptc_line_min_hz = line_audit.ptc_line_min_hz;
    typed_line_audit.ptc_line_max_hz = line_audit.ptc_line_max_hz;
    typed_line_audit.fixed_notch_enabled = line_audit.fixed_notch_enabled;
    typed_line_audit.fixed_notch_freqs_hz =
        line_audit.fixed_notch_freqs_hz;
    typed_line_audit.fixed_notch_widths_hz =
        line_audit.fixed_notch_widths_hz;
    typed_line_audit.fixed_notch_exclusion_half_width_hz =
        line_audit.fixed_notch_exclusion_half_width_hz;
    typed_line_audit.apply_shared_notches =
        line_audit.apply_shared_notches;
    typed_line_audit.apply_min_support_networks =
        static_cast<int>(line_audit.apply_min_support_networks);
    typed_line_audit.apply_min_detector_frac =
        line_audit.apply_min_detector_frac;
    typed_line_audit.apply_min_common_mode_prominence =
        line_audit.apply_min_common_mode_prominence;
    typed_line_audit.apply_width_scale = line_audit.apply_width_scale;
    typed_line_audit.apply_min_width_hz = line_audit.apply_min_width_hz;
    typed_line_audit.apply_max_width_hz = line_audit.apply_max_width_hz;
    typed_line_audit.apply_max_notches =
        static_cast<int>(line_audit.apply_max_notches);
    typed_line_audit.apply_cluster_tol_hz =
        line_audit.apply_cluster_tol_hz;
    typed_line_audit.detector_notch_min_prominence =
        line_audit.detector_notch_min_prominence;
    typed_line_audit.detector_notch_min_line_power_frac =
        line_audit.detector_notch_min_line_power_frac;
    typed_line_audit.detector_notch_max_notches =
        static_cast<int>(line_audit.detector_notch_max_notches);
    typed_line_audit.detector_notch_width_scale =
        line_audit.detector_notch_width_scale;
    typed_line_audit.detector_notch_min_width_hz =
        line_audit.detector_notch_min_width_hz;
    typed_line_audit.detector_notch_max_width_hz =
        line_audit.detector_notch_max_width_hz;
    typed_line_audit.detector_notch_context_samples =
        static_cast<int>(line_audit.detector_notch_context_samples);

    typed_raw.downsample.enabled = rtcproc.run_downsample;
    if (rtcproc.run_downsample) {
        typed_raw.downsample.factor = rtcproc.downsampler.factor;
        typed_raw.downsample.downsampled_freq_Hz =
            rtcproc.downsampler.downsampled_freq_Hz;
    }

    auto &typed_filter = typed_raw.filter;
    typed_filter.enabled = rtcproc.run_tod_filter;
    if (rtcproc.run_tod_filter) {
        typed_filter.a_gibbs = rtcproc.filter.a_gibbs;
        typed_filter.freq_low_Hz = rtcproc.filter.freq_low_Hz;
        typed_filter.freq_high_Hz = rtcproc.filter.freq_high_Hz;
        typed_filter.n_terms = static_cast<int>(rtcproc.filter.n_terms);
        typed_filter.notch.enabled = rtcproc.run_tod_notch;
        if (rtcproc.run_tod_notch) {
            typed_filter.notch.zero_phase = rtcproc.filter.notch_zero_phase;
            typed_filter.notch.freqs_Hz = rtcproc.filter.w0s;
            typed_filter.notch.delta_f_Hz.clear();
            typed_filter.notch.delta_f_Hz.reserve(rtcproc.filter.qs.size());
            for (std::size_t i = 0; i < rtcproc.filter.qs.size(); ++i) {
                const auto center_Hz = i < rtcproc.filter.w0s.size()
                                           ? rtcproc.filter.w0s[i]
                                           : 0.0;
                typed_filter.notch.delta_f_Hz.push_back(
                    rtcproc.filter.qs[i] > 0.0
                        ? center_Hz / rtcproc.filter.qs[i]
                        : 0.0);
            }
        }
    }

    auto &typed_iir_filter = typed_raw.iir_filter;
    typed_iir_filter.enabled = rtcproc.run_tod_iir_highpass;
    if (rtcproc.run_tod_iir_highpass) {
        typed_iir_filter.freq_Hz = rtcproc.filter.iir_highpass_freq_Hz;
        typed_iir_filter.order = rtcproc.filter.iir_highpass_order;
        typed_iir_filter.zero_phase = rtcproc.filter.iir_highpass_zero_phase;
    }

    typed_raw.flux_calibration_enabled = rtcproc.run_calibrate;
    typed_raw.extinction_correction_enabled = rtcproc.run_extinction;

    rtcproc.configure_filter_edge_guard(telescope.fsmp);
    auto &typed_edge_guard = typed_filter.edge_guard;
    typed_edge_guard.enabled = rtcproc.filter_edge_guard.enabled;
    if (auto parsed = citlali::config::parse_raw_filter_edge_guard_mode(
            rtcproc.filter_edge_guard.mode)) {
        typed_edge_guard.mode = *parsed;
    }
    if (auto parsed = citlali::config::parse_raw_filter_edge_guard_combine(
            rtcproc.filter_edge_guard.combine)) {
        typed_edge_guard.combine = *parsed;
    }
    typed_edge_guard.min_samples =
        static_cast<int>(rtcproc.filter_edge_guard.min_samples);
    typed_edge_guard.extra_samples =
        static_cast<int>(rtcproc.filter_edge_guard.extra_samples);
    typed_edge_guard.max_samples =
        static_cast<int>(rtcproc.filter_edge_guard.max_samples);
    typed_edge_guard.iir_settle_attenuation =
        rtcproc.filter_edge_guard.iir_settle_attenuation;
    typed_edge_guard.apply_fir = rtcproc.filter_edge_guard.apply_fir;
    typed_edge_guard.apply_notch = rtcproc.filter_edge_guard.apply_notch;
    typed_edge_guard.apply_dynamic_notch =
        rtcproc.filter_edge_guard.apply_dynamic_notch;
    typed_edge_guard.apply_iir_highpass =
        rtcproc.filter_edge_guard.apply_iir_highpass;
    typed_edge_guard.apply_downsample =
        rtcproc.filter_edge_guard.apply_downsample;
    telescope.inner_scans_chunk = rtcproc.filter_edge_guard.context_samples;
    telescope.outer_scans_chunk = telescope.inner_scans_chunk;
    if (rtcproc.tod_output_outer) {
        telescope.outer_scans_chunk = std::max<Eigen::Index>(
            telescope.outer_scans_chunk,
            std::max<Eigen::Index>(0, rtcproc.tod_output_outer_context_samples));
    }
    if (rtcproc.line_audit.enabled &&
        rtcproc.line_audit.post_filter_enabled &&
        rtcproc.line_audit.post_filter_apply_detector_notches) {
        telescope.outer_scans_chunk = std::max<Eigen::Index>(
            telescope.outer_scans_chunk,
            std::max<Eigen::Index>(0, rtcproc.line_audit.detector_notch_context_samples));
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
    auto &typed_fruit_loops = typed_timestream_config.fruit_loops;
    typed_fruit_loops.enabled = ptcproc.run_fruit_loops;
    if (ptcproc.run_fruit_loops) {
        typed_fruit_loops.save_all_iters = ptcproc.save_all_iters;
        typed_fruit_loops.path = ptcproc.fruit_loops_path;
        typed_fruit_loops.type = ptcproc.fruit_loops_type;
        if (auto parsed = citlali::config::parse_fruit_loops_mode(
                ptcproc.fruit_mode)) {
            typed_fruit_loops.mode = *parsed;
        }
        typed_fruit_loops.sig2noise_limit = ptcproc.fruit_loops_sig2noise;
        typed_fruit_loops.array_flux_limit.clear();
        typed_fruit_loops.array_flux_limit.reserve(
            static_cast<std::size_t>(ptcproc.fruit_loops_flux.size()));
        for (Eigen::Index i = 0; i < ptcproc.fruit_loops_flux.size(); ++i) {
            typed_fruit_loops.array_flux_limit.push_back(
                ptcproc.fruit_loops_flux(i));
        }
        typed_fruit_loops.peak_fraction_limit =
            ptcproc.fruit_loops_peak_fraction_limit;
        typed_fruit_loops.local_snr_floor =
            ptcproc.fruit_loops_local_snr_floor;
        typed_fruit_loops.local_sigma_inner_radius_arcsec =
            ptcproc.fruit_loops_local_sigma_inner_radius_arcsec;
        typed_fruit_loops.local_sigma_outer_radius_arcsec =
            ptcproc.fruit_loops_local_sigma_outer_radius_arcsec;
        typed_fruit_loops.local_sigma_inner_fwhm =
            ptcproc.fruit_loops_local_sigma_inner_fwhm;
        typed_fruit_loops.local_sigma_outer_fwhm =
            ptcproc.fruit_loops_local_sigma_outer_fwhm;
        typed_fruit_loops.local_sigma_edge_guard_arcsec =
            ptcproc.fruit_loops_local_sigma_edge_guard_arcsec;
        typed_fruit_loops.local_sigma_min_pixels =
            ptcproc.fruit_loops_local_sigma_min_pixels;
        typed_fruit_loops.adaptive_support_radius_arcsec =
            ptcproc.fruit_loops_adaptive_support_radius_arcsec;
        typed_fruit_loops.adaptive_support_radius_fwhm =
            ptcproc.fruit_loops_adaptive_support_radius_fwhm;
        typed_fruit_loops.weight_feedback.enabled =
            ptcproc.fruit_loops_weight_feedback_enabled;
        if (auto parsed =
                citlali::config::parse_fruit_loops_weight_feedback_reference(
                    ptcproc.fruit_loops_weight_feedback_reference)) {
            typed_fruit_loops.weight_feedback.reference = *parsed;
        }
        typed_fruit_loops.weight_feedback.low_relative_weight =
            ptcproc.fruit_loops_weight_feedback_low_relative_weight;
        typed_fruit_loops.weight_feedback.high_relative_weight =
            ptcproc.fruit_loops_weight_feedback_high_relative_weight;
        typed_fruit_loops.center_keep_radius_arcsec =
            ptcproc.fruit_loops_center_keep_radius_arcsec;
        if (auto parsed =
                citlali::config::parse_fruit_loops_interp_mode_override(
                    ptcproc.fruit_loops_interp_mode_override)) {
            typed_fruit_loops.interp_mode_override = *parsed;
        }
        typed_fruit_loops.legacy_center = ptcproc.fruit_loops_legacy_center;
        typed_fruit_loops.recompute_weights_after_addback =
            ptcproc.fruit_loops_recompute_weights_after_addback;
        typed_fruit_loops.max_iters = ptcproc.fruit_loops_iters;
    }
    auto &typed_clean = typed_timestream_config.processed_time_chunk.clean;
    typed_clean.enabled = ptcproc.run_clean;
    if (ptcproc.run_clean) {
        if (auto parsed = citlali::config::parse_processed_cleaner_mode(
                ptcproc.cleaner.active_cleaner_label())) {
            typed_clean.active = *parsed;
        }
        typed_clean.grouping = ptcproc.cleaner.grouping;
        typed_clean.mask_radius_arcsec = ptcproc.mask_radius_arcsec;
        typed_clean.tau = ptcproc.cleaner.tau;
        typed_clean.standard_pca.enabled =
            ptcproc.cleaner.standard_pca.enabled;
        typed_clean.standard_pca.stddev_limit = ptcproc.cleaner.stddev_limit;
        typed_clean.standard_pca.n_calc = ptcproc.cleaner.n_calc;
        typed_clean.standard_pca.n_eig_to_cut.clear();
        for (const auto &[arr_index, arr_name] : toltec_io.array_name_map) {
            const auto it = ptcproc.cleaner.n_eig_to_cut.find(arr_index);
            if (it == ptcproc.cleaner.n_eig_to_cut.end()) {
                continue;
            }
            std::vector<int> n_eig_to_cut;
            n_eig_to_cut.reserve(static_cast<std::size_t>(it->second.size()));
            for (Eigen::Index i = 0; i < it->second.size(); ++i) {
                n_eig_to_cut.push_back(static_cast<int>(it->second(i)));
            }
            typed_clean.standard_pca.n_eig_to_cut[arr_name] =
                std::move(n_eig_to_cut);
        }
        auto &typed_corr_grouping = typed_clean.corr_grouping;
        typed_corr_grouping.enabled = ptcproc.cleaner.corr_grouping.enabled;
        if (auto parsed = citlali::config::parse_processed_corr_grouping_metric(
                ptcproc.cleaner.corr_grouping.metric)) {
            typed_corr_grouping.metric = *parsed;
        }
        typed_corr_grouping.corr_min = ptcproc.cleaner.corr_grouping.corr_min;
        typed_corr_grouping.min_overlap =
            ptcproc.cleaner.corr_grouping.min_overlap;
        typed_corr_grouping.min_good_frac =
            ptcproc.cleaner.corr_grouping.min_good_frac;
        typed_corr_grouping.min_group_size =
            ptcproc.cleaner.corr_grouping.min_group_size;
        typed_corr_grouping.max_samples =
            ptcproc.cleaner.corr_grouping.max_samples;
        typed_corr_grouping.clean_residual =
            ptcproc.cleaner.corr_grouping.clean_residual;

        auto &typed_null_model = typed_clean.null_model;
        typed_null_model.enabled = ptcproc.cleaner.null_model.enabled;
        typed_null_model.n_surrogates =
            ptcproc.cleaner.null_model.n_surrogates;
        typed_null_model.quantile = ptcproc.cleaner.null_model.quantile;
        typed_null_model.min_good_frac =
            ptcproc.cleaner.null_model.min_good_frac;
        typed_null_model.max_modes = ptcproc.cleaner.null_model.max_modes;
        typed_null_model.max_samples = ptcproc.cleaner.null_model.max_samples;
        typed_null_model.seed = static_cast<int>(ptcproc.cleaner.null_model.seed);
        typed_null_model.grouping = ptcproc.cleaner.null_model.grouping;

        auto &typed_mp = typed_clean.marchenko_pastur;
        typed_mp.enabled = ptcproc.cleaner.marchenko_pastur.enabled;
        typed_mp.min_good_frac =
            ptcproc.cleaner.marchenko_pastur.min_good_frac;
        typed_mp.max_modes = ptcproc.cleaner.marchenko_pastur.max_modes;
        typed_mp.max_samples = ptcproc.cleaner.marchenko_pastur.max_samples;
        typed_mp.band_low_Hz = ptcproc.cleaner.marchenko_pastur.band_low_Hz;
        typed_mp.band_high_Hz = ptcproc.cleaner.marchenko_pastur.band_high_Hz;
        typed_mp.clip_z = ptcproc.cleaner.marchenko_pastur.clip_z;
        typed_mp.bulk_keep_frac =
            ptcproc.cleaner.marchenko_pastur.bulk_keep_frac;
        typed_mp.q_grid_size = ptcproc.cleaner.marchenko_pastur.q_grid_size;
        typed_mp.grouping = ptcproc.cleaner.marchenko_pastur.grouping;

        auto &typed_adaptive = typed_clean.adaptive_selector;
        typed_adaptive.enabled = ptcproc.cleaner.adaptive_selector.enabled;
        typed_adaptive.min_good_frac =
            ptcproc.cleaner.adaptive_selector.min_good_frac;
        typed_adaptive.max_det = ptcproc.cleaner.adaptive_selector.max_det;
        typed_adaptive.max_samples =
            ptcproc.cleaner.adaptive_selector.max_samples;
        typed_adaptive.max_pairs = ptcproc.cleaner.adaptive_selector.max_pairs;
        typed_adaptive.seed =
            static_cast<int>(ptcproc.cleaner.adaptive_selector.seed);
        typed_adaptive.clip_z = ptcproc.cleaner.adaptive_selector.clip_z;
        typed_adaptive.low_weight =
            ptcproc.cleaner.adaptive_selector.low_weight;
        typed_adaptive.tail_weight =
            ptcproc.cleaner.adaptive_selector.tail_weight;
        typed_adaptive.topmode_weight =
            ptcproc.cleaner.adaptive_selector.topmode_weight;
        typed_adaptive.reg_weight =
            ptcproc.cleaner.adaptive_selector.reg_weight;
        typed_adaptive.low_band_Hz =
            ptcproc.cleaner.adaptive_selector.low_band_Hz;
        typed_adaptive.mid_band_Hz =
            ptcproc.cleaner.adaptive_selector.mid_band_Hz;
        typed_adaptive.candidate_offsets =
            ptcproc.cleaner.adaptive_selector.candidate_offsets;
        typed_adaptive.grouping = ptcproc.cleaner.adaptive_selector.grouping;
        typed_adaptive.log_candidates =
            ptcproc.cleaner.adaptive_selector.log_candidates;
    }
    auto &typed_weighting =
        typed_timestream_config.processed_time_chunk.weighting;
    if (auto parsed =
            citlali::config::parse_processed_weighting_type(ptcproc.weighting_type)) {
        typed_weighting.type = *parsed;
    }
    typed_weighting.source_mask_radius_arcsec =
        ptcproc.source_mask_radius_arcsec;
    typed_weighting.hybrid_correction_min_factor =
        ptcproc.hybrid_correction_min_factor;
    typed_weighting.hybrid_correction_max_factor =
        ptcproc.hybrid_correction_max_factor;
    typed_weighting.median_map_weight_factor = ptcproc.med_weight_factor;
    typed_weighting.lower_map_weight_factor = ptcproc.lower_weight_factor;
    typed_weighting.upper_map_weight_factor = ptcproc.upper_weight_factor;
    auto &typed_flagging =
        typed_timestream_config.processed_time_chunk.flagging;
    typed_flagging.lower_tod_inv_var_factor = ptcproc.lower_inv_var_factor;
    typed_flagging.upper_tod_inv_var_factor = ptcproc.upper_inv_var_factor;
    auto &typed_busy_row = typed_weighting.busy_row_suppression;
    typed_busy_row.enabled = ptcproc.busy_row_suppression.enabled;
    typed_busy_row.require_busy_veto =
        ptcproc.busy_row_suppression.require_busy_veto;
    typed_busy_row.min_candidate_clusters =
        ptcproc.busy_row_suppression.min_candidate_clusters;
    typed_busy_row.min_max_unflagged_residual_z =
        ptcproc.busy_row_suppression.min_max_unflagged_residual_z;
    typed_busy_row.factor = ptcproc.busy_row_suppression.factor;
    const auto &weight_validation = ptcproc.weight_validation;
    auto &typed_weight_validation = typed_weighting.validation;
    typed_weight_validation.enabled = weight_validation.enabled;
    typed_weight_validation.accumulation_iters =
        weight_validation.accumulation_iters;
    typed_weight_validation.apply_start_iter =
        weight_validation.apply_start_iter;
    typed_weight_validation.min_valid_scans =
        weight_validation.min_valid_scans;
    typed_weight_validation.min_factor = weight_validation.min_factor;
    typed_weight_validation.unvalidated_factor =
        weight_validation.unvalidated_factor;
    typed_weight_validation.require_fruitloops_model =
        weight_validation.require_fruitloops_model;
    typed_weight_validation.transient_ratio_enabled =
        weight_validation.transient_ratio_enabled;
    typed_weight_validation.ratio_power = weight_validation.ratio_power;
    typed_weight_validation.transient_ratio_power =
        weight_validation.transient_ratio_power;
    typed_weight_validation.upward_enabled = weight_validation.upward_enabled;
    typed_weight_validation.upward_max_factor =
        weight_validation.upward_max_factor;
    typed_weight_validation.upward_power = weight_validation.upward_power;
    typed_weight_validation.upward_min_base_factor =
        weight_validation.upward_min_base_factor;
    typed_weight_validation.upward_require_atmospheric =
        weight_validation.upward_require_atmospheric;
    typed_weight_validation.upward_min_atmospheric_factor =
        weight_validation.upward_min_atmospheric_factor;
    typed_weight_validation.atmospheric_correlation_enabled =
        weight_validation.atmospheric_correlation_enabled;
    if (auto parsed = citlali::config::parse_processed_weight_grouping(
            weight_validation.atmospheric_grouping)) {
        typed_weight_validation.atmospheric_grouping = *parsed;
    }
    typed_weight_validation.atmospheric_min_detectors =
        weight_validation.atmospheric_min_detectors;
    typed_weight_validation.atmospheric_ref = weight_validation.atmospheric_ref;
    typed_weight_validation.atmospheric_span =
        weight_validation.atmospheric_span;
    typed_weight_validation.atmospheric_power =
        weight_validation.atmospheric_power;
    typed_weight_validation.min_good_frac = weight_validation.min_good_frac;
    typed_weight_validation.min_overlap = weight_validation.min_overlap;
    typed_weight_validation.max_samples = weight_validation.max_samples;
    typed_weight_validation.high_weight_validation_enabled =
        weight_validation.high_weight_validation_enabled;
    typed_weight_validation.high_weight_apply_caps =
        weight_validation.high_weight_apply_caps;
    if (auto parsed = citlali::config::parse_processed_weight_grouping(
            weight_validation.high_weight_grouping)) {
        typed_weight_validation.high_weight_grouping = *parsed;
    }
    typed_weight_validation.high_weight_min_group_detectors =
        weight_validation.high_weight_min_group_detectors;
    typed_weight_validation.high_weight_log_robust_z =
        weight_validation.high_weight_log_robust_z;
    typed_weight_validation.high_weight_max_median_factor =
        weight_validation.high_weight_max_median_factor;
    typed_weight_validation.high_weight_cap_median_factor =
        weight_validation.high_weight_cap_median_factor;
    typed_weight_validation.high_weight_min_validated_factor =
        weight_validation.high_weight_min_validated_factor;

    const auto &weight_corr_penalty = ptcproc.weight_corr_penalty;
    auto &typed_corr_penalty = typed_weighting.corr_penalty;
    typed_corr_penalty.enabled = weight_corr_penalty.enabled;
    typed_corr_penalty.min_good_frac = weight_corr_penalty.min_good_frac;
    typed_corr_penalty.min_overlap = weight_corr_penalty.min_overlap;
    typed_corr_penalty.max_samples = weight_corr_penalty.max_samples;
    typed_corr_penalty.max_pairs = weight_corr_penalty.max_pairs;
    typed_corr_penalty.seed = static_cast<int>(weight_corr_penalty.seed);
    typed_corr_penalty.floor = weight_corr_penalty.floor;
    typed_corr_penalty.exponent = weight_corr_penalty.exponent;
    typed_corr_penalty.pair_corr.enabled =
        weight_corr_penalty.pair_corr.enabled;
    typed_corr_penalty.pair_corr.ref = weight_corr_penalty.pair_corr.ref;
    typed_corr_penalty.pair_corr.span = weight_corr_penalty.pair_corr.span;
    typed_corr_penalty.pair_corr.weight = weight_corr_penalty.pair_corr.weight;
    typed_corr_penalty.cm_el_corr.enabled =
        weight_corr_penalty.cm_el_corr.enabled;
    typed_corr_penalty.cm_el_corr.ref = weight_corr_penalty.cm_el_corr.ref;
    typed_corr_penalty.cm_el_corr.span = weight_corr_penalty.cm_el_corr.span;
    typed_corr_penalty.cm_el_corr.weight =
        weight_corr_penalty.cm_el_corr.weight;
    typed_corr_penalty.cm_low_mid_ratio.enabled =
        weight_corr_penalty.cm_low_mid_ratio.enabled;
    typed_corr_penalty.cm_low_mid_ratio.ref =
        weight_corr_penalty.cm_low_mid_ratio.ref;
    typed_corr_penalty.cm_low_mid_ratio.span =
        weight_corr_penalty.cm_low_mid_ratio.span;
    typed_corr_penalty.cm_low_mid_ratio.weight =
        weight_corr_penalty.cm_low_mid_ratio.weight;
    typed_corr_penalty.cm_low_mid_ratio.low_band_Hz = {
        weight_corr_penalty.cm_low_mid_ratio.low_min_Hz,
        weight_corr_penalty.cm_low_mid_ratio.low_max_Hz};
    typed_corr_penalty.cm_low_mid_ratio.mid_band_Hz = {
        weight_corr_penalty.cm_low_mid_ratio.mid_min_Hz,
        weight_corr_penalty.cm_low_mid_ratio.mid_max_Hz};

    auto &typed_second_pass =
        typed_timestream_config.processed_time_chunk.flagging.second_pass_local;
    typed_second_pass.enabled = ptcproc.second_pass_local.enabled;
    typed_second_pass.min_spike_sigma =
        ptcproc.second_pass_local.min_spike_sigma;
    typed_second_pass.min_good_frac = ptcproc.second_pass_local.min_good_frac;
    typed_second_pass.baseline_window_sec =
        ptcproc.second_pass_local.baseline_window_sec;
    typed_second_pass.sigma_scale = ptcproc.second_pass_local.sigma_scale;
    typed_second_pass.delta_sigma_scale =
        ptcproc.second_pass_local.delta_sigma_scale;
    typed_second_pass.raw_candidate_rel_sigma_scale =
        ptcproc.second_pass_local.raw_candidate_rel_sigma_scale;
    typed_second_pass.raw_window_sec =
        ptcproc.second_pass_local.raw_window_sec;
    typed_second_pass.raw_half_peak_frac =
        ptcproc.second_pass_local.raw_half_peak_frac;
    typed_second_pass.raw_max_width_sec =
        ptcproc.second_pass_local.raw_max_width_sec;
    typed_second_pass.delta_window_sec =
        ptcproc.second_pass_local.delta_window_sec;
    typed_second_pass.delta_half_peak_frac =
        ptcproc.second_pass_local.delta_half_peak_frac;
    typed_second_pass.delta_max_width_sec =
        ptcproc.second_pass_local.delta_max_width_sec;
    typed_second_pass.max_step_shift_z =
        ptcproc.second_pass_local.max_step_shift_z;
    typed_second_pass.high_score_event_override =
        ptcproc.second_pass_local.high_score_event_override;
    typed_second_pass.merge_within_detector_sec =
        ptcproc.second_pass_local.merge_within_detector_sec;
    typed_second_pass.cluster_events_sec =
        ptcproc.second_pass_local.cluster_events_sec;
    typed_second_pass.min_cluster_detectors =
        ptcproc.second_pass_local.min_cluster_detectors;
    typed_second_pass.high_score_cluster_override =
        ptcproc.second_pass_local.high_score_cluster_override;
    typed_second_pass.max_auto_flag_clusters_per_network =
        ptcproc.second_pass_local.max_auto_flag_clusters_per_network;
    typed_second_pass.selective_busy_network_acceptance_enabled =
        ptcproc.second_pass_local.selective_busy_network_acceptance_enabled;
    typed_second_pass.source_protection.enabled =
        ptcproc.second_pass_local.source_protection_config_enabled;
    typed_second_pass.source_protection.radius_arcsec =
        ptcproc.second_pass_local.source_protection_radius_arcsec;

    // copy tod output bool for eigenvalues
    ptcproc.run_tod_output = run_tod_output;
    ptcproc.write_evals = diagnostics.write_evals;
}

inline double Engine::processed_time_chunk_fs_hz() const {
    double fs_hz = telescope.fsmp;
    if (rtcproc.run_downsample && rtcproc.downsampler.factor > 1) {
        fs_hz /= static_cast<double>(rtcproc.downsampler.factor);
    }
    return fs_hz;
}

template <class calib_t>
Eigen::Index Engine::apply_model_protected_ptc_line_audit(
    TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
    calib_t &calib_for_scan,
    bool model_subtracted) {

    const auto &base_audit = rtcproc.line_audit;
    if (!base_audit.enabled || !base_audit.ptc_model_protected_enabled) {
        return 0;
    }
    if (base_audit.ptc_require_model_subtracted && !model_subtracted) {
        logger->debug(
            "skipping model-protected PTC line-audit notch pass for scan {} because no model was subtracted",
            ptcdata.index.data + 1);
        return 0;
    }
    if (!base_audit.ptc_apply_fixed_notches &&
        !base_audit.ptc_apply_shared_notches &&
        !base_audit.ptc_apply_detector_notches) {
        return 0;
    }

    auto audit = base_audit;
    audit.pre_filter_enabled = false;
    audit.post_filter_enabled = false;
    audit.apply_shared_notches = audit.ptc_apply_shared_notches;
    audit.post_filter_apply_detector_notches = audit.ptc_apply_detector_notches;
    audit.fixed_notch_enabled = audit.fixed_notch_enabled && audit.ptc_apply_fixed_notches;
    if (std::isfinite(base_audit.ptc_line_min_hz)) {
        audit.line_min_hz = base_audit.ptc_line_min_hz;
    }
    if (std::isfinite(base_audit.ptc_line_max_hz)) {
        audit.line_max_hz = base_audit.ptc_line_max_hz;
    }

    const double fs_hz = processed_time_chunk_fs_hz();
    if (!std::isfinite(fs_hz) || fs_hz <= 0.0) {
        logger->warn("skipping model-protected PTC line-audit notch pass; invalid fs_hz={}", fs_hz);
        return 0;
    }

    Eigen::Index total_notches = 0;
    Eigen::Index max_notches_per_timestream = 0;

    if (audit.fixed_notch_enabled) {
        const Eigen::Index n_fixed_sections =
            rtcproc.count_rtc_line_audit_fixed_notches(fs_hz, audit);
        const auto n_fixed =
            rtcproc.apply_rtc_line_audit_fixed_notches(ptcdata, fs_hz, audit);
        total_notches += n_fixed;
        if (n_fixed > 0) {
            max_notches_per_timestream += n_fixed_sections;
        }
    }

    if (audit.apply_shared_notches) {
        const Eigen::Index n_iters = std::max<Eigen::Index>(1, audit.ptc_apply_iterations);
        for (Eigen::Index iter = 0; iter < n_iters; ++iter) {
            rtcproc.capture_rtc_line_audit(
                ptcdata, calib_for_scan, 0, ptcdata.scans.data.rows(), audit, true);
            const auto n_shared =
                rtcproc.apply_rtc_line_audit_shared_notches(ptcdata, fs_hz, audit, true);
            total_notches += n_shared;
            if (n_shared > 0) {
                max_notches_per_timestream += n_shared;
            }
            if (n_shared <= 0) {
                break;
            }
        }
    }

    if (audit.post_filter_apply_detector_notches) {
        const auto n_detector =
            rtcproc.apply_rtc_line_audit_detector_notches(
                ptcdata, fs_hz, audit, 0, ptcdata.scans.data.rows());
        total_notches += n_detector;
        if (n_detector > 0) {
            if (audit.detector_notch_max_notches > 0) {
                max_notches_per_timestream +=
                    std::min<Eigen::Index>(audit.detector_notch_max_notches, n_detector);
            }
            else {
                max_notches_per_timestream += n_detector;
            }
        }
    }

    if (total_notches > 0) {
        ptcdata.status.tod_filtered = true;
        if (rtcproc.filter_edge_guard.enabled &&
            rtcproc.filter_edge_guard.apply_dynamic_notch &&
            max_notches_per_timestream > 0) {
            const double min_width_hz =
                std::min(audit.apply_min_width_hz, audit.detector_notch_min_width_hz);
            Eigen::Index guard_samples =
                max_notches_per_timestream *
                timestream::Filter::notch_settle_samples_for_width(
                    fs_hz, min_width_hz, rtcproc.filter_edge_guard.iir_settle_attenuation);
            guard_samples = std::max(guard_samples, rtcproc.filter_edge_guard.min_samples);
            guard_samples += rtcproc.filter_edge_guard.extra_samples;
            if (rtcproc.filter_edge_guard.max_samples > 0) {
                guard_samples = std::min(guard_samples, rtcproc.filter_edge_guard.max_samples);
            }
            guard_samples = std::max<Eigen::Index>(0, guard_samples);
            if (guard_samples > 0) {
                rtcproc.apply_filter_edge_guard(ptcdata, 0, ptcdata.scans.data.rows(), guard_samples);
            }
        }
        logger->info(
            "model-protected PTC line-audit notch pass scan {}: total_notches={} fs_hz={} model_subtracted={} fixed={} shared={} detector={}",
            ptcdata.index.data + 1,
            total_notches,
            fs_hz,
            model_subtracted,
            audit.fixed_notch_enabled,
            audit.apply_shared_notches,
            audit.post_filter_apply_detector_notches);
    }

    return total_notches;
}

template<typename CT>
void Engine::get_learning_config(CT &config) {
    ReductionLearningState::Options options;

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return missing_keys.size() == missing_before && invalid_keys.size() == invalid_before;
    };
    auto mirror_if_parsed = [&](auto &target, const auto &source,
                                std::size_t missing_before,
                                std::size_t invalid_before) {
        if (parsed_cleanly(missing_before, invalid_before)) {
            target = source;
        }
    };

    if (config.template has_typed<bool>(std::tuple{"timestream","learning","enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.enabled = options.enabled;
        }
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","diagnostics_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.diagnostics_enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","diagnostics_enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.diagnostics_enabled =
                options.diagnostics_enabled;
        }
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","learn_iters"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.learn_iters, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","learn_iters"}, {}, {0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.learn_iters = options.learn_iters;
        }
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","apply_start_iter"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.apply_start_iter, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","apply_start_iter"}, {}, {0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.apply_start_iter =
                options.apply_start_iter;
        }
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","max_records_per_type"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.max_records_per_type, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","max_records_per_type"}, {}, {0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.max_records_per_type =
                options.max_records_per_type;
        }
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","apply_sample_masks_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.apply_sample_masks_enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","apply_sample_masks_enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.apply_sample_masks_enabled =
                options.apply_sample_masks_enabled;
        }
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","apply_max_new_flagged_fraction"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.apply_max_new_flagged_fraction, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","apply_max_new_flagged_fraction"}, {}, {0.0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.apply_max_new_flagged_fraction =
                options.apply_max_new_flagged_fraction;
        }
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","map_pixel_outlier_diagnostics_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_diagnostics_enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_diagnostics_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.diagnostics_enabled,
            options.map_pixel_outlier_diagnostics_enabled, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","map_pixel_outlier_contributor_diagnostics_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_contributor_diagnostics_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_contributor_diagnostics_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.contributor_diagnostics_enabled,
            options.map_pixel_outlier_contributor_diagnostics_enabled,
            missing_before, invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_diagnostics_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_targeted_contributor_diagnostics_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_diagnostics_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.targeted_contributor_diagnostics_enabled,
            options.map_pixel_outlier_targeted_contributor_diagnostics_enabled,
            missing_before, invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_detector_exclusion_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.detector_exclusion_enabled,
            options.map_pixel_outlier_detector_exclusion_enabled,
            missing_before, invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","map_pixel_outlier_top_n"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_top_n, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_top_n"}, {}, {0});
        mirror_if_parsed(typed_timestream_config.learning.map_pixel_outlier.top_n,
                         options.map_pixel_outlier_top_n, missing_before,
                         invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_max_pixels"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_targeted_contributor_max_pixels,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_max_pixels"},
                         {}, {0});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.targeted_contributor_max_pixels,
            options.map_pixel_outlier_targeted_contributor_max_pixels,
            missing_before, invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_min_pixels"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_detector_exclusion_min_pixels,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_min_pixels"},
                         {}, {1});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.detector_exclusion_min_pixels,
            options.map_pixel_outlier_detector_exclusion_min_pixels,
            missing_before, invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","map_pixel_outlier_min_abs_z"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_min_abs_z, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_min_abs_z"}, {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.min_abs_z,
            options.map_pixel_outlier_min_abs_z, missing_before,
            invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","map_pixel_outlier_min_n_eff"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_min_n_eff, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_min_n_eff"}, {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.min_n_eff,
            options.map_pixel_outlier_min_n_eff, missing_before,
            invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","map_pixel_outlier_source_radius_arcsec"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_source_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_source_radius_arcsec"}, {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.source_radius_arcsec,
            options.map_pixel_outlier_source_radius_arcsec, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","busy_detector_exclusion_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.busy_detector_exclusion_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","busy_detector_exclusion_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.busy_detector.exclusion_enabled,
            options.busy_detector_exclusion_enabled, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","scan_network_pathology_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.enabled,
            options.scan_network_pathology_enabled, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","scan_network_pathology_apply_pre_rtc"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_apply_pre_rtc,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_apply_pre_rtc"});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.apply_pre_rtc,
            options.scan_network_pathology_apply_pre_rtc, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","scan_network_pathology_apply_pre_ptc"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_apply_pre_ptc,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_apply_pre_ptc"});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.apply_pre_ptc,
            options.scan_network_pathology_apply_pre_ptc, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","scan_network_pathology_apply_pre_mapmaking"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_apply_pre_mapmaking,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_apply_pre_mapmaking"});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.apply_pre_mapmaking,
            options.scan_network_pathology_apply_pre_mapmaking, missing_before,
            invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","scan_network_pathology_min_candidate_clusters"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_min_candidate_clusters,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_min_candidate_clusters"},
                         {}, {0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.min_candidate_clusters,
            options.scan_network_pathology_min_candidate_clusters,
            missing_before, invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","scan_network_pathology_min_candidate_events"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_min_candidate_events,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_min_candidate_events"},
                         {}, {0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.min_candidate_events,
            options.scan_network_pathology_min_candidate_events, missing_before,
            invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","scan_network_pathology_min_max_residual_z"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_min_max_residual_z,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_min_max_residual_z"},
                         {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.min_max_residual_z,
            options.scan_network_pathology_min_max_residual_z, missing_before,
            invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","scan_network_pathology_severe_candidate_events"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_severe_candidate_events,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_severe_candidate_events"},
                         {}, {0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.severe_candidate_events,
            options.scan_network_pathology_severe_candidate_events,
            missing_before, invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","scan_network_pathology_severe_max_residual_z"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_severe_max_residual_z,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_severe_max_residual_z"},
                         {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.severe_max_residual_z,
            options.scan_network_pathology_severe_max_residual_z,
            missing_before, invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","scan_network_pathology_max_new_flagged_fraction"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_max_new_flagged_fraction,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_max_new_flagged_fraction"},
                         {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.max_new_flagged_fraction,
            options.scan_network_pathology_max_new_flagged_fraction,
            missing_before, invalid_before);
    }

    reduction_learning.configure(options);
    const bool map_contribution_diag =
        reduction_learning.options.enabled &&
        reduction_learning.options.diagnostics_enabled &&
        reduction_learning.options.map_pixel_outlier_diagnostics_enabled &&
        reduction_learning.options.map_pixel_outlier_contributor_diagnostics_enabled;
    omb.contribution_diag_enabled = map_contribution_diag;
    cmb.contribution_diag_enabled = map_contribution_diag;
    logger->info(
        "reduction learning state configured: enabled={} diagnostics_enabled={} "
        "learn_iters={} apply_start_iter={} max_records_per_type={} "
        "apply_sample_masks_enabled={} apply_max_new_flagged_fraction={:.4g} "
        "map_pixel_outliers(enabled={} contributors={} targeted_contributors={} detector_exclusion={} top_n={} target_max={} exclude_min_pixels={} min_abs_z={} min_n_eff={} source_radius_arcsec={}) "
        "busy_detector_exclusion_enabled={} scan_network_pathology(enabled={} pre_rtc={} pre_ptc={} pre_mapmaking={} min_clusters={} min_events={} min_resid_z={} severe_events={} severe_resid_z={} max_new_flagged_fraction={:.4g})",
        reduction_learning.options.enabled,
        reduction_learning.options.diagnostics_enabled,
        reduction_learning.options.learn_iters,
        reduction_learning.options.apply_start_iter,
        reduction_learning.options.max_records_per_type,
        reduction_learning.options.apply_sample_masks_enabled,
        reduction_learning.options.apply_max_new_flagged_fraction,
        reduction_learning.options.map_pixel_outlier_diagnostics_enabled,
        reduction_learning.options.map_pixel_outlier_contributor_diagnostics_enabled,
        reduction_learning.options.map_pixel_outlier_targeted_contributor_diagnostics_enabled,
        reduction_learning.options.map_pixel_outlier_detector_exclusion_enabled,
        reduction_learning.options.map_pixel_outlier_top_n,
        reduction_learning.options.map_pixel_outlier_targeted_contributor_max_pixels,
        reduction_learning.options.map_pixel_outlier_detector_exclusion_min_pixels,
        reduction_learning.options.map_pixel_outlier_min_abs_z,
        reduction_learning.options.map_pixel_outlier_min_n_eff,
        reduction_learning.options.map_pixel_outlier_source_radius_arcsec,
        reduction_learning.options.busy_detector_exclusion_enabled,
        reduction_learning.options.scan_network_pathology_enabled,
        reduction_learning.options.scan_network_pathology_apply_pre_rtc,
        reduction_learning.options.scan_network_pathology_apply_pre_ptc,
        reduction_learning.options.scan_network_pathology_apply_pre_mapmaking,
        reduction_learning.options.scan_network_pathology_min_candidate_clusters,
        reduction_learning.options.scan_network_pathology_min_candidate_events,
        reduction_learning.options.scan_network_pathology_min_max_residual_z,
        reduction_learning.options.scan_network_pathology_severe_candidate_events,
        reduction_learning.options.scan_network_pathology_severe_max_residual_z,
        reduction_learning.options.scan_network_pathology_max_new_flagged_fraction);
}

void Engine::configure_map_pixel_contribution_targets(mapmaking::MapBuffer &mb,
                                                      const std::string &stage_name) {
    const bool full_contribution_diag =
        reduction_learning.options.enabled &&
        reduction_learning.options.diagnostics_enabled &&
        reduction_learning.options.map_pixel_outlier_diagnostics_enabled &&
        reduction_learning.options.map_pixel_outlier_contributor_diagnostics_enabled;

    mb.clear_contribution_targets();
    mb.contribution_diag_enabled = full_contribution_diag;

    if (full_contribution_diag) {
        return;
    }
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled() ||
        !reduction_learning.options.map_pixel_outlier_diagnostics_enabled ||
        !reduction_learning.options.map_pixel_outlier_targeted_contributor_diagnostics_enabled ||
        reduction_learning.options.map_pixel_outlier_targeted_contributor_max_pixels <= 0 ||
        fruit_iter <= 0 ||
        mb.signal.empty() ||
        mb.n_rows <= 0 ||
        mb.n_cols <= 0) {
        return;
    }

    const std::string producer = "mapdiag:" + stage_name;
    int target_iter = -1;
    {
        std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
        for (const auto &record : reduction_learning.map_pixel_outliers) {
            if (record.obsnum == obsnum &&
                record.producer == producer &&
                record.iter >= 0 &&
                record.iter < fruit_iter &&
                record.map_index >= 0 &&
                record.map_index < static_cast<int>(mb.signal.size()) &&
                record.row >= 0 &&
                record.row < mb.n_rows &&
                record.col >= 0 &&
                record.col < mb.n_cols) {
                target_iter = std::max(target_iter, record.iter);
            }
        }
    }
    if (target_iter < 0) {
        return;
    }

    struct target_candidate_t {
        Eigen::Index map_index = -1;
        Eigen::Index row = -1;
        Eigen::Index col = -1;
        double score = 0.0;
    };
    std::vector<target_candidate_t> candidates;
    {
        std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
        for (const auto &record : reduction_learning.map_pixel_outliers) {
            if (record.obsnum != obsnum ||
                record.producer != producer ||
                record.iter != target_iter ||
                record.map_index < 0 ||
                record.map_index >= static_cast<int>(mb.signal.size()) ||
                record.row < 0 ||
                record.row >= mb.n_rows ||
                record.col < 0 ||
                record.col >= mb.n_cols) {
                continue;
            }
            const double raw_score =
                std::isfinite(record.leave_one_out_z)
                    ? std::abs(record.leave_one_out_z)
                    : std::abs(record.value);
            const double score = std::isfinite(raw_score) ? raw_score : 0.0;
            candidates.push_back({
                static_cast<Eigen::Index>(record.map_index),
                static_cast<Eigen::Index>(record.row),
                static_cast<Eigen::Index>(record.col),
                score});
        }
    }
    if (candidates.empty()) {
        return;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) {
                  return a.score > b.score;
              });

    std::vector<std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>> targets;
    targets.reserve(static_cast<std::size_t>(
        reduction_learning.options.map_pixel_outlier_targeted_contributor_max_pixels));
    auto have_target = [&](const auto &candidate) {
        return std::find_if(targets.begin(), targets.end(),
                            [&](const auto &target) {
                                return std::get<0>(target) == candidate.map_index &&
                                       std::get<1>(target) == candidate.row &&
                                       std::get<2>(target) == candidate.col;
                            }) != targets.end();
    };
    const std::size_t max_targets = static_cast<std::size_t>(
        reduction_learning.options.map_pixel_outlier_targeted_contributor_max_pixels);
    for (const auto &candidate : candidates) {
        if (targets.size() >= max_targets) {
            break;
        }
        if (have_target(candidate)) {
            continue;
        }
        targets.emplace_back(candidate.map_index, candidate.row, candidate.col);
    }
    if (targets.empty()) {
        return;
    }

    mb.set_contribution_targets(static_cast<Eigen::Index>(mb.signal.size()), targets);
    if (mb.contribution_diag_targeted) {
        mb.contribution_diag_enabled = true;
        logger->info(
            "map-pixel targeted contributor tracing enabled stage={} obsnum={} iter={} source_iter={} targets={}",
            stage_name, obsnum, fruit_iter, target_iter, targets.size());
    }
}

template <class apt_t>
static Eigen::Index citlali_learning_find_det_by_uid(const apt_t &apt, int uid) {
    if (uid == timestream::kTransientFillInt || uid < 0) {
        return -1;
    }
    const auto uid_it = apt.find("uid");
    if (uid_it == apt.end()) {
        return static_cast<Eigen::Index>(uid);
    }
    for (Eigen::Index i = 0; i < uid_it->second.size(); ++i) {
        if (std::isfinite(uid_it->second(i)) &&
            static_cast<int>(std::llround(uid_it->second(i))) == uid) {
            return i;
        }
    }
    return -1;
}

template <class apt_t>
static int citlali_learning_apt_int(const apt_t &apt, const std::string &key,
                                    Eigen::Index det, int fallback) {
    const auto it = apt.find(key);
    if (it == apt.end() || det < 0 || det >= it->second.size() ||
        !std::isfinite(it->second(det))) {
        return fallback;
    }
    return static_cast<int>(std::llround(it->second(det)));
}

template <class apt_t>
static int citlali_learning_array_for_nw(const apt_t &apt, int nw, int fallback) {
    const auto nw_it = apt.find("nw");
    const auto array_it = apt.find("array");
    if (nw_it == apt.end() || array_it == apt.end()) {
        return fallback;
    }
    const Eigen::Index n =
        std::min<Eigen::Index>(nw_it->second.size(), array_it->second.size());
    for (Eigen::Index det = 0; det < n; ++det) {
        if (!std::isfinite(nw_it->second(det)) ||
            !std::isfinite(array_it->second(det))) {
            continue;
        }
        if (static_cast<int>(std::llround(nw_it->second(det))) == nw) {
            return static_cast<int>(std::llround(array_it->second(det)));
        }
    }
    return fallback;
}

template <class rtc_t, class calib_t>
void Engine::apply_learned_rtc_sample_masks(rtc_t &rtcdata, calib_t &calib_scan) {
    apply_learned_detector_exclusions(
        rtcdata, calib_scan, "pre_rtc_detector_exclusion", true, false,
        true, true);
    apply_learned_sample_masks(
        rtcdata, calib_scan, true, "pre_rtc",
        rtcproc.despiker.source_protection_enabled,
        rtcproc.despiker.source_protection_radius_arcsec);
}

template <class ptc_t, class calib_t>
void Engine::apply_learned_ptc_sample_masks(ptc_t &ptcdata, calib_t &calib_scan) {
    apply_learned_sample_masks(
        ptcdata, calib_scan, false, "pre_ptc",
        ptcproc.second_pass_local.source_protection_enabled,
        ptcproc.second_pass_local.source_protection_radius_arcsec);
}

template <class ptc_t, class calib_t>
void Engine::apply_learned_ptc_detector_exclusions(ptc_t &ptcdata,
                                                   calib_t &calib_scan) {
    apply_learned_detector_exclusions(
        ptcdata, calib_scan, "pre_ptc_detector_exclusion", false, true,
        true, true);
}

template <class tc_t, class calib_t>
void Engine::apply_learned_mapmaking_detector_exclusions(tc_t &tcdata,
                                                         calib_t &calib_scan) {
    apply_learned_detector_exclusions(
        tcdata, calib_scan, "pre_mapmaking_detector_exclusion", false, false,
        false, true);
}

template <class tc_t, class calib_t>
void Engine::apply_learned_detector_exclusions(tc_t &tcdata,
                                               calib_t &calib_scan,
                                               const std::string &stage,
                                               bool pre_rtc,
                                               bool update_apt_flags,
                                               bool include_detector_records,
                                               bool include_network_records) {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.apply_active()) {
        return;
    }
    if (tcdata.flags.data.rows() <= 0 || tcdata.flags.data.cols() <= 0) {
        return;
    }

    const bool mapdiag_detector_exclusion =
        include_detector_records &&
        reduction_learning.options.map_pixel_outlier_detector_exclusion_enabled;
    const bool busy_detector_exclusion =
        include_detector_records &&
        reduction_learning.options.busy_detector_exclusion_enabled;
    const bool network_exclusion =
        include_network_records &&
        reduction_learning.options.scan_network_pathology_enabled &&
        (stage == "pre_mapmaking_detector_exclusion"
             ? reduction_learning.options.scan_network_pathology_apply_pre_mapmaking
             : ((!pre_rtc && reduction_learning.options.scan_network_pathology_apply_pre_ptc) ||
                (pre_rtc && reduction_learning.options.scan_network_pathology_apply_pre_rtc)));
    if (!mapdiag_detector_exclusion && !busy_detector_exclusion &&
        !network_exclusion) {
        return;
    }

    const int scan_id = static_cast<int>(tcdata.index.data);
    std::vector<ReductionLearningState::DetectorPenalty> records;
    {
        std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
        for (const auto &record : reduction_learning.detector_penalties) {
            if (record.obsnum != obsnum ||
                !record.scan_local ||
                record.scan != scan_id ||
                record.iter < 0 ||
                record.iter >= fruit_iter ||
                !std::isfinite(record.factor) ||
                record.factor > 0.0) {
                continue;
            }
            const bool is_mapdiag_detector =
                mapdiag_detector_exclusion &&
                record.uid >= 0 &&
                record.reason == "map_pixel_outlier_detector_dominance" &&
                record.producer.rfind("mapdiag:", 0) == 0;
            const bool is_busy_detector =
                busy_detector_exclusion &&
                record.uid >= 0 &&
                record.reason == "busy_vetoed_residual" &&
                record.producer == "ptc_second_pass";
            const bool is_network =
                network_exclusion &&
                record.uid < 0 &&
                record.nw >= 0 &&
                record.reason == "busy_network_pathology" &&
                record.producer == "ptc_second_pass";
            if (is_mapdiag_detector || is_busy_detector || is_network) {
                records.push_back(record);
            }
        }
    }
    if (records.empty()) {
        return;
    }

    ReductionLearningState::LearnedMaskApplicationSummary summary;
    summary.obsnum = obsnum;
    summary.producer = "learning_state";
    summary.stage = stage;
    summary.iter = fruit_iter;
    summary.scan = scan_id;
    summary.candidate_records = static_cast<int>(records.size());
    const bool has_network_record = std::any_of(
        records.begin(), records.end(),
        [](const auto &record) {
            return record.uid < 0 &&
                   record.reason == "busy_network_pathology";
        });
    summary.max_new_flagged_fraction = has_network_record
        ? reduction_learning.options.scan_network_pathology_max_new_flagged_fraction
        : reduction_learning.options.apply_max_new_flagged_fraction;

    const Eigen::Index n_pts = tcdata.flags.data.rows();
    const Eigen::Index n_dets = tcdata.flags.data.cols();
    std::set<Eigen::Index> proposed_dets;
    std::set<Eigen::Index> network_proposed_dets;
    for (const auto &record : records) {
        if (record.uid >= 0) {
            const Eigen::Index det =
                citlali_learning_find_det_by_uid(calib_scan.apt, record.uid);
            if (det < 0 || det >= n_dets) {
                ++summary.invalid_records;
                continue;
            }
            ++summary.matched_records;
            proposed_dets.insert(det);
        }
        else if (record.nw >= 0) {
            bool matched_network = false;
            for (Eigen::Index det = 0; det < n_dets; ++det) {
                const int det_nw =
                    citlali_learning_apt_int(calib_scan.apt, "nw", det, -1);
                if (det_nw == record.nw) {
                    matched_network = true;
                    proposed_dets.insert(det);
                    network_proposed_dets.insert(det);
                }
            }
            if (matched_network) {
                ++summary.matched_records;
            }
            else {
                ++summary.invalid_records;
            }
        }
    }
    if (proposed_dets.empty()) {
        reduction_learning.record_learned_mask_application(summary);
        return;
    }

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> source_mask;
    bool have_network_source_protection = false;
    if (!network_proposed_dets.empty() &&
        stage == "pre_mapmaking_detector_exclusion") {
        const double radius_arcsec =
            std::max(20.0, ptcproc.second_pass_local.source_protection_radius_arcsec);
        auto [mask, source_info] = engine_utils::calc_source_protection_mask(
            tcdata, calib_scan.apt, telescope.pixel_axes, map_grouping,
            "map_center_radius", radius_arcsec);
        (void) source_info;
        source_mask = std::move(mask);
        have_network_source_protection =
            source_mask.rows() == n_pts && source_mask.cols() == n_dets;
        if (!have_network_source_protection) {
            logger->warn(
                "learned {} source-protection mask shape mismatch scan {}: mask=({}, {}) flags=({}, {})",
                stage, scan_id, source_mask.rows(), source_mask.cols(), n_pts, n_dets);
        }
    }

    for (const auto det : proposed_dets) {
        for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
            if (have_network_source_protection &&
                network_proposed_dets.find(det) != network_proposed_dets.end() &&
                source_mask(sample, det)) {
                ++summary.source_protected_samples;
                continue;
            }
            ++summary.proposed_samples;
            if (tcdata.flags.data(sample, det)) {
                ++summary.already_flagged_samples;
            }
            else {
                ++summary.newly_flagged_samples;
            }
        }
    }

    const double denom = static_cast<double>(std::max<Eigen::Index>(1, n_pts * n_dets));
    summary.newly_flagged_fraction =
        static_cast<double>(summary.newly_flagged_samples) / denom;
    const bool over_cap =
        summary.max_new_flagged_fraction > 0.0 &&
        summary.newly_flagged_fraction >
            summary.max_new_flagged_fraction;
    if (!over_cap) {
        auto flag_it = calib_scan.apt.find("flag");
        std::set<Eigen::Index> apt_flag_dets;
        Eigen::Index apt_flag_preserved = 0;
        if (update_apt_flags &&
            flag_it != calib_scan.apt.end() &&
            flag_it->second.size() > 0) {
            std::map<int, Eigen::Index> unflagged_by_nw;
            std::map<int, Eigen::Index> unflagged_by_array;
            const Eigen::Index n_apt =
                std::min<Eigen::Index>(n_dets, flag_it->second.size());
            for (Eigen::Index det = 0; det < n_apt; ++det) {
                if (flag_it->second(det) != 0.0) {
                    continue;
                }
                const int nw =
                    citlali_learning_apt_int(calib_scan.apt, "nw", det, -1);
                const int array =
                    citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
                if (nw >= 0) {
                    ++unflagged_by_nw[nw];
                }
                if (array >= 0) {
                    ++unflagged_by_array[array];
                }
            }

            for (const auto det : proposed_dets) {
                if (network_proposed_dets.find(det) != network_proposed_dets.end()) {
                    continue;
                }
                if (det < 0 ||
                    det >= n_apt ||
                    flag_it->second(det) != 0.0) {
                    continue;
                }
                const int nw =
                    citlali_learning_apt_int(calib_scan.apt, "nw", det, -1);
                const int array =
                    citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
                const bool preserves_nw =
                    nw < 0 ||
                    unflagged_by_nw.find(nw) == unflagged_by_nw.end() ||
                    unflagged_by_nw[nw] > 1;
                const bool preserves_array =
                    array < 0 ||
                    unflagged_by_array.find(array) == unflagged_by_array.end() ||
                    unflagged_by_array[array] > 1;
                if (!preserves_nw || !preserves_array) {
                    ++apt_flag_preserved;
                    continue;
                }
                apt_flag_dets.insert(det);
                if (nw >= 0) {
                    --unflagged_by_nw[nw];
                }
                if (array >= 0) {
                    --unflagged_by_array[array];
                }
            }
        }

        for (const auto det : proposed_dets) {
            if (have_network_source_protection &&
                network_proposed_dets.find(det) != network_proposed_dets.end()) {
                for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
                    if (!source_mask(sample, det)) {
                        tcdata.flags.data(sample, det) = true;
                    }
                }
            }
            else {
                tcdata.flags.data.col(det).setOnes();
            }
            if (apt_flag_dets.find(det) != apt_flag_dets.end()) {
                flag_it->second(det) = 1.0;
            }
        }
        summary.applied = true;
        if (apt_flag_preserved > 0) {
            logger->info(
                "learned {} preserved {} scan-local APT flags in scan {} iter {} to keep nw/array groups valid",
                stage, apt_flag_preserved, scan_id + 1, fruit_iter);
        }
    }

    reduction_learning.record_learned_mask_application(summary);
    if (over_cap) {
        logger->warn(
            "learned {} rejected scan {} iter {}: candidates={} matched={} dets={} newly_flagged={} newly_flagged_fraction={:.4f} cap={:.4f}",
            stage, scan_id + 1, fruit_iter, summary.candidate_records,
            summary.matched_records, proposed_dets.size(),
            summary.newly_flagged_samples, summary.newly_flagged_fraction,
            summary.max_new_flagged_fraction);
    }
    else {
        logger->info(
            "learned {} applied scan {} iter {}: candidates={} matched={} dets={} newly_flagged={} already_flagged={} newly_flagged_fraction={:.4f}",
            stage, scan_id + 1, fruit_iter, summary.candidate_records,
            summary.matched_records, proposed_dets.size(),
            summary.newly_flagged_samples, summary.already_flagged_samples,
            summary.newly_flagged_fraction);
    }
}

template <class tc_t, class calib_t>
void Engine::apply_learned_sample_masks(tc_t &tcdata, calib_t &calib_scan,
                                        bool apply_pre_rtc,
                                        const std::string &stage,
                                        bool source_protection_enabled,
                                        double source_protection_radius_arcsec) {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.options.apply_sample_masks_enabled ||
        !reduction_learning.apply_active()) {
        return;
    }
    if (tcdata.flags.data.rows() <= 0 || tcdata.flags.data.cols() <= 0) {
        return;
    }

    const int scan_id = static_cast<int>(tcdata.index.data);
    std::vector<ReductionLearningState::LearnedSampleMask> records;
    {
        std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
        for (const auto &record : reduction_learning.learned_sample_masks) {
            if (record.obsnum == obsnum &&
                record.scan == scan_id &&
                record.iter >= 0 &&
                record.iter < fruit_iter &&
                record.apply_pre_rtc == apply_pre_rtc) {
                records.push_back(record);
            }
        }
    }
    if (records.empty()) {
        return;
    }

    ReductionLearningState::LearnedMaskApplicationSummary summary;
    summary.obsnum = obsnum;
    summary.producer = "learning_state";
    summary.stage = stage;
    summary.iter = fruit_iter;
    summary.scan = scan_id;
    summary.candidate_records = static_cast<int>(records.size());
    summary.max_new_flagged_fraction =
        reduction_learning.options.apply_max_new_flagged_fraction;

    const Eigen::Index n_pts = tcdata.flags.data.rows();
    const Eigen::Index n_dets = tcdata.flags.data.cols();
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> proposed(n_pts, n_dets);
    proposed.setZero();

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> source_mask;
    bool have_source_protection = false;
    if (source_protection_enabled && source_protection_radius_arcsec > 0.0) {
        auto [mask, source_info] = engine_utils::calc_source_protection_mask(
            tcdata, calib_scan.apt, telescope.pixel_axes, map_grouping,
            "map_center_radius", source_protection_radius_arcsec);
        (void) source_info;
        source_mask = std::move(mask);
        have_source_protection =
            source_mask.rows() == n_pts && source_mask.cols() == n_dets;
        if (source_protection_enabled && !have_source_protection) {
            logger->warn(
                "learned mask {} source-protection mask shape mismatch scan {}: mask=({}, {}) flags=({}, {})",
                stage, scan_id, source_mask.rows(), source_mask.cols(), n_pts, n_dets);
        }
    }

    for (const auto &record : records) {
        if (record.source_protected) {
            ++summary.invalid_records;
            continue;
        }
        const Eigen::Index det = citlali_learning_find_det_by_uid(calib_scan.apt, record.uid);
        const long long raw_start = apply_pre_rtc ? record.raw_start : record.ptc_start;
        const long long raw_stop = apply_pre_rtc ? record.raw_stop : record.ptc_stop;
        if (det < 0 || det >= n_dets || raw_start < 0 || raw_stop < raw_start ||
            raw_stop < 0 || raw_start >= n_pts) {
            ++summary.invalid_records;
            continue;
        }
        const Eigen::Index start =
            std::max<Eigen::Index>(0, static_cast<Eigen::Index>(raw_start));
        const Eigen::Index stop =
            std::min<Eigen::Index>(n_pts - 1, static_cast<Eigen::Index>(raw_stop));
        if (stop < start) {
            ++summary.invalid_records;
            continue;
        }

        ++summary.matched_records;
        for (Eigen::Index sample = start; sample <= stop; ++sample) {
            if (have_source_protection && source_mask(sample, det)) {
                ++summary.source_protected_samples;
                continue;
            }
            if (!proposed(sample, det)) {
                proposed(sample, det) = true;
                ++summary.proposed_samples;
            }
        }
    }

    if (summary.proposed_samples > 0) {
        for (Eigen::Index det = 0; det < n_dets; ++det) {
            for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
                if (!proposed(sample, det)) {
                    continue;
                }
                if (tcdata.flags.data(sample, det)) {
                    ++summary.already_flagged_samples;
                }
                else {
                    ++summary.newly_flagged_samples;
                }
            }
        }
    }

    const double denom = static_cast<double>(std::max<Eigen::Index>(1, n_pts * n_dets));
    summary.newly_flagged_fraction =
        static_cast<double>(summary.newly_flagged_samples) / denom;
    const bool over_cap =
        reduction_learning.options.apply_max_new_flagged_fraction > 0.0 &&
        summary.newly_flagged_fraction >
            reduction_learning.options.apply_max_new_flagged_fraction;
    if (!over_cap) {
        for (Eigen::Index det = 0; det < n_dets; ++det) {
            for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
                if (proposed(sample, det)) {
                    tcdata.flags.data(sample, det) = true;
                }
            }
        }
        summary.applied = true;
    }

    reduction_learning.record_learned_mask_application(summary);
    if (over_cap) {
        logger->warn(
            "learned {} sample-mask application rejected scan {} iter {}: candidates={} matched={} proposed={} newly_flagged={} newly_flagged_fraction={:.4f} cap={:.4f}",
            stage, scan_id + 1, fruit_iter, summary.candidate_records,
            summary.matched_records, summary.proposed_samples,
            summary.newly_flagged_samples, summary.newly_flagged_fraction,
            reduction_learning.options.apply_max_new_flagged_fraction);
    }
    else if (summary.proposed_samples > 0) {
        logger->info(
            "learned {} sample masks applied scan {} iter {}: candidates={} matched={} proposed={} newly_flagged={} already_flagged={} source_protected={} newly_flagged_fraction={:.4f}",
            stage, scan_id + 1, fruit_iter, summary.candidate_records,
            summary.matched_records, summary.proposed_samples,
            summary.newly_flagged_samples, summary.already_flagged_samples,
            summary.source_protected_samples, summary.newly_flagged_fraction);
    }
}

template <class rtc_t, class ptc_t, class calib_t>
void Engine::collect_rtc_learning_diagnostics(rtc_t &rtcdata, ptc_t &ptcdata,
                                              calib_t &calib_scan,
                                              const std::vector<timestream::RTCProc::RTCDetectorDiagSummary> &det_summary) {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled()) {
        return;
    }

    const auto scan_id = ptcdata.index.data;
    if (det_summary.empty()) {
        return;
    }

    const auto rtc_source_summary =
        rtcproc.snapshot_source_protection_diag_summary(scan_id);
    if (rtc_source_summary.enabled) {
        ReductionLearningState::SourceProtectionSummary source_summary;
        source_summary.obsnum = obsnum;
        source_summary.producer = "rtc_despike";
        source_summary.mode = "map_center_radius";
        source_summary.iter = fruit_iter;
        source_summary.scan = static_cast<int>(scan_id);
        source_summary.protected_samples = rtc_source_summary.protected_samples;
        source_summary.total_samples = rtc_source_summary.total_samples;
        source_summary.radius_arcsec = rtc_source_summary.radius_arcsec;
        reduction_learning.record_source_protection_summary(std::move(source_summary));
    }

    auto record_event = [&](const auto &event, Eigen::Index det,
                            const std::string &reason) {
        const auto uid_it = calib_scan.apt.find("uid");
        if (!event.valid() || !event.accepted || uid_it == calib_scan.apt.end() ||
            det < 0 || det >= uid_it->second.size()) {
            return;
        }
        ReductionLearningState::LearnedSampleMask record;
        record.obsnum = obsnum;
        record.producer = "rtc_despike";
        record.reason = reason;
        record.iter = fruit_iter;
        record.scan = static_cast<int>(scan_id);
        record.uid = citlali_learning_apt_int(calib_scan.apt, "uid", det,
                                              static_cast<int>(det));
        record.nw = citlali_learning_apt_int(calib_scan.apt, "nw", det, -1);
        record.array = citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
        record.raw_start = event.start_sample;
        record.raw_stop = event.end_sample;
        record.score = event.score;
        record.z = event.score;
        record.confidence = 1.0;
        record.source_protected = false;
        record.apply_pre_rtc = true;
        reduction_learning.record_learned_sample_mask(std::move(record));
    };

    for (const auto &row : det_summary) {
        const Eigen::Index det = row.det;
        record_event(row.local_raw_event, det, "local_raw_accepted");
        record_event(row.local_delta_event, det, "local_delta_accepted");
    }
}

template <class ptc_t, class calib_t>
void Engine::collect_ptc_learning_diagnostics(
    ptc_t &ptcdata, calib_t &calib_scan,
    const std::vector<timestream::PTCProc::SecondPassDiagSummary> &second_pass_summary,
    const std::vector<timestream::PTCProc::HighWeightDiagSummary> &high_weight_summary) {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled()) {
        return;
    }

    const auto scan_id = ptcdata.index.data;

    if (ptcproc.second_pass_local.source_protection_enabled) {
        ReductionLearningState::SourceProtectionSummary source_summary;
        source_summary.obsnum = obsnum;
        source_summary.producer = "ptc_second_pass";
        source_summary.mode = "map_center_radius";
        source_summary.iter = fruit_iter;
        source_summary.scan = static_cast<int>(scan_id);
        source_summary.total_samples =
            static_cast<int>(ptcdata.scans.data.rows() * ptcdata.scans.data.cols());
        source_summary.radius_arcsec =
            ptcproc.second_pass_local.source_protection_radius_arcsec;
        auto [source_mask, source_info] = engine_utils::calc_source_protection_mask(
            ptcdata, calib_scan.apt, telescope.pixel_axes, map_grouping,
            "map_center_radius",
            ptcproc.second_pass_local.source_protection_radius_arcsec);
        (void) source_mask;
        source_summary.protected_samples =
            static_cast<int>(source_info.protected_samples);
        reduction_learning.record_source_protection_summary(std::move(source_summary));
    }

    for (const auto &summary : high_weight_summary) {
        ReductionLearningState::HighWeightDetector record;
        record.obsnum = obsnum;
        record.grouping = summary.grouping;
        record.reason = summary.reason;
        record.iter = fruit_iter;
        record.scan = static_cast<int>(scan_id);
        record.uid = summary.uid;
        record.nw = static_cast<int>(summary.nw);
        record.array = static_cast<int>(summary.array);
        record.weight = summary.approximate_weight;
        record.final_weight = summary.final_weight;
        record.group_median = summary.group_median_weight;
        record.robust_z = summary.robust_z;
        record.cap = summary.applied_cap;
        record.validation_factor = summary.validation_factor;
        record.cap_recommended = summary.cap_recommended;
        record.cap_applied = summary.cap_applied;
        record.validated = summary.validated;
        reduction_learning.record_high_weight_detector(std::move(record));
    }

    if (second_pass_summary.empty()) {
        return;
    }

    for (const auto &summary : second_pass_summary) {
        const bool has_candidate = summary.n_candidate_clusters > 0 ||
                                   summary.n_candidate_events > 0;
        const bool has_residual =
            std::isfinite(summary.max_unflagged_residual_z) &&
            summary.max_unflagged_residual_uid != timestream::kTransientFillInt;
        const bool selective_acceptance_recommended =
            summary.busy_network_vetoed &&
            ((std::isfinite(summary.top_candidate_cluster_peak_score) &&
              summary.top_candidate_cluster_peak_score >=
                  ptcproc.second_pass_local.high_score_cluster_override) ||
             (std::isfinite(summary.max_unflagged_residual_z) &&
              summary.max_unflagged_residual_z >=
                  ptcproc.second_pass_local.high_score_event_override));
        if (has_candidate || has_residual || summary.busy_network_vetoed) {
            ReductionLearningState::BusyNetworkSummary record;
            record.obsnum = obsnum;
            record.producer = "ptc_second_pass";
            record.reason = summary.busy_network_vetoed
                ? "busy_network_vetoed"
                : "candidate_or_residual";
            record.iter = fruit_iter;
            record.scan = static_cast<int>(scan_id);
            record.nw = static_cast<int>(summary.nw);
            record.n_candidate_clusters =
                static_cast<int>(summary.n_candidate_clusters);
            record.n_candidate_events =
                static_cast<int>(summary.n_candidate_events);
            record.n_accepted_clusters =
                static_cast<int>(summary.n_accepted_clusters);
            record.n_accepted_events =
                static_cast<int>(summary.n_accepted_events);
            record.n_rejected_clusters =
                static_cast<int>(summary.n_rejected_clusters);
            record.n_rejected_events =
                static_cast<int>(summary.n_rejected_events);
            record.n_source_protected_clusters =
                static_cast<int>(summary.n_source_protected_clusters);
            record.n_source_protected_events =
                static_cast<int>(summary.n_source_protected_events);
            record.max_unflagged_residual_uid = summary.max_unflagged_residual_uid;
            record.top_candidate_sample = summary.top_candidate_cluster_sample;
            record.top_candidate_score = summary.top_candidate_cluster_peak_score;
            record.max_unflagged_residual_z = summary.max_unflagged_residual_z;
            record.busy_vetoed = summary.busy_network_vetoed;
            record.selective_acceptance_recommended = selective_acceptance_recommended;
            reduction_learning.record_busy_network_summary(std::move(record));
        }

        if (reduction_learning.options.scan_network_pathology_enabled &&
            summary.nw >= 0) {
            const int off_source_candidate_events = std::max<Eigen::Index>(
                0, summary.n_candidate_events - summary.n_source_protected_events);
            const double max_residual_z = std::isfinite(summary.max_unflagged_residual_z)
                ? summary.max_unflagged_residual_z
                : 0.0;
            const bool busy_pathology =
                summary.busy_network_vetoed &&
                summary.n_candidate_clusters >=
                    reduction_learning.options.scan_network_pathology_min_candidate_clusters &&
                off_source_candidate_events >=
                    reduction_learning.options.scan_network_pathology_min_candidate_events &&
                max_residual_z >=
                    reduction_learning.options.scan_network_pathology_min_max_residual_z;
            const bool severe_pathology =
                off_source_candidate_events >=
                    reduction_learning.options.scan_network_pathology_severe_candidate_events &&
                max_residual_z >=
                    reduction_learning.options.scan_network_pathology_severe_max_residual_z;
            if (busy_pathology || severe_pathology) {
                ReductionLearningState::DetectorPenalty penalty;
                penalty.obsnum = obsnum;
                penalty.producer = "ptc_second_pass";
                penalty.reason = "busy_network_pathology";
                penalty.iter = fruit_iter;
                penalty.scan = static_cast<int>(scan_id);
                penalty.uid = -1;
                penalty.nw = static_cast<int>(summary.nw);
                penalty.array = citlali_learning_array_for_nw(
                    calib_scan.apt, penalty.nw, -1);
                penalty.factor = 0.0;
                penalty.score = std::max(
                    max_residual_z,
                    std::isfinite(summary.top_candidate_cluster_peak_score)
                        ? summary.top_candidate_cluster_peak_score
                        : 0.0);
                penalty.scan_local = true;
                reduction_learning.record_detector_penalty(std::move(penalty));
            }
        }

        for (const auto &event : summary.candidate_events) {
            if (event.uid == timestream::kTransientFillInt ||
                event.start_sample < 0 ||
                event.end_sample < event.start_sample) {
                continue;
            }
            if (!event.accepted || event.source_protected) {
                continue;
            }
            const Eigen::Index det =
                citlali_learning_find_det_by_uid(calib_scan.apt, event.uid);
            ReductionLearningState::LearnedSampleMask candidate_record;
            candidate_record.obsnum = obsnum;
            candidate_record.producer = "ptc_second_pass";
            candidate_record.reason = event.busy_network_vetoed
                ? "busy_selective_accepted_event"
                : "candidate_event";
            candidate_record.iter = fruit_iter;
            candidate_record.scan = static_cast<int>(scan_id);
            candidate_record.uid = event.uid;
            candidate_record.nw = static_cast<int>(summary.nw);
            candidate_record.array =
                citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
            candidate_record.ptc_start = event.start_sample;
            candidate_record.ptc_stop = event.end_sample;
            candidate_record.score = event.score;
            candidate_record.z = event.score;
            candidate_record.value = event.cluster_score;
            candidate_record.confidence = event.busy_network_vetoed ? 0.8 : 1.0;
            candidate_record.source_protected = event.source_protected;
            candidate_record.apply_pre_rtc = false;
            reduction_learning.record_learned_sample_mask(std::move(candidate_record));
        }

        if (summary.top_event.valid() && summary.top_event.accepted &&
            summary.top_event_uid != timestream::kTransientFillInt) {
            const Eigen::Index det =
                citlali_learning_find_det_by_uid(calib_scan.apt, summary.top_event_uid);
            ReductionLearningState::LearnedSampleMask sample_record;
            sample_record.obsnum = obsnum;
            sample_record.producer = "ptc_second_pass";
            sample_record.reason = "accepted_event";
            sample_record.iter = fruit_iter;
            sample_record.scan = static_cast<int>(scan_id);
            sample_record.uid = summary.top_event_uid;
            sample_record.nw = static_cast<int>(summary.nw);
            sample_record.array =
                citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
            sample_record.ptc_start = summary.top_event.start_sample;
            sample_record.ptc_stop = summary.top_event.end_sample;
            sample_record.score = summary.top_event.score;
            sample_record.z = summary.top_event.score;
            sample_record.confidence = 1.0;
            sample_record.source_protected = false;
            sample_record.apply_pre_rtc = false;
            reduction_learning.record_learned_sample_mask(std::move(sample_record));
        }

        if (summary.busy_network_vetoed && has_residual &&
            summary.max_unflagged_residual_z >=
                ptcproc.second_pass_local.high_score_event_override) {
            const Eigen::Index det = citlali_learning_find_det_by_uid(
                calib_scan.apt, summary.max_unflagged_residual_uid);
            ReductionLearningState::DetectorPenalty penalty;
            penalty.obsnum = obsnum;
            penalty.producer = "ptc_second_pass";
            penalty.reason = "busy_vetoed_residual";
            penalty.iter = fruit_iter;
            penalty.scan = static_cast<int>(scan_id);
            penalty.uid = summary.max_unflagged_residual_uid;
            penalty.nw = static_cast<int>(summary.nw);
            penalty.array =
                citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
            penalty.factor = 0.0;
            penalty.score = summary.max_unflagged_residual_z;
            penalty.scan_local = true;
            reduction_learning.record_detector_penalty(std::move(penalty));
        }
    }
}

inline void Engine::write_learning_summary() {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled() ||
        redu_dir_name.empty()) {
        return;
    }

    std::ostringstream filename;
    filename << redu_dir_name << "/learning_iter_" << fruit_iter << ".csv";
    std::ofstream out(filename.str());
    if (!out) {
        logger->warn("failed to open learning summary output {}", filename.str());
        return;
    }

    auto csv = [](const std::string &s) {
        std::string escaped = "\"";
        for (const char ch : s) {
            if (ch == '"') {
                escaped += "\"\"";
            }
            else {
                escaped += ch;
            }
        }
        escaped += "\"";
        return escaped;
    };

    enum {
        ColRecordType,
        ColIter,
        ColObsnum,
        ColProducer,
        ColReason,
        ColScan,
        ColUid,
        ColNw,
        ColArray,
        ColRawStart,
        ColRawStop,
        ColPtcStart,
        ColPtcStop,
        ColScore,
        ColZ,
        ColValue,
        ColConfidence,
        ColSourceDistanceArcsec,
        ColSourceProtected,
        ColApplyPreRtc,
        ColCandidateClusters,
        ColCandidateEvents,
        ColAcceptedClusters,
        ColAcceptedEvents,
        ColRejectedClusters,
        ColRejectedEvents,
        ColSourceProtectedClusters,
        ColSourceProtectedEvents,
        ColMaxResidualUid,
        ColTopCandidateSample,
        ColTopCandidateScore,
        ColMaxResidualZ,
        ColBusyVetoed,
        ColSelectiveAcceptanceRecommended,
        ColFactor,
        ColScanLocal,
        ColProtectedSamples,
        ColTotalSamples,
        ColRadiusArcsec,
        ColSupportNpix,
        ColApplicationStage,
        ColCandidateRecords,
        ColMatchedRecords,
        ColInvalidRecords,
        ColProposedSamples,
        ColNewlyFlaggedSamples,
        ColAlreadyFlaggedSamples,
        ColSourceProtectedSamples,
        ColNewlyFlaggedFraction,
        ColMaxNewFlaggedFraction,
        ColApplied,
        ColGrouping,
        ColWeight,
        ColFinalWeight,
        ColGroupMedian,
        ColRobustZ,
        ColCap,
        ColValidationFactor,
        ColCapRecommended,
        ColCapApplied,
        ColValidated,
        ColMapIndex,
        ColRow,
        ColCol,
        ColSample,
        ColNEff,
        ColLeaveOneOutZ,
        ColCount
    };

    const std::vector<std::string> header = {
        "record_type", "iter", "obsnum", "producer", "reason", "scan", "uid",
        "nw", "array", "raw_start", "raw_stop", "ptc_start", "ptc_stop",
        "score", "z", "value", "confidence", "source_distance_arcsec",
        "source_protected", "apply_pre_rtc", "n_candidate_clusters",
        "n_candidate_events", "n_accepted_clusters", "n_accepted_events",
        "n_rejected_clusters", "n_rejected_events",
        "n_source_protected_clusters", "n_source_protected_events",
        "max_unflagged_residual_uid", "top_candidate_sample",
        "top_candidate_score", "max_unflagged_residual_z", "busy_vetoed",
        "selective_acceptance_recommended", "factor", "scan_local",
        "protected_samples", "total_samples", "radius_arcsec", "support_npix",
        "application_stage", "candidate_records", "matched_records",
        "invalid_records", "proposed_samples", "newly_flagged_samples",
        "already_flagged_samples", "source_protected_samples",
        "newly_flagged_fraction", "max_new_flagged_fraction", "applied",
        "grouping", "weight", "final_weight", "group_median", "robust_z",
        "cap", "validation_factor", "cap_recommended", "cap_applied",
        "validated", "map_index", "row", "col", "sample", "n_eff",
        "leave_one_out_z"
    };

    auto text = [](const auto &value) {
        std::ostringstream stream;
        stream << value;
        return stream.str();
    };

    auto write_row = [&](const std::vector<std::string> &row) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            if (i > 0) {
                out << ',';
            }
            out << row[i];
        }
        out << '\n';
    };

    auto new_row = [&]() {
        return std::vector<std::string>(ColCount);
    };

    auto write_common_header = [&]() {
        write_row(header);
    };

    auto write_base = [&](std::vector<std::string> &row,
                          const std::string &record_type, int iter,
                          const std::string &obsnum_value,
                          const std::string &producer,
                          const std::string &reason, int scan, int uid,
                          int nw, int array) {
        row[ColRecordType] = csv(record_type);
        row[ColIter] = text(iter);
        row[ColObsnum] = csv(obsnum_value);
        row[ColProducer] = csv(producer);
        row[ColReason] = csv(reason);
        row[ColScan] = text(scan);
        row[ColUid] = text(uid);
        row[ColNw] = text(nw);
        row[ColArray] = text(array);
    };

    std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
    write_common_header();

    for (const auto &record : reduction_learning.learned_sample_masks) {
        auto row = new_row();
        write_base(row, "sample_mask", record.iter, record.obsnum, record.producer,
                   record.reason, record.scan, record.uid, record.nw, record.array);
        row[ColRawStart] = text(record.raw_start);
        row[ColRawStop] = text(record.raw_stop);
        row[ColPtcStart] = text(record.ptc_start);
        row[ColPtcStop] = text(record.ptc_stop);
        row[ColScore] = text(record.score);
        row[ColZ] = text(record.z);
        row[ColValue] = text(record.value);
        row[ColConfidence] = text(record.confidence);
        row[ColSourceDistanceArcsec] = text(record.source_distance_arcsec);
        row[ColSourceProtected] = text(record.source_protected ? 1 : 0);
        row[ColApplyPreRtc] = text(record.apply_pre_rtc ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.busy_network_summaries) {
        auto row = new_row();
        write_base(row, "busy_network", record.iter, record.obsnum, record.producer,
                   record.reason, record.scan, -1, record.nw, -1);
        row[ColScore] = text(record.top_candidate_score);
        row[ColZ] = text(record.max_unflagged_residual_z);
        row[ColCandidateClusters] = text(record.n_candidate_clusters);
        row[ColCandidateEvents] = text(record.n_candidate_events);
        row[ColAcceptedClusters] = text(record.n_accepted_clusters);
        row[ColAcceptedEvents] = text(record.n_accepted_events);
        row[ColRejectedClusters] = text(record.n_rejected_clusters);
        row[ColRejectedEvents] = text(record.n_rejected_events);
        row[ColSourceProtectedClusters] = text(record.n_source_protected_clusters);
        row[ColSourceProtectedEvents] = text(record.n_source_protected_events);
        row[ColMaxResidualUid] = text(record.max_unflagged_residual_uid);
        row[ColTopCandidateSample] = text(record.top_candidate_sample);
        row[ColTopCandidateScore] = text(record.top_candidate_score);
        row[ColMaxResidualZ] = text(record.max_unflagged_residual_z);
        row[ColBusyVetoed] = text(record.busy_vetoed ? 1 : 0);
        row[ColSelectiveAcceptanceRecommended] =
            text(record.selective_acceptance_recommended ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.detector_penalties) {
        auto row = new_row();
        write_base(row, "detector_penalty", record.iter, record.obsnum,
                   record.producer, record.reason, record.scan, record.uid,
                   record.nw, record.array);
        row[ColScore] = text(record.score);
        row[ColZ] = text(record.score);
        row[ColFactor] = text(record.factor);
        row[ColScanLocal] = text(record.scan_local ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.high_weight_detectors) {
        auto row = new_row();
        write_base(row, "high_weight_detector", record.iter, record.obsnum,
                   "weight_validation", record.reason, record.scan, record.uid,
                   record.nw, record.array);
        row[ColScore] = text(record.robust_z);
        row[ColZ] = text(record.robust_z);
        row[ColValue] = text(record.weight);
        row[ColFactor] = text(record.validation_factor);
        row[ColGrouping] = csv(record.grouping);
        row[ColWeight] = text(record.weight);
        row[ColFinalWeight] = text(record.final_weight);
        row[ColGroupMedian] = text(record.group_median);
        row[ColRobustZ] = text(record.robust_z);
        row[ColCap] = text(record.cap);
        row[ColValidationFactor] = text(record.validation_factor);
        row[ColCapRecommended] = text(record.cap_recommended ? 1 : 0);
        row[ColCapApplied] = text(record.cap_applied ? 1 : 0);
        row[ColValidated] = text(record.validated ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.map_pixel_outliers) {
        auto row = new_row();
        write_base(row, "map_pixel_outlier", record.iter, record.obsnum,
                   record.producer, record.reason, record.scan, record.uid,
                   -1, -1);
        row[ColScore] = text(record.leave_one_out_z);
        row[ColZ] = text(record.leave_one_out_z);
        row[ColValue] = text(record.value);
        row[ColWeight] = text(record.weight);
        row[ColMapIndex] = text(record.map_index);
        row[ColRow] = text(record.row);
        row[ColCol] = text(record.col);
        row[ColSample] = text(record.sample);
        row[ColNEff] = text(record.n_eff);
        row[ColLeaveOneOutZ] = text(record.leave_one_out_z);
        row[ColSourceDistanceArcsec] = text(record.source_distance_arcsec);
        row[ColSourceProtected] = text(record.source_protected ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.source_protection_summaries) {
        auto row = new_row();
        write_base(row, "source_protection", record.iter, record.obsnum,
                   record.producer, record.mode, record.scan, -1, -1, -1);
        row[ColSourceProtected] = text(1);
        row[ColApplyPreRtc] = text(0);
        row[ColProtectedSamples] = text(record.protected_samples);
        row[ColTotalSamples] = text(record.total_samples);
        row[ColRadiusArcsec] = text(record.radius_arcsec);
        row[ColSupportNpix] = text(record.support_npix);
        write_row(row);
    }

    for (const auto &record : reduction_learning.learned_mask_applications) {
        auto row = new_row();
        const bool detector_exclusion =
            record.stage.find("detector_exclusion") != std::string::npos;
        write_base(row,
                   detector_exclusion
                       ? "detector_penalty_application"
                       : "sample_mask_application",
                   record.iter, record.obsnum, record.producer,
                   detector_exclusion
                       ? "apply_learned_detector_exclusion"
                       : "apply_learned_sample_mask",
                   record.scan, -1, -1, -1);
        row[ColApplicationStage] = csv(record.stage);
        row[ColCandidateRecords] = text(record.candidate_records);
        row[ColMatchedRecords] = text(record.matched_records);
        row[ColInvalidRecords] = text(record.invalid_records);
        row[ColProposedSamples] = text(record.proposed_samples);
        row[ColNewlyFlaggedSamples] = text(record.newly_flagged_samples);
        row[ColAlreadyFlaggedSamples] = text(record.already_flagged_samples);
        row[ColSourceProtectedSamples] = text(record.source_protected_samples);
        row[ColNewlyFlaggedFraction] = text(record.newly_flagged_fraction);
        row[ColMaxNewFlaggedFraction] = text(record.max_new_flagged_fraction);
        row[ColApplied] = text(record.applied ? 1 : 0);
        write_row(row);
    }

    logger->info("wrote reduction learning summary {}", filename.str());
}

template<typename CT>
void Engine::get_timestream_config(CT &config) {
    logger->info("getting timestream config options");
    typed_timestream_config = citlali::config::TimestreamConfig{};

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return missing_keys.size() == missing_before && invalid_keys.size() == invalid_before;
    };

    // run tod processing
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_tod, missing_keys, invalid_keys,
                         std::tuple{"timestream","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.enabled = run_tod;
        }
    }
    if (!run_tod) {
        logger->error("timestream.enabled is false. This reduction requires TOD processing; set "
                      "low_level.timestream.enabled: true in your reduce config.");
        std::exit(EXIT_FAILURE);
    }
    // tod type (xs, rs, is, qs)
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, tod_type, missing_keys, invalid_keys,
                         std::tuple{"timestream","type"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_tod_type(tod_type)) {
                typed_timestream_config.type = *parsed;
            }
        }
    }

    // run rtc or ptc tod output?
    // output rtc
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_tod_output_rtc, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","output","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.output.raw_time_chunk_enabled = run_tod_output_rtc;
            typed_timestream_config.output.raw_time_chunk.enabled = run_tod_output_rtc;
        }
    }
    rtcproc.tod_output_mini = false;
    rtcproc.tod_output_outer = false;
    rtcproc.tod_output_outer_context_samples = 0;
    std::string rtc_output_mode = "full";
    if (run_tod_output_rtc && config.has(std::tuple{"timestream","raw_time_chunk","output","mode"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, rtc_output_mode, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","output","mode"},
                         {"full","mini","full_outer","mini_outer"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_tod_stream_output_mode(rtc_output_mode)) {
                typed_timestream_config.output.raw_time_chunk.mode = *parsed;
            }
        }
        rtcproc.tod_output_mini = (rtc_output_mode == "mini" || rtc_output_mode == "mini_outer");
        rtcproc.tod_output_outer = (rtc_output_mode == "full_outer" || rtc_output_mode == "mini_outer");
    }
    if (run_tod_output_rtc && config.has(std::tuple{"timestream","raw_time_chunk","output","outer_context_samples"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, rtcproc.tod_output_outer_context_samples, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","output","outer_context_samples"},
                         {}, {0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.output.raw_time_chunk.outer_context_samples =
                static_cast<int>(rtcproc.tod_output_outer_context_samples);
        }
    }
    // output ptc
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_tod_output_ptc, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","output","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.output.processed_time_chunk_enabled = run_tod_output_ptc;
            typed_timestream_config.output.processed_time_chunk.enabled = run_tod_output_ptc;
        }
    }
    ptcproc.tod_output_mini = false;
    ptcproc.tod_output_outer = false;
    ptcproc.tod_output_outer_context_samples = 0;
    std::string ptc_output_mode = "full";
    if (run_tod_output_ptc && config.has(std::tuple{"timestream","processed_time_chunk","output","mode"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, ptc_output_mode, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","output","mode"}, {"full","mini"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_tod_stream_output_mode(ptc_output_mode)) {
                typed_timestream_config.output.processed_time_chunk.mode = *parsed;
            }
        }
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
    if (run_tod_output) {
        if (auto parsed = citlali::config::parse_tod_output_type(tod_output_type)) {
            typed_timestream_config.output.type = *parsed;
        }
    }

    // tod subdirectory name
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, tod_output_subdir_name, missing_keys, invalid_keys,
                         std::tuple{"timestream","output", "subdir_name"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.output.subdir_name = tod_output_subdir_name;
        }
    }
    // write eigenvalues to stats file
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, diagnostics.write_evals, missing_keys, invalid_keys,
                         std::tuple{"timestream","output", "stats","eigenvalues"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.output.write_eigenvalues = diagnostics.write_evals;
        }
    }

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

    auto read_tod_selection_count = [&](const auto &key, const std::string &config_path,
                                        int &value) {
        if (!config.template has_typed<int>(key)) {
            return;
        }
        value = config.template get_typed<int>(key);
        if (value < 0) {
            logger->error("{} must be non-negative. Found {}", config_path, value);
            std::exit(EXIT_FAILURE);
        }
    };

    auto parse_tod_selection_mode = [&](const auto &mode_key,
                                        const auto &n_uniform_key,
                                        const auto &n_source_dense_key,
                                        bool output_enabled,
                                        const std::string &mode_path,
                                        const std::string &n_uniform_path,
                                        const std::string &n_source_dense_path,
                                        std::string &mode,
                                        int &n_uniform,
                                        int &n_source_dense) {
        mode = "indices";
        n_uniform = 10;
        n_source_dense = 10;
        if (!output_enabled) {
            return;
        }
        if (config.has(mode_key)) {
            get_config_value(config, mode, missing_keys, invalid_keys, mode_key,
                             {"indices", "all", "uniform_plus_source_crossing"});
        }
        read_tod_selection_count(n_uniform_key, n_uniform_path, n_uniform);
        read_tod_selection_count(n_source_dense_key, n_source_dense_path, n_source_dense);
        if (mode == "uniform_plus_source_crossing" && n_uniform + n_source_dense <= 0) {
            logger->error("{} selects uniform_plus_source_crossing but {} + {} is zero",
                          mode_path, n_uniform_path, n_source_dense_path);
            std::exit(EXIT_FAILURE);
        }
    };

    parse_tod_selection_mode(
        std::tuple{"timestream","raw_time_chunk","output","selection","mode"},
        std::tuple{"timestream","raw_time_chunk","output","selection","n_uniform"},
        std::tuple{"timestream","raw_time_chunk","output","selection","n_source_dense"},
        run_tod_output_rtc,
        "timestream.raw_time_chunk.output.selection.mode",
        "timestream.raw_time_chunk.output.selection.n_uniform",
        "timestream.raw_time_chunk.output.selection.n_source_dense",
        tod_output_selection_mode_rtc,
        tod_output_uniform_count_rtc,
        tod_output_source_dense_count_rtc);
    parse_tod_selection_mode(
        std::tuple{"timestream","processed_time_chunk","output","selection","mode"},
        std::tuple{"timestream","processed_time_chunk","output","selection","n_uniform"},
        std::tuple{"timestream","processed_time_chunk","output","selection","n_source_dense"},
        run_tod_output_ptc,
        "timestream.processed_time_chunk.output.selection.mode",
        "timestream.processed_time_chunk.output.selection.n_uniform",
        "timestream.processed_time_chunk.output.selection.n_source_dense",
        tod_output_selection_mode_ptc,
        tod_output_uniform_count_ptc,
        tod_output_source_dense_count_ptc);

    auto mirror_tod_output_selection = [](const std::vector<Eigen::Index> &chunks_1based,
                                          bool chunk_select_enabled,
                                          const std::string &selection_mode,
                                          int n_uniform,
                                          int n_source_dense,
                                          citlali::config::TodStreamOutputConfig &target) {
        target.chunk_select_enabled = chunk_select_enabled;
        target.chunks_1based.clear();
        target.chunks_1based.reserve(chunks_1based.size());
        for (const auto chunk : chunks_1based) {
            target.chunks_1based.push_back(static_cast<int>(chunk));
        }
        if (auto parsed = citlali::config::parse_tod_output_selection_mode(selection_mode)) {
            target.selection_mode = *parsed;
        }
        target.selection_n_uniform = n_uniform;
        target.selection_n_source_dense = n_source_dense;
    };

    mirror_tod_output_selection(rtc_output_chunks, rtc_chunk_select_enabled,
                                tod_output_selection_mode_rtc,
                                tod_output_uniform_count_rtc,
                                tod_output_source_dense_count_rtc,
                                typed_timestream_config.output.raw_time_chunk);
    mirror_tod_output_selection(ptc_output_chunks, ptc_chunk_select_enabled,
                                tod_output_selection_mode_ptc,
                                tod_output_uniform_count_ptc,
                                tod_output_source_dense_count_ptc,
                                typed_timestream_config.output.processed_time_chunk);

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
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, telescope.chunk_mode, missing_keys, invalid_keys,
                         std::tuple{"timestream","chunking", "chunk_mode"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.chunking.mode = telescope.chunk_mode;
        }
    }
    // get time chunk size
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, telescope.chunking_value, missing_keys, invalid_keys,
                         std::tuple{"timestream","chunking", "value"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.chunking.value = telescope.chunking_value;
        }
    }
    // force chunking?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, telescope.force_chunk, missing_keys, invalid_keys,
                         std::tuple{"timestream","chunking", "force_chunking"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.chunking.force = telescope.force_chunk;
        }
    }

    /* get raw time chunk config */
    get_rtc_config(config);

    /* get processed time chunk config */
    get_ptc_config(config);

    /* get shared reduction-learning config */
    get_learning_config(config);
}

template<typename CT>
void Engine::get_mapmaking_config(CT &config) {
    logger->info("getting mapmaking config options");
    typed_mapmaking_config = citlali::config::MapmakingConfig{};
    typed_coadd_config = citlali::config::CoaddConfig{};
    typed_noise_config = citlali::config::NoiseConfig{};

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return missing_keys.size() == missing_before && invalid_keys.size() == invalid_before;
    };

    // enable mapmaking?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_mapmaking, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_mapmaking_config.enabled = run_mapmaking;
        }
    }
    // map grouping
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, map_grouping, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","grouping"},{"auto","array","nw","detector","fg"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_map_grouping(map_grouping)) {
                typed_mapmaking_config.grouping = *parsed;
            }
        }
    }

    // optional expected sky regime for interpreting map diagnostics
    map_regime = "unknown";
    if (config.template has_typed<std::string>(std::tuple{"source", "map_regime"})) {
        map_regime = config.template get_typed<std::string>(std::tuple{"source", "map_regime"});
        check_allowed(map_regime, missing_keys, invalid_keys,
                      std::vector<std::string>{"source_dominant", "source_faint", "blank_field", "unknown"},
                      std::tuple{"source", "map_regime"});
    }

    // polarization is disabled for detector grouping
    if (rtcproc.run_polarization && ((redu_type=="beammap" && map_grouping=="auto") || map_grouping=="detector")) {
        logger->error("Detector grouping reductions do not currently support polarimetry mode");
        std::exit(EXIT_FAILURE);
    }

    // set rtcproc map_grouping
    rtcproc.kernel.map_grouping = map_grouping;
    ptcproc.active_map_grouping = map_grouping;

    // map_method
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, map_method, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","method"},{"naive","jinc","maximum_likelihood"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_map_method(map_method)) {
                typed_mapmaking_config.method = *parsed;
            }
        }
    }
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
    logger->info("fruit loops weight feedback: enabled={} reference={} relative=[{}, {}]",
                 ptcproc.fruit_loops_weight_feedback_enabled,
                 ptcproc.fruit_loops_weight_feedback_reference,
                 ptcproc.fruit_loops_weight_feedback_low_relative_weight,
                 ptcproc.fruit_loops_weight_feedback_high_relative_weight);
    ptcproc.fruit_loops_jinc_r_max = 0.0;
    ptcproc.fruit_loops_jinc_subpixel_n = 1;
    ptcproc.fruit_loops_jinc_shape_params.clear();

    // map reference frame (radec, altaz, galactic)
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, telescope.pixel_axes, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","pixel_axes"},{"radec","altaz", "galactic"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_mapmaking_config.pixel_axes = telescope.pixel_axes;
        }
    }
    if (redu_type == "beammap" && telescope.pixel_axes != "altaz") {
        logger->error(
            "beammap reductions require mapmaking.pixel_axes='altaz'; got '{}'",
            telescope.pixel_axes);
        std::exit(EXIT_FAILURE);
    }

    // get config for omb
    logger->info("getting omb config options");
    const auto omb_missing_before = missing_keys.size();
    const auto omb_invalid_before = invalid_keys.size();
    omb.get_config(config, missing_keys, invalid_keys, telescope.pixel_axes, redu_type);
    if (parsed_cleanly(omb_missing_before, omb_invalid_before)) {
        typed_mapmaking_config.coverage_cut = omb.cov_cut;
        typed_mapmaking_config.pixel_size_arcsec = omb.pixel_size_rad * RAD_TO_ASEC;
        typed_mapmaking_config.unit = omb.sig_unit;
        if (omb.wcs.naxis.size() >= 2) {
            typed_mapmaking_config.x_size_pix = static_cast<int>(omb.wcs.naxis[0]);
            typed_mapmaking_config.y_size_pix = static_cast<int>(omb.wcs.naxis[1]);
        }
        if (omb.wcs.crpix.size() >= 2) {
            typed_mapmaking_config.crpix1 = omb.wcs.crpix[0];
            typed_mapmaking_config.crpix2 = omb.wcs.crpix[1];
        }
        if (omb.crval_config.size() >= 2) {
            typed_mapmaking_config.crval1_j2000 = omb.crval_config[0];
            typed_mapmaking_config.crval2_j2000 = omb.crval_config[1];
        }
        typed_post_processing_config.map_histogram_n_bins = omb.hist_n_bins;
    }

    // run coaddition?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_coadd, missing_keys, invalid_keys,
                         std::tuple{"coadd","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_coadd_config.enabled = run_coadd;
        }
    }
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
            if (jinc_shape_vec.size() != 3) {
                invalid_keys.push_back({"mapmaking","jinc_filter","shape_params",arr_name});
                jinc_shape_vec.resize(3, 0.0);
            }
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
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_noise, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_noise_config.enabled = run_noise;
        }
    }
    if (run_noise) {
        // number of noise maps
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.n_noise, missing_keys, invalid_keys,
                             std::tuple{"noise_maps","n_noise_maps"},{},{0},{});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_noise_config.n_noise_maps = static_cast<int>(omb.n_noise);
            }
        }
        // randomize noise maps on detector as well as time chunk
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.randomize_dets, missing_keys, invalid_keys,
                             std::tuple{"noise_maps","randomize_dets"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_noise_config.randomize_dets = omb.randomize_dets;
            }
        }

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
        typed_noise_config.n_noise_maps = 0;
    }

    write_noise_realizations = false;
    if (config.template has_typed<bool>(std::tuple{"noise_maps","write_realizations"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, write_noise_realizations, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","write_realizations"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_noise_config.write_realizations = write_noise_realizations;
        }
    }
    run_noise_products = run_noise;
    typed_noise_config.products_enabled = run_noise_products;
    if (config.template has_typed<bool>(std::tuple{"noise_maps","products","enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_noise_products, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","products","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_noise_config.products_enabled = run_noise_products;
        }
    }
    apply_empirical_noise_weights = run_noise;
    typed_noise_config.apply_empirical_weights = apply_empirical_noise_weights;
    if (config.template has_typed<bool>(std::tuple{"noise_maps","products","apply_empirical_weights"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, apply_empirical_noise_weights, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","products","apply_empirical_weights"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_noise_config.apply_empirical_weights = apply_empirical_noise_weights;
        }
    }

    // set mapmaker polarization
    naive_mm.run_polarization = rtcproc.run_polarization;
    jinc_mm.run_polarization = rtcproc.run_polarization;
}

template<typename CT>
void Engine::get_pointing_config(CT &config) {
    logger->info("getting pointing config options");
    typed_pointing_config = citlali::config::PointingConfig{};

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return missing_keys.size() == missing_before && invalid_keys.size() == invalid_before;
    };

    pointing_source_strategy = "standard";
    if (config.template has_typed<std::string>(std::tuple{"pointing","source_strategy","mode"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_source_strategy, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","mode"},
                         {"standard", "psf_preserve"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_pointing_source_strategy(
                    pointing_source_strategy)) {
                typed_pointing_config.source_strategy = *parsed;
            }
        }
    }

    pointing_fit_gaussian_enabled = (pointing_source_strategy == "standard");
    typed_pointing_config.fit_gaussian = pointing_fit_gaussian_enabled;
    if (config.template has_typed<bool>(std::tuple{"pointing","source_strategy","fit_gaussian"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_fit_gaussian_enabled, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","fit_gaussian"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_pointing_config.fit_gaussian = pointing_fit_gaussian_enabled;
        }
    }

    pointing_fruitloops_center_mode =
        (pointing_source_strategy == "psf_preserve") ? "map_center" : "auto";
    if (auto parsed = citlali::config::parse_fruit_loops_center_mode(
            pointing_fruitloops_center_mode)) {
        typed_pointing_config.fruitloops_center_mode = *parsed;
    }
    if (config.template has_typed<std::string>(std::tuple{"pointing","source_strategy","fruitloops_center_mode"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_fruitloops_center_mode, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","fruitloops_center_mode"},
                         {"auto", "header", "peak", "map_center"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_fruit_loops_center_mode(
                    pointing_fruitloops_center_mode)) {
                typed_pointing_config.fruitloops_center_mode = *parsed;
            }
        }
    }

    pointing_header_center_max_radius_arcsec = 0.0;
    if (pointing_source_strategy == "standard" &&
        std::isfinite(map_fitter.fitting_region_pix) && map_fitter.fitting_region_pix > 0.0 &&
        std::isfinite(omb.pixel_size_rad) && omb.pixel_size_rad > 0.0) {
        pointing_header_center_max_radius_arcsec =
            map_fitter.fitting_region_pix * omb.pixel_size_rad * RAD_TO_ASEC;
    }
    typed_pointing_config.header_max_radius_arcsec =
        pointing_header_center_max_radius_arcsec;
    if (config.template has_typed<double>(std::tuple{"pointing","source_strategy","header_max_radius_arcsec"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_header_center_max_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","header_max_radius_arcsec"},
                         {}, {0.0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_pointing_config.header_max_radius_arcsec =
                pointing_header_center_max_radius_arcsec;
        }
    }

    pointing_header_center_require_coverage = true;
    typed_pointing_config.header_require_coverage =
        pointing_header_center_require_coverage;
    if (config.template has_typed<bool>(std::tuple{"pointing","source_strategy","header_require_coverage"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_header_center_require_coverage, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","header_require_coverage"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_pointing_config.header_require_coverage =
                pointing_header_center_require_coverage;
        }
    }

    ptcproc.fruit_loops_source_center_mode = pointing_fruitloops_center_mode;
    ptcproc.fruit_loops_header_center_max_radius_arcsec =
        pointing_header_center_max_radius_arcsec;
    ptcproc.fruit_loops_header_center_require_coverage =
        pointing_header_center_require_coverage;

    logger->info("pointing source strategy: mode={} fit_gaussian={} fruitloops_center_mode={} "
                 "header_max_radius_arcsec={} header_require_coverage={}",
                 pointing_source_strategy, pointing_fit_gaussian_enabled,
                 pointing_fruitloops_center_mode,
                 pointing_header_center_max_radius_arcsec,
                 pointing_header_center_require_coverage);

    if (!ptcproc.run_fruit_loops) {
        logger->warn("pointing source strategy is configured but timestream.fruit_loops.enabled=false");
    }
    else if (ptcproc.fruit_loops_iters < 2) {
        logger->warn("pointing source-aware fruit loops uses previous maps; max_iters={} will not run a measurement iteration",
                     ptcproc.fruit_loops_iters);
    }

    if (pointing_source_strategy == "psf_preserve" && pointing_fit_gaussian_enabled) {
        logger->warn("pointing.source_strategy.mode=psf_preserve with fit_gaussian=true; "
                     "Gaussian fits remain diagnostics only and do not constrain fruit loops");
    }
    if (pointing_source_strategy == "psf_preserve" &&
        pointing_fruitloops_center_mode == "peak") {
        logger->warn("pointing.source_strategy.mode=psf_preserve with fruitloops_center_mode=peak; "
                     "messy out-of-focus maps may bias the fruit loops source support");
    }
    if (!pointing_fit_gaussian_enabled &&
        (pointing_fruitloops_center_mode == "header" ||
         pointing_fruitloops_center_mode == "auto")) {
        logger->warn("pointing Gaussian fitting is disabled; later fruit loops iterations will not "
                     "get new valid POINTING header centers from this run");
    }
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
    beammap_convergence_radius_arcsec = 10.0;
    if (config.template has_typed<double>(std::tuple{"beammap","convergence_radius_arcsec"})) {
        get_config_value(config, beammap_convergence_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"beammap","convergence_radius_arcsec"},
                         {}, {0.0});
    }

    beammap_phase_split_enabled = true;
    beammap_locator_iter = 0;
    beammap_measurement_start_iter = 1;
    if (config.template has_typed<bool>(std::tuple{"beammap","phase_strategy","enabled"})) {
        get_config_value(config, beammap_phase_split_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","phase_strategy","enabled"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","phase_strategy","locator_iter"})) {
        get_config_value(config, beammap_locator_iter, missing_keys, invalid_keys,
                         std::tuple{"beammap","phase_strategy","locator_iter"},
                         {}, {0});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","phase_strategy","measurement_start_iter"})) {
        get_config_value(config, beammap_measurement_start_iter, missing_keys, invalid_keys,
                         std::tuple{"beammap","phase_strategy","measurement_start_iter"},
                         {}, {1});
    }
    if (beammap_locator_iter != 0) {
        logger->warn(
            "beammap.phase_strategy.locator_iter={} requested, but the locator pass must be iter 0; using 0",
            beammap_locator_iter);
        beammap_locator_iter = 0;
    }
    if (beammap_measurement_start_iter <= beammap_locator_iter) {
        logger->warn(
            "beammap.phase_strategy.measurement_start_iter={} must be after locator_iter={}; using {}",
            beammap_measurement_start_iter, beammap_locator_iter, beammap_locator_iter + 1);
        beammap_measurement_start_iter = beammap_locator_iter + 1;
    }
    if (beammap_iter_max <= beammap_measurement_start_iter) {
        logger->warn(
            "beammap.iter_max={} will not run a measurement pass with measurement_start_iter={}",
            beammap_iter_max, beammap_measurement_start_iter);
    }

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

    beammap_detector_weighting_mode = "const";
    if (config.template has_typed<std::string>(std::tuple{"beammap","detector_weighting","mode"})) {
        get_config_value(config, beammap_detector_weighting_mode, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_weighting","mode"},
                         {"const", "ptc", "ptc_after_iter0"});
    }

    beammap_fit_radius_fwhm = 0.0;
    if (config.template has_typed<double>(std::tuple{"beammap","fitting","fit_radius_fwhm"})) {
        get_config_value(config, beammap_fit_radius_fwhm, missing_keys, invalid_keys,
                         std::tuple{"beammap","fitting","fit_radius_fwhm"},
                         {}, {0.0});
    }
    map_fitter.beammap_fit_radius_fwhm = beammap_fit_radius_fwhm;

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
    beammap_priors_max_d2_iter0 = 25.0;
    beammap_priors_max_d2_after_iter0 = 25.0;
    beammap_priors_score_lambda = 2.0;
    beammap_priors_score_lambda_iter0 = 2.0;
    beammap_priors_score_lambda_after_iter0 = 2.0;
    beammap_priors_fallback_blind = true;
    beammap_priors_align_after_iter0 = true;
    beammap_priors_alignment_scope = "array";
    beammap_priors_alignment_common_support = "all";
    beammap_priors_alignment_common_support_quantile = 0.02;
    beammap_priors_alignment_min_matches = 30;
    beammap_priors_alignment_max_d2 = 25.0;
    beammap_priors_alignment_fit_rotation = true;
    beammap_priors_alignment_max_rotation_deg = 8.0;

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
    beammap_priors_max_d2_iter0 = beammap_priors_max_d2;
    beammap_priors_max_d2_after_iter0 = beammap_priors_max_d2;
    if (config.template has_typed<double>(std::tuple{"beammap","priors","score_lambda"})) {
        get_config_value(config, beammap_priors_score_lambda, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","score_lambda"},
                         {}, {0.0});
    }
    beammap_priors_score_lambda_iter0 = beammap_priors_score_lambda;
    beammap_priors_score_lambda_after_iter0 = beammap_priors_score_lambda;
    if (config.template has_typed<double>(std::tuple{"beammap","priors","max_d2_iter0"})) {
        get_config_value(config, beammap_priors_max_d2_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","max_d2_iter0"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","max_d2_after_iter0"})) {
        get_config_value(config, beammap_priors_max_d2_after_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","max_d2_after_iter0"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","score_lambda_iter0"})) {
        get_config_value(config, beammap_priors_score_lambda_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","score_lambda_iter0"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","score_lambda_after_iter0"})) {
        get_config_value(config, beammap_priors_score_lambda_after_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","score_lambda_after_iter0"},
                         {}, {0.0});
    }
    if (config.template has_typed<bool>(std::tuple{"beammap","priors","fallback_blind"})) {
        get_config_value(config, beammap_priors_fallback_blind, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","fallback_blind"});
    }
    if (config.template has_typed<bool>(std::tuple{"beammap","priors","align_after_iter0"})) {
        get_config_value(config, beammap_priors_align_after_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","align_after_iter0"});
    }
    if (config.template has_typed<std::string>(std::tuple{"beammap","priors","alignment_scope"})) {
        get_config_value(config, beammap_priors_alignment_scope, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_scope"},
                         {"array", "common"});
    }
    if (config.template has_typed<std::string>(std::tuple{"beammap","priors","alignment_common_support"})) {
        get_config_value(config, beammap_priors_alignment_common_support, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_common_support"},
                         {"all", "overlap_box"});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","alignment_common_support_quantile"})) {
        get_config_value(config, beammap_priors_alignment_common_support_quantile, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_common_support_quantile"},
                         {}, {0.0}, {0.45});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","priors","alignment_min_matches"})) {
        get_config_value(config, beammap_priors_alignment_min_matches, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_min_matches"},
                         {}, {3});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","alignment_max_d2"})) {
        get_config_value(config, beammap_priors_alignment_max_d2, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_max_d2"},
                         {}, {0.0});
    }
    if (config.template has_typed<bool>(std::tuple{"beammap","priors","alignment_fit_rotation"})) {
        get_config_value(config, beammap_priors_alignment_fit_rotation, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_fit_rotation"});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","alignment_max_rotation_deg"})) {
        get_config_value(config, beammap_priors_alignment_max_rotation_deg, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_max_rotation_deg"},
                         {}, {0.0});
    }
    if (beammap_priors_enabled && beammap_priors_filepath == "null") {
        logger->warn("beammap.priors.enabled=true but beammap.priors.filepath is null; disabling priors");
        beammap_priors_enabled = false;
    }

    auto get_fixed_beammap_vector = [&](const std::vector<std::string> &path,
                                        std::size_t expected_size) {
        std::vector<double> values;
        if (path.size() == 2) {
            values = config.template get_typed<std::vector<double>>(std::make_tuple(path[0], path[1]));
        }
        else {
            values = config.template get_typed<std::vector<double>>(std::make_tuple(path[0], path[1], path[2]));
        }
        if (values.size() != expected_size) {
            invalid_keys.push_back(path);
            values.resize(expected_size, 0.0);
        }
        return values;
    };

    const std::size_t n_toltec_arrays = toltec_io.array_name_map.size();
    // lower fwhm limit
    auto lower_fwhm_arcsec_vec = get_fixed_beammap_vector({"beammap","flagging","array_lower_fwhm_arcsec"},
                                                          n_toltec_arrays);
    // upper fwhm limit
    auto upper_fwhm_arcsec_vec = get_fixed_beammap_vector({"beammap","flagging","array_upper_fwhm_arcsec"},
                                                          n_toltec_arrays);
    // lower signal-to-noise limit
    auto lower_sig2noise_vec = get_fixed_beammap_vector({"beammap","flagging","array_lower_sig2noise"},
                                                        n_toltec_arrays);
    // upper signal-to-noise limit
    auto upper_sig2noise_vec = get_fixed_beammap_vector({"beammap","flagging","array_upper_sig2noise"},
                                                        n_toltec_arrays);
    // maximum allowed distance limit
    auto max_dist_arcsec_vec = get_fixed_beammap_vector({"beammap","flagging","array_max_dist_arcsec"},
                                                        n_toltec_arrays);
    // per-array post-derotation network geometry cut
    auto network_robust_z_vec = get_fixed_beammap_vector({"beammap","flagging","array_network_robust_z"},
                                                         n_toltec_arrays);
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
    auto sens_factors_vec = get_fixed_beammap_vector({"beammap","flagging","sens_factors"}, 2);
    lower_sens_factor = sens_factors_vec[0];
    upper_sens_factor = sens_factors_vec[1];

    // upper and lower frequencies over which to calculate sensitivity
    sens_psd_limits_Hz.resize(2);
    // get psd limits for sens from config
    auto sens_psd_limits_Hz_vec = get_fixed_beammap_vector({"beammap","sens_psd_limits_Hz"}, 2);
    // map sens limits back to Eigen vector
    sens_psd_limits_Hz = (Eigen::Map<Eigen::VectorXd>(sens_psd_limits_Hz_vec.data(), sens_psd_limits_Hz_vec.size()));

    // Beammap PTC TOD/diagnostics are written after the convergence decision.
    // The default is the actual last attempted iteration, including early
    // convergence, so the saved PTC reflects the final cleaning state.
    beammap_tod_output_iter = -1;

    beammap_detector_tod_output_enabled = false;
    beammap_detector_tod_output_subdir_name = "source_crossing_tod";
    beammap_detector_tod_output_n_uniform = 10;
    beammap_detector_tod_output_n_source_dense = 10;
    if (config.template has_typed<bool>(std::tuple{"beammap","detector_tod_output","enabled"})) {
        get_config_value(config, beammap_detector_tod_output_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_tod_output","enabled"});
    }
    if (config.template has_typed<std::string>(std::tuple{"beammap","detector_tod_output","subdir_name"})) {
        get_config_value(config, beammap_detector_tod_output_subdir_name, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_tod_output","subdir_name"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","detector_tod_output","n_uniform"})) {
        get_config_value(config, beammap_detector_tod_output_n_uniform, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_tod_output","n_uniform"},
                         {}, {0});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","detector_tod_output","n_source_dense"})) {
        get_config_value(config, beammap_detector_tod_output_n_source_dense, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_tod_output","n_source_dense"},
                         {}, {0});
    }

    typed_beammap_config = citlali::config::BeammapConfig{};
    typed_beammap_config.iteration.max_iterations = beammap_iter_max;
    typed_beammap_config.iteration.tolerance = beammap_iter_tolerance;
    typed_beammap_config.iteration.convergence_radius_arcsec =
        beammap_convergence_radius_arcsec;
    typed_beammap_config.phase_strategy.enabled = beammap_phase_split_enabled;
    typed_beammap_config.phase_strategy.locator_iter = beammap_locator_iter;
    typed_beammap_config.phase_strategy.measurement_start_iter =
        beammap_measurement_start_iter;
    typed_beammap_config.reference.subtract_reference_detector =
        beammap_subtract_reference;
    typed_beammap_config.reference.reference_detector =
        static_cast<long>(beammap_reference_det);
    typed_beammap_config.reference.derotate = beammap_derotate;
    typed_beammap_config.rfi_mask.enabled = beammap_rfi_mask_enabled;
    typed_beammap_config.rfi_mask.block_size_samples =
        beammap_rfi_mask_block_size_samples;
    typed_beammap_config.rfi_mask.min_good_samples =
        beammap_rfi_mask_min_good_samples;
    typed_beammap_config.rfi_mask.dilate_blocks = beammap_rfi_mask_dilate_blocks;
    typed_beammap_config.rfi_mask.sigma_threshold =
        beammap_rfi_mask_sigma_threshold;
    typed_beammap_config.rfi_mask.sigma_floor = beammap_rfi_mask_sigma_floor;
    typed_beammap_config.rfi_mask.max_flagged_fraction =
        beammap_rfi_mask_max_flagged_fraction;
    if (auto parsed = citlali::config::parse_beammap_detector_weighting_mode(
            beammap_detector_weighting_mode)) {
        typed_beammap_config.detector_weighting_mode = *parsed;
    }
    typed_beammap_config.fitting.fit_radius_fwhm = beammap_fit_radius_fwhm;
    typed_beammap_config.scan_band_mask.enabled = beammap_scan_band_mask_enabled;
    typed_beammap_config.scan_band_mask.edge_rows = beammap_scan_band_mask_edge_rows;
    typed_beammap_config.scan_band_mask.min_row_pixels =
        beammap_scan_band_mask_min_row_pixels;
    typed_beammap_config.scan_band_mask.min_contiguous_rows =
        beammap_scan_band_mask_min_contiguous_rows;
    typed_beammap_config.scan_band_mask.row_median_sigma_threshold =
        beammap_scan_band_mask_row_median_sigma_threshold;
    typed_beammap_config.scan_band_mask.row_sigma_ratio_threshold =
        beammap_scan_band_mask_row_sigma_ratio_threshold;
    typed_beammap_config.scan_band_mask.max_flagged_fraction =
        beammap_scan_band_mask_max_flagged_fraction;
    typed_beammap_config.split_fits_by_flag.enabled = beammap_split_fits_by_flag;
    typed_beammap_config.split_fits_by_flag.flag_values = beammap_split_flag_values;
    typed_beammap_config.priors.enabled = beammap_priors_enabled;
    typed_beammap_config.priors.filepath = beammap_priors_filepath;
    typed_beammap_config.priors.candidate_top_n =
        beammap_priors_candidate_top_n;
    typed_beammap_config.priors.min_snr = beammap_priors_min_snr;
    typed_beammap_config.priors.max_d2 = beammap_priors_max_d2;
    typed_beammap_config.priors.max_d2_iter0 = beammap_priors_max_d2_iter0;
    typed_beammap_config.priors.max_d2_after_iter0 =
        beammap_priors_max_d2_after_iter0;
    typed_beammap_config.priors.score_lambda = beammap_priors_score_lambda;
    typed_beammap_config.priors.score_lambda_iter0 =
        beammap_priors_score_lambda_iter0;
    typed_beammap_config.priors.score_lambda_after_iter0 =
        beammap_priors_score_lambda_after_iter0;
    typed_beammap_config.priors.fallback_blind = beammap_priors_fallback_blind;
    typed_beammap_config.priors.align_after_iter0 =
        beammap_priors_align_after_iter0;
    typed_beammap_config.priors.alignment_scope =
        beammap_priors_alignment_scope;
    typed_beammap_config.priors.alignment_common_support =
        beammap_priors_alignment_common_support;
    typed_beammap_config.priors.alignment_common_support_quantile =
        beammap_priors_alignment_common_support_quantile;
    typed_beammap_config.priors.alignment_min_matches =
        beammap_priors_alignment_min_matches;
    typed_beammap_config.priors.alignment_max_d2 =
        beammap_priors_alignment_max_d2;
    typed_beammap_config.priors.alignment_fit_rotation =
        beammap_priors_alignment_fit_rotation;
    typed_beammap_config.priors.alignment_max_rotation_deg =
        beammap_priors_alignment_max_rotation_deg;
    typed_beammap_config.detector_tod_output.enabled =
        beammap_detector_tod_output_enabled;
    typed_beammap_config.detector_tod_output.subdir_name =
        beammap_detector_tod_output_subdir_name;
    typed_beammap_config.detector_tod_output.n_uniform =
        beammap_detector_tod_output_n_uniform;
    typed_beammap_config.detector_tod_output.n_source_dense =
        beammap_detector_tod_output_n_source_dense;
    typed_beammap_config.flagging.array_lower_fwhm_arcsec =
        lower_fwhm_arcsec_vec;
    typed_beammap_config.flagging.array_upper_fwhm_arcsec =
        upper_fwhm_arcsec_vec;
    typed_beammap_config.flagging.array_lower_sig2noise =
        lower_sig2noise_vec;
    typed_beammap_config.flagging.array_upper_sig2noise =
        upper_sig2noise_vec;
    typed_beammap_config.flagging.array_max_dist_arcsec =
        max_dist_arcsec_vec;
    typed_beammap_config.flagging.array_network_robust_z =
        network_robust_z_vec;
    typed_beammap_config.flagging.sens_factors = sens_factors_vec;
    typed_beammap_config.flagging.sens_psd_limits_hz = sens_psd_limits_Hz_vec;
    typed_beammap_config.flagging.max_prior_d2 = beammap_flag_max_prior_d2;
}

template<typename CT>
void Engine::get_map_filter_config(CT &config) {
    logger->info("getting map filtering config options");
    // get wiener filter config options
    wiener_filter.get_config(config, missing_keys, invalid_keys);

    auto &typed_map_filter = typed_post_processing_config.map_filtering;
    typed_map_filter.enabled = run_map_filter;
    if (auto parsed = citlali::config::parse_map_filter_type(wiener_filter.filter_type)) {
        typed_map_filter.type = *parsed;
    }
    if (auto parsed = citlali::config::parse_map_filter_template_type(
            wiener_filter.template_type)) {
        typed_map_filter.template_type = *parsed;
    }
    typed_map_filter.lowpass_only = wiener_filter.run_lowpass;
    typed_map_filter.normalize_errors = wiener_filter.normalize_error;
    typed_map_filter.edge_guard.enabled = wiener_filter.edge_guard_enabled;
    typed_map_filter.edge_guard.weight_threshold_mode =
        wiener_filter.edge_weight_threshold_mode;
    typed_map_filter.edge_guard.hits_threshold_mode =
        wiener_filter.edge_hits_threshold_mode;
    typed_map_filter.edge_guard.hits_core_fraction =
        wiener_filter.edge_hits_core_fraction;
    typed_map_filter.edge_guard.guard_radius_fwhm =
        wiener_filter.edge_guard_radius_fwhm;
    typed_map_filter.edge_guard.fill_mode = wiener_filter.edge_fill_mode;
    if (auto parsed = citlali::config::parse_map_filter_edge_taper_mode(
            wiener_filter.edge_taper_mode)) {
        typed_map_filter.edge_guard.taper_mode = *parsed;
    }
    typed_map_filter.edge_guard.taper_min_fraction =
        wiener_filter.edge_taper_min_fraction;
    typed_map_filter.denom_rel_tol = wiener_filter.denom_rel_tol;
    typed_map_filter.tail_frac_tol = wiener_filter.tail_frac_tol;
    typed_map_filter.max_loops = wiener_filter.max_loops;
    typed_map_filter.denom_check_iters = wiener_filter.denom_check_iters;
    typed_map_filter.max_denom_iters = wiener_filter.max_denom_iters;
    typed_map_filter.template_fwhm_arcsec.clear();
    for (const auto &[array_name, fwhm_rad] : wiener_filter.template_fwhm_rad) {
        typed_map_filter.template_fwhm_arcsec[array_name] =
            fwhm_rad * RAD_TO_ASEC;
    }

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
citlali::config::RuntimeConfig Engine::get_runtime_config(CT &config) {
    citlali::config::RuntimeConfig runtime_config;

    // verbose mode?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, verbose_mode, missing_keys, invalid_keys,
                         std::tuple{"runtime","verbose"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            runtime_config.verbose = verbose_mode;
        }
    }
    // output directory
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, output_dir, missing_keys, invalid_keys,
                         std::tuple{"runtime","output_dir"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            runtime_config.output_dir = output_dir;
        }
    }
    // number of threads to use
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, n_threads, missing_keys, invalid_keys,
                         std::tuple{"runtime","n_threads"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            runtime_config.n_threads = n_threads;
        }
    }
    // overall parallel policy
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, parallel_policy, missing_keys, invalid_keys,
                         std::tuple{"runtime","parallel_policy"},{"seq","omp"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            if (auto parsed = citlali::config::parse_parallel_policy(parallel_policy)) {
                runtime_config.parallel_policy = *parsed;
            }
        }
    }
    // reduction type (science, pointing, beammap)
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, redu_type, missing_keys, invalid_keys,
                         std::tuple{"runtime","reduction_type"},{"science","pointing","beammap"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            if (auto parsed = citlali::config::parse_reduction_type(redu_type)) {
                runtime_config.reduction_type = *parsed;
            }
        }
    }
    // create redu00, redu01... subdirectories
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, use_subdir, missing_keys, invalid_keys,
                         std::tuple{"runtime","use_subdir"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            runtime_config.use_subdir = use_subdir;
        }
    }
    // interp over gaps in align_timestream
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, interp_over_gaps, missing_keys, invalid_keys,
                         std::tuple{"runtime","interp_over_gaps"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            runtime_config.interp_over_gaps = interp_over_gaps;
        }
    }
    return runtime_config;
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

    typed_runtime_config = get_runtime_config(config);
    if (!typed_runtime_config.interp_over_gaps) {
        logger->error("runtime.interp_over_gaps=false is unsupported; set runtime.interp_over_gaps: true");
        std::exit(EXIT_FAILURE);
    }

    /* get timestream config */
    get_timestream_config(config);
    {
        // The pointing pipeline also covers PSF-preserving focus and holography-style reductions.
        const bool source_aware_reduction = (redu_type == "pointing");
        rtcproc.despiker.source_protection_enabled =
            rtcproc.run_despike &&
            rtcproc.despike_source_protection_config_enabled &&
            source_aware_reduction;
        ptcproc.second_pass_local.source_protection_enabled =
            ptcproc.second_pass_local.enabled &&
            ptcproc.second_pass_local.source_protection_config_enabled &&
            source_aware_reduction;
        typed_timestream_config.raw_time_chunk.despike.source_protection.active =
            rtcproc.despiker.source_protection_enabled;
        typed_timestream_config.processed_time_chunk.flagging.second_pass_local
            .source_protection.active =
            ptcproc.second_pass_local.source_protection_enabled;
        if (rtcproc.run_despike && rtcproc.despike_source_protection_config_enabled) {
            logger->info(
                "raw_time_chunk.despike source protection active={} reduction_type={} radius_arcsec={:.4g}",
                rtcproc.despiker.source_protection_enabled, redu_type,
                rtcproc.despiker.source_protection_radius_arcsec);
        }
        if (ptcproc.second_pass_local.enabled &&
            ptcproc.second_pass_local.source_protection_config_enabled) {
            logger->info(
                "processed_time_chunk.flagging.second_pass_local source protection active={} reduction_type={} radius_arcsec={:.4g}",
                ptcproc.second_pass_local.source_protection_enabled, redu_type,
                ptcproc.second_pass_local.source_protection_radius_arcsec);
        }
    }

    /* get mapmaking config */
    typed_post_processing_config = citlali::config::PostProcessingConfig{};
    get_mapmaking_config(config);

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return missing_keys.size() == missing_before && invalid_keys.size() == invalid_before;
    };

    // run map filter?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_map_filter, missing_keys, invalid_keys,
                         std::tuple{"post_processing","map_filtering","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_post_processing_config.map_filtering_enabled = run_map_filter;
            typed_post_processing_config.map_filtering.enabled = run_map_filter;
        }
    }

    // run source finder?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_source_finder, missing_keys, invalid_keys,
                         std::tuple{"post_processing","source_finding","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_post_processing_config.source_finding_enabled = run_source_finder;
            typed_post_processing_config.source_finding.enabled = run_source_finder;
        }
    }

    // map fitter options if in pointing or beammap mode or if map filtering or source finding are enabled
    if (redu_type=="pointing" || redu_type=="beammap" || run_map_filter || run_source_finder) {
        typed_post_processing_config.source_fitting.active = true;
        // size of region around found source to fit
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, map_fitter.bounding_box_pix, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_fitting","bounding_box_arcsec"},{},{0});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_fitting.bounding_box_arcsec =
                    map_fitter.bounding_box_pix;
            }
        }
        // radius around center of map to find source within
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, map_fitter.fitting_region_pix, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_fitting","fitting_radius_arcsec"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_fitting.fitting_radius_arcsec =
                    map_fitter.fitting_region_pix;
            }
        }
        // fit 2d gaussian rotation angle
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, map_fitter.fit_angle, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_fitting", "gauss_model","fit_rotation_angle"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_fitting.fit_rotation_angle =
                    map_fitter.fit_angle;
            }
        }

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
            typed_post_processing_config.source_fitting.amp_limit_factors[static_cast<std::size_t>(i)] =
                map_fitter.flux_limits(i);
            // fwhm limit
            map_fitter.fwhm_limits(i) = config.template get_typed<double>(std::tuple{"post_processing","source_fitting",
                                                                                     "gauss_model","fwhm_limit_factors",i});
            typed_post_processing_config.source_fitting.fwhm_limit_factors[static_cast<std::size_t>(i)] =
                map_fitter.fwhm_limits(i);
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
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.source_sigma, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_finding","source_sigma"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_finding.source_sigma =
                    omb.source_sigma;
            }
        }
        // window around source to exclude other sources
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.source_window_rad, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_finding","source_window_arcsec"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_finding.source_window_arcsec =
                    omb.source_window_rad;
            }
        }
        // search map, negative of map, or both
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.source_finder_mode, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_finding","mode"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_finding.mode =
                    omb.source_finder_mode;
            }
        }

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

    /* get pointing config */
    if (redu_type=="pointing") {
        get_pointing_config(config);
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
        typed_coadd_config.enabled = false;
        typed_noise_config.enabled = false;
        typed_post_processing_config.map_filtering_enabled = false;
        typed_post_processing_config.map_filtering.enabled = false;
        typed_post_processing_config.source_finding_enabled = false;
        typed_post_processing_config.source_finding.enabled = false;
        typed_post_processing_config.source_fitting.active = false;
        // we don't need to do iterations if no maps are made
        beammap_iter_max = 1;
        typed_beammap_config.iteration.max_iterations = 1;
    }
}

template<typename CT>
void Engine::get_photometry_config(CT &config) {
    typed_beammap_config.source = citlali::config::BeammapSourceConfig{};

    // beammap source name
    get_config_value(config, beammap_source_name, missing_keys, invalid_keys,
                     std::tuple{"beammap_source","name"});
    typed_beammap_config.source.name = beammap_source_name;
    // beammap source ra
    get_config_value(config, beammap_ra_rad, missing_keys, invalid_keys,
                     std::tuple{"beammap_source","ra_deg"});
    typed_beammap_config.source.ra_deg = beammap_ra_rad;
    // convert ra to radians
    beammap_ra_rad = beammap_ra_rad*DEG_TO_RAD;

    // beammap source dec
    get_config_value(config, beammap_dec_rad, missing_keys, invalid_keys,
                     std::tuple{"beammap_source","dec_deg"});
    typed_beammap_config.source.dec_deg = beammap_dec_rad;
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
        typed_beammap_config.source.fluxes.push_back(
            citlali::config::BeammapSourceFluxConfig{array, flux, uncertainty_mJy});
    }

    if (redu_type == "beammap") {
        bool valid_flux_config = true;
        for (auto const& entry : toltec_io.array_name_map) {
            const auto &arr_name = entry.second;
            auto flux_it = beammap_fluxes_mJy_beam.find(arr_name);
            if (flux_it == beammap_fluxes_mJy_beam.end()) {
                logger->error(
                    "beammap reductions require a positive source flux for {}; no beammap_source.fluxes entry was found",
                    arr_name);
                valid_flux_config = false;
                continue;
            }
            const double flux = flux_it->second;
            if (!std::isfinite(flux) || flux <= 0.0) {
                logger->error(
                    "beammap reductions require positive finite source fluxes; {} value_mJy={}",
                    arr_name, flux);
                valid_flux_config = false;
            }
        }
        if (!valid_flux_config) {
            std::exit(EXIT_FAILURE);
        }
    }
}

template<typename CT>
void Engine::get_astrometry_config(CT &config) {
    typed_astrometry_config = citlali::config::AstrometryConfig{};

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

        auto &typed_offsets = typed_astrometry_config.pointing_offsets;
        typed_offsets.enabled = true;
        const auto &az_offsets = pointing_offsets_arcsec["az"];
        typed_offsets.az_arcsec.assign(
            az_offsets.data(), az_offsets.data() + az_offsets.size());
        const auto &alt_offsets = pointing_offsets_arcsec["alt"];
        typed_offsets.alt_arcsec.assign(
            alt_offsets.data(), alt_offsets.data() + alt_offsets.size());
        typed_offsets.modified_julian_date.assign(
            pointing_offsets_modified_julian_date.data(),
            pointing_offsets_modified_julian_date.data() +
                pointing_offsets_modified_julian_date.size());
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
            if (run_noise && write_noise_realizations) {
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
                if (run_noise && write_noise_realizations) {
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
            citlali::pipeline::add_unit_conversion_basis_vars(fo);
            for (const auto &val: calib.arrays) {
                auto name = toltec_io.array_name_map[val];
                // conversion to Rayleigh-Jeans uK brightness temperature
                auto fwhm = (std::get<0>(calib.array_fwhms[val]) + std::get<1>(calib.array_fwhms[val]))/2;
                auto mJy_beam_to_uK = engine_utils::mJy_beam_to_uK(1, toltec_io.array_freq_map[val], fwhm);

                // beam area in steradians
                auto beam_area_rad = 2.*pi*pow(fwhm*FWHM_TO_STD*ASEC_TO_RAD,2);
                // get Jy/pixel
                auto mJy_beam_to_Jy_px = 1e-3/beam_area_rad*pow(omb.pixel_size_rad,2);

                citlali::pipeline::add_unit_conversion_array_vars(
                    fo, name, omb.sig_unit,
                    calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC,
                    mJy_beam_to_uK, mJy_beam_to_Jy_px);
            }
        }

        citlali::pipeline::add_observation_date_source_vars(
            fo, date_obs.back(), telescope.source_name);

        // add source flux for beammaps
        if (redu_type == "beammap") {
            citlali::pipeline::add_beammap_tod_header_vars(
                fo, calib, toltec_io.array_name_map,
                beammap_fluxes_mJy_beam, beammap_fluxes_MJy_Sr,
                beammap_iter_tolerance, beammap_convergence_radius_arcsec,
                beammap_iter_max, beammap_phase_split_enabled,
                beammap_locator_iter, beammap_measurement_start_iter,
                beammap_derotate, beammap_subtract_reference,
                beammap_reference_det);
        }

        citlali::pipeline::add_tod_identity_geometry_vars(
            fo, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION, TULA_GIT_VERSION,
            telescope.project_id, redu_type, telescope.obs_goal, tod_type,
            calib.run_hwpr, map_grouping, map_method, omb.exposure_time,
            telescope.pixel_axes, telescope.tel_header["Header.Source.Ra"][0],
            telescope.tel_header["Header.Source.Dec"][0],
            RAD_TO_DEG * telescope.tel_data["TelElAct"].mean(),
            RAD_TO_DEG * telescope.tel_data["TelAzAct"].mean(),
            RAD_TO_DEG * telescope.tel_data["ActParAng"].mean(),
            calib.arrays, calib.array_fwhms, calib.array_pas,
            toltec_io.array_name_map, RAD_TO_DEG, pi / 2, omb.sig_unit);

        citlali::pipeline::add_jinc_shape_config_vars_if_needed(
            fo, map_method, calib.arrays, jinc_mm.shape_params,
            toltec_io.array_name_map, jinc_mm.r_max);

        citlali::pipeline::add_tod_mean_tau_vars(
            fo, rtcproc, telescope.tel_data, telescope.tau_225_GHz,
            calib, toltec_io.array_name_map);

        citlali::pipeline::add_tod_auxiliary_metadata_vars(
            fo, telescope.fsmp,
            citlali::pipeline::apt_table_header_name(
                calib.apt_filepath, logger),
            fruit_iter);

        // add control/runtime parameters
        citlali::pipeline::add_tod_initial_runtime_config_vars(
            fo, verbose_mode, rtcproc.run_polarization, rtcproc.run_despike);
        const bool run_any_tod_filter = rtcproc.run_tod_filter || rtcproc.run_tod_iir_highpass;
        citlali::pipeline::add_rtc_local_despike_config_vars(
            fo, rtcproc.despiker.local_residual);
        citlali::pipeline::add_tod_filter_runtime_config_vars(
            fo, rtcproc, run_any_tod_filter);
        citlali::pipeline::add_tod_filter_edge_guard_config_vars(
            fo, rtcproc.filter_edge_guard, telescope.outer_scans_chunk,
            rtcproc.tod_output_outer_context_samples);
        citlali::pipeline::add_tod_processing_config_vars(fo, rtcproc);
        citlali::pipeline::add_weight_selection_config_vars(fo, ptcproc);
        citlali::pipeline::add_reduction_learning_config_vars(
            fo, reduction_learning);
        add_netcdf_var(fo, "CONFIG.INV_VAR.RTC.WTLOW", rtcproc.lower_inv_var_factor);
        add_netcdf_var(fo, "CONFIG.INV_VAR.RTC.WTHIGH", rtcproc.upper_inv_var_factor);
        citlali::pipeline::add_rtc_event_mask_config_vars(fo, rtcproc);
        citlali::pipeline::add_rtc_line_audit_config_vars_if_absent(
            fo, rtcproc.line_audit);
        citlali::pipeline::add_ptc_cleaning_header_config_vars(
            fo, ptcproc, calib, toltec_io.array_name_map);

        citlali::pipeline::add_oof_header_vars_if_observed(
            fo, telescope.sim_obs, telescope.tel_header, mb, redu_type,
            run_mapmaking, calib, toltec_io.array_name_map,
            toltec_io.array_wavelength_map);

        citlali::pipeline::add_fruit_loop_header_config_vars(
            fo, ptcproc, calib, toltec_io.array_name_map);

        fo.close();
    }
}

template <engine_utils::toltecIO::ProdType prod_t>
void Engine::create_tod_files() {
    // name for std map
    std::string name;
    const std::string dir_name = citlali::pipeline::tod_output_directory(
        obsnum_dir_name, tod_output_subdir_name);

    // rtc tod output filename setup
    if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
        auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                  engine_utils::toltecIO::rtc_timestream,
                                                  engine_utils::toltecIO::raw>(dir_name, redu_type, "",
                                                                               obsnum, telescope.sim_obs);

        name = citlali::pipeline::register_tod_output_file(
            tod_filename, "rtc", filename);
    }

    // ptc tod output filename setup
    else if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
        auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                  engine_utils::toltecIO::ptc_timestream,
                                                  engine_utils::toltecIO::raw>(dir_name, redu_type, "",
                                                                               obsnum, telescope.sim_obs);

        name = citlali::pipeline::register_tod_output_file(
            tod_filename, "ptc", filename);
    }

    write_netcdf_atomic(tod_filename[name], [&](netCDF::NcFile &fo) {

    if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
        citlali::pipeline::add_tod_output_type_label(fo, "rtc");
    }
    else if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
        citlali::pipeline::add_tod_output_type_label(fo, "ptc");
        citlali::pipeline::add_ptc_eigenvalue_dim(fo, ptcproc.cleaner.n_calc);
    }

    citlali::pipeline::add_observation_identity_vars(
        fo, std::stoi(obsnum), telescope.tel_header["Header.Source.Ra"](0),
        telescope.tel_header["Header.Source.Dec"](0));

    if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
        // Keep the RTC line-audit tuning alongside the RTC TOD so offline audits
        // can recover the exact per-run thresholds without the sidecar YAML.
        citlali::pipeline::add_rtc_line_audit_config_vars(
            fo, rtcproc.line_audit);
    }

    const auto tod_stream_layout = citlali::pipeline::tod_stream_layout(
        prod_t == engine_utils::toltecIO::rtc_timestream,
        n_tod_output_scans_rtc, n_tod_output_scans_ptc, rtcproc, ptcproc);
    const Eigen::Index n_tod_output_scans_for_stream =
        tod_stream_layout.n_output_scans;
    const bool tod_output_mini = tod_stream_layout.mini_output;
    const bool tod_output_outer = tod_stream_layout.outer_output;
    const auto tod_file_counts = citlali::pipeline::tod_file_counts(
        n_tod_output_scans_for_stream, telescope.scan_indices.rows(),
        calib.n_dets);

    const auto tod_dims = citlali::pipeline::add_tod_file_dims(
        fo, tod_file_counts.n_output_scans,
        tod_file_counts.n_raw_scan_indices, tod_file_counts.n_dets);
    netCDF::NcDim n_pts_dim = tod_dims.n_pts;
    netCDF::NcDim n_scans_dim = tod_dims.n_scans;
    netCDF::NcDim n_dets_dim = tod_dims.n_dets;
    std::vector<netCDF::NcDim> dims = tod_dims.signal;
    std::vector<netCDF::NcDim> raw_scans_dims = tod_dims.raw_scans;
    std::vector<netCDF::NcDim> scans_dims = tod_dims.scans;

    citlali::pipeline::add_tod_scan_index_placeholders(
        fo, raw_scans_dims, scans_dims, n_scans_dim,
        tod_file_counts.n_output_scans, tod_file_counts.n_raw_scan_indices,
        tod_output_outer, citlali::pipeline::tod_output_fill_int());

    citlali::pipeline::add_tod_filter_edge_guard_scan_placeholders(
        fo, n_scans_dim, tod_file_counts.n_output_scans,
        citlali::pipeline::tod_output_fill_int(),
        citlali::pipeline::tod_output_fill_double());

    const auto tod_chunking = citlali::pipeline::tod_data_chunking(
        telescope.scan_indices, tod_file_counts.n_dets);
    const auto chunkMode = tod_chunking.mode;
    const auto &chunkSizes = tod_chunking.sizes;

    citlali::pipeline::add_tod_core_data_vars(
        fo, dims, tod_output_mini, omb.sig_unit, rtcproc.run_kernel,
        telescope.pixel_axes, chunkMode, chunkSizes);

    citlali::pipeline::add_tod_static_metadata_vars(
        fo, calib.apt, calib.apt_header_units, telescope.tel_data,
        pointing_offsets_arcsec, logger, n_dets_dim, n_pts_dim, chunkMode,
        chunkSizes);

    if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
        const int fill_int = citlali::pipeline::rtcdiag_fill_int();
        const double fill_double = citlali::pipeline::rtcdiag_fill_double();
        const double rtc_stream_fsmp =
            citlali::pipeline::rtc_tod_stream_sample_rate(
                rtcproc, telescope.fsmp, telescope.d_fsmp);
        citlali::pipeline::add_rtcdiag_tod_stream_diag(
            fo, calib, rtcproc, n_scans_dim, n_dets_dim,
            n_tod_output_scans_for_stream, rtc_stream_fsmp,
            fill_int, fill_double);
    }

    // add weights
    if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
        std::vector<netCDF::NcDim> weight_dims = {n_scans_dim, n_dets_dim};
        citlali::pipeline::add_ptc_weights_var(fo, weight_dims, omb.sig_unit);
        const int ptc_stream_fill_int = citlali::pipeline::ptcdiag_fill_int();
        const double ptc_stream_fill_double =
            citlali::pipeline::ptcdiag_fill_double();

        citlali::pipeline::add_ptcdiag_tod_optional_diag(
            fo, calib, ptcproc, dims, chunkMode, chunkSizes,
            n_scans_dim, n_dets_dim, n_tod_output_scans_for_stream,
            ptc_stream_fill_int, ptc_stream_fill_double);
    }

    citlali::pipeline::add_tod_hwpr_var_if_requested(
        fo, rtcproc.run_polarization, calib.run_hwpr, n_pts_dim);

    // add tel header
    citlali::pipeline::add_telescope_header_vars(fo, telescope.tel_header);

    });
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
    double omb_size = citlali::pipeline::map_buffer_memory_gb(omb);

    logger->info("estimated size of map buffer {:.2f} GB", omb_size);

    mb_size_total = mb_size_total + omb_size;

    // print info if coadd is requested
    if (run_coadd) {
        logger->info("coadd map buffer rows: {}", cmb.n_rows);
        logger->info("coadd map buffer cols: {}", cmb.n_cols);

        // make a rough estimate of memory usage for coadd map buffer
        double cmb_size = citlali::pipeline::map_buffer_memory_gb(cmb);

        logger->info("estimated size of coadd buffer {:.2f} GB", cmb_size);

        mb_size_total = mb_size_total + cmb_size;

        // output info if coadd noise maps are requested
        if (run_noise) {
            logger->info("coadd map buffer noise maps: {}", cmb.n_noise);
            // make a rough estimate of memory usage for coadd noise maps
            double nmb_size = citlali::pipeline::noise_buffer_memory_gb(cmb);
            logger->info("estimated size of noise buffer {:.2f} GB", nmb_size);
            mb_size_total = mb_size_total + nmb_size;
        }
    }
    else {
        // output info if obs noise maps are requested
        if (run_noise) {
            logger->info("observation map buffer noise maps: {}", omb.n_noise);
            // make a rough estimate of memory usage for obs noise maps
            double nmb_size = citlali::pipeline::noise_buffer_memory_gb(omb);
            logger->info("estimated size of noise buffer {:.2f} GB", nmb_size);
            mb_size_total = mb_size_total + nmb_size;
        }
    }

    logger->info("estimated size of all maps {:.2f} GB", mb_size_total);
    logger->info("number of scans: {}",telescope.scan_indices.cols());
    if (run_tod_output) {
        if (citlali::pipeline::should_report_rtc_tod_output(tod_output_type)) {
            citlali::pipeline::log_rtc_tod_output_summary(
                logger, n_tod_output_scans_rtc, rtcproc.tod_output_mini,
                rtcproc.tod_output_outer);
        }
        if (citlali::pipeline::should_report_ptc_tod_output(tod_output_type)) {
            citlali::pipeline::log_ptc_tod_output_summary(
                logger, n_tod_output_scans_ptc, ptcproc.tod_output_mini);
        }
    }
    citlali::pipeline::log_diagnostics_sidecar_summary(logger);

    // test getting memory usage for fun
    /*struct sysinfo memInfo;
    long long totalPhysMem = memInfo.totalram;
    totalPhysMem *= memInfo.mem_unit;

    logger->info("total physical memory available {} GB", (totalPhysMem/1024)/1e7);*/
    auto phys_memory_kb = engine_utils::get_phys_memory();
    citlali::pipeline::log_physical_memory_summary(logger, phys_memory_kb);
}

template <TCDataKind tc_t>
void Engine::write_chunk_summary(TCData<tc_t, Eigen::MatrixXd> &in) {

    logger->debug("writing summary files for chunk {}",in.index.data);

    const auto filename =
        citlali::pipeline::chunk_summary_filename(in.index.data);

    // write summary log file
    std::ofstream f;
    f.open(citlali::pipeline::summary_log_path(obsnum_dir_name, filename));

    f << "Summary file for scan " << in.index.data << "\n";
    citlali::pipeline::write_pipeline_version_summary(
        f, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION);
    citlali::pipeline::write_chunk_time_summary(
        f, in.creation_time, engine_utils::current_date_time());

    citlali::pipeline::write_chunk_identity_summary(
        f, redu_type, tod_type, omb.sig_unit, in.name);

    citlali::pipeline::write_chunk_processing_status_summary(f, in.status);
    citlali::pipeline::write_chunk_tod_filter_summary(
        f, rtcproc, telescope.outer_scans_chunk);
    citlali::pipeline::write_chunk_ptc_model_line_audit_summary(
        f, rtcproc.line_audit);
    citlali::pipeline::write_chunk_scan_shape_summary(
        f, in.scans.data.rows(), in.scans.data.cols());
    citlali::pipeline::write_chunk_detector_flag_summary(
        f, (calib.apt["flag"].array()!=0).count(), in.n_dets_low,
        in.n_dets_high, in.scans.data.cols());

    citlali::pipeline::write_chunk_nonfinite_summary(f, in.scans.data);
    citlali::pipeline::write_chunk_data_stat_summary(
        f, in.scans.data.minCoeff(), in.scans.data.maxCoeff(),
        in.scans.data.mean(), tula::alg::median(in.scans.data),
        engine_utils::calc_std_dev(in.scans.data), omb.sig_unit);

    citlali::pipeline::write_chunk_kernel_summary_if_generated(
        f, in.status.kernel_generated, in.kernel, omb.sig_unit);

    f.close();
}

template <typename map_buffer_t>
void Engine::write_map_summary(map_buffer_t &mb) {

    logger->debug("writing map summary files");

    const auto filename = citlali::pipeline::map_summary_filename();
    std::ofstream f;
    f.open(citlali::pipeline::summary_log_path(obsnum_dir_name, filename));

    f << "Summary file for maps\n";
    citlali::pipeline::write_pipeline_version_summary(
        f, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION);
    citlali::pipeline::write_file_time_summary(
        f, engine_utils::current_date_time());

    citlali::pipeline::write_map_identity_summary(
        f, redu_type, tod_type, map_grouping, mb.n_rows, mb.n_cols, n_maps,
        mb.sig_unit);
    citlali::pipeline::write_map_product_presence_summary(f, mb);

    const auto nonfinite_counts =
        citlali::pipeline::count_map_summary_nonfinite(mb);
    citlali::pipeline::write_map_nonfinite_summary(f, nonfinite_counts);
}

template <mapmaking::MapType map_t, engine_utils::toltecIO::DataType data_t, engine_utils::toltecIO::ProdType prod_t>
auto Engine::setup_filenames(std::string dir_name) {
    return citlali::pipeline::map_output_filename<map_t, data_t, prod_t>(
        toltec_io, dir_name, redu_type, obsnum, telescope.sim_obs);
}

auto Engine::get_map_name(int i) {
    return citlali::pipeline::map_layer_name(i, map_grouping, calib);
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

    const auto array_id = citlali::pipeline::phdu_array_id(calib.arrays, i);

    // array name
    std::string name = citlali::pipeline::phdu_array_name(
        toltec_io.array_name_map, array_id);
    auto &fits_entry = fits_io->at(i);

    try {
    logger->debug("adding unit conversions");

    // conversion to Rayleigh-Jeans uK brightness temperature
    auto fwhm = citlali::pipeline::mean_beam_fwhm_arcsec(
        calib.array_fwhms[array_id]);
    auto mJy_beam_to_uK = engine_utils::mJy_beam_to_uK(
        1, toltec_io.array_freq_map[array_id], fwhm);

    // beam area in steradians
    auto beam_area_rad = citlali::pipeline::gaussian_beam_area_sr(
        fwhm, FWHM_TO_STD, ASEC_TO_RAD, pi);
    // get Jy/pixel
    auto mJy_beam_to_Jy_px =
        citlali::pipeline::mjy_beam_to_jy_pixel_factor(
            beam_area_rad, mb->pixel_size_rad);

    auto get_tel_header_scalar = [&](const std::string &key, double fallback) {
        return citlali::pipeline::telescope_header_scalar(
            telescope.tel_header, key, fallback, logger);
    };

    auto get_tel_data_mean = [&](const std::string &key, double fallback) {
        return citlali::pipeline::telescope_data_mean(
            telescope.tel_data, key, fallback, logger);
    };

    auto add_double_key = [&](const std::string &key, double value, const std::string &comment,
                              double fallback = 0.0) {
        citlali::pipeline::add_phdu_double_key(
            fits_entry, name, logger, key, value, comment, fallback);
    };

    // add unit conversions
    citlali::pipeline::add_phdu_unit_conversion_config(
        fits_entry, name, logger, rtcproc.run_calibrate, mb->sig_unit,
        calib.array_beam_areas[array_id]*MJY_SR_TO_mJY_ASEC,
        mJy_beam_to_uK, mJy_beam_to_Jy_px);

    // add source flux for beammaps
    if (redu_type == "beammap") {
        citlali::pipeline::add_phdu_beammap_source_flux(
            fits_entry, name, logger, beammap_fluxes_mJy_beam[name],
            beammap_fluxes_MJy_Sr[name]);

        citlali::pipeline::add_phdu_beammap_tuning(
            fits_entry, name, logger, beammap_iter_tolerance,
            beammap_convergence_radius_arcsec, beammap_iter_max,
            beammap_phase_split_enabled, beammap_locator_iter,
            beammap_measurement_start_iter, beammap_derotate);
        // add reference detector information
        citlali::pipeline::BeammapReferenceHeaderValues reference_values;
        if (beammap_subtract_reference) {
            reference_values =
                citlali::pipeline::beammap_reference_header_values(
                    calib, beammap_reference_det);
        }
        citlali::pipeline::add_phdu_beammap_reference(
            fits_entry, name, logger, beammap_subtract_reference,
            reference_values);
    }

    logger->debug("adding obsnums");

    // add obsnums
    citlali::pipeline::add_phdu_obsnum_keys(fits_entry, mb->obsnums);

    // add date and time of obs
    citlali::pipeline::add_phdu_date_obs_keys(
        fits_entry, mb->obsnums, date_obs);

    logger->debug("adding obs info");

    citlali::pipeline::add_phdu_pipeline_identity_keys(
        fits_entry, telescope.source_name, calib.run_hwpr, name,
        CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION, TULA_GIT_VERSION,
        telescope.project_id, redu_type, telescope.obs_goal, tod_type,
        map_grouping, map_method);
    const double source_ra = get_tel_header_scalar("Header.Source.Ra", 0.0);
    const double source_dec = get_tel_header_scalar("Header.Source.Dec", 0.0);
    citlali::pipeline::add_phdu_map_geometry_keys(
        fits_entry, name, logger, mb->exposure_time, telescope.pixel_axes,
        source_ra, source_dec, RAD_TO_DEG*get_tel_data_mean("TelElAct", 0.0),
        RAD_TO_DEG*get_tel_data_mean("TelAzAct", 0.0),
        RAD_TO_DEG*get_tel_data_mean("ActParAng", 0.0));

    logger->debug("adding beamsizes");

    // add beamsizes
    citlali::pipeline::add_phdu_beam_geometry_keys(
        fits_entry, name, logger, calib.array_fwhms[array_id],
        calib.array_pas[array_id], RAD_TO_DEG, pi/2);

    citlali::pipeline::add_phdu_auxiliary_scalar_keys(
        fits_entry, mb->sig_unit, telescope.fsmp, fruit_iter);

    // add jinc shape params
    if (map_method=="jinc") {
        logger->debug("adding jinc params");

        citlali::pipeline::add_phdu_jinc_shape_keys(
            fits_entry, name, logger, jinc_mm.r_max,
            jinc_mm.shape_params[array_id]);
    }

    // add mean tau
    logger->debug("adding extinction");
    const double mean_tau = citlali::pipeline::phdu_mean_tau(
        rtcproc, telescope, calib, i, logger);
    add_double_key("MEAN_TAU", mean_tau, "mean tau (" + name + ")");

    citlali::pipeline::add_phdu_apt_key_if_single_observation(
        fits_entry, mb->obsnums, calib.apt_filepath, logger);

    const double rms = citlali::pipeline::phdu_oof_rms(
        mb, i, redu_type, name, fits_io->at(i).filepath, logger);

    // out-of-focus holography parameters
    if (! telescope.sim_obs) {
        logger->debug("adding oof params");
        citlali::pipeline::add_phdu_oof_keys(
            fits_entry, name, logger, rms, mb->sig_unit,
            toltec_io.array_wavelength_map[array_id]/1000.,
            static_cast<int>(toltec_io.array_wavelength_map[array_id]*1000),
            get_tel_header_scalar("Header.M2.XReq", 0.0)/1000.*1e6,
            get_tel_header_scalar("Header.M2.YReq", 0.0)/1000.*1e6,
            get_tel_header_scalar("Header.M2.ZReq", 0.0)/1000.*1e6);
    }
    // add control/runtime parameters
    logger->debug("adding config params");
    const bool run_any_tod_filter =
        citlali::pipeline::phdu_any_tod_filter_enabled(rtcproc);
    citlali::pipeline::add_phdu_initial_runtime_config(
        fits_entry, verbose_mode, rtcproc.run_polarization,
        rtcproc.run_despike);
    citlali::pipeline::add_phdu_rtc_local_despike_config(
        fits_entry, name, logger, rtcproc.despiker.local_residual);
    citlali::pipeline::add_phdu_tod_filter_runtime_config(
        fits_entry, name, logger, rtcproc, run_any_tod_filter);
    citlali::pipeline::add_phdu_tod_edge_guard_config(
        fits_entry, rtcproc.filter_edge_guard, telescope.outer_scans_chunk);
    citlali::pipeline::add_phdu_tod_processing_config(fits_entry, rtcproc);
    citlali::pipeline::add_phdu_weight_selection_config(
        fits_entry, name, logger, ptcproc, rtcproc);
    citlali::pipeline::add_phdu_rtc_event_mask_config(
        fits_entry, name, logger, rtcproc);
    citlali::pipeline::add_phdu_reduction_learning_config(
        fits_entry, name, logger, reduction_learning);
    citlali::pipeline::add_phdu_weight_corr_penalty_config(
        fits_entry, name, logger, ptcproc.weight_corr_penalty);
    citlali::pipeline::add_phdu_busy_row_suppression_config(
        fits_entry, name, logger, ptcproc.busy_row_suppression);
    const auto n_eig_removed =
        ptcproc.run_clean ? ptcproc.cleaner.n_eig_to_cut[array_id].sum()
                          : 0;
    citlali::pipeline::add_phdu_cleaner_config(
        fits_entry, name, logger, ptcproc, n_eig_removed);

    const double fruit_loops_flux_limit =
        citlali::pipeline::phdu_fruit_loop_flux_limit(
            ptcproc, calib.arrays, i, array_id);
    citlali::pipeline::add_phdu_fruit_loops_config(
        fits_entry, name, logger, ptcproc, fruit_loops_flux_limit,
        mb->sig_unit);

    if (redu_type == "pointing") {
        citlali::pipeline::add_phdu_pointing_config(
            fits_entry, name, logger, pointing_source_strategy,
            pointing_fit_gaussian_enabled, pointing_fruitloops_center_mode,
            pointing_header_center_max_radius_arcsec,
            pointing_header_center_require_coverage);
    }

    citlali::pipeline::add_phdu_telescope_header_keys_if_single_observation(
        fits_entry, mb->obsnums, name, logger, telescope.tel_header);
    } catch (const CCfits::FitsError &e) {
        throw std::runtime_error(
            fmt::format("failed to add PHDU/header for array '{}' (file={}): {}",
                        name, fits_io->at(i).filepath, e.message()));
    } catch (const std::exception &e) {
        throw std::runtime_error(
            fmt::format("failed to add PHDU/header for array '{}' (file={}): {}",
                        name, fits_io->at(i).filepath, e.what()));
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
    const Eigen::Index array_index = maps_to_arrays(i);
    if (array_index < 0 || array_index >= calib.arrays.size()) {
        logger->error("write_maps maps_to_arrays index out of range: maps_to_arrays(i)={} calib.arrays.size={} map_i={}",
                      static_cast<long long>(array_index),
                      static_cast<long long>(calib.arrays.size()),
                      static_cast<long long>(i));
        std::exit(EXIT_FAILURE);
    }

    const double source_epoch =
        citlali::pipeline::wcs_source_epoch_or_default(telescope.tel_header,
                                                       logger);

    // update wcs ctypes for frequency and stokes params
    mb->wcs.crval[2] =
        citlali::pipeline::map_wcs_frequency(toltec_io.array_freq_map,
                                             calib.arrays, array_index);
    mb->wcs.crval[3] = stokes_index;
    const std::string &stokes_suffix = rtcproc.polarization.stokes_params[stokes_index];

    try {
        auto add_map_hdu_with_wcs = [&](const std::string &hdu_name, auto &data) {
            fits_io->at(map_index).add_hdu(hdu_name, data);
            fits_io->at(map_index).add_wcs(
                fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
        };

        // signal map
        add_map_hdu_with_wcs(
            citlali::pipeline::signal_map_hdu_name(map_name, stokes_suffix),
            mb->signal[i]);
        citlali::pipeline::add_signal_map_metadata(
            *fits_io->at(map_index).hdus.back(), mb->sig_unit);

        // weight map
        add_map_hdu_with_wcs(
            citlali::pipeline::weight_map_hdu_name(map_name, stokes_suffix),
            mb->weight[i]);
        const std::string weight_unit =
            citlali::pipeline::map_weight_unit(mb->sig_unit);
        const bool empirical_weight_calibration =
            citlali::pipeline::empirical_weight_calibration_enabled(
                run_noise_products, run_noise,
                apply_empirical_noise_weights);
        citlali::pipeline::add_weight_map_metadata(
            *fits_io->at(map_index).hdus.back(), weight_unit,
            empirical_weight_calibration);
        if (i < mb->noise_weight_scale.size()) {
            citlali::pipeline::add_empirical_weight_scale_key(
                *fits_io->at(map_index).hdus.back(), mb->noise_weight_scale(i));
        }
        if (i < mb->noise_weight_median_ratio.size()) {
            citlali::pipeline::add_weight_variance_median_key(
                *fits_io->at(map_index).hdus.back(),
                mb->noise_weight_median_ratio(i));
        }
        const bool is_beammap = redu_type == "beammap";
        const double median_err_value = mb->median_err(i);
        const double median_err =
            citlali::pipeline::map_median_error_or_zero(median_err_value,
                                                        is_beammap);
        if (citlali::pipeline::has_negative_map_median_error(
                median_err_value, is_beammap)) {
            logger->warn("negative median_err for map {} in {}; using 0", map_name,
                         fits_io->at(map_index).filepath);
        }
        citlali::pipeline::add_image_median_error_key(
            *fits_io->at(map_index).hdus.back(), median_err, mb->sig_unit);

        if (citlali::pipeline::has_map_image_slot(
                mb->weight_formal, i, mb->n_rows, mb->n_cols)) {
            add_map_hdu_with_wcs(
                citlali::pipeline::formal_weight_map_hdu_name(
                    map_name, stokes_suffix),
                mb->weight_formal[i]);
            citlali::pipeline::add_formal_weight_map_metadata(
                *fits_io->at(map_index).hdus.back(), weight_unit);
        }

        if (citlali::pipeline::has_map_image_slot(
                mb->noise_variance, i, mb->n_rows, mb->n_cols)) {
            add_map_hdu_with_wcs(
                citlali::pipeline::noise_variance_map_hdu_name(
                    map_name, stokes_suffix),
                mb->noise_variance[i]);
            const std::string variance_unit =
                citlali::pipeline::map_variance_unit(mb->sig_unit);
            citlali::pipeline::add_noise_variance_map_metadata(
                *fits_io->at(map_index).hdus.back(), variance_unit);
        }

        // kernel map
        if (rtcproc.run_kernel) {
            fits_io->at(map_index).add_hdu(
                citlali::pipeline::kernel_map_hdu_name(map_name, stokes_suffix),
                mb->kernel[i]);
            citlali::pipeline::add_image_type_key(
                *fits_io->at(map_index).hdus.back(), rtcproc.kernel.type,
                citlali::pipeline::kernel_type_comment());

            double fwhm = citlali::pipeline::kernel_fwhm_arcsec(
                rtcproc.kernel.type, rtcproc.kernel.fwhm_rad,
                calib.array_fwhms[calib.arrays(i)], RAD_TO_ASEC);
            if (citlali::pipeline::has_nonfinite_kernel_fwhm(fwhm)) {
                logger->warn("non-finite kernel FWHM for map {} in {}; using -99", map_name,
                             fits_io->at(map_index).filepath);
                fwhm = citlali::pipeline::invalid_kernel_fwhm_arcsec();
            }
            citlali::pipeline::add_kernel_fwhm_key(
                *fits_io->at(map_index).hdus.back(), fwhm);
            fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
            citlali::pipeline::add_kernel_map_metadata(
                *fits_io->at(map_index).hdus.back(), mb->sig_unit);
        }

        // coverage map
        if (!mb->coverage.empty()) {
            add_map_hdu_with_wcs(
                citlali::pipeline::coverage_map_hdu_name(
                    map_name, stokes_suffix),
                mb->coverage[i]);
            citlali::pipeline::add_coverage_map_metadata(
                *fits_io->at(map_index).hdus.back());
        }

        /* coverage bool and signal-to-noise maps */
        if (!mb->coverage.empty()) {
            // get weight threshold for current map
            auto cov_region = mb->calc_cov_region(i);
            auto weight_threshold = std::get<0>(cov_region);
            if (citlali::pipeline::has_nonfinite_weight_threshold(
                    weight_threshold)) {
                logger->warn("non-finite weight threshold for map {} in {}; using 0", map_name,
                             fits_io->at(map_index).filepath);
            }
            weight_threshold =
                citlali::pipeline::weight_threshold_or_zero(weight_threshold);
            Eigen::MatrixXd coverage_bool =
                citlali::pipeline::coverage_mask_from_weight(
                    mb->weight[i], weight_threshold);

            // coverage bool map
            add_map_hdu_with_wcs(
                citlali::pipeline::coverage_mask_map_hdu_name(
                    map_name, stokes_suffix),
                coverage_bool);
            citlali::pipeline::add_coverage_mask_map_metadata(
                *fits_io->at(map_index).hdus.back());
            citlali::pipeline::add_image_weight_threshold_key(
                *fits_io->at(map_index).hdus.back(), weight_threshold);

            // legacy signal-to-noise map name retained for compatibility; this is pixel S/N.
            Eigen::MatrixXd sig2noise =
                citlali::pipeline::pixel_snr_image_or_fallback(
                    mb->sig2noise_pixel, i, mb->n_rows, mb->n_cols,
                    mb->signal[i], mb->weight[i]);
            add_map_hdu_with_wcs(
                citlali::pipeline::legacy_pixel_snr_map_hdu_name(
                    map_name, stokes_suffix),
                sig2noise);
            citlali::pipeline::add_legacy_pixel_snr_map_metadata(
                *fits_io->at(map_index).hdus.back());

            add_map_hdu_with_wcs(
                citlali::pipeline::pixel_snr_map_hdu_name(
                    map_name, stokes_suffix),
                sig2noise);
            citlali::pipeline::add_pixel_snr_map_metadata(
                *fits_io->at(map_index).hdus.back());

            const bool is_filtered_output =
                citlali::pipeline::is_filtered_map_output(
                    fits_io, filtered_fits_io_vec, filtered_coadd_fits_io_vec);
            if (is_filtered_output &&
                citlali::pipeline::has_map_image_slot(
                    mb->point_source_uncertainty, i, mb->n_rows,
                    mb->n_cols)) {
                add_map_hdu_with_wcs(
                    citlali::pipeline::point_source_flux_map_hdu_name(
                        map_name, stokes_suffix),
                    mb->signal[i]);
                citlali::pipeline::add_point_source_flux_map_metadata(
                    *fits_io->at(map_index).hdus.back(), mb->sig_unit);
                citlali::pipeline::add_point_source_response_norm_key(
                    *fits_io->at(map_index).hdus.back(), 1.0);

                add_map_hdu_with_wcs(
                    citlali::pipeline::point_source_uncertainty_map_hdu_name(
                        map_name, stokes_suffix),
                    mb->point_source_uncertainty[i]);
                citlali::pipeline::add_point_source_uncertainty_map_metadata(
                    *fits_io->at(map_index).hdus.back(), mb->sig_unit);

                add_map_hdu_with_wcs(
                    citlali::pipeline::point_source_snr_map_hdu_name(
                        map_name, stokes_suffix),
                    mb->sig2noise_point_source[i]);
                citlali::pipeline::add_point_source_snr_map_metadata(
                    *fits_io->at(map_index).hdus.back());
            }
        }

        // write noise maps
        if (citlali::pipeline::should_write_noise_maps(mb->noise,
                                                       noise_fits_io)) {
            if (!citlali::pipeline::has_noise_fits_slot(noise_fits_io,
                                                        map_index)) {
                logger->error("write_maps noise file index out of range: map_index={} noise_fits_io_size={} map_i={}",
                              static_cast<long long>(map_index),
                              static_cast<long long>(noise_fits_io->size()),
                              static_cast<long long>(i));
                std::exit(EXIT_FAILURE);
            }
            if (!citlali::pipeline::has_noise_map_slot(mb->noise, i)) {
                logger->error("write_maps noise map index out of range: i={} noise_size={}",
                              static_cast<long long>(i), static_cast<long long>(mb->noise.size()));
                std::exit(EXIT_FAILURE);
            }
            const double median_rms =
                citlali::pipeline::map_median_rms_or_zero(mb->median_rms, i);
            if (citlali::pipeline::has_nonfinite_map_median_rms(
                    mb->median_rms, i)) {
                logger->warn("non-finite median_rms for map {} in {}; using 0", map_name,
                             noise_fits_io->at(map_index).filepath);
            }
            auto add_noise_map_hdu_with_wcs = [&](const std::string &hdu_name, auto &data) {
                noise_fits_io->at(map_index).add_hdu(hdu_name, data);
                noise_fits_io->at(map_index).add_wcs(
                    noise_fits_io->at(map_index).hdus.back(), mb->wcs,
                    source_epoch);
            };
            for (Eigen::Index n=0; n<mb->n_noise; ++n) {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(mb->noise[i].data() + n * mb->n_rows * mb->n_cols,
                                                                                               mb->n_rows, mb->n_cols);

                add_noise_map_hdu_with_wcs(
                    citlali::pipeline::noise_signal_map_hdu_name(
                        map_name, n, stokes_suffix),
                    noise_matrix);
                citlali::pipeline::add_noise_image_summary_keys(
                    *noise_fits_io->at(map_index).hdus.back(), mb->sig_unit,
                    median_rms);
            }
        }
    } catch (const CCfits::FitsError &e) {
        throw std::runtime_error(
            fmt::format("failed to write map '{}' (map_i={} file={} noise_file={}): {}",
                        map_name,
                        static_cast<long long>(i),
                        fits_io->at(map_index).filepath,
                        citlali::pipeline::noise_file_path_or_na(
                            mb->noise, noise_fits_io, map_index),
                        e.message()));
    } catch (const std::exception &e) {
        throw std::runtime_error(
            fmt::format("failed to write map '{}' (map_i={} file={} noise_file={}): {}",
                        map_name,
                        static_cast<long long>(i),
                        fits_io->at(map_index).filepath,
                        citlali::pipeline::noise_file_path_or_na(
                            mb->noise, noise_fits_io, map_index),
                        e.what()));
    }
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_psd(map_buffer_t &mb, std::string dir_name) {
    // get filename
    const std::string filename =
        setup_filenames<map_t, engine_utils::toltecIO::toltec,
                        engine_utils::toltecIO::psd>(dir_name);

    write_netcdf_atomic(filename + ".nc", [&](netCDF::NcFile &fo) {

    // loop through psd vector
    for (Eigen::Index i = 0; i < mb->psds.size(); ++i) {
        const std::string map_name = get_map_name(i);

        const Eigen::Index map_index = arrays_to_maps(i);
        const Eigen::Index stokes_index = maps_to_stokes(i);

        const std::string name = citlali::pipeline::spectral_product_name(
            toltec_io.array_name_map, calib.arrays,
            rtcproc.polarization.stokes_params, map_name, map_index,
            stokes_index);

        citlali::pipeline::add_spectral_psd_product(
            fo, mb->noise, name, mb->psds, mb->psd_freqs, mb->psd_2ds,
            mb->psd_2d_freqs, mb->noise_psds, mb->noise_psd_freqs,
            mb->noise_psd_2ds, mb->noise_psd_2d_freqs, i);
    }
    });
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_hist(map_buffer_t &mb, std::string dir_name) {
    const std::string filename =
        setup_filenames<map_t, engine_utils::toltecIO::toltec,
                        engine_utils::toltecIO::hist>(dir_name);

    write_netcdf_atomic(filename + ".nc", [&](netCDF::NcFile &fo) {
    netCDF::NcDim hist_bins_dim =
        citlali::pipeline::add_spectral_histogram_bins_dim(
            fo, mb->hist_n_bins);

    // loop through stored histograms
    for (Eigen::Index i = 0; i < mb->hists.size(); ++i) {
        const std::string map_name = get_map_name(i);

        const Eigen::Index map_index = arrays_to_maps(i);
        const Eigen::Index stokes_index = maps_to_stokes(i);

        const std::string name = citlali::pipeline::spectral_product_name(
            toltec_io.array_name_map, calib.arrays,
            rtcproc.polarization.stokes_params, map_name, map_index,
            stokes_index);

        citlali::pipeline::add_spectral_histogram_product(
            fo, mb->noise, name, hist_bins_dim, mb->hist_bins, mb->hists,
            mb->noise_hists, i);
    }
    });
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_mapdiag(map_buffer_t &mb, std::string dir_name) {
    const std::string filename =
        setup_filenames<map_t, engine_utils::toltecIO::toltec,
                        engine_utils::toltecIO::mapdiag>(dir_name);
    const auto mapdiag_context = citlali::pipeline::make_mapdiag_size_context(
        static_cast<std::size_t>(n_maps),
        std::max<std::size_t>(1, mb->obsnums.size()),
        map_t == mapmaking::RawCoadd || map_t == mapmaking::FilteredCoadd);
    const double fill_double = citlali::pipeline::mapdiag_fill_double();
    const int fill_int = citlali::pipeline::mapdiag_fill_int();
    const auto n_mapdiag_maps = mapdiag_context.n_maps;

    std::vector<std::string> array_names(n_mapdiag_maps);
    std::vector<std::string> stokes_names(n_mapdiag_maps);
    std::vector<std::string> map_names(n_mapdiag_maps);
    std::vector<double> median_err(n_mapdiag_maps, fill_double);
    std::vector<double> median_rms(n_mapdiag_maps, fill_double);
    std::vector<double> weight_thresholds(n_mapdiag_maps, fill_double);
    std::vector<double> weight_sum(n_mapdiag_maps, fill_double);
    std::vector<double> core_weight_sum(n_mapdiag_maps, fill_double);
    std::vector<double> coverage_sum(n_mapdiag_maps, fill_double);
    std::vector<double> coverage_max(n_mapdiag_maps, fill_double);
    std::vector<double> coverage_median_core(n_mapdiag_maps, fill_double);
    citlali::pipeline::MapdiagCoverageRefs coverage_refs{
        coverage_sum,
        coverage_max,
        coverage_median_core};
    std::vector<double> empirical_to_formal_noise_ratio(
        n_mapdiag_maps, fill_double);
    citlali::pipeline::MapdiagFormalNoiseRefs formal_noise_refs{
        median_err,
        median_rms,
        empirical_to_formal_noise_ratio};
    std::vector<double> noise_weight_median_ratio(n_mapdiag_maps, fill_double);
    std::vector<double> noise_weight_scale(n_mapdiag_maps, fill_double);
    std::vector<double> noise_products_s2n_sigma(n_mapdiag_maps, fill_double);
    std::vector<double> noise_products_valid_pixels(
        n_mapdiag_maps, fill_double);
    citlali::pipeline::MapdiagNoiseProductRefs noise_product_refs{
        noise_weight_median_ratio,
        noise_weight_scale,
        noise_products_s2n_sigma,
        noise_products_valid_pixels};
    std::vector<double> peak_signal(n_mapdiag_maps, fill_double);
    std::vector<double> peak_abs_sig2noise(n_mapdiag_maps, fill_double);
    std::vector<double> core_peak_abs_sig2noise(n_mapdiag_maps, fill_double);
    std::vector<double> noise_rms_p16(n_mapdiag_maps, fill_double);
    std::vector<double> noise_rms_p84(n_mapdiag_maps, fill_double);
    std::vector<double> core_tail_frac_abs3(n_mapdiag_maps, fill_double);
    std::vector<double> core_tail_frac_pos3(n_mapdiag_maps, fill_double);
    std::vector<double> core_tail_frac_neg3(n_mapdiag_maps, fill_double);
    std::vector<double> core_tail_excess_abs3(n_mapdiag_maps, fill_double);
    std::vector<double> core_tail_excess_pos3(n_mapdiag_maps, fill_double);
    std::vector<double> core_tail_excess_neg3(n_mapdiag_maps, fill_double);
    std::vector<double> core_sig2noise_skew(n_mapdiag_maps, fill_double);
    citlali::pipeline::MapdiagCoreTailRefs core_tail_refs{
        core_tail_frac_abs3,
        core_tail_frac_pos3,
        core_tail_frac_neg3,
        core_tail_excess_abs3,
        core_tail_excess_pos3,
        core_tail_excess_neg3,
        core_sig2noise_skew};
    std::vector<double> noise_tail_frac_abs3(n_mapdiag_maps, fill_double);
    std::vector<double> noise_tail_frac_pos3(n_mapdiag_maps, fill_double);
    std::vector<double> noise_tail_frac_neg3(n_mapdiag_maps, fill_double);
    std::vector<double> noise_tail_excess_abs3(n_mapdiag_maps, fill_double);
    std::vector<double> noise_tail_excess_pos3(n_mapdiag_maps, fill_double);
    std::vector<double> noise_tail_excess_neg3(n_mapdiag_maps, fill_double);
    std::vector<double> noise_sig2noise_skew(n_mapdiag_maps, fill_double);
    citlali::pipeline::MapdiagNoiseTailRefs noise_tail_refs{
        noise_rms_p16,
        noise_rms_p84,
        noise_tail_frac_abs3,
        noise_tail_frac_pos3,
        noise_tail_frac_neg3,
        noise_tail_excess_abs3,
        noise_tail_excess_pos3,
        noise_tail_excess_neg3,
        noise_sig2noise_skew};
    std::vector<double> edge_guard_weight_thresholds(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_hits_thresholds(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_background_levels(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_science_frac(n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_support_frac(n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_guardband_rms_pre(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_guardband_rms_post(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_exterior_rms_pre(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_exterior_rms_post(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_exterior_max_abs_pre(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_exterior_max_abs_post(
        n_mapdiag_maps, fill_double);
    citlali::pipeline::MapdiagEdgeGuardDoubleRefs edge_guard_double_refs{
        edge_guard_weight_thresholds,
        edge_guard_hits_thresholds,
        edge_guard_background_levels,
        edge_guard_science_frac,
        edge_guard_support_frac,
        edge_guard_guardband_rms_pre,
        edge_guard_guardband_rms_post,
        edge_guard_exterior_rms_pre,
        edge_guard_exterior_rms_post,
        edge_guard_exterior_max_abs_pre,
        edge_guard_exterior_max_abs_post};
    std::vector<int> n_valid_pixels(n_mapdiag_maps, 0);
    std::vector<int> n_core_pixels(n_mapdiag_maps, 0);
    citlali::pipeline::MapdiagWeightRefs weight_refs{
        weight_sum,
        core_weight_sum,
        n_valid_pixels,
        n_core_pixels};
    std::vector<int> peak_row(n_mapdiag_maps, fill_int);
    std::vector<int> peak_col(n_mapdiag_maps, fill_int);
    citlali::pipeline::MapdiagPeakRefs peak_refs{
        peak_abs_sig2noise,
        core_peak_abs_sig2noise,
        peak_row,
        peak_col};
    std::vector<int> edge_guard_applied(n_mapdiag_maps, 0);
    std::vector<int> edge_guard_support_radius_pix(n_mapdiag_maps, 0);
    std::vector<int> edge_guard_science_npix(n_mapdiag_maps, 0);
    std::vector<int> edge_guard_support_npix(n_mapdiag_maps, 0);
    std::vector<int> edge_guard_guardband_npix(n_mapdiag_maps, 0);
    citlali::pipeline::MapdiagEdgeGuardIntRefs edge_guard_int_refs{
        edge_guard_applied,
        edge_guard_support_radius_pix,
        edge_guard_science_npix,
        edge_guard_support_npix,
        edge_guard_guardband_npix};
    citlali::pipeline::MapdiagMapIntValues map_int_values{
        n_valid_pixels,
        n_core_pixels,
        peak_row,
        peak_col,
        edge_guard_applied,
        edge_guard_support_radius_pix,
        edge_guard_science_npix,
        edge_guard_support_npix,
        edge_guard_guardband_npix};

    const std::size_t obs_table_size =
        citlali::pipeline::mapdiag_obs_table_size(mapdiag_context);
    std::vector<double> obs_weight_sum(obs_table_size, fill_double);
    std::vector<double> obs_weight_frac(obs_table_size, fill_double);
    std::vector<double> obs_core_weight_sum(obs_table_size, fill_double);
    std::vector<double> obs_core_weight_frac(obs_table_size, fill_double);
    std::vector<int> obs_valid_pixels(obs_table_size, fill_int);
    std::vector<int> obs_core_pixels(obs_table_size, fill_int);
    citlali::pipeline::MapdiagObsTableRefs obs_tables{
        obs_weight_sum,
        obs_core_weight_sum,
        obs_valid_pixels,
        obs_core_pixels};
    citlali::pipeline::MapdiagObservationDoubleValues obs_double_values{
        obs_weight_sum,
        obs_weight_frac,
        obs_core_weight_sum,
        obs_core_weight_frac};
    citlali::pipeline::MapdiagObservationIntValues obs_int_values{
        obs_valid_pixels,
        obs_core_pixels};
    citlali::pipeline::MapdiagMapDoubleValues map_double_values{
        median_err,
        median_rms,
        weight_thresholds,
        weight_sum,
        core_weight_sum,
        coverage_sum,
        coverage_max,
        coverage_median_core,
        empirical_to_formal_noise_ratio,
        noise_weight_median_ratio,
        noise_weight_scale,
        noise_products_s2n_sigma,
        noise_products_valid_pixels,
        peak_signal,
        peak_abs_sig2noise,
        core_peak_abs_sig2noise,
        noise_rms_p16,
        noise_rms_p84,
        core_tail_frac_abs3,
        core_tail_frac_pos3,
        core_tail_frac_neg3,
        core_tail_excess_abs3,
        core_tail_excess_pos3,
        core_tail_excess_neg3,
        core_sig2noise_skew,
        noise_tail_frac_abs3,
        noise_tail_frac_pos3,
        noise_tail_frac_neg3,
        noise_tail_excess_abs3,
        noise_tail_excess_pos3,
        noise_tail_excess_neg3,
        noise_sig2noise_skew,
        edge_guard_weight_thresholds,
        edge_guard_hits_thresholds,
        edge_guard_background_levels,
        edge_guard_science_frac,
        edge_guard_support_frac,
        edge_guard_guardband_rms_pre,
        edge_guard_guardband_rms_post,
        edge_guard_exterior_rms_pre,
        edge_guard_exterior_rms_post,
        edge_guard_exterior_max_abs_pre,
        edge_guard_exterior_max_abs_post};

    const std::string stage_name =
        citlali::pipeline::mapdiag_stage_name<map_t>();
    citlali::pipeline::MapdiagMetadataVars mapdiag_metadata{
        {stage_name, mb->name, map_regime, telescope.source_name,
         telescope.project_id, telescope.obs_goal},
        {mb->pixel_size_rad, mb->cov_cut, mb->sig_unit},
        {wiener_filter.edge_guard_enabled,
         wiener_filter.edge_weight_threshold_mode,
         wiener_filter.edge_hits_threshold_mode,
         wiener_filter.edge_fill_mode,
         wiener_filter.edge_taper_mode,
         wiener_filter.edge_hits_core_fraction,
         wiener_filter.edge_guard_radius_fwhm,
         wiener_filter.edge_taper_min_fraction}};
    citlali::pipeline::MapdiagLabelVars mapdiag_labels{
        array_names,
        stokes_names,
        map_names,
        mb->obsnums,
        obsnum,
        date_obs,
        mapdiag_context.n_obsnums};
    citlali::pipeline::MapdiagValueVars mapdiag_values{
        map_double_values,
        map_int_values,
        obs_double_values,
        obs_int_values};

    const citlali::pipeline::MapdiagStatsContext mapdiag_stats{fill_double};
    const std::string mapdiag_record_producer =
        citlali::pipeline::mapdiag_record_producer(stage_name);

    for (Eigen::Index i = 0; i < n_maps; ++i) {
        const std::size_t idx = citlali::pipeline::mapdiag_size_index(i);
        const auto map_index = arrays_to_maps(i);
        const auto stokes_index = maps_to_stokes(i);
        const auto labels = citlali::pipeline::make_mapdiag_map_labels(
            toltec_io.array_name_map[calib.arrays[map_index]],
            rtcproc.polarization.stokes_params[stokes_index],
            get_map_name(i));
        citlali::pipeline::assign_mapdiag_map_labels(
            idx, labels, {array_names, stokes_names, map_names});

        const auto cov_region = mb->calc_cov_region(i);
        auto weight_threshold = std::get<0>(cov_region);
        weight_threshold =
            citlali::pipeline::mapdiag_weight_threshold_or_zero(
                weight_threshold);
        weight_thresholds[idx] = weight_threshold;
        if (citlali::pipeline::mapdiag_has_edge_guard_entry(idx, *mb)) {
            citlali::pipeline::assign_mapdiag_edge_guard_int_entry(
                idx, *mb, edge_guard_int_refs);
            citlali::pipeline::assign_mapdiag_edge_guard_double_entry(
                idx, *mb, edge_guard_double_refs);
        }

        const auto weight_arr = mb->weight[i].array();
        const auto valid_mask =
            citlali::pipeline::mapdiag_valid_weight_mask(weight_arr);
        const auto core_mask =
            citlali::pipeline::mapdiag_core_weight_mask(
                weight_arr, weight_threshold);
        citlali::pipeline::assign_mapdiag_weight_stats(
            idx,
            citlali::pipeline::mapdiag_weight_stats(
                weight_arr, valid_mask, core_mask),
            weight_refs);

        citlali::pipeline::assign_mapdiag_formal_noise_stats(
            idx,
            citlali::pipeline::mapdiag_formal_noise_stats_or_fill(
                mb->median_err, mb->median_rms, i, fill_double),
            formal_noise_refs);
        const auto noise_product_stats =
            citlali::pipeline::mapdiag_noise_product_stats_or_fill(
                mb->noise_weight_median_ratio, mb->noise_weight_scale,
                mb->noise_s2n_sigma, mb->noise_valid_pixels, i,
                fill_double);
        citlali::pipeline::assign_mapdiag_noise_product_stats(
            idx, noise_product_stats, noise_product_refs);

        if (citlali::pipeline::mapdiag_has_coverage_map(
                mb->coverage, i)) {
            citlali::pipeline::assign_mapdiag_coverage_stats(
                idx, mb->coverage[i], core_mask, fill_double,
                coverage_refs);
        }
        peak_signal[idx] = citlali::pipeline::mapdiag_peak_signal_or_fill(
            mb->signal[i], fill_double);
        if (citlali::pipeline::mapdiag_has_signal_weight_samples(
                mb->signal[i], mb->weight[i])) {
            const Eigen::MatrixXd sig2noise =
                citlali::pipeline::mapdiag_sig2noise_image(
                    mb->signal[i], mb->weight[i]);
            citlali::pipeline::assign_mapdiag_peak_stats(
                idx,
                citlali::pipeline::mapdiag_peak_stats(
                    sig2noise, core_mask, n_core_pixels[idx], fill_double),
                peak_refs);
            const auto core_values =
                mapdiag_stats.collect_masked_values(sig2noise, core_mask);
            const auto signal_tail = mapdiag_stats.tail_stats(core_values);
            citlali::pipeline::assign_mapdiag_core_tail_stats(
                idx, signal_tail, core_tail_refs);

            if (citlali::pipeline::mapdiag_outlier_diagnostics_enabled(
                    reduction_learning)) {
                const auto source_distance_context =
                    citlali::pipeline::mapdiag_source_distance_context(
                        mb, RAD_TO_ASEC, fill_double);

                const double protect_radius =
                    citlali::pipeline::mapdiag_source_protect_radius_arcsec(
                        reduction_learning);
                const Eigen::ArrayXXd off_source_core_mask =
                    citlali::pipeline::mapdiag_off_source_core_mask(
                        core_mask, source_distance_context, protect_radius);

                const auto off_source_values =
                    mapdiag_stats.collect_masked_values(
                        sig2noise, off_source_core_mask);
                if (citlali::pipeline::mapdiag_has_enough_off_source_values(
                        off_source_values)) {
                    const auto robust_stats =
                        citlali::pipeline::mapdiag_robust_center_stats(
                            mapdiag_stats, off_source_values);
                    if (citlali::pipeline::
                            mapdiag_has_valid_robust_center_stats(
                                robust_stats)) {
                        auto candidates =
                            citlali::pipeline::make_mapdiag_pixel_candidates();
                        const bool has_contribution_products =
                            citlali::pipeline::
                                mapdiag_has_contribution_products(mb, i);
                        const double ptc_fs_hz = processed_time_chunk_fs_hz();
                        const Eigen::Index n_mapdiag_rows =
                            citlali::pipeline::mapdiag_n_rows(mb);
                        const Eigen::Index n_mapdiag_cols =
                            citlali::pipeline::mapdiag_n_cols(mb);
                        const double min_effective_samples =
                            citlali::pipeline::mapdiag_min_effective_samples(
                                reduction_learning);
                        const double min_abs_z =
                            citlali::pipeline::mapdiag_min_abs_z(
                                reduction_learning);

                        for (Eigen::Index r = 0; r < n_mapdiag_rows; ++r) {
                            for (Eigen::Index c = 0; c < n_mapdiag_cols; ++c) {
                                if (!citlali::pipeline::
                                        mapdiag_mask_pixel_is_selected(
                                            off_source_core_mask, r, c)) {
                                    continue;
                                }

                                const double value =
                                    citlali::pipeline::
                                        mapdiag_matrix_double_value(
                                            mb->signal[i], r, c);
                                const double wt =
                                    citlali::pipeline::
                                        mapdiag_matrix_double_value(
                                            mb->weight[i], r, c);
                                const double sn =
                                    citlali::pipeline::
                                        mapdiag_matrix_double_value(
                                            sig2noise, r, c);
                                if (!citlali::pipeline::
                                        mapdiag_is_valid_outlier_pixel_value(
                                            value, wt, sn)) {
                                    continue;
                                }

                                const double n_eff =
                                    citlali::pipeline::
                                        mapdiag_effective_samples_or_fill(
                                            mb->coverage, i, r, c,
                                            mb->n_rows, mb->n_cols,
                                            ptc_fs_hz, fill_double);
                                if (!citlali::pipeline::
                                        mapdiag_passes_min_effective_samples(
                                            n_eff, min_effective_samples)) {
                                    continue;
                                }

                                const double z =
                                    citlali::pipeline::mapdiag_robust_z(
                                        sn, robust_stats);
                                if (!citlali::pipeline::
                                        mapdiag_passes_min_abs_z(z,
                                                                 min_abs_z)) {
                                    continue;
                                }

                                const double source_distance_arcsec =
                                    citlali::pipeline::
                                        mapdiag_source_distance_arcsec(
                                            r, c, source_distance_context);
                                auto candidate =
                                    citlali::pipeline::
                                        make_mapdiag_map_pixel_candidate(
                                            r, c, value, wt, n_eff, z,
                                            source_distance_arcsec,
                                            fill_int, fill_double);

                                if (has_contribution_products) {
                                    const auto contribution_map_index =
                                        citlali::pipeline::
                                            mapdiag_contribution_map_index(i);
                                    const int uid =
                                        citlali::pipeline::
                                            mapdiag_matrix_value(
                                                mb->contribution_uid[
                                                    contribution_map_index],
                                                r, c);
                                    const double contrib_signal =
                                        citlali::pipeline::
                                            mapdiag_matrix_double_value(
                                                mb->contribution_signal[
                                                    contribution_map_index],
                                                r, c);
                                    const double contrib_weight =
                                        citlali::pipeline::
                                            mapdiag_matrix_double_value(
                                                mb->contribution_weight[
                                                    contribution_map_index],
                                                r, c);
                                    const double contrib_variance_weight =
                                        citlali::pipeline::
                                            mapdiag_matrix_double_value(
                                                mb->contribution_variance_weight[
                                                    contribution_map_index],
                                                r, c);
                                    if (citlali::pipeline::
                                            mapdiag_has_valid_contributor(
                                                uid, fill_int,
                                                contrib_signal)) {
                                        citlali::pipeline::
                                            assign_mapdiag_candidate_contributor_from_products(
                                                candidate, uid,
                                                mb->contribution_scan[
                                                    contribution_map_index],
                                                mb->contribution_sample[
                                                    contribution_map_index],
                                                r, c);
                                        const double total_signal =
                                            citlali::pipeline::
                                                mapdiag_matrix_double_value(
                                                    mb->contribution_total_signal[
                                                        contribution_map_index],
                                                    r, c);
                                        const double total_weight =
                                            citlali::pipeline::
                                                mapdiag_matrix_double_value(
                                                    mb->contribution_total_weight[
                                                        contribution_map_index],
                                                    r, c);
                                        const double total_variance_weight =
                                            citlali::pipeline::
                                                mapdiag_matrix_double_value(
                                                    mb->contribution_total_variance_weight[
                                                        contribution_map_index],
                                                    r, c);
                                        const double remaining_weight =
                                            citlali::pipeline::
                                                mapdiag_remaining_contribution_weight(
                                                    total_weight,
                                                    contrib_weight);
                                        if (citlali::pipeline::
                                                mapdiag_has_full_leave_one_out_inputs(
                                                    total_signal,
                                                    total_weight,
                                                    contrib_weight,
                                                    contrib_variance_weight,
                                                    total_variance_weight,
                                                    remaining_weight)) {
                                            const double loo_value =
                                                citlali::pipeline::
                                                    mapdiag_full_leave_one_out_value(
                                                        total_signal,
                                                        contrib_signal,
                                                        remaining_weight);
                                            citlali::pipeline::
                                                mapdiag_assign_leave_one_out_z(
                                                    value, wt, loo_value,
                                                    candidate.leave_one_out_z);
                                        }
                                        else if (citlali::pipeline::
                                                     mapdiag_has_fallback_leave_one_out_inputs(
                                                         wt, contrib_weight)) {
                                            const double raw_sum =
                                                citlali::pipeline::
                                                    mapdiag_raw_weighted_signal(
                                                        value, wt);
                                            const double loo_value =
                                                citlali::pipeline::
                                                    mapdiag_fallback_leave_one_out_value(
                                                        raw_sum,
                                                        contrib_signal, wt,
                                                        contrib_weight);
                                            citlali::pipeline::
                                                mapdiag_assign_leave_one_out_z(
                                                    value, wt, loo_value,
                                                    candidate.leave_one_out_z);
                                        }
                                    }
                                }
                                citlali::pipeline::
                                    append_mapdiag_pixel_candidate(
                                        candidates, candidate);
                            }
                        }

                        citlali::pipeline::sort_mapdiag_pixel_candidates(
                            candidates);
                        const std::size_t candidate_top_n =
                            citlali::pipeline::mapdiag_candidate_top_n(
                                reduction_learning);
                        const std::size_t n_emitted_candidates =
                            citlali::pipeline::mapdiag_candidate_emit_count(
                                candidates.size(), candidate_top_n);
                        auto dominance =
                            citlali::pipeline::
                                make_mapdiag_detector_dominance_list();

                        for (std::size_t ci = 0; ci < n_emitted_candidates;
                             ++ci) {
                            const auto &candidate =
                                citlali::pipeline::mapdiag_emitted_candidate(
                                    candidates, ci);
                            const auto outlier_reason =
                                citlali::pipeline::
                                    mapdiag_map_pixel_outlier_reason(
                                        candidate, mb);
                            const auto record_map_index =
                                citlali::pipeline::mapdiag_record_map_index(i);
                            auto record =
                                citlali::pipeline::make_mapdiag_outlier_record<
                                    ReductionLearningState::MapPixelOutlier>(
                                    obsnum, mapdiag_record_producer,
                                    outlier_reason, fruit_iter,
                                    record_map_index, candidate);
                            reduction_learning.record_map_pixel_outlier(
                                std::move(record));
                            citlali::pipeline::
                                update_mapdiag_detector_dominance(
                                    dominance, candidate, fill_int);
                        }

                        const bool detector_exclusion_enabled =
                            citlali::pipeline::
                                mapdiag_detector_exclusion_enabled(
                                    reduction_learning);
                        if (detector_exclusion_enabled) {
                            const int detector_exclusion_min_pixels =
                                citlali::pipeline::
                                    mapdiag_detector_exclusion_min_pixels(
                                        reduction_learning);
                            const int array_id =
                                citlali::pipeline::mapdiag_array_id_or_default(
                                    map_index, calib.arrays, -1);
                            for (const auto &entry : dominance) {
                                if (!citlali::pipeline::
                                        mapdiag_dominance_meets_min_pixels(
                                            entry,
                                            detector_exclusion_min_pixels)) {
                                    continue;
                                }
                                const auto penalty_reason =
                                    citlali::pipeline::
                                        mapdiag_detector_dominance_penalty_reason();
                                auto penalty =
                                    citlali::pipeline::
                                        make_mapdiag_detector_penalty<
                                            ReductionLearningState::
                                                DetectorPenalty>(
                                            obsnum, mapdiag_record_producer,
                                            penalty_reason,
                                            fruit_iter, entry, array_id);
                                reduction_learning.record_detector_penalty(
                                    std::move(penalty), true);
                                const auto display_scan_index =
                                    citlali::pipeline::
                                        mapdiag_display_scan_index(entry.scan);
                                logger->info(
                                    "mapdiag learned scan-local detector exclusion candidate stage={} iter={} map={} uid={} scan={} outlier_pixels={} max_abs_value={:.4g} max_abs_leave_one_out_z={:.4g}",
                                    stage_name, fruit_iter, i, entry.uid,
                                    display_scan_index,
                                    entry.count, entry.max_abs_value,
                                    entry.max_abs_leave_one_out_z);
                            }
                        }
                    }
                }
            }

            const bool has_noise_realizations =
                citlali::pipeline::mapdiag_has_noise_realizations(
                    mb->noise, i, mb->n_noise);
            if (has_noise_realizations) {
                auto noise_samples =
                    citlali::pipeline::make_mapdiag_noise_tail_samples(mb);

                const auto valid_core =
                    citlali::pipeline::mapdiag_valid_core_noise_mask(
                        core_mask);
                const double valid_core_count =
                    citlali::pipeline::mapdiag_valid_core_noise_count(
                        valid_core);
                const Eigen::Index n_noise_realizations =
                    citlali::pipeline::mapdiag_noise_realization_count(mb);
                for (Eigen::Index n = 0; n < n_noise_realizations; ++n) {
                    const auto noise_matrix =
                        citlali::pipeline::mapdiag_noise_matrix(mb, i, n);
                    citlali::pipeline::add_mapdiag_noise_realization_samples(
                        noise_samples, mapdiag_stats, noise_matrix,
                        valid_core, valid_core_count, core_mask);
                }
                citlali::pipeline::assign_mapdiag_noise_tail_samples(
                    idx, mapdiag_stats, noise_samples, noise_tail_refs);
            }
        }

        const bool is_single_observation_mapdiag = !mapdiag_context.is_coadd;
        if (is_single_observation_mapdiag) {
            citlali::pipeline::assign_mapdiag_single_obs_entry(
                mapdiag_context, idx, weight_sum[idx],
                core_weight_sum[idx], n_valid_pixels[idx],
                n_core_pixels[idx], obs_tables);
        }
        else {
            const auto n_obsnums = mb->obsnums.size();
            for (std::size_t obs_idx = 0; obs_idx < n_obsnums; ++obs_idx) {
                const auto &coadd_obsnum = mb->obsnums[obs_idx];
                const auto obs_dir =
                    citlali::pipeline::mapdiag_obs_raw_dir(
                        redu_dir_name, coadd_obsnum);
                const auto obs_weight_path =
                    toltec_io
                        .create_filename<engine_utils::toltecIO::toltec,
                                         engine_utils::toltecIO::map,
                                         engine_utils::toltecIO::raw>(
                        obs_dir, redu_type, array_names[idx], coadd_obsnum,
                        telescope.sim_obs) + ".fits";
                const auto weight_hdu_name =
                    citlali::pipeline::mapdiag_weight_hdu_name(
                        map_names[idx], stokes_names[idx]);
                try {
                    fitsIO<file_type_enum::read_fits, CCfits::ExtHDU*>
                        obs_fits(obs_weight_path);
                    const auto obs_weight = obs_fits.get_hdu(weight_hdu_name);
                    citlali::pipeline::accumulate_mapdiag_obs_weight(
                        i, mapdiag_context.n_obsnums, mb->n_rows, mb->n_cols,
                        core_mask, obs_weight, obs_idx, obs_tables);
                } catch (const std::exception &e) {
                    logger->warn(
                        "failed to derive mapdiag contribution from {} [{}]: {}",
                        obs_weight_path, weight_hdu_name, e.what());
                    citlali::pipeline::zero_mapdiag_obs_entry(
                        mapdiag_context, idx, obs_idx, obs_tables);
                }
            }
        }
        const auto obs_totals =
            citlali::pipeline::sum_mapdiag_obs_weight_totals(
                obs_weight_sum, obs_core_weight_sum, mapdiag_context, idx);
        citlali::pipeline::assign_mapdiag_obs_fraction_pair(
            obs_weight_sum, obs_totals.weight, obs_core_weight_sum,
            obs_totals.core_weight, fill_double, mapdiag_context, idx,
            obs_weight_frac, obs_core_weight_frac);
    }

    write_netcdf_atomic(
        citlali::pipeline::mapdiag_netcdf_filename(filename),
        [&](netCDF::NcFile &fo) {
            citlali::pipeline::add_mapdiag_netcdf_vars(
                fo,
                {mapdiag_context, obsnum, mapdiag_metadata,
                 mapdiag_labels, mapdiag_values});
        });
}

void Engine::create_ptcdiag_file() {
    std::string dir_name = obsnum_dir_name + "raw/";
    if (tod_output_subdir_name != "null") {
        dir_name = dir_name + tod_output_subdir_name + "/";
    }

    const auto filename =
        toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                  engine_utils::toltecIO::ptcdiag,
                                  engine_utils::toltecIO::raw>(
            dir_name, redu_type, "", obsnum, telescope.sim_obs);
    ptcdiag_filename = filename + ".nc";

    write_netcdf_atomic(ptcdiag_filename, [&](netCDF::NcFile &fo) {
    const int fill_int = citlali::pipeline::ptcdiag_fill_int();
    const double fill_double = citlali::pipeline::ptcdiag_fill_double();
    const Eigen::Index n_scans = telescope.scan_indices.cols();
    const std::vector<std::size_t> det_chunks = {
        1, TULA_SIZET(calib.n_dets)};

    citlali::pipeline::add_tod_output_type_label(fo, "ptcdiag");

    citlali::pipeline::add_observation_identity_vars(
        fo, std::stoi(obsnum), telescope.tel_header["Header.Source.Ra"](0),
        telescope.tel_header["Header.Source.Dec"](0));

    netCDF::NcDim n_scans_dim = fo.addDim("n_scans", n_scans);
    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", calib.n_dets);
    std::vector<netCDF::NcDim> det_dims = {
        n_scans_dim, n_dets_dim};

    citlali::pipeline::add_diagnostic_output_scan_index(
        fo, n_scans_dim, n_scans, fill_int);

    auto add_det_meta_int = [&](const std::string &name,
                                const std::string &comment,
                                const std::vector<int> &values) {
        citlali::pipeline::add_ptcdiag_det_meta_int(
            fo, name, comment, n_dets_dim, values);
    };
    auto apt_int_values = [&](const std::string &key) {
        return citlali::pipeline::ptcdiag_apt_int_values(
            calib, key, fill_int);
    };
    citlali::pipeline::add_ptcdiag_det_meta_vars(
        add_det_meta_int, apt_int_values);

    citlali::pipeline::add_pipeline_identity_vars(
        fo, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION, TULA_GIT_VERSION,
        telescope.project_id, redu_type, telescope.obs_goal, tod_type);
    add_netcdf_var(fo, "SAMPRATE", telescope.fsmp);

    citlali::pipeline::add_weight_selection_config_vars(fo, ptcproc);
    citlali::pipeline::add_reduction_learning_config_vars(
        fo, reduction_learning);
    citlali::pipeline::add_ptc_weight_cutoff_config_vars(fo, ptcproc, true);
    citlali::pipeline::add_ptcdiag_compact_config_vars(fo, ptcproc);

    const auto n_ptc_scan_values = static_cast<std::size_t>(n_scans);
    const auto n_ptc_det_values = static_cast<std::size_t>(calib.n_dets);
    const std::size_t ptc_det_value_count =
        n_ptc_scan_values * n_ptc_det_values;
    auto add_det_double = [&](const std::string &name,
                              const std::string &comment) {
        citlali::pipeline::add_ptcdiag_det_double(
            fo, name, comment, det_dims, det_chunks, ptc_det_value_count,
            fill_double);
    };
    auto add_det_int = [&](const std::string &name,
                           const std::string &comment) {
        citlali::pipeline::add_ptcdiag_det_int(
            fo, name, comment, det_dims, det_chunks, ptc_det_value_count,
            fill_int);
    };
    citlali::pipeline::add_ptcdiag_detector_core_diag(add_det_double);
    citlali::pipeline::add_ptcdiag_detector_invvar_window_diag(
        add_det_int, add_det_double);

    citlali::pipeline::add_ptcdiag_corr_network_block(
        fo, calib, n_scans_dim, n_scans, fill_int, fill_double);

    const std::string weight_corr_comment =
        "multiplicative weight penalty factor applied per network in each scan";
    citlali::pipeline::add_ptcdiag_weight_corr_network_block(
        fo, calib, n_scans_dim, n_scans, weight_corr_comment, fill_int,
        fill_double);

    citlali::pipeline::add_ptcdiag_busy_row_network_block(
        fo, calib, n_scans_dim, n_scans, fill_int, fill_double);

    citlali::pipeline::add_ptcdiag_adaptive_pca_network_block(
        fo, calib, n_scans_dim, n_scans, fill_int, fill_double);

    const std::string second_pass_comment =
        "1 if this network had more candidate second-pass clusters than the normal auto-flag limit";
    citlali::pipeline::add_ptcdiag_second_pass_network_block(
        fo, calib, n_scans_dim, n_scans, second_pass_comment, true,
        fill_int, fill_double);
    });
}

void Engine::create_rtcdiag_file() {
    std::string dir_name = obsnum_dir_name + "raw/";
    if (tod_output_subdir_name != "null") {
        dir_name = dir_name + tod_output_subdir_name + "/";
    }

    const auto filename =
        toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                  engine_utils::toltecIO::rtcdiag,
                                  engine_utils::toltecIO::raw>(
            dir_name, redu_type, "", obsnum, telescope.sim_obs);
    rtcdiag_filename = filename + ".nc";

    write_netcdf_atomic(rtcdiag_filename, [&](netCDF::NcFile &fo) {

    citlali::pipeline::add_tod_output_type_label(fo, "rtcdiag");

    const int fill_int = citlali::pipeline::rtcdiag_fill_int();
    const double fill_double = citlali::pipeline::rtcdiag_fill_double();
    const Eigen::Index n_scans = telescope.scan_indices.cols();
    const auto n_scan_values = static_cast<std::size_t>(n_scans);
    const auto n_array_values = static_cast<std::size_t>(calib.n_arrays);
    const auto n_scan_array_values = n_scan_values * n_array_values;
    const double rtc_fsmp =
        rtcproc.run_downsample ? telescope.d_fsmp : telescope.fsmp;

    citlali::pipeline::add_observation_identity_vars(
        fo, std::stoi(obsnum), telescope.tel_header["Header.Source.Ra"](0),
        telescope.tel_header["Header.Source.Dec"](0));

    netCDF::NcDim n_scans_dim = fo.addDim("n_scans", n_scans);
    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", calib.n_dets);
    netCDF::NcDim n_arrays_dim = fo.addDim("n_arrays", calib.n_arrays);
    netCDF::NcDim n_nws_rtcdiag_dim =
        fo.addDim("n_nws_rtcdiag", calib.n_nws);
    const std::vector<std::size_t> scan_chunks = {
        TULA_SIZET(std::max<Eigen::Index>(n_scans, 1))};
    const std::vector<std::size_t> scan_array_chunks = {
        1, TULA_SIZET(std::max<Eigen::Index>(calib.n_arrays, 1))};
    const std::vector<std::size_t> rtc_det_chunks = {
        1, TULA_SIZET(calib.n_dets)};
    const std::vector<std::size_t> rtc_nw_chunks = {
        1, TULA_SIZET(calib.n_nws)};

    citlali::pipeline::add_diagnostic_output_scan_index(
        fo, n_scans_dim, n_scans, fill_int);

    citlali::pipeline::add_rtcdiag_array_ids(
        fo, calib, n_arrays_dim, fill_int);

    auto add_scan_double = [&](const std::string &name,
                               const std::string &units,
                               const std::string &comment,
                               const std::vector<double> &values) {
        citlali::pipeline::add_rtcdiag_scan_double(
            fo, name, units, comment, n_scans_dim, scan_chunks, values);
    };

    std::vector<double> scan_duration_s(n_scan_values, fill_double);
    std::vector<double> scan_speed_p50_arcsec_s(n_scan_values, fill_double);
    std::vector<double> scan_speed_p95_arcsec_s(n_scan_values, fill_double);
    std::vector<double> scan_speed_p995_arcsec_s(n_scan_values, fill_double);
    constexpr double max_tel_sample_step_s = 0.1;
    constexpr double max_pointing_step_rad = 0.01;

    const auto tel_time_it = telescope.tel_data.find("TelTime");
    const auto az_it = telescope.tel_data.find("az_phys");
    const auto alt_it = telescope.tel_data.find("alt_phys");
    const bool has_telescope_motion_data =
        tel_time_it != telescope.tel_data.end() &&
        az_it != telescope.tel_data.end() &&
        alt_it != telescope.tel_data.end();
    if (has_telescope_motion_data) {
        const auto &tel_time = tel_time_it->second;
        const auto &az_phys = az_it->second;
        const auto &alt_phys = alt_it->second;
        const Eigen::Index n_tel =
            std::min({tel_time.size(), az_phys.size(), alt_phys.size()});
        for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
            const auto scan_index = static_cast<std::size_t>(scan);
            const Eigen::Index start =
                std::max<Eigen::Index>(0, telescope.scan_indices(0, scan));
            const Eigen::Index stop =
                std::min<Eigen::Index>(n_tel - 1,
                                       telescope.scan_indices(1, scan));
            const bool has_valid_scan_bounds =
                stop > start && start >= 0 && stop < n_tel;
            if (!has_valid_scan_bounds) {
                continue;
            }
            const double duration = tel_time(stop) - tel_time(start);
            if (std::isfinite(duration) && duration > 0.0) {
                scan_duration_s[scan_index] = duration;
            }
            const auto n_scan_samples =
                std::max<Eigen::Index>(stop - start, 0);
            std::vector<double> speed_arcsec_s;
            speed_arcsec_s.reserve(static_cast<std::size_t>(n_scan_samples));
            for (Eigen::Index i = start; i < stop; ++i) {
                const double dt = tel_time(i + 1) - tel_time(i);
                const double daz = az_phys(i + 1) - az_phys(i);
                const double dalt = alt_phys(i + 1) - alt_phys(i);
                if (!std::isfinite(dt) || !std::isfinite(daz) ||
                    !std::isfinite(dalt) || dt <= 0.0 ||
                    dt > max_tel_sample_step_s ||
                    std::abs(daz) > max_pointing_step_rad ||
                    std::abs(dalt) > max_pointing_step_rad) {
                    continue;
                }
                speed_arcsec_s.push_back(
                    std::hypot(daz, dalt) / dt * RAD_TO_ASEC);
            }
            if (!speed_arcsec_s.empty()) {
                std::sort(speed_arcsec_s.begin(), speed_arcsec_s.end());
                scan_speed_p50_arcsec_s[scan_index] =
                    citlali::pipeline::rtcdiag_percentile_sorted(
                        speed_arcsec_s, 50.0);
                scan_speed_p95_arcsec_s[scan_index] =
                    citlali::pipeline::rtcdiag_percentile_sorted(
                        speed_arcsec_s, 95.0);
                scan_speed_p995_arcsec_s[scan_index] =
                    citlali::pipeline::rtcdiag_percentile_sorted(
                        speed_arcsec_s, 99.5);
            }
        }
    }
    else {
        logger->warn(
            "rtcdiag scan-speed diagnostics skipped: missing TelTime, "
            "az_phys, or alt_phys telescope data");
    }

    citlali::pipeline::add_rtcdiag_scan_summary_vars(
        add_scan_double,
        {scan_duration_s,
         scan_speed_p50_arcsec_s,
         scan_speed_p95_arcsec_s,
         scan_speed_p995_arcsec_s});

    std::vector<netCDF::NcDim> scan_array_dims = {
        n_scans_dim, n_arrays_dim};
    std::vector<double> source_power_half_bandwidth_hz(
        n_scan_array_values, fill_double);
    std::vector<double> tod_lowpass_to_source_power_half_ratio(
        n_scan_array_values, fill_double);
    for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
        const auto scan_index = static_cast<std::size_t>(scan);
        const double speed = scan_speed_p995_arcsec_s[scan_index];
        if (!std::isfinite(speed) || speed <= 0.0) {
            continue;
        }
        for (Eigen::Index arr_i = 0; arr_i < calib.n_arrays; ++arr_i) {
            const Eigen::Index array = calib.arrays(arr_i);
            const auto fwhm_it = calib.array_fwhms.find(array);
            if (fwhm_it == calib.array_fwhms.end()) {
                continue;
            }
            const double fwhm_arcsec =
                0.5 * (std::get<0>(fwhm_it->second) +
                       std::get<1>(fwhm_it->second));
            if (!std::isfinite(fwhm_arcsec) || fwhm_arcsec <= 0.0) {
                continue;
            }
            const double f_half_hz =
                (std::sqrt(std::log(2.0)) /
                 (2.0 * pi * fwhm_arcsec * FWHM_TO_STD)) *
                speed;
            const auto flat_i = scan_index * n_array_values +
                                static_cast<std::size_t>(arr_i);
            source_power_half_bandwidth_hz[flat_i] = f_half_hz;
            const bool has_lowpass_ratio =
                rtcproc.run_tod_filter &&
                rtcproc.filter.freq_high_Hz > 0.0 && f_half_hz > 0.0;
            if (has_lowpass_ratio) {
                tod_lowpass_to_source_power_half_ratio[flat_i] =
                    rtcproc.filter.freq_high_Hz / f_half_hz;
            }
        }
    }
    auto add_scan_array_double = [&](const std::string &name,
                                     const std::string &units,
                                     const std::string &comment,
                                     const std::vector<double> &values) {
        citlali::pipeline::add_rtcdiag_scan_array_double(
            fo, name, units, comment, scan_array_dims, scan_array_chunks,
            values);
    };
    citlali::pipeline::add_rtcdiag_scan_array_summary_vars(
        add_scan_array_double,
        {source_power_half_bandwidth_hz,
         tod_lowpass_to_source_power_half_ratio});

    citlali::pipeline::add_rtcdiag_network_ids(
        fo, calib, n_nws_rtcdiag_dim, fill_int);

    citlali::pipeline::add_pipeline_identity_vars(
        fo, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION, TULA_GIT_VERSION,
        telescope.project_id, redu_type, telescope.obs_goal, tod_type);
    add_netcdf_var(fo, "SAMPRATE", telescope.fsmp);
    add_netcdf_var(fo, "RTC_SAMPRATE", rtc_fsmp);
    add_netcdf_var(fo, "CONFIG.TODFILTERED", rtcproc.run_tod_filter);
    add_netcdf_var(fo, "CONFIG.TODFILTER.FREQ_HIGH_HZ",
                   rtcproc.filter.freq_high_Hz);
    add_netcdf_var(fo, "CONFIG.TODFILTER.FREQ_LOW_HZ",
                   rtcproc.filter.freq_low_Hz);
    add_netcdf_var(fo, "CONFIG.TODFILTER.N_TERMS", rtcproc.filter.n_terms);
    citlali::pipeline::add_tod_filter_edge_guard_config_vars(
        fo, rtcproc.filter_edge_guard, telescope.outer_scans_chunk,
        rtcproc.tod_output_outer_context_samples);

    // Keep a compact provenance subset so rtcdiag is interpretable without the RTC TOD.
    add_netcdf_var(fo, "CONFIG.VERBOSE", verbose_mode);
    citlali::pipeline::add_reduction_learning_config_vars(
        fo, reduction_learning, false);
    add_netcdf_var(fo, "CONFIG.DESPIKED", rtcproc.run_despike);
    citlali::pipeline::add_rtc_local_despike_config_vars(
        fo, rtcproc.despiker.local_residual);
    citlali::pipeline::add_rtc_event_mask_config_vars(fo, rtcproc);
    citlali::pipeline::add_rtc_line_audit_config_vars(
        fo, rtcproc.line_audit);
    add_netcdf_var(fo, "CONFIG.INV_VAR.WINDOW_SEC",
                   rtcproc.remove_bad_dets_window_sec);

    citlali::pipeline::add_rtcdiag_apt_double_vars(fo, calib, n_dets_dim);

    std::vector<netCDF::NcDim> rtc_det_dims = {
        n_scans_dim, n_dets_dim};
    const auto n_rtc_det_values =
        n_scan_values * static_cast<std::size_t>(calib.n_dets);
    auto add_rtc_det_double = [&](const std::string &name,
                                  const std::string &comment) {
        citlali::pipeline::add_rtcdiag_det_double(
            fo, name, comment, rtc_det_dims, rtc_det_chunks,
            n_rtc_det_values, fill_double);
    };
    auto add_rtc_det_int = [&](const std::string &name,
                               const std::string &comment) {
        citlali::pipeline::add_rtcdiag_det_int(
            fo, name, comment, rtc_det_dims, rtc_det_chunks,
            n_rtc_det_values, fill_int);
    };

    citlali::pipeline::add_rtcdiag_detector_core_diag(
        add_rtc_det_int, add_rtc_det_double);
    citlali::pipeline::add_rtcdiag_detector_invvar_window_diag(
        add_rtc_det_int, add_rtc_det_double);

    std::vector<netCDF::NcDim> rtc_nw_dims = {
        n_scans_dim, n_nws_rtcdiag_dim};
    const auto n_rtc_nw_values =
        n_scan_values * static_cast<std::size_t>(calib.n_nws);
    auto add_rtc_nw_double = [&](const std::string &name,
                                 const std::string &comment) {
        citlali::pipeline::add_rtcdiag_network_double(
            fo, name, comment, rtc_nw_dims, rtc_nw_chunks,
            n_rtc_nw_values, fill_double);
    };
    auto add_rtc_nw_int = [&](const std::string &name,
                              const std::string &comment) {
        citlali::pipeline::add_rtcdiag_network_int(
            fo, name, comment, rtc_nw_dims, rtc_nw_chunks,
            n_rtc_nw_values, fill_int);
    };

    citlali::pipeline::add_rtcdiag_standard_network_diag(
        add_rtc_nw_int, add_rtc_nw_double);

    const bool write_impulsive_capture_diag =
        rtcproc.impulsive_capture.enabled;
    if (write_impulsive_capture_diag) {
        const auto max_events_per_network =
            rtcproc.impulsive_capture.max_events_per_network;
        const auto n_slots =
            static_cast<std::size_t>(
                std::max<Eigen::Index>(max_events_per_network, 1));
        const double snippet_pre_window_sec =
            rtcproc.impulsive_capture.snippet_pre_window_sec;
        const auto snippet_pre =
            citlali::pipeline::rtcdiag_impulsive_window_samples(
                snippet_pre_window_sec, rtc_fsmp);
        const double snippet_post_window_sec =
            rtcproc.impulsive_capture.snippet_post_window_sec;
        const auto snippet_post =
            citlali::pipeline::rtcdiag_impulsive_window_samples(
                snippet_post_window_sec, rtc_fsmp);
        const auto n_snippet =
            citlali::pipeline::rtcdiag_impulsive_snippet_sample_count(
                snippet_pre, snippet_post);
        netCDF::NcDim n_rtc_impulsive_slots_dim =
            fo.addDim("n_rtc_impulsive_slots", n_slots);
        netCDF::NcDim n_rtc_impulsive_samples_dim =
            fo.addDim("n_rtc_impulsive_samples", n_snippet);

        netCDF::NcVar offset_v =
            fo.addVar("rtc_impulsive_snippet_offset_samples", netCDF::ncInt,
                      n_rtc_impulsive_samples_dim);
        offset_v.putAtt("units", "samples");
        offset_v.putAtt(
            "comment",
            "sample offsets relative to rtc_impulsive_slot_event_sample");
        const auto offsets =
            citlali::pipeline::rtcdiag_impulsive_snippet_offsets(
                n_snippet, snippet_pre, fill_int);
        offset_v.putVar(offsets.data());

        const auto n_impulsive_networks =
            static_cast<std::size_t>(calib.n_nws);
        std::vector<netCDF::NcDim> rtc_impulsive_slot_dims = {
            n_scans_dim, n_nws_rtcdiag_dim, n_rtc_impulsive_slots_dim};
        std::vector<netCDF::NcDim> rtc_impulsive_snippet_dims = {
            n_scans_dim, n_nws_rtcdiag_dim, n_rtc_impulsive_slots_dim,
            n_rtc_impulsive_samples_dim};
        const std::vector<std::size_t> rtc_impulsive_slot_chunks = {
            1, n_impulsive_networks, n_slots};
        const std::vector<std::size_t> rtc_impulsive_snippet_chunks = {
            1, n_impulsive_networks, n_slots, n_snippet};
        const auto n_rtc_impulsive_slot_values =
            static_cast<std::size_t>(n_scans) *
            n_impulsive_networks * n_slots;
        const auto n_rtc_impulsive_snippet_values =
            n_rtc_impulsive_slot_values * n_snippet;

        auto add_rtc_imp_slot_double = [&](const std::string &name,
                                           const std::string &comment) {
            citlali::pipeline::add_rtcdiag_impulsive_slot_double(
                fo, name, comment, rtc_impulsive_slot_dims,
                rtc_impulsive_slot_chunks, n_rtc_impulsive_slot_values,
                fill_double);
        };
        auto add_rtc_imp_slot_int = [&](const std::string &name,
                                        const std::string &comment) {
            citlali::pipeline::add_rtcdiag_impulsive_slot_int(
                fo, name, comment, rtc_impulsive_slot_dims,
                rtc_impulsive_slot_chunks, n_rtc_impulsive_slot_values,
                fill_int);
        };
        auto add_rtc_imp_snip_double = [&](const std::string &name,
                                           const std::string &comment) {
            citlali::pipeline::add_rtcdiag_impulsive_snippet_double(
                fo, name, comment, rtc_impulsive_snippet_dims,
                rtc_impulsive_snippet_chunks, n_rtc_impulsive_snippet_values,
                fill_double);
        };
        auto add_rtc_imp_snip_int = [&](const std::string &name,
                                        const std::string &comment) {
            citlali::pipeline::add_rtcdiag_impulsive_snippet_int(
                fo, name, comment, rtc_impulsive_snippet_dims,
                rtc_impulsive_snippet_chunks, n_rtc_impulsive_snippet_values,
                fill_int);
        };

        citlali::pipeline::add_rtcdiag_impulsive_capture_diag(
            add_rtc_imp_slot_int, add_rtc_imp_slot_double,
            add_rtc_imp_snip_double, add_rtc_imp_snip_int,
            citlali::pipeline::rtcdiag_impulsive_capture_file_comments());
    }

    });
}

void Engine::write_stats() {
    std::string stats_dir = obsnum_dir_name + "raw/";
    // if using tod subdir, put stats file in it
    const bool has_tod_output_subdir = tod_output_subdir_name != "null";
    if (has_tod_output_subdir) {
        const auto stats_subdir_path = stats_dir + tod_output_subdir_name;
        if (!fs::exists(fs::status(stats_subdir_path))) {
            fs::create_directories(stats_subdir_path);
            stats_dir = stats_subdir_path + "/";
        }
    }
    // create stats filename
    const auto stats_filename =
        toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                  engine_utils::toltecIO::stats,
                                  engine_utils::toltecIO::raw>(
            stats_dir, redu_type, "", obsnum, telescope.sim_obs);

    // det stats header
    const auto det_stats_header_units =
        citlali::pipeline::detector_stats_units(omb.sig_unit);
    // group stats header
    const auto grp_stats_header_units =
        citlali::pipeline::group_stats_units(omb.sig_unit);
    const auto stats_netcdf_filename = stats_filename + ".nc";
    write_netcdf_atomic(stats_netcdf_filename, [&](netCDF::NcFile &fo) {

    citlali::pipeline::add_obsnum_var(fo, std::stoi(obsnum));

    // add dimensions
    const auto n_stats_chunks = telescope.scan_indices.cols();
    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", calib.n_dets);
    netCDF::NcDim n_arrays_dim = fo.addDim("n_arrays", calib.n_arrays);
    netCDF::NcDim n_chunks_dim = fo.addDim("n_chunks", n_stats_chunks);

    const std::vector<netCDF::NcDim> det_stat_dims = {
        n_chunks_dim, n_dets_dim};
    const std::vector<netCDF::NcDim> grp_stat_dims = {
        n_chunks_dim, n_arrays_dim};

    // add det stats
    for (const auto &stat : diagnostics.det_stats_header) {
        citlali::pipeline::add_stats_double_var(
            fo, stat, det_stat_dims, diagnostics.stats[stat],
            citlali::pipeline::stats_unit_or_empty(
                det_stats_header_units, stat));
    }
    // add group stats
    for (const auto &stat : diagnostics.grp_stats_header) {
        citlali::pipeline::add_stats_double_var(
            fo, stat, grp_stat_dims, diagnostics.stats[stat],
            citlali::pipeline::stats_unit_or_empty(
                grp_stats_header_units, stat));
    }

    // add apt table
    citlali::pipeline::add_stats_apt_double_vars(fo, calib, n_dets_dim);

    // add adc
    citlali::pipeline::add_stats_adc_snap_vars(
        fo, calib, diagnostics.adc_snap_data);

    // add eigenvalues
    const bool has_eigenvalue_diagnostics =
        citlali::pipeline::should_write_stats_eigenvalues(
            diagnostics, ptcproc.cleaner);
    if (has_eigenvalue_diagnostics) {
        const bool has_eigenvalue_groups =
            citlali::pipeline::has_stats_eigenvalue_groups(
                diagnostics.evals);
        if (has_eigenvalue_groups) {
            const auto first_it = diagnostics.evals.begin();
            const Eigen::Index n_cleaner_eigenvalues =
                ptcproc.cleaner.n_calc;
            const auto &cleaner_grouping = ptcproc.cleaner.grouping;
            const double eigenvalue_fill_value =
                citlali::pipeline::ptcdiag_fill_double();
            const auto n_eig_groups = first_it->second[0].size();
            const auto eval_dims =
                citlali::pipeline::add_stats_eigenvalue_dims(
                    fo, n_cleaner_eigenvalues, n_eig_groups);

            // loop through chunks
            for (const auto &[chunk_index, eval_groups] : diagnostics.evals) {
                // loop through cleaner grouping
                const auto n_eval_groupings = eval_groups.size();
                for (Eigen::Index i=0; i<n_eval_groupings; ++i) {
                    const auto &cleaner_grouping_name = cleaner_grouping[i];
                    const auto eval_var_name =
                        citlali::pipeline::stats_eigenvalue_var_name(
                            cleaner_grouping_name, i, chunk_index);
                    citlali::pipeline::add_stats_eigenvalue_group_var(
                        fo, eval_var_name, eval_dims, eval_groups[i],
                        n_cleaner_eigenvalues, eigenvalue_fill_value);
                }
            }
        }
        else {
            logger->warn("evals requested but empty; skipping eval/evec output");
        }
    }
    });
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::run_wiener_filter(map_buffer_t &mb) {
    const auto n_maps_local = static_cast<std::size_t>(mb.signal.size());
    mapmaking::reset_edge_guard_storage(mb, n_maps_local);

    // pointer to map buffer
    mapmaking::MapBuffer *pmb = &mb;
    using FitsVector =
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>>;
    // pointer to data file fits vector
    FitsVector *f_io = nullptr;
    // pointer to noise file fits vector
    FitsVector *n_io = nullptr;
    // directory name
    std::string filtered_dir_name;
    // logging label
    const char *map_label = "filtered maps";

    // filtered obs maps
    if constexpr (map_t == mapmaking::FilteredObs) {
        f_io = &filtered_fits_io_vec;
        n_io = &filtered_noise_fits_io_vec;
        filtered_dir_name = obsnum_dir_name + "filtered/";
        map_label = "filtered obs maps";
    }

    // filtered coadded maps
    else if constexpr (map_t == mapmaking::FilteredCoadd) {
        f_io = &filtered_coadd_fits_io_vec;
        n_io = &filtered_coadd_noise_fits_io_vec;
        filtered_dir_name = coadd_dir_name + "filtered/";
        map_label = "filtered coadded maps";
    }

    const auto n_filtered_fits = static_cast<Eigen::Index>(f_io->size());
    logger->info("preparing {} FITS headers ({} files)", map_label,
                 f_io->size());
    for (Eigen::Index i=0; i<n_filtered_fits; ++i) {
        add_phdu(f_io, pmb, i);

        if (!pmb->noise.empty() && !n_io->empty()) {
            add_phdu(n_io, pmb, i);
        }
    }

    // loop through maps and run wiener filter
    for (Eigen::Index i=0; i<n_maps; ++i) {
        // current array
        const auto array = maps_to_arrays(i);
        const auto &array_name = toltec_io.array_name_map[array];
        // get file index
        const auto map_index = arrays_to_maps(i);
        logger->info("starting {} map {}/{} (array={})",
                     map_label, i + 1, n_maps, array_name);
        // init fwhm in pixels
        wiener_filter.init_fwhm =
            toltec_io.array_fwhm_arcsec[array] * ASEC_TO_RAD /
            mb.pixel_size_rad;
        // make wiener filter template
        logger->info("building Wiener template for {} map {}/{} (array={})",
                     map_label, i + 1, n_maps, array_name);
        double template_fwhm_rad = 0.0;
        const bool template_uses_fwhm =
            wiener_filter.template_type == "gaussian" ||
            wiener_filter.template_type == "airy";
        if (template_uses_fwhm) {
            const auto it = wiener_filter.template_fwhm_rad.find(array_name);
            if (it == wiener_filter.template_fwhm_rad.end()) {
                logger->error("missing Wiener template_fwhm_rad for array {}",
                              array_name);
                std::exit(EXIT_FAILURE);
            }
            template_fwhm_rad = it->second;
        }
        wiener_filter.make_template(mb, calib.apt, template_fwhm_rad, i);
        logger->info("Wiener template ready for {} map {}/{} (array={})",
                     map_label, i + 1, n_maps, array_name);
        // run the filter for the current map
        logger->info("running Wiener filter core for {} map {}/{} (array={})",
                     map_label, i + 1, n_maps, array_name);
        wiener_filter.filter_maps(mb, i);
        logger->info("map filtering complete for {} map {}/{}",
                     map_label, i + 1, n_maps);

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

            const bool should_calculate_noise_products =
                write_filtered_maps_partial &&
                (run_noise_products || wiener_filter.normalize_error);
            if (should_calculate_noise_products) {
                const bool apply_scale =
                    apply_empirical_noise_weights ||
                    wiener_filter.normalize_error;
                logger->info("calculating empirical noise products for {} map {}/{}",
                             map_label, i + 1, n_maps);
                mb.calc_noise_products(i, apply_scale);
                const bool has_noise_weight_summary =
                    i < mb.noise_weight_median_ratio.size();
                if (has_noise_weight_summary) {
                    logger->info(
                        "noise products: median(w_formal*var)={:.4g} "
                        "scale={:.4g} noise_s2n_sigma={:.4g}",
                        mb.noise_weight_median_ratio(i),
                        mb.noise_weight_scale(i),
                        mb.noise_s2n_sigma(i));
                }
                mb.calc_median_err();
                mb.calc_median_rms();
            }
        }

        if (write_filtered_maps_partial) {
            // only write if saving all iterations or on last iteration
            // write maps immediately after filtering due to computation time
            logger->info("writing {} map {}/{} to disk",
                         map_label, i + 1, n_maps);
            write_maps(f_io, n_io, pmb, i);

            const auto &filtered_map_path = f_io->at(map_index).filepath;
            logger->info("file has been written to:");
            logger->info("{}.fits", filtered_map_path);

            // explicitly destroy the fits file after we're done with it
            bool should_close_file = true;
            if (rtcproc.run_polarization) {
                if (rtcproc.polarization.stokes_params[maps_to_stokes(i)] !=
                    "U") {
                    should_close_file = false;
                }
            }
            // check if we're moving onto a new file
            if (i < n_maps - 1) {
                const bool next_map_opens_new_file =
                    arrays_to_maps(i + 1) > arrays_to_maps(i);
                if (next_map_opens_new_file && should_close_file) {
                    logger->info("closing FITS handle for {}",
                                 filtered_map_path);
                    f_io->at(map_index).pfits->destroy();
                    logger->info("closed FITS handle for {}",
                                 filtered_map_path);
                }
            }
        }

        logger->info("completed {} map {}/{}", map_label, i + 1, n_maps);
    }

    if (write_filtered_maps_partial) {
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
    constexpr int missing_source_location = -99;
    // loop through maps
    for (Eigen::Index i=0; i<n_maps; ++i) {
        // update source vectors
        mb.n_sources.push_back(0);
        mb.row_source_locs.push_back(Eigen::VectorXi::Ones(1));
        mb.col_source_locs.push_back(Eigen::VectorXi::Ones(1));

        // default missing value keeps vector sizes aligned with maps
        mb.row_source_locs.back() *= missing_source_location;
        mb.col_source_locs.back() *= missing_source_location;

        // run source finder
        const auto sources_found = mb.find_sources(i);

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
    for (const auto &sources : mb.n_sources) {
        n_sources += sources;
    }

    // matrix to store source parameters
    mb.source_params.setZero(n_sources, map_fitter.n_params);
    mb.source_perror.setZero(n_sources, map_fitter.n_params);

    // keep track of row in total source count
    Eigen::Index source_row_start = 0;

    // now loop through and fit the sources
    for (Eigen::Index i=0; i<n_maps; ++i) {
        // skip map if no sources found
        const auto n_map_sources = mb.n_sources[i];
        if (n_map_sources > 0) {
            // current array
            const auto array = maps_to_arrays(i);
            // init fwhm in pixels
            const auto init_fwhm =
                toltec_io.array_fwhm_arcsec[array] * ASEC_TO_RAD /
                mb.pixel_size_rad;

            // placeholder vectors for grppi map
            std::vector<int> source_in_vec;
            std::vector<int> source_out_vec;

            source_in_vec.resize(n_map_sources);
            std::iota(source_in_vec.begin(), source_in_vec.end(), 0);
            source_out_vec.resize(n_map_sources);

            // loop through sources and fit them
            grppi::map(tula::grppi_utils::dyn_ex(parallel_policy),
                       source_in_vec, source_out_vec, [&](auto j) {
                // update source rows and cols
                const double init_row = mb.row_source_locs[i](j);
                const double init_col = mb.col_source_locs[i](j);

                // fit source
                auto [params, perrors, good_fit] =
                    map_fitter.fit_to_gaussian<
                        engine_utils::mapFitter::pointing>(
                            mb.signal[i], mb.weight[i], init_fwhm,
                            init_row, init_col);
                if (good_fit) {
                    const double pixel_to_arcsec =
                        RAD_TO_ASEC * mb.pixel_size_rad;
                    const double source_fwhm_to_arcsec =
                        RAD_TO_ASEC * STD_TO_FWHM * mb.pixel_size_rad;
                    // rescale fit params from pixel to on-sky units
                    params(1) = pixel_to_arcsec *
                                (params(1) - (mb.n_cols - 1)/2.0);
                    params(2) = pixel_to_arcsec *
                                (params(2) - (mb.n_rows - 1)/2.0);
                    params(3) = source_fwhm_to_arcsec * params(3);
                    params(4) = source_fwhm_to_arcsec * params(4);

                    // rescale fit errors from pixel to on-sky units
                    perrors(1) = pixel_to_arcsec * perrors(1);
                    perrors(2) = pixel_to_arcsec * perrors(2);
                    perrors(3) = source_fwhm_to_arcsec * perrors(3);
                    perrors(4) = source_fwhm_to_arcsec * perrors(4);

                    // if in radec calculate absolute pointing
                    const bool use_radec_projection =
                        telescope.pixel_axes == "radec";
                    if (use_radec_projection) {
                        Eigen::VectorXd lat(1), lon(1);
                        lat << params(2) * ASEC_TO_RAD;
                        lon << params(1) * ASEC_TO_RAD;

                        auto [adec, ara] =
                            engine_utils::tangent_to_abs(
                                lat, lon, mb.wcs.crval[0] * DEG_TO_RAD,
                                mb.wcs.crval[1] * DEG_TO_RAD);

                        params(1) = ara(0) * RAD_TO_DEG;
                        params(2) = adec(0) * RAD_TO_DEG;

                        perrors(1) = perrors(1) * ASEC_TO_DEG;
                        perrors(2) = perrors(2) * ASEC_TO_DEG;
                    }

                    // add source params and errors to table
                    mb.source_params.row(source_row_start + j) = params;
                    mb.source_perror.row(source_row_start + j) = perrors;
                }
                return 0;
            });

            // update row
            source_row_start += n_map_sources;
        }
    }
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_sources(map_buffer_t &mb, std::string dir_name) {
    // get filename for source table
    const std::string source_filename =
        setup_filenames<map_t, engine_utils::toltecIO::source,
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
    const std::string pos_units =
        (telescope.pixel_axes == "radec") ? "deg" : "arcsec";

    // units for source header
    std::map<std::string, std::string> source_header_units = {
        {"array", "N/A"},
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
        {"sig2noise", "N/A"},
    };

    // meta information for source table
    YAML::Node source_meta;

    // add obsnums
    for (Eigen::Index i = 0; i < mb->obsnums.size(); ++i) {
        // add obsnum to meta data
        const auto obsnum_key = "obsnum" + std::to_string(i);
        source_meta[obsnum_key] = mb->obsnums[i];
    }

    // add source name
    source_meta["Source"] = telescope.source_name;

    // add date of file creation
    source_meta["creation_date"] = engine_utils::current_date_time();

    // add observation date
    source_meta["date"] = date_obs.back();


    // populate source meta information
    for (const auto &[key, val] : source_header_units) {
        source_meta[key].push_back("units: " + val);
        // description from apt
        const auto description = calib.apt_header_description[key];
        source_meta[key].push_back(description);
    }

    // count up the total number of sources
    Eigen::Index n_sources = 0;
    for (const auto &sources : mb->n_sources) {
        n_sources += sources;
    }

    // matrix to hold source information (floats for readability)
    const auto source_table_cols = 2 * map_fitter.n_params + 2;
    Eigen::MatrixXf source_table(n_sources, source_table_cols);

    // loop through params and add arrays
    Eigen::Index k = 0;
    for (Eigen::Index i = 0; i < mb->n_sources.size(); ++i) {
        if (mb->n_sources[i] != 0) {
            // calculate map standard deviation
            const double map_std_dev =
                engine_utils::calc_std_dev(mb->signal[i]);

            for (Eigen::Index j = 0; j < mb->n_sources[i]; ++j) {
                source_table(k, 0) = maps_to_arrays(i);
                // set signal to noise
                const auto sig2noise_col = 2 * map_fitter.n_params + 1;
                source_table(k, sig2noise_col) =
                    mb->source_params(k, 0) / map_std_dev;

                ++k;
            }
        }
    }

    // populate source table
    Eigen::Index source_param_index = 0;
    for (Eigen::Index i = 1; i < 2 * map_fitter.n_params; i = i + 2) {
        const auto param_col = i;
        const auto error_col = i + 1;
        source_table.col(param_col) =
            mb->source_params.col(source_param_index).template cast<float>();
        source_table.col(error_col) =
            mb->source_perror.col(source_param_index).template cast<float>();
        ++source_param_index;
    }

    // write source table
    to_ecsv_from_matrix(source_filename, source_table, source_header, source_meta);
}
