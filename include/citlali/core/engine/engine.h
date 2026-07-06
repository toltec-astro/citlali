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

#include <citlali/core/config/reduction_config.h>
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
#include <citlali/core/pipeline/learning_apt_helpers.h>
#include <citlali/core/pipeline/map_filename.h>
#include <citlali/core/pipeline/map_filtering.h>
#include <citlali/core/pipeline/map_source_finding.h>
#include <citlali/core/pipeline/mapdiag_edge_guard.h>
#include <citlali/core/pipeline/map_layer_name.h>
#include <citlali/core/pipeline/map_summary_stats.h>
#include <citlali/core/pipeline/mapdiag_labels.h>
#include <citlali/core/pipeline/mapdiag_netcdf.h>
#include <citlali/core/pipeline/mapdiag_observation_weight.h>
#include <citlali/core/pipeline/mapdiag_stage.h>
#include <citlali/core/pipeline/mapdiag_stats.h>
#include <citlali/core/pipeline/observation_map_files.h>
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
#include <citlali/core/pipeline/tod_stream_netcdf.h>
#include <citlali/core/pipeline/tod_output_selection.h>

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

    // typed config mirror for staged config migration
    citlali::config::ReductionConfig typed_config;

    // obsnum
    std::string obsnum;

    // write filtered maps as they complete
    bool write_filtered_maps_partial;

    // rtc or ptc types
    std::string tod_output_subdir_name;
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
    citlali::config::TodOutputSelectionMode tod_output_selection_mode_rtc =
        citlali::config::TodOutputSelectionMode::indices;
    citlali::config::TodOutputSelectionMode tod_output_selection_mode_ptc =
        citlali::config::TodOutputSelectionMode::indices;
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

#include <citlali/core/engine/detail/observation_setup_impl.h>
#include <citlali/core/engine/detail/tod_output_selection_impl.h>
#include <citlali/core/engine/detail/timestream_config_impl.h>
#include <citlali/core/engine/detail/learning_impl.h>
#include <citlali/core/engine/detail/config_loading_impl.h>
#include <citlali/core/engine/detail/map_output_impl.h>
#include <citlali/core/engine/detail/map_post_processing_impl.h>
