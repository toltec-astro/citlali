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
#include <cmath>
#include <omp.h>
#include <fstream>
#include <limits>
#include <sstream>

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

    // get pointing config options
    template<typename CT>
    void get_pointing_config(CT &);

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
    if (tod_output_subdir_name!="null") {
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

    rtcproc.configure_filter_edge_guard(telescope.fsmp);
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
    rtcproc.tod_output_outer = false;
    rtcproc.tod_output_outer_context_samples = 0;
    if (run_tod_output_rtc && config.has(std::tuple{"timestream","raw_time_chunk","output","mode"})) {
        std::string rtc_output_mode = "full";
        get_config_value(config, rtc_output_mode, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","output","mode"},
                         {"full","mini","full_outer","mini_outer"});
        rtcproc.tod_output_mini = (rtc_output_mode == "mini" || rtc_output_mode == "mini_outer");
        rtcproc.tod_output_outer = (rtc_output_mode == "full_outer" || rtc_output_mode == "mini_outer");
    }
    if (run_tod_output_rtc && config.has(std::tuple{"timestream","raw_time_chunk","output","outer_context_samples"})) {
        get_config_value(config, rtcproc.tod_output_outer_context_samples, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","output","outer_context_samples"},
                         {}, {0});
    }
    // output ptc
    get_config_value(config, run_tod_output_ptc, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","output","enabled"});
    ptcproc.tod_output_mini = false;
    ptcproc.tod_output_outer = false;
    ptcproc.tod_output_outer_context_samples = 0;
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
    if (redu_type == "beammap" && telescope.pixel_axes != "altaz") {
        logger->error(
            "beammap reductions require mapmaking.pixel_axes='altaz'; got '{}'",
            telescope.pixel_axes);
        std::exit(EXIT_FAILURE);
    }

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

    write_noise_realizations = false;
    if (config.template has_typed<bool>(std::tuple{"noise_maps","write_realizations"})) {
        get_config_value(config, write_noise_realizations, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","write_realizations"});
    }
    run_noise_products = run_noise;
    if (config.template has_typed<bool>(std::tuple{"noise_maps","products","enabled"})) {
        get_config_value(config, run_noise_products, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","products","enabled"});
    }
    apply_empirical_noise_weights = run_noise;
    if (config.template has_typed<bool>(std::tuple{"noise_maps","products","apply_empirical_weights"})) {
        get_config_value(config, apply_empirical_noise_weights, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","products","apply_empirical_weights"});
    }

    // set mapmaker polarization
    naive_mm.run_polarization = rtcproc.run_polarization;
    jinc_mm.run_polarization = rtcproc.run_polarization;
}

template<typename CT>
void Engine::get_pointing_config(CT &config) {
    logger->info("getting pointing config options");

    pointing_source_strategy = "standard";
    if (config.template has_typed<std::string>(std::tuple{"pointing","source_strategy","mode"})) {
        get_config_value(config, pointing_source_strategy, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","mode"},
                         {"standard", "psf_preserve"});
    }

    pointing_fit_gaussian_enabled = (pointing_source_strategy == "standard");
    if (config.template has_typed<bool>(std::tuple{"pointing","source_strategy","fit_gaussian"})) {
        get_config_value(config, pointing_fit_gaussian_enabled, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","fit_gaussian"});
    }

    pointing_fruitloops_center_mode =
        (pointing_source_strategy == "psf_preserve") ? "map_center" : "auto";
    if (config.template has_typed<std::string>(std::tuple{"pointing","source_strategy","fruitloops_center_mode"})) {
        get_config_value(config, pointing_fruitloops_center_mode, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","fruitloops_center_mode"},
                         {"auto", "header", "peak", "map_center"});
    }

    pointing_header_center_max_radius_arcsec = 0.0;
    if (pointing_source_strategy == "standard" &&
        std::isfinite(map_fitter.fitting_region_pix) && map_fitter.fitting_region_pix > 0.0 &&
        std::isfinite(omb.pixel_size_rad) && omb.pixel_size_rad > 0.0) {
        pointing_header_center_max_radius_arcsec =
            map_fitter.fitting_region_pix * omb.pixel_size_rad * RAD_TO_ASEC;
    }
    if (config.template has_typed<double>(std::tuple{"pointing","source_strategy","header_max_radius_arcsec"})) {
        get_config_value(config, pointing_header_center_max_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","header_max_radius_arcsec"},
                         {}, {0.0});
    }

    pointing_header_center_require_coverage = true;
    if (config.template has_typed<bool>(std::tuple{"pointing","source_strategy","header_require_coverage"})) {
        get_config_value(config, pointing_header_center_require_coverage, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","header_require_coverage"});
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
    if (!interp_over_gaps) {
        logger->error("runtime.interp_over_gaps=false is unsupported; set runtime.interp_over_gaps: true");
        std::exit(EXIT_FAILURE);
    }

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
            add_netcdf_var<std::string>(fo, "UNITCONV.UK_CONVENTION", "Rayleigh-Jeans brightness temperature");
            add_netcdf_var<std::string>(fo, "UNITCONV.UK_BASIS",
                                        "monochromatic array center frequency; mJy/beam uses Gaussian beam solid angle to Jy/sr");
            for (const auto &val: calib.arrays) {
                auto name = toltec_io.array_name_map[val];
                // conversion to Rayleigh-Jeans uK brightness temperature
                auto fwhm = (std::get<0>(calib.array_fwhms[val]) + std::get<1>(calib.array_fwhms[val]))/2;
                auto mJy_beam_to_uK = engine_utils::mJy_beam_to_uK(1, toltec_io.array_freq_map[val], fwhm);

                // beam area in steradians
                auto beam_area_rad = 2.*pi*pow(fwhm*FWHM_TO_STD*ASEC_TO_RAD,2);
                // get Jy/pixel
                auto mJy_beam_to_Jy_px = 1e-3/beam_area_rad*pow(omb.pixel_size_rad,2);

                if (omb.sig_unit == "mJy/beam") {
                    // conversion to mJy/beam
                    add_netcdf_var(fo, "to_mJy_beam_"+name, 1);
                    // conversion to MJy/sr
                    add_netcdf_var(fo, "to_MJy_sr_"+name, 1/(calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC));
                    // conversion to Rayleigh-Jeans uK
                    add_netcdf_var(fo, "to_uK_"+name, mJy_beam_to_uK);
                    // conversion to Jy/pixel
                    add_netcdf_var(fo, "to_Jy_pixel_"+name, mJy_beam_to_Jy_px);
                }
                else if (omb.sig_unit == "MJy/sr") {
                    // conversion to mJy/beam
                    add_netcdf_var(fo, "to_mJy_beam_"+name, calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC);
                    // conversion to MJy/Sr
                    add_netcdf_var(fo, "to_MJy_sr_"+name, 1);
                    // conversion to Rayleigh-Jeans uK
                    add_netcdf_var(fo, "to_uK_"+name, calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC*mJy_beam_to_uK);
                    // conversion to Jy/pixel
                    add_netcdf_var(fo, "to_Jy_pixel_"+name, calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC*mJy_beam_to_Jy_px);
                }
                else if (omb.sig_unit == "uK") {
                    // conversion to mJy/beam
                    add_netcdf_var(fo, "to_mJy_beam_"+name, 1/mJy_beam_to_uK);
                    // conversion to MJy/sr
                    add_netcdf_var(fo, "to_MJy_sr_"+name, 1/mJy_beam_to_uK/(calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC));
                    // conversion to Rayleigh-Jeans uK
                    add_netcdf_var(fo, "to_uK_"+name, 1);
                    // conversion to Jy/pixel
                    add_netcdf_var(fo, "to_Jy_pixel_"+name, (1/mJy_beam_to_uK)*mJy_beam_to_Jy_px);
                }
                else if (omb.sig_unit == "Jy/pixel") {
                    // conversion to mJy/beam
                    add_netcdf_var(fo, "to_mJy_beam_"+name, 1/mJy_beam_to_Jy_px);
                    // conversion to MJy/sr
                    add_netcdf_var(fo, "to_MJy_sr_"+name, (1/mJy_beam_to_Jy_px)/(calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC));
                    // conversion to Rayleigh-Jeans uK
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
            add_netcdf_var(fo, "BEAMMAP.CONVERGENCE_RADIUS_ARCSEC", beammap_convergence_radius_arcsec);
            add_netcdf_var(fo, "BEAMMAP.ITER_MAX", beammap_iter_max);
            add_netcdf_var(fo, "BEAMMAP.PHASE_SPLIT_ENABLED", beammap_phase_split_enabled);
            add_netcdf_var(fo, "BEAMMAP.LOCATOR_ITER", beammap_locator_iter);
            add_netcdf_var(fo, "BEAMMAP.MEASUREMENT_START_ITER", beammap_measurement_start_iter);
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
            for (const auto &arr: calib.arrays) {
                auto name = toltec_io.array_name_map[arr];
                add_netcdf_var(fo, "JINC_A_"+name, jinc_mm.shape_params[arr][0]);
                add_netcdf_var(fo, "JINC_B_"+name, jinc_mm.shape_params[arr][1]);
                add_netcdf_var(fo, "JINC_C_"+name, jinc_mm.shape_params[arr][2]);
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
        add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.ENABLED", rtcproc.filter_edge_guard.enabled);
        add_netcdf_var<std::string>(fo, "CONFIG.TODFILTER.EDGE_GUARD.MODE", rtcproc.filter_edge_guard.mode);
        add_netcdf_var<std::string>(fo, "CONFIG.TODFILTER.EDGE_GUARD.COMBINE", rtcproc.filter_edge_guard.combine);
        add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.CONTEXT_SAMPLES", rtcproc.filter_edge_guard.context_samples);
        add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.GUARD_SAMPLES", rtcproc.filter_edge_guard.guard_samples);
        add_netcdf_var(fo, "CONFIG.TOD.OUTER_CONTEXT_SAMPLES", telescope.outer_scans_chunk);
        add_netcdf_var(fo, "CONFIG.TOD.OUTPUT_OUTER_CONTEXT_SAMPLES", rtcproc.tod_output_outer_context_samples);
        add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.MIN_SAMPLES", rtcproc.filter_edge_guard.min_samples);
        add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.EXTRA_SAMPLES", rtcproc.filter_edge_guard.extra_samples);
        add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.MAX_SAMPLES", rtcproc.filter_edge_guard.max_samples);
        add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.IIR_SETTLE_ATTENUATION", rtcproc.filter_edge_guard.iir_settle_attenuation);
        add_netcdf_var(fo, "CONFIG.DOWNSAMPLED", rtcproc.run_downsample);
        add_netcdf_var(fo, "CONFIG.CALIBRATED", rtcproc.run_calibrate);
        add_netcdf_var(fo, "CONFIG.EXTINCTION", rtcproc.run_extinction);
        add_netcdf_var<std::string>(fo, "CONFIG.EXTINCTION.EXTMODEL", rtcproc.calibration.extinction_model);
        add_netcdf_var<std::string>(fo, "CONFIG.WEIGHT.TYPE", ptcproc.weighting_type);
        add_netcdf_var(fo, "CONFIG.WEIGHT.SOURCE_MASK_RADIUS_ARCSEC", ptcproc.source_mask_radius_arcsec);
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
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.PRE_WINDOW_SEC", rtcproc.impulsive_capture.snippet_pre_window_sec);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.POST_WINDOW_SEC", rtcproc.impulsive_capture.snippet_post_window_sec);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.ENABLED", rtcproc.impulsive_coincidence.enabled);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_GOOD_FRAC", rtcproc.impulsive_coincidence.min_good_frac);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.EVENT_SCORE_THRESH", rtcproc.impulsive_coincidence.event_score_thresh);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_USED", rtcproc.impulsive_coincidence.min_det_used);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_FRAC", rtcproc.impulsive_coincidence.min_impulsive_det_frac);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_ALIGNMENT_FRAC", rtcproc.impulsive_coincidence.min_alignment_frac);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_NETWORKS_ALIGNED", rtcproc.impulsive_coincidence.min_networks_aligned);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_OVERRIDE_THRESH", rtcproc.impulsive_coincidence.high_score_override_thresh);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_MIN_NETWORKS", rtcproc.impulsive_coincidence.high_score_min_networks_aligned);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.CLUSTER_TOL_SEC", rtcproc.impulsive_coincidence.cluster_tol_sec);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.PRE_WINDOW_SEC", rtcproc.impulsive_coincidence.mask_pre_window_sec);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.POST_WINDOW_SEC", rtcproc.impulsive_coincidence.mask_post_window_sec);
        add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MAX_FLAGGED_FRAC", rtcproc.impulsive_coincidence.max_flagged_fraction);
        // The RTC TOD now records line-audit provenance at file creation time.
        // When raw obs output is enabled, add_tod_header() reopens the same file;
        // skip re-adding those vars to avoid NetCDF duplicate-name failures.
        if (fo.getVar("CONFIG.RTC.LINE_AUDIT.ENABLED").isNull()) {
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.ENABLED", rtcproc.line_audit.enabled);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.LINE_MIN_HZ", rtcproc.line_audit.line_min_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.LINE_MAX_HZ", rtcproc.line_audit.line_max_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.SEGMENT_SEC", rtcproc.line_audit.segment_sec);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_SEGMENT_SEC", rtcproc.line_audit.min_segment_sec);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.OVERLAP_FRAC", rtcproc.line_audit.overlap_frac);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CONTINUUM_RADIUS_BINS", rtcproc.line_audit.continuum_radius_bins);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PROMINENCE_THRESH", rtcproc.line_audit.prominence_thresh);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CM_PROMINENCE_THRESH", rtcproc.line_audit.cm_prominence_thresh);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_GOOD_FRAC", rtcproc.line_audit.min_good_frac);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_WINDOWS", rtcproc.line_audit.min_windows);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MAX_PEAKS_PER_DETECTOR", rtcproc.line_audit.max_peaks_per_detector);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MAX_DET", rtcproc.line_audit.max_det);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_DET_FOR_NETWORK", rtcproc.line_audit.min_det_for_network);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CLUSTER_TOL_HZ", rtcproc.line_audit.cluster_tol_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTOR_FRAC", rtcproc.line_audit.notch_min_detector_frac);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTORS", rtcproc.line_audit.notch_min_detectors);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_CM_PROMINENCE", rtcproc.line_audit.notch_min_cm_prominence);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_MIN_PROMINENCE", rtcproc.line_audit.detector_min_prominence);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_MIN_LINE_POWER_FRAC", rtcproc.line_audit.detector_min_line_power_frac);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.BAD_DETECTOR_MAX_CLUSTER_FRAC", rtcproc.line_audit.bad_detector_max_cluster_frac);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PRE_FILTER_ENABLED", rtcproc.line_audit.pre_filter_enabled);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_ENABLED", rtcproc.line_audit.post_filter_enabled);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_SHARED_NOTCHES", rtcproc.line_audit.post_filter_apply_shared_notches);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_DETECTOR_NOTCHES", rtcproc.line_audit.post_filter_apply_detector_notches);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_ITERATIONS", rtcproc.line_audit.post_filter_apply_iterations);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_LINE_MIN_HZ", rtcproc.line_audit.post_filter_line_min_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_LINE_MAX_HZ", rtcproc.line_audit.post_filter_line_max_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_MODEL_PROTECTED_ENABLED", rtcproc.line_audit.ptc_model_protected_enabled);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_REQUIRE_MODEL_SUBTRACTED", rtcproc.line_audit.ptc_require_model_subtracted);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_FIXED_NOTCHES", rtcproc.line_audit.ptc_apply_fixed_notches);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_SHARED_NOTCHES", rtcproc.line_audit.ptc_apply_shared_notches);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_DETECTOR_NOTCHES", rtcproc.line_audit.ptc_apply_detector_notches);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_ITERATIONS", rtcproc.line_audit.ptc_apply_iterations);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_LINE_MIN_HZ", rtcproc.line_audit.ptc_line_min_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_LINE_MAX_HZ", rtcproc.line_audit.ptc_line_max_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_ENABLED", rtcproc.line_audit.fixed_notch_enabled);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_COUNT", static_cast<int>(rtcproc.line_audit.fixed_notch_freqs_hz.size()));
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_WIDTH_COUNT", static_cast<int>(rtcproc.line_audit.fixed_notch_widths_hz.size()));
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_EXCLUSION_HALF_WIDTH_HZ", rtcproc.line_audit.fixed_notch_exclusion_half_width_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_SHARED_NOTCHES", rtcproc.line_audit.apply_shared_notches);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_SUPPORT_NETWORKS", rtcproc.line_audit.apply_min_support_networks);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_DETECTOR_FRAC", rtcproc.line_audit.apply_min_detector_frac);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_CM_PROMINENCE", rtcproc.line_audit.apply_min_common_mode_prominence);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_WIDTH_SCALE", rtcproc.line_audit.apply_width_scale);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_WIDTH_HZ", rtcproc.line_audit.apply_min_width_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MAX_WIDTH_HZ", rtcproc.line_audit.apply_max_width_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MAX_NOTCHES", rtcproc.line_audit.apply_max_notches);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_CLUSTER_TOL_HZ", rtcproc.line_audit.apply_cluster_tol_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_PROMINENCE", rtcproc.line_audit.detector_notch_min_prominence);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_LINE_POWER_FRAC", rtcproc.line_audit.detector_notch_min_line_power_frac);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MAX_NOTCHES", rtcproc.line_audit.detector_notch_max_notches);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_WIDTH_SCALE", rtcproc.line_audit.detector_notch_width_scale);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_WIDTH_HZ", rtcproc.line_audit.detector_notch_min_width_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MAX_WIDTH_HZ", rtcproc.line_audit.detector_notch_max_width_hz);
            add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_CONTEXT_SAMPLES", rtcproc.line_audit.detector_notch_context_samples);
        }
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
        add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.ENABLED", ptcproc.busy_row_suppression.enabled);
        add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.REQUIRE_BUSY_VETO", ptcproc.busy_row_suppression.require_busy_veto);
        add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.MIN_CAND_CLUSTERS", ptcproc.busy_row_suppression.min_candidate_clusters);
        add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.MIN_MAX_RESID_Z", ptcproc.busy_row_suppression.min_max_unflagged_residual_z);
        add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.FACTOR", ptcproc.busy_row_suppression.factor);
        add_netcdf_var(fo, "CONFIG.CLEANED", ptcproc.run_clean);
        add_netcdf_var<std::string>(fo, "CONFIG.CLEANED.MODESEL", ptcproc.cleaner.active_cleaner_label());
        add_netcdf_var(fo, "CONFIG.CLEANED.MP.ENABLED", ptcproc.cleaner.marchenko_pastur.enabled);
        add_netcdf_var(fo, "CONFIG.CLEANED.MP.BANDLOW_HZ", ptcproc.cleaner.marchenko_pastur.band_low_Hz);
        add_netcdf_var(fo, "CONFIG.CLEANED.MP.BANDHIGH_HZ", ptcproc.cleaner.marchenko_pastur.band_high_Hz);
        add_netcdf_var(fo, "CONFIG.CLEANED.MP.MAXMODES", ptcproc.cleaner.marchenko_pastur.max_modes);
        std::string adaptive_offsets_joined;
        for (std::size_t i = 0; i < ptcproc.cleaner.adaptive_selector.candidate_offsets.size(); ++i) {
            if (i > 0) {
                adaptive_offsets_joined += ",";
            }
            adaptive_offsets_joined += std::to_string(ptcproc.cleaner.adaptive_selector.candidate_offsets[i]);
        }
        std::string adaptive_grouping_joined;
        for (std::size_t i = 0; i < ptcproc.cleaner.adaptive_selector.grouping.size(); ++i) {
            if (i > 0) {
                adaptive_grouping_joined += ",";
            }
            adaptive_grouping_joined += ptcproc.cleaner.adaptive_selector.grouping[i];
        }
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.ENABLED", ptcproc.cleaner.adaptive_selector.enabled);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.MIN_GOOD_FRAC", ptcproc.cleaner.adaptive_selector.min_good_frac);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.MAX_DET", ptcproc.cleaner.adaptive_selector.max_det);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.MAX_SAMPLES", ptcproc.cleaner.adaptive_selector.max_samples);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.MAX_PAIRS", ptcproc.cleaner.adaptive_selector.max_pairs);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.CLIP_Z", ptcproc.cleaner.adaptive_selector.clip_z);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.LOW_WEIGHT", ptcproc.cleaner.adaptive_selector.low_weight);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.TAIL_WEIGHT", ptcproc.cleaner.adaptive_selector.tail_weight);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.TOPMODE_WEIGHT", ptcproc.cleaner.adaptive_selector.topmode_weight);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.REG_WEIGHT", ptcproc.cleaner.adaptive_selector.reg_weight);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.LOWMIN_HZ", ptcproc.cleaner.adaptive_selector.low_band_Hz[0]);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.LOWMAX_HZ", ptcproc.cleaner.adaptive_selector.low_band_Hz[1]);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.MIDMIN_HZ", ptcproc.cleaner.adaptive_selector.mid_band_Hz[0]);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.MIDMAX_HZ", ptcproc.cleaner.adaptive_selector.mid_band_Hz[1]);
        add_netcdf_var<std::string>(fo, "CONFIG.CLEANED.ADAPT.CANDIDATE_OFFSETS", adaptive_offsets_joined);
        add_netcdf_var<std::string>(fo, "CONFIG.CLEANED.ADAPT.GROUPING", adaptive_grouping_joined);
        add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.LOG_CANDIDATES", ptcproc.cleaner.adaptive_selector.log_candidates);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.ENABLED", ptcproc.second_pass_local.enabled);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_SPIKE_SIGMA", ptcproc.second_pass_local.min_spike_sigma);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_GOOD_FRAC", ptcproc.second_pass_local.min_good_frac);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.BASELINE_WINDOW_SEC", ptcproc.second_pass_local.baseline_window_sec);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.SIGMA_SCALE", ptcproc.second_pass_local.sigma_scale);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.DELTA_SIGMA_SCALE", ptcproc.second_pass_local.delta_sigma_scale);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.RAW_CAND_REL_SIGMA_SCALE", ptcproc.second_pass_local.raw_candidate_rel_sigma_scale);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.RAW_WINDOW_SEC", ptcproc.second_pass_local.raw_window_sec);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.RAW_HALF_PEAK_FRAC", ptcproc.second_pass_local.raw_half_peak_frac);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.RAW_MAX_WIDTH_SEC", ptcproc.second_pass_local.raw_max_width_sec);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.DELTA_WINDOW_SEC", ptcproc.second_pass_local.delta_window_sec);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.DELTA_HALF_PEAK_FRAC", ptcproc.second_pass_local.delta_half_peak_frac);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.DELTA_MAX_WIDTH_SEC", ptcproc.second_pass_local.delta_max_width_sec);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MAX_STEP_SHIFT_Z", ptcproc.second_pass_local.max_step_shift_z);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MERGE_WITHIN_DET_SEC", ptcproc.second_pass_local.merge_within_detector_sec);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.CLUSTER_EVENTS_SEC", ptcproc.second_pass_local.cluster_events_sec);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_CLUSTER_DETECTORS", ptcproc.second_pass_local.min_cluster_detectors);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.HIGH_SCORE_CLUSTER_OVERRIDE", ptcproc.second_pass_local.high_score_cluster_override);
        add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MAX_AUTO_FLAG_CLUSTERS", ptcproc.second_pass_local.max_auto_flag_clusters_per_network);

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
        add_netcdf_var(fo, "CONFIG.FRUITLOOPS.PEAKFRAC", ptcproc.fruit_loops_peak_fraction_limit);
        add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSNR", ptcproc.fruit_loops_local_snr_floor);
        add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_INNER", ptcproc.fruit_loops_local_sigma_inner_radius_arcsec);
        add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_OUTER", ptcproc.fruit_loops_local_sigma_outer_radius_arcsec);
        add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_EDGE", ptcproc.fruit_loops_local_sigma_edge_guard_arcsec);
        add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_MINPIX", ptcproc.fruit_loops_local_sigma_min_pixels);
        add_netcdf_var(fo, "CONFIG.FRUITLOOPS.ADAPT_SUPPORT_RAD", ptcproc.fruit_loops_adaptive_support_radius_arcsec);
        add_netcdf_var(fo, "CONFIG.FRUITLOOPS.ADAPT_SUPPORT_FWHM", ptcproc.fruit_loops_adaptive_support_radius_fwhm);
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

    write_netcdf_atomic(tod_filename[name], [&](netCDF::NcFile &fo) {

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

    if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
        // Keep the RTC line-audit tuning alongside the RTC TOD so offline audits
        // can recover the exact per-run thresholds without the sidecar YAML.
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.ENABLED", rtcproc.line_audit.enabled);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.LINE_MIN_HZ", rtcproc.line_audit.line_min_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.LINE_MAX_HZ", rtcproc.line_audit.line_max_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.SEGMENT_SEC", rtcproc.line_audit.segment_sec);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_SEGMENT_SEC", rtcproc.line_audit.min_segment_sec);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.OVERLAP_FRAC", rtcproc.line_audit.overlap_frac);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CONTINUUM_RADIUS_BINS", rtcproc.line_audit.continuum_radius_bins);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PROMINENCE_THRESH", rtcproc.line_audit.prominence_thresh);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CM_PROMINENCE_THRESH", rtcproc.line_audit.cm_prominence_thresh);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_GOOD_FRAC", rtcproc.line_audit.min_good_frac);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_WINDOWS", rtcproc.line_audit.min_windows);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MAX_PEAKS_PER_DETECTOR", rtcproc.line_audit.max_peaks_per_detector);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MAX_DET", rtcproc.line_audit.max_det);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_DET_FOR_NETWORK", rtcproc.line_audit.min_det_for_network);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CLUSTER_TOL_HZ", rtcproc.line_audit.cluster_tol_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTOR_FRAC", rtcproc.line_audit.notch_min_detector_frac);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTORS", rtcproc.line_audit.notch_min_detectors);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_CM_PROMINENCE", rtcproc.line_audit.notch_min_cm_prominence);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_MIN_PROMINENCE", rtcproc.line_audit.detector_min_prominence);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_MIN_LINE_POWER_FRAC", rtcproc.line_audit.detector_min_line_power_frac);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.BAD_DETECTOR_MAX_CLUSTER_FRAC", rtcproc.line_audit.bad_detector_max_cluster_frac);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PRE_FILTER_ENABLED", rtcproc.line_audit.pre_filter_enabled);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_ENABLED", rtcproc.line_audit.post_filter_enabled);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_SHARED_NOTCHES", rtcproc.line_audit.post_filter_apply_shared_notches);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_DETECTOR_NOTCHES", rtcproc.line_audit.post_filter_apply_detector_notches);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_ITERATIONS", rtcproc.line_audit.post_filter_apply_iterations);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_LINE_MIN_HZ", rtcproc.line_audit.post_filter_line_min_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_LINE_MAX_HZ", rtcproc.line_audit.post_filter_line_max_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_MODEL_PROTECTED_ENABLED", rtcproc.line_audit.ptc_model_protected_enabled);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_REQUIRE_MODEL_SUBTRACTED", rtcproc.line_audit.ptc_require_model_subtracted);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_FIXED_NOTCHES", rtcproc.line_audit.ptc_apply_fixed_notches);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_SHARED_NOTCHES", rtcproc.line_audit.ptc_apply_shared_notches);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_DETECTOR_NOTCHES", rtcproc.line_audit.ptc_apply_detector_notches);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_ITERATIONS", rtcproc.line_audit.ptc_apply_iterations);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_LINE_MIN_HZ", rtcproc.line_audit.ptc_line_min_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_LINE_MAX_HZ", rtcproc.line_audit.ptc_line_max_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_ENABLED", rtcproc.line_audit.fixed_notch_enabled);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_COUNT", static_cast<int>(rtcproc.line_audit.fixed_notch_freqs_hz.size()));
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_WIDTH_COUNT", static_cast<int>(rtcproc.line_audit.fixed_notch_widths_hz.size()));
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_EXCLUSION_HALF_WIDTH_HZ", rtcproc.line_audit.fixed_notch_exclusion_half_width_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_SHARED_NOTCHES", rtcproc.line_audit.apply_shared_notches);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_SUPPORT_NETWORKS", rtcproc.line_audit.apply_min_support_networks);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_DETECTOR_FRAC", rtcproc.line_audit.apply_min_detector_frac);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_CM_PROMINENCE", rtcproc.line_audit.apply_min_common_mode_prominence);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_WIDTH_SCALE", rtcproc.line_audit.apply_width_scale);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_WIDTH_HZ", rtcproc.line_audit.apply_min_width_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MAX_WIDTH_HZ", rtcproc.line_audit.apply_max_width_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MAX_NOTCHES", rtcproc.line_audit.apply_max_notches);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_CLUSTER_TOL_HZ", rtcproc.line_audit.apply_cluster_tol_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_PROMINENCE", rtcproc.line_audit.detector_notch_min_prominence);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_LINE_POWER_FRAC", rtcproc.line_audit.detector_notch_min_line_power_frac);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MAX_NOTCHES", rtcproc.line_audit.detector_notch_max_notches);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_WIDTH_SCALE", rtcproc.line_audit.detector_notch_width_scale);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_WIDTH_HZ", rtcproc.line_audit.detector_notch_min_width_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MAX_WIDTH_HZ", rtcproc.line_audit.detector_notch_max_width_hz);
        add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_CONTEXT_SAMPLES", rtcproc.line_audit.detector_notch_context_samples);
    }

    const Eigen::Index n_tod_output_scans_for_stream =
        (prod_t == engine_utils::toltecIO::rtc_timestream) ? n_tod_output_scans_rtc : n_tod_output_scans_ptc;
    const bool tod_output_mini =
        (prod_t == engine_utils::toltecIO::rtc_timestream) ? rtcproc.tod_output_mini : ptcproc.tod_output_mini;
    const bool tod_output_outer =
        (prod_t == engine_utils::toltecIO::rtc_timestream) ? rtcproc.tod_output_outer : ptcproc.tod_output_outer;

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
    raw_scan_indices_v.putAtt(
        "comment",
        tod_output_outer
            ? "indices in output timebase: inner_start, inner_end, outer_start, outer_end"
            : "indices in output timebase; outer=inner (output stores inner scans only)");
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

    auto add_scan_int_var = [&](const std::string &name, const std::string &comment) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, n_scans_dim);
        v.putAtt("units", "samples");
        v.putAtt("comment", comment);
        std::vector<int> init(static_cast<std::size_t>(n_tod_output_scans_for_stream), -2147483647);
        v.putVar(init.data());
    };
    auto add_scan_double_var = [&](const std::string &name, const std::string &units,
                                   const std::string &comment) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, n_scans_dim);
        v.putAtt("units", units);
        v.putAtt("comment", comment);
        std::vector<double> init(static_cast<std::size_t>(n_tod_output_scans_for_stream),
                                 std::numeric_limits<double>::quiet_NaN());
        v.putVar(init.data());
    };
    add_scan_int_var("tod_filter_edge_guard_pre_samples", "samples flagged at the start of this output scan by the TOD filter edge guard");
    add_scan_int_var("tod_filter_edge_guard_post_samples", "samples flagged at the end of this output scan by the TOD filter edge guard");
    add_scan_int_var("tod_filter_edge_guard_flagged_samples", "detector-samples flagged by the TOD filter edge guard");
    add_scan_double_var("tod_filter_edge_guard_flagged_frac", "N/A", "fraction of time samples guarded at this output scan edge");

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
        add_rtc_det_int("rtc_despike_local_raw_accepted_event_count",
                        "per-detector count of locally detrended raw candidate events accepted by the compact-raw gate");
        add_rtc_det_int("rtc_despike_local_flagged_sample_count",
                        "per-detector count of samples flagged by accepted compact-raw local-residual events");
        add_rtc_det_int("rtc_despike_local_exceed_count",
                        "legacy alias for rtc_despike_local_flagged_sample_count");
        add_rtc_det_int("rtc_despike_local_raw_reject_count",
                        "per-detector count of locally detrended raw candidate events rejected by the compact-raw gate");
        add_rtc_det_int("rtc_despike_delta_spike_count",
                        "per-detector count of delta-domain spikes identified by the RTC despiker");
        add_rtc_det_int("rtc_despike_local_delta_candidate_count",
                        "per-detector count of locally detrended delta candidate events considered by the compact-delta gate");
        add_rtc_det_int("rtc_despike_local_delta_accepted_event_count",
                        "per-detector count of locally detrended delta candidate events accepted by the compact-delta gate");
        add_rtc_det_int("rtc_despike_local_delta_exceed_count",
                        "legacy alias for rtc_despike_local_delta_accepted_event_count");
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
        add_rtc_det_int("rtc_detector_notch_n_applied",
                        "per-detector count of post-filter detector-local RTC notches applied");
        add_rtc_det_double("rtc_detector_notch_primary_freq_hz",
                           "frequency of the strongest detector-local post-filter RTC notch applied");
        add_rtc_det_double("rtc_detector_notch_primary_width_hz",
                           "bandwidth of the strongest detector-local post-filter RTC notch applied");
        add_rtc_det_double("rtc_detector_notch_primary_prominence",
                           "PSD prominence of the strongest detector-local post-filter RTC notch applied");
        add_rtc_det_double("rtc_detector_notch_primary_line_power_frac",
                           "line-power fraction of the strongest detector-local post-filter RTC notch applied");
        add_rtc_det_double("rtc_detector_notch_rms_before",
                           "robust RMS of the detector RTC timestream before detector-local post-filter notching");
        add_rtc_det_double("rtc_detector_notch_rms_after",
                           "robust RMS of the detector RTC timestream after detector-local post-filter notching");

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
                       "detectors passing the step-mask valid-sample threshold and finite robust scale");
        add_rtc_nw_int("rtc_network_impulsive_n_det_used",
                       "detectors passing the impulsive-coincidence valid-sample threshold and finite robust scale");
        add_rtc_nw_int("rtc_network_line_audit_n_det_used",
                       "detectors analyzed by the pre-filter RTC line audit in each network block");
        add_rtc_nw_double("rtc_network_line_audit_shared_freq_hz",
                          "frequency of the strongest shared narrowband RTC line family in each network block");
        add_rtc_nw_int("rtc_network_line_audit_shared_detector_count",
                       "number of detectors participating in the strongest shared narrowband RTC line family");
        add_rtc_nw_double("rtc_network_line_audit_shared_detector_frac",
                          "fraction of audited detectors participating in the strongest shared narrowband RTC line family");
        add_rtc_nw_double("rtc_network_line_audit_shared_median_prominence",
                          "median detector-level PSD prominence of the strongest shared narrowband RTC line family");
        add_rtc_nw_double("rtc_network_line_audit_shared_max_prominence",
                          "maximum detector-level PSD prominence of the strongest shared narrowband RTC line family");
        add_rtc_nw_double("rtc_network_line_audit_shared_width_hz",
                          "median linewidth of the strongest shared narrowband RTC line family");
        add_rtc_nw_double("rtc_network_line_audit_shared_line_power_frac",
                          "median detector-level line-power fraction of the strongest shared narrowband RTC line family");
        add_rtc_nw_double("rtc_network_line_audit_shared_common_mode_freq_hz",
                          "matched common-mode line frequency for the strongest shared narrowband RTC line family");
        add_rtc_nw_double("rtc_network_line_audit_shared_common_mode_prominence",
                          "matched common-mode PSD prominence for the strongest shared narrowband RTC line family");
        add_rtc_nw_double("rtc_network_line_audit_shared_notch_score",
                          "shared-line notch score, detector fraction times median prominence");
        add_rtc_nw_int("rtc_network_line_audit_shared_recommend_notch",
                       "1 if the strongest shared narrowband RTC line family met the current notch-candidate criteria");
        add_rtc_nw_int("rtc_network_line_audit_n_applied_notches",
                       "number of chunk-level shared-line RTC notches actually applied to this scan");
        add_rtc_nw_int("rtc_network_line_audit_shared_applied_notch",
                       "1 if the strongest shared narrowband RTC line family in this network matched an applied chunk-level RTC notch");
        add_rtc_nw_double("rtc_network_line_audit_shared_applied_freq_hz",
                          "center frequency of the applied chunk-level RTC notch matched to the strongest shared narrowband RTC line family");
        add_rtc_nw_double("rtc_network_line_audit_shared_applied_width_hz",
                          "full-width bandwidth of the applied chunk-level RTC notch matched to the strongest shared narrowband RTC line family");
        add_rtc_nw_int("rtc_network_line_audit_shared_applied_support_network_count",
                       "number of networks supporting the applied chunk-level RTC notch matched to the strongest shared narrowband RTC line family");
        add_rtc_nw_int("rtc_network_line_audit_detector_candidate_uid",
                       "UID of the strongest detector-local RTC line candidate in each network block; -2147483647 means none");
        add_rtc_nw_double("rtc_network_line_audit_detector_candidate_freq_hz",
                          "frequency of the strongest detector-local RTC line candidate");
        add_rtc_nw_double("rtc_network_line_audit_detector_candidate_prominence",
                          "PSD prominence of the strongest detector-local RTC line candidate");
        add_rtc_nw_double("rtc_network_line_audit_detector_candidate_line_power_frac",
                          "line-power fraction of the strongest detector-local RTC line candidate");
        add_rtc_nw_double("rtc_network_line_audit_detector_candidate_cluster_detector_frac",
                          "shared-cluster detector fraction associated with the strongest detector-local RTC line candidate");
        add_rtc_nw_int("rtc_network_line_audit_detector_candidate_recommend_flag",
                       "1 if the strongest detector-local RTC line candidate met the current bad-detector criteria");
        auto add_rtc_nw_line_audit_diag = [&](const std::string &prefix, const std::string &stage) {
            add_rtc_nw_int(prefix + "_n_det_used",
                           "detectors analyzed by the " + stage + " RTC line audit in each network block");
            add_rtc_nw_double(prefix + "_shared_freq_hz",
                              "frequency of the strongest shared narrowband " + stage + " RTC line family in each network block");
            add_rtc_nw_int(prefix + "_shared_detector_count",
                           "number of detectors participating in the strongest shared narrowband " + stage + " RTC line family");
            add_rtc_nw_double(prefix + "_shared_detector_frac",
                              "fraction of audited detectors participating in the strongest shared narrowband " + stage + " RTC line family");
            add_rtc_nw_double(prefix + "_shared_median_prominence",
                              "median detector-level PSD prominence of the strongest shared narrowband " + stage + " RTC line family");
            add_rtc_nw_double(prefix + "_shared_max_prominence",
                              "maximum detector-level PSD prominence of the strongest shared narrowband " + stage + " RTC line family");
            add_rtc_nw_double(prefix + "_shared_width_hz",
                              "median linewidth of the strongest shared narrowband " + stage + " RTC line family");
            add_rtc_nw_double(prefix + "_shared_line_power_frac",
                              "median detector-level line-power fraction of the strongest shared narrowband " + stage + " RTC line family");
            add_rtc_nw_double(prefix + "_shared_common_mode_freq_hz",
                              "matched common-mode line frequency for the strongest shared narrowband " + stage + " RTC line family");
            add_rtc_nw_double(prefix + "_shared_common_mode_prominence",
                              "matched common-mode PSD prominence for the strongest shared narrowband " + stage + " RTC line family");
            add_rtc_nw_double(prefix + "_shared_notch_score",
                              "shared-line notch score, detector fraction times median prominence");
            add_rtc_nw_int(prefix + "_shared_recommend_notch",
                           "1 if the strongest shared narrowband " + stage + " RTC line family met the current notch-candidate criteria");
            add_rtc_nw_int(prefix + "_n_applied_notches",
                           "number of chunk-level shared-line RTC notches actually applied in the " + stage + " stage");
            add_rtc_nw_int(prefix + "_shared_applied_notch",
                           "1 if the strongest shared narrowband " + stage + " RTC line family in this network matched an applied chunk-level RTC notch");
            add_rtc_nw_double(prefix + "_shared_applied_freq_hz",
                              "center frequency of the applied chunk-level RTC notch matched to the strongest shared narrowband " + stage + " RTC line family");
            add_rtc_nw_double(prefix + "_shared_applied_width_hz",
                              "full-width bandwidth of the applied chunk-level RTC notch matched to the strongest shared narrowband " + stage + " RTC line family");
            add_rtc_nw_int(prefix + "_shared_applied_support_network_count",
                           "number of networks supporting the applied chunk-level RTC notch matched to the strongest shared narrowband " + stage + " RTC line family");
            add_rtc_nw_int(prefix + "_detector_candidate_uid",
                           "UID of the strongest detector-local " + stage + " RTC line candidate in each network block; -2147483647 means none");
            add_rtc_nw_double(prefix + "_detector_candidate_freq_hz",
                              "frequency of the strongest detector-local " + stage + " RTC line candidate");
            add_rtc_nw_double(prefix + "_detector_candidate_prominence",
                              "PSD prominence of the strongest detector-local " + stage + " RTC line candidate");
            add_rtc_nw_double(prefix + "_detector_candidate_line_power_frac",
                              "line-power fraction of the strongest detector-local " + stage + " RTC line candidate");
            add_rtc_nw_double(prefix + "_detector_candidate_cluster_detector_frac",
                              "shared-cluster detector fraction associated with the strongest detector-local " + stage + " RTC line candidate");
            add_rtc_nw_int(prefix + "_detector_candidate_recommend_flag",
                           "1 if the strongest detector-local " + stage + " RTC line candidate met the current bad-detector criteria");
        };
        add_rtc_nw_line_audit_diag("rtc_network_post_line_audit", "post-filter");
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
        add_rtc_nw_double("rtc_network_impulsive_score_median",
                          "median detector impulsive-event score within each RTC network block");
        add_rtc_nw_double("rtc_network_impulsive_score_max",
                          "maximum detector impulsive-event score within each RTC network block");
        add_rtc_nw_double("rtc_network_impulsive_det_frac",
                          "fraction of diagnostic-used detectors with impulsive-event score above the impulsive coincidence threshold");
        add_rtc_nw_double("rtc_network_impulsive_alignment_frac",
                          "fraction of impulsive-active detectors aligned in the dominant impulsive time cluster");
        add_rtc_nw_int("rtc_network_impulsive_dominant_sample",
                       "dominant aligned impulsive sample within each RTC network block; -2147483647 means unavailable");
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
        add_rtc_nw_int("rtc_network_impulsive_mask_applied",
                       "1 if impulsive_coincidence_mask flagged a time window for this RTC network block, else 0");
        add_rtc_nw_int("rtc_network_impulsive_mask_start_sample",
                       "inclusive starting sample of the applied impulsive_coincidence_mask window; -2147483647 means none");
        add_rtc_nw_int("rtc_network_impulsive_mask_end_sample",
                       "inclusive ending sample of the applied impulsive_coincidence_mask window; -2147483647 means none");
        add_rtc_nw_int("rtc_network_impulsive_mask_window_samples",
                       "number of RTC time samples in the applied impulsive_coincidence_mask window");
        add_rtc_nw_int("rtc_network_impulsive_mask_n_det_masked",
                       "number of detectors included in the applied impulsive_coincidence_mask window");
        add_rtc_nw_int("rtc_network_impulsive_mask_n_det_samples_flagged",
                       "number of previously good detector-samples newly flagged by impulsive_coincidence_mask");
        add_rtc_nw_double("rtc_network_impulsive_mask_flagged_fraction",
                          "fraction of previously good detector-samples in the network block newly flagged by impulsive_coincidence_mask");
        add_rtc_nw_int("rtc_network_impulsive_mask_candidate_available",
                       "1 if impulsive_coincidence_mask found a candidate for this RTC network block, else 0");
        add_rtc_nw_int("rtc_network_impulsive_mask_local_trigger",
                       "1 if the selected impulsive candidate satisfied the within-network trigger thresholds, else 0");
        add_rtc_nw_int("rtc_network_impulsive_mask_cross_network_trigger",
                       "1 if the selected impulsive candidate satisfied a cross-network alignment trigger, else 0");
        add_rtc_nw_int("rtc_network_impulsive_mask_high_score_override_trigger",
                       "1 if the selected impulsive candidate satisfied the looser high-score cross-network override, else 0");
        add_rtc_nw_int("rtc_network_impulsive_mask_rejected_max_fraction",
                       "1 if the selected impulsive candidate was rejected only because its proposed flagged fraction exceeded the configured limit");
        add_rtc_nw_int("rtc_network_impulsive_mask_candidate_center_sample",
                       "center sample of the selected impulsive candidate before any cross-network recentering; -2147483647 means unavailable");
        add_rtc_nw_int("rtc_network_impulsive_mask_cluster_center_sample",
                       "median aligned sample of the selected cross-network impulsive cluster; -2147483647 means unavailable");
        add_rtc_nw_int("rtc_network_impulsive_mask_cluster_network_count",
                       "number of distinct networks participating in the selected impulsive candidate cluster");
        add_rtc_nw_int("rtc_network_impulsive_mask_cluster_active_count",
                       "number of detector-level impulsive events in the selected within-network cluster");
        add_rtc_nw_int("rtc_network_impulsive_mask_total_active_count",
                       "total number of detector-level impulsive events above threshold in the selected network block");
        add_rtc_nw_double("rtc_network_impulsive_mask_cluster_peak_score",
                          "maximum impulsive-event score found within the selected cross-network impulsive cluster");
        add_rtc_nw_double("rtc_network_impulsive_mask_override_score",
                          "score used by the high-score override path after combining the selected cluster peak with the strongest candidate score seen in participating networks");
        add_rtc_nw_int("rtc_network_impulsive_mask_override_uses_network_peak",
                       "1 if rtc_network_impulsive_mask_override_score came from a participating network's strongest candidate rather than the selected cluster peak");
        add_rtc_nw_double("rtc_network_impulsive_mask_proposed_flagged_fraction",
                          "fraction of previously good detector-samples that the selected impulsive mask window would newly flag before any rejection");

        if (rtcproc.impulsive_capture.enabled) {
            const auto n_slots = static_cast<std::size_t>(std::max<Eigen::Index>(rtcproc.impulsive_capture.max_events_per_network, 1));
            const double rtc_fsmp = rtcproc.run_downsample ? telescope.d_fsmp : telescope.fsmp;
            const auto snippet_pre = static_cast<std::size_t>(std::max(0.0, std::round(rtcproc.impulsive_capture.snippet_pre_window_sec * rtc_fsmp)));
            const auto snippet_post = static_cast<std::size_t>(std::max(0.0, std::round(rtcproc.impulsive_capture.snippet_post_window_sec * rtc_fsmp)));
            const auto n_snippet = snippet_pre + snippet_post + 1;
            netCDF::NcDim n_rtc_impulsive_slots_dim = fo.addDim("n_rtc_impulsive_slots", n_slots);
            netCDF::NcDim n_rtc_impulsive_samples_dim = fo.addDim("n_rtc_impulsive_samples", n_snippet);

            netCDF::NcVar offset_v = fo.addVar("rtc_impulsive_snippet_offset_samples", netCDF::ncInt, n_rtc_impulsive_samples_dim);
            offset_v.putAtt("units", "samples");
            offset_v.putAtt("comment", "sample offsets relative to rtc_impulsive_slot_event_sample");
            std::vector<int> offsets(n_snippet, fill_int);
            for (std::size_t i = 0; i < n_snippet; ++i) {
                offsets[i] = static_cast<int>(i) - static_cast<int>(snippet_pre);
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
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_raw_accepted_event_count",
                                 "count of locally detrended raw candidate events accepted by the compact-raw gate for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_flagged_sample_count",
                                 "count of samples flagged by accepted compact-raw local-residual events for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_exceed_count",
                                 "legacy alias for rtc_impulsive_slot_local_flagged_sample_count");
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_raw_reject_count",
                                 "count of locally detrended raw candidate events rejected by the compact-raw gate for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_delta_spike_count",
                                 "count of delta-domain spikes for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_delta_candidate_count",
                                 "count of locally detrended delta candidate events considered by the compact-delta gate for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_delta_accepted_event_count",
                                 "count of locally detrended delta candidate events accepted by the compact-delta gate for a captured detector slot");
            add_rtc_imp_slot_int("rtc_impulsive_slot_local_delta_exceed_count",
                                 "legacy alias for rtc_impulsive_slot_local_delta_accepted_event_count");
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

        if (ptcproc.second_pass_local.enabled) {
            netCDF::NcVar added_flag_v = fo.addVar("ptc_second_pass_added_flag", netCDF::ncByte, dims);
            added_flag_v.putAtt("units", "N/A");
            added_flag_v.putAtt("comment",
                                "0=not added by PTC second-pass residual deglitching, 1=newly flagged by that pass");
            added_flag_v.setChunking(chunkMode, chunkSizes);

            const int fill_value = -2147483647;
            const double fill_double = std::numeric_limits<double>::quiet_NaN();
            netCDF::NcDim n_nws_ptc_second_pass_dim = fo.addDim("n_nws_ptc_second_pass", calib.n_nws);
            netCDF::NcVar nw_ids_v = fo.addVar("ptc_second_pass_network_ids", netCDF::ncInt, n_nws_ptc_second_pass_dim);
            nw_ids_v.putAtt("units", "N/A");
            nw_ids_v.putAtt("comment", "network IDs corresponding to n_nws_ptc_second_pass axis");
            std::vector<int> nw_ids(static_cast<std::size_t>(calib.n_nws), fill_value);
            for (Eigen::Index i = 0; i < calib.n_nws; ++i) {
                nw_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
            }
            nw_ids_v.putVar(nw_ids.data());

            std::vector<netCDF::NcDim> ptc_second_pass_dims = {n_scans_dim, n_nws_ptc_second_pass_dim};
            auto add_ptc_second_pass_int = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, ptc_second_pass_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<int> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                      static_cast<std::size_t>(calib.n_nws), fill_value);
                v.putVar(init.data());
            };
            auto add_ptc_second_pass_double = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, ptc_second_pass_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<double> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                         static_cast<std::size_t>(calib.n_nws), fill_double);
                v.putVar(init.data());
            };

            add_ptc_second_pass_int("ptc_second_pass_busy_network_vetoed",
                                    "1 if this network had more candidate second-pass clusters than the auto-flag limit and was diagnostic-only");
            add_ptc_second_pass_int("ptc_second_pass_n_candidate_clusters",
                                    "number of candidate second-pass residual clusters in this scan/network");
            add_ptc_second_pass_int("ptc_second_pass_n_candidate_events",
                                    "number of candidate detector-local residual events contributing to candidate clusters");
            add_ptc_second_pass_int("ptc_second_pass_n_accepted_clusters",
                                    "number of candidate clusters accepted for auto-flagging after the busy-network veto");
            add_ptc_second_pass_int("ptc_second_pass_n_accepted_events",
                                    "number of accepted detector-local residual events contributing to auto-flagging");
            add_ptc_second_pass_int("ptc_second_pass_n_det_with_added_flags",
                                    "number of detectors in this scan/network with at least one sample newly flagged by the PTC second pass");
            add_ptc_second_pass_int("ptc_second_pass_max_unflagged_residual_uid",
                                    "UID of the detector with the largest absolute unflagged post-PCA residual in this scan/network");
            add_ptc_second_pass_int("ptc_second_pass_top_candidate_cluster_sample",
                                    "median sample of the strongest candidate second-pass cluster; -2147483647 means none");
            add_ptc_second_pass_int("ptc_second_pass_top_candidate_cluster_n_detectors",
                                    "number of distinct detectors contributing to the strongest candidate second-pass cluster");
            add_ptc_second_pass_int("ptc_second_pass_top_candidate_cluster_n_events",
                                    "number of merged detector events contributing to the strongest candidate second-pass cluster");
            add_ptc_second_pass_int("ptc_second_pass_top_event_kind",
                                    "kind code of the strongest accepted second-pass event (0=raw_like,1=delta_like,-2147483647 means none)");
            add_ptc_second_pass_int("ptc_second_pass_top_event_uid",
                                    "UID of the strongest accepted second-pass event; -2147483647 means none");
            add_ptc_second_pass_int("ptc_second_pass_top_event_sample",
                                    "sample of the strongest accepted second-pass event; -2147483647 means none");
            add_ptc_second_pass_double("ptc_second_pass_existing_flagged_fraction",
                                       "fraction of detector-samples already flagged before the PTC second pass in this scan/network");
            add_ptc_second_pass_double("ptc_second_pass_proposed_flagged_fraction",
                                       "fraction of detector-samples that the accepted PTC second-pass flags would cover in this scan/network");
            add_ptc_second_pass_double("ptc_second_pass_newly_flagged_fraction",
                                       "fraction of previously good detector-samples newly flagged by the PTC second pass in this scan/network");
            add_ptc_second_pass_double("ptc_second_pass_max_unflagged_residual_z",
                                       "largest absolute standardized residual remaining on previously unflagged PTC samples in this scan/network");
            add_ptc_second_pass_double("ptc_second_pass_top_candidate_cluster_peak_score",
                                       "peak event score of the strongest candidate second-pass cluster in this scan/network");
            add_ptc_second_pass_double("ptc_second_pass_top_event_score",
                                       "score of the strongest accepted second-pass event; NaN means none");
        }

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

        if (ptcproc.busy_row_suppression.enabled) {
            const int fill_int = -2147483647;
            const double fill_double = std::numeric_limits<double>::quiet_NaN();
            netCDF::NcDim n_nws_wbusy_dim = fo.addDim("n_nws_busy_row_suppression", calib.n_nws);
            netCDF::NcVar nw_ids_v = fo.addVar("weight_busy_row_suppression_network_ids", netCDF::ncInt, n_nws_wbusy_dim);
            nw_ids_v.putAtt("units", "N/A");
            nw_ids_v.putAtt("comment", "network IDs corresponding to n_nws_busy_row_suppression axis");
            std::vector<int> nw_ids(static_cast<std::size_t>(calib.n_nws), fill_int);
            for (Eigen::Index i = 0; i < calib.n_nws; ++i) {
                nw_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
            }
            nw_ids_v.putVar(nw_ids.data());

            std::vector<netCDF::NcDim> wbusy_dims = {n_scans_dim, n_nws_wbusy_dim};
            auto add_wbusy_int = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, wbusy_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<int> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                      static_cast<std::size_t>(calib.n_nws), fill_int);
                v.putVar(init.data());
            };
            auto add_wbusy_double = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, wbusy_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<double> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                         static_cast<std::size_t>(calib.n_nws), fill_double);
                v.putVar(init.data());
            };

            add_wbusy_int("weight_busy_row_suppression_applied",
                          "1 if busy-row weight suppression was applied to this scan/network block, else 0");
            add_wbusy_int("weight_busy_row_suppression_busy_network_vetoed",
                          "1 if this scan/network exceeded the second-pass busy-network veto threshold, else 0");
            add_wbusy_int("weight_busy_row_suppression_n_candidate_clusters",
                          "candidate second-pass residual cluster count used by the busy-row suppression rule");
            add_wbusy_int("weight_busy_row_suppression_n_det_weighted",
                          "detectors with positive map weight multiplied by the busy-row suppression factor");
            add_wbusy_double("weight_busy_row_suppression_factor",
                             "multiplicative factor applied by busy-row suppression to positive detector map weights");
            add_wbusy_double("weight_busy_row_suppression_max_unflagged_residual_z",
                             "largest absolute unflagged post-PCA residual z used by the busy-row suppression rule");
        }

        if (ptcproc.cleaner.adaptive_selector.enabled) {
            const int fill_int = -2147483647;
            const double fill_double = std::numeric_limits<double>::quiet_NaN();
            netCDF::NcDim n_nws_adapt_dim = fo.addDim("n_nws_adaptive_pca", calib.n_nws);
            netCDF::NcVar nw_ids_v = fo.addVar("adaptive_pca_network_ids", netCDF::ncInt, n_nws_adapt_dim);
            nw_ids_v.putAtt("units", "N/A");
            nw_ids_v.putAtt("comment", "network IDs corresponding to n_nws_adaptive_pca axis");
            std::vector<int> nw_ids(static_cast<std::size_t>(calib.n_nws), fill_int);
            for (Eigen::Index i = 0; i < calib.n_nws; ++i) {
                nw_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
            }
            nw_ids_v.putVar(nw_ids.data());

            std::vector<netCDF::NcDim> adapt_dims = {n_scans_dim, n_nws_adapt_dim};
            auto add_adapt_int = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, adapt_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<int> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                      static_cast<std::size_t>(calib.n_nws), fill_int);
                v.putVar(init.data());
            };
            auto add_adapt_double = [&](const std::string &name, const std::string &comment) {
                netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, adapt_dims);
                v.putAtt("units", "N/A");
                v.putAtt("comment", comment);
                std::vector<double> init(static_cast<std::size_t>(n_tod_output_scans_for_stream) *
                                         static_cast<std::size_t>(calib.n_nws), fill_double);
                v.putVar(init.data());
            };

            add_adapt_int("adaptive_pca_selector_used",
                          "1 if the bounded adaptive PCA selector evaluated this scan/network block, else 0");
            add_adapt_int("adaptive_pca_selector_fallback",
                          "1 if adaptive PCA selector fell back to the configured baseline cut, else 0");
            add_adapt_int("adaptive_pca_baseline_k",
                          "configured baseline PCA cut for this scan/network block");
            add_adapt_int("adaptive_pca_chosen_k",
                          "adaptive PCA cut selected for this scan/network block");
            add_adapt_int("adaptive_pca_runnerup_k",
                          "second-best adaptive PCA cut for this scan/network block");
            add_adapt_int("adaptive_pca_n_candidates",
                          "number of candidate PCA cuts evaluated for this scan/network block");
            add_adapt_int("adaptive_pca_n_det_input",
                          "input detector count in this scan/network block before selector filtering");
            add_adapt_int("adaptive_pca_n_det_used",
                          "detector count retained for adaptive selector scoring");
            add_adapt_int("adaptive_pca_n_time_used",
                          "sample count retained for adaptive selector scoring");
            add_adapt_int("adaptive_pca_sample_step",
                          "time decimation factor used by the adaptive selector");
            add_adapt_double("adaptive_pca_chosen_score",
                             "final normalized adaptive selector score for the chosen PCA cut");
            add_adapt_double("adaptive_pca_runnerup_score",
                             "final normalized adaptive selector score for the runner-up PCA cut");
            add_adapt_double("adaptive_pca_score_margin",
                             "chosen minus runner-up score margin; more negative is a clearer adaptive choice");
            add_adapt_double("adaptive_pca_chosen_med_abs_corr",
                             "median absolute detector-detector correlation for the chosen adaptive PCA cut");
            add_adapt_double("adaptive_pca_chosen_cm_low_mid_ratio",
                             "common-mode low/mid bandpower ratio for the chosen adaptive PCA cut");
            add_adapt_double("adaptive_pca_chosen_tail4_binom_z",
                             "tail-excess metric for the chosen adaptive PCA cut");
            add_adapt_double("adaptive_pca_chosen_top_mode_frac",
                             "top residual covariance mode fraction for the chosen adaptive PCA cut");
            add_adapt_double("adaptive_pca_eig_solve_msec",
                             "milliseconds spent solving eigenmodes before adaptive scoring");
            add_adapt_double("adaptive_pca_candidate_eval_msec",
                             "milliseconds spent scoring candidate PCA cuts after eigen solve");
            add_adapt_double("adaptive_pca_total_msec",
                             "total adaptive PCA milliseconds for this scan/network block");
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
    double omb_size = 8*omb.n_rows*omb.n_cols*(omb.signal.size() + omb.weight.size() +
                                               omb.kernel.size() + omb.coverage.size() +
                                               omb.grid_weight.size())/1e9;

    logger->info("estimated size of map buffer {:.2f} GB", omb_size);

    mb_size_total = mb_size_total + omb_size;

    // print info if coadd is requested
    if (run_coadd) {
        logger->info("coadd map buffer rows: {}", cmb.n_rows);
        logger->info("coadd map buffer cols: {}", cmb.n_cols);

        // make a rough estimate of memory usage for coadd map buffer
        double cmb_size = 8*cmb.n_rows*cmb.n_cols*(cmb.signal.size() + cmb.weight.size() +
                                                   cmb.kernel.size() + cmb.coverage.size() +
                                                   cmb.grid_weight.size())/1e9;

        logger->info("estimated size of coadd buffer {:.2f} GB", cmb_size);

        mb_size_total = mb_size_total + cmb_size;

        // output info if coadd noise maps are requested
        if (run_noise) {
            logger->info("coadd map buffer noise maps: {}", cmb.n_noise);
            // make a rough estimate of memory usage for coadd noise maps
            double nmb_size = 8*cmb.n_rows*cmb.n_cols*cmb.noise.size()*cmb.n_noise/1e9;
            logger->info("estimated size of noise buffer {:.2f} GB", nmb_size);
            mb_size_total = mb_size_total + nmb_size;
        }
    }
    else {
        // output info if obs noise maps are requested
        if (run_noise) {
            logger->info("observation map buffer noise maps: {}", omb.n_noise);
            // make a rough estimate of memory usage for obs noise maps
            double nmb_size = 8*omb.n_rows*omb.n_cols*omb.noise.size()*omb.n_noise/1e9;
            logger->info("estimated size of noise buffer {:.2f} GB", nmb_size);
            mb_size_total = mb_size_total + nmb_size;
        }
    }

    logger->info("estimated size of all maps {:.2f} GB", mb_size_total);
    logger->info("number of scans: {}",telescope.scan_indices.cols());
    if (run_tod_output) {
        if (tod_output_type == "rtc" || tod_output_type == "both") {
            logger->info("RTC TOD output scans: {}", n_tod_output_scans_rtc);
            logger->info("RTC TOD output mode: {}{}",
                         rtcproc.tod_output_mini ? "mini" : "full",
                         rtcproc.tod_output_outer ? "_outer" : "");
        }
        if (tod_output_type == "ptc" || tod_output_type == "both") {
            logger->info("PTC TOD output scans: {}", n_tod_output_scans_ptc);
            logger->info("PTC TOD output mode: {}", ptcproc.tod_output_mini ? "mini" : "full");
        }
    }
    logger->info("RTC diagnostics sidecar output: standard");
    logger->info("PTC diagnostics sidecar output: standard");
    logger->info("Map diagnostics sidecar output: standard");

    // test getting memory usage for fun
    /*struct sysinfo memInfo;
    long long totalPhysMem = memInfo.totalram;
    totalPhysMem *= memInfo.mem_unit;

    logger->info("total physical memory available {} GB", (totalPhysMem/1024)/1e7);*/
    auto phys_memory_kb = engine_utils::get_phys_memory();
    if (phys_memory_kb >= 0) {
        logger->info("physical memory used {:.2f} GB", static_cast<double>(phys_memory_kb) / 1e7);
    } else {
        logger->debug("physical memory used unavailable on this platform");
    }
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
    f << "-TOD filter edge guard enabled: " << rtcproc.filter_edge_guard.enabled << "\n";
    f << "-TOD filter edge guard context samples: " << rtcproc.filter_edge_guard.context_samples << "\n";
    f << "-TOD filter edge guard samples per edge: " << rtcproc.filter_edge_guard.guard_samples << "\n";
    f << "-TOD loaded outer context samples: " << telescope.outer_scans_chunk << "\n";
    f << "-RTC detector notch context samples: " << rtcproc.line_audit.detector_notch_context_samples << "\n";
    f << "-RTC fixed line-audit notch enabled: " << rtcproc.line_audit.fixed_notch_enabled << "\n";
    f << "-RTC fixed line-audit notch count: " << rtcproc.line_audit.fixed_notch_freqs_hz.size() << "\n";
    f << "-PTC model-protected line-audit notch enabled: " << rtcproc.line_audit.ptc_model_protected_enabled << "\n";
    f << "-PTC model-protected line-audit require model: " << rtcproc.line_audit.ptc_require_model_subtracted << "\n";
    f << "-PTC model-protected fixed/shared/detector notches: "
      << rtcproc.line_audit.ptc_apply_fixed_notches << "/"
      << rtcproc.line_audit.ptc_apply_shared_notches << "/"
      << rtcproc.line_audit.ptc_apply_detector_notches << "\n";
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

    try {
    logger->debug("adding unit conversions");

    // conversion to Rayleigh-Jeans uK brightness temperature
    auto fwhm = (std::get<0>(calib.array_fwhms[calib.arrays(i)]) + std::get<1>(calib.array_fwhms[calib.arrays(i)]))/2;
    auto mJy_beam_to_uK = engine_utils::mJy_beam_to_uK(1, toltec_io.array_freq_map[calib.arrays(i)], fwhm);

    // beam area in steradians
    auto beam_area_rad = 2.*pi*pow(fwhm*FWHM_TO_STD*ASEC_TO_RAD,2);
    // get Jy/pixel
    auto mJy_beam_to_Jy_px = 1e-3/beam_area_rad*pow(mb->pixel_size_rad,2);

    fits_io->at(i).pfits->pHDU().addKey("UKCONV", "RJ", "uK convention: Rayleigh-Jeans brightness temperature");
    fits_io->at(i).pfits->pHDU().addKey("UKBASIS", "Jy/sr", "uK basis: monochromatic intensity per steradian");

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

    auto add_double_key = [&](const std::string &key, double value, const std::string &comment,
                              double fallback = 0.0) {
        if (!std::isfinite(value)) {
            logger->warn("PHDU key '{}' non-finite ({}) for array {} in {}; using fallback {}",
                         key, value, name, fits_io->at(i).filepath, fallback);
            value = fallback;
        }
        try {
            fits_io->at(i).pfits->pHDU().addKey(key, value, comment);
        } catch (const CCfits::FitsError &e) {
            throw std::runtime_error(
                fmt::format("failed PHDU float key '{}' for array '{}' (file={} value={}): {}",
                            key, name, fits_io->at(i).filepath, value, e.message()));
        }
    };

    // add unit conversions
    if (rtcproc.run_calibrate) {
        if (mb->sig_unit == "mJy/beam") {
            // conversion to mJy/beam
            fits_io->at(i).pfits->pHDU().addKey("to_mJy/beam", 1, "Conversion to mJy/beam");
            // conversion to MJy/sr
            add_double_key("to_MJy/sr",
                           1/(calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC),
                           "Conversion to MJy/sr");
            // conversion to Rayleigh-Jeans uK
            add_double_key("to_uK", mJy_beam_to_uK, "Conversion to Rayleigh-Jeans uK");
            // conversion to Jy/pixel
            add_double_key("to_Jy/pixel", mJy_beam_to_Jy_px, "Conversion to Jy/pixel");
        }
        else if (mb->sig_unit == "MJy/sr") {
            // conversion to mJy/beam
            add_double_key("to_mJy/beam",
                           calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC,
                           "Conversion to mJy/beam");
            // conversion to MJy/Sr
            fits_io->at(i).pfits->pHDU().addKey("to_MJy/sr", 1, "Conversion to MJy/sr");
            // conversion to Rayleigh-Jeans uK
            add_double_key("to_uK",
                           calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC*mJy_beam_to_uK,
                           "Conversion to Rayleigh-Jeans uK");
            // conversion to Jy/pixel
            add_double_key("to_Jy/pixel",
                           calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC*mJy_beam_to_Jy_px,
                           "Conversion to Jy/pixel");
        }
        else if (mb->sig_unit == "uK") {
            // conversion to mJy/beam
            add_double_key("to_mJy/beam", 1/mJy_beam_to_uK, "Conversion to mJy/beam");
            // conversion to MJy/sr
            add_double_key("to_MJy/sr",
                           1/mJy_beam_to_uK/(calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC),
                           "Conversion to MJy/sr");
            // conversion to Rayleigh-Jeans uK
            fits_io->at(i).pfits->pHDU().addKey("to_uK", 1, "Conversion to Rayleigh-Jeans uK");
            // conversion to Jy/pixel
            add_double_key("to_Jy/pixel", (1/mJy_beam_to_uK)*mJy_beam_to_Jy_px, "Conversion to Jy/pixel");
        }
        else if (mb->sig_unit == "Jy/pixel") {
            // conversion to mJy/beam
            add_double_key("to_mJy/beam", 1/mJy_beam_to_Jy_px, "Conversion to mJy/beam");
            // conversion to MJy/sr
            add_double_key("to_MJy/sr",
                           (1/mJy_beam_to_Jy_px)/(calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC),
                           "Conversion to MJy/sr");
            // conversion to Rayleigh-Jeans uK
            add_double_key("to_uK", mJy_beam_to_uK/mJy_beam_to_Jy_px, "Conversion to Rayleigh-Jeans uK");
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
        add_double_key("HEADER.SOURCE.FLUX_MJYPERBEAM", beammap_fluxes_mJy_beam[name], "Source flux (mJy/beam)");
        add_double_key("HEADER.SOURCE.FLUX_MJYPERSR", beammap_fluxes_MJy_Sr[name], "Source flux (MJy/sr)");

        add_double_key("BEAMMAP.ITER_TOLERANCE", beammap_iter_tolerance, "Beammap iteration tolerance");
        add_double_key("BEAMMAP.CONVERGENCE_RADIUS_ARCSEC", beammap_convergence_radius_arcsec,
                       "Beammap convergence aperture radius (arcsec)");
        fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.ITER_MAX", beammap_iter_max, "Beammap max iterations");
        fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.PHASE_SPLIT_ENABLED", beammap_phase_split_enabled,
                                            "Beammap locator/measurement phases enabled");
        fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.LOCATOR_ITER", beammap_locator_iter,
                                            "Beammap locator iteration");
        fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.MEASUREMENT_START_ITER", beammap_measurement_start_iter,
                                            "Beammap first measurement iteration");
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
            add_double_key("BEAMMAP.REF_X_T", ref_x_t, "Az rotation center (arcsec)");
            add_double_key("BEAMMAP.REF_Y_T", ref_y_t, "Alt rotation center (arcsec)");
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
    add_double_key("EXPTIME", mb->exposure_time, "Exposure time (sec)");
    // add pixel axes
    fits_io->at(i).pfits->pHDU().addKey("RADESYS", telescope.pixel_axes, "Coord Reference Frame");
    const double source_ra = get_tel_header_scalar("Header.Source.Ra", 0.0);
    const double source_dec = get_tel_header_scalar("Header.Source.Dec", 0.0);
    // add source ra
    add_double_key("SRC_RA", source_ra, "Source RA (radians)");
    // add source dec
    add_double_key("SRC_DEC", source_dec, "Source Dec (radians)");
    // add map tangent point ra
    add_double_key("TAN_RA", source_ra, "Map Tangent Point RA (radians)");
    //add map tangent point dec
    add_double_key("TAN_DEC", source_dec, "Map Tangent Point Dec (radians)");
    // add mean alt
    add_double_key("MEAN_EL", RAD_TO_DEG*get_tel_data_mean("TelElAct", 0.0), "Mean Elevation (deg)");
    // add mean az
    add_double_key("MEAN_AZ", RAD_TO_DEG*get_tel_data_mean("TelAzAct", 0.0), "Mean Azimuth (deg)");
    // add mean parallactic angle
    add_double_key("MEAN_PA", RAD_TO_DEG*get_tel_data_mean("ActParAng", 0.0), "Mean Parallactic angle (deg)");

    logger->debug("adding beamsizes");

    // add beamsizes
    if (std::get<0>(calib.array_fwhms[calib.arrays(i)]) >= std::get<1>(calib.array_fwhms[calib.arrays(i)])) {
        add_double_key("BMAJ", std::get<0>(calib.array_fwhms[calib.arrays(i)]), "beammaj (arcsec)");
        add_double_key("BMIN", std::get<1>(calib.array_fwhms[calib.arrays(i)]), "beammin (arcsec)");
        add_double_key("BPA", calib.array_pas[calib.arrays(i)]*RAD_TO_DEG, "beampa (deg)");
    }
    else {
        add_double_key("BMAJ", std::get<1>(calib.array_fwhms[calib.arrays(i)]), "beammaj (arcsec)");
        add_double_key("BMIN", std::get<0>(calib.array_fwhms[calib.arrays(i)]), "beammin (arcsec)");
        add_double_key("BPA", (calib.array_pas[calib.arrays(i)] + pi/2)*RAD_TO_DEG, "beampa (deg)");
    }

    fits_io->at(i).pfits->pHDU().addKey("BUNIT", mb->sig_unit, "bunit");

    // add jinc shape params
    if (map_method=="jinc") {
        logger->debug("adding jinc params");

        add_double_key("JINC_R", jinc_mm.r_max, "Jinc filter R_max");
        add_double_key("JINC_A", jinc_mm.shape_params[calib.arrays(i)][0], "Jinc filter param a");
        add_double_key("JINC_B", jinc_mm.shape_params[calib.arrays(i)][1], "Jinc filter param b");
        add_double_key("JINC_C", jinc_mm.shape_params[calib.arrays(i)][2], "Jinc filter param c");
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
    add_double_key("MEAN_TAU", mean_tau, "mean tau (" + name + ")");

    // add sample rate
    add_double_key("SAMPRATE", telescope.fsmp, "sample rate (Hz)");

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

    double rms = 0.0;

    if (redu_type != "beammap" && std::isfinite(mb->median_err(i)) &&
        mb->median_err(i) > std::numeric_limits<double>::epsilon()) {
        rms = pow(mb->median_err(i), 0.5);
    }
    else if (redu_type != "beammap" && std::isfinite(mb->median_err(i)) &&
             mb->median_err(i) < 0.0) {
        logger->warn("negative median_err for PHDU {} in {}; using OOF_RMS=0", name,
                     fits_io->at(i).filepath);
    }

    // out-of-focus holography parameters
    if (! telescope.sim_obs) {
	    logger->debug("adding oof params");
	    add_double_key("OOF_RMS", rms, "rms of map background (" + mb->sig_unit +")");
	    add_double_key("OOF_W", toltec_io.array_wavelength_map[calib.arrays(i)]/1000., "wavelength (m)");
	    fits_io->at(i).pfits->pHDU().addKey("OOF_ID", static_cast<int>(toltec_io.array_wavelength_map[calib.arrays(i)]*1000), "instrument id");
	    add_double_key("OOF_T", 3.0, "taper (dB)");
	    add_double_key("OOF_M2X", get_tel_header_scalar("Header.M2.XReq", 0.0)/1000.*1e6, "oof m2x (microns)");
	    add_double_key("OOF_M2Y", get_tel_header_scalar("Header.M2.YReq", 0.0)/1000.*1e6, "oof m2y (microns)");
	    add_double_key("OOF_M2Z", get_tel_header_scalar("Header.M2.ZReq", 0.0)/1000.*1e6, "oof m2z (microns)");

	    add_double_key("OOF_RO", 25., "outer diameter of the antenna (m)");
	    add_double_key("OOF_RI", 1.65, "inner diameter of the antenna (m)");
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
    add_double_key("CONFIG.DESPIKE.LOCAL.WINDOW_SEC",
                   rtcproc.despiker.local_residual.window_sec,
                   "Local-residual despike smoothing window");
    add_double_key("CONFIG.DESPIKE.LOCAL.SIGMA_SCALE",
                   rtcproc.despiker.local_residual.sigma_scale,
                   "Local-residual despike raw threshold scale");
    add_double_key("CONFIG.DESPIKE.LOCAL.DELTA_SIGMA_SCALE",
                   rtcproc.despiker.local_residual.delta_sigma_scale,
                   "Local-residual despike delta threshold scale");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.RAW_GATE.ENABLED",
                                        rtcproc.despiker.local_residual.compact_raw_gate.enabled,
                                        "Enable compact morphology gate for local-residual raw candidates");
    add_double_key("CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_REL_SIGMA_SCALE",
                   rtcproc.despiker.local_residual.compact_raw_gate.candidate_rel_sigma_scale,
                   "Candidate threshold scale relative to the accepted local-residual raw threshold");
    add_double_key("CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_SIGMA_SCALE",
                   rtcproc.despiker.local_residual.compact_raw_gate.candidate_rel_sigma_scale *
                       rtcproc.despiker.local_residual.sigma_scale,
                   "Effective candidate threshold scale in units of min_spike_sigma for compact local-residual raw gate");
    add_double_key("CONFIG.DESPIKE.LOCAL.RAW_GATE.WINDOW_SEC",
                   rtcproc.despiker.local_residual.compact_raw_gate.window_sec,
                   "Window used to score compactness of local-residual raw candidates");
    add_double_key("CONFIG.DESPIKE.LOCAL.RAW_GATE.HALF_PEAK_FRAC",
                   rtcproc.despiker.local_residual.compact_raw_gate.half_peak_frac,
                   "Half-peak fraction used to measure local-residual raw candidate width");
    add_double_key("CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_WIDTH_SEC",
                   rtcproc.despiker.local_residual.compact_raw_gate.max_width_sec,
                   "Maximum width allowed for compact local-residual raw candidates");
    add_double_key("CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_STEP_SHIFT_Z",
                   rtcproc.despiker.local_residual.compact_raw_gate.max_step_shift_z,
                   "Maximum allowed pre/post baseline shift for compact local-residual raw candidates");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKE.LOCAL.DELTA_GATE.ENABLED",
                                        rtcproc.despiker.local_residual.compact_delta_gate.enabled,
                                        "Enable compact morphology gate for local-residual delta candidates");
    add_double_key("CONFIG.DESPIKE.LOCAL.DELTA_GATE.WINDOW_SEC",
                   rtcproc.despiker.local_residual.compact_delta_gate.window_sec,
                   "Window used to score compactness of local-residual delta candidates");
    add_double_key("CONFIG.DESPIKE.LOCAL.DELTA_GATE.HALF_PEAK_FRAC",
                   rtcproc.despiker.local_residual.compact_delta_gate.half_peak_frac,
                   "Half-peak fraction used to measure local-residual delta candidate width");
    add_double_key("CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_WIDTH_SEC",
                   rtcproc.despiker.local_residual.compact_delta_gate.max_width_sec,
                   "Maximum width allowed for compact local-residual delta candidates");
    add_double_key("CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_STEP_SHIFT_Z",
                   rtcproc.despiker.local_residual.compact_delta_gate.max_step_shift_z,
                   "Maximum allowed pre/post baseline shift for compact local-residual delta candidates");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODFILTERED", run_any_tod_filter, "TOD Filtered");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODNOTCH", rtcproc.run_tod_notch, "TOD notch enabled");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODIIRHP", rtcproc.run_tod_iir_highpass, "TOD IIR highpass enabled");
    add_double_key("CONFIG.TODIIRHP.FREQ_HZ", rtcproc.filter.iir_highpass_freq_Hz, "TOD IIR highpass cutoff frequency");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODIIRHP.ORDER", rtcproc.filter.iir_highpass_order, "TOD IIR highpass cascaded order");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODIIRHP.ZEROPHASE", rtcproc.filter.iir_highpass_zero_phase, "TOD IIR highpass forward-backward");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODFILTER.EDGE_GUARD.ENABLED", rtcproc.filter_edge_guard.enabled, "TOD filter edge guard enabled");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODFILTER.EDGE_GUARD.MODE", rtcproc.filter_edge_guard.mode, "TOD filter edge guard mode");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODFILTER.EDGE_GUARD.COMBINE", rtcproc.filter_edge_guard.combine, "TOD filter edge guard combine rule");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODFILTER.EDGE_GUARD.CONTEXT_SAMPLES", static_cast<int>(rtcproc.filter_edge_guard.context_samples), "TOD filter context samples");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODFILTER.EDGE_GUARD.GUARD_SAMPLES", static_cast<int>(rtcproc.filter_edge_guard.guard_samples), "TOD filter guarded samples per edge");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TOD.OUTER_CONTEXT_SAMPLES", static_cast<int>(telescope.outer_scans_chunk), "TOD loaded outer context samples");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DOWNSAMPLED", rtcproc.run_downsample, "Downsampled");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CALIBRATED", rtcproc.run_calibrate, "Calibrated");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.EXTINCTION", rtcproc.run_extinction, "Extinction corrected");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.EXTINCTION.EXTMODEL", rtcproc.calibration.extinction_model, "Extinction model");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.TYPE", ptcproc.weighting_type, "Weighting scheme");
    add_double_key("CONFIG.INV_VAR.RTC.WTLOW", rtcproc.lower_inv_var_factor, "RTC lower inv var cutoff");
    add_double_key("CONFIG.INV_VAR.RTC.WTHIGH", rtcproc.upper_inv_var_factor, "RTC upper inv var cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.STEP_MASK.ENABLED",
                                        rtcproc.network_step_mask.enabled,
                                        "Enable RTC network-window step masking");
    add_double_key("CONFIG.RTC.STEP_MASK.STEP_WINDOW_SEC",
                   rtcproc.network_step_mask.step_window_sec,
                   "Window used for RTC step-score estimation");
    add_double_key("CONFIG.RTC.STEP_MASK.STEP_SCORE_THRESH",
                   rtcproc.network_step_mask.step_score_thresh,
                   "Detector step-score threshold for RTC step masking");
    add_double_key("CONFIG.RTC.STEP_MASK.MIN_GOOD_FRAC",
                   rtcproc.network_step_mask.min_good_frac,
                   "Minimum good-sample fraction for RTC step-mask metrics");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.STEP_MASK.MIN_DET_USED",
                                        static_cast<int>(rtcproc.network_step_mask.min_det_used),
                                        "Minimum detectors required in a network for RTC step masking");
    add_double_key("CONFIG.RTC.STEP_MASK.MIN_STEP_DET_FRAC",
                   rtcproc.network_step_mask.min_step_det_frac,
                   "Minimum step-like detector fraction for RTC step masking");
    add_double_key("CONFIG.RTC.STEP_MASK.MIN_ALIGNMENT_FRAC",
                   rtcproc.network_step_mask.min_alignment_frac,
                   "Minimum aligned-step detector fraction for RTC step masking");
    add_double_key("CONFIG.RTC.STEP_MASK.CLUSTER_TOL_SEC",
                   rtcproc.network_step_mask.cluster_tol_sec,
                   "Allowed timing tolerance for aligned RTC step clusters");
    add_double_key("CONFIG.RTC.STEP_MASK.HALF_WIDTH_SEC",
                   rtcproc.network_step_mask.mask_half_width_sec,
                   "Half-width of the applied RTC step-mask window");
    add_double_key("CONFIG.RTC.STEP_MASK.MAX_FLAGGED_FRAC",
                   rtcproc.network_step_mask.max_flagged_fraction,
                   "Maximum allowed newly flagged detector-sample fraction per RTC network mask");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.IMPULSIVE.ENABLED",
                                        rtcproc.impulsive_capture.enabled,
                                        "Enable RTC impulsive-event snippet capture");
    add_double_key("CONFIG.RTC.IMPULSIVE.MIN_GOOD_FRAC",
                   rtcproc.impulsive_capture.min_good_frac,
                   "Minimum good-sample fraction for RTC impulsive capture");
    add_double_key("CONFIG.RTC.IMPULSIVE.MIN_EVENT_Z",
                   rtcproc.impulsive_capture.min_event_z,
                   "Minimum event score for RTC impulsive capture");
    add_double_key("CONFIG.RTC.IMPULSIVE.NEAR_EVENT_Z",
                   rtcproc.impulsive_capture.near_event_z,
                   "Near-threshold z for RTC impulsive counts");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.IMPULSIVE.MAX_EVENTS",
                                        static_cast<int>(rtcproc.impulsive_capture.max_events_per_network),
                                        "Maximum captured impulsive detectors per network");
    add_double_key("CONFIG.RTC.IMPULSIVE.PRE_WINDOW_SEC",
                   rtcproc.impulsive_capture.snippet_pre_window_sec,
                   "Pre-event window of captured RTC impulsive snippets");
    add_double_key("CONFIG.RTC.IMPULSIVE.POST_WINDOW_SEC",
                   rtcproc.impulsive_capture.snippet_post_window_sec,
                   "Post-event window of captured RTC impulsive snippets");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.IMPULSIVE_COINCIDENCE.ENABLED",
                                        rtcproc.impulsive_coincidence.enabled,
                                        "Enable RTC impulsive coincidence masking");
    add_double_key("CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_GOOD_FRAC",
                   rtcproc.impulsive_coincidence.min_good_frac,
                   "Minimum good-sample fraction for RTC impulsive coincidence metrics");
    add_double_key("CONFIG.RTC.IMPULSIVE_COINCIDENCE.EVENT_SCORE_THRESH",
                   rtcproc.impulsive_coincidence.event_score_thresh,
                   "Detector impulsive-event score threshold for RTC impulsive coincidence masking");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_USED",
                                        static_cast<int>(rtcproc.impulsive_coincidence.min_det_used),
                                        "Minimum detectors required in a network for RTC impulsive coincidence masking");
    add_double_key("CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_FRAC",
                   rtcproc.impulsive_coincidence.min_impulsive_det_frac,
                   "Minimum impulsive-active detector fraction for RTC impulsive coincidence masking");
    add_double_key("CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_ALIGNMENT_FRAC",
                   rtcproc.impulsive_coincidence.min_alignment_frac,
                   "Minimum aligned-impulsive detector fraction for RTC impulsive coincidence masking");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_NETWORKS_ALIGNED",
                                        static_cast<int>(rtcproc.impulsive_coincidence.min_networks_aligned),
                                        "Minimum aligned networks required for cross-network RTC impulsive coincidence masking");
    add_double_key("CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_OVERRIDE_THRESH",
                   rtcproc.impulsive_coincidence.high_score_override_thresh,
                   "High-score threshold enabling a looser cross-network RTC impulsive coincidence trigger");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_MIN_NETWORKS",
                                        static_cast<int>(rtcproc.impulsive_coincidence.high_score_min_networks_aligned),
                                        "Minimum aligned networks for the high-score override RTC impulsive coincidence trigger");
    add_double_key("CONFIG.RTC.IMPULSIVE_COINCIDENCE.CLUSTER_TOL_SEC",
                   rtcproc.impulsive_coincidence.cluster_tol_sec,
                   "Allowed timing tolerance for aligned RTC impulsive coincidence clusters");
    add_double_key("CONFIG.RTC.IMPULSIVE_COINCIDENCE.PRE_WINDOW_SEC",
                   rtcproc.impulsive_coincidence.mask_pre_window_sec,
                   "Pre-event window of the applied RTC impulsive coincidence mask");
    add_double_key("CONFIG.RTC.IMPULSIVE_COINCIDENCE.POST_WINDOW_SEC",
                   rtcproc.impulsive_coincidence.mask_post_window_sec,
                   "Post-event window of the applied RTC impulsive coincidence mask");
    add_double_key("CONFIG.RTC.IMPULSIVE_COINCIDENCE.MAX_FLAGGED_FRAC",
                   rtcproc.impulsive_coincidence.max_flagged_fraction,
                   "Maximum allowed newly flagged detector-sample fraction per RTC impulsive coincidence mask");
    add_double_key("CONFIG.INV_VAR.PTC.WTLOW", ptcproc.lower_inv_var_factor, "PTC lower inv var cutoff");
    add_double_key("CONFIG.INV_VAR.PTC.WTHIGH", ptcproc.upper_inv_var_factor, "PTC upper inv var cutoff");
    add_double_key("CONFIG.WEIGHT.PTC.WTLOW", ptcproc.lower_weight_factor, "PTC lower weight cutoff");
    add_double_key("CONFIG.WEIGHT.PTC.WTHIGH", ptcproc.upper_weight_factor, "PTC upper weight cutoff");
    add_double_key("CONFIG.WEIGHT.MEDWTFACTOR", ptcproc.med_weight_factor, "Median weight factor");
    add_double_key("CONFIG.WEIGHT.SRCMASK_ARCSEC", ptcproc.source_mask_radius_arcsec,
                   "Source mask radius for full-weight variance estimation");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.ENABLED",
                                        ptcproc.weight_corr_penalty.enabled,
                                        "Enable per-network corr-based weight penalties");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.MIN_GOOD_FRAC",
                   ptcproc.weight_corr_penalty.min_good_frac,
                   "Minimum unflagged sample fraction per detector");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.MIN_OVERLAP",
                   ptcproc.weight_corr_penalty.min_overlap,
                   "Minimum overlap for pairwise corr metric");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.MAX_SAMPLES",
                                        ptcproc.weight_corr_penalty.max_samples,
                                        "Max sampled timestream points for penalty metrics");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.MAX_PAIRS",
                                        ptcproc.weight_corr_penalty.max_pairs,
                                        "Max sampled detector pairs for corr metric");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.FLOOR",
                   ptcproc.weight_corr_penalty.floor,
                   "Minimum per-network multiplicative weight factor");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.EXPONENT",
                   ptcproc.weight_corr_penalty.exponent,
                   "Exponent shaping corr penalty response");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.PAIR.ENABLED",
                                        ptcproc.weight_corr_penalty.pair_corr.enabled,
                                        "Enable pairwise corr penalty term");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.PAIR.REF",
                   ptcproc.weight_corr_penalty.pair_corr.ref,
                   "Pairwise corr reference value");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.PAIR.SPAN",
                   ptcproc.weight_corr_penalty.pair_corr.span,
                   "Pairwise corr scale span");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.PAIR.WEIGHT",
                   ptcproc.weight_corr_penalty.pair_corr.weight,
                   "Pairwise corr term weight");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.CM_EL.ENABLED",
                                        ptcproc.weight_corr_penalty.cm_el_corr.enabled,
                                        "Enable common-mode elevation corr penalty term");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.CM_EL.REF",
                   ptcproc.weight_corr_penalty.cm_el_corr.ref,
                   "Common-mode elevation corr reference");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.CM_EL.SPAN",
                   ptcproc.weight_corr_penalty.cm_el_corr.span,
                   "Common-mode elevation corr scale span");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.CM_EL.WEIGHT",
                   ptcproc.weight_corr_penalty.cm_el_corr.weight,
                   "Common-mode elevation corr term weight");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.ENABLED",
                                        ptcproc.weight_corr_penalty.cm_low_mid_ratio.enabled,
                                        "Enable common-mode low/mid ratio penalty term");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.REF",
                   ptcproc.weight_corr_penalty.cm_low_mid_ratio.ref,
                   "Common-mode low/mid ratio reference");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.SPAN",
                   ptcproc.weight_corr_penalty.cm_low_mid_ratio.span,
                   "Common-mode low/mid ratio scale span");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.WEIGHT",
                   ptcproc.weight_corr_penalty.cm_low_mid_ratio.weight,
                   "Common-mode low/mid ratio term weight");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMIN_HZ",
                   ptcproc.weight_corr_penalty.cm_low_mid_ratio.low_min_Hz,
                   "Low-band minimum frequency for low/mid ratio");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMAX_HZ",
                   ptcproc.weight_corr_penalty.cm_low_mid_ratio.low_max_Hz,
                   "Low-band maximum frequency for low/mid ratio");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMIN_HZ",
                   ptcproc.weight_corr_penalty.cm_low_mid_ratio.mid_min_Hz,
                   "Mid-band minimum frequency for low/mid ratio");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMAX_HZ",
                   ptcproc.weight_corr_penalty.cm_low_mid_ratio.mid_max_Hz,
                   "Mid-band maximum frequency for low/mid ratio");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.ENABLED",
                                        ptcproc.busy_row_suppression.enabled,
                                        "Enable busy scan/network row weight suppression");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.REQUIRE_BUSY_VETO",
                                        ptcproc.busy_row_suppression.require_busy_veto,
                                        "Require second-pass busy-network veto before suppression");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.MIN_CAND_CLUSTERS",
                                        ptcproc.busy_row_suppression.min_candidate_clusters,
                                        "Minimum candidate residual clusters for suppression");
    add_double_key("CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.MIN_MAX_RESID_Z",
                   ptcproc.busy_row_suppression.min_max_unflagged_residual_z,
                   "Minimum max unflagged residual z for suppression");
    add_double_key("CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.FACTOR",
                   ptcproc.busy_row_suppression.factor,
                   "Busy-row multiplicative weight suppression factor");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED", ptcproc.run_clean, "Cleaned");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.MODESEL",
                                        ptcproc.cleaner.active_cleaner_label(),
                                        "PTC cleaner method");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.MP.ENABLED",
                                        ptcproc.cleaner.marchenko_pastur.enabled,
                                        "Marchenko-Pastur mode selection enabled");
    add_double_key("CONFIG.CLEANED.MP.BANDLOW_HZ",
                   ptcproc.cleaner.marchenko_pastur.band_low_Hz,
                   "MP covariance low-band edge (Hz)");
    add_double_key("CONFIG.CLEANED.MP.BANDHIGH_HZ",
                   ptcproc.cleaner.marchenko_pastur.band_high_Hz,
                   "MP covariance high-band edge (Hz)");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.MP.MAXMODES",
                                        ptcproc.cleaner.marchenko_pastur.max_modes,
                                        "MP max modes considered");
    std::string adaptive_offsets_joined;
    for (std::size_t j = 0; j < ptcproc.cleaner.adaptive_selector.candidate_offsets.size(); ++j) {
        if (j > 0) {
            adaptive_offsets_joined += ",";
        }
        adaptive_offsets_joined += std::to_string(ptcproc.cleaner.adaptive_selector.candidate_offsets[j]);
    }
    std::string adaptive_grouping_joined;
    for (std::size_t j = 0; j < ptcproc.cleaner.adaptive_selector.grouping.size(); ++j) {
        if (j > 0) {
            adaptive_grouping_joined += ",";
        }
        adaptive_grouping_joined += ptcproc.cleaner.adaptive_selector.grouping[j];
    }
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.ADAPT.ENABLED",
                                        ptcproc.cleaner.adaptive_selector.enabled,
                                        "Bounded adaptive PCA selector enabled");
    add_double_key("CONFIG.CLEANED.ADAPT.MIN_GOOD_FRAC",
                   ptcproc.cleaner.adaptive_selector.min_good_frac,
                   "Adaptive PCA minimum unflagged detector fraction");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.ADAPT.MAX_DET",
                                        ptcproc.cleaner.adaptive_selector.max_det,
                                        "Adaptive PCA max detectors used for scoring");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.ADAPT.MAX_SAMPLES",
                                        ptcproc.cleaner.adaptive_selector.max_samples,
                                        "Adaptive PCA max time samples used for scoring");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.ADAPT.MAX_PAIRS",
                                        ptcproc.cleaner.adaptive_selector.max_pairs,
                                        "Adaptive PCA max detector pairs used for scoring");
    add_double_key("CONFIG.CLEANED.ADAPT.CLIP_Z",
                   ptcproc.cleaner.adaptive_selector.clip_z,
                   "Adaptive PCA residual clip threshold");
    add_double_key("CONFIG.CLEANED.ADAPT.LOW_WEIGHT",
                   ptcproc.cleaner.adaptive_selector.low_weight,
                   "Adaptive PCA low-band selector weight");
    add_double_key("CONFIG.CLEANED.ADAPT.TAIL_WEIGHT",
                   ptcproc.cleaner.adaptive_selector.tail_weight,
                   "Adaptive PCA tail selector weight");
    add_double_key("CONFIG.CLEANED.ADAPT.TOPMODE_WEIGHT",
                   ptcproc.cleaner.adaptive_selector.topmode_weight,
                   "Adaptive PCA top-mode selector weight");
    add_double_key("CONFIG.CLEANED.ADAPT.REG_WEIGHT",
                   ptcproc.cleaner.adaptive_selector.reg_weight,
                   "Adaptive PCA regularization-to-baseline weight");
    add_double_key("CONFIG.CLEANED.ADAPT.LOWMIN_HZ",
                   ptcproc.cleaner.adaptive_selector.low_band_Hz[0],
                   "Adaptive PCA low-band minimum frequency");
    add_double_key("CONFIG.CLEANED.ADAPT.LOWMAX_HZ",
                   ptcproc.cleaner.adaptive_selector.low_band_Hz[1],
                   "Adaptive PCA low-band maximum frequency");
    add_double_key("CONFIG.CLEANED.ADAPT.MIDMIN_HZ",
                   ptcproc.cleaner.adaptive_selector.mid_band_Hz[0],
                   "Adaptive PCA mid-band minimum frequency");
    add_double_key("CONFIG.CLEANED.ADAPT.MIDMAX_HZ",
                   ptcproc.cleaner.adaptive_selector.mid_band_Hz[1],
                   "Adaptive PCA mid-band maximum frequency");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.ADAPT.OFFSETS",
                                        adaptive_offsets_joined,
                                        "Adaptive PCA candidate cut offsets");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.ADAPT.GROUPING",
                                        adaptive_grouping_joined,
                                        "Grouping subset where adaptive PCA is active");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.ADAPT.LOGCAND",
                                        ptcproc.cleaner.adaptive_selector.log_candidates,
                                        "Adaptive PCA per-candidate logging enabled");
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
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS.SRCMODE",
                                        ptcproc.fruit_loops_source_center_mode,
                                        "Fruit loops source center mode");
    add_double_key("CONFIG.FRUITLOOPS.HDRMAXR",
                   ptcproc.fruit_loops_header_center_max_radius_arcsec,
                   "Fruit loops header center max radius");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS.HDRCOV",
                                        ptcproc.fruit_loops_header_center_require_coverage,
                                        "Require coverage at header center");
    add_double_key("CONFIG.FRUITLOOPS.S2N", ptcproc.fruit_loops_sig2noise, "Fruit loops S/N");
    add_double_key("CONFIG.FRUITLOOPS.PEAKFRAC", ptcproc.fruit_loops_peak_fraction_limit,
                   "Fruit loops peak fraction");
    add_double_key("CONFIG.FRUITLOOPS.LOCALSNR", ptcproc.fruit_loops_local_snr_floor,
                   "Fruit loops local sigma S/N floor");
    add_double_key("CONFIG.FRUITLOOPS.LOCALSIG_INNER", ptcproc.fruit_loops_local_sigma_inner_radius_arcsec,
                   "Fruit loops local sigma inner annulus");
    add_double_key("CONFIG.FRUITLOOPS.LOCALSIG_OUTER", ptcproc.fruit_loops_local_sigma_outer_radius_arcsec,
                   "Fruit loops local sigma outer annulus");
    add_double_key("CONFIG.FRUITLOOPS.LOCALSIG_EDGE", ptcproc.fruit_loops_local_sigma_edge_guard_arcsec,
                   "Fruit loops local sigma edge guard");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS.LOCALSIG_MINPIX",
                                        ptcproc.fruit_loops_local_sigma_min_pixels,
                                        "Fruit loops local sigma minimum pixels");
    add_double_key("CONFIG.FRUITLOOPS.ADAPT_SUPPORT_RAD",
                   ptcproc.fruit_loops_adaptive_support_radius_arcsec,
                   "Fruit loops adaptive support radius");
    add_double_key("CONFIG.FRUITLOOPS.ADAPT_SUPPORT_FWHM",
                   ptcproc.fruit_loops_adaptive_support_radius_fwhm,
                   "Fruit loops adaptive support FWHM factor");
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
        add_double_key("CONFIG.FRUITLOOPS.FLUX", flux_limit,
                       "Fruit loops flux (" + mb->sig_unit + ")");
    }
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.FRUITLOOPS.MAXITER", ptcproc.fruit_loops_iters, "Fruit loops iterations");

    if (redu_type == "pointing") {
        fits_io->at(i).pfits->pHDU().addKey("CONFIG.POINTING.STRATEGY",
                                            pointing_source_strategy,
                                            "Pointing source strategy");
        fits_io->at(i).pfits->pHDU().addKey("CONFIG.POINTING.FITGAUSS",
                                            pointing_fit_gaussian_enabled,
                                            "Pointing Gaussian fit enabled");
        fits_io->at(i).pfits->pHDU().addKey("CONFIG.POINTING.SRCMODE",
                                            pointing_fruitloops_center_mode,
                                            "Pointing fruit loops source mode");
        add_double_key("CONFIG.POINTING.HDRMAXR",
                       pointing_header_center_max_radius_arcsec,
                       "Pointing header center max radius");
        fits_io->at(i).pfits->pHDU().addKey("CONFIG.POINTING.HDRCOV",
                                            pointing_header_center_require_coverage,
                                            "Pointing header coverage guard");
    }

    // add telescope file header information
    if (mb->obsnums.size()==1) {
        logger->debug("adding tel params");
        for (auto const& [key, val] : telescope.tel_header) {
            if (val.size() < 1 || !std::isfinite(val(0))) {
                logger->warn("skipping tel_header '{}' due to empty/non-finite value", key);
                continue;
            }
            logger->debug("adding {}: {}", key, val);
            add_double_key(key, val(0), key);
        }
    }
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

    try {
        // signal map
        fits_io->at(map_index).add_hdu("signal_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->signal[i]);
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
        fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");
        fits_io->at(map_index).hdus.back()->addKey("BUNIT", mb->sig_unit, "Physical unit of image values");
        fits_io->at(map_index).hdus.back()->addKey("DESCRIP", "Signal map in map units", "Image product description");

        // weight map
        fits_io->at(map_index).add_hdu("weight_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->weight[i]);
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
        const std::string weight_unit = "1/("+mb->sig_unit+")^2";
        fits_io->at(map_index).hdus.back()->addKey("UNIT", weight_unit, "Unit of map");
        fits_io->at(map_index).hdus.back()->addKey("BUNIT", weight_unit, "Physical unit of image values");
        fits_io->at(map_index).hdus.back()->addKey("TYPE",
            (run_noise_products && run_noise && apply_empirical_noise_weights) ? "empirical" : "formal",
            "Weight calibration type");
        fits_io->at(map_index).hdus.back()->addKey(
            "DESCRIP",
            (run_noise_products && run_noise && apply_empirical_noise_weights)
                ? "Jackknife-calibrated inverse variance weight map"
                : "Formal mapmaker inverse variance weight map",
            "Image product description");
        if (i < mb->noise_weight_scale.size()) {
            fits_io->at(map_index).hdus.back()->addKey("EMP_SCALE", mb->noise_weight_scale(i),
                                                       "Empirical weight scale");
        }
        if (i < mb->noise_weight_median_ratio.size()) {
            fits_io->at(map_index).hdus.back()->addKey("WVARMED", mb->noise_weight_median_ratio(i),
                                                       "Median formal weight times jackknife variance");
        }
        double median_err = 0.0;
        if (redu_type != "beammap" && std::isfinite(mb->median_err(i)) &&
            mb->median_err(i) > std::numeric_limits<double>::epsilon()) {
            median_err = pow(mb->median_err(i), 0.5);
        }
        else if (redu_type != "beammap" && std::isfinite(mb->median_err(i)) &&
                 mb->median_err(i) < 0.0) {
            logger->warn("negative median_err for map {} in {}; using 0", map_name,
                         fits_io->at(map_index).filepath);
        }
        fits_io->at(map_index).hdus.back()->addKey("MEDERR", median_err, "Median Error ("+mb->sig_unit+")");

        if (i < static_cast<Eigen::Index>(mb->weight_formal.size()) &&
            mb->weight_formal[i].rows() == mb->n_rows &&
            mb->weight_formal[i].cols() == mb->n_cols) {
            fits_io->at(map_index).add_hdu("weight_formal_" + map_name + rtcproc.polarization.stokes_params[stokes_index],
                                           mb->weight_formal[i]);
            fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
            fits_io->at(map_index).hdus.back()->addKey("UNIT", weight_unit, "Unit of map");
            fits_io->at(map_index).hdus.back()->addKey("BUNIT", weight_unit, "Physical unit of image values");
            fits_io->at(map_index).hdus.back()->addKey("TYPE", "formal", "Weight calibration type");
            fits_io->at(map_index).hdus.back()->addKey(
                "DESCRIP",
                "Formal mapmaker inverse variance before empirical calibration",
                "Image product description");
        }

        if (i < static_cast<Eigen::Index>(mb->noise_variance.size()) &&
            mb->noise_variance[i].rows() == mb->n_rows &&
            mb->noise_variance[i].cols() == mb->n_cols) {
            fits_io->at(map_index).add_hdu("noise_variance_" + map_name + rtcproc.polarization.stokes_params[stokes_index],
                                           mb->noise_variance[i]);
            fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
            const std::string variance_unit = "("+mb->sig_unit+")^2";
            fits_io->at(map_index).hdus.back()->addKey("UNIT", variance_unit, "Unit of map");
            fits_io->at(map_index).hdus.back()->addKey("BUNIT", variance_unit, "Physical unit of image values");
            fits_io->at(map_index).hdus.back()->addKey(
                "DESCRIP",
                "Per-pixel variance estimated from jackknife noise maps",
                "Image product description");
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
            if (!std::isfinite(fwhm)) {
                logger->warn("non-finite kernel FWHM for map {} in {}; using -99", map_name,
                             fits_io->at(map_index).filepath);
                fwhm = -99.0;
            }
            fits_io->at(map_index).hdus.back()->addKey("FWHM",fwhm,"Kernel fwhm (arcsec)");
            fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
            fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");
            fits_io->at(map_index).hdus.back()->addKey("BUNIT", mb->sig_unit, "Physical unit of image values");
            fits_io->at(map_index).hdus.back()->addKey("DESCRIP", "Mapmaking or filtering kernel image", "Image product description");
        }

        // coverage map
        if (!mb->coverage.empty()) {
            fits_io->at(map_index).add_hdu("coverage_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->coverage[i]);
            fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
            fits_io->at(map_index).hdus.back()->addKey("UNIT", "sec", "Unit of map");
            fits_io->at(map_index).hdus.back()->addKey("BUNIT", "sec", "Physical unit of image values");
            fits_io->at(map_index).hdus.back()->addKey("DESCRIP", "Effective integration time coverage map", "Image product description");
        }

        /* coverage bool and signal-to-noise maps */
        if (!mb->coverage.empty()) {
            // need these to use eigen select
            Eigen::MatrixXd ones, zeros;
            ones.setOnes(mb->weight[i].rows(), mb->weight[i].cols());
            zeros.setZero(mb->weight[i].rows(), mb->weight[i].cols());

            // get weight threshold for current map
            auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = mb->calc_cov_region(i);
            if (!std::isfinite(weight_threshold)) {
                logger->warn("non-finite weight threshold for map {} in {}; using 0", map_name,
                             fits_io->at(map_index).filepath);
                weight_threshold = 0.0;
            }
            // if weight is less than threshold, set to zero, otherwise set to one
            Eigen::MatrixXd coverage_bool = (mb->weight[i].array() < weight_threshold).select(zeros,ones);

            // coverage bool map
            fits_io->at(map_index).add_hdu("coverage_bool_" + map_name + rtcproc.polarization.stokes_params[stokes_index], coverage_bool);
            fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
            fits_io->at(map_index).hdus.back()->addKey("UNIT", "N/A", "Unit of map");
            fits_io->at(map_index).hdus.back()->addKey("BUNIT", "N/A", "Physical unit of image values");
            fits_io->at(map_index).hdus.back()->addKey("DESCRIP", "Boolean valid-coverage support mask", "Image product description");
            fits_io->at(map_index).hdus.back()->addKey("WTTHRESH", weight_threshold, "Weight threshold");

            // legacy signal-to-noise map name retained for compatibility; this is pixel S/N.
            Eigen::MatrixXd sig2noise;
            if (i < static_cast<Eigen::Index>(mb->sig2noise_pixel.size()) &&
                mb->sig2noise_pixel[i].rows() == mb->n_rows &&
                mb->sig2noise_pixel[i].cols() == mb->n_cols) {
                sig2noise = mb->sig2noise_pixel[i];
            }
            else {
                sig2noise = mb->signal[i].array()*sqrt(mb->weight[i].array());
            }
            fits_io->at(map_index).add_hdu("sig2noise_" + map_name + rtcproc.polarization.stokes_params[stokes_index], sig2noise);
            fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
            fits_io->at(map_index).hdus.back()->addKey("UNIT", "N/A", "Unit of map");
            fits_io->at(map_index).hdus.back()->addKey("BUNIT", "N/A", "Physical unit of image values");
            fits_io->at(map_index).hdus.back()->addKey("TYPE", "pixel", "S/N estimator type");
            fits_io->at(map_index).hdus.back()->addKey(
                "DESCRIP",
                "Legacy pixel S/N: signal times sqrt(weight)",
                "Image product description");

            fits_io->at(map_index).add_hdu("sig2noise_pixel_" + map_name + rtcproc.polarization.stokes_params[stokes_index],
                                           sig2noise);
            fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
            fits_io->at(map_index).hdus.back()->addKey("UNIT", "N/A", "Unit of map");
            fits_io->at(map_index).hdus.back()->addKey("BUNIT", "N/A", "Physical unit of image values");
            fits_io->at(map_index).hdus.back()->addKey("TYPE", "pixel", "S/N estimator type");
            fits_io->at(map_index).hdus.back()->addKey(
                "DESCRIP",
                "Pixel S/N map: signal times sqrt(empirical weight)",
                "Image product description");

            const bool is_filtered_output =
                (fits_io == &filtered_fits_io_vec) ||
                (fits_io == &filtered_coadd_fits_io_vec);
            if (is_filtered_output &&
                i < static_cast<Eigen::Index>(mb->point_source_uncertainty.size()) &&
                mb->point_source_uncertainty[i].rows() == mb->n_rows &&
                mb->point_source_uncertainty[i].cols() == mb->n_cols) {
                fits_io->at(map_index).add_hdu("point_source_flux_" + map_name + rtcproc.polarization.stokes_params[stokes_index],
                                               mb->signal[i]);
                fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
                fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");
                fits_io->at(map_index).hdus.back()->addKey("BUNIT", mb->sig_unit, "Physical unit of image values");
                fits_io->at(map_index).hdus.back()->addKey(
                    "DESCRIP",
                    "Point-source flux estimate after filter response normalization",
                    "Image product description");
                fits_io->at(map_index).hdus.back()->addKey("RESPNORM", 1.0, "Point-source response normalization applied");

                fits_io->at(map_index).add_hdu("point_source_uncertainty_" + map_name + rtcproc.polarization.stokes_params[stokes_index],
                                               mb->point_source_uncertainty[i]);
                fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
                fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");
                fits_io->at(map_index).hdus.back()->addKey("BUNIT", mb->sig_unit, "Physical unit of image values");
                fits_io->at(map_index).hdus.back()->addKey(
                    "DESCRIP",
                    "Point-source 1-sigma uncertainty from jackknife maps",
                    "Image product description");

                fits_io->at(map_index).add_hdu("sig2noise_point_source_" + map_name + rtcproc.polarization.stokes_params[stokes_index],
                                               mb->sig2noise_point_source[i]);
                fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
                fits_io->at(map_index).hdus.back()->addKey("UNIT", "N/A", "Unit of map");
                fits_io->at(map_index).hdus.back()->addKey("BUNIT", "N/A", "Physical unit of image values");
                fits_io->at(map_index).hdus.back()->addKey("TYPE", "point_source", "S/N estimator type");
                fits_io->at(map_index).hdus.back()->addKey(
                    "DESCRIP",
                    "Point-source S/N from flux divided by jackknife uncertainty",
                    "Image product description");
            }
        }

        // write noise maps
        if (!mb->noise.empty() && !noise_fits_io->empty()) {
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
            double median_rms = 0.0;
            if (i < mb->median_rms.size() && std::isfinite(mb->median_rms(i))) {
                median_rms = mb->median_rms(i);
            }
            else if (i < mb->median_rms.size()) {
                logger->warn("non-finite median_rms for map {} in {}; using 0", map_name,
                             noise_fits_io->at(map_index).filepath);
            }
            for (Eigen::Index n=0; n<mb->n_noise; ++n) {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(mb->noise[i].data() + n * mb->n_rows * mb->n_cols,
                                                                                               mb->n_rows, mb->n_cols);

                noise_fits_io->at(map_index).add_hdu("signal_" + map_name + std::to_string(n) + "_" + rtcproc.polarization.stokes_params[stokes_index],
                                                     noise_matrix);
                noise_fits_io->at(map_index).add_wcs(noise_fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
                noise_fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");
                noise_fits_io->at(map_index).hdus.back()->addKey("MEDRMS", median_rms, "Median RMS of noise maps");
            }
        }
    } catch (const CCfits::FitsError &e) {
        throw std::runtime_error(
            fmt::format("failed to write map '{}' (map_i={} file={} noise_file={}): {}",
                        map_name,
                        static_cast<long long>(i),
                        fits_io->at(map_index).filepath,
                        (!mb->noise.empty() && map_index < static_cast<Eigen::Index>(noise_fits_io->size()))
                            ? noise_fits_io->at(map_index).filepath
                            : std::string("N/A"),
                        e.message()));
    } catch (const std::exception &e) {
        throw std::runtime_error(
            fmt::format("failed to write map '{}' (map_i={} file={} noise_file={}): {}",
                        map_name,
                        static_cast<long long>(i),
                        fits_io->at(map_index).filepath,
                        (!mb->noise.empty() && map_index < static_cast<Eigen::Index>(noise_fits_io->size()))
                            ? noise_fits_io->at(map_index).filepath
                            : std::string("N/A"),
                        e.what()));
    }
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_psd(map_buffer_t &mb, std::string dir_name) {
    // get filename
    std::string filename = setup_filenames<map_t,engine_utils::toltecIO::toltec,engine_utils::toltecIO::psd>(dir_name);

    write_netcdf_atomic(filename + ".nc", [&](netCDF::NcFile &fo) {

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
    });
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_hist(map_buffer_t &mb, std::string dir_name) {
    std::string filename = setup_filenames<map_t,engine_utils::toltecIO::toltec,engine_utils::toltecIO::hist>(dir_name);

    write_netcdf_atomic(filename + ".nc", [&](netCDF::NcFile &fo) {
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
    });
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_mapdiag(map_buffer_t &mb, std::string dir_name) {
    std::string filename = setup_filenames<map_t, engine_utils::toltecIO::toltec, engine_utils::toltecIO::mapdiag>(dir_name);
    const std::size_t n_maps_local = static_cast<std::size_t>(n_maps);
    const std::size_t n_obsnums = std::max<std::size_t>(1, mb->obsnums.size());
    const bool is_coadd = (map_t == mapmaking::RawCoadd || map_t == mapmaking::FilteredCoadd);
    const double fill_double = std::numeric_limits<double>::quiet_NaN();
    const int fill_int = -2147483647;

    std::vector<std::string> array_names(n_maps_local);
    std::vector<std::string> stokes_names(n_maps_local);
    std::vector<std::string> map_names(n_maps_local);
    std::vector<double> median_err(n_maps_local, fill_double);
    std::vector<double> median_rms(n_maps_local, fill_double);
    std::vector<double> weight_thresholds(n_maps_local, fill_double);
    std::vector<double> weight_sum(n_maps_local, fill_double);
    std::vector<double> core_weight_sum(n_maps_local, fill_double);
    std::vector<double> coverage_sum(n_maps_local, fill_double);
    std::vector<double> coverage_max(n_maps_local, fill_double);
    std::vector<double> coverage_median_core(n_maps_local, fill_double);
    std::vector<double> empirical_to_formal_noise_ratio(n_maps_local, fill_double);
    std::vector<double> noise_weight_median_ratio(n_maps_local, fill_double);
    std::vector<double> noise_weight_scale(n_maps_local, fill_double);
    std::vector<double> noise_products_s2n_sigma(n_maps_local, fill_double);
    std::vector<double> noise_products_valid_pixels(n_maps_local, fill_double);
    std::vector<double> peak_signal(n_maps_local, fill_double);
    std::vector<double> peak_abs_sig2noise(n_maps_local, fill_double);
    std::vector<double> core_peak_abs_sig2noise(n_maps_local, fill_double);
    std::vector<double> noise_rms_p16(n_maps_local, fill_double);
    std::vector<double> noise_rms_p84(n_maps_local, fill_double);
    std::vector<double> core_tail_frac_abs3(n_maps_local, fill_double);
    std::vector<double> core_tail_frac_pos3(n_maps_local, fill_double);
    std::vector<double> core_tail_frac_neg3(n_maps_local, fill_double);
    std::vector<double> core_tail_excess_abs3(n_maps_local, fill_double);
    std::vector<double> core_tail_excess_pos3(n_maps_local, fill_double);
    std::vector<double> core_tail_excess_neg3(n_maps_local, fill_double);
    std::vector<double> core_sig2noise_skew(n_maps_local, fill_double);
    std::vector<double> noise_tail_frac_abs3(n_maps_local, fill_double);
    std::vector<double> noise_tail_frac_pos3(n_maps_local, fill_double);
    std::vector<double> noise_tail_frac_neg3(n_maps_local, fill_double);
    std::vector<double> noise_tail_excess_abs3(n_maps_local, fill_double);
    std::vector<double> noise_tail_excess_pos3(n_maps_local, fill_double);
    std::vector<double> noise_tail_excess_neg3(n_maps_local, fill_double);
    std::vector<double> noise_sig2noise_skew(n_maps_local, fill_double);
    std::vector<double> edge_guard_weight_thresholds(n_maps_local, fill_double);
    std::vector<double> edge_guard_hits_thresholds(n_maps_local, fill_double);
    std::vector<double> edge_guard_background_levels(n_maps_local, fill_double);
    std::vector<double> edge_guard_science_frac(n_maps_local, fill_double);
    std::vector<double> edge_guard_support_frac(n_maps_local, fill_double);
    std::vector<double> edge_guard_guardband_rms_pre(n_maps_local, fill_double);
    std::vector<double> edge_guard_guardband_rms_post(n_maps_local, fill_double);
    std::vector<double> edge_guard_exterior_rms_pre(n_maps_local, fill_double);
    std::vector<double> edge_guard_exterior_rms_post(n_maps_local, fill_double);
    std::vector<double> edge_guard_exterior_max_abs_pre(n_maps_local, fill_double);
    std::vector<double> edge_guard_exterior_max_abs_post(n_maps_local, fill_double);
    std::vector<int> n_valid_pixels(n_maps_local, 0);
    std::vector<int> n_core_pixels(n_maps_local, 0);
    std::vector<int> peak_row(n_maps_local, fill_int);
    std::vector<int> peak_col(n_maps_local, fill_int);
    std::vector<int> edge_guard_applied(n_maps_local, 0);
    std::vector<int> edge_guard_support_radius_pix(n_maps_local, 0);
    std::vector<int> edge_guard_science_npix(n_maps_local, 0);
    std::vector<int> edge_guard_support_npix(n_maps_local, 0);
    std::vector<int> edge_guard_guardband_npix(n_maps_local, 0);

    std::vector<double> obs_weight_sum(n_maps_local * n_obsnums, fill_double);
    std::vector<double> obs_weight_frac(n_maps_local * n_obsnums, fill_double);
    std::vector<double> obs_core_weight_sum(n_maps_local * n_obsnums, fill_double);
    std::vector<double> obs_core_weight_frac(n_maps_local * n_obsnums, fill_double);
    std::vector<int> obs_valid_pixels(n_maps_local * n_obsnums, fill_int);
    std::vector<int> obs_core_pixels(n_maps_local * n_obsnums, fill_int);

    auto put_string_1d = [](netCDF::NcFile &fo, const std::string &name, netCDF::NcDim dim,
                            const std::vector<std::string> &values, const std::string &comment = "") {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncString, dim);
        if (!comment.empty()) {
            v.putAtt("comment", comment);
        }
        for (std::size_t i = 0; i < values.size(); ++i) {
            const std::vector<std::size_t> idx = {i};
            std::string value = values[i];
            v.putVar(idx, value);
        }
    };

    auto accumulate_obs_weight = [&](Eigen::Index map_i,
                                     const Eigen::ArrayXXd &core_mask,
                                     const Eigen::MatrixXd &obs_weight,
                                     std::size_t obs_index) {
        const Eigen::Index block_row = (mb->n_rows - obs_weight.rows()) / 2;
        const Eigen::Index block_col = (mb->n_cols - obs_weight.cols()) / 2;
        Eigen::Index row0 = std::max<Eigen::Index>(0, block_row);
        Eigen::Index col0 = std::max<Eigen::Index>(0, block_col);
        Eigen::Index src_row0 = std::max<Eigen::Index>(0, -block_row);
        Eigen::Index src_col0 = std::max<Eigen::Index>(0, -block_col);
        Eigen::Index rows = std::min<Eigen::Index>(mb->n_rows - row0, obs_weight.rows() - src_row0);
        Eigen::Index cols = std::min<Eigen::Index>(mb->n_cols - col0, obs_weight.cols() - src_col0);
        const std::size_t flat = static_cast<std::size_t>(map_i) * n_obsnums + obs_index;
        if (rows <= 0 || cols <= 0) {
            obs_weight_sum[flat] = 0.0;
            obs_core_weight_sum[flat] = 0.0;
            obs_valid_pixels[flat] = 0;
            obs_core_pixels[flat] = 0;
            return;
        }

        const auto block = obs_weight.block(src_row0, src_col0, rows, cols);
        const auto valid = (block.array() > 0.0).template cast<double>();
        const auto core_block = core_mask.block(row0, col0, rows, cols);
        obs_weight_sum[flat] = (block.array() * valid).sum();
        obs_core_weight_sum[flat] = (block.array() * valid * core_block).sum();
        obs_valid_pixels[flat] = static_cast<int>(valid.sum());
        obs_core_pixels[flat] = static_cast<int>((valid * core_block).sum());
    };

    struct tail_stats_t {
        double frac_abs3 = std::numeric_limits<double>::quiet_NaN();
        double frac_pos3 = std::numeric_limits<double>::quiet_NaN();
        double frac_neg3 = std::numeric_limits<double>::quiet_NaN();
        double excess_abs3 = std::numeric_limits<double>::quiet_NaN();
        double excess_pos3 = std::numeric_limits<double>::quiet_NaN();
        double excess_neg3 = std::numeric_limits<double>::quiet_NaN();
        double skew = std::numeric_limits<double>::quiet_NaN();
    };

    auto vector_median = [&](const std::vector<double> &values) -> double {
        if (values.empty()) {
            return fill_double;
        }
        Eigen::Map<const Eigen::VectorXd> mapped(values.data(), static_cast<Eigen::Index>(values.size()));
        return tula::alg::median(mapped);
    };

    auto vector_quantile = [&](std::vector<double> values, double q) -> double {
        if (values.empty()) {
            return fill_double;
        }
        q = std::clamp(q, 0.0, 1.0);
        std::sort(values.begin(), values.end());
        const double pos = q * static_cast<double>(values.size() - 1);
        const std::size_t i0 = static_cast<std::size_t>(std::floor(pos));
        const std::size_t i1 = static_cast<std::size_t>(std::ceil(pos));
        const double frac = pos - static_cast<double>(i0);
        return values[i0] * (1.0 - frac) + values[i1] * frac;
    };

    auto collect_masked_values = [&](const Eigen::MatrixXd &matrix, const Eigen::ArrayXXd &mask) {
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(mask.sum()));
        for (Eigen::Index r = 0; r < matrix.rows(); ++r) {
            for (Eigen::Index c = 0; c < matrix.cols(); ++c) {
                const double value = matrix(r, c);
                if (mask(r, c) > 0.0 && std::isfinite(value)) {
                    values.push_back(value);
                }
            }
        }
        return values;
    };

    auto calc_tail_stats = [&](const std::vector<double> &values) {
        tail_stats_t stats;
        if (values.size() < 8) {
            return stats;
        }
        const double center = vector_median(values);
        if (!std::isfinite(center)) {
            return stats;
        }
        std::vector<double> abs_dev;
        abs_dev.reserve(values.size());
        for (const auto &value : values) {
            abs_dev.push_back(std::abs(value - center));
        }
        const double mad = vector_median(abs_dev);
        const double robust_sigma = 1.4826 * mad;
        if (!std::isfinite(robust_sigma) || robust_sigma <= std::numeric_limits<double>::epsilon()) {
            return stats;
        }

        std::size_t n_abs = 0;
        std::size_t n_pos = 0;
        std::size_t n_neg = 0;
        double skew_sum = 0.0;
        for (const auto &value : values) {
            const double z = (value - center) / robust_sigma;
            if (!std::isfinite(z)) {
                continue;
            }
            if (std::abs(z) >= 3.0) {
                ++n_abs;
            }
            if (z >= 3.0) {
                ++n_pos;
            }
            if (z <= -3.0) {
                ++n_neg;
            }
            skew_sum += z * z * z;
        }

        const double n = static_cast<double>(values.size());
        stats.frac_abs3 = static_cast<double>(n_abs) / n;
        stats.frac_pos3 = static_cast<double>(n_pos) / n;
        stats.frac_neg3 = static_cast<double>(n_neg) / n;
        constexpr double gauss_pos3 = 1.3498980316300959e-3;
        constexpr double gauss_abs3 = 2.6997960632601918e-3;
        stats.excess_abs3 = stats.frac_abs3 / gauss_abs3;
        stats.excess_pos3 = stats.frac_pos3 / gauss_pos3;
        stats.excess_neg3 = stats.frac_neg3 / gauss_pos3;
        stats.skew = skew_sum / n;
        return stats;
    };

    for (Eigen::Index i = 0; i < n_maps; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const auto map_index = arrays_to_maps(i);
        const auto stokes_index = maps_to_stokes(i);
        array_names[idx] = toltec_io.array_name_map[calib.arrays[map_index]];
        stokes_names[idx] = rtcproc.polarization.stokes_params[stokes_index];
        map_names[idx] = get_map_name(i);

        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = mb->calc_cov_region(i);
        if (!std::isfinite(weight_threshold) || weight_threshold < 0.0) {
            weight_threshold = 0.0;
        }
        weight_thresholds[idx] = weight_threshold;
        if (idx < mb->edge_guard_applied.size()) {
            edge_guard_applied[idx] = mb->edge_guard_applied[idx];
            edge_guard_support_radius_pix[idx] = mb->edge_guard_support_radius_pix[idx];
            edge_guard_science_npix[idx] = mb->edge_guard_science_npix[idx];
            edge_guard_support_npix[idx] = mb->edge_guard_support_npix[idx];
            edge_guard_guardband_npix[idx] = mb->edge_guard_guardband_npix[idx];
            edge_guard_weight_thresholds[idx] = mb->edge_guard_weight_threshold[idx];
            edge_guard_hits_thresholds[idx] = mb->edge_guard_hits_threshold[idx];
            edge_guard_background_levels[idx] = mb->edge_guard_background_level[idx];
            edge_guard_science_frac[idx] = mb->edge_guard_science_frac[idx];
            edge_guard_support_frac[idx] = mb->edge_guard_support_frac[idx];
            edge_guard_guardband_rms_pre[idx] = mb->edge_guard_guardband_rms_pre[idx];
            edge_guard_guardband_rms_post[idx] = mb->edge_guard_guardband_rms_post[idx];
            edge_guard_exterior_rms_pre[idx] = mb->edge_guard_exterior_rms_pre[idx];
            edge_guard_exterior_rms_post[idx] = mb->edge_guard_exterior_rms_post[idx];
            edge_guard_exterior_max_abs_pre[idx] = mb->edge_guard_exterior_max_abs_pre[idx];
            edge_guard_exterior_max_abs_post[idx] = mb->edge_guard_exterior_max_abs_post[idx];
        }

        const auto weight_arr = mb->weight[i].array();
        const auto valid_mask = (weight_arr > 0.0).template cast<double>();
        const auto core_mask = ((weight_arr >= weight_threshold) && (weight_arr > 0.0)).template cast<double>();
        n_valid_pixels[idx] = static_cast<int>(valid_mask.sum());
        n_core_pixels[idx] = static_cast<int>(core_mask.sum());
        weight_sum[idx] = (weight_arr * valid_mask).sum();
        core_weight_sum[idx] = (weight_arr * core_mask).sum();

        if (i < mb->median_err.size() && std::isfinite(mb->median_err(i)) &&
            mb->median_err(i) > std::numeric_limits<double>::epsilon()) {
            median_err[idx] = std::sqrt(mb->median_err(i));
        }
        if (i < mb->median_rms.size() && std::isfinite(mb->median_rms(i))) {
            median_rms[idx] = mb->median_rms(i);
        }
        if (std::isfinite(median_err[idx]) && std::isfinite(median_rms[idx]) &&
            median_err[idx] > std::numeric_limits<double>::epsilon()) {
            empirical_to_formal_noise_ratio[idx] = median_rms[idx] / median_err[idx];
        }
        if (i < mb->noise_weight_median_ratio.size()) {
            noise_weight_median_ratio[idx] = mb->noise_weight_median_ratio(i);
        }
        if (i < mb->noise_weight_scale.size()) {
            noise_weight_scale[idx] = mb->noise_weight_scale(i);
        }
        if (i < mb->noise_s2n_sigma.size()) {
            noise_products_s2n_sigma[idx] = mb->noise_s2n_sigma(i);
        }
        if (i < mb->noise_valid_pixels.size()) {
            noise_products_valid_pixels[idx] = mb->noise_valid_pixels(i);
        }

        if (!mb->coverage.empty() && i < static_cast<Eigen::Index>(mb->coverage.size())) {
            coverage_sum[idx] = mb->coverage[i].sum();
            coverage_max[idx] = mb->coverage[i].maxCoeff();
            std::vector<double> core_cov;
            core_cov.reserve(static_cast<std::size_t>(n_core_pixels[idx]));
            for (Eigen::Index r = 0; r < mb->coverage[i].rows(); ++r) {
                for (Eigen::Index c = 0; c < mb->coverage[i].cols(); ++c) {
                    if (core_mask(r, c) > 0.0 && std::isfinite(mb->coverage[i](r, c))) {
                        core_cov.push_back(mb->coverage[i](r, c));
                    }
                }
            }
            if (!core_cov.empty()) {
                coverage_median_core[idx] = tula::alg::median(Eigen::Map<Eigen::VectorXd>(core_cov.data(), core_cov.size()));
            }
        }

        peak_signal[idx] = mb->signal[i].size() > 0 ? mb->signal[i].maxCoeff() : fill_double;
        if (mb->signal[i].size() > 0 && mb->weight[i].size() > 0) {
            Eigen::MatrixXd sig2noise = mb->signal[i].array() * mb->weight[i].array().max(0.0).sqrt();
            Eigen::Index r_peak = 0;
            Eigen::Index c_peak = 0;
            peak_abs_sig2noise[idx] = sig2noise.cwiseAbs().maxCoeff(&r_peak, &c_peak);
            peak_row[idx] = static_cast<int>(r_peak);
            peak_col[idx] = static_cast<int>(c_peak);
            if (n_core_pixels[idx] > 0) {
                const Eigen::MatrixXd core_sig2noise = (sig2noise.cwiseAbs().array() * core_mask).matrix();
                core_peak_abs_sig2noise[idx] = core_sig2noise.maxCoeff();
            }
            const auto core_values = collect_masked_values(sig2noise, core_mask);
            const auto signal_tail = calc_tail_stats(core_values);
            core_tail_frac_abs3[idx] = signal_tail.frac_abs3;
            core_tail_frac_pos3[idx] = signal_tail.frac_pos3;
            core_tail_frac_neg3[idx] = signal_tail.frac_neg3;
            core_tail_excess_abs3[idx] = signal_tail.excess_abs3;
            core_tail_excess_pos3[idx] = signal_tail.excess_pos3;
            core_tail_excess_neg3[idx] = signal_tail.excess_neg3;
            core_sig2noise_skew[idx] = signal_tail.skew;

            if (!mb->noise.empty() && i < static_cast<Eigen::Index>(mb->noise.size()) && mb->n_noise > 0) {
                std::vector<double> noise_rms_values;
                noise_rms_values.reserve(static_cast<std::size_t>(mb->n_noise));
                std::vector<double> tail_abs_values;
                std::vector<double> tail_pos_values;
                std::vector<double> tail_neg_values;
                std::vector<double> excess_abs_values;
                std::vector<double> excess_pos_values;
                std::vector<double> excess_neg_values;
                std::vector<double> skew_values;
                tail_abs_values.reserve(static_cast<std::size_t>(mb->n_noise));
                tail_pos_values.reserve(static_cast<std::size_t>(mb->n_noise));
                tail_neg_values.reserve(static_cast<std::size_t>(mb->n_noise));
                excess_abs_values.reserve(static_cast<std::size_t>(mb->n_noise));
                excess_pos_values.reserve(static_cast<std::size_t>(mb->n_noise));
                excess_neg_values.reserve(static_cast<std::size_t>(mb->n_noise));
                skew_values.reserve(static_cast<std::size_t>(mb->n_noise));

                const auto valid_core = (core_mask > 0.0);
                const double valid_core_count = valid_core.count();
                for (Eigen::Index n = 0; n < mb->n_noise; ++n) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(
                        mb->noise[i].data() + n * mb->n_rows * mb->n_cols, mb->n_rows, mb->n_cols);
                    if (valid_core_count > 0.0) {
                        const double rms_sq = (valid_core.select(noise_matrix.array().square(), 0.0)).sum();
                        noise_rms_values.push_back(std::sqrt(rms_sq / valid_core_count));
                    }
                    const auto noise_values = collect_masked_values(noise_matrix, core_mask);
                    const auto noise_tail = calc_tail_stats(noise_values);
                    if (std::isfinite(noise_tail.frac_abs3)) {
                        tail_abs_values.push_back(noise_tail.frac_abs3);
                    }
                    if (std::isfinite(noise_tail.frac_pos3)) {
                        tail_pos_values.push_back(noise_tail.frac_pos3);
                    }
                    if (std::isfinite(noise_tail.frac_neg3)) {
                        tail_neg_values.push_back(noise_tail.frac_neg3);
                    }
                    if (std::isfinite(noise_tail.excess_abs3)) {
                        excess_abs_values.push_back(noise_tail.excess_abs3);
                    }
                    if (std::isfinite(noise_tail.excess_pos3)) {
                        excess_pos_values.push_back(noise_tail.excess_pos3);
                    }
                    if (std::isfinite(noise_tail.excess_neg3)) {
                        excess_neg_values.push_back(noise_tail.excess_neg3);
                    }
                    if (std::isfinite(noise_tail.skew)) {
                        skew_values.push_back(noise_tail.skew);
                    }
                }
                noise_rms_p16[idx] = vector_quantile(noise_rms_values, 0.16);
                noise_rms_p84[idx] = vector_quantile(noise_rms_values, 0.84);
                noise_tail_frac_abs3[idx] = vector_median(tail_abs_values);
                noise_tail_frac_pos3[idx] = vector_median(tail_pos_values);
                noise_tail_frac_neg3[idx] = vector_median(tail_neg_values);
                noise_tail_excess_abs3[idx] = vector_median(excess_abs_values);
                noise_tail_excess_pos3[idx] = vector_median(excess_pos_values);
                noise_tail_excess_neg3[idx] = vector_median(excess_neg_values);
                noise_sig2noise_skew[idx] = vector_median(skew_values);
            }
        }

        if (!is_coadd) {
            obs_weight_sum[idx * n_obsnums] = weight_sum[idx];
            obs_core_weight_sum[idx * n_obsnums] = core_weight_sum[idx];
            obs_valid_pixels[idx * n_obsnums] = n_valid_pixels[idx];
            obs_core_pixels[idx * n_obsnums] = n_core_pixels[idx];
        }
        else {
            for (std::size_t obs_idx = 0; obs_idx < mb->obsnums.size(); ++obs_idx) {
                const auto &obsnum_i = mb->obsnums[obs_idx];
                const auto obs_dir = redu_dir_name + "/" + obsnum_i + "/raw/";
                const auto obs_weight_path = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                                       engine_utils::toltecIO::map,
                                                                       engine_utils::toltecIO::raw>(
                    obs_dir, redu_type, array_names[idx], obsnum_i, telescope.sim_obs) + ".fits";
                try {
                    fitsIO<file_type_enum::read_fits, CCfits::ExtHDU*> obs_fits(obs_weight_path);
                    const auto weight_hdu_name = "weight_" + map_names[idx] + stokes_names[idx];
                    auto obs_weight = obs_fits.get_hdu(weight_hdu_name);
                    accumulate_obs_weight(i, core_mask, obs_weight, obs_idx);
                } catch (const std::exception &e) {
                    logger->warn("failed to derive mapdiag contribution from {} [{}]: {}", obs_weight_path,
                                 "weight_" + map_names[idx] + stokes_names[idx], e.what());
                    const std::size_t flat = idx * n_obsnums + obs_idx;
                    obs_weight_sum[flat] = 0.0;
                    obs_core_weight_sum[flat] = 0.0;
                    obs_valid_pixels[flat] = 0;
                    obs_core_pixels[flat] = 0;
                }
            }
        }

        double total_weight = 0.0;
        double total_core_weight = 0.0;
        for (std::size_t obs_idx = 0; obs_idx < n_obsnums; ++obs_idx) {
            total_weight += obs_weight_sum[idx * n_obsnums + obs_idx];
            total_core_weight += obs_core_weight_sum[idx * n_obsnums + obs_idx];
        }
        for (std::size_t obs_idx = 0; obs_idx < n_obsnums; ++obs_idx) {
            const std::size_t flat = idx * n_obsnums + obs_idx;
            obs_weight_frac[flat] = (total_weight > 0.0) ? obs_weight_sum[flat] / total_weight : fill_double;
            obs_core_weight_frac[flat] = (total_core_weight > 0.0) ? obs_core_weight_sum[flat] / total_core_weight : fill_double;
        }
    }

    write_netcdf_atomic(filename + ".nc", [&](netCDF::NcFile &fo) {
    netCDF::NcVar obsnum_v = fo.addVar("obsnum", netCDF::ncInt);
    obsnum_v.putAtt("units", "N/A");
    int obsnum_int = is_coadd ? -1 : std::stoi(obsnum);
    obsnum_v.putVar(&obsnum_int);

    netCDF::NcDim n_maps_dim = fo.addDim("n_maps", n_maps_local);
    netCDF::NcDim n_obsnums_dim = fo.addDim("n_obsnums", n_obsnums);
    std::vector<netCDF::NcDim> map_obs_dims = {n_maps_dim, n_obsnums_dim};

    std::string stage_name = "raw_obs";
    if constexpr (map_t == mapmaking::FilteredObs) {
        stage_name = "filtered_obs";
    }
    else if constexpr (map_t == mapmaking::RawCoadd) {
        stage_name = "raw_coadd";
    }
    else if constexpr (map_t == mapmaking::FilteredCoadd) {
        stage_name = "filtered_coadd";
    }
    add_netcdf_var<std::string>(fo, "MAP_STAGE", stage_name);
    add_netcdf_var<std::string>(fo, "MAP_BUFFER", mb->name);
    add_netcdf_var<std::string>(fo, "MAP_REGIME", map_regime);
    add_netcdf_var<std::string>(fo, "SOURCE", telescope.source_name);
    add_netcdf_var<std::string>(fo, "PROJID", telescope.project_id);
    add_netcdf_var<std::string>(fo, "OBSGOAL", telescope.obs_goal);
    add_netcdf_var(fo, "MAP_PIXEL_SIZE_RAD", mb->pixel_size_rad);
    add_netcdf_var(fo, "MAP_COVERAGE_CUT", mb->cov_cut);
    add_netcdf_var<std::string>(fo, "MAP_SIG_UNIT", mb->sig_unit);
    add_netcdf_var(fo, "MAP_EDGE_GUARD_ENABLED", wiener_filter.edge_guard_enabled);
    add_netcdf_var<std::string>(fo, "MAP_EDGE_GUARD_WEIGHT_THRESHOLD_MODE", wiener_filter.edge_weight_threshold_mode);
    add_netcdf_var<std::string>(fo, "MAP_EDGE_GUARD_HITS_THRESHOLD_MODE", wiener_filter.edge_hits_threshold_mode);
    add_netcdf_var<std::string>(fo, "MAP_EDGE_GUARD_FILL_MODE", wiener_filter.edge_fill_mode);
    add_netcdf_var<std::string>(fo, "MAP_EDGE_GUARD_TAPER_MODE", wiener_filter.edge_taper_mode);
    add_netcdf_var(fo, "MAP_EDGE_GUARD_HITS_CORE_FRACTION", wiener_filter.edge_hits_core_fraction);
    add_netcdf_var(fo, "MAP_EDGE_GUARD_RADIUS_FWHM", wiener_filter.edge_guard_radius_fwhm);
    add_netcdf_var(fo, "MAP_EDGE_GUARD_TAPER_MIN_FRACTION", wiener_filter.edge_taper_min_fraction);

    put_string_1d(fo, "map_array_name", n_maps_dim, array_names, "array label for each map row");
    put_string_1d(fo, "map_stokes", n_maps_dim, stokes_names, "stokes parameter label for each map row");
    put_string_1d(fo, "map_name", n_maps_dim, map_names, "grouping-derived map label prefix for each map row");

    std::vector<std::string> obsnum_strings = mb->obsnums;
    if (obsnum_strings.empty()) {
        obsnum_strings.push_back(obsnum);
    }
    put_string_1d(fo, "coadd_obsnum", n_obsnums_dim, obsnum_strings, "obsnum ordering for map x obsnum contribution tables");

    std::vector<std::string> dateobs_strings = date_obs;
    if (dateobs_strings.empty()) {
        dateobs_strings.push_back("");
    }
    if (dateobs_strings.size() > n_obsnums) {
        dateobs_strings.resize(n_obsnums);
    }
    if (dateobs_strings.size() < n_obsnums) {
        dateobs_strings.resize(n_obsnums, "");
    }
    put_string_1d(fo, "coadd_dateobs", n_obsnums_dim, dateobs_strings, "DATEOBS ordering matching coadd_obsnum");

    auto add_map_double = [&](const std::string &name, const std::string &comment, const std::vector<double> &values) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, n_maps_dim);
        v.putAtt("comment", comment);
        v.putVar(values.data());
    };
    auto add_map_int = [&](const std::string &name, const std::string &comment, const std::vector<int> &values) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, n_maps_dim);
        v.putAtt("comment", comment);
        v.putVar(values.data());
    };
    auto add_map_obs_double = [&](const std::string &name, const std::string &comment, const std::vector<double> &values) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, map_obs_dims);
        v.putAtt("comment", comment);
        v.putVar(values.data());
    };
    auto add_map_obs_int = [&](const std::string &name, const std::string &comment, const std::vector<int> &values) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, map_obs_dims);
        v.putAtt("comment", comment);
        v.putVar(values.data());
    };

    add_map_double("map_median_err", "median error derived from the map weight product", median_err);
    add_map_double("map_median_rms", "median RMS of the map noise realization or background estimator", median_rms);
    add_map_double("map_weight_threshold", "coverage-derived weight threshold used to define the core map support", weight_thresholds);
    add_map_double("map_weight_sum", "sum of positive map weights over all valid pixels", weight_sum);
    add_map_double("map_core_weight_sum", "sum of positive map weights over pixels above map_weight_threshold", core_weight_sum);
    add_map_double("map_coverage_sum", "sum of coverage values over the map; NaN if no coverage map exists", coverage_sum);
    add_map_double("map_coverage_max", "maximum coverage value in the map; NaN if no coverage map exists", coverage_max);
    add_map_double("map_core_coverage_median", "median coverage over the core support; NaN if no coverage map exists", coverage_median_core);
    add_map_double("map_empirical_to_formal_noise_ratio", "ratio of map_median_rms to map_median_err over the core support", empirical_to_formal_noise_ratio);
    add_map_double("map_noise_weight_median_ratio", "median of formal weight times jackknife variance over the valid support", noise_weight_median_ratio);
    add_map_double("map_noise_weight_scale", "empirical scalar applied to formal weights", noise_weight_scale);
    add_map_double("map_noise_products_s2n_sigma", "standard deviation of jackknife noise multiplied by sqrt(formal weight)", noise_products_s2n_sigma);
    add_map_double("map_noise_products_valid_pixels", "number of pixels used for empirical noise-product calibration", noise_products_valid_pixels);
    add_map_double("map_peak_signal", "maximum signal value in the map", peak_signal);
    add_map_double("map_peak_abs_sig2noise", "maximum absolute signal-to-noise value in the map", peak_abs_sig2noise);
    add_map_double("map_core_peak_abs_sig2noise", "maximum absolute signal-to-noise value over pixels with weight >= map_weight_threshold", core_peak_abs_sig2noise);
    add_map_double("map_noise_rms_p16", "16th percentile of core RMS values across noise realizations", noise_rms_p16);
    add_map_double("map_noise_rms_p84", "84th percentile of core RMS values across noise realizations", noise_rms_p84);
    add_map_double("map_core_tail_fraction_abs_gt3", "fraction of core sig2noise pixels with |robust-z| >= 3", core_tail_frac_abs3);
    add_map_double("map_core_tail_fraction_pos_gt3", "fraction of core sig2noise pixels with robust-z >= 3", core_tail_frac_pos3);
    add_map_double("map_core_tail_fraction_neg_lt3", "fraction of core sig2noise pixels with robust-z <= -3", core_tail_frac_neg3);
    add_map_double("map_core_tail_excess_abs_gt3", "ratio of map_core_tail_fraction_abs_gt3 to Gaussian expectation", core_tail_excess_abs3);
    add_map_double("map_core_tail_excess_pos_gt3", "ratio of map_core_tail_fraction_pos_gt3 to Gaussian expectation", core_tail_excess_pos3);
    add_map_double("map_core_tail_excess_neg_lt3", "ratio of map_core_tail_fraction_neg_lt3 to Gaussian expectation", core_tail_excess_neg3);
    add_map_double("map_core_sig2noise_skew", "mean robust-z^3 of core sig2noise pixels", core_sig2noise_skew);
    add_map_double("map_noise_tail_fraction_abs_gt3", "median fraction across noise realizations with |robust-z| >= 3 in the core support", noise_tail_frac_abs3);
    add_map_double("map_noise_tail_fraction_pos_gt3", "median fraction across noise realizations with robust-z >= 3 in the core support", noise_tail_frac_pos3);
    add_map_double("map_noise_tail_fraction_neg_lt3", "median fraction across noise realizations with robust-z <= -3 in the core support", noise_tail_frac_neg3);
    add_map_double("map_noise_tail_excess_abs_gt3", "median ratio across noise realizations of abs tail fraction to Gaussian expectation", noise_tail_excess_abs3);
    add_map_double("map_noise_tail_excess_pos_gt3", "median ratio across noise realizations of positive tail fraction to Gaussian expectation", noise_tail_excess_pos3);
    add_map_double("map_noise_tail_excess_neg_lt3", "median ratio across noise realizations of negative tail fraction to Gaussian expectation", noise_tail_excess_neg3);
    add_map_double("map_noise_sig2noise_skew", "median mean robust-z^3 across noise realizations in the core support", noise_sig2noise_skew);
    add_map_double("map_edge_guard_weight_threshold", "runtime weight threshold used by the filter edge guard; NaN when not applied", edge_guard_weight_thresholds);
    add_map_double("map_edge_guard_hits_threshold", "runtime coverage threshold used by the filter edge guard; NaN when not applied or no coverage map exists", edge_guard_hits_thresholds);
    add_map_double("map_edge_guard_background_level", "background fill level applied outside the edge-guard support mask before filtering", edge_guard_background_levels);
    add_map_double("map_edge_guard_science_fraction", "fraction of map pixels in the edge-guard science mask", edge_guard_science_frac);
    add_map_double("map_edge_guard_support_fraction", "fraction of map pixels in the edge-guard support mask", edge_guard_support_frac);
    add_map_double("map_edge_guard_guardband_rms_pre", "RMS of signal values in the effective edge-guard guard band before applying fill/taper", edge_guard_guardband_rms_pre);
    add_map_double("map_edge_guard_guardband_rms_post", "RMS of signal values in the effective edge-guard guard band after applying fill/taper and before filtering", edge_guard_guardband_rms_post);
    add_map_double("map_edge_guard_exterior_rms_pre", "RMS of signal values outside the effective edge-guard support before applying fill/taper", edge_guard_exterior_rms_pre);
    add_map_double("map_edge_guard_exterior_rms_post", "RMS of signal values outside the effective edge-guard support after applying fill/taper and before filtering", edge_guard_exterior_rms_post);
    add_map_double("map_edge_guard_exterior_max_abs_pre", "maximum absolute signal value outside the effective edge-guard support before applying fill/taper", edge_guard_exterior_max_abs_pre);
    add_map_double("map_edge_guard_exterior_max_abs_post", "maximum absolute signal value outside the effective edge-guard support after applying fill/taper and before filtering", edge_guard_exterior_max_abs_post);
    add_map_int("map_n_valid_pixels", "count of pixels with strictly positive weight", n_valid_pixels);
    add_map_int("map_n_core_pixels", "count of pixels with weight >= map_weight_threshold", n_core_pixels);
    add_map_int("map_peak_row", "row index of the maximum absolute signal-to-noise pixel", peak_row);
    add_map_int("map_peak_col", "column index of the maximum absolute signal-to-noise pixel", peak_col);
    add_map_int("map_edge_guard_applied", "1 when the filter edge guard was applied to this map, 0 otherwise", edge_guard_applied);
    add_map_int("map_edge_guard_support_radius_pix", "support-mask dilation radius in pixels used by the filter edge guard", edge_guard_support_radius_pix);
    add_map_int("map_edge_guard_science_npix", "number of pixels in the filter edge-guard science mask", edge_guard_science_npix);
    add_map_int("map_edge_guard_support_npix", "number of pixels in the filter edge-guard support mask", edge_guard_support_npix);
    add_map_int("map_edge_guard_guardband_npix", "number of pixels in the filter edge-guard guard band (support minus science)", edge_guard_guardband_npix);

    add_map_obs_double("coadd_obs_weight_sum", "sum of positive observation-level raw weight values aligned onto this map grid", obs_weight_sum);
    add_map_obs_double("coadd_obs_weight_frac", "fractional contribution of each obsnum to coadd_obs_weight_sum for a given map", obs_weight_frac);
    add_map_obs_double("coadd_obs_core_weight_sum", "sum of positive observation-level raw weight values within the final map core support", obs_core_weight_sum);
    add_map_obs_double("coadd_obs_core_weight_frac", "fractional contribution of each obsnum within the final map core support", obs_core_weight_frac);
    add_map_obs_int("coadd_obs_n_valid_pixels", "count of aligned observation pixels with positive raw weight", obs_valid_pixels);
    add_map_obs_int("coadd_obs_n_core_pixels", "count of aligned observation pixels with positive raw weight inside the final map core support", obs_core_pixels);
    });
}

void Engine::create_ptcdiag_file() {
    std::string dir_name = obsnum_dir_name + "raw/";
    if (tod_output_subdir_name != "null") {
        dir_name = dir_name + tod_output_subdir_name + "/";
    }

    auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                              engine_utils::toltecIO::ptcdiag,
                                              engine_utils::toltecIO::raw>(dir_name, redu_type, "",
                                                                           obsnum, telescope.sim_obs);
    ptcdiag_filename = filename + ".nc";

    write_netcdf_atomic(ptcdiag_filename, [&](netCDF::NcFile &fo) {
    const int fill_int = -2147483647;
    const double fill_double = std::numeric_limits<double>::quiet_NaN();
    const Eigen::Index n_scans = telescope.scan_indices.cols();
    std::vector<std::size_t> det_chunks = {1, TULA_SIZET(calib.n_dets)};

    netCDF::NcDim n_tod_output_type_dim = fo.addDim("n_tod_output_type", 1);
    netCDF::NcVar tod_output_type_var = fo.addVar("tod_output_type", netCDF::ncString, n_tod_output_type_dim);
    const std::vector<size_t> tod_output_type_index = {0};
    std::string tod_output_type_name = "ptcdiag";
    tod_output_type_var.putVar(tod_output_type_index, tod_output_type_name);

    netCDF::NcVar obsnum_v = fo.addVar("obsnum", netCDF::ncInt);
    obsnum_v.putAtt("units", "N/A");
    int obsnum_int = std::stoi(obsnum);
    obsnum_v.putVar(&obsnum_int);

    netCDF::NcVar source_ra_v = fo.addVar("SourceRa", netCDF::ncDouble);
    source_ra_v.putAtt("units", "rad");
    source_ra_v.putVar(&telescope.tel_header["Header.Source.Ra"](0));

    netCDF::NcVar source_dec_v = fo.addVar("SourceDec", netCDF::ncDouble);
    source_dec_v.putAtt("units", "rad");
    source_dec_v.putVar(&telescope.tel_header["Header.Source.Dec"](0));

    netCDF::NcDim n_scans_dim = fo.addDim("n_scans", n_scans);
    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", calib.n_dets);
    std::vector<netCDF::NcDim> det_dims = {n_scans_dim, n_dets_dim};

    netCDF::NcVar output_scan_index_v = fo.addVar("output_scan_index", netCDF::ncInt, n_scans_dim);
    output_scan_index_v.putAtt("units", "N/A");
    output_scan_index_v.putAtt("comment", "1-based original scan index from the full observation");
    std::vector<int> output_scan_index(static_cast<std::size_t>(n_scans), fill_int);
    for (Eigen::Index i = 0; i < n_scans; ++i) {
        output_scan_index[static_cast<std::size_t>(i)] = static_cast<int>(i + 1);
    }
    output_scan_index_v.putVar(output_scan_index.data());

    auto add_det_meta_int = [&](const std::string &name, const std::string &comment, const std::vector<int> &values) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, n_dets_dim);
        v.putAtt("units", "N/A");
        v.putAtt("comment", comment);
        v.putVar(values.data());
    };
    auto apt_int_values = [&](const std::string &key) {
        std::vector<int> values(static_cast<std::size_t>(calib.n_dets), fill_int);
        auto it = calib.apt.find(key);
        if (it != calib.apt.end() && it->second.size() == calib.n_dets) {
            for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
                values[static_cast<std::size_t>(i)] = static_cast<int>(std::lround(it->second(i)));
            }
        }
        return values;
    };
    add_det_meta_int("ptc_diag_uid", "detector UID along n_dets", apt_int_values("uid"));
    add_det_meta_int("ptc_diag_array", "array index along n_dets", apt_int_values("array"));
    add_det_meta_int("ptc_diag_network", "network index along n_dets", apt_int_values("nw"));
    add_det_meta_int("ptc_diag_apt_flag", "APT detector flag along n_dets", apt_int_values("flag"));

    add_netcdf_var<std::string>(fo, "INSTRUME", "TolTEC");
    add_netcdf_var<std::string>(fo, "TELESCOP", "LMT");
    add_netcdf_var<std::string>(fo, "PIPELINE", "CITLALI");
    add_netcdf_var<std::string>(fo, "VERSION", CITLALI_GIT_VERSION);
    add_netcdf_var<std::string>(fo, "KIDS", KIDSCPP_GIT_VERSION);
    add_netcdf_var<std::string>(fo, "TULA", TULA_GIT_VERSION);
    add_netcdf_var<std::string>(fo, "PROJID", telescope.project_id);
    add_netcdf_var<std::string>(fo, "GOAL", redu_type);
    add_netcdf_var<std::string>(fo, "OBSGOAL", telescope.obs_goal);
    add_netcdf_var<std::string>(fo, "TYPE", tod_type);
    add_netcdf_var(fo, "SAMPRATE", telescope.fsmp);

    add_netcdf_var(fo, "CONFIG.WEIGHT.TYPE", ptcproc.weighting_type);
    add_netcdf_var(fo, "CONFIG.WEIGHT.SOURCE_MASK_RADIUS_ARCSEC", ptcproc.source_mask_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.INV_VAR.PTC.WTLOW", ptcproc.lower_inv_var_factor);
    add_netcdf_var(fo, "CONFIG.INV_VAR.PTC.WTHIGH", ptcproc.upper_inv_var_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.PTC.WTLOW", ptcproc.lower_weight_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.PTC.WTHIGH", ptcproc.upper_weight_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.MEDWTFACTOR", ptcproc.med_weight_factor);
    add_netcdf_var(fo, "CONFIG.INV_VAR.WINDOW_SEC", ptcproc.remove_bad_dets_window_sec);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.ENABLED", ptcproc.weight_corr_penalty.enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.ENABLED", ptcproc.busy_row_suppression.enabled);
    add_netcdf_var(fo, "CONFIG.CLEANED", ptcproc.run_clean);
    add_netcdf_var<std::string>(fo, "CONFIG.CLEANED.MODESEL", ptcproc.cleaner.active_cleaner_label());
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.ENABLED", ptcproc.cleaner.adaptive_selector.enabled);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.ENABLED", ptcproc.second_pass_local.enabled);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_SPIKE_SIGMA", ptcproc.second_pass_local.min_spike_sigma);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_CLUSTER_DETECTORS", ptcproc.second_pass_local.min_cluster_detectors);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MAX_AUTO_FLAG_CLUSTERS", ptcproc.second_pass_local.max_auto_flag_clusters_per_network);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS", ptcproc.run_fruit_loops);
    add_netcdf_var<std::string>(fo, "CONFIG.FRUITLOOPS.PATH", ptcproc.fruit_loops_path);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.S2N", ptcproc.fruit_loops_sig2noise);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.PEAKFRAC", ptcproc.fruit_loops_peak_fraction_limit);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSNR", ptcproc.fruit_loops_local_snr_floor);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_INNER", ptcproc.fruit_loops_local_sigma_inner_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_OUTER", ptcproc.fruit_loops_local_sigma_outer_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_EDGE", ptcproc.fruit_loops_local_sigma_edge_guard_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_MINPIX", ptcproc.fruit_loops_local_sigma_min_pixels);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.ADAPT_SUPPORT_RAD", ptcproc.fruit_loops_adaptive_support_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.ADAPT_SUPPORT_FWHM", ptcproc.fruit_loops_adaptive_support_radius_fwhm);

    auto add_det_double = [&](const std::string &name, const std::string &comment) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, det_dims);
        v.putAtt("units", "N/A");
        v.putAtt("comment", comment);
        v.setChunking(netCDF::NcVar::nc_CHUNKED, det_chunks);
        std::vector<double> init(static_cast<std::size_t>(n_scans) * static_cast<std::size_t>(calib.n_dets), fill_double);
        v.putVar(init.data());
    };
    add_det_double("ptc_detector_weight", "final detector map weight used by PTC for this scan");
    add_det_double("ptc_detector_rms", "per-detector RMS of the PTC timestream written for this scan");
    add_det_double("ptc_detector_stddev", "per-detector standard deviation of the PTC timestream written for this scan");
    add_det_double("ptc_detector_median", "per-detector median of the PTC timestream written for this scan");
    add_det_double("ptc_detector_flagged_fraction", "fraction of detector samples flagged in the PTC timestream for this scan");
    add_det_double("ptc_invvar_window_valid_fraction",
                   "fraction of remove_bad_dets diagnostic windows with enough unflagged samples to estimate inverse variance in the PTC timestream");
    add_det_double("ptc_invvar_window_median",
                   "median per-window inverse variance used for PTC remove_bad_dets diagnostics");
    add_det_double("ptc_invvar_window_q10",
                   "10th percentile of per-window inverse variance used for PTC remove_bad_dets diagnostics");
    add_det_double("ptc_invvar_window_q90",
                   "90th percentile of per-window inverse variance used for PTC remove_bad_dets diagnostics");
    add_det_double("ptc_invvar_window_flagged_frac_median",
                   "median flagged fraction across remove_bad_dets diagnostic windows in the PTC timestream");
    add_det_double("ptc_invvar_window_flagged_frac_max",
                   "maximum flagged fraction across remove_bad_dets diagnostic windows in the PTC timestream");
    add_det_double("ptc_invvar_window_heavy_flagged_fraction",
                   "fraction of remove_bad_dets diagnostic windows in the PTC timestream with at least 50 percent flagged samples");
    {
        netCDF::NcVar v = fo.addVar("ptc_invvar_window_n_total", netCDF::ncInt, det_dims);
        v.putAtt("units", "N/A");
        v.putAtt("comment", "total number of fixed windows evaluated for PTC remove_bad_dets diagnostics");
        v.setChunking(netCDF::NcVar::nc_CHUNKED, det_chunks);
        std::vector<int> init(static_cast<std::size_t>(n_scans) * static_cast<std::size_t>(calib.n_dets), fill_int);
        v.putVar(init.data());
    }
    {
        netCDF::NcVar v = fo.addVar("ptc_invvar_window_n_valid", netCDF::ncInt, det_dims);
        v.putAtt("units", "N/A");
        v.putAtt("comment", "number of fixed windows with a finite inverse-variance estimate for PTC remove_bad_dets diagnostics");
        v.setChunking(netCDF::NcVar::nc_CHUNKED, det_chunks);
        std::vector<int> init(static_cast<std::size_t>(n_scans) * static_cast<std::size_t>(calib.n_dets), fill_int);
        v.putVar(init.data());
    }

    auto add_network_block = [&](const std::string &dim_name,
                                 const std::string &id_name,
                                 const std::string &id_comment,
                                 const std::vector<std::pair<std::string, std::string>> &int_vars,
                                 const std::vector<std::pair<std::string, std::string>> &double_vars) {
        netCDF::NcDim n_nws_dim = fo.addDim(dim_name, calib.n_nws);
        netCDF::NcVar nw_ids_v = fo.addVar(id_name, netCDF::ncInt, n_nws_dim);
        nw_ids_v.putAtt("units", "N/A");
        nw_ids_v.putAtt("comment", id_comment);
        std::vector<int> nw_ids(static_cast<std::size_t>(calib.n_nws), fill_int);
        for (Eigen::Index i = 0; i < calib.n_nws; ++i) {
            nw_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
        }
        nw_ids_v.putVar(nw_ids.data());
        std::vector<netCDF::NcDim> dims = {n_scans_dim, n_nws_dim};
        for (const auto &[name, comment] : int_vars) {
            netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, dims);
            v.putAtt("units", "N/A");
            v.putAtt("comment", comment);
            std::vector<int> init(static_cast<std::size_t>(n_scans) * static_cast<std::size_t>(calib.n_nws), fill_int);
            v.putVar(init.data());
        }
        for (const auto &[name, comment] : double_vars) {
            netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, dims);
            v.putAtt("units", "N/A");
            v.putAtt("comment", comment);
            std::vector<double> init(static_cast<std::size_t>(n_scans) * static_cast<std::size_t>(calib.n_nws), fill_double);
            v.putVar(init.data());
        }
    };

    add_network_block(
        "n_nws_corr",
        "corr_nw_network_ids",
        "network IDs corresponding to n_nws_corr axis",
        {
            {"corr_nw_n_groups", "number of final corr_nw cleaning groups per network"},
            {"corr_nw_n_groups_raw", "number of raw connected components before min_group_size filtering"},
            {"corr_nw_n_det_input", "input detector count in each network block"},
            {"corr_nw_n_det_candidates", "detectors passing apt flag and min_good_frac"},
            {"corr_nw_n_det_used", "candidate detectors with finite non-zero std for correlation"},
            {"corr_nw_n_det_grouped", "detectors included in final cleaned corr_nw groups"},
            {"corr_nw_n_det_ungrouped", "detectors excluded from final cleaned corr_nw groups"},
            {"corr_nw_sample_step", "time decimation factor used for corr_nw grouping"},
        },
        {});

    add_network_block(
        "n_nws_wcorr",
        "weight_corr_penalty_network_ids",
        "network IDs corresponding to n_nws_wcorr axis",
        {
            {"weight_corr_penalty_n_det_input", "detector count in each network block"},
            {"weight_corr_penalty_n_det_candidates", "detectors passing apt flag and min_good_frac"},
            {"weight_corr_penalty_n_det_used", "candidate detectors with finite non-zero std"},
            {"weight_corr_penalty_n_det_weighted", "detectors with positive map weight multiplied by penalty factor"},
            {"weight_corr_penalty_sample_step", "time decimation factor used for penalty metrics"},
        },
        {
            {"weight_corr_penalty_factor", "multiplicative weight penalty factor applied per network in each scan"},
            {"weight_corr_penalty_severity", "normalized [0,1] severity used to derive weight_corr_penalty_factor"},
            {"weight_corr_penalty_pair_med_abs_corr", "median absolute sampled detector-detector correlation per network"},
            {"weight_corr_penalty_cm_el_abs_corr", "absolute correlation between network common mode and TelElAct"},
            {"weight_corr_penalty_cm_low_mid_ratio", "common-mode low/mid bandpower ratio"},
        });

    add_network_block(
        "n_nws_busy_row_suppression",
        "weight_busy_row_suppression_network_ids",
        "network IDs corresponding to n_nws_busy_row_suppression axis",
        {
            {"weight_busy_row_suppression_applied", "1 if busy-row weight suppression was applied to this scan/network block, else 0"},
            {"weight_busy_row_suppression_busy_network_vetoed", "1 if this scan/network exceeded the second-pass busy-network veto threshold, else 0"},
            {"weight_busy_row_suppression_n_candidate_clusters", "candidate second-pass residual cluster count used by the busy-row suppression rule"},
            {"weight_busy_row_suppression_n_det_weighted", "detectors with positive map weight multiplied by the busy-row suppression factor"},
        },
        {
            {"weight_busy_row_suppression_factor", "multiplicative factor applied by busy-row suppression to positive detector map weights"},
            {"weight_busy_row_suppression_max_unflagged_residual_z", "largest absolute unflagged post-PCA residual z used by the busy-row suppression rule"},
        });

    add_network_block(
        "n_nws_adaptive_pca",
        "adaptive_pca_network_ids",
        "network IDs corresponding to n_nws_adaptive_pca axis",
        {
            {"adaptive_pca_selector_used", "1 if the bounded adaptive PCA selector evaluated this scan/network block, else 0"},
            {"adaptive_pca_selector_fallback", "1 if adaptive PCA selector fell back to the configured baseline cut, else 0"},
            {"adaptive_pca_baseline_k", "configured baseline PCA cut for this scan/network block"},
            {"adaptive_pca_chosen_k", "adaptive PCA cut selected for this scan/network block"},
            {"adaptive_pca_runnerup_k", "second-best adaptive PCA cut for this scan/network block"},
            {"adaptive_pca_n_candidates", "number of candidate PCA cuts evaluated for this scan/network block"},
            {"adaptive_pca_n_det_input", "input detector count in this scan/network block before selector filtering"},
            {"adaptive_pca_n_det_used", "detector count retained for adaptive selector scoring"},
            {"adaptive_pca_n_time_used", "sample count retained for adaptive selector scoring"},
            {"adaptive_pca_sample_step", "time decimation factor used by the adaptive selector"},
        },
        {
            {"adaptive_pca_chosen_score", "final normalized adaptive selector score for the chosen PCA cut"},
            {"adaptive_pca_runnerup_score", "final normalized adaptive selector score for the runner-up PCA cut"},
            {"adaptive_pca_score_margin", "chosen minus runner-up score margin; more negative is a clearer adaptive choice"},
            {"adaptive_pca_chosen_med_abs_corr", "median absolute detector-detector correlation for the chosen adaptive PCA cut"},
            {"adaptive_pca_chosen_cm_low_mid_ratio", "common-mode low/mid bandpower ratio for the chosen adaptive PCA cut"},
            {"adaptive_pca_chosen_tail4_binom_z", "tail-excess metric for the chosen adaptive PCA cut"},
            {"adaptive_pca_chosen_top_mode_frac", "top residual covariance mode fraction for the chosen adaptive PCA cut"},
            {"adaptive_pca_eig_solve_msec", "milliseconds spent solving eigenmodes before adaptive scoring"},
            {"adaptive_pca_candidate_eval_msec", "milliseconds spent scoring candidate PCA cuts after eigen solve"},
            {"adaptive_pca_total_msec", "total adaptive PCA milliseconds for this scan/network block"},
        });

    add_network_block(
        "n_nws_ptc_second_pass",
        "ptc_second_pass_network_ids",
        "network IDs corresponding to n_nws_ptc_second_pass axis",
        {
            {"ptc_second_pass_busy_network_vetoed", "1 if this network had more candidate second-pass clusters than the auto-flag limit and was diagnostic-only"},
            {"ptc_second_pass_n_candidate_clusters", "number of candidate second-pass residual clusters in this scan/network"},
            {"ptc_second_pass_n_candidate_events", "number of candidate detector-local residual events contributing to candidate clusters"},
            {"ptc_second_pass_n_accepted_clusters", "number of candidate clusters accepted for auto-flagging after the busy-network veto"},
            {"ptc_second_pass_n_accepted_events", "number of accepted detector-local residual events contributing to auto-flagging"},
            {"ptc_second_pass_n_det_with_added_flags", "number of detectors in this scan/network with at least one sample newly flagged by the PTC second pass"},
            {"ptc_second_pass_max_unflagged_residual_uid", "UID of the detector with the largest absolute unflagged post-PCA residual in this scan/network"},
            {"ptc_second_pass_top_candidate_cluster_sample", "median sample of the strongest candidate second-pass cluster; -2147483647 means none"},
            {"ptc_second_pass_top_candidate_cluster_n_detectors", "number of distinct detectors contributing to the strongest candidate second-pass cluster"},
            {"ptc_second_pass_top_candidate_cluster_n_events", "number of merged detector events contributing to the strongest candidate second-pass cluster"},
            {"ptc_second_pass_top_event_kind", "kind code of the strongest accepted second-pass event (0=raw_like,1=delta_like,-2147483647 means none)"},
            {"ptc_second_pass_top_event_uid", "UID of the strongest accepted second-pass event; -2147483647 means none"},
            {"ptc_second_pass_top_event_sample", "sample of the strongest accepted second-pass event; -2147483647 means none"},
        },
        {
            {"ptc_second_pass_existing_flagged_fraction", "fraction of detector-samples already flagged before the PTC second pass in this scan/network"},
            {"ptc_second_pass_proposed_flagged_fraction", "fraction of detector-samples that the accepted PTC second-pass flags would cover in this scan/network"},
            {"ptc_second_pass_newly_flagged_fraction", "fraction of previously good detector-samples newly flagged by the PTC second pass in this scan/network"},
            {"ptc_second_pass_max_unflagged_residual_z", "largest absolute standardized residual remaining on previously unflagged PTC samples in this scan/network"},
            {"ptc_second_pass_top_candidate_cluster_peak_score", "peak event score of the strongest candidate second-pass cluster in this scan/network"},
            {"ptc_second_pass_top_event_score", "score of the strongest accepted second-pass event; NaN means none"},
        });
    });
}

void Engine::create_rtcdiag_file() {
    std::string dir_name = obsnum_dir_name + "raw/";
    if (tod_output_subdir_name != "null") {
        dir_name = dir_name + tod_output_subdir_name + "/";
    }

    auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                              engine_utils::toltecIO::rtcdiag,
                                              engine_utils::toltecIO::raw>(dir_name, redu_type, "",
                                                                           obsnum, telescope.sim_obs);
    rtcdiag_filename = filename + ".nc";

    write_netcdf_atomic(rtcdiag_filename, [&](netCDF::NcFile &fo) {

    netCDF::NcDim n_tod_output_type_dim = fo.addDim("n_tod_output_type", 1);
    netCDF::NcVar tod_output_type_var = fo.addVar("tod_output_type", netCDF::ncString, n_tod_output_type_dim);
    const std::vector<size_t> tod_output_type_index = {0};
    std::string tod_output_type_name = "rtcdiag";
    tod_output_type_var.putVar(tod_output_type_index, tod_output_type_name);

    const int fill_int = -2147483647;
    const double fill_double = std::numeric_limits<double>::quiet_NaN();
    const Eigen::Index n_scans = telescope.scan_indices.cols();
    const double rtc_fsmp = rtcproc.run_downsample ? telescope.d_fsmp : telescope.fsmp;

    netCDF::NcVar obsnum_v = fo.addVar("obsnum", netCDF::ncInt);
    obsnum_v.putAtt("units", "N/A");
    int obsnum_int = std::stoi(obsnum);
    obsnum_v.putVar(&obsnum_int);

    netCDF::NcVar source_ra_v = fo.addVar("SourceRa", netCDF::ncDouble);
    source_ra_v.putAtt("units", "rad");
    source_ra_v.putVar(&telescope.tel_header["Header.Source.Ra"](0));

    netCDF::NcVar source_dec_v = fo.addVar("SourceDec", netCDF::ncDouble);
    source_dec_v.putAtt("units", "rad");
    source_dec_v.putVar(&telescope.tel_header["Header.Source.Dec"](0));

    netCDF::NcDim n_scans_dim = fo.addDim("n_scans", n_scans);
    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", calib.n_dets);
    netCDF::NcDim n_arrays_dim = fo.addDim("n_arrays", calib.n_arrays);
    netCDF::NcDim n_nws_rtcdiag_dim = fo.addDim("n_nws_rtcdiag", calib.n_nws);
    const std::vector<std::size_t> scan_chunks = {TULA_SIZET(std::max<Eigen::Index>(n_scans, 1))};
    const std::vector<std::size_t> scan_array_chunks = {
        1, TULA_SIZET(std::max<Eigen::Index>(calib.n_arrays, 1))};
    const std::vector<std::size_t> rtc_det_chunks = {1, TULA_SIZET(calib.n_dets)};
    const std::vector<std::size_t> rtc_nw_chunks = {1, TULA_SIZET(calib.n_nws)};

    netCDF::NcVar output_scan_index_v = fo.addVar("output_scan_index", netCDF::ncInt, n_scans_dim);
    output_scan_index_v.putAtt("units", "N/A");
    output_scan_index_v.putAtt("comment", "1-based original scan index from the full observation");
    std::vector<int> output_scan_index(static_cast<std::size_t>(n_scans), fill_int);
    for (Eigen::Index i = 0; i < n_scans; ++i) {
        output_scan_index[static_cast<std::size_t>(i)] = static_cast<int>(i + 1);
    }
    output_scan_index_v.putVar(output_scan_index.data());

    netCDF::NcVar array_ids_v = fo.addVar("rtc_diag_array_ids", netCDF::ncInt, n_arrays_dim);
    array_ids_v.putAtt("units", "N/A");
    array_ids_v.putAtt("comment", "array IDs corresponding to n_arrays axis");
    std::vector<int> array_ids(static_cast<std::size_t>(calib.n_arrays), fill_int);
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        array_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.arrays(i));
    }
    array_ids_v.putVar(array_ids.data());

    auto percentile_sorted = [](const std::vector<double> &sorted_values, double pct) {
        if (sorted_values.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (sorted_values.size() == 1) {
            return sorted_values.front();
        }
        pct = std::min(100.0, std::max(0.0, pct));
        const double pos = (pct / 100.0) * static_cast<double>(sorted_values.size() - 1);
        const auto lo = static_cast<std::size_t>(std::floor(pos));
        const auto hi = static_cast<std::size_t>(std::ceil(pos));
        const double frac = pos - static_cast<double>(lo);
        return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac;
    };

    auto add_scan_double = [&](const std::string &name, const std::string &units,
                               const std::string &comment, const std::vector<double> &values) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, n_scans_dim);
        v.putAtt("units", units);
        v.putAtt("comment", comment);
        set_netcdf_chunking_and_compression(v, scan_chunks, 1);
        v.putVar(values.data());
    };

    std::vector<double> scan_duration_s(static_cast<std::size_t>(n_scans), fill_double);
    std::vector<double> scan_speed_p50_arcsec_s(static_cast<std::size_t>(n_scans), fill_double);
    std::vector<double> scan_speed_p95_arcsec_s(static_cast<std::size_t>(n_scans), fill_double);
    std::vector<double> scan_speed_p995_arcsec_s(static_cast<std::size_t>(n_scans), fill_double);

    const auto tel_time_it = telescope.tel_data.find("TelTime");
    const auto az_it = telescope.tel_data.find("az_phys");
    const auto alt_it = telescope.tel_data.find("alt_phys");
    if (tel_time_it != telescope.tel_data.end() &&
        az_it != telescope.tel_data.end() &&
        alt_it != telescope.tel_data.end()) {
        const auto &tel_time = tel_time_it->second;
        const auto &az_phys = az_it->second;
        const auto &alt_phys = alt_it->second;
        const Eigen::Index n_tel = std::min({tel_time.size(), az_phys.size(), alt_phys.size()});
        for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
            const Eigen::Index start = std::max<Eigen::Index>(0, telescope.scan_indices(0, scan));
            const Eigen::Index stop = std::min<Eigen::Index>(n_tel - 1, telescope.scan_indices(1, scan));
            if (stop <= start || start < 0 || stop >= n_tel) {
                continue;
            }
            const double duration = tel_time(stop) - tel_time(start);
            if (std::isfinite(duration) && duration > 0.0) {
                scan_duration_s[static_cast<std::size_t>(scan)] = duration;
            }
            std::vector<double> speed_arcsec_s;
            speed_arcsec_s.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(stop - start, 0)));
            for (Eigen::Index i = start; i < stop; ++i) {
                const double dt = tel_time(i + 1) - tel_time(i);
                const double daz = az_phys(i + 1) - az_phys(i);
                const double dalt = alt_phys(i + 1) - alt_phys(i);
                if (!std::isfinite(dt) || !std::isfinite(daz) || !std::isfinite(dalt) ||
                    dt <= 0.0 || dt > 0.1 || std::abs(daz) > 0.01 || std::abs(dalt) > 0.01) {
                    continue;
                }
                speed_arcsec_s.push_back(std::hypot(daz, dalt) / dt * RAD_TO_ASEC);
            }
            if (!speed_arcsec_s.empty()) {
                std::sort(speed_arcsec_s.begin(), speed_arcsec_s.end());
                scan_speed_p50_arcsec_s[static_cast<std::size_t>(scan)] =
                    percentile_sorted(speed_arcsec_s, 50.0);
                scan_speed_p95_arcsec_s[static_cast<std::size_t>(scan)] =
                    percentile_sorted(speed_arcsec_s, 95.0);
                scan_speed_p995_arcsec_s[static_cast<std::size_t>(scan)] =
                    percentile_sorted(speed_arcsec_s, 99.5);
            }
        }
    }
    else {
        logger->warn("rtcdiag scan-speed diagnostics skipped: missing TelTime, az_phys, or alt_phys telescope data");
    }

    add_scan_double("scan_duration_s", "s",
                    "inner scan duration used for scan-speed diagnostics", scan_duration_s);
    add_scan_double("scan_speed_altaz_p50_arcsec_s", "arcsec/s",
                    "per-scan median boresight speed in the delta-source altaz frame",
                    scan_speed_p50_arcsec_s);
    add_scan_double("scan_speed_altaz_p95_arcsec_s", "arcsec/s",
                    "per-scan 95th percentile boresight speed in the delta-source altaz frame",
                    scan_speed_p95_arcsec_s);
    add_scan_double("scan_speed_altaz_p995_arcsec_s", "arcsec/s",
                    "per-scan robust peak (99.5th percentile) boresight speed in the delta-source altaz frame",
                    scan_speed_p995_arcsec_s);

    std::vector<netCDF::NcDim> scan_array_dims = {n_scans_dim, n_arrays_dim};
    std::vector<double> source_power_half_bandwidth_hz(
        static_cast<std::size_t>(n_scans) * static_cast<std::size_t>(calib.n_arrays), fill_double);
    std::vector<double> tod_lowpass_to_source_power_half_ratio(
        static_cast<std::size_t>(n_scans) * static_cast<std::size_t>(calib.n_arrays), fill_double);
    for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
        const double speed = scan_speed_p995_arcsec_s[static_cast<std::size_t>(scan)];
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
                0.5 * (std::get<0>(fwhm_it->second) + std::get<1>(fwhm_it->second));
            if (!std::isfinite(fwhm_arcsec) || fwhm_arcsec <= 0.0) {
                continue;
            }
            const double f_half_hz =
                (std::sqrt(std::log(2.0)) / (2.0 * pi * fwhm_arcsec * FWHM_TO_STD)) * speed;
            const auto flat_i = static_cast<std::size_t>(scan) * static_cast<std::size_t>(calib.n_arrays) +
                                static_cast<std::size_t>(arr_i);
            source_power_half_bandwidth_hz[flat_i] = f_half_hz;
            if (rtcproc.run_tod_filter && rtcproc.filter.freq_high_Hz > 0.0 && f_half_hz > 0.0) {
                tod_lowpass_to_source_power_half_ratio[flat_i] =
                    rtcproc.filter.freq_high_Hz / f_half_hz;
            }
        }
    }
    auto add_scan_array_double = [&](const std::string &name, const std::string &units,
                                     const std::string &comment, const std::vector<double> &values) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, scan_array_dims);
        v.putAtt("units", units);
        v.putAtt("comment", comment);
        set_netcdf_chunking_and_compression(v, scan_array_chunks, 1);
        v.putVar(values.data());
    };
    add_scan_array_double(
        "scan_source_power_half_bandwidth_hz", "Hz",
        "Gaussian compact-source temporal power half-bandwidth from scan_speed_altaz_p995_arcsec_s and array mean FWHM",
        source_power_half_bandwidth_hz);
    add_scan_array_double(
        "scan_tod_lowpass_to_source_power_half_ratio", "N/A",
        "configured RTC FIR low-pass cutoff divided by scan_source_power_half_bandwidth_hz; values much larger than 1 indicate extra high-frequency noise admitted relative to compact-source half-power bandwidth",
        tod_lowpass_to_source_power_half_ratio);

    netCDF::NcVar nw_ids_v = fo.addVar("rtc_diag_network_ids", netCDF::ncInt, n_nws_rtcdiag_dim);
    nw_ids_v.putAtt("units", "N/A");
    nw_ids_v.putAtt("comment", "network IDs corresponding to n_nws_rtcdiag axis");
    std::vector<int> nw_ids(static_cast<std::size_t>(calib.n_nws), fill_int);
    for (Eigen::Index i = 0; i < calib.n_nws; ++i) {
        nw_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
    }
    nw_ids_v.putVar(nw_ids.data());

    add_netcdf_var<std::string>(fo, "INSTRUME", "TolTEC");
    add_netcdf_var<std::string>(fo, "TELESCOP", "LMT");
    add_netcdf_var<std::string>(fo, "PIPELINE", "CITLALI");
    add_netcdf_var<std::string>(fo, "VERSION", CITLALI_GIT_VERSION);
    add_netcdf_var<std::string>(fo, "KIDS", KIDSCPP_GIT_VERSION);
    add_netcdf_var<std::string>(fo, "TULA", TULA_GIT_VERSION);
    add_netcdf_var<std::string>(fo, "PROJID", telescope.project_id);
    add_netcdf_var<std::string>(fo, "GOAL", redu_type);
    add_netcdf_var<std::string>(fo, "OBSGOAL", telescope.obs_goal);
    add_netcdf_var<std::string>(fo, "TYPE", tod_type);
    add_netcdf_var(fo, "SAMPRATE", telescope.fsmp);
    add_netcdf_var(fo, "RTC_SAMPRATE", rtc_fsmp);
    add_netcdf_var(fo, "CONFIG.TODFILTERED", rtcproc.run_tod_filter);
    add_netcdf_var(fo, "CONFIG.TODFILTER.FREQ_HIGH_HZ", rtcproc.filter.freq_high_Hz);
    add_netcdf_var(fo, "CONFIG.TODFILTER.FREQ_LOW_HZ", rtcproc.filter.freq_low_Hz);
    add_netcdf_var(fo, "CONFIG.TODFILTER.N_TERMS", rtcproc.filter.n_terms);
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.ENABLED", rtcproc.filter_edge_guard.enabled);
    add_netcdf_var<std::string>(fo, "CONFIG.TODFILTER.EDGE_GUARD.MODE", rtcproc.filter_edge_guard.mode);
    add_netcdf_var<std::string>(fo, "CONFIG.TODFILTER.EDGE_GUARD.COMBINE", rtcproc.filter_edge_guard.combine);
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.CONTEXT_SAMPLES", rtcproc.filter_edge_guard.context_samples);
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.GUARD_SAMPLES", rtcproc.filter_edge_guard.guard_samples);
    add_netcdf_var(fo, "CONFIG.TOD.OUTER_CONTEXT_SAMPLES", telescope.outer_scans_chunk);
    add_netcdf_var(fo, "CONFIG.TOD.OUTPUT_OUTER_CONTEXT_SAMPLES", rtcproc.tod_output_outer_context_samples);
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.MIN_SAMPLES", rtcproc.filter_edge_guard.min_samples);
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.EXTRA_SAMPLES", rtcproc.filter_edge_guard.extra_samples);
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.MAX_SAMPLES", rtcproc.filter_edge_guard.max_samples);
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.IIR_SETTLE_ATTENUATION", rtcproc.filter_edge_guard.iir_settle_attenuation);

    // Keep a compact provenance subset so rtcdiag is interpretable without the RTC TOD.
    add_netcdf_var(fo, "CONFIG.VERBOSE", verbose_mode);
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
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.PRE_WINDOW_SEC", rtcproc.impulsive_capture.snippet_pre_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.POST_WINDOW_SEC", rtcproc.impulsive_capture.snippet_post_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.ENABLED", rtcproc.impulsive_coincidence.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_GOOD_FRAC", rtcproc.impulsive_coincidence.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.EVENT_SCORE_THRESH", rtcproc.impulsive_coincidence.event_score_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_USED", rtcproc.impulsive_coincidence.min_det_used);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_FRAC", rtcproc.impulsive_coincidence.min_impulsive_det_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_ALIGNMENT_FRAC", rtcproc.impulsive_coincidence.min_alignment_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_NETWORKS_ALIGNED", rtcproc.impulsive_coincidence.min_networks_aligned);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_OVERRIDE_THRESH", rtcproc.impulsive_coincidence.high_score_override_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_MIN_NETWORKS", rtcproc.impulsive_coincidence.high_score_min_networks_aligned);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.CLUSTER_TOL_SEC", rtcproc.impulsive_coincidence.cluster_tol_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.PRE_WINDOW_SEC", rtcproc.impulsive_coincidence.mask_pre_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.POST_WINDOW_SEC", rtcproc.impulsive_coincidence.mask_post_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MAX_FLAGGED_FRAC", rtcproc.impulsive_coincidence.max_flagged_fraction);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.ENABLED", rtcproc.line_audit.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.LINE_MIN_HZ", rtcproc.line_audit.line_min_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.LINE_MAX_HZ", rtcproc.line_audit.line_max_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.SEGMENT_SEC", rtcproc.line_audit.segment_sec);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_SEGMENT_SEC", rtcproc.line_audit.min_segment_sec);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.OVERLAP_FRAC", rtcproc.line_audit.overlap_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CONTINUUM_RADIUS_BINS", rtcproc.line_audit.continuum_radius_bins);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PROMINENCE_THRESH", rtcproc.line_audit.prominence_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CM_PROMINENCE_THRESH", rtcproc.line_audit.cm_prominence_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_GOOD_FRAC", rtcproc.line_audit.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_WINDOWS", rtcproc.line_audit.min_windows);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MAX_PEAKS_PER_DETECTOR", rtcproc.line_audit.max_peaks_per_detector);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MAX_DET", rtcproc.line_audit.max_det);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_DET_FOR_NETWORK", rtcproc.line_audit.min_det_for_network);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CLUSTER_TOL_HZ", rtcproc.line_audit.cluster_tol_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTOR_FRAC", rtcproc.line_audit.notch_min_detector_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTORS", rtcproc.line_audit.notch_min_detectors);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_CM_PROMINENCE", rtcproc.line_audit.notch_min_cm_prominence);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_MIN_PROMINENCE", rtcproc.line_audit.detector_min_prominence);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_MIN_LINE_POWER_FRAC", rtcproc.line_audit.detector_min_line_power_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.BAD_DETECTOR_MAX_CLUSTER_FRAC", rtcproc.line_audit.bad_detector_max_cluster_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PRE_FILTER_ENABLED", rtcproc.line_audit.pre_filter_enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_ENABLED", rtcproc.line_audit.post_filter_enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_SHARED_NOTCHES", rtcproc.line_audit.post_filter_apply_shared_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_DETECTOR_NOTCHES", rtcproc.line_audit.post_filter_apply_detector_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_ITERATIONS", rtcproc.line_audit.post_filter_apply_iterations);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_LINE_MIN_HZ", rtcproc.line_audit.post_filter_line_min_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_LINE_MAX_HZ", rtcproc.line_audit.post_filter_line_max_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_MODEL_PROTECTED_ENABLED", rtcproc.line_audit.ptc_model_protected_enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_REQUIRE_MODEL_SUBTRACTED", rtcproc.line_audit.ptc_require_model_subtracted);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_FIXED_NOTCHES", rtcproc.line_audit.ptc_apply_fixed_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_SHARED_NOTCHES", rtcproc.line_audit.ptc_apply_shared_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_DETECTOR_NOTCHES", rtcproc.line_audit.ptc_apply_detector_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_ITERATIONS", rtcproc.line_audit.ptc_apply_iterations);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_LINE_MIN_HZ", rtcproc.line_audit.ptc_line_min_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_LINE_MAX_HZ", rtcproc.line_audit.ptc_line_max_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_ENABLED", rtcproc.line_audit.fixed_notch_enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_COUNT", static_cast<int>(rtcproc.line_audit.fixed_notch_freqs_hz.size()));
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_WIDTH_COUNT", static_cast<int>(rtcproc.line_audit.fixed_notch_widths_hz.size()));
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_EXCLUSION_HALF_WIDTH_HZ", rtcproc.line_audit.fixed_notch_exclusion_half_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_SHARED_NOTCHES", rtcproc.line_audit.apply_shared_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_SUPPORT_NETWORKS", rtcproc.line_audit.apply_min_support_networks);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_DETECTOR_FRAC", rtcproc.line_audit.apply_min_detector_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_CM_PROMINENCE", rtcproc.line_audit.apply_min_common_mode_prominence);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_WIDTH_SCALE", rtcproc.line_audit.apply_width_scale);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_WIDTH_HZ", rtcproc.line_audit.apply_min_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MAX_WIDTH_HZ", rtcproc.line_audit.apply_max_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MAX_NOTCHES", rtcproc.line_audit.apply_max_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_CLUSTER_TOL_HZ", rtcproc.line_audit.apply_cluster_tol_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_PROMINENCE", rtcproc.line_audit.detector_notch_min_prominence);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_LINE_POWER_FRAC", rtcproc.line_audit.detector_notch_min_line_power_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MAX_NOTCHES", rtcproc.line_audit.detector_notch_max_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_WIDTH_SCALE", rtcproc.line_audit.detector_notch_width_scale);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_WIDTH_HZ", rtcproc.line_audit.detector_notch_min_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MAX_WIDTH_HZ", rtcproc.line_audit.detector_notch_max_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_CONTEXT_SAMPLES", rtcproc.line_audit.detector_notch_context_samples);
    add_netcdf_var(fo, "CONFIG.INV_VAR.WINDOW_SEC", rtcproc.remove_bad_dets_window_sec);

    for (auto const &x : calib.apt) {
        netCDF::NcVar apt_v = fo.addVar("apt_" + x.first, netCDF::ncDouble, n_dets_dim);
        apt_v.putAtt("units", calib.apt_header_units[x.first]);
        apt_v.putVar(x.second.data());
    }

    std::vector<netCDF::NcDim> rtc_det_dims = {n_scans_dim, n_dets_dim};
    auto add_rtc_det_double = [&](const std::string &name, const std::string &comment) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, rtc_det_dims);
        v.putAtt("units", "N/A");
        v.putAtt("comment", comment);
        set_netcdf_chunking_and_compression(v, rtc_det_chunks, 1);
        std::vector<double> init(static_cast<std::size_t>(n_scans) *
                                 static_cast<std::size_t>(calib.n_dets), fill_double);
        v.putVar(init.data());
    };
    auto add_rtc_det_int = [&](const std::string &name, const std::string &comment) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, rtc_det_dims);
        v.putAtt("units", "N/A");
        v.putAtt("comment", comment);
        set_netcdf_chunking_and_compression(v, rtc_det_chunks, 1);
        std::vector<int> init(static_cast<std::size_t>(n_scans) *
                              static_cast<std::size_t>(calib.n_dets), fill_int);
        v.putVar(init.data());
    };

    add_rtc_det_int("rtc_despike_raw_exceed_count",
                    "per-detector count of raw-sample MAD-threshold exceedances before despike expansion");
    add_rtc_det_int("rtc_despike_local_raw_candidate_count",
                    "per-detector count of locally detrended raw candidate events considered by the compact-raw gate");
    add_rtc_det_int("rtc_despike_local_raw_accepted_event_count",
                    "per-detector count of locally detrended raw candidate events accepted by the compact-raw gate");
    add_rtc_det_int("rtc_despike_local_flagged_sample_count",
                    "per-detector count of samples flagged by accepted compact-raw local-residual events");
    add_rtc_det_int("rtc_despike_local_exceed_count",
                    "legacy alias for rtc_despike_local_flagged_sample_count");
    add_rtc_det_int("rtc_despike_local_raw_reject_count",
                    "per-detector count of locally detrended raw candidate events rejected by the compact-raw gate");
    add_rtc_det_int("rtc_despike_delta_spike_count",
                    "per-detector count of delta-domain spikes identified by the RTC despiker");
    add_rtc_det_int("rtc_despike_local_delta_candidate_count",
                    "per-detector count of locally detrended delta candidate events considered by the compact-delta gate");
    add_rtc_det_int("rtc_despike_local_delta_accepted_event_count",
                    "per-detector count of locally detrended delta candidate events accepted by the compact-delta gate");
    add_rtc_det_int("rtc_despike_local_delta_exceed_count",
                    "legacy alias for rtc_despike_local_delta_accepted_event_count");
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
    add_rtc_det_int("rtc_detector_notch_n_applied",
                    "per-detector count of post-filter detector-local RTC notches applied");
    add_rtc_det_double("rtc_detector_notch_primary_freq_hz",
                       "frequency of the strongest detector-local post-filter RTC notch applied");
    add_rtc_det_double("rtc_detector_notch_primary_width_hz",
                       "bandwidth of the strongest detector-local post-filter RTC notch applied");
    add_rtc_det_double("rtc_detector_notch_primary_prominence",
                       "PSD prominence of the strongest detector-local post-filter RTC notch applied");
    add_rtc_det_double("rtc_detector_notch_primary_line_power_frac",
                       "line-power fraction of the strongest detector-local post-filter RTC notch applied");
    add_rtc_det_double("rtc_detector_notch_rms_before",
                       "robust RMS of the detector RTC timestream before detector-local post-filter notching");
    add_rtc_det_double("rtc_detector_notch_rms_after",
                       "robust RMS of the detector RTC timestream after detector-local post-filter notching");
    add_rtc_det_double("rtc_invvar_window_valid_fraction",
                       "fraction of remove_bad_dets diagnostic windows with enough unflagged samples to estimate inverse variance in the RTC timestream");
    add_rtc_det_double("rtc_invvar_window_median",
                       "median per-window inverse variance used for RTC remove_bad_dets diagnostics");
    add_rtc_det_double("rtc_invvar_window_q10",
                       "10th percentile of per-window inverse variance used for RTC remove_bad_dets diagnostics");
    add_rtc_det_double("rtc_invvar_window_q90",
                       "90th percentile of per-window inverse variance used for RTC remove_bad_dets diagnostics");
    add_rtc_det_double("rtc_invvar_window_flagged_frac_median",
                       "median flagged fraction across remove_bad_dets diagnostic windows in the RTC timestream");
    add_rtc_det_double("rtc_invvar_window_flagged_frac_max",
                       "maximum flagged fraction across remove_bad_dets diagnostic windows in the RTC timestream");
    add_rtc_det_double("rtc_invvar_window_heavy_flagged_fraction",
                       "fraction of remove_bad_dets diagnostic windows in the RTC timestream with at least 50 percent flagged samples");
    add_rtc_det_int("rtc_invvar_window_n_total",
                    "total number of fixed windows evaluated for RTC remove_bad_dets diagnostics");
    add_rtc_det_int("rtc_invvar_window_n_valid",
                    "number of fixed windows with a finite inverse-variance estimate for RTC remove_bad_dets diagnostics");

    std::vector<netCDF::NcDim> rtc_nw_dims = {n_scans_dim, n_nws_rtcdiag_dim};
    auto add_rtc_nw_double = [&](const std::string &name, const std::string &comment) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, rtc_nw_dims);
        v.putAtt("units", "N/A");
        v.putAtt("comment", comment);
        set_netcdf_chunking_and_compression(v, rtc_nw_chunks, 1);
        std::vector<double> init(static_cast<std::size_t>(n_scans) *
                                 static_cast<std::size_t>(calib.n_nws), fill_double);
        v.putVar(init.data());
    };
    auto add_rtc_nw_int = [&](const std::string &name, const std::string &comment) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, rtc_nw_dims);
        v.putAtt("units", "N/A");
        v.putAtt("comment", comment);
        set_netcdf_chunking_and_compression(v, rtc_nw_chunks, 1);
        std::vector<int> init(static_cast<std::size_t>(n_scans) *
                              static_cast<std::size_t>(calib.n_nws), fill_int);
        v.putVar(init.data());
    };

    add_rtc_nw_int("rtc_network_n_det_input",
                   "input detector count in each RTC network block");
    add_rtc_nw_int("rtc_network_n_det_used",
                   "detectors passing the step-mask valid-sample threshold and finite robust scale");
    add_rtc_nw_int("rtc_network_impulsive_n_det_used",
                   "detectors passing the impulsive-coincidence valid-sample threshold and finite robust scale");
    add_rtc_nw_int("rtc_network_line_audit_n_det_used",
                   "detectors analyzed by the pre-filter RTC line audit in each network block");
    add_rtc_nw_double("rtc_network_line_audit_shared_freq_hz",
                      "frequency of the strongest shared narrowband RTC line family in each network block");
    add_rtc_nw_int("rtc_network_line_audit_shared_detector_count",
                   "number of detectors participating in the strongest shared narrowband RTC line family");
    add_rtc_nw_double("rtc_network_line_audit_shared_detector_frac",
                      "fraction of audited detectors participating in the strongest shared narrowband RTC line family");
    add_rtc_nw_double("rtc_network_line_audit_shared_median_prominence",
                      "median detector-level PSD prominence of the strongest shared narrowband RTC line family");
    add_rtc_nw_double("rtc_network_line_audit_shared_max_prominence",
                      "maximum detector-level PSD prominence of the strongest shared narrowband RTC line family");
    add_rtc_nw_double("rtc_network_line_audit_shared_width_hz",
                      "median linewidth of the strongest shared narrowband RTC line family");
    add_rtc_nw_double("rtc_network_line_audit_shared_line_power_frac",
                      "median detector-level line-power fraction of the strongest shared narrowband RTC line family");
    add_rtc_nw_double("rtc_network_line_audit_shared_common_mode_freq_hz",
                      "matched common-mode line frequency for the strongest shared narrowband RTC line family");
    add_rtc_nw_double("rtc_network_line_audit_shared_common_mode_prominence",
                      "matched common-mode PSD prominence for the strongest shared narrowband RTC line family");
    add_rtc_nw_double("rtc_network_line_audit_shared_notch_score",
                      "shared-line notch score, detector fraction times median prominence");
    add_rtc_nw_int("rtc_network_line_audit_shared_recommend_notch",
                   "1 if the strongest shared narrowband RTC line family met the current notch-candidate criteria");
    add_rtc_nw_int("rtc_network_line_audit_n_applied_notches",
                   "number of chunk-level shared-line RTC notches actually applied to this scan");
    add_rtc_nw_int("rtc_network_line_audit_shared_applied_notch",
                   "1 if the strongest shared narrowband RTC line family in this network matched an applied chunk-level RTC notch");
    add_rtc_nw_double("rtc_network_line_audit_shared_applied_freq_hz",
                      "center frequency of the applied chunk-level RTC notch matched to the strongest shared narrowband RTC line family");
    add_rtc_nw_double("rtc_network_line_audit_shared_applied_width_hz",
                      "full-width bandwidth of the applied chunk-level RTC notch matched to the strongest shared narrowband RTC line family");
    add_rtc_nw_int("rtc_network_line_audit_shared_applied_support_network_count",
                   "number of networks supporting the applied chunk-level RTC notch matched to the strongest shared narrowband RTC line family");
    add_rtc_nw_int("rtc_network_line_audit_detector_candidate_uid",
                   "UID of the strongest detector-local RTC line candidate in each network block; -2147483647 means none");
    add_rtc_nw_double("rtc_network_line_audit_detector_candidate_freq_hz",
                      "frequency of the strongest detector-local RTC line candidate");
    add_rtc_nw_double("rtc_network_line_audit_detector_candidate_prominence",
                      "PSD prominence of the strongest detector-local RTC line candidate");
    add_rtc_nw_double("rtc_network_line_audit_detector_candidate_line_power_frac",
                      "line-power fraction of the strongest detector-local RTC line candidate");
    add_rtc_nw_double("rtc_network_line_audit_detector_candidate_cluster_detector_frac",
                      "shared-cluster detector fraction associated with the strongest detector-local RTC line candidate");
    add_rtc_nw_int("rtc_network_line_audit_detector_candidate_recommend_flag",
                   "1 if the strongest detector-local RTC line candidate met the current bad-detector criteria");
    auto add_rtc_nw_line_audit_diag = [&](const std::string &prefix, const std::string &stage) {
        add_rtc_nw_int(prefix + "_n_det_used",
                       "detectors analyzed by the " + stage + " RTC line audit in each network block");
        add_rtc_nw_double(prefix + "_shared_freq_hz",
                          "frequency of the strongest shared narrowband " + stage + " RTC line family in each network block");
        add_rtc_nw_int(prefix + "_shared_detector_count",
                       "number of detectors participating in the strongest shared narrowband " + stage + " RTC line family");
        add_rtc_nw_double(prefix + "_shared_detector_frac",
                          "fraction of audited detectors participating in the strongest shared narrowband " + stage + " RTC line family");
        add_rtc_nw_double(prefix + "_shared_median_prominence",
                          "median detector-level PSD prominence of the strongest shared narrowband " + stage + " RTC line family");
        add_rtc_nw_double(prefix + "_shared_max_prominence",
                          "maximum detector-level PSD prominence of the strongest shared narrowband " + stage + " RTC line family");
        add_rtc_nw_double(prefix + "_shared_width_hz",
                          "median linewidth of the strongest shared narrowband " + stage + " RTC line family");
        add_rtc_nw_double(prefix + "_shared_line_power_frac",
                          "median detector-level line-power fraction of the strongest shared narrowband " + stage + " RTC line family");
        add_rtc_nw_double(prefix + "_shared_common_mode_freq_hz",
                          "matched common-mode line frequency for the strongest shared narrowband " + stage + " RTC line family");
        add_rtc_nw_double(prefix + "_shared_common_mode_prominence",
                          "matched common-mode PSD prominence for the strongest shared narrowband " + stage + " RTC line family");
        add_rtc_nw_double(prefix + "_shared_notch_score",
                          "shared-line notch score, detector fraction times median prominence");
        add_rtc_nw_int(prefix + "_shared_recommend_notch",
                       "1 if the strongest shared narrowband " + stage + " RTC line family met the current notch-candidate criteria");
        add_rtc_nw_int(prefix + "_n_applied_notches",
                       "number of chunk-level shared-line RTC notches actually applied in the " + stage + " stage");
        add_rtc_nw_int(prefix + "_shared_applied_notch",
                       "1 if the strongest shared narrowband " + stage + " RTC line family in this network matched an applied chunk-level RTC notch");
        add_rtc_nw_double(prefix + "_shared_applied_freq_hz",
                          "center frequency of the applied chunk-level RTC notch matched to the strongest shared narrowband " + stage + " RTC line family");
        add_rtc_nw_double(prefix + "_shared_applied_width_hz",
                          "full-width bandwidth of the applied chunk-level RTC notch matched to the strongest shared narrowband " + stage + " RTC line family");
        add_rtc_nw_int(prefix + "_shared_applied_support_network_count",
                       "number of networks supporting the applied chunk-level RTC notch matched to the strongest shared narrowband " + stage + " RTC line family");
        add_rtc_nw_int(prefix + "_detector_candidate_uid",
                       "UID of the strongest detector-local " + stage + " RTC line candidate in each network block; -2147483647 means none");
        add_rtc_nw_double(prefix + "_detector_candidate_freq_hz",
                          "frequency of the strongest detector-local " + stage + " RTC line candidate");
        add_rtc_nw_double(prefix + "_detector_candidate_prominence",
                          "PSD prominence of the strongest detector-local " + stage + " RTC line candidate");
        add_rtc_nw_double(prefix + "_detector_candidate_line_power_frac",
                          "line-power fraction of the strongest detector-local " + stage + " RTC line candidate");
        add_rtc_nw_double(prefix + "_detector_candidate_cluster_detector_frac",
                          "shared-cluster detector fraction associated with the strongest detector-local " + stage + " RTC line candidate");
        add_rtc_nw_int(prefix + "_detector_candidate_recommend_flag",
                       "1 if the strongest detector-local " + stage + " RTC line candidate met the current bad-detector criteria");
    };
    add_rtc_nw_line_audit_diag("rtc_network_post_line_audit", "post-filter");
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
    add_rtc_nw_double("rtc_network_impulsive_score_median",
                      "median detector impulsive-event score within each RTC network block");
    add_rtc_nw_double("rtc_network_impulsive_score_max",
                      "maximum detector impulsive-event score within each RTC network block");
    add_rtc_nw_double("rtc_network_impulsive_det_frac",
                      "fraction of diagnostic-used detectors with impulsive-event score above the impulsive coincidence threshold");
    add_rtc_nw_double("rtc_network_impulsive_alignment_frac",
                      "fraction of impulsive-active detectors aligned in the dominant impulsive time cluster");
    add_rtc_nw_int("rtc_network_impulsive_dominant_sample",
                   "dominant aligned impulsive sample within each RTC network block; -2147483647 means unavailable");
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
    add_rtc_nw_int("rtc_network_impulsive_mask_applied",
                   "1 if impulsive_coincidence_mask flagged a time window for this RTC network block, else 0");
    add_rtc_nw_int("rtc_network_impulsive_mask_start_sample",
                   "inclusive starting sample of the applied impulsive_coincidence_mask window; -2147483647 means none");
    add_rtc_nw_int("rtc_network_impulsive_mask_end_sample",
                   "inclusive ending sample of the applied impulsive_coincidence_mask window; -2147483647 means none");
    add_rtc_nw_int("rtc_network_impulsive_mask_window_samples",
                   "number of RTC time samples in the applied impulsive_coincidence_mask window");
    add_rtc_nw_int("rtc_network_impulsive_mask_n_det_masked",
                   "number of detectors included in the applied impulsive_coincidence_mask window");
    add_rtc_nw_int("rtc_network_impulsive_mask_n_det_samples_flagged",
                   "number of previously good detector-samples newly flagged by impulsive_coincidence_mask");
    add_rtc_nw_double("rtc_network_impulsive_mask_flagged_fraction",
                      "fraction of previously good detector-samples in the network block newly flagged by impulsive_coincidence_mask");
    add_rtc_nw_int("rtc_network_impulsive_mask_candidate_available",
                   "1 if impulsive_coincidence_mask found a candidate for this RTC network block, else 0");
    add_rtc_nw_int("rtc_network_impulsive_mask_local_trigger",
                   "1 if the selected impulsive candidate satisfied the within-network trigger thresholds, else 0");
    add_rtc_nw_int("rtc_network_impulsive_mask_cross_network_trigger",
                   "1 if the selected impulsive candidate satisfied a cross-network alignment trigger, else 0");
    add_rtc_nw_int("rtc_network_impulsive_mask_high_score_override_trigger",
                   "1 if the selected impulsive candidate satisfied the looser high-score cross-network override, else 0");
    add_rtc_nw_int("rtc_network_impulsive_mask_rejected_max_fraction",
                   "1 if the selected impulsive candidate was rejected only because its proposed flagged fraction exceeded the configured limit");
    add_rtc_nw_int("rtc_network_impulsive_mask_candidate_center_sample",
                   "center sample of the selected impulsive candidate before any cross-network recentering; -2147483647 means unavailable");
    add_rtc_nw_int("rtc_network_impulsive_mask_cluster_center_sample",
                   "median aligned sample of the selected cross-network impulsive cluster; -2147483647 means unavailable");
    add_rtc_nw_int("rtc_network_impulsive_mask_cluster_network_count",
                   "number of distinct networks participating in the selected impulsive candidate cluster");
    add_rtc_nw_int("rtc_network_impulsive_mask_cluster_active_count",
                   "number of detector-level impulsive events in the selected within-network cluster");
    add_rtc_nw_int("rtc_network_impulsive_mask_total_active_count",
                   "total number of detector-level impulsive events above threshold in the selected network block");
    add_rtc_nw_double("rtc_network_impulsive_mask_cluster_peak_score",
                      "maximum impulsive-event score found within the selected cross-network impulsive cluster");
    add_rtc_nw_double("rtc_network_impulsive_mask_override_score",
                      "score used by the high-score override path after combining the selected cluster peak with the strongest candidate score seen in participating networks");
    add_rtc_nw_int("rtc_network_impulsive_mask_override_uses_network_peak",
                   "1 if rtc_network_impulsive_mask_override_score came from a participating network's strongest candidate rather than the selected cluster peak");
    add_rtc_nw_double("rtc_network_impulsive_mask_proposed_flagged_fraction",
                      "fraction of previously good detector-samples that the selected impulsive mask window would newly flag before any rejection");

    if (rtcproc.impulsive_capture.enabled) {
        const auto n_slots =
            static_cast<std::size_t>(std::max<Eigen::Index>(rtcproc.impulsive_capture.max_events_per_network, 1));
        const auto snippet_pre =
            static_cast<std::size_t>(std::max(0.0, std::round(rtcproc.impulsive_capture.snippet_pre_window_sec * rtc_fsmp)));
        const auto snippet_post =
            static_cast<std::size_t>(std::max(0.0, std::round(rtcproc.impulsive_capture.snippet_post_window_sec * rtc_fsmp)));
        const auto n_snippet = snippet_pre + snippet_post + 1;
        netCDF::NcDim n_rtc_impulsive_slots_dim = fo.addDim("n_rtc_impulsive_slots", n_slots);
        netCDF::NcDim n_rtc_impulsive_samples_dim = fo.addDim("n_rtc_impulsive_samples", n_snippet);

        netCDF::NcVar offset_v = fo.addVar("rtc_impulsive_snippet_offset_samples", netCDF::ncInt, n_rtc_impulsive_samples_dim);
        offset_v.putAtt("units", "samples");
        offset_v.putAtt("comment", "sample offsets relative to rtc_impulsive_slot_event_sample");
        std::vector<int> offsets(n_snippet, fill_int);
        for (std::size_t i = 0; i < n_snippet; ++i) {
            offsets[i] = static_cast<int>(i) - static_cast<int>(snippet_pre);
        }
        offset_v.putVar(offsets.data());

        std::vector<netCDF::NcDim> rtc_impulsive_slot_dims = {n_scans_dim, n_nws_rtcdiag_dim, n_rtc_impulsive_slots_dim};
        std::vector<netCDF::NcDim> rtc_impulsive_snippet_dims = {n_scans_dim, n_nws_rtcdiag_dim, n_rtc_impulsive_slots_dim, n_rtc_impulsive_samples_dim};
        const std::vector<std::size_t> rtc_impulsive_slot_chunks = {1, TULA_SIZET(calib.n_nws), n_slots};
        const std::vector<std::size_t> rtc_impulsive_snippet_chunks = {1, TULA_SIZET(calib.n_nws), n_slots, n_snippet};

        auto add_rtc_imp_slot_double = [&](const std::string &name, const std::string &comment) {
            netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, rtc_impulsive_slot_dims);
            v.putAtt("units", "N/A");
            v.putAtt("comment", comment);
            set_netcdf_chunking_and_compression(v, rtc_impulsive_slot_chunks, 1);
            std::vector<double> init(static_cast<std::size_t>(n_scans) *
                                     static_cast<std::size_t>(calib.n_nws) * n_slots, fill_double);
            v.putVar(init.data());
        };
        auto add_rtc_imp_slot_int = [&](const std::string &name, const std::string &comment) {
            netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, rtc_impulsive_slot_dims);
            v.putAtt("units", "N/A");
            v.putAtt("comment", comment);
            set_netcdf_chunking_and_compression(v, rtc_impulsive_slot_chunks, 1);
            std::vector<int> init(static_cast<std::size_t>(n_scans) *
                                  static_cast<std::size_t>(calib.n_nws) * n_slots, fill_int);
            v.putVar(init.data());
        };
        auto add_rtc_imp_snip_double = [&](const std::string &name, const std::string &comment) {
            netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, rtc_impulsive_snippet_dims);
            v.putAtt("units", "N/A");
            v.putAtt("comment", comment);
            set_netcdf_chunking_and_compression(v, rtc_impulsive_snippet_chunks, 1);
            std::vector<double> init(static_cast<std::size_t>(n_scans) *
                                     static_cast<std::size_t>(calib.n_nws) * n_slots * n_snippet, fill_double);
            v.putVar(init.data());
        };
        auto add_rtc_imp_snip_int = [&](const std::string &name, const std::string &comment) {
            netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, rtc_impulsive_snippet_dims);
            v.putAtt("units", "N/A");
            v.putAtt("comment", comment);
            set_netcdf_chunking_and_compression(v, rtc_impulsive_snippet_chunks, 1);
            std::vector<int> init(static_cast<std::size_t>(n_scans) *
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
                                "absolute robust-z peak of a captured impulsive RTC event");
        add_rtc_imp_slot_double("rtc_impulsive_slot_peak_delta_abs_z",
                                "absolute delta robust-z peak of a captured impulsive RTC event");
        add_rtc_imp_slot_double("rtc_impulsive_slot_added_flagged_frac",
                                "newly added flagged-sample fraction for the captured detector");
        add_rtc_imp_slot_int("rtc_impulsive_slot_raw_exceed_count",
                             "native raw-threshold exceedance count for the captured detector");
        add_rtc_imp_slot_int("rtc_impulsive_slot_local_raw_candidate_count",
                             "compact-raw local candidate count for the captured detector");
        add_rtc_imp_slot_int("rtc_impulsive_slot_local_raw_accepted_event_count",
                             "accepted compact-raw local-event count for the captured detector");
        add_rtc_imp_slot_int("rtc_impulsive_slot_local_flagged_sample_count",
                             "samples flagged by accepted compact-raw local events for the captured detector");
        add_rtc_imp_slot_int("rtc_impulsive_slot_local_exceed_count",
                             "legacy alias for rtc_impulsive_slot_local_flagged_sample_count");
        add_rtc_imp_slot_int("rtc_impulsive_slot_local_raw_reject_count",
                             "rejected compact-raw local-event count for the captured detector");
        add_rtc_imp_slot_int("rtc_impulsive_slot_delta_spike_count",
                             "native delta-spike count for the captured detector");
        add_rtc_imp_slot_int("rtc_impulsive_slot_local_delta_candidate_count",
                             "compact-delta local candidate count for the captured detector");
        add_rtc_imp_slot_int("rtc_impulsive_slot_local_delta_accepted_event_count",
                             "accepted compact-delta local-event count for the captured detector");
        add_rtc_imp_slot_int("rtc_impulsive_slot_local_delta_exceed_count",
                             "legacy alias for rtc_impulsive_slot_local_delta_accepted_event_count");
        add_rtc_imp_slot_int("rtc_impulsive_slot_local_delta_reject_count",
                             "rejected compact-delta local-event count for the captured detector");
        add_rtc_imp_snip_double("rtc_impulsive_slot_snippet_z",
                                "standardized RTC snippet around each captured impulsive event");
        add_rtc_imp_snip_int("rtc_impulsive_slot_snippet_flag",
                             "RTC flag state for each sample in the captured impulsive-event snippet");
    }

    });
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

    write_netcdf_atomic(stats_filename + ".nc", [&](netCDF::NcFile &fo) {

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
    });
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::run_wiener_filter(map_buffer_t &mb) {
    const auto n_maps_local = static_cast<std::size_t>(mb.signal.size());
    mb.edge_guard_applied.assign(n_maps_local, 0);
    mb.edge_guard_support_radius_pix.assign(n_maps_local, 0);
    mb.edge_guard_science_npix.assign(n_maps_local, 0);
    mb.edge_guard_support_npix.assign(n_maps_local, 0);
    mb.edge_guard_guardband_npix.assign(n_maps_local, 0);
    mb.edge_guard_weight_threshold.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
    mb.edge_guard_hits_threshold.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
    mb.edge_guard_background_level.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
    mb.edge_guard_science_frac.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
    mb.edge_guard_support_frac.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
    mb.edge_guard_guardband_rms_pre.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
    mb.edge_guard_guardband_rms_post.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
    mb.edge_guard_exterior_rms_pre.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
    mb.edge_guard_exterior_rms_post.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
    mb.edge_guard_exterior_max_abs_pre.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
    mb.edge_guard_exterior_max_abs_post.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
    mb.edge_guard_window.resize(n_maps_local);

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
        if (!pmb->noise.empty() && !n_io->empty()) {
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

            if (write_filtered_maps_partial && (run_noise_products || wiener_filter.normalize_error)) {
                const bool apply_scale = apply_empirical_noise_weights || wiener_filter.normalize_error;
                logger->info("calculating empirical noise products for {} map {}/{}",
                             map_label, i + 1, n_maps);
                mb.calc_noise_products(i, apply_scale);
                if (i < mb.noise_weight_median_ratio.size()) {
                    logger->info("noise products: median(w_formal*var)={:.4g} scale={:.4g} noise_s2n_sigma={:.4g}",
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
