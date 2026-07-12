#include <citlali/core/config/calibration_config.h>
#include <citlali/core/config/reduction_config.h>
#include <citlali/core/config/reduction_config_validation.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>
#include <citlali/core/pipeline/config_diagnostics_state.h>
#include <citlali/core/pipeline/output_path_state.h>
#include <citlali/core/cli/config_loading.h>
#include <citlali/core/cli/reduction_runtime.h>
#include <citlali/core/cli/runtime_setup.h>
#include <citlali/core/cli/tod_processor_selection.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/beammap_source_flux_config.h>
#include <citlali/core/pipeline/fruit_loop_paths.h>
#include <citlali/core/pipeline/iteration_lifecycle.h>
#include <citlali/core/pipeline/map_geometry.h>
#include <citlali/core/pipeline/observation_execution.h>
#include <citlali/core/pipeline/observation_preflight.h>
#include <citlali/core/pipeline/output_layout.h>
#include <citlali/core/pipeline/output_netcdf_metadata.h>
#include <citlali/core/pipeline/phdu_reduction_config.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/pipeline/processed_clean_config_read.h>
#include <citlali/core/pipeline/processed_clean_resolution.h>
#include <citlali/core/pipeline/processed_timestream_config_serialization.h>
#include <citlali/core/pipeline/processed_timestream_execution_plan.h>
#include <citlali/core/pipeline/processed_weighting_config_read.h>
#include <citlali/core/pipeline/processed_weighting_resolution.h>
#include <citlali/core/pipeline/runtime_provenance_output.h>
#include <citlali/core/pipeline/source_protection_activation.h>
#include <citlali/core/pipeline/timestream_output_provenance.h>
#include <citlali/core/pipeline/timestream_config_mirror.h>
#include <citlali/core/pipeline/timestream_config_adapter_processed.h>
#include <citlali/core/pipeline/timestream_run_context.h>
#include <citlali/core/pipeline/tod_output_state.h>

#include <gtest/gtest.h>

#include <functional>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace {

#define EXPECT_TEMPLATE_TRUE(...) EXPECT_TRUE((__VA_ARGS__))
#define EXPECT_TEMPLATE_FALSE(...) EXPECT_FALSE((__VA_ARGS__))

struct FakeLogger {
    int error_calls = 0;
    int info_calls = 0;
    int debug_calls = 0;
    int warn_calls = 0;

    template <class... Args>
    void error(const char *, Args &&...) { ++error_calls; }

    template <class... Args>
    void info(const char *, Args &&...) { ++info_calls; }

    template <class... Args>
    void debug(const char *, Args &&...) { ++debug_calls; }

    template <class... Args>
    void warn(const char *, Args &&...) {
        ++warn_calls;
    }
};

struct FakeAptColumn {
    double value = 1.0;

    FakeAptColumn &array() { return *this; }

    FakeAptColumn &operator*=(double factor) {
        value *= factor;
        return *this;
    }
};

struct FakeTelHeaderValue {
    double value = 0.0;
    bool set = false;

    void setConstant(double new_value) {
        value = new_value;
        set = true;
    }
};

struct FakeTelTime {
    std::vector<double> values = {0.0, 1.0};

    double operator()(std::size_t index) const { return values.at(index); }
    std::size_t size() const { return values.size(); }
};

struct FakeRawObsMeta {
    double fsmp = 0.0;
    int obsid = 0;

    template <class T>
    T get_typed(const std::string &key) const {
        if (key == "fsmp") {
            return static_cast<T>(fsmp);
        }
        if (key == "obsid") {
            return static_cast<T>(obsid);
        }
        return T{};
    }
};

struct FakeCalib {
    std::map<std::string, FakeAptColumn> apt;
    std::string ignore_hwpr = "false";
    bool run_hwpr = true;
    int get_apt_calls = 0;
    std::string loaded_apt_path;
    std::vector<std::string> loaded_raw_filenames;
    std::vector<std::string> loaded_interfaces;
    bool loaded_hwpr = false;
    std::string loaded_hwpr_filepath;
    bool loaded_hwpr_sim_obs = false;
    int calc_flux_calibration_calls = 0;
    std::string loaded_flux_units;
    double loaded_flux_pixel_size_rad = 0.0;

    void get_apt(const std::string &apt_path,
                 const std::vector<std::string> &raw_filenames,
                 const std::vector<std::string> &interfaces) {
        ++get_apt_calls;
        loaded_apt_path = apt_path;
        loaded_raw_filenames = raw_filenames;
        loaded_interfaces = interfaces;
    }

    void get_hwpr(const std::string &filepath, bool sim_obs) {
        loaded_hwpr = true;
        loaded_hwpr_filepath = filepath;
        loaded_hwpr_sim_obs = sim_obs;
    }

    void calc_flux_calibration(const std::string &units,
                               double pixel_size_rad) {
        ++calc_flux_calibration_calls;
        loaded_flux_units = units;
        loaded_flux_pixel_size_rad = pixel_size_rad;
    }
};

struct FakeEngine {
    citlali::config::ReductionConfig typed_config = [] {
        citlali::config::ReductionConfig config;
        config.runtime.verbose = false;
        config.runtime.interp_over_gaps = false;
        config.runtime.n_threads = 4;
        config.mapmaking.method = citlali::config::MapMethod::jinc;
        config.mapmaking.grouping = citlali::config::MapGrouping::array;
        config.noise.enabled = true;
        return config;
    }();
    citlali::config::RuntimeConfigProvenance runtime_config_provenance =
        citlali::config::make_runtime_config_provenance(typed_config.runtime,
                                                        false);
    citlali::pipeline::TimestreamAlignmentState alignment = [] {
        citlali::pipeline::TimestreamAlignmentState state;
        state.start_indices = {7};
        state.end_indices = {9};
        state.hwpr_start_index = -1;
        state.hwpr_end_index = -1;
        return state;
    }();
    citlali::pipeline::OutputPathState output_paths = [] {
        citlali::pipeline::OutputPathState paths;
        paths.redu_dir_name = "/tmp/redu01";
        paths.redu_dir_num = 1;
        return paths;
    }();
    struct {
        std::string obsnum;
    } observation_identity;
    std::string redu_type = "science";
    std::string redu_dir_name = "/tmp/redu01";
    std::string output_dir = "/tmp";
    int redu_dir_num = 1;
    std::string obsnum_dir_name;
    bool run_coadd = false;
    bool run_map_filter = false;
    bool run_mapmaking = true;
    bool run_tod = true;
    bool run_noise = true;
    bool run_noise_products = true;
    bool run_source_finder = false;
    bool write_filtered_maps_partial = false;
    bool apply_empirical_noise_weights = false;
    struct {
        int fruit_iter = 0;
    } iteration;
    int configure_map_pixel_contribution_targets_calls = 0;
    std::string last_map_pixel_contribution_target;
    int create_obs_map_files_calls = 0;
    int output_calls = 0;
    int run_wiener_filter_calls = 0;
    int find_sources_calls = 0;
    int fit_maps_calls = 0;
    int setup_calls = 0;
    int pipeline_calls = 0;
    int get_astrometry_config_calls = 0;
    int get_photometry_config_calls = 0;
    int get_citlali_config_calls = 0;
    std::string loaded_astrometry_config;
    std::string loaded_photometry_config;
    citlali::pipeline::ConfigDiagnosticsState config_diagnostics;
    struct {
        std::vector<std::string> date_obs;
    } observation_dates;
    int write_learning_summary_calls = 0;

    struct {
        std::vector<std::string> obsnums;
        std::vector<double> crval_config = {0.0, 0.0};
        std::string sig_unit = "mJy/beam";
        double exposure_time = 0.0;
        double cov_cut = 0.0;
        double pixel_size_rad = 0.0;
        int calc_noise_products_calls = 0;
        int calc_map_psd_calls = 0;
        int calc_map_hist_calls = 0;
        int calc_median_err_calls = 0;
        int calc_median_rms_calls = 0;
        bool last_apply_empirical_noise_weights = false;

        void calc_noise_products(bool apply_empirical_noise_weights) {
            ++calc_noise_products_calls;
            last_apply_empirical_noise_weights =
                apply_empirical_noise_weights;
        }

        void calc_map_psd() { ++calc_map_psd_calls; }
        void calc_map_hist() { ++calc_map_hist_calls; }
        void calc_median_err() { ++calc_median_err_calls; }
        void calc_median_rms() { ++calc_median_rms_calls; }
    } omb;

    struct {
        bool normalize_error = false;
    } wiener_filter;

    struct {
        std::vector<std::string> obsnums;
        double exposure_time = 0.0;
        int normalize_maps_calls = 0;
        int normalize_polarized_maps_calls = 0;
        int calc_noise_products_calls = 0;
        int calc_map_psd_calls = 0;
        int calc_map_hist_calls = 0;
        int calc_median_err_calls = 0;
        int calc_median_rms_calls = 0;
        bool last_apply_empirical_noise_weights = false;

        void normalize_maps() { ++normalize_maps_calls; }
        void normalize_polarized_maps() { ++normalize_polarized_maps_calls; }

        void calc_noise_products(bool apply_empirical_noise_weights) {
            ++calc_noise_products_calls;
            last_apply_empirical_noise_weights =
                apply_empirical_noise_weights;
        }

        void calc_map_psd() { ++calc_map_psd_calls; }
        void calc_map_hist() { ++calc_map_hist_calls; }
        void calc_median_err() { ++calc_median_err_calls; }
        void calc_median_rms() { ++calc_median_rms_calls; }
    } cmb;

    FakeCalib calib;

    struct {
        double fsmp = 100.0;
        double d_fsmp = -1.0;
        bool sim_obs = false;
        std::string pixel_axes = "pixel_axes";
        int get_tel_data_calls = 0;
        int calc_tan_pointing_calls = 0;
        int calc_scan_indices_calls = 0;
        std::string scan_chunk_mode;
        double scan_chunk_value = 0.0;
        bool scan_force_chunk = false;
        std::string loaded_tel_path;
        std::map<std::string, FakeTelHeaderValue> tel_header;
        std::map<std::string, FakeTelTime> tel_data;

        template <class ChunkingConfig>
        void get_tel_data(const std::string &tel_path,
                          const ChunkingConfig &chunking) {
            ++get_tel_data_calls;
            loaded_tel_path = tel_path;
            scan_chunk_mode = chunking.mode;
            scan_chunk_value = chunking.value;
            scan_force_chunk = chunking.force;
        }

        void calc_tan_pointing() { ++calc_tan_pointing_calls; }
        template <class ChunkingConfig>
        void calc_scan_indices(const ChunkingConfig &chunking) {
            ++calc_scan_indices_calls;
            scan_chunk_mode = chunking.mode;
            scan_chunk_value = chunking.value;
            scan_force_chunk = chunking.force;
        }
    } telescope;

    struct {
        bool run_downsample = false;
        bool run_polarization = false;
        struct {
            int factor = 1;
            double downsampled_freq_Hz = 0.0;
        } downsampler;
        struct {
            double freq_high_Hz = 0.0;
        } filter;
    } rtcproc;

    struct {
        bool run_fruit_loops = false;
        int fruit_loops_iters = 3;
        bool save_all_iters = false;
        std::string fruit_loops_path = "null";
        std::string fruit_loops_type = "obsnum/raw";
        std::string fruit_loops_interp_mode = "bilinear";
        struct {
            double cov_cut = 0.0;
            std::vector<double> signal;
        } tod_mb;
        bool fruit_loops_recompute_weights_after_addback = true;
        int load_mb_calls = 0;
        int begin_weight_validation_iter = -1;
        int finalize_weight_validation_iter = -1;
        std::string loaded_filepath;
        std::string loaded_noise_filepath;

        template <class Calib, class PixelAxes>
        void load_mb(const std::string &filepath,
                     const std::string &noise_filepath, Calib &,
                     const std::string &, const PixelAxes &, double) {
            ++load_mb_calls;
            loaded_filepath = filepath;
            loaded_noise_filepath = noise_filepath;
        }

        void begin_weight_validation_iteration(int iter) {
            begin_weight_validation_iter = iter;
        }

        void finalize_weight_validation_iteration(int iter) {
            finalize_weight_validation_iter = iter;
        }
    } ptcproc;

    struct {
        bool enabled = false;
        bool diagnostics = false;
        int begin_calls = 0;
        int begin_iter = -1;
        int finalize_calls = 0;
        int finalize_iter = -1;
        bool source_model_available = false;
        std::string redu_type;

        void begin_iteration(int iter, bool source_available,
                             citlali::config::ReductionType type) {
            ++begin_calls;
            begin_iter = iter;
            source_model_available = source_available;
            redu_type = std::string(citlali::config::to_string(type));
        }

        void finalize_iteration(int iter) {
            ++finalize_calls;
            finalize_iter = iter;
        }

        bool is_enabled() const { return enabled; }
        bool diagnostics_enabled() const { return diagnostics; }
        std::string summary_string() const { return "fake summary"; }
    } learning;

    template <class MapBuffer>
    void configure_map_pixel_contribution_targets(
        MapBuffer &, const std::string &target) {
        ++configure_map_pixel_contribution_targets_calls;
        last_map_pixel_contribution_target = target;
    }

    void create_obs_map_files() { ++create_obs_map_files_calls; }

    template <auto MapType>
    void output() {
        ++output_calls;
    }

    template <auto MapType, class MapBuffer>
    void run_wiener_filter(MapBuffer &) {
        ++run_wiener_filter_calls;
    }

    template <auto MapType, class MapBuffer>
    void find_sources(MapBuffer &) {
        ++find_sources_calls;
    }

    void fit_maps() { ++fit_maps_calls; }

    void setup() { ++setup_calls; }

    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, const RawObs &) {
        ++pipeline_calls;
    }

    void get_astrometry_config(const std::string &config) {
        ++get_astrometry_config_calls;
        loaded_astrometry_config = config;
    }

    void get_photometry_config(const std::string &config) {
        ++get_photometry_config_calls;
        loaded_photometry_config = config;
    }

    template <class Config>
    void get_citlali_config(Config &) {
        ++get_citlali_config_calls;
        runtime_config_provenance =
            citlali::config::make_runtime_config_provenance(
                typed_config.runtime, false);
    }

    void write_learning_summary() { ++write_learning_summary_calls; }
};

struct FakeFlxscaleCorrection {
    double factor = 1.0;
    double value() const { return factor; }
};

struct FakeHwpData {
    std::string path = "hwpr.nc";
    std::string filepath() const { return path; }
};

struct FakeRawObs {
    struct FakeArrayPropTable {
        std::string path = "apt.ecsv";
        std::string filepath() const { return path; }
    };

    struct FakeDataItem {
        std::string path;
        std::string iface;
        std::string filepath() const { return path; }
        std::string interface() const { return iface; }
    };

    struct FakeConfigInfo {
        std::string value = "config";
        const std::string &config() const { return value; }
    };

    struct FakeTelData {
        std::string path = "tel.nc";
        std::string filepath() const { return path; }
    };

    const FakeFlxscaleCorrection *correction = nullptr;
    std::string obs_name = "fake_obs";
    FakeArrayPropTable apt;
    std::vector<FakeDataItem> kids_items = {
        {"toltec0.nc", "toltec0"},
        {"toltec1.nc", "toltec1"},
    };
    FakeConfigInfo astrometry;
    FakeConfigInfo photometry;
    FakeTelData tel;
    std::optional<FakeHwpData> hwp;

    const FakeArrayPropTable &array_prop_table() const { return apt; }

    const std::vector<FakeDataItem> &kidsdata() const { return kids_items; }

    const FakeConfigInfo &astrometry_calib_info() const { return astrometry; }

    const FakeConfigInfo &photometry_calib_info() const { return photometry; }

    const FakeTelData &teldata() const { return tel; }

    const FakeFlxscaleCorrection *flxscale_correction() const {
        return correction;
    }

    const std::string &name() const { return obs_name; }

    std::optional<FakeHwpData> hwpdata() const { return hwp; }
};

struct FakeReferenceWrappedRawObs : FakeRawObs {
    std::vector<std::reference_wrapper<const FakeDataItem>> kidsdata() const {
        std::vector<std::reference_wrapper<const FakeDataItem>> refs;
        for (const auto &item : kids_items) {
            refs.push_back(std::cref(item));
        }
        return refs;
    }
};

struct FakeKidsProc {
    int get_rawobs_meta_calls = 0;
    int loaded_config_value = 0;
    std::vector<FakeRawObsMeta> meta = {
        {100.0, 101},
        {122.0, 102},
    };

    struct Config {
        int value = 0;
    };

    static FakeKidsProc from_config(const Config &config) {
        FakeKidsProc kidsproc;
        kidsproc.loaded_config_value = config.value;
        return kidsproc;
    }

    std::vector<FakeRawObsMeta> get_rawobs_meta(const FakeRawObs &) {
        ++get_rawobs_meta_calls;
        return meta;
    }
};

struct FakeFailingKidsProc : FakeKidsProc {
    static FakeFailingKidsProc from_config(const Config &) {
        return {};
    }

    std::vector<FakeRawObsMeta> get_rawobs_meta(const FakeRawObs &) {
        throw std::runtime_error("No such file or directory");
    }
};

struct FakeCitlaliConfig {
    int get_config_calls = 0;
    std::string requested_key;

    FakeKidsProc::Config get_config(const std::string &key) {
        ++get_config_calls;
        requested_key = key;
        return {42};
    }
};

struct FakeConfigNode {
    std::string value;

    template <class T>
    T as() const {
        return T{value};
    }
};

struct FakeRuntimeConfig {
    std::vector<FakeConfigNode> config_nodes;

    std::vector<FakeConfigNode> get_node(const std::string &key) const {
        if (key == "config_file") {
            return config_nodes;
        }
        return {};
    }
};

struct FakeLoadedConfig {
    std::vector<std::string> loaded_paths;
};

struct FakeGeometryTodProc {
    using map_extent_t = int;
    using map_coord_t = double;
};

struct FakeTodConfig {
    int value = 0;
    bool has_reduction_type = true;
    std::string reduction_type = "science";

    template <class Key>
    bool has(const Key &) const {
        return has_reduction_type;
    }

    template <class Key>
    std::string get_str(const Key &) {
        return reduction_type;
    }
};

struct FakeScienceTodProc {
    int loaded_value = 0;

    static FakeScienceTodProc from_config(FakeTodConfig &config) {
        return {config.value};
    }
};

struct FakePointingTodProc {
    int loaded_value = 0;

    static FakePointingTodProc from_config(FakeTodConfig &config) {
        return {config.value};
    }
};

struct FakeBeammapTodProc {
    int loaded_value = 0;

    static FakeBeammapTodProc from_config(FakeTodConfig &config) {
        return {config.value};
    }
};

struct FakeIOCoordinator {
    std::vector<FakeRawObs> raw_inputs;

    const std::vector<FakeRawObs> &inputs() const { return raw_inputs; }

    std::size_t n_inputs() const { return raw_inputs.size(); }
};

struct FakeExecutionEngine {
    citlali::config::ReductionConfig typed_config;
    bool run_tod = true;
    int setup_calls = 0;
    int pipeline_calls = 0;
    std::vector<std::string> event_order;

    void setup() {
        ++setup_calls;
        event_order.push_back("setup");
    }

    void pipeline(FakeKidsProc &, const FakeRawObs &) {
        ++pipeline_calls;
        event_order.push_back("pipeline");
    }
};

struct FakeIterationPtcProc {
    bool run_fruit_loops = false;
    bool save_all_iters = false;
    int fruit_loops_iters = 3;
    std::string fruit_loops_path = "null";
    int begin_weight_validation_iter = -1;
    int finalize_weight_validation_iter = -1;

    void begin_weight_validation_iteration(int iter) {
        begin_weight_validation_iter = iter;
    }

    void finalize_weight_validation_iteration(int iter) {
        finalize_weight_validation_iter = iter;
    }
};

struct FakeFruitLoopsAdapterPtcProc {
    bool run_fruit_loops = false;
    bool fruit_loops_recompute_weights_after_addback = false;
    bool save_all_iters = false;
    std::string fruit_loops_path;
    std::string fruit_loops_type;
    std::string fruit_mode;
    double fruit_loops_sig2noise = 0.0;
    Eigen::VectorXd fruit_loops_flux;
    double fruit_loops_peak_fraction_limit = 0.0;
    double fruit_loops_local_snr_floor = 0.0;
    double fruit_loops_local_sigma_inner_radius_arcsec = 0.0;
    double fruit_loops_local_sigma_outer_radius_arcsec = 0.0;
    double fruit_loops_local_sigma_inner_fwhm = 0.0;
    double fruit_loops_local_sigma_outer_fwhm = 0.0;
    double fruit_loops_local_sigma_edge_guard_arcsec = 0.0;
    int fruit_loops_local_sigma_min_pixels = 0;
    double fruit_loops_adaptive_support_radius_arcsec = 0.0;
    double fruit_loops_adaptive_support_radius_fwhm = 0.0;
    bool fruit_loops_weight_feedback_enabled = false;
    std::string fruit_loops_weight_feedback_reference;
    double fruit_loops_weight_feedback_low_relative_weight = 0.0;
    double fruit_loops_weight_feedback_high_relative_weight = 0.0;
    double fruit_loops_center_keep_radius_arcsec = 0.0;
    std::string fruit_loops_interp_mode_override;
    bool fruit_loops_legacy_center = false;
    int fruit_loops_iters = 0;
};

struct FakeProcessedCleanAdapterPtcProc {
    bool run_clean = false;
    double mask_radius_arcsec = 0.0;
    struct Cleaner {
        std::vector<std::string> grouping;
        double tau = 0.0;
        struct StandardPca {
            bool enabled = false;
        } standard_pca;
        double stddev_limit = 0.0;
        int n_calc = 0;
        std::map<int, Eigen::VectorXI> n_eig_to_cut;
        struct CorrGrouping {
            bool enabled = false;
            std::string metric;
            double corr_min = 0.0;
            int min_overlap = 0;
            double min_good_frac = 0.0;
            int min_group_size = 0;
            int max_samples = 0;
            bool clean_residual = false;
        } corr_grouping;
        struct NullModel {
            bool enabled = false;
            int n_surrogates = 0;
            double quantile = 0.0;
            double min_good_frac = 0.0;
            int max_modes = 0;
            int max_samples = 0;
            std::uint32_t seed = 0;
            std::vector<std::string> grouping;
        } null_model;
        struct MarchenkoPastur {
            bool enabled = false;
            double min_good_frac = 0.0;
            int max_modes = 0;
            int max_samples = 0;
            double band_low_Hz = 0.0;
            double band_high_Hz = 0.0;
            double clip_z = 0.0;
            double bulk_keep_frac = 0.0;
            int q_grid_size = 0;
            std::vector<std::string> grouping;
        } marchenko_pastur;
        struct AdaptiveSelector {
            bool enabled = false;
            double min_good_frac = 0.0;
            int max_det = 0;
            int max_samples = 0;
            int max_pairs = 0;
            std::uint32_t seed = 0;
            double clip_z = 0.0;
            double low_weight = 0.0;
            double tail_weight = 0.0;
            double topmode_weight = 0.0;
            double reg_weight = 0.0;
            std::array<double, 2> low_band_Hz{};
            std::array<double, 2> mid_band_Hz{};
            std::vector<int> candidate_offsets;
            std::vector<std::string> grouping;
            bool log_candidates = false;
        } adaptive_selector;
    } cleaner;
};

struct FakeProcessedAdapterPtcProc {
    std::string weighting_type;
    double source_mask_radius_arcsec = 0.0;
    double hybrid_correction_min_factor = 0.0;
    double hybrid_correction_max_factor = 0.0;
    double med_weight_factor = 0.0;
    double lower_weight_factor = 0.0;
    double upper_weight_factor = 0.0;
    double lower_inv_var_factor = 0.0;
    double upper_inv_var_factor = 0.0;
    struct WeightValidation {
        bool enabled = false;
        int accumulation_iters = 0, apply_start_iter = 0, min_valid_scans = 0;
        double min_factor = 0.0, unvalidated_factor = 0.0;
        bool require_fruitloops_model = false, transient_ratio_enabled = false;
        double ratio_power = 0.0, transient_ratio_power = 0.0;
        bool upward_enabled = false;
        double upward_max_factor = 0.0, upward_power = 0.0;
        double upward_min_base_factor = 0.0;
        bool upward_require_atmospheric = false;
        double upward_min_atmospheric_factor = 0.0;
        bool atmospheric_correlation_enabled = false;
        std::string atmospheric_grouping;
        int atmospheric_min_detectors = 0;
        double atmospheric_ref = 0.0, atmospheric_span = 0.0;
        double atmospheric_power = 0.0, min_good_frac = 0.0;
        int min_overlap = 0, max_samples = 0;
        bool high_weight_validation_enabled = false;
        bool high_weight_apply_caps = false;
        std::string high_weight_grouping;
        int high_weight_min_group_detectors = 0;
        double high_weight_log_robust_z = 0.0;
        double high_weight_max_median_factor = 0.0;
        double high_weight_cap_median_factor = 0.0;
        double high_weight_min_validated_factor = 0.0;
    } weight_validation;
    struct PenaltyTerm {
        bool enabled = false;
        double ref = 0.0, span = 0.0, weight = 0.0;
    };
    struct PenaltyBand : PenaltyTerm {
        double low_min_Hz = 0.0, low_max_Hz = 0.0;
        double mid_min_Hz = 0.0, mid_max_Hz = 0.0;
    };
    struct WeightPenalty {
        bool enabled = false;
        double min_good_frac = 0.0;
        int min_overlap = 0, max_samples = 0, max_pairs = 0;
        std::uint32_t seed = 0;
        double floor = 0.0, exponent = 0.0;
        PenaltyTerm pair_corr, cm_el_corr;
        PenaltyBand cm_low_mid_ratio;
    } weight_corr_penalty;
    struct BusyRow {
        bool enabled = false, require_busy_veto = false;
        int min_candidate_clusters = 0;
        double min_max_unflagged_residual_z = 0.0, factor = 0.0;
    } busy_row_suppression;
    struct SecondPass {
        bool enabled = false;
        double min_spike_sigma = 0.0, min_good_frac = 0.0;
        double baseline_window_sec = 0.0, sigma_scale = 0.0;
        double delta_sigma_scale = 0.0, raw_candidate_rel_sigma_scale = 0.0;
        double raw_window_sec = 0.0, raw_half_peak_frac = 0.0;
        double raw_max_width_sec = 0.0, delta_window_sec = 0.0;
        double delta_half_peak_frac = 0.0, delta_max_width_sec = 0.0;
        double max_step_shift_z = 0.0, high_score_event_override = 0.0;
        double merge_within_detector_sec = 0.0, cluster_events_sec = 0.0;
        int min_cluster_detectors = 0;
        double high_score_cluster_override = 0.0;
        int max_auto_flag_clusters_per_network = 0;
        bool selective_busy_network_acceptance_enabled = false;
        bool source_protection_config_enabled = false;
        bool source_protection_enabled = true;
        double source_protection_radius_arcsec = 0.0;
    } second_pass_local;
};

struct FakeReductionLearning {
    bool enabled = false;
    bool diagnostics = false;
    int begin_calls = 0;
    int begin_iter = -1;
    int finalize_calls = 0;
    int finalize_iter = -1;
    bool source_model_available = false;
    std::string redu_type;

    void begin_iteration(int iter, bool source_available,
                         citlali::config::ReductionType type) {
        ++begin_calls;
        begin_iter = iter;
        source_model_available = source_available;
        redu_type = std::string(citlali::config::to_string(type));
    }

    void finalize_iteration(int iter) {
        ++finalize_calls;
        finalize_iter = iter;
    }

    bool is_enabled() const { return enabled; }
    bool diagnostics_enabled() const { return diagnostics; }
    std::string summary_string() const { return "fake summary"; }
};

struct FakeIterationEngine {
    citlali::config::ReductionConfig typed_config;
    citlali::config::RuntimeConfigProvenance runtime_config_provenance =
        citlali::config::make_runtime_config_provenance(typed_config.runtime,
                                                        false);
    citlali::pipeline::OutputPathState output_paths = [] {
        citlali::pipeline::OutputPathState paths;
        paths.redu_dir_name = "/tmp/redu01";
        return paths;
    }();
    struct {
        int fruit_iter = 0;
    } iteration;
    std::string redu_type = "science";
    std::string redu_dir_name = "/tmp/redu01";
    FakeIterationPtcProc ptcproc;
    FakeReductionLearning learning;
    int write_learning_summary_calls = 0;

    void write_learning_summary() { ++write_learning_summary_calls; }
};

struct FakeIterationTodProc {
    FakeIterationEngine engine_state;
    int make_index_file_calls = 0;
    std::string indexed_path;

    FakeIterationEngine &engine() { return engine_state; }

    void make_index_file(const std::string &path) {
        ++make_index_file_calls;
        indexed_path = path;
    }
};

struct FakeReductionIterationEngine {
    citlali::config::ReductionConfig typed_config = [] {
        citlali::config::ReductionConfig config;
        config.coadd.enabled = true;
        config.noise.enabled = true;
        return config;
    }();
    citlali::config::RuntimeConfigProvenance runtime_config_provenance =
        citlali::config::make_runtime_config_provenance(typed_config.runtime,
                                                        false);
    citlali::pipeline::OutputPathState output_paths = [] {
        citlali::pipeline::OutputPathState paths;
        paths.redu_dir_name = "/tmp/redu01";
        return paths;
    }();
    struct {
        int fruit_iter = 0;
    } iteration;
    std::string redu_type = "science";
    std::string redu_dir_name = "/tmp/redu01";
    bool run_coadd = true;
    bool run_noise = true;
    bool apply_empirical_noise_weights = false;
    bool run_source_finder = false;
    struct {
        std::vector<std::string> date_obs = {"old"};
    } observation_dates;
    FakeIterationPtcProc ptcproc;
    FakeReductionLearning learning;
    int write_learning_summary_calls = 0;
    int output_calls = 0;
    int run_wiener_filter_calls = 0;
    int find_sources_calls = 0;

    struct {
        bool run_polarization = false;
    } rtcproc;

    struct {
        bool normalize_error = false;
    } wiener_filter;

    struct {
        std::vector<std::string> obsnums = {"101"};
        double exposure_time = 6.0;
        void normalize_maps() {}
        void normalize_polarized_maps() {}
        void calc_noise_products(bool) {}
        void calc_map_psd() {}
        void calc_map_hist() {}
        void calc_median_err() {}
        void calc_median_rms() {}
    } cmb;

    void write_learning_summary() { ++write_learning_summary_calls; }

    template <auto MapType>
    void output() {
        ++output_calls;
    }

    template <auto MapType, class MapBuffer>
    void run_wiener_filter(MapBuffer &) {
        ++run_wiener_filter_calls;
    }

    template <auto MapType, class MapBuffer>
    void find_sources(MapBuffer &) {
        ++find_sources_calls;
    }
};

template <class Engine>
void sync_fake_runtime_provenance(Engine &engine) {
    engine.runtime_config_provenance =
        citlali::config::make_runtime_config_provenance(
            engine.typed_config.runtime, false);
}

struct FakeReductionIterationTodProc {
    FakeReductionIterationEngine engine_state;
    int create_output_dir_calls = 0;
    int allocate_cmb_calls = 0;
    int allocate_nmb_calls = 0;
    int make_index_file_calls = 0;
    int create_coadded_map_files_calls = 0;
    std::string indexed_path;

    FakeReductionIterationEngine &engine() { return engine_state; }

    void create_output_dir() { ++create_output_dir_calls; }

    void allocate_cmb() { ++allocate_cmb_calls; }

    void create_coadded_map_files() { ++create_coadded_map_files_calls; }

    template <class MapBuffer>
    void allocate_nmb(MapBuffer &) {
        ++allocate_nmb_calls;
    }

    void make_index_file(const std::string &path) {
        ++make_index_file_calls;
        indexed_path = path;
    }
};

struct FakeCoaddTodProc {
    FakeEngine engine_state;
    int calc_cmb_size_calls = 0;
    int create_output_dir_calls = 0;
    int allocate_cmb_calls = 0;
    int allocate_nmb_calls = 0;
    int coadd_calls = 0;
    int create_coadded_map_files_calls = 0;
    int last_map_coord_count = 0;

    FakeEngine &engine() { return engine_state; }

    void create_output_dir() { ++create_output_dir_calls; }

    template <class MapCoords>
    void calc_cmb_size(MapCoords &map_coords) {
        ++calc_cmb_size_calls;
        last_map_coord_count = static_cast<int>(map_coords.size());
    }

    void allocate_cmb() { ++allocate_cmb_calls; }

    void coadd() { ++coadd_calls; }

    template <class MapBuffer>
    void allocate_nmb(MapBuffer &) {
        ++allocate_nmb_calls;
    }

    void create_coadded_map_files() { ++create_coadded_map_files_calls; }
};

struct FakeCalibrationTodProc {
    FakeEngine engine_state;
    int get_apt_from_files_calls = 0;

    FakeEngine &engine() { return engine_state; }

    void get_apt_from_files(const FakeRawObs &) { ++get_apt_from_files_calls; }
};

struct FakeTelescopeTodProc {
    FakeEngine engine_state;
    int check_inputs_calls = 0;
    int align_timestreams_calls = 0;
    int align_timestreams_gaps_calls = 0;
    int interp_pointing_calls = 0;
    int get_tone_freqs_from_files_calls = 0;
    int get_adc_snap_from_files_calls = 0;

    FakeEngine &engine() { return engine_state; }

    void check_inputs(const FakeRawObs &) { ++check_inputs_calls; }

    void align_timestreams(const FakeRawObs &) {
        ++align_timestreams_calls;
    }

    void align_timestreams_gaps(const FakeRawObs &) {
        ++align_timestreams_gaps_calls;
    }

    void interp_pointing() { ++interp_pointing_calls; }

    void get_tone_freqs_from_files(const FakeRawObs &) {
        ++get_tone_freqs_from_files_calls;
    }

    void get_adc_snap_from_files(const FakeRawObs &) {
        ++get_adc_snap_from_files_calls;
    }
};

struct FakeObservationMapTodProc {
    FakeEngine engine_state;
    int calc_map_num_calls = 0;
    int calc_omb_size_calls = 0;
    int allocate_omb_calls = 0;
    int allocate_nmb_calls = 0;
    int last_map_extent = 0;
    int last_map_coord = 0;

    FakeEngine &engine() { return engine_state; }

    void calc_map_num() { ++calc_map_num_calls; }

    template <class MapExtents, class MapCoords>
    void calc_omb_size(MapExtents &map_extents, MapCoords &map_coords) {
        ++calc_omb_size_calls;
        map_extents.push_back(101);
        map_coords.push_back(202);
    }

    void allocate_omb(int &map_extent, int &map_coord) {
        ++allocate_omb_calls;
        last_map_extent = map_extent;
        last_map_coord = map_coord;
    }

    template <class MapBuffer>
    void allocate_nmb(MapBuffer &) {
        ++allocate_nmb_calls;
    }
};

struct FakeInitialObservationTodProc : FakeTelescopeTodProc {
    int calc_map_num_calls = 0;
    int calc_omb_size_calls = 0;
    int calc_cmb_size_calls = 0;
    int create_output_dir_calls = 0;
    int allocate_omb_calls = 0;
    int allocate_nmb_calls = 0;
    int get_apt_from_files_calls = 0;
    int make_index_file_calls = 0;
    int allocate_cmb_calls = 0;
    int coadd_calls = 0;
    int create_coadded_map_files_calls = 0;
    int last_map_extent = 0;
    int last_map_coord = 0;
    int last_map_coord_count = 0;
    std::string indexed_path;

    void calc_map_num() { ++calc_map_num_calls; }

    template <class MapExtents, class MapCoords>
    void calc_omb_size(MapExtents &map_extents, MapCoords &map_coords) {
        ++calc_omb_size_calls;
        map_extents.push_back(303);
        map_coords.push_back(404);
    }

    template <class MapCoords>
    void calc_cmb_size(MapCoords &map_coords) {
        ++calc_cmb_size_calls;
        last_map_coord_count = static_cast<int>(map_coords.size());
    }

    void get_apt_from_files(const FakeRawObs &) {
        ++get_apt_from_files_calls;
    }

    void create_output_dir() { ++create_output_dir_calls; }

    void allocate_cmb() { ++allocate_cmb_calls; }

    void coadd() { ++coadd_calls; }

    void create_coadded_map_files() { ++create_coadded_map_files_calls; }

    void make_index_file(const std::string &path) {
        ++make_index_file_calls;
        indexed_path = path;
    }

    void allocate_omb(int &map_extent, int &map_coord) {
        ++allocate_omb_calls;
        last_map_extent = map_extent;
        last_map_coord = map_coord;
    }

    template <class MapBuffer>
    void allocate_nmb(MapBuffer &) {
        ++allocate_nmb_calls;
    }
};

enum class FakeMapType {
    RawObs,
    FilteredObs,
    RawCoadd,
    FilteredCoadd,
};

TEST(config_scaffold, formats_config_paths) {
    EXPECT_EQ(citlali::config::format_path({"runtime", "n_threads"}),
              "runtime.n_threads");
    EXPECT_EQ(citlali::config::format_path({}), "<config>");
}

TEST(config_scaffold, validation_report_tracks_errors_and_warnings) {
    citlali::config::ValidationReport report;

    EXPECT_TRUE(report.ok());
    report.add_warning({"runtime", "verbose"}, "example warning");
    EXPECT_TRUE(report.ok());
    EXPECT_EQ(report.warning_count(), 1U);

    report.add_error({"runtime", "n_threads"}, "must be positive");
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 1U);

    auto errors = report.errors();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors.front().path,
              (citlali::config::ConfigPath{"runtime", "n_threads"}));
}

TEST(config_scaffold, parses_existing_runtime_enum_values) {
    EXPECT_EQ(citlali::config::parse_parallel_policy("seq").value(),
              citlali::config::ParallelPolicy::seq);
    EXPECT_EQ(citlali::config::parse_parallel_policy("omp").value(),
              citlali::config::ParallelPolicy::omp);
    EXPECT_FALSE(citlali::config::parse_parallel_policy("threads").has_value());

    EXPECT_EQ(citlali::config::parse_reduction_type("science").value(),
              citlali::config::ReductionType::science);
    EXPECT_EQ(citlali::config::parse_reduction_type("pointing").value(),
              citlali::config::ReductionType::pointing);
    EXPECT_EQ(citlali::config::parse_reduction_type("beammap").value(),
              citlali::config::ReductionType::beammap);
}

TEST(config_scaffold, parses_existing_mapmaking_enum_values) {
    EXPECT_EQ(citlali::config::parse_map_grouping("auto").value(),
              citlali::config::MapGrouping::automatic);
    EXPECT_EQ(citlali::config::parse_map_grouping("detector").value(),
              citlali::config::MapGrouping::detector);
    EXPECT_EQ(citlali::config::parse_map_grouping("nw").value(),
              citlali::config::MapGrouping::network);
    EXPECT_EQ(citlali::config::parse_map_grouping("array").value(),
              citlali::config::MapGrouping::array);
    EXPECT_EQ(citlali::config::parse_map_grouping("fg").value(),
              citlali::config::MapGrouping::frequency_group);

    EXPECT_EQ(citlali::config::parse_map_method("naive").value(),
              citlali::config::MapMethod::naive);
    EXPECT_EQ(citlali::config::parse_map_method("jinc").value(),
              citlali::config::MapMethod::jinc);
    EXPECT_EQ(citlali::config::parse_map_method("maximum_likelihood").value(),
              citlali::config::MapMethod::maximum_likelihood);
}

TEST(config_scaffold, parses_existing_timestream_enum_values) {
    EXPECT_EQ(citlali::config::parse_tod_type("xs").value(),
              citlali::config::TodType::xs);
    EXPECT_EQ(citlali::config::parse_tod_type("rs").value(),
              citlali::config::TodType::rs);
    EXPECT_EQ(citlali::config::parse_tod_type("is").value(),
              citlali::config::TodType::is);
    EXPECT_EQ(citlali::config::parse_tod_type("qs").value(),
              citlali::config::TodType::qs);

    EXPECT_EQ(citlali::config::parse_tod_output_type("none").value(),
              citlali::config::TodOutputType::none);
    EXPECT_EQ(citlali::config::parse_tod_output_type("rtc").value(),
              citlali::config::TodOutputType::rtc);
    EXPECT_EQ(citlali::config::parse_tod_output_type("ptc").value(),
              citlali::config::TodOutputType::ptc);
    EXPECT_EQ(citlali::config::parse_tod_output_type("both").value(),
              citlali::config::TodOutputType::both);

    EXPECT_EQ(citlali::config::parse_tod_stream_output_mode("full").value(),
              citlali::config::TodStreamOutputMode::full);
    EXPECT_EQ(citlali::config::parse_tod_stream_output_mode("mini").value(),
              citlali::config::TodStreamOutputMode::mini);
    EXPECT_EQ(citlali::config::parse_tod_stream_output_mode("full_outer").value(),
              citlali::config::TodStreamOutputMode::full_outer);
    EXPECT_EQ(citlali::config::parse_tod_stream_output_mode("mini_outer").value(),
              citlali::config::TodStreamOutputMode::mini_outer);
    EXPECT_FALSE(citlali::config::parse_tod_stream_output_mode("outer").has_value());

    EXPECT_EQ(citlali::config::parse_tod_output_selection_mode("indices").value(),
              citlali::config::TodOutputSelectionMode::indices);
    EXPECT_EQ(citlali::config::parse_tod_output_selection_mode("all").value(),
              citlali::config::TodOutputSelectionMode::all);
    EXPECT_EQ(citlali::config::parse_tod_output_selection_mode(
                  "uniform_plus_source_crossing").value(),
              citlali::config::TodOutputSelectionMode::uniform_plus_source_crossing);
    EXPECT_FALSE(citlali::config::parse_tod_output_selection_mode("source").has_value());

    EXPECT_EQ(citlali::config::parse_raw_filter_edge_guard_mode("flag").value(),
              citlali::config::RawTimeChunkFilterEdgeGuardMode::flag);
    EXPECT_EQ(citlali::config::parse_raw_filter_edge_guard_mode("none").value(),
              citlali::config::RawTimeChunkFilterEdgeGuardMode::none);
    EXPECT_FALSE(citlali::config::parse_raw_filter_edge_guard_mode("mask").has_value());

    EXPECT_EQ(citlali::config::parse_raw_filter_edge_guard_combine("sum").value(),
              citlali::config::RawTimeChunkFilterEdgeGuardCombine::sum);
    EXPECT_EQ(citlali::config::parse_raw_filter_edge_guard_combine("max").value(),
              citlali::config::RawTimeChunkFilterEdgeGuardCombine::max);
    EXPECT_FALSE(citlali::config::parse_raw_filter_edge_guard_combine("mean").has_value());

    EXPECT_EQ(citlali::config::parse_processed_weighting_type("full").value(),
              citlali::config::ProcessedTimeChunkWeightingType::full);
    EXPECT_EQ(citlali::config::parse_processed_weighting_type("approximate").value(),
              citlali::config::ProcessedTimeChunkWeightingType::approximate);
    EXPECT_EQ(citlali::config::parse_processed_weighting_type("hybrid").value(),
              citlali::config::ProcessedTimeChunkWeightingType::hybrid);
    EXPECT_EQ(citlali::config::parse_processed_weighting_type("validated").value(),
              citlali::config::ProcessedTimeChunkWeightingType::validated);
    EXPECT_EQ(citlali::config::parse_processed_weighting_type("const").value(),
              citlali::config::ProcessedTimeChunkWeightingType::constant);
    EXPECT_FALSE(citlali::config::parse_processed_weighting_type("constant").has_value());

    EXPECT_EQ(citlali::config::parse_processed_weight_grouping("array").value(),
              citlali::config::ProcessedTimeChunkWeightGrouping::array);
    EXPECT_EQ(citlali::config::parse_processed_weight_grouping("nw").value(),
              citlali::config::ProcessedTimeChunkWeightGrouping::network);
    EXPECT_EQ(citlali::config::parse_processed_weight_grouping("all").value(),
              citlali::config::ProcessedTimeChunkWeightGrouping::all);
    EXPECT_FALSE(citlali::config::parse_processed_weight_grouping("fg").has_value());

    EXPECT_EQ(citlali::config::parse_processed_cleaner_mode("none").value(),
              citlali::config::ProcessedTimeChunkCleanerMode::none);
    EXPECT_EQ(citlali::config::parse_processed_cleaner_mode("standard_pca").value(),
              citlali::config::ProcessedTimeChunkCleanerMode::standard_pca);
    EXPECT_EQ(citlali::config::parse_processed_cleaner_mode("null_model").value(),
              citlali::config::ProcessedTimeChunkCleanerMode::null_model);
    EXPECT_EQ(citlali::config::parse_processed_cleaner_mode("marchenko_pastur").value(),
              citlali::config::ProcessedTimeChunkCleanerMode::marchenko_pastur);
    EXPECT_EQ(citlali::config::parse_processed_cleaner_mode("adaptive_selector").value(),
              citlali::config::ProcessedTimeChunkCleanerMode::adaptive_selector);
    EXPECT_FALSE(citlali::config::parse_processed_cleaner_mode("pca").has_value());

    EXPECT_EQ(citlali::config::parse_processed_corr_grouping_metric("abs").value(),
              citlali::config::ProcessedTimeChunkCorrGroupingMetric::abs);
    EXPECT_EQ(citlali::config::parse_processed_corr_grouping_metric("signed").value(),
              citlali::config::ProcessedTimeChunkCorrGroupingMetric::signed_metric);
    EXPECT_FALSE(citlali::config::parse_processed_corr_grouping_metric("pearson").has_value());

    EXPECT_EQ(citlali::config::parse_fruit_loops_mode("upper").value(),
              citlali::config::FruitLoopsMode::upper);
    EXPECT_EQ(citlali::config::parse_fruit_loops_mode("lower").value(),
              citlali::config::FruitLoopsMode::lower);
    EXPECT_EQ(citlali::config::parse_fruit_loops_mode("both").value(),
              citlali::config::FruitLoopsMode::both);
    EXPECT_FALSE(citlali::config::parse_fruit_loops_mode("absolute").has_value());

    EXPECT_EQ(citlali::config::parse_fruit_loops_weight_feedback_reference("p95").value(),
              citlali::config::FruitLoopsWeightFeedbackReference::p95);
    EXPECT_EQ(citlali::config::parse_fruit_loops_weight_feedback_reference("median").value(),
              citlali::config::FruitLoopsWeightFeedbackReference::median);
    EXPECT_EQ(citlali::config::parse_fruit_loops_weight_feedback_reference("peak").value(),
              citlali::config::FruitLoopsWeightFeedbackReference::peak);
    EXPECT_FALSE(citlali::config::parse_fruit_loops_weight_feedback_reference("mean").has_value());

    EXPECT_EQ(citlali::config::parse_fruit_loops_interp_mode_override("auto").value(),
              citlali::config::FruitLoopsInterpModeOverride::automatic);
    EXPECT_EQ(citlali::config::parse_fruit_loops_interp_mode_override("nearest").value(),
              citlali::config::FruitLoopsInterpModeOverride::nearest);
    EXPECT_EQ(citlali::config::parse_fruit_loops_interp_mode_override("bilinear").value(),
              citlali::config::FruitLoopsInterpModeOverride::bilinear);
    EXPECT_EQ(citlali::config::parse_fruit_loops_interp_mode_override("jinc").value(),
              citlali::config::FruitLoopsInterpModeOverride::jinc);
    EXPECT_EQ(citlali::config::parse_fruit_loops_interp_mode_override("trunc").value(),
              citlali::config::FruitLoopsInterpModeOverride::trunc);
    EXPECT_FALSE(citlali::config::parse_fruit_loops_interp_mode_override("legacy_nearest").has_value());
}

TEST(config_scaffold, parses_existing_pointing_enum_values) {
    EXPECT_EQ(citlali::config::parse_pointing_source_strategy("standard").value(),
              citlali::config::PointingSourceStrategy::standard);
    EXPECT_EQ(citlali::config::parse_pointing_source_strategy("psf_preserve").value(),
              citlali::config::PointingSourceStrategy::psf_preserve);

    EXPECT_EQ(citlali::config::parse_fruit_loops_center_mode("auto").value(),
              citlali::config::FruitLoopsCenterMode::automatic);
    EXPECT_EQ(citlali::config::parse_fruit_loops_center_mode("header").value(),
              citlali::config::FruitLoopsCenterMode::header);
    EXPECT_EQ(citlali::config::parse_fruit_loops_center_mode("peak").value(),
              citlali::config::FruitLoopsCenterMode::peak);
    EXPECT_EQ(citlali::config::parse_fruit_loops_center_mode("map_center").value(),
              citlali::config::FruitLoopsCenterMode::map_center);
}

TEST(config_scaffold, parses_existing_map_filter_enum_values) {
    EXPECT_EQ(citlali::config::parse_map_filter_type("wiener_filter").value(),
              citlali::config::MapFilterType::wiener_filter);
    EXPECT_EQ(citlali::config::parse_map_filter_type("convolve").value(),
              citlali::config::MapFilterType::convolve);
    EXPECT_EQ(citlali::config::parse_map_filter_type("destripe").value(),
              citlali::config::MapFilterType::destripe);
    EXPECT_FALSE(citlali::config::parse_map_filter_type("smooth").has_value());

    EXPECT_EQ(citlali::config::parse_map_filter_template_type("kernel").value(),
              citlali::config::MapFilterTemplateType::kernel);
    EXPECT_EQ(citlali::config::parse_map_filter_template_type("gaussian").value(),
              citlali::config::MapFilterTemplateType::gaussian);
    EXPECT_EQ(citlali::config::parse_map_filter_template_type("airy").value(),
              citlali::config::MapFilterTemplateType::airy);
    EXPECT_EQ(citlali::config::parse_map_filter_template_type("highpass").value(),
              citlali::config::MapFilterTemplateType::highpass);

    EXPECT_EQ(citlali::config::parse_map_filter_edge_taper_mode("none").value(),
              citlali::config::MapFilterEdgeTaperMode::none);
    EXPECT_EQ(citlali::config::parse_map_filter_edge_taper_mode("cosine").value(),
              citlali::config::MapFilterEdgeTaperMode::cosine);
}

TEST(config_scaffold, parses_existing_beammap_enum_values) {
    EXPECT_EQ(citlali::config::parse_beammap_detector_weighting_mode("const").value(),
              citlali::config::BeammapDetectorWeightingMode::constant);
    EXPECT_EQ(citlali::config::parse_beammap_detector_weighting_mode("ptc").value(),
              citlali::config::BeammapDetectorWeightingMode::ptc);
    EXPECT_EQ(citlali::config::parse_beammap_detector_weighting_mode("ptc_after_iter0").value(),
              citlali::config::BeammapDetectorWeightingMode::ptc_after_iter0);
    EXPECT_FALSE(citlali::config::parse_beammap_detector_weighting_mode("weights").has_value());
}

TEST(config_scaffold, parses_polarimetry_enum_values) {
    EXPECT_EQ(citlali::config::parse_polarimetry_grouping("fg").value(),
              citlali::config::PolarimetryGrouping::frequency_group);
    EXPECT_EQ(citlali::config::parse_polarimetry_grouping("loc").value(),
              citlali::config::PolarimetryGrouping::detector_location);
    EXPECT_FALSE(
        citlali::config::parse_polarimetry_grouping("network").has_value());

    EXPECT_EQ(citlali::config::parse_polarimetry_hwpr_policy("auto").value(),
              citlali::config::PolarimetryHwprPolicy::automatic);
    EXPECT_EQ(citlali::config::parse_polarimetry_hwpr_policy("true").value(),
              citlali::config::PolarimetryHwprPolicy::ignore);
    EXPECT_EQ(citlali::config::parse_polarimetry_hwpr_policy("false").value(),
              citlali::config::PolarimetryHwprPolicy::require);
}

TEST(config_scaffold, mirrors_legacy_polarimetry_runtime_config) {
    struct FakeRtcProc {
        struct FakePolarization {
            std::string grouping = "loc";
        } polarization;
        bool run_polarization = true;
    } rtcproc;
    citlali::config::TimestreamPolarimetryConfig config;

    citlali::pipeline::mirror_polarimetry_config(config, rtcproc);

    EXPECT_TRUE(config.enabled);
    EXPECT_EQ(config.grouping,
              citlali::config::PolarimetryGrouping::detector_location);
    EXPECT_EQ(config.hwpr_policy,
              citlali::config::PolarimetryHwprPolicy::automatic);
}

TEST(config_scaffold, validates_beammap_source_fluxes) {
    const std::map<int, std::string> array_names = {
        {0, "a1100"}, {1, "a1400"}, {2, "a2000"}};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::validate_beammap_source_fluxes(
        std::map<std::string, double>{{"a1100", 1.0},
                                      {"a1400", 2.0},
                                      {"a2000", 3.0}},
        array_names, logger));
    EXPECT_EQ(logger->error_calls, 0);

    EXPECT_FALSE(citlali::pipeline::validate_beammap_source_fluxes(
        std::map<std::string, double>{{"a1100", 1.0}, {"a1400", 0.0}},
        array_names, logger));
    EXPECT_EQ(logger->error_calls, 2);
}

TEST(config_scaffold, validates_top_level_config_values) {
    citlali::config::ReductionConfig config;
    EXPECT_TRUE(citlali::config::validate(config).ok());

    config.runtime.n_threads = 0;
    config.timestream.enabled = false;
    config.mapmaking.pixel_size_arcsec = -1.0;
    config.noise.enabled = true;
    config.noise.n_noise_maps = -1;
    config.post_processing.source_fitting.active = true;
    config.post_processing.source_fitting.bounding_box_arcsec = -1.0;
    config.post_processing.map_histogram_n_bins = -1;
    config.pointing.header_max_radius_arcsec = -1.0;
    config.beammap.iteration.max_iterations = 0;

    auto report = citlali::config::validate(config);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 8U);
}

TEST(config_scaffold, validates_timestream_output_selection_values) {
    citlali::config::TimestreamConfig config;
    config.output.raw_time_chunk.outer_context_samples = -1;
    config.output.raw_time_chunk.chunks_1based.push_back(0);
    config.output.raw_time_chunk.selection_mode =
        citlali::config::TodOutputSelectionMode::uniform_plus_source_crossing;
    config.output.raw_time_chunk.selection_n_uniform = 0;
    config.output.raw_time_chunk.selection_n_source_dense = 0;
    config.output.processed_time_chunk.selection_n_uniform = -1;
    config.chunking.value = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 5U);
}

TEST(config_scaffold, validates_timestream_source_protection_values) {
    citlali::config::TimestreamConfig config;
    config.raw_time_chunk.despike.source_protection.radius_arcsec = -1.0;
    config.processed_time_chunk.flagging.second_pass_local
        .source_protection.radius_arcsec = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 2U);
}

TEST(config_scaffold, validates_timestream_despike_local_residual_values) {
    citlali::config::RawTimeChunkDespikeLocalResidualConfig config;
    config.window_sec = -1.0;
    config.sigma_scale = -1.0;
    config.delta_sigma_scale = -1.0;
    config.event_padding_sec = -1.0;
    config.high_score_event_override = -1.0;
    config.max_added_flagged_fraction = 2.0;
    config.compact_raw_gate.candidate_rel_sigma_scale = -1.0;
    config.compact_raw_gate.window_sec = -1.0;
    config.compact_raw_gate.half_peak_frac = 2.0;
    config.compact_raw_gate.max_width_sec = -1.0;
    config.compact_raw_gate.max_step_shift_z = -1.0;
    config.compact_delta_gate.window_sec = -1.0;
    config.compact_delta_gate.half_peak_frac = 2.0;
    config.compact_delta_gate.max_width_sec = -1.0;
    config.compact_delta_gate.max_step_shift_z = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 15U);
}

TEST(config_scaffold, validates_raw_time_chunk_filter_values) {
    citlali::config::RawTimeChunkConfig config;
    config.downsample.enabled = true;
    config.downsample.factor = -1;
    config.downsample.downsampled_freq_Hz = -1.0;
    config.filter.enabled = true;
    config.filter.a_gibbs = -1.0;
    config.filter.freq_low_Hz = 3.0;
    config.filter.freq_high_Hz = 2.0;
    config.filter.n_terms = -1;
    config.filter.notch.enabled = true;
    config.filter.notch.zero_phase = false;
    config.filter.notch.freqs_Hz = {-1.0};
    config.filter.edge_guard.enabled = true;
    config.filter.edge_guard.min_samples = -1;
    config.filter.edge_guard.extra_samples = -1;
    config.filter.edge_guard.max_samples = -1;
    config.filter.edge_guard.iir_settle_attenuation = 2.0;
    config.iir_filter.enabled = true;
    config.iir_filter.freq_Hz = -1.0;
    config.iir_filter.order = 0;
    config.iir_filter.zero_phase = false;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 15U);
}

TEST(config_scaffold, validates_raw_downsample_requires_filter) {
    citlali::config::RawTimeChunkConfig config;
    config.downsample.enabled = true;
    config.filter.enabled = false;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 1U);
}

TEST(config_scaffold, validates_raw_time_chunk_flagging_diagnostics) {
    citlali::config::RawTimeChunkFlaggingConfig config;
    config.network_step_mask.enabled = true;
    config.network_step_mask.step_window_sec = 0.0;
    config.network_step_mask.step_score_thresh = -1.0;
    config.network_step_mask.min_good_frac = 2.0;
    config.network_step_mask.min_det_used = 0;
    config.network_step_mask.min_step_det_frac = -1.0;
    config.network_step_mask.min_alignment_frac = 2.0;
    config.network_step_mask.cluster_tol_sec = -1.0;
    config.network_step_mask.mask_half_width_sec = -1.0;
    config.network_step_mask.max_flagged_fraction = -1.0;
    config.impulsive_capture.enabled = true;
    config.impulsive_capture.min_good_frac = -1.0;
    config.impulsive_capture.min_event_z = -1.0;
    config.impulsive_capture.near_event_z = -1.0;
    config.impulsive_capture.max_events_per_network = 0;
    config.impulsive_capture.snippet_pre_window_sec = -1.0;
    config.impulsive_capture.snippet_post_window_sec = -1.0;
    config.impulsive_coincidence.enabled = true;
    config.impulsive_coincidence.min_good_frac = 2.0;
    config.impulsive_coincidence.event_score_thresh = -1.0;
    config.impulsive_coincidence.min_det_used = 0;
    config.impulsive_coincidence.min_impulsive_det_frac = -1.0;
    config.impulsive_coincidence.min_alignment_frac = 2.0;
    config.impulsive_coincidence.min_networks_aligned = 0;
    config.impulsive_coincidence.high_score_override_thresh = -1.0;
    config.impulsive_coincidence.high_score_min_networks_aligned = -1;
    config.impulsive_coincidence.cluster_tol_sec = -1.0;
    config.impulsive_coincidence.mask_pre_window_sec = -1.0;
    config.impulsive_coincidence.mask_post_window_sec = -1.0;
    config.impulsive_coincidence.max_flagged_fraction = 2.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 27U);
}

TEST(config_scaffold, validates_raw_time_chunk_altaz_destripe_values) {
    citlali::config::RawTimeChunkAltAzDestripeConfig config;
    config.enabled = true;
    config.min_samples = 3;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 1U);
}

TEST(config_scaffold, validates_raw_time_chunk_line_audit_values) {
    citlali::config::RawTimeChunkLineAuditConfig config;
    config.line_min_hz = -1.0;
    config.line_max_hz = -1.0;
    config.segment_sec = 0.0;
    config.min_segment_sec = 0.0;
    config.overlap_frac = 1.0;
    config.continuum_radius_bins = 0;
    config.prominence_thresh = 0.0;
    config.cm_prominence_thresh = 0.0;
    config.min_good_frac = -1.0;
    config.min_windows = 0;
    config.max_peaks_per_detector = 0;
    config.max_det = -1;
    config.min_det_for_network = 0;
    config.cluster_tol_hz = -1.0;
    config.notch_min_detector_frac = 2.0;
    config.notch_min_detectors = 0;
    config.notch_min_cm_prominence = 0.0;
    config.detector_min_prominence = 0.0;
    config.detector_min_line_power_frac = -1.0;
    config.bad_detector_max_cluster_frac = 2.0;
    config.post_filter_apply_iterations = 0;
    config.post_filter_line_min_hz = -1.0;
    config.post_filter_line_max_hz = -1.0;
    config.ptc_apply_iterations = 0;
    config.ptc_line_min_hz = 2.0;
    config.ptc_line_max_hz = 1.0;
    config.fixed_notch_enabled = true;
    config.fixed_notch_widths_hz = {-1.0};
    config.fixed_notch_exclusion_half_width_hz = -1.0;
    config.apply_min_support_networks = 0;
    config.apply_min_detector_frac = -1.0;
    config.apply_min_common_mode_prominence = 0.0;
    config.apply_width_scale = 0.0;
    config.apply_min_width_hz = 2.0;
    config.apply_max_width_hz = 1.0;
    config.apply_max_notches = -1;
    config.apply_cluster_tol_hz = -1.0;
    config.detector_notch_min_prominence = 0.0;
    config.detector_notch_min_line_power_frac = 2.0;
    config.detector_notch_max_notches = -1;
    config.detector_notch_width_scale = 0.0;
    config.detector_notch_min_width_hz = 2.0;
    config.detector_notch_max_width_hz = 1.0;
    config.detector_notch_context_samples = -1;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 41U);
}

TEST(config_scaffold, validates_processed_time_chunk_second_pass_local_values) {
    citlali::config::ProcessedTimeChunkSecondPassLocalConfig config;
    config.min_spike_sigma = -1.0;
    config.min_good_frac = 2.0;
    config.baseline_window_sec = -1.0;
    config.sigma_scale = -1.0;
    config.delta_sigma_scale = -1.0;
    config.raw_candidate_rel_sigma_scale = -1.0;
    config.raw_window_sec = -1.0;
    config.raw_half_peak_frac = -1.0;
    config.raw_max_width_sec = -1.0;
    config.delta_window_sec = -1.0;
    config.delta_half_peak_frac = -1.0;
    config.delta_max_width_sec = -1.0;
    config.max_step_shift_z = -1.0;
    config.high_score_event_override = -1.0;
    config.merge_within_detector_sec = -1.0;
    config.cluster_events_sec = -1.0;
    config.min_cluster_detectors = 0;
    config.high_score_cluster_override = -1.0;
    config.max_auto_flag_clusters_per_network = 0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 19U);
}

TEST(config_scaffold, validates_processed_time_chunk_weighting_values) {
    citlali::config::ProcessedTimeChunkWeightingConfig config;
    config.source_mask_radius_arcsec = -1.0;
    config.hybrid_correction_min_factor = 2.0;
    config.hybrid_correction_max_factor = 1.0;
    config.busy_row_suppression.enabled = true;
    config.busy_row_suppression.min_candidate_clusters = -1;
    config.busy_row_suppression.min_max_unflagged_residual_z = -1.0;
    config.busy_row_suppression.factor = 2.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 5U);
}

TEST(config_scaffold, validates_processed_time_chunk_clean_values) {
    citlali::config::ProcessedTimeChunkCleanConfig config;
    config.enabled = true;
    config.standard_pca.n_calc = -1;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 1U);
}

TEST(config_scaffold, requires_exactly_one_processed_cleaner) {
    citlali::config::ProcessedTimeChunkCleanConfig config;
    config.enabled = true;
    config.standard_pca.enabled = false;

    citlali::config::ValidationReport none_enabled;
    citlali::config::validate(config, none_enabled);
    EXPECT_EQ(none_enabled.error_count(), 1U);

    config.standard_pca.enabled = true;
    config.null_model.enabled = true;
    citlali::config::ValidationReport multiple_enabled;
    citlali::config::validate(config, multiple_enabled);
    EXPECT_EQ(multiple_enabled.error_count(), 1U);
}

TEST(config_scaffold, validates_processed_time_chunk_clean_expert_values) {
    citlali::config::ProcessedTimeChunkCleanConfig config;
    config.enabled = true;
    config.corr_grouping.enabled = true;
    config.corr_grouping.corr_min = 2.0;
    config.corr_grouping.min_overlap = 0;
    config.corr_grouping.min_good_frac = -1.0;
    config.corr_grouping.min_group_size = 1;
    config.corr_grouping.max_samples = -1;
    config.null_model.enabled = true;
    config.null_model.n_surrogates = 3;
    config.null_model.quantile = 0.4;
    config.null_model.min_good_frac = 2.0;
    config.null_model.max_modes = -1;
    config.null_model.max_samples = -1;
    config.null_model.seed = -1;
    config.marchenko_pastur.enabled = true;
    config.marchenko_pastur.min_good_frac = 2.0;
    config.marchenko_pastur.max_modes = -1;
    config.marchenko_pastur.max_samples = -1;
    config.marchenko_pastur.band_low_Hz = -1.0;
    config.marchenko_pastur.band_high_Hz = -1.0;
    config.marchenko_pastur.bulk_keep_frac = 0.0;
    config.marchenko_pastur.q_grid_size = 7;
    config.adaptive_selector.enabled = true;
    config.adaptive_selector.min_good_frac = 2.0;
    config.adaptive_selector.max_det = -1;
    config.adaptive_selector.max_samples = -1;
    config.adaptive_selector.max_pairs = -1;
    config.adaptive_selector.seed = -1;
    config.adaptive_selector.low_weight = -1.0;
    config.adaptive_selector.tail_weight = -1.0;
    config.adaptive_selector.topmode_weight = -1.0;
    config.adaptive_selector.reg_weight = -1.0;
    config.adaptive_selector.low_band_Hz = {-1.0, 0.5};
    config.adaptive_selector.mid_band_Hz = {2.0, 1.0};

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 30U);
}

TEST(config_scaffold, validates_processed_time_chunk_weight_validation_values) {
    citlali::config::ProcessedTimeChunkWeightValidationConfig config;
    config.enabled = true;
    config.accumulation_iters = 0;
    config.apply_start_iter = -1;
    config.min_valid_scans = 0;
    config.min_factor = 2.0;
    config.unvalidated_factor = -1.0;
    config.ratio_power = -1.0;
    config.transient_ratio_power = -1.0;
    config.upward_max_factor = 0.5;
    config.upward_power = -1.0;
    config.upward_min_base_factor = 2.0;
    config.upward_min_atmospheric_factor = -1.0;
    config.atmospheric_min_detectors = 1;
    config.atmospheric_ref = 2.0;
    config.atmospheric_span = 0.0;
    config.atmospheric_power = -1.0;
    config.min_good_frac = 2.0;
    config.min_overlap = 1;
    config.max_samples = -1;
    config.high_weight_min_group_detectors = 1;
    config.high_weight_log_robust_z = -1.0;
    config.high_weight_max_median_factor = 0.5;
    config.high_weight_cap_median_factor = 0.5;
    config.high_weight_min_validated_factor = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 23U);
}

TEST(config_scaffold, validates_processed_time_chunk_corr_penalty_values) {
    citlali::config::ProcessedTimeChunkWeightCorrPenaltyConfig config;
    config.enabled = true;
    config.min_good_frac = 2.0;
    config.min_overlap = 1;
    config.max_samples = -1;
    config.max_pairs = -1;
    config.seed = -1;
    config.floor = 2.0;
    config.exponent = -1.0;
    config.pair_corr.span = 0.0;
    config.pair_corr.weight = -1.0;
    config.cm_el_corr.span = 0.0;
    config.cm_el_corr.weight = -1.0;
    config.cm_low_mid_ratio.span = 0.0;
    config.cm_low_mid_ratio.weight = -1.0;
    config.cm_low_mid_ratio.low_band_Hz = {-1.0, 0.5};
    config.cm_low_mid_ratio.mid_band_Hz = {2.0, 1.0};

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 15U);
}

TEST(config_scaffold, validates_fruit_loops_values) {
    citlali::config::TimestreamFruitLoopsConfig config;
    config.enabled = true;
    config.peak_fraction_limit = -1.0;
    config.local_snr_floor = -1.0;
    config.local_sigma_inner_radius_arcsec = -1.0;
    config.local_sigma_outer_radius_arcsec = -1.0;
    config.local_sigma_inner_fwhm = -1.0;
    config.local_sigma_outer_fwhm = -1.0;
    config.local_sigma_edge_guard_arcsec = -1.0;
    config.local_sigma_min_pixels = 0;
    config.adaptive_support_radius_arcsec = -1.0;
    config.adaptive_support_radius_fwhm = -1.0;
    config.weight_feedback.enabled = true;
    config.weight_feedback.low_relative_weight = -1.0;
    config.weight_feedback.high_relative_weight = -2.0;
    config.center_keep_radius_arcsec = -1.0;
    config.max_iters = -1;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 15U);
}

TEST(config_scaffold, validates_timestream_learning_values) {
    citlali::config::TimestreamLearningConfig config;
    config.learn_iters = -1;
    config.apply_start_iter = -1;
    config.max_records_per_type = -1;
    config.apply_max_new_flagged_fraction = -1.0;
    config.map_pixel_outlier.top_n = -1;
    config.map_pixel_outlier.targeted_contributor_max_pixels = -1;
    config.map_pixel_outlier.detector_exclusion_min_pixels = 0;
    config.map_pixel_outlier.min_abs_z = -1.0;
    config.map_pixel_outlier.min_n_eff = -1.0;
    config.map_pixel_outlier.source_radius_arcsec = -1.0;
    config.scan_network_pathology.min_candidate_clusters = -1;
    config.scan_network_pathology.min_candidate_events = -1;
    config.scan_network_pathology.min_max_residual_z = -1.0;
    config.scan_network_pathology.severe_candidate_events = -1;
    config.scan_network_pathology.severe_max_residual_z = -1.0;
    config.scan_network_pathology.max_new_flagged_fraction = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 16U);
}

TEST(config_scaffold, validates_map_filter_config_values) {
    citlali::config::MapFilterConfig config;
    config.enabled = true;
    config.edge_guard.hits_core_fraction = -1.0;
    config.edge_guard.guard_radius_fwhm = -1.0;
    config.edge_guard.taper_min_fraction = 2.0;
    config.denom_rel_tol = 2.0;
    config.tail_frac_tol = -1.0;
    config.max_loops = 0;
    config.denom_check_iters = -1;
    config.max_denom_iters = -1;
    config.template_fwhm_arcsec[""] = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 10U);
}

TEST(config_scaffold, validates_source_finding_config_values) {
    citlali::config::SourceFindingConfig config;
    config.enabled = true;
    config.source_sigma = -1.0;
    config.source_window_arcsec = -1.0;
    config.mode.clear();

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 3U);
}

TEST(config_scaffold, validates_beammap_config_values) {
    citlali::config::BeammapConfig config;
    citlali::config::ValidationReport initial_report;
    citlali::config::validate(config, initial_report);
    EXPECT_TRUE(initial_report.ok());

    config.phase_strategy.measurement_start_iter = 0;
    config.rfi_mask.max_flagged_fraction = 1.5;
    config.scan_band_mask.edge_rows = 1;
    config.priors.candidate_top_n = 0;
    config.priors.alignment_common_support_quantile = 0.5;
    config.detector_tod_output.n_uniform = -1;
    config.flagging.max_prior_d2 = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 7U);
}

TEST(config_scaffold, validates_beammap_source_values) {
    citlali::config::BeammapSourceConfig config;
    config.fluxes.push_back(citlali::config::BeammapSourceFluxConfig{"", 0.0, -1.0});

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 3U);
}

TEST(config_scaffold, validates_astrometry_pointing_offsets_values) {
    citlali::config::AstrometryPointingOffsetsConfig config;
    citlali::config::ValidationReport initial_report;
    citlali::config::validate(citlali::config::AstrometryConfig{},
                              initial_report);
    EXPECT_TRUE(initial_report.ok());

    config.enabled = true;
    config.az_arcsec = {1.0, 2.0, 3.0};
    config.alt_arcsec = {1.0};
    config.modified_julian_date = {1.0};

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 3U);
}

TEST(error_scaffold, preserves_error_code_and_message) {
    auto error = citlali::error::invalid_config("bad config");
    EXPECT_EQ(error.code(), citlali::error::Code::invalid_config);
    EXPECT_STREQ(error.what(), "bad config");
}

TEST(cli_config_loading, loads_and_merges_config_files) {
    FakeRuntimeConfig runtime_config{
        {FakeConfigNode{"70_reduce.yaml"}, FakeConfigNode{"80_reduce.yaml"}}};
    std::vector<std::string> config_filepaths;
    auto logger = std::make_shared<FakeLogger>();

    auto config = citlali::cli::load_config_files<
        FakeRuntimeConfig, FakeLoadedConfig>(
        runtime_config, config_filepaths, logger,
        [](const std::string &filepath) {
            FakeLoadedConfig config;
            config.loaded_paths.push_back(filepath);
            return config;
        },
        [](FakeLoadedConfig lhs, FakeLoadedConfig rhs) {
            lhs.loaded_paths.insert(lhs.loaded_paths.end(),
                                    rhs.loaded_paths.begin(),
                                    rhs.loaded_paths.end());
            return lhs;
        });

    EXPECT_EQ(config_filepaths,
              (std::vector<std::string>{"70_reduce.yaml", "80_reduce.yaml"}));
    EXPECT_EQ(config.loaded_paths,
              (std::vector<std::string>{"70_reduce.yaml", "80_reduce.yaml"}));
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(cli_runtime_setup, derives_fftw_threads) {
    EXPECT_EQ(citlali::cli::fftw_threads_for_runtime(8, false), 8);
    EXPECT_EQ(citlali::cli::fftw_threads_for_runtime(8, true), 1);
}

TEST(cli_runtime_setup, separates_requested_effective_and_realized_runtime) {
    citlali::config::RuntimeConfig requested;
    requested.n_threads = 6;
    requested.parallel_policy = citlali::config::ParallelPolicy::omp;

    auto provenance = citlali::config::make_runtime_config_provenance(
        requested, true);

    EXPECT_TRUE(provenance.initialized);
    EXPECT_EQ(provenance.requested.n_threads, 6);
    EXPECT_EQ(provenance.effective.values.n_threads, 6);
    EXPECT_EQ(provenance.effective.threads.requested_threads, 6);
    EXPECT_EQ(provenance.effective.threads.omp_threads, 6);
    EXPECT_EQ(provenance.effective.threads.eigen_threads, 1);
    EXPECT_EQ(provenance.effective.threads.fftw_plan_threads, 1);
    EXPECT_TRUE(provenance.effective.threads.wiener_filter_omp);
    EXPECT_FALSE(provenance.realized.fftw_threads_initialized);

    provenance.effective.values.n_threads = 3;
    EXPECT_EQ(provenance.requested.n_threads, 6);
}

TEST(cli_runtime_setup, uses_effective_thread_plan_as_runtime_authority) {
    FakeEngine engine;
    engine.runtime_config_provenance =
        citlali::config::make_runtime_config_provenance(
            engine.typed_config.runtime, false);
    engine.runtime_config_provenance.requested.n_threads = 6;
    engine.runtime_config_provenance.effective.threads =
        citlali::config::make_runtime_thread_plan(3, false);

    EXPECT_EQ(engine.runtime_config_provenance.requested.n_threads, 6);
    EXPECT_EQ(citlali::pipeline::runtime_thread_count(engine), 3);
}

TEST(cli_runtime_setup, uses_effective_runtime_values_as_policy_authority) {
    FakeEngine engine;
    engine.runtime_config_provenance.requested.output_dir = "requested";
    engine.runtime_config_provenance.requested.reduction_type =
        citlali::config::ReductionType::pointing;
    engine.runtime_config_provenance.effective.values.output_dir = "effective";
    engine.runtime_config_provenance.effective.values.reduction_type =
        citlali::config::ReductionType::science;
    engine.runtime_config_provenance.effective.values.verbose = true;
    engine.runtime_config_provenance.effective.values.parallel_policy =
        citlali::config::ParallelPolicy::omp;

    EXPECT_TRUE(citlali::pipeline::verbose_runtime_enabled(engine));
    EXPECT_EQ(citlali::pipeline::runtime_output_dir(engine), "effective");
    EXPECT_EQ(citlali::pipeline::runtime_reduction_type(engine),
              citlali::config::ReductionType::science);
    EXPECT_EQ(citlali::pipeline::runtime_parallel_policy_name(engine), "omp");
}

TEST(cli_runtime_setup, serializes_stable_runtime_provenance_schema) {
    citlali::config::RuntimeConfig requested;
    requested.n_threads = 6;
    requested.parallel_policy = citlali::config::ParallelPolicy::omp;
    requested.reduction_type = citlali::config::ReductionType::pointing;
    auto provenance = citlali::config::make_runtime_config_provenance(
        requested, true);
    provenance.realized.omp_threads = 6;
    provenance.realized.eigen_threads = 1;
    provenance.realized.fftw_plan_threads = 1;
    provenance.realized.fftw_threads_initialized = true;
    provenance.realized.parallel_policy =
        citlali::config::ParallelPolicy::omp;
    provenance.realized.reduction_type =
        citlali::config::ReductionType::pointing;

    const auto node = citlali::pipeline::runtime_provenance_node(provenance);

    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-runtime-provenance-v1");
    EXPECT_EQ(node["requested"]["n_threads"].as<int>(), 6);
    EXPECT_EQ(node["requested"]["parallel_policy"].as<std::string>(),
              "omp");
    EXPECT_EQ(node["effective"]["threads"]["fftw_plan"].as<int>(), 1);
    EXPECT_TRUE(node["effective"]["threads"]["wiener_filter_omp"].as<bool>());
    EXPECT_TRUE(node["realized"]["threads"]["fftw_initialized"].as<bool>());
    EXPECT_EQ(node["realized"]["parallel_policy"].as<std::string>(), "omp");
    EXPECT_EQ(node["realized"]["reduction_type"].as<std::string>(),
              "pointing");
}

TEST(cli_runtime_setup, atomically_writes_runtime_provenance_sidecar) {
    const auto output_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_runtime_provenance_output_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    const auto provenance = citlali::config::make_runtime_config_provenance(
        citlali::config::RuntimeConfig{}, false);

    citlali::pipeline::write_runtime_provenance_file(output_dir, provenance);

    const auto output_path =
        citlali::pipeline::runtime_provenance_path(output_dir);
    EXPECT_TRUE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    const auto stored = YAML::LoadFile(output_path.string());
    EXPECT_EQ(stored["schema_version"].as<std::string>(),
              "citlali-runtime-provenance-v1");
    EXPECT_TRUE(stored["initialized"].as<bool>());
    std::filesystem::remove_all(output_dir);
}

TEST(cli_runtime_setup, runtime_provenance_write_failure_propagates) {
    const auto missing_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_missing_runtime_provenance_dir" / "nested";
    std::filesystem::remove_all(missing_dir.parent_path());
    const auto provenance = citlali::config::make_runtime_config_provenance(
        citlali::config::RuntimeConfig{}, false);

    EXPECT_THROW(citlali::pipeline::write_runtime_provenance_file(
                     missing_dir, provenance),
                 std::ios_base::failure);
    EXPECT_FALSE(std::filesystem::exists(
        citlali::pipeline::runtime_provenance_path(missing_dir)));
    EXPECT_FALSE(std::filesystem::exists(
        citlali::pipeline::runtime_provenance_path(missing_dir).string() +
        ".tmp"));
}

TEST(config_scaffold, typed_tod_output_shape_ignores_legacy_mirror_values) {
    struct OutputEngine {
        citlali::config::ReductionConfig typed_config;
        struct {
            bool tod_output_mini = false;
            bool tod_output_outer = false;
            int tod_output_outer_context_samples = 0;
        } rtcproc;
        struct {
            bool tod_output_mini = false;
        } ptcproc;
    } engine;
    engine.typed_config.timestream.output.raw_time_chunk.mode =
        citlali::config::TodStreamOutputMode::mini_outer;
    engine.typed_config.timestream.output.raw_time_chunk.outer_context_samples =
        17;
    engine.typed_config.timestream.output.processed_time_chunk.mode =
        citlali::config::TodStreamOutputMode::mini;

    EXPECT_TRUE(citlali::pipeline::raw_tod_mini_output(engine));
    EXPECT_TRUE(citlali::pipeline::raw_tod_outer_output(engine));
    EXPECT_EQ(citlali::pipeline::raw_tod_outer_context_samples(engine), 17);
    EXPECT_TRUE(citlali::pipeline::processed_tod_mini_output(engine));
    EXPECT_FALSE(engine.rtcproc.tod_output_mini);
    EXPECT_FALSE(engine.rtcproc.tod_output_outer);
    EXPECT_EQ(engine.rtcproc.tod_output_outer_context_samples, 0);
    EXPECT_FALSE(engine.ptcproc.tod_output_mini);
}

TEST(config_scaffold, tod_file_layout_uses_typed_stream_modes) {
    citlali::config::TodStreamOutputConfig rtc_output;
    rtc_output.mode = citlali::config::TodStreamOutputMode::mini_outer;
    citlali::config::TodStreamOutputConfig ptc_output;
    ptc_output.mode = citlali::config::TodStreamOutputMode::full;

    const auto rtc_layout = citlali::pipeline::tod_stream_layout(
        citlali::config::TodOutputStream::rtc, 4, 7, rtc_output, ptc_output);
    const auto ptc_layout = citlali::pipeline::tod_stream_layout(
        citlali::config::TodOutputStream::ptc, 4, 7, rtc_output, ptc_output);

    EXPECT_EQ(rtc_layout.n_output_scans, 4);
    EXPECT_TRUE(rtc_layout.mini_output);
    EXPECT_TRUE(rtc_layout.outer_output);
    EXPECT_EQ(ptc_layout.n_output_scans, 7);
    EXPECT_FALSE(ptc_layout.mini_output);
    EXPECT_FALSE(ptc_layout.outer_output);
}

TEST(config_scaffold, writes_observation_tod_output_provenance) {
    struct OutputEngine {
        citlali::config::ReductionConfig typed_config;
        citlali::pipeline::TodOutputState tod_outputs;
        citlali::pipeline::OutputPathState output_paths;
        struct {
            Eigen::MatrixXI scan_indices;
        } telescope;
    } engine;
    auto &output = engine.typed_config.timestream.output;
    output.raw_time_chunk_enabled = true;
    output.processed_time_chunk_enabled = true;
    output.type = citlali::config::TodOutputType::both;
    output.raw_time_chunk.enabled = true;
    output.raw_time_chunk.mode =
        citlali::config::TodStreamOutputMode::mini_outer;
    output.processed_time_chunk.enabled = true;
    output.processed_time_chunk.mode =
        citlali::config::TodStreamOutputMode::full;
    engine.typed_config.timestream.chunking = {"number", 3.0, true};
    engine.telescope.scan_indices.resize(4, 3);
    engine.tod_outputs.rtc_scan_to_output_scan.resize(3);
    engine.tod_outputs.rtc_scan_to_output_scan << 0, -1, 1;
    engine.tod_outputs.ptc_scan_to_output_scan.resize(3);
    engine.tod_outputs.ptc_scan_to_output_scan << 0, 1, 2;
    engine.tod_outputs.n_rtc_output_scans = 2;
    engine.tod_outputs.n_ptc_output_scans = 3;
    engine.output_paths.tod_filename["rtc"] = "/data/rtc.nc";
    engine.output_paths.tod_filename["ptc"] = "/data/ptc.nc";
    const auto output_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_timestream_output_provenance_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    engine.output_paths.obsnum_dir_name = output_dir.string();

    citlali::pipeline::write_timestream_output_provenance_file(engine);

    const auto output_path =
        citlali::pipeline::timestream_output_provenance_path(output_dir);
    const auto stored = YAML::LoadFile(output_path.string());
    EXPECT_EQ(stored["schema_version"].as<std::string>(),
              "citlali-timestream-output-provenance-v1");
    EXPECT_EQ(stored["requested"]["chunking"]["mode"].as<std::string>(),
              "number");
    EXPECT_EQ(stored["effective"]["output_type"].as<std::string>(), "both");
    EXPECT_EQ(stored["effective"]["raw_time_chunk"]
                    ["selected_chunks_1based"].as<std::vector<int>>(),
              (std::vector<int>{1, 3}));
    EXPECT_EQ(stored["realized"]["n_scans"].as<int>(), 3);
    EXPECT_EQ(stored["realized"]["raw_time_chunk"]
                    ["n_output_scans"].as<int>(),
              2);
    EXPECT_EQ(stored["realized"]["files"]["rtc"].as<std::string>(),
              "/data/rtc.nc");
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    std::filesystem::remove_all(output_dir);
}

TEST(cli_runtime_setup, configures_runtime_threads) {
    FakeEngine engine;
    engine.typed_config.runtime.n_threads = 6;
    const auto plan =
        citlali::config::make_runtime_thread_plan(6, false);
    auto logger = std::make_shared<FakeLogger>();
    int omp_threads = 0;
    int eigen_threads = 0;
    int fftw_threads = 0;

    const auto realized = citlali::cli::configure_runtime_threads(
        plan, logger,
        [&](int n_threads) { omp_threads = n_threads; },
        [&](int n_threads) { eigen_threads = n_threads; },
        []() { return 1; },
        [&](int n_threads) { fftw_threads = n_threads; });

    EXPECT_EQ(omp_threads, 6);
    EXPECT_EQ(eigen_threads, 1);
    EXPECT_EQ(fftw_threads, 6);
    EXPECT_EQ(logger->info_calls, 1);
    EXPECT_EQ(logger->warn_calls, 0);
    EXPECT_EQ(realized.omp_threads, 6);
    EXPECT_EQ(realized.eigen_threads, 1);
    EXPECT_EQ(realized.fftw_plan_threads, 6);
    EXPECT_TRUE(realized.fftw_threads_initialized);
}

TEST(cli_runtime_setup, configures_single_fftw_thread_for_wiener_omp) {
    FakeEngine engine;
    engine.typed_config.runtime.n_threads = 6;
    const auto plan =
        citlali::config::make_runtime_thread_plan(6, true);
    auto logger = std::make_shared<FakeLogger>();
    int fftw_threads = 0;

    const auto realized = citlali::cli::configure_runtime_threads(
        plan, logger,
        [](int) {},
        [](int) {},
        []() { return 1; },
        [&](int n_threads) { fftw_threads = n_threads; });

    EXPECT_EQ(fftw_threads, 1);
    EXPECT_EQ(realized.fftw_plan_threads, 1);
}

TEST(cli_runtime_setup, warns_when_fftw_thread_init_fails) {
    FakeEngine engine;
    engine.typed_config.runtime.n_threads = 6;
    const auto plan =
        citlali::config::make_runtime_thread_plan(6, false);
    auto logger = std::make_shared<FakeLogger>();
    int fftw_threads = 0;

    const auto realized = citlali::cli::configure_runtime_threads(
        plan, logger,
        [](int) {},
        [](int) {},
        []() { return 0; },
        [&](int n_threads) { fftw_threads = n_threads; });

    EXPECT_EQ(fftw_threads, 0);
    EXPECT_EQ(logger->warn_calls, 1);
    EXPECT_EQ(realized.omp_threads, 6);
    EXPECT_EQ(realized.eigen_threads, 1);
    EXPECT_EQ(realized.fftw_plan_threads, 0);
    EXPECT_FALSE(realized.fftw_threads_initialized);
}

TEST(cli_reduction_runtime, prepares_reduction_runtime) {
    FakeInitialObservationTodProc todproc;
    todproc.engine().typed_config.runtime.verbose = true;
    FakeCitlaliConfig config;
    auto logger = std::make_shared<FakeLogger>();
    int enable_debug_calls = 0;
    int configure_threads_calls = 0;

    EXPECT_TRUE(citlali::cli::prepare_reduction_runtime(
        todproc, config, logger, [&]() { ++enable_debug_calls; },
        [&](auto &) { ++configure_threads_calls; }));

    EXPECT_EQ(todproc.engine().get_citlali_config_calls, 1);
    EXPECT_EQ(enable_debug_calls, 1);
    EXPECT_EQ(configure_threads_calls, 1);
}

TEST(cli_reduction_runtime, rejects_invalid_reduction_runtime) {
    FakeInitialObservationTodProc todproc;
    todproc.engine().config_diagnostics.missing_keys = {{"runtime"}};
    FakeCitlaliConfig config;
    auto logger = std::make_shared<FakeLogger>();
    int enable_debug_calls = 0;
    int configure_threads_calls = 0;

    EXPECT_FALSE(citlali::cli::prepare_reduction_runtime(
        todproc, config, logger, [&]() { ++enable_debug_calls; },
        [&](auto &) { ++configure_threads_calls; }));

    EXPECT_EQ(enable_debug_calls, 0);
    EXPECT_EQ(configure_threads_calls, 0);
    EXPECT_EQ(logger->error_calls, 2);
}

TEST(pipeline_map_geometry, stores_typed_map_vectors) {
    citlali::pipeline::ReductionMapGeometry<FakeGeometryTodProc> geometry;

    geometry.extents.push_back(7);
    geometry.coords.push_back(1.5);

    EXPECT_EQ(geometry.extents, (std::vector<int>{7}));
    EXPECT_EQ(geometry.coords, (std::vector<double>{1.5}));
}

TEST(cli_tod_processor_selection, selects_science_processor) {
    std::variant<std::monostate, FakeScienceTodProc, FakePointingTodProc,
                 FakeBeammapTodProc>
        todproc;
    FakeTodConfig config{77};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::cli::emplace_tod_processor_for_reduction_type<
        decltype(todproc), FakeScienceTodProc, FakePointingTodProc,
        FakeBeammapTodProc>(todproc, citlali::config::ReductionType::science,
                            config, logger));

    ASSERT_TRUE(std::holds_alternative<FakeScienceTodProc>(todproc));
    EXPECT_EQ(std::get<FakeScienceTodProc>(todproc).loaded_value, 77);
    EXPECT_EQ(logger->info_calls, 1);
}

TEST(cli_tod_processor_selection, exposes_reduction_type_config_key) {
    auto key = citlali::cli::reduction_type_config_key();

    EXPECT_EQ(std::get<0>(key), "runtime");
    EXPECT_EQ(std::get<1>(key), "reduction_type");
    EXPECT_EQ(citlali::cli::reduction_type_config_key_path(),
              (std::vector<std::string>{"runtime", "reduction_type"}));
}

TEST(cli_tod_processor_selection, reads_reduction_type_config) {
    FakeTodConfig config;
    config.reduction_type = "pointing";

    auto reduction_type = citlali::cli::read_reduction_type_config(config);

    ASSERT_TRUE(reduction_type.has_value());
    EXPECT_EQ(*reduction_type, "pointing");
}

TEST(cli_tod_processor_selection, returns_empty_reduction_type_when_missing) {
    FakeTodConfig config;
    config.has_reduction_type = false;

    EXPECT_FALSE(citlali::cli::read_reduction_type_config(config).has_value());
}

TEST(cli_tod_processor_selection, selects_pointing_processor) {
    std::variant<std::monostate, FakeScienceTodProc, FakePointingTodProc,
                 FakeBeammapTodProc>
        todproc;
    FakeTodConfig config{88};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::cli::emplace_tod_processor_for_reduction_type<
        decltype(todproc), FakeScienceTodProc, FakePointingTodProc,
        FakeBeammapTodProc>(todproc, citlali::config::ReductionType::pointing,
                            config, logger));

    ASSERT_TRUE(std::holds_alternative<FakePointingTodProc>(todproc));
    EXPECT_EQ(std::get<FakePointingTodProc>(todproc).loaded_value, 88);
}

TEST(cli_tod_processor_selection, rejects_unknown_processor_type) {
    std::variant<std::monostate, FakeScienceTodProc, FakePointingTodProc,
                 FakeBeammapTodProc>
        todproc;
    FakeTodConfig config{99};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_FALSE(citlali::cli::emplace_tod_processor_for_reduction_type<
        decltype(todproc), FakeScienceTodProc, FakePointingTodProc,
        FakeBeammapTodProc>(
            todproc, static_cast<citlali::config::ReductionType>(-1), config,
            logger));

    EXPECT_TRUE(std::holds_alternative<std::monostate>(todproc));
    EXPECT_EQ(logger->info_calls, 0);
}

TEST(pipeline_preflight, applies_flxscale_correction_when_present) {
    FakeEngine engine;
    engine.calib.apt["flxscale"].value = 2.0;
    FakeFlxscaleCorrection correction{1.5};
    FakeRawObs rawobs{&correction, "obs"};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::apply_flxscale_correction(
        engine, rawobs, logger));
    EXPECT_DOUBLE_EQ(engine.calib.apt["flxscale"].value, 3.0);
}

TEST(pipeline_preflight, skips_absent_flxscale_correction) {
    FakeEngine engine;
    engine.calib.apt["flxscale"].value = 2.0;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::apply_flxscale_correction(
        engine, rawobs, logger));
    EXPECT_DOUBLE_EQ(engine.calib.apt["flxscale"].value, 2.0);
}

TEST(pipeline_preflight, rejects_invalid_flxscale_correction) {
    FakeEngine engine;
    engine.calib.apt["flxscale"].value = 2.0;
    FakeFlxscaleCorrection correction{-1.0};
    FakeRawObs rawobs{&correction, "obs"};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_FALSE(citlali::pipeline::apply_flxscale_correction(
        engine, rawobs, logger));
    EXPECT_DOUBLE_EQ(engine.calib.apt["flxscale"].value, 2.0);
}

TEST(pipeline_preflight, rejects_missing_flxscale_column) {
    FakeEngine engine;
    FakeFlxscaleCorrection correction{1.5};
    FakeRawObs rawobs{&correction, "obs"};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_FALSE(citlali::pipeline::apply_flxscale_correction(
        engine, rawobs, logger));
}

TEST(pipeline_preflight, loads_array_properties_table) {
    FakeEngine engine;
    FakeRawObs rawobs;
    rawobs.apt.path = "/data/apt.ecsv";
    rawobs.kids_items = {
        {"/data/toltec0.nc", "nw0"},
        {"/data/toltec1.nc", "nw1"},
    };
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_array_properties_table(
        engine, rawobs, logger);

    EXPECT_EQ(engine.calib.get_apt_calls, 1);
    EXPECT_EQ(engine.calib.loaded_apt_path, "/data/apt.ecsv");
    EXPECT_EQ(engine.calib.loaded_raw_filenames,
              (std::vector<std::string>{"/data/toltec0.nc",
                                        "/data/toltec1.nc"}));
    EXPECT_EQ(engine.calib.loaded_interfaces,
              (std::vector<std::string>{"nw0", "nw1"}));
    EXPECT_EQ(logger->info_calls, 1);
}

TEST(pipeline_preflight,
     loads_array_properties_table_from_reference_wrapped_kidsdata) {
    FakeEngine engine;
    FakeReferenceWrappedRawObs rawobs;
    rawobs.apt.path = "/data/apt.ecsv";
    rawobs.kids_items = {
        {"/data/toltec0.nc", "nw0"},
        {"/data/toltec1.nc", "nw1"},
    };
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_array_properties_table(
        engine, rawobs, logger);

    EXPECT_EQ(engine.calib.get_apt_calls, 1);
    EXPECT_EQ(engine.calib.loaded_raw_filenames,
              (std::vector<std::string>{"/data/toltec0.nc",
                                        "/data/toltec1.nc"}));
    EXPECT_EQ(engine.calib.loaded_interfaces,
              (std::vector<std::string>{"nw0", "nw1"}));
}

TEST(pipeline_preflight, configures_non_beammap_observation_calibration) {
    FakeCalibrationTodProc todproc;
    FakeRawObs rawobs;
    rawobs.astrometry.value = "astro";
    rawobs.photometry.value = "photo";
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::configure_observation_calibration<false>(
        todproc, rawobs, logger);

    EXPECT_EQ(todproc.engine().get_astrometry_config_calls, 1);
    EXPECT_EQ(todproc.engine().loaded_astrometry_config, "astro");
    EXPECT_EQ(todproc.engine().get_photometry_config_calls, 0);
    EXPECT_EQ(todproc.get_apt_from_files_calls, 0);
    EXPECT_EQ(todproc.engine().calib.get_apt_calls, 1);
}

TEST(pipeline_preflight, configures_beammap_detector_calibration_from_files) {
    FakeCalibrationTodProc todproc;
    todproc.engine().typed_config.mapmaking.grouping =
        citlali::config::MapGrouping::detector;
    FakeRawObs rawobs;
    rawobs.astrometry.value = "astro";
    rawobs.photometry.value = "photo";
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::configure_observation_calibration<true>(
        todproc, rawobs, logger);

    EXPECT_EQ(todproc.engine().get_astrometry_config_calls, 1);
    EXPECT_EQ(todproc.engine().get_photometry_config_calls, 1);
    EXPECT_EQ(todproc.engine().loaded_photometry_config, "photo");
    EXPECT_EQ(todproc.get_apt_from_files_calls, 1);
    EXPECT_EQ(todproc.engine().calib.get_apt_calls, 0);
}

TEST(pipeline_preflight, configures_beammap_array_calibration_from_apt) {
    FakeCalibrationTodProc todproc;
    todproc.engine().typed_config.mapmaking.grouping =
        citlali::config::MapGrouping::array;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::configure_observation_calibration<true>(
        todproc, rawobs, logger);

    EXPECT_EQ(todproc.engine().get_photometry_config_calls, 1);
    EXPECT_EQ(todproc.get_apt_from_files_calls, 0);
    EXPECT_EQ(todproc.engine().calib.get_apt_calls, 1);
}

TEST(pipeline_preflight, skips_reduction_calibration_when_not_needed) {
    FakeCalibrationTodProc todproc;
    FakeRawObs rawobs;
    std::vector<FakeRawObsMeta> rawobs_kids_meta = {{122.0, 102}};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(
        citlali::pipeline::configure_reduction_observation_calibration_if_needed<
            false>(todproc, rawobs, rawobs_kids_meta, false, logger));

    EXPECT_EQ(todproc.engine().get_astrometry_config_calls, 0);
    EXPECT_EQ(todproc.engine().calib.get_apt_calls, 0);
    EXPECT_DOUBLE_EQ(todproc.engine().telescope.fsmp, 100.0);
}

TEST(pipeline_preflight, configures_reduction_calibration_when_needed) {
    FakeCalibrationTodProc todproc;
    FakeRawObs rawobs;
    rawobs.astrometry.value = "astro";
    std::vector<FakeRawObsMeta> rawobs_kids_meta = {{122.0, 102}};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(
        citlali::pipeline::configure_reduction_observation_calibration_if_needed<
            false>(todproc, rawobs, rawobs_kids_meta, true, logger));

    EXPECT_EQ(todproc.engine().get_astrometry_config_calls, 1);
    EXPECT_EQ(todproc.engine().loaded_astrometry_config, "astro");
    EXPECT_EQ(todproc.engine().calib.get_apt_calls, 1);
    EXPECT_DOUBLE_EQ(todproc.engine().telescope.fsmp, 122.0);
}

TEST(pipeline_preflight, resets_simulated_observation_indices) {
    FakeEngine engine;
    FakeRawObs rawobs;
    rawobs.kids_items = {
        {"a.nc", "nw0"},
        {"b.nc", "nw1"},
        {"c.nc", "nw2"},
    };

    citlali::pipeline::reset_simulated_observation_indices(engine, rawobs);

    EXPECT_EQ(engine.alignment.start_indices,
              (std::vector<Eigen::Index>{0, 0, 0, 0, 0, 0}));
    EXPECT_TRUE(engine.alignment.end_indices.empty());
    EXPECT_EQ(engine.alignment.hwpr_start_index, 0);
    EXPECT_EQ(engine.alignment.hwpr_end_index, 0);
}

TEST(pipeline_preflight, leaves_hwpr_indices_when_hwpr_disabled) {
    FakeEngine engine;
    engine.calib.run_hwpr = false;
    FakeRawObs rawobs;

    citlali::pipeline::reset_simulated_observation_indices(engine, rawobs);

    EXPECT_EQ(engine.alignment.start_indices,
              (std::vector<Eigen::Index>{0, 0, 0, 0}));
    EXPECT_TRUE(engine.alignment.end_indices.empty());
    EXPECT_EQ(engine.alignment.hwpr_start_index, -1);
    EXPECT_EQ(engine.alignment.hwpr_end_index, -1);
}

TEST(pipeline_preflight, loads_and_aligns_telescope_data) {
    FakeTelescopeTodProc todproc;
    todproc.engine().typed_config.timestream.chunking.mode = "duration";
    todproc.engine().typed_config.timestream.chunking.value = 12.5;
    todproc.engine().typed_config.timestream.chunking.force = true;
    FakeRawObs rawobs;
    rawobs.tel.path = "/data/tel.nc";
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_and_align_telescope_data(
        todproc, rawobs, logger);

    EXPECT_EQ(todproc.engine().telescope.get_tel_data_calls, 1);
    EXPECT_EQ(todproc.engine().telescope.loaded_tel_path, "/data/tel.nc");
    EXPECT_EQ(todproc.engine().telescope.scan_chunk_mode, "duration");
    EXPECT_DOUBLE_EQ(todproc.engine().telescope.scan_chunk_value, 12.5);
    EXPECT_TRUE(todproc.engine().telescope.scan_force_chunk);
    EXPECT_EQ(todproc.align_timestreams_calls, 1);
    EXPECT_EQ(todproc.align_timestreams_gaps_calls, 0);
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(pipeline_preflight, aligns_telescope_data_over_gaps) {
    FakeTelescopeTodProc todproc;
    todproc.engine().typed_config.runtime.interp_over_gaps = true;
    sync_fake_runtime_provenance(todproc.engine());
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_and_align_telescope_data(
        todproc, rawobs, logger);

    EXPECT_EQ(todproc.align_timestreams_calls, 0);
    EXPECT_EQ(todproc.align_timestreams_gaps_calls, 1);
}

TEST(pipeline_preflight, resets_indices_for_simulated_telescope_data) {
    FakeTelescopeTodProc todproc;
    todproc.engine().telescope.sim_obs = true;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_and_align_telescope_data(
        todproc, rawobs, logger);

    EXPECT_EQ(todproc.align_timestreams_calls, 0);
    EXPECT_EQ(todproc.align_timestreams_gaps_calls, 0);
    EXPECT_EQ(todproc.engine().alignment.start_indices,
              (std::vector<Eigen::Index>{0, 0, 0, 0}));
    EXPECT_TRUE(todproc.engine().alignment.end_indices.empty());
    EXPECT_EQ(logger->info_calls, 1);
}

TEST(pipeline_preflight, calculates_telescope_pointing) {
    FakeTelescopeTodProc todproc;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::calculate_telescope_pointing(todproc, logger);

    EXPECT_EQ(todproc.engine().telescope.calc_tan_pointing_calls, 1);
    EXPECT_EQ(todproc.interp_pointing_calls, 1);
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(pipeline_preflight, skips_telescope_reload_when_not_needed) {
    FakeTelescopeTodProc todproc;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_and_point_telescope_data_if_needed(
        todproc, rawobs, false, logger);

    EXPECT_EQ(todproc.engine().telescope.get_tel_data_calls, 0);
    EXPECT_EQ(todproc.engine().telescope.calc_tan_pointing_calls, 0);
    EXPECT_EQ(todproc.interp_pointing_calls, 0);
    EXPECT_EQ(logger->info_calls, 0);
}

TEST(pipeline_preflight, reloads_and_points_telescope_when_needed) {
    FakeTelescopeTodProc todproc;
    FakeRawObs rawobs;
    rawobs.tel.path = "/data/tel.nc";
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_and_point_telescope_data_if_needed(
        todproc, rawobs, true, logger);

    EXPECT_EQ(todproc.engine().telescope.get_tel_data_calls, 1);
    EXPECT_EQ(todproc.engine().telescope.loaded_tel_path, "/data/tel.nc");
    EXPECT_EQ(todproc.engine().telescope.calc_tan_pointing_calls, 1);
    EXPECT_EQ(todproc.interp_pointing_calls, 1);
    EXPECT_EQ(logger->info_calls, 4);
}

TEST(pipeline_preflight, calculates_scan_indices) {
    FakeEngine engine;
    engine.typed_config.timestream.chunking.mode = "number";
    engine.typed_config.timestream.chunking.value = 7.0;
    engine.typed_config.timestream.chunking.force = true;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::calculate_scan_indices(engine, logger);

    EXPECT_EQ(engine.telescope.calc_scan_indices_calls, 1);
    EXPECT_EQ(engine.telescope.scan_chunk_mode, "number");
    EXPECT_DOUBLE_EQ(engine.telescope.scan_chunk_value, 7.0);
    EXPECT_TRUE(engine.telescope.scan_force_chunk);
    EXPECT_EQ(logger->info_calls, 1);
}

TEST(pipeline_preflight, skips_scan_indices_when_not_needed) {
    FakeEngine engine;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::calculate_scan_indices_if_needed(
        engine, false, logger);

    EXPECT_EQ(engine.telescope.calc_scan_indices_calls, 0);
    EXPECT_EQ(logger->info_calls, 0);
}

TEST(pipeline_preflight, calculates_scan_indices_when_needed) {
    FakeEngine engine;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::calculate_scan_indices_if_needed(
        engine, true, logger);

    EXPECT_EQ(engine.telescope.calc_scan_indices_calls, 1);
    EXPECT_EQ(logger->info_calls, 1);
}

TEST(pipeline_preflight, loads_rawobs_kids_meta) {
    FakeKidsProc kidsproc;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    const auto meta = citlali::pipeline::load_rawobs_kids_meta(
        kidsproc, rawobs, logger);

    EXPECT_EQ(kidsproc.get_rawobs_meta_calls, 1);
    ASSERT_EQ(meta.size(), 2U);
    EXPECT_DOUBLE_EQ(meta.back().fsmp, 122.0);
    EXPECT_EQ(meta.back().obsid, 102);
    EXPECT_EQ(logger->debug_calls, 1);
}

TEST(pipeline_preflight, makes_kids_data_proc_from_config) {
    FakeCitlaliConfig config;

    auto kidsproc =
        citlali::pipeline::make_kids_data_proc<FakeKidsProc>(config);

    EXPECT_EQ(config.get_config_calls, 1);
    EXPECT_EQ(config.requested_key, "kids");
    EXPECT_EQ(kidsproc.loaded_config_value, 42);
}

TEST(pipeline_preflight, loads_valid_engine_config) {
    FakeEngine engine;
    FakeCitlaliConfig config;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::load_and_validate_engine_config(
        engine, config, logger));

    EXPECT_EQ(engine.get_citlali_config_calls, 1);
    EXPECT_EQ(logger->info_calls, 1);
    EXPECT_EQ(logger->error_calls, 0);
}

TEST(pipeline_preflight, rejects_invalid_engine_config) {
    FakeEngine engine;
    engine.config_diagnostics.missing_keys = {{"runtime"}};
    engine.config_diagnostics.invalid_keys = {{"mapmaking", "pixel_size"}};
    FakeCitlaliConfig config;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_FALSE(citlali::pipeline::load_and_validate_engine_config(
        engine, config, logger));

    EXPECT_EQ(engine.get_citlali_config_calls, 1);
    EXPECT_EQ(logger->error_calls, 2);
}

TEST(pipeline_preflight, configures_verbose_logging_when_requested) {
    FakeEngine engine;
    engine.typed_config.runtime.verbose = true;
    engine.runtime_config_provenance =
        citlali::config::make_runtime_config_provenance(
            engine.typed_config.runtime, false);
    auto logger = std::make_shared<FakeLogger>();
    int enable_debug_calls = 0;

    citlali::pipeline::configure_verbose_logging_if_requested(
        engine, logger, [&]() { ++enable_debug_calls; });

    EXPECT_EQ(enable_debug_calls, 1);
    EXPECT_EQ(logger->debug_calls, 1);
}

TEST(pipeline_preflight, skips_verbose_logging_when_not_requested) {
    FakeEngine engine;
    engine.typed_config.runtime.verbose = false;
    auto logger = std::make_shared<FakeLogger>();
    int enable_debug_calls = 0;

    citlali::pipeline::configure_verbose_logging_if_requested(
        engine, logger, [&]() { ++enable_debug_calls; });

    EXPECT_EQ(enable_debug_calls, 0);
    EXPECT_EQ(logger->debug_calls, 0);
}

TEST(pipeline_preflight, checks_observation_inputs) {
    FakeTelescopeTodProc todproc;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::check_observation_inputs(
        todproc, rawobs, logger);

    EXPECT_EQ(todproc.check_inputs_calls, 1);
    EXPECT_EQ(logger->debug_calls, 1);
}

TEST(pipeline_preflight, updates_sample_rate_from_rawobs_meta) {
    FakeEngine engine;
    std::vector<FakeRawObsMeta> rawobs_kids_meta = {
        {75.0, 101},
        {122.0, 102},
    };
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::update_sample_rate_from_rawobs_meta(
        engine, rawobs_kids_meta, logger);

    EXPECT_DOUBLE_EQ(engine.telescope.fsmp, 122.0);
    EXPECT_EQ(logger->debug_calls, 1);
}

TEST(pipeline_preflight, loads_raw_detector_diagnostics) {
    FakeTelescopeTodProc todproc;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_raw_detector_diagnostics(
        todproc, rawobs, logger);

    EXPECT_EQ(todproc.get_tone_freqs_from_files_calls, 1);
    EXPECT_EQ(todproc.get_adc_snap_from_files_calls, 1);
    EXPECT_EQ(logger->debug_calls, 2);
}

TEST(pipeline_preflight, skips_adc_snap_for_simulated_observations) {
    FakeTelescopeTodProc todproc;
    todproc.engine().telescope.sim_obs = true;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_raw_detector_diagnostics(
        todproc, rawobs, logger);

    EXPECT_EQ(todproc.get_tone_freqs_from_files_calls, 1);
    EXPECT_EQ(todproc.get_adc_snap_from_files_calls, 0);
    EXPECT_EQ(logger->debug_calls, 1);
}

TEST(pipeline_preflight, calculates_flux_calibration) {
    FakeEngine engine;
    engine.omb.sig_unit = "Jy/pixel";
    engine.omb.pixel_size_rad = 0.001;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::calculate_flux_calibration(engine, logger);

    EXPECT_EQ(engine.calib.calc_flux_calibration_calls, 1);
    EXPECT_EQ(engine.calib.loaded_flux_units, "Jy/pixel");
    EXPECT_DOUBLE_EQ(engine.calib.loaded_flux_pixel_size_rad, 0.001);
    EXPECT_EQ(logger->info_calls, 1);
}

TEST(pipeline_preflight, configures_sample_rate_without_downsample) {
    FakeEngine engine;
    engine.telescope.fsmp = 122.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
    EXPECT_DOUBLE_EQ(engine.telescope.d_fsmp, 122.0);
}

TEST(pipeline_preflight, configures_sample_rate_with_downsample_factor) {
    FakeEngine engine;
    engine.telescope.fsmp = 100.0;
    auto &raw = engine.typed_config.timestream.raw_time_chunk;
    raw.downsample.enabled = true;
    raw.downsample.factor = 4;
    raw.filter.freq_high_Hz = 10.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
    EXPECT_DOUBLE_EQ(engine.telescope.d_fsmp, 25.0);
}

TEST(pipeline_preflight, derives_downsample_factor_from_frequency) {
    FakeEngine engine;
    engine.telescope.fsmp = 100.0;
    auto &raw = engine.typed_config.timestream.raw_time_chunk;
    raw.downsample.enabled = true;
    raw.downsample.factor = 0;
    raw.downsample.downsampled_freq_Hz = 30.0;
    raw.filter.freq_high_Hz = 10.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
    EXPECT_EQ(raw.downsample.factor, 3);
    EXPECT_EQ(engine.rtcproc.downsampler.factor, 3);
    EXPECT_DOUBLE_EQ(engine.telescope.d_fsmp, 100.0 / 3.0);
}

TEST(pipeline_preflight, rejects_invalid_downsample_frequency) {
    FakeEngine engine;
    auto &downsample =
        engine.typed_config.timestream.raw_time_chunk.downsample;
    downsample.enabled = true;
    downsample.factor = 0;
    downsample.downsampled_freq_Hz = 0.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_FALSE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
}

TEST(pipeline_preflight, rejects_downsample_frequency_above_sample_rate) {
    FakeEngine engine;
    engine.telescope.fsmp = 100.0;
    auto &downsample =
        engine.typed_config.timestream.raw_time_chunk.downsample;
    downsample.enabled = true;
    downsample.factor = 0;
    downsample.downsampled_freq_Hz = 200.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_FALSE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
}

TEST(pipeline_preflight, rejects_downsample_filter_above_nyquist) {
    FakeEngine engine;
    engine.telescope.fsmp = 100.0;
    auto &raw = engine.typed_config.timestream.raw_time_chunk;
    raw.downsample.enabled = true;
    raw.downsample.factor = 4;
    raw.filter.freq_high_Hz = 20.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_FALSE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
}

TEST(pipeline_preflight, sample_rate_policy_uses_typed_downsample_config) {
    FakeEngine engine;
    engine.telescope.fsmp = 100.0;
    auto &raw = engine.typed_config.timestream.raw_time_chunk;
    raw.downsample.enabled = true;
    raw.downsample.factor = 5;
    raw.filter.freq_high_Hz = 8.0;
    engine.rtcproc.run_downsample = false;
    engine.rtcproc.downsampler.factor = 2;
    engine.rtcproc.filter.freq_high_Hz = 40.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
    EXPECT_DOUBLE_EQ(engine.telescope.d_fsmp, 20.0);
}

TEST(pipeline_preflight, raw_filter_policy_uses_typed_config) {
    FakeEngine engine;
    engine.telescope.fsmp = 100.0;
    auto &raw = engine.typed_config.timestream.raw_time_chunk;
    raw.kernel.enabled = true;
    raw.flux_calibration_enabled = true;
    raw.extinction_correction_enabled = true;
    raw.filter.enabled = true;
    raw.filter.notch.enabled = true;
    raw.iir_filter.enabled = true;
    raw.iir_filter.freq_Hz = 10.0;
    engine.rtcproc.run_downsample = false;

    EXPECT_TRUE(citlali::pipeline::raw_kernel_enabled(engine));
    EXPECT_TRUE(citlali::pipeline::raw_flux_calibration_enabled(engine));
    EXPECT_TRUE(
        citlali::pipeline::raw_extinction_correction_enabled(engine));
    EXPECT_TRUE(citlali::pipeline::raw_fir_filter_enabled(engine));
    EXPECT_TRUE(citlali::pipeline::raw_notch_filter_enabled(engine));
    EXPECT_TRUE(citlali::pipeline::raw_iir_filter_enabled(engine));
    EXPECT_DOUBLE_EQ(citlali::pipeline::raw_iir_filter_frequency_hz(engine),
                     10.0);
    EXPECT_TRUE(citlali::pipeline::raw_iir_filter_below_nyquist(engine));

    raw.filter.enabled = false;
    raw.iir_filter.freq_Hz = 50.0;
    EXPECT_FALSE(citlali::pipeline::raw_fir_filter_enabled(engine));
    EXPECT_FALSE(citlali::pipeline::raw_notch_filter_enabled(engine));
    EXPECT_FALSE(citlali::pipeline::raw_iir_filter_below_nyquist(engine));
}

TEST(pipeline_preflight, source_protection_activation_uses_typed_config) {
    struct FakeRtcProc {
        struct {
            bool source_protection_enabled = false;
            double source_protection_radius_arcsec = 0.0;
        } despiker;
    } rtcproc;
    struct FakePtcProc {
        struct {
            bool source_protection_enabled = false;
            double source_protection_radius_arcsec = 0.0;
        } second_pass_local;
    } ptcproc;
    citlali::config::TimestreamConfig config;
    auto &raw = config.raw_time_chunk.despike;
    raw.enabled = true;
    raw.source_protection.enabled = true;
    raw.source_protection.radius_arcsec = 24.0;
    auto &processed =
        config.processed_time_chunk.flagging.second_pass_local;
    processed.enabled = true;
    processed.source_protection.enabled = true;
    processed.source_protection.radius_arcsec = 31.0;
    auto logger = std::make_shared<FakeLogger>();

    const auto pointing_resolution =
        citlali::pipeline::resolve_source_protection(
            citlali::config::ReductionType::pointing, config);
    EXPECT_TRUE(pointing_resolution.source_aware_reduction);
    EXPECT_TRUE(pointing_resolution.raw_activation_requested);
    EXPECT_TRUE(pointing_resolution.processed_activation_requested);
    EXPECT_TRUE(pointing_resolution.raw_active);
    EXPECT_TRUE(pointing_resolution.processed_active);

    citlali::pipeline::apply_source_protection_activation(
        citlali::config::ReductionType::pointing, rtcproc, ptcproc, config,
        logger);
    EXPECT_TRUE(raw.source_protection.active);
    EXPECT_TRUE(processed.source_protection.active);
    EXPECT_TRUE(rtcproc.despiker.source_protection_enabled);
    EXPECT_DOUBLE_EQ(rtcproc.despiker.source_protection_radius_arcsec, 24.0);
    EXPECT_TRUE(ptcproc.second_pass_local.source_protection_enabled);
    EXPECT_DOUBLE_EQ(
        ptcproc.second_pass_local.source_protection_radius_arcsec, 31.0);

    const auto science_resolution =
        citlali::pipeline::resolve_source_protection(
            citlali::config::ReductionType::science, config);
    EXPECT_FALSE(science_resolution.source_aware_reduction);
    EXPECT_TRUE(science_resolution.raw_activation_requested);
    EXPECT_TRUE(science_resolution.processed_activation_requested);
    EXPECT_FALSE(science_resolution.raw_active);
    EXPECT_FALSE(science_resolution.processed_active);

    citlali::pipeline::apply_source_protection_activation(
        citlali::config::ReductionType::science, rtcproc, ptcproc, config,
        logger);
    EXPECT_FALSE(raw.source_protection.active);
    EXPECT_FALSE(processed.source_protection.active);
    EXPECT_FALSE(rtcproc.despiker.source_protection_enabled);
    EXPECT_FALSE(ptcproc.second_pass_local.source_protection_enabled);
}

TEST(pipeline_preflight, loads_hwpr_data_for_polarized_observation) {
    FakeEngine engine;
    engine.rtcproc.run_polarization = true;
    engine.telescope.sim_obs = true;
    FakeRawObs rawobs;
    rawobs.hwp = FakeHwpData{"hwpr.nc"};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_hwpr_data_if_requested(engine, rawobs, logger);

    EXPECT_TRUE(engine.calib.run_hwpr);
    EXPECT_TRUE(engine.calib.loaded_hwpr);
    EXPECT_EQ(engine.calib.loaded_hwpr_filepath, "hwpr.nc");
    EXPECT_TRUE(engine.calib.loaded_hwpr_sim_obs);
}

TEST(pipeline_preflight, ignores_hwpr_when_configured) {
    FakeEngine engine;
    engine.rtcproc.run_polarization = true;
    engine.calib.ignore_hwpr = "true";
    FakeRawObs rawobs;
    rawobs.hwp = FakeHwpData{"hwpr.nc"};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_hwpr_data_if_requested(engine, rawobs, logger);

    EXPECT_FALSE(engine.calib.run_hwpr);
    EXPECT_FALSE(engine.calib.loaded_hwpr);
}

TEST(pipeline_preflight, ignores_null_hwpr_filepath) {
    FakeEngine engine;
    engine.rtcproc.run_polarization = true;
    FakeRawObs rawobs;
    rawobs.hwp = FakeHwpData{"null"};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_hwpr_data_if_requested(engine, rawobs, logger);

    EXPECT_FALSE(engine.calib.run_hwpr);
    EXPECT_FALSE(engine.calib.loaded_hwpr);
}

TEST(pipeline_preflight, leaves_hwpr_state_when_not_polarized) {
    FakeEngine engine;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_hwpr_data_if_requested(engine, rawobs, logger);

    EXPECT_TRUE(engine.calib.run_hwpr);
    EXPECT_FALSE(engine.calib.loaded_hwpr);
}

TEST(pipeline_preflight, leaves_map_center_when_not_configured) {
    FakeEngine engine;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::overwrite_map_center_if_configured(engine, logger);

    EXPECT_TRUE(engine.telescope.tel_header.empty());
}

TEST(pipeline_preflight, overwrites_map_center_when_configured) {
    FakeEngine engine;
    engine.omb.crval_config = {180.0, 45.0};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::overwrite_map_center_if_configured(engine, logger);

    EXPECT_TRUE(engine.telescope.tel_header["Header.Source.Ra"].set);
    EXPECT_TRUE(engine.telescope.tel_header["Header.Source.Dec"].set);
    EXPECT_NEAR(engine.telescope.tel_header["Header.Source.Ra"].value,
                citlali::pipeline::degrees_to_radians(180.0), 1e-12);
    EXPECT_NEAR(engine.telescope.tel_header["Header.Source.Dec"].value,
                citlali::pipeline::degrees_to_radians(45.0), 1e-12);
}

TEST(pipeline_preflight, updates_observation_exposure_time) {
    FakeEngine engine;
    engine.telescope.tel_data["TelTime"].values = {10.0, 12.5, 14.0};

    citlali::pipeline::update_observation_exposure_time(engine);

    EXPECT_DOUBLE_EQ(engine.omb.exposure_time, 4.0);
    EXPECT_DOUBLE_EQ(engine.cmb.exposure_time, 0.0);
}

TEST(pipeline_preflight, accumulates_observation_exposure_time_for_coadd) {
    FakeEngine engine;
    engine.typed_config.coadd.enabled = true;
    engine.cmb.exposure_time = 3.0;
    engine.telescope.tel_data["TelTime"].values = {10.0, 12.5, 14.0};

    citlali::pipeline::update_observation_exposure_time(engine);

    EXPECT_DOUBLE_EQ(engine.omb.exposure_time, 4.0);
    EXPECT_DOUBLE_EQ(engine.cmb.exposure_time, 7.0);
}

TEST(pipeline_preflight, appends_observation_date) {
    FakeEngine engine;
    engine.observation_dates.date_obs = {"old"};

    citlali::pipeline::append_observation_date(engine, std::string{"new"});

    EXPECT_EQ(engine.observation_dates.date_obs, (std::vector<std::string>{"old", "new"}));
}

TEST(pipeline_preflight, derives_date_obs_from_telescope_time) {
    FakeEngine engine;
    engine.telescope.tel_data["TelTime"].values = {123.0, 456.0};

    auto date_obs = citlali::pipeline::date_obs_from_telescope_time(
        engine, [](double unix_time) {
            return std::string{"utc:"} + std::to_string(
                static_cast<int>(unix_time));
        });

    EXPECT_EQ(date_obs, "utc:123");
}

TEST(pipeline_preflight, configures_non_fruit_loop_as_single_iteration) {
    FakeEngine engine;
    auto &fruit_loops = engine.typed_config.timestream.fruit_loops;
    fruit_loops.enabled = false;
    fruit_loops.max_iters = 5;
    fruit_loops.save_all_iters = false;
    auto logger = std::make_shared<FakeLogger>();

    const auto resolution =
        citlali::pipeline::resolve_fruit_loop_iteration_policy(
            fruit_loops, citlali::config::ReductionType::science);

    EXPECT_EQ(fruit_loops.max_iters, 5);
    EXPECT_FALSE(fruit_loops.save_all_iters);
    EXPECT_EQ(resolution.effective_max_iters, 1);
    EXPECT_TRUE(resolution.effective_save_all_iters);
    EXPECT_TRUE(resolution.forced_single_iteration_while_disabled);
    EXPECT_FALSE(resolution.forced_single_iteration_for_beammap);

    citlali::pipeline::configure_fruit_loop_iteration_policy(
        engine, logger);

    EXPECT_EQ(fruit_loops.max_iters, 1);
    EXPECT_TRUE(fruit_loops.save_all_iters);
    EXPECT_EQ(engine.ptcproc.fruit_loops_iters, 1);
    EXPECT_TRUE(engine.ptcproc.save_all_iters);
    EXPECT_EQ(logger->warn_calls, 0);
}

TEST(pipeline_preflight, configures_beammap_fruit_loop_as_single_iteration) {
    FakeEngine engine;
    engine.typed_config.runtime.reduction_type =
        citlali::config::ReductionType::beammap;
    sync_fake_runtime_provenance(engine);
    auto &fruit_loops = engine.typed_config.timestream.fruit_loops;
    fruit_loops.enabled = true;
    fruit_loops.max_iters = 5;
    fruit_loops.save_all_iters = false;
    auto logger = std::make_shared<FakeLogger>();

    const auto resolution =
        citlali::pipeline::resolve_fruit_loop_iteration_policy(
            fruit_loops, citlali::config::ReductionType::beammap);

    EXPECT_FALSE(resolution.forced_single_iteration_while_disabled);
    EXPECT_TRUE(resolution.forced_single_iteration_for_beammap);

    citlali::pipeline::configure_fruit_loop_iteration_policy(
        engine, logger);

    EXPECT_EQ(fruit_loops.max_iters, 1);
    EXPECT_TRUE(fruit_loops.save_all_iters);
    EXPECT_EQ(engine.ptcproc.fruit_loops_iters, 1);
    EXPECT_TRUE(engine.ptcproc.save_all_iters);
}

TEST(pipeline_preflight, warns_when_fruit_loop_noise_maps_disabled) {
    FakeEngine engine;
    engine.typed_config.timestream.fruit_loops.enabled = true;
    engine.typed_config.noise.enabled = false;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::configure_fruit_loop_iteration_policy(
        engine, logger);

    EXPECT_EQ(logger->warn_calls, 1);
}

TEST(pipeline_preflight, preserves_science_fruit_loop_iteration_policy) {
    FakeEngine engine;
    engine.typed_config.runtime.reduction_type =
        citlali::config::ReductionType::science;
    sync_fake_runtime_provenance(engine);
    auto &fruit_loops = engine.typed_config.timestream.fruit_loops;
    fruit_loops.enabled = true;
    fruit_loops.max_iters = 5;
    fruit_loops.save_all_iters = false;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::configure_fruit_loop_iteration_policy(
        engine, logger);

    EXPECT_EQ(fruit_loops.max_iters, 5);
    EXPECT_FALSE(fruit_loops.save_all_iters);
    EXPECT_EQ(engine.ptcproc.fruit_loops_iters, 5);
    EXPECT_FALSE(engine.ptcproc.save_all_iters);
    EXPECT_EQ(logger->warn_calls, 0);
}

TEST(pipeline_fruit_loop_paths, derives_obsnum_raw_map_dir) {
    EXPECT_EQ(citlali::pipeline::fruit_loop_map_dir(
                  "/data/redu01", "obsnum/raw", "123456"),
              "/data/redu01/123456/raw/");
}

TEST(pipeline_fruit_loop_paths, derives_obsnum_filtered_map_dir) {
    EXPECT_EQ(citlali::pipeline::fruit_loop_map_dir(
                  "/data/redu01", "obsnum/filtered", "123456"),
              "/data/redu01/123456/filtered/");
}

TEST(pipeline_fruit_loop_paths, derives_coadd_map_dirs) {
    EXPECT_EQ(citlali::pipeline::fruit_loop_map_dir(
                  "/data/redu01", "coadd/raw", "123456"),
              "/data/redu01/coadded/raw/");
    EXPECT_EQ(citlali::pipeline::fruit_loop_map_dir(
                  "/data/redu01", "coadd/filtered", "123456"),
              "/data/redu01/coadded/filtered/");
}

TEST(pipeline_fruit_loop_paths, preserves_empty_path_for_unknown_type) {
    EXPECT_EQ(citlali::pipeline::fruit_loop_map_dir(
                  "/data/redu01", "unknown", "123456"),
              "");
}

TEST(pipeline_fruit_loop_paths, derives_previous_iteration_map_dir) {
    EXPECT_EQ(citlali::pipeline::previous_fruit_loop_reduction_dir_name(2),
              "redu01");
    EXPECT_EQ(citlali::pipeline::previous_fruit_loop_map_dir(
                  "/data", 12, "obsnum/raw", "123456"),
              "/data/redu11/123456/raw/");
}

TEST(pipeline_iteration_lifecycle, detects_pending_fruit_loop_iteration) {
    FakeIterationEngine engine;
    engine.iteration.fruit_iter = 1;
    engine.typed_config.timestream.fruit_loops.max_iters = 3;

    EXPECT_TRUE(citlali::pipeline::fruit_loop_iteration_pending(
        engine, false));
}

TEST(pipeline_iteration_lifecycle, stops_when_fruit_loops_converge) {
    FakeIterationEngine engine;
    engine.iteration.fruit_iter = 1;
    engine.typed_config.timestream.fruit_loops.max_iters = 3;

    EXPECT_FALSE(citlali::pipeline::fruit_loop_iteration_pending(
        engine, true));
}

TEST(pipeline_iteration_lifecycle, stops_at_iteration_limit) {
    FakeIterationEngine engine;
    engine.iteration.fruit_iter = 3;
    engine.typed_config.timestream.fruit_loops.max_iters = 3;

    EXPECT_FALSE(citlali::pipeline::fruit_loop_iteration_pending(
        engine, false));
}

TEST(pipeline_iteration_lifecycle, begins_non_fruit_loop_iteration) {
    FakeIterationEngine engine;
    engine.iteration.fruit_iter = 0;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::begin_fruit_loop_iteration(engine, logger);

    EXPECT_EQ(engine.ptcproc.begin_weight_validation_iter, 0);
    EXPECT_EQ(engine.learning.begin_calls, 1);
    EXPECT_EQ(engine.learning.begin_iter, 0);
    EXPECT_FALSE(engine.learning.source_model_available);
    EXPECT_EQ(engine.learning.redu_type, "science");
    EXPECT_EQ(logger->info_calls, 0);
}

TEST(pipeline_iteration_lifecycle, begins_fruit_loop_iteration_with_source_model) {
    FakeIterationEngine engine;
    engine.iteration.fruit_iter = 1;
    engine.typed_config.timestream.fruit_loops.enabled = true;
    engine.learning.enabled = true;
    engine.learning.diagnostics = true;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::begin_fruit_loop_iteration(engine, logger);

    EXPECT_EQ(engine.ptcproc.begin_weight_validation_iter, 1);
    EXPECT_EQ(engine.learning.begin_calls, 1);
    EXPECT_TRUE(engine.learning.source_model_available);
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(pipeline_iteration_lifecycle, uses_configured_fruit_loop_path_as_source_model) {
    FakeIterationEngine engine;
    engine.iteration.fruit_iter = 0;
    engine.typed_config.timestream.fruit_loops.enabled = true;
    engine.typed_config.timestream.fruit_loops.path = "/data/redu00";
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::begin_fruit_loop_iteration(engine, logger);

    EXPECT_TRUE(engine.learning.source_model_available);
    EXPECT_EQ(logger->info_calls, 1);
}

TEST(pipeline_iteration_lifecycle, finalizes_iteration) {
    FakeIterationEngine engine;
    engine.iteration.fruit_iter = 3;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::finalize_fruit_loop_iteration(engine, logger);

    EXPECT_EQ(engine.ptcproc.finalize_weight_validation_iter, 3);
    EXPECT_EQ(engine.learning.finalize_calls, 1);
    EXPECT_EQ(engine.learning.finalize_iter, 3);
    EXPECT_EQ(engine.write_learning_summary_calls, 1);
    EXPECT_EQ(logger->info_calls, 0);
}

TEST(pipeline_iteration_lifecycle, logs_finalize_diagnostics_when_enabled) {
    FakeIterationEngine engine;
    engine.iteration.fruit_iter = 4;
    engine.learning.enabled = true;
    engine.learning.diagnostics = true;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::finalize_fruit_loop_iteration(engine, logger);

    EXPECT_EQ(engine.learning.finalize_iter, 4);
    EXPECT_EQ(engine.write_learning_summary_calls, 1);
    EXPECT_EQ(logger->info_calls, 1);
}

TEST(pipeline_iteration_lifecycle, finalizes_iteration_outputs) {
    FakeIterationTodProc todproc;
    todproc.engine().iteration.fruit_iter = 3;
    todproc.engine().output_paths.redu_dir_name = "/data/redu03";
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::finalize_iteration_outputs(todproc, logger);

    EXPECT_EQ(todproc.engine().ptcproc.finalize_weight_validation_iter, 3);
    EXPECT_EQ(todproc.engine().learning.finalize_iter, 3);
    EXPECT_EQ(todproc.engine().write_learning_summary_calls, 1);
    EXPECT_EQ(todproc.make_index_file_calls, 1);
    EXPECT_EQ(todproc.indexed_path, "/data/redu03");
    EXPECT_EQ(todproc.engine().iteration.fruit_iter, 4);
    EXPECT_EQ(logger->info_calls, 1);
}

TEST(pipeline_execution, setup_runs_before_enabled_pipeline) {
    FakeExecutionEngine engine;
    FakeKidsProc kidsproc;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::setup_and_run_observation_pipeline(
        engine, kidsproc, rawobs, logger);

    EXPECT_EQ(engine.setup_calls, 1);
    EXPECT_EQ(engine.pipeline_calls, 1);
    EXPECT_EQ(engine.event_order,
              (std::vector<std::string>{"setup", "pipeline"}));
}

TEST(pipeline_execution, setup_runs_when_tod_pipeline_disabled) {
    FakeExecutionEngine engine;
    engine.typed_config.timestream.enabled = false;
    FakeKidsProc kidsproc;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::setup_and_run_observation_pipeline(
        engine, kidsproc, rawobs, logger);

    EXPECT_EQ(engine.setup_calls, 1);
    EXPECT_EQ(engine.pipeline_calls, 0);
    EXPECT_EQ(engine.event_order, (std::vector<std::string>{"setup"}));
}

TEST(pipeline_execution, prepares_coadd_iteration_buffers) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.noise.enabled = true;
    todproc.engine().cmb.obsnums = {"101", "102"};
    todproc.engine().cmb.exposure_time = 12.0;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::prepare_coadd_iteration_buffers(todproc, logger);

    EXPECT_EQ(todproc.allocate_cmb_calls, 1);
    EXPECT_EQ(todproc.allocate_nmb_calls, 1);
    EXPECT_TRUE(todproc.engine().cmb.obsnums.empty());
    EXPECT_DOUBLE_EQ(todproc.engine().cmb.exposure_time, 0.0);
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(pipeline_execution, skips_coadd_noise_buffer_when_noise_disabled) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.noise.enabled = false;
    todproc.engine().cmb.obsnums = {"101"};
    todproc.engine().cmb.exposure_time = 6.0;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::prepare_coadd_iteration_buffers(todproc, logger);

    EXPECT_EQ(todproc.allocate_cmb_calls, 1);
    EXPECT_EQ(todproc.allocate_nmb_calls, 0);
    EXPECT_TRUE(todproc.engine().cmb.obsnums.empty());
    EXPECT_DOUBLE_EQ(todproc.engine().cmb.exposure_time, 0.0);
    EXPECT_EQ(logger->info_calls, 1);
}

TEST(pipeline_execution, prepares_iteration_observation_buffers_for_coadd) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = true;
    todproc.engine().observation_dates.date_obs = {"old"};
    todproc.engine().cmb.obsnums = {"101"};
    todproc.engine().cmb.exposure_time = 6.0;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::prepare_iteration_observation_buffers(
        todproc, logger);

    EXPECT_TRUE(todproc.engine().observation_dates.date_obs.empty());
    EXPECT_EQ(todproc.allocate_cmb_calls, 1);
    EXPECT_TRUE(todproc.engine().cmb.obsnums.empty());
    EXPECT_DOUBLE_EQ(todproc.engine().cmb.exposure_time, 0.0);
}

TEST(pipeline_execution, prepares_iteration_observation_buffers_without_coadd) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = false;
    todproc.engine().observation_dates.date_obs = {"old"};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::prepare_iteration_observation_buffers(
        todproc, logger);

    EXPECT_TRUE(todproc.engine().observation_dates.date_obs.empty());
    EXPECT_EQ(todproc.allocate_cmb_calls, 0);
    EXPECT_EQ(logger->info_calls, 0);
}

TEST(pipeline_execution, begins_reduction_iteration) {
    FakeReductionIterationTodProc todproc;
    std::vector<std::string> config_filepaths;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::begin_reduction_iteration(
        todproc, config_filepaths, logger);

    EXPECT_EQ(todproc.engine().ptcproc.begin_weight_validation_iter, 0);
    EXPECT_EQ(todproc.engine().learning.begin_calls, 1);
    EXPECT_EQ(todproc.create_output_dir_calls, 1);
    EXPECT_TRUE(todproc.engine().observation_dates.date_obs.empty());
    EXPECT_EQ(todproc.allocate_cmb_calls, 1);
    EXPECT_EQ(todproc.allocate_nmb_calls, 1);
    EXPECT_TRUE(todproc.engine().cmb.obsnums.empty());
    EXPECT_DOUBLE_EQ(todproc.engine().cmb.exposure_time, 0.0);
}

TEST(pipeline_execution, initializes_reduction_iterations) {
    FakeEngine engine;
    engine.iteration.fruit_iter = 7;
    engine.ptcproc.run_fruit_loops = false;
    engine.ptcproc.fruit_loops_iters = 5;
    engine.ptcproc.save_all_iters = false;
    bool fruit_loops_converged = true;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::initialize_reduction_iterations(
        engine, fruit_loops_converged, logger);

    EXPECT_EQ(engine.iteration.fruit_iter, 0);
    EXPECT_FALSE(fruit_loops_converged);
    EXPECT_EQ(engine.ptcproc.fruit_loops_iters, 1);
    EXPECT_TRUE(engine.ptcproc.save_all_iters);
}

TEST(pipeline_execution, calculates_initial_observation_map_dimensions) {
    FakeObservationMapTodProc todproc;
    std::vector<int> map_extents;
    std::vector<int> map_coords;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::calculate_initial_observation_map_dimensions(
        todproc, map_extents, map_coords, logger);

    EXPECT_EQ(todproc.calc_map_num_calls, 1);
    EXPECT_EQ(todproc.calc_omb_size_calls, 1);
    EXPECT_EQ(map_extents, (std::vector<int>{101}));
    EXPECT_EQ(map_coords, (std::vector<int>{202}));
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(pipeline_execution,
     skips_initial_observation_map_dimensions_when_mapmaking_disabled) {
    FakeObservationMapTodProc todproc;
    todproc.engine().typed_config.mapmaking.enabled = false;
    std::vector<int> map_extents = {11};
    std::vector<int> map_coords = {22};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::calculate_initial_observation_map_dimensions(
        todproc, map_extents, map_coords, logger);

    EXPECT_EQ(todproc.calc_map_num_calls, 0);
    EXPECT_EQ(todproc.calc_omb_size_calls, 0);
    EXPECT_EQ(map_extents, (std::vector<int>{11}));
    EXPECT_EQ(map_coords, (std::vector<int>{22}));
    EXPECT_EQ(logger->info_calls, 0);
}

TEST(pipeline_execution, prepares_initial_observation_setup) {
    FakeInitialObservationTodProc todproc;
    FakeRawObs rawobs;
    rawobs.astrometry.value = "astro";
    rawobs.tel.path = "/data/tel.nc";
    std::vector<FakeRawObsMeta> rawobs_kids_meta = {
        {75.0, 101},
        {122.0, 102},
    };
    std::vector<int> map_extents;
    std::vector<int> map_coords;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::prepare_initial_observation_setup<false>(
        todproc, rawobs, rawobs_kids_meta, map_extents, map_coords, logger));

    EXPECT_EQ(todproc.engine().get_astrometry_config_calls, 1);
    EXPECT_EQ(todproc.engine().calib.get_apt_calls, 1);
    EXPECT_EQ(todproc.check_inputs_calls, 1);
    EXPECT_DOUBLE_EQ(todproc.engine().telescope.fsmp, 122.0);
    EXPECT_EQ(todproc.engine().telescope.loaded_tel_path, "/data/tel.nc");
    EXPECT_EQ(todproc.engine().telescope.calc_tan_pointing_calls, 1);
    EXPECT_EQ(todproc.interp_pointing_calls, 1);
    EXPECT_EQ(todproc.engine().telescope.calc_scan_indices_calls, 1);
    EXPECT_EQ(todproc.calc_map_num_calls, 1);
    EXPECT_EQ(todproc.calc_omb_size_calls, 1);
    EXPECT_EQ(map_extents, (std::vector<int>{303}));
    EXPECT_EQ(map_coords, (std::vector<int>{404}));
}

TEST(pipeline_execution, prepares_initial_observation) {
    FakeInitialObservationTodProc todproc;
    FakeCitlaliConfig config;
    FakeRawObs rawobs;
    std::vector<int> map_extents;
    std::vector<int> map_coords;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::prepare_initial_observation<
        false, FakeKidsProc>(
        todproc, config, rawobs, map_extents, map_coords, logger));

    EXPECT_EQ(config.get_config_calls, 1);
    EXPECT_EQ(config.requested_key, "kids");
    EXPECT_EQ(todproc.check_inputs_calls, 1);
    EXPECT_EQ(todproc.calc_omb_size_calls, 1);
    EXPECT_EQ(map_extents, (std::vector<int>{303}));
    EXPECT_EQ(map_coords, (std::vector<int>{404}));
}

TEST(pipeline_execution, prepares_initial_observations) {
    FakeInitialObservationTodProc todproc;
    FakeCitlaliConfig config;
    FakeIOCoordinator co{{FakeRawObs{}, FakeRawObs{}}};
    std::vector<int> map_extents;
    std::vector<int> map_coords;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::prepare_initial_observations<
        false, FakeKidsProc>(
        todproc, co, config, map_extents, map_coords, logger));

    EXPECT_EQ(config.get_config_calls, 2);
    EXPECT_EQ(todproc.check_inputs_calls, 2);
    EXPECT_EQ(todproc.calc_omb_size_calls, 2);
    EXPECT_EQ(map_extents, (std::vector<int>{303, 303}));
    EXPECT_EQ(map_coords, (std::vector<int>{404, 404}));
    EXPECT_EQ(logger->info_calls, 19);
}

TEST(pipeline_execution, rejects_initial_observations_on_failure) {
    FakeInitialObservationTodProc todproc;
    FakeCitlaliConfig config;
    FakeFlxscaleCorrection correction{-1.0};
    FakeRawObs bad_rawobs{&correction, "bad_obs"};
    FakeIOCoordinator co{{bad_rawobs, FakeRawObs{}}};
    std::vector<int> map_extents;
    std::vector<int> map_coords;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_FALSE(citlali::pipeline::prepare_initial_observations<
        false, FakeKidsProc>(
        todproc, co, config, map_extents, map_coords, logger));

    EXPECT_EQ(config.get_config_calls, 1);
    EXPECT_EQ(todproc.check_inputs_calls, 0);
    EXPECT_TRUE(map_extents.empty());
    EXPECT_TRUE(map_coords.empty());
}

TEST(pipeline_execution, prepares_initial_reduction_geometry) {
    FakeInitialObservationTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = true;
    FakeCitlaliConfig config;
    FakeIOCoordinator co{{FakeRawObs{}, FakeRawObs{}}};
    std::vector<int> map_extents;
    std::vector<int> map_coords;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::prepare_initial_reduction_geometry<
        false, FakeKidsProc>(
        todproc, co, config, map_extents, map_coords, logger));

    EXPECT_EQ(todproc.calc_omb_size_calls, 2);
    EXPECT_EQ(todproc.calc_cmb_size_calls, 1);
    EXPECT_EQ(todproc.last_map_coord_count, 2);
}

TEST(pipeline_execution, rejects_initial_reduction_geometry_on_failure) {
    FakeInitialObservationTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = true;
    FakeCitlaliConfig config;
    FakeFlxscaleCorrection correction{-1.0};
    FakeRawObs bad_rawobs{&correction, "bad_obs"};
    FakeIOCoordinator co{{bad_rawobs}};
    std::vector<int> map_extents;
    std::vector<int> map_coords;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_FALSE(citlali::pipeline::prepare_initial_reduction_geometry<
        false, FakeKidsProc>(
        todproc, co, config, map_extents, map_coords, logger));

    EXPECT_EQ(todproc.calc_cmb_size_calls, 0);
}

TEST(pipeline_execution, rejects_initial_observation_setup_on_bad_flxscale) {
    FakeInitialObservationTodProc todproc;
    FakeFlxscaleCorrection correction{-1.0};
    FakeRawObs rawobs{&correction, "obs"};
    std::vector<FakeRawObsMeta> rawobs_kids_meta = {{122.0, 102}};
    std::vector<int> map_extents;
    std::vector<int> map_coords;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_FALSE(citlali::pipeline::prepare_initial_observation_setup<false>(
        todproc, rawobs, rawobs_kids_meta, map_extents, map_coords, logger));

    EXPECT_EQ(todproc.check_inputs_calls, 0);
    EXPECT_EQ(todproc.calc_map_num_calls, 0);
    EXPECT_TRUE(map_extents.empty());
    EXPECT_TRUE(map_coords.empty());
}

TEST(pipeline_execution, calculates_initial_coadd_map_dimensions) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = true;
    std::vector<int> map_coords = {1, 2, 3};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::calculate_initial_coadd_map_dimensions(
        todproc, map_coords, logger);

    EXPECT_EQ(todproc.calc_cmb_size_calls, 1);
    EXPECT_EQ(todproc.last_map_coord_count, 3);
    EXPECT_EQ(logger->info_calls, 1);
}

TEST(pipeline_execution,
     skips_initial_coadd_map_dimensions_when_coadd_disabled) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = false;
    std::vector<int> map_coords = {1, 2, 3};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::calculate_initial_coadd_map_dimensions(
        todproc, map_coords, logger);

    EXPECT_EQ(todproc.calc_cmb_size_calls, 0);
    EXPECT_EQ(todproc.last_map_coord_count, 0);
    EXPECT_EQ(logger->info_calls, 0);
}

TEST(pipeline_execution, allocates_observation_map_buffers) {
    FakeObservationMapTodProc todproc;
    int map_extent = 11;
    int map_coord = 22;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::allocate_observation_map_buffers(
        todproc, map_extent, map_coord, logger);

    EXPECT_EQ(todproc.calc_map_num_calls, 1);
    EXPECT_EQ(todproc.allocate_omb_calls, 1);
    EXPECT_EQ(todproc.last_map_extent, 11);
    EXPECT_EQ(todproc.last_map_coord, 22);
    EXPECT_EQ(todproc.engine().configure_map_pixel_contribution_targets_calls,
              1);
    EXPECT_EQ(todproc.engine().last_map_pixel_contribution_target, "raw_obs");
    EXPECT_EQ(todproc.allocate_nmb_calls, 1);
    EXPECT_EQ(logger->info_calls, 3);
}

TEST(pipeline_execution, skips_observation_noise_for_non_jinc_coadd) {
    FakeObservationMapTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = true;
    todproc.engine().typed_config.mapmaking.method =
        citlali::config::MapMethod::naive;
    int map_extent = 11;
    int map_coord = 22;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::allocate_observation_map_buffers(
        todproc, map_extent, map_coord, logger);

    EXPECT_EQ(todproc.calc_map_num_calls, 1);
    EXPECT_EQ(todproc.allocate_omb_calls, 1);
    EXPECT_EQ(todproc.allocate_nmb_calls, 0);
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(pipeline_execution, allocates_observation_map_buffers_by_index) {
    FakeObservationMapTodProc todproc;
    std::vector<int> map_extents = {11, 33};
    std::vector<int> map_coords = {22, 44};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::allocate_observation_map_buffers_if_needed(
        todproc, map_extents, map_coords, 1, logger);

    EXPECT_EQ(todproc.calc_map_num_calls, 1);
    EXPECT_EQ(todproc.allocate_omb_calls, 1);
    EXPECT_EQ(todproc.last_map_extent, 33);
    EXPECT_EQ(todproc.last_map_coord, 44);
    EXPECT_EQ(todproc.allocate_nmb_calls, 1);
    EXPECT_EQ(logger->info_calls, 3);
}

TEST(pipeline_execution,
     skips_observation_map_buffer_indexing_when_mapmaking_disabled) {
    FakeObservationMapTodProc todproc;
    todproc.engine().typed_config.mapmaking.enabled = false;
    std::vector<int> map_extents;
    std::vector<int> map_coords;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::allocate_observation_map_buffers_if_needed(
        todproc, map_extents, map_coords, 0, logger);

    EXPECT_EQ(todproc.calc_map_num_calls, 0);
    EXPECT_EQ(todproc.allocate_omb_calls, 0);
    EXPECT_EQ(todproc.allocate_nmb_calls, 0);
    EXPECT_EQ(logger->info_calls, 0);
}

TEST(pipeline_execution, prepares_reduction_observation_inputs) {
    FakeInitialObservationTodProc todproc;
    FakeRawObs rawobs;
    rawobs.tel.path = "/data/tel.nc";
    std::vector<FakeRawObsMeta> rawobs_kids_meta = {{122.0, 102}};
    std::vector<int> map_extents = {11};
    std::vector<int> map_coords = {22};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::prepare_reduction_observation_inputs<false>(
        todproc, rawobs, rawobs_kids_meta, true, map_extents, map_coords, 0,
        std::string{"2026-01-01T00:00:00"}, logger));

    EXPECT_EQ(todproc.engine().get_astrometry_config_calls, 1);
    EXPECT_DOUBLE_EQ(todproc.engine().telescope.fsmp, 122.0);
    EXPECT_DOUBLE_EQ(todproc.engine().telescope.d_fsmp, 122.0);
    EXPECT_EQ(todproc.get_tone_freqs_from_files_calls, 1);
    EXPECT_EQ(todproc.get_adc_snap_from_files_calls, 1);
    EXPECT_EQ(todproc.engine().observation_identity.obsnum, "000102");
    EXPECT_EQ(todproc.engine().calib.calc_flux_calibration_calls, 1);
    EXPECT_EQ(todproc.engine().telescope.get_tel_data_calls, 1);
    EXPECT_EQ(todproc.engine().telescope.calc_scan_indices_calls, 1);
    EXPECT_EQ(todproc.allocate_omb_calls, 1);
    EXPECT_EQ(todproc.last_map_extent, 11);
    EXPECT_EQ(todproc.last_map_coord, 22);
    EXPECT_EQ(todproc.engine().observation_dates.date_obs,
              (std::vector<std::string>{"2026-01-01T00:00:00"}));
    EXPECT_DOUBLE_EQ(todproc.engine().omb.exposure_time, 1.0);
}

TEST(pipeline_execution,
     rejects_reduction_observation_inputs_on_bad_sample_rate) {
    FakeInitialObservationTodProc todproc;
    auto &downsample = todproc.engine()
                           .typed_config.timestream.raw_time_chunk.downsample;
    downsample.enabled = true;
    downsample.factor = 0;
    downsample.downsampled_freq_Hz = 0.0;
    FakeRawObs rawobs;
    std::vector<FakeRawObsMeta> rawobs_kids_meta = {{122.0, 102}};
    std::vector<int> map_extents = {11};
    std::vector<int> map_coords = {22};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_FALSE(citlali::pipeline::prepare_reduction_observation_inputs<false>(
        todproc, rawobs, rawobs_kids_meta, true, map_extents, map_coords, 0,
        std::string{"2026-01-01T00:00:00"}, logger));

    EXPECT_EQ(todproc.get_tone_freqs_from_files_calls, 0);
    EXPECT_TRUE(todproc.engine().observation_dates.date_obs.empty());
    EXPECT_EQ(todproc.allocate_omb_calls, 0);
}

TEST(pipeline_execution, coadds_observation) {
    FakeCoaddTodProc todproc;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::coadd_observation(todproc, logger);

    EXPECT_EQ(todproc.coadd_calls, 1);
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(pipeline_execution, skips_coadd_for_polarization) {
    FakeCoaddTodProc todproc;
    todproc.engine().rtcproc.run_polarization = true;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::coadd_observation(todproc, logger);

    EXPECT_EQ(todproc.coadd_calls, 0);
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(pipeline_execution, writes_raw_observation_outputs) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.mapmaking.enabled = true;
    todproc.engine().typed_config.noise.products_enabled = true;
    todproc.engine().typed_config.noise.enabled = true;
    todproc.engine().typed_config.noise.apply_empirical_weights = true;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_raw_observation_outputs<FakeMapType::RawObs>(
        todproc, logger);

    EXPECT_EQ(todproc.engine().omb.calc_noise_products_calls, 1);
    EXPECT_TRUE(todproc.engine().omb.last_apply_empirical_noise_weights);
    EXPECT_EQ(todproc.engine().create_obs_map_files_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
    EXPECT_EQ(logger->info_calls, 5);
}

TEST(pipeline_execution, skips_raw_noise_products_when_disabled) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.mapmaking.enabled = true;
    todproc.engine().typed_config.noise.products_enabled = false;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_raw_observation_outputs<FakeMapType::RawObs>(
        todproc, logger);

    EXPECT_EQ(todproc.engine().omb.calc_noise_products_calls, 0);
    EXPECT_EQ(todproc.engine().create_obs_map_files_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
    EXPECT_EQ(logger->info_calls, 3);
}

TEST(pipeline_execution, skips_raw_outputs_when_mapmaking_disabled) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.mapmaking.enabled = false;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_raw_observation_outputs<FakeMapType::RawObs>(
        todproc, logger);

    EXPECT_EQ(todproc.engine().omb.calc_noise_products_calls, 0);
    EXPECT_EQ(todproc.engine().create_obs_map_files_calls, 0);
    EXPECT_EQ(todproc.engine().output_calls, 0);
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(pipeline_execution, writes_filtered_observation_outputs) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.runtime.reduction_type =
        citlali::config::ReductionType::pointing;
    sync_fake_runtime_provenance(todproc.engine());
    todproc.engine().typed_config.noise.products_enabled = true;
    todproc.engine().typed_config.noise.enabled = true;
    citlali::config::set_source_finding_enabled(
        todproc.engine().typed_config.post_processing, true);
    todproc.engine().wiener_filter.normalize_error = true;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_filtered_observation_outputs<
        FakeMapType::FilteredObs, false>(todproc, logger);

    EXPECT_EQ(todproc.engine().run_wiener_filter_calls, 1);
    EXPECT_EQ(todproc.engine().omb.calc_noise_products_calls, 1);
    EXPECT_TRUE(todproc.engine().omb.last_apply_empirical_noise_weights);
    EXPECT_EQ(todproc.engine().omb.calc_map_psd_calls, 1);
    EXPECT_EQ(todproc.engine().omb.calc_map_hist_calls, 1);
    EXPECT_EQ(todproc.engine().omb.calc_median_err_calls, 1);
    EXPECT_EQ(todproc.engine().omb.calc_median_rms_calls, 1);
    EXPECT_EQ(todproc.engine().find_sources_calls, 1);
    EXPECT_EQ(todproc.engine().fit_maps_calls, 0);
    EXPECT_EQ(todproc.engine().output_calls, 1);
}

TEST(pipeline_execution, fits_filtered_observation_maps_when_requested) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.runtime.reduction_type =
        citlali::config::ReductionType::pointing;
    sync_fake_runtime_provenance(todproc.engine());
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_filtered_observation_outputs<
        FakeMapType::FilteredObs, true>(todproc, logger);

    EXPECT_EQ(todproc.engine().fit_maps_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
}

TEST(pipeline_execution, skips_post_filter_observation_output_for_science) {
    FakeCoaddTodProc todproc;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_filtered_observation_outputs<
        FakeMapType::FilteredObs, false>(todproc, logger);

    EXPECT_EQ(todproc.engine().omb.calc_noise_products_calls, 0);
    EXPECT_EQ(todproc.engine().output_calls, 0);
    EXPECT_EQ(logger->info_calls, 10);
}

TEST(pipeline_execution, writes_observation_outputs_without_accumulation) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.mapmaking.enabled = true;
    todproc.engine().typed_config.coadd.enabled = false;
    citlali::config::set_map_filtering_enabled(
        todproc.engine().typed_config.post_processing, false);
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_observation_outputs_and_accumulate<
        FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
        todproc, logger);

    EXPECT_EQ(todproc.engine().output_calls, 1);
    EXPECT_EQ(todproc.coadd_calls, 0);
    EXPECT_EQ(todproc.engine().run_wiener_filter_calls, 0);
}

TEST(pipeline_execution, writes_observation_outputs_and_coadds) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.mapmaking.enabled = true;
    todproc.engine().typed_config.coadd.enabled = true;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_observation_outputs_and_accumulate<
        FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
        todproc, logger);

    EXPECT_EQ(todproc.engine().output_calls, 1);
    EXPECT_EQ(todproc.coadd_calls, 1);
    EXPECT_EQ(todproc.engine().run_wiener_filter_calls, 0);
}

TEST(pipeline_execution, writes_observation_outputs_and_filters) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.runtime.reduction_type =
        citlali::config::ReductionType::pointing;
    sync_fake_runtime_provenance(todproc.engine());
    todproc.engine().typed_config.mapmaking.enabled = true;
    todproc.engine().typed_config.coadd.enabled = false;
    citlali::config::set_map_filtering_enabled(
        todproc.engine().typed_config.post_processing, true);
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_observation_outputs_and_accumulate<
        FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
        todproc, logger);

    EXPECT_EQ(todproc.engine().output_calls, 2);
    EXPECT_EQ(todproc.coadd_calls, 0);
    EXPECT_EQ(todproc.engine().run_wiener_filter_calls, 1);
}

TEST(pipeline_execution, runs_reduction_observation_pipeline) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.mapmaking.enabled = true;
    todproc.engine().typed_config.coadd.enabled = true;
    todproc.engine().typed_config.timestream.fruit_loops.enabled = true;
    todproc.engine().typed_config.timestream.fruit_loops.path = "/data/fruit";
    todproc.engine().typed_config.timestream.fruit_loops.type = "obsnum/raw";
    todproc.engine().omb.obsnums = {"000123"};
    FakeKidsProc kidsproc;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::run_reduction_observation_pipeline<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
        todproc, kidsproc, rawobs, logger);

    EXPECT_EQ(todproc.engine().ptcproc.load_mb_calls, 1);
    EXPECT_EQ(todproc.engine().setup_calls, 1);
    EXPECT_EQ(todproc.engine().pipeline_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
    EXPECT_EQ(todproc.coadd_calls, 1);
}

TEST(pipeline_execution, runs_reduction_observation) {
    FakeInitialObservationTodProc todproc;
    todproc.engine().typed_config.mapmaking.enabled = true;
    FakeKidsProc kidsproc;
    FakeRawObs rawobs;
    std::vector<FakeRawObsMeta> rawobs_kids_meta = {{122.0, 102}};
    std::vector<int> map_extents = {11};
    std::vector<int> map_coords = {22};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::run_reduction_observation<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
        todproc, kidsproc, rawobs, rawobs_kids_meta, true, map_extents,
        map_coords, 0, std::string{"2026-01-01T00:00:00"}, logger));

    EXPECT_EQ(todproc.engine().setup_calls, 1);
    EXPECT_EQ(todproc.engine().pipeline_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
    EXPECT_EQ(todproc.allocate_omb_calls, 1);
}

TEST(pipeline_execution,
     rejects_reduction_observation_when_prepare_fails) {
    FakeInitialObservationTodProc todproc;
    auto &downsample = todproc.engine()
                           .typed_config.timestream.raw_time_chunk.downsample;
    downsample.enabled = true;
    downsample.factor = 0;
    downsample.downsampled_freq_Hz = 0.0;
    FakeKidsProc kidsproc;
    FakeRawObs rawobs;
    std::vector<FakeRawObsMeta> rawobs_kids_meta = {{122.0, 102}};
    std::vector<int> map_extents = {11};
    std::vector<int> map_coords = {22};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_FALSE(citlali::pipeline::run_reduction_observation<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
        todproc, kidsproc, rawobs, rawobs_kids_meta, true, map_extents,
        map_coords, 0, std::string{"2026-01-01T00:00:00"}, logger));

    EXPECT_EQ(todproc.engine().setup_calls, 0);
    EXPECT_EQ(todproc.engine().pipeline_calls, 0);
    EXPECT_EQ(todproc.engine().output_calls, 0);
}

TEST(pipeline_execution, runs_reduction_observation_at_index) {
    FakeInitialObservationTodProc todproc;
    FakeCitlaliConfig config;
    FakeIOCoordinator co{{FakeRawObs{}, FakeRawObs{}}};
    std::vector<int> map_extents = {11, 33};
    std::vector<int> map_coords = {22, 44};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::run_reduction_observation_at_index<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs, false,
        FakeKidsProc>(
        todproc, co, config, map_extents, map_coords, 1,
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        logger));

    EXPECT_EQ(config.get_config_calls, 1);
    EXPECT_EQ(todproc.last_map_extent, 33);
    EXPECT_EQ(todproc.last_map_coord, 44);
    EXPECT_EQ(todproc.engine().setup_calls, 1);
    EXPECT_EQ(todproc.engine().pipeline_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
    EXPECT_EQ(todproc.engine().observation_dates.date_obs,
              (std::vector<std::string>{"2026-01-01T00:00:00"}));
}

TEST(pipeline_execution, reports_observation_context_when_metadata_load_fails) {
    FakeInitialObservationTodProc todproc;
    FakeCitlaliConfig config;
    FakeRawObs rawobs;
    rawobs.obs_name = "science_152392";
    rawobs.tel.path = "missing_telescope.nc";
    FakeIOCoordinator co{{rawobs}};
    std::vector<int> map_extents = {11};
    std::vector<int> map_coords = {22};
    auto logger = std::make_shared<FakeLogger>();

    try {
        citlali::pipeline::run_reduction_observation_at_index<
            false, FakeMapType::RawObs, FakeMapType::FilteredObs, false,
            FakeFailingKidsProc>(
            todproc, co, config, map_extents, map_coords, 0,
            [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
            logger);
        FAIL() << "expected metadata load failure";
    } catch (const std::runtime_error &error) {
        const std::string message{error.what()};
        EXPECT_NE(message.find("observation index 0"), std::string::npos);
        EXPECT_NE(message.find("science_152392"), std::string::npos);
        EXPECT_NE(message.find("missing_telescope.nc"), std::string::npos);
        EXPECT_NE(message.find("No such file or directory"),
                  std::string::npos);
    }
}

TEST(pipeline_execution, runs_reduction_iteration_observations) {
    FakeInitialObservationTodProc todproc;
    FakeCitlaliConfig config;
    FakeIOCoordinator co{{FakeRawObs{}, FakeRawObs{}}};
    std::vector<int> map_extents = {11, 33};
    std::vector<int> map_coords = {22, 44};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::run_reduction_iteration_observations<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs, false,
        FakeKidsProc>(
        todproc, co, config, map_extents, map_coords,
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        logger));

    EXPECT_EQ(config.get_config_calls, 2);
    EXPECT_EQ(todproc.engine().setup_calls, 2);
    EXPECT_EQ(todproc.engine().pipeline_calls, 2);
    EXPECT_EQ(todproc.engine().output_calls, 2);
    EXPECT_EQ(todproc.engine().observation_dates.date_obs.size(), 2U);
}

TEST(pipeline_execution, rejects_reduction_iteration_observations_on_failure) {
    FakeInitialObservationTodProc todproc;
    auto &downsample = todproc.engine()
                           .typed_config.timestream.raw_time_chunk.downsample;
    downsample.enabled = true;
    downsample.factor = 0;
    downsample.downsampled_freq_Hz = 0.0;
    FakeCitlaliConfig config;
    FakeIOCoordinator co{{FakeRawObs{}, FakeRawObs{}}};
    std::vector<int> map_extents = {11, 33};
    std::vector<int> map_coords = {22, 44};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_FALSE(citlali::pipeline::run_reduction_iteration_observations<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs, false,
        FakeKidsProc>(
        todproc, co, config, map_extents, map_coords,
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        logger));

    EXPECT_EQ(config.get_config_calls, 1);
    EXPECT_EQ(todproc.engine().setup_calls, 0);
    EXPECT_EQ(todproc.engine().output_calls, 0);
}

TEST(pipeline_execution, runs_reduction_iteration) {
    FakeInitialObservationTodProc todproc;
    todproc.engine().output_paths.redu_dir_name = "/tmp/redu01";
    FakeCitlaliConfig config;
    FakeIOCoordinator co{{FakeRawObs{}, FakeRawObs{}}};
    std::vector<std::string> config_filepaths;
    std::vector<int> map_extents = {11, 33};
    std::vector<int> map_coords = {22, 44};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::run_reduction_iteration<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs,
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd, false,
        FakeKidsProc>(
        todproc, co, config, config_filepaths, map_extents, map_coords,
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        logger));

    EXPECT_EQ(todproc.create_output_dir_calls, 1);
    EXPECT_EQ(todproc.engine().setup_calls, 2);
    EXPECT_EQ(todproc.engine().pipeline_calls, 2);
    EXPECT_EQ(todproc.engine().output_calls, 2);
    EXPECT_EQ(todproc.engine().ptcproc.finalize_weight_validation_iter, 0);
    EXPECT_EQ(todproc.engine().learning.finalize_calls, 1);
    EXPECT_EQ(todproc.engine().write_learning_summary_calls, 1);
    EXPECT_EQ(todproc.make_index_file_calls, 1);
    EXPECT_EQ(todproc.indexed_path, "/tmp/redu01");
    EXPECT_EQ(todproc.engine().iteration.fruit_iter, 1);
}

TEST(pipeline_execution, runs_reduction_iterations) {
    FakeInitialObservationTodProc todproc;
    todproc.engine().typed_config.runtime.reduction_type =
        citlali::config::ReductionType::science;
    todproc.engine().typed_config.timestream.fruit_loops.enabled = true;
    todproc.engine().typed_config.timestream.fruit_loops.max_iters = 2;
    FakeCitlaliConfig config;
    FakeIOCoordinator co{{FakeRawObs{}}};
    std::vector<std::string> config_filepaths;
    std::vector<int> map_extents = {11};
    std::vector<int> map_coords = {22};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::run_reduction_iterations<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs,
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd, false,
        FakeKidsProc>(
        todproc, co, config, config_filepaths, map_extents, map_coords,
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        logger));

    EXPECT_EQ(todproc.engine().iteration.fruit_iter, 2);
    EXPECT_EQ(todproc.engine().setup_calls, 2);
    EXPECT_EQ(todproc.engine().pipeline_calls, 2);
    EXPECT_EQ(todproc.engine().output_calls, 2);
    EXPECT_EQ(todproc.make_index_file_calls, 2);
    EXPECT_EQ(todproc.create_output_dir_calls, 1);
}

TEST(pipeline_execution, runs_reduction_pipeline) {
    FakeInitialObservationTodProc todproc;
    FakeCitlaliConfig config;
    FakeIOCoordinator co{{FakeRawObs{}}};
    std::vector<std::string> config_filepaths;
    std::vector<int> map_extents;
    std::vector<int> map_coords;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::run_reduction_pipeline<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs,
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd, false,
        FakeKidsProc>(
        todproc, co, config, config_filepaths, map_extents, map_coords,
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        logger));

    EXPECT_EQ(todproc.calc_omb_size_calls, 1);
    EXPECT_EQ(map_extents, (std::vector<int>{303}));
    EXPECT_EQ(map_coords, (std::vector<int>{404}));
    EXPECT_EQ(todproc.engine().setup_calls, 1);
    EXPECT_EQ(todproc.engine().pipeline_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
    EXPECT_EQ(todproc.engine().iteration.fruit_iter, 1);
}

TEST(pipeline_execution, writes_raw_coadd_outputs) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.noise.apply_empirical_weights = true;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_raw_coadd_outputs<FakeMapType::RawCoadd>(
        todproc, logger);

    EXPECT_EQ(todproc.create_coadded_map_files_calls, 1);
    EXPECT_EQ(todproc.engine().cmb.normalize_maps_calls, 1);
    EXPECT_EQ(todproc.engine().cmb.normalize_polarized_maps_calls, 0);
    EXPECT_EQ(todproc.engine().cmb.calc_noise_products_calls, 1);
    EXPECT_TRUE(todproc.engine().cmb.last_apply_empirical_noise_weights);
    EXPECT_EQ(todproc.engine().cmb.calc_map_psd_calls, 1);
    EXPECT_EQ(todproc.engine().cmb.calc_map_hist_calls, 1);
    EXPECT_EQ(todproc.engine().cmb.calc_median_err_calls, 1);
    EXPECT_EQ(todproc.engine().cmb.calc_median_rms_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
}

TEST(pipeline_execution, writes_polarized_raw_coadd_outputs) {
    FakeCoaddTodProc todproc;
    todproc.engine().rtcproc.run_polarization = true;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_raw_coadd_outputs<FakeMapType::RawCoadd>(
        todproc, logger);

    EXPECT_EQ(todproc.engine().cmb.normalize_maps_calls, 0);
    EXPECT_EQ(todproc.engine().cmb.normalize_polarized_maps_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
}

TEST(pipeline_execution, skips_raw_coadd_noise_products_when_disabled) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.noise.products_enabled = false;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_raw_coadd_outputs<FakeMapType::RawCoadd>(
        todproc, logger);

    EXPECT_EQ(todproc.engine().cmb.calc_noise_products_calls, 0);
    EXPECT_EQ(todproc.engine().output_calls, 1);
}

TEST(pipeline_execution, writes_filtered_coadd_outputs) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.runtime.reduction_type =
        citlali::config::ReductionType::pointing;
    sync_fake_runtime_provenance(todproc.engine());
    citlali::config::set_source_finding_enabled(
        todproc.engine().typed_config.post_processing, true);
    todproc.engine().wiener_filter.normalize_error = true;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_filtered_coadd_outputs<
        FakeMapType::FilteredCoadd>(todproc, logger);

    EXPECT_EQ(todproc.engine().run_wiener_filter_calls, 1);
    EXPECT_EQ(todproc.engine().cmb.calc_noise_products_calls, 1);
    EXPECT_TRUE(todproc.engine().cmb.last_apply_empirical_noise_weights);
    EXPECT_EQ(todproc.engine().cmb.calc_map_psd_calls, 1);
    EXPECT_EQ(todproc.engine().cmb.calc_map_hist_calls, 1);
    EXPECT_EQ(todproc.engine().cmb.calc_median_err_calls, 1);
    EXPECT_EQ(todproc.engine().cmb.calc_median_rms_calls, 1);
    EXPECT_EQ(todproc.engine().find_sources_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
}

TEST(pipeline_execution, skips_post_filter_coadd_output_for_science) {
    FakeCoaddTodProc todproc;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_filtered_coadd_outputs<
        FakeMapType::FilteredCoadd>(todproc, logger);

    EXPECT_EQ(todproc.engine().cmb.calc_noise_products_calls, 0);
    EXPECT_EQ(todproc.engine().output_calls, 0);
    EXPECT_EQ(logger->info_calls, 10);
}

TEST(pipeline_execution, skips_iteration_coadd_outputs_when_coadd_disabled) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = false;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_iteration_coadd_outputs_if_needed<
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd>(todproc, logger);

    EXPECT_EQ(todproc.create_coadded_map_files_calls, 0);
    EXPECT_EQ(todproc.engine().output_calls, 0);
}

TEST(pipeline_execution, writes_iteration_raw_coadd_outputs) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = true;
    citlali::config::set_map_filtering_enabled(
        todproc.engine().typed_config.post_processing, false);
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_iteration_coadd_outputs_if_needed<
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd>(todproc, logger);

    EXPECT_EQ(todproc.create_coadded_map_files_calls, 1);
    EXPECT_EQ(todproc.engine().run_wiener_filter_calls, 0);
    EXPECT_EQ(todproc.engine().output_calls, 1);
}

TEST(pipeline_execution, writes_iteration_filtered_coadd_outputs) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.runtime.reduction_type =
        citlali::config::ReductionType::pointing;
    sync_fake_runtime_provenance(todproc.engine());
    todproc.engine().typed_config.coadd.enabled = true;
    citlali::config::set_map_filtering_enabled(
        todproc.engine().typed_config.post_processing, true);
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_iteration_coadd_outputs_if_needed<
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd>(todproc, logger);

    EXPECT_EQ(todproc.create_coadded_map_files_calls, 1);
    EXPECT_EQ(todproc.engine().run_wiener_filter_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 2);
}

TEST(pipeline_execution, finishes_reduction_iteration) {
    FakeReductionIterationTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = false;
    todproc.engine().iteration.fruit_iter = 2;
    todproc.engine().output_paths.redu_dir_name = "/data/redu02";
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::finish_reduction_iteration<
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd>(todproc, logger);

    EXPECT_EQ(todproc.engine().ptcproc.finalize_weight_validation_iter, 2);
    EXPECT_EQ(todproc.engine().learning.finalize_calls, 1);
    EXPECT_EQ(todproc.engine().write_learning_summary_calls, 1);
    EXPECT_EQ(todproc.make_index_file_calls, 1);
    EXPECT_EQ(todproc.indexed_path, "/data/redu02");
    EXPECT_EQ(todproc.engine().iteration.fruit_iter, 3);
}

TEST(pipeline_execution, loads_initial_fruit_loop_map_from_configured_path) {
    FakeEngine engine;
    engine.iteration.fruit_iter = 0;
    engine.typed_config.timestream.fruit_loops.enabled = true;
    engine.typed_config.timestream.fruit_loops.path = "/data/fruit";
    engine.typed_config.timestream.fruit_loops.type = "obsnum/raw";
    engine.omb.obsnums = {"152389"};
    engine.omb.cov_cut = 4.5;
    engine.omb.pixel_size_rad = 0.001;

    citlali::pipeline::load_initial_fruit_loop_maps_if_requested(engine);

    EXPECT_EQ(engine.ptcproc.load_mb_calls, 1);
    EXPECT_EQ(engine.ptcproc.loaded_filepath, "/data/fruit/152389/raw/");
    EXPECT_EQ(engine.ptcproc.loaded_noise_filepath,
              "/data/fruit/152389/raw/");
    EXPECT_DOUBLE_EQ(engine.ptcproc.tod_mb.cov_cut, 4.5);
}

TEST(pipeline_execution, uses_typed_fruit_loop_weight_policy) {
    FakeEngine engine;
    engine.typed_config.timestream.fruit_loops.enabled = true;
    engine.typed_config.timestream.fruit_loops
        .recompute_weights_after_addback = false;
    engine.ptcproc.run_fruit_loops = false;
    engine.ptcproc.fruit_loops_recompute_weights_after_addback = true;
    engine.ptcproc.tod_mb.signal = {1.0};

    const auto policy =
        citlali::pipeline::fruit_loop_weight_policy(engine);

    EXPECT_TRUE(policy.use_noise_weights);
    EXPECT_TRUE(policy.keep_source_subtracted_weights);
}

TEST(pipeline_execution, uses_typed_fruit_loop_interpolation_policy) {
    FakeEngine engine;
    engine.typed_config.timestream.fruit_loops.enabled = true;
    engine.typed_config.timestream.fruit_loops.interp_mode_override =
        citlali::config::FruitLoopsInterpModeOverride::bilinear;
    engine.ptcproc.run_fruit_loops = false;
    auto logger = std::make_shared<FakeLogger>();

    const auto resolution =
        citlali::pipeline::resolve_fruit_loop_interpolation(
            engine.typed_config.timestream.fruit_loops,
            citlali::config::MapMethod::jinc);

    EXPECT_EQ(
        resolution.requested,
        citlali::config::FruitLoopsInterpModeOverride::bilinear);
    EXPECT_EQ(resolution.mapmaking_default,
              citlali::config::FruitLoopsInterpModeOverride::jinc);
    EXPECT_EQ(resolution.effective,
              citlali::config::FruitLoopsInterpModeOverride::bilinear);
    EXPECT_TRUE(resolution.override_applied);
    EXPECT_FALSE(resolution.jinc_fell_back_to_bilinear);

    citlali::pipeline::configure_fruit_loop_interpolation_mode(
        engine, citlali::config::MapMethod::jinc, logger);

    EXPECT_EQ(engine.ptcproc.fruit_loops_interp_mode, "bilinear");

    engine.typed_config.timestream.fruit_loops.interp_mode_override =
        citlali::config::FruitLoopsInterpModeOverride::jinc;
    const auto fallback =
        citlali::pipeline::resolve_fruit_loop_interpolation(
            engine.typed_config.timestream.fruit_loops,
            citlali::config::MapMethod::naive);
    EXPECT_EQ(fallback.effective,
              citlali::config::FruitLoopsInterpModeOverride::bilinear);
    EXPECT_TRUE(fallback.override_applied);
    EXPECT_TRUE(fallback.jinc_fell_back_to_bilinear);
}

TEST(pipeline_execution, uses_typed_fruit_loop_flux_metadata) {
    citlali::config::TimestreamFruitLoopsConfig config;
    config.enabled = true;
    config.array_flux_limit = {11.0, 22.0};
    const std::vector<int> arrays{0, 2};

    EXPECT_DOUBLE_EQ(
        citlali::pipeline::phdu_fruit_loop_flux_limit(
            config, arrays, 1, 2),
        22.0);
}

TEST(pipeline_execution, adapts_typed_fruit_loop_policy_one_way) {
    citlali::config::TimestreamFruitLoopsConfig config;
    config.enabled = true;
    config.save_all_iters = true;
    config.recompute_weights_after_addback = true;
    config.path = "/typed/maps";
    config.type = "obsnum/raw";
    config.mode = citlali::config::FruitLoopsMode::both;
    config.sig2noise_limit = 7.5;
    config.array_flux_limit = {1.0, 2.0, 3.0};
    config.peak_fraction_limit = 0.8;
    config.local_snr_floor = 4.1;
    config.local_sigma_inner_radius_arcsec = 11.0;
    config.local_sigma_outer_radius_arcsec = 36.0;
    config.local_sigma_inner_fwhm = 1.6;
    config.local_sigma_outer_fwhm = 4.2;
    config.local_sigma_edge_guard_arcsec = 6.0;
    config.local_sigma_min_pixels = 55;
    config.adaptive_support_radius_arcsec = 13.0;
    config.adaptive_support_radius_fwhm = 1.7;
    config.weight_feedback.enabled = true;
    config.weight_feedback.reference =
        citlali::config::FruitLoopsWeightFeedbackReference::median;
    config.weight_feedback.low_relative_weight = 0.03;
    config.weight_feedback.high_relative_weight = 0.12;
    config.center_keep_radius_arcsec = 8.0;
    config.interp_mode_override =
        citlali::config::FruitLoopsInterpModeOverride::nearest;
    config.legacy_center = true;
    config.max_iters = 4;
    FakeFruitLoopsAdapterPtcProc ptcproc;

    citlali::pipeline::apply_fruit_loops_config_to_processor(
        config, ptcproc);

    EXPECT_TRUE(ptcproc.run_fruit_loops);
    EXPECT_TRUE(ptcproc.save_all_iters);
    EXPECT_TRUE(ptcproc.fruit_loops_recompute_weights_after_addback);
    EXPECT_EQ(ptcproc.fruit_loops_path, "/typed/maps");
    EXPECT_EQ(ptcproc.fruit_loops_type, "obsnum/raw");
    EXPECT_EQ(ptcproc.fruit_mode, "both");
    EXPECT_DOUBLE_EQ(ptcproc.fruit_loops_sig2noise, 7.5);
    ASSERT_EQ(ptcproc.fruit_loops_flux.size(), 3);
    EXPECT_DOUBLE_EQ(ptcproc.fruit_loops_flux(2), 3.0);
    EXPECT_DOUBLE_EQ(ptcproc.fruit_loops_peak_fraction_limit, 0.8);
    EXPECT_DOUBLE_EQ(ptcproc.fruit_loops_local_snr_floor, 4.1);
    EXPECT_DOUBLE_EQ(
        ptcproc.fruit_loops_local_sigma_inner_radius_arcsec, 11.0);
    EXPECT_DOUBLE_EQ(
        ptcproc.fruit_loops_local_sigma_outer_radius_arcsec, 36.0);
    EXPECT_DOUBLE_EQ(ptcproc.fruit_loops_local_sigma_inner_fwhm, 1.6);
    EXPECT_DOUBLE_EQ(ptcproc.fruit_loops_local_sigma_outer_fwhm, 4.2);
    EXPECT_DOUBLE_EQ(
        ptcproc.fruit_loops_local_sigma_edge_guard_arcsec, 6.0);
    EXPECT_EQ(ptcproc.fruit_loops_local_sigma_min_pixels, 55);
    EXPECT_DOUBLE_EQ(
        ptcproc.fruit_loops_adaptive_support_radius_arcsec, 13.0);
    EXPECT_DOUBLE_EQ(
        ptcproc.fruit_loops_adaptive_support_radius_fwhm, 1.7);
    EXPECT_TRUE(ptcproc.fruit_loops_weight_feedback_enabled);
    EXPECT_EQ(ptcproc.fruit_loops_weight_feedback_reference, "median");
    EXPECT_DOUBLE_EQ(
        ptcproc.fruit_loops_weight_feedback_low_relative_weight, 0.03);
    EXPECT_DOUBLE_EQ(
        ptcproc.fruit_loops_weight_feedback_high_relative_weight, 0.12);
    EXPECT_DOUBLE_EQ(ptcproc.fruit_loops_center_keep_radius_arcsec, 8.0);
    EXPECT_EQ(ptcproc.fruit_loops_interp_mode_override, "nearest");
    EXPECT_TRUE(ptcproc.fruit_loops_legacy_center);
    EXPECT_EQ(ptcproc.fruit_loops_iters, 4);
}

TEST(pipeline_execution, adapts_typed_processed_clean_policy_one_way) {
    citlali::config::ProcessedTimeChunkCleanConfig config;
    config.enabled = true;
    config.grouping = {"array", "nw"};
    config.mask_radius_arcsec = 19.0;
    config.tau = 0.23;
    config.standard_pca.enabled = true;
    config.standard_pca.stddev_limit = 4.5;
    config.standard_pca.n_calc = 17;
    config.standard_pca.n_eig_to_cut["a1100"] = {2, 3};
    auto &corr = config.corr_grouping;
    corr.enabled = true;
    corr.metric =
        citlali::config::ProcessedTimeChunkCorrGroupingMetric::signed_metric;
    corr.corr_min = 0.61;
    corr.min_overlap = 301;
    corr.min_good_frac = 0.81;
    corr.min_group_size = 11;
    corr.max_samples = 20001;
    corr.clean_residual = false;
    auto &null_model = config.null_model;
    null_model.enabled = true;
    null_model.n_surrogates = 18;
    null_model.quantile = 0.98;
    null_model.min_good_frac = 0.82;
    null_model.max_modes = 61;
    null_model.max_samples = 19000;
    null_model.seed = 23456;
    null_model.grouping = {"nw"};
    auto &mp = config.marchenko_pastur;
    mp.enabled = true;
    mp.min_good_frac = 0.83;
    mp.max_modes = 62;
    mp.max_samples = 18000;
    mp.band_low_Hz = 0.06;
    mp.band_high_Hz = 2.1;
    mp.clip_z = 11.0;
    mp.bulk_keep_frac = 0.79;
    mp.q_grid_size = 63;
    mp.grouping = {"array"};
    auto &adaptive = config.adaptive_selector;
    adaptive.enabled = true;
    adaptive.min_good_frac = 0.71;
    adaptive.max_det = 121;
    adaptive.max_samples = 1025;
    adaptive.max_pairs = 2001;
    adaptive.seed = 34567;
    adaptive.clip_z = 49.0;
    adaptive.low_weight = 0.9;
    adaptive.tail_weight = 0.2;
    adaptive.topmode_weight = 0.3;
    adaptive.reg_weight = 0.4;
    adaptive.low_band_Hz = {0.07, 0.51};
    adaptive.mid_band_Hz = {0.52, 2.2};
    adaptive.candidate_offsets = {-1, 1, 3};
    adaptive.grouping = {"all"};
    adaptive.log_candidates = true;
    const std::map<int, std::string> array_name_map{{0, "a1100"}};
    FakeProcessedCleanAdapterPtcProc ptcproc;

    citlali::pipeline::apply_processed_clean_config_to_processor(
        config, array_name_map, ptcproc);

    EXPECT_TRUE(ptcproc.run_clean);
    EXPECT_EQ(ptcproc.cleaner.grouping,
              (std::vector<std::string>{"array", "nw"}));
    EXPECT_DOUBLE_EQ(ptcproc.mask_radius_arcsec, 19.0);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.tau, 0.23);
    EXPECT_TRUE(ptcproc.cleaner.standard_pca.enabled);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.stddev_limit, 4.5);
    EXPECT_EQ(ptcproc.cleaner.n_calc, 17);
    ASSERT_EQ(ptcproc.cleaner.n_eig_to_cut.count(0), 1U);
    ASSERT_EQ(ptcproc.cleaner.n_eig_to_cut.at(0).size(), 2);
    EXPECT_EQ(ptcproc.cleaner.n_eig_to_cut.at(0)(0), 2);
    EXPECT_EQ(ptcproc.cleaner.n_eig_to_cut.at(0)(1), 3);
    EXPECT_TRUE(ptcproc.cleaner.corr_grouping.enabled);
    EXPECT_EQ(ptcproc.cleaner.corr_grouping.metric, "signed");
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.corr_grouping.corr_min, 0.61);
    EXPECT_EQ(ptcproc.cleaner.corr_grouping.min_overlap, 301);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.corr_grouping.min_good_frac, 0.81);
    EXPECT_EQ(ptcproc.cleaner.corr_grouping.min_group_size, 11);
    EXPECT_EQ(ptcproc.cleaner.corr_grouping.max_samples, 20001);
    EXPECT_FALSE(ptcproc.cleaner.corr_grouping.clean_residual);
    EXPECT_TRUE(ptcproc.cleaner.null_model.enabled);
    EXPECT_EQ(ptcproc.cleaner.null_model.n_surrogates, 18);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.null_model.quantile, 0.98);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.null_model.min_good_frac, 0.82);
    EXPECT_EQ(ptcproc.cleaner.null_model.max_modes, 61);
    EXPECT_EQ(ptcproc.cleaner.null_model.max_samples, 19000);
    EXPECT_EQ(ptcproc.cleaner.null_model.seed, 23456U);
    EXPECT_EQ(ptcproc.cleaner.null_model.grouping,
              (std::vector<std::string>{"nw"}));
    EXPECT_TRUE(ptcproc.cleaner.marchenko_pastur.enabled);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.marchenko_pastur.min_good_frac, 0.83);
    EXPECT_EQ(ptcproc.cleaner.marchenko_pastur.max_modes, 62);
    EXPECT_EQ(ptcproc.cleaner.marchenko_pastur.max_samples, 18000);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.marchenko_pastur.band_low_Hz, 0.06);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.marchenko_pastur.band_high_Hz, 2.1);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.marchenko_pastur.clip_z, 11.0);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.marchenko_pastur.bulk_keep_frac, 0.79);
    EXPECT_EQ(ptcproc.cleaner.marchenko_pastur.q_grid_size, 63);
    EXPECT_EQ(ptcproc.cleaner.marchenko_pastur.grouping,
              (std::vector<std::string>{"array"}));
    EXPECT_TRUE(ptcproc.cleaner.adaptive_selector.enabled);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.adaptive_selector.min_good_frac, 0.71);
    EXPECT_EQ(ptcproc.cleaner.adaptive_selector.max_det, 121);
    EXPECT_EQ(ptcproc.cleaner.adaptive_selector.max_samples, 1025);
    EXPECT_EQ(ptcproc.cleaner.adaptive_selector.max_pairs, 2001);
    EXPECT_EQ(ptcproc.cleaner.adaptive_selector.seed, 34567U);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.adaptive_selector.clip_z, 49.0);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.adaptive_selector.low_weight, 0.9);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.adaptive_selector.tail_weight, 0.2);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.adaptive_selector.topmode_weight, 0.3);
    EXPECT_DOUBLE_EQ(ptcproc.cleaner.adaptive_selector.reg_weight, 0.4);
    EXPECT_EQ(ptcproc.cleaner.adaptive_selector.low_band_Hz,
              (std::array<double, 2>{0.07, 0.51}));
    EXPECT_EQ(ptcproc.cleaner.adaptive_selector.mid_band_Hz,
              (std::array<double, 2>{0.52, 2.2}));
    EXPECT_EQ(ptcproc.cleaner.adaptive_selector.candidate_offsets,
              (std::vector<int>{-1, 1, 3}));
    EXPECT_EQ(ptcproc.cleaner.adaptive_selector.grouping,
              (std::vector<std::string>{"all"}));
    EXPECT_TRUE(ptcproc.cleaner.adaptive_selector.log_candidates);
}

TEST(pipeline_execution, adapts_typed_processed_weighting_policy_one_way) {
    citlali::config::ProcessedTimeChunkWeightingConfig weighting;
    citlali::config::ProcessedTimeChunkFlaggingConfig flagging;
    weighting.type = citlali::config::ProcessedTimeChunkWeightingType::hybrid;
    weighting.source_mask_radius_arcsec = 17.0;
    weighting.hybrid_correction_min_factor = 0.4;
    weighting.hybrid_correction_max_factor = 2.4;
    weighting.median_map_weight_factor = 1.1;
    weighting.lower_map_weight_factor = 0.2;
    weighting.upper_map_weight_factor = 3.2;
    auto &validation = weighting.validation;
    validation.enabled = true;
    validation.accumulation_iters = 2;
    validation.apply_start_iter = 3;
    validation.min_valid_scans = 4;
    validation.min_factor = 0.15;
    validation.unvalidated_factor = 0.85;
    validation.require_fruitloops_model = false;
    validation.transient_ratio_enabled = true;
    validation.ratio_power = 1.2;
    validation.transient_ratio_power = 1.3;
    validation.upward_enabled = true;
    validation.upward_max_factor = 1.4;
    validation.upward_power = 1.5;
    validation.upward_min_base_factor = 0.75;
    validation.upward_require_atmospheric = false;
    validation.upward_min_atmospheric_factor = 0.65;
    validation.atmospheric_correlation_enabled = false;
    validation.atmospheric_grouping =
        citlali::config::ProcessedTimeChunkWeightGrouping::network;
    validation.atmospheric_min_detectors = 7;
    validation.atmospheric_ref = 0.11;
    validation.atmospheric_span = 0.22;
    validation.atmospheric_power = 1.7;
    validation.min_good_frac = 0.55;
    validation.min_overlap = 123;
    validation.max_samples = 456;
    validation.high_weight_validation_enabled = false;
    validation.high_weight_apply_caps = false;
    validation.high_weight_grouping =
        citlali::config::ProcessedTimeChunkWeightGrouping::all;
    validation.high_weight_min_group_detectors = 8;
    validation.high_weight_log_robust_z = 5.5;
    validation.high_weight_max_median_factor = 7.5;
    validation.high_weight_cap_median_factor = 3.5;
    validation.high_weight_min_validated_factor = 0.92;
    auto &penalty = weighting.corr_penalty;
    penalty.enabled = true;
    penalty.min_good_frac = 0.66;
    penalty.min_overlap = 234;
    penalty.max_samples = 567;
    penalty.max_pairs = 89;
    penalty.seed = 42;
    penalty.floor = 0.07;
    penalty.exponent = 2.3;
    penalty.pair_corr = {false, 0.12, 0.23, 0.34};
    penalty.cm_el_corr = {true, 0.45, 0.56, 0.67};
    penalty.cm_low_mid_ratio.enabled = true;
    penalty.cm_low_mid_ratio.ref = 0.78;
    penalty.cm_low_mid_ratio.span = 0.89;
    penalty.cm_low_mid_ratio.weight = 0.91;
    penalty.cm_low_mid_ratio.low_band_Hz = {0.1, 0.4};
    penalty.cm_low_mid_ratio.mid_band_Hz = {0.5, 2.5};
    auto &busy = weighting.busy_row_suppression;
    busy.enabled = true;
    busy.require_busy_veto = false;
    busy.min_candidate_clusters = 9;
    busy.min_max_unflagged_residual_z = 23.0;
    busy.factor = 0.25;
    flagging.lower_tod_inv_var_factor = 0.2;
    flagging.upper_tod_inv_var_factor = 4.2;
    FakeProcessedAdapterPtcProc ptcproc;

    citlali::pipeline::apply_processed_weighting_config_to_processor(
        weighting, flagging, ptcproc);

    EXPECT_EQ(ptcproc.weighting_type, "hybrid");
    EXPECT_DOUBLE_EQ(ptcproc.source_mask_radius_arcsec, 17.0);
    EXPECT_DOUBLE_EQ(ptcproc.hybrid_correction_min_factor, 0.4);
    EXPECT_DOUBLE_EQ(ptcproc.hybrid_correction_max_factor, 2.4);
    EXPECT_DOUBLE_EQ(ptcproc.med_weight_factor, 1.1);
    EXPECT_DOUBLE_EQ(ptcproc.lower_weight_factor, 0.2);
    EXPECT_DOUBLE_EQ(ptcproc.upper_weight_factor, 3.2);
    EXPECT_DOUBLE_EQ(ptcproc.lower_inv_var_factor, 0.2);
    EXPECT_DOUBLE_EQ(ptcproc.upper_inv_var_factor, 4.2);

    const auto &actual_validation = ptcproc.weight_validation;
    EXPECT_TRUE(actual_validation.enabled);
    EXPECT_EQ(actual_validation.accumulation_iters, 2);
    EXPECT_EQ(actual_validation.apply_start_iter, 3);
    EXPECT_EQ(actual_validation.min_valid_scans, 4);
    EXPECT_DOUBLE_EQ(actual_validation.min_factor, 0.15);
    EXPECT_DOUBLE_EQ(actual_validation.unvalidated_factor, 0.85);
    EXPECT_FALSE(actual_validation.require_fruitloops_model);
    EXPECT_TRUE(actual_validation.transient_ratio_enabled);
    EXPECT_DOUBLE_EQ(actual_validation.ratio_power, 1.2);
    EXPECT_DOUBLE_EQ(actual_validation.transient_ratio_power, 1.3);
    EXPECT_TRUE(actual_validation.upward_enabled);
    EXPECT_DOUBLE_EQ(actual_validation.upward_max_factor, 1.4);
    EXPECT_DOUBLE_EQ(actual_validation.upward_power, 1.5);
    EXPECT_DOUBLE_EQ(actual_validation.upward_min_base_factor, 0.75);
    EXPECT_FALSE(actual_validation.upward_require_atmospheric);
    EXPECT_DOUBLE_EQ(actual_validation.upward_min_atmospheric_factor, 0.65);
    EXPECT_FALSE(actual_validation.atmospheric_correlation_enabled);
    EXPECT_EQ(actual_validation.atmospheric_grouping, "nw");
    EXPECT_EQ(actual_validation.atmospheric_min_detectors, 7);
    EXPECT_DOUBLE_EQ(actual_validation.atmospheric_ref, 0.11);
    EXPECT_DOUBLE_EQ(actual_validation.atmospheric_span, 0.22);
    EXPECT_DOUBLE_EQ(actual_validation.atmospheric_power, 1.7);
    EXPECT_DOUBLE_EQ(actual_validation.min_good_frac, 0.55);
    EXPECT_EQ(actual_validation.min_overlap, 123);
    EXPECT_EQ(actual_validation.max_samples, 456);
    EXPECT_FALSE(actual_validation.high_weight_validation_enabled);
    EXPECT_FALSE(actual_validation.high_weight_apply_caps);
    EXPECT_EQ(actual_validation.high_weight_grouping, "all");
    EXPECT_EQ(actual_validation.high_weight_min_group_detectors, 8);
    EXPECT_DOUBLE_EQ(actual_validation.high_weight_log_robust_z, 5.5);
    EXPECT_DOUBLE_EQ(actual_validation.high_weight_max_median_factor, 7.5);
    EXPECT_DOUBLE_EQ(actual_validation.high_weight_cap_median_factor, 3.5);
    EXPECT_DOUBLE_EQ(actual_validation.high_weight_min_validated_factor,
                     0.92);

    const auto &actual_penalty = ptcproc.weight_corr_penalty;
    EXPECT_TRUE(actual_penalty.enabled);
    EXPECT_DOUBLE_EQ(actual_penalty.min_good_frac, 0.66);
    EXPECT_EQ(actual_penalty.min_overlap, 234);
    EXPECT_EQ(actual_penalty.max_samples, 567);
    EXPECT_EQ(actual_penalty.max_pairs, 89);
    EXPECT_EQ(actual_penalty.seed, 42U);
    EXPECT_DOUBLE_EQ(actual_penalty.floor, 0.07);
    EXPECT_DOUBLE_EQ(actual_penalty.exponent, 2.3);
    EXPECT_FALSE(actual_penalty.pair_corr.enabled);
    EXPECT_DOUBLE_EQ(actual_penalty.pair_corr.ref, 0.12);
    EXPECT_DOUBLE_EQ(actual_penalty.pair_corr.span, 0.23);
    EXPECT_DOUBLE_EQ(actual_penalty.pair_corr.weight, 0.34);
    EXPECT_TRUE(actual_penalty.cm_el_corr.enabled);
    EXPECT_DOUBLE_EQ(actual_penalty.cm_el_corr.ref, 0.45);
    EXPECT_DOUBLE_EQ(actual_penalty.cm_el_corr.span, 0.56);
    EXPECT_DOUBLE_EQ(actual_penalty.cm_el_corr.weight, 0.67);
    EXPECT_TRUE(actual_penalty.cm_low_mid_ratio.enabled);
    EXPECT_DOUBLE_EQ(actual_penalty.cm_low_mid_ratio.ref, 0.78);
    EXPECT_DOUBLE_EQ(actual_penalty.cm_low_mid_ratio.span, 0.89);
    EXPECT_DOUBLE_EQ(actual_penalty.cm_low_mid_ratio.weight, 0.91);
    EXPECT_DOUBLE_EQ(actual_penalty.cm_low_mid_ratio.low_min_Hz, 0.1);
    EXPECT_DOUBLE_EQ(actual_penalty.cm_low_mid_ratio.low_max_Hz, 0.4);
    EXPECT_DOUBLE_EQ(actual_penalty.cm_low_mid_ratio.mid_min_Hz, 0.5);
    EXPECT_DOUBLE_EQ(actual_penalty.cm_low_mid_ratio.mid_max_Hz, 2.5);

    EXPECT_TRUE(ptcproc.busy_row_suppression.enabled);
    EXPECT_FALSE(ptcproc.busy_row_suppression.require_busy_veto);
    EXPECT_EQ(ptcproc.busy_row_suppression.min_candidate_clusters, 9);
    EXPECT_DOUBLE_EQ(
        ptcproc.busy_row_suppression.min_max_unflagged_residual_z, 23.0);
    EXPECT_DOUBLE_EQ(ptcproc.busy_row_suppression.factor, 0.25);
}

TEST(pipeline_execution, adapts_typed_second_pass_policy_one_way) {
    citlali::config::ProcessedTimeChunkSecondPassLocalConfig config;
    config.enabled = true;
    config.min_spike_sigma = 11.0;
    config.min_good_frac = 0.6;
    config.baseline_window_sec = 0.31;
    config.sigma_scale = 0.72;
    config.delta_sigma_scale = 0.73;
    config.raw_candidate_rel_sigma_scale = 1.2;
    config.raw_window_sec = 0.19;
    config.raw_half_peak_frac = 0.51;
    config.raw_max_width_sec = 0.21;
    config.delta_window_sec = 0.13;
    config.delta_half_peak_frac = 0.52;
    config.delta_max_width_sec = 0.11;
    config.max_step_shift_z = 3.2;
    config.high_score_event_override = 21.0;
    config.merge_within_detector_sec = 0.09;
    config.cluster_events_sec = 0.10;
    config.min_cluster_detectors = 6;
    config.high_score_cluster_override = 9.5;
    config.max_auto_flag_clusters_per_network = 4;
    config.selective_busy_network_acceptance_enabled = false;
    config.source_protection.enabled = true;
    config.source_protection.radius_arcsec = 31.0;
    FakeProcessedAdapterPtcProc ptcproc;

    citlali::pipeline::apply_second_pass_local_config_to_processor(
        config, ptcproc);

    EXPECT_TRUE(ptcproc.second_pass_local.enabled);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.min_spike_sigma, 11.0);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.min_good_frac, 0.6);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.baseline_window_sec, 0.31);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.sigma_scale, 0.72);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.delta_sigma_scale, 0.73);
    EXPECT_DOUBLE_EQ(
        ptcproc.second_pass_local.raw_candidate_rel_sigma_scale, 1.2);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.raw_window_sec, 0.19);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.raw_half_peak_frac, 0.51);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.raw_max_width_sec, 0.21);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.delta_window_sec, 0.13);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.delta_half_peak_frac, 0.52);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.delta_max_width_sec, 0.11);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.max_step_shift_z, 3.2);
    EXPECT_DOUBLE_EQ(
        ptcproc.second_pass_local.high_score_event_override, 21.0);
    EXPECT_DOUBLE_EQ(
        ptcproc.second_pass_local.merge_within_detector_sec, 0.09);
    EXPECT_DOUBLE_EQ(ptcproc.second_pass_local.cluster_events_sec, 0.10);
    EXPECT_EQ(ptcproc.second_pass_local.min_cluster_detectors, 6);
    EXPECT_DOUBLE_EQ(
        ptcproc.second_pass_local.high_score_cluster_override, 9.5);
    EXPECT_EQ(
        ptcproc.second_pass_local.max_auto_flag_clusters_per_network, 4);
    EXPECT_FALSE(
        ptcproc.second_pass_local.selective_busy_network_acceptance_enabled);
    EXPECT_TRUE(ptcproc.second_pass_local.source_protection_config_enabled);
    EXPECT_DOUBLE_EQ(
        ptcproc.second_pass_local.source_protection_radius_arcsec, 31.0);
    EXPECT_TRUE(ptcproc.second_pass_local.source_protection_enabled);
}

TEST(config_scaffold, normalizes_processed_clean_group_aliases) {
    EXPECT_EQ(
        citlali::pipeline::normalize_processed_clean_group("Network"),
        "nw");
    EXPECT_EQ(
        citlali::pipeline::normalize_processed_clean_group("CORR_NW"),
        "corr_nw");
    EXPECT_TRUE(citlali::pipeline::is_supported_processed_clean_group("fg"));
    EXPECT_FALSE(
        citlali::pipeline::is_supported_processed_clean_group("unknown"));

    const auto resolution =
        citlali::pipeline::resolve_processed_clean_grouping(
            {"Network", "nw", "ARRAY", "unknown"});
    EXPECT_EQ(resolution.effective,
              (std::vector<std::string>{"nw", "array"}));
    EXPECT_EQ(resolution.unsupported,
              (std::vector<std::string>{"unknown"}));
    EXPECT_EQ(resolution.duplicates, (std::vector<std::string>{"nw"}));
    EXPECT_EQ(resolution.aliases_normalized, 2);

    citlali::config::ProcessedTimeChunkCleanConfig cleaners;
    cleaners.enabled = true;
    cleaners.standard_pca.enabled = true;
    cleaners.null_model.enabled = true;
    const auto mode_resolution =
        citlali::pipeline::resolve_processed_cleaner_mode(cleaners);
    EXPECT_EQ(
        mode_resolution.effective,
        citlali::config::ProcessedTimeChunkCleanerMode::standard_pca);
    EXPECT_EQ(mode_resolution.enabled_mode_count, 2);
}

TEST(config_scaffold, resolves_processed_weighting_source_mask_inheritance) {
    const auto inherited =
        citlali::pipeline::resolve_processed_weighting_source_mask(
            std::nullopt, 24.0);
    EXPECT_FALSE(inherited.requested.has_value());
    EXPECT_DOUBLE_EQ(inherited.effective, 24.0);
    EXPECT_TRUE(inherited.inherited_from_cleaning);

    const auto explicit_zero =
        citlali::pipeline::resolve_processed_weighting_source_mask(0.0,
                                                                   24.0);
    ASSERT_TRUE(explicit_zero.requested.has_value());
    EXPECT_DOUBLE_EQ(*explicit_zero.requested, 0.0);
    EXPECT_DOUBLE_EQ(explicit_zero.effective, 0.0);
    EXPECT_FALSE(explicit_zero.inherited_from_cleaning);
}

TEST(config_scaffold, separates_processed_requested_and_effective_state) {
    citlali::config::TimestreamConfig requested;
    requested.fruit_loops.enabled = true;
    requested.fruit_loops.max_iters = 4;
    requested.processed_time_chunk.clean.enabled = true;
    requested.processed_time_chunk.clean.mask_radius_arcsec = 18.0;

    auto plan =
        citlali::pipeline::make_processed_timestream_execution_plan(
            requested);
    plan.effective.fruit_loops.max_iters = 1;
    plan.effective.processed_time_chunk.clean.mask_radius_arcsec = 24.0;
    plan.effective_resolutions.fruit_loop_iterations =
        citlali::pipeline::resolve_fruit_loop_iteration_policy(
            requested.fruit_loops,
            citlali::config::ReductionType::beammap);

    EXPECT_TRUE(plan.initialized);
    EXPECT_EQ(plan.requested.fruit_loops.max_iters, 4);
    EXPECT_DOUBLE_EQ(
        plan.requested.processed_time_chunk.clean.mask_radius_arcsec, 18.0);
    EXPECT_EQ(plan.effective.fruit_loops.max_iters, 1);
    EXPECT_DOUBLE_EQ(
        plan.effective.processed_time_chunk.clean.mask_radius_arcsec, 24.0);
    ASSERT_TRUE(
        plan.effective_resolutions.fruit_loop_iterations.has_value());
    EXPECT_TRUE(plan.effective_resolutions.fruit_loop_iterations
                    ->forced_single_iteration_for_beammap);
    EXPECT_FALSE(plan.realized.fruit_loop_iterations_completed.has_value());
}

TEST(config_scaffold, resets_all_processed_plan_state_between_runs) {
    citlali::config::TimestreamConfig first_request;
    first_request.fruit_loops.enabled = true;
    first_request.fruit_loops.max_iters = 4;
    auto plan =
        citlali::pipeline::make_processed_timestream_execution_plan(
            first_request);
    plan.effective.fruit_loops.max_iters = 1;
    plan.effective_resolutions.fruit_loop_iterations =
        citlali::pipeline::resolve_fruit_loop_iteration_policy(
            first_request.fruit_loops,
            citlali::config::ReductionType::beammap);
    plan.realized.fruit_loop_iterations_completed = 1;
    plan.realized.fruit_loops_converged = true;
    plan.realized.source_protection =
        citlali::pipeline::SourceProtectionActivationResolution{
            true, true, true, true, true};

    citlali::config::TimestreamConfig second_request;
    second_request.fruit_loops.enabled = false;
    second_request.fruit_loops.max_iters = 7;
    second_request.processed_time_chunk.weighting.validation.enabled = false;
    second_request.processed_time_chunk.weighting.validation.min_factor =
        0.37;

    citlali::pipeline::reset_processed_timestream_execution_plan(
        plan, second_request);

    EXPECT_TRUE(plan.initialized);
    EXPECT_FALSE(plan.requested.fruit_loops.enabled);
    EXPECT_EQ(plan.requested.fruit_loops.max_iters, 7);
    EXPECT_DOUBLE_EQ(
        plan.requested.processed_time_chunk.weighting.validation.min_factor,
        0.37);
    EXPECT_FALSE(plan.effective.fruit_loops.enabled);
    EXPECT_EQ(plan.effective.fruit_loops.max_iters, 7);
    EXPECT_FALSE(
        plan.effective_resolutions.fruit_loop_iterations.has_value());
    EXPECT_FALSE(
        plan.effective_resolutions.weighting_dependencies.has_value());
    EXPECT_FALSE(plan.realized.source_protection.has_value());
    EXPECT_FALSE(plan.realized.fruit_loop_iterations_completed.has_value());
    EXPECT_FALSE(plan.realized.fruit_loops_converged.has_value());
}

TEST(config_scaffold, serializes_processed_config_snapshot_deterministically) {
    citlali::config::TimestreamConfig config;
    config.fruit_loops.enabled = true;
    config.fruit_loops.mode = citlali::config::FruitLoopsMode::both;
    config.fruit_loops.array_flux_limit = {1.0, 2.0};
    config.fruit_loops.weight_feedback.reference =
        citlali::config::FruitLoopsWeightFeedbackReference::median;
    auto &clean = config.processed_time_chunk.clean;
    clean.enabled = true;
    clean.active =
        citlali::config::ProcessedTimeChunkCleanerMode::standard_pca;
    clean.grouping = {"array", "nw"};
    clean.standard_pca.n_eig_to_cut["a1100"] = {2, 3};
    clean.corr_grouping.metric =
        citlali::config::ProcessedTimeChunkCorrGroupingMetric::signed_metric;
    clean.null_model.grouping = {};
    clean.adaptive_selector.low_band_Hz = {0.1, 0.5};
    auto &weighting = config.processed_time_chunk.weighting;
    weighting.type =
        citlali::config::ProcessedTimeChunkWeightingType::validated;
    weighting.validation.atmospheric_grouping =
        citlali::config::ProcessedTimeChunkWeightGrouping::network;
    weighting.corr_penalty.seed = 42;
    weighting.corr_penalty.cm_low_mid_ratio.mid_band_Hz = {0.6, 2.1};
    weighting.busy_row_suppression.enabled = true;
    auto &second_pass =
        config.processed_time_chunk.flagging.second_pass_local;
    second_pass.enabled = true;
    second_pass.source_protection.enabled = true;
    second_pass.source_protection.active = true;
    const auto snapshot =
        citlali::pipeline::snapshot_processed_timestream_config(config);

    const auto node =
        citlali::pipeline::processed_timestream_config_snapshot_node(
            snapshot);

    EXPECT_TRUE(node["fruit_loops"]["enabled"].as<bool>());
    EXPECT_EQ(node["fruit_loops"]["mode"].as<std::string>(), "both");
    EXPECT_EQ(node["fruit_loops"]["array_flux_limit"].size(), 2U);
    EXPECT_EQ(node["fruit_loops"]["weight_feedback"]["reference"]
                  .as<std::string>(),
              "median");
    const auto clean_node = node["processed_time_chunk"]["clean"];
    EXPECT_EQ(clean_node["active"].as<std::string>(), "standard_pca");
    EXPECT_EQ(clean_node["grouping"].size(), 2U);
    EXPECT_EQ(clean_node["standard_pca"]["n_eig_to_cut"]["a1100"][1]
                  .as<int>(),
              3);
    EXPECT_EQ(clean_node["corr_grouping"]["metric"].as<std::string>(),
              "signed");
    EXPECT_TRUE(clean_node["null_model"]["grouping"].IsSequence());
    EXPECT_EQ(clean_node["null_model"]["grouping"].size(), 0U);
    EXPECT_DOUBLE_EQ(
        clean_node["adaptive_selector"]["low_band_Hz"][1].as<double>(),
        0.5);
    const auto weighting_node =
        node["processed_time_chunk"]["weighting"];
    EXPECT_EQ(weighting_node["type"].as<std::string>(), "validated");
    EXPECT_EQ(weighting_node["validation"]["atmospheric_grouping"]
                  .as<std::string>(),
              "nw");
    EXPECT_EQ(weighting_node["corr_penalty"]["seed"].as<int>(), 42);
    EXPECT_DOUBLE_EQ(
        weighting_node["corr_penalty"]["cm_low_mid_ratio"]
                      ["mid_band_Hz"][1]
                          .as<double>(),
        2.1);
    EXPECT_TRUE(
        weighting_node["busy_row_suppression"]["enabled"].as<bool>());
    const auto second_pass_node = node["processed_time_chunk"]["flagging"]
                                      ["second_pass_local"];
    EXPECT_TRUE(second_pass_node["enabled"].as<bool>());
    EXPECT_TRUE(second_pass_node["source_protection"]["enabled"].as<bool>());
    EXPECT_TRUE(second_pass_node["source_protection"]["active"].as<bool>());
    EXPECT_EQ(YAML::Dump(node), YAML::Dump(
        citlali::pipeline::processed_timestream_config_snapshot_node(
            snapshot)));
}

TEST(config_scaffold, resolves_processed_weighting_dependencies) {
    citlali::config::ProcessedTimeChunkWeightingConfig weighting;
    citlali::config::ProcessedTimeChunkFlaggingConfig flagging;
    weighting.type =
        citlali::config::ProcessedTimeChunkWeightingType::validated;
    weighting.validation.enabled = false;
    weighting.busy_row_suppression.enabled = true;
    flagging.second_pass_local.enabled = false;
    auto logger = std::make_shared<FakeLogger>();

    const auto resolution = citlali::pipeline::resolve_processed_weighting(
        weighting, flagging);

    EXPECT_FALSE(weighting.validation.enabled);
    EXPECT_TRUE(weighting.busy_row_suppression.enabled);
    EXPECT_TRUE(resolution.effective.validation.enabled);
    EXPECT_FALSE(resolution.effective.busy_row_suppression.enabled);
    EXPECT_TRUE(resolution.validation_forced_by_weighting_type);
    EXPECT_TRUE(resolution.busy_row_disabled_without_second_pass);

    citlali::pipeline::resolve_processed_weighting_dependencies(
        weighting, flagging, logger);

    EXPECT_TRUE(weighting.validation.enabled);
    EXPECT_FALSE(weighting.busy_row_suppression.enabled);
    EXPECT_EQ(logger->warn_calls, 2);
}

TEST(pipeline_execution, skips_initial_fruit_loop_map_without_path) {
    FakeEngine engine;
    engine.iteration.fruit_iter = 0;
    engine.typed_config.timestream.fruit_loops.enabled = true;
    engine.typed_config.timestream.fruit_loops.path = "null";
    engine.omb.obsnums = {"152389"};

    citlali::pipeline::load_initial_fruit_loop_maps_if_requested(engine);

    EXPECT_EQ(engine.ptcproc.load_mb_calls, 0);
}

TEST(pipeline_execution, loads_previous_saved_fruit_loop_map) {
    FakeEngine engine;
    engine.iteration.fruit_iter = 2;
    engine.typed_config.runtime.output_dir = "/data/out";
    engine.runtime_config_provenance =
        citlali::config::make_runtime_config_provenance(
            engine.typed_config.runtime, false);
    engine.output_paths.redu_dir_num = 12;
    engine.typed_config.timestream.fruit_loops.save_all_iters = true;
    engine.typed_config.timestream.fruit_loops.type = "obsnum/filtered";
    engine.omb.obsnums = {"152389"};
    engine.omb.cov_cut = 5.5;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_previous_fruit_loop_maps_if_needed(
        engine, logger);

    EXPECT_EQ(engine.ptcproc.load_mb_calls, 1);
    EXPECT_EQ(engine.ptcproc.loaded_filepath,
              "/data/out/redu11/152389/filtered/");
    EXPECT_EQ(engine.ptcproc.loaded_noise_filepath,
              "/data/out/redu11/152389/filtered/");
    EXPECT_DOUBLE_EQ(engine.ptcproc.tod_mb.cov_cut, 5.5);
    EXPECT_EQ(logger->info_calls, 1);
}

TEST(pipeline_execution, loads_previous_stored_fruit_loop_map) {
    FakeEngine engine;
    engine.iteration.fruit_iter = 3;
    engine.output_paths.redu_dir_name = "/data/current/redu12";
    engine.typed_config.timestream.fruit_loops.save_all_iters = false;
    engine.typed_config.timestream.fruit_loops.type = "coadd/raw";
    engine.omb.obsnums = {"152389"};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_previous_fruit_loop_maps_if_needed(
        engine, logger);

    EXPECT_EQ(engine.ptcproc.load_mb_calls, 1);
    EXPECT_EQ(engine.ptcproc.loaded_filepath,
              "/data/current/redu12/coadded/raw/");
    EXPECT_EQ(engine.ptcproc.loaded_noise_filepath,
              "/data/current/redu12/coadded/raw/");
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(pipeline_execution, skips_previous_fruit_loop_map_on_first_iteration) {
    FakeEngine engine;
    engine.iteration.fruit_iter = 0;
    engine.typed_config.timestream.fruit_loops.save_all_iters = true;
    engine.omb.obsnums = {"152389"};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_previous_fruit_loop_maps_if_needed(
        engine, logger);

    EXPECT_EQ(engine.ptcproc.load_mb_calls, 0);
    EXPECT_EQ(logger->info_calls, 0);
}

TEST(pipeline_execution, loads_observation_fruit_loop_maps_for_non_beammap) {
    FakeEngine engine;
    engine.iteration.fruit_iter = 0;
    engine.typed_config.timestream.fruit_loops.enabled = true;
    engine.typed_config.timestream.fruit_loops.path = "/data/fruit";
    engine.typed_config.timestream.fruit_loops.type = "obsnum/raw";
    engine.omb.obsnums = {"000123"};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_observation_fruit_loop_maps_if_needed<false>(
        engine, logger);

    EXPECT_EQ(engine.ptcproc.load_mb_calls, 1);
    EXPECT_EQ(engine.ptcproc.loaded_filepath, "/data/fruit/000123/raw/");
}

TEST(pipeline_execution, skips_observation_fruit_loop_maps_for_beammap) {
    FakeEngine engine;
    engine.iteration.fruit_iter = 0;
    engine.typed_config.timestream.fruit_loops.enabled = true;
    engine.typed_config.timestream.fruit_loops.path = "/data/fruit";
    engine.omb.obsnums = {"000123"};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::load_observation_fruit_loop_maps_if_needed<true>(
        engine, logger);

    EXPECT_EQ(engine.ptcproc.load_mb_calls, 0);
    EXPECT_EQ(logger->info_calls, 0);
}

TEST(pipeline_output_layout, derives_config_copy_destinations) {
    EXPECT_EQ(citlali::pipeline::config_copy_filename("70_reduce.yaml"),
              "70_reduce.yaml");
    EXPECT_EQ(citlali::pipeline::config_copy_filename(
                  "/tmp/redu/70_reduce.yaml"),
              "70_reduce.yaml");
    EXPECT_EQ(citlali::pipeline::config_copy_destination(
                  "/tmp/redu01", "/tmp/redu/70_reduce.yaml"),
              "/tmp/redu01/70_reduce.yaml");
}

TEST(pipeline_output_layout, formats_obsnums_with_legacy_padding) {
    EXPECT_EQ(citlali::pipeline::format_obsnum(42), "000042");
    EXPECT_EQ(citlali::pipeline::format_obsnum(1234567), "1234567");
}

TEST(pipeline_output_layout, configures_observation_output_layout) {
    FakeEngine engine;
    engine.output_paths.redu_dir_name = "/tmp/redu01";
    engine.omb.obsnums = {"old"};

    citlali::pipeline::configure_observation_output_layout(engine, 42);

    EXPECT_EQ(engine.observation_identity.obsnum, "000042");
    EXPECT_EQ(engine.output_paths.obsnum_dir_name, "/tmp/redu01/000042/");
    ASSERT_EQ(engine.omb.obsnums.size(), 1U);
    EXPECT_EQ(engine.omb.obsnums.front(), "000042");
    EXPECT_TRUE(engine.cmb.obsnums.empty());
}

TEST(pipeline_output_layout, adds_observation_number_to_coadd_layout) {
    FakeEngine engine;
    engine.typed_config.coadd.enabled = true;
    engine.cmb.obsnums = {"000001"};

    citlali::pipeline::configure_observation_output_layout(engine, 42);

    ASSERT_EQ(engine.cmb.obsnums.size(), 2U);
    EXPECT_EQ(engine.cmb.obsnums.back(), "000042");
}

TEST(pipeline_output_layout, reads_obsnum_from_rawobs_meta) {
    std::vector<FakeRawObsMeta> rawobs_kids_meta = {
        {75.0, 101},
        {122.0, 202},
    };
    auto logger = std::make_shared<FakeLogger>();

    const int obsnum = citlali::pipeline::obsnum_from_rawobs_meta(
        rawobs_kids_meta, logger);

    EXPECT_EQ(obsnum, 202);
    EXPECT_EQ(logger->debug_calls, 1);
}

TEST(pipeline_output_layout, prepares_observation_layout_from_rawobs_meta) {
    FakeEngine engine;
    engine.output_paths.redu_dir_name = "/tmp/citlali_scaffold_redu";
    std::vector<FakeRawObsMeta> rawobs_kids_meta = {
        {75.0, 101},
        {122.0, 202},
    };
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::prepare_observation_output_layout_from_rawobs_meta(
        engine, rawobs_kids_meta, logger);

    EXPECT_EQ(engine.observation_identity.obsnum, "000202");
    EXPECT_EQ(engine.output_paths.obsnum_dir_name,
              "/tmp/citlali_scaffold_redu/000202/");
    ASSERT_EQ(engine.omb.obsnums.size(), 1U);
    EXPECT_EQ(engine.omb.obsnums.front(), "000202");
    EXPECT_EQ(logger->debug_calls, 3);
}

TEST(pipeline_output_layout, prepares_iteration_output_layout_on_first_iter) {
    FakeCoaddTodProc todproc;
    todproc.engine().iteration.fruit_iter = 0;
    std::vector<std::string> config_filepaths;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::prepare_iteration_output_layout_if_needed(
        todproc, config_filepaths, logger);

    EXPECT_EQ(todproc.create_output_dir_calls, 1);
}

TEST(pipeline_output_layout, skips_iteration_output_layout_when_not_saved) {
    FakeCoaddTodProc todproc;
    todproc.engine().iteration.fruit_iter = 1;
    todproc.engine().ptcproc.save_all_iters = false;
    std::vector<std::string> config_filepaths;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::prepare_iteration_output_layout_if_needed(
        todproc, config_filepaths, logger);

    EXPECT_EQ(todproc.create_output_dir_calls, 0);
}

TEST(pipeline_output_layout, derives_gaps_log_filepath) {
    EXPECT_EQ(citlali::pipeline::gaps_log_filepath("/tmp/redu01/152389/"),
              "/tmp/redu01/152389//logs/gaps.log");
}

TEST(pipeline_output_layout, warns_when_timing_gaps_are_present) {
    FakeEngine engine;
    engine.observation_identity.obsnum = "152389";
    engine.alignment.gaps["roach0"] = 2;
    engine.typed_config.runtime.verbose = false;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::record_timing_gaps_if_needed(engine, logger);

    EXPECT_EQ(logger->warn_calls, 1);
}

}  // namespace
