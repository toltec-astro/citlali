#include <citlali_config/default_config.h>
#include <citlali/core/config/calibration_config.h>
#include <citlali/core/config/reduction_config.h>
#include <citlali/core/config/reduction_config_validation.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>
#include <citlali/core/pipeline/config_diagnostics_state.h>
#include <citlali/core/pipeline/config_schema_validation.h>
#include <citlali/core/pipeline/output_path_state.h>
#include <citlali/core/cli/config_loading.h>
#include <citlali/core/cli/reduction_runtime.h>
#include <citlali/core/cli/runtime_setup.h>
#include <citlali/core/cli/tod_processor_selection.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/beammap_execution_plan.h>
#include <citlali/core/pipeline/beammap_source_flux_config.h>
#include <citlali/core/pipeline/coadd_config_read.h>
#include <citlali/core/pipeline/coadd_execution_plan.h>
#include <citlali/core/pipeline/coadd_provenance.h>
#include <citlali/core/pipeline/citlali_config_read.h>
#include <citlali/core/pipeline/fruit_loop_paths.h>
#include <citlali/core/pipeline/iteration_lifecycle.h>
#include <citlali/core/pipeline/learning_config_adapter.h>
#include <citlali/core/pipeline/learning_config_read.h>
#include <citlali/core/pipeline/interface_sync_config_adapter.h>
#include <citlali/core/pipeline/map_geometry.h>
#include <citlali/core/pipeline/map_index_state.h>
#include <citlali/core/pipeline/mapmaking_execution_plan.h>
#include <citlali/core/pipeline/mapmaking_method_config.h>
#include <citlali/core/pipeline/mapmaking_output_config.h>
#include <citlali/core/pipeline/mapmaking_provenance.h>
#include <citlali/core/pipeline/mapmaking_provenance_lifecycle.h>
#include <citlali/core/pipeline/map_filter_config_policy.h>
#include <citlali/core/pipeline/noise_config_adapter.h>
#include <citlali/core/pipeline/noise_config_read.h>
#include <citlali/core/pipeline/noise_execution_plan.h>
#include <citlali/core/pipeline/noise_provenance.h>
#include <citlali/core/pipeline/pointing_config_adapter.h>
#include <citlali/core/pipeline/pointing_config_read.h>
#include <citlali/core/pipeline/pointing_execution_plan.h>
#include <citlali/core/pipeline/pointing_provenance.h>
#include <citlali/core/pipeline/pointing_offsets_config_read.h>
#include <citlali/core/pipeline/post_processing_config_read.h>
#include <citlali/core/pipeline/post_processing_execution_plan.h>
#include <citlali/core/pipeline/post_processing_provenance.h>
#include <citlali/core/pipeline/post_processing_provenance_lifecycle.h>
#include <citlali/core/pipeline/observation_execution.h>
#include <citlali/core/pipeline/observation_preflight.h>
#include <citlali/core/pipeline/output_layout.h>
#include <citlali/core/pipeline/output_netcdf_metadata.h>
#include <citlali/core/pipeline/phdu_reduction_config.h>
#include <citlali/core/pipeline/polarimetry_config_read.h>
#include <citlali/core/pipeline/polarimetry_execution_plan.h>
#include <citlali/core/pipeline/polarimetry_provenance.h>
#include <citlali/core/pipeline/raw_iir_filter_metadata.h>
#include <citlali/core/pipeline/raw_timestream_authority.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/pipeline/raw_timestream_execution_plan.h>
#include <citlali/core/pipeline/raw_timestream_observation_resolution.h>
#include <citlali/core/pipeline/raw_timestream_observation_shadow.h>
#include <citlali/core/pipeline/raw_timestream_provenance.h>
#include <citlali/core/pipeline/raw_timestream_provenance_lifecycle.h>
#include <citlali/core/pipeline/config_parse_tracking.h>
#include <citlali/core/pipeline/raw_filtering_config_read.h>
#include <citlali/core/pipeline/raw_flagging_config_read.h>
#include <citlali/core/pipeline/raw_timestream_config_serialization.h>
#include <citlali/core/pipeline/raw_timestream_config_read.h>
#include <citlali/core/pipeline/reduction_config_validation_logging.h>
#include <citlali/core/pipeline/processed_clean_config_read.h>
#include <citlali/core/pipeline/processed_clean_resolution.h>
#include <citlali/core/pipeline/processed_timestream_config_serialization.h>
#include <citlali/core/pipeline/processed_timestream_execution_plan.h>
#include <citlali/core/pipeline/processed_timestream_provenance.h>
#include <citlali/core/pipeline/processed_weighting_config_read.h>

#include <boost/random/mersenne_twister.hpp>
#include <citlali/core/pipeline/processed_weighting_resolution.h>
#include <citlali/core/pipeline/runtime_provenance_output.h>
#include <citlali/core/pipeline/source_finding_config_policy.h>
#include <citlali/core/pipeline/source_fitting_config_policy.h>
#include <citlali/core/pipeline/source_protection_activation.h>
#include <citlali/core/pipeline/timestream_output_provenance.h>
#include <citlali/core/pipeline/timestream_config_adapter_polarimetry.h>
#include <citlali/core/pipeline/timestream_config_adapter_processed.h>
#include <citlali/core/pipeline/timestream_config_adapter_raw.h>
#include <citlali/core/pipeline/timestream_run_context.h>
#include <citlali/core/pipeline/tod_output_state.h>
#include <citlali/core/utils/fits_io.h>
#include <kids/toltec/toltec.h>
#include <citlali/core/timestream/rtc/rtcproc.h>

#include <gtest/gtest.h>
#include <spdlog/sinks/null_sink.h>
#include <tula/config/yamlconfig.h>

#include <algorithm>
#include <array>
#include <functional>
#include <filesystem>
#include <initializer_list>
#include <limits>
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

struct FakeWienerFilterConfigTarget {
    std::string filter_type = "stale";
    std::string template_type = "stale";
    std::string kernel_template_tail_mode = "stale";
    bool run_lowpass = false;
    bool normalize_error = false;
    bool edge_guard_enabled = false;
    std::string edge_weight_threshold_mode = "stale";
    std::string edge_hits_threshold_mode = "stale";
    double edge_hits_core_fraction = -1.0;
    double edge_guard_radius_fwhm = -1.0;
    std::string edge_fill_mode = "stale";
    std::string edge_taper_mode = "stale";
    double edge_taper_min_fraction = -1.0;
    double denom_rel_tol = -1.0;
    double tail_frac_tol = -1.0;
    int max_loops = -1;
    int denom_check_iters = -1;
    int max_denom_iters = -1;
    int map_fitter = -1;
    std::string parallel_policy = "stale";
    std::map<std::string, double> template_fwhm_rad{{"stale", -1.0}};
};

struct FakeSourceFindingConfigTarget {
    double source_sigma = -1.0;
    double source_window_rad = -1.0;
    std::string source_finder_mode = "stale";
};

struct FakeSourceFittingConfigTarget {
    double bounding_box_pix = -1.0;
    double fitting_region_pix = -1.0;
    bool fit_angle = false;
    Eigen::VectorXd flux_limits;
    Eigen::VectorXd fwhm_limits;
    double flux_low = 0.1;
    double flux_high = 2.0;
    double fwhm_low = 0.1;
    double fwhm_high = 2.0;
};

void ensure_citlali_test_logger() {
    if (!spdlog::get("citlali_logger")) {
        auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        spdlog::register_logger(
            std::make_shared<spdlog::logger>("citlali_logger", sink));
    }
}

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
    citlali::pipeline::ProcessedTimestreamExecutionPlan
        processed_timestream_plan;
    citlali::pipeline::MapmakingExecutionPlan mapmaking_plan;
    citlali::pipeline::CoaddExecutionPlan coadd_plan;
    citlali::pipeline::NoiseExecutionPlan noise_plan;
    citlali::pipeline::MapIndexState map_indices;
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
    bool output_throws = false;
    int run_wiener_filter_calls = 0;
    int find_sources_calls = 0;
    int fit_maps_calls = 0;
    int setup_calls = 0;
    int pipeline_calls = 0;
    int get_astrometry_config_calls = 0;
    int get_photometry_config_calls = 0;
    int get_citlali_config_calls = 0;
    bool inject_config_error = false;
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
    void output(citlali::pipeline::StageProfileCollector &) {
        ++output_calls;
        if (output_throws) {
            throw std::runtime_error("injected map output failure");
        }
    }

    template <auto MapType, class MapBuffer>
    void run_wiener_filter(MapBuffer &) {
        ++run_wiener_filter_calls;
    }

    template <auto MapType, class MapBuffer>
    citlali::pipeline::SourceFitCardinality find_sources(MapBuffer &) {
        ++find_sources_calls;
        return {};
    }

    void fit_maps(citlali::pipeline::PointingFitStage) {
        ++fit_maps_calls;
    }

    void setup(citlali::pipeline::StageProfileCollector &) { ++setup_calls; }

    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, const RawObs &,
                  citlali::pipeline::StageProfileCollector &) {
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
        if (inject_config_error) {
            config_diagnostics.invalid_keys.push_back(
                {"mapmaking", "pixel_size"});
        }
        runtime_config_provenance =
            citlali::config::make_runtime_config_provenance(
                typed_config.runtime, false);
    }

    void write_learning_summary() { ++write_learning_summary_calls; }
};

struct FakeRawProvenanceEngine {
    citlali::config::ReductionConfig typed_config;
    citlali::pipeline::RawTimestreamExecutionPlan raw_timestream_plan;
    citlali::pipeline::OutputPathState output_paths;
    citlali::pipeline::TodOutputState tod_outputs;
    struct {
        Eigen::MatrixXi scan_indices;
    } telescope;
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
    YAML::Node root = YAML::Load("{}");

    const YAML::Node &get_node() const { return root; }

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

    void setup(citlali::pipeline::StageProfileCollector &) {
        ++setup_calls;
        event_order.push_back("setup");
    }

    void pipeline(FakeKidsProc &, const FakeRawObs &,
                  citlali::pipeline::StageProfileCollector &) {
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
    void output(citlali::pipeline::StageProfileCollector &) {
        ++output_calls;
    }

    template <auto MapType, class MapBuffer>
    void run_wiener_filter(MapBuffer &) {
        ++run_wiener_filter_calls;
    }

    template <auto MapType, class MapBuffer>
    citlali::pipeline::SourceFitCardinality find_sources(MapBuffer &) {
        ++find_sources_calls;
        return {};
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

    void create_output_dir(citlali::pipeline::StageProfileCollector &) {
        ++create_output_dir_calls;
    }

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

    void create_output_dir(citlali::pipeline::StageProfileCollector &) {
        ++create_output_dir_calls;
    }

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

    void create_output_dir(citlali::pipeline::StageProfileCollector &) {
        ++create_output_dir_calls;
    }

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

TEST(config_scaffold, reads_and_adapts_typed_jinc_filter_config) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    root["mapmaking"]["jinc_filter"]["r_max"] = 4.25;
    root["mapmaking"]["jinc_filter"]["subpixel_n"] = 3;
    root["mapmaking"]["jinc_filter"]["shape_params"]["a1100"] =
        std::vector<double>{1.2, 1.8, 2.4};
    auto yaml_config =
        tula::config::YamlConfig::from_str(YAML::Dump(root));
    const std::map<int, std::string> array_names = {
        {0, "a1100"}, {1, "a1400"}, {2, "a2000"}};
    citlali::config::MapmakingConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_mapmaking_method_request_config(
        yaml_config, citlali::config::MapMethod::jinc, array_names,
        request, diagnostics);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_DOUBLE_EQ(request.jinc_filter.r_max, 4.25);
    EXPECT_EQ(request.jinc_filter.subpixel_n, 3);
    EXPECT_EQ(request.jinc_filter.shape_params.at("a1100"),
              (std::array<double, 3>{1.2, 1.8, 2.4}));

    struct FakeJincMapmaker {
        double r_max = -1.0;
        int subpixel_n = -1;
        std::map<int, Eigen::VectorXd> shape_params = {
            {99, Eigen::VectorXd::Constant(3, -1.0)}};
    } legacy;
    citlali::pipeline::adapt_jinc_filter_config_one_way(
        request.jinc_filter, array_names, legacy);

    EXPECT_DOUBLE_EQ(legacy.r_max, 4.25);
    EXPECT_EQ(legacy.subpixel_n, 3);
    EXPECT_EQ(legacy.shape_params.size(), 3U);
    EXPECT_EQ(legacy.shape_params.at(0).size(), 3);
    EXPECT_DOUBLE_EQ(legacy.shape_params.at(0)(0), 1.2);
    EXPECT_DOUBLE_EQ(legacy.shape_params.at(0)(1), 1.8);
    EXPECT_DOUBLE_EQ(legacy.shape_params.at(0)(2), 2.4);
}

TEST(config_scaffold, rejects_malformed_typed_jinc_shape) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    root["mapmaking"]["jinc_filter"]["shape_params"]["a1400"] =
        std::vector<double>{1.0, 2.0};
    auto yaml_config =
        tula::config::YamlConfig::from_str(YAML::Dump(root));
    const std::map<int, std::string> array_names = {
        {0, "a1100"}, {1, "a1400"}, {2, "a2000"}};
    citlali::config::MapmakingConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_mapmaking_method_request_config(
        yaml_config, citlali::config::MapMethod::jinc, array_names,
        request, diagnostics);

    ASSERT_TRUE(diagnostics.has_errors());
    ASSERT_EQ(diagnostics.invalid_keys.size(), 1U);
    EXPECT_EQ(
        diagnostics.invalid_keys.front(),
        (std::vector<std::string>{
            "mapmaking", "jinc_filter", "shape_params", "a1400"}));
}

TEST(config_scaffold,
     reads_and_adapts_typed_maximum_likelihood_config) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    root["mapmaking"]["maximum_likelihood"]["max_iterations"] = 17;
    root["mapmaking"]["maximum_likelihood"]["tolerance"] = 2.5e-8;
    auto yaml_config =
        tula::config::YamlConfig::from_str(YAML::Dump(root));
    citlali::config::MapmakingConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;
    const std::map<int, std::string> array_names;

    citlali::pipeline::read_mapmaking_method_request_config(
        yaml_config, citlali::config::MapMethod::maximum_likelihood,
        array_names, request, diagnostics);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_EQ(request.maximum_likelihood.max_iterations, 17);
    EXPECT_DOUBLE_EQ(request.maximum_likelihood.tolerance, 2.5e-8);

    struct FakeMaximumLikelihoodMapmaker {
        int max_iterations = -1;
        double tolerance = -1.0;
    } legacy;
    citlali::pipeline::adapt_maximum_likelihood_config_one_way(
        request.maximum_likelihood, legacy);
    EXPECT_EQ(legacy.max_iterations, 17);
    EXPECT_DOUBLE_EQ(legacy.tolerance, 2.5e-8);
}

TEST(config_scaffold, reads_typed_mapmaking_output_request) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    root["mapmaking"]["coverage_cut"] = 0.35;
    root["mapmaking"]["pixel_size_arcsec"] = 2.0;
    root["mapmaking"]["cunit"] = "MJy/sr";
    root["mapmaking"]["x_size_pix"] = 7;
    root["mapmaking"]["y_size_pix"] = 9;
    root["mapmaking"]["crpix1"] = 3.5;
    root["mapmaking"]["crpix2"] = 4.5;
    root["mapmaking"]["crval1_J2000"] = 11.0;
    root["mapmaking"]["crval2_J2000"] = 12.0;
    root["mapmaking"]["tan_ra"] = 13.0;
    root["mapmaking"]["tan_dec"] = 14.0;
    auto yaml_config =
        tula::config::YamlConfig::from_str(YAML::Dump(root));
    citlali::config::MapmakingConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_mapmaking_output_request_config(
        yaml_config, request, diagnostics);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_DOUBLE_EQ(request.coverage_cut, 0.35);
    EXPECT_DOUBLE_EQ(request.pixel_size_arcsec, 2.0);
    EXPECT_EQ(request.unit, "MJy/sr");
    EXPECT_EQ(request.x_size_pix, 7);
    EXPECT_EQ(request.y_size_pix, 9);
    EXPECT_DOUBLE_EQ(request.crpix1, 3.5);
    EXPECT_DOUBLE_EQ(request.crpix2, 4.5);
    EXPECT_DOUBLE_EQ(request.crval1_j2000, 11.0);
    EXPECT_DOUBLE_EQ(request.crval2_j2000, 12.0);
    EXPECT_DOUBLE_EQ(request.tan_ra, 13.0);
    EXPECT_DOUBLE_EQ(request.tan_dec, 14.0);
}

TEST(config_scaffold, reads_complete_post_processing_request) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    root["post_processing"]["map_filtering"]["enabled"] = true;
    root["post_processing"]["map_filtering"]["type"] = "convolve";
    root["post_processing"]["map_filtering"]["edge_guard"]
        ["taper_mode"] = "cosine";
    root["post_processing"]["source_fitting"]["model"] = "gaussian";
    root["post_processing"]["source_finding"]["enabled"] = true;
    root["post_processing"]["source_finding"]["mode"] = "both";
    root["post_processing"]["map_histogram_n_bins"] = 31;
    root["wiener_filter"]["template_type"] = "airy";
    root["wiener_filter"]["kernel_template_tail_mode"] = "zero";
    root["wiener_filter"]["template_fwhm_arcsec"]["a1100"] = 7.5;
    auto yaml_config =
        tula::config::YamlConfig::from_str(YAML::Dump(root));
    citlali::config::PostProcessingConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_post_processing_request_config(
        yaml_config, request, diagnostics);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_TRUE(request.map_filtering.enabled);
    EXPECT_TRUE(request.map_filtering_enabled);
    EXPECT_EQ(
        request.map_filtering.type,
        citlali::config::MapFilterType::convolve);
    EXPECT_EQ(
        request.map_filtering.edge_guard.taper_mode,
        citlali::config::MapFilterEdgeTaperMode::cosine);
    EXPECT_EQ(
        request.map_filtering.template_type,
        citlali::config::MapFilterTemplateType::airy);
    EXPECT_EQ(
        request.map_filtering.kernel_template_tail_mode,
        citlali::config::MapFilterKernelTailMode::zero);
    EXPECT_DOUBLE_EQ(
        request.map_filtering.template_fwhm_arcsec.at("a1100"), 7.5);
    EXPECT_EQ(
        request.source_fitting.model,
        citlali::config::SourceFitModel::gaussian);
    EXPECT_TRUE(request.source_finding.enabled);
    EXPECT_TRUE(request.source_finding_enabled);
    EXPECT_EQ(request.source_finding.mode, "both");
    EXPECT_EQ(request.map_histogram_n_bins, 31);
}

TEST(config_scaffold, post_processing_request_preserves_disabled_values) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    root["post_processing"]["map_filtering"]["enabled"] = false;
    root["post_processing"]["source_finding"]["enabled"] = false;
    root["post_processing"]["source_finding"]["source_sigma"] = 8.5;
    root["post_processing"]["source_fitting"]["fitting_radius_arcsec"] =
        42.0;
    root["wiener_filter"]["denom_check_iters"] = 9;
    auto yaml_config =
        tula::config::YamlConfig::from_str(YAML::Dump(root));
    citlali::config::PostProcessingConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_post_processing_request_config(
        yaml_config, request, diagnostics);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_FALSE(request.map_filtering.enabled);
    EXPECT_EQ(request.map_filtering.denom_check_iters, 9);
    EXPECT_FALSE(request.source_finding.enabled);
    EXPECT_DOUBLE_EQ(request.source_finding.source_sigma, 8.5);
    EXPECT_DOUBLE_EQ(request.source_fitting.fitting_radius_arcsec, 42.0);
}

TEST(config_scaffold, post_processing_request_rejects_invalid_enum) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    root["wiener_filter"]["kernel_template_tail_mode"] = "invalid";
    auto yaml_config =
        tula::config::YamlConfig::from_str(YAML::Dump(root));
    citlali::config::PostProcessingConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_post_processing_request_config(
        yaml_config, request, diagnostics);

    ASSERT_TRUE(diagnostics.has_errors());
    EXPECT_EQ(
        diagnostics.invalid_keys,
        (std::vector<std::vector<std::string>>{{
            "wiener_filter", "kernel_template_tail_mode"}}));
}

TEST(config_scaffold,
     post_processing_plan_preserves_request_and_resolves_activation) {
    citlali::config::PostProcessingConfig request;
    citlali::config::set_map_filtering_enabled(request, true);
    citlali::config::set_source_finding_enabled(request, true);
    request.map_filtering.template_type =
        citlali::config::MapFilterTemplateType::gaussian;
    request.map_filtering.template_fwhm_arcsec["a1100"] = 5.0;
    citlali::pipeline::PostProcessingExecutionPlan plan;

    plan.reset_from_request(
        request, citlali::config::ReductionType::science, true, true);

    EXPECT_TRUE(plan.initialized);
    EXPECT_TRUE(plan.requested.map_filtering.enabled);
    EXPECT_TRUE(plan.requested.source_finding.enabled);
    EXPECT_TRUE(plan.effective.map_filtering.enabled);
    EXPECT_TRUE(plan.effective.source_finding.enabled);
    EXPECT_TRUE(plan.effective.source_fitting.active);
    EXPECT_TRUE(
        plan.effective_resolution.source_fitting_required_by_map_filtering);
    EXPECT_TRUE(
        plan.effective_resolution.source_fitting_required_by_source_finding);
    EXPECT_FALSE(
        plan.effective_resolution.source_fitting_required_by_reduction);
    EXPECT_TRUE(plan.effective_resolution.coadd_enabled);
    EXPECT_DOUBLE_EQ(
        plan.requested.map_filtering.template_fwhm_arcsec.at("a1100"),
        5.0);
}

TEST(config_scaffold,
     post_processing_plan_suppresses_products_without_mapmaking) {
    citlali::config::PostProcessingConfig request;
    citlali::config::set_map_filtering_enabled(request, true);
    citlali::config::set_source_finding_enabled(request, true);
    citlali::pipeline::PostProcessingExecutionPlan plan;

    plan.reset_from_request(
        request, citlali::config::ReductionType::pointing, false, false);

    EXPECT_TRUE(plan.requested.map_filtering.enabled);
    EXPECT_TRUE(plan.requested.source_finding.enabled);
    EXPECT_FALSE(plan.requested.source_fitting.active);
    EXPECT_FALSE(plan.effective.map_filtering.enabled);
    EXPECT_FALSE(plan.effective.source_finding.enabled);
    EXPECT_FALSE(plan.effective.source_fitting.active);
    EXPECT_TRUE(
        plan.effective_resolution.map_filtering_disabled_by_mapmaking);
    EXPECT_TRUE(
        plan.effective_resolution.source_finding_disabled_by_mapmaking);
    EXPECT_TRUE(
        plan.effective_resolution.source_fitting_required_by_reduction);
    EXPECT_TRUE(
        plan.effective_resolution.source_fitting_disabled_by_mapmaking);
}

TEST(config_scaffold,
     post_processing_plan_enables_pointing_fitter_without_filtering) {
    citlali::config::PostProcessingConfig request;
    citlali::pipeline::PostProcessingExecutionPlan plan;

    plan.reset_from_request(
        request, citlali::config::ReductionType::pointing, true, false);

    EXPECT_FALSE(plan.effective.map_filtering.enabled);
    EXPECT_FALSE(plan.effective.source_finding.enabled);
    EXPECT_TRUE(plan.effective.source_fitting.active);
    EXPECT_TRUE(
        plan.effective_resolution.source_fitting_required_by_reduction);
    EXPECT_FALSE(
        plan.effective_resolution.source_fitting_disabled_by_mapmaking);
}

TEST(config_scaffold, beammap_plan_preserves_request_without_mapmaking) {
    citlali::config::BeammapConfig request;
    request.iteration.max_iterations = 7;
    citlali::pipeline::BeammapExecutionPlan plan;

    plan.reset_from_request(request, {}, false);

    EXPECT_EQ(plan.requested().iteration.max_iterations, 7);
    EXPECT_EQ(plan.effective().iteration.max_iterations, 1);
}

TEST(config_scaffold, adapts_effective_map_filter_config_one_way) {
    citlali::config::MapFilterConfig config;
    config.enabled = true;
    config.type = citlali::config::MapFilterType::wiener_filter;
    config.template_type = citlali::config::MapFilterTemplateType::airy;
    config.kernel_template_tail_mode =
        citlali::config::MapFilterKernelTailMode::cosine;
    config.lowpass_only = true;
    config.normalize_errors = true;
    config.edge_guard.enabled = true;
    config.edge_guard.weight_threshold_mode = "coverage_cut";
    config.edge_guard.hits_threshold_mode = "core_median_fraction";
    config.edge_guard.hits_core_fraction = 0.2;
    config.edge_guard.guard_radius_fwhm = 1.5;
    config.edge_guard.fill_mode = "core_median";
    config.edge_guard.taper_mode =
        citlali::config::MapFilterEdgeTaperMode::cosine;
    config.edge_guard.taper_min_fraction = 0.3;
    config.denom_rel_tol = 2.0e-4;
    config.tail_frac_tol = 4.0e-2;
    config.max_loops = 321;
    config.denom_check_iters = 7;
    config.max_denom_iters = 123;
    config.template_fwhm_arcsec = {
        {"a1100", 5.0}, {"a1400", 6.3}, {"a2000", 9.5}};
    FakeWienerFilterConfigTarget target;

    citlali::pipeline::adapt_map_filter_config_one_way(
        config, 0.25, target);

    EXPECT_EQ(target.filter_type, "wiener_filter");
    EXPECT_EQ(target.template_type, "airy");
    EXPECT_EQ(target.kernel_template_tail_mode, "cosine");
    EXPECT_TRUE(target.run_lowpass);
    EXPECT_TRUE(target.normalize_error);
    EXPECT_TRUE(target.edge_guard_enabled);
    EXPECT_EQ(target.edge_weight_threshold_mode, "coverage_cut");
    EXPECT_EQ(target.edge_hits_threshold_mode, "core_median_fraction");
    EXPECT_DOUBLE_EQ(target.edge_hits_core_fraction, 0.2);
    EXPECT_DOUBLE_EQ(target.edge_guard_radius_fwhm, 1.5);
    EXPECT_EQ(target.edge_fill_mode, "core_median");
    EXPECT_EQ(target.edge_taper_mode, "cosine");
    EXPECT_DOUBLE_EQ(target.edge_taper_min_fraction, 0.3);
    EXPECT_DOUBLE_EQ(target.denom_rel_tol, 2.0e-4);
    EXPECT_DOUBLE_EQ(target.tail_frac_tol, 4.0e-2);
    EXPECT_EQ(target.max_loops, 321);
    EXPECT_EQ(target.denom_check_iters, 7);
    EXPECT_EQ(target.max_denom_iters, 123);
    EXPECT_DOUBLE_EQ(target.template_fwhm_rad.at("a1100"), 1.25);
    EXPECT_DOUBLE_EQ(target.template_fwhm_rad.at("a1400"), 1.575);
    EXPECT_DOUBLE_EQ(target.template_fwhm_rad.at("a2000"), 2.375);

    config.template_type = citlali::config::MapFilterTemplateType::kernel;
    citlali::pipeline::adapt_map_filter_config_one_way(
        config, 0.25, target);
    EXPECT_TRUE(target.template_fwhm_rad.empty());
}

TEST(config_scaffold, map_filter_prerequisites_throw_canonical_config_error) {
    citlali::config::NoiseConfig noise;
    noise.enabled = true;
    citlali::config::MapFilterConfig filter;
    filter.enabled = true;
    filter.template_type = citlali::config::MapFilterTemplateType::kernel;
    struct {
        bool run_kernel = false;
    } rtcproc;
    FakeWienerFilterConfigTarget target;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_THROW(
        citlali::pipeline::apply_map_filter_runtime_policy(
            noise, filter, rtcproc, 7, "seq", target, logger),
        citlali::error::Error);

    filter.template_type = citlali::config::MapFilterTemplateType::gaussian;
    filter.type = citlali::config::MapFilterType::wiener_filter;
    filter.lowpass_only = false;
    noise.enabled = false;
    EXPECT_THROW(
        citlali::pipeline::apply_map_filter_runtime_policy(
            noise, filter, rtcproc, 7, "seq", target, logger),
        citlali::error::Error);
    EXPECT_EQ(logger->error_calls, 2);
}

TEST(config_scaffold, adapts_effective_source_finding_config_one_way) {
    citlali::config::SourceFindingConfig config;
    config.enabled = true;
    config.source_sigma = 7.5;
    config.source_window_arcsec = 12.0;
    config.mode = "both";
    FakeSourceFindingConfigTarget observation_maps;
    FakeSourceFindingConfigTarget coadd_maps;

    citlali::pipeline::adapt_source_finding_config_one_way(
        config, 0.25, true, observation_maps, coadd_maps);

    EXPECT_DOUBLE_EQ(observation_maps.source_sigma, 7.5);
    EXPECT_DOUBLE_EQ(observation_maps.source_window_rad, 3.0);
    EXPECT_EQ(observation_maps.source_finder_mode, "both");
    EXPECT_DOUBLE_EQ(coadd_maps.source_sigma, 7.5);
    EXPECT_DOUBLE_EQ(coadd_maps.source_window_rad, 3.0);
    EXPECT_EQ(coadd_maps.source_finder_mode, "both");

    config.source_sigma = 6.0;
    config.source_window_arcsec = 8.0;
    config.mode = "negative";
    coadd_maps = {};
    citlali::pipeline::adapt_source_finding_config_one_way(
        config, 0.5, false, observation_maps, coadd_maps);

    EXPECT_DOUBLE_EQ(observation_maps.source_sigma, 6.0);
    EXPECT_DOUBLE_EQ(observation_maps.source_window_rad, 4.0);
    EXPECT_EQ(observation_maps.source_finder_mode, "negative");
    EXPECT_DOUBLE_EQ(coadd_maps.source_sigma, -1.0);
    EXPECT_DOUBLE_EQ(coadd_maps.source_window_rad, -1.0);
    EXPECT_EQ(coadd_maps.source_finder_mode, "stale");
}

TEST(config_scaffold, adapts_effective_source_fitting_config_one_way) {
    citlali::config::SourceFittingConfig config;
    config.active = true;
    config.bounding_box_arcsec = 30.0;
    config.fitting_radius_arcsec = 40.0;
    config.fit_rotation_angle = true;
    config.amp_limit_factors = {0.5, 2.5};
    config.fwhm_limit_factors = {0.6, 1.8};
    FakeSourceFittingConfigTarget target;

    citlali::pipeline::adapt_source_fitting_config_one_way(
        config, 0.25, 0.5, target);

    EXPECT_DOUBLE_EQ(target.bounding_box_pix, 60.0);
    EXPECT_DOUBLE_EQ(target.fitting_region_pix, 80.0);
    EXPECT_TRUE(target.fit_angle);
    ASSERT_EQ(target.flux_limits.size(), 2);
    ASSERT_EQ(target.fwhm_limits.size(), 2);
    EXPECT_DOUBLE_EQ(target.flux_limits(0), 0.5);
    EXPECT_DOUBLE_EQ(target.flux_limits(1), 2.5);
    EXPECT_DOUBLE_EQ(target.fwhm_limits(0), 0.6);
    EXPECT_DOUBLE_EQ(target.fwhm_limits(1), 1.8);
    EXPECT_DOUBLE_EQ(target.flux_low, 0.5);
    EXPECT_DOUBLE_EQ(target.flux_high, 2.5);
    EXPECT_DOUBLE_EQ(target.fwhm_low, 0.6);
    EXPECT_DOUBLE_EQ(target.fwhm_high, 1.8);

    config.amp_limit_factors = {0.0, 0.0};
    config.fwhm_limit_factors = {0.0, 0.0};
    FakeSourceFittingConfigTarget default_target;
    citlali::pipeline::adapt_source_fitting_config_one_way(
        config, 0.25, 0.5, default_target);
    EXPECT_DOUBLE_EQ(default_target.flux_low, 0.1);
    EXPECT_DOUBLE_EQ(default_target.flux_high, 2.0);
    EXPECT_DOUBLE_EQ(default_target.fwhm_low, 0.1);
    EXPECT_DOUBLE_EQ(default_target.fwhm_high, 2.0);
}

TEST(config_scaffold, records_complete_post_processing_point_iteration) {
    citlali::config::PostProcessingConfig request;
    citlali::config::set_map_filtering_enabled(request, true);
    citlali::config::set_source_finding_enabled(request, true);
    citlali::pipeline::PostProcessingExecutionPlan plan;
    plan.reset_from_request(
        request, citlali::config::ReductionType::pointing, true, false);
    plan.begin_iteration();

    citlali::config::MapmakingConfig mapmaking_request;
    mapmaking_request.enabled = true;
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        mapmaking_request, citlali::config::ReductionType::pointing);
    mapmaking.begin_iteration();
    mapmaking.begin_observation(0, "152389", 3, 1.0e-5, 6);
    citlali::pipeline::complete_mapmaking_observation(mapmaking);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);

    citlali::pipeline::record_post_processing_filter_completed(
        plan, citlali::pipeline::PostProcessingMapContext::observation, 3);
    citlali::pipeline::record_post_processing_catalog_fits_completed(
        plan, citlali::pipeline::PostProcessingMapContext::observation,
        195, 190);
    citlali::pipeline::record_post_processing_source_table_written(
        plan, citlali::pipeline::PostProcessingMapContext::observation,
        195);
    citlali::pipeline::record_post_processing_pointing_fits_completed(
        plan, false, 3, 3);
    citlali::pipeline::record_post_processing_pointing_fits_completed(
        plan, true, 3, 3);
    citlali::pipeline::record_post_processing_run_completed(
        plan, mapmaking);

    EXPECT_TRUE(plan.realized.reduction_completed);
    EXPECT_TRUE(plan.realized.outputs_completed);
    EXPECT_EQ(plan.realized.observation.filtered_map_count, 3U);
    EXPECT_EQ(plan.realized.observation.detected_source_count, 195U);
    EXPECT_EQ(plan.realized.observation.source_table_row_count, 195U);
    EXPECT_EQ(plan.realized.observation.catalog_fits.valid_count, 190U);
    EXPECT_EQ(plan.realized.pointing_raw_fits.attempt_count, 3U);
    EXPECT_EQ(plan.realized.pointing_filtered_fits.attempt_count, 3U);
}

TEST(config_scaffold, rejects_incomplete_post_processing_source_table) {
    citlali::config::PostProcessingConfig request;
    citlali::config::set_map_filtering_enabled(request, true);
    citlali::config::set_source_finding_enabled(request, true);
    citlali::pipeline::PostProcessingExecutionPlan plan;
    plan.reset_from_request(
        request, citlali::config::ReductionType::science, true, true);
    plan.begin_iteration();

    citlali::config::MapmakingConfig mapmaking_request;
    mapmaking_request.enabled = true;
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        mapmaking_request, citlali::config::ReductionType::science);
    mapmaking.begin_iteration();
    mapmaking.begin_observation(0, "science", 3, 1.0e-5, 3);
    citlali::pipeline::complete_mapmaking_observation(mapmaking);
    mapmaking.begin_coadd(3, 6);
    citlali::pipeline::complete_mapmaking_coadd(mapmaking);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);

    citlali::pipeline::record_post_processing_filter_completed(
        plan, citlali::pipeline::PostProcessingMapContext::coadd, 3);
    citlali::pipeline::record_post_processing_catalog_fits_completed(
        plan, citlali::pipeline::PostProcessingMapContext::coadd, 8, 7);
    EXPECT_THROW(
        citlali::pipeline::record_post_processing_run_completed(
            plan, mapmaking),
        std::logic_error);
    EXPECT_FALSE(plan.realized.reduction_completed);
}

TEST(config_scaffold, records_complete_post_processing_science_coadd) {
    citlali::config::PostProcessingConfig request;
    citlali::config::set_map_filtering_enabled(request, true);
    citlali::config::set_source_finding_enabled(request, true);
    citlali::pipeline::PostProcessingExecutionPlan plan;
    plan.reset_from_request(
        request, citlali::config::ReductionType::science, true, true);
    plan.begin_iteration();

    citlali::config::MapmakingConfig mapmaking_request;
    mapmaking_request.enabled = true;
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        mapmaking_request, citlali::config::ReductionType::science);
    mapmaking.begin_iteration();
    mapmaking.begin_observation(0, "science", 3, 1.0e-5, 3);
    citlali::pipeline::complete_mapmaking_observation(mapmaking);
    mapmaking.begin_coadd(3, 6);
    citlali::pipeline::complete_mapmaking_coadd(mapmaking);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);

    citlali::pipeline::record_post_processing_filter_completed(
        plan, citlali::pipeline::PostProcessingMapContext::coadd, 3);
    citlali::pipeline::record_post_processing_catalog_fits_completed(
        plan, citlali::pipeline::PostProcessingMapContext::coadd, 8, 7);
    citlali::pipeline::record_post_processing_source_table_written(
        plan, citlali::pipeline::PostProcessingMapContext::coadd, 8);
    citlali::pipeline::record_post_processing_run_completed(
        plan, mapmaking);

    EXPECT_EQ(plan.realized.observation.filter_context_count, 0U);
    EXPECT_EQ(plan.realized.coadd.filter_context_count, 1U);
    EXPECT_EQ(plan.realized.coadd.filtered_map_count, 3U);
    EXPECT_EQ(plan.realized.coadd.source_table_row_count, 8U);
}

TEST(config_scaffold, records_complete_post_processing_beammap_fits) {
    citlali::pipeline::PostProcessingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::PostProcessingConfig{},
        citlali::config::ReductionType::beammap, true, false);
    plan.begin_iteration();

    citlali::config::MapmakingConfig mapmaking_request;
    mapmaking_request.enabled = true;
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        mapmaking_request, citlali::config::ReductionType::beammap);
    mapmaking.begin_iteration();
    mapmaking.begin_observation(0, "beammap", 5234, 1.0e-5, 5234);
    citlali::pipeline::complete_mapmaking_observation(mapmaking);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);

    citlali::pipeline::record_post_processing_beammap_fits_completed(
        plan, 5234, 5135);
    citlali::pipeline::record_post_processing_run_completed(
        plan, mapmaking);

    EXPECT_EQ(plan.realized.beammap_fits.context_count, 1U);
    EXPECT_EQ(plan.realized.beammap_fits.attempt_count, 5234U);
    EXPECT_EQ(plan.realized.beammap_fits.valid_count, 5135U);
}

TEST(config_scaffold, atomically_writes_post_processing_provenance) {
    const auto output_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_post_processing_provenance_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);

    citlali::pipeline::PostProcessingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::PostProcessingConfig{},
        citlali::config::ReductionType::science, false, false);
    plan.begin_iteration();
    plan.realized.reduction_completed = true;
    plan.realized.outputs_completed = true;
    citlali::pipeline::write_post_processing_provenance_file(
        output_dir, plan);

    const auto output_path =
        citlali::pipeline::post_processing_provenance_path(output_dir);
    EXPECT_TRUE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    const auto stored = YAML::LoadFile(output_path.string());
    EXPECT_EQ(stored["schema_version"].as<std::string>(),
              "citlali-post-processing-provenance-v1");
    EXPECT_FALSE(stored["requested"]["map_filtering"]["enabled"]
                     .as<bool>());
    EXPECT_TRUE(stored["realized"]["outputs_completed"].as<bool>());
    std::filesystem::remove_all(output_dir);
}

TEST(config_scaffold, post_processing_provenance_failure_propagates) {
    const auto missing_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_missing_post_processing_provenance_dir" / "nested";
    std::filesystem::remove_all(missing_dir.parent_path());
    citlali::pipeline::PostProcessingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::PostProcessingConfig{},
        citlali::config::ReductionType::science, false, false);
    plan.realized.reduction_completed = true;
    plan.realized.outputs_completed = true;

    EXPECT_THROW(
        citlali::pipeline::write_post_processing_provenance_file(
            missing_dir, plan),
        std::exception);
}

TEST(config_scaffold, reads_typed_coadd_request) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    root["coadd"]["enabled"] = true;
    auto yaml_config =
        tula::config::YamlConfig::from_str(YAML::Dump(root));
    citlali::config::CoaddConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_coadd_request_config(
        yaml_config, request, diagnostics);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_TRUE(request.enabled);
}

TEST(config_scaffold, coadd_plan_preserves_disabled_request) {
    citlali::config::CoaddConfig request;
    request.enabled = true;
    citlali::pipeline::CoaddExecutionPlan plan;

    plan.reset_from_request(request, false);

    EXPECT_TRUE(plan.requested.enabled);
    EXPECT_FALSE(plan.effective.enabled);
    EXPECT_FALSE(plan.effective_resolution.mapmaking_enabled);
    EXPECT_TRUE(plan.effective_resolution.requested_enabled);
    EXPECT_FALSE(plan.effective_resolution.effective_enabled);
    EXPECT_TRUE(plan.effective_resolution.disabled_by_mapmaking);
}

TEST(config_scaffold, routes_coadd_accessor_through_effective_plan) {
    FakeEngine engine;
    engine.typed_config.coadd.enabled = true;
    engine.coadd_plan.reset_from_request(
        engine.typed_config.coadd, false);

    EXPECT_TRUE(engine.typed_config.coadd.enabled);
    EXPECT_TRUE(engine.coadd_plan.requested.enabled);
    EXPECT_FALSE(citlali::pipeline::coadd_config(engine).enabled);
}

TEST(config_scaffold, records_enabled_coadd_realized_cardinality) {
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::science);
    mapmaking.begin_iteration();
    mapmaking.begin_observation(0, "152390", 3, 9.696273622e-6, 3);
    citlali::pipeline::complete_mapmaking_observation(mapmaking);
    mapmaking.begin_coadd(3, 6);
    citlali::pipeline::complete_mapmaking_coadd(mapmaking);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);
    citlali::config::CoaddConfig request;
    request.enabled = true;
    citlali::pipeline::CoaddExecutionPlan coadd;
    coadd.reset_from_request(request, true);

    citlali::pipeline::record_coadd_run_completed(coadd, mapmaking);

    EXPECT_TRUE(coadd.realized.reduction_completed);
    EXPECT_TRUE(coadd.realized.coadd_executed);
    EXPECT_EQ(*coadd.realized.map_count, 3U);
    EXPECT_EQ(*coadd.realized.required_map_write_count, 6U);
    EXPECT_TRUE(coadd.realized.outputs_completed);
}

TEST(config_scaffold, records_effectively_disabled_coadd) {
    citlali::config::MapmakingConfig mapmaking_request;
    mapmaking_request.enabled = false;
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        mapmaking_request, citlali::config::ReductionType::science);
    mapmaking.begin_iteration();
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);
    citlali::config::CoaddConfig coadd_request;
    coadd_request.enabled = true;
    citlali::pipeline::CoaddExecutionPlan coadd;
    coadd.reset_from_request(coadd_request, false);

    citlali::pipeline::record_coadd_run_completed(coadd, mapmaking);

    EXPECT_TRUE(coadd.requested.enabled);
    EXPECT_FALSE(coadd.effective.enabled);
    EXPECT_TRUE(coadd.realized.reduction_completed);
    EXPECT_FALSE(coadd.realized.coadd_executed);
    EXPECT_FALSE(coadd.realized.map_count.has_value());
    EXPECT_FALSE(coadd.realized.required_map_write_count.has_value());
    EXPECT_FALSE(coadd.realized.outputs_completed);
}

TEST(config_scaffold, rejects_inconsistent_coadd_realized_state) {
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::science);
    mapmaking.begin_iteration();
    mapmaking.begin_observation(0, "152390", 3, 9.696273622e-6, 3);
    citlali::pipeline::complete_mapmaking_observation(mapmaking);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);
    citlali::config::CoaddConfig request;
    request.enabled = true;
    citlali::pipeline::CoaddExecutionPlan coadd;
    coadd.reset_from_request(request, true);

    EXPECT_THROW(
        citlali::pipeline::record_coadd_run_completed(coadd, mapmaking),
        std::logic_error);
}

TEST(config_scaffold, reads_complete_typed_noise_request) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    root["noise_maps"]["enabled"] = true;
    root["noise_maps"]["n_noise_maps"] = 7;
    root["noise_maps"]["randomize_dets"] = false;
    root["noise_maps"]["write_realizations"] = true;
    root["noise_maps"]["products"]["enabled"] = false;
    root["noise_maps"]["products"]["apply_empirical_weights"] = false;
    auto yaml_config =
        tula::config::YamlConfig::from_str(YAML::Dump(root));
    citlali::config::NoiseConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_noise_request_config(
        yaml_config, request, diagnostics);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_TRUE(request.enabled);
    EXPECT_EQ(request.n_noise_maps, 7);
    EXPECT_FALSE(request.randomize_dets);
    EXPECT_TRUE(request.write_realizations);
    EXPECT_FALSE(request.products_enabled);
    EXPECT_FALSE(request.apply_empirical_weights);
}

TEST(config_scaffold, preserves_legacy_noise_optional_defaults) {
    ensure_citlali_test_logger();
    auto yaml_config = tula::config::YamlConfig::from_str(
        "noise_maps:\n"
        "  enabled: false\n"
        "  n_noise_maps: 5\n"
        "  randomize_dets: true\n");
    citlali::config::NoiseConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_noise_request_config(
        yaml_config, request, diagnostics);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_FALSE(request.enabled);
    EXPECT_EQ(request.n_noise_maps, 5);
    EXPECT_TRUE(request.randomize_dets);
    EXPECT_FALSE(request.write_realizations);
    EXPECT_FALSE(request.products_enabled);
    EXPECT_FALSE(request.apply_empirical_weights);
}

TEST(config_scaffold, noise_plan_preserves_disabled_request) {
    citlali::config::NoiseConfig request;
    request.enabled = true;
    request.n_noise_maps = 10;
    citlali::pipeline::NoiseExecutionPlan plan;

    plan.reset_from_request(request, false);

    EXPECT_TRUE(plan.requested.enabled);
    EXPECT_EQ(plan.requested.n_noise_maps, 10);
    EXPECT_FALSE(plan.effective.enabled);
    EXPECT_EQ(plan.effective.n_noise_maps, 0);
    EXPECT_TRUE(plan.effective_resolution.disabled_by_mapmaking);
    EXPECT_TRUE(plan.effective_resolution.count_zeroed_while_disabled);
    EXPECT_EQ(
        plan.effective_resolution.random_seed,
        citlali::pipeline::noise_random_seed);
}

TEST(config_scaffold, routes_noise_accessor_through_effective_plan) {
    FakeEngine engine;
    engine.typed_config.noise.enabled = true;
    engine.typed_config.noise.n_noise_maps = 10;
    engine.noise_plan.reset_from_request(
        engine.typed_config.noise, false);

    EXPECT_TRUE(engine.typed_config.noise.enabled);
    EXPECT_EQ(engine.typed_config.noise.n_noise_maps, 10);
    EXPECT_FALSE(citlali::pipeline::noise_config(engine).enabled);
    EXPECT_EQ(citlali::pipeline::noise_config(engine).n_noise_maps, 0);
}

TEST(config_scaffold, adapts_effective_noise_config_one_way) {
    struct FakeMapBlock {
        int n_noise = -1;
        bool randomize_dets = true;
    } observation_maps, coadd_maps;
    citlali::config::NoiseConfig effective;
    effective.enabled = true;
    effective.n_noise_maps = 7;
    effective.randomize_dets = false;

    citlali::pipeline::adapt_noise_config_one_way(
        effective, false, observation_maps, coadd_maps);

    EXPECT_EQ(observation_maps.n_noise, 7);
    EXPECT_FALSE(observation_maps.randomize_dets);
    EXPECT_EQ(coadd_maps.n_noise, 0);
    EXPECT_FALSE(coadd_maps.randomize_dets);
    EXPECT_EQ(effective.n_noise_maps, 7);
}

TEST(config_scaffold, explicit_noise_seed_preserves_default_sequence) {
    boost::random::mt19937 legacy_default;
    boost::random::mt19937 explicit_seed{
        citlali::pipeline::noise_random_seed};

    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(legacy_default(), explicit_seed());
    }
}

TEST(config_scaffold, records_disabled_noise_run) {
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::pointing);
    mapmaking.begin_iteration();
    mapmaking.begin_observation(0, "152389", 3, 4.848136811e-6, 3);
    citlali::pipeline::complete_mapmaking_observation(mapmaking);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);
    citlali::pipeline::NoiseExecutionPlan noise;
    noise.reset_from_request(citlali::config::NoiseConfig{}, true);

    citlali::pipeline::record_noise_run_completed(
        noise, mapmaking, true);

    EXPECT_TRUE(noise.realized.reduction_completed);
    EXPECT_FALSE(noise.realized.generation_executed);
    EXPECT_FALSE(noise.realized.total_noise_realization_count.has_value());
    EXPECT_FALSE(noise.realized.outputs_completed);
}

TEST(config_scaffold, records_jinc_science_noise_cardinality) {
    citlali::config::MapmakingConfig mapmaking_request;
    mapmaking_request.method = citlali::config::MapMethod::jinc;
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        mapmaking_request, citlali::config::ReductionType::science);
    mapmaking.begin_iteration();
    mapmaking.begin_observation(0, "152390", 3, 4.848136811e-6, 3);
    citlali::pipeline::complete_mapmaking_observation(mapmaking);
    mapmaking.begin_observation(1, "152392", 3, 4.848136811e-6, 3);
    citlali::pipeline::complete_mapmaking_observation(mapmaking);
    mapmaking.begin_coadd(3, 6);
    citlali::pipeline::complete_mapmaking_coadd(mapmaking);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);
    citlali::config::NoiseConfig request;
    request.enabled = true;
    request.n_noise_maps = 10;
    request.products_enabled = false;
    citlali::pipeline::NoiseExecutionPlan noise;
    noise.reset_from_request(request, true);

    citlali::pipeline::record_noise_run_completed(
        noise, mapmaking, true);

    EXPECT_TRUE(noise.realized.generation_executed);
    EXPECT_EQ(*noise.realized.observation_scientific_map_count, 6U);
    EXPECT_EQ(*noise.realized.observation_noise_realization_count, 60U);
    EXPECT_EQ(*noise.realized.coadd_scientific_map_count, 3U);
    EXPECT_EQ(*noise.realized.coadd_noise_realization_count, 30U);
    EXPECT_EQ(*noise.realized.total_noise_realization_count, 90U);
    EXPECT_EQ(*noise.realized.empirical_product_map_count, 0U);
    EXPECT_EQ(*noise.realized.realization_image_write_count, 0U);
    EXPECT_TRUE(noise.realized.outputs_completed);
}

TEST(config_scaffold, records_full_observation_noise_outputs) {
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::pointing);
    mapmaking.begin_iteration();
    mapmaking.begin_observation(0, "152389", 3, 4.848136811e-6, 3);
    citlali::pipeline::complete_mapmaking_observation(mapmaking);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);
    citlali::config::NoiseConfig request;
    request.enabled = true;
    request.n_noise_maps = 2;
    request.products_enabled = true;
    request.write_realizations = true;
    citlali::pipeline::NoiseExecutionPlan noise;
    noise.reset_from_request(request, true);

    citlali::pipeline::record_noise_run_completed(
        noise, mapmaking, true);

    EXPECT_EQ(*noise.realized.observation_noise_realization_count, 6U);
    EXPECT_EQ(*noise.realized.total_noise_realization_count, 6U);
    EXPECT_EQ(*noise.realized.empirical_product_map_count, 6U);
    EXPECT_EQ(*noise.realized.realization_image_write_count, 12U);
}

TEST(config_scaffold, records_non_jinc_coadd_noise_cardinality) {
    citlali::config::MapmakingConfig mapmaking_request;
    mapmaking_request.method = citlali::config::MapMethod::naive;
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        mapmaking_request, citlali::config::ReductionType::science);
    mapmaking.begin_iteration();
    mapmaking.begin_observation(0, "152390", 3, 4.848136811e-6, 3);
    citlali::pipeline::complete_mapmaking_observation(mapmaking);
    mapmaking.begin_coadd(3, 3);
    citlali::pipeline::complete_mapmaking_coadd(mapmaking);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);
    citlali::config::NoiseConfig request;
    request.enabled = true;
    request.n_noise_maps = 2;
    citlali::pipeline::NoiseExecutionPlan noise;
    noise.reset_from_request(request, true);

    citlali::pipeline::record_noise_run_completed(
        noise, mapmaking, false);

    EXPECT_EQ(*noise.realized.observation_scientific_map_count, 0U);
    EXPECT_EQ(*noise.realized.observation_noise_realization_count, 0U);
    EXPECT_EQ(*noise.realized.coadd_noise_realization_count, 6U);
    EXPECT_EQ(*noise.realized.total_noise_realization_count, 6U);
    EXPECT_EQ(*noise.realized.empirical_product_map_count, 3U);
}

TEST(config_scaffold, rejects_noise_completion_before_mapmaking) {
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::science);
    citlali::config::NoiseConfig request;
    request.enabled = true;
    citlali::pipeline::NoiseExecutionPlan noise;
    noise.reset_from_request(request, true);

    EXPECT_THROW(
        citlali::pipeline::record_noise_run_completed(
            noise, mapmaking, false),
        std::logic_error);
}

TEST(config_scaffold,
     adapts_typed_mapmaking_output_to_legacy_wcs_one_way) {
    struct FakeWcs {
        std::vector<float> cdelt;
        std::vector<int> naxis;
        std::vector<float> crpix;
        std::vector<float> crval;
        std::vector<std::string> cunit;
        std::vector<std::string> ctype;
    };
    struct FakeOutputMapBlock {
        double cov_cut = -1.0;
        int hist_n_bins = -1;
        double pixel_size_rad = -1.0;
        std::string sig_unit = "stale";
        FakeWcs wcs;
        std::vector<float> crval_config;
    } legacy;
    citlali::config::MapmakingConfig request;
    request.coverage_cut = 0.35;
    request.pixel_size_arcsec = 2.0;
    request.unit = "MJy/sr";
    request.x_size_pix = 7;
    request.y_size_pix = 9;
    request.crpix1 = 3.5;
    request.crpix2 = 4.5;
    request.crval1_j2000 = 11.0;
    request.crval2_j2000 = 12.0;
    citlali::config::PostProcessingConfig post_processing;
    post_processing.map_histogram_n_bins = 31;

    citlali::pipeline::adapt_mapmaking_output_config_one_way(
        request, post_processing, citlali::config::MapPixelAxes::radec,
        citlali::config::ReductionType::science, 0.25, 4.0, 8.0,
        legacy);

    EXPECT_DOUBLE_EQ(legacy.cov_cut, 0.35);
    EXPECT_EQ(legacy.hist_n_bins, 31);
    EXPECT_DOUBLE_EQ(legacy.pixel_size_rad, 0.5);
    EXPECT_EQ(legacy.sig_unit, "MJy/sr");
    EXPECT_EQ(legacy.wcs.cdelt,
              (std::vector<float>{-2.0F, 2.0F, 1.0F, 1.0F}));
    EXPECT_EQ(legacy.wcs.naxis, (std::vector<int>{7, 9, 1, 1}));
    EXPECT_EQ(legacy.wcs.crpix,
              (std::vector<float>{3.5F, 4.5F, 0.0F, 0.0F}));
    EXPECT_EQ(legacy.wcs.crval,
              (std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F}));
    EXPECT_EQ(legacy.wcs.ctype,
              (std::vector<std::string>{
                  "RA---TAN", "DEC--TAN", "FREQ", "STOKES"}));
    EXPECT_EQ(legacy.wcs.cunit,
              (std::vector<std::string>{"deg", "deg", "Hz", ""}));
    EXPECT_EQ(legacy.crval_config,
              (std::vector<float>{11.0F, 12.0F}));

    citlali::pipeline::adapt_mapmaking_output_config_one_way(
        request, post_processing, citlali::config::MapPixelAxes::altaz,
        citlali::config::ReductionType::beammap, 0.25, 4.0, 8.0,
        legacy);

    EXPECT_EQ(legacy.wcs.cdelt,
              (std::vector<float>{-4.0F, 4.0F, 1.0F, 1.0F}));
    EXPECT_EQ(legacy.wcs.ctype,
              (std::vector<std::string>{
                  "AZOFFSET", "ELOFFSET", "FREQ", "STOKES"}));
    EXPECT_EQ(legacy.wcs.cunit,
              (std::vector<std::string>{
                  "arcsec", "arcsec", "Hz", ""}));
    EXPECT_EQ(legacy.crval_config,
              (std::vector<float>{11.0F, 12.0F}));
}

TEST(config_scaffold,
     mapmaking_execution_plan_preserves_request_and_resolves_grouping) {
    citlali::config::MapmakingConfig request;
    request.grouping = citlali::config::MapGrouping::automatic;
    citlali::pipeline::MapmakingExecutionPlan plan;

    plan.reset_from_request(
        request, citlali::config::ReductionType::science);
    EXPECT_EQ(plan.requested.grouping,
              citlali::config::MapGrouping::automatic);
    EXPECT_EQ(plan.effective.grouping,
              citlali::config::MapGrouping::array);
    EXPECT_TRUE(plan.effective_resolution.automatic_grouping_resolved);
    EXPECT_FALSE(
        plan.effective_resolution.detector_grouping_fell_back_to_array);

    plan.reset_from_request(
        request, citlali::config::ReductionType::beammap);
    EXPECT_EQ(plan.requested.grouping,
              citlali::config::MapGrouping::automatic);
    EXPECT_EQ(plan.effective.grouping,
              citlali::config::MapGrouping::detector);

    request.grouping = citlali::config::MapGrouping::detector;
    plan.reset_from_request(
        request, citlali::config::ReductionType::pointing);
    EXPECT_EQ(plan.requested.grouping,
              citlali::config::MapGrouping::detector);
    EXPECT_EQ(plan.effective.grouping,
              citlali::config::MapGrouping::array);
    EXPECT_TRUE(
        plan.effective_resolution.detector_grouping_fell_back_to_array);
}

TEST(config_scaffold, routes_mapmaking_accessor_through_effective_plan) {
    FakeEngine engine;
    engine.typed_config.mapmaking.grouping =
        citlali::config::MapGrouping::automatic;
    engine.mapmaking_plan.reset_from_request(
        engine.typed_config.mapmaking,
        citlali::config::ReductionType::beammap);

    EXPECT_EQ(engine.typed_config.mapmaking.grouping,
              citlali::config::MapGrouping::automatic);
    EXPECT_EQ(engine.mapmaking_plan.requested.grouping,
              citlali::config::MapGrouping::automatic);
    EXPECT_EQ(citlali::pipeline::mapmaking_config(engine).grouping,
              citlali::config::MapGrouping::detector);
}

TEST(config_scaffold, records_uncalibrated_effective_map_unit) {
    citlali::config::MapmakingConfig request;
    request.unit = "mJy/beam";
    citlali::pipeline::MapmakingExecutionPlan plan;

    plan.reset_from_request(
        request, citlali::config::ReductionType::science, false,
        citlali::config::TodType::rs);

    EXPECT_EQ(plan.requested.unit, "mJy/beam");
    EXPECT_EQ(plan.effective.unit, "rs");
    EXPECT_EQ(plan.effective_resolution.requested_unit, "mJy/beam");
    EXPECT_EQ(plan.effective_resolution.effective_unit, "rs");
    EXPECT_TRUE(
        plan.effective_resolution.uncalibrated_unit_substituted);
}

TEST(config_scaffold, serializes_versioned_mapmaking_provenance) {
    citlali::config::MapmakingConfig request;
    request.grouping = citlali::config::MapGrouping::automatic;
    request.method = citlali::config::MapMethod::jinc;
    request.pixel_size_arcsec = 1.5;
    citlali::pipeline::MapmakingExecutionPlan plan;
    plan.reset_from_request(
        request, citlali::config::ReductionType::beammap);
    plan.begin_iteration();
    plan.begin_observation(0, "148670", 5234, 4.848136811e-6,
                           10468);
    citlali::pipeline::complete_mapmaking_observation(plan);
    plan.begin_coadd(5234, 10468);
    citlali::pipeline::complete_mapmaking_coadd(plan);
    citlali::pipeline::record_mapmaking_run_completed(plan);

    const auto node =
        citlali::pipeline::mapmaking_provenance_node(plan);

    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-mapmaking-provenance-v2");
    EXPECT_TRUE(node["initialized"].as<bool>());
    EXPECT_EQ(node["requested"]["grouping"].as<std::string>(),
              "auto");
    EXPECT_EQ(node["requested"]["method"].as<std::string>(), "jinc");
    EXPECT_DOUBLE_EQ(
        node["requested"]["pixel_size_arcsec"].as<double>(), 1.5);
    EXPECT_EQ(node["effective"]["config"]["grouping"]
                  .as<std::string>(),
              "detector");
    EXPECT_EQ(node["effective"]["resolution"]["reduction_type"]
                  .as<std::string>(),
              "beammap");
    EXPECT_EQ(node["effective"]["resolution"]["effective_unit"]
                  .as<std::string>(),
              "mJy/beam");
    ASSERT_EQ(node["observations"].size(), 1U);
    EXPECT_EQ(node["observations"][0]["observation_index"]
                  .as<std::size_t>(),
              0U);
    EXPECT_EQ(node["observations"][0]["obsnum"].as<int>(), 148670);
    EXPECT_EQ(node["observations"][0]["map_count"].as<std::size_t>(),
              5234U);
    EXPECT_EQ(node["observations"][0]["required_map_write_count"]
                  .as<std::size_t>(),
              10468U);
    EXPECT_TRUE(
        node["observations"][0]["outputs_completed"].as<bool>());
    EXPECT_TRUE(node["coadd"]["available"].as<bool>());
    EXPECT_EQ(node["coadd"]["map_count"].as<std::size_t>(), 5234U);
    EXPECT_TRUE(node["realized"]["reduction_completed"].as<bool>());
    EXPECT_TRUE(node["realized"]["mapmaking_executed"].as<bool>());
    EXPECT_EQ(node["realized"]["completed_observation_count"]["value"]
                  .as<std::size_t>(),
              1U);
    EXPECT_EQ(node["realized"]["completed_coadd_count"]["value"]
                  .as<std::size_t>(),
              1U);
}

TEST(config_scaffold, resets_mapmaking_cardinality_per_iteration) {
    citlali::pipeline::MapmakingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::science);
    plan.begin_iteration();
    plan.begin_observation(0, "152389", 3, 9.696273622e-6, 3);
    citlali::pipeline::complete_mapmaking_observation(plan);
    plan.begin_coadd(3, 6);
    citlali::pipeline::complete_mapmaking_coadd(plan);

    plan.begin_iteration();

    EXPECT_TRUE(plan.observations.empty());
    EXPECT_FALSE(plan.coadd.has_value());
    ASSERT_TRUE(plan.realized.completed_observation_count.has_value());
    ASSERT_TRUE(plan.realized.completed_coadd_count.has_value());
    EXPECT_EQ(*plan.realized.completed_observation_count, 0U);
    EXPECT_EQ(*plan.realized.completed_coadd_count, 0U);
}

TEST(config_scaffold, serializes_zero_padded_obsnum_as_numeric_identity) {
    citlali::pipeline::MapmakingObservationState observation{
        0, "000042", 3, 4.848136811e-6, 3, true};

    const auto node =
        citlali::pipeline::mapmaking_observation_state_node(observation);

    EXPECT_EQ(node["obsnum"].as<int>(), 42);
}

TEST(config_scaffold, rejects_incomplete_mapmaking_cardinality) {
    citlali::pipeline::MapmakingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::science);

    EXPECT_THROW(
        citlali::pipeline::record_mapmaking_run_completed(plan),
        std::logic_error);

    plan.begin_iteration();
    plan.begin_observation(0, "152389", 3, 9.696273622e-6, 3);
    EXPECT_THROW(
        citlali::pipeline::record_mapmaking_run_completed(plan),
        std::logic_error);
}

TEST(config_scaffold, records_zero_cardinality_when_mapmaking_disabled) {
    citlali::config::MapmakingConfig request;
    request.enabled = false;
    citlali::pipeline::MapmakingExecutionPlan plan;
    plan.reset_from_request(
        request, citlali::config::ReductionType::science);
    plan.begin_iteration();

    citlali::pipeline::record_mapmaking_run_completed(plan);

    EXPECT_TRUE(plan.realized.reduction_completed);
    EXPECT_FALSE(plan.realized.mapmaking_executed);
    EXPECT_EQ(*plan.realized.completed_observation_count, 0U);
    EXPECT_EQ(*plan.realized.completed_coadd_count, 0U);
}

TEST(config_scaffold, calculates_required_mapmaking_write_count) {
    EXPECT_EQ(citlali::pipeline::required_mapmaking_write_count(3, 1),
              3U);
    EXPECT_EQ(citlali::pipeline::required_mapmaking_write_count(3, 2),
              6U);
    EXPECT_THROW(
        citlali::pipeline::required_mapmaking_write_count(0, 1),
        std::logic_error);
    EXPECT_THROW(
        citlali::pipeline::required_mapmaking_write_count(
            std::numeric_limits<std::size_t>::max(), 2),
        std::overflow_error);
}

TEST(config_scaffold, atomically_writes_mapmaking_provenance) {
    const auto output_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_mapmaking_provenance_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    citlali::pipeline::MapmakingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::science);
    plan.begin_iteration();
    plan.begin_observation(0, "152389", 3, 9.696273622e-6, 3);
    citlali::pipeline::complete_mapmaking_observation(plan);
    citlali::pipeline::record_mapmaking_run_completed(plan);

    citlali::pipeline::write_mapmaking_provenance_file(
        output_dir, plan);

    const auto output_path =
        citlali::pipeline::mapmaking_provenance_path(output_dir);
    EXPECT_TRUE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    const auto stored = YAML::LoadFile(output_path.string());
    EXPECT_EQ(stored["schema_version"].as<std::string>(),
              "citlali-mapmaking-provenance-v2");
    std::filesystem::remove_all(output_dir);
}

TEST(config_scaffold, mapmaking_provenance_failure_propagates) {
    const auto missing_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_missing_mapmaking_provenance_dir" / "nested";
    std::filesystem::remove_all(missing_dir.parent_path());
    citlali::pipeline::MapmakingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::science);
    plan.begin_iteration();
    plan.begin_observation(0, "152389", 3, 9.696273622e-6, 3);
    citlali::pipeline::complete_mapmaking_observation(plan);
    citlali::pipeline::record_mapmaking_run_completed(plan);

    EXPECT_THROW(
        citlali::pipeline::write_mapmaking_provenance_file(
            missing_dir, plan),
        std::ios_base::failure);
    EXPECT_FALSE(std::filesystem::exists(
        citlali::pipeline::mapmaking_provenance_path(missing_dir)));
    EXPECT_THROW(
        citlali::pipeline::write_mapmaking_provenance_file(
            missing_dir, citlali::pipeline::MapmakingExecutionPlan{}),
        std::logic_error);
}

TEST(config_scaffold, serializes_versioned_coadd_provenance) {
    citlali::config::CoaddConfig request;
    request.enabled = true;
    citlali::pipeline::CoaddExecutionPlan plan;
    plan.reset_from_request(request, true);
    plan.realized = citlali::pipeline::CoaddRealizedState{
        true, true, 3U, 6U, true};

    const auto node = citlali::pipeline::coadd_provenance_node(plan);

    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-coadd-provenance-v1");
    EXPECT_TRUE(node["requested"]["enabled"].as<bool>());
    EXPECT_TRUE(node["effective"]["config"]["enabled"].as<bool>());
    EXPECT_FALSE(node["effective"]["resolution"]
                        ["disabled_by_mapmaking"]
                            .as<bool>());
    EXPECT_TRUE(node["realized"]["coadd_executed"].as<bool>());
    EXPECT_EQ(node["realized"]["map_count"]["value"]
                  .as<std::size_t>(),
              3U);
    EXPECT_EQ(node["realized"]["required_map_write_count"]["value"]
                  .as<std::size_t>(),
              6U);
    EXPECT_TRUE(node["realized"]["outputs_completed"].as<bool>());
}

TEST(config_scaffold, atomically_writes_coadd_provenance) {
    const auto output_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_coadd_provenance_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    citlali::pipeline::CoaddExecutionPlan plan;
    plan.reset_from_request(citlali::config::CoaddConfig{}, true);
    plan.realized.reduction_completed = true;

    citlali::pipeline::write_coadd_provenance_file(output_dir, plan);

    const auto output_path =
        citlali::pipeline::coadd_provenance_path(output_dir);
    EXPECT_TRUE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    const auto stored = YAML::LoadFile(output_path.string());
    EXPECT_EQ(stored["schema_version"].as<std::string>(),
              "citlali-coadd-provenance-v1");
    std::filesystem::remove_all(output_dir);
}

TEST(config_scaffold, coadd_provenance_failure_propagates) {
    const auto missing_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_missing_coadd_provenance_dir" / "nested";
    std::filesystem::remove_all(missing_dir.parent_path());
    citlali::pipeline::CoaddExecutionPlan plan;
    plan.reset_from_request(citlali::config::CoaddConfig{}, true);
    plan.realized.reduction_completed = true;

    EXPECT_THROW(
        citlali::pipeline::write_coadd_provenance_file(
            missing_dir, plan),
        std::ios_base::failure);
    EXPECT_FALSE(std::filesystem::exists(
        citlali::pipeline::coadd_provenance_path(missing_dir)));
    EXPECT_THROW(
        citlali::pipeline::write_coadd_provenance_file(
            missing_dir, citlali::pipeline::CoaddExecutionPlan{}),
        std::logic_error);
}

TEST(config_scaffold, serializes_versioned_noise_provenance) {
    citlali::config::NoiseConfig request;
    request.enabled = true;
    request.n_noise_maps = 2;
    request.products_enabled = true;
    request.write_realizations = true;
    citlali::pipeline::NoiseExecutionPlan plan;
    plan.reset_from_request(request, true);
    plan.realized.reduction_completed = true;
    plan.realized.generation_executed = true;
    plan.realized.noise_maps_per_scientific_map = 2U;
    plan.realized.observation_scientific_map_count = 3U;
    plan.realized.observation_noise_realization_count = 6U;
    plan.realized.coadd_scientific_map_count = 0U;
    plan.realized.coadd_noise_realization_count = 0U;
    plan.realized.total_noise_realization_count = 6U;
    plan.realized.empirical_product_map_count = 6U;
    plan.realized.realization_image_write_count = 12U;
    plan.realized.outputs_completed = true;

    const auto node = citlali::pipeline::noise_provenance_node(plan);

    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-noise-products-provenance-v1");
    EXPECT_EQ(node["requested"]["n_noise_maps"].as<int>(), 2);
    EXPECT_EQ(node["effective"]["resolution"]["randomization"]["seed"]
                  .as<std::uint32_t>(),
              citlali::pipeline::noise_random_seed);
    EXPECT_EQ(node["realized"]["total_noise_realization_count"]["value"]
                  .as<std::size_t>(),
              6U);
    EXPECT_EQ(node["realized"]["realization_image_write_count"]["value"]
                  .as<std::size_t>(),
              12U);
}

TEST(config_scaffold, atomically_writes_noise_provenance) {
    const auto output_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_noise_provenance_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    citlali::pipeline::NoiseExecutionPlan plan;
    plan.reset_from_request(citlali::config::NoiseConfig{}, true);
    plan.realized.reduction_completed = true;

    citlali::pipeline::write_noise_provenance_file(output_dir, plan);

    const auto output_path =
        citlali::pipeline::noise_provenance_path(output_dir);
    EXPECT_TRUE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    const auto stored = YAML::LoadFile(output_path.string());
    EXPECT_EQ(stored["schema_version"].as<std::string>(),
              "citlali-noise-products-provenance-v1");
    std::filesystem::remove_all(output_dir);
}

TEST(config_scaffold, noise_provenance_failure_propagates) {
    const auto missing_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_missing_noise_provenance_dir" / "nested";
    std::filesystem::remove_all(missing_dir.parent_path());
    citlali::pipeline::NoiseExecutionPlan plan;
    plan.reset_from_request(citlali::config::NoiseConfig{}, true);
    plan.realized.reduction_completed = true;

    EXPECT_THROW(
        citlali::pipeline::write_noise_provenance_file(
            missing_dir, plan),
        std::ios_base::failure);
    EXPECT_FALSE(std::filesystem::exists(
        citlali::pipeline::noise_provenance_path(missing_dir)));
    EXPECT_THROW(
        citlali::pipeline::write_noise_provenance_file(
            missing_dir, citlali::pipeline::NoiseExecutionPlan{}),
        std::logic_error);
}

TEST(config_scaffold, resolves_pointing_request_without_mutating_it) {
    citlali::config::PointingConfig request;
    request.header_max_radius_arcsec = 0.0;
    citlali::pipeline::PointingRequestPresence presence;
    citlali::pipeline::PointingExecutionPlan plan;

    plan.reset_from_request(request, presence, true, true, false, 30.0);

    EXPECT_DOUBLE_EQ(plan.requested.header_max_radius_arcsec, 0.0);
    EXPECT_DOUBLE_EQ(plan.effective.header_max_radius_arcsec, 30.0);
    EXPECT_TRUE(plan.effective.fit_gaussian);
    EXPECT_TRUE(
        plan.effective_resolution.header_max_radius_defaulted);
    EXPECT_FALSE(plan.effective_resolution.fit_disabled_by_mapmaking);

    request.source_strategy =
        citlali::config::PointingSourceStrategy::psf_preserve;
    request.fit_gaussian = false;
    request.fruitloops_center_mode =
        citlali::config::FruitLoopsCenterMode::map_center;
    plan.reset_from_request(request, presence, true, true, false, 30.0);
    EXPECT_DOUBLE_EQ(plan.effective.header_max_radius_arcsec, 0.0);
}

TEST(config_scaffold, reads_complete_typed_pointing_request) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    root["pointing"]["source_strategy"]["mode"] = "psf_preserve";
    root["pointing"]["source_strategy"]["fit_gaussian"] = false;
    root["pointing"]["source_strategy"]["fruitloops_center_mode"] =
        "map_center";
    root["pointing"]["source_strategy"]
        ["header_max_radius_arcsec"] = 17.5;
    root["pointing"]["source_strategy"]
        ["header_require_coverage"] = false;
    auto yaml_config =
        tula::config::YamlConfig::from_str(YAML::Dump(root));
    citlali::config::PointingConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    const auto presence =
        citlali::pipeline::read_pointing_request_config(
            yaml_config, request, diagnostics);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_TRUE(presence.source_strategy);
    EXPECT_TRUE(presence.fit_gaussian);
    EXPECT_TRUE(presence.fruitloops_center_mode);
    EXPECT_TRUE(presence.header_max_radius_arcsec);
    EXPECT_TRUE(presence.header_require_coverage);
    EXPECT_EQ(
        request.source_strategy,
        citlali::config::PointingSourceStrategy::psf_preserve);
    EXPECT_FALSE(request.fit_gaussian);
    EXPECT_EQ(
        request.fruitloops_center_mode,
        citlali::config::FruitLoopsCenterMode::map_center);
    EXPECT_DOUBLE_EQ(request.header_max_radius_arcsec, 17.5);
    EXPECT_FALSE(request.header_require_coverage);
}

TEST(config_scaffold, records_pointing_observation_lifecycle) {
    citlali::pipeline::PointingExecutionPlan pointing;
    pointing.reset_from_request(
        citlali::config::PointingConfig{}, {}, true, true, false, 30.0);
    pointing.begin_iteration();
    pointing.begin_observation(0, "152389", 3);
    citlali::pipeline::record_pointing_fit_results(
        pointing, citlali::pipeline::PointingFitStage::raw_observation,
        3, 2);
    citlali::pipeline::record_pointing_fit_results(
        pointing,
        citlali::pipeline::PointingFitStage::filtered_observation,
        3, 1);
    citlali::pipeline::complete_pointing_observation(pointing);

    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::pointing);
    mapmaking.begin_iteration();
    mapmaking.begin_observation(0, "152389", 3, 1.0e-5, 3);
    citlali::pipeline::complete_mapmaking_observation(mapmaking);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);

    citlali::pipeline::record_pointing_run_completed(
        pointing, mapmaking);

    EXPECT_TRUE(pointing.realized.reduction_completed);
    EXPECT_TRUE(pointing.realized.pointing_executed);
    EXPECT_EQ(pointing.realized.completed_observation_count, 1U);
    EXPECT_EQ(pointing.realized.scientific_map_count, 3U);
    EXPECT_EQ(pointing.realized.raw_fit_attempt_count, 3U);
    EXPECT_EQ(pointing.realized.raw_valid_fit_count, 2U);
    EXPECT_EQ(pointing.realized.filtered_fit_attempt_count, 3U);
    EXPECT_EQ(pointing.realized.filtered_valid_fit_count, 1U);
    EXPECT_TRUE(pointing.realized.outputs_completed);
}

TEST(config_scaffold, keeps_pointing_fit_independent_of_filtered_outputs) {
    citlali::pipeline::PointingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::PointingConfig{}, {}, true, false, false, 30.0);
    plan.begin_iteration();
    plan.begin_observation(0, "152389", 3);

    citlali::pipeline::record_pointing_fit_results(
        plan, citlali::pipeline::PointingFitStage::raw_observation,
        3, 2);

    citlali::pipeline::complete_pointing_observation(plan);

    EXPECT_TRUE(plan.effective.fit_gaussian);
    EXPECT_TRUE(plan.effective_resolution.fit_output_path_available);
    EXPECT_FALSE(
        plan.effective_resolution.fit_disabled_by_output_policy);
    EXPECT_TRUE(plan.observations.front().raw_fit.recorded);
    EXPECT_EQ(plan.observations.front().raw_fit.attempt_count, 3U);
    EXPECT_FALSE(plan.observations.front().filtered_fit.recorded);
}

TEST(config_scaffold, rejects_duplicate_pointing_fit_stage_results) {
    citlali::pipeline::PointingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::PointingConfig{}, {}, true, true, false, 30.0);
    plan.begin_iteration();
    plan.begin_observation(0, "152389", 3);

    citlali::pipeline::record_pointing_fit_results(
        plan, citlali::pipeline::PointingFitStage::raw_observation,
        3, 2);

    EXPECT_THROW(
        citlali::pipeline::record_pointing_fit_results(
            plan,
            citlali::pipeline::PointingFitStage::raw_observation,
            3, 2),
        std::logic_error);
    EXPECT_NO_THROW(
        citlali::pipeline::record_pointing_fit_results(
            plan,
            citlali::pipeline::PointingFitStage::filtered_observation,
            3, 2));
}

TEST(config_scaffold, requires_filtered_pointing_fit_when_filtering) {
    citlali::pipeline::PointingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::PointingConfig{}, {}, true, true, false, 30.0);
    plan.begin_iteration();
    plan.begin_observation(0, "152389", 3);
    citlali::pipeline::record_pointing_fit_results(
        plan, citlali::pipeline::PointingFitStage::raw_observation,
        3, 2);

    EXPECT_THROW(
        citlali::pipeline::complete_pointing_observation(plan),
        std::logic_error);
}

TEST(config_scaffold, disables_pointing_fit_without_mapmaking) {
    citlali::pipeline::PointingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::PointingConfig{}, {}, false, true, false, 30.0);

    EXPECT_FALSE(plan.effective.fit_gaussian);
    EXPECT_FALSE(plan.effective_resolution.fit_output_path_available);
    EXPECT_TRUE(plan.effective_resolution.fit_disabled_by_mapmaking);
    EXPECT_FALSE(plan.effective_resolution.fit_disabled_by_output_policy);
}

TEST(config_scaffold, pointing_adapter_is_one_way) {
    struct Processor {
        std::string fruit_loops_source_center_mode;
        double fruit_loops_header_center_max_radius_arcsec = 0.0;
        bool fruit_loops_header_center_require_coverage = false;
    } processor;
    citlali::config::PointingConfig effective;
    effective.fruitloops_center_mode =
        citlali::config::FruitLoopsCenterMode::header;
    effective.header_max_radius_arcsec = 25.0;
    effective.header_require_coverage = true;

    citlali::pipeline::adapt_pointing_config_one_way(
        effective, processor);

    EXPECT_EQ(processor.fruit_loops_source_center_mode, "header");
    EXPECT_DOUBLE_EQ(
        processor.fruit_loops_header_center_max_radius_arcsec, 25.0);
    EXPECT_TRUE(
        processor.fruit_loops_header_center_require_coverage);
}

TEST(config_scaffold, serializes_versioned_pointing_provenance) {
    citlali::pipeline::PointingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::PointingConfig{}, {}, true, true, false, 30.0);
    plan.begin_iteration();
    plan.begin_observation(0, "152389", 3);
    citlali::pipeline::record_pointing_fit_results(
        plan, citlali::pipeline::PointingFitStage::raw_observation,
        3, 2);
    citlali::pipeline::record_pointing_fit_results(
        plan,
        citlali::pipeline::PointingFitStage::filtered_observation,
        3, 1);
    citlali::pipeline::complete_pointing_observation(plan);
    plan.realized.reduction_completed = true;
    plan.realized.pointing_executed = true;
    plan.realized.completed_observation_count = 1U;
    plan.realized.scientific_map_count = 3U;
    plan.realized.raw_fit_attempt_count = 3U;
    plan.realized.raw_valid_fit_count = 2U;
    plan.realized.filtered_fit_attempt_count = 3U;
    plan.realized.filtered_valid_fit_count = 1U;
    plan.realized.outputs_completed = true;

    const auto node =
        citlali::pipeline::pointing_provenance_node(plan);

    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-pointing-provenance-v2");
    EXPECT_DOUBLE_EQ(
        node["requested"]["header_max_radius_arcsec"].as<double>(),
        0.0);
    EXPECT_DOUBLE_EQ(
        node["effective"]["config"]
            ["header_max_radius_arcsec"].as<double>(),
        30.0);
    EXPECT_TRUE(
        node["effective"]["resolution"]
            ["header_max_radius_defaulted"].as<bool>());
    EXPECT_EQ(node["observations"][0]["raw_valid_fit_count"]
                  .as<std::size_t>(),
              2U);
    EXPECT_EQ(node["observations"][0]["filtered_valid_fit_count"]
                  .as<std::size_t>(),
              1U);
    EXPECT_TRUE(node["realized"]["outputs_completed"].as<bool>());
}

TEST(config_scaffold, pointing_provenance_write_contract) {
    const auto output_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_pointing_provenance_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    citlali::pipeline::PointingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::PointingConfig{}, {}, false, false, false, 0.0);
    plan.realized.reduction_completed = true;

    citlali::pipeline::write_pointing_provenance_file(
        output_dir, plan);

    const auto output_path =
        citlali::pipeline::pointing_provenance_path(output_dir);
    EXPECT_TRUE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    EXPECT_EQ(
        YAML::LoadFile(output_path.string())["schema_version"]
            .as<std::string>(),
        "citlali-pointing-provenance-v2");
    std::filesystem::remove_all(output_dir);

    const auto missing_dir = output_dir / "missing";
    EXPECT_THROW(
        citlali::pipeline::write_pointing_provenance_file(
            missing_dir, plan),
        std::ios_base::failure);
    EXPECT_THROW(
        citlali::pipeline::write_pointing_provenance_file(
            output_dir,
            citlali::pipeline::PointingExecutionPlan{}),
        std::logic_error);
}

TEST(config_scaffold, validates_typed_mapmaking_method_values) {
    citlali::config::MapmakingConfig config;
    config.jinc_filter.r_max = 0.0;
    config.jinc_filter.subpixel_n = 0;
    config.jinc_filter.shape_params["a1100"][1] =
        std::numeric_limits<double>::quiet_NaN();
    config.maximum_likelihood.max_iterations = 0;
    config.maximum_likelihood.tolerance = 0.0;
    citlali::config::ValidationReport report;

    citlali::config::validate(config, report);

    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 5U);
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

TEST(config_scaffold, reads_disabled_polarimetry_request) {
    ensure_citlali_test_logger();
    auto config = tula::config::YamlConfig::from_str(
        citlali::citlali_default_config_content);
    citlali::config::TimestreamPolarimetryConfig request;
    request.enabled = true;
    request.grouping =
        citlali::config::PolarimetryGrouping::detector_location;
    request.hwpr_policy =
        citlali::config::PolarimetryHwprPolicy::require;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_polarimetry_request_config(
        config, request, diagnostics);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_FALSE(request.enabled);
    EXPECT_EQ(request.grouping,
              citlali::config::PolarimetryGrouping::frequency_group);
    EXPECT_EQ(request.hwpr_policy,
              citlali::config::PolarimetryHwprPolicy::automatic);
}

TEST(config_scaffold, reads_enabled_polarimetry_request) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    root["timestream"]["polarimetry"]["enabled"] = true;
    root["timestream"]["polarimetry"]["grouping"] = "loc";
    root["timestream"]["polarimetry"]["ignore_hwpr"] = "true";
    auto config = tula::config::YamlConfig::from_str(YAML::Dump(root));
    citlali::config::TimestreamPolarimetryConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_polarimetry_request_config(
        config, request, diagnostics);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_TRUE(request.enabled);
    EXPECT_EQ(request.grouping,
              citlali::config::PolarimetryGrouping::detector_location);
    EXPECT_EQ(request.hwpr_policy,
              citlali::config::PolarimetryHwprPolicy::ignore);
}

TEST(config_scaffold, rejects_invalid_polarimetry_enum) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    root["timestream"]["polarimetry"]["grouping"] = "network";
    auto config = tula::config::YamlConfig::from_str(YAML::Dump(root));
    citlali::config::TimestreamPolarimetryConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_polarimetry_request_config(
        config, request, diagnostics);

    ASSERT_TRUE(diagnostics.has_errors());
    EXPECT_EQ(diagnostics.invalid_key_paths(),
              (citlali::pipeline::ConfigDiagnosticsState::key_vec_t{
                  {"timestream", "polarimetry", "grouping"}}));
}

TEST(config_scaffold, adapts_typed_polarimetry_config_one_way) {
    struct FakeRtcProc {
        struct FakePolarization {
            std::string grouping = "loc";
            std::map<int, std::string> stokes_params;
        } polarization;
        bool run_polarization = true;
    } rtcproc;
    struct FakeCalib {
        std::string ignore_hwpr = "false";
        int untouched = 41;
    } calib;
    citlali::config::TimestreamPolarimetryConfig config;

    citlali::pipeline::adapt_polarimetry_config(config, rtcproc, calib);

    EXPECT_FALSE(rtcproc.run_polarization);
    EXPECT_EQ(rtcproc.polarization.grouping, "fg");
    EXPECT_EQ(rtcproc.polarization.stokes_params,
              (std::map<int, std::string>{{0, "I"}}));
    EXPECT_EQ(calib.ignore_hwpr, "auto");
    EXPECT_EQ(calib.untouched, 41);
}

TEST(config_scaffold, rejects_enabled_polarimetry_capability) {
    citlali::config::TimestreamPolarimetryConfig request;
    request.enabled = true;
    citlali::pipeline::PolarimetryExecutionPlan plan;

    plan.reset_from_request(request);

    ASSERT_TRUE(plan.initialized);
    EXPECT_TRUE(plan.requested.enabled);
    EXPECT_FALSE(plan.effective.enabled);
    EXPECT_FALSE(plan.capability.enabled_capability_available);
    EXPECT_FALSE(plan.capability.request_accepted);
    EXPECT_TRUE(plan.capability.disabled_by_capability);
}

TEST(config_scaffold, records_disabled_polarimetry_provenance) {
    citlali::config::TimestreamPolarimetryConfig request;
    citlali::pipeline::PolarimetryExecutionPlan plan;
    plan.reset_from_request(request);

    citlali::pipeline::record_polarimetry_run_completed(plan);
    const auto node =
        citlali::pipeline::polarimetry_provenance_node(plan);

    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-polarimetry-provenance-v1");
    EXPECT_EQ(node["capability"]["status"].as<std::string>(),
              "planned-unavailable");
    EXPECT_FALSE(
        node["capability"]["enabled_supported"].as<bool>());
    EXPECT_TRUE(node["effective"]["capability_resolution"]
                    ["request_accepted"]
                        .as<bool>());
    EXPECT_TRUE(node["realized"]["reduction_completed"].as<bool>());
    EXPECT_FALSE(node["realized"]["polarimetry_executed"].as<bool>());
}

TEST(config_scaffold, polarimetry_provenance_write_contract) {
    const auto output_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_polarimetry_provenance_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    citlali::pipeline::PolarimetryExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::TimestreamPolarimetryConfig{});
    citlali::pipeline::record_polarimetry_run_completed(plan);

    citlali::pipeline::write_polarimetry_provenance_file(
        output_dir, plan);

    const auto output_path =
        citlali::pipeline::polarimetry_provenance_path(output_dir);
    EXPECT_TRUE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    EXPECT_EQ(YAML::LoadFile(output_path.string())["capability"]["status"]
                  .as<std::string>(),
              "planned-unavailable");
    std::filesystem::remove_all(output_dir);

    const auto missing_dir = output_dir / "missing" / "nested";
    EXPECT_THROW(
        citlali::pipeline::write_polarimetry_provenance_file(
            missing_dir, plan),
        std::ios_base::failure);
    EXPECT_THROW(
        citlali::pipeline::write_polarimetry_provenance_file(
            output_dir,
            citlali::pipeline::PolarimetryExecutionPlan{}),
        std::logic_error);
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

TEST(config_scaffold, reads_beammap_photometry_without_source_identity) {
    auto config = tula::config::YamlConfig::from_str(R"yaml(
beammap_source:
  fluxes:
    - array_name: a1100
      value_mJy: 1000.0
      uncertainty_mJy: 10.0
    - array_name: a1400
      value_mJy: 900.0
      uncertainty_mJy: 9.0
)yaml");

    const auto observation =
        citlali::pipeline::read_beammap_photometry_config(config);

    ASSERT_EQ(observation.photometry.fluxes.size(), 2U);
    EXPECT_EQ(
        observation.photometry.fluxes.front().array_name, "a1100");
    EXPECT_DOUBLE_EQ(
        observation.photometry.fluxes.front().value_mjy, 1000.0);
    EXPECT_EQ(
        observation.fluxes_mjy_beam,
        (std::map<std::string, double>{{"a1100", 1000.0},
                                       {"a1400", 900.0}}));
}

TEST(config_scaffold, rejects_invalid_beammap_source_observation) {
    const std::map<int, std::string> array_names = {
        {0, "a1100"}, {1, "a1400"}, {2, "a2000"}};
    auto logger = std::make_shared<FakeLogger>();
    citlali::pipeline::BeammapPhotometryObservationConfig observation;
    observation.fluxes_mjy_beam = {
        {"a1100", 1.0}, {"a1400", 2.0}};

    try {
        citlali::pipeline::require_valid_beammap_source_fluxes(
            observation, array_names, logger);
        FAIL() << "expected invalid beammap source configuration";
    }
    catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::invalid_config);
        EXPECT_STREQ(
            error.what(), "invalid beammap_source flux configuration");
    }
}

TEST(config_scaffold, rejects_invalid_beammap_flux_uncertainty) {
    const std::map<int, std::string> array_names = {{0, "a1100"}};
    auto logger = std::make_shared<FakeLogger>();
    citlali::pipeline::BeammapPhotometryObservationConfig observation;
    observation.photometry.fluxes = {{"a1100", 1.0, -1.0}};
    observation.fluxes_mjy_beam = {{"a1100", 1.0}};

    EXPECT_THROW(
        citlali::pipeline::require_valid_beammap_source_fluxes(
            observation, array_names, logger),
        citlali::error::Error);
    EXPECT_EQ(logger->error_calls, 1);
}

TEST(config_scaffold, replaces_beammap_photometry_atomically) {
    citlali::config::BeammapPhotometryConfig photometry;
    photometry.fluxes = {{"a1100", 99.0, 0.0}};
    std::map<std::string, double> fluxes_mjy_beam = {
        {"a1100", 99.0}, {"stale-array", 88.0}};
    std::map<std::string, double> fluxes_mjy_sr = {
        {"a1100", 9.0}, {"stale-array", 8.0}};

    citlali::pipeline::BeammapPhotometryObservationConfig observation;
    observation.photometry.fluxes = {{"a1400", 2.0, 0.1}};
    observation.fluxes_mjy_beam = {{"a1400", 2.0}};

    citlali::pipeline::install_beammap_photometry_config(
        std::move(observation), photometry, fluxes_mjy_beam,
        fluxes_mjy_sr);

    ASSERT_EQ(photometry.fluxes.size(), 1U);
    EXPECT_EQ(photometry.fluxes.front().array_name, "a1400");
    EXPECT_EQ(
        fluxes_mjy_beam,
        (std::map<std::string, double>{{"a1400", 2.0}}));
    EXPECT_TRUE(fluxes_mjy_sr.empty());
}

TEST(config_scaffold, reads_and_adapts_astrometry_atomically) {
    auto config = tula::config::YamlConfig::from_str(R"yaml(
pointing_offsets:
  - axes_name: AZ
    value_arcsec: [1.5, 2.5]
  - axes_name: alt
    value_arcsec: [-3.5, -4.5]
  - modified_julian_date: [60000.0, 60001.0]
)yaml");
    auto logger = std::make_shared<FakeLogger>();

    auto observation =
        citlali::pipeline::read_astrometry_config(config, logger);
    citlali::pipeline::require_valid_astrometry_config(
        observation, logger);

    citlali::config::AstrometryConfig target;
    target.pointing_offsets.az_arcsec = {99.0};
    citlali::pipeline::PointingOffsetState state;
    state.arcsec["stale"] = Eigen::VectorXd::Constant(1, 99.0);
    citlali::pipeline::install_astrometry_config(
        std::move(observation), target, state);

    EXPECT_EQ(target.pointing_offsets.az_arcsec,
              (std::vector<double>{1.5, 2.5}));
    EXPECT_EQ(target.pointing_offsets.alt_arcsec,
              (std::vector<double>{-3.5, -4.5}));
    EXPECT_EQ(target.pointing_offsets.modified_julian_date,
              (std::vector<double>{60000.0, 60001.0}));
    EXPECT_EQ(state.arcsec.count("stale"), 0U);
    EXPECT_TRUE(state.arcsec.at("az").isApprox(
        (Eigen::Vector2d{} << 1.5, 2.5).finished()));
    EXPECT_TRUE(state.arcsec.at("alt").isApprox(
        (Eigen::Vector2d{} << -3.5, -4.5).finished()));
    EXPECT_TRUE(state.modified_julian_date.matrix().isApprox(
        (Eigen::Vector2d{} << 60000.0, 60001.0).finished()));
}

TEST(config_scaffold, preserves_positional_astrometry_and_mjd_sentinel) {
    auto config = tula::config::YamlConfig::from_str(R"yaml(
pointing_offsets:
  - value_arcsec: [1.0]
  - value_arcsec: [2.0]
  - modified_julian_date: [-1.0]
)yaml");
    auto logger = std::make_shared<FakeLogger>();

    const auto observation =
        citlali::pipeline::read_astrometry_config(config, logger);
    citlali::pipeline::require_valid_astrometry_config(
        observation, logger);

    EXPECT_EQ(observation.pointing_offsets.az_arcsec,
              (std::vector<double>{1.0}));
    EXPECT_EQ(observation.pointing_offsets.alt_arcsec,
              (std::vector<double>{2.0}));
    EXPECT_EQ(observation.pointing_offsets.modified_julian_date,
              (std::vector<double>{0.0, 0.0}));
    EXPECT_EQ(logger->warn_calls, 4);
}

TEST(config_scaffold, rejects_nonfinite_astrometry_before_install) {
    auto config = tula::config::YamlConfig::from_str(R"yaml(
pointing_offsets:
  - axes_name: az
    value_arcsec: [.nan]
  - axes_name: alt
    value_arcsec: [0.0]
)yaml");
    auto logger = std::make_shared<FakeLogger>();
    const auto observation =
        citlali::pipeline::read_astrometry_config(config, logger);

    EXPECT_THROW(
        citlali::pipeline::require_valid_astrometry_config(
            observation, logger),
        citlali::error::Error);
    EXPECT_EQ(logger->error_calls, 1);
}

TEST(config_scaffold, rejects_explicit_empty_astrometry_mjd) {
    auto config = tula::config::YamlConfig::from_str(R"yaml(
pointing_offsets:
  - axes_name: az
    value_arcsec: [0.0]
  - axes_name: alt
    value_arcsec: [0.0]
  - modified_julian_date: []
)yaml");
    auto logger = std::make_shared<FakeLogger>();
    const auto observation =
        citlali::pipeline::read_astrometry_config(config, logger);

    EXPECT_THROW(
        citlali::pipeline::require_valid_astrometry_config(
            observation, logger),
        citlali::error::Error);
    EXPECT_EQ(logger->error_calls, 1);
}

TEST(config_scaffold, rejects_incomplete_astrometry_before_install) {
    auto config = tula::config::YamlConfig::from_str(R"yaml(
pointing_offsets:
  - axes_name: az
    value_arcsec: [1.0]
)yaml");
    auto logger = std::make_shared<FakeLogger>();
    const auto observation =
        citlali::pipeline::read_astrometry_config(config, logger);

    EXPECT_THROW(
        citlali::pipeline::require_valid_astrometry_config(
            observation, logger),
        citlali::error::Error);
    EXPECT_EQ(logger->error_calls, 1);
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

TEST(config_scaffold, accepts_checked_low_level_config_schema) {
    const auto config = tula::config::YamlConfig::from_str(
        citlali::citlali_default_config_content);
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    EXPECT_TRUE(citlali::pipeline::validate_low_level_config_schema(
        config, diagnostics));
    EXPECT_FALSE(diagnostics.has_errors());
}

TEST(config_scaffold, rejects_unknown_low_level_config_nodes) {
    const auto config = tula::config::YamlConfig::from_str(R"yaml(
runtime:
  reduction_type: pointing
  unexpected_policy: true
unknown_empty_section: {}
)yaml");
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    EXPECT_FALSE(citlali::pipeline::validate_low_level_config_schema(
        config, diagnostics));
    EXPECT_EQ(
        diagnostics.invalid_key_paths(),
        (citlali::pipeline::ConfigDiagnosticsState::key_vec_t{
            {"runtime", "unexpected_policy"},
            {"unknown_empty_section"},
        }));
}

TEST(config_scaffold, accepts_known_optional_low_level_config_node) {
    const auto config = tula::config::YamlConfig::from_str(R"yaml(
pointing:
  source_strategy:
    fit_gaussian: false
)yaml");
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    EXPECT_TRUE(citlali::pipeline::validate_low_level_config_schema(
        config, diagnostics));
}

TEST(config_scaffold, leaves_tolteca_input_subtree_to_external_schema) {
    const auto config = tula::config::YamlConfig::from_str(R"yaml(
inputs:
  - meta:
      future_tolteca_metadata: retained
)yaml");
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    EXPECT_TRUE(citlali::pipeline::validate_low_level_config_schema(
        config, diagnostics));
    EXPECT_FALSE(diagnostics.has_errors());
}

TEST(config_scaffold, typed_validation_errors_are_fatal_diagnostics) {
    citlali::config::ReductionConfig config;
    config.interface_sync.toltec_offset_sec[3] =
        std::numeric_limits<double>::quiet_NaN();
    citlali::pipeline::ConfigDiagnosticsState diagnostics;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::validate_typed_config(
        config, diagnostics, logger);

    ASSERT_TRUE(diagnostics.has_errors());
    EXPECT_EQ(
        diagnostics.invalid_key_paths().front(),
        (std::vector<std::string>{"interface_sync_offset", "toltec3"}));
    EXPECT_GT(logger->error_calls, 0);
}

TEST(config_scaffold, reads_learning_into_typed_request) {
    ensure_citlali_test_logger();
    auto config = tula::config::YamlConfig::from_str(R"yaml(
timestream:
  learning:
    enabled: true
    learn_iters: 4
    map_pixel_outlier_top_n: 17
    scan_network_pathology_max_new_flagged_fraction: 0.2
)yaml");
    citlali::config::TimestreamLearningConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_learning_config(
        config, request, diagnostics);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_TRUE(request.enabled);
    EXPECT_EQ(request.learn_iters, 4);
    EXPECT_EQ(request.map_pixel_outlier.top_n, 17);
    EXPECT_DOUBLE_EQ(
        request.scan_network_pathology.max_new_flagged_fraction, 0.2);
}

TEST(config_scaffold, adapts_learning_request_one_way) {
    struct FakeOptions {
        bool enabled = false;
        bool diagnostics_enabled = false;
        int learn_iters = 0;
        int apply_start_iter = 0;
        int max_records_per_type = 0;
        bool apply_sample_masks_enabled = false;
        double apply_max_new_flagged_fraction = 0.0;
        bool map_pixel_outlier_diagnostics_enabled = false;
        bool map_pixel_outlier_contributor_diagnostics_enabled = false;
        bool map_pixel_outlier_targeted_contributor_diagnostics_enabled = false;
        bool map_pixel_outlier_detector_exclusion_enabled = false;
        int map_pixel_outlier_top_n = 0;
        int map_pixel_outlier_targeted_contributor_max_pixels = 0;
        int map_pixel_outlier_detector_exclusion_min_pixels = 0;
        double map_pixel_outlier_min_abs_z = 0.0;
        double map_pixel_outlier_min_n_eff = 0.0;
        double map_pixel_outlier_source_radius_arcsec = 0.0;
        bool busy_detector_exclusion_enabled = false;
        bool scan_network_pathology_enabled = false;
        bool scan_network_pathology_apply_pre_rtc = false;
        bool scan_network_pathology_apply_pre_ptc = false;
        bool scan_network_pathology_apply_pre_mapmaking = false;
        int scan_network_pathology_min_candidate_clusters = 0;
        int scan_network_pathology_min_candidate_events = 0;
        double scan_network_pathology_min_max_residual_z = 0.0;
        int scan_network_pathology_severe_candidate_events = 0;
        double scan_network_pathology_severe_max_residual_z = 0.0;
        double scan_network_pathology_max_new_flagged_fraction = 0.0;
    };
    struct FakeLearning {
        using Options = FakeOptions;
        FakeOptions options;
        void configure(FakeOptions value) { options = value; }
    } learning;
    citlali::config::TimestreamLearningConfig request;
    request.enabled = true;
    request.learn_iters = 5;
    request.map_pixel_outlier.top_n = 19;
    request.scan_network_pathology.max_new_flagged_fraction = 0.25;

    citlali::pipeline::adapt_learning_config_one_way(request, learning);

    EXPECT_TRUE(learning.options.enabled);
    EXPECT_EQ(learning.options.learn_iters, 5);
    EXPECT_EQ(learning.options.map_pixel_outlier_top_n, 19);
    EXPECT_DOUBLE_EQ(
        learning.options.scan_network_pathology_max_new_flagged_fraction,
        0.25);
    EXPECT_EQ(request.learn_iters, 5);
}

TEST(config_scaffold, reads_interface_sync_offsets_into_typed_request) {
    ensure_citlali_test_logger();
    auto config = tula::config::YamlConfig::from_str(R"yaml(
interface_sync_offset:
  - toltec0: 0.25
  - toltec12: -0.125
  - hwpr: 0.5
)yaml");
    citlali::config::InterfaceSyncOffsetConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    const bool clean = citlali::pipeline::read_interface_sync_offsets(
        config, request, diagnostics,
        spdlog::get("citlali_logger"));

    EXPECT_TRUE(clean);
    EXPECT_FALSE(diagnostics.has_errors());
    EXPECT_DOUBLE_EQ(request.toltec_offset_sec[0], 0.25);
    EXPECT_DOUBLE_EQ(request.toltec_offset_sec[12], -0.125);
    EXPECT_DOUBLE_EQ(request.hwpr_offset_sec, 0.5);
}

TEST(config_scaffold, rejects_interface_sync_duplicates_atomically) {
    ensure_citlali_test_logger();
    auto config = tula::config::YamlConfig::from_str(R"yaml(
interface_sync_offset:
  - toltec0: 0.25
  - toltec0: 0.5
)yaml");
    citlali::config::InterfaceSyncOffsetConfig request;
    request.toltec_offset_sec[0] = 9.0;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    const bool clean = citlali::pipeline::read_interface_sync_offsets(
        config, request, diagnostics,
        spdlog::get("citlali_logger"));

    EXPECT_FALSE(clean);
    EXPECT_TRUE(diagnostics.has_errors());
    EXPECT_DOUBLE_EQ(request.toltec_offset_sec[0], 9.0);
}

TEST(config_scaffold, rejects_unknown_interface_sync_entry) {
    ensure_citlali_test_logger();
    auto config = tula::config::YamlConfig::from_str(R"yaml(
interface_sync_offset:
  - unknown_interface: 0.25
)yaml");
    citlali::config::InterfaceSyncOffsetConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    EXPECT_FALSE(citlali::pipeline::read_interface_sync_offsets(
        config, request, diagnostics,
        spdlog::get("citlali_logger")));
    EXPECT_TRUE(diagnostics.has_errors());
}

TEST(config_scaffold, rejects_nonfinite_interface_sync_offset) {
    ensure_citlali_test_logger();
    auto config = tula::config::YamlConfig::from_str(R"yaml(
interface_sync_offset:
  - toltec0: .nan
)yaml");
    citlali::config::InterfaceSyncOffsetConfig request;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    EXPECT_FALSE(citlali::pipeline::read_interface_sync_offsets(
        config, request, diagnostics,
        spdlog::get("citlali_logger")));
    EXPECT_TRUE(diagnostics.has_errors());
}

TEST(config_scaffold, adapts_interface_sync_request_one_way) {
    citlali::config::InterfaceSyncOffsetConfig request;
    request.toltec_offset_sec[0] = 0.25;
    request.toltec_offset_sec[12] = -0.125;
    request.hwpr_offset_sec = 0.5;
    std::map<std::string, double> offsets;

    citlali::pipeline::adapt_interface_sync_config_one_way(
        request, offsets);

    EXPECT_EQ(offsets.size(), 14U);
    EXPECT_DOUBLE_EQ(offsets.at("toltec0"), 0.25);
    EXPECT_DOUBLE_EQ(offsets.at("toltec12"), -0.125);
    EXPECT_DOUBLE_EQ(offsets.at("hwpr"), 0.5);
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

TEST(config_scaffold, source_finding_requires_map_filtering) {
    citlali::config::PostProcessingConfig config;
    citlali::config::set_source_finding_enabled(config, true);
    config.source_finding.source_sigma = 5.0;
    config.source_finding.source_window_arcsec = 30.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);

    ASSERT_FALSE(report.ok());
    ASSERT_EQ(report.error_count(), 1U);
    EXPECT_NE(report.format_for_cli().find(
                  "post_processing.source_finding.enabled"),
              std::string::npos);
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

TEST(config_scaffold, validates_beammap_photometry_values) {
    citlali::config::BeammapPhotometryConfig config;
    config.fluxes.push_back(
        citlali::config::BeammapArrayFluxConfig{"", 0.0, -1.0});

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
    todproc.engine().inject_config_error = true;
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
            false>(todproc, rawobs, rawobs_kids_meta, false, 0, logger));

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
            false>(todproc, rawobs, rawobs_kids_meta, true, 0, logger));

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
    engine.inject_config_error = true;
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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::setup_and_run_observation_pipeline(
        engine, kidsproc, rawobs, stage_profile, logger);

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::setup_and_run_observation_pipeline(
        engine, kidsproc, rawobs, stage_profile, logger);

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::begin_reduction_iteration(
        todproc, config_filepaths, stage_profile, logger);

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
        todproc, rawobs, rawobs_kids_meta, map_extents, map_coords, 0, logger));

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
        todproc, config, rawobs, map_extents, map_coords, 0, logger));

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
        todproc, rawobs, rawobs_kids_meta, map_extents, map_coords, 0, logger));

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

TEST(pipeline_execution, records_observation_mapmaking_cardinality) {
    FakeObservationMapTodProc todproc;
    auto &engine = todproc.engine();
    engine.mapmaking_plan.reset_from_request(
        engine.typed_config.mapmaking,
        citlali::config::ReductionType::pointing);
    engine.mapmaking_plan.begin_iteration();
    engine.map_indices.n_maps = 3;
    engine.omb.pixel_size_rad = 4.848136811e-6;
    engine.observation_identity.obsnum = "152389";
    citlali::config::set_map_filtering_enabled(
        engine.typed_config.post_processing, true);
    std::vector<int> map_extents = {11};
    std::vector<int> map_coords = {22};
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::allocate_observation_map_buffers_if_needed(
        todproc, map_extents, map_coords, 0, logger);
    citlali::pipeline::complete_mapmaking_observation_if_available(engine);

    ASSERT_EQ(engine.mapmaking_plan.observations.size(), 1U);
    const auto &observation = engine.mapmaking_plan.observations.front();
    EXPECT_EQ(observation.observation_index, 0U);
    EXPECT_EQ(observation.obsnum, "152389");
    EXPECT_EQ(observation.map_count, 3U);
    EXPECT_DOUBLE_EQ(observation.effective_pixel_size_rad,
                     4.848136811e-6);
    EXPECT_EQ(observation.required_map_write_count, 6U);
    EXPECT_TRUE(observation.outputs_completed);
    EXPECT_EQ(*engine.mapmaking_plan.realized.completed_observation_count,
              1U);
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
        [](auto &engine) {
            return engine.telescope.get_tel_data_calls == 1
                       ? std::string{"2026-01-01T00:00:00"}
                       : std::string{"stale-telescope-state"};
        },
        logger));

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
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        logger));

    EXPECT_EQ(todproc.get_tone_freqs_from_files_calls, 0);
    EXPECT_TRUE(todproc.engine().observation_dates.date_obs.empty());
    EXPECT_EQ(todproc.allocate_omb_calls, 0);
}

TEST(pipeline_execution, coadds_observation) {
    FakeCoaddTodProc todproc;
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::coadd_observation(todproc, stage_profile, logger);

    EXPECT_EQ(todproc.coadd_calls, 1);
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(pipeline_execution, skips_coadd_for_polarization) {
    FakeCoaddTodProc todproc;
    todproc.engine().rtcproc.run_polarization = true;
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::coadd_observation(todproc, stage_profile, logger);

    EXPECT_EQ(todproc.coadd_calls, 0);
    EXPECT_EQ(logger->info_calls, 2);
}

TEST(pipeline_execution, writes_raw_observation_outputs) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.mapmaking.enabled = true;
    todproc.engine().typed_config.noise.products_enabled = true;
    todproc.engine().typed_config.noise.enabled = true;
    todproc.engine().typed_config.noise.apply_empirical_weights = true;
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_raw_observation_outputs<FakeMapType::RawObs>(
        todproc, stage_profile, logger);

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_raw_observation_outputs<FakeMapType::RawObs>(
        todproc, stage_profile, logger);

    EXPECT_EQ(todproc.engine().omb.calc_noise_products_calls, 0);
    EXPECT_EQ(todproc.engine().create_obs_map_files_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
    EXPECT_EQ(logger->info_calls, 3);
}

TEST(pipeline_execution, skips_raw_outputs_when_mapmaking_disabled) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.mapmaking.enabled = false;
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_raw_observation_outputs<FakeMapType::RawObs>(
        todproc, stage_profile, logger);

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_filtered_observation_outputs<
        FakeMapType::FilteredObs, false>(todproc, stage_profile, logger);

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_filtered_observation_outputs<
        FakeMapType::FilteredObs, true>(todproc, stage_profile, logger);

    EXPECT_EQ(todproc.engine().fit_maps_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
}

TEST(pipeline_execution, skips_post_filter_observation_output_for_science) {
    FakeCoaddTodProc todproc;
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_filtered_observation_outputs<
        FakeMapType::FilteredObs, false>(todproc, stage_profile, logger);

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_observation_outputs_and_accumulate<
        FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
        todproc, stage_profile, logger);

    EXPECT_EQ(todproc.engine().output_calls, 1);
    EXPECT_EQ(todproc.coadd_calls, 0);
    EXPECT_EQ(todproc.engine().run_wiener_filter_calls, 0);
}

TEST(pipeline_execution, writes_observation_outputs_and_coadds) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.mapmaking.enabled = true;
    todproc.engine().typed_config.coadd.enabled = true;
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_observation_outputs_and_accumulate<
        FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
        todproc, stage_profile, logger);

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_observation_outputs_and_accumulate<
        FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
        todproc, stage_profile, logger);

    EXPECT_EQ(todproc.engine().output_calls, 2);
    EXPECT_EQ(todproc.coadd_calls, 0);
    EXPECT_EQ(todproc.engine().run_wiener_filter_calls, 1);
}

TEST(pipeline_execution,
     map_output_failure_does_not_complete_observation_cardinality) {
    FakeCoaddTodProc todproc;
    auto &engine = todproc.engine();
    engine.mapmaking_plan.reset_from_request(
        engine.typed_config.mapmaking,
        citlali::config::ReductionType::pointing);
    engine.mapmaking_plan.begin_iteration();
    engine.mapmaking_plan.begin_observation(
        0, "152389", 3, 4.848136811e-6, 3);
    engine.output_throws = true;
    engine.typed_config.coadd.enabled = false;
    citlali::config::set_map_filtering_enabled(
        engine.typed_config.post_processing, false);
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_THROW(
        (citlali::pipeline::write_observation_outputs_and_accumulate<
            FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
            todproc, stage_profile, logger)),
        std::runtime_error);

    EXPECT_FALSE(
        engine.mapmaking_plan.observations.front().outputs_completed);
    EXPECT_EQ(*engine.mapmaking_plan.realized.completed_observation_count,
              0U);
    EXPECT_THROW(
        citlali::pipeline::record_mapmaking_run_completed(
            engine.mapmaking_plan),
        std::logic_error);
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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::run_reduction_observation_pipeline<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
        todproc, kidsproc, rawobs, stage_profile, logger);

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::run_reduction_observation<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
        todproc, kidsproc, rawobs, rawobs_kids_meta, true, map_extents,
        map_coords, 0,
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        stage_profile, logger));

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_FALSE(citlali::pipeline::run_reduction_observation<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs, false>(
        todproc, kidsproc, rawobs, rawobs_kids_meta, true, map_extents,
        map_coords, 0,
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        stage_profile, logger));

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::run_reduction_observation_at_index<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs, false,
        FakeKidsProc>(
        todproc, co, config, map_extents, map_coords, 1,
        [](auto &engine) {
            return "telescope-loaded-" +
                   std::to_string(engine.telescope.get_tel_data_calls);
        },
        stage_profile, logger));

    EXPECT_EQ(config.get_config_calls, 1);
    EXPECT_EQ(todproc.last_map_extent, 33);
    EXPECT_EQ(todproc.last_map_coord, 44);
    EXPECT_EQ(todproc.engine().setup_calls, 1);
    EXPECT_EQ(todproc.engine().pipeline_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
    EXPECT_EQ(todproc.engine().observation_dates.date_obs,
              (std::vector<std::string>{"telescope-loaded-1"}));
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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    try {
        citlali::pipeline::run_reduction_observation_at_index<
            false, FakeMapType::RawObs, FakeMapType::FilteredObs, false,
            FakeFailingKidsProc>(
            todproc, co, config, map_extents, map_coords, 0,
            [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
            stage_profile, logger);
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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::run_reduction_iteration_observations<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs, false,
        FakeKidsProc>(
        todproc, co, config, map_extents, map_coords,
        [](auto &engine) {
            return "telescope-loaded-" +
                   std::to_string(engine.telescope.get_tel_data_calls);
        },
        stage_profile, logger));

    EXPECT_EQ(config.get_config_calls, 2);
    EXPECT_EQ(todproc.engine().setup_calls, 2);
    EXPECT_EQ(todproc.engine().pipeline_calls, 2);
    EXPECT_EQ(todproc.engine().output_calls, 2);
    EXPECT_EQ(todproc.engine().observation_dates.date_obs,
              (std::vector<std::string>{"telescope-loaded-1",
                                        "telescope-loaded-2"}));
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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_FALSE(citlali::pipeline::run_reduction_iteration_observations<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs, false,
        FakeKidsProc>(
        todproc, co, config, map_extents, map_coords,
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        stage_profile, logger));

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::run_reduction_iteration<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs,
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd, false,
        FakeKidsProc>(
        todproc, co, config, config_filepaths, map_extents, map_coords,
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        stage_profile, logger));

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
    citlali::pipeline::reset_processed_timestream_execution_plan(
        todproc.engine().processed_timestream_plan,
        todproc.engine().typed_config.timestream);
    FakeCitlaliConfig config;
    FakeIOCoordinator co{{FakeRawObs{}}};
    std::vector<std::string> config_filepaths;
    std::vector<int> map_extents = {11};
    std::vector<int> map_coords = {22};
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::run_reduction_iterations<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs,
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd, false,
        FakeKidsProc>(
        todproc, co, config, config_filepaths, map_extents, map_coords,
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        stage_profile, logger));

    EXPECT_EQ(todproc.engine().iteration.fruit_iter, 2);
    EXPECT_EQ(todproc.engine().setup_calls, 2);
    EXPECT_EQ(todproc.engine().pipeline_calls, 2);
    EXPECT_EQ(todproc.engine().output_calls, 2);
    EXPECT_EQ(todproc.make_index_file_calls, 2);
    EXPECT_EQ(todproc.create_output_dir_calls, 1);
    ASSERT_TRUE(todproc.engine().processed_timestream_plan.realized
                    .fruit_loop_iterations_completed.has_value());
    EXPECT_EQ(*todproc.engine().processed_timestream_plan.realized
                   .fruit_loop_iterations_completed,
              2);
    ASSERT_TRUE(todproc.engine().processed_timestream_plan.realized
                    .fruit_loops_converged.has_value());
    EXPECT_FALSE(*todproc.engine().processed_timestream_plan.realized
                      .fruit_loops_converged);
}

TEST(pipeline_execution, runs_reduction_pipeline) {
    FakeInitialObservationTodProc todproc;
    FakeCitlaliConfig config;
    FakeIOCoordinator co{{FakeRawObs{}}};
    std::vector<std::string> config_filepaths;
    std::vector<int> map_extents;
    std::vector<int> map_coords;
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();
    stage_profile.record("session.pre_pipeline", {}, 0.25, logger);

    EXPECT_TEMPLATE_TRUE(citlali::pipeline::run_reduction_pipeline<
        false, FakeMapType::RawObs, FakeMapType::FilteredObs,
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd, false,
        FakeKidsProc>(
        todproc, co, config, config_filepaths, map_extents, map_coords,
        [](auto &) { return std::string{"2026-01-01T00:00:00"}; },
        stage_profile, logger));

    EXPECT_EQ(todproc.calc_omb_size_calls, 1);
    EXPECT_EQ(map_extents, (std::vector<int>{303}));
    EXPECT_EQ(map_coords, (std::vector<int>{404}));
    EXPECT_EQ(todproc.engine().setup_calls, 1);
    EXPECT_EQ(todproc.engine().pipeline_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
    EXPECT_EQ(todproc.engine().iteration.fruit_iter, 1);
    const auto profile_records = stage_profile.records();
    const auto has_profile_stage = [&](const std::string &stage) {
        return std::any_of(
            profile_records.begin(), profile_records.end(),
            [&](const auto &record) { return record.stage == stage; });
    };
    EXPECT_TRUE(has_profile_stage("reduction.iterations"));
    EXPECT_TRUE(has_profile_stage("observation.pipeline"));
    EXPECT_TRUE(has_profile_stage("map.output"));
    EXPECT_TRUE(has_profile_stage("session.pre_pipeline"));
}

TEST(pipeline_execution, writes_raw_coadd_outputs) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.noise.apply_empirical_weights = true;
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_raw_coadd_outputs<FakeMapType::RawCoadd>(
        todproc, stage_profile, logger);

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_raw_coadd_outputs<FakeMapType::RawCoadd>(
        todproc, stage_profile, logger);

    EXPECT_EQ(todproc.engine().cmb.normalize_maps_calls, 0);
    EXPECT_EQ(todproc.engine().cmb.normalize_polarized_maps_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 1);
}

TEST(pipeline_execution, skips_raw_coadd_noise_products_when_disabled) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.noise.products_enabled = false;
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_raw_coadd_outputs<FakeMapType::RawCoadd>(
        todproc, stage_profile, logger);

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_filtered_coadd_outputs<
        FakeMapType::FilteredCoadd>(todproc, stage_profile, logger);

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_filtered_coadd_outputs<
        FakeMapType::FilteredCoadd>(todproc, stage_profile, logger);

    EXPECT_EQ(todproc.engine().cmb.calc_noise_products_calls, 0);
    EXPECT_EQ(todproc.engine().output_calls, 0);
    EXPECT_EQ(logger->info_calls, 10);
}

TEST(pipeline_execution, skips_iteration_coadd_outputs_when_coadd_disabled) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = false;
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_iteration_coadd_outputs_if_needed<
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd>(
        todproc, stage_profile, logger);

    EXPECT_EQ(todproc.create_coadded_map_files_calls, 0);
    EXPECT_EQ(todproc.engine().output_calls, 0);
}

TEST(pipeline_execution, writes_iteration_raw_coadd_outputs) {
    FakeCoaddTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = true;
    citlali::config::set_map_filtering_enabled(
        todproc.engine().typed_config.post_processing, false);
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_iteration_coadd_outputs_if_needed<
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd>(
        todproc, stage_profile, logger);

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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_iteration_coadd_outputs_if_needed<
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd>(
        todproc, stage_profile, logger);

    EXPECT_EQ(todproc.create_coadded_map_files_calls, 1);
    EXPECT_EQ(todproc.engine().run_wiener_filter_calls, 1);
    EXPECT_EQ(todproc.engine().output_calls, 2);
}

TEST(pipeline_execution, records_coadd_mapmaking_cardinality) {
    FakeCoaddTodProc todproc;
    auto &engine = todproc.engine();
    engine.typed_config.runtime.reduction_type =
        citlali::config::ReductionType::pointing;
    sync_fake_runtime_provenance(engine);
    engine.typed_config.coadd.enabled = true;
    citlali::config::set_map_filtering_enabled(
        engine.typed_config.post_processing, true);
    engine.mapmaking_plan.reset_from_request(
        engine.typed_config.mapmaking,
        citlali::config::ReductionType::pointing);
    engine.mapmaking_plan.begin_iteration();
    engine.map_indices.n_maps = 3;
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::write_iteration_coadd_outputs_if_needed<
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd>(
        todproc, stage_profile, logger);

    ASSERT_TRUE(engine.mapmaking_plan.coadd.has_value());
    EXPECT_EQ(engine.mapmaking_plan.coadd->map_count, 3U);
    EXPECT_EQ(engine.mapmaking_plan.coadd->required_map_write_count, 6U);
    EXPECT_TRUE(engine.mapmaking_plan.coadd->outputs_completed);
    EXPECT_EQ(*engine.mapmaking_plan.realized.completed_coadd_count, 1U);
}

TEST(pipeline_execution, finishes_reduction_iteration) {
    FakeReductionIterationTodProc todproc;
    todproc.engine().typed_config.coadd.enabled = false;
    todproc.engine().iteration.fruit_iter = 2;
    todproc.engine().output_paths.redu_dir_name = "/data/redu02";
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::finish_reduction_iteration<
        FakeMapType::RawCoadd, FakeMapType::FilteredCoadd>(
        todproc, stage_profile, logger);

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

TEST(config_scaffold, separates_raw_requested_effective_and_observation_state) {
    citlali::config::RawTimeChunkConfig request;
    request.filter.enabled = false;
    request.filter.freq_high_Hz = 23.5;
    request.downsample.enabled = true;
    request.downsample.factor = 0;
    request.downsample.downsampled_freq_Hz = 30.0;

    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(request);
    plan.effective.downsample.factor = 4;
    auto &observation = plan.begin_observation();
    observation.native_sample_rate_hz = 488.0;
    observation.effective_sample_rate_hz = 122.0;
    observation.downsample_factor = 4;
    plan.realized.completed_scan_count = 3;

    EXPECT_TRUE(plan.initialized);
    EXPECT_FALSE(plan.requested.filter.enabled);
    EXPECT_DOUBLE_EQ(plan.requested.filter.freq_high_Hz, 23.5);
    EXPECT_EQ(plan.requested.downsample.factor, 0);
    EXPECT_EQ(plan.effective.downsample.factor, 4);
    ASSERT_TRUE(plan.observation.has_value());
    EXPECT_DOUBLE_EQ(*plan.observation->native_sample_rate_hz, 488.0);
    EXPECT_EQ(*plan.observation->downsample_factor, 4);
    EXPECT_EQ(plan.realized.completed_scan_count, 3U);
}

TEST(config_scaffold, resolves_raw_context_free_request_without_observation_data) {
    citlali::config::RawTimeChunkConfig request;
    request.filter.enabled = false;
    request.filter.notch.enabled = true;
    request.filter.edge_guard.enabled = true;
    request.iir_filter.enabled = true;
    request.downsample.enabled = true;
    request.downsample.factor = 0;
    request.downsample.downsampled_freq_Hz = 40.0;
    request.despike.enabled = false;
    request.despike.source_protection.enabled = true;
    request.flux_calibration_enabled = true;
    request.extinction_correction_enabled = true;

    const auto resolution =
        citlali::pipeline::resolve_raw_timestream_effective_request(
            request);

    EXPECT_FALSE(resolution.filtering.fir_requested);
    EXPECT_TRUE(resolution.filtering.fixed_notch_requested);
    EXPECT_FALSE(resolution.filtering.fixed_notch_effective);
    EXPECT_TRUE(resolution.filtering.iir_highpass_requested);
    EXPECT_TRUE(resolution.filtering.edge_guard_requested);
    EXPECT_TRUE(resolution.filtering.downsample_requested);
    EXPECT_FALSE(
        resolution.filtering.downsample_filter_dependency_satisfied);
    EXPECT_EQ(
        resolution.downsampling.kind,
        citlali::pipeline::RawDownsampleRequestKind::target_frequency);
    EXPECT_EQ(resolution.downsampling.requested_factor, 0);
    EXPECT_DOUBLE_EQ(
        resolution.downsampling.requested_frequency_hz, 40.0);
    EXPECT_FALSE(resolution.source_protection.despike_requested);
    EXPECT_FALSE(resolution.source_protection.source_protection_requested);
    EXPECT_TRUE(resolution.corrections.flux_calibration_requested);
    EXPECT_TRUE(resolution.corrections.extinction_correction_requested);
}

TEST(config_scaffold, reads_raw_filtering_request_without_activation_loss) {
    ensure_citlali_test_logger();
    auto config = tula::config::YamlConfig::from_str(R"yaml(
timestream:
  raw_time_chunk:
    kernel:
      enabled: true
      filepath: kernel.fits
      type: fits
      fwhm_arcsec: 7.5
      image_ext_names: [a1100, a1400, a2000]
    filter:
      enabled: false
      a_gibbs: 42.0
      freq_low_Hz: 0.25
      freq_high_Hz: 18.5
      n_terms: 48
      notch:
        enabled: true
        zero_phase: true
        freqs_Hz: [10.0, 12.0]
        delta_f_Hz: [0.2]
      edge_guard:
        enabled: true
        mode: flag
        combine: max
        min_samples: 3
        extra_samples: 4
        max_samples: 96
        iir_settle_attenuation: 0.02
        apply_fir: true
        apply_notch: false
        apply_dynamic_notch: true
        apply_iir_highpass: false
        apply_downsample: true
    IIR_filter:
      enabled: false
      freq_Hz: 0.35
      order: 3
      zero_phase: true
    downsample:
      enabled: true
      factor: 0
      downsampled_freq_Hz: 40.0
    flux_calibration:
      enabled: true
    extinction_correction:
      enabled: false
    altaz_destripe:
      enabled: true
      grouping: array
      fit_time_trend: false
      fit_derivs: true
      min_samples: 80
)yaml");
    citlali::config::RawTimeChunkConfig raw;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_raw_filtering_request_config(
        config, raw, diagnostics);

    EXPECT_FALSE(diagnostics.has_errors());
    EXPECT_TRUE(raw.kernel.enabled);
    EXPECT_EQ(raw.kernel.filepath, "kernel.fits");
    EXPECT_DOUBLE_EQ(raw.kernel.fwhm_arcsec, 7.5);
    EXPECT_EQ(raw.kernel.image_ext_names.size(), 3U);
    EXPECT_FALSE(raw.filter.enabled);
    EXPECT_DOUBLE_EQ(raw.filter.freq_high_Hz, 18.5);
    EXPECT_EQ(raw.filter.n_terms, 48);
    EXPECT_TRUE(raw.filter.notch.enabled);
    EXPECT_EQ(raw.filter.notch.freqs_Hz.size(), 2U);
    EXPECT_EQ(raw.filter.notch.delta_f_Hz.size(), 1U);
    EXPECT_TRUE(raw.filter.edge_guard.enabled);
    EXPECT_TRUE(citlali::config::is_max_raw_filter_edge_guard_combine(
        raw.filter.edge_guard.combine));
    EXPECT_EQ(raw.filter.edge_guard.max_samples, 96);
    EXPECT_FALSE(raw.filter.edge_guard.apply_notch);
    EXPECT_FALSE(raw.iir_filter.enabled);
    EXPECT_DOUBLE_EQ(raw.iir_filter.freq_Hz, 0.35);
    EXPECT_EQ(raw.iir_filter.order, 3);
    EXPECT_TRUE(raw.iir_filter.zero_phase);
    EXPECT_TRUE(raw.downsample.enabled);
    EXPECT_EQ(raw.downsample.factor, 0);
    EXPECT_DOUBLE_EQ(raw.downsample.downsampled_freq_Hz, 40.0);
    EXPECT_TRUE(raw.flux_calibration_enabled);
    EXPECT_FALSE(raw.extinction_correction_enabled);
    EXPECT_TRUE(raw.altaz_destripe.enabled);
    EXPECT_EQ(raw.altaz_destripe.grouping, "array");
    EXPECT_FALSE(raw.altaz_destripe.fit_time_trend);
    EXPECT_EQ(raw.altaz_destripe.min_samples, 80);
}

TEST(config_scaffold, reads_raw_flagging_request_and_legacy_threshold_alias) {
    ensure_citlali_test_logger();
    auto config = tula::config::YamlConfig::from_str(R"yaml(
timestream:
  raw_time_chunk:
    despike:
      enabled: false
      min_spike_sigma: 9.0
      time_constant_sec: 0.025
      window_size: 40.0
      legacy:
        enabled: false
      source_protection:
        enabled: true
        radius_arcsec: 28.0
      local_residual:
        enabled: false
        sigma_scale: 0.5
        compact_raw_gate:
          candidate_sigma_scale: 1.5
        compact_delta_gate:
          enabled: true
          window_sec: 0.2
    flagging:
      delta_f_min_Hz: 70000.0
      lower_tod_inv_var_factor: 0.25
      upper_tod_inv_var_factor: 4.0
      network_step_mask:
        enabled: true
        min_det_used: 24
      impulsive_capture:
        enabled: true
        max_events_per_network: 5
      impulsive_coincidence:
        enabled: true
        min_networks_aligned: 4
        max_flagged_fraction: 0.2
)yaml");
    citlali::config::RawTimeChunkConfig raw;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_raw_flagging_and_despike_request_config(
        config, raw, diagnostics);

    EXPECT_FALSE(diagnostics.has_errors());
    EXPECT_FALSE(raw.despike.enabled);
    EXPECT_DOUBLE_EQ(raw.despike.min_spike_sigma, 9.0);
    EXPECT_FALSE(raw.despike.legacy_enabled);
    EXPECT_TRUE(raw.despike.source_protection.enabled);
    EXPECT_FALSE(raw.despike.source_protection.active);
    EXPECT_DOUBLE_EQ(raw.despike.source_protection.radius_arcsec, 28.0);
    EXPECT_FALSE(raw.despike.local_residual.enabled);
    EXPECT_DOUBLE_EQ(raw.despike.local_residual.sigma_scale, 0.5);
    EXPECT_DOUBLE_EQ(
        raw.despike.local_residual.compact_raw_gate
            .candidate_rel_sigma_scale,
        3.0);
    EXPECT_TRUE(raw.despike.local_residual.compact_delta_gate.enabled);
    EXPECT_DOUBLE_EQ(
        raw.despike.local_residual.compact_delta_gate.window_sec, 0.2);
    EXPECT_DOUBLE_EQ(raw.flagging.delta_f_min_Hz, 70000.0);
    EXPECT_EQ(raw.flagging.network_step_mask.min_det_used, 24);
    EXPECT_EQ(raw.flagging.impulsive_capture.max_events_per_network, 5);
    EXPECT_EQ(
        raw.flagging.impulsive_coincidence.min_networks_aligned, 4);
    EXPECT_DOUBLE_EQ(
        raw.flagging.impulsive_coincidence.max_flagged_fraction, 0.2);
}

TEST(config_scaffold, reads_complete_raw_request_through_typed_boundary) {
    ensure_citlali_test_logger();
    auto config = tula::config::YamlConfig::from_str(R"yaml(
timestream:
  raw_time_chunk:
    filter:
      enabled: false
      freq_high_Hz: 17.0
    despike:
      enabled: false
      min_spike_sigma: 11.0
    flagging:
      delta_f_min_Hz: 65000.0
    line_audit:
      enabled: false
      line_min_hz: 2.0
      line_max_hz: 55.0
      post_filter_enabled: true
      post_filter_apply_iterations: 2
      fixed_notch_enabled: true
      fixed_notch_freqs_hz: [12.0, 24.0]
      fixed_notch_widths_hz: [0.25]
      apply_shared_notches: true
      apply_max_notches: 5
      detector_notch_context_samples: 32
)yaml");
    citlali::config::RawTimeChunkConfig raw;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    citlali::pipeline::read_raw_timestream_request_config(
        config, raw, diagnostics);

    EXPECT_FALSE(diagnostics.has_errors());
    EXPECT_FALSE(raw.filter.enabled);
    EXPECT_DOUBLE_EQ(raw.filter.freq_high_Hz, 17.0);
    EXPECT_FALSE(raw.despike.enabled);
    EXPECT_DOUBLE_EQ(raw.despike.min_spike_sigma, 11.0);
    EXPECT_DOUBLE_EQ(raw.flagging.delta_f_min_Hz, 65000.0);
    EXPECT_FALSE(raw.line_audit.enabled);
    EXPECT_DOUBLE_EQ(raw.line_audit.line_min_hz, 2.0);
    EXPECT_DOUBLE_EQ(raw.line_audit.line_max_hz, 55.0);
    EXPECT_TRUE(raw.line_audit.post_filter_enabled);
    EXPECT_EQ(raw.line_audit.post_filter_apply_iterations, 2);
    EXPECT_TRUE(raw.line_audit.fixed_notch_enabled);
    EXPECT_EQ(raw.line_audit.fixed_notch_freqs_hz.size(), 2U);
    EXPECT_EQ(raw.line_audit.fixed_notch_widths_hz.size(), 1U);
    EXPECT_TRUE(raw.line_audit.apply_shared_notches);
    EXPECT_EQ(raw.line_audit.apply_max_notches, 5);
    EXPECT_EQ(raw.line_audit.detector_notch_context_samples, 32);
}

TEST(config_scaffold, serializes_raw_request_without_observation_state) {
    citlali::config::RawTimeChunkConfig request;
    request.despike.enabled = false;
    request.despike.min_spike_sigma = 9.5;
    request.despike.source_protection.enabled = true;
    request.despike.source_protection.active = true;
    request.despike.source_protection.radius_arcsec = 26.0;
    request.filter.enabled = false;
    request.filter.freq_high_Hz = 21.0;
    request.filter.notch.freqs_Hz = {10.0, 12.0};
    request.iir_filter.enabled = false;
    request.iir_filter.freq_Hz = 0.4;
    request.downsample.factor = 0;
    request.line_audit.fixed_notch_widths_hz = {0.2, 0.3};
    request.extinction_correction_enabled = true;
    request.extinction_model = "observation-derived-model";

    const auto node =
        citlali::pipeline::raw_timestream_request_node(request);

    EXPECT_FALSE(node["despike"]["enabled"].as<bool>());
    EXPECT_DOUBLE_EQ(
        node["despike"]["min_spike_sigma"].as<double>(), 9.5);
    EXPECT_TRUE(
        node["despike"]["source_protection"]["enabled"].as<bool>());
    EXPECT_DOUBLE_EQ(
        node["despike"]["source_protection"]["radius_arcsec"]
            .as<double>(),
        26.0);
    EXPECT_FALSE(
        node["despike"]["source_protection"]["active"].IsDefined());
    EXPECT_FALSE(node["filter"]["enabled"].as<bool>());
    EXPECT_DOUBLE_EQ(node["filter"]["freq_high_Hz"].as<double>(), 21.0);
    EXPECT_EQ(node["filter"]["notch"]["freqs_Hz"].size(), 2U);
    EXPECT_FALSE(node["IIR_filter"]["enabled"].as<bool>());
    EXPECT_DOUBLE_EQ(node["IIR_filter"]["freq_Hz"].as<double>(), 0.4);
    EXPECT_EQ(node["downsample"]["factor"].as<int>(), 0);
    EXPECT_EQ(node["line_audit"]["fixed_notch_widths_hz"].size(), 2U);
    EXPECT_TRUE(node["extinction_correction"]["enabled"].as<bool>());
    EXPECT_FALSE(node["extinction_correction"]["model"].IsDefined());
}

TEST(config_scaffold, resets_raw_observation_and_realized_state) {
    citlali::config::RawTimeChunkConfig first_request;
    first_request.filter.enabled = true;
    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(first_request);
    plan.begin_observation().filter_edge_guard_samples = 24;
    plan.realized.dynamic_notch_count = 3;

    citlali::config::RawTimeChunkConfig second_request;
    second_request.filter.enabled = false;
    second_request.filter.freq_high_Hz = 19.0;
    plan.reset_from_request(second_request);

    EXPECT_TRUE(plan.initialized);
    EXPECT_FALSE(plan.requested.filter.enabled);
    EXPECT_DOUBLE_EQ(plan.requested.filter.freq_high_Hz, 19.0);
    EXPECT_FALSE(plan.effective.filter.enabled);
    EXPECT_FALSE(plan.effective_resolutions.filtering.fir_requested);
    EXPECT_EQ(
        plan.effective_resolutions.downsampling.kind,
        citlali::pipeline::RawDownsampleRequestKind::disabled);
    EXPECT_FALSE(plan.observation.has_value());
    EXPECT_FALSE(plan.realized.dynamic_notch_count.has_value());

    plan.begin_observation().filter_edge_guard_samples = 8;
    plan.realized.completed_scan_count = 2;
    plan.begin_observation();
    ASSERT_TRUE(plan.observation.has_value());
    EXPECT_FALSE(plan.observation->filter_edge_guard_samples.has_value());
    EXPECT_FALSE(plan.realized.completed_scan_count.has_value());
}

TEST(config_scaffold, shadows_raw_observation_legacy_state_without_mutation) {
    citlali::config::RawTimeChunkConfig request;
    request.downsample.enabled = true;
    request.downsample.factor = 2;
    request.filter.edge_guard.enabled = true;
    request.filter.edge_guard.min_samples = 4;
    request.despike.enabled = true;
    request.despike.source_protection.enabled = true;
    request.extinction_correction_enabled = true;

    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(request);
    timestream::RTCProc rtcproc;
    rtcproc.run_downsample = true;
    rtcproc.downsampler.factor = 2;
    rtcproc.filter_edge_guard.guard_samples = 4;
    rtcproc.filter_edge_guard.context_samples = 4;
    rtcproc.despiker.source_protection_enabled = true;

    const auto begin =
        citlali::pipeline::begin_raw_timestream_observation_shadow(
            plan, citlali::config::ReductionType::pointing,
            100.0, 50.0, rtcproc);

    EXPECT_TRUE(begin.exact) << begin.diagnostic();
    EXPECT_FALSE(begin.edge_guard_deferred);
    ASSERT_TRUE(plan.observation.has_value());
    EXPECT_DOUBLE_EQ(*plan.observation->native_sample_rate_hz, 100.0);
    EXPECT_DOUBLE_EQ(*plan.observation->effective_sample_rate_hz, 50.0);
    EXPECT_EQ(*plan.observation->downsample_factor, 2);
    EXPECT_EQ(*plan.observation->filter_edge_guard_samples, 4);
    EXPECT_FALSE(plan.observation->filter_edge_guard_parity_deferred);
    EXPECT_TRUE(*plan.observation->source_protection_active);
    EXPECT_FALSE(plan.observation->extinction_active.has_value());
    EXPECT_EQ(plan.requested.downsample.factor, 2);

    rtcproc.run_extinction = true;
    rtcproc.calibration.setup(0.1);
    const auto extinction =
        citlali::pipeline::complete_raw_timestream_extinction_shadow(
            plan, 0.1, rtcproc.calibration.tx_225_zenith,
            rtcproc.run_extinction,
            rtcproc.calibration.extinction_model);

    EXPECT_TRUE(extinction.exact) << extinction.diagnostic();
    EXPECT_TRUE(*plan.observation->extinction_active);
    EXPECT_EQ(*plan.observation->extinction_model,
              rtcproc.calibration.extinction_model);

    plan.realized.completed_scan_count = 7;
    const auto second =
        citlali::pipeline::begin_raw_timestream_observation_shadow(
            plan, citlali::config::ReductionType::pointing,
            120.0, 60.0, rtcproc);

    EXPECT_TRUE(second.exact) << second.diagnostic();
    ASSERT_TRUE(plan.observation.has_value());
    EXPECT_DOUBLE_EQ(*plan.observation->native_sample_rate_hz, 120.0);
    EXPECT_FALSE(plan.observation->extinction_active.has_value());
    EXPECT_FALSE(plan.observation->extinction_model.has_value());
    EXPECT_FALSE(plan.realized.completed_scan_count.has_value());
}

TEST(config_scaffold,
     ignores_inactive_legacy_downsample_factor_in_observation_shadow) {
    citlali::config::RawTimeChunkConfig request;
    request.downsample.enabled = false;
    request.downsample.factor = 1;

    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(request);
    timestream::RTCProc rtcproc;
    rtcproc.run_downsample = false;
    rtcproc.downsampler.factor = 0;

    const auto report =
        citlali::pipeline::begin_raw_timestream_observation_shadow(
            plan, citlali::config::ReductionType::science,
            122.0703125, 122.0703125, rtcproc);

    EXPECT_TRUE(report.exact) << report.diagnostic();
    ASSERT_TRUE(plan.observation.has_value());
    EXPECT_EQ(*plan.observation->downsample_factor, 1);
    EXPECT_DOUBLE_EQ(
        *plan.observation->effective_sample_rate_hz, 122.0703125);
    EXPECT_FALSE(plan.observation->filter_edge_guard_parity_deferred);
}

TEST(config_scaffold, reports_raw_observation_shadow_divergence) {
    citlali::config::RawTimeChunkConfig request;
    request.downsample.enabled = true;
    request.downsample.factor = 2;
    request.filter.edge_guard.enabled = true;
    request.filter.edge_guard.min_samples = 4;
    request.despike.enabled = true;
    request.despike.source_protection.enabled = true;
    request.extinction_correction_enabled = true;

    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(request);
    timestream::RTCProc rtcproc;
    rtcproc.run_downsample = true;
    rtcproc.downsampler.factor = 3;
    rtcproc.filter_edge_guard.guard_samples = 5;
    rtcproc.filter_edge_guard.context_samples = 6;
    rtcproc.despiker.source_protection_enabled = false;

    const auto begin =
        citlali::pipeline::begin_raw_timestream_observation_shadow(
            plan, citlali::config::ReductionType::pointing,
            100.0, 49.0, rtcproc);

    EXPECT_FALSE(begin.exact);
    EXPECT_NE(begin.diagnostic().find("effective_sample_rate_hz"),
              std::string::npos);
    EXPECT_NE(begin.diagnostic().find("downsample.factor"),
              std::string::npos);
    EXPECT_NE(begin.diagnostic().find("source_protection.active"),
              std::string::npos);
    EXPECT_NE(begin.diagnostic().find("filter_edge_guard.guard_samples"),
              std::string::npos);
    EXPECT_NE(begin.diagnostic().find("filter_edge_guard.context_samples"),
              std::string::npos);

    const auto extinction =
        citlali::pipeline::complete_raw_timestream_extinction_shadow(
            plan, 0.1, rtcproc.calibration.tx_225_zenith,
            false, "N/A");
    EXPECT_FALSE(extinction.exact);
    EXPECT_NE(extinction.diagnostic().find("extinction.active"),
              std::string::npos);
    EXPECT_NE(extinction.diagnostic().find("extinction.model"),
              std::string::npos);
}

TEST(config_scaffold,
     defers_frequency_derived_downsample_edge_guard_shadow_parity) {
    citlali::config::RawTimeChunkConfig request;
    request.downsample.enabled = true;
    request.downsample.factor = 0;
    request.downsample.downsampled_freq_Hz = 40.0;
    request.filter.edge_guard.enabled = true;
    request.filter.edge_guard.apply_downsample = true;

    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(request);
    timestream::RTCProc rtcproc;
    rtcproc.run_downsample = true;
    rtcproc.downsampler.factor = 3;
    rtcproc.filter_edge_guard.guard_samples = 0;
    rtcproc.filter_edge_guard.context_samples = 0;

    const auto report =
        citlali::pipeline::begin_raw_timestream_observation_shadow(
            plan, citlali::config::ReductionType::science,
            120.0, 40.0, rtcproc);

    EXPECT_TRUE(report.exact) << report.diagnostic();
    EXPECT_TRUE(report.edge_guard_deferred);
    ASSERT_TRUE(plan.observation.has_value());
    EXPECT_EQ(*plan.observation->downsample_factor, 3);
    EXPECT_EQ(*plan.observation->filter_edge_guard_samples, 2);
    EXPECT_TRUE(plan.observation->filter_edge_guard_parity_deferred);
    EXPECT_EQ(plan.requested.downsample.factor, 0);
}

TEST(config_scaffold, serializes_versioned_raw_timestream_provenance) {
    citlali::config::RawTimeChunkConfig request;
    request.downsample.enabled = true;
    request.downsample.factor = 0;
    request.downsample.downsampled_freq_Hz = 40.0;
    request.despike.enabled = true;
    request.despike.source_protection.enabled = true;
    request.extinction_correction_enabled = true;

    citlali::pipeline::RawTimestreamExecutionPlan plan;
    citlali::config::InterfaceSyncOffsetConfig interface_sync_request;
    interface_sync_request.toltec_offset_sec[0] = 0.25;
    interface_sync_request.hwpr_offset_sec = -0.5;
    plan.reset_from_request(request, interface_sync_request);
    plan.effective.downsample.factor = 3;
    auto &observation = plan.begin_observation();
    observation.native_sample_rate_hz = 120.0;
    observation.effective_sample_rate_hz = 40.0;
    observation.downsample_factor = 3;
    observation.filter_edge_guard_samples = 2;
    observation.filter_outer_context_samples = 4;
    observation.filter_edge_guard_parity_deferred = true;
    observation.source_protection_active = false;
    observation.extinction_active = true;
    observation.extinction_model = "am_q50";
    plan.realized.execution_completed = true;
    plan.realized.completed_scan_count = 12;
    plan.realized.flagged_sample_count = 34;
    plan.realized.dynamic_notch_count = 2;
    plan.realized.required_timestream_write_count = 24;

    const auto node =
        citlali::pipeline::raw_timestream_provenance_node(plan);

    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-raw-timestream-provenance-v2");
    EXPECT_TRUE(node["initialized"].as<bool>());
    EXPECT_EQ(node["requested"]["downsample"]["factor"].as<int>(), 0);
    EXPECT_EQ(
        node["requested"]["interface_sync_offset"]["unit"]
            .as<std::string>(),
        "s");
    EXPECT_DOUBLE_EQ(
        node["requested"]["interface_sync_offset"]["offsets"]
            ["toltec0"]
                .as<double>(),
        0.25);
    EXPECT_DOUBLE_EQ(
        node["effective"]["config"]["interface_sync_offset"]
            ["offsets"]["hwpr"]
                .as<double>(),
        -0.5);
    EXPECT_EQ(
        node["effective"]["config"]["downsample"]["factor"].as<int>(),
        3);
    EXPECT_EQ(node["effective"]["resolutions"]["downsampling"]["kind"]
                  .as<std::string>(),
              "target_frequency");
    EXPECT_TRUE(node["observation"]["available"].as<bool>());
    EXPECT_DOUBLE_EQ(
        node["observation"]["value"]["effective_sample_rate_hz"]
            ["value"]
                .as<double>(),
        40.0);
    EXPECT_TRUE(node["observation"]["value"]
                    ["filter_edge_guard_parity_deferred"]
                        .as<bool>());
    EXPECT_EQ(node["observation"]["value"]["extinction_model"]["value"]
                  .as<std::string>(),
              "am_q50");
    EXPECT_TRUE(node["realized"]["execution_completed"].as<bool>());
    EXPECT_EQ(node["realized"]["completed_scan_count"]["value"]
                  .as<std::size_t>(),
              12U);
    EXPECT_EQ(node["realized"]["required_timestream_write_count"]["value"]
                  .as<std::size_t>(),
              24U);
}

TEST(config_scaffold, serializes_unavailable_raw_observation_explicitly) {
    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(citlali::config::RawTimeChunkConfig{});

    const auto node =
        citlali::pipeline::raw_timestream_provenance_node(plan);

    EXPECT_FALSE(node["observation"]["available"].as<bool>());
    EXPECT_FALSE(node["observation"]["value"].IsDefined());
    EXPECT_FALSE(node["realized"]["execution_completed"].as<bool>());
    EXPECT_FALSE(
        node["realized"]["completed_scan_count"]["available"].as<bool>());
}

TEST(config_scaffold, atomically_writes_raw_timestream_provenance) {
    const auto output_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_raw_timestream_provenance_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(citlali::config::RawTimeChunkConfig{});
    plan.begin_observation();
    citlali::pipeline::complete_raw_timestream_observation(plan, 0, 0);

    citlali::pipeline::write_raw_timestream_provenance_file(
        output_dir, plan);

    const auto output_path =
        citlali::pipeline::raw_timestream_provenance_path(output_dir);
    EXPECT_TRUE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    const auto stored = YAML::LoadFile(output_path.string());
    EXPECT_EQ(stored["schema_version"].as<std::string>(),
              "citlali-raw-timestream-provenance-v2");
    EXPECT_TRUE(stored["initialized"].as<bool>());
    std::filesystem::remove_all(output_dir);
}

TEST(config_scaffold, raw_timestream_provenance_failure_propagates) {
    const auto missing_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_missing_raw_provenance_dir" / "nested";
    std::filesystem::remove_all(missing_dir.parent_path());
    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(citlali::config::RawTimeChunkConfig{});
    plan.begin_observation();
    citlali::pipeline::complete_raw_timestream_observation(plan, 0, 0);

    EXPECT_THROW(
        citlali::pipeline::write_raw_timestream_provenance_file(
            missing_dir, plan),
        std::ios_base::failure);
    EXPECT_FALSE(std::filesystem::exists(
        citlali::pipeline::raw_timestream_provenance_path(missing_dir)));
    EXPECT_FALSE(std::filesystem::exists(
        citlali::pipeline::raw_timestream_provenance_path(missing_dir)
            .string() +
        ".tmp"));

    EXPECT_THROW(
        citlali::pipeline::write_raw_timestream_provenance_file(
            missing_dir,
            citlali::pipeline::RawTimestreamExecutionPlan{}),
        std::logic_error);
}

TEST(config_scaffold, rejects_incomplete_raw_timestream_provenance) {
    const auto output_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_incomplete_raw_timestream_provenance_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);

    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(citlali::config::RawTimeChunkConfig{});
    EXPECT_THROW(
        citlali::pipeline::write_raw_timestream_provenance_file(
            output_dir, plan),
        std::logic_error);

    plan.begin_observation();
    EXPECT_THROW(
        citlali::pipeline::write_raw_timestream_provenance_file(
            output_dir, plan),
        std::logic_error);

    plan.realized.execution_completed = true;
    EXPECT_THROW(
        citlali::pipeline::write_raw_timestream_provenance_file(
            output_dir, plan),
        std::logic_error);
    EXPECT_FALSE(std::filesystem::exists(
        citlali::pipeline::raw_timestream_provenance_path(output_dir)));
    std::filesystem::remove_all(output_dir);
}

TEST(config_scaffold, completes_raw_timestream_realized_state_explicitly) {
    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(citlali::config::RawTimeChunkConfig{});

    EXPECT_THROW(
        citlali::pipeline::complete_raw_timestream_observation(
            plan, 4, 12),
        std::logic_error);

    plan.begin_observation();
    citlali::pipeline::complete_raw_timestream_observation(
        plan, 4, 12);

    EXPECT_TRUE(plan.realized.execution_completed);
    EXPECT_EQ(plan.realized.completed_scan_count, 4U);
    EXPECT_EQ(plan.realized.required_timestream_write_count, 12U);
    EXPECT_FALSE(plan.realized.flagged_sample_count.has_value());
    EXPECT_FALSE(plan.realized.dynamic_notch_count.has_value());

    plan.begin_observation();
    EXPECT_FALSE(plan.realized.execution_completed);
    EXPECT_FALSE(plan.realized.completed_scan_count.has_value());
    EXPECT_FALSE(
        plan.realized.required_timestream_write_count.has_value());
}

TEST(config_scaffold, counts_required_raw_timestream_writes_by_mode) {
    const citlali::pipeline::TimestreamOutputExpectations standard{
        2, 3, 4, 4};
    const citlali::pipeline::TimestreamOutputExpectations beammap{
        2, 0, 4, 0};

    EXPECT_EQ(
        citlali::pipeline::raw_required_timestream_write_count(standard),
        13U);
    EXPECT_EQ(
        citlali::pipeline::raw_required_timestream_write_count(beammap),
        6U);
    EXPECT_THROW(
        citlali::pipeline::raw_required_timestream_write_count(
            {-1, 0, 0, 0}),
        std::logic_error);
}

TEST(config_scaffold,
     publishes_sequential_raw_observation_provenance_without_state_leakage) {
    const auto root =
        std::filesystem::path(testing::TempDir()) /
        "citlali_raw_observation_provenance_lifecycle_test";
    std::filesystem::remove_all(root);
    const auto first_dir = root / "000101";
    const auto second_dir = root / "000102";
    std::filesystem::create_directories(first_dir);
    std::filesystem::create_directories(second_dir);

    FakeRawProvenanceEngine engine;
    engine.typed_config.timestream.output.type =
        citlali::config::TodOutputType::both;
    engine.output_paths.tod_filename["rtc"] = "rtc.nc";
    engine.output_paths.rtcdiag_filename = "rtcdiag.nc";
    engine.output_paths.ptcdiag_filename = "ptcdiag.nc";
    engine.tod_outputs.n_rtc_output_scans = 2;
    engine.tod_outputs.n_ptc_output_scans = 3;
    engine.telescope.scan_indices.resize(2, 4);
    engine.raw_timestream_plan.reset_from_request(
        citlali::config::RawTimeChunkConfig{});
    engine.raw_timestream_plan.begin_observation()
        .native_sample_rate_hz = 100.0;
    engine.output_paths.obsnum_dir_name = first_dir.string();

    const auto first_published =
        citlali::pipeline::publish_completed_raw_timestream_provenance<false>(
            engine);

    const auto first_path =
        citlali::pipeline::raw_timestream_provenance_path(first_dir);
    ASSERT_TRUE(first_published.has_value());
    EXPECT_EQ(*first_published, first_path);
    ASSERT_TRUE(std::filesystem::exists(first_path));
    const auto first = YAML::LoadFile(first_path.string());
    EXPECT_TRUE(first["realized"]["execution_completed"].as<bool>());
    EXPECT_EQ(first["realized"]["completed_scan_count"]["value"]
                  .as<std::size_t>(),
              4U);
    EXPECT_EQ(first["realized"]["required_timestream_write_count"]
                   ["value"]
                       .as<std::size_t>(),
              13U);
    EXPECT_FALSE(first["realized"]["flagged_sample_count"]["available"]
                     .as<bool>());

    engine.raw_timestream_plan.begin_observation()
        .native_sample_rate_hz = 120.0;
    engine.tod_outputs.n_rtc_output_scans = 1;
    engine.tod_outputs.n_ptc_output_scans = 1;
    engine.telescope.scan_indices.resize(2, 3);
    engine.output_paths.obsnum_dir_name = second_dir.string();

    const auto second_published =
        citlali::pipeline::publish_completed_raw_timestream_provenance<false>(
            engine);

    const auto second_path =
        citlali::pipeline::raw_timestream_provenance_path(second_dir);
    ASSERT_TRUE(second_published.has_value());
    EXPECT_EQ(*second_published, second_path);
    ASSERT_TRUE(std::filesystem::exists(second_path));
    const auto second = YAML::LoadFile(second_path.string());
    EXPECT_DOUBLE_EQ(
        second["observation"]["value"]["native_sample_rate_hz"]["value"]
            .as<double>(),
        120.0);
    EXPECT_EQ(second["realized"]["completed_scan_count"]["value"]
                  .as<std::size_t>(),
              3U);
    EXPECT_EQ(second["realized"]["required_timestream_write_count"]
                   ["value"]
                       .as<std::size_t>(),
              8U);

    const auto first_after_second = YAML::LoadFile(first_path.string());
    EXPECT_DOUBLE_EQ(
        first_after_second["observation"]["value"]
                          ["native_sample_rate_hz"]["value"]
                              .as<double>(),
        100.0);
    std::filesystem::remove_all(root);
}

TEST(config_scaffold, raw_observation_provenance_write_failure_propagates) {
    const auto missing_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_missing_raw_observation_provenance" / "nested";
    std::filesystem::remove_all(missing_dir.parent_path());

    FakeRawProvenanceEngine engine;
    engine.raw_timestream_plan.reset_from_request(
        citlali::config::RawTimeChunkConfig{});
    engine.raw_timestream_plan.begin_observation();
    engine.telescope.scan_indices.resize(2, 1);
    engine.output_paths.obsnum_dir_name = missing_dir.string();

    EXPECT_THROW(
        citlali::pipeline::publish_completed_raw_timestream_provenance<false>(
            engine),
        std::ios_base::failure);
    EXPECT_FALSE(std::filesystem::exists(
        citlali::pipeline::raw_timestream_provenance_path(missing_dir)));
    EXPECT_FALSE(std::filesystem::exists(
        citlali::pipeline::raw_timestream_provenance_path(missing_dir)
            .string() +
        ".tmp"));
}

TEST(config_scaffold, rejects_raw_observation_before_plan_initialization) {
    citlali::pipeline::RawTimestreamExecutionPlan plan;
    EXPECT_THROW(plan.begin_observation(), std::logic_error);
}

TEST(config_scaffold, adapts_complete_raw_request_to_legacy_rtc_one_way) {
    citlali::config::RawTimeChunkConfig request;
    request.kernel.enabled = true;
    request.kernel.filepath = "kernel.fits";
    request.kernel.type = "fits";
    request.kernel.fwhm_arcsec = 7.5;
    request.kernel.image_ext_names = {"a1100", "a1400", "a2000"};
    request.filter.enabled = true;
    request.filter.a_gibbs = 43.0;
    request.filter.freq_low_Hz = 0.2;
    request.filter.freq_high_Hz = 18.0;
    request.filter.n_terms = 48;
    request.filter.notch.enabled = true;
    request.filter.notch.zero_phase = true;
    request.filter.notch.freqs_Hz = {10.0, 20.0};
    request.filter.notch.delta_f_Hz = {0.5, 1.0};
    request.filter.edge_guard.enabled = true;
    request.filter.edge_guard.mode =
        citlali::config::RawTimeChunkFilterEdgeGuardMode::flag;
    request.filter.edge_guard.combine =
        citlali::config::RawTimeChunkFilterEdgeGuardCombine::max;
    request.filter.edge_guard.min_samples = 3;
    request.filter.edge_guard.extra_samples = 5;
    request.filter.edge_guard.max_samples = 96;
    request.filter.edge_guard.iir_settle_attenuation = 0.02;
    request.filter.edge_guard.apply_notch = false;
    request.iir_filter.enabled = true;
    request.iir_filter.freq_Hz = 0.25;
    request.iir_filter.order = 3;
    request.iir_filter.zero_phase = true;
    request.downsample.enabled = true;
    request.downsample.factor = 4;
    request.downsample.downsampled_freq_Hz = 30.0;
    request.despike.enabled = true;
    request.despike.min_spike_sigma = 9.0;
    request.despike.time_constant_sec = 0.02;
    request.despike.window_size = 48.0;
    request.despike.legacy_enabled = false;
    request.despike.source_protection.enabled = true;
    request.despike.source_protection.radius_arcsec = 27.0;
    request.despike.local_residual.enabled = true;
    request.despike.local_residual.window_sec = 0.3;
    request.despike.local_residual.compact_raw_gate.max_width_sec = 0.2;
    request.flagging.delta_f_min_Hz = 70000.0;
    request.flagging.lower_tod_inv_var_factor = 0.2;
    request.flagging.upper_tod_inv_var_factor = 4.5;
    request.flagging.network_step_mask.enabled = true;
    request.flagging.network_step_mask.min_det_used = 24;
    request.flagging.impulsive_capture.enabled = true;
    request.flagging.impulsive_capture.max_events_per_network = 5;
    request.flagging.impulsive_coincidence.enabled = true;
    request.flagging.impulsive_coincidence.min_networks_aligned = 4;
    request.flagging.impulsive_coincidence.high_score_override_thresh = 8.0;
    request.altaz_destripe.enabled = true;
    request.altaz_destripe.grouping = "array";
    request.altaz_destripe.fit_time_trend = false;
    request.altaz_destripe.min_samples = 80;
    request.line_audit.enabled = true;
    request.line_audit.line_min_hz = 2.0;
    request.line_audit.post_filter_enabled = true;
    request.line_audit.post_filter_apply_iterations = 2;
    request.line_audit.fixed_notch_enabled = true;
    request.line_audit.fixed_notch_freqs_hz = {12.0, 24.0};
    request.line_audit.fixed_notch_widths_hz = {0.2, 0.3};
    request.line_audit.apply_shared_notches = true;
    request.line_audit.detector_notch_context_samples = 32;
    request.flux_calibration_enabled = true;
    request.extinction_correction_enabled = true;

    timestream::RTCProc rtcproc;
    citlali::pipeline::adapt_raw_timestream_config_one_way(
        request, rtcproc, 1.0, 1.0);

    EXPECT_TRUE(rtcproc.run_kernel);
    EXPECT_EQ(rtcproc.kernel.filepath, "kernel.fits");
    EXPECT_EQ(rtcproc.kernel.img_ext_names, request.kernel.image_ext_names);
    EXPECT_TRUE(rtcproc.run_tod_filter);
    EXPECT_EQ(rtcproc.filter.n_terms, 48);
    EXPECT_TRUE(rtcproc.run_tod_notch);
    EXPECT_EQ(rtcproc.filter.qs, (std::vector<double>{20.0, 20.0}));
    EXPECT_TRUE(rtcproc.run_tod_iir_highpass);
    EXPECT_DOUBLE_EQ(rtcproc.filter.iir_highpass_freq_Hz, 0.25);
    EXPECT_TRUE(rtcproc.run_downsample);
    EXPECT_EQ(rtcproc.downsampler.factor, 4);
    EXPECT_TRUE(rtcproc.run_despike);
    EXPECT_DOUBLE_EQ(rtcproc.despiker.min_spike_sigma, 9.0);
    EXPECT_TRUE(rtcproc.network_step_mask.enabled);
    EXPECT_TRUE(rtcproc.impulsive_capture.enabled);
    EXPECT_TRUE(rtcproc.impulsive_coincidence.enabled);
    EXPECT_TRUE(rtcproc.altaz_destripe.enabled);
    EXPECT_TRUE(rtcproc.line_audit.enabled);
    EXPECT_TRUE(rtcproc.run_calibrate);
    EXPECT_TRUE(rtcproc.run_extinction);
    EXPECT_FALSE(rtcproc.despiker.source_protection_enabled);
    EXPECT_DOUBLE_EQ(rtcproc.kernel.sigma_rad, 7.5);
}

TEST(config_scaffold, keeps_disabled_raw_request_values_out_of_rtc_sentinels) {
    citlali::config::RawTimeChunkConfig request;
    request.filter.enabled = false;
    request.filter.n_terms = 73;
    request.filter.notch.enabled = true;
    request.filter.notch.freqs_Hz = {15.0};
    request.filter.notch.delta_f_Hz = {0.25};
    request.iir_filter.enabled = false;
    request.iir_filter.freq_Hz = 0.4;
    request.iir_filter.order = 5;
    request.iir_filter.zero_phase = true;
    request.despike.enabled = false;
    request.despike.window_size = 55.0;
    request.despike.source_protection.enabled = false;
    request.despike.source_protection.radius_arcsec = 37.0;
    request.kernel.enabled = true;
    request.kernel.type = "gaussian";
    request.kernel.image_ext_names = {"ignored-request-value"};

    timestream::RTCProc rtcproc;
    citlali::pipeline::adapt_raw_timestream_config_one_way(
        request, rtcproc, 1.0, 1.0);

    EXPECT_FALSE(rtcproc.run_tod_filter);
    EXPECT_EQ(rtcproc.filter.n_terms, 0);
    EXPECT_FALSE(rtcproc.run_tod_notch);
    EXPECT_TRUE(rtcproc.filter.qs.empty());
    EXPECT_FALSE(rtcproc.run_tod_iir_highpass);
    EXPECT_DOUBLE_EQ(rtcproc.filter.iir_highpass_freq_Hz, 0.0);
    EXPECT_EQ(rtcproc.filter.iir_highpass_order, 1);
    EXPECT_FALSE(rtcproc.filter.iir_highpass_zero_phase);
    EXPECT_FALSE(rtcproc.run_despike);
    EXPECT_DOUBLE_EQ(rtcproc.despiker.window_size, 55.0);
    EXPECT_TRUE(rtcproc.despike_source_protection_config_enabled);
    EXPECT_DOUBLE_EQ(
        rtcproc.despiker.source_protection_radius_arcsec, 20.0);
    EXPECT_TRUE(rtcproc.kernel.img_ext_names.empty());
    EXPECT_EQ(request.filter.n_terms, 73);
    EXPECT_DOUBLE_EQ(request.iir_filter.freq_Hz, 0.4);
    EXPECT_FALSE(request.despike.source_protection.enabled);
    EXPECT_DOUBLE_EQ(
        request.despike.source_protection.radius_arcsec, 37.0);
    EXPECT_EQ(request.kernel.image_ext_names.size(), 1U);
}

TEST(config_scaffold, expands_legacy_line_audit_width_only_in_rtc_target) {
    citlali::config::RawTimeChunkConfig request;
    request.line_audit.fixed_notch_enabled = true;
    request.line_audit.fixed_notch_freqs_hz = {12.0, 24.0};
    request.line_audit.fixed_notch_widths_hz = {0.3};

    timestream::RTCProc rtcproc;
    citlali::pipeline::adapt_raw_timestream_config_one_way(
        request, rtcproc, 1.0, 1.0);

    EXPECT_EQ(request.line_audit.fixed_notch_widths_hz,
              (std::vector<double>{0.3}));
    EXPECT_EQ(rtcproc.line_audit.fixed_notch_widths_hz,
              (std::vector<double>{0.3, 0.3}));
}

TEST(config_scaffold, overlays_raw_observation_state_without_mutating_plan) {
    citlali::config::RawTimeChunkConfig request;
    request.downsample.factor = 0;
    request.despike.source_protection.enabled = true;
    request.extinction_correction_enabled = true;
    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(request);
    auto &observation = plan.begin_observation();
    observation.native_sample_rate_hz = 488.0;
    observation.effective_sample_rate_hz = 122.0;
    observation.downsample_factor = 4;
    observation.filter_edge_guard_samples = 19;
    observation.filter_outer_context_samples = 23;
    observation.source_protection_active = true;
    observation.extinction_active = false;
    observation.extinction_model = "am_q25";

    timestream::RTCProc rtcproc;
    citlali::pipeline::adapt_raw_timestream_config_one_way(
        plan.effective, rtcproc, 1.0, 1.0);
    citlali::pipeline::adapt_raw_timestream_observation_state_one_way(
        *plan.observation, rtcproc);

    EXPECT_DOUBLE_EQ(rtcproc.despiker.fsmp, 488.0);
    EXPECT_EQ(rtcproc.downsampler.factor, 4);
    EXPECT_EQ(rtcproc.filter_edge_guard.guard_samples, 19);
    EXPECT_EQ(rtcproc.filter_edge_guard.context_samples, 23);
    EXPECT_TRUE(rtcproc.despiker.source_protection_enabled);
    EXPECT_FALSE(rtcproc.run_extinction);
    EXPECT_EQ(rtcproc.calibration.extinction_model, "am_q25");
    EXPECT_EQ(plan.requested.downsample.factor, 0);
    EXPECT_EQ(plan.effective.downsample.factor, 0);

    auto &next_observation = plan.begin_observation();
    ASSERT_FALSE(next_observation.native_sample_rate_hz.has_value());
    ASSERT_FALSE(next_observation.source_protection_active.has_value());
}

TEST(config_scaffold, resolves_raw_sample_rate_without_mutating_request) {
    citlali::config::RawTimeChunkConfig request;
    request.filter.enabled = true;
    request.filter.freq_high_Hz = 10.0;
    request.downsample.enabled = true;
    request.downsample.factor = 0;
    request.downsample.downsampled_freq_Hz = 30.0;

    const auto resolution =
        citlali::pipeline::resolve_raw_sample_rate(request, 100.0);

    ASSERT_TRUE(resolution.valid());
    EXPECT_DOUBLE_EQ(resolution.native_sample_rate_hz, 100.0);
    EXPECT_EQ(resolution.downsample_factor, 3);
    EXPECT_DOUBLE_EQ(
        resolution.effective_sample_rate_hz, 100.0 / 3.0);
    EXPECT_DOUBLE_EQ(
        resolution.downsample_nyquist_hz, 100.0 / 6.0);
    EXPECT_EQ(request.downsample.factor, 0);
}

TEST(config_scaffold, classifies_raw_sample_rate_resolution_failures) {
    using Error = citlali::pipeline::RawSampleRateResolutionError;
    citlali::config::RawTimeChunkConfig request;
    request.filter.enabled = true;
    request.filter.freq_high_Hz = 10.0;
    request.downsample.enabled = true;

    EXPECT_EQ(citlali::pipeline::resolve_raw_sample_rate(request, 0.0).error,
              Error::invalid_native_sample_rate);

    request.downsample.factor = 0;
    request.downsample.downsampled_freq_Hz = 0.0;
    EXPECT_EQ(
        citlali::pipeline::resolve_raw_sample_rate(request, 100.0).error,
        Error::invalid_target_frequency);

    request.downsample.downsampled_freq_Hz = 101.0;
    EXPECT_EQ(
        citlali::pipeline::resolve_raw_sample_rate(request, 100.0).error,
        Error::target_frequency_above_native);

    request.downsample.downsampled_freq_Hz =
        std::numeric_limits<double>::min();
    EXPECT_EQ(
        citlali::pipeline::resolve_raw_sample_rate(request, 100.0).error,
        Error::invalid_downsample_factor);

    request.downsample.factor = 4;
    request.filter.freq_high_Hz = 20.0;
    EXPECT_EQ(
        citlali::pipeline::resolve_raw_sample_rate(request, 100.0).error,
        Error::antialias_filter_above_nyquist);
}

TEST(config_scaffold, raw_filter_edge_guard_matches_legacy_rtc_resolution) {
    citlali::config::RawTimeChunkConfig request;
    request.filter.enabled = true;
    request.filter.freq_high_Hz = 10.0;
    request.filter.n_terms = 32;
    request.filter.notch.enabled = true;
    request.filter.notch.freqs_Hz = {8.0, 12.0};
    request.filter.notch.delta_f_Hz = {0.25};
    request.filter.edge_guard.enabled = true;
    request.filter.edge_guard.combine =
        citlali::config::RawTimeChunkFilterEdgeGuardCombine::sum;
    request.filter.edge_guard.min_samples = 4;
    request.filter.edge_guard.extra_samples = 3;
    request.filter.edge_guard.max_samples = 100000;
    request.iir_filter.enabled = true;
    request.iir_filter.freq_Hz = 0.2;
    request.iir_filter.order = 2;
    request.iir_filter.zero_phase = true;
    request.downsample.enabled = true;
    request.downsample.factor = 4;
    request.line_audit.enabled = true;
    request.line_audit.pre_filter_enabled = true;
    request.line_audit.fixed_notch_enabled = true;
    request.line_audit.fixed_notch_freqs_hz = {5.0, 15.0, 60.0};
    request.line_audit.fixed_notch_widths_hz = {0.3};
    request.line_audit.apply_shared_notches = true;
    request.line_audit.apply_max_notches = 2;
    request.line_audit.apply_min_width_hz = 0.25;

    const auto sample_rate =
        citlali::pipeline::resolve_raw_sample_rate(request, 100.0);
    ASSERT_TRUE(sample_rate.valid());
    const auto typed = citlali::pipeline::resolve_raw_filter_edge_guard(
        request, sample_rate);

    timestream::RTCProc rtcproc;
    citlali::pipeline::adapt_raw_timestream_config_one_way(
        request, rtcproc, 1.0, 1.0);
    rtcproc.configure_filter_edge_guard(100.0);

    EXPECT_EQ(typed.guard_samples, rtcproc.filter_edge_guard.guard_samples);
    EXPECT_EQ(
        typed.context_samples, rtcproc.filter_edge_guard.context_samples);
    EXPECT_GT(typed.fixed_notch_samples, 0);
    EXPECT_GT(typed.line_audit_fixed_notch_samples, 0);
    EXPECT_GT(typed.line_audit_dynamic_notch_samples, 0);
    EXPECT_GT(typed.iir_highpass_samples, 0);
    EXPECT_EQ(typed.downsample_samples, 3);
}

TEST(config_scaffold, raw_filter_edge_guard_max_policy_matches_legacy_rtc) {
    citlali::config::RawTimeChunkConfig request;
    request.filter.enabled = true;
    request.filter.freq_high_Hz = 8.0;
    request.filter.n_terms = 16;
    request.filter.notch.enabled = true;
    request.filter.notch.freqs_Hz = {6.0};
    request.filter.notch.delta_f_Hz = {0.4};
    request.filter.edge_guard.enabled = true;
    request.filter.edge_guard.combine =
        citlali::config::RawTimeChunkFilterEdgeGuardCombine::max;
    request.filter.edge_guard.extra_samples = 2;

    const auto sample_rate =
        citlali::pipeline::resolve_raw_sample_rate(request, 100.0);
    const auto typed = citlali::pipeline::resolve_raw_filter_edge_guard(
        request, sample_rate);
    timestream::RTCProc rtcproc;
    citlali::pipeline::adapt_raw_timestream_config_one_way(
        request, rtcproc, 1.0, 1.0);
    rtcproc.configure_filter_edge_guard(100.0);

    EXPECT_EQ(typed.guard_samples, rtcproc.filter_edge_guard.guard_samples);
    EXPECT_EQ(
        typed.context_samples, rtcproc.filter_edge_guard.context_samples);
}

TEST(config_scaffold, raw_source_protection_matches_shared_resolution) {
    citlali::config::TimestreamConfig config;
    auto &raw = config.raw_time_chunk.despike;
    raw.enabled = true;
    raw.source_protection.enabled = true;

    for (const auto reduction_type : {
             citlali::config::ReductionType::pointing,
             citlali::config::ReductionType::science}) {
        const auto raw_resolution =
            citlali::pipeline::resolve_raw_source_protection_observation(
                reduction_type, raw);
        const auto shared_resolution =
            citlali::pipeline::resolve_source_protection(
                reduction_type, config);
        EXPECT_EQ(raw_resolution.requested,
                  shared_resolution.raw_activation_requested);
        EXPECT_EQ(raw_resolution.source_aware_reduction,
                  shared_resolution.source_aware_reduction);
        EXPECT_EQ(raw_resolution.active, shared_resolution.raw_active);
    }
}

TEST(config_scaffold, raw_extinction_model_matches_legacy_calibration) {
    timestream::Calibration calibration;
    for (const double tau_225_ghz : {0.0, 0.03, 0.08, 0.2}) {
        const auto resolution =
            citlali::pipeline::resolve_raw_extinction_observation(
                true, tau_225_ghz, calibration.tx_225_zenith);
        calibration.setup(tau_225_ghz);
        EXPECT_TRUE(resolution.requested);
        EXPECT_TRUE(resolution.active);
        EXPECT_EQ(resolution.model, calibration.extinction_model);
    }

    const auto disabled =
        citlali::pipeline::resolve_raw_extinction_observation(
            false, 0.08, calibration.tx_225_zenith);
    EXPECT_FALSE(disabled.requested);
    EXPECT_FALSE(disabled.active);
    EXPECT_EQ(disabled.model, "N/A");
}

TEST(config_scaffold, constructs_complete_raw_observation_state) {
    citlali::config::RawTimeChunkConfig request;
    request.filter.enabled = true;
    request.filter.freq_high_Hz = 10.0;
    request.downsample.enabled = true;
    request.downsample.factor = 4;
    const auto sample_rate =
        citlali::pipeline::resolve_raw_sample_rate(request, 100.0);
    const auto edge_guard =
        citlali::pipeline::resolve_raw_filter_edge_guard(
            request, sample_rate);
    const auto source =
        citlali::pipeline::resolve_raw_source_protection_observation(
            citlali::config::ReductionType::pointing, request.despike);
    timestream::Calibration calibration;
    const auto extinction =
        citlali::pipeline::resolve_raw_extinction_observation(
            true, 0.08, calibration.tx_225_zenith);

    const auto state =
        citlali::pipeline::make_raw_timestream_observation_state(
            sample_rate, edge_guard, source, extinction);

    ASSERT_TRUE(state.native_sample_rate_hz.has_value());
    EXPECT_DOUBLE_EQ(*state.native_sample_rate_hz, 100.0);
    EXPECT_DOUBLE_EQ(*state.effective_sample_rate_hz, 25.0);
    EXPECT_EQ(*state.downsample_factor, 4);
    EXPECT_EQ(*state.filter_edge_guard_samples, edge_guard.guard_samples);
    EXPECT_EQ(
        *state.filter_outer_context_samples, edge_guard.context_samples);
    EXPECT_FALSE(*state.source_protection_active);
    EXPECT_TRUE(*state.extinction_active);
    EXPECT_EQ(*state.extinction_model, extinction.model);
}

TEST(config_scaffold, initializes_typed_raw_execution_authority) {
    citlali::config::RawTimeChunkConfig request;
    request.filter.enabled = true;
    request.filter.freq_high_Hz = 10.0;
    request.filter.n_terms = 32;
    request.filter.edge_guard.enabled = true;
    request.downsample.enabled = true;
    request.downsample.factor = 4;
    request.despike.enabled = true;
    request.despike.min_spike_sigma = 9.0;

    citlali::pipeline::RawTimestreamExecutionPlan plan;
    citlali::config::RawTimeChunkConfig effective;
    effective.filter.n_terms = 7;
    timestream::RTCProc production;
    production.filter.n_terms = 11;
    production.despiker.min_spike_sigma = 3.0;

    citlali::pipeline::initialize_raw_timestream_authority(
        request, {}, plan, effective, production, 100.0, 1.0, 1.0);

    EXPECT_TRUE(plan.initialized);
    EXPECT_EQ(plan.requested.filter.n_terms, 32);
    EXPECT_EQ(plan.effective.filter.n_terms, 32);
    EXPECT_EQ(effective.filter.n_terms, 32);
    EXPECT_EQ(production.filter.n_terms, 32);
    EXPECT_DOUBLE_EQ(production.despiker.min_spike_sigma, 9.0);
    EXPECT_EQ(production.downsampler.factor, 4);
    EXPECT_GT(production.filter_edge_guard.guard_samples, 0);
}

TEST(config_scaffold, raw_authority_preserves_disabled_request_values) {
    citlali::config::RawTimeChunkConfig request;
    request.filter.enabled = false;
    request.filter.n_terms = 73;
    request.iir_filter.enabled = false;
    request.iir_filter.freq_Hz = 0.4;
    request.downsample.enabled = false;
    request.downsample.factor = 1;

    citlali::pipeline::RawTimestreamExecutionPlan plan;
    citlali::config::RawTimeChunkConfig effective;
    timestream::RTCProc production;
    citlali::pipeline::initialize_raw_timestream_authority(
        request, {}, plan, effective, production, 100.0, 1.0, 1.0);

    EXPECT_FALSE(production.run_tod_filter);
    EXPECT_EQ(production.filter.n_terms, 0);
    EXPECT_FALSE(production.run_tod_iir_highpass);
    EXPECT_DOUBLE_EQ(production.filter.iir_highpass_freq_Hz, 0.0);
    EXPECT_FALSE(production.run_downsample);
    EXPECT_EQ(plan.requested.filter.n_terms, 73);
    EXPECT_EQ(plan.effective.filter.n_terms, 73);
    EXPECT_EQ(effective.filter.n_terms, 73);
    EXPECT_EQ(effective.downsample.factor, 1);
}

TEST(config_scaffold, projects_effective_raw_iir_output_metadata) {
    citlali::config::RawTimeChunkIirFilterConfig config;
    config.enabled = false;
    config.freq_Hz = 0.4;
    config.order = 5;
    config.zero_phase = true;

    const auto disabled =
        citlali::pipeline::raw_iir_filter_metadata(config);
    EXPECT_FALSE(disabled.enabled);
    EXPECT_DOUBLE_EQ(disabled.frequency_hz, 0.0);
    EXPECT_EQ(disabled.order, 1);
    EXPECT_FALSE(disabled.zero_phase);
    EXPECT_DOUBLE_EQ(config.freq_Hz, 0.4);
    EXPECT_EQ(config.order, 5);
    EXPECT_TRUE(config.zero_phase);

    config.enabled = true;
    const auto enabled =
        citlali::pipeline::raw_iir_filter_metadata(config);
    EXPECT_TRUE(enabled.enabled);
    EXPECT_DOUBLE_EQ(enabled.frequency_hz, 0.4);
    EXPECT_EQ(enabled.order, 5);
    EXPECT_TRUE(enabled.zero_phase);
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

TEST(config_scaffold, routes_processed_accessors_through_effective_plan) {
    FakeEngine engine;
    engine.typed_config.timestream.fruit_loops.max_iters = 7;
    engine.typed_config.timestream.processed_time_chunk.clean
        .mask_radius_arcsec = 18.0;
    citlali::pipeline::reset_processed_timestream_execution_plan(
        engine.processed_timestream_plan,
        engine.typed_config.timestream);
    engine.processed_timestream_plan.effective.fruit_loops.max_iters = 2;
    engine.processed_timestream_plan.effective.processed_time_chunk.clean
        .mask_radius_arcsec = 24.0;

    EXPECT_EQ(citlali::pipeline::fruit_loops_config(engine).max_iters, 2);
    EXPECT_DOUBLE_EQ(
        citlali::pipeline::processed_time_chunk_config(engine)
            .clean.mask_radius_arcsec,
        24.0);
    EXPECT_EQ(engine.typed_config.timestream.fruit_loops.max_iters, 7);
    EXPECT_DOUBLE_EQ(
        engine.typed_config.timestream.processed_time_chunk.clean
            .mask_radius_arcsec,
        18.0);
}

TEST(config_scaffold, records_processed_iteration_result) {
    citlali::config::TimestreamConfig config;
    auto plan =
        citlali::pipeline::make_processed_timestream_execution_plan(config);

    citlali::pipeline::record_processed_timestream_iteration_result(
        plan, 3, true);

    ASSERT_TRUE(plan.realized.fruit_loop_iterations_completed.has_value());
    EXPECT_EQ(*plan.realized.fruit_loop_iterations_completed, 3);
    ASSERT_TRUE(plan.realized.fruit_loops_converged.has_value());
    EXPECT_TRUE(*plan.realized.fruit_loops_converged);
}

TEST(config_scaffold, serializes_versioned_processed_provenance) {
    citlali::config::TimestreamConfig config;
    config.fruit_loops.enabled = true;
    config.fruit_loops.max_iters = 3;
    auto plan =
        citlali::pipeline::make_processed_timestream_execution_plan(config);
    plan.effective.fruit_loops.max_iters = 1;
    plan.effective_resolutions.fruit_loop_iterations =
        citlali::pipeline::resolve_fruit_loop_iteration_policy(
            config.fruit_loops,
            citlali::config::ReductionType::beammap);
    citlali::pipeline::record_processed_timestream_iteration_result(
        plan, 1, false);

    const auto node =
        citlali::pipeline::processed_timestream_provenance_node(plan);

    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-processed-timestream-provenance-v1");
    EXPECT_TRUE(node["initialized"].as<bool>());
    EXPECT_EQ(node["requested"]["fruit_loops"]["max_iters"].as<int>(),
              3);
    EXPECT_EQ(
        node["effective"]["config"]["fruit_loops"]["max_iters"]
            .as<int>(),
        1);
    EXPECT_TRUE(node["effective"]["resolutions"]
                    ["fruit_loop_iterations"]["available"]
                        .as<bool>());
    EXPECT_EQ(node["realized"]["fruit_loop_iterations_completed"]
                       ["value"]
                           .as<int>(),
              1);
}

TEST(config_scaffold, atomically_writes_processed_provenance) {
    const auto output_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_processed_timestream_provenance_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    const auto plan =
        citlali::pipeline::make_processed_timestream_execution_plan(
            citlali::config::TimestreamConfig{});

    citlali::pipeline::write_processed_timestream_provenance_file(
        output_dir, plan);

    const auto output_path =
        citlali::pipeline::processed_timestream_provenance_path(output_dir);
    EXPECT_TRUE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    const auto stored = YAML::LoadFile(output_path.string());
    EXPECT_EQ(stored["schema_version"].as<std::string>(),
              "citlali-processed-timestream-provenance-v1");
    EXPECT_TRUE(stored["initialized"].as<bool>());
    std::filesystem::remove_all(output_dir);
}

TEST(config_scaffold, processed_provenance_write_failure_propagates) {
    const auto missing_dir =
        std::filesystem::path(testing::TempDir()) /
        "citlali_missing_processed_provenance_dir" / "nested";
    std::filesystem::remove_all(missing_dir.parent_path());
    const auto plan =
        citlali::pipeline::make_processed_timestream_execution_plan(
            citlali::config::TimestreamConfig{});

    EXPECT_THROW(
        citlali::pipeline::write_processed_timestream_provenance_file(
            missing_dir, plan),
        std::ios_base::failure);
    EXPECT_FALSE(std::filesystem::exists(
        citlali::pipeline::processed_timestream_provenance_path(
            missing_dir)));
    EXPECT_FALSE(std::filesystem::exists(
        citlali::pipeline::processed_timestream_provenance_path(
            missing_dir)
            .string() +
        ".tmp"));

    EXPECT_THROW(
        citlali::pipeline::write_processed_timestream_provenance_file(
            missing_dir,
            citlali::pipeline::ProcessedTimestreamExecutionPlan{}),
        std::logic_error);
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

TEST(config_scaffold, serializes_processed_resolution_availability) {
    citlali::config::TimestreamConfig config;
    auto plan =
        citlali::pipeline::make_processed_timestream_execution_plan(config);

    const auto empty_effective =
        citlali::pipeline::processed_timestream_effective_resolutions_node(
            plan.effective_resolutions);
    const auto empty_realized =
        citlali::pipeline::processed_timestream_realized_state_node(
            plan.realized);
    EXPECT_FALSE(empty_effective["cleaner_mode"]["available"].as<bool>());
    EXPECT_FALSE(
        empty_effective["fruit_loop_interpolation"]["available"].as<bool>());
    EXPECT_FALSE(
        empty_realized["fruit_loop_iterations_completed"]["available"]
            .as<bool>());

    config.processed_time_chunk.clean.enabled = true;
    config.processed_time_chunk.clean.standard_pca.enabled = true;
    plan.effective_resolutions.cleaner_mode =
        citlali::pipeline::resolve_processed_cleaner_mode(
            config.processed_time_chunk.clean);
    plan.effective_resolutions.weighting_source_mask =
        citlali::pipeline::resolve_processed_weighting_source_mask(
            std::nullopt, 18.0);
    plan.effective_resolutions.weighting_dependencies =
        citlali::pipeline::resolve_processed_weighting(
            config.processed_time_chunk.weighting,
            config.processed_time_chunk.flagging);
    plan.effective_resolutions.fruit_loop_iterations =
        citlali::pipeline::resolve_fruit_loop_iteration_policy(
            config.fruit_loops,
            citlali::config::ReductionType::beammap);
    plan.effective_resolutions.fruit_loop_interpolation =
        citlali::pipeline::resolve_fruit_loop_interpolation(
            config.fruit_loops, citlali::config::MapMethod::jinc);
    plan.realized.source_protection =
        citlali::pipeline::resolve_source_protection(
            citlali::config::ReductionType::pointing, config);
    plan.realized.fruit_loop_iterations_completed = 3;
    plan.realized.fruit_loops_converged = false;

    const auto effective =
        citlali::pipeline::processed_timestream_effective_resolutions_node(
            plan.effective_resolutions);
    const auto realized =
        citlali::pipeline::processed_timestream_realized_state_node(
            plan.realized);
    EXPECT_TRUE(effective["cleaner_mode"]["available"].as<bool>());
    EXPECT_EQ(effective["cleaner_mode"]["value"]["effective"]
                  .as<std::string>(),
              "standard_pca");
    EXPECT_TRUE(
        effective["weighting_source_mask"]["value"]
                 ["inherited_from_cleaning"]
                     .as<bool>());
    EXPECT_TRUE(effective["fruit_loop_iterations"]["value"]
                         ["forced_single_iteration_for_beammap"]
                             .as<bool>());
    EXPECT_EQ(effective["fruit_loop_interpolation"]["value"]["effective"]
                  .as<std::string>(),
              "jinc");
    EXPECT_TRUE(realized["source_protection"]["available"].as<bool>());
    EXPECT_FALSE(
        realized["source_protection"]["value"]["raw_active"].as<bool>());
    EXPECT_EQ(realized["fruit_loop_iterations_completed"]["value"].as<int>(),
              3);
    EXPECT_FALSE(realized["fruit_loops_converged"]["value"].as<bool>());
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
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::prepare_iteration_output_layout_if_needed(
        todproc, config_filepaths, stage_profile, logger);

    EXPECT_EQ(todproc.create_output_dir_calls, 1);
}

TEST(pipeline_output_layout, skips_iteration_output_layout_when_not_saved) {
    FakeCoaddTodProc todproc;
    todproc.engine().iteration.fruit_iter = 1;
    todproc.engine().ptcproc.save_all_iters = false;
    std::vector<std::string> config_filepaths;
    citlali::pipeline::StageProfileCollector stage_profile;
    auto logger = std::make_shared<FakeLogger>();

    citlali::pipeline::prepare_iteration_output_layout_if_needed(
        todproc, config_filepaths, stage_profile, logger);

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
