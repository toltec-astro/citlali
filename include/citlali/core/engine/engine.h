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
#include <citlali/core/engine/control_state.h>
#include <citlali/core/engine/learning.h>
#include <citlali/core/engine/runtime_state.h>
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

#include <citlali/core/engine/component_state.h>
#include <citlali/core/engine/io.h>
#include <citlali/core/engine/kidsproc.h>
#include <citlali/core/engine/todproc.h>

class Engine: public ReductionControls,
              public ReductionComponents,
              public BeammapFluxState,
              public PointingControls,
              public EngineRuntimeState {
public:
    // type for missing/invalid keys
    using key_vec_t = EngineRuntimeState::key_vec_t;

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
    Eigen::Index tod_output_scan_row(
        Eigen::Index, citlali::config::TodOutputStream) const;

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
    Eigen::Index write_maps(fits_io_type &, fits_io_type &, map_buffer_t &, Eigen::Index);

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
