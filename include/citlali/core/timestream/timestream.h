#pragma once

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <optional>
#include <algorithm>
#include <sstream>
#include <limits>
#include <cctype>
#include <cmath>
#include <numeric>
#include <string_view>
#include <vector>

#include <boost/math/special_functions/bessel.hpp>

#include <unsupported/Eigen/CXX11/Tensor>

#include <CCfits/CCfits>

#include <tula/enum.h>
#include <tula/nddata/labelmapper.h>
#include <tula/formatter/enum.h>
#include <tula/formatter/matrix.h>
#include <tula/formatter/utils.h>
#include <tula/logging.h>
#include <kids/core/wcs.h>

#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/pointing.h>
#include <citlali/core/utils/toltec_io.h>

#include <citlali/core/mapmaking/map.h>

namespace timestream {

enum TimestreamFlags {
    Good = 0,
    D21FitsBetter   = 1 << 0,
    D21LargeOffset  = 1 << 1,
    D21NotConverged = 1 << 2,
    D21OutOfRange   = 1 << 3,
    D21QrOutOfRange = 1 << 4,
    LargeOffset     = 1 << 5,
    NotConverged    = 1 << 6,
    OutOfRange      = 1 << 7,
    QrOutOfRange    = 1 << 8,
    LowGain         = 1 << 9,
    APT             = 1 << 10,
    Spike           = 1 << 11,
    Freq            = 1 << 12
};

namespace wcs = kids::wcs;

// clang-format off
TULA_BITFLAG(TCDataKind, int,  0xFFFF,
             RTC               = 1 << 0,
             PTC               = 1 << 1,
             Any               = RTC | PTC
             );
// clang-format on

/// @brief TC data class.
template <TCDataKind kind_ = TCDataKind::Any, typename = void>
struct TCData;

} // namespace timestream

namespace std {

// register TCData as a variant type
// below are mandatory to inherit from variant on gcc
template <timestream::TCDataKind kind>
struct variant_size<timestream::TCData<kind>>
    : variant_size<typename timestream::TCData<kind>::variant_t> {};

template <size_t _Np, auto kind>
struct variant_alternative<_Np, timestream::TCData<kind>>
    : variant_alternative<_Np, typename timestream::TCData<kind>::variant_t> {};

#if defined(__GNUC__) && !defined(__clang__)
#if (__GNUC__ >= 9 && __GNUC__ < 13)
// this is need to allow inherit from std::variant on GCC
namespace __detail {
namespace __variant {

template <typename _Ret, typename _Visitor, auto kind, size_t __first>
struct _Multi_array<_Ret (*)(_Visitor, timestream::TCData<kind>), __first>
    : _Multi_array<_Ret (*)(_Visitor, typename timestream::TCData<kind>::variant_t),
                   __first> {
    static constexpr int __do_cookie = 0;
};
template <typename _Maybe_variant_cookie, auto kind>
struct _Extra_visit_slot_needed<_Maybe_variant_cookie, timestream::TCData<kind>>
    : _Extra_visit_slot_needed<_Maybe_variant_cookie,
                               typename timestream::TCData<kind>::variant_t> {};

template <typename _Maybe_variant_cookie, auto kind>
struct _Extra_visit_slot_needed<_Maybe_variant_cookie, timestream::TCData<kind> &>
    : _Extra_visit_slot_needed<_Maybe_variant_cookie,
                               typename timestream::TCData<kind>::variant_t &> {};
} // namespace __variant
} // namespace __detail
#else
#endif
#endif

} // namespace std

namespace timestream {

namespace internal {

template <typename Derived>
struct TCDataBase;

template <TCDataKind kind_>
struct TCDataBase<TCData<kind_>> {
    static constexpr auto kind() { return kind_; }
    //using meta_t = typename internal::impl_traits<TCData<kind_>>::meta_t;
    //meta_t meta;
};

} // namespace internal

// data status
struct Status {
    bool demodulated = false;
    bool kernel_generated = false;
    bool despiked = false;
    bool tod_filtered = false;
    bool filter_edge_guarded = false;
    int filter_edge_guard_pre_samples = 0;
    int filter_edge_guard_post_samples = 0;
    int filter_edge_guard_flagged_samples = 0;
    double filter_edge_guard_flagged_frac = std::numeric_limits<double>::quiet_NaN();
    bool downsampled = false;
    bool calibrated = false;
    bool extinction_corrected = false;
    bool cleaned = false;
};

// wcs objects
struct DetectorAxis : wcs::Axis<DetectorAxis, wcs::CoordsKind::Column>,
                      wcs::LabeledData<DetectorAxis> {
    DetectorAxis() = default;
    DetectorAxis(Eigen::MatrixXd data_, tula::nddata::LabelMapper<DetectorAxis> row_labels_)
        : data{std::move(data_)}, row_labels{std::move(row_labels_)} {}
    std::string_view name{"detector"};
    Eigen::MatrixXd data;
    tula::nddata::LabelMapper<DetectorAxis> row_labels;
};

struct TimeAxis : wcs::Axis<TimeAxis, wcs::CoordsKind::Row>,
                  wcs::LabeledData<TimeAxis> {
    TimeAxis() = default;
    TimeAxis(Eigen::MatrixXd data_, tula::nddata::LabelMapper<TimeAxis> col_labels_)
        : data{std::move(data_)}, col_labels{std::move(col_labels_)} {}
    std::string_view name{"time"};
    Eigen::MatrixXd data;
    tula::nddata::LabelMapper<TimeAxis> col_labels;
};

struct TimeStreamFrame : wcs::Frame2D<TimeStreamFrame, TimeAxis, DetectorAxis> {
    TimeAxis time_axis;
    DetectorAxis detector_axis;

    // Frame2D impl
    const TimeAxis &row_axis() const { return time_axis; }
    const DetectorAxis &col_axis() const { return detector_axis; }
};

// data objects

/// @brief base class for time stream data
template <typename Derived>
struct TimeStream : internal::TCDataBase<Derived>,
                    tula::nddata::NDData<TimeStream<Derived>> {
    TimeStreamFrame wcs;
    // the timestream is stored in row major for efficient r/w
    template <typename PlainObject>
    struct dataref_t : tula::nddata::NDData<dataref_t<PlainObject>> {
        PlainObject data{nullptr, 0, 0};
    };

    template <typename PlainObject>
    struct data_t : tula::nddata::NDData<data_t<PlainObject>> {
        PlainObject data;
    };

    // time of creation
    std::string creation_time = engine_utils::current_date_time();

    // number of detectors lower than weight limit
    int n_dets_low, n_dets_high;

    // data status struct
    Status status;

    // kernel timestreams
    data_t<Eigen::MatrixXd> kernel;
    // flag timestream
    data_t<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> flags;
    // noise timestreams
    data_t<Eigen::MatrixXi> noise;
    // bitwise flags
    data_t<Eigen::Matrix<TimestreamFlags,Eigen::Dynamic,Eigen::Dynamic>> flags2;
    // current scan indices
    data_t<Eigen::Matrix<Eigen::Index,Eigen::Dynamic,1>> scan_indices;
    // scan index
    data_t<Eigen::Index> index;
    // telescope data for scan
    data_t<std::map<std::string, Eigen::VectorXd>> tel_data;
    // pointing offsets
    data_t<std::map<std::string, Eigen::VectorXd>> pointing_offsets_arcsec;
    // hwpr angle for scan
    data_t<Eigen::VectorXd> hwpr_angle;
    // detector angle
    data_t<Eigen::VectorXd> angle;
    // fcf
    data_t<Eigen::VectorXd> fcf;
    // vectors for mapping apt table onto timestreams
    data_t<Eigen::VectorXI> map_indices;
    // detector pointing
    data_t<std::map<std::string, Eigen::MatrixXd>> pointing;
};

template <typename RefType>
struct TCData<TCDataKind::RTC,RefType>
    : TimeStream<TCData<TCDataKind::RTC>> {
    using Base = TimeStream<TCData<TCDataKind::RTC>>;
    using data_t = std::conditional_t<tula::eigen_utils::is_plain_v<RefType>,Base::data_t<RefType>,
                                      Base::dataref_t<RefType>>;
    // time chunk type
    std::string_view name{"RTC"};
    // data timestreams
    data_t scans;
};

template <typename RefType>
struct TCData<TCDataKind::PTC, RefType>
    : TimeStream<TCData<TCDataKind::PTC>> {
    using Base = TimeStream<TCData<TCDataKind::PTC>>;
    using data_t = std::conditional_t<tula::eigen_utils::is_plain_v<RefType>,Base::data_t<RefType>,
                                      Base::dataref_t<RefType>>;
    // time chunk type
    std::string_view name{"PTC"};
    // data timestreams
    data_t scans;
    // weights for current scan
    Base::data_t<Eigen::VectorXd> weights;
    // eigenvalues for scan
    Base::data_t<std::vector<std::vector<Eigen::VectorXd>>> evals;
    // eigenvectors for scan
    Base::data_t<std::vector<std::vector<Eigen::MatrixXd>>> evecs;
    // medians of good detector weights
    Base::data_t<std::vector<double>> median_weights;
};

/// @brief data class of runtime variant kind.
template <TCDataKind kind_>
struct TCData<kind_, std::enable_if_t<tula::enum_utils::is_compound_v<kind_>>>
    : tula::enum_utils::enum_to_variant_t<kind_, TCData> {
    using Base = tula::enum_utils::enum_to_variant_t<kind_, TCData>;
    using variant_t = tula::enum_utils::enum_to_variant_t<kind_, TCData>;

    const variant_t &variant() const { return *this; }
    static constexpr auto kind() { return kind_; }
};

struct KernelMatrixDiag {
    Eigen::Index n_total = 0;
    Eigen::Index n_sampled = 0;
    Eigen::Index n_finite = 0;
    Eigen::Index n_negative = 0;
    Eigen::Index n_positive = 0;
    double min = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
    double mean = std::numeric_limits<double>::quiet_NaN();
    double rms = std::numeric_limits<double>::quiet_NaN();
    double abs_max = std::numeric_limits<double>::quiet_NaN();
};

template <typename Derived>
KernelMatrixDiag summarize_kernel_matrix(const Eigen::MatrixBase<Derived> &matrix,
                                         Eigen::Index max_samples = 250000) {
    KernelMatrixDiag diag;
    diag.n_total = matrix.size();
    if (diag.n_total <= 0 || matrix.rows() <= 0 || matrix.cols() <= 0) {
        return diag;
    }
    const Eigen::Index stride =
        (max_samples > 0 && diag.n_total > max_samples)
            ? std::max<Eigen::Index>(1, (diag.n_total + max_samples - 1) / max_samples)
            : 1;
    double sum = 0.0;
    double sumsq = 0.0;
    double abs_max = 0.0;
    for (Eigen::Index idx = 0; idx < diag.n_total; idx += stride) {
        const Eigen::Index r = idx % matrix.rows();
        const Eigen::Index c = idx / matrix.rows();
        const double v = static_cast<double>(matrix(r, c));
        ++diag.n_sampled;
        if (!std::isfinite(v)) {
            continue;
        }
        if (diag.n_finite == 0) {
            diag.min = v;
            diag.max = v;
        }
        else {
            diag.min = std::min(diag.min, v);
            diag.max = std::max(diag.max, v);
        }
        if (v < 0.0) {
            ++diag.n_negative;
        }
        else if (v > 0.0) {
            ++diag.n_positive;
        }
        abs_max = std::max(abs_max, std::abs(v));
        sum += v;
        sumsq += v * v;
        ++diag.n_finite;
    }
    if (diag.n_finite > 0) {
        diag.mean = sum / static_cast<double>(diag.n_finite);
        diag.rms = std::sqrt(sumsq / static_cast<double>(diag.n_finite));
        diag.abs_max = abs_max;
    }
    return diag;
}

template <typename Derived>
void log_kernel_matrix_diag(const std::shared_ptr<spdlog::logger> &logger,
                            const std::string &stage,
                            const Eigen::MatrixBase<Derived> &matrix,
                            Eigen::Index scan_index = -1) {
    if (!logger || matrix.size() == 0) {
        return;
    }
    const auto diag = summarize_kernel_matrix(matrix);
    const double negative_frac =
        diag.n_finite > 0
            ? static_cast<double>(diag.n_negative) / static_cast<double>(diag.n_finite)
            : std::numeric_limits<double>::quiet_NaN();
    logger->info(
        "kernel_tod_diag stage='{}' scan={} shape={}x{} sampled={}/{} finite={}/{} neg={} neg_frac={} pos={} min={} max={} mean={} rms={} absmax={}",
        stage,
        scan_index,
        matrix.rows(),
        matrix.cols(),
        diag.n_sampled,
        diag.n_total,
        diag.n_finite,
        diag.n_sampled,
        diag.n_negative,
        negative_frac,
        diag.n_positive,
        diag.min,
        diag.max,
        diag.mean,
        diag.rms,
        diag.abs_max);
}

inline void log_kernel_map_diag(
    const std::shared_ptr<spdlog::logger> &logger,
    const std::string &stage,
    const std::vector<Eigen::MatrixXd> &kernel_maps,
    const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps = nullptr) {
    if (!logger || kernel_maps.empty()) {
        return;
    }
    Eigen::Index n_active = 0;
    Eigen::Index n_center_finite = 0;
    Eigen::Index n_center_negative = 0;
    Eigen::Index n_center_positive = 0;
    Eigen::Index n_center_zero = 0;
    Eigen::Index worst_center_map = -1;
    double center_min = std::numeric_limits<double>::quiet_NaN();
    double center_max = std::numeric_limits<double>::quiet_NaN();
    double center_sum = 0.0;

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(kernel_maps.size()); ++i) {
        if (active_maps != nullptr && (i >= active_maps->size() || !(*active_maps)(i))) {
            continue;
        }
        const auto &map = kernel_maps[static_cast<std::size_t>(i)];
        if (map.size() == 0) {
            continue;
        }
        ++n_active;
        const Eigen::Index center_row =
            std::clamp<Eigen::Index>(static_cast<Eigen::Index>(std::llround((map.rows() - 1) / 2.0)),
                                     0, map.rows() - 1);
        const Eigen::Index center_col =
            std::clamp<Eigen::Index>(static_cast<Eigen::Index>(std::llround((map.cols() - 1) / 2.0)),
                                     0, map.cols() - 1);
        const double center = map(center_row, center_col);
        if (std::isfinite(center)) {
            if (n_center_finite == 0) {
                center_min = center;
                center_max = center;
                worst_center_map = i;
            }
            else {
                if (center < center_min) {
                    center_min = center;
                    worst_center_map = i;
                }
                center_max = std::max(center_max, center);
            }
            if (center < 0.0) {
                ++n_center_negative;
            }
            else if (center > 0.0) {
                ++n_center_positive;
            }
            else {
                ++n_center_zero;
            }
            center_sum += center;
            ++n_center_finite;
        }
    }

    const double center_mean =
        n_center_finite > 0
            ? center_sum / static_cast<double>(n_center_finite)
            : std::numeric_limits<double>::quiet_NaN();
    const double center_negative_frac =
        n_center_finite > 0
            ? static_cast<double>(n_center_negative) / static_cast<double>(n_center_finite)
            : std::numeric_limits<double>::quiet_NaN();

    logger->info(
        "kernel_map_diag stage='{}' maps={} active={} center_finite={} center_neg={} center_neg_frac={} center_pos={} center_zero={} center_min={} center_max={} center_mean={} worst_center_map={}",
        stage,
        kernel_maps.size(),
        n_active,
        n_center_finite,
        n_center_negative,
        center_negative_frac,
        n_center_positive,
        n_center_zero,
        center_min,
        center_max,
        center_mean,
        worst_center_map);
}

// class for tod processing
class TCProc {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // toltec io class for array names
    engine_utils::toltecIO toltec_io;

    // add or subtract gaussian source
    enum SourceType {
        Gaussian = 0,
        NegativeGaussian = 1,
        Airy = 2,
        NegativeAiry = 3,
        Map = 4,
        NegativeMap = 5
    };

    // tod output
    bool run_tod_output, write_evals;
    // compact TOD output mode (float signal, byte flags, no per-detector pointing/kernel vars)
    bool tod_output_mini = false;
    // include the loaded outer scan context in the TOD output instead of only the inner science scan
    bool tod_output_outer = false;
    // minimum context samples per side to load when writing *_outer TOD output
    Eigen::Index tod_output_outer_context_samples = 0;

    // run fruit loops
    bool run_fruit_loops;
    // path for input images
    std::string fruit_loops_path;
    // paths for first set of images
    std::vector<std::string> init_fruit_loops_path;
    // fruit loops type and mode
    std::string fruit_loops_type, fruit_mode;
    // number of fruit loops iterations
    int fruit_loops_iters = 0;
    // signal-to-noise cut for fruit loops algorithm
    double fruit_loops_sig2noise = 0;
    // flux density cut for fruit loops algorithm
    Eigen::VectorXd fruit_loops_flux;
    // fractional and local-noise cuts for adaptive fruit loops source support
    double fruit_loops_peak_fraction_limit = 0.0;
    double fruit_loops_local_snr_floor = 0.0;
    double fruit_loops_local_sigma_inner_radius_arcsec = 10.0;
    double fruit_loops_local_sigma_outer_radius_arcsec = 35.0;
    double fruit_loops_local_sigma_inner_fwhm = 1.5;
    double fruit_loops_local_sigma_outer_fwhm = 4.0;
    double fruit_loops_local_sigma_edge_guard_arcsec = 5.0;
    int fruit_loops_local_sigma_min_pixels = 50;
    double fruit_loops_adaptive_support_radius_arcsec = 12.0;
    double fruit_loops_adaptive_support_radius_fwhm = 1.5;
    Eigen::VectorXd fruit_loops_local_sigma_map;
    Eigen::VectorXi fruit_loops_local_sigma_npix;
    Eigen::VectorXd fruit_loops_amp_ref;
    Eigen::VectorXd fruit_loops_adaptive_threshold;
    Eigen::VectorXd fruit_loops_adaptive_support_radius_rad;
    // preserve a central map region from coverage-cut masking when loading
    // fruit loops templates
    double fruit_loops_center_keep_radius_arcsec = 0.0;
    // interpolation used when projecting fruit loops maps back to the TOD
    std::string fruit_loops_interp_mode = "bilinear";
    // optional override for fruit loops interpolation mode
    // (auto|nearest|bilinear|jinc|trunc)
    std::string fruit_loops_interp_mode_override = "auto";
    // use pre-Mar-2026 center convention (n/2) for map->tod projection
    bool fruit_loops_legacy_center = false;
    // if true, recompute weights after map add-back (pre-Mar-2026 behavior)
    bool fruit_loops_recompute_weights_after_addback = false;
    // current map grouping, used by helpers that need detector pointing
    std::string active_map_grouping = "array";
    // jinc interpolation settings copied from the active mapmaker config
    double fruit_loops_jinc_r_max = 0.0;
    int fruit_loops_jinc_subpixel_n = 1;
    std::map<Eigen::Index, Eigen::VectorXd> fruit_loops_jinc_shape_params;
    std::map<Eigen::Index, Eigen::MatrixXd> fruit_loops_jinc_weights_mat;
    std::map<Eigen::Index, std::vector<Eigen::MatrixXd>> fruit_loops_jinc_weights_mat_subpix;
    // source positions inferred from loaded fruit loops maps, in map-frame radians
    Eigen::VectorXd fruit_loops_source_lat;
    Eigen::VectorXd fruit_loops_source_lon;
    Eigen::VectorXi fruit_loops_source_valid;
    // save all iterations
    bool save_all_iters;

    // map buffer for map to tod approach
    mapmaking::MapBuffer tod_mb;

    // number of weight outlier iterations
    int iter_lim = 0;

    // upper and lower inv var limits for outliers
    double lower_inv_var_factor, upper_inv_var_factor;

    // mask radius in arcseconds
    double mask_radius_arcsec = 0.0;

    struct RemoveBadDetsWindowDiagSummary {
        int n_total_windows = 0;
        int n_valid_windows = 0;
        double valid_window_fraction = std::numeric_limits<double>::quiet_NaN();
        double inv_var_median = std::numeric_limits<double>::quiet_NaN();
        double inv_var_q10 = std::numeric_limits<double>::quiet_NaN();
        double inv_var_q90 = std::numeric_limits<double>::quiet_NaN();
        double flagged_frac_median = std::numeric_limits<double>::quiet_NaN();
        double flagged_frac_max = std::numeric_limits<double>::quiet_NaN();
        double heavily_flagged_window_fraction = std::numeric_limits<double>::quiet_NaN();
    };

    // diagnostic window size for scan-local inverse-variance summaries
    double remove_bad_dets_window_sec = 0.5;
    std::map<Eigen::Index, std::vector<RemoveBadDetsWindowDiagSummary>>
        remove_bad_dets_window_summary_by_scan;

    // create a map buffer from a citlali reduction directory
    template <class calib_t>
    void load_mb(std::string, std::string, calib_t &, const std::string &,
                 const std::string & = "", double = 0.0);
    template <class mb_t, class calib_t>
    void configure_fruit_loops_adaptive_gate(mb_t &, calib_t &, const std::string &, bool = true);
    double fruit_loops_jinc_func(double, double, double, double, double, double);
    void allocate_fruit_loops_jinc_matrix(double);
    double sample_map_bilinear(const Eigen::MatrixXd &, double, double) const;
    double sample_map_jinc(const Eigen::MatrixXd &, Eigen::Index, double, double) const;

    // get limits for a particular grouping
    template <class calib_t>
    auto get_grouping(std::string, calib_t &, int);

    // compute and store pointing of all detectors
    template <TCDataKind tcdata_t, class calib_t>
    void precompute_pointing(TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, std::string, std::string);

    // translate citlali map buffer to timestream and add/subtract from TCData scans
    template <TCProc::SourceType source_type, class mb_t, TCDataKind tcdata_t, class calib_t, typename Derived>
    void map_to_tod(mb_t &, TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &, std::string, std::string);

    // remove detectors with outlier weights
    template <TCDataKind tcdata_t, class calib_t>
    auto remove_bad_dets(TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, std::string);

    // remove detectors with small correlations
    template <TCDataKind tcdata_t, class calib_t, typename Derived>
    auto remove_uncorrelated(TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, std::string);

    // add or subtract gaussian to timestream
    template <SourceType source_type, TCDataKind tcdata_t, typename Derived, typename apt_t>
    void add_gaussian(TCData<tcdata_t, Eigen::MatrixXd> &, Eigen::DenseBase<Derived> &, std::string &,
                      std::string &, apt_t &, double, Eigen::Index, Eigen::Index);

    // resolve the center of the source-masked region in map-frame radians
    template <TCDataKind tcdata_t, class calib_t>
    bool resolve_mask_center_rad(const TCData<tcdata_t, Eigen::MatrixXd> &,
                                 const calib_t &, std::string_view,
                                 Eigen::Index, double &, double &) const;

    // flag a region around the center of the map or a detector-specific source prior
    template <TCDataKind tcdata_t, class calib_t>
    auto mask_region(TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, std::string, std::string, int, int, int);

    // append time chunk params common to rtcs and ptcs
    template <TCDataKind tcdata_t, class calib_t, typename pointing_offset_t>
    void append_base_to_netcdf(netCDF::NcFile &, TCData<tcdata_t, Eigen::MatrixXd> &, std::string,
                               std::string &, pointing_offset_t &, calib_t &, bool apply_det_offsets = false,
                               Eigen::Index scan_row_index = -1, bool output_outer_scan = false);
};

template <class calib_t>
void TCProc::load_mb(std::string filepath, std::string noise_filepath, calib_t &calib,
                     const std::string &expected_map_grouping, const std::string &expected_pixel_axes,
                     double expected_pixel_size_rad) {

    namespace fs = std::filesystem;

    fruit_loops_source_lat.resize(0);
    fruit_loops_source_lon.resize(0);
    fruit_loops_source_valid.resize(0);
    fruit_loops_local_sigma_map.resize(0);
    fruit_loops_local_sigma_npix.resize(0);
    fruit_loops_amp_ref.resize(0);
    fruit_loops_adaptive_threshold.resize(0);

    if (expected_map_grouping.empty()) {
        logger->error("expected map grouping not provided for fruit loops map loading");
        std::exit(EXIT_FAILURE);
    }

    // clear map buffer
    std::vector<Eigen::MatrixXd>().swap(tod_mb.signal);
    std::vector<Eigen::MatrixXd>().swap(tod_mb.weight);
    std::vector<Eigen::MatrixXd>().swap(tod_mb.kernel);
    std::vector<Eigen::Tensor<double,3>>().swap(tod_mb.noise);
    std::vector<std::string>().swap(tod_mb.wcs.cunit);

    tod_mb.median_rms.resize(0);
    tod_mb.n_noise = 0;
    tod_mb.map_grouping = expected_map_grouping;

    // resize wcs params
    tod_mb.wcs.naxis.resize(4,0.);
    tod_mb.wcs.crpix.resize(4,0.);
    tod_mb.wcs.crval.resize(4,0.);
    tod_mb.wcs.cdelt.resize(4,0.);
    tod_mb.wcs.cunit.resize(2, "N/A");

    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
        return s;
    };

    auto split_tokens = [](const std::string &s) {
        std::vector<std::string> tokens;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, '_')) {
            if (!item.empty()) {
                tokens.push_back(item);
            }
        }
        return tokens;
    };

    auto read_key_long = [&](CCfits::ExtHDU &ext, const std::string &key, long &out) {
        try {
            ext.readKey(key, out);
        } catch (const CCfits::Keyword::WrongKeywordValueType &) {
            std::string tmp;
            try {
                ext.readKey(key, tmp);
                out = std::stol(tmp);
            } catch (const CCfits::FitsException &e) {
                logger->error("failed to read {} from fruit loops map header: {}", key, e.message());
                std::exit(EXIT_FAILURE);
            } catch (...) {
                logger->error("invalid value for {} in fruit loops map header", key);
                std::exit(EXIT_FAILURE);
            }
        } catch (const CCfits::FitsException &e) {
            logger->error("failed to read {} from fruit loops map header: {}", key, e.message());
            std::exit(EXIT_FAILURE);
        }
    };

    auto read_key_double = [&](CCfits::ExtHDU &ext, const std::string &key, double &out) {
        try {
            ext.readKey(key, out);
        } catch (const CCfits::Keyword::WrongKeywordValueType &) {
            std::string tmp;
            try {
                ext.readKey(key, tmp);
                out = std::stod(tmp);
            } catch (const CCfits::FitsException &e) {
                logger->error("failed to read {} from fruit loops map header: {}", key, e.message());
                std::exit(EXIT_FAILURE);
            } catch (...) {
                logger->error("invalid value for {} in fruit loops map header", key);
                std::exit(EXIT_FAILURE);
            }
        } catch (const CCfits::FitsException &e) {
            logger->error("failed to read {} from fruit loops map header: {}", key, e.message());
            std::exit(EXIT_FAILURE);
        }
    };

    auto read_key_string = [&](CCfits::ExtHDU &ext, const std::string &key, std::string &out) {
        try {
            ext.readKey(key, out);
        } catch (const CCfits::Keyword::WrongKeywordValueType &) {
            double tmp = 0.0;
            try {
                ext.readKey(key, tmp);
                out = std::to_string(tmp);
            } catch (const CCfits::FitsException &e) {
                logger->error("failed to read {} from fruit loops map header: {}", key, e.message());
                std::exit(EXIT_FAILURE);
            }
        } catch (const CCfits::FitsException &e) {
            logger->error("failed to read {} from fruit loops map header: {}", key, e.message());
            std::exit(EXIT_FAILURE);
        }
    };

    const auto grouping = to_lower(expected_map_grouping);
    Eigen::Index expected_n_maps = 0;
    if (grouping == "array") {
        expected_n_maps = calib.arrays.size();
    }
    else if (grouping == "nw") {
        expected_n_maps = calib.nws.size();
    }
    else if (grouping == "fg") {
        expected_n_maps = calib.fg.size() * calib.arrays.size();
    }
    else if (grouping == "detector") {
        expected_n_maps = calib.n_dets;
    }
    else {
        logger->error("unsupported map grouping '{}' for fruit loops", expected_map_grouping);
        std::exit(EXIT_FAILURE);
    }

    std::unordered_map<Eigen::Index, Eigen::Index> array_to_index;
    for (Eigen::Index i=0; i<calib.arrays.size(); ++i) {
        array_to_index[calib.arrays(i)] = i;
    }

    std::unordered_map<Eigen::Index, Eigen::Index> nw_to_index;
    for (Eigen::Index i=0; i<calib.nws.size(); ++i) {
        nw_to_index[calib.nws(i)] = i;
    }

    std::unordered_map<Eigen::Index, Eigen::Index> fg_to_index;
    for (Eigen::Index i=0; i<calib.fg.size(); ++i) {
        fg_to_index[calib.fg(i)] = i;
    }

    auto parse_map_index = [&](const std::string &ext_name, const std::string &prefix,
                               Eigen::Index array_id) -> std::optional<Eigen::Index> {
        if (ext_name.rfind(prefix + "_", 0) != 0) {
            return std::nullopt;
        }
        auto tokens = split_tokens(ext_name);
        if (tokens.size() < 2) {
            return std::nullopt;
        }
        const auto stokes = tokens.back();
        if (stokes != "I") {
            return std::nullopt;
        }

        if (grouping == "array") {
            if (tokens.size() != 2) {
                return std::nullopt;
            }
            auto it = array_to_index.find(array_id);
            if (it == array_to_index.end()) {
                logger->error("array {} not found in calib arrays for fruit loops", array_id);
                std::exit(EXIT_FAILURE);
            }
            return it->second;
        }

        if (tokens.size() < 4) {
            return std::nullopt;
        }

        const auto &group_token = tokens[1];
        Eigen::Index group_id = -1;
        try {
            group_id = static_cast<Eigen::Index>(std::stol(tokens[2]));
        } catch (const std::exception &) {
            return std::nullopt;
        }

        if (grouping == "nw") {
            if (group_token != "nw") {
                return std::nullopt;
            }
            auto it = nw_to_index.find(group_id);
            if (it == nw_to_index.end()) {
                logger->error("nw {} not found in calib nws for fruit loops", group_id);
                std::exit(EXIT_FAILURE);
            }
            return it->second;
        }
        if (grouping == "fg") {
            if (group_token != "fg") {
                return std::nullopt;
            }
            auto fg_it = fg_to_index.find(group_id);
            if (fg_it == fg_to_index.end()) {
                logger->error("fg {} not found in calib fgs for fruit loops", group_id);
                std::exit(EXIT_FAILURE);
            }
            auto arr_it = array_to_index.find(array_id);
            if (arr_it == array_to_index.end()) {
                logger->error("array {} not found in calib arrays for fruit loops", array_id);
                std::exit(EXIT_FAILURE);
            }
            return fg_it->second + calib.fg.size() * arr_it->second;
        }
        if (grouping == "detector") {
            if (group_token != "det") {
                return std::nullopt;
            }
            if (group_id < 0 || group_id >= expected_n_maps) {
                logger->error("detector map id {} out of range for fruit loops", group_id);
                std::exit(EXIT_FAILURE);
            }
            return group_id;
        }
        return std::nullopt;
    };

    std::vector<std::optional<Eigen::MatrixXd>> signal_maps(expected_n_maps);
    std::vector<std::optional<Eigen::MatrixXd>> weight_maps(expected_n_maps);
    std::vector<std::optional<Eigen::MatrixXd>> kernel_maps(expected_n_maps);
    bool any_kernel = false;

    bool wcs_set = false;
    std::string file_grouping_lower;
    std::string file_pixel_axes_lower;

    // vector to hold mean rms per array
    std::vector<double> median_rms_vec(calib.arrays.size(),
                                       std::numeric_limits<double>::quiet_NaN());
    bool found_any_rms = false;

    // loop through arrays in current obs
    for (const auto &arr: calib.arrays) {
        try {
            std::vector<fs::path> map_files;
            for (const auto& entry : fs::directory_iterator(filepath)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                auto path_str = entry.path().string();
                if (path_str.find(".fits") == std::string::npos) {
                    continue;
                }
                if (path_str.find("_citlali") == std::string::npos) {
                    continue;
                }
                if (path_str.find("_noise") != std::string::npos) {
                    continue;
                }
                if (path_str.find(toltec_io.array_name_map[arr]) == std::string::npos) {
                    continue;
                }
                map_files.push_back(entry.path());
            }
            std::sort(map_files.begin(), map_files.end());
            if (map_files.empty()) {
                logger->error("no map FITS found for array {} in {}", toltec_io.array_name_map[arr], filepath);
                std::exit(EXIT_FAILURE);
            }
            if (map_files.size() > 1) {
                logger->error("multiple map FITS found for array {} in {}", toltec_io.array_name_map[arr], filepath);
                std::exit(EXIT_FAILURE);
            }

            auto map_path = map_files.front();
            fitsIO<file_type_enum::read_fits, CCfits::ExtHDU*> fits_io(map_path.string());

            // read grouping and pixel axes from primary hdu if present
            try {
                std::string grouping_key;
                fits_io.pfits->pHDU().readKey("GROUPING", grouping_key);
                if (!grouping_key.empty()) {
                    auto grouping_lower = to_lower(grouping_key);
                    if (!file_grouping_lower.empty() && grouping_lower != file_grouping_lower) {
                        logger->error("mismatched GROUPING across fruit loops maps: {} vs {}", file_grouping_lower, grouping_lower);
                        std::exit(EXIT_FAILURE);
                    }
                    file_grouping_lower = grouping_lower;
                    if (grouping_lower != grouping) {
                        logger->error("fruit loops maps GROUPING '{}' does not match expected '{}'",
                                      grouping_key, expected_map_grouping);
                        std::exit(EXIT_FAILURE);
                    }
                }
            } catch (...) {
                // ignore if missing
            }

            try {
                std::string axes_key;
                fits_io.pfits->pHDU().readKey("RADESYS", axes_key);
                if (!axes_key.empty()) {
                    auto axes_lower = to_lower(axes_key);
                    if (!file_pixel_axes_lower.empty() && axes_lower != file_pixel_axes_lower) {
                        logger->error("mismatched RADESYS across fruit loops maps: {} vs {}", file_pixel_axes_lower, axes_lower);
                        std::exit(EXIT_FAILURE);
                    }
                    file_pixel_axes_lower = axes_lower;
                    if (!expected_pixel_axes.empty() && axes_lower != to_lower(expected_pixel_axes)) {
                        logger->error("fruit loops maps RADESYS '{}' does not match expected '{}'",
                                      axes_key, expected_pixel_axes);
                        std::exit(EXIT_FAILURE);
                    }
                }
            } catch (...) {
                // ignore if missing
            }

            // get number of extensions other than primary extension
            int num_extensions = 0;
            bool keep_going = true;
            while (keep_going) {
                try {
                    CCfits::ExtHDU& ext = fits_io.pfits->extension(num_extensions + 1);
                    num_extensions++;
                } catch (CCfits::FITS::NoSuchHDU) {
                    keep_going = false;
                }
            }

            if (num_extensions == 0) {
                logger->error("{} is empty", map_path.string());
                std::exit(EXIT_FAILURE);
            }

            // get wcs (should be same for all maps)
            CCfits::ExtHDU& extension = fits_io.pfits->extension(1);
            long naxis1 = 0;
            long naxis2 = 0;
            double crpix1 = 0.0;
            double crpix2 = 0.0;
            double crval1 = 0.0;
            double crval2 = 0.0;
            double cdelt1 = 0.0;
            double cdelt2 = 0.0;
            std::string cunit1;
            std::string cunit2;
            read_key_long(extension, "NAXIS1", naxis1);
            read_key_long(extension, "NAXIS2", naxis2);
            read_key_double(extension, "CRPIX1", crpix1);
            read_key_double(extension, "CRPIX2", crpix2);
            read_key_double(extension, "CRVAL1", crval1);
            read_key_double(extension, "CRVAL2", crval2);
            read_key_double(extension, "CDELT1", cdelt1);
            read_key_double(extension, "CDELT2", cdelt2);
            read_key_string(extension, "CUNIT1", cunit1);
            read_key_string(extension, "CUNIT2", cunit2);

            // convert CRPIX to 0-based
            crpix1 -= 1.0;
            crpix2 -= 1.0;

            logger->debug("fruit loops WCS {}: naxis=({}, {}) crpix=({}, {}) crval=({}, {}) cdelt=({}, {}) cunit=({}, {})",
                          map_path.string(), naxis1, naxis2, crpix1, crpix2,
                          crval1, crval2, cdelt1, cdelt2, cunit1, cunit2);

            if (!wcs_set) {
                tod_mb.wcs.naxis[0] = static_cast<int>(naxis1);
                tod_mb.wcs.naxis[1] = static_cast<int>(naxis2);
                tod_mb.wcs.crpix[0] = crpix1;
                tod_mb.wcs.crpix[1] = crpix2;
                tod_mb.wcs.crval[0] = crval1;
                tod_mb.wcs.crval[1] = crval2;
                tod_mb.wcs.cdelt[0] = cdelt1;
                tod_mb.wcs.cdelt[1] = cdelt2;
                tod_mb.wcs.cunit[0] = cunit1;
                tod_mb.wcs.cunit[1] = cunit2;
                wcs_set = true;
                logger->debug("fruit loops WCS reference set from {}",
                              map_path.string());
            }
            else {
                if (tod_mb.wcs.naxis[0] != naxis1 || tod_mb.wcs.naxis[1] != naxis2) {
                    logger->error("inconsistent map dimensions across fruit loops maps in {}",
                                  map_path.string());
                    std::exit(EXIT_FAILURE);
                }
                double cdelt0_ref = std::abs(tod_mb.wcs.cdelt[0]);
                double cdelt1_ref = std::abs(tod_mb.wcs.cdelt[1]);
                double cdelt0_new = std::abs(cdelt1);
                double cdelt1_new = std::abs(cdelt2);
                double cdelt0_diff = std::abs(cdelt0_ref - cdelt0_new);
                double cdelt1_diff = std::abs(cdelt1_ref - cdelt1_new);
                double cdelt0_rel = cdelt0_diff / std::max({cdelt0_ref, cdelt0_new, 1e-12});
                double cdelt1_rel = cdelt1_diff / std::max({cdelt1_ref, cdelt1_new, 1e-12});
                constexpr double cdelt_rel_tol = 1e-6;
                if (cdelt0_rel > cdelt_rel_tol || cdelt1_rel > cdelt_rel_tol) {
                    logger->error("inconsistent CDELT across fruit loops maps in {}: "
                                  "ref=({}, {}) new=({}, {}) rel_diff=({}, {})",
                                  map_path.string(),
                                  tod_mb.wcs.cdelt[0], tod_mb.wcs.cdelt[1], cdelt1, cdelt2,
                                  cdelt0_rel, cdelt1_rel);
                    std::exit(EXIT_FAILURE);
                }
                if (to_lower(tod_mb.wcs.cunit[0]) != to_lower(cunit1) ||
                    to_lower(tod_mb.wcs.cunit[1]) != to_lower(cunit2)) {
                    logger->error("inconsistent CUNIT across fruit loops maps in {}",
                                  map_path.string());
                    std::exit(EXIT_FAILURE);
                }
            }

            // get maps, including all fg maps
            for (int i=0; i<num_extensions; ++i) {
                CCfits::ExtHDU& ext = fits_io.pfits->extension(i+1);
                std::string extName;
                ext.readKey("EXTNAME", extName);

                if (extName.rfind("signal_", 0) == 0) {
                    auto map_index = parse_map_index(extName, "signal", arr);
                    if (map_index) {
                        if (signal_maps[*map_index].has_value()) {
                            logger->error("duplicate signal map index {} in {}", *map_index, map_path.string());
                            std::exit(EXIT_FAILURE);
                        }
                        signal_maps[*map_index] = fits_io.get_hdu(extName);
                        logger->info("found {} [{}]", map_path.filename().string(), extName);
                    }
                }
                else if (extName.rfind("weight_", 0) == 0) {
                    auto map_index = parse_map_index(extName, "weight", arr);
                    if (map_index) {
                        if (weight_maps[*map_index].has_value()) {
                            logger->error("duplicate weight map index {} in {}", *map_index, map_path.string());
                            std::exit(EXIT_FAILURE);
                        }
                        weight_maps[*map_index] = fits_io.get_hdu(extName);
                        logger->info("found {} [{}]", map_path.filename().string(), extName);
                    }
                }
                else if (extName.rfind("kernel_", 0) == 0) {
                    auto map_index = parse_map_index(extName, "kernel", arr);
                    if (map_index) {
                        if (kernel_maps[*map_index].has_value()) {
                            logger->error("duplicate kernel map index {} in {}", *map_index, map_path.string());
                            std::exit(EXIT_FAILURE);
                        }
                        kernel_maps[*map_index] = fits_io.get_hdu(extName);
                        any_kernel = true;
                        logger->info("found {} [{}]", map_path.filename().string(), extName);
                    }
                }
            }

            // get noise maps for median rms
            std::vector<fs::path> noise_files;
            for (const auto& entry : fs::directory_iterator(noise_filepath)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                auto path_str = entry.path().string();
                if (path_str.find(".fits") == std::string::npos) {
                    continue;
                }
                // Accept both raw and filtered noise map filenames.
                if (path_str.find("_noise") == std::string::npos) {
                    continue;
                }
                if (path_str.size() < 13 ||
                    path_str.compare(path_str.size() - 13, 13, "_citlali.fits") != 0) {
                    continue;
                }
                if (path_str.find(toltec_io.array_name_map[arr]) == std::string::npos) {
                    continue;
                }
                noise_files.push_back(entry.path());
            }
            std::sort(noise_files.begin(), noise_files.end());
            if (!noise_files.empty()) {
                // only use the first noise file for this array
                fitsIO<file_type_enum::read_fits, CCfits::ExtHDU*> noise_io(noise_files.front().string());

                int num_noise_ext = 0;
                keep_going = true;
                while (keep_going) {
                    try {
                        CCfits::ExtHDU& ext = noise_io.pfits->extension(num_noise_ext + 1);
                        num_noise_ext++;
                    } catch (CCfits::FITS::NoSuchHDU) {
                        keep_going = false;
                    }
                }

                if (num_noise_ext == 0) {
                    logger->error("{} is empty", noise_files.front().string());
                    std::exit(EXIT_FAILURE);
                }

                double median_rms = std::numeric_limits<double>::quiet_NaN();
                for (int i=0; i<num_noise_ext; ++i) {
                    CCfits::ExtHDU& ext = noise_io.pfits->extension(i+1);
                    std::string extName;
                    ext.readKey("EXTNAME", extName);
                    if (extName.find("signal") != std::string::npos && extName.find("_0_I") != std::string::npos) {
                        ext.readKey("MEDRMS", median_rms);
                        found_any_rms = true;
                        break;
                    }
                }
                auto arr_it = array_to_index.find(arr);
                if (arr_it != array_to_index.end()) {
                    median_rms_vec[arr_it->second] = median_rms;
                }
            }
            else {
                logger->warn("no noise FITS found for array {} in {}; fruit loops S/N gating may be disabled",
                             toltec_io.array_name_map[arr], noise_filepath);
            }

        } catch (const fs::filesystem_error& err) {
            logger->error("{}", err.what());
            std::exit(EXIT_FAILURE);
        }
    }

    // check if we found any maps
    if (expected_n_maps == 0) {
        logger->error("no maps expected for fruit loops");
        std::exit(EXIT_FAILURE);
    }

    for (Eigen::Index i=0; i<expected_n_maps; ++i) {
        if (!signal_maps[i].has_value()) {
            logger->error("missing signal map index {} in {}", i, filepath);
            std::exit(EXIT_FAILURE);
        }
        if (!weight_maps[i].has_value()) {
            logger->error("missing weight map index {} in {}", i, filepath);
            std::exit(EXIT_FAILURE);
        }
        tod_mb.signal.push_back(std::move(*signal_maps[i]));
        tod_mb.weight.push_back(std::move(*weight_maps[i]));
    }

    if (any_kernel) {
        bool missing_kernel = false;
        for (Eigen::Index i=0; i<expected_n_maps; ++i) {
            if (kernel_maps[i].has_value()) {
                tod_mb.kernel.push_back(std::move(*kernel_maps[i]));
            }
            else {
                missing_kernel = true;
                break;
            }
        }
        if (missing_kernel) {
            logger->warn("kernel maps incomplete; disabling kernel subtraction for fruit loops");
            std::vector<Eigen::MatrixXd>().swap(tod_mb.kernel);
        }
    }

    if (found_any_rms) {
        tod_mb.median_rms = Eigen::Map<Eigen::VectorXd>(median_rms_vec.data(), median_rms_vec.size());
    }
    else {
        logger->warn("fruit loops did not load MEDRMS from noise maps in {}; S/N gating will be disabled", noise_filepath);
    }

    // set dimensions
    tod_mb.n_cols = tod_mb.wcs.naxis[0];
    tod_mb.n_rows = tod_mb.wcs.naxis[1];

    // get pixel size in radians
    if (to_lower(tod_mb.wcs.cunit[0]) == "deg") {
        tod_mb.pixel_size_rad = std::abs(tod_mb.wcs.cdelt[0])*DEG_TO_RAD;
    }
    else if (to_lower(tod_mb.wcs.cunit[0]) == "arcsec") {
        tod_mb.pixel_size_rad = std::abs(tod_mb.wcs.cdelt[0])*ASEC_TO_RAD;
    }
    else {
        logger->error("unsupported CUNIT '{}' in fruit loops maps", tod_mb.wcs.cunit[0]);
        std::exit(EXIT_FAILURE);
    }

    if (expected_pixel_size_rad > 0.0) {
        double diff = std::abs(tod_mb.pixel_size_rad - expected_pixel_size_rad);
        double tol = std::max(1e-12, expected_pixel_size_rad * 1e-6);
        if (diff > tol) {
            logger->error("fruit loops map pixel size {} rad does not match expected {} rad",
                          tod_mb.pixel_size_rad, expected_pixel_size_rad);
            std::exit(EXIT_FAILURE);
        }
    }

    if (fruit_loops_interp_mode == "jinc") {
        allocate_fruit_loops_jinc_matrix(tod_mb.pixel_size_rad);
    }
    else {
        fruit_loops_jinc_weights_mat.clear();
        fruit_loops_jinc_weights_mat_subpix.clear();
    }

    double expected_row = (tod_mb.n_rows - 1) / 2.0;
    double expected_col = (tod_mb.n_cols - 1) / 2.0;
    if (std::isfinite(tod_mb.wcs.crpix[0]) && tod_mb.wcs.crpix[0] > 0.0 &&
        std::abs(tod_mb.wcs.crpix[0] - expected_col) > 1.0) {
        logger->error("fruit loops map CRPIX1 ({}) does not match expected map center ({})",
                      tod_mb.wcs.crpix[0], expected_col);
        std::exit(EXIT_FAILURE);
    }
    if (std::isfinite(tod_mb.wcs.crpix[1]) && tod_mb.wcs.crpix[1] > 0.0 &&
        std::abs(tod_mb.wcs.crpix[1] - expected_row) > 1.0) {
        logger->error("fruit loops map CRPIX2 ({}) does not match expected map center ({})",
                      tod_mb.wcs.crpix[1], expected_row);
        std::exit(EXIT_FAILURE);
    }

    Eigen::MatrixXd ones, zeros;
    ones.setOnes(tod_mb.weight[0].rows(), tod_mb.weight[0].cols());
    zeros.setZero(tod_mb.weight[0].rows(), tod_mb.weight[0].cols());

    Eigen::MatrixXd center_keep_mask;
    const bool use_center_keep_mask =
        fruit_loops_center_keep_radius_arcsec > 0.0 && tod_mb.pixel_size_rad > 0.0;
    if (use_center_keep_mask) {
        center_keep_mask.setZero(tod_mb.weight[0].rows(), tod_mb.weight[0].cols());
        const double keep_radius_pix =
            fruit_loops_center_keep_radius_arcsec * ASEC_TO_RAD / tod_mb.pixel_size_rad;
        const double keep_radius_sq = keep_radius_pix * keep_radius_pix;
        const double center_row = (tod_mb.n_rows - 1) / 2.0;
        const double center_col = (tod_mb.n_cols - 1) / 2.0;

        for (Eigen::Index row = 0; row < tod_mb.n_rows; ++row) {
            const double drow = static_cast<double>(row) - center_row;
            for (Eigen::Index col = 0; col < tod_mb.n_cols; ++col) {
                const double dcol = static_cast<double>(col) - center_col;
                if (drow * drow + dcol * dcol <= keep_radius_sq) {
                    center_keep_mask(row, col) = 1.0;
                }
            }
        }
        logger->info("fruit loops preserving central {:.3f} arcsec from coverage cut (radius {:.3f} pix)",
                     fruit_loops_center_keep_radius_arcsec, keep_radius_pix);
    }

    // calculate coverage bool map
    for (int i=0; i<tod_mb.weight.size(); ++i) {
        // get weight threshold for current map
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = tod_mb.calc_cov_region(i);
        // if weight is less than threshold, set to zero, otherwise set to one
        Eigen::MatrixXd cov_bool =
            (tod_mb.weight[i].array() < weight_threshold).select(zeros,ones);
        if (use_center_keep_mask) {
            cov_bool = (cov_bool.array() + center_keep_mask.array()).min(1.0).matrix();
        }
        tod_mb.signal[i] = tod_mb.signal[i].array() * cov_bool.array();
        if (!tod_mb.kernel.empty()) {
            tod_mb.kernel[i] = tod_mb.kernel[i].array() * cov_bool.array();
        }
    }
    fruit_loops_source_lat = Eigen::VectorXd::Zero(tod_mb.signal.size());
    fruit_loops_source_lon = Eigen::VectorXd::Zero(tod_mb.signal.size());
    fruit_loops_source_valid = Eigen::VectorXi::Zero(tod_mb.signal.size());
    const double row_offset = (tod_mb.n_rows - 1) / 2.0;
    const double col_offset = (tod_mb.n_cols - 1) / 2.0;
    auto apt_position_value = [&](const std::string &key, Eigen::Index index) {
        auto it = calib.apt.find(key);
        if (it == calib.apt.end() || index < 0 || index >= it->second.size()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return it->second(index);
    };
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(tod_mb.signal.size()); ++i) {
        if (grouping == "detector") {
            const double x_arcsec = apt_position_value("x_t", i);
            const double y_arcsec = apt_position_value("y_t", i);
            if (std::isfinite(x_arcsec) && std::isfinite(y_arcsec)) {
                fruit_loops_source_lat(i) = y_arcsec * ASEC_TO_RAD;
                fruit_loops_source_lon(i) = x_arcsec * ASEC_TO_RAD;
                fruit_loops_source_valid(i) = 1;
                continue;
            }
        }
        double peak_val = -std::numeric_limits<double>::infinity();
        Eigen::Index peak_row = 0;
        Eigen::Index peak_col = 0;
        bool found_peak = false;
        for (Eigen::Index row = 0; row < tod_mb.signal[i].rows(); ++row) {
            for (Eigen::Index col = 0; col < tod_mb.signal[i].cols(); ++col) {
                const double value = tod_mb.signal[i](row, col);
                if (std::isfinite(value) && value > peak_val) {
                    peak_val = value;
                    peak_row = row;
                    peak_col = col;
                    found_peak = true;
                }
            }
        }
        if (found_peak && peak_val > 0.0) {
            fruit_loops_source_lat(i) =
                (static_cast<double>(peak_row) - row_offset) * tod_mb.pixel_size_rad;
            fruit_loops_source_lon(i) =
                (static_cast<double>(peak_col) - col_offset) * tod_mb.pixel_size_rad;
            fruit_loops_source_valid(i) = 1;
        }
    }

    configure_fruit_loops_adaptive_gate(tod_mb, calib, grouping);
    // clear weight maps to save memory
    std::vector<Eigen::MatrixXd>().swap(tod_mb.weight);
}

template <class mb_t, class calib_t>
void TCProc::configure_fruit_loops_adaptive_gate(mb_t &mb, calib_t &calib,
                                                 const std::string &map_grouping,
                                                 bool allow_peak_source_fallback) {

    const bool use_adaptive_threshold =
        fruit_loops_peak_fraction_limit > 0.0 || fruit_loops_local_snr_floor > 0.0;
    const auto n_maps = static_cast<Eigen::Index>(mb.signal.size());
    if (!use_adaptive_threshold || n_maps <= 0) {
        fruit_loops_local_sigma_map.resize(0);
        fruit_loops_local_sigma_npix.resize(0);
        fruit_loops_amp_ref.resize(0);
        fruit_loops_adaptive_threshold.resize(0);
        fruit_loops_adaptive_support_radius_rad.resize(0);
        return;
    }

    const double fill = std::numeric_limits<double>::quiet_NaN();
    fruit_loops_local_sigma_map = Eigen::VectorXd::Constant(n_maps, fill);
    fruit_loops_local_sigma_npix = Eigen::VectorXi::Zero(n_maps);
    fruit_loops_amp_ref = Eigen::VectorXd::Constant(n_maps, fill);
    fruit_loops_adaptive_threshold = Eigen::VectorXd::Constant(n_maps, fill);
    fruit_loops_adaptive_support_radius_rad = Eigen::VectorXd::Constant(n_maps, fill);

    const double pix_arcsec = mb.pixel_size_rad * RAD_TO_ASEC;
    if (!std::isfinite(pix_arcsec) || pix_arcsec <= 0.0 ||
        mb.n_rows <= 0 || mb.n_cols <= 0) {
        logger->warn("fruit loops adaptive gate disabled: invalid map geometry");
        return;
    }

    if (fruit_loops_source_valid.size() != n_maps ||
        fruit_loops_source_lat.size() != n_maps ||
        fruit_loops_source_lon.size() != n_maps) {
        fruit_loops_source_lat = Eigen::VectorXd::Zero(n_maps);
        fruit_loops_source_lon = Eigen::VectorXd::Zero(n_maps);
        fruit_loops_source_valid = Eigen::VectorXi::Zero(n_maps);
    }

    const double row_offset = (mb.n_rows - 1) / 2.0;
    const double col_offset = (mb.n_cols - 1) / 2.0;
    const double edge_guard_pix = fruit_loops_local_sigma_edge_guard_arcsec / pix_arcsec;
    const Eigen::Index edge_guard =
        std::max<Eigen::Index>(0, static_cast<Eigen::Index>(std::ceil(edge_guard_pix)));

    auto median_of = [](std::vector<double> values) {
        if (values.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const auto mid_index = values.size() / 2;
        auto mid = values.begin() + static_cast<std::ptrdiff_t>(mid_index);
        std::nth_element(values.begin(), mid, values.end());
        double med = *mid;
        if (values.size() % 2 == 0 && mid_index > 0) {
            auto lo = values.begin() + static_cast<std::ptrdiff_t>(mid_index - 1);
            std::nth_element(values.begin(), lo, mid);
            med = 0.5 * (med + *lo);
        }
        return med;
    };

    auto robust_sigma = [&](const std::vector<double> &values) {
        if (values.size() < static_cast<std::size_t>(fruit_loops_local_sigma_min_pixels)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double med = median_of(values);
        if (!std::isfinite(med)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::vector<double> deviations;
        deviations.reserve(values.size());
        for (const auto value : values) {
            deviations.push_back(std::abs(value - med));
        }
        double sigma = 1.4826 * median_of(std::move(deviations));
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            const double mean =
                std::accumulate(values.begin(), values.end(), 0.0) /
                static_cast<double>(values.size());
            double var = 0.0;
            for (const auto value : values) {
                const double dv = value - mean;
                var += dv * dv;
            }
            sigma = std::sqrt(var / static_cast<double>(values.size()));
        }
        return (std::isfinite(sigma) && sigma > 0.0)
                   ? sigma
                   : std::numeric_limits<double>::quiet_NaN();
    };

    auto apt_value = [&](const std::string &key, Eigen::Index index) {
        auto it = calib.apt.find(key);
        if (it == calib.apt.end() || index < 0 || index >= it->second.size()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double value = it->second(index);
        return (std::isfinite(value) && value > 0.0)
                   ? value
                   : std::numeric_limits<double>::quiet_NaN();
    };

    auto set_peak_center_if_needed = [&](Eigen::Index map_index) {
        if (fruit_loops_source_valid(map_index) != 0) {
            return;
        }
        double peak_val = -std::numeric_limits<double>::infinity();
        Eigen::Index peak_row = 0;
        Eigen::Index peak_col = 0;
        bool found_peak = false;
        for (Eigen::Index row = 0; row < mb.signal[map_index].rows(); ++row) {
            for (Eigen::Index col = 0; col < mb.signal[map_index].cols(); ++col) {
                const double value = mb.signal[map_index](row, col);
                if (std::isfinite(value) && value > peak_val) {
                    peak_val = value;
                    peak_row = row;
                    peak_col = col;
                    found_peak = true;
                }
            }
        }
        if (found_peak && peak_val > 0.0) {
            fruit_loops_source_lat(map_index) =
                (static_cast<double>(peak_row) - row_offset) * mb.pixel_size_rad;
            fruit_loops_source_lon(map_index) =
                (static_cast<double>(peak_col) - col_offset) * mb.pixel_size_rad;
            fruit_loops_source_valid(map_index) = 1;
        }
    };

    Eigen::Index n_valid_thresholds = 0;
    Eigen::Index n_valid_sigma = 0;
    Eigen::Index n_valid_amp = 0;
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        if (allow_peak_source_fallback) {
            set_peak_center_if_needed(i);
        }
        if (fruit_loops_source_valid(i) == 0) {
            continue;
        }
        const double center_row = fruit_loops_source_lat(i) / mb.pixel_size_rad + row_offset;
        const double center_col = fruit_loops_source_lon(i) / mb.pixel_size_rad + col_offset;
        if (!std::isfinite(center_row) || !std::isfinite(center_col)) {
            continue;
        }

        double fwhm_arcsec = std::numeric_limits<double>::quiet_NaN();
        if (map_grouping == "detector") {
            const double a_fwhm = apt_value("a_fwhm", i);
            const double b_fwhm = apt_value("b_fwhm", i);
            if (std::isfinite(a_fwhm) && std::isfinite(b_fwhm)) {
                fwhm_arcsec = std::max(a_fwhm, b_fwhm);
            }
            else if (std::isfinite(a_fwhm)) {
                fwhm_arcsec = a_fwhm;
            }
            else if (std::isfinite(b_fwhm)) {
                fwhm_arcsec = b_fwhm;
            }
        }
        if (!std::isfinite(fwhm_arcsec) || fwhm_arcsec <= 0.0) {
            fwhm_arcsec = 0.0;
        }

        double inner_arcsec = std::max(
            fruit_loops_local_sigma_inner_radius_arcsec,
            fruit_loops_local_sigma_inner_fwhm * fwhm_arcsec);
        double outer_arcsec = std::max(
            fruit_loops_local_sigma_outer_radius_arcsec,
            fruit_loops_local_sigma_outer_fwhm * fwhm_arcsec);
        if (!(outer_arcsec > inner_arcsec)) {
            outer_arcsec = inner_arcsec + std::max(5.0, fwhm_arcsec);
        }

        std::vector<double> annulus_values;
        annulus_values.reserve(static_cast<std::size_t>(mb.n_rows * mb.n_cols / 8));
        const Eigen::Index row_begin = std::min(edge_guard, mb.n_rows);
        const Eigen::Index row_end = std::max(row_begin, mb.n_rows - edge_guard);
        const Eigen::Index col_begin = std::min(edge_guard, mb.n_cols);
        const Eigen::Index col_end = std::max(col_begin, mb.n_cols - edge_guard);
        const bool have_weight_map =
            i < static_cast<Eigen::Index>(mb.weight.size()) &&
            mb.weight[i].rows() == mb.n_rows &&
            mb.weight[i].cols() == mb.n_cols;
        for (Eigen::Index row = row_begin; row < row_end; ++row) {
            const double drow = (static_cast<double>(row) - center_row) * pix_arcsec;
            for (Eigen::Index col = col_begin; col < col_end; ++col) {
                if (have_weight_map) {
                    const double weight = mb.weight[i](row, col);
                    if (!std::isfinite(weight) || weight <= 0.0) {
                        continue;
                    }
                }
                const double signal = mb.signal[i](row, col);
                if (!std::isfinite(signal)) {
                    continue;
                }
                const double dcol = (static_cast<double>(col) - center_col) * pix_arcsec;
                const double radius_arcsec = std::sqrt(drow * drow + dcol * dcol);
                if (radius_arcsec >= inner_arcsec && radius_arcsec <= outer_arcsec) {
                    annulus_values.push_back(signal);
                }
            }
        }

        fruit_loops_local_sigma_npix(i) =
            static_cast<int>(std::min<std::size_t>(
                annulus_values.size(), static_cast<std::size_t>(std::numeric_limits<int>::max())));
        const double sigma = robust_sigma(annulus_values);
        if (std::isfinite(sigma) && sigma > 0.0) {
            fruit_loops_local_sigma_map(i) = sigma;
            n_valid_sigma++;
        }

        double amp_ref = std::numeric_limits<double>::quiet_NaN();
        if (map_grouping == "detector") {
            for (const auto &key : {"cal_amp", "template_amp", "amp", "map_peak_amp"}) {
                amp_ref = apt_value(key, i);
                if (std::isfinite(amp_ref) && amp_ref > 0.0) {
                    break;
                }
            }
        }
        if (!std::isfinite(amp_ref) || amp_ref <= 0.0) {
            const double peak_radius_arcsec = std::max(5.0, inner_arcsec);
            double local_peak = -std::numeric_limits<double>::infinity();
            for (Eigen::Index row = 0; row < mb.n_rows; ++row) {
                const double drow = (static_cast<double>(row) - center_row) * pix_arcsec;
                for (Eigen::Index col = 0; col < mb.n_cols; ++col) {
                    const double signal = mb.signal[i](row, col);
                    if (!std::isfinite(signal)) {
                        continue;
                    }
                    const double dcol = (static_cast<double>(col) - center_col) * pix_arcsec;
                    if (std::sqrt(drow * drow + dcol * dcol) <= peak_radius_arcsec &&
                        signal > local_peak) {
                        local_peak = signal;
                    }
                }
            }
            if (std::isfinite(local_peak) && local_peak > 0.0) {
                amp_ref = local_peak;
            }
        }
        if (std::isfinite(amp_ref) && amp_ref > 0.0) {
            fruit_loops_amp_ref(i) = amp_ref;
            n_valid_amp++;
        }

        double threshold = std::numeric_limits<double>::quiet_NaN();
        if (fruit_loops_peak_fraction_limit > 0.0 &&
            std::isfinite(amp_ref) && amp_ref > 0.0) {
            threshold = fruit_loops_peak_fraction_limit * amp_ref;
        }
        if (fruit_loops_local_snr_floor > 0.0 &&
            std::isfinite(sigma) && sigma > 0.0) {
            const double snr_threshold = fruit_loops_local_snr_floor * sigma;
            threshold = std::isfinite(threshold)
                            ? std::max(threshold, snr_threshold)
                            : snr_threshold;
        }
        if (std::isfinite(threshold) && threshold > 0.0) {
            fruit_loops_adaptive_threshold(i) = threshold;
            n_valid_thresholds++;
        }

        double support_radius_arcsec = fruit_loops_adaptive_support_radius_arcsec;
        if (fruit_loops_adaptive_support_radius_fwhm > 0.0 &&
            std::isfinite(fwhm_arcsec) && fwhm_arcsec > 0.0) {
            support_radius_arcsec = std::max(
                support_radius_arcsec,
                fruit_loops_adaptive_support_radius_fwhm * fwhm_arcsec);
        }
        if (std::isfinite(support_radius_arcsec) && support_radius_arcsec > 0.0) {
            fruit_loops_adaptive_support_radius_rad(i) =
                support_radius_arcsec * ASEC_TO_RAD;
        }
    }

    logger->info("fruit loops adaptive gate: peak_fraction={} local_snr_floor={} "
                 "local_sigma_annulus=[{}={}, {}={}] arcsec edge_guard={} arcsec "
                 "support_radius_min={} arcsec support_radius_fwhm={} "
                 "valid_thresholds={}/{} valid_sigma={} valid_amp={}",
                 fruit_loops_peak_fraction_limit, fruit_loops_local_snr_floor,
                 "inner", fruit_loops_local_sigma_inner_radius_arcsec,
                 "outer", fruit_loops_local_sigma_outer_radius_arcsec,
                 fruit_loops_local_sigma_edge_guard_arcsec,
                 fruit_loops_adaptive_support_radius_arcsec,
                 fruit_loops_adaptive_support_radius_fwhm,
                 n_valid_thresholds, n_maps, n_valid_sigma, n_valid_amp);
}

inline double TCProc::fruit_loops_jinc_func(double r, double a, double b, double c,
                                            double r_max, double l_d) {
    if (r != 0.0) {
        r = r / l_d;
        auto jinc_1 = 2.0 * boost::math::cyl_bessel_j(1, 2.0 * pi * r / a) /
                      (2.0 * pi * r / a);
        auto exp_func = std::exp(-std::pow(2.0 * r / b, c));
        auto jinc_2 = 2.0 * boost::math::cyl_bessel_j(1, 3.831706 * r / r_max) /
                      (3.831706 * r / r_max);
        return jinc_1 * exp_func * jinc_2;
    }
    return 1.0;
}

inline void TCProc::allocate_fruit_loops_jinc_matrix(double pixel_size_rad) {
    fruit_loops_jinc_weights_mat.clear();
    fruit_loops_jinc_weights_mat_subpix.clear();

    if (pixel_size_rad <= 0.0 || fruit_loops_jinc_r_max <= 0.0 ||
        fruit_loops_jinc_shape_params.empty()) {
        return;
    }

    static const std::map<Eigen::Index, double> l_d = {
        {0, (1.1 / 1000) / 45},
        {1, (1.4 / 1000) / 45},
        {2, (2.0 / 1000) / 45},
    };

    const int subpixel_n = std::max(1, fruit_loops_jinc_subpixel_n);
    std::vector<double> subpixel_offsets;
    if (subpixel_n > 1) {
        subpixel_offsets.resize(subpixel_n);
        for (int i = 0; i < subpixel_n; ++i) {
            subpixel_offsets[i] =
                -0.5 + (static_cast<double>(i) + 0.5) / static_cast<double>(subpixel_n);
        }
    }

    for (const auto &[array_index, shape_params] : fruit_loops_jinc_shape_params) {
        auto ld_it = l_d.find(array_index);
        if (ld_it == l_d.end() || shape_params.size() < 3) {
            continue;
        }

        const auto a = shape_params(0);
        const auto b = shape_params(1);
        const auto c = shape_params(2);
        const auto ld = ld_it->second;
        const int r_max_pix = std::max(
            0, static_cast<int>(std::floor(fruit_loops_jinc_r_max * ld / pixel_size_rad)));
        const Eigen::VectorXd pixels =
            Eigen::VectorXd::LinSpaced(2 * r_max_pix + 1, -r_max_pix, r_max_pix);

        auto &kernel = fruit_loops_jinc_weights_mat[array_index];
        kernel.setZero(pixels.size(), pixels.size());

        for (Eigen::Index i = 0; i < pixels.size(); ++i) {
            for (Eigen::Index j = 0; j < pixels.size(); ++j) {
                const double radius =
                    pixel_size_rad * std::sqrt(std::pow(pixels(i), 2) + std::pow(pixels(j), 2));
                kernel(i, j) =
                    fruit_loops_jinc_func(radius, a, b, c, fruit_loops_jinc_r_max, ld);
            }
        }

        if (subpixel_n > 1) {
            auto &subpix_vec = fruit_loops_jinc_weights_mat_subpix[array_index];
            subpix_vec.resize(subpixel_n * subpixel_n);
            for (int sr = 0; sr < subpixel_n; ++sr) {
                for (int sc = 0; sc < subpixel_n; ++sc) {
                    auto &subpix_kernel = subpix_vec[static_cast<size_t>(sr * subpixel_n + sc)];
                    subpix_kernel.setZero(pixels.size(), pixels.size());
                    for (Eigen::Index i = 0; i < pixels.size(); ++i) {
                        for (Eigen::Index j = 0; j < pixels.size(); ++j) {
                            const double radius =
                                pixel_size_rad *
                                std::sqrt(std::pow(pixels(i) - subpixel_offsets[sr], 2) +
                                          std::pow(pixels(j) - subpixel_offsets[sc], 2));
                            subpix_kernel(i, j) = fruit_loops_jinc_func(
                                radius, a, b, c, fruit_loops_jinc_r_max, ld);
                        }
                    }
                }
            }
        }
    }
}

inline double TCProc::sample_map_bilinear(const Eigen::MatrixXd &map, double row,
                                          double col) const {
    if (!std::isfinite(row) || !std::isfinite(col) || map.size() == 0) {
        return 0.0;
    }
    if (row < 0.0 || col < 0.0 ||
        row > static_cast<double>(map.rows() - 1) ||
        col > static_cast<double>(map.cols() - 1)) {
        return 0.0;
    }

    const auto row0 = static_cast<Eigen::Index>(std::floor(row));
    const auto col0 = static_cast<Eigen::Index>(std::floor(col));
    const auto row1 = row0 + 1;
    const auto col1 = col0 + 1;
    const double frac_row = row - static_cast<double>(row0);
    const double frac_col = col - static_cast<double>(col0);

    double weighted_sum = 0.0;
    double norm = 0.0;

    for (int dr = 0; dr <= 1; ++dr) {
        const auto rr = (dr == 0) ? row0 : row1;
        const double row_weight = (dr == 0) ? (1.0 - frac_row) : frac_row;
        if (rr < 0 || rr >= map.rows()) {
            continue;
        }
        for (int dc = 0; dc <= 1; ++dc) {
            const auto cc = (dc == 0) ? col0 : col1;
            const double col_weight = (dc == 0) ? (1.0 - frac_col) : frac_col;
            if (cc < 0 || cc >= map.cols()) {
                continue;
            }
            const double weight = row_weight * col_weight;
            weighted_sum += weight * map(rr, cc);
            norm += weight;
        }
    }

    if (norm > 0.0) {
        return weighted_sum / norm;
    }

    const auto ir = std::clamp(static_cast<Eigen::Index>(std::llround(row)), Eigen::Index{0},
                               map.rows() - 1);
    const auto ic = std::clamp(static_cast<Eigen::Index>(std::llround(col)), Eigen::Index{0},
                               map.cols() - 1);
    return map(ir, ic);
}

inline double TCProc::sample_map_jinc(const Eigen::MatrixXd &map, Eigen::Index array_index,
                                      double row, double col) const {
    if (map.size() == 0) {
        return 0.0;
    }

    auto kernel_it = fruit_loops_jinc_weights_mat.find(array_index);
    if (kernel_it == fruit_loops_jinc_weights_mat.end()) {
        return sample_map_bilinear(map, row, col);
    }

    const auto ir = static_cast<Eigen::Index>(std::llround(row));
    const auto ic = static_cast<Eigen::Index>(std::llround(col));

    const Eigen::MatrixXd *kernel = &kernel_it->second;
    auto subpix_it = fruit_loops_jinc_weights_mat_subpix.find(array_index);
    const bool use_subpix =
        fruit_loops_jinc_subpixel_n > 1 && subpix_it != fruit_loops_jinc_weights_mat_subpix.end() &&
        !subpix_it->second.empty();
    if (use_subpix) {
        auto subpix_index = [&](double d) {
            int idx = static_cast<int>(std::floor((d + 0.5) * fruit_loops_jinc_subpixel_n));
            if (idx < 0) {
                idx = 0;
            }
            else if (idx >= fruit_loops_jinc_subpixel_n) {
                idx = fruit_loops_jinc_subpixel_n - 1;
            }
            return idx;
        };
        const int sr = subpix_index(row - static_cast<double>(ir));
        const int sc = subpix_index(col - static_cast<double>(ic));
        const auto idx = static_cast<size_t>(sr * fruit_loops_jinc_subpixel_n + sc);
        if (idx < subpix_it->second.size()) {
            kernel = &subpix_it->second[idx];
        }
    }

    const auto mat_rows = kernel->rows();
    const auto mat_cols = kernel->cols();
    const auto mat_rows_center = mat_rows / 2;
    const auto mat_cols_center = mat_cols / 2;

    auto lower_row = ir - mat_rows_center;
    auto upper_row = ir + mat_rows - 1 - mat_rows_center;
    auto lower_col = ic - mat_cols_center;
    auto upper_col = ic + mat_cols - 1 - mat_cols_center;

    const auto jinc_lower_row = std::abs(std::min<Eigen::Index>(0, lower_row));
    const auto jinc_lower_col = std::abs(std::min<Eigen::Index>(0, lower_col));

    lower_row = std::max<Eigen::Index>(0, lower_row);
    upper_row = std::min<Eigen::Index>(map.rows() - 1, upper_row);
    lower_col = std::max<Eigen::Index>(0, lower_col);
    upper_col = std::min<Eigen::Index>(map.cols() - 1, upper_col);

    if (lower_row > upper_row || lower_col > upper_col) {
        return 0.0;
    }

    const auto size_rows = upper_row - lower_row + 1;
    const auto size_cols = upper_col - lower_col + 1;

    double weighted_sum = 0.0;
    double norm = 0.0;
    for (Eigen::Index r = 0; r < size_rows; ++r) {
        for (Eigen::Index c = 0; c < size_cols; ++c) {
            const double weight = (*kernel)(jinc_lower_row + r, jinc_lower_col + c);
            weighted_sum += weight * map(lower_row + r, lower_col + c);
            norm += weight;
        }
    }

    // Memo-style jinc interpolation is a kernel-weighted average, not a matched-amplitude estimate.
    if (std::abs(norm) > 1e-8) {
        return weighted_sum / norm;
    }
    return sample_map_bilinear(map, row, col);
}

template <class calib_t>
auto TCProc::get_grouping(std::string grp, calib_t &calib, int n_dets) {
    std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grp_limits;

    // initial group value is value for the first det index
    Eigen::Index grp_i = calib.apt[grp](0);
    std::unordered_set<Eigen::Index> seen;
    seen.insert(grp_i);
    // set up first group
    grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{0, 0};
    Eigen::Index j = 0;
    // loop through apt table arrays, get highest index for current array
    for (Eigen::Index i=0; i<n_dets; ++i) {
        auto det_index = i;
        // if we're still on the current group
        if (calib.apt[grp](det_index) == grp_i) {
            std::get<1>(grp_limits[grp_i]) = i + 1;
        }
        // otherwise increment and start the next group
        else {
            grp_i = calib.apt[grp](det_index);
            if (seen.find(grp_i) != seen.end()) {
                logger->error("non-contiguous grouping detected for '{}' value {}", grp, grp_i);
                std::exit(EXIT_FAILURE);
            }
            seen.insert(grp_i);
            j += 1;
            grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{i,0};
        }
    }
    return grp_limits;
}

// compute pointing for all detectors
template <TCDataKind tcdata_t, class calib_t>
void TCProc::precompute_pointing(TCData<tcdata_t, Eigen::MatrixXd> &in, calib_t &calib, std::string pixel_axes, std::string map_grouping) {

    // dimensions of data
    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    in.pointing.data["lat"].resize(n_pts,n_dets);
    in.pointing.data["lon"].resize(n_pts,n_dets);

    for (Eigen::Index i=0; i<n_dets; ++i) {
        // current detector index in apt
        auto det_index = i;
        double az_off = calib.apt["x_t"](det_index);
        double el_off = calib.apt["y_t"](det_index);

        // get detector pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off, pixel_axes,
                                                          in.pointing_offsets_arcsec.data, map_grouping);

        in.pointing.data["lat"].col(i) = std::move(lat);
        in.pointing.data["lon"].col(i) = std::move(lon);
    }
}

template <TCProc::SourceType source_type, class mb_t, TCDataKind tcdata_t, class calib_t, typename Derived>
void TCProc::map_to_tod(mb_t &mb, TCData<tcdata_t, Eigen::MatrixXd> &in, calib_t &calib,
                        Eigen::DenseBase<Derived> &map_indices, std::string pixel_axes,
                        std::string map_grouping) {

    // dimensions of data
    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // add or subtract timestream
    int factor = 1;
    if constexpr (source_type==NegativeMap) {
        factor = -1;
    }

    // run kernel through fruit loops
    bool run_kernel = in.kernel.data.size() !=0;
    if (run_kernel && mb.kernel.size() != mb.signal.size()) {
        logger->warn("kernel map count ({}) does not match signal map count ({}); disabling kernel subtraction",
                     mb.kernel.size(), mb.signal.size());
        run_kernel = false;
    }
    // if mean rms is filled use S/N limit
    bool run_noise = mb.median_rms.size() != 0;

    std::unordered_map<Eigen::Index, Eigen::Index> array_to_index;
    for (Eigen::Index i=0; i<calib.arrays.size(); ++i) {
        array_to_index[calib.arrays(i)] = i;
    }

    bool warned_rms = false;
    bool warned_flux = false;
    bool warned_adaptive = false;

    double row_offset = fruit_loops_legacy_center ? (mb.n_rows / 2.0) : ((mb.n_rows - 1) / 2.0);
    double col_offset = fruit_loops_legacy_center ? (mb.n_cols / 2.0) : ((mb.n_cols - 1) / 2.0);

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; ++i) {
        // current detector index in apt
        auto map_index = map_indices(i);
        auto array_id = static_cast<Eigen::Index>(calib.apt["array"](i));
        auto array_it = array_to_index.find(array_id);
        if (array_it == array_to_index.end()) {
            logger->error("array {} not found in calib arrays for fruit loops", array_id);
            std::exit(EXIT_FAILURE);
        }
        auto array_pos = array_it->second;

        if (map_index < 0 || map_index >= static_cast<Eigen::Index>(mb.signal.size())) {
            logger->error("map index {} out of range for fruit loops (signal maps: {})", map_index, mb.signal.size());
            std::exit(EXIT_FAILURE);
        }

        double adaptive_support_radius_rad = std::numeric_limits<double>::quiet_NaN();
        bool use_adaptive_support = false;
        bool have_adaptive_support_center = false;
        double adaptive_support_lat = 0.0;
        double adaptive_support_lon = 0.0;
        if (fruit_loops_adaptive_support_radius_rad.size() ==
                static_cast<Eigen::Index>(mb.signal.size()) &&
            map_index < fruit_loops_adaptive_support_radius_rad.size()) {
            adaptive_support_radius_rad =
                fruit_loops_adaptive_support_radius_rad(map_index);
            use_adaptive_support =
                std::isfinite(adaptive_support_radius_rad) &&
                adaptive_support_radius_rad > 0.0;
        }
        if (use_adaptive_support &&
            fruit_loops_source_valid.size() == static_cast<Eigen::Index>(mb.signal.size()) &&
            map_index < fruit_loops_source_valid.size() &&
            fruit_loops_source_valid(map_index) != 0 &&
            map_index < fruit_loops_source_lat.size() &&
            map_index < fruit_loops_source_lon.size()) {
            adaptive_support_lat = fruit_loops_source_lat(map_index);
            adaptive_support_lon = fruit_loops_source_lon(map_index);
            have_adaptive_support_center =
                std::isfinite(adaptive_support_lat) &&
                std::isfinite(adaptive_support_lon);
        }

        // check if detector is not flagged
        if (calib.apt["flag"](i) == 0 && (in.flags.data.col(i).array() == 0).any()) {
            double az_off = calib.apt["x_t"](i);
            double el_off = calib.apt["y_t"](i);

            // calc tangent plane pointing
            auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off, pixel_axes,
                                                              in.pointing_offsets_arcsec.data, map_grouping);

            // get map buffer row and col indices for lat and lon vectors
            Eigen::VectorXd irows = lat.array()/mb.pixel_size_rad + row_offset;
            Eigen::VectorXd icols = lon.array()/mb.pixel_size_rad + col_offset;

            // loop through data points
            for (Eigen::Index j=0; j<n_pts; ++j) {
                const double map_row = irows(j);
                const double map_col = icols(j);

                bool on_image = false;
                double signal = 0.0;
                Eigen::Index trunc_ir = 0;
                Eigen::Index trunc_ic = 0;

                if (fruit_loops_interp_mode == "trunc") {
                    // Legacy v4.x behavior: cast to integer pixel indices (truncate toward zero).
                    trunc_ir = static_cast<Eigen::Index>(map_row);
                    trunc_ic = static_cast<Eigen::Index>(map_col);
                    on_image = !in.flags.data(j, i) &&
                               trunc_ir >= 0 && trunc_ir < mb.n_rows &&
                               trunc_ic >= 0 && trunc_ic < mb.n_cols;
                    if (on_image) {
                        signal = mb.signal[map_index](trunc_ir, trunc_ic);
                    }
                }
                else {
                    // check if current sample is on the image and add to the timestream
                    on_image = !in.flags.data(j,i) && map_row >= 0.0 && map_col >= 0.0 &&
                               map_row <= static_cast<double>(mb.n_rows - 1) &&
                               map_col <= static_cast<double>(mb.n_cols - 1);
                    if (on_image) {
                        if (fruit_loops_interp_mode == "jinc") {
                            signal = sample_map_jinc(mb.signal[map_index], array_id, map_row, map_col);
                        }
                        else if (fruit_loops_interp_mode == "nearest") {
                            const auto ir = std::clamp(static_cast<Eigen::Index>(std::llround(map_row)),
                                                       Eigen::Index{0}, mb.n_rows - 1);
                            const auto ic = std::clamp(static_cast<Eigen::Index>(std::llround(map_col)),
                                                       Eigen::Index{0}, mb.n_cols - 1);
                            signal = mb.signal[map_index](ir, ic);
                        }
                        else {
                            signal = sample_map_bilinear(mb.signal[map_index], map_row, map_col);
                        }
                    }
                }

                if (on_image) {
                    // check whether we should include pixel
                    bool run_pix_s2n = false;
                    bool run_pix_flux = false;
                    bool run_pix_adaptive = false;

                    double rms = std::numeric_limits<double>::quiet_NaN();
                    bool have_rms = false;
                    if (run_noise) {
                        if (mb.median_rms.size() == calib.arrays.size()) {
                            rms = mb.median_rms(array_pos);
                            have_rms = std::isfinite(rms) && rms > 0;
                        }
                        else if (mb.median_rms.size() > map_index) {
                            if (!warned_rms) {
                                logger->warn("median_rms size ({}) does not match arrays ({}) - using map index fallback",
                                             mb.median_rms.size(), calib.arrays.size());
                                warned_rms = true;
                            }
                            rms = mb.median_rms(map_index);
                            have_rms = std::isfinite(rms) && rms > 0;
                        }
                        else if (!warned_rms) {
                            logger->warn("median_rms size ({}) insufficient for map index {}; disabling S/N gate",
                                         mb.median_rms.size(), map_index);
                            warned_rms = true;
                        }
                    }

                    double flux_limit = 0.0;
                    bool have_flux = false;
                    if (fruit_loops_flux.size() == calib.arrays.size()) {
                        flux_limit = fruit_loops_flux(array_pos);
                        have_flux = std::isfinite(flux_limit) && std::abs(flux_limit) > 0.0;
                    }
                    else if (array_id < fruit_loops_flux.size()) {
                        if (!warned_flux) {
                            logger->warn("fruit_loops_flux size ({}) does not match arrays ({}); using array id indexing",
                                         fruit_loops_flux.size(), calib.arrays.size());
                            warned_flux = true;
                        }
                        flux_limit = fruit_loops_flux(array_id);
                        have_flux = std::isfinite(flux_limit) && std::abs(flux_limit) > 0.0;
                    }
                    else if (!warned_flux) {
                        logger->warn("fruit_loops_flux size ({}) insufficient for array {}; disabling flux gate",
                                     fruit_loops_flux.size(), array_id);
                        warned_flux = true;
                    }

                    double adaptive_limit = std::numeric_limits<double>::quiet_NaN();
                    bool have_adaptive = false;
                    if (fruit_loops_adaptive_threshold.size() == static_cast<Eigen::Index>(mb.signal.size()) &&
                        map_index < fruit_loops_adaptive_threshold.size()) {
                        adaptive_limit = fruit_loops_adaptive_threshold(map_index);
                        have_adaptive = std::isfinite(adaptive_limit) && adaptive_limit > 0.0;
                    }
                    else if ((fruit_loops_peak_fraction_limit > 0.0 ||
                              fruit_loops_local_snr_floor > 0.0) &&
                             !warned_adaptive) {
                        logger->warn("fruit loops adaptive threshold unavailable; disabling adaptive gate");
                        warned_adaptive = true;
                    }

                    const bool have_s2n = have_rms && std::abs(fruit_loops_sig2noise) > 0.0;
                    bool run_adaptive_support = true;
                    if (use_adaptive_support) {
                        run_adaptive_support = false;
                        if (have_adaptive_support_center) {
                            const double dlat = lat(j) - adaptive_support_lat;
                            const double dlon = lon(j) - adaptive_support_lon;
                            run_adaptive_support =
                                std::sqrt(dlat * dlat + dlon * dlon) <=
                                adaptive_support_radius_rad;
                        }
                    }
                    if (fruit_mode == "upper") {
                        run_pix_s2n = have_s2n && (signal / rms >= fruit_loops_sig2noise);
                        run_pix_flux = have_flux && (signal >= flux_limit);
                        run_pix_adaptive =
                            have_adaptive && run_adaptive_support &&
                            (signal >= adaptive_limit);
                    }
                    else if (fruit_mode == "lower") {
                        run_pix_s2n = have_s2n && (signal / rms <= fruit_loops_sig2noise);
                        run_pix_flux = have_flux && (signal <= flux_limit);
                        run_pix_adaptive =
                            have_adaptive && run_adaptive_support &&
                            (signal <= -std::abs(adaptive_limit));
                    }
                    else if (fruit_mode == "both") {
                        run_pix_s2n = have_s2n && (std::abs(signal / rms) >= std::abs(fruit_loops_sig2noise));
                        run_pix_flux = have_flux && (std::abs(signal) >= std::abs(flux_limit));
                        run_pix_adaptive =
                            have_adaptive && run_adaptive_support &&
                            (std::abs(signal) >= adaptive_limit);
                    }

                    // if signal flux is higher than S/N limit or flux limit
                    if (run_pix_s2n || run_pix_flux || run_pix_adaptive) {
                        // add/subtract signal pixel from signal timestream
                        in.scans.data(j,i) += factor * signal;
                        // add/subtract kernel pixel from kernel timestream
                        if (run_kernel) {
                            double kernel_value;
                            if (fruit_loops_interp_mode == "jinc") {
                                kernel_value = sample_map_jinc(mb.kernel[map_index], array_id,
                                                               map_row, map_col);
                            }
                            else if (fruit_loops_interp_mode == "trunc") {
                                kernel_value = mb.kernel[map_index](trunc_ir, trunc_ic);
                            }
                            else if (fruit_loops_interp_mode == "nearest") {
                                const auto ir = std::clamp(static_cast<Eigen::Index>(std::llround(map_row)),
                                                           Eigen::Index{0}, mb.n_rows - 1);
                                const auto ic = std::clamp(static_cast<Eigen::Index>(std::llround(map_col)),
                                                           Eigen::Index{0}, mb.n_cols - 1);
                                kernel_value = mb.kernel[map_index](ir, ic);
                            }
                            else {
                                kernel_value = sample_map_bilinear(mb.kernel[map_index], map_row,
                                                                   map_col);
                            }
                            in.kernel.data(j,i) += factor * kernel_value;
                        }
                    }
                }
            }
        }
    }
}

template <TCDataKind tcdata_t, class calib_t>
auto TCProc::remove_bad_dets(TCData<tcdata_t, Eigen::MatrixXd> &in, calib_t &calib, std::string map_grouping) {

    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    // only run if limits are not zero
    if (lower_inv_var_factor !=0 || upper_inv_var_factor !=0) {
        logger->info("removing outlier dets");
        auto &window_diag = remove_bad_dets_window_summary_by_scan[in.index.data];
        if (window_diag.size() != static_cast<std::size_t>(in.scans.data.cols())) {
            window_diag.assign(static_cast<std::size_t>(in.scans.data.cols()),
                               RemoveBadDetsWindowDiagSummary{});
        }

        auto infer_dt_sec = [&]() {
            auto it = in.tel_data.data.find("TelTime");
            if (it == in.tel_data.data.end() || it->second.size() < 2) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            std::vector<double> dt;
            dt.reserve(static_cast<std::size_t>(it->second.size() - 1));
            for (Eigen::Index i = 1; i < it->second.size(); ++i) {
                const double delta = it->second(i) - it->second(i - 1);
                if (std::isfinite(delta) && delta > 0.0) {
                    dt.push_back(delta);
                }
            }
            if (dt.empty()) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return tula::alg::median(Eigen::Map<Eigen::VectorXd>(dt.data(), dt.size()));
        };

        auto vector_quantile = [](std::vector<double> values, double q) {
            if (values.empty()) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            std::sort(values.begin(), values.end());
            q = std::clamp(q, 0.0, 1.0);
            const double pos = q * static_cast<double>(values.size() - 1);
            const auto lo = static_cast<std::size_t>(std::floor(pos));
            const auto hi = static_cast<std::size_t>(std::ceil(pos));
            if (lo == hi) {
                return values[lo];
            }
            const double frac = pos - static_cast<double>(lo);
            return values[lo] * (1.0 - frac) + values[hi] * frac;
        };

        const double dt_sec = infer_dt_sec();
        Eigen::Index window_samples = in.scans.data.rows();
        if (std::isfinite(dt_sec) && dt_sec > 0.0 && remove_bad_dets_window_sec > 0.0) {
            window_samples = std::max<Eigen::Index>(
                8, static_cast<Eigen::Index>(std::llround(remove_bad_dets_window_sec / dt_sec)));
        }
        window_samples = std::min<Eigen::Index>(window_samples, in.scans.data.rows());
        window_samples = std::max<Eigen::Index>(1, window_samples);

        auto summarize_windows = [&](const auto &scans, const auto &flags) {
            RemoveBadDetsWindowDiagSummary summary;
            if (scans.size() <= 0 || flags.size() != scans.size()) {
                return summary;
            }

            summary.n_total_windows = static_cast<int>((scans.size() + window_samples - 1) / window_samples);
            std::vector<double> inv_vars;
            std::vector<double> flagged_fracs;
            inv_vars.reserve(static_cast<std::size_t>(summary.n_total_windows));
            flagged_fracs.reserve(static_cast<std::size_t>(summary.n_total_windows));

            for (Eigen::Index start = 0; start < scans.size(); start += window_samples) {
                const Eigen::Index stop = std::min<Eigen::Index>(scans.size(), start + window_samples);
                const Eigen::Index len = stop - start;
                if (len <= 0) {
                    continue;
                }
                int n_flagged = 0;
                for (Eigen::Index i = start; i < stop; ++i) {
                    if (flags(i)) {
                        ++n_flagged;
                    }
                }
                const double flagged_frac = static_cast<double>(n_flagged) / static_cast<double>(len);
                flagged_fracs.push_back(flagged_frac);

                Eigen::VectorXd scan_window = scans.segment(start, len);
                Eigen::Matrix<bool, Eigen::Dynamic, 1> flag_window = flags.segment(start, len);
                const double stddev = engine_utils::calc_std_dev(scan_window, flag_window);
                if (std::isfinite(stddev) && stddev > 0.0) {
                    inv_vars.push_back(std::pow(stddev, -2));
                }
            }

            summary.n_valid_windows = static_cast<int>(inv_vars.size());
            if (summary.n_total_windows > 0) {
                summary.valid_window_fraction =
                    static_cast<double>(summary.n_valid_windows) /
                    static_cast<double>(summary.n_total_windows);
            }
            if (!inv_vars.empty()) {
                summary.inv_var_median = vector_quantile(inv_vars, 0.5);
                summary.inv_var_q10 = vector_quantile(inv_vars, 0.1);
                summary.inv_var_q90 = vector_quantile(inv_vars, 0.9);
            }
            if (!flagged_fracs.empty()) {
                summary.flagged_frac_median = vector_quantile(flagged_fracs, 0.5);
                summary.flagged_frac_max = *std::max_element(flagged_fracs.begin(), flagged_fracs.end());
                const auto n_heavy = std::count_if(
                    flagged_fracs.begin(), flagged_fracs.end(),
                    [](double v) { return std::isfinite(v) && v >= 0.5; });
                summary.heavily_flagged_window_fraction =
                    static_cast<double>(n_heavy) /
                    static_cast<double>(flagged_fracs.size());
            }
            return summary;
        };

        // number of detectors
        Eigen::Index n_dets = in.scans.data.cols();

        // get grouping
        auto grp_limits = get_grouping("array", calib, n_dets);

        in.n_dets_low = 0;
        in.n_dets_high = 0;

        // loop through group limits
        for (auto const& [key, val] : grp_limits) {
            // control for iteration
            bool keep_going = true;
            Eigen::Index n_iter = 0;

            while (keep_going) {
                // number of unflagged detectors
                Eigen::Index n_good_dets = 0;

                // get good dets in group
                for (Eigen::Index j=std::get<0>(grp_limits[key]); j<std::get<1>(grp_limits[key]); ++j) {
                    if (calib.apt["flag"](j)==0 && (in.flags.data.col(j).array()==0).any()) {
                        n_good_dets++;
                    }
                }

                Eigen::VectorXd det_std_dev(n_good_dets);
                Eigen::VectorXI dets(n_good_dets);
                std::vector<double> finite_inv_vars;
                finite_inv_vars.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(n_good_dets, 0)));
                Eigen::Index k = 0;

                // collect standard deviation from good detectors
                for (Eigen::Index j=std::get<0>(grp_limits[key]); j<std::get<1>(grp_limits[key]); ++j) {
                    Eigen::Index det_index = j;
                    if (calib.apt["flag"](det_index)==0 && (in.flags.data.col(j).array()==0).any()) {
                        // make Eigen::Maps for each detector's scan
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                            in.scans.data.col(j).data(), in.scans.data.rows());
                        Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                            in.flags.data.col(j).data(), in.flags.data.rows());

                        if (map_grouping == "detector" && mask_radius_arcsec > 0.0) {
                            Eigen::Matrix<bool, Eigen::Dynamic, 1> masked_flags = flags;
                            double az_off = calib.apt["x_t"](det_index);
                            double el_off = calib.apt["y_t"](det_index);
                            auto [lat, lon] = engine_utils::calc_det_pointing(
                                in.tel_data.data,
                                az_off,
                                el_off,
                                std::string{"altaz"},
                                in.pointing_offsets_arcsec.data,
                                map_grouping);
                            double source_lat = 0.0;
                            double source_lon = 0.0;
                            resolve_mask_center_rad(in, calib, map_grouping, det_index,
                                                    source_lat, source_lon);
                            const double radius_rad = mask_radius_arcsec * ASEC_TO_RAD;
                            for (Eigen::Index sample = 0; sample < masked_flags.size(); ++sample) {
                                const double dlat = lat(sample) - source_lat;
                                const double dlon = lon(sample) - source_lon;
                                if (std::sqrt(dlat * dlat + dlon * dlon) < radius_rad) {
                                    masked_flags(sample) = true;
                                }
                            }
                            det_std_dev(k) = engine_utils::calc_std_dev(scans, masked_flags);
                            if (n_iter == 0) {
                                window_diag[static_cast<std::size_t>(det_index)] =
                                    summarize_windows(scans, masked_flags);
                            }
                        }
                        else {
                            // calc standard deviation
                            det_std_dev(k) = engine_utils::calc_std_dev(scans, flags);
                            if (n_iter == 0) {
                                window_diag[static_cast<std::size_t>(det_index)] =
                                    summarize_windows(scans, flags);
                            }
                        }

                        // convert to 1/variance so it is a weight
                        if (std::isfinite(det_std_dev(k)) && det_std_dev(k) > 0.0) {
                            det_std_dev(k) = std::pow(det_std_dev(k),-2);
                            finite_inv_vars.push_back(det_std_dev(k));
                        }
                        else {
                            det_std_dev(k) = 0;
                        }

                        dets(k) = j;
                        k++;
                    }
                }

                if (finite_inv_vars.empty()) {
                    logger->warn("array {} iter {}: skipped inv var cut; no finite positive detector variances", key, n_iter);
                    break;
                }

                Eigen::Map<Eigen::VectorXd> finite_inv_var_map(
                    finite_inv_vars.data(),
                    static_cast<Eigen::Index>(finite_inv_vars.size()));
                // get median inverse variance
                double median_std_dev = tula::alg::median(finite_inv_var_map);
                if (!std::isfinite(median_std_dev) || median_std_dev <= 0.0) {
                    logger->warn("array {} iter {}: skipped inv var cut; median inverse variance is {}", key, n_iter, median_std_dev);
                    break;
                }

                int n_dets_low = 0;
                int n_dets_high = 0;

                // loop through good detectors and flag those that have std devs beyond the limits
                for (Eigen::Index j=0; j<n_good_dets; ++j) {
                    Eigen::Index det_index = dets(j);
                    // only run if unflagged already
                    if (calib.apt["flag"](det_index)==0) {
                        // flag those below limit
                        if ((det_std_dev(j) < (lower_inv_var_factor*median_std_dev)) && lower_inv_var_factor!=0) {
                            in.flags.data.col(dets(j)).setOnes();
                            if (map_grouping=="detector") {
                                calib_scan.apt["flag"](det_index) = 1;
                            }
                            in.n_dets_low++;
                            n_dets_low++;
                        }

                        // flag those above limit
                        if ((det_std_dev(j) > (upper_inv_var_factor*median_std_dev)) && upper_inv_var_factor!=0) {
                            in.flags.data.col(dets(j)).setOnes();
                            if (map_grouping=="detector") {
                                calib_scan.apt["flag"](det_index) = 1;
                            }
                            in.n_dets_high++;
                            n_dets_high++;
                        }
                    }
                }

                logger->info("array {} iter {}: {}/{} dets below inv var limit. {}/{} dets above inv var limit.", key, n_iter,
                            n_dets_low, n_good_dets, n_dets_high, n_good_dets);

                // increment iteration
                n_iter++;
                // check if no more detectors are above limit
                if ((n_dets_low==0 && n_dets_high==0) || n_iter > iter_lim) {
                    keep_going = false;
                }
            }
        }
        // set up scan calib
        calib_scan.setup();
    }

    return std::move(calib_scan);
}

template <TCDataKind tcdata_t, class calib_t, typename Derived>
auto TCProc::remove_uncorrelated(TCData<tcdata_t, Eigen::MatrixXd> &in, calib_t &calib, std::string map_grouping) {
    //Eigen::Index n_dets = in.scans.data.cols();
    //Eigen::Index n_pts = in.scans.data.rows();

    // make copy of flags
    //Eigen::MatrixXd f = abs(flags.derived().template cast<double> ().array() - 1);

    // container for covariance matrix
    //Eigen::MatrixXd pca_cov(n_dets, n_dets);

    // calculate the covariance matrix
    //pca_cov.noalias() = ((scans.adjoint() * scans).array()).matrix();// / denom.array()).matrix();
}

template <TCProc::SourceType source_type, TCDataKind tcdata_t, typename Derived, typename apt_t>
void TCProc::add_gaussian(TCData<tcdata_t, Eigen::MatrixXd> &in, Eigen::DenseBase<Derived> &params, std::string &pixel_axes,
                          std::string &map_grouping, apt_t &apt, double pixel_size_rad, Eigen::Index n_rows, Eigen::Index n_cols) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; ++i) {
        // detector index in apt
        auto det_index = i;
        // map index
        auto map_index = in.map_indices.data(i);

        double az_off = apt["x_t"](det_index);
        double el_off = apt["y_t"](det_index);

        // get pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off, pixel_axes,
                                                          in.pointing_offsets_arcsec.data, map_grouping);

        // get parameters from current map
        double amp = params(map_index,0);
        // rows
        double off_lat = params(map_index,2);
        // cols
        double off_lon = params(map_index,1);
        // row fwhm
        double sigma_lat = params(map_index,4);
        // col fwhm
        double sigma_lon = params(map_index,3);
        // rot angle
        double rot_ang = params(map_index,5);

        // use maximum of sigmas due to atmospheric cleaning
        double sigma = std::max(sigma_lat, sigma_lon);

        // subtract source
        if constexpr (source_type==NegativeGaussian) {
            amp = -amp;
        }

        // rescale offsets and stddev to on-sky units
        off_lat = pixel_size_rad*(off_lat - (n_rows - 1)/2.0);
        off_lon = pixel_size_rad*(off_lon - (n_cols - 1)/2.0);

        // convert to on-sky units
        sigma_lon = pixel_size_rad*sigma;
        sigma_lat = pixel_size_rad*sigma;
        sigma = pixel_size_rad*sigma;

        // get 2d elliptical gaussian angles
        auto cost2 = cos(rot_ang) * cos(rot_ang);
        auto sint2 = sin(rot_ang) * sin(rot_ang);
        auto sin2t = sin(2. * rot_ang);
        auto xstd2 = sigma * sigma;
        auto ystd2 = sigma * sigma;
        auto a = - 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
        auto b = - 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
        auto c = - 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));

        // calculate distance to source to truncate it
        //auto dist = ((lat.array() - off_lat).pow(2) + (lon.array() - off_lon).pow(2)).sqrt();

        Eigen::VectorXd gauss(n_pts);
        // make timestream from 2d gaussian
        for (Eigen::Index j=0; j<n_pts; ++j) {
            gauss(j) = amp*exp(pow(lon(j) - off_lon, 2) * a +
                                 (lon(j) - off_lon) * (lat(j) - off_lat) * b +
                                 pow(lat(j) - off_lat, 2) * c);
        }

        // check for bad fit?
        if (!gauss.array().isNaN().any()) {
            // add gaussian to detector scan
            in.scans.data.col(i) = in.scans.data.col(i).array() + gauss.array();
        }
    }
}

template <TCDataKind tcdata_t, class calib_t>
bool TCProc::resolve_mask_center_rad(const TCData<tcdata_t, Eigen::MatrixXd> &in,
                                     const calib_t &calib, std::string_view map_grouping,
                                     Eigen::Index det_index, double &source_lat,
                                     double &source_lon) const {

    source_lat = 0.0;
    source_lon = 0.0;

    if (det_index >= 0 && det_index < in.map_indices.data.size()) {
        const Eigen::Index map_index = in.map_indices.data(det_index);
        if (map_index >= 0 &&
            map_index < fruit_loops_source_valid.size() &&
            fruit_loops_source_valid(map_index) != 0 &&
            map_index < fruit_loops_source_lat.size() &&
            map_index < fruit_loops_source_lon.size()) {
            const double lat = fruit_loops_source_lat(map_index);
            const double lon = fruit_loops_source_lon(map_index);
            if (std::isfinite(lat) && std::isfinite(lon)) {
                source_lat = lat;
                source_lon = lon;
                return true;
            }
        }
    }

    if (map_grouping == "detector") {
        auto x_it = calib.apt.find("x_t_raw");
        auto y_it = calib.apt.find("y_t_raw");
        if (x_it == calib.apt.end() || y_it == calib.apt.end()) {
            x_it = calib.apt.find("x_t");
            y_it = calib.apt.find("y_t");
        }
        if (x_it != calib.apt.end() && y_it != calib.apt.end() &&
            det_index >= 0 &&
            det_index < x_it->second.size() &&
            det_index < y_it->second.size()) {
            const double x_arcsec = x_it->second(det_index);
            const double y_arcsec = y_it->second(det_index);
            if (std::isfinite(x_arcsec) && std::isfinite(y_arcsec)) {
                source_lat = y_arcsec * ASEC_TO_RAD;
                source_lon = x_arcsec * ASEC_TO_RAD;
                return true;
            }
        }
    }

    return false;
}

template <TCDataKind tcdata_t, class calib_t>
auto TCProc::mask_region(TCData<tcdata_t, Eigen::MatrixXd> &in, calib_t &calib, std::string pixel_axes, std::string map_grouping,
                         int n_pts, int n_dets, int start_index) {

    // copy of tel data
    std::map<std::string, Eigen::VectorXd> tel_data_copy;

    // populate copy of tel data
    for (const auto &[key,val]: in.tel_data.data) {
        tel_data_copy[key] = in.tel_data.data[key].segment(0,n_pts);
    }

    // copy of pointing offsets
    std::map<std::string, Eigen::VectorXd> pointing_offset_copy;

    // populate copy of pointing offsets
    for (const auto &[key,val]: in.pointing_offsets_arcsec.data) {
        pointing_offset_copy[key] = in.pointing_offsets_arcsec.data[key].segment(0,n_pts);
    }

    // make a copy of the timestream flags
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> masked_flags = in.flags.data.block(0, start_index, n_pts, n_dets);

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; ++i) {
        // current detector index in apt
        auto det_index = i + start_index;

        double az_off = calib.apt["x_t"](det_index);
        double el_off = calib.apt["y_t"](det_index);

        // calc tangent plane pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(tel_data_copy, az_off, el_off, pixel_axes,
                                                          pointing_offset_copy, map_grouping);

        double source_lat = 0.0;
        double source_lon = 0.0;
        resolve_mask_center_rad(in, calib, map_grouping, det_index, source_lat, source_lon);

        // distance to the masked source region
        auto dist = ((lat.array() - source_lat).pow(2) +
                     (lon.array() - source_lon).pow(2)).sqrt();

        // loop through samples
        for (Eigen::Index j=0; j<n_pts; ++j) {
            // flag samples within radius as bad
            if (dist(j) < mask_radius_arcsec*ASEC_TO_RAD) {
                masked_flags(j,i) = 1;
            }
        }
    }

    return std::move(masked_flags);
}

template <TCDataKind tcdata_t, class calib_t, typename pointing_offset_t>
void TCProc::append_base_to_netcdf(netCDF::NcFile &fo, TCData<tcdata_t, Eigen::MatrixXd> &in, std::string map_grouping,
                                   std::string &pixel_axes, pointing_offset_t &pointing_offsets_arcsec, calib_t &calib,
                                   bool apply_det_offsets, Eigen::Index scan_row_index, bool output_outer_scan) {
    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();

    // get absolute coords
    double cra, cdec;
    fo.getVar("SourceRa").getVar(&cra);
    fo.getVar("SourceDec").getVar(&cdec);

    // get dimensions
    NcDim n_pts_dim = fo.getDim("n_pts");
    NcDim n_dets_dim = fo.getDim("n_dets");

    // number of samples currently in file
    unsigned long n_pts_exists = n_pts_dim.getSize();
    // number of detectors currently in file
    unsigned long n_dets_exists = n_dets_dim.getSize();

    // start indices for data
    std::vector<std::size_t> start_index = {n_pts_exists, 0};
    // size for data
    std::vector<std::size_t> size = {1, TULA_SIZET(n_dets)};

    // start index for telescope data
    std::vector<std::size_t> start_index_tel = {n_pts_exists};
    // size for telescope data
    std::vector<std::size_t> size_tel = {TULA_SIZET(n_pts)};

    // start index for apt table
    std::vector<std::size_t> start_index_apt = {0};
    // size for apt
    std::vector<std::size_t> size_apt = {1};

    // get timestream variables
    NcVar signal_v = fo.getVar("signal");
    NcVar flags_v = fo.getVar("flags");
    NcVar kernel_v = fo.getVar("kernel");

    // detector tangent plane pointing
    NcVar det_lat_v = fo.getVar("det_lat");
    NcVar det_lon_v = fo.getVar("det_lon");

    // detector absolute pointing
    NcVar det_ra_v = fo.getVar("det_ra");
    NcVar det_dec_v = fo.getVar("det_dec");

    const bool write_det_pointing = (!det_lat_v.isNull() && !det_lon_v.isNull()) ||
                                    (!det_ra_v.isNull() && !det_dec_v.isNull());
    Eigen::MatrixXd lat, lon;
    if (write_det_pointing) {
        // tangent plane pointing for each detector
        lat.resize(n_pts, n_dets);
        lon.resize(n_pts, n_dets);

        // loop through detectors and get tangent plane pointing
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // detector index in apt
            auto det_index = i;
            double az_off = calib.apt["x_t"](det_index);
            double el_off = calib.apt["y_t"](det_index);

            // get tangent pointing
            auto [det_lat, det_lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off, pixel_axes,
                                                                      pointing_offsets_arcsec, map_grouping, apply_det_offsets);
            lat.col(i) = std::move(det_lat);
            lon.col(i) = std::move(det_lon);
        }
    }

    // append data (doing this per row is way faster than transposing
    // and populating them at once)
    for (std::size_t i=0; i<TULA_SIZET(n_pts); ++i) {
        start_index[0] = n_pts_exists + i;
        // append scans
        if (tod_output_mini) {
            Eigen::VectorXf scans = in.scans.data.row(i).template cast<float>();
            signal_v.putVar(start_index, size, scans.data());
        }
        else {
            Eigen::VectorXd scans = in.scans.data.row(i);
            signal_v.putVar(start_index, size, scans.data());
        }

        // append flags
        if (tod_output_mini) {
            Eigen::Matrix<signed char, 1, Eigen::Dynamic> flags_byte =
                in.flags.data.row(i).template cast<signed char>();
            flags_v.putVar(start_index, size, flags_byte.data());
        }
        else {
            Eigen::VectorXi flags_int = in.flags.data.row(i).template cast<int> ();
            flags_v.putVar(start_index, size, flags_int.data());
        }

        // append kernel
        if (!kernel_v.isNull()) {
            Eigen::VectorXd kernel = in.kernel.data.row(i);
            kernel_v.putVar(start_index, size, kernel.data());
        }

        if (write_det_pointing) {
            // append detector latitudes
            Eigen::VectorXd lat_row = lat.row(i);
            if (!det_lat_v.isNull()) {
                det_lat_v.putVar(start_index, size, lat_row.data());
            }

            // append detector longitudes
            Eigen::VectorXd lon_row = lon.row(i);
            if (!det_lon_v.isNull()) {
                det_lon_v.putVar(start_index, size, lon_row.data());
            }

            if (pixel_axes == "radec" && !det_ra_v.isNull() && !det_dec_v.isNull()) {
                // get absolute pointing
                auto [dec, ra] = engine_utils::tangent_to_abs(lat_row, lon_row, cra, cdec);

                // append detector ra
                det_ra_v.putVar(start_index, size, ra.data());

                // append detector dec
                det_dec_v.putVar(start_index, size, dec.data());
            }
        }
    }

    // append telescope
    for (auto const& x: in.tel_data.data) {
        NcVar tel_data_v = fo.getVar(x.first);
        tel_data_v.putVar(start_index_tel, size_tel, x.second.data());
    }

    // append pointing offsets
    for (auto const& x: in.pointing_offsets_arcsec.data) {
        NcVar offset_v = fo.getVar("pointing_offset_"+x.first);
        offset_v.putVar(start_index_tel, size_tel, x.second.data());
    }

    // append hwpr angle
    if (calib.run_hwpr) {
        NcVar hwpr_v = fo.getVar("hwpr");
        hwpr_v.putVar(start_index_tel, size_tel, in.hwpr_angle.data.data());
    }


    // overwrite apt table (can be updated between beammap iterations)
    for (auto const& x: calib.apt) {
        netCDF::NcVar apt_v = fo.getVar("apt_" + x.first);
        if (!apt_v.isNull()) {
            for (std::size_t i=0; i<TULA_SIZET(n_dets_exists); ++i) {
                start_index_apt[0] = i;
                apt_v.putVar(start_index_apt, size_apt, &calib.apt[x.first](i));
            }
        }
    }

    // vector to hold current scan indices
    Eigen::VectorXd scan_indices(2);
    Eigen::VectorXi raw_scan_indices(4);
    const Eigen::Index scan_row = (scan_row_index >= 0) ? scan_row_index : in.index.data;

    // if not on first scan, grab last scan and add size of current scan
    if (scan_row > 0) {
        if (output_outer_scan) {
            Eigen::VectorXi previous_raw_scan_indices(4);
            std::vector<std::size_t> raw_scan_indices_start_index = {TULA_SIZET(scan_row-1), 0};
            std::vector<std::size_t> raw_scan_indices_size = {1, 4};
            fo.getVar("raw_scan_indices").getVar(
                raw_scan_indices_start_index, raw_scan_indices_size, previous_raw_scan_indices.data());
            scan_indices(0) = previous_raw_scan_indices(3) + 1;
            scan_indices(1) = scan_indices(0) + in.scans.data.rows() - 1;
        }
        else {
            // start indices for data
            std::vector<std::size_t> scan_indices_start_index = {TULA_SIZET(scan_row-1), 0};
            // size for data
            std::vector<std::size_t> scan_indices_size = {1, 2};
            fo.getVar("scan_indices").getVar(scan_indices_start_index, scan_indices_size, scan_indices.data());

            scan_indices = scan_indices.array() + in.scans.data.rows();
        }
    }

    // otherwise, use size of this scan
    else {
        scan_indices(0) = 0;
        scan_indices(1) = in.scans.data.rows() - 1;
    }

    if (output_outer_scan && in.scan_indices.data.size() >= 4) {
        const Eigen::Index outer_start = static_cast<Eigen::Index>(scan_indices(0));
        const Eigen::Index outer_end = static_cast<Eigen::Index>(scan_indices(1));
        const Eigen::Index inner_offset =
            std::max<Eigen::Index>(0, in.scan_indices.data(0) - in.scan_indices.data(2));
        const Eigen::Index inner_len =
            std::max<Eigen::Index>(0, in.scan_indices.data(1) - in.scan_indices.data(0) + 1);
        const Eigen::Index inner_start =
            std::min<Eigen::Index>(outer_end, outer_start + inner_offset);
        const Eigen::Index inner_end =
            std::min<Eigen::Index>(outer_end, inner_start + std::max<Eigen::Index>(0, inner_len - 1));
        raw_scan_indices << static_cast<int>(inner_start), static_cast<int>(inner_end),
                            static_cast<int>(outer_start), static_cast<int>(outer_end);
        scan_indices(0) = inner_start;
        scan_indices(1) = inner_end;
    }
    else {
        raw_scan_indices << static_cast<int>(scan_indices(0)), static_cast<int>(scan_indices(1)),
                            static_cast<int>(scan_indices(0)), static_cast<int>(scan_indices(1));
    }

    // add current raw scan indices row (output timebase)
    std::vector<std::size_t> raw_scan_indices_start_index = {TULA_SIZET(scan_row), 0};
    std::vector<std::size_t> raw_scan_indices_size = {1, 4};
    NcVar raw_scan_indices_v = fo.getVar("raw_scan_indices");
    raw_scan_indices_v.putVar(raw_scan_indices_start_index, raw_scan_indices_size, raw_scan_indices.data());

    // add current scan indices row
    std::vector<std::size_t> scan_indices_start_index = {TULA_SIZET(scan_row), 0};
    std::vector<std::size_t> scan_indices_size = {1, 2};
    NcVar scan_indices_v = fo.getVar("scan_indices");
    scan_indices_v.putVar(scan_indices_start_index, scan_indices_size,scan_indices.data());

    // add mapping to original scan number (1-based)
    std::vector<std::size_t> output_scan_index_start_index = {TULA_SIZET(scan_row)};
    std::vector<std::size_t> output_scan_index_size = {1};
    NcVar output_scan_index_v = fo.getVar("output_scan_index");
    int output_scan_index = static_cast<int>(in.index.data + 1);
    output_scan_index_v.putVar(output_scan_index_start_index, output_scan_index_size, &output_scan_index);

    auto write_scan_int = [&](const std::string &name, int value) {
        NcVar v = fo.getVar(name);
        if (!v.isNull()) {
            v.putVar(output_scan_index_start_index, output_scan_index_size, &value);
        }
    };
    auto write_scan_double = [&](const std::string &name, double value) {
        NcVar v = fo.getVar(name);
        if (!v.isNull()) {
            v.putVar(output_scan_index_start_index, output_scan_index_size, &value);
        }
    };
    write_scan_int("tod_filter_edge_guard_pre_samples", in.status.filter_edge_guard_pre_samples);
    write_scan_int("tod_filter_edge_guard_post_samples", in.status.filter_edge_guard_post_samples);
    write_scan_int("tod_filter_edge_guard_flagged_samples", in.status.filter_edge_guard_flagged_samples);
    write_scan_double("tod_filter_edge_guard_flagged_frac", in.status.filter_edge_guard_flagged_frac);
}
} // namespace timestream
