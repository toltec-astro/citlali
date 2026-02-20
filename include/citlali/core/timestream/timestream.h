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
    // compact rtc TOD output mode (float signal, byte flags, no per-detector pointing/kernel vars)
    bool tod_output_mini = false;

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
    // save all iterations
    bool save_all_iters;

    // map buffer for map to tod approach
    mapmaking::MapBuffer tod_mb;

    // number of weight outlier iterations
    int iter_lim = 0;

    // upper and lower inv var limits for outliers
    double lower_inv_var_factor, upper_inv_var_factor;

    // mask radius in arcseconds
    double mask_radius_arcsec;

    // create a map buffer from a citlali reduction directory
    template <class calib_t>
    void load_mb(std::string, std::string, calib_t &, const std::string &,
                 const std::string & = "", double = 0.0);

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

    // flag a region around the center of the map
    template <TCDataKind tcdata_t, class calib_t>
    auto mask_region(TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, std::string, std::string, int, int, int);

    // append time chunk params common to rtcs and ptcs
    template <TCDataKind tcdata_t, class calib_t, typename pointing_offset_t>
    void append_base_to_netcdf(netCDF::NcFile &, TCData<tcdata_t, Eigen::MatrixXd> &, std::string,
                               std::string &, pointing_offset_t &, calib_t &, bool apply_det_offsets = false,
                               Eigen::Index scan_row_index = -1);
};

template <class calib_t>
void TCProc::load_mb(std::string filepath, std::string noise_filepath, calib_t &calib,
                     const std::string &expected_map_grouping, const std::string &expected_pixel_axes,
                     double expected_pixel_size_rad) {

    namespace fs = std::filesystem;

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
                if (path_str.find("_noise_citlali.fits") == std::string::npos) {
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

    double expected_row = tod_mb.n_rows / 2.0;
    double expected_col = tod_mb.n_cols / 2.0;
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

    // calculate coverage bool map
    for (int i=0; i<tod_mb.weight.size(); ++i) {
        // get weight threshold for current map
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = tod_mb.calc_cov_region(i);
        // if weight is less than threshold, set to zero, otherwise set to one
        auto cov_bool = (tod_mb.weight[i].array() < weight_threshold).select(zeros,ones);
        tod_mb.signal[i] = tod_mb.signal[i].array() * cov_bool.array();
        if (!tod_mb.kernel.empty()) {
            tod_mb.kernel[i] = tod_mb.kernel[i].array() * cov_bool.array();
        }
    }
    // clear weight maps to save memory
    std::vector<Eigen::MatrixXd>().swap(tod_mb.weight);
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

    double row_offset = (mb.n_rows)/2.;
    double col_offset = (mb.n_cols)/2.;

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
                // row and col pixel from signal image
                Eigen::Index ir = static_cast<Eigen::Index>(std::llround(irows(j)));
                Eigen::Index ic = static_cast<Eigen::Index>(std::llround(icols(j)));

                // check if current sample is on the image and add to the timestream
                if (!in.flags.data(j,i) && (ir >= 0) && (ir < mb.n_rows) && (ic >= 0) && (ic < mb.n_cols)) {
                    double signal = mb.signal[map_index](ir,ic);
                    // check whether we should include pixel
                    bool run_pix_s2n = false;
                    bool run_pix_flux = false;

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
                        have_flux = true;
                    }
                    else if (array_id < fruit_loops_flux.size()) {
                        if (!warned_flux) {
                            logger->warn("fruit_loops_flux size ({}) does not match arrays ({}); using array id indexing",
                                         fruit_loops_flux.size(), calib.arrays.size());
                            warned_flux = true;
                        }
                        flux_limit = fruit_loops_flux(array_id);
                        have_flux = true;
                    }
                    else if (!warned_flux) {
                        logger->warn("fruit_loops_flux size ({}) insufficient for array {}; disabling flux gate",
                                     fruit_loops_flux.size(), array_id);
                        warned_flux = true;
                    }

                    if (fruit_mode == "upper") {
                        run_pix_s2n = have_rms && (signal / rms >= fruit_loops_sig2noise);
                        run_pix_flux = have_flux && (signal >= flux_limit);
                    }
                    else if (fruit_mode == "lower") {
                        run_pix_s2n = have_rms && (signal / rms <= fruit_loops_sig2noise);
                        run_pix_flux = have_flux && (signal <= flux_limit);
                    }
                    else if (fruit_mode == "both") {
                        run_pix_s2n = have_rms && (std::abs(signal / rms) >= std::abs(fruit_loops_sig2noise));
                        run_pix_flux = have_flux && (std::abs(signal) >= std::abs(flux_limit));
                    }

                    // if signal flux is higher than S/N limit or flux limit
                    if (run_pix_s2n || run_pix_flux) {
                        // add/subtract signal pixel from signal timestream
                        in.scans.data(j,i) += factor * signal;
                        // add/subtract kernel pixel from kernel timestream
                        if (run_kernel) {
                            in.kernel.data(j,i) += factor * mb.kernel[map_index](ir,ic);
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

                        // calc standard deviation
                        det_std_dev(k) = engine_utils::calc_std_dev(scans, flags);

                        // convert to 1/variance so it is a weight
                        if (det_std_dev(k) !=0) {
                            det_std_dev(k) = std::pow(det_std_dev(k),-2);
                        }
                        else {
                            det_std_dev(k) = 0;
                        }

                        dets(k) = j;
                        k++;
                    }
                }

                // get median standard deviation
                double median_std_dev = tula::alg::median(det_std_dev);

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
        off_lat = pixel_size_rad*(off_lat - (n_rows)/2);
        off_lon = pixel_size_rad*(off_lon - (n_cols)/2);

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

        // distance to center of map
        auto dist = (lat.array().pow(2) + lon.array().pow(2)).sqrt();

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
                                   bool apply_det_offsets, Eigen::Index scan_row_index) {
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
        // start indices for data
        std::vector<std::size_t> scan_indices_start_index = {TULA_SIZET(scan_row-1), 0};
        // size for data
        std::vector<std::size_t> scan_indices_size = {1, 2};
        fo.getVar("scan_indices").getVar(scan_indices_start_index, scan_indices_size, scan_indices.data());

        scan_indices = scan_indices.array() + in.scans.data.rows();
    }

    // otherwise, use size of this scan
    else {
        scan_indices(0) = 0;
        scan_indices(1) = in.scans.data.rows() - 1;
    }

    raw_scan_indices << static_cast<int>(scan_indices(0)), static_cast<int>(scan_indices(1)),
                        static_cast<int>(scan_indices(0)), static_cast<int>(scan_indices(1));

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
}
} // namespace timestream
