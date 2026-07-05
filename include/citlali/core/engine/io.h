#pragma once

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

#include <mutex>

#if defined(__has_include)
#if __has_include(<hdf5.h>)
#include <hdf5.h>
#define CITLALI_ENGINE_IO_HAS_HDF5 1
#elif __has_include(<hdf5/serial/hdf5.h>)
#include <hdf5/serial/hdf5.h>
#define CITLALI_ENGINE_IO_HAS_HDF5 1
#else
#define CITLALI_ENGINE_IO_HAS_HDF5 0
#endif
#else
#define CITLALI_ENGINE_IO_HAS_HDF5 0
#endif

#include <citlali/core/timestream/timestream.h>

namespace predefs {

/// This namespace contains global settings for common types and constants.

// We need to be careful about the int type used here as we may
// have very large array in the memory.
// check eigen index type
static_assert(std::is_same_v<std::ptrdiff_t, Eigen::Index>,
              "UNEXPECTED EIGEN INDEX TYPE");
using index_t = std::ptrdiff_t;
using shape_t = Eigen::Matrix<index_t, 2, 1>;
using data_t = double;

// IO spec
namespace kidsdata = kids::toltec;

// TcData is the data structure of which RTCData and PTCData are a part
using timestream::TCData;

// Selects the type of TCData
using timestream::TCDataKind;

inline std::mutex &netcdf_io_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline void suppress_hdf5_diagnostics_for_this_thread() {
#if CITLALI_ENGINE_IO_HAS_HDF5
    static thread_local bool suppressed = false;
    if (!suppressed) {
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
        suppressed = true;
    }
#endif
}

} // namespace predefs

template <typename Derived>
using ConfigMapper =
    tula::config::ConfigValidatorMixin<Derived, tula::config::YamlConfig>;

/**
 * @brief The raw obs struct
 * This represents a single observation that contains a set of data items
 * and calibration items.
 */
struct RawObs : ConfigMapper<RawObs> {
    using Base = ConfigMapper<RawObs>;

    /**
     * @brief The DataItem struct
     * This represent a single data item that belongs to a particular
     * observation
     */
    struct DataItem : ConfigMapper<DataItem> {
        using Base = ConfigMapper<DataItem>;
        DataItem(config_t config)
            : Base{std::move(config)}, m_interface(this->config().get_str(
                                           std::tuple{"meta", "interface"})),
              m_filepath(this->config().get_filepath("filepath")) {}

        static auto check_config(config_t &config)
            -> std::optional<std::string> {
            std::vector<std::string> missing_keys;
            SPDLOG_INFO("check data item config\n{}", config);
            if (!config.has(std::tuple{"meta", "interface"})) {
                missing_keys.push_back("meta.interface");
            }
            if (!config.has("filepath")) {
                missing_keys.push_back("filepath");
            }
            if (missing_keys.empty()) {
                return std::nullopt;
            }
            return fmt::format("invalid or missing keys={}", missing_keys);
        }
        const std::string &interface() const { return m_interface; }
        const std::string &filepath() const { return m_filepath; }

    private:
        std::string m_interface{};
        std::string m_filepath{};
    };

    // clang-format off
    TULA_ENUM_DECL(CalItemType, int,
           array_prop_table,
           photometry,
           astrometry,
           flxscale_correction,
           unresolved
          );
    // clang-format on
    using CalItemTypes = tula::meta::cases<CalItemType::array_prop_table,
                                           CalItemType::photometry,
                                           CalItemType::astrometry,
                                           CalItemType::flxscale_correction,
                                           CalItemType::unresolved>;
    struct ArrayPropTable;
    struct PhotometryCalibInfo;
    struct AstrometryCalibInfo;
    struct FlxscaleCorrection;

    struct CalItem;
    template <auto type>
    using cal_item_t = tula::meta::switch_t<
        type,
        tula::meta::case_t<CalItemType::array_prop_table, ArrayPropTable>,
        tula::meta::case_t<CalItemType::photometry, PhotometryCalibInfo>,
        tula::meta::case_t<CalItemType::astrometry, AstrometryCalibInfo>,
        tula::meta::case_t<CalItemType::flxscale_correction, FlxscaleCorrection>,
        tula::meta::case_t<CalItemType::unresolved, CalItem>>;
    using cal_item_var_t =
        std::variant<
        std::monostate,
        cal_item_t<CalItemType::array_prop_table>,
        cal_item_t<CalItemType::photometry>,
        cal_item_t<CalItemType::astrometry>,
        cal_item_t<CalItemType::flxscale_correction>
        >;

    struct ArrayPropTable : ConfigMapper<ArrayPropTable> {
        using Base = ConfigMapper<ArrayPropTable>;
        ArrayPropTable(config_t config)
            : Base{std::move(config)},
              m_filepath(this->config().get_str("filepath")) {}

        static auto check_config(config_t &config)
            -> std::optional<std::string> {
            std::vector<std::string> missing_keys;
            SPDLOG_INFO("check array prop table config\n{}", config);
            if (!config.has("filepath")) {
                missing_keys.push_back("filepath");
            }
            if (missing_keys.empty()) {
                return std::nullopt;
            }
            return fmt::format("invalid or missing keys={}", missing_keys);
        }
        const std::string &filepath() const { return m_filepath; }

    private:
        std::string m_filepath{};
    };

    struct PhotometryCalibInfo : ConfigMapper<PhotometryCalibInfo> {
        using Base = ConfigMapper<PhotometryCalibInfo>;
        PhotometryCalibInfo(config_t config)
            : Base{std::move(config)}{}
        static auto check_config(config_t &config)
            -> std::optional<std::string> {
            std::vector<std::string> missing_keys;
            SPDLOG_INFO("check photometry calib info\n{}", config);
            // do the checks here
            if (missing_keys.empty()) {
                return std::nullopt;
            }
            return fmt::format("invalid or missing keys={}", missing_keys);
        }
        template <typename OStream>
        friend auto operator<<(OStream &os, const PhotometryCalibInfo &d)
            -> decltype(auto) {
            return os << fmt::format("PhotometryCalibInfo()");
        }
    };

    struct AstrometryCalibInfo : ConfigMapper<AstrometryCalibInfo> {
        using Base = ConfigMapper<AstrometryCalibInfo>;
        AstrometryCalibInfo(config_t config)
            : Base{std::move(config)}{}
        static auto check_config(config_t &config)
            -> std::optional<std::string> {
            std::vector<std::string> missing_keys;
            SPDLOG_INFO("check astrometry calib info\n{}", config);
            // do the checks here
            if (missing_keys.empty()) {
                return std::nullopt;
            }
            return fmt::format("invalid or missing keys={}", missing_keys);
        }
        template <typename OStream>
        friend auto operator<<(OStream &os, const AstrometryCalibInfo &d)
            -> decltype(auto) {
            return os << fmt::format("AstrometryCalibInfo()");
        }
    };

    struct FlxscaleCorrection : ConfigMapper<FlxscaleCorrection> {
        using Base = ConfigMapper<FlxscaleCorrection>;
        FlxscaleCorrection(config_t config)
            : Base{std::move(config)} {
            // accept either key name for convenience
            if (this->config().has("value")) {
                m_value = this->config().get_typed<double>("value");
            } else {
                m_value = this->config().get_typed<double>("flxscale_correction");
            }
        }
        static auto check_config(config_t &config)
            -> std::optional<std::string> {
            std::vector<std::string> missing_keys;
            SPDLOG_INFO("check flxscale correction info\n{}", config);
            if (!config.has("value") && !config.has("flxscale_correction")) {
                missing_keys.push_back("value");
            }
            if (missing_keys.empty()) {
                return std::nullopt;
            }
            return fmt::format(
                "invalid or missing keys={} (expected one of: value, "
                "flxscale_correction)",
                missing_keys);
        }
        double value() const { return m_value; }
        template <typename OStream>
        friend auto operator<<(OStream &os, const FlxscaleCorrection &d)
            -> decltype(auto) {
            return os << fmt::format("FlxscaleCorrection(value={})", d.value());
        }

    private:
        double m_value{1.0};
    };

    /// @breif a generic cal item holder
    struct CalItem : ConfigMapper<CalItem> {
        using Base = ConfigMapper<CalItem>;
        CalItem(config_t config)
            : Base{std::move(config)},
              m_typestr(this->config().get_str("type")) {
            resolve();
        }

        static auto check_config(config_t &config)
            -> std::optional<std::string> {
            std::vector<std::string> missing_keys;
            SPDLOG_INFO("check cal item config\n{}", config);
            if (!config.has("type")) {
                missing_keys.push_back("type");
            }
            if (missing_keys.empty()) {
                return std::nullopt;
            }
            return fmt::format("invalid or missing keys={}", missing_keys);
        }
        const std::string &typestr() const { return m_typestr; }

        auto type() const {
            if (auto opt_type_meta = CalItemType_meta::from_name(typestr());
                opt_type_meta.has_value()) {
                return opt_type_meta.value().value;
            }
            return CalItemType::unresolved;
        }

        template <auto type_>
        auto is_type() -> bool {
            return type() == type_;
        }

        template <auto type_>
        auto get() const -> const auto & {
            return std::get<cal_item_t<type_>>(m_cal_item);
        }

    private:
        std::string m_typestr{};
        cal_item_var_t m_cal_item{};
        void resolve() {
            tula::meta::switch_invoke<CalItemTypes>(
                [&](auto _) {
                    constexpr auto type_ = std::decay_t<decltype(_)>::value;
                    if constexpr (type_ == CalItemType::unresolved) {
                        m_cal_item = std::monostate{};
                    } else {
                        m_cal_item = cal_item_t<type_>{this->config()};
                    }
                },
                type());
        }
    };

    RawObs(config_t config)
        : Base{std::move(config)}, m_name{this->config().get_str(
                                       std::tuple{"meta", "name"})} {
        collect_data_items();
        collect_cal_items();
    }

    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        std::vector<std::string> missing_keys;
        SPDLOG_INFO("check raw obs config\n{}", config);
        if (!config.has(std::tuple{"meta", "name"})) {
            missing_keys.push_back("meta.name");
        }
        if (!config.has_list("data_items")) {
            missing_keys.push_back("data_items");
        }
        if (!config.has_list("cal_items")) {
            missing_keys.push_back("cal_items");
        }
        if (missing_keys.empty()) {
            return std::nullopt;
        }
        return fmt::format("invalid or missing keys={}", missing_keys);
    }
    const std::string &name() const { return m_name; }
    auto n_data_items() const { return m_data_items.size(); }
    const std::vector<DataItem> &data_items() const { return m_data_items; }
    const DataItem &teldata() const {
        return m_data_items[m_teldata_index.value()];
    }
    std::optional<DataItem> hwpdata() const {
        if (m_hwpdata_index) {
            return std::optional{
                DataItem{m_data_items[m_hwpdata_index.value()]}};
        }
        return std::nullopt;
    }

    auto kidsdata() const -> decltype(auto) {
        std::vector<std::reference_wrapper<const DataItem>> result{};
        for (auto i : m_kidsdata_indices) {
            result.push_back(std::cref(m_data_items[i]));
        }
        return result;
    }

    auto n_cal_items() const { return m_cal_items.size(); }
    const std::vector<CalItem> &cal_items() const { return m_cal_items; }
    const ArrayPropTable &array_prop_table() const {
        return m_cal_items[m_apt_index.value()]
            .get<CalItemType::array_prop_table>();
    }
    const PhotometryCalibInfo &photometry_calib_info() const {
        return m_cal_items[m_phot_cal_index.value()]
            .get<CalItemType::photometry>();
    }
    const AstrometryCalibInfo &astrometry_calib_info() const {
        return m_cal_items[m_astro_cal_index.value()]
            .get<CalItemType::astrometry>();
    }
    const FlxscaleCorrection *flxscale_correction() const {
        if (!m_flxscale_corr_index) {
            return nullptr;
        }
        return &m_cal_items[m_flxscale_corr_index.value()]
                    .get<CalItemType::flxscale_correction>();
    }

private:
    inline const static std::regex re_interface_kidsdata{"toltec\\d{1,2}"};
    inline const static std::regex re_interface_teldata{"lmt"};
    inline const static std::regex re_interface_hwpdata{"hwpr"};

    std::string m_name;
    std::vector<DataItem> m_data_items{};
    std::vector<std::size_t> m_kidsdata_indices{};
    std::optional<std::size_t> m_teldata_index{std::nullopt};
    std::optional<std::size_t> m_hwpdata_index{std::nullopt};

    void collect_data_items();
    std::vector<CalItem> m_cal_items{};
    std::optional<std::size_t> m_apt_index{std::nullopt};
    std::optional<std::size_t> m_phot_cal_index{std::nullopt};
    std::optional<std::size_t> m_astro_cal_index{std::nullopt};
    std::optional<std::size_t> m_flxscale_corr_index{std::nullopt};

    void collect_cal_items();
};

TULA_ENUM_REGISTER(RawObs::CalItemType);

namespace fmt {

template <typename T>
struct formatter<std::reference_wrapper<T>>
    : tula::fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const std::reference_wrapper<T> &ref,
                FormatContext &ctx) const noexcept -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "{}", ref.get());
    }
};

template <>
struct formatter<RawObs>
    : tula::fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const RawObs &obs, FormatContext &ctx) const noexcept
        -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "RawObs(name={}, n_data_items={})",
                         obs.name(), obs.n_data_items());
    }
};

template <>
struct formatter<RawObs::DataItem>
    : tula::fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const RawObs::DataItem &item, FormatContext &ctx) const noexcept
        -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "DataItem(interface={}, filepath={})",
                         item.interface(), item.filepath());
    }
};

template <>
struct formatter<RawObs::CalItem>
    : tula::fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const RawObs::CalItem &item, FormatContext &ctx) const noexcept
        -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "CalItem(typestr={})",
                         item.typestr());
    }
};

template <>
struct formatter<RawObs::ArrayPropTable>
    : tula::fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const RawObs::ArrayPropTable &apt, FormatContext &ctx) const noexcept
        -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "ArrayPropTable(filepath={})",
                         apt.filepath());
    }
};

template <>
struct formatter<RawObs::FlxscaleCorrection>
    : tula::fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const RawObs::FlxscaleCorrection &corr,
                FormatContext &ctx) const noexcept -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "FlxscaleCorrection(value={})",
                         corr.value());
    }
};

} // namespace fmt

#include <citlali/core/engine/detail/rawobs_collection_impl.h>
#include <citlali/core/engine/detail/seq_io_coordinator_impl.h>
