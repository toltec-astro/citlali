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
            SPDLOG_TRACE("check data item config\n{}", config);
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
           unresolved
          );
    // clang-format on
    using CalItemTypes = tula::meta::cases<CalItemType::array_prop_table,
                                           CalItemType::photometry,
                                           CalItemType::astrometry,
                                           CalItemType::unresolved>;
    struct ArrayPropTable;
    struct PhotometryCalibInfo;
    struct AstrometryCalibInfo;

    struct CalItem;
    template <auto type>
    using cal_item_t = tula::meta::switch_t<
        type,
        tula::meta::case_t<CalItemType::array_prop_table, ArrayPropTable>,
        tula::meta::case_t<CalItemType::photometry, PhotometryCalibInfo>,
        tula::meta::case_t<CalItemType::astrometry, AstrometryCalibInfo>,
        tula::meta::case_t<CalItemType::unresolved, CalItem>>;
    using cal_item_var_t =
        std::variant<
        std::monostate,
        cal_item_t<CalItemType::array_prop_table>,
        cal_item_t<CalItemType::photometry>,
        cal_item_t<CalItemType::astrometry>
        >;

    struct ArrayPropTable : ConfigMapper<ArrayPropTable> {
        using Base = ConfigMapper<ArrayPropTable>;
        ArrayPropTable(config_t config)
            : Base{std::move(config)},
              m_filepath(this->config().get_str("filepath")) {}

        static auto check_config(config_t &config)
            -> std::optional<std::string> {
            std::vector<std::string> missing_keys;
            SPDLOG_TRACE("check array prop table config\n{}", config);
            if (!config.has("filepath")) {
                missing_keys.push_back("filepath");
            }
            if (missing_keys.empty()) {
                return std::nullopt;
            }
            return fmt::format("invalid or missing keys={}", missing_keys);
        }
        const std::string &filepath() const { return m_filepath; }

        template <typename OStream>
        friend auto operator<<(OStream &os, const ArrayPropTable &d)
            -> decltype(auto) {
            return os << fmt::format("ArrayPropTable(filepath={})",
                                     d.filepath());
        }

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
            SPDLOG_TRACE("check photometry calib info\n{}", config);
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
            SPDLOG_TRACE("check astrometry calib info\n{}", config);
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
            SPDLOG_TRACE("check cal item config\n{}", config);
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

        template <typename OStream>
        friend auto operator<<(OStream &os, const CalItem &d)
            -> decltype(auto) {
            // SPDLOG_DEBUG("calitem type: {}", d.type());
            // return os << fmt::format("CalItem(type={}, typestr={})",
            // d.type(),
            //                          d.typestr());
            return os << fmt::format("CalItem(typestr={})", d.typestr());
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
        SPDLOG_TRACE("check raw obs config\n{}", config);
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

    void collect_cal_items();
};

TULA_ENUM_REGISTER(RawObs::CalItemType);

namespace fmt {

template <typename T>
struct formatter<std::reference_wrapper<T>, char, void>
    : tula::fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const std::reference_wrapper<T> &ref,
                FormatContext &ctx) noexcept -> decltype(ctx.out()) {
        return format_to(ctx.out(), "{}", ref.get());
    }
};

template <>
struct formatter<RawObs, char, void>
    : tula::fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const RawObs &obs, FormatContext &ctx) noexcept
        -> decltype(ctx.out()) {
        return format_to(ctx.out(), "RawObs(name={}, n_data_items={})",
                         obs.name(), obs.n_data_items());
    }
};

template <>
struct formatter<RawObs::DataItem, char, void>
    : tula::fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const RawObs::DataItem &item, FormatContext &ctx) noexcept
        -> decltype(ctx.out()) {
        return format_to(ctx.out(), "DataItem(interface={}, filepath={})",
                         item.interface(), item.filepath());
    }
};

} // namespace fmt

// collect data items impl
void RawObs::collect_data_items() {
    m_data_items.clear();
    std::vector<DataItem> data_items{};
    auto node_data_items = this->config().get_node("data_items");
    auto n_data_items = node_data_items.size();
    for (std::size_t i = 0; i < n_data_items; ++i) {
        SPDLOG_TRACE("add data item {} of {}", i, n_data_items);
        data_items.emplace_back(
            config_t{node_data_items[i], this->config().filepath()});
    }
    m_data_items = std::move(data_items);
    SPDLOG_DEBUG("collected n_data_items={}\n{}", this->n_data_items(),
                 this->data_items());
    // update the data indices
    m_kidsdata_indices.clear();
    m_teldata_index.reset();
    m_hwpdata_index.reset();
    std::smatch m;
    for (std::size_t i = 0; i < m_data_items.size(); ++i) {
        if (std::regex_match(m_data_items[i].interface(), m,
                             re_interface_kidsdata)) {
            m_kidsdata_indices.push_back(i);
        }
        if (std::regex_match(m_data_items[i].interface(), m,
                             re_interface_teldata)) {
            if (m_teldata_index.has_value()) {
                throw std::runtime_error("found too many telescope data items");
            }
            m_teldata_index = i;
        }
        if (std::regex_match(m_data_items[i].interface(), m,
                             re_interface_hwpdata)) {
            if (m_hwpdata_index.has_value()) {
                throw std::runtime_error(
                    "found too many halfwave plate data items");
            }
            m_hwpdata_index = i;
        }
    }
    if (!m_teldata_index) {
        throw std::runtime_error("no telescope data item found");
    }
    // The hwp data is optional
    if (!m_hwpdata_index) {
        SPDLOG_TRACE("no hwp data item found");
    }
    SPDLOG_TRACE("kidsdata_indices={} teldata_index={} hwpdata_index={}",
                 m_kidsdata_indices, m_teldata_index, m_hwpdata_index);
    SPDLOG_TRACE("kidsdata={} teldata={} hwpdata={}", kidsdata(), teldata(),
                 hwpdata());
}

// collect cal items impl

void RawObs::collect_cal_items() {
    m_cal_items.clear();
    std::vector<CalItem> cal_items{};
    auto node_cal_items = this->config().get_node("cal_items");
    auto n_cal_items = node_cal_items.size();
    for (std::size_t i = 0; i < n_cal_items; ++i) {
        cal_items.emplace_back(
            config_t{node_cal_items[i], this->config().filepath()});
    }
    m_cal_items = std::move(cal_items);
    SPDLOG_DEBUG("collected n_cal_items={}\n{}", this->n_cal_items(),
                 this->cal_items());
    // update the data indices
    m_apt_index.reset();
    m_phot_cal_index.reset();
    m_astro_cal_index.reset();
    for (std::size_t i = 0; i < m_cal_items.size(); ++i) {
        if (m_cal_items[i].is_type<CalItemType::array_prop_table>()) {
            if (m_apt_index.has_value()) {
                throw std::runtime_error("found too many array prop tables");
            }
            m_apt_index = i;
        }
        if (m_cal_items[i].is_type<CalItemType::photometry>()) {
            if (m_phot_cal_index.has_value()) {
                throw std::runtime_error("found too many photometry calib info.");
            }
            m_phot_cal_index = i;
        }
        if (m_cal_items[i].is_type<CalItemType::astrometry>()) {
            if (m_astro_cal_index.has_value()) {
                throw std::runtime_error("found too many astrometry calib info.");
            }
            m_astro_cal_index = i;
        }
    }
    if (!m_apt_index) {
        throw std::runtime_error("no array prop table found");
    }
    SPDLOG_TRACE("apt_index={}", m_apt_index);
    SPDLOG_TRACE("apt={}", array_prop_table());
}
/**
 * @brief The Coordinator struct
 * This wraps around the config object and provides
 * high level methods in various ways to setup the MPI runtime
 * with node-local and cross-node environment.
 */
struct SeqIOCoordinator : ConfigMapper<SeqIOCoordinator> {
    using Base = ConfigMapper<SeqIOCoordinator>;
    using index_t = predefs::index_t;
    using shape_t = predefs::shape_t;
    using input_t = RawObs;
    using payload_t = RawObs::DataItem;

    SeqIOCoordinator(config_t config) : Base{std::move(config)} {
        collect_inputs();
    }

    // io_buffer
    using payloads_buffer_data_t = predefs::data_t;

    auto n_inputs() const { return m_inputs.size(); }

    const std::vector<input_t> &inputs() const { return m_inputs; };

    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        if (config.has_list("inputs")) {
            return std::nullopt;
        }
        return fmt::format("invalid or missing key \"inputs\"");
    }
    friend std::ostream &operator<<(std::ostream &os,
                                    const SeqIOCoordinator &co) {
        return os << fmt::format("SeqIOCoordinator(n_inputs={})",
                                 co.n_inputs());
    }

private:
    std::vector<input_t> m_inputs{}; // stores all the inputs

    // collect input data
    void collect_inputs() {
        m_inputs.clear();
        std::vector<input_t> inputs;
        auto node_inputs = this->config().get_node("inputs");
        auto n_inputs = node_inputs.size();
        for (std::size_t i = 0; i < n_inputs; ++i) {
            SPDLOG_TRACE("add input {} of {}", i, n_inputs);
            inputs.emplace_back(
                config_t{node_inputs[i], this->config().filepath()});
        }
        m_inputs = std::move(inputs);

        SPDLOG_DEBUG("collected n_inputs={}\n{}", this->n_inputs(),
                     this->inputs());
    }
};
