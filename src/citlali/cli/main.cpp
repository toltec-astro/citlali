#include <citlali_config/config.h>
#include <citlali_config/gitversion.h>
#include <citlali_config/default_config.h>
#include <kids/core/kidsdata.h>
#include <kids/sweep/fitter.h>
#include <kids/timestream/solver.h>
#include <kids/toltec/toltec.h>
#include <kidscpp_config/gitversion.h>
#include <tula/cli.h>
#include <tula/config/core.h>
#include <tula/config/flatconfig.h>
#include <tula/config/yamlconfig.h>
#include <tula/enum.h>
#include <tula/formatter/container.h>
#include <tula/formatter/enum.h>
#include <tula/grppi.h>
#include <tula/logging.h>
#include <tula/switch_invoke.h>

#include <cstdlib>
#include <omp.h>
#include <regex>
#include <tuple>

#include <citlali/core/engine/beammap.h>
#include <citlali/core/engine/lali.h>
#include <citlali/core/utils/constants.h>

using rc_t = tula::config::YamlConfig;

auto parse_args(int argc, char *argv[]) {
    // disable logger before parse
    spdlog::set_level(spdlog::level::off);
    using namespace tula::cli::clipp_builder;

    // some of the option specs
    auto ver_str =
        fmt::format("{} ({})", CITLALI_GIT_VERSION, CITLALI_BUILD_TIMESTAMP);
    auto kids_ver_str = fmt::format("kids {} ({})", KIDSCPP_GIT_VERSION,
                                    KIDSCPP_BUILD_TIMESTAMP);
    constexpr auto level_names = tula::logging::active_level_names;
    auto default_level_name = []() {
        auto v = spdlog::level::debug;
        if (v < tula::logging::active_level) {
            v = tula::logging::active_level;
        }
        return tula::logging::get_level_name(v);
    }();
    using ex_config = tula::grppi_utils::ex_config;
    // clang-format off
    auto parse = config_parser<rc_t, tula::config::FlatConfig>{};
    auto screen = tula::cli::screen{
    // =======================================================================
                      "citlali" , CITLALI_PROJECT_NAME, ver_str,
                                  CITLALI_PROJECT_DESCRIPTION};
    auto [cli, rc, cc] = parse([&](auto &r, auto &c) { return (
    // rc -- runtime config
    // cc -- cli config
    // =======================================================================
    c(p(           "h", "help"), "Print help information and exit."),
    c(p(             "version"), "Print version information and exit."),
    // =======================================================================
    r(             "config_file" , "The path of input config file. "
                                 "Multiple config file are merged in order.",
                                 opt_strs()),
    c(p(          "dump_config"), "Print the default config file to STDOUT."),
    // =======================================================================
              "common options" % g(
    c(p(      "l", "log_level"), "Set the log level.",
                                 default_level_name, list(level_names)),
    r(p(             "grppiex"), "GRPPI execution policy.",
                                 ex_config::default_mode(),
                                 list(ex_config::mode_names_supported())))
    // =======================================================================
    );}, screen, argc, argv);
    // clang-format on
    // SPDLOG_TRACE("cc: {}", cc.pformat());
    if (cc.get_typed<bool>("help")) {
        screen.manpage(cli);
        std::exit(EXIT_SUCCESS);
    } else if (cc.get_typed<bool>("version")) {
        screen.version();
        // also print the kids version
        fmt::print("{}\n", kids_ver_str);
        std::exit(EXIT_SUCCESS);
    }
    {
        auto log_level_str = cc.get_str("log_level");
        auto log_level = spdlog::level::from_str(log_level_str);
        spdlog::set_level(log_level);
        SPDLOG_INFO("reconfigure logger to level={}", log_level_str);
    }
    // pass on the runtime config
    return std::move(rc);
}

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
    inline const static std::regex re_interface_hwpdata{"hwp"};

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

/**
 * @brief The KIDs data solver struct
 * This wraps around the kids config
 */

bool extra_output = 0;
struct KidsDataProc : ConfigMapper<KidsDataProc> {
    using Base = ConfigMapper<KidsDataProc>;
    using Fitter = kids::SweepFitter;
    using Solver = kids::TimeStreamSolver;
    KidsDataProc(config_t config)
        : Base{std::move(config)},
          m_fitter{Fitter::Config{
              {"weight_window_type", this->config().get_str(std::tuple{
                                         "fitter", "weight_window", "type"})},
              {"weight_window_fwhm",
               this->config().get_typed<double>(
                   std::tuple{"fitter", "weight_window", "fwhm_Hz"})},
              {"modelspec",
               config.get_str(std::tuple{"fitter", "modelspec"})}}},
          m_solver{Solver::Config{
              {"fitreportdir", "/dev/null"},
              {"exmode", "seq"},
              {"extra_output", extra_output},
          }} {}

    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        std::vector<std::string> missing_keys;
        SPDLOG_TRACE("check kids data proc config\n{}", config);
        if (!config.has("fitter")) {
            missing_keys.push_back("fitter");
        }
        if (!config.has("solver")) {
            missing_keys.push_back("solver");
        }
        if (missing_keys.empty()) {
            return std::nullopt;
        }
        return fmt::format("invalid or missing keys={}", missing_keys);
    }

    auto get_data_item_meta(const RawObs::DataItem &data_item) {
        namespace kidsdata = predefs::kidsdata;
        auto source = data_item.filepath();
        auto [kind, meta] = kidsdata::get_meta<>(source);
        return meta;
    }

    auto get_rawobs_meta(const RawObs &rawobs) {
        std::vector<kids::KidsData<>::meta_t> result;
        for (const auto &data_item : rawobs.kidsdata()) {
            result.push_back(get_data_item_meta(data_item));
        }
        return result;
    }

    auto populate_rtc_meta(const RawObs &rawobs) {
        std::vector<kids::KidsData<>::meta_t> result;
        for (const auto &data_item : rawobs.kidsdata()) {
            result.push_back(get_data_item_meta(data_item));
        }
        return result;
    }

    auto reduce_data_item(const RawObs::DataItem &data_item,
                          const tula::container_utils::Slice<int> &slice) {
        SPDLOG_TRACE("kids reduce data_item {}", data_item);
        // read data
        namespace kidsdata = predefs::kidsdata;
        auto source = data_item.filepath();
        auto [kind, meta] = kidsdata::get_meta<>(source);
        if (!(kind & kids::KidsDataKind::TimeStream)) {
            throw std::runtime_error(
                fmt::format("wrong type of kids data {}", kind));
        }
        auto rts = kidsdata::read_data_slice<kids::KidsDataKind::RawTimeStream>(
            source, slice);
        auto result = this->solver()(rts, Solver::Config{});
        return result;
    }

    auto reduce_rawobs(const RawObs &rawobs,
                       const tula::container_utils::Slice<int> &slice) {
        SPDLOG_TRACE("kids reduce rawobs {}", rawobs);
        std::vector<kids::TimeStreamSolverResult> result;
        for (const auto &data_item : rawobs.kidsdata()) {
            result.push_back(reduce_data_item(data_item, slice));
        }
        return result;
    }

    auto load_data_item(const RawObs::DataItem &data_item,
                        const tula::container_utils::Slice<int> &slice) {
        SPDLOG_TRACE("kids reduce data_item {}", data_item);
        // read data
        namespace kidsdata = predefs::kidsdata;
        auto source = data_item.filepath();
        auto [kind, meta] = kidsdata::get_meta<>(source);
        if (!(kind & kids::KidsDataKind::TimeStream)) {
            throw std::runtime_error(
                fmt::format("wrong type of kids data {}", kind));
        }
        auto rts = kidsdata::read_data_slice<kids::KidsDataKind::RawTimeStream>(
            source, slice);
        return rts;
    }

    auto load_rawobs(const RawObs &rawobs,
                     const tula::container_utils::Slice<int> &slice) {
        SPDLOG_TRACE("kids load rawobs {}", rawobs);
        std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> result;
        for (const auto &data_item : rawobs.kidsdata()) {
            result.push_back(load_data_item(data_item, slice));
        }
        return result;
    }

    template <typename scanindices_t>
    auto populate_rtc(const RawObs &rawobs, scanindices_t &scanindex,
                      const int scanlength, const int n_detectors) {
        // call reduce rawobs, get the data into rtc
        auto slice = tula::container_utils::Slice<int>{
            scanindex(2), scanindex(3) + 1, std::nullopt};
        auto reduced = reduce_rawobs(rawobs, slice);

        Eigen::MatrixXd xs(scanlength, n_detectors);

        Eigen::Index i = 0;
        for (std::vector<kids::TimeStreamSolverResult>::iterator it =
                 reduced.begin();
             it != reduced.end(); ++it) {
            auto nrows = it->data_out.xs.data.rows();
            auto ncols = it->data_out.xs.data.cols();
            xs.block(0, i, nrows, ncols) = it->data_out.xs.data;
            i += ncols;
        }

        return std::move(xs);
    }

    template <typename loaded_t, typename scanindices_t>
    auto populate_rtc_load(loaded_t &loaded, scanindices_t &scanindex,
                           const int scanlength, const int n_detectors) {
        // call reduce rawobs, get the data into rtc
        auto slice = tula::container_utils::Slice<int>{
            scanindex(2), scanindex(3) + 1, std::nullopt};
        // auto loaded = load_rawobs(rawobs, slice);

        Eigen::MatrixXd xs(scanlength, n_detectors);

        Eigen::Index i = 0;
        for (std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>::
                 iterator it = loaded.begin();
             it != loaded.end(); ++it) {
            auto result = this->solver()(*it, Solver::Config{});
            auto nrows = result.data_out.xs.data.rows();
            auto ncols = result.data_out.xs.data.cols();
            xs.block(0, i, nrows, ncols) = result.data_out.xs.data;
            i += ncols;
        }

        return std::move(xs);
    }

    // TODO fix the const correctness
    Fitter &fitter() { return m_fitter; }
    Solver &solver() { return m_solver; }

    const Fitter &fitter() const { return m_fitter; }
    const Solver &solver() const { return m_solver; }

    template <typename OStream>
    friend OStream &operator<<(OStream &os, const KidsDataProc &kidsproc) {
        return os << fmt::format("KidsDataProc(fitter={}, solver={})",
                                 kidsproc.fitter().config.pformat(),
                                 kidsproc.solver().config.pformat());
    }

private:
    // fitter and solver
    Fitter m_fitter;
    Solver m_solver;
};

struct DummyEngine {
    template <typename OStream>
    friend OStream &operator<<(OStream &os, const DummyEngine &e) {
        return os << fmt::format("DummyEngine()");
    }
};

/**
 * @brief The time ordered data processing struct
 * This wraps around the lali config
 */

template <class EngineType>
struct TimeOrderedDataProc : ConfigMapper<TimeOrderedDataProc<EngineType>> {
    using Base = ConfigMapper<TimeOrderedDataProc<EngineType>>;
    using config_t = typename Base::config_t;
    using Engine = EngineType;
    using scanindicies_t = Eigen::MatrixXI;
    using map_extent_t = std::vector<double>;
    using map_coord_t = std::vector<Eigen::VectorXd>;
    using map_count_t = std::size_t;
    using array_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;
    using det_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;

    TimeOrderedDataProc(config_t config) : Base{std::move(config)} {}

    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        std::vector<std::string> missing_keys;
        SPDLOG_TRACE("check TOD proc config\n{}", config);
        if (!config.has("runtime")) {
            missing_keys.push_back("runtime");
        }
        if (!config.has("timestream")) {
            missing_keys.push_back("timestream");
        }
        if (!config.has("mapmaking")) {
            missing_keys.push_back("mapmaking");
        }
        if (!config.has("beammap")) {
            missing_keys.push_back("beammap");
        }
        if (!config.has("coadd")) {
            missing_keys.push_back("coadd");
        }
        if (missing_keys.empty()) {
            return std::nullopt;
        }
        return fmt::format("invalid or missing keys={}", missing_keys);
    }

    // get rows/cols for observation maps
    auto get_map_extent(const RawObs &rawobs) {
        map_extent_t map_extent;
        map_coord_t map_coord;

        auto [nr, nc, rcp, ccp] = engine().get_dims(
            engine().tel_meta_data, engine().calib_data, engine().scanindices,
            engine().ex_name, engine().reduction_type, engine().x_size_pix,
            engine().y_size_pix, engine().pointing_offsets);

        map_extent.push_back(nr);
        map_extent.push_back(nc);

        map_coord.push_back(rcp);
        map_coord.push_back(ccp);

        return std::tuple{map_extent, map_coord};
    }

    // get number of maps and grouping indices
    /*auto get_map_count(const RawObs &rawobs) {
        map_count_t map_count;
        array_indices_t array_indices;
        det_indices_t det_indices;

        array_indices.push_back(std::tuple{0, 0});

        Eigen::Index ai = 0;

        for (Eigen::Index i = 0; i < engine().calib_data["array"].size(); i++) {
            if (engine().calib_data["array"](i) == ai) {
                std::get<1>(array_indices.at(ai)) = i;
            } else {
                array_indices.push_back(std::tuple{i, 0});
                ai += 1;
            }
        }

        if ((std::strcmp("science", engine().reduction_type.c_str()) == 0) ||
            (std::strcmp("pointing", engine().reduction_type.c_str()) == 0)) {
            det_indices = array_indices;

        }

        else if ((std::strcmp("beammap", engine().reduction_type.c_str()) ==
                  0) ||
                 (std::strcmp("wyatt", engine().reduction_type.c_str()) == 0)) {
            for (Eigen::Index i = 0; i < engine().calib_data["array"].size();
                 i++) {
                det_indices.push_back(std::tuple{i, i + 1});
            }
        }

        map_count = det_indices.size();
        SPDLOG_INFO("map_count {}", map_count);

        Eigen::Index new_map_count = 3;
        if (engine().run_polarization) {
            new_map_count = new_map_count*3;
        }


        return std::tuple{new_map_count, array_indices, det_indices};
    }*/

    // get number of maps and grouping indices
    auto get_map_count(const RawObs &rawobs) {
        map_count_t map_count = 0;
        array_indices_t array_indices;
        det_indices_t det_indices;

        ToltecIO toltec_io;

        if (std::strcmp("beammap", engine().reduction_type.c_str()) == 0) {
            map_count = engine().ndet;
        }

        Eigen::Index narrays = 3;

        if ((std::strcmp("science", engine().reduction_type.c_str()) == 0) ||
            (std::strcmp("pointing", engine().reduction_type.c_str()) == 0)) {

            for (Eigen::Index arr=0; arr<narrays; arr++) {
                if ((engine().calib_data["array"].array()==arr).any()) {
                    map_count++;
                    engine().arrays[toltec_io.name_keys[arr]] = arr;
                }
            }
        }

        array_indices.push_back(std::tuple{0, 0});

        Eigen::Index ai = 0;

        for (Eigen::Index i = 0; i < engine().calib_data["array"].size(); i++) {
            if (engine().calib_data["array"](i) == ai) {
                std::get<1>(array_indices.at(ai)) = i;
            }
            else {
                array_indices.push_back(std::tuple{i, 0});
                ai += 1;
            }
        }

        if ((std::strcmp("science", engine().reduction_type.c_str()) == 0) ||
            (std::strcmp("pointing", engine().reduction_type.c_str()) == 0)) {
            det_indices = array_indices;
        }

        // multiply by number of stokes parameters
        map_count = map_count*engine().polarization.stokes_params.size();

        SPDLOG_INFO("map count {}", map_count);
        SPDLOG_INFO("array_indices {}", array_indices);
        SPDLOG_INFO("det_indices {}", det_indices);

        return std::tuple{map_count, array_indices, det_indices};
    }

    // get scan indices
    auto get_scanindicies(const RawObs &rawobs) {

        engine().get_scanindices(engine().tel_meta_data, engine().source_center,
                                 engine().map_pattern_type, engine().fsmp,
                                 engine().time_chunk, engine().filter.nterms);
    }

    // allocate the coadd map buffer
    void setup_coadd_map_buffer(const std::vector<map_coord_t> &map_coords,
                                const map_count_t &map_count) {

        engine().cmb.setup_maps(map_coords, map_count);
    }

    // allocate the noise map buffer
    void setup_noise_map_buffer(const map_count_t &map_count,
                                const Eigen::Index nrows,
                                const Eigen::Index ncols) {

        engine().nmb.setup_maps(map_count, nrows, ncols);
    }

    // allocate a single observation's map buffer
    auto setup_map_buffer(const map_extent_t &map_extent,
                          const map_coord_t &map_coord,
                          const map_count_t &map_count) {

        engine().mb.setup_maps(map_extent, map_coord, map_count,
                               engine().run_kernel, engine().map_grouping);
    }

    // TODO fix the const correctness
    Engine &engine() { return m_engine; }

    const Engine &engine() const { return m_engine; }

    template <typename OStream>
    friend OStream &operator<<(OStream &os,
                               const TimeOrderedDataProc &todproc) {
        return os << fmt::format("TimeOrderedDataProc(engine={})",
                                 todproc.engine());
    }

private:
    Engine m_engine;
};

// @brief Run citlali reduction.
/// @param rc The runtime config.
int run(const rc_t &rc) {
    using kids::KidsData;
    using kids::KidsDataKind;
    using tula::logging::timeit;
    SPDLOG_INFO("use KIDs data spec: {}", predefs::kidsdata::name);

    std::vector<std::string> config_filepaths;

    // load the yaml citlali config
    // this will merge the list of config files in rc
    tula::config::YamlConfig citlali_config;
    {
        auto node_config_files = rc.get_node("config_file");
        for (const auto & n: node_config_files) {
            auto filepath = n.as<std::string>();
            config_filepaths.push_back(filepath);
            SPDLOG_TRACE("load config from file {}", filepath);
            citlali_config = tula::config::merge(citlali_config, tula::config::YamlConfig::from_filepath(filepath));
        }
    }
    // auto citlali_config =
    //    tula::config::YamlConfig::from_filepath(rc.get_str("config_file"));

    SPDLOG_INFO("citlali config file {}", citlali_config);

    // set up the IO coorindator
    auto co = SeqIOCoordinator::from_config(citlali_config);

    // set up KIDs data proc
    auto kidsproc =
        KidsDataProc::from_config(citlali_config.get_config("kids"));

    // set up TOD proc
    using todproc_var_t =
        std::variant<std::monostate, TimeOrderedDataProc<Lali>,
                     TimeOrderedDataProc<Beammap>>;

    todproc_var_t todproc;

    // set todproc to variant depending on the config file map grouping
    // type

    // check if config file has a grouping parameter
    if (citlali_config.has(std::tuple{"runtime", "reduction_type"})) {
        try {
            auto reduction_type =
                citlali_config.get_str(std::tuple{"runtime", "reduction_type"});

            // check for science mode
            if (reduction_type == "science") {
                SPDLOG_INFO("reducing in science mode");
                todproc =
                    TimeOrderedDataProc<Lali>::from_config(citlali_config);
            }

            // check for pointing mode
            else if (reduction_type == "pointing") {
                SPDLOG_INFO("reducing in pointing mode");
                todproc =
                    TimeOrderedDataProc<Lali>::from_config(citlali_config);
            }

            // check for beammap mode
            else if (reduction_type == "beammap") {
                SPDLOG_INFO("reducing in beammap mode");
                todproc =
                    TimeOrderedDataProc<Beammap>::from_config(citlali_config);
            }

            else {
                std::vector<std::string> invalid_keys;
                // push back invalid keys into temp vector
                engine_utils::for_each_in_tuple(
                    std::tuple{"runtime", "reduction_type"},
                    [&](const auto &x) { invalid_keys.push_back(x); });

                std::cerr << fmt::format("invalid keys={}", invalid_keys)
                          << "\n";
                return EXIT_FAILURE;
            }

            // catch bad yaml type conversion and mark as invalid
        } catch (YAML::TypedBadConversion<std::string>) {
            std::vector<std::string> invalid_keys;
            // push back invalid keys into temp vector
            engine_utils::for_each_in_tuple(
                std::tuple{"mapmaking", "grouping"},
                [&](const auto &x) { invalid_keys.push_back(x); });

            std::cerr << fmt::format("invalid keys={}", invalid_keys) << "\n";
            return EXIT_FAILURE;
        }
    }

    // else mark as missing
    else {
        std::vector<std::string> missing_keys;
        // push back invalid keys into temp vector
        engine_utils::for_each_in_tuple(
            std::tuple{"mapmaking", "grouping"},
            [&](const auto &x) { missing_keys.push_back(x); });

        std::cerr << fmt::format("missing keys={}", missing_keys) << "\n";
        return EXIT_FAILURE;
    }

    // start the main process
    auto exitcode = std::visit(
        [&](auto &todproc) {
            using todproc_t = std::decay_t<decltype(todproc)>;

            // if todproc type is not one of the allowed std::variant states,
            // exit
            if constexpr (std::is_same_v<todproc_t, std::monostate>) {
                return EXIT_FAILURE;
            }

            else {
                // type definitions for map vectors
                using map_extent_t = typename todproc_t::map_extent_t;
                using map_coord_t = typename todproc_t::map_coord_t;
                using map_count_t = typename todproc_t::map_count_t;
                using array_indices_t = typename todproc_t::array_indices_t;

                // create vectors for map size and grouping parameters
                std::vector<map_extent_t> map_extents{};
                std::vector<map_coord_t> map_coords{};
                std::vector<map_count_t> map_counts{};
                std::vector<array_indices_t> array_indices{};
                std::vector<array_indices_t> det_indices{};

                // set up the coadded map buffer (CMB) by reading in each
                // observation
                for (const auto &rawobs : co.inputs()) {
                    // this is needed to figure out the data sample rate
                    // and number of detectors
                    auto rawobs_kids_meta = kidsproc.get_rawobs_meta(rawobs);

                    // get path to apt table
                    auto cal_path = rawobs.array_prop_table().filepath();
                    SPDLOG_INFO("cal_path {}", cal_path);

                    // load apt table into engine
                    todproc.engine().get_calib(cal_path);

                    // get sample rate
                    todproc.engine().fsmp = rawobs_kids_meta.back().get_typed<double>("fsmp");
                    SPDLOG_INFO("todproc.engine().fsmp {}",todproc.engine().fsmp);

                    // get config options from citlali_config
                    todproc.engine().from_config(citlali_config);

                    // exit if missing or invalid config options
                    if (!todproc.engine().missing_keys.empty() ||
                        !todproc.engine().invalid_keys.empty()) {
                        std::cerr << fmt::format("missing keys={}", todproc.engine().missing_keys) << "\n";
                        std::cerr << fmt::format("invalid keys={}", todproc.engine().invalid_keys) << "\n";
                        return EXIT_FAILURE;
                    }

                    // load telescope file
                    todproc.engine().get_telescope(rawobs.teldata().filepath());

                    // coadded exposure time
                    todproc.engine().c_t_exp += todproc.engine().tel_meta_params["t_exp"];

                    // calculate physical pointing vectors
                    todproc.engine().get_phys_pointing(todproc.engine().tel_meta_data, todproc.engine().source_center,
                                                       todproc.engine().map_type);

                    // get scanindices
                    todproc.get_scanindicies(rawobs);
                    SPDLOG_INFO("todproc.engine().scanindices {}", todproc.engine().scanindices);

                    // get map extents for each observation
                    SPDLOG_INFO("getting map extents");
                    {
                        tula::logging::scoped_timeit timer("getting map extents");
                        auto [me, mcoord] = todproc.get_map_extent(rawobs);
                        map_extents.push_back(std::move(me));
                        map_coords.push_back(std::move(mcoord));
                    }

                    // get map counts for each observation
                    SPDLOG_INFO("calculating map count");
                    auto [mc, ai, di] = todproc.get_map_count(rawobs);
                    map_counts.push_back(std::move(mc));
                    array_indices.push_back(std::move(ai));
                    det_indices.push_back(std::move(di));
                }

                // toltec i/o class for filename generation
                ToltecIO toltec_io;

                todproc.engine().redu_num = 0;
                std::stringstream ss_redu, ss_redu_next;

                ss_redu << std::setfill('0') << std::setw(2) << todproc.engine().redu_num;
                std::string hdname = "redu" + ss_redu.str() + "/";

                ss_redu_next << std::setfill('0') << std::setw(2) << todproc.engine().redu_num + 1;
                std::string hdname_next = "redu" + ss_redu_next.str() + "/";

                while (fs::exists(fs::status(todproc.engine().filepath + hdname)) ||
                       fs::exists(fs::status(todproc.engine().filepath + hdname_next))) {
                    todproc.engine().redu_num++;
                    std::stringstream ss_redu_i;
                    ss_redu_i << std::setfill('0') << std::setw(2) << todproc.engine().redu_num;
                    hdname = "redu" + ss_redu_i.str() + "/";

                    ss_redu_next << std::setfill('0') << std::setw(2) << todproc.engine().redu_num + 1;
                    hdname_next = "redu" + ss_redu_next.str() + "/";
                }

                toltec_io.setup_output_directory(todproc.engine().filepath, hdname);

                for (std::string config_filepath : config_filepaths) {
                    SPDLOG_INFO("config_filepath {}", config_filepath);
                    std::size_t found = config_filepath.rfind("/");
                    if (found!=std::string::npos) {
                        std::string config_name = config_filepath.substr(found);
                        SPDLOG_INFO("config_name {}",config_name);
                        fs::copy(config_filepath, todproc.engine().filepath + hdname + config_name);
                    }
                    else {
                        fs::copy(config_filepath, todproc.engine().filepath + hdname + config_filepath);
                    }

                }

                // set up coadded map buffer
                if (todproc.engine().run_coadd) {
                    SPDLOG_INFO("setup coadded map buffer");
                    todproc.setup_coadd_map_buffer(map_coords, map_counts.front());

                    // toltec i/o class for filename generation
                    ToltecIO toltec_io;

                    std::string rdname = hdname + "coadded/raw/";
                    toltec_io.setup_output_directory(todproc.engine().filepath,rdname);

                    std::string fdname = hdname + "coadded/filtered/";
                    if (todproc.engine().run_coadd_filter) {
                        toltec_io.setup_output_directory(todproc.engine().filepath, fdname);
                    }

                    // create files for each member of the array_indices group
                    // uses filepath from last config read
                    for (Eigen::Index i = 0; i < todproc.engine().cmb.map_count; i++) {
                        std::string coadd_filename;
                        // generate filename for coadded maps
                        coadd_filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu, ToltecIO::no_obs_type, ToltecIO::raw,
                            ToltecIO::obsnum_false>(todproc.engine().filepath + rdname, todproc.engine().obsnum, i);

                        // push the file classes into a vector for storage
                        FitsIO<fileType::write_fits, CCfits::ExtHDU *>coadd_fits_io(coadd_filename);
                        todproc.engine().coadd_fits_ios.push_back(std::move(coadd_fits_io));

                        if (todproc.engine().run_coadd_filter) {
                            // generate filename for filtered coadded maps
                            coadd_filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,ToltecIO::no_obs_type,
                                                                      ToltecIO::filtered,ToltecIO::obsnum_false>(
                                todproc.engine().filepath + fdname, todproc.engine().obsnum, i);

                            // push the file classes into a vector for storage
                            FitsIO<fileType::write_fits, CCfits::ExtHDU *> filtered_coadd_fits_ios(coadd_filename);
                            todproc.engine().filtered_coadd_fits_ios.push_back(std::move(filtered_coadd_fits_ios));
                        }

                        // check if noise maps requested
                        if (todproc.engine().run_noise) {
                            std::string noise_filename;
                            noise_filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu, ToltecIO::no_obs_type,
                                                                      ToltecIO::noise_raw,ToltecIO::obsnum_false>(
                                todproc.engine().filepath + rdname, todproc.engine().obsnum, i);

                            // push the file classes into a vector for storage
                            FitsIO<fileType::write_fits, CCfits::ExtHDU *> noise_fits_io(noise_filename);
                            todproc.engine().noise_fits_ios.push_back(std::move(noise_fits_io));

                            // check if filter is requested
                            if (todproc.engine().run_coadd_filter) {
                                noise_filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                    ToltecIO::no_obs_type, ToltecIO::noise_filtered, ToltecIO::obsnum_false>(
                                    todproc.engine().filepath + fdname, todproc.engine().obsnum, i);

                                // push the file classes into a vector for storage
                                FitsIO<fileType::write_fits, CCfits::ExtHDU *> filtered_noise_fits_io(noise_filename);
                                todproc.engine().filtered_noise_fits_ios.push_back(std::move(filtered_noise_fits_io));
                            }
                        }
                    }
                }

                // run the reduction for each observation
                for (std::size_t i = 0; i < co.n_inputs(); ++i) {
                    SPDLOG_INFO("starting reduction of observation {}/{}", i + 1, co.n_inputs());
                    const auto &rawobs = co.inputs()[i];

                    // keep track of what observation is being reduced
                    todproc.engine().nobs = i;

                    // set up map buffer for current observation
                    todproc.setup_map_buffer(map_extents[i], map_coords[i], map_counts[i]);

                    // this is needed to figure out the data sample rate
                    // and number of detectors
                    auto rawobs_kids_meta = kidsproc.get_rawobs_meta(rawobs);

                    // get path to apt table
                    auto cal_path = rawobs.array_prop_table().filepath();
                    SPDLOG_INFO("cal_path {}", cal_path);


                    // beammap source name
                    todproc.engine().beammap_source_name =
                        rawobs.photometry_calib_info().config().get_typed<std::string>(std::tuple{"beammap_source","name"});

                    // beammap source ra
                    todproc.engine().beammap_ra =
                        rawobs.photometry_calib_info().config().get_typed<double>(std::tuple{"beammap_source","ra_deg"});
                    // beammap source dec
                    todproc.engine().beammap_dec =
                        rawobs.photometry_calib_info().config().get_typed<double>(std::tuple{"beammap_source","dec_deg"});

                    Eigen::Index nfluxes =
                        rawobs.photometry_calib_info().config().get_node(std::tuple{"beammap_source","fluxes"}).size();

                    for (Eigen::Index nf=0; nf<nfluxes; nf++) {
                        // fluxes
                        todproc.engine().beammap_fluxes[toltec_io.name_keys[nf]] =
                            rawobs.photometry_calib_info().config().get_typed<double>(std::tuple{"beammap_source","fluxes",i,"value_mJy"});
                        // flux uncertainties
                        todproc.engine().beammap_uncer[toltec_io.name_keys[nf]] =
                            rawobs.photometry_calib_info().config().get_typed<double>(std::tuple{"beammap_source","fluxes",i,"uncertainty_mJy"});
                    }

                    // extinction model name
                    todproc.engine().beammap_source_name =
                        rawobs.photometry_calib_info().config().get_typed<std::string>(std::tuple{"extinction","model_name"});
                    // tau
                    todproc.engine().tau =
                        rawobs.photometry_calib_info().config().get_typed<double>(std::tuple{"extinction","tau_220"});

                    todproc.engine().pointing_offsets["az"] =
                        rawobs.astrometry_calib_info().config().get_typed<double>(std::tuple{"pointing_offsets",0,"value_arcsec"});

                    todproc.engine().pointing_offsets["alt"] =
                        rawobs.astrometry_calib_info().config().get_typed<double>(std::tuple{"pointing_offsets",1,"value_arcsec"});

                    // load apt table into engine
                    todproc.engine().get_calib(cal_path);

                    if (todproc.engine().run_polarization) {
                        SPDLOG_INFO("getting hwpdata");
                        auto hwp_filepath = rawobs.hwpdata()->filepath();
                        todproc.engine().get_hwp(hwp_filepath);
                    }

                    // get sample rate
                    todproc.engine().fsmp = rawobs_kids_meta.back().get_typed<double>("fsmp");
                    SPDLOG_INFO("todproc.engine().fsmp {}", todproc.engine().fsmp);

                    // get config options from citlali_config
                    todproc.engine().from_config(citlali_config);

                    // get obsnum for output filename
                    todproc.engine().obsnum = rawobs_kids_meta.back().get_typed<int>("obsid");

                    // load telescope file
                    todproc.engine().get_telescope(rawobs.teldata().filepath());

                    // calculate physical pointing vectors
                    todproc.engine().get_phys_pointing(todproc.engine().tel_meta_data, todproc.engine().source_center,
                                                       todproc.engine().map_type);
                    // get scanindices
                    todproc.get_scanindicies(rawobs);
                    SPDLOG_INFO("todproc.engine().scanindices {}", todproc.engine().scanindices);

                    // copy array and detector indices into engine
                    todproc.engine().array_indices = array_indices.at(i);
                    todproc.engine().det_indices = det_indices.at(i);

                    todproc.engine().cflux.resize(map_counts[i]);

                    Eigen::Index k = 0;
                    Eigen::Index l = 0;
                    for (Eigen::Index j=0; j<todproc.engine().cflux.size(); j++) {
                        if (todproc.engine().cunit == "mJy/beam") {
                             todproc.engine().cflux(j) = toltec_io.barea_keys[l]*MJY_SR_TO_mJY_ASEC;
                            if (k == toltec_io.barea_keys.size() - 1) {
                                k = 0;
                                l++;
                             }
                            else {
                                k++;
                            }
                        }
                        else if (todproc.engine().cunit == "MJy/Sr") {
                             todproc.engine().cflux(j) = 1;
                        }
                    }

                    SPDLOG_INFO("cflux {}",todproc.engine().cflux);

                    // do general setup that is only run once per rawobs before grppi pipeline
                    {
                        tula::logging::scoped_timeit timer("engine setup()");
                        todproc.engine().setup();
                    }

                    // run the pipeline
                    {
                        tula::logging::scoped_timeit timer("engine pipeline()");
                        todproc.engine().pipeline(kidsproc, rawobs);
                    }

                    // generate observation output files
                    {
                        tula::logging::scoped_timeit timer("engine obs output()");
                        todproc.engine().template output<EngineBase::obs>(todproc.engine().mb, todproc.engine().fits_ios,
                                                                          todproc.engine().noise_fits_ios,false);
                    }

                    // coadd current map buffer into coadded map buffer
                    if (todproc.engine().run_coadd) {
                        {
                            tula::logging::scoped_timeit timer("engine coadd()");
                            todproc.engine().cmb.coadd(todproc.engine().mb, todproc.engine().dfsmp, todproc.engine().run_kernel);
                        }
                    }
                }

                if (todproc.engine().run_coadd) {
                    // normalize coadded maps
                    todproc.engine().cmb.normalize_maps(todproc.engine().run_kernel);

                    // coadd histogram and psd
                    for (Eigen::Index i = 0; i < todproc.engine().cmb.map_count;i++) {
                        SPDLOG_INFO("calculating coadded map psds for map {}",i);
                        PSD psd;
                        psd.weight_type = todproc.engine().weighting_type;
                        psd.cov_cut = todproc.engine().cmb.cov_cut;
                        psd.exmode = todproc.engine().ex_name;
                        psd.calc_map_psd(todproc.engine().cmb.signal.at(i), todproc.engine().cmb.weight.at(i), todproc.engine().cmb.rcphys,
                                         todproc.engine().cmb.ccphys);
                        todproc.engine().cmb.psd.push_back(std::move(psd));

                        Histogram hist;
                        hist.weight_type = todproc.engine().weighting_type;
                        hist.cov_cut = todproc.engine().cmb.cov_cut;
                        hist.calc_hist(todproc.engine().cmb.signal.at(i), todproc.engine().cmb.weight.at(i));
                        todproc.engine().cmb.histogram.push_back(std::move(hist));

                        SPDLOG_INFO("calculating noise map psds and histograms for map {}", i);
                        if (todproc.engine().run_noise) {
                            // noise average histogram and psd
                            std::vector<PSD> psd_vec;
                            std::vector<Histogram> hist_vec;

                            todproc.engine().cmb.noise_psd.push_back(psd_vec);
                            todproc.engine().cmb.noise_hist.push_back(hist_vec);

                            PSD noise_avg_psd;
                            Histogram noise_avg_hist;

                            // loop through noise maps and get psd
                            for (Eigen::Index j = 0; j < todproc.engine().cmb.nnoise; j++) {
                                PSD psd;
                                psd.cov_cut = todproc.engine().cmb.cov_cut;
                                psd.exmode = todproc.engine().ex_name;
                                psd.weight_type = todproc.engine().weighting_type;

                                Histogram hist;
                                hist.cov_cut = todproc.engine().cmb.cov_cut;
                                hist.weight_type = todproc.engine().weighting_type;

                                Eigen::Tensor<double, 2> out = todproc.engine().cmb.noise.at(i).chip(j, 2);
                                Eigen::Map<Eigen::MatrixXd> noise(out.data(), out.dimension(0), out.dimension(1));

                                psd.calc_map_psd(noise, todproc.engine().cmb.weight.at(i), todproc.engine().cmb.rcphys,
                                                 todproc.engine().cmb.ccphys);
                                todproc.engine().cmb.noise_psd.at(i).push_back(std::move(psd));

                                hist.calc_hist(noise, todproc.engine().cmb.weight.at(i));
                                todproc.engine().cmb.noise_hist.at(i).push_back(std::move(hist));

                                if (j == 0) {
                                    noise_avg_psd.psd = todproc.engine().cmb.noise_psd.at(i).back().psd;
                                    noise_avg_psd.psd_freq = todproc.engine().cmb.noise_psd.at(i).back().psd_freq;
                                    noise_avg_psd.psd2d = todproc.engine().cmb.noise_psd.at(i).back().psd2d;
                                    noise_avg_psd.psd2d_freq = todproc.engine().cmb.noise_psd.at(i).back().psd2d_freq;

                                    noise_avg_hist.hist_vals = todproc.engine().cmb.noise_hist.at(i).back().hist_vals;
                                    noise_avg_hist.hist_bins = todproc.engine().cmb.noise_hist.at(i).back().hist_bins;

                                }

                                else {
                                    noise_avg_psd.psd = noise_avg_psd.psd + todproc.engine().cmb.noise_psd.at(i).back().psd;
                                    noise_avg_psd.psd2d = noise_avg_psd.psd2d + todproc.engine().cmb.noise_psd.at(i).back().psd2d /
                                            todproc.engine().cmb.nnoise;
                                    noise_avg_psd.psd2d_freq = noise_avg_psd.psd2d_freq + todproc.engine()
                                                .cmb.noise_psd.at(i).back().psd2d_freq / todproc.engine().cmb.nnoise;

                                    noise_avg_hist.hist_vals = noise_avg_hist.hist_vals + todproc.engine().cmb.noise_hist.at(i).back().hist_vals;
                                }
                            }

                            noise_avg_psd.psd = noise_avg_psd.psd / todproc.engine().cmb.nnoise;
                            todproc.engine().cmb.noise_avg_psd.push_back(noise_avg_psd);

                            noise_avg_hist.hist_vals = noise_avg_hist.hist_vals / todproc.engine().cmb.nnoise;
                            todproc.engine().cmb.noise_avg_hist.push_back(noise_avg_hist);
                        }
                    }

                    // generate coadd output files
                    {
                        tula::logging::scoped_timeit timer("engine coadd output()");
                        todproc.engine().template output<EngineBase::coadd>(todproc.engine().cmb, todproc.engine().coadd_fits_ios,
                                                                            todproc.engine().noise_fits_ios,false);
                    }

                    if (todproc.engine().run_coadd_filter) {
                        // filter coadd maps
                        {
                            ToltecIO toltec_io;
                            tula::logging::scoped_timeit timer("wiener filter");
                            todproc.engine().wiener_filter.exmode = todproc.engine().ex_name;

                            for (Eigen::Index i = 0; i < todproc.engine().cmb.map_count; i++) {
                                todproc.engine().wiener_filter.make_template(todproc.engine().cmb,todproc.engine().calib_data,
                                                                             todproc.engine().gaussian_template_fwhm_rad[toltec_io.name_keys[i]],i);
                                todproc.engine().wiener_filter.filter_coaddition(todproc.engine().cmb, i);

                                // filter noise maps
                                if (todproc.engine().run_noise) {
                                    tula::logging::scoped_timeit timer("filter_noise()");
                                    for (Eigen::Index j = 0;j < todproc.engine().cmb.nnoise; j++) {
                                        todproc.engine().wiener_filter.filter_noise(todproc.engine().cmb, i, j);
                                    }
                                }
                            }

                            if (todproc.engine().run_noise) {
                                if (todproc.engine().wiener_filter.normalize_error) {
                                    SPDLOG_INFO("normalizing noise map errors");
                                    todproc.engine().cmb.normalize_noise_map_errors(todproc.engine().weighting_type);
                                    SPDLOG_INFO("calculating average filtered rms");
                                    todproc.engine().cmb.calc_average_filtered_rms(todproc.engine().weighting_type);
                                    SPDLOG_INFO("normalizing errors");
                                    todproc.engine().cmb.normalize_errors(todproc.engine().weighting_type);
                                }


                                // clear psd and histogram vectors
                                todproc.engine().cmb.psd.clear();
                                todproc.engine().cmb.histogram.clear();

                                todproc.engine().cmb.noise_psd.clear();
                                todproc.engine().cmb.noise_hist.clear();

                                todproc.engine().cmb.noise_avg_psd.clear();
                                todproc.engine().cmb.noise_avg_hist.clear();

                                // coadd histogram and psd
                                for (Eigen::Index i = 0; i < todproc.engine().cmb.map_count;i++) {
                                    SPDLOG_INFO("calculating coadded map psds for map {}",i);
                                    PSD psd;
                                    psd.weight_type = todproc.engine().weighting_type;
                                    psd.cov_cut = todproc.engine().cmb.cov_cut;
                                    psd.exmode = todproc.engine().ex_name;
                                    psd.calc_map_psd(todproc.engine().cmb.signal.at(i), todproc.engine().cmb.weight.at(i), todproc.engine().cmb.rcphys,
                                                     todproc.engine().cmb.ccphys);
                                    todproc.engine().cmb.psd.push_back(std::move(psd));

                                    Histogram hist;
                                    hist.weight_type = todproc.engine().weighting_type;
                                    hist.cov_cut = todproc.engine().cmb.cov_cut;
                                    hist.calc_hist(todproc.engine().cmb.signal.at(i), todproc.engine().cmb.weight.at(i));
                                    todproc.engine().cmb.histogram.push_back(std::move(hist));

                                    SPDLOG_INFO("calculating noise map psds and histograms for map {}", i);
                                    if (todproc.engine().run_noise) {
                                        // noise average histogram and psd
                                        std::vector<PSD> psd_vec;
                                        std::vector<Histogram> hist_vec;

                                        todproc.engine().cmb.noise_psd.push_back(psd_vec);
                                        todproc.engine().cmb.noise_hist.push_back(hist_vec);

                                        PSD noise_avg_psd;
                                        Histogram noise_avg_hist;

                                        // loop through noise maps and get psd
                                        for (Eigen::Index j = 0; j < todproc.engine().cmb.nnoise; j++) {
                                            PSD psd;
                                            psd.weight_type = todproc.engine().weighting_type;
                                            psd.cov_cut = todproc.engine().cmb.cov_cut;
                                            psd.exmode = todproc.engine().ex_name;

                                            Histogram hist;
                                            hist.weight_type = todproc.engine().weighting_type;
                                            hist.cov_cut = todproc.engine().cmb.cov_cut;

                                            Eigen::Tensor<double, 2> out = todproc.engine().cmb.noise.at(i).chip(j, 2);
                                            Eigen::Map<Eigen::MatrixXd> noise(out.data(), out.dimension(0), out.dimension(1));

                                            psd.calc_map_psd(noise, todproc.engine().cmb.weight.at(i), todproc.engine().cmb.rcphys,
                                                             todproc.engine().cmb.ccphys);
                                            todproc.engine().cmb.noise_psd.at(i).push_back(std::move(psd));

                                            hist.calc_hist(noise, todproc.engine().cmb.weight.at(i));
                                            todproc.engine().cmb.noise_hist.at(i).push_back(std::move(hist));

                                            if (j == 0) {
                                                noise_avg_psd.psd = todproc.engine().cmb.noise_psd.at(i).back().psd;
                                                noise_avg_psd.psd_freq = todproc.engine().cmb.noise_psd.at(i).back().psd_freq;
                                                noise_avg_psd.psd2d = todproc.engine().cmb.noise_psd.at(i).back().psd2d;
                                                noise_avg_psd.psd2d_freq = todproc.engine().cmb.noise_psd.at(i).back().psd2d_freq;

                                                noise_avg_hist.hist_vals = todproc.engine().cmb.noise_hist.at(i).back().hist_vals;
                                                noise_avg_hist.hist_bins = todproc.engine().cmb.noise_hist.at(i).back().hist_bins;
                                            }

                                            else {
                                                noise_avg_psd.psd = noise_avg_psd.psd + todproc.engine().cmb.noise_psd.at(i).back().psd;
                                                noise_avg_psd.psd2d = noise_avg_psd.psd2d + todproc.engine().cmb.noise_psd.at(i).back().psd2d /
                                                                                                todproc.engine().cmb.nnoise;
                                                noise_avg_psd.psd2d_freq = noise_avg_psd.psd2d_freq +
                                                                           todproc.engine().cmb.noise_psd.at(i).back().psd2d_freq / todproc.engine().cmb.nnoise;

                                                noise_avg_hist.hist_vals = noise_avg_hist.hist_vals + todproc.engine().cmb.noise_hist.at(i).back().hist_vals;

                                            }
                                        }

                                        noise_avg_psd.psd = noise_avg_psd.psd / todproc.engine().cmb.nnoise;
                                        todproc.engine().cmb.noise_avg_psd.push_back(noise_avg_psd);

                                        noise_avg_hist.hist_vals = noise_avg_hist.hist_vals / todproc.engine().cmb.nnoise;
                                        todproc.engine().cmb.noise_avg_hist.push_back(noise_avg_hist);
                                    }
                                }
                            }
                        }

                        // generate filtered coadd output files (cmb is overwritten with filtered maps)
                        {
                            tula::logging::scoped_timeit timer("engine filtered coadd output()");
                            todproc.engine().template output<EngineBase::coadd>(todproc.engine().cmb,
                                                                                todproc.engine().filtered_coadd_fits_ios,
                                                                                todproc.engine().filtered_noise_fits_ios, true);
                        }
                    }
                }

                return EXIT_SUCCESS;
            }
        },
        todproc);

    return exitcode;
}

int main(int argc, char *argv[]) {
    // to do the dump_config, we need to make sure the output is
    // not contaminated with any logging message. Therefore this has
    // to go first
    bool exit_dump_config{false};
    clipp::parse(argc, argv, (
        clipp::option("--dump_config").call([&exit_dump_config] () {
            auto preamble = fmt::format(
                "# Default config.yaml of Citlali {} ({})",
                CITLALI_GIT_VERSION, CITLALI_BUILD_TIMESTAMP
                );
            fmt::print("{}\n{}", preamble, citlali::citlali_default_config_content);
            exit_dump_config = true;
            }),
        clipp::any_other()
    ));
    if (exit_dump_config) {
        return EXIT_SUCCESS;
    }
    // now with normal CLI interface
    tula::logging::init();
    auto rc = parse_args(argc, argv);
    SPDLOG_TRACE("rc {}", rc.pformat());
    if (rc.get_node("config_file").size() > 0) {
        tula::logging::scoped_timeit TULA_X{"Citlali Process"};
        return run(rc);
    } else {
        std::cout << "Invalid argument. Type --help for usage.\n";
    }
}
