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

#include <citlali/core/engine/wyatt.h>


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
                     const tula::container_utils::Slice<int> &slice,std::string ex_name="seq") {
        std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> result;
        /*for (const auto &data_item : rawobs.kidsdata()) {
            result.push_back(load_data_item(data_item, slice));
        }*/

        std::vector<int> result_in_vec, result_out_vec;
        result_in_vec.resize(rawobs.kidsdata().size());
        std::iota(result_in_vec.begin(), result_in_vec.end(), 0);
        result_out_vec.resize(rawobs.kidsdata().size());

        grppi::map(tula::grppi_utils::dyn_ex(ex_name), result_in_vec, result_out_vec, [&](auto r) {
            result.push_back(load_data_item(rawobs.kidsdata()[r], slice));
            return 0;
        });

        return std::move(result);
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
        //auto slice = tula::container_utils::Slice<int>{
        //scanindex(2), scanindex(3) + 1, std::nullopt};
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

    if (std::strcmp("beammap", engine().reduction_type.c_str()) == 0) {
        map_count = engine().ndet;
    }

    Eigen::Index narrays = 3;

    for (Eigen::Index arr=0; arr<narrays; arr++) {
        if ((engine().calib_data["array"].array()==arr).any()) {
            map_count++;
            engine().arrays[engine().toltec_io.name_keys[arr]] = arr;
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


int run(const rc_t &rc) {

    tula::config::YamlConfig wyatt_config;
    std::vector<std::string> config_filepaths;

    {
        auto node_config_files = rc.get_node("config_file");
        for (const auto & n: node_config_files) {
            auto filepath = n.as<std::string>();
            config_filepaths.push_back(filepath);
            SPDLOG_TRACE("load config from file {}", filepath);
            wyatt_config = tula::config::merge(wyatt_config, tula::config::YamlConfig::from_filepath(filepath));
        }
    }

    SPDLOG_INFO(wyatt_config);

    std::size_t found = config_filepaths[0].find("coadd");
    if (found!=std::string::npos) {
        SPDLOG_INFO("Doing coadd");
        wyatt::coadd(wyatt_config);
        return EXIT_SUCCESS;
    }

    else {
        // wyatt config information
        auto obsnum = wyatt_config.get_typed<int>("obsnum");
        auto wyatt_filepath = wyatt_config.get_str("wyatt_filepath");
        auto nrows = wyatt_config.get_typed<int>("nrows");
        auto ncols = wyatt_config.get_typed<int>("ncols");
        auto start_time = wyatt_config.get_typed<double>("start_time");
        auto sampdown = wyatt_config.get_typed<int>("sampdown");
        auto lowerfreq_on = wyatt_config.get_typed<double>("lowerfreq_on");
        auto upperfreq_on = wyatt_config.get_typed<double>("upperfreq_on");
        auto lowerfreq_off = wyatt_config.get_typed<double>("lowerfreq_off");
        auto upperfreq_off = wyatt_config.get_typed<double>("upperfreq_off");
        auto fwhm_upper_lim = wyatt_config.get_typed<double>("fwhm_upper_lim");
        auto fwhm_lower_lim = wyatt_config.get_typed<double>("fwhm_lower_lim");

        SPDLOG_INFO("Got config options");

        auto input_filepath = wyatt_config.get_str("input_filepath");
        SPDLOG_INFO("Input filepath {}", input_filepath);
        netCDF::NcFile fo(input_filepath, netCDF::NcFile::read);
        auto vars = fo.getVars();

        SPDLOG_INFO("Got raw file");

        auto input_filepath_proc = wyatt_config.get_str("input_filepath_proc");
        SPDLOG_INFO("Processed file {}", input_filepath_proc);
        netCDF::NcFile fop(input_filepath_proc, netCDF::NcFile::read);
        auto vars_p = fop.getVars();

        SPDLOG_INFO("Got processed file");

        std::string t0_filename = wyatt_filepath+std::to_string(obsnum)+"_ts0.out";
        std::string t1_filename = wyatt_filepath+std::to_string(obsnum)+"_ts1.out";

        SPDLOG_INFO("t0_filename {}", t0_filename);
        SPDLOG_INFO("t1_filename {}",t1_filename);

        Eigen::VectorXd t0s = wyatt::get_wyatt(t0_filename);
        Eigen::VectorXd t1s = wyatt::get_wyatt(t1_filename);

        SPDLOG_INFO("t0s {}",t0s);
        SPDLOG_INFO("t1s {}",t1s);

        double SampleFreqRaw;

        if (sampdown == 1){
            const auto& SampleFreqRaw_var =  vars.find("Header.Toltec.SampleFreqRaw")->second;
            SampleFreqRaw_var.getVar(&SampleFreqRaw);
        }
        else {
            const auto& SampleFreqRaw_var =  vars.find("Header.Toltec.SampleFreq")->second;
            SampleFreqRaw_var.getVar(&SampleFreqRaw);
        }

        double SampleFreq;
        const auto& SampleFreq_var =  vars.find("Header.Toltec.SampleFreq")->second;
        SampleFreq_var.getVar(&SampleFreq);

        SPDLOG_INFO("SampleFreqRaw {}",SampleFreqRaw);
        SPDLOG_INFO("SampleFreq {}",SampleFreq);

        Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Ts;
        const auto& Ts_var = vars.find("Data.Toltec.Ts")->second;

        Eigen::Index Ts_dim_0 = Ts_var.getDim(0).getSize();
        Eigen::Index Ts_dim_1 = Ts_var.getDim(1).getSize();

        Ts.resize(Ts_dim_0,Ts_dim_1);
        Ts_var.getVar(Ts.data());
        SPDLOG_INFO(Ts);
        for(int i=0;i<Ts.cols();i++)
            SPDLOG_INFO(Ts.col(i));
        Eigen::VectorXd time = wyatt::calc_time(Ts, Ts_dim_0, SampleFreqRaw, start_time);

        SPDLOG_INFO("time {}",time);

        Eigen::VectorXd LoFreq;
        LoFreq.resize(Ts_dim_0);
        const auto& LoFreq_var = vars.find("Data.Toltec.LoFreq")->second;
        LoFreq_var.getVar(LoFreq.data());

        SPDLOG_INFO("LoFreq {}", LoFreq);

        double LoFreq_mean;
        LoFreq_mean = LoFreq.mean();

        size_t Xs_dim;
        const auto& Xs = vars_p.find("Data.Kids.xs")->second;
        Xs_dim = Xs.getDim(1).getSize();
        SPDLOG_INFO("D");
        SPDLOG_INFO("Xs_dim: {}",Xs_dim);
        SPDLOG_INFO("nrows: {} ncols: {}",nrows,ncols);

        size_t Rs_dim;
        const auto& Rs = vars_p.find("Data.Kids.rs")->second;
        Rs_dim = Rs.getDim(1).getSize();

        SPDLOG_INFO("Rs_dim: {}",Rs_dim);

        Eigen::VectorXd tf;
        size_t tone_freq_dim;
        const auto& tone_freq = vars.find("Header.Toltec.ToneFreq")->second;
        tone_freq_dim = tone_freq.getDim(1).getSize();
        tf.resize(tone_freq_dim);
        tone_freq.getVar(tf.data());

        SPDLOG_INFO("tone_freq_dim {}", tone_freq_dim);
        SPDLOG_INFO("tone_freqs {}", tf);

        wyatt::maps maps;
        maps.resize(nrows,ncols,t0s.size(),Xs_dim);
        SPDLOG_INFO("Generated Maps");

        wyatt::populate_indices(maps.si,maps.ei,t0s,t1s,time,Ts_dim_0);

        SPDLOG_INFO("si {}",maps.si);
        SPDLOG_INFO("ei {}",maps.ei);

        std::vector<int> detvec_in(Xs_dim);
        std::iota(detvec_in.begin(), detvec_in.end(), 0);
        std::vector<int> detvec_out(Xs_dim);

        int s = 0;

        Eigen::MatrixXd params(6,Xs_dim);

        int nthreads = Eigen::nbThreads();

        //Begin timestream pipeline
        SPDLOG_INFO("Starting Pipeline");
        grppi::pipeline(grppi::sequential_execution(), [&]() -> std::optional<TCData<TCDataKind::WTC, Eigen::MatrixXd>> {
                while (s<t0s.size()){
                    SPDLOG_INFO("On pixel {}/{}",s,t0s.size());
                    size_t scanlength = maps.ei[s] - maps.si[s];
                    size_t sii = maps.si[s];
                    size_t eii = maps.ei[s];

                    std::vector<size_t> sis;
                    sis = {sii,0};
                    std::vector<size_t> count;
                    count = {scanlength,Xs_dim};

                    TCData<TCDataKind::WTC, Eigen::MatrixXd> r;
                    r.xscans.data.resize(scanlength,Xs_dim);
                    Xs.getVar(sis,count,r.xscans.data.data());

                    r.rscans.data.resize(scanlength,Rs_dim);
                    Rs.getVar(sis,count,r.rscans.data.data());

                    SPDLOG_INFO("r.scans {}", r.xscans.data);

                    s++;

                    return r;
                }
                return {};
            },

            [&](auto in) {
                grppi::map(grppi::parallel_execution_omp(), detvec_in, detvec_out, [&] (auto i) {
                    Eigen::VectorXd x_in_col = in.xscans.data.col(i);
                    Eigen::VectorXd r_in_col = in.rscans.data.col(i);

                    Eigen::VectorXd x_psdvec = wyatt::psd(x_in_col, maps, 512, SampleFreq);
                    Eigen::VectorXd r_psdvec = wyatt::psd(r_in_col, maps, 512, SampleFreq);

                    Eigen::Index nfreqs = x_psdvec.size();
                    double dfreq = SampleFreq / x_psdvec.size()/2.;
                    Eigen::VectorXd freqs = dfreq * Eigen::VectorXd::LinSpaced(nfreqs, 0, x_psdvec.size());

                    Eigen::Index lfi_on = lowerfreq_on/dfreq;
                    Eigen::Index lfi_off= lowerfreq_off/dfreq;

                    Eigen::Index on_rng = (upperfreq_on/dfreq) - lfi_on;
                    Eigen::Index off_rng = (upperfreq_off/dfreq) - lfi_off;

                    double x_peak = x_psdvec.segment(lfi_on,on_rng).maxCoeff();
                    double x_avg = tula::alg::median(x_psdvec.segment(lfi_off,off_rng));

                    double r_peak = r_psdvec.segment(lfi_on,on_rng).maxCoeff();
                    double r_avg = tula::alg::median(r_psdvec.segment(lfi_off,off_rng));

                    maps.x_onoff(s-1,i) = x_peak/x_avg;
                    maps.r_onoff(s-1,i) = r_peak/r_avg;

                    maps.x_off(s-1,i) = x_avg;
                    maps.r_off(s-1,i) = r_avg;

                    maps.xmap(s-1,i) = tula::alg::median(x_in_col);
                    maps.rmap(s-1,i) = tula::alg::median(r_in_col);

                    return 0;
                });
                return in;});

        grppi::pipeline(grppi::parallel_execution_omp(), [&]() -> std::optional<int> {
                static int dets = 0;
                while(dets<Xs_dim){
                    return dets++;
                }
                return {};
            },

            grppi::farm(nthreads,[&](auto &dets) {
                SPDLOG_INFO("Fitting det {}", dets);
                Eigen::VectorXd xrng = Eigen::VectorXd::LinSpaced(nrows, 0, nrows-1);
                Eigen::VectorXd yrng = Eigen::VectorXd::LinSpaced(ncols, 0, ncols-1);

                Eigen::VectorXd onoff_col = maps.x_onoff.col(dets);
                Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> onoff_matrix = Eigen::Map<Eigen::MatrixXd> (onoff_col.data(),nrows,ncols);

                     //Eigen::VectorXd off_col = maps.off.col(dets);
                     //Eigen::VectorXd on_col(off_col.size());

                     //on_col = (onoff_col.array() * off_col.array()).eval();

                     //Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> on_matrix = Eigen::Map<Eigen::MatrixXd> (on_col.data(),nrows,ncols);

                int l = 0;
                for (int k=0;k<nrows;k++) {
                    for (int m =0;m<ncols;m++) {
                        onoff_matrix(k,m) = onoff_col(l);
                        //on_matrix(k,m) = on_col(l);
                        l++;
                    }
                }

                double temp = onoff_matrix.mean();
                onoff_matrix = onoff_matrix.array() - temp;

                     //on_matrix = (on_matrix.array()/on_matrix.mean()).eval();

                for (int i=0;i<nrows;i=i+2) {
                    Eigen::VectorXd tmprv = onoff_matrix.row(i).reverse();
                    //Eigen::VectorXd tmprv = on_matrix.row(i).reverse();
                    onoff_matrix.row(i) = tmprv;
                    //on_matrix.row(i) = tmprv;

                }
                double th = onoff_matrix.maxCoeff();
                //double th = on_matrix.maxCoeff();

                int tmppeak_i = 0;
                int tmppeak_j = 0;
                for (int k=0;k<nrows;k++) {
                    for (int m=0;m<ncols;m++) {
                        if(onoff_matrix(k,m) == th){
                            //if(on_matrix(k,m) == th){
                            tmppeak_i = k;
                            tmppeak_j = m;
                        }
                    }
                }

                Eigen::VectorXd init_p(6);
                double init_fwhm = 1.5;//fwhm_upper_lim;
                init_p << th, tmppeak_j,tmppeak_i, init_fwhm, init_fwhm, 0.;

                SPDLOG_INFO("Initial Peak: {}",th);
                SPDLOG_INFO("Initial x: {}",tmppeak_i);
                SPDLOG_INFO("Initial y: {}",tmppeak_j);
                SPDLOG_INFO("Initial fwhmx: {}",init_fwhm);

                Eigen::VectorXd lower_limits(6);
                Eigen::VectorXd upper_limits(6);

                int range = 100;
                int minsize = 0;

                fwhm_upper_lim = 3.0;
                fwhm_lower_lim = 1.0;

                lower_limits << 0.75*th, std::max(tmppeak_j - range,minsize),std::max(tmppeak_i - range,minsize),fwhm_lower_lim, fwhm_lower_lim, 0.;
                upper_limits << 1.25*th, std::min(tmppeak_j + range,ncols),std::min(tmppeak_i + range,nrows), fwhm_upper_lim, fwhm_upper_lim, 3.1415/4.;

                Eigen::MatrixXd limits(6, 2);

                limits.col(0) << 0.75*th, std::max(tmppeak_j - range,minsize),std::max(tmppeak_i - range,minsize),fwhm_lower_lim, fwhm_lower_lim, 0.;
                limits.col(1) << 1.25*th, std::min(tmppeak_j + range,ncols),std::min(tmppeak_i + range,nrows), fwhm_upper_lim, fwhm_upper_lim, 3.1415/4.;

                auto g = gaussfit::modelgen<gaussfit::Gaussian2D>(init_p);

                Eigen::MatrixXd sigmamatrix(nrows,ncols);
                sigmamatrix.setOnes();

                auto _p = g.params;
                auto xy = g.meshgrid(yrng, xrng);
                //Run the fit
                auto [g_fit, cov] = curvefit_ceres(g, _p, xy, onoff_matrix, sigmamatrix,limits);

                //auto g_fit = curvefit_ceres(g, _p, xy, on_matrix, sigmamatrix,lower_limits,upper_limits);
                params.col(dets) = g_fit.params;

                return 0;}));

        using namespace netCDF;
        using namespace netCDF::exceptions;

        auto output_filepath = wyatt_config.get_str("output_filepath");
        auto nw = wyatt_config.get_str("nw");

        std::string output_filename = output_filepath+std::to_string(obsnum)+"_"+nw+".nc";

        try {
            //Create NetCDF file
            SPDLOG_INFO("output_filename {}",output_filename);
            NcFile fo(output_filename, NcFile::replace);

            int nalls = ncols*nrows;

            NcDim nall = fo.addDim("nall", nalls);
            NcDim ndet_dim = fo.addDim("ndet", Xs_dim);

            NcDim nrowsdim = fo.addDim("nrows", nrows);
            NcDim ncolsdim = fo.addDim("ncols", ncols);

            std::vector<NcDim> dims;
            dims.push_back(nall);
            dims.push_back(ndet_dim);

            NcDim LoFreq_dim = fo.addDim("LoFreq_dim",1);
            auto LoFreq_var = "LoFreq";
            NcVar LoFreq_data = fo.addVar(LoFreq_var,ncDouble,LoFreq_dim);
            LoFreq_data.putVar(&LoFreq_mean);


            auto tone_freq_var = "tone_freq";
            NcVar tone_freq_data = fo.addVar(tone_freq_var,ncDouble,ndet_dim);
            tone_freq_data.putVar(tf.data());

            auto x_onoff_var = "x_onoff";
            NcVar x_onoff_data = fo.addVar(x_onoff_var, ncDouble, dims);
            x_onoff_data.putVar(maps.x_onoff.data());

            auto x_off_var = "x_off";
            NcVar x_off_data = fo.addVar(x_off_var, ncDouble, dims);
            x_off_data.putVar(maps.x_off.data());

            auto r_onoff_var = "r_onoff";
            NcVar r_onoff_data = fo.addVar(r_onoff_var, ncDouble, dims);
            r_onoff_data.putVar(maps.r_onoff.data());

            auto r_off_var = "r_off";
            NcVar r_off_data = fo.addVar(r_off_var, ncDouble, dims);
            r_off_data.putVar(maps.r_off.data());

            auto xmap_var = "xmap";
            NcVar xmap_data = fo.addVar(xmap_var, ncDouble, dims);
            xmap_data.putVar(maps.xmap.data());

            auto rmap_var = "rmap";
            NcVar rmap_data = fo.addVar(rmap_var, ncDouble, dims);
            rmap_data.putVar(maps.rmap.data());

            NcVar si_var = fo.addVar("si",ncDouble,nall);
            NcVar ei_var = fo.addVar("ei",ncDouble,nall);

            si_var.putVar(maps.si.data());
            ei_var.putVar(maps.ei.data());

            for (int i = 0;i<Xs_dim;i++) {
                auto mapfitvar = "map_fits" + std::to_string(i);
                NcVar mapfitdata = fo.addVar(mapfitvar, ncDouble);

                mapfitdata.putAtt("amplitude",ncDouble,params(0,i));
                mapfitdata.putAtt("offset_x",ncDouble,params(1,i));
                mapfitdata.putAtt("offset_y",ncDouble,params(2,i));
                mapfitdata.putAtt("FWHM_x",ncDouble,params(3,i));
                mapfitdata.putAtt("FWHM_y",ncDouble,params(4,i));
            }

            fo.close();
            return 0;
        }
        catch (NcException &e) {
            SPDLOG_ERROR("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", output_filename)};
        }

    }

    return EXIT_SUCCESS;
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
