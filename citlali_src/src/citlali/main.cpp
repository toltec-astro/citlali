#include "kids/cli/utils.h"
#include "kids/core/kidsdata.h"
#include "kids/sweep/fitter.h"
#include "kids/toltec/toltec.h"
#include "utils/config.h"
#include "utils/grppiex.h"
#include "utils/logging.h"
#include "utils/yamlconfig.h"
#include <citlali/gitversion.h>
#include <cstdlib>
#include <kids/gitversion.h>
#include <tuple>

auto parse_args(int argc, char *argv[]) {

    using FlatConfig = config::Config;
    // Runtime config container, this is returned.
    FlatConfig rc{};
    // CLI config container, this is consumed and discarded.
    FlatConfig cc{};
    // some of the option specs
    auto citlali_ver = fmt::format("citlali {} ({})", CITLALI_GIT_VERSION,
                                   CITLALI_BUILD_TIMESTAMP);
    auto kids_ver =
        fmt::format("kids {} ({})", KIDS_GIT_VERSION, KIDS_BUILD_TIMESTAMP);
    using namespace config_utils::clipp_builder;
    // clang-format off
    auto screen = clipp_utils::screen{
    // =======================================================================
                      "citlali" , "citlali", citlali_ver,
                                  "TolTEC data reduction pipeline"           };
    auto cli = (
    // =======================================================================
    _(cc, p(       "h", "help" ), "Print help information and exit"   )       ,
    // =======================================================================
    _(cc, p(         "version" ), "Print version information and exit")       ,
    // =======================================================================
                                                                             (
    _(rc,        "config_file"  , "The path of input config file", str()     ),
              "common options"                                          % __(
    _(cc, p(  "l", "log_level" ), "Set the log level",
                                  logging::active_level_str,
                                  list(logging::level_names)            ),
    _(rc, p(            "plot" ), "Make diagnostic plot"                ),
    _(rc, p(    "plot_backend" ), "Matplotlib backend to use",
                                  "default", str()                      ),
    _(rc, p(         "grppiex" ), "GRPPI executioon policy",
                                  grppiex::modes::default_(),
                                  list(grppiex::modes::names())         )
                                                                            )
                                                                              )
    );
    // =======================================================================
    // clang-format on

    screen.parse(cli, argc, argv);
    // handle CLI config
    SPDLOG_TRACE("cc: {}", cc.pformat());
    if (cc.get_typed<bool>("help")) {
        screen.manpage(cli);
        std::exit(EXIT_SUCCESS);
    } else if (cc.get_typed<bool>("version")) {
        screen.version();
        // also print the kids version
        screen.os << kids_ver << '\n';
        std::exit(EXIT_SUCCESS);
    }
    {
        auto log_level_str = cc.get_str("log_level");
        auto log_level = spdlog::level::from_str(log_level_str);
        SPDLOG_DEBUG("reconfigure logger to level={}", log_level_str);
        spdlog::set_level(log_level);
    }
    // pass on the runtime config
    return rc;
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

} // namespace predefs

/// @brief The mixin class stores and validates config
template <typename Derived> struct ConfigValidatorMixin {
    using config_t = config::YamlConfig;

private:
    config_t m_config{};
    struct derived_has_check_config {
        define_has_member_traits(Derived, check_config);
        constexpr static auto value = has_check_config::value;
    };

public:
    ConfigValidatorMixin() = default;
    template <typename... Args> ConfigValidatorMixin(Args &&...args) {
        set_config(FWD(args)...);
    }
    const config_t &config() const { return m_config; }
    void set_config(config_t config, bool check = true) {
        if (check) {
            if constexpr (derived_has_check_config::value) {
                if (auto opt_errors = Derived::check_config(config);
                    opt_errors.has_value()) {
                    throw std::runtime_error(
                        fmt::format("invalid config:\n{}\nerrors: {}", config,
                                    opt_errors.value()));
                }
                SPDLOG_TRACE("set config check passed");
            } else {
                SPDLOG_WARN("set config check requested but no "
                            "check_config found");
            }
        } else {
            SPDLOG_TRACE("set config without check");
        }
        m_config = std::move(config);
        SPDLOG_TRACE("m_config:\n{}", m_config.pformat());
    }
    static auto from_config(config_t config, bool check = true) {
        if (check) {
            if constexpr (derived_has_check_config::value) {
                if (auto opt_errors = Derived::check_config(config);
                    opt_errors.has_value()) {
                    throw std::runtime_error(
                        fmt::format("invalid config:\n{}\nerrors: {}", config,
                                    opt_errors.value()));
                }
                SPDLOG_TRACE("config check passed");
            } else {
                SPDLOG_WARN("config check requested but no "
                            "check_config found");
            }
        } else {
            SPDLOG_TRACE("config check skipped");
        }
        return Derived{std::move(config)};
    }
};

/**
 * @brief The raw obs struct
 * This represents a single observation that contains a set of data items.
 */
struct RawObs : ConfigValidatorMixin<RawObs> {
    /**
     * @brief The DataItem struct
     * This represent a single data item that belongs to a particular
     * observation
     */
    struct DataItem : ConfigValidatorMixin<DataItem> {
        using Base = ConfigValidatorMixin<DataItem>;
        DataItem(config_t config)
            : Base{std::move(config)}, interface(this->config().get_str(
                                           std::tuple{"meta", "interface"})),
              filepath(this->config().get_str("filepath")) {}
        const std::string interface{};
        const std::string filepath{};
        friend std::ostream &operator<<(std::ostream &os, const DataItem &d) {
            return os << fmt::format("DataItem(interface={} filepath={})",
                                     d.interface, d.filepath);
        }

        static auto check_config(config_t &config)
            -> std::optional<std::string> {
            std::vector<std::string> missing_keys;
            for (const auto &key : {"meta", "filepath"}) {
                if (!config.has(key)) {
                    missing_keys.push_back(key);
                }
            }
            if (missing_keys.empty()) {
                return std::nullopt;
            }
            return fmt::format("missing keys={}", missing_keys);
        }
    };
    using Base = ConfigValidatorMixin<RawObs>;
    RawObs(config_t config)
        : Base{std::move(config)}, name{this->config().get_str(
                                       std::tuple{"meta", "name"})} {

        // initialize the data_items
        auto node_data_items = this->config()["data_items"];
        for (std::size_t i = 0; i < node_data_items.size(); ++i) {
            data_items.emplace_back(config_t(node_data_items[i]));
        }
    }
    std::string name;
    std::vector<DataItem> data_items{};
    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        if (config.has("data_items") && config["data_items"].IsSequence()) {
            return std::nullopt;
        }
        if (config.has("data_items")) {
            return fmt::format("\"data_items\" has to be a list");
        }
        return fmt::format("missing key \"data_items\"");
    }
    friend std::ostream &operator<<(std::ostream &os, const RawObs &obs) {
        return os << fmt::format("RawObs(name={}, n_data_items={})", obs.name,
                                 obs.data_items.size());
    }
};

/**
 * @brief The Coordinator struct
 * This wraps around the config object and provides
 * high level methods in various ways to setup the MPI runtime
 * with node-local and cross-node environment.
 */
struct SeqIOCoordinator : ConfigValidatorMixin<SeqIOCoordinator> {
    using Base = ConfigValidatorMixin<SeqIOCoordinator>;
    using index_t = predefs::index_t;
    using shape_t = predefs::shape_t;
    using input_t = RawObs;
    using payload_t = RawObs::DataItem;

    SeqIOCoordinator(config_t config) : Base{std::move(config)} {
        collect_inputs();
    }

    // io_buffer
    using payloads_buffer_data_t = predefs::data_t;

    constexpr auto n_inputs() const { return m_inputs.size(); }

    const std::vector<input_t> &inputs() const { return m_inputs; };

    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        if (config.has("inputs") && config["inputs"].IsSequence()) {
            return std::nullopt;
        }
        return fmt::format("missing key \"inputs\"");
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
        auto node_inputs = this->config()["inputs"];
        assert(node_inputs.IsSequence());
        for (std::size_t i = 0; i < node_inputs.size(); i++) {
            inputs.emplace_back(config_t{node_inputs[i]});
        }
        m_inputs = std::move(inputs);

        SPDLOG_DEBUG("collected n_inputs={}", n_inputs());
    }
};

/// @brief Run citlali reduction.
/// @param rc The runtime config.
int run(const config::Config &rc) {
    using kids::KidsData;
    using kids::KidsDataKind;
    using logging::timeit;
    // IO spec
    namespace kidsdata = kids::toltec;
    SPDLOG_INFO("use KIDs data spec: {}", kidsdata::name);

    // load the yaml citlali config
    auto citlali_config =
        config::YamlConfig::from_filepath(rc.get_str("config_file"));
    SPDLOG_TRACE("citlali config:\n{}", citlali_config.pformat());

    // set up the IO coorindator
    auto co = SeqIOCoordinator::from_config(citlali_config);
    SPDLOG_TRACE("pipeline coordinator: {}", co);
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
    logging::init<>(true);

    auto rc = parse_args(argc, argv);
    SPDLOG_TRACE("rc {}", rc.pformat());
    if (rc.is_set("config_file")) {
        return run(rc);
    } else {
        std::cout << "Invalid argument. Type --help for usage.\n";
    }
}
