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

using Config = config::Config;
using YamlConfig = config::YamlConfig;

auto parse_args(int argc, char *argv[]) {
    // runtime config container
    Config rc{};
    // cli config container
    Config cc{};
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
    // handle cc
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
    return rc;
}

/// @brief The mixin class for validating config
template <typename Derived> struct ConfigMixin {
    using config_t = YamlConfig;

private:
    using Self = ConfigMixin<Derived>;
    config_t m_config;
    struct derived_has_validate_config {
        define_has_member_traits(Derived, validate_config);
        constexpr static auto value = has_validate_config::value;
    };

public:
    ConfigMixin() = default;
    template <typename... Args> ConfigMixin(Args &&...args) {
        set_config(FWD(args)...);
    }
    const config_t &config() { return m_config; }
    void set_config(config_t config, bool validate = true) {
        if (validate) {
            if constexpr (derived_has_validate_config::value) {
                if (auto opt_errors = Derived::validate_config(config);
                    opt_errors.has_value()) {
                    SPDLOG_ERROR("invalid config:\n{}\nerrors: {}", config,
                                 opt_errors.value());
                } else {
                    SPDLOG_TRACE("set config validated");
                }
            } else {
                SPDLOG_WARN(
                    "set config validation requested but no validator found");
            }
        } else {
            SPDLOG_TRACE("set config without validation");
        }
        m_config = std::move(config);
    }
};

/**
 * @brief The raw obs struct
 * This represents a single observation that contains a set of data items.
 */
struct RawObs : ConfigMixin<RawObs> {
    /**
     * @brief The DataItem struct
     * This represent a single data item that belongs to a particular
     * observation
     */
    struct DataItem : ConfigMixin<DataItem> {

        DataItem(config_t config_)
            : ConfigMixin<DataItem>{std::move(config_)},
              interface(config().get_str("interface")),
              filepath(config().get_str("filepath")) {
            // initalize io
            //                 meta::switch_invoke<InterfaceRegistry::InterfaceKind>(
            //                     [&](auto interfacekind_) {
            //                         constexpr auto interfacekind =
            //                             DECAY(interfacekind_)::value;
            //                         io = InterfaceRegistry::template
            //                         io_t<interfacekind>();
            //                     },
            //                     interface_kind());
        }
        std::string interface{};
        std::string filepath{};
        friend std::ostream &operator<<(std::ostream &os, const DataItem &d) {
            return os << fmt::format("DataItem(interface={} filepath={})",
                                     d.interface, d.filepath);
        }
        InterfaceRegistry::variant_t io;
        static auto validate_config(const config_t &config)
            -> std::optional<std::string> {
            std::vector<std::string> missing_keys;
            for (const auto &key : {"interface", "filepath"}) {
                if (!config.has(key)) {
                    missing_keys.push_back(key);
                }
            }
            if (missing_keys.empty()) {
                return std::nullopt;
            }
            return fmt::format("missing keys={}", missing_keys);
        }
        index_t buffer_size() const { return 4880 * 500; }
        shape_t buffer_shape() const {
            shape_t s;
            s << 4880, 500;
            return s;
        }
    };
    Observation(config_t config_)
        : ConfigMixin<Observation>{config_}, name{config().get_str("name",
                                                                   "unnamed")} {
        // initialize the data_items
        //             auto node = config()["data_items"];
        //             assert(node.IsSequence());
        //             for (std::size_t i = 0; i < node.size(); ++i) {
        //                 data_items.emplace_back(config_t(node[i]));
        //             }
    }
    std::string name;
    // std::vector<DataItem> data_items{};
    static auto validate_config(const config_t &config)
        -> std::optional<std::string> {
        if (config.has("data_items")) {
            return std::nullopt;
        }
        return fmt::format("missing key={}", "data_items");
    }
    friend std::ostream &operator<<(std::ostream &os, const Observation &obs) {
        return os << fmt::format("Observation(name={})", obs.name);
    }
};

/// @brief Run citlali reduction.
/// @param rc The runtime config.
int run(const config::Config &rc) {
    using kids::KidsData;
    using kids::KidsDataKind;
    using logging::timeit;
    // IO spec
    namespace spec = kids::toltec;
    SPDLOG_INFO("use data spec: {}", spec::name);

    // load the config
    auto config_filepath = rc.get_str("config_file");
    auto config = YamlConfig(YAML::LoadFile(config_filepath));
    SPDLOG_TRACE("citlali config:\n{}", config.pformat());
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
