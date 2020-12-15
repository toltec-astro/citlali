#include "kids/cli/utils.h"
#include "kids/core/kidsdata.h"
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
    // disable logger in this function
    // auto _0 = logging::scoped_quiet{};
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
                 "server mode"                                          % __(
    _(rc, p(       "p", "port" ), "The port to use", 55437, opt_int()   )
                                                                            )|
                    "cmd mode"                                          % __(
    _(rc,        "config_file"  , "The path of input config file", str())
                                                                            ),
              "common options"                                          % __(
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
    return rc;
}

int run_server(int port) {
    SPDLOG_INFO("start server on port {}", port);
    SPDLOG_WARN("this functionality is not implemented yet");
    std::cin.get();
    SPDLOG_INFO("server shutdown");
    return EXIT_SUCCESS;
}

int run_cmdproc(const config::Config &rc) {
    using kids::KidsData;
    using kids::KidsDataKind;
    using logging::timeit;
    // IO spec
    namespace spec = kids::toltec;
    SPDLOG_INFO("use data spec: {}", spec::name);

    // load the config
    auto config_filepath = rc.get_str("config_file");
    auto config = YamlConfig(YAML::LoadFile(config_filepath));
    SPDLOG_TRACE("load config: {}", config.pformat());
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
    std::cout << fmt::format("log level at compile time: {}\n",
                             SPDLOG_ACTIVE_LEVEL);
    spdlog::set_level(
        static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));
    // spdlog::set_error_handler([](const std::string& msg) {
    //     throw std::runtime_error(msg);
    // });

    auto rc = parse_args(argc, argv);
    SPDLOG_TRACE("rc {}", rc.pformat());
    // server mode
    if (rc.is_set("port")) {
        return run_server(rc.get_typed<int>("port"));
    } else if (rc.is_set("config_file")) {
        // cmd mode
        return run_cmdproc(rc);
    } else {
        std::cout << "Invalid argument. Type --help for usage.\n";
    }
}
