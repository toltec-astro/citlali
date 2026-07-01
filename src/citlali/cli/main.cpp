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

#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <regex>
#include <tuple>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/cli/abort_backtrace.h>
#include <citlali/core/cli/config_loading.h>
#include <citlali/core/cli/default_config_dump.h>
#include <citlali/core/cli/hdf5_diagnostics.h>
#include <citlali/core/cli/reduction_runtime.h>
#include <citlali/core/cli/run_logging.h>
#include <citlali/core/cli/runtime_setup.h>
#include <citlali/core/cli/tod_processor_selection.h>
#include <citlali/core/engine/lali.h>
#include <citlali/core/engine/pointing.h>
#include <citlali/core/engine/beammap.h>
#include <citlali/core/pipeline/map_geometry.h>
#include <citlali/core/pipeline/observation_execution.h>
#include <citlali/core/pipeline/observation_date.h>

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
        auto v = spdlog::level::info;
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


// @brief Run citlali reduction.
/// @param rc The runtime config.
int run(const rc_t &rc) {
    citlali::cli::suppress_optional_hdf5_diagnostics();

    // get current level
    auto log_level = spdlog::get_level();

    auto run_loggers = citlali::cli::configure_run_loggers(log_level);
    auto logger = run_loggers.logger;
    // set pattern for logger
    //spdlog::set_pattern("[%H:%M:%S %z] [%s] %v");

    citlali::cli::install_abort_backtrace_handler();

    logger->info("use KIDs data spec: {}", predefs::kidsdata::name);

    std::vector<std::string> config_filepaths;

    auto citlali_config = citlali::cli::load_merged_yaml_config_files(
        rc, config_filepaths, logger);

    // set up the IO coorindator
    auto co = SeqIOCoordinator::from_config(citlali_config);

    // set up KIDs data proc
    //auto kidsproc =
    //    KidsDataProc::from_config(citlali_config.get_config("kids"));

    // set up todproc
    using todproc_var_t =
        std::variant<std::monostate, TimeOrderedDataProc<Lali>, TimeOrderedDataProc<Pointing>,
                     TimeOrderedDataProc<Beammap>>;

    // declare todproc variable
    todproc_var_t todproc;

    // set todproc to variant depending on the config file reduction type
    auto selection_status =
        citlali::cli::select_tod_processor_from_config<
            todproc_var_t, TimeOrderedDataProc<Lali>,
            TimeOrderedDataProc<Pointing>, TimeOrderedDataProc<Beammap>>(
            todproc, citlali_config, logger);

    if (citlali::cli::report_tod_processor_selection_failure(
            selection_status, std::cerr)) {
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
                citlali::pipeline::ReductionMapGeometry<todproc_t>
                    map_geometry;

                if (!citlali::cli::prepare_reduction_runtime(
                        todproc, citlali_config, logger,
                        []() { spdlog::set_level(spdlog::level::debug); },
                        [&](const auto &engine) {
                            citlali::cli::configure_runtime_threads(
                                engine, logger,
#if defined(CITLALI_USE_WIENER_FILTER_OMP)
                                true,
#else
                                false,
#endif
                                [](int n_threads) {
                                    omp_set_num_threads(n_threads);
                                },
                                [](int n_threads) {
                                    Eigen::setNbThreads(n_threads);
                                },
                                []() { return fftw_init_threads(); },
                                [](int n_threads) {
                                    fftw_plan_with_nthreads(n_threads);
                                });
                        })) {
                    citlali::cli::report_engine_config_errors(
                        todproc.engine(), std::cerr);
                    return EXIT_FAILURE;
                }

                if (!citlali::pipeline::run_reduction_pipeline<
                        std::is_same_v<todproc_t,
                                       TimeOrderedDataProc<Beammap>>,
                        mapmaking::RawObs, mapmaking::FilteredObs,
                        mapmaking::RawCoadd, mapmaking::FilteredCoadd,
                        std::is_same_v<todproc_t,
                                       TimeOrderedDataProc<Pointing>>,
                        KidsDataProc>(
                        todproc, co, citlali_config, config_filepaths,
                        map_geometry.extents, map_geometry.coords,
                        [](auto &engine) {
                            return citlali::pipeline::date_obs_from_telescope_time(
                                engine, [](double unix_time) {
                                    return engine_utils::unix_to_utc(
                                        unix_time);
                                });
                        },
                        logger)) {
                    return EXIT_FAILURE;
                }

                logger->info("citlali is done!  going to sleep now...wake me when you need me.");
                return EXIT_SUCCESS;
            }
        },
        todproc);

    // re-enable default logger
    citlali::cli::restore_default_sink_level(run_loggers, log_level);

    return exitcode;
}

int main(int argc, char *argv[]) {
     // to do the dump_config, we need to make sure the output is
    // not contaminated with any logging message. Therefore this has
    // to go first
    if (citlali::cli::dump_default_config_if_requested(
            argc, argv, CITLALI_GIT_VERSION, CITLALI_BUILD_TIMESTAMP,
            citlali::citlali_default_config_content)) {
        return EXIT_SUCCESS;
    }
    // now with normal CLI interface
    try {
        tula::logging::init();
        auto rc = parse_args(argc, argv);
        SPDLOG_INFO("rc {}", rc.pformat());
        if (rc.get_node("config_file").size() > 0) {
            tula::logging::scoped_timeit TULA_X{"Citlali Process"};
            return run(rc);
        } else {
            std::cout << "Invalid argument. Type --help for usage.\n";
            return EXIT_FAILURE;
        }
    } catch (const CCfits::FitsError &e) {
        SPDLOG_CRITICAL("Unhandled CCfits::FitsError: {}", e.message());
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        SPDLOG_CRITICAL("Unhandled exception: {}", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        SPDLOG_CRITICAL("Unhandled non-standard exception");
        return EXIT_FAILURE;
    }
}
