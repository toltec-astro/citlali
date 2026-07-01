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

#if defined(__linux__)
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#endif

#if defined(__has_include)
#if __has_include(<hdf5.h>)
#include <hdf5.h>
#define CITLALI_HAS_HDF5 1
#elif __has_include(<hdf5/serial/hdf5.h>)
#include <hdf5/serial/hdf5.h>
#define CITLALI_HAS_HDF5 1
#else
#define CITLALI_HAS_HDF5 0
#endif
#else
#define CITLALI_HAS_HDF5 0
#endif

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/cli/config_loading.h>
#include <citlali/core/engine/lali.h>
#include <citlali/core/engine/pointing.h>
#include <citlali/core/engine/beammap.h>
#include <citlali/core/pipeline/fruit_loop_paths.h>
#include <citlali/core/pipeline/iteration_lifecycle.h>
#include <citlali/core/pipeline/observation_execution.h>
#include <citlali/core/pipeline/observation_preflight.h>
#include <citlali/core/pipeline/output_layout.h>

using rc_t = tula::config::YamlConfig;

namespace {

#if defined(__linux__)
void abort_backtrace_handler(int sig) {
    void *frames[128];
    int n = ::backtrace(frames, static_cast<int>(sizeof(frames) / sizeof(frames[0])));
    const char msg[] = "\n[citlali] fatal signal received; stack trace follows:\n";
    const ssize_t nw = ::write(STDERR_FILENO, msg, sizeof(msg) - 1);
    if (nw < 0) {
        // best-effort only in signal context
    }
    const auto &crumb = mapmaking::get_jinc_debug_breadcrumb();
    if (crumb.valid) {
        std::fprintf(stderr,
                     "[citlali] jinc breadcrumb: stage=%s det_col=%lld det_uid=%d sample=%lld map_index=%lld array=%lld "
                     "pixel=(%d,%d) subpix=%d map_block=[%d:%d,%d:%d] jinc_offset=(%d,%d) size=%dx%d\n",
                     crumb.stage,
                     crumb.det_col,
                     crumb.det_uid,
                     crumb.sample,
                     crumb.map_index,
                     crumb.array_index,
                     crumb.pixel_row,
                     crumb.pixel_col,
                     crumb.subpix_idx,
                     crumb.lower_row,
                     crumb.upper_row,
                     crumb.lower_col,
                     crumb.upper_col,
                     crumb.jinc_lower_row,
                     crumb.jinc_lower_col,
                     crumb.size_rows,
                     crumb.size_cols);
    }
    ::backtrace_symbols_fd(frames, n, STDERR_FILENO);
    ::signal(sig, SIG_DFL);
    ::raise(sig);
}

void install_abort_backtrace_handler() {
    ::signal(SIGABRT, abort_backtrace_handler);
    ::signal(SIGBUS, abort_backtrace_handler);
    ::signal(SIGFPE, abort_backtrace_handler);
    ::signal(SIGILL, abort_backtrace_handler);
    ::signal(SIGSEGV, abort_backtrace_handler);
}
#else
void install_abort_backtrace_handler() {}
#endif

} // namespace

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
    using kids::KidsData;
    using kids::KidsDataKind;
    using tula::logging::timeit;

#if CITLALI_HAS_HDF5
    // netCDF may probe optional HDF5 quantization attributes; suppress noisy
    // HDF5 diagnostics when those attributes are absent.
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
#endif

    // get current level
    auto log_level = spdlog::get_level();

    // vector to hold sink pointers
    std::vector<spdlog::sink_ptr> sinks_default;
    // create sink for default logger
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    // only show kidscpp critical logs
    sink->set_level(spdlog::level::critical);
    sinks_default.push_back(sink);
    // create default logger
    auto default_logger = std::make_shared<spdlog::logger>("console", begin(sinks_default), end(sinks_default));
    // register logger
    spdlog::register_logger(default_logger);
    // overwrite default logger
    spdlog::set_default_logger(default_logger);
    default_logger->flush_on(spdlog::level::info);

    // vector to hold sink pointers
    std::vector<spdlog::sink_ptr> sinks;
    // create console sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.push_back(console_sink);
    // create citlali logger
    auto logger = std::make_shared<spdlog::logger>("citlali_logger", begin(sinks), end(sinks));
    spdlog::register_logger(logger);
    logger->flush_on(spdlog::level::info);

    // set global level
    spdlog::set_level(log_level);
    // set pattern for logger
    //spdlog::set_pattern("[%H:%M:%S %z] [%s] %v");

    install_abort_backtrace_handler();

    logger->info("use KIDs data spec: {}", predefs::kidsdata::name);

    std::vector<std::string> config_filepaths;

    auto citlali_config =
        citlali::cli::load_config_files<rc_t, tula::config::YamlConfig>(
            rc, config_filepaths, logger,
            [](const std::string &filepath) {
                return tula::config::YamlConfig::from_filepath(filepath);
            },
            [](tula::config::YamlConfig lhs,
               const tula::config::YamlConfig &rhs) {
                return tula::config::merge(lhs, rhs);
            });

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
    // check if config file has a reduction type parameter
    if (citlali_config.has(std::tuple{"runtime", "reduction_type"})) {
        try {
            auto reduction_type =
                citlali_config.get_str(std::tuple{"runtime", "reduction_type"});

            // check for science mode
            if (reduction_type == "science") {
                logger->info("reducing in science mode");
                todproc.emplace<TimeOrderedDataProc<Lali>>(
                    TimeOrderedDataProc<Lali>::from_config(citlali_config));
            }

            // check for pointing mode
            else if (reduction_type == "pointing") {
                logger->info("reducing in pointing mode");
                todproc.emplace<TimeOrderedDataProc<Pointing>>(
                    TimeOrderedDataProc<Pointing>::from_config(citlali_config));
            }

            // check for beammap mode
            else if (reduction_type == "beammap") {
                logger->info("reducing in beammap mode");
                todproc.emplace<TimeOrderedDataProc<Beammap>>(
                    TimeOrderedDataProc<Beammap>::from_config(citlali_config));
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
                std::tuple{"runtime", "reduction_type"},
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
            std::tuple{"runtime", "reduction_type"},
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
                using array_indices_t = typename todproc_t::array_indices_t;

                // create vectors for map size and grouping parameters
                std::vector<map_extent_t> map_extents{};
                std::vector<map_coord_t> map_coords{};

                if (!citlali::pipeline::load_and_validate_engine_config(
                        todproc.engine(), citlali_config, logger)) {
                    std::cerr << fmt::format("missing keys={}", todproc.engine().missing_keys) << "\n";
                    std::cerr << fmt::format("invalid keys={}", todproc.engine().invalid_keys) << "\n";

                    return EXIT_FAILURE;
                }

                citlali::pipeline::configure_verbose_logging_if_requested(
                    todproc.engine(), logger, []() {
                        spdlog::set_level(spdlog::level::debug);
                    });

                // set omp parallelization explicitly
                omp_set_num_threads(todproc.engine().n_threads);
                // disable eigen underlying parallelization
                Eigen::setNbThreads(1);

                // set fftw threads
                const int fftw_init_ok = fftw_init_threads();
                if (!fftw_init_ok) {
                    logger->warn("unable to initialize FFTW threading; using default FFTW behavior");
                }
                int fftw_n_threads = todproc.engine().n_threads;
#if defined(CITLALI_USE_WIENER_FILTER_OMP)
                // Avoid nested parallelism: Wiener OMP path already parallelizes over work units.
                fftw_n_threads = 1;
#endif
                if (fftw_init_ok) {
                    fftw_plan_with_nthreads(fftw_n_threads);
                    logger->info("configured FFTW plan threads={}", fftw_n_threads);
                }

                if (!citlali::pipeline::prepare_initial_reduction_geometry<
                        std::is_same_v<todproc_t,
                                       TimeOrderedDataProc<Beammap>>,
                        KidsDataProc>(
                        todproc, co, citlali_config, map_extents,
                        map_coords, logger)) {
                    return EXIT_FAILURE;
                }

                if (!citlali::pipeline::run_reduction_iterations<
                        std::is_same_v<todproc_t,
                                       TimeOrderedDataProc<Beammap>>,
                        mapmaking::RawObs, mapmaking::FilteredObs,
                        mapmaking::RawCoadd, mapmaking::FilteredCoadd,
                        std::is_same_v<todproc_t,
                                       TimeOrderedDataProc<Pointing>>,
                        KidsDataProc>(
                        todproc, co, citlali_config, config_filepaths,
                        map_extents, map_coords,
                        [](auto &engine) {
                            return engine_utils::unix_to_utc(
                                engine.telescope.tel_data["TelTime"](0));
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
    sink->set_level(log_level);

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
