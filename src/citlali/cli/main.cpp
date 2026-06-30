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

    // load the yaml citlali config
    // this will merge the list of config files in rc
    tula::config::YamlConfig citlali_config;
    auto node_config_files = rc.get_node("config_file");
    for (const auto & n: node_config_files) {
        auto filepath = n.as<std::string>();
        config_filepaths.push_back(filepath);
        logger->info("load config from file {}", filepath);
        citlali_config = tula::config::merge(citlali_config, tula::config::YamlConfig::from_filepath(filepath));
    }

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

                // get config options from citlali_config
                logger->info("getting citlali config");
                todproc.engine().get_citlali_config(citlali_config);

                // exit if missing or invalid config options
                if (!todproc.engine().missing_keys.empty() || !todproc.engine().invalid_keys.empty()) {
                    logger->error("missing or invalid keys were found!");
                    logger->error("see for default config: https://github.com/toltec-astro/citlali/blob/v4.x/data/config.yaml");
                    std::cerr << fmt::format("missing keys={}", todproc.engine().missing_keys) << "\n";
                    std::cerr << fmt::format("invalid keys={}", todproc.engine().invalid_keys) << "\n";

                    return EXIT_FAILURE;
                }

                // if running in verbose mode, set log level to debug
                if (todproc.engine().verbose_mode) {
                    spdlog::set_level(spdlog::level::debug);
                    logger->debug("running in verbose mode. setting log level=debug.");
                }

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

                // set up the coadded map buffer by reading in each observation
                int i = 0;
                logger->info("starting initial loop through input obs");
                for (const auto &rawobs : co.inputs()) {
                    logger->info("starting setup of observation {}/{}", i + 1, co.n_inputs());
                    // set up KIDs data proc
                    auto kidsproc =
                        KidsDataProc::from_config(citlali_config.get_config("kids"));
                    i++;
                    // this is needed to figure out the data sample rate
                    // and number of detectors
                    logger->debug("getting rawobs kids meta info");
                    auto rawobs_kids_meta = kidsproc.get_rawobs_meta(rawobs);

                    citlali::pipeline::configure_observation_calibration<
                        std::is_same_v<todproc_t, TimeOrderedDataProc<Beammap>>>(
                        todproc, rawobs, logger);

                    if (!citlali::pipeline::apply_flxscale_correction(
                            todproc.engine(), rawobs, logger)) {
                        return EXIT_FAILURE;
                    }

                    // check input files
                    logger->debug("checking inputs");
                    todproc.check_inputs(rawobs);

                    // get sample rate
                    logger->debug("getting sample rate");
                    todproc.engine().telescope.fsmp = rawobs_kids_meta.back().get_typed<double>("fsmp");

                    citlali::pipeline::load_and_align_telescope_data(
                        todproc, rawobs, logger);

                    // calc tangent plane pointing
                    logger->info("calculating tangent plane pointing");
                    todproc.engine().telescope.calc_tan_pointing();

                    // calc pointing offsets
                    logger->info("calculating pointing offsets");
                    todproc.interp_pointing();

                    // calc scan indices
                    logger->info("calculating scan indices");
                    todproc.engine().telescope.calc_scan_indices();

                    if (todproc.engine().run_mapmaking) {
                        // determine number of maps
                        logger->info("calculating number of maps");
                        todproc.calc_map_num();

                        // determine omb map sizes
                        logger->info("calculating obs map dimensions");
                        todproc.calc_omb_size(map_extents, map_coords);
                    }
                }

                if (todproc.engine().run_coadd) {
                    // get size of coadd buffer
                    logger->info("calculating cmb dimensions");
                    todproc.calc_cmb_size(map_coords);
                }

                // current fruit loops iteration
                todproc.engine().fruit_iter = 0;
                // fruit loops convergence check
                bool fruit_loops_converged = false;

                citlali::pipeline::configure_fruit_loop_iteration_policy(
                    todproc.engine(), logger);

                // loop through fruit loops iterations
                while ((todproc.engine().fruit_iter < todproc.engine().ptcproc.fruit_loops_iters) && !fruit_loops_converged) {
                    citlali::pipeline::begin_fruit_loop_iteration(
                        todproc.engine(), logger);

                    // setup redu dirs if saving outputs or on first iter
                    if (todproc.engine().ptcproc.save_all_iters || todproc.engine().fruit_iter == 0) {
                        // setup reduction directories
                        todproc.create_output_dir();

                        citlali::pipeline::copy_config_files_to_reduction_dir(
                            config_filepaths, todproc.engine().redu_dir_name,
                            logger);
                    }

                    // clear obs dates
                    todproc.engine().date_obs.clear();

                    if (todproc.engine().run_coadd) {
                        citlali::pipeline::prepare_coadd_iteration_buffers(
                            todproc, logger);
                    }

                    // run the reduction for each observation
                    for (std::size_t i=0; i<co.n_inputs(); ++i) {
                        logger->info("starting reduction of observation {}/{}", i + 1, co.n_inputs());
                        // set up KIDs data proc
                        auto kidsproc =
                            KidsDataProc::from_config(citlali_config.get_config("kids"));

                        // get current rawobs
                        const auto &rawobs = co.inputs()[i];

                        // this is needed to figure out the data sample rate
                        // and number of detectors
                        logger->debug("getting rawobs kids meta info");
                        auto rawobs_kids_meta = kidsproc.get_rawobs_meta(rawobs);

                        if (co.n_inputs() > 1) {
                            citlali::pipeline::configure_observation_calibration<
                                std::is_same_v<todproc_t,
                                               TimeOrderedDataProc<Beammap>>>(
                                todproc, rawobs, logger);

                            if (!citlali::pipeline::apply_flxscale_correction(
                                    todproc.engine(), rawobs, logger)) {
                                return EXIT_FAILURE;
                            }

                            // get sample rate
                            logger->debug("getting sample rate");
                            todproc.engine().telescope.fsmp = rawobs_kids_meta.back().get_typed<double>("fsmp");
                        }

                        if (!citlali::pipeline::configure_effective_sample_rate(
                                todproc.engine(), logger)) {
                            return EXIT_FAILURE;
                        }

                        // get tone frequencies from raw files for flagging nearby tones
                        logger->debug("getting tone frequencies");
                        todproc.get_tone_freqs_from_files(rawobs);

                        // get adc snap data for stats file
                        if (!todproc.engine().telescope.sim_obs) {
                            logger->debug("getting adc snap data");
                            todproc.get_adc_snap_from_files(rawobs);
                        }

                        // get obsnum
                        logger->debug("getting obsnum");
                        const int obsnum = rawobs_kids_meta.back().get_typed<int>("obsid");
                        citlali::pipeline::prepare_observation_output_layout(
                            todproc.engine(), obsnum, logger);

                        citlali::pipeline::load_hwpr_data_if_requested(
                            todproc.engine(), rawobs, logger);

                        // get flux calibration
                        logger->info("calculating flux calibration");
                        todproc.engine().calib.calc_flux_calibration(todproc.engine().omb.sig_unit,todproc.engine().omb.pixel_size_rad);

                        // get telescope file
                        if (co.n_inputs() > 1) {
                            citlali::pipeline::load_and_align_telescope_data(
                                todproc, rawobs, logger);

                            // calc tangent plane pointing
                            logger->info("calculating tangent plane pointing");
                            todproc.engine().telescope.calc_tan_pointing();

                            // calc pointing offsets
                            logger->info("calculating pointing offsets");
                            todproc.interp_pointing();
                        }

                        // get date time of observation
                        todproc.engine().date_obs.push_back(engine_utils::unix_to_utc(todproc.engine().telescope.tel_data["TelTime"](0)));

                        citlali::pipeline::record_timing_gaps_if_needed(
                            todproc.engine(), logger);

                        if (co.n_inputs() > 1) {
                            // calc scan indices
                            logger->info("calculating scan indices");
                            todproc.engine().telescope.calc_scan_indices();
                        }

                        // allocate observation map buffer
                        if (todproc.engine().run_mapmaking) {
                            citlali::pipeline::allocate_observation_map_buffers(
                                todproc, map_extents[i], map_coords[i], logger);
                        }

                        citlali::pipeline::update_observation_exposure_time(
                            todproc.engine());

                        if constexpr (!std::is_same_v<todproc_t, TimeOrderedDataProc<Beammap>>) {
                            // if on first fruit loops iteration and a path is specified
                            citlali::pipeline::load_initial_fruit_loop_model_if_requested(
                                todproc.engine());

                            // if on iteration >0 get the maps from the previous iteration
                            citlali::pipeline::load_previous_fruit_loop_model_if_needed(
                                todproc.engine(), logger);
                        }

                        citlali::pipeline::setup_and_run_observation_pipeline(
                            todproc.engine(), kidsproc, rawobs, logger);

                        citlali::pipeline::write_raw_observation_outputs<
                            mapmaking::RawObs>(todproc, logger);

                        // coadd
                        if (todproc.engine().run_coadd) {
                            logger->info("coadding");
                            if (!todproc.engine().rtcproc.run_polarization) {
                                todproc.coadd();
                            }
                        }

                        // filter obs map
                        else if (todproc.engine().run_map_filter) {
                            citlali::pipeline::write_filtered_observation_outputs<
                                mapmaking::FilteredObs,
                                std::is_same_v<todproc_t,
                                               TimeOrderedDataProc<Pointing>>>(
                                todproc, logger);
                        }
                    }

                    if (todproc.engine().run_coadd) {
                        citlali::pipeline::write_raw_coadd_outputs<
                            mapmaking::RawCoadd>(todproc, logger);

                        if (todproc.engine().run_map_filter) {
                            citlali::pipeline::write_filtered_coadd_outputs<
                                mapmaking::FilteredCoadd>(todproc, logger);
                        }
                    }

                    citlali::pipeline::finalize_fruit_loop_iteration(
                        todproc.engine(), logger);

                    logger->info("making index files");
                    // make index files for each directory recursively
                    todproc.make_index_file(todproc.engine().redu_dir_name);

                    // increment fruit loops iteration
                    todproc.engine().fruit_iter++;
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
