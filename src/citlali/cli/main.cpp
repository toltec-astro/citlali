#include <regex>
#include <fftw3.h>
#include <CCfits/CCfits>
#ifdef _OPENMP
# include <omp.h>
#endif

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
#include <tula/algorithm/mlinterp/mlinterp.hpp>

#include <citlali/core/pipeline/controls.h>
#include <citlali/core/utils/threads.h>
#include <citlali/core/pipeline/engine.h>

#include <citlali/core/pipeline/io.h>
#include <citlali/core/pipeline/kidsproc.h>
#include <citlali/core/pipeline/todproc.h>

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
                      "citlali" , CITLALI_PROJECT_NAME, ver_str,
                                  CITLALI_PROJECT_DESCRIPTION};
    auto [cli, rc, cc] = parse([&](auto &r, auto &c) {
    return (
        // rc -- runtime config
        // cc -- cli config
        c(p("h", "help"), "Print help information and exit."),
        c(p("version"), "Print version information and exit."),
        r("config_file", "The path of the input config file. Multiple config files are merged in order.", opt_strs()),
        c(p("dump_config"), "Print the default config file to STDOUT."),

        "common options" % g(
            c(p("l", "log_level"), "Set the log level.", default_level_name, list(level_names)),
            r(p("grppiex"), "GRPPI execution policy.", ex_config::default_mode(), list(ex_config::mode_names_supported()))
        )
    );
    }, screen, argc, argv);
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

auto setup_citlali_logger() {
    // get default log level
    auto log_level = spdlog::get_level();
    // vector to hold sink pointers
    std::vector<spdlog::sink_ptr> sinks_default;
    // create sink for default logger
    auto default_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    // disable logging
    default_sink->set_level(spdlog::level::off);
    sinks_default.push_back(default_sink);
    // create default logger
    auto default_logger = std::make_shared<spdlog::logger>("console", begin(sinks_default), end(sinks_default));
    // register logger
    spdlog::register_logger(default_logger);
    // overwrite default logger
    spdlog::set_default_logger(default_logger);

    // vector to hold sink pointers
    std::vector<spdlog::sink_ptr> sinks;
    // create console sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.push_back(console_sink);
    // create citlali logger
    auto logger = std::make_shared<spdlog::logger>("citlali_logger", begin(sinks), end(sinks));
    spdlog::register_logger(logger);

    // set global level
    spdlog::set_level(log_level);

    return default_sink;
}

int run(const rc_t& rc) {
    using kids::KidsData;
    using kids::KidsDataKind;
    using tula::logging::timeit;

    // manage logging
    auto default_sink = setup_citlali_logger();
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // merge the list of config files in rc
    std::vector<std::string> config_filepaths;

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
    auto kidsproc =
        KidsDataProc::from_config(citlali_config.get_config("kids"));

    // get todproc
    TimeOrderedDataProc<Engine> todproc = TimeOrderedDataProc<Engine>::from_config(citlali_config);

    // create an alias for engine
    auto& engine = todproc.engine();

    // gets and checks config values
    ConfigValidator config_container(citlali_config);

    // get shared config options
    logger->info("getting shared config options");
    engine.get_configs(config_container);

    // build the pipelines while also populating the configuration options for the reduction stages
    logger->info("building pipelnes");
    engine.build_pipelines(config_container);

    // exit if missing or invalid config options
    if (!config_container.missing_keys.empty() || !config_container.invalid_keys.empty()) {
        logger->error("missing keys: {}", config_container.missing_keys);
        logger->error("invalid keys: {}", config_container.invalid_keys);;

        throw std::runtime_error(
            "missing or invalid keys were found! "
            "see for default config: https://github.com/toltec-astro/citlali/blob/v5.x/data/config.yaml"
            );
    }

    // if running in verbose mode, set log level to debug
    if (verbose) {
        spdlog::set_level(spdlog::level::debug);
        logger->debug("running in verbose mode. setting log level=debug.");
    }

    // preliminary setup for all maps
    logger->info("setting up maps");
    todproc.setup_maps();

    // set up threads for Eigen and threading library
    logger->info("setting parallelization");
    todproc.set_parallelization();    

    // set up the coadded map buffer by reading in each observation
    logger->info("starting initial loop through input obs");
    for (int i = 0; i < co.n_inputs(); ++i) {
        logger->info("starting setup of observation {}/{}", i + 1, co.n_inputs());

        // get current rawobs
        const auto& rawobs = co.inputs()[i];

        // kids data meta information
        logger->info("getting rawobs kids meta info");
        auto rawobs_kids_meta = kidsproc.get_rawobs_meta(rawobs);

        // get sample rate
        engine.toltec.data_fs_hz = rawobs_kids_meta.back().get_typed<double>("fsmp");

        // get obsnum
        todproc.obsnums.push_back(fmt::format("{:06}", rawobs_kids_meta.back().get_typed<int>("obsid")));

        // run tod setup processes for current observation
        logger->info("running obs setup");
        todproc.setup_obs_tod(rawobs);

        // calc number and size of obs map for current observation
        logger->info("calculating obs map number and dimensions");
        todproc.calc_obs_map_dims();
    }

    // only one iteration if not running fruit loops
    if (!run_fruit_loops) {
        fruit_iters = 1;
        save_all_fruit_iters = true;
    }

    // fruit loops iterations
    for (int iter = 0; iter < fruit_iters; ++iter) {
        logger->info("starting fruit iteration {}", iter);
        // allocate coadd maps
        if (run_map_coadd) {
            logger->info("setting up and allocating coadded maps");
            // reset maps
            engine.coadd_maps = ObsMaps<>();
            todproc.allocate_coadded_maps();

            // allocate coadd noise maps
            if (run_noise_maps) {
                logger->info("setting up and allocating coadded noise maps");
                // reset maps
                engine.noise_maps = ObsMaps<MapKey, std::vector<ObsMatrix<MapKey>>>();
                todproc.allocate_noise_maps(engine.coadd_maps);
            }
        }

        // create reduction directories
        if (save_all_fruit_iters || iter == 0) {
            logger->info("creating output directories");
            todproc.setup_directories();

            // copy config files to reduction directory
            for (const auto &config_filepath : config_filepaths) {
                logger->debug("copying config files into reduction directory");

                // extract filename from path
                std::string config_name = fs::path(config_filepath).filename().string();

                // copy file to reduction directory with overwrite
                fs::copy(config_filepath, engine.reduction_directory + "/" + config_name,
                         fs::copy_options::overwrite_existing);
            }
        }

        // run the reduction for each observation
        for (int i = 0; i < co.n_inputs(); ++i) {
            logger->info("starting reduction of observation {}/{}", i + 1, co.n_inputs());

            // copy obsnum
            engine.obsnum = todproc.obsnums[i];

            // get fruit loops path
            if (run_fruit_loops) {
                auto result = engine.ptc_pipeline.get_component("FruitLoops");
                if (result) {
                    auto& [index, component] = result.value();
                    auto fruit_ptr = dynamic_cast<FruitLoops<TCData>*>(component);

                    if (fruit_ptr) {
                        fruit_ptr->fruit_iter = iter;
                        std::string base_path;

                        if (iter == 0 && fruit_ptr->fruit_path != "null") {
                            const auto& fruit_source = fruit_ptr->fruit_source;
                            const std::string& fruit_path = fruit_ptr->fruit_path;

                            if (fruit_source == "obsnum/raw") {
                                base_path = fruit_path + "/" + engine.obsnum + "/raw/";
                            } else if (fruit_source == "obsnum/filtered") {
                                base_path = fruit_path + "/" + engine.obsnum + "/filtered/";
                            } else if (fruit_source == "coadded/raw") {
                                base_path = fruit_path + "/coadded/raw/";
                            } else if (fruit_source == "coadded/filtered") {
                                base_path = fruit_path + "/coadded/filtered/";
                            }
                        } else {
                            std::string redu_path = engine.reduction_directory;
                            if (save_all_fruit_iters) {
                                redu_path = redu_path.substr(0, redu_path.size() - 2) +
                                            fmt::format("{:02}", std::stoi(redu_path.substr(redu_path.size() - 2)) - 1);
                            }

                            const auto& fruit_source = fruit_ptr->fruit_source;
                            if (fruit_source == "obsnum/raw") {
                                base_path = redu_path + "/" + engine.obsnum + "/raw/";
                            } else if (fruit_source == "obsnum/filtered") {
                                base_path = redu_path + "/" + engine.obsnum + "/filtered/";
                            } else if (fruit_source == "coadded/raw") {
                                base_path = redu_path + "/coadded/raw/";
                            } else if (fruit_source == "coadded/filtered") {
                                base_path = redu_path + "/coadded/filtered/";
                            }
                        }

                        fruit_ptr->curr_fruit_dir = base_path;
                    }
                }
            }

            // get current rawobs
            const auto& rawobs = co.inputs()[i];

            // kids data meta information
            logger->info("getting rawobs kids meta info");
            auto rawobs_kids_meta = kidsproc.get_rawobs_meta(rawobs);

            // only re-run if more than one obs
            if (co.n_inputs() > 1) {
                // get sample rate
                engine.toltec.data_fs_hz = rawobs_kids_meta.back().get_typed<double>("fsmp");

                // run tod setup processes for current observation
                logger->info("running obs setup");
                todproc.setup_obs_tod(rawobs);
            }

            logger->info("allocating obs maps");
            // reset maps
            engine.obs_maps = ObsMaps<>();
            for (const auto& array: engine.toltec.apt.arrays) {
                for (const auto& unique_key : todproc.unique_map_keys[array]) {
                    MapKey i_key(array, unique_key, "I");
                    engine.obs_maps.add(i_key, {todproc.map_extents[i].first, todproc.map_extents[i].second},
                                        true, run_kernel, map_grouping !="uid");

                    if (run_polarization) {
                        MapKey q_key(array, unique_key, "Q");
                        engine.obs_maps.add(q_key, {todproc.map_extents[i].first, todproc.map_extents[i].second},
                                            true, false, false);

                        MapKey u_key(array, unique_key, "U");
                        engine.obs_maps.add(u_key, {todproc.map_extents[i].first, todproc.map_extents[i].second},
                                            true, false, false);
                    }
                }
            }

            // setup map wcs
            engine.obs_maps.wcs.set(engine.telescope.pixel_axes, engine.telescope.x0, engine.telescope.y0,
                                    todproc.map_extents[i].first, todproc.map_extents[i].second, pix_size_radians,
                                    engine.telescope.header.at("Source.Epoch")(0));

            // absolute coordinates
            engine.obs_maps.rows = todproc.map_coords[i].first;
            engine.obs_maps.cols = todproc.map_coords[i].second;

            // allocate noise maps if not coadding
            if (!run_map_coadd && run_noise_maps) {
                logger->info("setting up and allocating obs noise maps");
                // reset maps
                engine.noise_maps = ObsMaps<MapKey, std::vector<ObsMatrix<MapKey>>>();
                todproc.allocate_noise_maps(engine.obs_maps);
            }

            // divide threads between chunks, maps, and detectors
            citlali::utils::threads::set_optimal_threads(n_threads, engine.telescope.n_chunks, engine.obs_maps.n_maps, exec_mode);
            logger->debug("using {} thread(s) for time chunks", citlali::utils::threads::n_chunk_threads);
            logger->debug("using {} thread(s) for time chunk remainder", citlali::utils::threads::n_chunk_remainder_threads);
            logger->debug("using {} thread(s) for maps", citlali::utils::threads::n_map_threads);
            logger->debug("using {} thread(s) for map remainder", citlali::utils::threads::n_map_remainder_threads);

            // run the pipelines
            logger->info("running pipelines");
            engine.run_obs(kidsproc, rawobs);

            // output obs maps
            logger->info("outputting obs maps");
            todproc.output_maps(engine.obs_maps, "obs_maps", false);

            if (!run_map_coadd && run_noise_maps) {
                // output obs noise maps
                logger->info("outputting obs noise maps");
                todproc.output_maps(engine.noise_maps, "obs_noise", false);
            }
        }

        // run the map pipeline on the coadded maps
        if (run_map_coadd) {
            logger->info("processing coadded maps");
            engine.run_coadd();
            todproc.output_maps(engine.coadd_maps, "coadd_maps", false);

            if (run_map_coadd && run_noise_maps) {
                // output coadd noise maps
                logger->info("outputting coadd noise maps");
                todproc.output_maps(engine.noise_maps, "coadd_noise", false);
            }
        }

        logger->info("making index files");
        // make index files for each directory recursively
        todproc.make_index_file(engine.reduction_directory);
    }

    logger->info("I'm done...going to sleep now...wake me when you need me.");

    // re-enable default logger
    default_sink->set_level(spdlog::get_level());

    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
    // make sure dump_config output is not contaminated with any logging message.
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
    SPDLOG_INFO("rc {}", rc.pformat());
    if (rc.get_node("config_file").size() > 0) {
        tula::logging::scoped_timeit TULA_X{"Citlali Process"};
        return run(rc);
    } else {
        std::cout << "Invalid argument. Type --help for usage.\n";
    }
}
