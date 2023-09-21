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
#include <omp.h>
#include <regex>
#include <tuple>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/engine/lali.h>
#include <citlali/core/engine/pointing.h>
#include <citlali/core/engine/beammap.h>

#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"

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
    using kids::KidsData;
    using kids::KidsDataKind;
    using tula::logging::timeit;
    SPDLOG_INFO("use KIDs data spec: {}", predefs::kidsdata::name);

    std::vector<std::string> config_filepaths;

    // load the yaml citlali config
    // this will merge the list of config files in rc
    tula::config::YamlConfig citlali_config;
    auto node_config_files = rc.get_node("config_file");
    for (const auto & n: node_config_files) {
        auto filepath = n.as<std::string>();
        config_filepaths.push_back(filepath);
        SPDLOG_TRACE("load config from file {}", filepath);
        citlali_config = tula::config::merge(citlali_config, tula::config::YamlConfig::from_filepath(filepath));
    }

    // set up the IO coorindator
    auto co = SeqIOCoordinator::from_config(citlali_config);

    // set up KIDs data proc
    auto kidsproc =
        KidsDataProc::from_config(citlali_config.get_config("kids"));

    // set up todproc
    using todproc_var_t =
        std::variant<std::monostate, TimeOrderedDataProc<Lali>, TimeOrderedDataProc<Pointing>,
                     TimeOrderedDataProc<Beammap>>;

    todproc_var_t todproc;

    // set todproc to variant depending on the config file reduction type

    // check if config file has a reduction type parameter
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
                    TimeOrderedDataProc<Pointing>::from_config(citlali_config);
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
                using map_coord_abs_t = typename todproc_t::map_coord_abs_t;
                using map_count_t = typename todproc_t::map_count_t;
                using array_indices_t = typename todproc_t::array_indices_t;

                // create vectors for map size and grouping parameters
                std::vector<map_extent_t> map_extents{};
                std::vector<map_coord_t> map_coords{};
                std::vector<map_coord_abs_t> map_coords_abs{};
                std::vector<map_count_t> map_counts{};

                // get config options from citlali_config
                SPDLOG_INFO("getting citlali config");
                todproc.engine().get_citlali_config(citlali_config);

                // exit if missing or invalid config options
                if (!todproc.engine().missing_keys.empty() || !todproc.engine().invalid_keys.empty()) {
                    std::cerr << fmt::format("missing keys={}", todproc.engine().missing_keys) << "\n";
                    std::cerr << fmt::format("invalid keys={}", todproc.engine().invalid_keys) << "\n";

                    return EXIT_FAILURE;
                }

                // if running in verbose mode, set log level to debug
                if (todproc.engine().verbose_mode) {
                    spdlog::set_level(spdlog::level::debug);
                    SPDLOG_DEBUG("running in verbose mode. setting log level=debug.");
                }

                // set parallelization explicitly
                omp_set_num_threads(todproc.engine().n_threads);
                Eigen::setNbThreads(1);

                // set fftw threads
                int fftw_threads = fftw_init_threads();
                fftw_plan_with_nthreads(todproc.engine().n_threads);

                // setup reduction directories
                todproc.create_output_dir();

                // copy config files to reduction directory
                for (std::string &config_filepath : config_filepaths) {
                    std::size_t found = config_filepath.rfind("/");
                    if (found!=std::string::npos) {
                        std::string config_name = config_filepath.substr(found);
                        fs::copy(config_filepath, todproc.engine().redu_dir_name + "/" + config_name);
                    }
                    else {
                        fs::copy(config_filepath, todproc.engine().redu_dir_name + "/" + config_filepath);
                    }
                }

                // set up the coadded map buffer by reading in each observation
                for (const auto &rawobs : co.inputs()) {
                    // this is needed to figure out the data sample rate
                    // and number of detectors
                    SPDLOG_DEBUG("getting rawobs kids meta info");
                    auto rawobs_kids_meta = kidsproc.get_rawobs_meta(rawobs);

                    // get astrometry config options
                    SPDLOG_DEBUG("getting astrometry config");
                    todproc.engine().get_astrometry_config(rawobs.astrometry_calib_info().config());
                    // get photometry config options
                    if constexpr (std::is_same_v<todproc_t, TimeOrderedDataProc<Beammap>>) {
                        todproc.engine().get_photometry_config(rawobs.photometry_calib_info().config());

                        // if beammap generate the apt table from the files
                        if (todproc.engine().map_grouping=="detector" || todproc.engine().map_grouping=="auto") {
                            SPDLOG_INFO("making apt file from raw nc files");
                            todproc.get_apt_from_files(rawobs);
                        }
                        else {
                            auto apt_path = rawobs.array_prop_table().filepath();
                            SPDLOG_INFO("getting array properties table {}", apt_path);

                            // get raw files and interfaces
                            std::vector<std::string> raw_filenames, interfaces;
                            for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
                                raw_filenames.push_back(data_item.filepath());
                                interfaces.push_back(data_item.interface());
                            }

                            // get and setup apt table
                            todproc.engine().calib.get_apt(apt_path, raw_filenames, interfaces);
                        }
                    }

                    else {
                        // get apt table
                        auto apt_path = rawobs.array_prop_table().filepath();
                        SPDLOG_INFO("getting array properties table {}", apt_path);

                        std::vector<std::string> raw_filenames, interfaces;
                        for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
                            raw_filenames.push_back(data_item.filepath());
                            interfaces.push_back(data_item.interface());
                        }

                        todproc.engine().calib.get_apt(apt_path, raw_filenames, interfaces);
                    }

                    // check input files
                    SPDLOG_DEBUG("checking inputs");
                    todproc.check_inputs(rawobs);

                    // get sample rate
                    SPDLOG_DEBUG("getting sample rate");
                    todproc.engine().telescope.fsmp = rawobs_kids_meta.back().get_typed<double>("fsmp");

                    // get telescope file
                    auto tel_path = rawobs.teldata().filepath();
                    SPDLOG_INFO("getting telescope file {}", tel_path);
                    todproc.engine().telescope.get_tel_data(tel_path);

                    // overwrite map center
                    if (todproc.engine().omb.crval_config[0]!=0 && todproc.engine().omb.crval_config[1]!=0) {
                        SPDLOG_INFO("overwriting map center to ({}, {})",todproc.engine().omb.crval_config[0],
                                    todproc.engine().omb.crval_config[1]);
                        todproc.engine().telescope.tel_header["Header.Source.Ra"].setConstant(todproc.engine().omb.crval_config[0]);
                        todproc.engine().telescope.tel_header["Header.Source.Dec"].setConstant(todproc.engine().omb.crval_config[1]);
                    }

                    // align tod
                    if (!todproc.engine().telescope.sim_obs) {
                        SPDLOG_INFO("aligning timestreams");
                        todproc.align_timestreams(rawobs);
                    }

                    // if simu, set start and end indices to 0
                    else {
                        todproc.engine().start_indices.clear();
                        todproc.engine().end_indices.clear();

                        for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
                            todproc.engine().start_indices.push_back(0);
                            todproc.engine().start_indices.push_back(0);
                        }

                        if (todproc.engine().calib.run_hwp) {
                            todproc.engine().hwpr_start_indices = 0;
                            todproc.engine().hwpr_end_indices = 0;
                        }
                    }

                    // calc tangent plane pointing
                    SPDLOG_INFO("calculating tangent plane pointing");
                    todproc.engine().telescope.calc_tan_pointing();

                    // calc pointing offsets
                    SPDLOG_INFO("calculating pointing offsets");
                    todproc.interp_pointing();

                    // calc scan indices
                    SPDLOG_INFO("calculating scan indices");
                    todproc.engine().telescope.calc_scan_indices();

                    if (todproc.engine().run_mapmaking) {
                        // determine number of maps
                        SPDLOG_INFO("calculating number of maps");
                        todproc.calc_map_num();

                        // determine omb map sizes
                        SPDLOG_INFO("calculating omb dimensions");
                        todproc.calc_omb_size(map_extents, map_coords, map_coords_abs);
                    }
                }

                if (todproc.engine().run_coadd) {
                    // get size of coadd buffer
                    SPDLOG_DEBUG("calculating cmb dimensions");
                    todproc.calc_cmb_size(map_coords, map_coords_abs);
                    // make coadd buffer
                    SPDLOG_DEBUG("allocating cmb");
                    todproc.allocate_cmb();
                    // make noise maps for coadd map buffer
                    if (todproc.engine().run_noise) {
                        SPDLOG_DEBUG("allocating nmb");
                        todproc.allocate_nmb(todproc.engine().cmb);
                    }

                    // create output coadded map files
                    SPDLOG_DEBUG("creating cmb filenames");
                    todproc.create_coadded_map_files();
                }

                // run the reduction for each observation
                for (std::size_t i=0; i<co.n_inputs(); ++i) {
                    SPDLOG_INFO("starting reduction of observation {}/{}", i + 1, co.n_inputs());

                    // get current rawobs
                    const auto &rawobs = co.inputs()[i];

                    // this is needed to figure out the data sample rate
                    // and number of detectors
                    SPDLOG_DEBUG("getting rawobs kids meta info");
                    auto rawobs_kids_meta = kidsproc.get_rawobs_meta(rawobs);

                    if (co.n_inputs() > 1) {
                        // get astrometry config options
                        SPDLOG_DEBUG("getting astrometry config");
                        todproc.engine().get_astrometry_config(rawobs.astrometry_calib_info().config());
                        // get photometry config options
                        if constexpr (std::is_same_v<todproc_t, TimeOrderedDataProc<Beammap>>) {
                            todproc.engine().get_photometry_config(rawobs.photometry_calib_info().config());

                            // if beammap generate the apt table from the files
                            if (todproc.engine().map_grouping=="detector" || todproc.engine().map_grouping=="auto") {
                                SPDLOG_INFO("making apt file from raw nc files");
                                todproc.get_apt_from_files(rawobs);
                            }
                            else {
                                auto apt_path = rawobs.array_prop_table().filepath();
                                SPDLOG_INFO("getting array properties table {}", apt_path);

                                // get raw files and interfaces
                                std::vector<std::string> raw_filenames, interfaces;
                                for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
                                    raw_filenames.push_back(data_item.filepath());
                                    interfaces.push_back(data_item.interface());
                                }

                                // get and setup apt table
                                todproc.engine().calib.get_apt(apt_path, raw_filenames, interfaces);
                            }
                        }

                        // get apt file
                        else {
                            auto apt_path = rawobs.array_prop_table().filepath();
                            SPDLOG_INFO("getting array properties table {}", apt_path);

                            // get raw files and interfaces
                            std::vector<std::string> raw_filenames, interfaces;
                            for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
                                raw_filenames.push_back(data_item.filepath());
                                interfaces.push_back(data_item.interface());
                            }

                            // get and setup apt table
                            todproc.engine().calib.get_apt(apt_path, raw_filenames, interfaces);
                        }

                        // get sample rate
                        SPDLOG_DEBUG("getting sample rate");
                        todproc.engine().telescope.fsmp = rawobs_kids_meta.back().get_typed<double>("fsmp");
                    }

                    // calculate downsampling factor
                    if (todproc.engine().rtcproc.run_downsample) {
                        if (todproc.engine().rtcproc.downsampler.factor <= 0) {
                            if (todproc.engine().rtcproc.downsampler.downsampled_freq_Hz > todproc.engine().telescope.fsmp) {
                                SPDLOG_ERROR("downsampled freq ({} Hz) must be less than sample rate ({} Hz)",
                                             todproc.engine().rtcproc.downsampler.downsampled_freq_Hz,
                                             todproc.engine().telescope.fsmp);
                                return EXIT_FAILURE;
                            }
                            todproc.engine().rtcproc.downsampler.factor = std::floor(todproc.engine().telescope.fsmp /
                                                                                     todproc.engine().rtcproc.downsampler.downsampled_freq_Hz);
                        }
                    }

                    // calculate downsampled sample rate
                    if (todproc.engine().rtcproc.run_downsample) {
                        todproc.engine().telescope.d_fsmp = todproc.engine().telescope.fsmp/todproc.engine().rtcproc.downsampler.factor;
                    }
                    else {
                        todproc.engine().telescope.d_fsmp = todproc.engine().telescope.fsmp;
                    }

                    // get tone frequencies from raw files for flagging nearby tones
                    SPDLOG_DEBUG("getting tone frequencies");
                    todproc.get_tone_freqs_from_files(rawobs);

                    // get adc snap data for stats file
                    SPDLOG_DEBUG("getting adc snap data");
                    todproc.get_adc_snap_from_files(rawobs);

                    // get obsnum
                    SPDLOG_DEBUG("getting obsnum");
                    int obsnum = rawobs_kids_meta.back().get_typed<int>("obsid");

                    // convert obsnum to string with leading zeros
                    std::stringstream ss;
                    ss << std::setfill('0') << std::setw(6) << obsnum;

                    // add obsnum to todproc for file/directory names
                    todproc.engine().obsnum = ss.str();
                    // set up obsnum directory name
                    todproc.engine().obsnum_dir_name = todproc.engine().redu_dir_name + "/" + todproc.engine().obsnum +"/";

                    // add obsnum to omb for fits hdu headers
                    todproc.engine().omb.obsnums.clear();
                    todproc.engine().omb.obsnums.push_back(todproc.engine().obsnum);

                    // add current obsnum to cmb for fits hdu headers
                    todproc.engine().cmb.obsnums.push_back(todproc.engine().obsnum);

                    // create obsnum directory
                    SPDLOG_DEBUG("creating obsnum directory");
                    fs::create_directories(todproc.engine().obsnum_dir_name);

                    // create raw obsnum directory
                    SPDLOG_DEBUG("creating obsnum raw directory");
                    fs::create_directories(todproc.engine().obsnum_dir_name + "raw/");

                    // create filtered obsnum directory
                    if (!todproc.engine().run_coadd) {
                        if (todproc.engine().run_map_filter) {
                            SPDLOG_DEBUG("creating obsnum filtered directory");
                            fs::create_directories(todproc.engine().obsnum_dir_name + "filtered/");
                        }
                    }
                    // create log directory for verbose mode
                    if (todproc.engine().verbose_mode) {
                        SPDLOG_DEBUG("creating obsnum logs directory");
                        fs::create_directories(todproc.engine().obsnum_dir_name + "logs/");

                        /*
                        std::vector<spdlog::sink_ptr> sinks;

                        // console sink
                        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
                        console_sink->set_level(spdlog::level::debug);
                        sinks.push_back(console_sink);

                        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(todproc.engine().obsnum_dir_name +
                                                                                                 "logs/test.txt", true);
                        file_sink->set_level(spdlog::level::debug);

                        sinks.push_back(file_sink);

                        auto logger = std::make_shared<spdlog::logger>("logger", begin(sinks), end(sinks));
                        spdlog::register_logger(logger);
                        spdlog::set_default_logger(logger);
                        */
                    }

                    // get hwpr if polarized reduction is requested
                    if (todproc.engine().rtcproc.run_polarization) {
                        std::string hwpr_filepath;
                        // if hwpr file dict is found in config and we're not ignoring it
                        if (rawobs.hwpdata().has_value() && todproc.engine().calib.ignore_hwpr!="true") {
                            // get hwpr filepath
                            hwpr_filepath = rawobs.hwpdata()->filepath();
                            // if filepath is not null, get the hwpr data
                            if (hwpr_filepath != "null") {
                                SPDLOG_INFO("getting hwpr file");
                                todproc.engine().calib.get_hwpr(hwpr_filepath, todproc.engine().telescope.sim_obs);
                            }
                            else {
                                todproc.engine().calib.run_hwp = false;
                            }
                        }
                        else {
                            todproc.engine().calib.run_hwp = false;
                        }

                        if (!todproc.engine().calib.run_hwp) {
                            SPDLOG_INFO("ignoring hwpr");
                        }
                    }

                    // get flux calibration
                    SPDLOG_INFO("calculating flux calibration");
                    todproc.engine().calib.calc_flux_calibration(todproc.engine().omb.sig_unit);

                    // get telescope file
                    if (co.n_inputs() > 1) {
                        auto tel_path = rawobs.teldata().filepath();
                        SPDLOG_INFO("getting telescope file {}", tel_path);
                        todproc.engine().telescope.get_tel_data(tel_path);

                        // overwrite map center
                        if (todproc.engine().omb.crval_config[0]!=0 && todproc.engine().omb.crval_config[1]!=0) {
                            SPDLOG_INFO("overwriting map center to ({}, {})",todproc.engine().omb.crval_config[0],
                                        todproc.engine().omb.crval_config[1]);
                            todproc.engine().telescope.tel_header["Header.Source.Ra"].setConstant(todproc.engine().omb.crval_config[0]);
                            todproc.engine().telescope.tel_header["Header.Source.Dec"].setConstant(todproc.engine().omb.crval_config[1]);
                        }

                        // align tod
                        if (!todproc.engine().telescope.sim_obs) {
                            SPDLOG_INFO("aligning timestreams");
                            todproc.align_timestreams(rawobs);
                        }

                        // if simu, set start and end indices to 0
                        else {
                            todproc.engine().start_indices.clear();
                            todproc.engine().end_indices.clear();

                            for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
                                todproc.engine().start_indices.push_back(0);
                                todproc.engine().start_indices.push_back(0);

                                if (todproc.engine().calib.run_hwp) {
                                    todproc.engine().hwpr_start_indices = 0;
                                    todproc.engine().hwpr_end_indices = 0;
                                }
                            }
                        }

                        // calc tangent plane pointing
                        SPDLOG_INFO("calculating tangent plane pointing");
                        todproc.engine().telescope.calc_tan_pointing();

                        // calc pointing offsets
                        SPDLOG_INFO("calculating pointing offsets");
                        todproc.interp_pointing();
                    }

                    // get date time of observation
                    todproc.engine().date_obs.push_back(engine_utils::unix_to_utc(todproc.engine().telescope.tel_data["TelTime"](0)));

                    // warning for gaps in data
                    if (todproc.engine().gaps.size() > 0) {
                        SPDLOG_WARN("gaps found in obnsum {} data file timing!", todproc.engine().obsnum);
                        // write gaps.log file if in verbose mode
                        if (todproc.engine().verbose_mode) {
                            SPDLOG_DEBUG("writing gaps.log file");
                            std::ofstream f;
                            f.open(todproc.engine().obsnum_dir_name+"/logs/gaps.log");
                            f << "Summary of timing gaps\n";
                            for (auto const& [key, val] : todproc.engine().gaps) {
                                SPDLOG_DEBUG("{} gaps: {}", key, val);
                                f << "-" + key + " gaps: " << val << "\n";
                            }
                            f.close();
                        }
                    }

                    if (co.n_inputs() > 1) {
                        // calc scan indices
                        SPDLOG_INFO("calculating scan indices");
                        todproc.engine().telescope.calc_scan_indices();
                    }

                    // allocate observation map buffer
                    if (todproc.engine().run_mapmaking) {
                        SPDLOG_INFO("allocating obs map buffer");
                        todproc.allocate_omb(map_extents[i], map_coords[i], map_coords_abs[i]);

                        // make noise maps for observation map buffer
                        if (!todproc.engine().run_coadd) {
                            if (todproc.engine().run_noise) {
                                SPDLOG_INFO("allocating obs noise maps");
                                todproc.allocate_nmb(todproc.engine().omb);
                            }
                        }
                    }

                    // calc exposure time estimate
                    auto t0 = todproc.engine().telescope.tel_data["TelTime"](0);
                    auto tn = todproc.engine().telescope.tel_data["TelTime"](todproc.engine().telescope.tel_data["TelTime"].size()-1);

                    todproc.engine().omb.exposure_time = tn - t0;
                    todproc.engine().cmb.exposure_time = todproc.engine().cmb.exposure_time + todproc.engine().omb.exposure_time;

                    // setup
                    SPDLOG_INFO("pipeline setup");
                    todproc.engine().setup();

                    // run
                    if (todproc.engine().run_tod) {
                        SPDLOG_INFO("running pipeline");
                        todproc.engine().pipeline(kidsproc, rawobs);
                    }

                    // output
                    if (todproc.engine().run_mapmaking) {
                        SPDLOG_INFO("outputting raw obs files");
                        todproc.engine().template output<mapmaking::RawObs>();
                    }

                    // coadd
                    if (todproc.engine().run_coadd) {
                        SPDLOG_INFO("coadding");
                        todproc.coadd();
                    }

                    // filter obs map
                    else if (todproc.engine().run_map_filter) {
                        SPDLOG_INFO("filtering obs maps");
                        todproc.engine().template run_wiener_filter<mapmaking::FilteredObs>(todproc.engine().omb);

                        // calculate filtered obs map psds
                        SPDLOG_INFO("calculating filtered obs map psds");
                        todproc.engine().omb.calc_map_psd();
                        // calculate filtered obs map histograms
                        SPDLOG_INFO("calculating filtered obs map histograms");
                        todproc.engine().omb.calc_map_hist();

                        // find filtered obs map sources
                        if (todproc.engine().run_source_finder) {
                            SPDLOG_INFO("finding filtered obs map sources");
                            todproc.engine().template find_sources<mapmaking::FilteredObs>(todproc.engine().omb);
                        }

                        // if pointing, fit filtered maps
                        if constexpr (std::is_same_v<todproc_t, TimeOrderedDataProc<Pointing>>) {
                            todproc.engine().fit_maps();
                        }

                        // output filtered maps
                        SPDLOG_INFO("outputting filtered obs files");
                        todproc.engine().template output<mapmaking::FilteredObs>();
                    }
                }

                if (todproc.engine().run_coadd) {
                    // normalize coadded maps
                    SPDLOG_INFO("normalizing coadded maps");
                    todproc.engine().cmb.normalize_maps();

                    // calculate coadded map psds
                    SPDLOG_INFO("calculating coadded map psd");
                    todproc.engine().cmb.calc_map_psd();
                    // calculate coadded map histograms
                    SPDLOG_INFO("calculating coadded map histogram");
                    todproc.engine().cmb.calc_map_hist();

                    // output coadded maps
                    SPDLOG_INFO("outputting raw coadded files");
                    todproc.engine().template output<mapmaking::RawCoadd>();

                    if (todproc.engine().run_map_filter) {
                        SPDLOG_INFO("filtering coadded maps");
                        // filter coadded maps
                        todproc.engine().template run_wiener_filter<mapmaking::FilteredCoadd>(todproc.engine().cmb);

                        // calculate filtered coadded map psds
                        SPDLOG_INFO("calculating filtered coadded map psds");
                        todproc.engine().cmb.calc_map_psd();
                        // calculate filtered coadded map histograms
                        SPDLOG_INFO("calculating filtered coadded map histograms");
                        todproc.engine().cmb.calc_map_hist();

                        if (todproc.engine().run_source_finder) {
                            // find coadded map sources
                            SPDLOG_INFO("finding filtered coadded map sources");
                            todproc.engine().template find_sources<mapmaking::FilteredCoadd>(todproc.engine().cmb);
                        }

                        // output filtered coadded maps
                        SPDLOG_INFO("outputting filtered coadded files");
                        todproc.engine().template output<mapmaking::FilteredCoadd>();
                    }
                }

                SPDLOG_INFO("making index files");
                // make index files for each directory recursively
                todproc.make_index_file(todproc.engine().redu_dir_name);
                SPDLOG_INFO("citlali is done!");
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
    SPDLOG_INFO("rc {}", rc.pformat());
    if (rc.get_node("config_file").size() > 0) {
        tula::logging::scoped_timeit TULA_X{"Citlali Process"};
        return run(rc);
    } else {
        std::cout << "Invalid argument. Type --help for usage.\n";
    }
}
