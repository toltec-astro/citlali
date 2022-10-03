#include "cli/utils.h"
#include "core/kidsdata.h"
#include "sweep/finder.h"
#include "sweep/fitter.h"
#include "timestream/solver.h"
#include "toltec/toltec.h"
#include "utils/config.h"
#include "utils/grppiex.h"
#include "utils/logging.h"
#include <tula/filename.h>

#include <cstdlib>
#include <kids/gitversion.h>

// Some implementation typedefs
using keymap_t = std::unordered_map<std::string, std::string>;
using Config = config::Config;

/// @brief Helper to construct Finder from CLI.
struct finder_options {

    using Finder = kids::SweepKidsFinder;

    constexpr static auto cli = [](auto &rc) {
        using namespace config_utils::clipp_builder;
        // clang-format off
        return "finder configs" % __(
        _(rc, p("finder_threshold"     ), "Detection threshold",
                                           10., doub()),
        _(rc, p("finder_resample"      ), "Frequency grid to use for the resampling,"
                                          " specified in '[start]:[end][:step]'",
                                          ":", str()),
        _(rc, p("output_d21"           ), "Save the computed D21",
                                          "{stem}{suffix}.{ext}", opt_str("dest")),
        _(rc, p("output_processed"     ), "Save the reduced data",
                                          "{stem}{suffix}.{ext}", opt_str("dest"))
        ); // clang-format on
    };
    constexpr static auto finder = [](auto &rc) {
        Finder::Config conf{};
        for (auto &[confkey, rckey] : keymap_t{
                 {"output_processed", "output_processed"},
                 {"output_d21", "output_d21"},
                 {"threshold", "finder_threshold"},
                 {"resample", "finder_resample"},
                 {"fitter_weight_window_type", "fitter_weight_window_type"},
                 {"fitter_weight_window_fwhm", "fitter_weight_window_fwhm"},
                 {"exmode", "grppiex"}}) {
            if (rc.is_set(rckey)) {
                conf.at_or_add(confkey) = rc.at(rckey);
            }
        }
        return Finder{std::move(conf)};
    };
};

/// @brief Helper to construct Fitter from CLI.
struct fitter_options {

    using Fitter = kids::SweepFitter;

    constexpr static auto cli = [](auto &rc) {
        using namespace config_utils::clipp_builder;
        // clang-format off
        return "fitter configs" % __(
        _(rc, p("fitter_weight_window_type"), "Fit with weight window of this"
                                              " type applied",
                                              Fitter::WeightOption::lorentz,
                                              list(Fitter::WeightOption{})),
        _(rc, p("fitter_weight_window_fwhm"), "The FWHM of the weight window in Hz",
                                              1.5e4, doub()),
        _(rc, p("fitter_modelspec"), "The spec of S21 model to use",
                                              Fitter::ModelSpec::gainlintrend,
                                              list(Fitter::ModelSpec{})),
        _(rc, p("output_processed"     ), "Save the reduced data",
                                          "{stem}{suffix}.{ext}", opt_str("dest"))
                                      ); // clang-format on
    };
    constexpr static auto fitter = [](auto &rc) {
        Fitter::Config conf{};
        for (auto &[confkey, rckey] :
             keymap_t{{"output_processed", "output_processed"},
                      {"weight_window_type", "fitter_weight_window_type"},
                      {"weight_window_fwhm", "fitter_weight_window_fwhm"},
                      {"modelspec", "fitter_modelspec"},
                      {"exmode", "grppiex"}}) {
            if (rc.is_set(rckey)) {
                conf.at_or_add(confkey) = rc.at(rckey);
            }
        }
        return Fitter{std::move(conf)};
    };
};

/// @brief Helper to construct Solver from CLI.
struct solver_options {

    using Solver = kids::TimeStreamSolver;

    constexpr static auto cli = [](auto &rc) {
        using namespace config_utils::clipp_builder;
        // clang-format off
        return "solver configs" % __(
        _(rc, p("solver_fitreportdir"), "Look for fitreport file in this directory",
                                         ".", str("dir")),
        _(rc, p("solver_fitreportfile"), "Use this fitreport file",
                                         undef{}, str("file")),
        _(rc, p("solver_extra_output"), "Compute extra output"),
        _(rc, p("solver_chunk_size"), "Solve timestream by chunk",
                                         undef{}, opt_int()),
        _(rc, p("solver_sample_slice"), "Use this range of samples.",
                                         ":", str("slice"))
                                      ); // clang-format on
    };
    constexpr static auto solver = [](auto &rc) {
        Solver::Config conf{};
        for (auto &[confkey, rckey] :
             keymap_t{{"fitreportdir", "solver_fitreportdir"},
                      {"fitreportfile", "solver_fitreportfile"},
                      {"extra_output", "solver_extra_output"},
                      {"exmode", "grppiex"},
		      }) {
            if (rc.is_set(rckey)) {
                conf.at_or_add(confkey) = rc.at(rckey);
            }
        }
        return Solver{std::move(conf)};
    };
};

auto parse_args(int argc, char *argv[]) {
    // disable logger in this function
    auto _0 = logging::scoped_quiet{};
    // runtime config container
    Config rc{};
    // cli config container
    Config cc{};
    // some of the option specs
    auto ver = fmt::format("{} ({})", GIT_VERSION, BUILD_TIMESTAMP);
    using namespace config_utils::clipp_builder;
    // clang-format off
    auto screen = clipp_utils::screen{
    // =======================================================================
                        "kids" ,  "kids_c++", ver,
                                  "Process KIDs I/Q data"                   };
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
    _(rc,             "source"  , "The path or uri of input data", str())
                                                                            ),
              "common options"                                          % __(
    _(rc, p(     "o", "output" ), "Output dest",
                                  "{stem}{suffix}.{ext}", opt_str("dest")),
    _(rc, p(            "plot" ), "Make diagnostic plot"                ),
    _(rc, p(    "plot_backend" ), "Matplotlib backend to use",
                                  "default", str()                      ),
    _(rc, p(     "plot_output" ), "Plot output dest",
                                  "{stem}.png", opt_str("dest")         ),
    _(rc, p(         "grppiex" ), "GRPPI executioon policy",
                                  tula::grppi_utils::modes::default_(),
                                  list(tula::grppi_utils::modes::names())         )
                                                                            ),
              finder_options::cli(rc),
              fitter_options::cli(rc),
              solver_options::cli(rc)                                         )
    );
    // =======================================================================
    // clang-format on

    screen.parse(cli, argc, argv);
    // handle cc
    // SPDLOG_TRACE("cc: {}", cc.pformat());
    if (cc.get_typed<bool>("help")) {
        screen.manpage(cli);
        std::exit(EXIT_SUCCESS);
    } else if (cc.get_typed<bool>("version")) {
        screen.version();
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
    try {
        // read data
        auto [kind, meta] = spec::get_meta<>(rc.get_str("source"));
        KidsData<> kidsdata;
        if (kind & KidsDataKind::TimeStream) {
            using index_t = Eigen::Index;
            // check solver range and pass that to the reader
            auto sample_slice = tula::container_utils::parse_slice<index_t>(
                rc.get_str("solver_sample_slice"));
            auto ntimes = meta.template get_typed<int>("ntimes_all");
            auto sample_range =
                tula::container_utils::to_indices(sample_slice, ntimes);
            SPDLOG_INFO("solve range {} out of {}", sample_range, ntimes);
            using slice_t = DECAY(sample_slice);
            using range_t = DECAY(sample_range);
            auto &[start, stop, step, size] = sample_range;

            // check size, if chunk size is larger than data size, ignore chunk
            bool solve_by_chunk{false};
            if (rc.is_set("solver_chunk_size")) {
                auto chunksize = rc.get_typed<int>("solver_chunk_size");
		    SPDLOG_INFO("solver chunk size: {}", chunksize);
                solve_by_chunk = (chunksize < size);
            }
            SPDLOG_INFO("solve by chunk: {}", solve_by_chunk?"yes":"no");
            // make chunks
            if (solve_by_chunk) {
                if (rc.get_typed<bool>("solver_extra_output")) {
                    SPDLOG_WARN(
                        "solve by chunk cannot produce extra output, ignored");
                }
                if (!rc.is_set("output")) {
                    throw std::runtime_error("solve by chunk requires output");
                }
                std::vector<range_t> chunks{};
                auto chunksize = rc.get_typed<int>("solver_chunk_size");
                if (chunksize > size) {
                    // one chunk
                    chunks.push_back(sample_range);
                } else {
                    // split to multiple chunks
                    index_t c_start = start;
                    index_t c_stop = start;
                    while (true) {
                        // make t_slice the chunk size
                        c_stop = c_start + step * chunksize;
                        if (c_stop > stop) {
                            c_stop = stop;
                        }
                        chunks.emplace_back(container_utils::to_indices(
                            slice_t{c_start, c_stop, step}, ntimes));
                        c_start = c_stop;
                        assert((std::get<3>(chunks.back()) == chunksize) ||
                               (c_stop == stop));
                        if (c_start == stop) {
                            break;
                        }
                    }
                }
                auto nchunks = chunks.size();
                SPDLOG_INFO("solve by chunks size={} nchunks={}", chunksize,
                            nchunks);
                auto ex = tula::grppi_utils::dyn_ex(rc.get_str("grppiex"));
                {
                    logging::scoped_timeit l0("solve by chunk");
                    logging::progressbar pb0(
                        [](const auto &msg) { SPDLOG_INFO("{}", msg); }, 60,
                        "solve by chunk ");
                    using rts_t =
                        kids::KidsData<kids::KidsDataKind::RawTimeStream>;
                    using netCDF::NcFile;
                    auto filepath = filename_utils::parse_pattern(
                        rc.get_str("output"), meta.get_str("source"),
                        fmt::arg("ext", "nc"),
                        fmt::arg("suffix", "_processed"));
                    SPDLOG_INFO("save result to {}", filepath);
                    auto solver = solver_options::solver(rc);
                    kids::TimeStreamSolverResult::NcFileIO io{filepath};
                    std::size_t i = 0;
                    std::mutex io_mutex;
                    grppi::pipeline(
                        ex,
                        [&]() mutable -> std::optional<rts_t> {
                            std::scoped_lock lock(io_mutex);
                            logging::scoped_loglevel<spdlog::level::debug> l0{};
                            if (i >= chunks.size()) {
                                return std::nullopt;
                            }
                            auto [start, stop, step, size] = chunks.at(i);
                            ++i;
                            slice_t slice{start, stop, step};
                            SPDLOG_INFO("read slice {}", slice);
                            return timeit(
                                "read kids data slice",
                                spec::read_data_slice<
                                    kids::KidsDataKind::RawTimeStream>,
                                rc.get_str("source"), slice);
                        },
                        [&](auto kidsdata) {
                            SPDLOG_DEBUG("kidsdata ntimes={}",
                                         kidsdata.meta.template get_typed<int>(
                                             "sample_slice_size"));
                            auto result = timeit(
                                "solve xs", solver, kidsdata,
                                config::Config{{"extra_output", {false}}});
                            {
                                std::scoped_lock lock(io_mutex);
                                result.append_to_nc(io);
                            }
                            pb0.count(nchunks, nchunks / 10);
                            return EXIT_SUCCESS;
                        });
                }
                return EXIT_SUCCESS;
            } else {
                kidsdata =
                    timeit("read kids data slice", spec::read_data_slice<>,
                           rc.get_str("source"), sample_slice);
            }
        } else {
            kidsdata = timeit("read kids data all", spec::read_data<>,
                              rc.get_str("source"));
        }
        SPDLOG_TRACE("kids data: {}", kidsdata);
        // logging::scoped_loglevel<spdlog::level::info> _0{};
        // process result callback
        auto handle_result = [&rc](const auto &result) {
            if (rc.is_set("output")) {
                // logging::scoped_loglevel<spdlog::level::trace> _1{};
                result.save(rc.get_str("output"));
            }
            if (rc.is_set("plot") and rc.get_typed<bool>("plot")) {
                result.plot();
            }
            return EXIT_SUCCESS;
        };
        // dispatch and do process
        auto exitcode = std::visit(
            [&rc, &handle_result](const auto &data) {
                using Data = DECAY(data);
                constexpr auto kind = Data::kind();
                using kids::KidsDataKind;
                if constexpr (kind == KidsDataKind::VnaSweep) {
                    auto finder = finder_options::finder(rc);
                    auto result = timeit("detect kids", finder, data);
                    if (rc.is_set("output_d21")) {
                        result.save_d21(rc.get_str("output_d21"));
                    }
                    if (rc.is_set("output_processed")) {
                        result.save_processed(rc.get_str("output_processed"));
                    }
                    return handle_result(result);
                } else if constexpr (kind == KidsDataKind::TargetSweep) {
                    auto fitter = fitter_options::fitter(rc);
                    auto result = timeit("fit s21 model", fitter, data);
                    if (rc.is_set("output_processed")) {
                        result.save_processed(rc.get_str("output_processed"));
                    }
                    return handle_result(result);
                    // return EXIT_SUCCESS;
                } else if constexpr (kind == KidsDataKind::RawTimeStream) {
                    auto solver = solver_options::solver(rc);
                    auto result = timeit("solve xs", solver, data);
                    return handle_result(result);
                } else {
                    SPDLOG_TRACE(
                        "processing of data kind {} is not yet implemented",
                        kind);
                    return EXIT_FAILURE;
                }
            },
            kidsdata);
        return exitcode;
    } catch (const std::runtime_error &re) {
        SPDLOG_ERROR("{}. abort", re.what());
        return EXIT_FAILURE;
    }
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
    } else if (rc.is_set("source")) {
        // cmd mode
        return run_cmdproc(rc);
    } else {
        std::cout << "Invalid argument. Type --help for usage.\n";
    }
}
