#include "citlali/core/lali.h"
#include "kids/cli/utils.h"
#include "kids/core/kidsdata.h"
#include "kids/sweep/fitter.h"
#include "kids/timestream/solver.h"
#include "kids/toltec/toltec.h"
#include "utils/config.h"
#include "utils/grppiex.h"
#include "utils/logging.h"
#include "utils/yamlconfig.h"
#include <citlali/gitversion.h>
#include <cstdlib>
#include <kids/gitversion.h>
#include <regex>
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

// IO spec
namespace kidsdata = kids::toltec;

} // namespace predefs

template <typename Derived>
using ConfigMapper = config::ConfigValidatorMixin<Derived, config::YamlConfig>;

/**
 * @brief The raw obs struct
 * This represents a single observation that contains a set of data items.
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
              m_filepath(this->config().get_str("filepath")) {}

        static auto check_config(config_t &config)
            -> std::optional<std::string> {
            std::vector<std::string> missing_keys;
            SPDLOG_TRACE("check data items config\n{}", config);
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

        template <typename OStream>
        friend auto operator<<(OStream &os, const DataItem &d)
            -> decltype(auto) {
            return os << fmt::format("DataItem(interface={} filepath={})",
                                     d.interface(), d.filepath());
        }

    private:
        std::string m_interface{};
        std::string m_filepath{};
    };
    RawObs(config_t config)
        : Base{std::move(config)}, m_name{this->config().get_str(
                                       std::tuple{"meta", "name"})} {
        collect_data_items();
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
        if (missing_keys.empty()) {
            return std::nullopt;
        }
        return fmt::format("invalid or missing keys={}", missing_keys);
    }
    const std::string &name() const { return m_name; }
    constexpr auto n_data_items() const { return m_data_items.size(); }
    const std::vector<DataItem> &data_items() const { return m_data_items; }
    const DataItem &teldata() const {
        return m_data_items[m_teldata_index.value()];
    }
    auto kidsdata() const -> decltype(auto) {
        std::vector<std::reference_wrapper<const DataItem>> result{};
        for (auto i : m_kidsdata_indices) {
            result.push_back(std::cref(m_data_items[i]));
        }
        return result;
    }

    template <typename OStream>
    friend auto operator<<(OStream &os, const RawObs &obs) -> decltype(auto) {
        return os << fmt::format("RawObs(name={}, n_data_items={})", obs.name(),
                                 obs.n_data_items());
    }

private:
    inline const static std::regex re_interface_kidsdata{"toltec\\d{1,2}"};
    inline const static std::regex re_interface_teldata{"lmt"};

    std::string m_name;
    std::vector<DataItem> m_data_items{};
    std::vector<std::size_t> m_kidsdata_indices{};
    std::optional<std::size_t> m_teldata_index{std::nullopt};
    // collect data items
    void collect_data_items() {
        m_data_items.clear();
        std::vector<DataItem> data_items{};
        auto node_data_items = this->config().get_node("data_items");
        auto n_data_items = node_data_items.size();
        for (std::size_t i = 0; i < n_data_items; ++i) {
            SPDLOG_TRACE("add data item {} of {}", i, n_data_items);
            data_items.emplace_back(config_t{node_data_items[i]});
        }
        m_data_items = std::move(data_items);
        SPDLOG_DEBUG("collected n_data_items={}\n{}", this->n_data_items(),
                     this->data_items());
        // update the data indices
        m_kidsdata_indices.clear();
        m_teldata_index.reset();
        std::smatch m;
        for (std::size_t i = 0; i < m_data_items.size(); ++i) {
            if (std::regex_match(m_data_items[i].interface(), m,
                                 re_interface_kidsdata)) {
                m_kidsdata_indices.push_back(i);
            }
            if (std::regex_match(m_data_items[i].interface(), m,
                                 re_interface_teldata)) {
                if (m_teldata_index.has_value()) {
                    throw std::runtime_error(
                        "found two many telescope data items");
                }
                m_teldata_index = i;
            }
        }
        if (!m_teldata_index) {
            throw std::runtime_error("no telescope data item found");
        }
        SPDLOG_TRACE("kidsdata_indices={} teldata_index={}", m_kidsdata_indices,
                     m_teldata_index);
        SPDLOG_TRACE("kidsdata={} teldata={}", kidsdata(), teldata());
    }
};

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

    constexpr auto n_inputs() const { return m_inputs.size(); }

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
            inputs.emplace_back(config_t{node_inputs[i]});
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
struct KidsDataProc : ConfigMapper<KidsDataProc> {
    using Base = ConfigMapper<KidsDataProc>;
    using Fitter = kids::SweepFitter;
    using Solver = kids::TimeStreamSolver;
    KidsDataProc(config_t config)
        : Base{std::move(config)},
          m_fitter{Fitter::Config{
              {"weight_window_type",
               this->config().get_str(
                   std::tuple{"fitter", "weight_window", "type"})},
              {"weight_window_fwhm",
               this->config().get_typed<double>(
                   std::tuple{"fitter", "weight_window", "fwhm_Hz"})},
              {"modelspec",
               config.get_str(std::tuple{"fitter", "modelspec"})}}},
          m_solver{Solver::Config{}} {}

    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        std::vector<std::string> missing_keys;
        SPDLOG_TRACE("check kids data solver config\n{}", config);
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

    auto reduce_data_item(const RawObs::DataItem &data_item, slice) {
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
            source, container_utils::Slice<int>{});
        auto result = this->solver()(rts, Solver::Config{});
        return result;
    }

    auto reduce_rawobs(const RawObs &rawobs, slice) {
        SPDLOG_TRACE("kids reduce rawobs {}", rawobs);
        std::vector<kids::TimeStreamSolverResult> result;
        for (const auto &data_item : rawobs.kidsdata()) {
            result.push_back(reduce_data_item(data_item));
        }
    }

    template <typename RTC_t>
    auto populate_rtc(const RawObs &rawobs, RTC_t &rtc) {
        // call reduce rawobs, get the data into rtc
        auto [start, stop] = rtc.scanindices;
        slice = {start, stop} + ikids_data_start_index
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

/// @brief Run citlali reduction.
/// @param rc The runtime config.
int run(const config::Config &rc) {
    using kids::KidsData;
    using kids::KidsDataKind;
    using logging::timeit;
    SPDLOG_INFO("use KIDs data spec: {}", predefs::kidsdata::name);

    // load the yaml citlali config
    auto citlali_config =
        config::YamlConfig::from_filepath(rc.get_str("config_file"));
    SPDLOG_TRACE("citlali config:\n{}", citlali_config.pformat());

    // set up the IO coorindator
    auto co = SeqIOCoordinator::from_config(citlali_config);
    SPDLOG_TRACE("pipeline coordinator: {}", co);

    // set up KIDs data proc
    auto kidsproc =
        KidsDataProc::from_config(citlali_config.get_config("kids"));

    SPDLOG_TRACE("kids proc: {}", kidsproc);

    SPDLOG_INFO("Making Lali Class");
    lali::Lali LC;

    LC.setup_coadd_map_buffer();
    for (const auto &rawobs : co.inputs()) {
        // extract the map size
        // extract the map group number
        auto mapsisze = [ 100, 100 ]
    }
    /*
     *
    1. setup lali co-add map buffer
        std::vector<rawobs size>
        vector<n_map_gourps>
    [done]  for loop of co.inputs().teldata().filepath()
               get the per rawobs size
                {
              [zma]  * check map groupping with apt.ecsv fitgure out number of
    layers in map buffer n_map_groups = 3;
                }

    2. for loop of co.inputs() for each raw obs,

        auto rawobs = co.inputs()[i]

        2.1 map buffer
        * allocate map buffer (one layer for each gorup)


        auto rawobs_kids_meta = kidsproc.get_rawobs_meta(rawobs)

        2.2 calc scanindices
        //// tel pps:   [000011111122...........99100]
        ///  time range      [0s                  98s]
        /// use kids sample rate generate
        /// scanindicies [(0, 4880, 32, 4880-32), (4880-32, 9760 + 32, 4880,
    9760)]
        ///

        2.3 get rtc buffer info
            get number of all detectors from from meta["ntones"]

        2.3 do reduction
            grppi {
                scanindices item [start, stop,]
                rtc [n_samples_per_chunk x ndetector_total]
                kidsproc.populate_rtc(rawobs, rtc, scanindices item)
                        {
                        /// kidsdata 0 ts [0011111122...........99100]
                        /// kidsdata 1 ts[00011111122...........99100]
                        /// kidsdata 2 ts  [011111122...........99100]
                        /// kidsdata_start_vector<13>{2, 3, 1, ...}

                        kids_indics [scanindices + kidsdat_sart_vector]
                        for dataitem in rawobs.data_items()
                            read_data_slice(filepath, slice)
                            reduce kids
                            rtc.block[detector i] = tsresult.data_out.xs()
                        }
                return rtc
            2.5 add map buffer content to coadd map buffer
    */
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
