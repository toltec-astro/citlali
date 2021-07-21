// #include "citlali/core/lali.h"
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
#include <omp.h>

constexpr auto pi = static_cast<double>(EIGEN_PI);

// Arcseconds in 360 degrees
#define ASEC_CIRC 1296000.0
// rad per arcsecond
#define RAD_ASEC (2.0*pi / ASEC_CIRC)
// Degrees to radians
#define DEG_TO_RAD 3600.*RAD_ASEC

#include "citlali/core/read.h"
#include "citlali/core/observation.h"
#include "citlali/core/TCData.h"
#include "citlali/core/ecsv_reader.h"
#include "citlali/core/lali.h"
#include "citlali/core/beammap.h"
#include "citlali/core/source.h"

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
    _(rc,        "config_file"  , "The path of input config file", opt_str() ),
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

// TcData is the data structure of which RTCData and PTCData are a part
using timestream::TCData;

// Selects the type of TCData
using timestream::LaliDataKind;

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
              m_filepath(this->config().get_filepath("filepath")) {}

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
    const auto n_data_items() const { return m_data_items.size(); }
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
            data_items.emplace_back(
                config_t{node_data_items[i], this->config().filepath()});
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

    const auto n_inputs() const { return m_inputs.size(); }

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
            inputs.emplace_back(
                config_t{node_inputs[i], this->config().filepath()});
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

bool extra_output = 0;
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
          m_solver{Solver::Config{
                   {"fitreportdir", "/"},
                   {"exmode", "omp"},
                   {"extra_output", extra_output},}} {}

    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        std::vector<std::string> missing_keys;
        SPDLOG_TRACE("check kids data proc config\n{}", config);
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

    auto reduce_data_item(const RawObs::DataItem &data_item,
                          const container_utils::Slice<int> &slice) {
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
            source, slice);
        auto result = this->solver()(rts, Solver::Config{});
        return result;
    }

    auto reduce_rawobs(const RawObs &rawobs,
                       const container_utils::Slice<int> &slice) {
        SPDLOG_TRACE("kids reduce rawobs {}", rawobs);
        std::vector<kids::TimeStreamSolverResult> result;
        for (const auto &data_item : rawobs.kidsdata()) {
            result.push_back(reduce_data_item(data_item, slice));
        }
        return result;
    }

    template <typename scanindices_t>
    auto populate_rtc(const RawObs &rawobs, scanindices_t &scanindex, const int scanlength, const int n_detectors) {
        // call reduce rawobs, get the data into rtc

        auto slice = container_utils::Slice<int>{scanindex(2), scanindex(3)+1, std::nullopt};
        auto reduced = reduce_rawobs(rawobs, slice);

        Eigen::MatrixXd xs(scanlength, n_detectors);

        Eigen::Index i = 0;
        for(std::vector<kids::TimeStreamSolverResult>::iterator it = reduced.begin(); it != reduced.end(); ++it) {
            auto nrows = it->data_out.xs.data.rows();
            auto ncols = it->data_out.xs.data.cols();
            xs.block(0, i, nrows, ncols) = it->data_out.xs.data;
            i += ncols;
         }

        return std::move(xs);

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

struct DummyEngine {
    template <typename OStream>
    friend OStream &operator<<(OStream &os, const DummyEngine &e) {
        return os << fmt::format("DummyEngine()");
    }
};

/**
 * @brief The time ordered data processing struct
 * This wraps around the lali config
 */
struct TimeOrderedDataProc : ConfigMapper<TimeOrderedDataProc> {
    using Base = ConfigMapper<TimeOrderedDataProc>;
    using Engine = lali::Lali;
    // using Engine = beammap::Beammap;
    // using Engine = DummyEngine;
    using map_extent_t = std::vector<double>;
    using map_coord_t = std::vector<Eigen::VectorXd>;
    using map_count_t = std::size_t;
    using array_indices_t = std::vector<std::tuple<int,int>>;
    using scanindicies_t = Eigen::MatrixXI;

    TimeOrderedDataProc(config_t config) : Base{std::move(config)} {}

    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        std::vector<std::string> missing_keys;
        SPDLOG_TRACE("check TOD proc config\n{}", config);
        if (!config.has("tod")) {
            missing_keys.push_back("tod");
        }
        if (!config.has("map")) {
            missing_keys.push_back("map");
        }
        if (missing_keys.empty()) {
            return std::nullopt;
        }
        return fmt::format("invalid or missing keys={}", missing_keys);
    }

    auto get_map_extent(const RawObs &rawobs) {
        // implement this to return the map size
        std::vector<double> map_extent;
        std::vector<Eigen::VectorXd> map_coord;

        mapmaking::MapUtils mu;
        auto [nr, nc, rcp, ccp] = mu.getRowsCols<mu.Individual>(engine().telMD.telMetaData,
                                                                engine().offsets,
                                                                engine().config);

        map_extent.push_back(nr);
        map_extent.push_back(nc);

        map_coord.push_back(rcp);
        map_coord.push_back(ccp);

        return std::tuple{map_extent, map_coord};
    }

    auto get_map_count(const RawObs &rawobs) {
        // implement the logic to look into apt.ecsv and the groupping config
        // to figure out the number of maps requested for this rawobs
        auto grouping = engine().config.get_str(std::tuple{"map","grouping"});
        SPDLOG_INFO("grouping {}", grouping);

        Eigen::Index ai = 0;
        std::vector<std::tuple<int,int>> array_index;
        std::vector<std::tuple<int,int>> det_index;

        //if (std::strcmp("array_name", grouping.c_str()) == 0) {
        array_index.push_back(std::tuple{0,0});
        for(Eigen::Index i = 0; i < engine().array_name.size(); i++) {
            if (engine().array_name(i) == ai){
                std::get<1>(array_index.at(ai)) = i;
            }
            else {
                ai += 1;
                array_index.push_back(std::tuple{i,0});
            }
        }

        if (std::strcmp("array_name", grouping.c_str()) == 0) {
            det_index = array_index;
        }

        else if (std::strcmp("beammap", grouping.c_str()) == 0) {
            for(Eigen::Index i = 0; i < engine().array_name.size(); i++) {
                det_index.push_back(std::tuple{i,i+1});
            }
        }

        auto map_count = det_index.size();

        return std::tuple{map_count, array_index, det_index};
    }

    auto get_scanindicies(const RawObs &rawobs) {
        // implement the logic to setup map buffer for the engine
        scanindicies_t scanindices;
        auto offset = engine().config.get_typed<double>(std::tuple{"map","offset"});
        auto timechunk = engine().config.get_typed<double>(std::tuple{"map","timechunk"});

        lali::obs(scanindices, engine(), timechunk, offset);

        return scanindices;
    }

    void setup_coadd_map_buffer(const map_extent_t &map_extent,
                                const map_count_t &map_count) {
        // implement the logic to setup coadd map buffer for the engine
    }

    auto setup_map_buffer(const map_extent_t &map_extent,
                          const map_coord_t &map_coord,
                          const map_count_t &map_count) {
        // implement the logic to setup map buffer for the engine
        engine().Maps.nrows = map_extent.at(0);
        engine().Maps.ncols = map_extent.at(1);
        engine().Maps.rcphys = map_coord.at(0);
        engine().Maps.ccphys = map_coord.at(1);
        engine().Maps.map_count = map_count;

        engine().Maps.allocateMaps(engine().telMD, engine().offsets, engine().config);
    }

    // TODO fix the const correctness
    Engine &engine() { return m_engine; }

    const Engine &engine() const { return m_engine; }

    template <typename OStream>
    friend OStream &operator<<(OStream &os,
                               const TimeOrderedDataProc &todproc) {
        return os << fmt::format("TimeOrderedDataProc(engine={})",
                                 todproc.engine());
    }

private:
    Engine m_engine;
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
    //SPDLOG_INFO("citlali config:\n{}", citlali_config);

    // set up the IO coorindator
    auto co = SeqIOCoordinator::from_config(citlali_config);
    SPDLOG_TRACE("pipeline coordinator: {}", co);

    // set up KIDs data proc
    auto kidsproc =
        KidsDataProc::from_config(citlali_config.get_config("kids"));
    SPDLOG_INFO("kids proc: {}", kidsproc);

    // set up TOD proc
    auto todproc = TimeOrderedDataProc::from_config(citlali_config);
    //SPDLOG_INFO("tod proc: {}", todproc);

    // Set todproc config
    todproc.engine().config = citlali_config;

    // Path to apt table from config file
    auto cal_path = citlali_config.get_filepath(std::tuple{"inputs",0,"cal_items",0,"filepath"});
    SPDLOG_INFO("cal_path {}", cal_path);

    // Get apt table (error with ecsv, so using ascii for now)
    auto apt_table = get_aptable_from_ecsv(cal_path, citlali_config);
    SPDLOG_INFO("apt_table {}", apt_table);

    // Put apt table into lali engine (temporary maybe)
    todproc.engine().nw = apt_table.col(0);
    todproc.engine().array_name = apt_table.col(1);
    todproc.engine().fluxscale = apt_table.col(2);
    todproc.engine().offsets["azOffset"] = apt_table.col(3)*3600.;
    todproc.engine().offsets["elOffset"] = apt_table.col(4)*3600.;
    todproc.engine().fwhms["a_fwhm"] = apt_table.col(5)*3600.;
    todproc.engine().fwhms["b_fwhm"] = apt_table.col(6)*3600.;

    // containers to store some pre-computed info for all inputs
    using map_extent_t = TimeOrderedDataProc::map_extent_t;
    using map_coord_t = TimeOrderedDataProc::map_coord_t;
    using map_count_t = TimeOrderedDataProc::map_count_t;
    using array_indices_t = TimeOrderedDataProc::array_indices_t;

    std::vector<map_extent_t> map_extents{};
    std::vector<map_coord_t> map_coords{};
    std::vector<map_count_t> map_counts{};
    std::vector<array_indices_t> array_indices{};
    std::vector<array_indices_t> det_indices{};

    // 1. coadd map buffer
    //{
        // this block of code is to get the relavant info from all the inputs
        // and initialize the coadding buffer

        // populate the inputs info
        for (const auto &rawobs : co.inputs()) {
            todproc.engine().telMD = lali::TelData::fromNcFile(rawobs.teldata().filepath());
            auto maptype = todproc.engine().config.get_str(std::tuple{"map","type"});

            if (std::strcmp("RaDec", maptype.c_str()) == 0) {
                lali::internal::absToPhysEqPointing(todproc.engine().telMD.telMetaData,todproc.engine().telMD.srcCenter);
            }

            else if (std::strcmp("AzEl", maptype.c_str()) == 0) {
                lali::internal::absToPhysHorPointing<lali::pointing>(todproc.engine().telMD.telMetaData);
            }

            auto [me, mco] = todproc.get_map_extent(rawobs);
            map_extents.push_back(std::move(me));
            map_coords.push_back(std::move(mco));

            // map_extents.push_back(todproc.get_map_extent(rawobs));
            auto [mc, ai, di] = todproc.get_map_count(rawobs);
            map_counts.push_back(std::move(mc));
            array_indices.push_back(std::move(ai));
            det_indices.push_back(std::move(di));
        }

        // combine the map extents to the coadd map extent
        // TODO implement this
        auto coadd_map_extent = map_extents.front();
        // combine the map counts to the coadd map counts
        auto coadd_map_count =
            *std::max_element(map_counts.begin(), map_counts.end());

        todproc.setup_coadd_map_buffer(coadd_map_extent, coadd_map_count);
    //}


    // 2. loop over all the inputs to do the reduction
    //{
        for (std::size_t i = 0; i < co.n_inputs(); ++i) {
            const auto &rawobs = co.inputs()[i];
            // map buffer
            // just use the stored values here avoid repeated calculation
            todproc.setup_map_buffer(map_extents[i], map_coords[i], map_counts[i]);

            // this is needed to figure out the data sample rate
            // and number of detectors for creating the scanindices and rtc
            // buffers
            auto rawobs_kids_meta = kidsproc.get_rawobs_meta(rawobs);

            // Get obsid for output filename
            todproc.engine().obsid = rawobs_kids_meta.back().get_typed<int>("obsid");

            // Put it in the correct format with leading zeros (move)
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(6) << todproc.engine().obsid;
            std::string s = ss.str();

            // Get telescope file pointing and time vectors
            todproc.engine().telMD = lali::TelData::fromNcFile(rawobs.teldata().filepath());
            SPDLOG_INFO("got telescope file");

            // TODO implement this to be the actual time chunk size
            todproc.engine().samplerate = rawobs_kids_meta.back().get_typed<double>("fsmp");
            SPDLOG_INFO("tod_sample_rate {}", todproc.engine().samplerate);

            auto scanindicies = todproc.get_scanindicies(rawobs);
            SPDLOG_INFO("scanindicies {}", scanindicies);

            // Copy array and detector indices into engine
            todproc.engine().array_index = array_indices.at(i);
            todproc.engine().det_index = det_indices.at(i);

            // TODO implement to get the number of detectors to create rtc buffer
            todproc.engine().n_detectors = apt_table.rows();
            SPDLOG_INFO("n_detectors {}", todproc.engine().n_detectors);

            // Do general setup that is only run once per rawobs before grppi pipeline
            {
                logging::scoped_timeit timer("engine setup()");
                todproc.engine().setup();
            }

            // Run the actual pipeline
            {
                logging::scoped_timeit timer("engine pipeline()");
                todproc.engine().pipeline(scanindicies, kidsproc, rawobs);
            }

            // Generate output files
            {
                logging::scoped_timeit timer("engine output()");
                todproc.engine().output();
            }
        }
    //}

    /*
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

    {
    logging::scoped_timeit timer("Citlali Process");

        auto rc = parse_args(argc, argv);
        SPDLOG_TRACE("rc {}", rc.pformat());
        if (rc.is_set("config_file")) {
            return run(rc);
        } else {
            std::cout << "Invalid argument. Type --help for usage.\n";
        }
    }
}
