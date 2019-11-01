#include <Eigen/Core>
#include <chrono>
#include <cstdlib>
#include <exception>
// #include <kids/toltec/toltec.h>
#include <thread>
#include <utils/config.h>
#include <utils/container.h>
#include <utils/formatter/matrix.h>
#include <utils/formatter/ptr.h>
#include <utils/grppiex.h>
#include <utils/mpi.h>
#include <yaml-cpp/yaml.h>

/**
 * @brief The BlockAccessor struct
 * This provides methods to access array in block notions
 */
template <typename Array, auto ndims_ = 2> struct BlockAccessor {
    using array_t = Array;
    using index_t = Eigen::Index;
    using bins_t = Eigen::Matrix<index_t, 2, Eigen::Dynamic>;
    constexpr static const index_t ndims = ndims_;
    static_assert(ndims > 0, "NDIMS HAS TO BE POSITIVE");
    static_assert(ndims <= 2, "NDIMS > 2 IS NOT IMPLEMENTED");
    constexpr static const index_t left = 0;
    constexpr static const index_t right = 1;
    BlockAccessor(array_t array_) : array(std::move(array_)) {}
    array_t array;
    std::array<bins_t, ndims> all_bins;
    /**
     * @brief Return the bins for dim.
     */
    template <auto dim> constexpr auto bins() -> decltype(auto) {
        return all_bins[dim];
    }
    /**
     * @brief Return the number of bins for dim.
     */
    template <auto dim> constexpr index_t nbins() const {
        return bins<dim>().cols();
    }
    /**
     * @brief Set bins with bin matrix of shape (2, nbins).
     */
    template <auto dim> void set_bins(bins_t bins_) {
        bins<dim>() = std::move(bins_);
    }
    /**
     * @brief Set bins with vector of pairs of (left, right).
     */
    template <auto dim>
    void set_bins(const std::vector<std::pair<index_t, index_t>> &bins_) {
        set_bins(eigen_utils::tomat(bins_));
    }
    /**
     * @brief Set bins with vector of bin edges of size (nbins + 1).
     */
    template <auto dim> void set_bins(const std::vector<index_t> &bin_edges) {
        static_assert(dim > 0 && dim < ndims, "INVAID DIM");
        auto edges = eigen_utils::asvec(bin_edges);
        auto nbins = edges.size() - 1;
        bins<dim>().resize(2, nbins);
        bins<dim>().row(left) = edges.head(nbins);
        bins<dim>().row(right) = edges.tail(nbins);
    }

    /**
     * @brief Returns number of blocks
     */
    constexpr index_t nblocks() const {
        return std::reduce(all_bins.begin(), all_bins.end(), 1,
                           [](const auto &lhs, const auto &rhs) {
                               return lhs.cols() * rhs.cols();
                           });
    }
    template <REQUIRES_V_(ndims == 1)> auto block(index_t i) {
        constexpr decltype(auto) bins_ = bins<0>();
        assert(i < bins_.cols());
        auto i0 = bins_.coeff(left, i);
        auto ni = bins_.coeff(right, i) - i0;
        return array.segment(i0, ni);
    }
    template <REQUIRES_V_(ndims == 2)> auto block(index_t i, index_t j) {
        constexpr decltype(auto) ibins = bins<0>();
        assert(i < ibins.cols());
        auto i0 = ibins.coeff(left, i);
        auto ni = ibins.coeff(right, i) - i0;

        constexpr decltype(auto) jbins = bins<1>();
        assert(j < jbins.cols());
        auto j0 = jbins.coeff(left, j);
        auto nj = jbins.coeff(right, j) - j0;

        return array.block(i0, j0, ni, nj);
    }
};

/**
 * @brief The YamlConfig struct
 * This is a thin wrapper around YAML::Node.
 */
struct YamlConfig {
    using key_t = std::string;
    using value_t = YAML::Node;
    using storage_t = YAML::Node;
    YamlConfig() = default;
    explicit YamlConfig(storage_t node) : m_node(std::move(node)) {}
    std::string dump_to_str() const { return fmt::format("{}", m_node); }
    static auto load_from_str(std::string s) {
        return YamlConfig(YAML::Load(s));
    }
    friend std::ostream &operator<<(std::ostream &os,
                                    const YamlConfig &config) {
        return os << config.dump_to_str();
    }

    template <typename T> auto get_typed(const key_t &key) const {
        return m_node[key].as<T>(key);
    }
    template <typename T> auto get_typed(const key_t &key, T &&defval) const {
        decltype(auto) node = m_node[key];
        if (node.IsDefined() && !node.IsNull()) {
            return node.as<T>(key);
        }
        return FWD(defval);
    }

    auto get_str(const key_t &key) const { return get_typed<std::string>(key); }
    auto get_str(const key_t &key, const std::string &defval) const {
        return get_typed<std::string>(key, std::string(defval));
    }

    decltype(auto) operator[](const key_t &key) const { return m_node[key]; }

    bool has(const key_t &key) const { return m_node[key].IsDefined(); }
    template <typename T> bool has_typed(const key_t &key) const {
        if (!has(key)) {
            return false;
        }
        try {
            get_typed<T>(key);
        } catch (YAML::BadConversion) {
            return false;
        }
        return true;
    }

private:
    storage_t m_node;
};

/// @brief The mixin class for config handling.
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
    template <typename... Args> ConfigMixin(Args &&... args) {
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

/// @brief Interface base class
template <typename Derived, typename IO_> struct InterfaceBase {
    using IO = IO_;
    IO io;
};

/// @ brief Interface to TolTEC detector network
struct ToltecInterface : InterfaceBase<ToltecInterface, int> {
    using Base = InterfaceBase<ToltecInterface, int>;
};

/// @ brief Interface to HWP
struct HwpInterface : InterfaceBase<HwpInterface, int> {
    using Base = InterfaceBase<HwpInterface, int>;
};

/// @ brief Interface to LMT
struct LmtInterface : InterfaceBase<LmtInterface, int> {
    using Base = InterfaceBase<LmtInterface, int>;
};

/**
 * @brief The InterfaceRegistry struct
 * This is a helper class to access the actual interface implementation.
 */
struct InterfaceRegistry {
    enum class Interface;

    // dispatcher helper
    template <typename Case, typename... Rest> struct dispatcher_impl {
        template <Interface interface>
        using io_t = meta::switch_t<interface, Case, Rest...>;
        using variant_t =
            std::variant<typename Case::type, typename Rest::type...>;
    };
    /////////////////////////////////////////////
    // Register interface classes here
    META_ENUM(Interface, int, Toltec, Hwp, Lmt);
    using dispatcher =
        dispatcher_impl<meta::case_t<Interface::Toltec, ToltecInterface>,
                        meta::case_t<Interface::Hwp, HwpInterface>,
                        meta::case_t<Interface::Lmt, LmtInterface>>;
    /////////////////////////////////////////////
    template <Interface interface> using io_t = dispatcher::io_t<interface>;
    using variant_t = dispatcher::variant_t;
};

/**
 * @brief The Coordinator struct
 * This wraps around the config object and provides
 * high level methods in various ways to setup the MPI runtime
 * with node-local and cross-node environment.
 */
struct Coordinator : ConfigMixin<Coordinator> {
    using config_t = ConfigMixin::config_t;
    using comm_t = mpi_utils::comm;
    using index_t = int;
    constexpr static index_t not_a_worker = -1;

    Coordinator(const comm_t &comm_, const comm_t &comm_local_)
        : ConfigMixin<Coordinator>{}, comm(comm_), comm_local(comm_local_) {}
    const comm_t &comm;       // This is all, including cross-node
    const comm_t &comm_local; // node-local comm

    constexpr auto n_nodes() const { return comm.n_nodes(); }
    constexpr auto node_index() const { return comm.node_index(); }
    const auto &node_name() const { return comm.node_name(); }

    // global mpi context
    constexpr auto rank() const { return comm.rank(); }
    constexpr auto master_rank() const { return 0; }
    constexpr auto is_master() const { return rank() == master_rank(); }

    // local mpi context
    constexpr auto rank_local() const { return comm_local.rank(); }
    constexpr auto master_rank_local() const { return 0; }
    constexpr auto is_master_local() const {
        return rank_local() == master_rank_local();
    }
    constexpr auto worker_index_local() const {
        if (is_master_local()) {
            return not_a_worker;
        }
        return rank_local() - ((rank_local() > master_rank_local()) ? 1 : 0);
    }
    constexpr auto n_workers_local() const { return comm_local.size() - 1; }

    /**
     * Sync the config.
     * This is done to the global comm.
     */
    auto &sync() {

        auto config_str = comm.bcast_str(
            master_rank(), [&]() { return config().dump_to_str(); });

        if (is_master()) {
            SPDLOG_TRACE("config:\n{}", config());
        } else {
            // here we just disable the validation since it is already done
            set_config(config_t::load_from_str(config_str), false);
            SPDLOG_TRACE("synced config from master");
        }
        comm.barrier();
        // collect inputs
        const auto &inputs = collect_inputs();
        SPDLOG_TRACE("inputs: {}", inputs);
        return *this;
    }

    /**
     * @brief The Observation struct
     * This represents a single observation that contains a set of data items
     */
    struct Observation : ConfigMixin<Observation> {
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
            friend std::ostream &operator<<(std::ostream &os,
                                            const DataItem &d) {
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
        };
        Observation(config_t config_)
            : ConfigMixin<Observation>{config_}, name{config().get_str(
                                                     "name", "unnamed")} {
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
        friend std::ostream &operator<<(std::ostream &os,
                                        const Observation &obs) {
            return os << fmt::format("Observation(name={})", obs.name);
        }
    };

    // io_buffer
    using io_buffer_data_t = double;
    std::size_t io_buffer_size() { return 4000 * 4880 * 10; }

    constexpr auto n_inputs() const { return m_inputs.size(); }

private:
    using input_t = Observation;
    using payload_t = Observation::DataItem;
    // this is all the inputs
    std::vector<input_t> m_inputs;
    // this is the current payload
    std::vector<payload_t> m_payloads;

    // collect input data
    const std::vector<input_t> &collect_inputs() {
        m_inputs.clear();
        // this is run for all processes so that everyone has the necessary info
        // to collect payloads
        auto index = node_index();
        SPDLOG_TRACE("get input data for node={} name={}", index, node_name());
        std::vector<input_t> results;
        auto c = config()["io"]["inputs"];
        assert(c.IsSequence());
        for (std::size_t i = 0; i < c.size(); i++) {
            if (index == (meta::size_cast<int>(i) % n_nodes())) {
                results.emplace_back(config_t{c[i]});
            }
        }
        m_inputs = std::move(results);

        if (is_master_local()) {
            // report number of obs to process
            SPDLOG_DEBUG("collected n_inputs={} on node={} name={}",
                         m_inputs.size(), index, node_name());
        }
        return m_inputs;
    }

public:
    // collect payload
    const std::vector<payload_t> &collect_payloads(std::size_t input_index) {
        m_payloads.clear();
        if (is_master_local()) {
            // master local does not get payload
        } else {
            // construct payload
            // distribute inputs through out worker ranks
            auto index = worker_index_local();
            SPDLOG_TRACE("get payload for worker={} local_rank={}", index,
                         rank_local());
            std::vector<payload_t> results;
            auto c = m_inputs[input_index].config()["data_items"];
            assert(c.IsSequence());
            for (std::size_t i = 0; i < c.size(); i++) {
                if (index == (meta::size_cast<int>(i) % n_workers_local())) {
                    results.emplace_back(config_t{c[i]});
                }
            }
            m_payloads = std::move(results);
            SPDLOG_DEBUG("collected n_payloads={} on worker={} local_rank{} "
                         "node={} name={}",
                         m_payloads.size(), index, rank_local(), node_index(),
                         node_name());
        }
        return m_payloads;
    }
};

int main(int argc, char **argv) {

    mpi_utils::env mpi_env(argc, argv); // RAII MPI env
    mpi_env.set_exception_on_error();
    mpi_utils::comm comm; // MPI_COMM_WORLD
    MPI_Win win;    // window object
    mpi_utils::logging_init(comm);
    mpi_utils::pprint_node_ranks(0, comm, [&](const auto &node_ranks) {
        SPDLOG_INFO("{}", mpi_env);
        SPDLOG_INFO("{}", node_ranks);
    });

    mpi_utils::comm comm_shm{comm.split_shared()};
    {
        // check if shm comm is valid.
        const auto master_rank = 0;
        const auto is_master = comm.rank() == master_rank;
        // check env
        // we need to the io procs and the master on the same node
        if (is_master) {
            if (mpi_env.memory_model !=
                mpi_utils::env::WinMemoryModel::Unified) {
                SPDLOG_ERROR("unified memory model is required, found {}",
                             mpi_env.memory_model);
                MPI_Abort(comm, EXIT_FAILURE);
            }
            if (comm_shm.size() != comm.size()) {
                SPDLOG_ERROR("shared memory access across all processes is "
                             "required, found {}/{}",
                             comm_shm.size(), comm.size());
                MPI_Abort(comm, EXIT_FAILURE);
            }
        }
    }

    // io config
    using coordinator_t = Coordinator;
    coordinator_t co{comm, comm_shm};
    using config_t = coordinator_t::config_t;
    if (co.is_master()) {
        const char *test_config{
            R"(---
description: |-
    This file is provided as an example of the citlali config file.
io:
    inputs:
      - name: obs1
        data_items:
          - interface: toltec0
            filepath: "toltec0.nc"
          - interface: toltec1
            filepath: "toltec1.nc"
          - interface: toltec2
            filepath: "toltec2.nc"
          - interface: lmt 
            filepath: "lmt.nc"
          - interface: hwp 
            filepath: "hwp.nc"
      - name: obs2
        data_items:
          - interface: toltec0
            filepath: "toltec0.nc"
          - interface: toltec1
            filepath: "toltec1.nc"
          - interface: toltec2
            filepath: "toltec2.nc"
          - interface: lmt 
            filepath: "lmt.nc"
          - interface: hwp 
            filepath: "hwp.nc"
    time_chunking:
        enabled: yes
        method:
            value: "fixed"
            choises: ["hold_signal", "fixed_length"]
            description: |-
                hold_signal: Define chunks using the telescope hold signal.
                fixed_length: Each chunk has specified length.
        parameters:
           hold_signal:
                value: 1
                description: |-
                    value: Continuous samples with the hold signal of this value will be recongnized as one chunk.
           fixed_length:
                value: 10
                unit: second
                description: |-
                    value: The length of the time chunk.
                    unit: The unit of said value, e.g., sample, second, etc.
)"};
        SPDLOG_TRACE("yaml config in:\n{}", test_config);
        co.set_config(config_t::load_from_str(test_config));
        /// VALIDATE_CONFIG(config, )
        /// for all types that consumes config.
    }

    // make sure all ranks have access to the config.
    co.sync();
    auto n_obs = co.n_inputs();
    for (std::size_t i = 0; i < n_obs; ++i) {
        const auto &payloads = co.collect_payloads(i);
        SPDLOG_TRACE("payloads: {}", payloads);
    }

    MPI_Abort(comm, 0);
    // coordinator settings

    int arg{0};

    if (co.is_master()) {
        arg = 5;
        SPDLOG_TRACE("load config wait={}", arg);
        std::this_thread::sleep_for(std::chrono::seconds(arg));
        MPI_Bcast(&arg, 1, MPI_INT, 0, comm);
    } else {
        MPI_Bcast(&arg, 1, MPI_INT, 0, comm);
        SPDLOG_TRACE("got config wait={}", arg);
    }

    // intialize the shared memory buffer for io
    auto io_buffer =
        mpi_utils::Span<coordinator_t::io_buffer_data_t>::allocate_shared(
            co.master_rank(), co.io_buffer_size(), comm_shm, &win);
    //     auto coor =
    //         mpi_utils::Span<coordinator_t::io_buffer_data_t>::allocate_shared(
    //             co.master_rank(), co.io_buffer_size(), comm_shm, &win);

    if (co.is_master()) {
        auto ex = grppiex::Mode::omp;
        SPDLOG_TRACE("use grppi ex {}", ex);
        grppi::pipeline(
            grppiex::dyn_ex(ex),
            [&]() -> std::optional<int> {
                static int rtc = 0;
                // returns upon each recieved rtc
                std::this_thread::sleep_for(std::chrono::seconds(arg));
                MPI_Win_sync(win);
                SPDLOG_TRACE("io_buffer{}",
                             io_buffer.asvec().head(comm.size()));
                return ++rtc;
            },
            [&](int rtc) {
                SPDLOG_TRACE("processing rtc={}", rtc);
                std::this_thread::sleep_for(std::chrono::seconds(arg));
            });
    } else {
        SPDLOG_TRACE("reading rtc={}", 0);
        std::this_thread::sleep_for(std::chrono::seconds(arg));
        SPDLOG_TRACE("reading rtc={} done", 0);
        auto rank = comm_shm.rank();
        io_buffer.data[rank] = rank;
        MPI_Win_sync(win);
    }
    MPI_Win_free(&win);
    return EXIT_SUCCESS;
}