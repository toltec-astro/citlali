#include <Eigen/Core>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <kids/toltec/toltec.h>
#include <thread>
#include <utils/config.h>
#include <utils/container.h>
#include <utils/formatter/matrix.h>
#include <utils/formatter/ptr.h>
#include <utils/grppiex.h>
#include <utils/mpi_utils.h>
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
    std::string to_str() const { return fmt::format("{}", m_node); }
    static auto from_str(std::string s) { return YamlConfig(YAML::Load(s)); }
    friend std::ostream &operator<<(std::ostream &os,
                                    const YamlConfig &config) {
        return os << config.to_str();
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
    constexpr const auto &config() const { return m_config; }
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

/**
 * @brief The Coordinator struct
 * This wraps around the config object and provides
 * high level methods in various ways to setup the runtime.
 */
struct Coordinator : ConfigMixin<Coordinator> {
    using config_t = ConfigMixin::config_t;
    Coordinator(const mxx::comm &comm_)
        : ConfigMixin<Coordinator>{}, comm(comm_) {}
    const mxx::comm &comm;
    constexpr auto master_rank() { return 0; }
    constexpr auto is_master() { return comm.rank() == master_rank(); }
    constexpr auto worker_index() {
        if (is_master()) {
            return -1;
        }
        return comm.rank() - ((comm.rank() > master_rank()) ? 1 : 0);
    }
    constexpr auto n_workers() { return comm.size() - 1; }
    /**
     * Sync the config.
     */
    auto &sync() {
        {
            // config
            std::string c{};
            int s{0};
            if (is_master()) {
                c = config().to_str();
                s = meta::size_cast<int>(c.size());
            }
            MPI_Bcast(&s, 1, MPI_INT, master_rank(), comm);
            if (!is_master()) {
                c.resize(SIZET(s));
            }
            MPI_Bcast(const_cast<char *>(c.data()), s, MPI_CHAR, master_rank(),
                      comm);
            if (is_master()) {
                SPDLOG_TRACE("config:\n{}", config());
            } else {
                // here we just disable the validation since it is already done
                set_config(config_t::from_str(c), false);
                SPDLOG_TRACE("synced config from master");
            }
        }
        comm.barrier();
        return *this;
    }
    /**
     * @brief Initialize the coordinator
     */
    auto &initialize() {
        const auto &inputs = collect_inputs();
        SPDLOG_TRACE("inputs: {}", inputs);
        return *this;
    }

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

        // dispatcher
        template <typename Case, typename... Rest> struct dispatcher_impl {
            template <Interface interface>
            using io_t = meta::switch_t<interface, Case, Rest...>;
            using variant_t =
                std::variant<typename Case::type, typename Rest::type...>;
        };
        // Register interface classes here
        META_ENUM(Interface, int, Toltec, Hwp, Lmt);
        using dispatcher =
            dispatcher_impl<meta::case_t<Interface::Toltec, ToltecInterface>,
                            meta::case_t<Interface::Hwp, HwpInterface>,
                            meta::case_t<Interface::Lmt, LmtInterface>>;
        template <Interface interface> using io_t = dispatcher::io_t<interface>;
        using variant_t = dispatcher::variant_t;
    };

    struct RawData : ConfigMixin<RawData> {
        RawData(const config_t &config_)
            : ConfigMixin<RawData>{config_},
              interface(config()["interface"].template as<std::string>()),
              filepath(config()["filepath"].template as<std::string>()) {
            // initalize io
            meta::switch_invoke<InterfaceRegistry::InterfaceKind>(
                [&](auto interfacekind_) {
                    constexpr auto interfacekind = DECAY(interfacekind_)::value;
                    io = InterfaceRegistry::template io_t<interfacekind>();
                },
                interface_kind());
            std::string interface{};
            std::string filepath{};
            friend std::ostream &operator<<(std::ostream &os,
                                            const InputData &d) {
                return os << fmt::format("InputData(interface={} filepath={})",
                                         d.interface, d.filepath);
            }
        };

        /**
         * @brief The InputData struct
         * This holds informatation for an input data object
         */
        struct Input {
            using InputData(const config_t &config_)
                : m_config(config_),
                  interface(config()["interface"].template as<std::string>()),
                  filepath(config()["filepath"].template as<std::string>()) {
                // initalize io
                meta::switch_invoke<InterfaceRegistry::InterfaceKind>(
                    [&](auto interfacekind_) {
                        constexpr auto interfacekind =
                            DECAY(interfacekind_)::value;
                        io = InterfaceRegistry::template io_t<interfacekind>();
                    },
                    interface_kind());
            }
            std::string interface{};
            std::string filepath{};
            friend std::ostream &operator<<(std::ostream &os,
                                            const InputData &d) {
                return os << fmt::format("InputData(interface={} filepath={})",
                                         d.interface, d.filepath);
            }
            using io_t = typename InterfaceRegistry::IO_variant;
            io_t io;
            static bool validate_config(const config_t &config) {
                std::vector<std::string> required_keys{"interface", "filepath"};
                std::vector<std::string> optioanl_keys{};
                std::sstream report;
                for (auto &it = config.begin(); it != config.end(); ++it) {
                }
                SPDLOG_DEBUG("", report);
                return config.;
            }
            config_t m_config;
            constexpr const auto &config() const { return m_config; }
        };

        // io_buffer
        using io_buffer_data_t = double;
        std::size_t io_buffer_size() { return 4000 * 4880 * 10; }

    private:
        using input_t = InputData;
        // this is all the inputs
        std::vector<input_t> m_inputs;

        // collect input data
        const std::vector<input_t> &collect_inputs() {
            m_inputs.clear();
            if (is_master()) {
                // master does not get data.
            } else {
                // distribute inputs through out worker ranks as round robin
                auto index = worker_index();
                SPDLOG_TRACE("get input data for worker={} rank={}", index,
                             comm.rank());
                std::vector<input_t> results;
                auto c = config()["io"]["inputs"];
                assert(c.IsSequence());
                for (std::size_t i = 0; i < c.size(); i++) {
                    if (index == (meta::size_cast<int>(i) % n_workers())) {
                        results.emplace_back(c[i]);
                    }
                }
                m_inputs = std::move(results);
            }
            return m_inputs;
        }
    };

int main(int argc, char **argv) {

    mpi_utils::env mpi_env(argc, argv); // RAII MPI env
    mpi_env.set_exception_on_error();
    mxx::comm comm; // MPI_COMM_WORLD
    MPI_Win win;    // window object
    mpi_utils::logging_init(comm);
    mpi_utils::pprint_node_ranks(comm, 0, [&](const auto &node_ranks) {
        SPDLOG_INFO("{}", mpi_env);
        SPDLOG_INFO("{}", node_ranks);
    });

    auto comm_shm = comm.split_shared();
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
    coordinator_t co{comm_shm};
    if (co.is_master()) {
        const char *test_config{
            R"(---
description: |-
    This file is provided as an example of the citlali config file.
io:
    inputs:
      - name: obs1
        files:
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
        files:
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
        co.set_config(YAML::Load(test_config));
        // validate the cofnig
        coordinator_t::InputData::validate(config);
        lali::RTCProc::validate(config);

        rtcproc = RTCProc(config);
        rtcproc.process_chunk()
        /// VALIDATE_CONFIG(config, )
        /// for all types that consumes config.
    }

    // make sure all ranks have access to the config.
    co.sync().initialize();
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
    auto coor =
        mpi_utils::Span<coordinator_t::io_buffer_data_t>::allocate_shared(
            co.master_rank(), co.io_buffer_size(), comm_shm, &win);

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