#include "enum.h"
#include "formatter/ptr.h"
#include "formatter/utils.h"
#include "logging.h"
#include <Eigen/Core>
#include <mxx/comm.hpp>
#include <mxx/env.hpp>
#include <mxx/utils.hpp>
#include <sstream>
#include <string>

namespace mpi_utils {

inline auto logging_init(const mxx::comm &comm) {
    logging::init();
    auto pattern =
        fmt::format("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#:%!] [{}/{}] %v",
                    comm.rank(), comm.size());
    spdlog::set_pattern(pattern);
}

struct env : mxx::env {
    META_ENUM(WinMemoryModel, int, Unified, Separate, Unknown, NotSupported);
    template <typename... Args> env(Args... args) : mxx::env(args...) {
        // get memory model
        int *attr_val;
        int attr_flag;
        MPI_Win win;
        MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        /* this function will always return flag=false in MPI-2 */
        MPI_Win_get_attr(win, MPI_WIN_MODEL, &attr_val, &attr_flag);
        MPI_Win_free(&win);
        if (attr_flag) {
            if ((*attr_val) == MPI_WIN_UNIFIED) {
                memory_model = WinMemoryModel::Unified;
            } else if ((*attr_val) == MPI_WIN_SEPARATE) {
                memory_model = WinMemoryModel::Separate;
            } else {
                memory_model = WinMemoryModel::Unknown;
            }
        } else {
            memory_model = WinMemoryModel::NotSupported;
        }
        // api version
        MPI_Get_version(&api_version.first, &api_version.second);
        // lib info
        {
            char lib_info_[MPI_MAX_LIBRARY_VERSION_STRING];
            int lib_info_len;
            MPI_Get_library_version(lib_info_, &lib_info_len);
            lib_info = lib_info_;
        }
    }
    ~env();
    std::pair<int, int> api_version{0, 0};
    std::string lib_info{""};
    WinMemoryModel memory_model{WinMemoryModel::Unknown};
};
REGISTER_META_ENUM(env::WinMemoryModel);

env::~env() = default;

template <typename Func>
auto call_with_node_ranks(int rank, const mxx::comm &comm, Func &&func) {
    using index_t = decltype(comm.size());
    char node_name[MPI_MAX_PROCESSOR_NAME];
    int node_len;
    MPI_Get_processor_name(node_name, &node_len);
    // gather all processor names to rank
    std::vector<char> node_names_raw =
        mxx::gatherv(node_name, SIZET(MPI_MAX_PROCESSOR_NAME), rank, comm);
    if (rank == comm.rank()) {
        std::vector<std::string> node_names;

        for (index_t i = 0; i < comm.size(); ++i) {
            node_names.emplace_back(node_names_raw.data() +
                                        i * MPI_MAX_PROCESSOR_NAME,
                                    MPI_MAX_PROCESSOR_NAME);
        }
        // SPDLOG_TRACE("node names: {}", node_names);
        std::unordered_map<std::string, std::vector<index_t>> node_ranks_;
        for (index_t i = 0; i < comm.size(); ++i) {
            node_ranks_[node_names[SIZET(i)]].push_back(i);
        }
        std::vector<std::pair<std::string, std::vector<index_t>>> node_ranks;
        for (auto &[node, ranks] : node_ranks_) {
            std::sort(ranks.begin(), ranks.end());
            node_ranks.emplace_back(node, ranks);
        }
        // sort the vector of node names by first rank
        std::sort(node_ranks.begin(), node_ranks.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.second.front() < rhs.second.front();
                  });
        FWD(func)(std::move(node_ranks));
    }
}

template <typename Func>
auto pprint_node_ranks(int rank, const mxx::comm &comm, Func &&func) {
    call_with_node_ranks(
        rank, comm, [&comm, func = FWD(func)](auto &&node_ranks) {
            std::stringstream ss;
            ss << fmt::format("MPI comm layout:"
                              "\n  n_procs: {}\n  n_nodes: {}",
                              comm.size(), node_ranks.size());
            for (std::size_t i = 0; i < node_ranks.size(); ++i) {
                auto &[node, ranks] = node_ranks[i];
                ss << fmt::format("\n  {}: {}\n      ranks: {}", i, node,
                                  ranks);
            }
            func(ss.str());
        });
}

struct comm : mxx::comm {
    template <typename... Args>
    comm(Args &&... args) : mxx::comm{FWD(args)...} {
        // initialize node info
        constexpr auto master_rank = 0;
        constexpr auto tag = 0;
        call_with_node_ranks(master_rank, *this, [this](auto &&node_ranks) {
            this->m_n_nodes = node_ranks.size();
            for (std::size_t i = 0; i < node_ranks.size(); ++i) {
                auto &[node, ranks] = node_ranks[i];
                for (auto r : ranks) {
                    auto m_node_index = meta::size_cast<index_t>(i);
                    auto m_node_name = node;
                    auto m_node_name_size =
                        meta::size_cast<int>(m_node_name.size());
                    if (r == master_rank) {
                        this->m_node_index = m_node_index;
                        this->m_node_name = m_node_name;
                    } else {
                        // send to other processes
                        MPI_Send(&m_node_index, 1, MPI_INT, r, tag, *this);
                        MPI_Send(&m_node_name_size, 1, MPI_INT, r, tag, *this);
                        MPI_Send(const_cast<char *>(m_node_name.data()),
                                 m_node_name_size, MPI_CHAR, r, tag, *this);
                    }
                }
            }
        });
        MPI_Bcast(&this->m_n_nodes, 1, MPI_INT, master_rank, *this);
        if (rank() != master_rank) {
            MPI_Recv(&this->m_node_index, 1, MPI_INT, master_rank, tag, *this,
                     MPI_STATUS_IGNORE);
            int m_node_name_size{0};
            MPI_Recv(&m_node_name_size, 1, MPI_INT, master_rank, tag, *this,
                     MPI_STATUS_IGNORE);
            this->m_node_name.resize(SIZET(m_node_name_size));
            MPI_Recv(const_cast<char *>(this->m_node_name.data()),
                     m_node_name_size, MPI_CHAR, master_rank, tag, *this,
                     MPI_STATUS_IGNORE);
        }
        assert(n_nodes() > 0);
        assert(node_index() >= 0);
    }
    using index_t = int;
    // getters
    constexpr index_t n_nodes() const { return m_n_nodes; }
    constexpr index_t node_index() const { return m_node_index; }
    constexpr const std::string &node_name() const { return m_node_name; }
    // node info
    index_t m_n_nodes{-1};
    index_t m_node_index{-1};
    std::string m_node_name{""};

    template <typename Func, typename... Args>
    auto bcast_str(index_t from_rank, Func &&func, Args &&... args) const {
        std::string str{};
        int size{0};
        if (rank() == from_rank) {
            str = FWD(func)(FWD(args)...);
            size = meta::size_cast<int>(str.size());
        }
        MPI_Bcast(&size, 1, MPI_INT, from_rank, *this);
        if (rank() != from_rank) {
            str.resize(SIZET(size));
        }
        MPI_Bcast(const_cast<char *>(str.data()), size, MPI_CHAR, from_rank,
                  *this);
        return str;
    }
};

/**
 * @brief The Span struct
 * This provides contexts to a continuous block of memory.
 */
template <typename T, typename Index> struct Span {
    using data_t = T;
    using index_t = Index;
    using disp_unit_t = int; // set by MPI API
    static constexpr disp_unit_t disp_unit{sizeof(data_t)};
    using vec_t = Eigen::Map<Eigen::Matrix<data_t, Eigen::Dynamic, 1>>;
    using mat_t =
        Eigen::Map<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>>;
    Span(data_t *data_, index_t size_) : data(data_), size(size_) {}
    data_t *data{nullptr};
    index_t size{0};

    auto asvec() { return vec_t(data, size); }
    auto asmat(index_t nrows, index_t ncols) {
        assert(nrows * ncols <= size);
        return mat_t(data, nrows, ncols);
    }
    template <typename shape_t> auto asmat(const shape_t &shape) {
        assert(shape.size() == 2);
        return asmat(shape[0], shape[1]);
    }

    /**
     * @brief Create span as MPI shared memory.
     * @param rank The rank at which the memory is allocated
     * @param size The size of the memory block
     * @param comm The MPI comm across which the memory is shared
     * @param p_win The RMA context.
     */
    static auto allocate_shared(int master_rank, index_t size,
                                const mxx::comm &comm, MPI_Win *p_win) {
        // here the MPI APIs works with size of type MPI_Aint, so we
        // need to be careful about overflow issue for large array
        Span arr(nullptr, size);
        auto mem_size = meta::size_cast<MPI_Aint>(arr.size * disp_unit);
        // allocate
        MPI_Win_allocate_shared(comm.rank() == master_rank ? mem_size : 0,
                                arr.disp_unit, MPI_INFO_NULL, comm, &arr.data,
                                p_win);
        if (comm.rank() == master_rank) {
            SPDLOG_TRACE("shared memory allocated {}", arr);
        } else {
            // update the pointer to point to the shared memory on the
            // creation rank
            MPI_Aint _mem_size;
            Span::disp_unit_t _disp_unit;
            MPI_Win_shared_query(*p_win, master_rank, &_mem_size, &_disp_unit,
                                 &arr.data);
            // update size
            arr.size = meta::size_cast<Span::index_t>(_mem_size) / disp_unit;
            assert(_disp_unit == arr.disp_unit);
            SPDLOG_TRACE("shared memroy connected from rank={} {}", comm.rank(),
                         arr);
        }
        return arr;
    }
};

#define MPI_UTILS_DECLTYPE(v)                                                  \
    mxx::get_datatype<std::decay_t<decltype(v)>>().type()
#define MPI_UTILS_GETTYPE(T) mxx::get_datatype<T>().type()

} // namespace mpi_utils

namespace fmt {

template <>
struct formatter<mpi_utils::env, char> : fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const mpi_utils::env &e, FormatContext &ctx)
        -> decltype(ctx.out()) {
        return format_to(
            ctx.out(),
            "MPI environment:\n  MPI-API v{}.{}\n  {}\n  Memory model: {}",
            e.api_version.first, e.api_version.second, e.lib_info,
            e.memory_model);
    }
};

template <typename... Ts>
struct formatter<mpi_utils::Span<Ts...>, char>
    : fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const mpi_utils::Span<Ts...> &arr, FormatContext &ctx)
        -> decltype(ctx.out()) {
        return format_to(ctx.out(), "@{:z} size={} disp_unit={}",
                         fmt_utils::ptr(arr.data), arr.size, arr.disp_unit);
    }
};
} // namespace fmt
