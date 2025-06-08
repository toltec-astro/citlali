#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <type_traits>
#include <string>
#include <unordered_map>
#include <functional>

#include <Eigen/Dense>
#include <Eigen/Sparse>

// which maps to make
enum class MapMode : int {
    None = 0,
    Obs = 1 << 0,
    Noise = 2 << 0,
    Both = Obs | Noise
};

inline MapMode operator|(MapMode lhs, MapMode rhs) {
    return static_cast<MapMode>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline MapMode operator&(MapMode lhs, MapMode rhs) {
    return static_cast<MapMode>(static_cast<int>(lhs) & static_cast<int>(rhs));
}

inline MapMode operator^(MapMode lhs, MapMode rhs) {
    return static_cast<MapMode>(static_cast<int>(lhs) ^ static_cast<int>(rhs));
}

inline MapMode& operator|=(MapMode& lhs, MapMode rhs) {
    lhs = lhs | rhs;
    return lhs;
}

inline MapMode& operator&=(MapMode& lhs, MapMode rhs) {
    lhs = lhs & rhs;
    return lhs;
}

inline MapMode& operator^=(MapMode& lhs, MapMode rhs) {
    lhs = lhs ^ rhs;
    return lhs;
}

inline bool get_map_mode(MapMode value, MapMode flag) {
    return (static_cast<int>(value) & static_cast<int>(flag)) != 0;
}

// key for unordered maps
struct MapKey {
    int array_index;
    int group_index;
    std::string stokes;

    MapKey(int ai, int gi, const std::string& s)
        : array_index(ai), group_index(gi), stokes(s) {}

    bool operator==(const MapKey& other) const {
        return array_index == other.array_index &&
               group_index == other.group_index &&
               stokes == other.stokes;
    }
};


namespace std {
template <>
struct hash<MapKey> {
    std::size_t operator()(const MapKey& k) const {
        std::size_t h1 = std::hash<int>{}(k.array_index);
        std::size_t h2 = std::hash<int>{}(k.group_index);
        std::size_t h3 = std::hash<std::string>{}(k.stokes);

        // Combine hashes using boost::hash_combine-like technique
        std::size_t seed = h1;
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
}

using DefaultMapKeyType = MapKey;

template <typename MapKeyType = DefaultMapKeyType>
struct ObsSparse {
    MapKeyType key;
    std::vector<Eigen::Triplet<double, Eigen::Index>> data;
    int n;

    ObsSparse(const MapKeyType& k, const std::vector<int>& dims)
        : key(k), n(dims[0]*dims[1]) {
        data.reserve(n);
    }

    void operator()(int row, int col, double value) {
        data.emplace_back(row, col, value);
    }

    template <typename OtherMapKeyType>
    void operator+=(const ObsSparse<OtherMapKeyType>& other) {
        data.insert(data.end(), other.data.begin(), other.data.end());
    }

    void set_zero() {
        data.clear();
        data.reserve(n);
    }
};

template <typename MapKeyType = DefaultMapKeyType>
struct ObsMatrix {
    MapKeyType key;
    Eigen::MatrixXd data;
    Eigen::Index n_rows, n_cols;

    ObsMatrix(const MapKeyType& k, const std::vector<int>& dims)
    : key(k), n_rows(dims[0]), n_cols(dims[1]) {
        data = Eigen::MatrixXd::Zero(n_rows, n_cols);
    }

    void operator()(int row, int col, double value) {
        data(row, col) += value;
    }

    template <typename OtherMapKeyType>
    void operator+=(const ObsMatrix<OtherMapKeyType>& other) {
        data.array() += other.data.array();
    }

    template <typename OtherMapKeyType>
    void operator+=(const ObsSparse<OtherMapKeyType>& other) {
        Eigen::SparseMatrix<double> sparse(n_rows, n_cols);
        sparse.setFromTriplets(other.data.begin(), other.data.end());
        data += sparse;
    }

    void set_zero() {
        data.setZero();
    }
};


template <typename MapKeyType = DefaultMapKeyType, typename MapType = ObsMatrix<MapKeyType>>
class ObsMaps {
public:
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    ObsMaps() = default;
    std::vector<MapType> signal, weight, kernel, coverage;
    std::unordered_map<MapKeyType, int> signal_lookup, weight_lookup, kernel_lookup, coverage_lookup;

    int n_maps = 0;

    Eigen::VectorXd rows, cols;

    WCS wcs;

    // n_params x n_maps
    Eigen::MatrixXd params, errors;
    std::vector<Eigen::VectorXd> radial_psd, radial_freqs;
    std::vector<Eigen::MatrixXd> psd, freqs;
    std::vector<Eigen::VectorXd> bins;
    std::vector<Eigen::VectorXi> counts;

    void add(MapKeyType key, const std::vector<int>& dims, bool add_weight = true,
             bool add_kernel = false, bool add_coverage = false) {

        add_to(signal, signal_lookup, key, dims);

        if (add_weight) add_to(weight, weight_lookup, key, dims);
        if (add_kernel) add_to(kernel, kernel_lookup, key, dims);
        if (add_coverage) add_to(coverage, coverage_lookup, key, dims);

        n_maps++;
    }

    void set_zero() {
        zero_container(signal);
        zero_container(weight);
        zero_container(kernel);
        zero_container(coverage);
    }

    template <typename OtherMapKeyType, typename OtherMapType>
    void operator+=(ObsMaps<OtherMapKeyType, OtherMapType> &other) {
        add_containers(signal, other.signal);
        add_containers(weight, other.weight);
        add_containers(kernel, other.kernel);
        add_containers(coverage, other.coverage);
    }

private:
    template <typename Container>
    void add_to(Container& container,
                std::unordered_map<MapKeyType, int>& lookup,
                const MapKeyType& key, const std::vector<int>& dims) {
        if constexpr (is_std_vector<MapType>::value) {
            container.emplace_back();
            for (int i = 0; i < dims[2]; ++i) {
                container.back().emplace_back(key, dims);
            }
        }
        else {
            container.emplace_back(key, dims);
        }

        lookup[key] = static_cast<int>(container.size()) - 1;
    }

    template <typename Container0, typename Container1>
    void add_containers(Container0& lhs, const Container1& rhs) {
        for (int i = 0; i < lhs.size(); ++i) {
            if constexpr (is_std_vector<MapType>::value) {
                for (int j = 0; j < lhs[i].size(); ++j) {
                    lhs[i][j] += rhs[i][j];
                }
            }
            else {
                lhs[i] += rhs[i];
            }
        }
    }

    template <typename Container>
    void zero_container(Container& container) {
        for (int i = 0; i < container.size(); ++i) {
            if constexpr (is_std_vector<MapType>::value) {
                for (int j = 0; j < container[i].size(); ++j) {
                    container[i][j].set_zero();
                }
            }
            else {
                container[i].set_zero();
            }
        }
    }
};
