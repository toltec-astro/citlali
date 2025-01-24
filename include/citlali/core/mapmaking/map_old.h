# pragma once

#include <citlali/core/mapmaking/wcs.h>

// which maps to make
enum MapType {
    None = 0,
    Obs = 2,
    Noise = 3,
    Both = 4
};

// hold data for the Stokes params (I, Q, U)
template <typename DataType>
struct StokesBase {
    DataType i;
    std::optional<DataType> q, u;
};

// holds all data map types
struct DataMap {
    StokesBase<Eigen::MatrixXd> signal, weight;
    std::optional<StokesBase<Eigen::MatrixXd>> kernel, coverage;
};

// holds noise maps
struct NoiseMap {
    std::optional<std::vector<StokesBase<Eigen::MatrixXd>>> noise;
};

// holds fitted parameters for maps
struct MapParam {
    StokesBase<Eigen::VectorXd> params, error;
};

// holds psds of maps
struct MapPSD {
    StokesBase<Eigen::VectorXd> radial_psd, radial_freqs;
    StokesBase<Eigen::MatrixXd> psd, freqs;
};

// holds map of apt column value to a DataMap class containing all map types
class ArrayMap {
public:
    // overload the [] operator to access DataMap at a particular index
    DataMap& operator[](int key) {
        return maps[key];
    }

    // const version of the operator[] for const objects
    const DataMap& operator[](int key) const {
        return maps.at(key);
    }

    template<typename Derived>
    void init_maps(const Eigen::DenseBase<Derived>& map_keys,
                   int n_rows, int n_cols,
                   bool include_data = true,
                   bool include_kernel = false,
                   bool include_coverage = false,
                   bool include_polarization = false,
                   int n_noise_maps = 0) {

        for (auto& map_key : map_keys) {
            auto& data_map = maps[map_key];

            if (include_data) {
                // initialize signal and weight maps
                data_map.signal.i = Eigen::MatrixXd::Zero(n_rows, n_cols);
                data_map.weight.i = Eigen::MatrixXd::Zero(n_rows, n_cols);

                if (include_polarization) {
                    data_map.signal.q.emplace(Eigen::MatrixXd::Zero(n_rows, n_cols));
                    data_map.weight.q.emplace(Eigen::MatrixXd::Zero(n_rows, n_cols));
                    data_map.signal.u.emplace(Eigen::MatrixXd::Zero(n_rows, n_cols));
                    data_map.weight.u.emplace(Eigen::MatrixXd::Zero(n_rows, n_cols));
                }

                // optionally initialize kernel map
                if (include_kernel) {
                    data_map.kernel.emplace();
                    data_map.kernel->i = Eigen::MatrixXd::Zero(n_rows, n_cols);

                    if (include_polarization) {
                        data_map.kernel->q.emplace(Eigen::MatrixXd::Zero(n_rows, n_cols));
                        data_map.kernel->u.emplace(Eigen::MatrixXd::Zero(n_rows, n_cols));
                    }
                }

                // optionally initialize coverage map
                if (include_coverage) {
                    data_map.coverage.emplace();
                    data_map.coverage->i = Eigen::MatrixXd::Zero(n_rows, n_cols);

                    if (include_polarization) {
                        data_map.coverage->q.emplace(Eigen::MatrixXd::Zero(n_rows, n_cols));
                        data_map.coverage->u.emplace(Eigen::MatrixXd::Zero(n_rows, n_cols));
                    }
                }
            }

            // optionally initialize noise maps
            if (n_noise_maps > 0) {
                data_map.noise.emplace(n_noise_maps);
                for (auto& noise_map : *data_map.noise) {
                    noise_map.i = Eigen::MatrixXd::Zero(n_rows, n_cols);
                    if (include_polarization) {
                        noise_map.q.emplace(Eigen::MatrixXd::Zero(n_rows, n_cols));
                        noise_map.u.emplace(Eigen::MatrixXd::Zero(n_rows, n_cols));
                    }
                }
            }
        }
    }

    // map holding all MapSet objects for different array keys
    std::map<int, DataMap> maps;
    std::map<int, MapParams> params;
    std::map<int, MapPSD> psd;
};

// holds all array maps and shared metadata
class MapContainer {
public:
    // overload the [] operator to access ArrayMap at a particular index
    ArrayMap& operator[](int key) {
        return array_maps[key];
    }

    // const version of the operator[] for const objects
    const ArrayMap& operator[](int key) const {
        return array_maps.at(key);
    }

    // initialize maps for a given array
    template <typename Derived>
    void init_maps(int array, const Eigen::DenseBase<Derived>& map_keys, bool include_data) {
        (*this)[array].init_maps(map_keys, n_rows, n_cols, include_data, include_kernel, include_coverage, include_polarization, n_noise_maps);
    }

    // collect all signal, weight, and kernel references from all array maps
    void build_vectors() {

        using Map = Eigen::Map<Eigen::MatrixXd>;

        // clear existing references
        signal.clear();
        weight.clear();
        kernel.clear();
        arrays.clear();
        groups.clear();

        // loop through all array maps
        for (auto& [array_key, array_map] : array_maps) {
            // loop through all map sets in the array map
            for (auto& [map_key, data_map] : array_map.maps) {
                // collect signal references
                signal.push_back(Map(data_map.signal.i.data(), n_rows, n_cols));
                if (data_map.signal.q.has_value()) signal.push_back(Map(data_map.signal.q->data(), n_rows, n_cols));
                if (data_map.signal.u.has_value()) signal.push_back(Map(data_map.signal.u->data(), n_rows, n_cols));

                // collect weight references
                weight.push_back(Map(data_map.weight.i.data(), n_rows, n_cols));
                if (data_map.weight.q.has_value()) weight.push_back(Map(data_map.weight.q->data(), n_rows, n_cols));
                if (data_map.weight.u.has_value()) weight.push_back(Map(data_map.weight.u->data(), n_rows, n_cols));

                // collect kernel references
                if (data_map.kernel.has_value()) {
                    kernel.push_back(Map(data_map.kernel->i.data(), n_rows, n_cols));
                }

                if (data_map.coverage.has_value()) {
                    coverage.push_back(Map(data_map.coverage->i.data(), n_rows, n_cols));
                }

                arrays.push_back(array_key);
                groups.push_back(map_key);
            }
        }
    }

    void operator+(const MapContainer& other) {
        // loop through all array maps
        for (auto& [array_key, array_map] : array_maps) {
            // loop through all maps in the array map
            for (auto& [map_key, data_map] : array_map.maps) {
                data_map.signal.i += other[array_key][map_key].signal.i;
                data_map.weight.i += other[array_key][map_key].weight.i;

                if (data_map.kernel.has_value()) {
                    data_map.kernel->i += other[array_key][map_key].kernel->i;
                }
                if (data_map.coverage.has_value()) {
                    data_map.coverage->i += other[array_key][map_key].coverage->i;
                }
                if (data_map.signal.q.has_value()) {
                    (*data_map.signal.q) += (*other[array_key][map_key].signal.q);
                }
                if (data_map.signal.u.has_value()) {
                    (*data_map.signal.u) += (*other[array_key][map_key].signal.u);
                }
                if (data_map.weight.q.has_value()) {
                    (*data_map.weight.q) += (*other[array_key][map_key].weight.q);
                }
                if (data_map.weight.u.has_value()) {
                    (*data_map.weight.u) += (*other[array_key][map_key].weight.u);
                }
            }
        }
    }

    // shared WCS object
    WCS wcs;
    // pixel size in radians
    double pix_size_radians;
    // dimensions of maps
    int n_rows, n_cols;
    // coordinates of each pixel
    Eigen::VectorXd row_coords, col_coords;
    // flags for optional maps
    bool include_kernel, include_coverage, include_polarization;
    // number of noise maps
    int n_noise_maps;
    // apt key for grouping (i.e. uid, nw, array, fg)
    std::string map_grouping;

    // map holding all ArrayMap objects for different array keys
    std::map<int, ArrayMap> array_maps;

    // vectors to store map references for loops
    std::vector<Eigen::Map<Eigen::MatrixXd>> signal, weight, kernel, coverage;
    std::vector<Eigen::Map<Eigen::VectorXd>> params;

    std::vector<int> arrays, groups;
};
