#pragma once

#include <citlali/core/mapmaking/wcs.h>

// which maps to make
enum MapMode {
    None = 0,
    Obs = 2,
    Noise = 3,
    Both = 4
};

template <typename DataType>
struct StokesBase {
    DataType i, q, u;
};

struct DataMaps {
    StokesBase<Eigen::MatrixXd> signal, weight;
    StokesBase<Eigen::MatrixXd> kernel;
    StokesBase<Eigen::MatrixXd> coverage;

    void init(int n_rows, int n_cols, bool include_kernel,
              bool include_coverage, bool include_polarization) {
        signal.i = Eigen::MatrixXd::Zero(n_rows, n_cols);
        weight.i = Eigen::MatrixXd::Zero(n_rows, n_cols);

        if (include_kernel) {
            kernel.i = Eigen::MatrixXd::Zero(n_rows, n_cols);
        }
        if (include_coverage) {
            coverage.i = Eigen::MatrixXd::Zero(n_rows, n_cols);
        }
        if (include_polarization) {
            signal.q = Eigen::MatrixXd::Zero(n_rows, n_cols);
            weight.q = Eigen::MatrixXd::Zero(n_rows, n_cols);
            signal.u = Eigen::MatrixXd::Zero(n_rows, n_cols);
            weight.u = Eigen::MatrixXd::Zero(n_rows, n_cols);
        }
    }

    DataMaps& operator+=(const DataMaps& other) {

        signal.i += other.signal.i;
        weight.i += other.weight.i;

        if (kernel.i.size() > 0) {
            kernel.i += other.kernel.i;
        }

        if (coverage.i.size() > 0) {
            coverage.i += other.coverage.i;
        }

        if (signal.q.size() > 0 && weight.q.size() > 0) {
            signal.q += other.signal.q;
            weight.q += other.weight.q;
        }

        if (signal.u.size() > 0 && weight.u.size() > 0) {
            signal.u += other.signal.u;
            weight.u += other.weight.u;
        }

        return *this;
    }
};

struct NoiseMaps {
    StokesBase<std::vector<Eigen::MatrixXd>> noise;

    void init(int n_rows, int n_cols, int n_noise_maps, bool include_polarization) {
        noise.i.reserve(n_noise_maps);

        if (include_polarization) {
            noise.q.reserve(n_noise_maps);
            noise.u.reserve(n_noise_maps);
        }

        for (int i = 0; i < n_noise_maps; ++i) {
            noise.i[i] = Eigen::MatrixXd::Zero(n_rows, n_cols);

            if (include_polarization) {
                noise.q[i] = Eigen::MatrixXd::Zero(n_rows, n_cols);
                noise.u[i] = Eigen::MatrixXd::Zero(n_rows, n_cols);
            }
        }
    }

    NoiseMaps& operator+=(const NoiseMaps& other) {

        for (int i = 0; i < noise.i.size(); ++i) {
            noise.i[i] += other.noise.i[i];

            if (!noise.q.empty()) {
                noise.q[i] += other.noise.q[i];
            }

            if (!noise.u.empty()) {
                noise.u[i] += other.noise.u[i];
            }
        }

        return *this;
    }
};

struct MapPsds {
    StokesBase<Eigen::VectorXd> radial_psd, radial_freqs;
    StokesBase<Eigen::MatrixXd> psd, freqs;
};

struct MapsHists {
    StokesBase<Eigen::VectorXd> bins;
    StokesBase<Eigen::VectorXi> hist;
};

struct MapFit {
    StokesBase<Eigen::VectorXd> params, errors;

    void init(int n_params, bool include_polarization) {
        params.i.setZero(n_params);
        errors.i.setZero(n_params);

        if (include_polarization) {
            params.q.setZero(n_params);
            errors.q.setZero(n_params);

            params.u.setZero(n_params);
            errors.u.setZero(n_params);
        }
    }
};

template <typename MapType>
class MapsBase {
public:
    // map of maps to allow duplicate keys
    std::map<int, std::map<int, MapType>> maps;
    std::map<int, std::map<int, MapPsds>> psds;
    std::map<int, std::map<int, MapsHists>> hists;
    std::vector<int> arrays, groups;

    int n_rows, n_cols, n_pixels;
    double pix_size_radians;
    std::string map_grouping;
    // wcs shared for all maps
    WCS wcs;

    Eigen::VectorXd row_coords, col_coords;
    Eigen::VectorXi keys;

    bool include_polarization;

    // overload the [] operator to access maps at a particular index
    std::map<int, MapType>& operator[](int key) {
        return maps[key];
    }

    // const version of the operator[] for const objects
    const std::map<int, MapType>& operator[](int key) const {
        return maps.at(key);
    }

    MapsBase<MapType>& operator+=(MapsBase<MapType>& other) {
        for (auto &[key, upper_map] : maps) {
            for (auto& [lower_key, key_map] : upper_map) {
                key_map += other[key][lower_key];
            }
        }
        return *this;
    }
};

class DataMapsContainer : public MapsBase<DataMaps> {
public:
    std::map<int, std::map<int, MapFit>> fits;

    bool include_kernel, include_coverage;
    int n_params = 0;

    std::vector<Eigen::Map<Eigen::MatrixXd>> signal_map, weight_map, kernel_map, coverage_map;
    std::vector<Eigen::Map<Eigen::VectorXd>> params_map, errors_map;

    DataMapsContainer() {}
    DataMapsContainer(DataMapsContainer& other) {
        n_rows = other.n_rows;
        n_cols = other.n_cols;
        n_params = other.n_params;
        wcs = other.wcs;
        row_coords = other.row_coords;
        col_coords = other.col_coords;

        include_kernel = other.include_kernel;
        include_coverage = other.include_coverage;
        include_polarization = other.include_polarization;

        for (const auto& [key, other_map] : other.maps) {
            Eigen::VectorXi lower_keys(other_map.size());
            int i = 0;
            for (const auto& [lower_key, other_key_map] : other_map) {
                lower_keys(i) = lower_key;
                i++;
            }

            init_array(key, lower_keys);
            init_fit(key, lower_keys);
        }
    }

    template <typename Derived>
    void init_array(int key, Eigen::DenseBase<Derived>& lower_keys) {
        for (const auto &lower_key : lower_keys) {
            auto& key_map = maps[key][lower_key];
            key_map.init(n_rows, n_cols, include_kernel, include_coverage, include_polarization);
        }
    }

    template <typename Derived>
    void init_fit(int key, Eigen::DenseBase<Derived>& lower_keys) {
        for (const auto &lower_key : lower_keys) {
            fits[key][lower_key].init(n_params, include_polarization);
        }
    }

    void build_vectors() {
        arrays.clear();
        groups.clear();
        signal_map.clear();
        weight_map.clear();
        kernel_map.clear();
        coverage_map.clear();
        params_map.clear();
        errors_map.clear();

        for (const auto& [key, lower_keys] : maps) {
            for (const auto& [lower_key, lower_key_map] : lower_keys) {
                auto& key_map = maps[key][lower_key];

                arrays.push_back(key);
                groups.push_back(lower_key);

                signal_map.push_back(Eigen::Map<Eigen::MatrixXd>(key_map.signal.i.data(), n_rows, n_cols));
                weight_map.push_back(Eigen::Map<Eigen::MatrixXd>(key_map.weight.i.data(), n_rows, n_cols));

                params_map.push_back(Eigen::Map<Eigen::VectorXd>(fits[key][lower_key].params.i.data(), n_params));
                errors_map.push_back(Eigen::Map<Eigen::VectorXd>(fits[key][lower_key].errors.i.data(), n_params));

                if (include_kernel) {
                    kernel_map.push_back(Eigen::Map<Eigen::MatrixXd>(key_map.kernel.i.data(), n_rows, n_cols));
                }

                if (include_coverage) {
                    coverage_map.push_back(Eigen::Map<Eigen::MatrixXd>(key_map.coverage.i.data(), n_rows, n_cols));
                }

                if (include_polarization) {
                    arrays.push_back(key);
                    groups.push_back(lower_key);
                    signal_map.push_back(Eigen::Map<Eigen::MatrixXd>(key_map.signal.q.data(), n_rows, n_cols));
                    weight_map.push_back(Eigen::Map<Eigen::MatrixXd>(key_map.weight.q.data(), n_rows, n_cols));

                    arrays.push_back(key);
                    groups.push_back(lower_key);
                    signal_map.push_back(Eigen::Map<Eigen::MatrixXd>(key_map.signal.u.data(), n_rows, n_cols));
                    weight_map.push_back(Eigen::Map<Eigen::MatrixXd>(key_map.weight.u.data(), n_rows, n_cols));

                    params_map.push_back(Eigen::Map<Eigen::VectorXd>(fits[key][lower_key].params.q.data(), n_params));
                    errors_map.push_back(Eigen::Map<Eigen::VectorXd>(fits[key][lower_key].errors.q.data(), n_params));
                    params_map.push_back(Eigen::Map<Eigen::VectorXd>(fits[key][lower_key].params.u.data(), n_params));
                    errors_map.push_back(Eigen::Map<Eigen::VectorXd>(fits[key][lower_key].errors.u.data(), n_params));
                }
            }
        }
    }
};

class NoiseMapsContainer : public MapsBase<NoiseMaps> {
public:
    int n_noise_maps = 0;

    std::vector<Eigen::Map<Eigen::MatrixXd>> noise_map;

    NoiseMapsContainer() {}
    NoiseMapsContainer(NoiseMapsContainer& other) {
        n_rows = other.n_rows;
        n_cols = other.n_cols;
        n_noise_maps = other.n_noise_maps;
        wcs = other.wcs;
        row_coords = other.row_coords;
        col_coords = other.col_coords;

        include_polarization = other.include_polarization;

        for (const auto& [key, other_map] : other.maps) {
            Eigen::VectorXi lower_keys(other_map.size());
            int i = 0;
            for (const auto& [lower_key, other_key_map] : other_map) {
                lower_keys(i) = lower_key;
                i++;
            }

            init_array(key, lower_keys);
        }
    }

    template <typename Derived>
    void init_array(int key, Eigen::DenseBase<Derived>& lower_keys) {
        for (const auto &lower_key : lower_keys) {
            auto& key_map = maps[key][lower_key];

            key_map.init(n_rows, n_cols, n_noise_maps, include_polarization);
        }
    }

    void build_vectors() {
        arrays.clear();
        groups.clear();
        noise_map.clear();

        for (const auto& [key, lower_keys] : maps) {
            for (const auto& [lower_key, lower_key_map] : lower_keys) {
                auto& key_map = maps[key][lower_key];

                for (int i = 0; i < n_noise_maps; ++i) {
                    arrays.push_back(key);
                    groups.push_back(lower_key);
                    noise_map.push_back(Eigen::Map<Eigen::MatrixXd>(key_map.noise.i[i].data(), n_rows, n_cols));

                    if (include_polarization) {
                        arrays.push_back(key);
                        groups.push_back(lower_key);
                        noise_map.push_back(Eigen::Map<Eigen::MatrixXd>(key_map.noise.q[i].data(), n_rows, n_cols));

                        arrays.push_back(key);
                        groups.push_back(lower_key);
                        noise_map.push_back(Eigen::Map<Eigen::MatrixXd>(key_map.noise.u[i].data(), n_rows, n_cols));
                    }
                }
            }
        }
    }
};
