# pragma once

#include <mutex>

std::tuple<int, int, int, int, int, int> calc_bounds(
    int n_rows, int n_cols, int j_rows, int j_cols, int center_row, int center_col) {

    // calculate the start position in the map
    int start_row = center_row - j_rows / 2;
    int start_col = center_col - j_cols / 2;

    // clamp the map start positions to the map boundaries
    int map_start_row = std::max(0, start_row);
    int map_start_col = std::max(0, start_col);

    // calculate the end position and clamp it to the map boundaries
    int map_end_row = std::min(n_rows, start_row + j_rows);
    int map_end_col = std::min(n_cols, start_col + j_cols);

    // calculate the size for the map region
    int map_size_row = map_end_row - map_start_row;
    int map_size_col = map_end_col - map_start_col;

    // calculate the start position for the array (clamped to 0 if out of bounds)
    int array_start_row = std::max(0, -start_row);
    int array_start_col = std::max(0, -start_col);

    return std::make_tuple(map_start_row, map_start_col, map_size_row, map_size_col,
                           array_start_row, array_start_col);
}

// Jinc Mapmaker
template <typename TCDataType>
class JincMapmaker : public PipelineComponent<TCDataType> {
public:
    DataMapsContainer& obs_maps, coadd_maps;
    NoiseMapsContainer& noise_maps;
    Instrument& toltec;
    Telescope& telescope;

    bool randomize_dets;
    bool run_noise;

    MapMode map_type;

    // jinc shape parameters
    std::map<int, std::vector<double>> jinc_shape;
    double r_max;
    double pix_size_arcsec;

    // map array index to jinc array
    std::map<int, Eigen::MatrixXd> jinc_filter_matrix;

    // for adding thread local map values to map containers
    std::unique_ptr<std::mutex> jinc_mutex = std::make_unique<std::mutex>();

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    template <typename ConfigType>
    JincMapmaker(MapMode _map_type, Instrument& toltec_ref, Telescope& telescope_ref, DataMapsContainer& obs_map_ref,
                  DataMapsContainer& coadd_map_ref, NoiseMapsContainer& noise_map_ref, ConfigType& config)
        : map_type(_map_type), toltec(toltec_ref), telescope(telescope_ref), obs_maps(obs_map_ref), coadd_maps(coadd_map_ref), noise_maps(noise_map_ref) {

        config.get(randomize_dets, std::tuple{"noise_maps","randomize_dets"});

        for (const auto& [array_index, array_name] : toltec.array_index_to_name) {
            std::vector<double> jinc_shape_array;
            config.get(jinc_shape_array, std::tuple{"mapmaking","jinc_filter","shape_params",array_name});
            jinc_shape[array_index] = jinc_shape_array;
        }

        config.get(r_max, std::tuple{"mapmaking","jinc_filter","r_max"});
        config.get(randomize_dets, std::tuple{"noise_maps","randomize_dets"});
        config.get(pix_size_arcsec, std::tuple{"mapmaking","pixel_size_arcsec"});

        // only run if no bad keys found
        if (config.missing_keys.empty() && config.invalid_keys.empty()) {
            allocate_jinc_matrix();
        }
    }

    void init() {}
    void add_chunk_to_map(TCDataType&);
    void process(TCDataType& tcdata) override {
        logger->info("jinc mapmaker processing");
        add_chunk_to_map(tcdata);
    }

    double jinc_func(double r, double a, double b, double c, double r_max) const {
        if (r == 0.0) {
            return 1.0;
        }

        // calculate jinc function components
        double jinc_1 = gsl_sf_bessel_J1(2.0 * pi * r / a) / (pi * r / a);
        double exp_func = std::exp(-std::pow(2.0 * r / b, c));
        double jinc_2 = gsl_sf_bessel_J1(3.831706 * r / r_max) / (3.831706 * r / r_max);

        return jinc_1 * exp_func * jinc_2;
    }

    void allocate_jinc_matrix() {
        for (const auto& [array_index, wavelength_m] : toltec.array_index_to_wavelength) {
            // get shape params
            double a = jinc_shape[array_index][0];
            double b = jinc_shape[array_index][1];
            double c = jinc_shape[array_index][2];

            double wave_d = wavelength_m / telescope.lmt_diameter_m;
            int r_max_pix = std::floor(r_max * wave_d / (pix_size_arcsec * ASEC_TO_RAD));
            int n_pts = 2.0 * r_max_pix + 1;
            Eigen::VectorXd pix_range = Eigen::VectorXd::LinSpaced(n_pts, -r_max_pix, r_max_pix);

            jinc_filter_matrix[array_index].setZero(n_pts, n_pts);

            for (int i = 0; i < n_pts; ++i) {
                for (int j = 0; j < n_pts; ++j) {
                    double r = (pix_size_arcsec * ASEC_TO_RAD) * std::sqrt(pow(pix_range(i), 2) + pow(pix_range(j), 2));
                    jinc_filter_matrix[array_index](i,j) = jinc_func(r / wave_d, a, b, c, r_max);
                }
            }
        }
    }
};

// populate map
template <typename TCDataType>
void JincMapmaker<TCDataType>::add_chunk_to_map(TCDataType& tcdata) {
    int n_dets = tcdata.n_dets();
    int n_pts = tcdata.n_pts();

    // if calibration beammapping (no coverage) clear flags so all maps are made
    if (!obs_maps[toltec.apt["array"].data(0)][toltec.apt[obs_maps.map_grouping].data(0)].coverage.i.size() > 0) {
        tcdata.apt_flag.setZero();
    }

    DataMapsContainer chunk_maps(obs_maps);
    NoiseMapsContainer chunk_noise_maps(noise_maps);

    // 1 or -1 vector or matrix for noise timestreams
    Eigen::MatrixXi noise;
    if (noise_maps.n_noise_maps > 0) {
        if (randomize_dets) {
            noise.resize(noise_maps.n_noise_maps, n_dets);
        } else {
            noise.resize(noise_maps.n_noise_maps, 1);
        }
        // populates noise
        tcdata.generate_noise(noise);
    }

    // loop through dets
    for (int det = 0; det < n_dets; ++det) {
        // don't run detectors flagged for this chunk
        if (tcdata.apt_flag(det)) continue;

        // keys of current detector
        int array = toltec.apt["array"].data(det);
        int group = toltec.apt[obs_maps.map_grouping].data(det);

        // jinc filter for current array
        const auto& jinc_array = jinc_filter_matrix[array];

        // which map detector belongs to
        auto& obs_map = obs_maps[array][group];

        // get detector pointing
        auto xy = calc_pointing(toltec.apt["x_t"].data(det), toltec.apt["y_t"].data(det), tcdata.tel_data,
                                telescope.pixel_axes);

        // pixels for samples
        Eigen::VectorXI pix_x = (xy.first.array() / obs_maps.pix_size_radians + obs_maps.n_cols / 2.0).template cast<Eigen::Index>();
        Eigen::VectorXI pix_y = (xy.second.array() / obs_maps.pix_size_radians + obs_maps.n_rows / 2.0).template cast<Eigen::Index>();

        // pixels in noise maps
        Eigen::VectorXI noise_pix_x, noise_pix_y;
        if (noise_maps.n_noise_maps > 0) {
            noise_pix_x = (xy.first.array() / noise_maps.pix_size_radians + noise_maps.n_cols / 2.0).template cast<Eigen::Index>();
            noise_pix_y = (xy.second.array() / noise_maps.pix_size_radians + noise_maps.n_rows / 2.0).template cast<Eigen::Index>();
        }

        // loop through data points
        for (int i = 0; i < n_pts; ++i) {
            // don't run flagged detectors
            if (tcdata.flag(i, det)) continue;

            if (map_type == MapMode::Obs || map_type == MapMode::Both) {
                // if pixel in map
                if (pix_x(i) >= 0 && pix_x(i) < obs_maps.n_cols && pix_y(i) >= 0 && pix_y(i) < obs_maps.n_rows) {
                    // get bounds
                    auto [map_start_row, map_start_col, map_size_row, map_size_col, array_start_row, array_start_col] =
                        calc_bounds(obs_maps.n_rows, obs_maps.n_cols, jinc_array.rows(), jinc_array.cols(), pix_y(i), pix_x(i));

                    // blocks to relevant regions of maps and jinc array
                    auto jinc_block = jinc_array.block(array_start_row, array_start_col, map_size_row, map_size_col);
                    auto signal_block = chunk_maps[array][group].signal.i.block(map_start_row, map_start_col, map_size_row, map_size_col);
                    auto weight_block = chunk_maps[array][group].weight.i.block(map_start_row, map_start_col, map_size_row, map_size_col);

                    signal_block += jinc_block * tcdata.signal(i, det) * tcdata.weight(det);
                    weight_block += jinc_block * tcdata.weight(det);

                    if (obs_map.kernel.i.size() > 0) {
                        auto kernel_block = chunk_maps[array][group].kernel.i.block(map_start_row, map_start_col, map_size_row, map_size_col);
                        kernel_block += jinc_block * tcdata.kernel(i, det) * tcdata.weight(det);
                    }
                    if (obs_map.coverage.i.size() > 0) {
                        auto coverage_block = chunk_maps[array][group].coverage.i.block(map_start_row, map_start_col, map_size_row, map_size_col);
                        coverage_block += jinc_block * 1.0 / tcdata.data_fs_hz;
                    }

                    // only add to stokes maps if loc != -1
                    if (obs_maps.include_polarization && toltec.apt["loc"].data(det) != -1) {
                        if (obs_map.signal.q.size() > 0) {
                            auto signal_q_block = chunk_maps[array][group].signal.q.block(map_start_row, map_start_col, map_size_row, map_size_col);
                            signal_q_block += jinc_block * (*tcdata.signal_q)(i, det) * (*tcdata.weight_q)(det);
                        }
                        if (obs_map.weight.q.size() > 0) {
                            auto weight_q_block = chunk_maps[array][group].weight.q.block(map_start_row, map_start_col, map_size_row, map_size_col);
                            weight_q_block += jinc_block * (*tcdata.weight_q)(det);
                        }
                        if (obs_map.signal.u.size() > 0) {
                            auto signal_u_block = chunk_maps[array][group].signal.u.block(map_start_row, map_start_col, map_size_row, map_size_col);
                            signal_u_block += jinc_block * (*tcdata.signal_u)(i, det) * (*tcdata.weight_u)(det);
                        }
                        if (obs_map.weight.u.size() > 0) {
                            auto weight_u_block = chunk_maps[array][group].weight.u.block(map_start_row, map_start_col, map_size_row, map_size_col);
                            weight_u_block += jinc_block * (*tcdata.weight_u)(det);
                        }
                    }
                }
            }

            // populate noise maps
            if (noise_maps.n_noise_maps > 0) {
                if (noise_pix_x(i) >= 0 && noise_pix_x(i) < noise_maps.n_cols && noise_pix_y(i) >= 0 && noise_pix_y(i) < noise_maps.n_rows) {
                    // get bounds
                    auto [map_start_row, map_start_col, map_size_row, map_size_col, array_start_row, array_start_col] =
                        calc_bounds(noise_maps.n_rows, noise_maps.n_cols, jinc_array.rows(), jinc_array.cols(), noise_pix_y(i), noise_pix_x(i));

                    auto jinc_block = jinc_array.block(array_start_row, array_start_col, map_size_row, map_size_col);

                    // same for all noise maps
                    double signal = tcdata.signal(i, det) * tcdata.weight(det);
                    double signal_q, signal_u;

                    if (!noise_maps[array][group].noise.q.empty()) {
                        signal_q = (*tcdata.signal_q)(i, det) * (*tcdata.weight_q)(det);
                    }
                    if (!noise_maps[array][group].noise.q.empty()) {
                        signal_u = (*tcdata.signal_u)(i, det) * (*tcdata.weight_u)(det);
                    }
                    // loop through noise maps
                    for (int k = 0; k < noise_maps.n_noise_maps; ++k) {
                        int sign_value;

                        if (randomize_dets) {
                            sign_value = noise(k, det);
                        } else {
                            sign_value = noise(k);
                        }

                        auto noise_block = chunk_noise_maps[array][group].noise.i[k].block(map_start_row, map_start_col, map_size_row, map_size_col);
                        noise_block += sign_value * jinc_block * signal;

                        if (noise_maps.include_polarization && toltec.apt["loc"].data(det) != -1) {
                            if (!noise_maps[array][group].noise.q.empty()) {
                                noise_block = chunk_noise_maps[array][group].noise.q[k].block(map_start_row, map_start_col, map_size_row, map_size_col);
                                noise_block += sign_value * jinc_block * signal_q;
                            }
                            if (!noise_maps[array][group].noise.u.empty()) {
                                noise_block = chunk_noise_maps[array][group].noise.u[k].block(map_start_row, map_start_col, map_size_row, map_size_col);
                                noise_block += sign_value * jinc_block * signal_u;
                            }
                        }
                    }
                }
            }
        }
    }

    {
        // lock thread while adding into map
        std::scoped_lock<std::mutex> lock(*jinc_mutex);

        obs_maps += chunk_maps;

        if (noise_maps.n_noise_maps > 0) {
            noise_maps += chunk_noise_maps;
        }
    }
}
