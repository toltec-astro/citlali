#pragma once

#include <mutex>

std::tuple<int, int, int, int, int, int> find_overlap(
    int n_rows, int n_cols, int b_rows, int b_cols, int center_row, int center_col) {

    // calculate the start position
    int start_row = center_row - b_rows / 2;
    int start_col = center_col - b_cols / 2;

    // clamp the map start positions to the map boundaries
    start_row = std::max(0, start_row);
    start_col = std::max(0, start_col);

    // calculate the size for the map region
    int size_row = std::min(n_rows, start_row + b_rows) - start_row;
    int size_col = std::min(n_cols, start_col + b_cols) - start_col;

    // calculate the start position for the box (clamped to 0 if out of bounds)
    int box_start_row = std::max(0, -start_row);
    int box_start_col = std::max(0, -start_col);

    return std::make_tuple(start_row, start_col, size_row, size_col,
                           box_start_row, box_start_col);
}

// Jinc Mapmaker
template <typename TCDataType>
class JincMapmaker : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");
    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();

    Instrument& toltec;
    Telescope& telescope;
    ObsMaps<MapKey>& obs_maps, coadd_maps;
    ObsMaps<MapKey, std::vector<ObsMatrix<MapKey>>>& noise_maps;

    MapMode map_mode;

    std::map<int, std::vector<double>> jinc_shape;
    double radius_max;

    // map array index to jinc array
    std::map<int, Eigen::MatrixXd> array_to_jinc_map;

    template <typename ConfigType>
    JincMapmaker(Instrument& toltec_, Telescope& telescope_, ObsMaps<>& om_, ObsMaps<>& cm_, ObsMaps<MapKey, std::vector<ObsMatrix<MapKey>>>& nm_,
                  MapMode mm_, ConfigType& config)
        : toltec(toltec_), telescope(telescope_), obs_maps(om_), coadd_maps(cm_), noise_maps(nm_), map_mode(mm_) {

        for (const auto& [array_index, array_name] : toltec.array_index_to_name) {
            std::vector<double> jinc_shape_vec;
            config.get(jinc_shape_vec, std::tuple{"mapmaking","jinc_filter","shape_params",array_name});
            jinc_shape[array_index] = jinc_shape_vec;
        }

        config.get(radius_max, std::tuple{"mapmaking","jinc_filter","r_max"});

        if (config.missing_keys.empty() && config.invalid_keys.empty()) {
            allocate_jinc_matrix();
        }
    }

    double jinc_function(double radius, double radius_max, double a, double b, double c) {
        if (std::fabs(radius) < 1e-9)
            return 1.0;

        double jinc_0 = gsl_sf_bessel_J1(2.0 * pi * radius / a) / (pi * radius / a);
        double exp_func = std::exp(-std::pow(2.0 * radius / b, c));
        double jinc_1 = gsl_sf_bessel_J1(3.831706 * radius / radius_max) / (3.831706 * radius / radius_max);

        return jinc_0 * exp_func * jinc_1;
    }

    void allocate_jinc_matrix() {
        for (const auto& [array_index, wavelength_m] : toltec.array_index_to_wavelength) {
            double a = jinc_shape.at(array_index)[0];
            double b = jinc_shape.at(array_index)[1];
            double c = jinc_shape.at(array_index)[2];

            double wave_d = wavelength_m / telescope.lmt_diameter_m;
            int radius_max_pix = std::floor(radius_max * wave_d / (pix_size_arcsec * ASEC_TO_RAD));
            int n_pts = 2.0 * radius_max_pix + 1;
            Eigen::VectorXd pix_range = Eigen::VectorXd::LinSpaced(n_pts, -radius_max_pix, radius_max_pix);

            array_to_jinc_map[array_index].setZero(n_pts, n_pts);

            for (int i = 0; i < n_pts; ++i) {
                for (int j = 0; j < n_pts; ++j) {
                    double radius = (pix_size_arcsec * ASEC_TO_RAD) * std::sqrt(pow(pix_range(i), 2) + pow(pix_range(j), 2));
                    if (radius <= radius_max_pix) {
                        array_to_jinc_map.at(array_index)(i,j) = jinc_function(radius / wave_d, radius_max, a, b, c);
                    }
                }
            }
        }
    }

    void init() override {}

    void process(TCDataType& tcdata) override {
        logger->info("jinc mapmaker processing");

        ObsMaps<MapKey, ObsMatrix<>> sp_obs_maps;
        ObsMaps<MapKey, std::vector<ObsMatrix<>>> sp_noise_maps;

        bool run_obs_maps = get_map_mode(map_mode, MapMode::Obs) || get_map_mode(map_mode, MapMode::Both);
        bool run_noise_maps = (get_map_mode(map_mode, MapMode::Noise) || get_map_mode(map_mode, MapMode::Both)) && (noise_maps.signal.size() > 0);
        bool run_coverage = obs_maps.coverage.size() > 0;

        auto [n_pts, n_dets] = tcdata.dims();

        // if calibration beammapping (no coverage) clear apt flags so all maps are made
        if (obs_maps.coverage.empty()) {
            tcdata.apt_flag.setZero();
        }

        // allocate maps for current chunk
        for (const auto& [key, index] : obs_maps.signal_lookup) {
            sp_obs_maps.add(key, {obs_maps.wcs.naxis[1], obs_maps.wcs.naxis[0]}, true, run_kernel, run_coverage);

            if (run_noise_maps) {
                sp_noise_maps.add(key, {noise_maps.wcs.naxis[1], noise_maps.wcs.naxis[0]}, false, false, false);
            }
        }

        // random sign values for noise maps
        Eigen::MatrixXi signs;

        if (run_noise_maps) {
            if (randomize_dets) {
                signs.resize(n_noise_maps, n_dets);
            } else {
                signs.resize(n_noise_maps, 1);
            }
            tcdata.random_sign(signs);
        }

        for (int det = 0; det < n_dets; ++det) {
            // don't run detectors flagged for this chunk
            if (!tcdata.apt_flag(det)) {
                // get indices of maps
                int sig_i_index = obs_maps.signal_lookup.at(MapKey(toltec.apt["array"].data(det), toltec.apt[map_grouping].data(det), "I"));
                int sig_q_index, sig_u_index;

                if (run_polarization) {
                    sig_q_index = obs_maps.signal_lookup.at(MapKey(toltec.apt["array"].data(det), toltec.apt[map_grouping].data(det), "Q"));
                    sig_u_index = obs_maps.signal_lookup.at(MapKey(toltec.apt["array"].data(det), toltec.apt[map_grouping].data(det), "U"));
                }

                int kernel_i_index, coverage_i_index;

                if (run_kernel) {
                    kernel_i_index = obs_maps.kernel_lookup.at(MapKey(toltec.apt["array"].data(det), toltec.apt[map_grouping].data(det), "I"));
                }
                if (run_coverage) {
                    coverage_i_index = obs_maps.coverage_lookup.at(MapKey(toltec.apt["array"].data(det), toltec.apt[map_grouping].data(det), "I"));
                }

                // get detector pointing
                auto xy = calc_pointing(toltec.apt["x_t"].data(det), toltec.apt["y_t"].data(det), tcdata.tel_data, telescope.pixel_axes);

                // pixels for obs maps
                Eigen::VectorXI pix_x, pix_y;
                if (run_obs_maps) {
                    pix_x = (xy.first.array() / pix_size_radians + obs_maps.wcs.naxis[0] / 2.0).template cast<Eigen::Index>();
                    pix_y = (xy.second.array() / pix_size_radians + obs_maps.wcs.naxis[1] / 2.0).template cast<Eigen::Index>();
                }

                // pixels for noise maps
                Eigen::VectorXI noise_pix_x, noise_pix_y;
                if (run_noise_maps) {
                    noise_pix_x = (xy.first.array() / pix_size_radians + noise_maps.wcs.naxis[0] / 2.0).template cast<Eigen::Index>();
                    noise_pix_y = (xy.second.array() / pix_size_radians + noise_maps.wcs.naxis[1] / 2.0).template cast<Eigen::Index>();
                }

                // jinc filter for current array
                const auto& jinc_array = array_to_jinc_map.at(toltec.apt["array"].data(det));

                for (int i = 0; i < n_pts; ++i) {
                    // don't run flagged samples
                    if (tcdata.flag(i, det)) continue;

                    if (run_obs_maps) {
                        if (pix_x(i) >= 0 && pix_x(i) < obs_maps.wcs.naxis[0] && pix_y(i) >= 0 && pix_y(i) < obs_maps.wcs.naxis[1]) {
                            // get bounds
                            auto [start_row, start_col, n_rows, n_cols, array_start_row, array_start_col] =
                                find_overlap(obs_maps.wcs.naxis[1], obs_maps.wcs.naxis[0], jinc_array.rows(), jinc_array.cols(), pix_y(i), pix_x(i));

                            // blocks to relevant regions of maps and jinc array
                            auto jinc_block = jinc_array.block(array_start_row, array_start_col, n_rows, n_cols);

                            auto signal_block = sp_obs_maps.signal[sig_i_index].data.block(start_row, start_col, n_rows, n_cols);
                            auto weight_block = sp_obs_maps.weight[sig_i_index].data.block(start_row, start_col, n_rows, n_cols);

                            signal_block += jinc_block * tcdata.signal(i, det) * tcdata.weight(det);
                            weight_block += jinc_block * tcdata.weight(det);

                            if (run_kernel) {
                                auto kernel_block = sp_obs_maps.kernel[kernel_i_index].data.block(start_row, start_col, n_rows, n_cols);
                                kernel_block += jinc_block * tcdata.kernel(i, det) * tcdata.weight(det);
                            }

                            if (run_coverage) {
                                auto coverage_block = sp_obs_maps.coverage[coverage_i_index].data.block(start_row, start_col, n_rows, n_cols);
                                coverage_block += jinc_block * 1.0 / tcdata.data_fs_hz;
                            }

                            if (run_polarization) {
                                auto signal_q_block = sp_obs_maps.signal[sig_q_index].data.block(start_row, start_col, n_rows, n_cols);
                                auto weight_q_block = sp_obs_maps.weight[sig_q_index].data.block(start_row, start_col, n_rows, n_cols);

                                signal_q_block += jinc_block * tcdata.signal_q.value()(i, det) * tcdata.weight_q.value()(det);
                                weight_q_block += jinc_block * tcdata.weight_q.value()(det);

                                auto signal_u_block = sp_obs_maps.signal[sig_u_index].data.block(start_row, start_col, n_rows, n_cols);
                                auto weight_u_block = sp_obs_maps.weight[sig_u_index].data.block(start_row, start_col, n_rows, n_cols);

                                signal_u_block += jinc_block * tcdata.signal_u.value()(i, det) * tcdata.weight_u.value()(det);
                                weight_u_block += jinc_block * tcdata.weight_u.value()(det);
                            }
                        }
                    }

                    if (run_noise_maps) {
                        if (noise_pix_x(i) >= 0 && noise_pix_x(i) < noise_maps.wcs.naxis[0] && noise_pix_y(i) >= 0 && noise_pix_y(i) < noise_maps.wcs.naxis[1]) {
                            // get bounds
                            auto [start_row, start_col, n_rows, n_cols, array_start_row, array_start_col] =
                                find_overlap(noise_maps.wcs.naxis[1], noise_maps.wcs.naxis[0], jinc_array.rows(), jinc_array.cols(), pix_y(i), pix_x(i));

                            // blocks to relevant regions of maps and jinc array
                            auto jinc_block = jinc_array.block(array_start_row, array_start_col, n_rows, n_cols);

                            double signal = tcdata.signal(i,det) * tcdata.weight(det);
                            double signal_q, signal_u;

                            if (run_polarization) {
                                signal_q = (*tcdata.signal_q)(i, det) * (*tcdata.weight_q)(det);
                                signal_u = (*tcdata.signal_u)(i, det) * (*tcdata.weight_u)(det);
                            }

                            for (int n = 0; n < n_noise_maps; ++n) {
                                int sign;
                                if (randomize_dets) {
                                    sign = signs(n, det);
                                } else {
                                    sign = signs(n);
                                }
                                auto noise_block = sp_noise_maps.signal[sig_i_index][n].data.block(start_row, start_col, n_rows, n_cols);
                                noise_block += jinc_block * signal;

                                if (run_polarization) {
                                    auto noise_q_block = sp_noise_maps.signal[sig_q_index][n].data.block(start_row, start_col, n_rows, n_cols);
                                    noise_q_block += jinc_block * signal_q;

                                    auto noise_u_block = sp_noise_maps.signal[sig_u_index][n].data.block(start_row, start_col, n_rows, n_cols);
                                    noise_u_block += jinc_block * signal_u;
                                }
                            }
                        }
                    }
                }
            }
        }
        {
            std::scoped_lock<std::mutex> lock(*mutex);
            obs_maps += sp_obs_maps;
            if (run_noise_maps) {
                noise_maps += sp_noise_maps;
            }
        }
    }
};
