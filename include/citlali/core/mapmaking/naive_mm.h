#pragma once

#include <mutex>

using namespace citlali::config::options;

// Naive Mapmaker
template <typename TCDataType>
class NaiveMapmaker : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");
    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();

    Instrument& toltec;
    Telescope& telescope;
    ObsMaps<MapKey>& obs_maps, coadd_maps;
    ObsMaps<MapKey, std::vector<ObsMatrix<MapKey>>>& noise_maps;

    MapMode map_mode;

    template <typename ConfigType>
    NaiveMapmaker(Instrument& toltec_, Telescope& telescope_, ObsMaps<>& om_, ObsMaps<>& cm_, ObsMaps<MapKey, std::vector<ObsMatrix<MapKey>>>& nm_,
                  MapMode mm_, ConfigType& config)
        : toltec(toltec_), telescope(telescope_), obs_maps(om_), coadd_maps(cm_), noise_maps(nm_), map_mode(mm_) {}

    void init() override {}

    void process(TCDataType& tcdata) override {
        logger->info("naive mapmaker processing");

        ObsMaps<MapKey, ObsSparse<>> sp_obs_maps;
        ObsMaps<MapKey, std::vector<ObsSparse<>>> sp_noise_maps;

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
            sp_obs_maps.add(key, {static_cast<int>(n_pts), static_cast<int>(n_dets)}, true, run_kernel, run_coverage);

            if (run_noise_maps) {
                sp_noise_maps.add(key, {static_cast<int>(n_pts), static_cast<int>(n_dets), n_noise_maps}, false, false, false);
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

        // auto [in, out] = citlali::utils::threads::get_grppi_vectors(n_dets);
        // auto exec_mode = citlali::utils::threads::get_chunk_remainder_exec_mode();

        // if (map_grouping != "uid") {
        //     exec_mode = citlali::utils::threads::get_seq_exec_mode();
        // }

        // grppi::map(exec_mode, in, out, [&](int det) {
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

                for (int i = 0; i < n_pts; ++i) {
                    // don't run flagged samples
                    if (tcdata.flag(i, det)) continue;

                    if (run_obs_maps) {
                        if (pix_x(i) >= 0 && pix_x(i) < obs_maps.wcs.naxis[0] && pix_y(i) >= 0 && pix_y(i) < obs_maps.wcs.naxis[1]) {
                            sp_obs_maps.signal[sig_i_index](pix_y(i), pix_x(i), tcdata.signal(i, det) * tcdata.weight(det));
                            sp_obs_maps.weight[sig_i_index](pix_y(i), pix_x(i), tcdata.weight(det));

                            if (run_kernel) {
                                sp_obs_maps.kernel[kernel_i_index](pix_y(i), pix_x(i), tcdata.kernel(i, det) * tcdata.weight(det));
                            }

                            if (run_coverage) {
                                sp_obs_maps.coverage[coverage_i_index](pix_y(i), pix_x(i), 1.0 / tcdata.data_fs_hz);
                            }

                            if (run_polarization) {
                                sp_obs_maps.signal[sig_q_index](pix_y(i), pix_x(i), tcdata.signal_q.value()(i, det) * tcdata.weight_q.value()(det));
                                sp_obs_maps.weight[sig_q_index](pix_y(i), pix_x(i), tcdata.weight_q.value()(det));
                                sp_obs_maps.signal[sig_u_index](pix_y(i), pix_x(i), tcdata.signal_u.value()(i, det) * tcdata.weight_u.value()(det));
                                sp_obs_maps.weight[sig_u_index](pix_y(i), pix_x(i), tcdata.weight_u.value()(det));
                            }
                        }
                    }

                    if (run_noise_maps) {
                        if (noise_pix_x(i) >= 0 && noise_pix_x(i) < noise_maps.wcs.naxis[0] && noise_pix_y(i) >= 0 && noise_pix_y(i) < noise_maps.wcs.naxis[1]) {
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
                                sp_noise_maps.signal[sig_i_index][n](noise_pix_y(i), noise_pix_x(i), sign*signal);

                                if (run_polarization) {
                                    sp_noise_maps.signal[sig_q_index][n](noise_pix_y(i), noise_pix_x(i), sign*signal_q);
                                    sp_noise_maps.signal[sig_u_index][n](noise_pix_y(i), noise_pix_x(i), sign*signal_u);
                                }
                            }
                        }
                    }
                }
            }
        //     return 0;
        // });
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
