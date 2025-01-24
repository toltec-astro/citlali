# pragma once

#include <mutex>

// Naive Mapmaker
template <typename TCDataType>
class NaiveMapmaker : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    DataMapsContainer& obs_maps, coadd_maps;
    NoiseMapsContainer& noise_maps;
    Instrument& toltec;
    Telescope& telescope;

    bool randomize_dets;
    bool run_noise;

    MapMode map_type;

    // for adding thread local map values to map containers
    std::unique_ptr<std::mutex> naive_mutex = std::make_unique<std::mutex>();

    template <typename ConfigType>
    NaiveMapmaker(MapMode _map_type, Instrument& toltec_ref, Telescope& telescope_ref, DataMapsContainer& obs_map_ref,
                  DataMapsContainer& coadd_map_ref, NoiseMapsContainer& noise_map_ref, ConfigType& config)
        : map_type(_map_type), toltec(toltec_ref), telescope(telescope_ref), obs_maps(obs_map_ref), coadd_maps(coadd_map_ref), noise_maps(noise_map_ref) {

        config.get(randomize_dets, std::tuple{"noise_maps","randomize_dets"});
    }

    void init() {}
    void add_chunk_to_map(TCDataType&);

    void process(TCDataType& tcdata) override {
        logger->info("naive mapmaker processing");
        add_chunk_to_map(tcdata);
    }
};

// populate map
template <typename TCDataType>
void NaiveMapmaker<TCDataType>::add_chunk_to_map(TCDataType& tcdata) {
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

        // which map detector belongs to
        auto& obs_map = obs_maps[array][group];

        // get detector pointing
        auto xy = telescope.calc_pointing(toltec.apt["x_t"].data(det), toltec.apt["y_t"].data(det), tcdata.tel_data);

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

            // if pixel in map
            if (map_type == MapMode::Obs || map_type == MapMode::Both) {
                if (pix_x(i) >= 0 && pix_x(i) < obs_maps.n_cols && pix_y(i) >= 0 && pix_y(i) < obs_maps.n_rows) {
                    chunk_maps[array][group].signal.i(pix_y(i), pix_x(i)) += tcdata.signal(i, det) * tcdata.weight(det);
                    chunk_maps[array][group].weight.i(pix_y(i), pix_x(i)) += tcdata.weight(det);

                    if (obs_map.kernel.i.size() > 0) {
                        chunk_maps[array][group].kernel.i(pix_y(i), pix_x(i)) += tcdata.kernel(i, det) * tcdata.weight(det);
                    }
                    if (obs_map.coverage.i.size() > 0) {
                        chunk_maps[array][group].coverage.i(pix_y(i), pix_x(i)) += 1.0 / tcdata.data_fs_hz;
                    }

                    // only add to stokes maps if loc != -1
                    if (toltec.apt["loc"].data(det) != -1) {
                        if (obs_map.signal.q.size() > 0) {
                            chunk_maps[array][group].signal.q(pix_y(i), pix_x(i)) += (*tcdata.signal_q)(i, det) * (*tcdata.weight_q)(det);
                        }
                        if (obs_map.signal.u.size() > 0) {
                            chunk_maps[array][group].signal.u(pix_y(i), pix_x(i)) += (*tcdata.signal_u)(i, det) * (*tcdata.weight_u)(det);
                        }
                        if (obs_map.weight.q.size() > 0) {
                            chunk_maps[array][group].weight.q(pix_y(i), pix_x(i)) += (*tcdata.weight_q)(det);
                        }
                        if (obs_map.weight.u.size() > 0) {
                            chunk_maps[array][group].weight.u(pix_y(i), pix_x(i)) += (*tcdata.weight_u)(det);
                        }
                    }
                }
            }

            // populate noise maps
            if (noise_maps.n_noise_maps > 0) {
                if (noise_pix_x(i) >= 0 && noise_pix_x(i) < noise_maps.n_cols && noise_pix_y(i) >= 0 && noise_pix_y(i) < noise_maps.n_rows) {
                    // same for all noise maps
                    double signal = tcdata.signal(i, det) * tcdata.weight(det);
                    // loop through noise maps
                    for (int k = 0; k < noise_maps.n_noise_maps; ++k) {
                        int sign_value;

                        if (randomize_dets) {
                            sign_value = noise(k, det);
                        } else {
                            sign_value = noise(k);
                        }

                        chunk_noise_maps[array][group].noise.i[k](noise_pix_y(i), noise_pix_x(i)) += sign_value * signal;
                    }
                }
            }
        }
    }

    {
        // lock thread while adding into map
        std::scoped_lock<std::mutex> lock(*naive_mutex);

        obs_maps += chunk_maps;

        if (noise_maps.n_noise_maps > 0) {
            noise_maps += chunk_noise_maps;
        }
    }
}
