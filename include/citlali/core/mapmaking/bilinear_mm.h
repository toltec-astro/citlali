# pragma once

#include <citlali/core/mapmaking/mm.h>

#include <mutex>

using namespace citlali::config::options;

// Bilinear Mapmaker
template <typename TCDataType>
class BilinearMapmaker : public PipelineComponent<TCDataType>, public Mapmaker {
public:

    template <typename ConfigType>
    BilinearMapmaker(Instrument& toltec_, Telescope& telescope_,
                     ObsMaps<>& om_, ObsMaps<>& cm_,
                     ObsMaps<MapKey, std::vector<ObsMatrix<MapKey>>>& nm_,
                     MapMode mm_, ConfigType& config)
        : Mapmaker(toltec_, telescope_, om_, cm_, nm_, mm_) {}

    auto get_bilinear_weights(double dx, double dy) {
        double w00 = (1 - dx) * (1 - dy);
        double w10 = dx * (1 - dy);
        double w01 = (1 - dx) * dy;
        double w11 = dx * dy;

        return std::make_tuple(w00, w10, w01, w11);
    }

    template <typename Derived>
    void accumulate(Eigen::DenseBase<Derived> &data, double value, int x0, int y0,
                    double w00, double w10, double w01, double w11) {
        data(y0, x0) += value * w00;
        data(y0, x0 + 1) += value * w10;
        data(y0 + 1, x0) += value * w01;
        data(y0 + 1, x0 + 1) += value * w11;
    }

    void init() override {}

    void process(TCDataType& tcdata) override {
        logger->info("bilinear mapmaker processing");

        ObsMaps<MapKey, ObsMatrix<>> sp_obs_maps;
        ObsMaps<MapKey, std::vector<ObsMatrix<>>> sp_noise_maps;

        bool run_obs_maps = get_map_mode(map_mode, MapMode::Obs) || get_map_mode(map_mode, MapMode::Both);
        bool run_noise_maps = (get_map_mode(map_mode, MapMode::Noise) || get_map_mode(map_mode, MapMode::Both)) && (!noise_maps.signal.empty());
        bool run_coverage = !obs_maps.coverage.empty();

        auto [n_pts, n_dets] = tcdata.dims();

        // if calibration beammapping (no coverage) clear apt flags so all maps are made
        if (run_coverage) {
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
        auto signs = get_noise(tcdata);

        for (int det = 0; det < n_dets; ++det) {
            // don't run detectors flagged for this chunk
            if (!tcdata.apt_flag(det)) {
                // get lookup indices
                auto [sig_i_index, sig_q_index, sig_u_index, kernel_i_index, coverage_i_index] = get_indices(det, run_kernel, run_coverage);
                // get detector pointing
                auto [xy, pix_x, pix_y, noise_pix_x, noise_pix_y] = get_pointing(tcdata, det, run_obs_maps, run_noise_maps);

                for (int i = 0; i < n_pts; ++i) {
                    // don't run flagged samples
                    if (tcdata.flag(i, det)) continue;
                    if (run_obs_maps) {
                        int x0 = static_cast<int>(pix_x(i));
                        int y0 = static_cast<int>(pix_y(i));

                        if (x0 < 0 || x0 + 1 >= obs_maps.wcs.naxis[0] || y0 < 0 || y0 + 1 >= obs_maps.wcs.naxis[1])
                            continue;

                        double dx = (xy.first(i) / pix_size_radians + obs_maps.wcs.naxis[0] / 2.0) - x0;
                        double dy = (xy.second(i) / pix_size_radians + obs_maps.wcs.naxis[1] / 2.0) - y0;

                        auto [w00, w10, w01, w11] = get_bilinear_weights(dx, dy);

                        accumulate(sp_obs_maps.signal[sig_i_index].data, tcdata.signal(i, det) * tcdata.weight(det), x0, y0, w00, w10, w01, w11);
                        accumulate(sp_obs_maps.weight[sig_i_index].data, tcdata.weight(det), x0, y0, w00, w10, w01, w11);

                        if (run_kernel) {
                            accumulate(sp_obs_maps.kernel[kernel_i_index].data, tcdata.kernel(i, det) * tcdata.weight(det), x0, y0, w00, w10, w01, w11);
                        }

                        if (run_coverage) {
                            accumulate(sp_obs_maps.coverage[coverage_i_index].data, 1.0 / tcdata.data_fs_hz, x0, y0, w00, w10, w01, w11);
                        }

                        if (run_polarization) {
                            accumulate(sp_obs_maps.signal[sig_q_index].data, tcdata.signal_q.value()(i, det) * tcdata.weight_q.value()(det),
                                       x0, y0, w00, w10, w01, w11);
                            accumulate(sp_obs_maps.signal[sig_u_index].data, tcdata.signal_u.value()(det) * tcdata.weight_u.value()(det),
                                       x0, y0, w00, w10, w01, w11);
                        }
                    }

                    if (run_noise_maps) {
                        int x0 = static_cast<int>(noise_pix_x(i));
                        int y0 = static_cast<int>(noise_pix_y(i));

                        if (x0 < 0 || x0 + 1  >= noise_maps.wcs.naxis[0] || y0 < 0 || y0 + 1 >= noise_maps.wcs.naxis[1])
                            continue;

                        double dx = (xy.first(i) / pix_size_radians + noise_maps.wcs.naxis[0] / 2.0) - x0;
                        double dy = (xy.second(i) / pix_size_radians + noise_maps.wcs.naxis[1] / 2.0) - y0;

                        auto [w00, w10, w01, w11] = get_bilinear_weights(dx, dy);

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
                            accumulate(sp_noise_maps.signal[sig_i_index][n].data, sign*signal, x0, y0, w00, w10, w01, w11);

                            if (run_polarization) {
                                accumulate(sp_noise_maps.signal[sig_q_index][n].data, sign*signal_q, x0, y0, w00, w10, w01, w11);
                                accumulate(sp_noise_maps.signal[sig_u_index][n].data, sign*signal_u, x0, y0, w00, w10, w01, w11);
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
