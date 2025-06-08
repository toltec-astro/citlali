# pragma once

class Mapmaker {
public:
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");
    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();

    Instrument& toltec;
    Telescope& telescope;
    ObsMaps<MapKey>& obs_maps, coadd_maps;
    ObsMaps<MapKey, std::vector<ObsMatrix<MapKey>>>& noise_maps;

    MapMode map_mode;

    Mapmaker(Instrument& toltec_, Telescope& telescope_, ObsMaps<>& om_, ObsMaps<>& cm_, ObsMaps<MapKey, std::vector<ObsMatrix<MapKey>>>& nm_,
                 MapMode mm_)
        : toltec(toltec_), telescope(telescope_), obs_maps(om_), coadd_maps(cm_), noise_maps(nm_), map_mode(mm_) {}

    template <typename TCDataType>
    auto get_noise(TCDataType& tcdata) {
        auto [n_pts, n_dets] = tcdata.dims();
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

        return signs;
    }

    auto get_indices(const int det, const bool run_kernel, bool run_coverage) {
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

        return std::make_tuple(
            sig_i_index,
            sig_q_index,
            sig_u_index,
            kernel_i_index,
            coverage_i_index
            );
    }

    template <typename TCDataType>
    auto get_pointing(const TCDataType& tcdata, const int det, const bool run_obs_maps, const bool run_noise_maps) {
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

        return std::make_tuple(
            xy,
            pix_x,
            pix_y,
            noise_pix_x,
            noise_pix_y
            );
    }
};
