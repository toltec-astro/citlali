#pragma once

#include <citlali/core/utils/beam.h>

// Kernel
template <typename TCDataType>
class Kernel : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    std::string type, filepath;
    double fwhm_radians;
    double fwhm_limit_factor_ = 3.0;
    Instrument& toltec;
    Telescope& telescope;
    Beam beam;

    template <typename ConfigType>
    Kernel(Instrument& toltec_ref, Telescope& telescope_ref, ConfigType& config)
        : toltec(toltec_ref), telescope(telescope_ref) {

        config.get(filepath, std::tuple{"timestream","raw_time_chunk","kernel","filepath"});
        config.get(type, std::tuple{"timestream","raw_time_chunk","kernel","type"});
        config.get(fwhm_radians, std::tuple{"timestream","raw_time_chunk","kernel","fwhm_arcsec"});

        // convert FWHM to radians
        fwhm_radians *= ASEC_TO_RAD;
    }

    void init() {}

    void process(TCDataType& tcdata) override {
        logger->info("kernel processing");
        if (type == "gaussian") {
            add_gaussian_beam_to_time_chunk(tcdata);
        } else if (type == "airy") {
            add_airy_beam_to_time_chunk(tcdata);
        } else if (type == "elliptical_gaussian") {
            add_elliptical_gaussian_beam_to_time_chunk(tcdata);
        }
    }

    void add_gaussian_beam_to_time_chunk(TCDataType&);
    void add_airy_beam_to_time_chunk(TCDataType&);
    void add_elliptical_gaussian_beam_to_time_chunk(TCDataType&);
};

// populate kernel for Gaussian beam
template <typename TCDataType>
void Kernel<TCDataType>::add_gaussian_beam_to_time_chunk(TCDataType& tcdata) {
    int n_dets = tcdata.n_dets();
    int n_pts = tcdata.n_pts();
    tcdata.kernel.setZero(n_pts, n_dets);

    for (int det = 0; det < n_dets; ++det) {
        if (tcdata.apt_flag(det)) continue;

        auto xy = telescope.calc_pointing(toltec.apt["x_t"].data(det), toltec.apt["y_t"].data(det), tcdata.tel_data);

        Eigen::VectorXd beam_params(6);
        for (int i = 0; i < n_pts; ++i) {
            double fwhm = (fwhm_radians > 0) ? fwhm_radians : ASEC_TO_RAD * (toltec.apt["a_fwhm"].data(det) + toltec.apt["b_fwhm"].data(det)) / 2;
            beam_params << fwhm * FWHM_TO_STD, fwhm * FWHM_TO_STD, 0, 0, 0, 1;
            double beam_value = beam.calculate_gaussian(xy.first(i), xy.second(i), beam_params, fwhm_limit_factor_ * fwhm);
            tcdata.kernel(i, det) = beam_value;
        }
    }
}

// populate kernel for Airy beam
template <typename TCDataType>
void Kernel<TCDataType>::add_airy_beam_to_time_chunk(TCDataType& tcdata) {
    int n_dets = tcdata.n_dets();
    int n_pts = tcdata.n_pts();
    tcdata.kernel.setZero(n_pts, n_dets);

    for (int det = 0; det < n_dets; ++det) {
        if (tcdata.apt_flag(det)) continue;

        auto xy = telescope.calc_pointing(toltec.apt["x_t"].data(det), toltec.apt["y_t"].data(det), tcdata.tel_data);

        Eigen::VectorXd beam_params(4);
        for (int i = 0; i < n_pts; ++i) {
            double fwhm = (fwhm_radians > 0) ? fwhm_radians : ASEC_TO_RAD * (toltec.apt["a_fwhm"].data(det) + toltec.apt["b_fwhm"].data(det)) / 2;
            beam_params << fwhm, 0, 0, 1;
            double beam_value = beam.calculate_airy(xy.first(i), xy.second(i), beam_params, fwhm_limit_factor_ * fwhm);
            tcdata.kernel(i, det) = beam_value;
        }
    }
}

// populate kernel for Elliptical Gaussian beam
template <typename TCDataType>
void Kernel<TCDataType>::add_elliptical_gaussian_beam_to_time_chunk(TCDataType& tcdata) {
    int n_dets = tcdata.n_dets();
    int n_pts = tcdata.n_pts();
    tcdata.kernel.setZero(n_pts, n_dets);

    for (int det = 0; det < n_dets; ++det) {
        if (tcdata.apt_flag(det)) continue;

        auto xy = telescope.calc_pointing(toltec.apt["x_t"].data(det), toltec.apt["y_t"].data(det), tcdata.tel_data);

        Eigen::VectorXd beam_params(6);
        for (int i = 0; i < n_pts; ++i) {

            double fwhm = (fwhm_radians > 0) ? fwhm_radians : ASEC_TO_RAD * (toltec.apt["a_fwhm"].data(det) + toltec.apt["b_fwhm"].data(det)) / 2;

            beam_params << toltec.apt["a_fwhm"].data(det) * ASEC_TO_RAD * FWHM_TO_STD,
                toltec.apt["b_fwhm"].data(det) * ASEC_TO_RAD * FWHM_TO_STD,
                toltec.apt["angle"].data(det),  0, 0, 1;

            double beam_value = beam.calculate_gaussian(xy.first(i), xy.second(i), beam_params, fwhm_limit_factor_ * fwhm);
            tcdata.kernel(i, det) = beam_value;
        }
    }
}
