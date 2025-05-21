# pragma once

// Demodulate
template <typename TCDataType>
class Demodulate : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    Instrument& toltec;
    Telescope& telescope;

    int filter_order;
    double gibbs_factor;
    double low_cutoff_Hz;
    double high_cutoff_Hz;
    Eigen::VectorXd filter;

    template <typename ConfigType>
    Demodulate(Instrument& toltec, Telescope& telescope, ConfigType& config)
        : toltec(toltec), telescope(telescope) {

        config.get(gibbs_factor, std::tuple{"timestream","polarimetry","filter","a_gibbs"});
        config.get(low_cutoff_Hz, std::tuple{"timestream","polarimetry","filter","freq_low_Hz"});
        config.get(high_cutoff_Hz, std::tuple{"timestream","polarimetry","filter","freq_high_Hz"});
        config.get(filter_order, std::tuple{"timestream","polarimetry","filter","n_terms"});
    }

    void init() override {
        filter = create_kaiser_filter(toltec.data_fs_hz, filter_order, gibbs_factor, low_cutoff_Hz, high_cutoff_Hz);
    }

    void process(TCDataType& tcdata) override {
        logger->info("demodulate processing");

        int n_dets = tcdata.n_dets();
        int n_pts = tcdata.n_pts();
        bool run_hwpr = tcdata.hwpr_theta.size() > 0;

        // initialize signal q and u timestreams
        if (!tcdata.signal_q.has_value()) {
            tcdata.signal_q.emplace(n_pts, n_dets);
        }
        if (!tcdata.signal_u.has_value()) {
            tcdata.signal_u.emplace(n_pts, n_dets);
        }

        // parallactic angle + boresight elevation angle
        auto base_angle = tcdata.tel_data.at("ActParAng") + tcdata.tel_data.at("TelElAct");
        // holds individual detector total angle
        Eigen::VectorXd det_angle(n_pts);

        for (int det = 0; det < n_dets; ++det) {
            if (tcdata.apt_flag(det)) continue;

            // get detector altaz
            auto xy = calc_pointing(toltec.apt["x_t"].data(det), toltec.apt["y_t"].data(det), tcdata.tel_data, "altaz");

            // total angle for detector (PA + El_boresight + El_det + fg_angle + array_angle
            det_angle = base_angle.array() + xy.first.array() + toltec.fg_to_detector_angle[toltec.apt["fg"].data(det)]
                        + toltec.array_index_to_install_angle[toltec.apt["array"].data(det)];

            if (run_hwpr) {
                det_angle = 4 * tcdata.hwpr_theta - 2 * det_angle;
            } else {
                det_angle = 2 * det_angle;
            }

            if (tcdata.signal_q) {
                (*tcdata.signal_q).col(det) = cos(det_angle.array()) * tcdata.signal.col(det).array();
            }
            if (tcdata.signal_u) {
                (*tcdata.signal_u).col(det) = sin(det_angle.array()) * tcdata.signal.col(det).array();
            }
        }

        // convolve with lowpass filter
        convolve_filter((*tcdata.signal_q), filter);
        convolve_filter((*tcdata.signal_u), filter);

        tcdata.shrink(filter_order, n_pts - 2*filter_order);
    }
};
