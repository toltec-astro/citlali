#pragma once

// Extinction
template <typename TCDataType>
class Extinction : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    std::map<std::string, double> transmission_225_80_deg = {
        {"25",0.9500275}, {"50",0.9142065},
        {"75",0.8515054}, {"95",0.7337698},};

    std::string extinction_model;

    Instrument& toltec;
    Telescope& telescope;

    Extinction(Instrument& toltec_ref, Telescope& telescope_ref)
        : toltec(toltec_ref), telescope(telescope_ref) {}

    void init() {
        // precompute constants
        const double zenith_angle_rad = 80.0 * DEG_TO_RAD;
        const double cos_zenith = cos(pi / 2 - zenith_angle_rad);
        const double sec_zenith = 1.0 / cos_zenith;

        const double airmass = sec_zenith * (1.0 - 0.0012 * (pow(sec_zenith, 2) - 1.0));

        // get tau from the models at 80 degrees
        Eigen::VectorXd tau_225_80_deg_model(transmission_225_80_deg.size());

        int i = 0;
        for (const auto& [key, val] : transmission_225_80_deg) {
            tau_225_80_deg_model(i) = -log(val/airmass);
            i++;
        }

        // set initial extinction model
        extinction_model = "25";

        // loop through and find the model where calculated tau < telescope tau
        i = 0;
        for (const auto& [key, val] : transmission_225_80_deg) {
            if (tau_225_80_deg_model(i) <= telescope.header["Radiometer.Tau"](0)) {
                extinction_model = key;
            }
            i++;
        }
    }

    void process(TCDataType& tcdata) override {
        logger->info("extinction processing");

        auto cos_elev = cos(pi/2 - tcdata.tel_data.at("TelElAct").array());
        auto sec_elev = 1. / cos_elev;
        auto airmass = sec_elev * (1. - 0.0012 * (pow(sec_elev.array(), 2) - 1.));

        std::map<int, Eigen::VectorXd> tau_toltec;

        // get transmission in toltec bands for telescope boresight
        for (const auto& [index, name] : toltec.array_index_to_name) {
            auto transmission_toltec = polynomial_model(get_coefficients(name), tcdata.tel_data.at("TelElAct"));
            tau_toltec[index] = -(transmission_toltec.array().log()) / airmass.array();
        }

        for (int det = 0; det < tcdata.n_dets(); ++det) {
            // inverse extinction (exp(tau)) in current band for each elevation
            auto inv_extinction = (tau_toltec[toltec.apt["array"].data(det)].array()).exp();

            // multiply detector data points by the extinction
            tcdata.signal.col(det) = tcdata.signal.col(det).array() * inv_extinction;

            // fcf for approximate weights
            tcdata.fcf(det) *= inv_extinction.mean();
        }
    }

    Eigen::VectorXd get_coefficients(const std::string& model_name) {
        Eigen::VectorXd coeffs(7);

        if (model_name == "a1100") {
            if (extinction_model == "25") {
                coeffs << -0.12008024, 0.72422015, -1.81734478, 2.45313012, -1.92159695, 0.86918801, 0.78604295;
            } else if (extinction_model == "50") {
                coeffs << -0.18770822, 1.13390437, -2.85173457, 3.8617083, -3.03996805, 1.38624303, 0.65300169;
            } else if (extinction_model == "75") {
                coeffs << -0.28189529, 1.70842347, -4.31606883, 5.88248549, -4.67702093, 2.16747228, 0.4393435;
            } else if (extinction_model == "95") {
                coeffs << -1.21882233, 6.67068453, -14.96466875, 17.78045563, -12.10288687, 4.76050807, -0.06765066;
            }
        }  else if (model_name == "a1400") {
            if (extinction_model == "25") {
                coeffs << 0.02619509, -0.15757661, 0.39400473, -0.52912696, 0.411213, -0.18360141, 1.04398466;
            } else if (extinction_model == "50") {
                coeffs << 0.04292884, -0.25817762, 0.64533115, -0.86622214, 0.67267823, -0.29996916, 1.07167603;
            } else if (extinction_model == "75") {
                coeffs << 0.07581154, -0.45574885, 1.13852458, -1.52697451, 1.18425865, -0.5269455, 1.12531753;
            } else if (extinction_model == "95") {
                coeffs << 0.76090502, -4.05867663, 8.78487281, -9.90872343, 6.2198602, -2.13790165, 1.3668983;
            }
        } else if (model_name == "a2000") {
            if (extinction_model == "25") {
                coeffs << 0.16726241, -1.00436302, 2.50507317, -3.35219659, 2.59080373, -1.14622096, 1.26931683;
            } else if (extinction_model == "50") {
                coeffs << 0.35178447, -2.10859714, 5.24620825, -6.9952531, 5.37645792, -2.35675076, 1.54286813;
            } else if (extinction_model == "75") {
                coeffs << 0.79869908, -4.77265095, 11.82393401, -15.67007557, 11.93052031, -5.14788907, 2.14595898;
            } else if (extinction_model == "95") {
                coeffs << 16.0063036, -84.30325144, 179.28096414, -197.05751682, 118.73627425, -37.99279818, 6.55457576;
            }
        }

        return coeffs;
    }
};
