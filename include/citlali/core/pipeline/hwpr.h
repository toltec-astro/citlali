#pragma once

class Hwpr {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    bool run_hwpr;

    Eigen::VectorXd hwpr_theta, hwpr_uts;
    Eigen::MatrixXi hwpr_ts;

    double hwpr_fpga_freq;

    void load_hwpr(const std::string&, bool);
    Eigen::VectorXd calc_time_vector(double);
};

void Hwpr::load_hwpr(const std::string& filepath, bool sim_obs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    try {
        NcFile fo(filepath, NcFile::read, NcFile::classic);
        auto vars = fo.getVars();

        // determine the correct HWPR installation variable based on sim_obs flag
        const std::string hwpr_install_var = sim_obs ? "Header.Hwp.Installed" : "Header.Toltec.HwpInstalled";

        // check if HWPR is enabled
        vars.find(hwpr_install_var)->second.getVar(&run_hwpr);

        // early return if HWPR is not running
        if (!run_hwpr) return;

        // get HWP signal
        const auto& hwpr_data_var = fo.getVar("Data.Hwp.");
        Eigen::Index n_pts = hwpr_data_var.getDim(0).getSize();
        hwpr_theta.resize(n_pts);
        hwpr_data_var.getVar(hwpr_theta.data());

        if (!sim_obs) {
            // get HWP timing data
            hwpr_ts.resize(n_pts, 6);
            vars.find("Data.Hwp.Ts")->second.getVar(hwpr_ts.data());

            hwpr_ts.transposeInPlace();

            // get HWPR uts time and FPGA frequency
            Eigen::Index uts_n_pts = vars.find("Data.Hwp.Uts")->second.getDim(0).getSize();

            hwpr_uts.resize(uts_n_pts);
            vars.find("Data.Hwp.Uts")->second.getVar(hwpr_uts.data());

            vars.find("Header.Toltec.FpgaFreq")->second.getVar(&hwpr_fpga_freq);
        }

    } catch (const NcException &e) {
        throw std::runtime_error(fmt::format("Failed to load data from netCDF file {} with error", filepath, e.what()));
    }
}

Eigen::VectorXd Hwpr::calc_time_vector(double offset) {
    // cast once to double
    Eigen::MatrixXd hwpr_ts_double = hwpr_ts.cast<double>();

    // extract columns with descriptive names
    Eigen::VectorXd sec = hwpr_ts_double.col(0);        // ClockTime (sec)
    Eigen::VectorXd nsec = hwpr_ts_double.col(5);       // ClockTimeNanoSec (nsec)
    Eigen::VectorXd pps = hwpr_ts_double.col(1);        // PpsCount (pps ticks)
    Eigen::VectorXd msec = hwpr_ts_double.col(2) / hwpr_fpga_freq;  // ClockCount (clock ticks) to seconds
    //Eigen::VectorXd count = hwpr_ts_double.col(3);      // PacketCount (packet ticks)
    Eigen::VectorXd pps_msec = hwpr_ts_double.col(4) / hwpr_fpga_freq; // PpsTime (clock ticks) to seconds

    // determine start time with empirical offset
    double start_time_dbl = sec[0] + nsec[0] * 1e-9;
    int start_time = int(start_time_dbl - 0.5);
    start_time_dbl = start_time;

    // calculate clock count difference (dt)
    Eigen::VectorXd dt = msec - pps_msec;

    // handle overflow due to int32, using Eigen array logic
    dt = (dt.array() < 0).select(msec.array() - pps_msec.array() + (pow(2.0, 32) - 1) / hwpr_fpga_freq, msec - pps_msec);

    // build the time vector for the current network
    Eigen::VectorXd hwpr_time = start_time_dbl + pps.array() + dt.array() + offset;

    return hwpr_time;
}
