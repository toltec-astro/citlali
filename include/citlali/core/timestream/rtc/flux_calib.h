# pragma once

// FluxCalib
template <typename TCDataType>
class FluxCalib : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    Instrument& toltec;
    Telescope& telescope;

    FluxCalib(Instrument& toltec_ref, Telescope& telescope_ref)
        : toltec(toltec_ref), telescope(telescope_ref) {}

    void init() {}
    void process(TCDataType& tcdata) override {
        logger->info("flux calibration processing");

        // map to row vector for array multiplication
        auto flxscale_map = Eigen::Map<Eigen::RowVectorXd>(toltec.apt["flxscale"].data.data(), toltec.apt["flxscale"].data.size());

        // multiply signal by flux scale
        tcdata.signal = tcdata.signal.array().rowwise() * flxscale_map.array();
    }
};
