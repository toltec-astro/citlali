#pragma once

// Downsample stage for TCData
template <typename TCDataType>
class Downsample : public PipelineComponent<TCDataType> {
public:
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    int factor;
    double downsampled_fs_hz;
    Instrument& toltec;
    Telescope& telescope;

    template <typename ConfigType>
    Downsample(Instrument& toltec_ref, Telescope& telescope_ref, ConfigType& config)
        : toltec(toltec_ref), telescope(telescope_ref) {

        config.get(factor, std::tuple{"timestream","raw_time_chunk","downsample","factor"});
        config.get(downsampled_fs_hz, std::tuple{"timestream","raw_time_chunk","downsample","downsampled_freq_Hz"});
    }

    void init() override {
        if (factor <= 0) {
            // need downsample frequency to be smaller than sample rate
            if (downsampled_fs_hz > toltec.data_fs_hz) {
                throw std::runtime_error(fmt::format("downsampled freq ({} Hz) must be less than sample rate ({} Hz)",
                                                     downsampled_fs_hz, toltec.data_fs_hz));
            }
            // downsample factor = (sample rate)/(downsampled freq)
            factor = std::floor(toltec.data_fs_hz / downsampled_fs_hz);
        }
        else {
            downsampled_fs_hz = toltec.data_fs_hz / factor;
        }
    }

    // process function to downsample signal and flag matrices/vectors
    void process(TCDataType& tcdata) override {
        logger->info("downsample processing");

        // downsample signal and flags
        downsample(tcdata.signal);
        downsample(tcdata.flag);

        // downsample kernel
        if (tcdata.kernel.size() > 0) {
            downsample(tcdata.kernel);
        }
        // downsample each telescope key
        for (auto& [key, value]: tcdata.tel_data) {
            downsample(value);
        }

        // downsample hwpr angle
        if (tcdata.hwpr_theta.size() > 0) {
            downsample(tcdata.hwpr_theta);
        }

        // update sampling rate
        tcdata.data_fs_hz = downsampled_fs_hz;
    }

    template <typename Derived>
    void downsample(Eigen::MatrixBase<Derived>& data) {
        // define types
        using Eigen::Dynamic;
        using Eigen::Map;
        using Eigen::Stride;

        // define the type of the matrix and stride map
        using EigenStrideMap = Map<Eigen::Matrix<typename Derived::Scalar, Dynamic, Dynamic>, 0, Stride<Dynamic, Dynamic>>;

        // calculate the new downsampled dimensions
        Eigen::Index downsampled_rows = (data.rows() + factor - 1) / factor;
        Eigen::Index downsampled_cols = data.cols();

        // create a temporary matrix to store the downsampled data
        Eigen::Matrix<typename Derived::Scalar, Dynamic, Dynamic> downsampled_data(downsampled_rows, downsampled_cols);

        // create a stride map over the original data
        EigenStrideMap stride_map(const_cast<typename Derived::Scalar*>(data.derived().data()),
                                  downsampled_rows,
                                  downsampled_cols,
                                  Stride<Dynamic, Dynamic>(data.outerStride(), data.innerStride() * factor));

        // copy the downsampled data into the temporary matrix
        downsampled_data = stride_map;

        // assign the downsampled data back to the original matrix
        data.derived() = downsampled_data;
    }
};
