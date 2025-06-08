# pragma once

// TodOutput
template <typename TCDataType>
class TodOutput : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    TodOutput(std::string type_, Instrument& toltec_, Telescope& telescope_)
        : type(type_), toltec(toltec_), telescope(telescope_) {}

    void init() {
        // create netcdf file
        /*netCDF::NcFile fo(filepath, netCDF::NcFile::replace);

        netCDF::NcDim tod_type_dim = fo.addDim("tod_type_dim", 1);
        netCDF::NcVar tod_type_var = fo.addVar("tod_type", netCDF::ncString, tod_type_dim);
        tod_type_var.putVar({0}, type);

        // add telescope parameters
        for (auto const& x: telescope.data) {
            netCDF::NcVar var = fo.addVar(x.first, netCDF::ncDouble, n_pts_dim);
            var.setChunking(chunk_mode, chunk_size);
        }*/
    }
    void process(TCDataType& tcdata) override {
        logger->info("tod output processing");
    }

private:
    Instrument& toltec;
    Telescope& telescope;

    std::string type, filepath;
};
