#pragma once

bool is_point_in_box(double a, double b, double x, double y, double theta) {
    // rotate point (a, b) by -theta
    double xp = a * std::cos(theta) + b * std::sin(theta);
    double yp = -a * std::sin(theta) + b * std::cos(theta);

    // Check horizontal bounds
    if (-x / 2 <= xp && xp <= x / 2) {
        // Check vertical bounds
        if (-y / 2 <= yp && yp <= y / 2) {
            return true;
        }
    }
    return false;
}

template <typename VarType>
void read_nc_var_to_string(VarType &var, std::size_t buffer_size, std::string &result) {
    // buffer to store the variable data
    char buffer[buffer_size];

    // get the variable and read it into the buffer
    var.getVar(&buffer);

    // ensure null termination
    buffer[buffer_size - 1] = '\0';

    // convert to string
    result = std::string(buffer);

    // remove trailing spaces
    result.erase(std::remove(result.begin(), result.end(), ' '), result.end());
}

// function to calculate the gnomonic projection for vectors
template <typename Derived>
void gnomonic_projection(const Eigen::DenseBase<Derived> &l, const Eigen::DenseBase<Derived> &b,
                         double l0, double b0, Eigen::DenseBase<Derived> &x, Eigen::DenseBase<Derived> &y) {

    // precompute cosines and sines
    Eigen::VectorXd cos_b = b.derived().array().cos();
    Eigen::VectorXd sin_b = b.derived().array().sin();
    double cos_b0 = std::cos(b0);
    double sin_b0 = std::sin(b0);

    // calculate angular distance c
    Eigen::VectorXd cos_c = sin_b.array() * sin_b0 + cos_b.array() * cos_b0 * (l.derived().array() - l0).cos();

    // avoid division by zero or near zero
    for (int i = 0; i < cos_c.size(); ++i) {
        if (std::abs(cos_c(i)) < std::numeric_limits<double>::epsilon()) {
            x(i) = 0;
            y(i) = 0;
        }
        else {
            x(i) = cos_b(i) * std::sin(l(i) - l0) / cos_c(i);
            y(i) = (cos_b0 * sin_b(i) - sin_b0 * cos_b(i) * std::cos(l(i) - l0)) / cos_c(i);
        }
    }
}

class Telescope {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // keys to telescope data vectors (angles in radians)
    // first key is the name in the telescope file, second is for citlali.
    std::map<std::string, std::string> data_keys {
        {"TelTime", "TelTime"},
        {"TelUtc","TelUTC"},
        {"TelRaAct", "TelRa"},
        {"TelDecAct", "TelDec"},
        {"SourceLAct", "TelL"},
        {"SourceBAct", "TelB"},
        {"SourceRaAct", "TelRa"},
        {"SourceDecAct", "TelDec"},
        {"TelAzAct", "TelAzAct"},
        {"TelElAct", "TelElAct"},
        {"SourceAz", "SourceAz"},
        {"SourceEl", "SourceEl"},
        {"ActParAng", "ActParAng"},
        {"ActGalAng", "ActGalAng"},
        {"Hold", "Hold"},
        {"TelAzCor", "TelAzCor"},
        {"TelElCor", "TelElCor"},
        {"TelAzDes", "TelAzDes"},
        {"TelElDes", "TelElDes"},
        {"PpsTime","PpsTime"},
        {"TelAzMap", "TelAzMap"},
        {"TelElMap", "TelElMap"}
    };

    // keys for Header parameters
    std::vector<std::string> header_keys = {
        "Source.Ra", "Source.Dec", "Source.L", "Source.B", "Source.Epoch",
        "Source.CoordSys", "Source.Velocity", "Source.VelSys", "Source.Planet",
        "Source.RaProperMotionCor", "Source.DecProperMotionCor", "Source.ElObsMin",
        "Source.ElObsMax", "Sky.ObsVel", "Sky.BaryVel", "Sky.ParAng",
        "Sky.RaOffsetSys", "Telescope.PointingTolerance", "Telescope.AzDesPos",
        "Telescope.ElDesPos", "Telescope.AzActPos", "Telescope.ElActPos",
        "Telescope.CraneInBeam", "M1.ModelEnabled", "M1.ZernikeEnabled",
        "M1.ModelMode", "M2.XAct", "M2.YAct", "M2.ZAct", "M2.TipAct",
        "M2.TiltAct", "M2.XReq", "M2.YReq", "M2.ZReq", "M2.TipReq",
        "M2.TiltReq", "M2.XDes", "M2.YDes", "M2.ZDes", "M2.TipDes",
        "M2.TiltDes", "M2.XPcor", "M2.YPcor", "M2.ZPcor", "M2.TipPcor",
        "M2.TiltPcor", "M2.XCmd", "M2.YCmd", "M2.ZCmd", "M2.TipCmd",
        "M2.TiltCmd", "M2.ElCmd", "M2.AzPcor", "M2.ElPcor", "M2.CorEnabled",
        "M2.Follow", "M2.M2Heartbeat", "M2.AcuHeartbeat", "M2.Alive",
        "M2.Hold", "M2.ModelMode", "M3.ElDesEnabled", "M3.Alive", "M3.Fault",
        "M3.M3Heartbeat", "M3.AcuHeartbeat", "M3.M3OffPos", "TimePlace.LST",
        "TimePlace.UTDate", "TimePlace.UT1", "TimePlace.ObsLongitude",
        "TimePlace.ObsLatitude", "TimePlace.ObsElevation", "Gps.IgnoreLock",
        "PointModel.ModRev", "PointModel.AzPointModelCor", "PointModel.ElPointModelCor",
        "PointModel.AzPaddleOff", "PointModel.ElPaddleOff", "PointModel.AzReceiverOff",
        "PointModel.ElReceiverOff", "PointModel.AzReceiverCor", "PointModel.ElReceiverCor",
        "PointModel.AzUserOff", "PointModel.ElUserOff", "PointModel.AzM2Cor",
        "PointModel.ElM2Cor", "PointModel.ElRefracCor", "PointModel.AzTiltCor",
        "PointModel.ElTiltCor", "PointModel.AzTotalCor", "PointModel.ElTotalCor",
        "PointModel.PointModelCorEnabled", "PointModel.M2CorEnabled",
        "PointModel.RefracCorEnabled", "PointModel.TiltCorEnabled",
        "PointModel.ReceiverOffEnabled", "Dcs.ObsNum", "Dcs.SubObsNum",
        "Dcs.ScanNum", "Dcs.ObsType", "Dcs.ObsMode", "Dcs.CalMode",
        "Dcs.IntegrationTime", "Dcs.RequestedTime", "Tiltmeter_0_.TiltX",
        "Tiltmeter_0_.TiltY", "Tiltmeter_0_.Temp", "Tiltmeter_1_.TiltX",
        "Tiltmeter_1_.TiltY", "Tiltmeter_1_.Temp", "Weather.Temperature",
        "Weather.Humidity", "Weather.Pressure", "Weather.Precipitation",
        "Weather.Radiation", "Weather.WindDir1", "Weather.WindSpeed1",
        "Weather.WindDir2", "Weather.WindSpeed2", "Weather.TimeOfDay",
        "Radiometer.Tau", "Radiometer.Tau2", "Toltec.BeamSelected", "Toltec.NumBands",
        "Toltec.NumBeams", "Toltec.NumPixels", "Toltec.AzPointOff", "Toltec.ElPointOff",
        "Toltec.AzPointCor", "Toltec.ElPointCor", "Toltec.M3Dir", "Toltec.Remote",
        "TelescopeBackend.Master", "TelescopeBackend.ObsNum", "TelescopeBackend.SubObsNum",
        "TelescopeBackend.ScanNum", "TelescopeBackend.CalObsNum", "TelescopeBackend.NumPixels",
        "Map.NumRepeats", "Map.NumScans", "Map.HPBW", "Map.ScanAngle",
        "Map.XLength", "Map.YLength", "Map.XOffset", "Map.YOffset", "Map.XStep",
        "Map.YStep", "Map.XRamp", "Map.YRamp", "Map.TSamp", "Map.TRef", "Map.TCal",
        "Map.RowsPerScan", "Map.ScansPerCal", "Map.ScansToSkip", "Map.TurnTime",
        "Map.NumPass", "Map.ScanRate", "Map.ScanXStep", "Map.ScanYStep", "Map.ExecMode",
        "Lissajous.XLength", "Lissajous.YLength", "Lissajous.XOmega", "Lissajous.YOmega",
        "Lissajous.XDelta", "Lissajous.XLengthMinor", "Lissajous.YLengthMinor",
        "Lissajous.XOmegaMinor", "Lissajous.YOmegaMinor", "Lissajous.XDeltaMinor",
        "Lissajous.XOmegaNorm", "Lissajous.YOmegaNorm", "Lissajous.XOmegaMinorNorm",
        "Lissajous.YOmegaMinorNorm", "Lissajous.ScanRate", "Lissajous.TScan",
        "Lissajous.ExecMode", "ScanFile.Valid", "M1.ZernikeC", "M1.ActPos",
        "M1.CmdPos", "Sim.Jobkey"
    };

    // dish diameter
    double lmt_diameter_m = 50.0;

    // std map for telescope data and header
    std::map<std::string, Eigen::VectorXd> data, header;
    // center of map
    double x0, y0;
    // number of time chunks
    int n_chunks;
    // simulated obs or not
    bool sim_obs;
    // observation descriptors (from telescope file)
    std::string obs_goal, obs_pgm, map_coord, source_name, project_id;
    // pixel coordinate frame
    std::string pixel_axes;
    // raster or rastajous
    int map_exec_mode;
    // chunk indices matrix (4 x nscans)
    Eigen::MatrixXI chunk_indices;
    // force time chunk mode
    bool force_chunk;
    // how to divide up chunks
    std::string chunk_mode;
    // size or number of chunks
    double chunking_value;

    // size of tod bandpass filter
    Eigen::Index tod_filter_order;

    // pointing offsets MJD (defaults to zero)
    Eigen::VectorXd pointing_offset_mjd;

    void load_telescope(const std::string&);
    template <typename ConfigType>
    void get_configs(ConfigType &);
    template <typename ConfigType>
    void get_pointing_offsets(ConfigType&);
    void calc_tangent_plane_radec();
    void calc_tangent_plane_altaz();
    void calc_tangent_plane_galactic();
    void calc_tangent_plane_pointing();
    void calc_raster_chunk_indices();
    void calc_time_chunk_indices(const double);
    void calc_chunk_indices(const double, const Eigen::Index);
};

void Telescope::load_telescope(const std::string &filepath) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    try {
        // get telescope file
        NcFile fo(filepath, NcFile::read, NcFile::classic);
        auto vars = fo.getVars();

        // check if simulation job key is found.
        if (!fo.getVar("Header.Sim.Jobkey").isNull()) {
            logger->info("found Header.Sim.Jobkey. reducing as a simulation.");
            sim_obs = true;
        } else {
            logger->info("cannot find Header.Sim.Jobkey. reducing as real data.");
            sim_obs = false;
            // get obs goal
            netCDF::NcVar var = fo.getVar("Header.Dcs.ObsGoal");
            read_nc_var_to_string(var, 129, obs_goal);
        }

        // get observation mapping pattern
        netCDF::NcVar var = fo.getVar("Header.Dcs.ObsPgm");
        read_nc_var_to_string(var, 129, obs_pgm);

        // get exec mode if doing raster or rastajous
        if (obs_pgm == "Map") {
            vars.find("Header.Map.ExecMode")->second.getVar(&map_exec_mode);
            netCDF::NcVar var = fo.getVar("Header.Map.MapCoord");
            read_nc_var_to_string(var, 129, map_coord);
        } else {
            map_exec_mode = 1;
        }

        // get source name
        var = fo.getVar("Header.Source.SourceName");
        read_nc_var_to_string(var, 129, source_name);

        // get project id
        if (!sim_obs) {
            var = fo.getVar("Header.Dcs.ProjectId");
            read_nc_var_to_string(var, 129, project_id);
        } else {
            project_id = "simu";
        }

        // get tel data keys
        for (const auto& pair : data_keys) {
            auto var_name = "Data.TelescopeBackend." + pair.first;
            auto var = fo.getVar(var_name);
            if (!var.isNull()) {
                netCDF::NcType var_type = var.getType();
                if (var_type != netCDF::ncChar && var_type != netCDF::ncString) {
                    int num_dims = var.getDimCount();
                    int n_pts = 1;
                    if (num_dims == 1) {
                        n_pts = vars.find(var_name)->second.getDim(0).getSize();
                    }
                    data[pair.second].resize(n_pts);
                    vars.find(var_name)->second.getVar(data[pair.second].data());
                }
            } else {
                logger->debug("cannot find {}", pair.first);
            }
        }

        // get tel header keys
        for (const auto& key : header_keys) {
            auto var_name = "Header." + key;
            auto var = fo.getVar(var_name);
            if (!var.isNull()) {
                netCDF::NcType var_type = var.getType();
                if (var_type != netCDF::ncChar && var_type != netCDF::ncString) {
                    int num_dims = var.getDimCount();
                    int n_pts = 1;
                    if (num_dims == 1) {
                        n_pts = vars.find(var_name)->second.getDim(0).getSize();
                    }
                    header[key].resize(n_pts);
                    vars.find(var_name)->second.getVar(header[key].data());
                }
            } else {
                logger->debug("cannot find {}", key);
            }
        }

    } catch (NcException &e) {
        throw std::runtime_error(fmt::format("Failed to load data from netCDF file {}: {}", filepath, e.what()));
    }
}

template <typename ConfigType>
void Telescope::get_configs(ConfigType &config) {
    // pixel axes
    config.get(pixel_axes, std::tuple{"mapmaking", "pixel_axes"});
    // force chunking?
    config.get(force_chunk, std::tuple{"timestream", "chunking", "force_chunking"});
    // chunk mode
    config.get(chunk_mode, std::tuple{"timestream", "chunking", "mode"});
    // get number of time chunks or length in seconds
    config.get(chunking_value, std::tuple{"timestream", "chunking", "value"});
}

template <typename ConfigType>
void Telescope::get_pointing_offsets(ConfigType& config) {
    // check if config file has pointing_offsets
    if (!config.has("pointing_offsets")) {
        throw std::runtime_error("pointing_offsets not found in config");
    }

    logger->debug("{}", config);

    try {
        // get azimuth offset
        auto az_offset = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets", 0, "value_arcsec"});
        data["pointing_offset_az_arcsec"] = Eigen::Map<Eigen::VectorXd>(az_offset.data(), az_offset.size());

        // get altitude offset
        auto alt_offset = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets", 1, "value_arcsec"});
        data["pointing_offset_alt_arcsec"] = Eigen::Map<Eigen::VectorXd>(alt_offset.data(), alt_offset.size());

    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("failed to retrieve azimuth offset: {}", e.what()));
    }

    try {
        // get mjd of pointing offsets
        auto mjd_offset = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets", 2, "modified_julian_date"});
        pointing_offset_mjd = Eigen::Map<Eigen::VectorXd>(mjd_offset.data(), mjd_offset.size());
    } catch (const std::exception& e) {
        logger->warn("failed to retrieve modified_julian_date, defaulting to zeros: {}", e.what());
        pointing_offset_mjd.setZero(2);
    }
}

void Telescope::calc_tangent_plane_radec() {
    logger->debug("calculating radec tangent plane pointing");
    Eigen::Index n_pts = data.at("Hold").size();

    // resize vectors to hold physical (tangent plane) coordinates
    data["ra_tan"].resize(n_pts);
    data["dec_tan"].resize(n_pts);

    // copy RA and Dec data
    Eigen::VectorXd ra = data.at("TelRa");
    Eigen::VectorXd& dec = data.at("TelDec");

    // rescale RA values to the range [-pi, pi]
    ra = (ra.array() > pi).select(ra.array() - 2.0 * pi, ra.array());

    // get the center RA and Dec positions
    double ra0 = header.at("Source.Ra")(0);
    double dec0 = header.at("Source.Dec")(0);

    // rescale center RA to the range [-pi, pi]
    if (ra0 > pi) {
        ra0 -= 2.0 * pi;
    }

    // calculate gnomonic projection
    gnomonic_projection(ra, dec, ra0, dec0, data.at("ra_tan"), data.at("dec_tan"));
}

void Telescope::calc_tangent_plane_altaz() {
    logger->debug("calculating altaz tangent plane pointing");
    // adjust TelAzAct to avoid aliasing issues and handle wrapping around 2*pi
    Eigen::VectorXd& tel_az_act = data.at("TelAzAct");
    Eigen::VectorXd& source_az = data.at("SourceAz");

    for (Eigen::Index i = 0; i < tel_az_act.size(); ++i) {
        if ((tel_az_act(i) - source_az(i)) > 0.9 * 2.0 * pi) {
            tel_az_act(i) -= 2.0 * pi;
        }
    }

    // calculate azimuth difference
    Eigen::VectorXd az_diff = tel_az_act.array() - source_az.array();

    // calculate tangent plane longitude (azimuth)
    data["az_tan"] = (cos(data.at("TelElAct").array() - data.at("TelElCor").array()) * az_diff.array() - data.at("TelAzCor").array()).matrix();

    // calculate tangent plane latitude (altitude)
    data["alt_tan"] = (data.at("TelElAct").array() - data.at("SourceEl").array() - data.at("TelElCor").array()).matrix();
}

void Telescope::calc_tangent_plane_galactic() {
    logger->debug("calculating galactic tangent plane pointing");
    Eigen::Index n_pts = data.at("Hold").size();

    // resize vectors to hold physical (tangent plane) coordinates
    data["l_tan"].resize(n_pts);
    data["b_tan"].resize(n_pts);

    // copy l and b data
    Eigen::VectorXd l = data.at("TelL");
    Eigen::VectorXd& b = data.at("TelB");

    // Rescale l values to the range [-pi, pi]
    l = (l.array() > pi).select(l.array() - 2.0 * pi, l.array());

    // get the center l and b positions
    double l0 = header.at("Source.L")(0);
    double b0 = header.at("Source.B")(0);

    // rescale center l to the range [-pi, pi]
    if (l0 > pi) {
        l0 -= 2.0 * pi;
    }

    // calculate gnomonic projection
    gnomonic_projection(l, b, l0, b0, data.at("l_tan"), data.at("b_tan"));
}

void Telescope::calc_tangent_plane_pointing() {
    calc_tangent_plane_radec();
    calc_tangent_plane_altaz();
    calc_tangent_plane_galactic();

    // set the default coordinate vectors
    if (pixel_axes == "radec") {
        data["x"] = data.at("ra_tan");
        data["y"] = data.at("dec_tan");

        x0 = header.at("Source.RA")(0);
        y0 = header.at("Source.DEC")(0);

    } else if (pixel_axes == "altaz") {
        data["x"] = data.at("az_tan");
        data["y"] = data.at("alt_tan");

        x0 = 0.0;
        y0 = 0.0;

    } else if (pixel_axes == "galactic") {
        data["x"] = data.at("l_tan");
        data["y"] = data.at("b_tan");

        x0 = header.at("Source.L")(0);
        y0 = header.at("Source.B")(0);
    }

    // apply corrections
    data.at("TelElAct") -= data.at("TelElCor");
    data.at("TelAzAct") -= data.at("TelAzCor");
}

void Telescope::calc_raster_chunk_indices() {
    logger->info("calculating scans for raster mode");

    // convert the hold signal to a boolean matrix
    Eigen::Matrix<bool, Eigen::Dynamic, 1> hold_bool = data.at("Hold").template cast<bool>();

    // determine coordinate keys based on map_coord
    std::string coord1_key, coord2_key;
    if (map_coord == "Ra") {
        coord1_key = "ra_tan";
        coord2_key = "dec_tan";
    } else if (map_coord == "Az") {
        coord1_key = "az_tan";
        coord2_key = "alt_tan";
    }

    // update hold_bool if a point is outside the map bounds
    for (Eigen::Index i = 0; i < hold_bool.size(); ++i) {
        if (!is_point_in_box(data.at(coord1_key)(i), data.at(coord2_key)(i), header.at("Map.XLength")(0),
                             header.at("Map.YLength")(0), header.at("Map.ScanAngle")(0))) {
            hold_bool(i) = true;
        }
    }

    // count the number of chunk based on changes in hold_bool
    n_chunks = 0;
    for (Eigen::Index i = 1; i < hold_bool.size(); ++i) {
        if (hold_bool(i) && !hold_bool(i - 1)) {
            n_chunks++;
        }
    }

    // increment chunk number if the last element is not held
    if (!hold_bool(hold_bool.size() - 1)) {
        n_chunks++;
    }

    chunk_indices.resize(4, n_chunks);

    // populate chunk indices based on hold_bool transitions
    int counter = -1;
    if (!hold_bool(0)) {
        chunk_indices(0, 0) = 1;
        counter++;
    }

    // find the indices where hold_bool changes value
    for (Eigen::Index i = 1; i < hold_bool.size(); ++i) {
        if (!hold_bool(i) && hold_bool(i - 1)) {
            counter++;
            chunk_indices(0, counter) = i + 1;
        } else if (hold_bool(i) && !hold_bool(i - 1)) {
            chunk_indices(1, counter) = i - 1;
        }
    }

    // populate the final chunk if the last element is not held
    if (!hold_bool(hold_bool.size() - 1)) {
        chunk_indices(1, n_chunks - 1) = hold_bool.size() - 1;
    }
}

void Telescope::calc_time_chunk_indices(const double data_fs_hz) {
    logger->info("calculating scans for lissajous/rastajous mode");

    // index of the last chunk
    int n_pts = data.at("Hold").size();

    // length of a chunk
    double period;

    if (chunk_mode == "duration") {
        // calculate the period based on the time_chunk and sample rate (data_fs_hz)
        period = std::floor(chunking_value * data_fs_hz);

        // adjust the period if it exceeds the available data range or if time_chunk < 0
        if ((period > n_pts) || period < 0) {
            period = n_pts;
        }

        // calculate the number of scans
        n_chunks = static_cast<Eigen::Index>(std::floor(n_pts / period));
    }
    else if (chunk_mode == "number") {
        n_chunks = chunking_value;

        period = n_pts / n_chunks;
    }

    // resize and assign inner scan indices
    chunk_indices.resize(4, n_chunks);
    chunk_indices.row(0) = Eigen::Vector<Eigen::Index, Eigen::Dynamic>::LinSpaced(n_chunks, 0, n_chunks - 1).array() * static_cast<int>(period);
    chunk_indices.row(1) = chunk_indices.row(0).array() + static_cast<int>(period) - 1;
}

void Telescope::calc_chunk_indices(const double data_fs_hz, const Eigen::Index tod_filter_order) {
    // if mapping was in raster mode and force chunking disabled
    if ((obs_pgm == "Map" && map_exec_mode == 0) && !force_chunk) {
        calc_raster_chunk_indices();
    }

    // if lissajous, rasstajous, or force chunking is enabled
    else if (obs_pgm == "Lissajous" || (obs_pgm == "Map" && map_exec_mode == 1) || force_chunk) {
        calc_time_chunk_indices(data_fs_hz);
    }

    // create a copy of the chunk indices matrix
    Eigen::MatrixXI chunk_indices_temp = chunk_indices;

    // initialize the count of bad chunks
    Eigen::Index n_chunks_bad = 0;

    // create a boolean matrix to mark bad chunks
    Eigen::Matrix<bool, Eigen::Dynamic, 1> bad_chunk(n_chunks);

    // loop over all chunks to identify small chunks
    for (Eigen::Index i = 0; i < n_chunks; ++i) {
        int chunk_size = chunk_indices_temp(1, i) - chunk_indices_temp(0, i) + 1;
        if (chunk_size < data_fs_hz) {
            n_chunks_bad++;
            bad_chunk(i) = true;
        } else {
            bad_chunk(i) = false;
        }
    }

    // rebuild the chunk indices matrix excluding bad chunks
    Eigen::Index c = 0;
    chunk_indices.resize(4, n_chunks - n_chunks_bad);
    for (Eigen::Index i = 0; i < n_chunks; ++i) {
        if (!bad_chunk(i)) {
            chunk_indices(0, c) = chunk_indices_temp(0, i);
            chunk_indices(1, c) = chunk_indices_temp(1, i);
            c++;
        }
    }

    // calculate the number of good chunks
    n_chunks -= n_chunks_bad;

    // ensure that we have at least one chunk
    if (n_chunks < 1) {
        throw std::runtime_error("Number of chunks must be greater than or equal to 1.");
    }

    // set up the 3rd and 4th chunk indices rows to account for low-pass filtering
    chunk_indices.row(2) = chunk_indices.row(0).array() - tod_filter_order;
    chunk_indices.row(3) = chunk_indices.row(1).array() + tod_filter_order;

    // adjust the first and last outer chunk positions to match the inner chunks
    chunk_indices(2, 0) = chunk_indices(0, 0);
    chunk_indices(3, n_chunks - 1) = chunk_indices(1, n_chunks - 1);

    // adjust the first and last inner chunk positions by the filter length
    chunk_indices(0, 0) += tod_filter_order;
    chunk_indices(1, n_chunks - 1) -= tod_filter_order;

    // log the final chunk indices
    logger->debug("chunk_indices {}", chunk_indices);
}
