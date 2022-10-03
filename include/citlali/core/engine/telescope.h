#pragma once

#include <netcdf>

#include <citlali/core/utils/netcdf_io.h>

#include <citlali/core/utils/constants.h>

namespace engine {

class Telescope {
public:
    // mapping pattern
    char obs_pgm_char [128];
    // source name
    char source_name_char [128];

    // simulation jobkey
    char sim_job_key [128];
    // is this a simulation?
    bool sim_obs;

    // strings for mapping pattern and source name
    std::string obs_pgm, source_name;

    // time chunk size for lissajous/rastajous
    double time_chunk;

    // sample rate
    double fsmp, d_fsmp;

    // tau at 225 GHZ
    double tau_225_GHz;

    // number of samples to remove for inner scans
    Eigen::Index inner_scans_chunk;

    // scan indices matrix (4 x nscans)
    Eigen::MatrixXI scan_indices;

    // std map for telescope data vectors
    std::map<std::string, Eigen::VectorXd> tel_data;
    // std map for telescope header vectors
    std::map<std::string, Eigen::VectorXd> tel_header;

    // pixel axes (icrs, altaz, etc)
    std::string pixel_axes;

    // keys to telescope data vectors
    std::map<std::string, std::string> tel_data_keys {
        {"Data.TelescopeBackend.TelTime", "TelTime"},
        {"Data.TelescopeBackend.TelRaAct", "TelRa"},
        {"Data.TelescopeBackend.TelDecAct", "TelDec"},
        {"Data.TelescopeBackend.SourceRaAct", "TelRa"},
        {"Data.TelescopeBackend.SourceDecAct", "TelDec"},
        {"Data.TelescopeBackend.TelAzAct", "TelAzAct"},
        {"Data.TelescopeBackend.TelElAct", "TelElAct"},
        {"Data.TelescopeBackend.SourceAz", "SourceAz"},
        {"Data.TelescopeBackend.SourceEl", "SourceEl"},
        {"Data.TelescopeBackend.ActParAng", "ActParAng"},
        {"Data.TelescopeBackend.ParAng", "ParAng"},
        {"Data.TelescopeBackend.Hold", "Hold"},
        {"Data.TelescopeBackend.TelAzCor", "TelAzCor"},
        {"Data.TelescopeBackend.TelElCor", "TelElCor"},
        {"Data.TelescopeBackend.TelAzDes", "TelAzDes"},
        {"Data.TelescopeBackend.TelElDes", "TelElDes"},
        {"Data.TelescopeBackend.PpsTime","PpsTime"},
        {"Data.TelescopeBackend.TelAzMap", "TelAzMap"},
        {"Data.TelescopeBackend.TelElMap", "TelElMap"}
    };

    // keys to telescope header values
    std::map<std::string, std::string> tel_header_keys {
        {"Header.Source.Ra", "Header.Source.Ra"},
        {"Header.Source.Dec", "Header.Source.Dec"},
        {"Header.Source.Epoch", "Header.Source.Epoch"},
        {"Header.Source.CoordSys", "Header.Source.CoordSys"},
        {"Header.Source.Velocity", "Header.Source.Velocity"},
        {"Header.Source.VelSys", "Header.Source.VelSys"},
        {"Header.Source.Planet", "Header.Source.Planet"},
        {"Header.Source.RaProperMotionCor", "Header.Source.RaProperMotionCor"},
        {"Header.Source.DecProperMotionCor", "Header.Source.DecProperMotionCor"},
        {"Header.Source.ElObsMin", "Header.Source.ElObsMin"},
        {"Header.Source.ElObsMax", "Header.Source.ElObsMax"},
        {"Header.Sky.ObsVel", "Header.Sky.ObsVel"},
        {"Header.Sky.BaryVel", "Header.Sky.BaryVel"},
        {"Header.Sky.ParAng", "Header.Sky.ParAng"},
        {"Header.Sky.RaOffsetSys", "Header.Sky.RaOffsetSys"},
        {"Header.Telescope.PointingTolerance", "Header.Telescope.PointingTolerance"},
        {"Header.Telescope.AzDesPos", "Header.Telescope.AzDesPos"},
        {"Header.Telescope.ElDesPos", "Header.Telescope.ElDesPos"},
        {"Header.Telescope.AzActPos", "Header.Telescope.AzActPos"},
        {"Header.Telescope.ElActPos", "Header.Telescope.ElActPos"},
        {"Header.Telescope.CraneInBeam", "Header.Telescope.CraneInBeam"},
        {"Header.M1.ModelEnabled", "Header.M1.ModelEnabled"},
        {"Header.M1.ZernikeEnabled", "Header.M1.ZernikeEnabled"},
        {"Header.M1.ModelMode", "Header.M1.ModelMode"},
        {"Header.M2.XAct", "Header.M2.XAct"},
        {"Header.M2.YAct", "Header.M2.YAct"},
        {"Header.M2.ZAct", "Header.M2.ZAct"},
        {"Header.M2.TipAct", "Header.M2.TipAct"},
        {"Header.M2.TiltAct", "Header.M2.TiltAct"},
        {"Header.M2.XReq", "Header.M2.XReq"},
        {"Header.M2.YReq", "Header.M2.YReq"},
        {"Header.M2.ZReq", "Header.M2.ZReq"},
        {"Header.M2.TipReq", "Header.M2.TipReq"},
        {"Header.M2.TiltReq", "Header.M2.TiltReq"},
        {"Header.M2.XDes", "Header.M2.XDes"},
        {"Header.M2.YDes", "Header.M2.YDes"},
        {"Header.M2.ZDes", "Header.M2.ZDes"},
        {"Header.M2.TipDes", "Header.M2.TipDes"},
        {"Header.M2.TiltDes", "Header.M2.TiltDes"},
        {"Header.M2.XPcor", "Header.M2.XPcor"},
        {"Header.M2.YPcor", "Header.M2.YPcor"},
        {"Header.M2.ZPcor", "Header.M2.ZPcor"},
        {"Header.M2.TipPcor", "Header.M2.TipPcor"},
        {"Header.M2.TiltPcor", "Header.M2.TiltPcor"},
        {"Header.M2.XCmd", "Header.M2.XCmd"},
        {"Header.M2.YCmd", "Header.M2.YCmd"},
        {"Header.M2.ZCmd", "Header.M2.ZCmd"},
        {"Header.M2.TipCmd", "Header.M2.TipCmd"},
        {"Header.M2.TiltCmd", "Header.M2.TiltCmd"},
        {"Header.M2.ElCmd", "Header.M2.ElCmd"},
        {"Header.M2.AzPcor", "Header.M2.AzPcor"},
        {"Header.M2.ElPcor", "Header.M2.ElPcor"},
        {"Header.M2.CorEnabled", "Header.M2.CorEnabled"},
        {"Header.M2.Follow", "Header.M2.Follow"},
        {"Header.M2.M2Heartbeat", "Header.M2.M2Heartbeat"},
        {"Header.M2.AcuHeartbeat", "Header.M2.AcuHeartbeat"},
        {"Header.M2.Alive", "Header.M2.Alive"},
        {"Header.M2.Hold", "Header.M2.Hold"},
        {"Header.M2.ModelMode", "Header.M2.ModelMode"},
        {"Header.M3.ElDesEnabled", "Header.M3.ElDesEnabled"},
        {"Header.M3.Alive", "Header.M3.Alive"},
        {"Header.M3.Fault", "Header.M3.Fault"},
        {"Header.M3.M3Heartbeat", "Header.M3.M3Heartbeat"},
        {"Header.M3.AcuHeartbeat", "Header.M3.AcuHeartbeat"},
        {"Header.M3.M3OffPos", "Header.M3.M3OffPos"},
        {"Header.TimePlace.LST", "Header.TimePlace.LST"},
        {"Header.TimePlace.UTDate", "Header.TimePlace.UTDate"},
        {"Header.TimePlace.UT1", "Header.TimePlace.UT1"},
        {"Header.TimePlace.ObsLongitude", "Header.TimePlace.ObsLongitude"},
        {"Header.TimePlace.ObsLatitude", "Header.TimePlace.ObsLatitude"},
        {"Header.TimePlace.ObsElevation", "Header.TimePlace.ObsElevation"},
        {"Header.Gps.IgnoreLock", "Header.Gps.IgnoreLock"},
        {"Header.PointModel.ModRev", "Header.PointModel.ModRev"},
        {"Header.PointModel.AzPointModelCor", "Header.PointModel.AzPointModelCor"},
        {"Header.PointModel.ElPointModelCor", "Header.PointModel.ElPointModelCor"},
        {"Header.PointModel.AzPaddleOff", "Header.PointModel.AzPaddleOff"},
        {"Header.PointModel.ElPaddleOff", "Header.PointModel.ElPaddleOff"},
        {"Header.PointModel.AzReceiverOff", "Header.PointModel.AzReceiverOff"},
        {"Header.PointModel.ElReceiverOff", "Header.PointModel.ElReceiverOff"},
        {"Header.PointModel.AzReceiverCor", "Header.PointModel.AzReceiverCor"},
        {"Header.PointModel.ElReceiverCor", "Header.PointModel.ElReceiverCor"},
        {"Header.PointModel.AzUserOff", "Header.PointModel.AzUserOff"},
        {"Header.PointModel.ElUserOff", "Header.PointModel.ElUserOff"},
        {"Header.PointModel.AzM2Cor", "Header.PointModel.AzM2Cor"},
        {"Header.PointModel.ElM2Cor", "Header.PointModel.ElM2Cor"},
        {"Header.PointModel.ElRefracCor", "Header.PointModel.ElRefracCor"},
        {"Header.PointModel.AzTiltCor", "Header.PointModel.AzTiltCor"},
        {"Header.PointModel.ElTiltCor", "Header.PointModel.ElTiltCor"},
        {"Header.PointModel.AzTotalCor", "Header.PointModel.AzTotalCor"},
        {"Header.PointModel.ElTotalCor", "Header.PointModel.ElTotalCor"},
        {"Header.PointModel.PointModelCorEnabled", "Header.PointModel.PointModelCorEnabled"},
        {"Header.PointModel.M2CorEnabled", "Header.PointModel.M2CorEnabled"},
        {"Header.PointModel.RefracCorEnabled", "Header.PointModel.RefracCorEnabled"},
        {"Header.PointModel.TiltCorEnabled", "Header.PointModel.TiltCorEnabled"},
        {"Header.PointModel.ReceiverOffEnabled", "Header.PointModel.ReceiverOffEnabled"},
        {"Header.Dcs.ObsNum", "Header.Dcs.ObsNum"},
        {"Header.Dcs.SubObsNum", "Header.Dcs.SubObsNum"},
        {"Header.Dcs.ScanNum", "Header.Dcs.ScanNum"},
        {"Header.Dcs.ObsType", "Header.Dcs.ObsType"},
        {"Header.Dcs.ObsMode", "Header.Dcs.ObsMode"},
        {"Header.Dcs.CalMode", "Header.Dcs.CalMode"},
        {"Header.Dcs.IntegrationTime", "Header.Dcs.IntegrationTime"},
        {"Header.Dcs.RequestedTime", "Header.Dcs.RequestedTime"},
        {"Header.Tiltmeter_0_.TiltX", "Header.Tiltmeter_0_.TiltX"},
        {"Header.Tiltmeter_0_.TiltY", "Header.Tiltmeter_0_.TiltY"},
        {"Header.Tiltmeter_0_.Temp", "Header.Tiltmeter_0_.Temp"},
        {"Header.Tiltmeter_1_.TiltX", "Header.Tiltmeter_1_.TiltX"},
        {"Header.Tiltmeter_1_.TiltY", "Header.Tiltmeter_1_.TiltY"},
        {"Header.Tiltmeter_1_.Temp", "Header.Tiltmeter_1_.Temp"},
        {"Header.Weather.Temperature", "Header.Weather.Temperature"},
        {"Header.Weather.Humidity", "Header.Weather.Humidity"},
        {"Header.Weather.Pressure", "Header.Weather.Pressure"},
        {"Header.Weather.Precipitation", "Header.Weather.Precipitation"},
        {"Header.Weather.Radiation", "Header.Weather.Radiation"},
        {"Header.Weather.WindDir1", "Header.Weather.WindDir1"},
        {"Header.Weather.WindSpeed1", "Header.Weather.WindSpeed1"},
        {"Header.Weather.WindDir2", "Header.Weather.WindDir2"},
        {"Header.Weather.WindSpeed2", "Header.Weather.WindSpeed2"},
        {"Header.Weather.TimeOfDay", "Header.Weather.TimeOfDay"},
        {"Header.Radiometer.Tau", "Header.Radiometer.Tau"},
        {"Header.Radiometer.Tau2", "Header.Radiometer.Tau2"},
        {"Header.Toltec.BeamSelected", "Header.Toltec.BeamSelected"},
        {"Header.Toltec.NumBands", "Header.Toltec.NumBands"},
        {"Header.Toltec.NumBeams", "Header.Toltec.NumBeams"},
        {"Header.Toltec.NumPixels", "Header.Toltec.NumPixels"},
        {"Header.Toltec.AzPointOff", "Header.Toltec.AzPointOff"},
        {"Header.Toltec.ElPointOff", "Header.Toltec.ElPointOff"},
        {"Header.Toltec.AzPointCor", "Header.Toltec.AzPointCor"},
        {"Header.Toltec.ElPointCor", "Header.Toltec.ElPointCor"},
        {"Header.Toltec.M3Dir", "Header.Toltec.M3Dir"},
        {"Header.Toltec.Remote", "Header.Toltec.Remote"},
        {"Header.TelescopeBackend.Master", "Header.TelescopeBackend.Master"},
        {"Header.TelescopeBackend.ObsNum", "Header.TelescopeBackend.ObsNum"},
        {"Header.TelescopeBackend.SubObsNum", "Header.TelescopeBackend.SubObsNum"},
        {"Header.TelescopeBackend.ScanNum", "Header.TelescopeBackend.ScanNum"},
        {"Header.TelescopeBackend.CalObsNum", "Header.TelescopeBackend.CalObsNum"},
        {"Header.TelescopeBackend.NumPixels", "Header.TelescopeBackend.NumPixels"},
        {"Header.Map.NumRepeats", "Header.Map.NumRepeats"},
        {"Header.Map.NumScans", "Header.Map.NumScans"},
        {"Header.Map.HPBW", "Header.Map.HPBW"},
        {"Header.Map.ScanAngle", "Header.Map.ScanAngle"},
        {"Header.Map.XLength", "Header.Map.XLength"},
        {"Header.Map.YLength", "Header.Map.YLength"},
        {"Header.Map.XOffset", "Header.Map.XOffset"},
        {"Header.Map.YOffset", "Header.Map.YOffset"},
        {"Header.Map.XStep", "Header.Map.XStep"},
        {"Header.Map.YStep", "Header.Map.YStep"},
        {"Header.Map.XRamp", "Header.Map.XRamp"},
        {"Header.Map.YRamp", "Header.Map.YRamp"},
        {"Header.Map.TSamp", "Header.Map.TSamp"},
        {"Header.Map.TRef", "Header.Map.TRef"},
        {"Header.Map.TCal", "Header.Map.TCal"},
        {"Header.Map.RowsPerScan", "Header.Map.RowsPerScan"},
        {"Header.Map.ScansPerCal", "Header.Map.ScansPerCal"},
        {"Header.Map.ScansToSkip", "Header.Map.ScansToSkip"},
        {"Header.Map.TurnTime", "Header.Map.TurnTime"},
        {"Header.Map.NumPass", "Header.Map.NumPass"},
        {"Header.Map.ScanRate", "Header.Map.ScanRate"},
        {"Header.Map.ScanXStep", "Header.Map.ScanXStep"},
        {"Header.Map.ScanYStep", "Header.Map.ScanYStep"},
        {"Header.Map.ExecMode", "Header.Map.ExecMode"},
        {"Header.Lissajous.XLength", "Header.Lissajous.XLength"},
        {"Header.Lissajous.YLength", "Header.Lissajous.YLength"},
        {"Header.Lissajous.XOmega", "Header.Lissajous.XOmega"},
        {"Header.Lissajous.YOmega", "Header.Lissajous.YOmega"},
        {"Header.Lissajous.XDelta", "Header.Lissajous.XDelta"},
        {"Header.Lissajous.XLengthMinor", "Header.Lissajous.XLengthMinor"},
        {"Header.Lissajous.YLengthMinor", "Header.Lissajous.YLengthMinor"},
        {"Header.Lissajous.XOmegaMinor", "Header.Lissajous.XOmegaMinor"},
        {"Header.Lissajous.YOmegaMinor", "Header.Lissajous.YOmegaMinor"},
        {"Header.Lissajous.XDeltaMinor", "Header.Lissajous.XDeltaMinor"},
        {"Header.Lissajous.XOmegaNorm", "Header.Lissajous.XOmegaNorm"},
        {"Header.Lissajous.YOmegaNorm", "Header.Lissajous.YOmegaNorm"},
        {"Header.Lissajous.XOmegaMinorNorm", "Header.Lissajous.XOmegaMinorNorm"},
        {"Header.Lissajous.YOmegaMinorNorm", "Header.Lissajous.YOmegaMinorNorm"},
        {"Header.Lissajous.ScanRate", "Header.Lissajous.ScanRate"},
        {"Header.Lissajous.TScan", "Header.Lissajous.TScan"},
        {"Header.Lissajous.ExecMode", "Header.Lissajous.ExecMode"},
        {"Header.ScanFile.Valid", "Header.ScanFile.Valid"},
        {"Header.M1.ZernikeC","Header.M1.ZernikeC"},
        {"Header.Sim.Jobkey","Header.Sim.Jobkey"}
    };

    void get_tel_data(std::string &);
    void calc_tan_pointing();
    void calc_tan_icrs();
    void calc_tan_altaz();
    void calc_scan_indices();
};

void Telescope::get_tel_data(std::string &filepath) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    try {
        // get telescope file
        NcFile fo(filepath, NcFile::read, NcFile::classic);
        auto vars = fo.getVars();

        // get mapping pattern
        vars.find("Header.Dcs.ObsPgm")->second.getVar(&obs_pgm_char);
        obs_pgm = std::string(obs_pgm_char);

        // get source name
        vars.find("Header.Source.SourceName")->second.getVar(&source_name_char);
        source_name = std::string(source_name_char);

        // check if simulation job key is found.
        try {
            vars.find("Header.Sim.Jobkey")->second.getVar(&sim_job_key);
            SPDLOG_WARN("found Header.Sim.Jobkey");
            sim_obs = true;
        } catch (NcException &e) {
            SPDLOG_WARN("cannot find Header.Sim.Jobkey");
            sim_obs = false;
        }

        // loop through telescope data keys and populate vectors
        for (auto const& pair : tel_data_keys) {
            try {
                Eigen::Index n_pts = vars.find(pair.first)->second.getDim(0).getSize();
                tel_data[pair.second].resize(n_pts);
                vars.find(pair.first)->second.getVar(tel_data[pair.second].data());

            } catch (NcException &e) {
                SPDLOG_WARN("cannot find {}", pair.first);
            }
        }

        // adjust parallactic angle parameters
        //tel_data["ParAng"] = pi - tel_data["ParAng"].array();
        //tel_data["ActParAng"] = pi - tel_data["ActParAng"].array();

        // loop through telescope header keys and populate vectors
        for (auto const& pair : tel_header_keys) {
            try {
                Eigen::Index n_pts = vars.find(pair.first)->second.getDim(0).getSize();
                tel_header[pair.second].resize(n_pts);
                vars.find(pair.first)->second.getVar(tel_header[pair.second].data());

            } catch (NcException &e) {
                SPDLOG_WARN("cannot find {}", pair.first);
            }
        }

    } catch (NcException &e) {
        SPDLOG_WARN("{}", e.what());
        throw DataIOError{fmt::format(
            "failed to load data from netCDF file {}", filepath)};
    }
}

void Telescope::calc_tan_pointing() {

    // get icrs tan pointing
    if (std::strcmp("icrs", pixel_axes.c_str()) == 0) {
        calc_tan_icrs();
    }

    // get altaz tan pointing
    else if (std::strcmp("altaz", pixel_axes.c_str()) == 0) {
        calc_tan_altaz();
    }
}

void Telescope::calc_tan_icrs() {

    Eigen::Index n_pts = tel_data["TelRa"].size();

    tel_data["lat_phys"].resize(n_pts);
    tel_data["lon_phys"].resize(n_pts);

    // copy ra
    Eigen::VectorXd ra = tel_data["TelRa"];
    // copy dec
    Eigen::VectorXd dec = tel_data["TelDec"];

    // rescale ra
    (ra.array() > pi).select(tel_data["TelRa"].array() - 2.0*pi, tel_data["TelRa"].array());

    // copy center ra
    double center_ra = tel_header["Header.Source.Ra"](0);
    // copy center dec
    double center_dec = tel_header["Header.Source.Dec"](0);

    // rescale center ra
    center_ra = (center_ra > pi) ? center_ra - (2.0*pi) : center_ra;

    auto cosc = sin(center_dec)*sin(tel_data["TelDec"].array()) +
                cos(center_dec)*cos(tel_data["TelDec"].array())*cos(ra.array() - center_ra);

    // calc tangent coordinates
    for (Eigen::Index i=0; i<n_pts; i++) {
        if (cosc(i)==0.) {
            tel_data["lat_phys"](i) = 0.;
            tel_data["lon_phys"](i) = 0.;
        }

        else {
            tel_data["lat_phys"](i) = (cos(center_dec)*sin(tel_data["TelDec"](i)) -
                                              sin(center_dec)*cos(tel_data["TelDec"](i))*cos(ra(i)-center_ra))/cosc(i);

            tel_data["lon_phys"](i) = cos(tel_data["TelDec"](i))*sin(ra(i)-center_ra)/cosc(i);
        }
    }
}

void Telescope::calc_tan_altaz() {
    for (Eigen::Index i=0; i<tel_data["TelAzAct"].size(); i++) {
        if ((tel_data["TelAzAct"](i) - tel_data["SourceAz"](i)) > 0.9*2.0*pi) {
            tel_data["TelAzAct"](i) = tel_data["TelAzAct"](i) - 2.0*pi;
        }
    }

    // calculate tangent coordinates
    tel_data["lat_phys"] = (tel_data["TelElAct"] - tel_data["SourceEl"]) - tel_data["TelElCor"];
    tel_data["lon_phys"] = cos(tel_data["TelElDes"].array())*(tel_data["TelAzAct"].array() - tel_data["SourceAz"].array())
                                  - tel_data["TelAzCor"].array();

}

void Telescope::calc_scan_indices() {

    // number of scans
    Eigen::Index nscans;

    if (std::strcmp("Map", obs_pgm.c_str()) == 0) {
        SPDLOG_INFO("calculating scans for raster mode");
        // cast hold signal to boolean
        Eigen::Matrix<bool,Eigen::Dynamic,1> hold_bool = tel_data["Hold"].template cast<bool>();

        for (Eigen::Index i=1; i<hold_bool.size(); i++) {
            if (hold_bool(i) - hold_bool(i-1) == 1) {
                nscans++;
            }
        }

        if (hold_bool(hold_bool.size()-1) == 0){
            nscans++;
        }
        scan_indices.resize(4,nscans);

        int counter = -1;
        if (!hold_bool(0)) {
            scan_indices(0,0) = 1;
            counter++;
        }

        for (Eigen::Index i=1; i<hold_bool.size(); i++) {
            if (hold_bool(i) - hold_bool(i-1) < 0) {
                counter++;
                scan_indices(0,counter) = i + 1;
            }

            else if (hold_bool(i) - hold_bool(i-1) > 0) {
                scan_indices(1,counter) = i - 1;
            }
        }
        scan_indices(1,nscans - 1) = hold_bool.size() - 1;
    }

    // get scan indices for Lissajous/Rastajous pattern
    else if (std::strcmp("Lissajous", obs_pgm.c_str()) == 0) {
        SPDLOG_INFO("calculating scans for lissajous/rastajous mode");

        // index of first scan
        Eigen::Index first_scan_i = 0;
        // index of last scan
        Eigen::Index last_scan_i = tel_data["Hold"].size() - 1;

        // period (time_chunk/fsmp in seconds/Hz)
        Eigen::Index period_i = floor(time_chunk*fsmp);

        double period = floor(time_chunk*fsmp);
        // calculate number of scans
        nscans = floor((last_scan_i - first_scan_i + 1)*1./period);

        // assign scans to scan_indices matrix
        scan_indices.resize(4,nscans);
        scan_indices.row(0) =
            Eigen::Vector<Eigen::Index,Eigen::Dynamic>::LinSpaced(nscans,0,nscans-1).array()*period_i + first_scan_i;
        scan_indices.row(1) = scan_indices.row(0).array() + period_i - 1;
    }

    // copy of scan indices matrix
    Eigen::MatrixXI scan_indices_temp = scan_indices;

    // number of bad scans
    Eigen::Index n_bad_scans = 0;

    // size of scan
    int sum = 0;

    Eigen::Matrix<bool, Eigen::Dynamic, 1> is_bad_scan(nscans);
    for (Eigen::Index i=0; i<nscans; i++) {
        sum = 0;
        for (Eigen::Index j=scan_indices_temp(0,i); j<(scan_indices_temp(1,i)+1); j++) {
            sum += 1;
        }
        if(sum < 2.*fsmp) {
            n_bad_scans++;
            is_bad_scan(i)=1;
        }
        else {
            is_bad_scan(i) = 0;
        }
    }

    int c = 0;
    scan_indices.resize(4,nscans-n_bad_scans);
    for (Eigen::Index i=0; i<nscans; i++) {
        if (!is_bad_scan(i)) {
            scan_indices(0,c) = scan_indices_temp(0,i);
            scan_indices(1,c) = scan_indices_temp(1,i);
            c++;
        }
    }

    // calculate the number of good scans
    nscans = nscans - n_bad_scans;

    // set up the 3rd and 4th scan indices rows so that we don't lose data during lowpassing
    // inner_scans_chunk is zero if lowpassing is not enabled
    scan_indices.row(2) = scan_indices.row(0).array() - inner_scans_chunk;
    scan_indices.row(3) = scan_indices.row(1).array() + inner_scans_chunk;

    // set first and last outer scan positions to the same as inner scans
    scan_indices(2,0) = scan_indices(0,0);
    scan_indices(3,nscans-1) = scan_indices(1,nscans-1);

    scan_indices(0,0) = scan_indices(0,0) + inner_scans_chunk;
    scan_indices(1,nscans-1) = scan_indices(1,nscans-1) - inner_scans_chunk;
}

} // namespace engine
