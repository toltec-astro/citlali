#pragma once

#include <netcdf>

#include <tula/eigen.h>

#include <citlali/core/utils/netcdf_io.h>
#include <citlali/core/utils/constants.h>

namespace engine {

class Telescope {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // mapping pattern
    char obs_pgm_char [128] = "";
    // obs goal
    char obs_goal_char [128] = "";
    // source name
    char source_name_char [128] = "";

    // simulation jobkey
    char sim_job_key [128] = "";
    // is this a simulation?
    bool sim_obs;

    // strings for mapping pattern and source name
    std::string obs_pgm, source_name, obs_goal;

    // tangent plane center
    //std::map<std::string, double> tan_center_rad;

    // time chunk size for lissajous/rastajous
    double time_chunk;

    // force chunking scan indices
    bool force_chunk;

    // remove samples based on a velocity limit
    double velocity_limit;

    // sample rate, downsampled sample rate
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

    // keys to telescope data vectors (angles in radians)
    std::map<std::string, std::string> tel_data_keys {
        {"Data.TelescopeBackend.TelTime", "TelTime"},
        {"Data.TelescopeBackend.TelUtc","TelUTC"},
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

    // keys to telescope header values (angles in radians)
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

} // namespace engine
