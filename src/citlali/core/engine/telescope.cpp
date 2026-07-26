#include <boost/algorithm/string/trim.hpp>

#include <tula/algorithm/ei_stats.h>
#include <tula/formatter/matrix.h>
#include <tula/logging.h>

#include <citlali/core/engine/telescope.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/pointing.h>

namespace engine {

void Telescope::get_tel_data(std::string &filepath) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    try {
        // get telescope file
        NcFile fo(filepath, NcFile::read, NcFile::classic);
        auto vars = fo.getVars();

        // check if simulation job key is found.
        try {
            vars.find("Header.Sim.Jobkey")->second.getVar(&sim_job_key);
            logger->warn("found Header.Sim.Jobkey");
            sim_obs = true;
        } catch (NcException &e) {
            logger->warn("cannot find Header.Sim.Jobkey. reducing as real data.");
            sim_obs = false;
        }

        // get obs goal
        if (!sim_obs) {
            char obs_goal_char [129];
            vars.find("Header.Dcs.ObsGoal")->second.getVar(&obs_goal_char);
            obs_goal_char[128] = '\0';
            obs_goal = std::string(obs_goal_char);
            std::string::iterator end_pos = std::remove(obs_goal.begin(), obs_goal.end(), ' ');
            obs_goal.erase(end_pos, obs_goal.end());
        }

        // get map pattern
        char obs_pgm_char [129];
        // get mapping pattern
        vars.find("Header.Dcs.ObsPgm")->second.getVar(&obs_pgm_char);
        obs_pgm_char[128] = '\0';
        obs_pgm = std::string(obs_pgm_char);
        // try and remove end characters
        std::string::iterator end_pos = std::remove(obs_pgm.begin(), obs_pgm.end(), ' ');
        obs_pgm.erase(end_pos, obs_pgm.end());

        if (obs_pgm=="Map") {
            vars.find("Header.Map.ExecMode")->second.getVar(&exec_mode);

            char map_coord_char [129];
            // get mapping pattern
            vars.find("Header.Map.MapCoord")->second.getVar(&map_coord_char);
            map_coord_char[128] = '\0';
            map_coord = std::string(map_coord_char);
            // try and remove end characters
            end_pos = std::remove(map_coord.begin(), map_coord.end(), ' ');
            map_coord.erase(end_pos, map_coord.end());
        }
        else {
            exec_mode = 1;
        }

        // cannot reduce in lissajous mode if chunk less than or equal to zero
        if ((obs_pgm=="Lissajous" || (obs_pgm=="Map" && exec_mode==1)) && chunking_value<=0) {
            logger->error("mapping mode is lissajous and time chunk size is zero");
            std::exit(EXIT_FAILURE);
        }

        // get source name
        char source_name_char [129];
        vars.find("Header.Source.SourceName")->second.getVar(&source_name_char);
        source_name_char[128] = '\0';
        source_name = std::string(source_name_char);
        // try and remove end characters
        end_pos = std::remove(source_name.begin(), source_name.end(), ' ');
        source_name.erase(end_pos, source_name.end());

        // get project id
        if (!sim_obs) {
            char project_id_name_char [129];
            vars.find("Header.Dcs.ProjectId")->second.getVar(&project_id_name_char);
            project_id_name_char[128] = '\0';
            project_id = std::string(project_id_name_char);
            // try and remove end characters
            end_pos = std::remove(project_id.begin(), project_id.end(), ' ');
            project_id.erase(end_pos, project_id.end());
        }
        else {
            project_id = "simu";
        }

        // loop through telescope data keys and populate vectors
        for (const auto& pair : tel_data_keys) {
            try {
                logger->info("tel_data key {}",pair.first);
                Eigen::Index n_pts = vars.find(pair.first)->second.getDim(0).getSize();
                tel_data[pair.second].resize(n_pts);
                Eigen::VectorXd data_temp(n_pts);
                vars.find(pair.first)->second.getVar(data_temp.data());
                tel_data[pair.second] = data_temp;

            } catch (NcException &e) {
                logger->warn("cannot find {}", pair.first);
            }
        }

        // loop through telescope header keys and populate vectors
        for (const auto& pair : tel_header_keys) {
            // set for scalars
            Eigen::Index n_pts = 1;
            try {
                // try to get dimensions, otherwise keep n_pts at 1
                try {
                    n_pts = vars.find(pair.first)->second.getDim(0).getSize();
                } catch(...) {}

                Eigen::VectorXd header_temp(n_pts);
                vars.find(pair.first)->second.getVar(header_temp.data());
                tel_header[pair.second] = header_temp;

            } catch (NcException &e) {
                // ignore if simulation
                if (!sim_obs) {
                    logger->warn("cannot find {}", pair.first);
                }
            }
        }

        // set tau 225 GHz
        tau_225_GHz = tel_header["Header.Radiometer.Tau"](0);

        // close netcdf file
        fo.close();

    } catch (NcException &e) {
        logger->warn("{}", e.what());
        throw DataIOError{fmt::format(
            "failed to load data from netCDF file {}", filepath)};
    }

    if (!sim_obs) {
        // convert TelUTC to unix time
        engine_utils::utc_to_unix(tel_data["TelUTC"],tel_header["Header.TimePlace.UTDate"]);

        // keys for fixing periodic boundary conditions
        std::vector<std::string> periodic_keys = {
            "TelRa","TelDec",
            "TelL", "TelB",
            "TelAzAct","TelElAct",
            "TelAzCor","TelElCor",
            "SourceAz","SourceEl",
        };

        // fix periodic boundary conditions
        for (const auto &key: periodic_keys) {
            engine_utils::fix_periodic_boundary(tel_data[key],pi,1.99*pi,2.0*pi);
        }

        // calculate galactic l and b for source
        engine_utils::equatorial_to_galactic(tel_header["Header.Source.Ra"](0),
                                             tel_header["Header.Source.Dec"](0),
                                             tel_header["Header.Source.L"](0),
                                             tel_header["Header.Source.B"](0));
    }

    // manually set epoch to J2000 for simulations
    else {
        tel_header["Header.Source.Epoch"](0) = 2000.0;
    }
}

void Telescope::calc_tan_pointing() {
    // get radec tangent pointing
    calc_tan_radec();
    // get altaz tangent pointing
    calc_tan_altaz();

    if (!sim_obs) {
        // get galactic tangent pointing
        calc_tan_galactic();
    }

    // set tangential projection to radec
    if (pixel_axes=="radec") {
        logger->info("using radec frame");
        tel_data["lat_phys"] = tel_data["dec_phys"];
        tel_data["lon_phys"] = tel_data["ra_phys"];
    }
    // set tangential projection to altaz
    else if (pixel_axes=="altaz") {
        logger->info("using altaz frame");
        tel_data["lat_phys"] = tel_data["alt_phys"];
        tel_data["lon_phys"] = tel_data["az_phys"];
    }
    // set tangential projection to galactic
    else if (pixel_axes=="galactic") {
        logger->info("using galactic frame");
        tel_data["lat_phys"] = tel_data["b_phys"];
        tel_data["lon_phys"] = tel_data["l_phys"];
    }

    // apply corrections
    tel_data["TelElAct"] -= tel_data["TelElCor"];
    tel_data["TelAzAct"] -= tel_data["TelAzCor"];
}

void Telescope::calc_tan_radec() {
    // size of data
    Eigen::Index n_pts = tel_data["TelRa"].size();

    // vectors to hold physical (tangent plane) coordinates
    tel_data["dec_phys"].resize(n_pts);
    tel_data["ra_phys"].resize(n_pts);

    // copy radec
    Eigen::VectorXd ra = tel_data["TelRa"];
    auto& dec = tel_data["TelDec"];

    // rescale ra
    (ra.array() > pi).select(tel_data["TelRa"].array() - 2.0*pi, tel_data["TelRa"].array());

    // center positions
    double ra0 = tel_header["Header.Source.Ra"](0);
    double dec0 = tel_header["Header.Source.Dec"](0);

    // rescale center ra
    ra0 = (ra0 > pi) ? ra0 - (2.0*pi) : ra0;

    // calculate gnomonic projection
    engine_utils::gnomonic_projection(ra, dec, ra0, dec0, tel_data["ra_phys"], tel_data["dec_phys"]);
}

void Telescope::calc_tan_altaz() {
    // use loop to avoid annoying eigen aliasing issues with select
    for (Eigen::Index i=0; i<tel_data["TelAzAct"].size(); ++i) {
        if ((tel_data["TelAzAct"](i) - tel_data["SourceAz"](i)) > 0.9*2.0*pi) {
            tel_data["TelAzAct"](i) = tel_data["TelAzAct"](i) - 2.0*pi;
        }
    }

    // subtract source az
    auto az_diff = tel_data["TelAzAct"].array() - tel_data["SourceAz"].array();

    // tangent plane lat (alt)
    tel_data["alt_phys"] = (tel_data["TelElAct"].array() - tel_data["SourceEl"].array() - tel_data["TelElCor"].array()).matrix();

    // tangent plane lon (az)
    tel_data["az_phys"] = (cos(tel_data["TelElAct"].array() - tel_data["TelElCor"].array()) * az_diff - tel_data["TelAzCor"].array()).matrix();
}

void Telescope::calc_tan_galactic() {
    // size of data
    Eigen::Index n_pts = tel_data["TelL"].size();

    // vectors to hold physical (tangent plane) coordinates
    tel_data["l_phys"].resize(n_pts);
    tel_data["b_phys"].resize(n_pts);

    // copy lb
    Eigen::VectorXd l = tel_data["TelL"];
    auto b = tel_data["TelB"];

    // rescale l
    (l.array() > pi).select(tel_data["TelL"].array() - 2.0*pi, tel_data["TelL"].array());

    // center positions
    double l0 = tel_header["Header.Source.L"](0);
    double b0 = tel_header["Header.Source.B"](0);

    // rescale center l
    l0 = (l0 > pi) ? l0 - (2.0*pi) : l0;

    // calculate gnomonic projection
    engine_utils::gnomonic_projection(l, b, l0, b0, tel_data["l_phys"], tel_data["b_phys"]);
}

void Telescope::calc_scan_indices() {
    // number of scans
    Eigen::Index n_scans = 0;

    // get scans for raster pattern
    if ((obs_pgm=="Map" && exec_mode==0) && !force_chunk) {
        logger->info("calculating scans for raster mode");

        // convert the hold signal to a bool
        Eigen::Matrix<bool,Eigen::Dynamic,1> hold_bool = tel_data["Hold"].template cast<bool>();

        // get velocities
        /*auto x_vel = engine_utils::compute_numerical_derivative(tel_data["TelTime"],tel_data["az_phys"]);
        auto y_vel = engine_utils::compute_numerical_derivative(tel_data["TelTime"],tel_data["alt_phys"]);

        auto vel = sqrt(pow(x_vel.array(),2) + pow(y_vel.array(),2));

        double med_vel = tula::alg::median(vel);

        for (Eigen::Index i=0; i<vel.size(); ++i) {
            if (vel(i) < 0.5*med_vel) {
                //hold_bool(i) = 1;
            }
        }*/

        std::string coord1_key, coord2_key;
        if (map_coord == "Ra") {
            coord1_key = "ra_phys";
            coord2_key = "dec_phys";
        }
        else if (map_coord == "Az") {
            coord1_key = "az_phys";
            coord2_key = "alt_phys";
        }

        for (Eigen::Index i = 0; i < hold_bool.size(); ++i) {
            if (!engine_utils::is_point_in_box(tel_data[coord1_key](i), tel_data[coord2_key](i),
                                              tel_header["Header.Map.XLength"](0), tel_header["Header.Map.YLength"](0),
                                              tel_header["Header.Map.ScanAngle"](0))) {
                hold_bool(i) = 1;
            }
        }

        // find where the change in the hold signal is 1 and increment scans
        for (Eigen::Index i=1; i<hold_bool.size(); ++i) {
            if (hold_bool(i) - hold_bool(i-1) == 1) {
                n_scans++;
            }
        }

        // increment scan number if last element is zero
        if (hold_bool(hold_bool.size() - 1) == 0) {
            n_scans++;
        }
        // resize matrix to hold scans
        scan_indices.resize(4,n_scans);

        // populate first scan
        int counter = -1;
        if (!hold_bool(0)) {
            scan_indices(0,0) = 1;
            counter++;
        }

        for (Eigen::Index i=1; i<hold_bool.size(); ++i) {
            // get start of scan
            if (hold_bool(i) - hold_bool(i-1) < 0) {
                counter++;
                scan_indices(0,counter) = i + 1;
            }
            // get end of scan
            else if (hold_bool(i) - hold_bool(i-1) > 0) {
                scan_indices(1,counter) = i - 1;
            }
        }

        if (hold_bool(hold_bool.size()-1) == 0) {
            // populate final scan
            scan_indices(1,n_scans - 1) = hold_bool.size() - 1;
        }
    }

    // get scan indices for Lissajous/Rastajous pattern
    else if (obs_pgm=="Lissajous" || (obs_pgm=="Map" && exec_mode==1) || force_chunk) {
        logger->info("calculating scans for lissajous/rastajous mode");

        // index of first scan
        Eigen::Index first_scan_i = 0;
        // index of last scan
        Eigen::Index last_scan_i = tel_data["Hold"].size() - 1;

        double period;
        Eigen::Index period_i;

        if (chunk_mode == "duration") {

            // period (time_chunk x fsmp in seconds x Hz)
            period_i = std::floor(chunking_value*fsmp);

            period = std::floor(chunking_value*fsmp);

            if (period > (last_scan_i - first_scan_i + 1)) {
                period = last_scan_i - first_scan_i + 1;
                period_i = last_scan_i - first_scan_i + 1;
            }

            // calculate number of scans
            n_scans = std::floor((last_scan_i - first_scan_i + 1)*1./period);
        }
        else if (chunk_mode == "number") {
            n_scans = chunking_value;

            period = (last_scan_i - first_scan_i + 1) / n_scans;
            period_i = (last_scan_i - first_scan_i + 1) / n_scans;
        }

        // assign scans to scan_indices matrix
        scan_indices.resize(4,n_scans);
        scan_indices.row(0) =
            Eigen::Vector<Eigen::Index,Eigen::Dynamic>::LinSpaced(n_scans,0,n_scans-1).array()*period_i + first_scan_i;
        scan_indices.row(1) = scan_indices.row(0).array() + period_i - 1;
    }

    // copy of scan indices matrix
    Eigen::MatrixXI scan_indices_temp = scan_indices;

    // number of bad scans
    Eigen::Index n_bad_scans = 0;

    // size of scan
    int sum = 0;

    // check for small scans
    Eigen::Matrix<bool, Eigen::Dynamic, 1> is_bad_scan(n_scans);
    for (Eigen::Index i=0; i<n_scans; ++i) {
        sum = 0;
        for (Eigen::Index j=scan_indices_temp(0,i); j<(scan_indices_temp(1,i)+1); ++j) {
            sum += 1;
        }
        if (sum < 2.*fsmp) {
            n_bad_scans++;
            is_bad_scan(i) = 1;
        }
        else {
            is_bad_scan(i) = 0;
        }
    }

    // rebuild scan matrix excluding bad scans
    Eigen::Index c = 0;
    scan_indices.resize(4,n_scans-n_bad_scans);
    for (Eigen::Index i=0; i<n_scans; ++i) {
        if (!is_bad_scan(i)) {
            scan_indices(0,c) = scan_indices_temp(0,i);
            scan_indices(1,c) = scan_indices_temp(1,i);
            c++;
        }
    }

    // calculate the number of good scans
    n_scans = n_scans - n_bad_scans;

    // set up the 3rd and 4th scan indices rows so that we don't lose data
    // during lowpassing inner_scans_chunk is zero if lowpassing is not enabled
    scan_indices.row(2) = scan_indices.row(0).array() - inner_scans_chunk;
    scan_indices.row(3) = scan_indices.row(1).array() + inner_scans_chunk;

    // set first and last outer scan positions to the same as inner scans
    scan_indices(2,0) = scan_indices(0,0);
    scan_indices(3,n_scans-1) = scan_indices(1,n_scans-1);

    // add/subtract the filter length from first/last inner scan positions
    scan_indices(0,0) = scan_indices(0,0) + inner_scans_chunk;
    scan_indices(1,n_scans-1) = scan_indices(1,n_scans-1) - inner_scans_chunk;

    logger->info("scan_indices {}",scan_indices);
}

} // namespace engine
