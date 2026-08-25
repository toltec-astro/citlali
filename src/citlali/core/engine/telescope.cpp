#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include <tula/logging.h>
#include <tula/algorithm/ei_stats.h>

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/engine/telescope.h>
#include <citlali/core/error/error.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/pointing.h>

namespace engine {

void Telescope::get_tel_data(
    std::string &filepath,
    const citlali::config::TimestreamChunkingConfig &chunking) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    try {
        // get telescope file
        NcFile fo(filepath, NcFile::read, NcFile::classic);

        // check if simulation job key is found.
        try {
            fo.getVar("Header.Sim.Jobkey").getVar(&sim_job_key);
            logger->info("found Header.Sim.Jobkey");
            sim_obs = true;
        } catch (NcException &e) {
            logger->info("Header.Sim.Jobkey is absent; treating input as real data");
            sim_obs = false;
        }

        // get obs goal
        if (!sim_obs) {
            char obs_goal_char [129];
            fo.getVar("Header.Dcs.ObsGoal").getVar(&obs_goal_char);
            obs_goal_char[128] = '\0';
            obs_goal = std::string(obs_goal_char);
            std::string::iterator end_pos = std::remove(obs_goal.begin(), obs_goal.end(), ' ');
            obs_goal.erase(end_pos, obs_goal.end());
        }

        // get map pattern
        char obs_pgm_char [129];
        // get mapping pattern
        fo.getVar("Header.Dcs.ObsPgm").getVar(&obs_pgm_char);
        obs_pgm_char[128] = '\0';
        obs_pgm = std::string(obs_pgm_char);
        // try and remove end characters
        std::string::iterator end_pos = std::remove(obs_pgm.begin(), obs_pgm.end(), ' ');
        obs_pgm.erase(end_pos, obs_pgm.end());

        if (obs_pgm=="Map") {
            fo.getVar("Header.Map.ExecMode").getVar(&exec_mode);

            char map_coord_char [129];
            // get mapping pattern
            fo.getVar("Header.Map.MapCoord").getVar(&map_coord_char);
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
        if ((obs_pgm=="Lissajous" || (obs_pgm=="Map" && exec_mode==1)) &&
            chunking.value <= 0) {
            throw citlali::error::invalid_config(
                "lissajous mapping requires a positive time chunk size");
        }

        // get source name
        char source_name_char [129];
        fo.getVar("Header.Source.SourceName").getVar(&source_name_char);
        source_name_char[128] = '\0';
        source_name = std::string(source_name_char);
        // try and remove end characters
        end_pos = std::remove(source_name.begin(), source_name.end(), ' ');
        source_name.erase(end_pos, source_name.end());

        // get project id
        if (!sim_obs) {
            char project_id_name_char [129];
            fo.getVar("Header.Dcs.ProjectId").getVar(&project_id_name_char);
            project_id_name_char[128] = '\0';
            project_id = std::string(project_id_name_char);
            // try and remove end characters
            end_pos = std::remove(project_id.begin(), project_id.end(), ' ');
            project_id.erase(end_pos, project_id.end());
        }
        else {
            project_id = "simu";
        }

        std::vector<std::string> missing_data_keys;
        std::vector<std::string> missing_header_keys;

        // loop through telescope data keys and populate vectors
        for (const auto& pair : tel_data_keys) {
            try {
                logger->info("tel_data key {}",pair.first);
                Eigen::Index n_pts = fo.getVar(pair.first).getDim(0).getSize();
                tel_data[pair.second].resize(n_pts);
                Eigen::VectorXd data_temp(n_pts);
                fo.getVar(pair.first).getVar(data_temp.data());
                tel_data[pair.second] = data_temp;

            } catch (NcException &e) {
                missing_data_keys.push_back(pair.first);
                logger->debug("optional telescope data variable is absent: {}",
                              pair.first);
            }
        }

        // loop through telescope header keys and populate vectors
        for (const auto& pair : tel_header_keys) {
            // set for scalars
            Eigen::Index n_pts = 1;
            try {
                // try to get dimensions, otherwise keep n_pts at 1
                try {
                    n_pts = fo.getVar(pair.first).getDim(0).getSize();
                } catch(...) {}

                Eigen::VectorXd header_temp(n_pts);
                fo.getVar(pair.first).getVar(header_temp.data());
                tel_header[pair.second] = header_temp;

            } catch (NcException &e) {
                // ignore if simulation
                if (!sim_obs) {
                    missing_header_keys.push_back(pair.first);
                    logger->debug("optional telescope header is absent: {}",
                                  pair.first);
                }
            }
        }

        if (!missing_data_keys.empty() || !missing_header_keys.empty()) {
            logger->info(
                "telescope input {} omits {} configured data variables and {} configured header values; individual optional names are available at debug level",
                filepath, missing_data_keys.size(), missing_header_keys.size());
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
    if (citlali::config::is_radec_map_pixel_axes(pixel_axes)) {
        logger->info("using radec frame");
        tel_data["lat_phys"] = tel_data["dec_phys"];
        tel_data["lon_phys"] = tel_data["ra_phys"];
    }
    // set tangential projection to altaz
    else if (citlali::config::is_altaz_map_pixel_axes(pixel_axes)) {
        logger->info("using altaz frame");
        tel_data["lat_phys"] = tel_data["alt_phys"];
        tel_data["lon_phys"] = tel_data["az_phys"];
    }
    // set tangential projection to galactic
    else if (citlali::config::is_galactic_map_pixel_axes(pixel_axes)) {
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
    ra = (ra.array() > pi).select(ra.array() - 2.0*pi, ra.array());

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
    l = (l.array() > pi).select(l.array() - 2.0*pi, l.array());

    // center positions
    double l0 = tel_header["Header.Source.L"](0);
    double b0 = tel_header["Header.Source.B"](0);

    // rescale center l
    l0 = (l0 > pi) ? l0 - (2.0*pi) : l0;

    // calculate gnomonic projection
    engine_utils::gnomonic_projection(l, b, l0, b0, tel_data["l_phys"], tel_data["b_phys"]);
}

void Telescope::calc_scan_indices(
    const citlali::config::TimestreamChunkingConfig &chunking) {
    // number of scans
    Eigen::Index n_scans = 0;

    auto require_tel_series = [&](const std::string &key) -> Eigen::VectorXd & {
        auto it = tel_data.find(key);
        if (it == tel_data.end() || it->second.size() == 0) {
            throw std::runtime_error(fmt::format(
                "cannot calculate scan indices: telescope series '{}' is missing or empty", key));
        }
        return it->second;
    };

    auto require_header_scalar = [&](const std::string &key) -> double {
        auto it = tel_header.find(key);
        if (it == tel_header.end() || it->second.size() == 0) {
            throw std::runtime_error(fmt::format(
                "cannot calculate scan indices: telescope header '{}' is missing or empty", key));
        }
        return it->second(0);
    };

    // get scans for raster pattern
    if ((obs_pgm=="Map" && exec_mode==0) && !chunking.force) {
        logger->info("calculating scans for raster mode");

        // convert the hold signal to a bool
        auto &hold = require_tel_series("Hold");
        Eigen::Matrix<bool,Eigen::Dynamic,1> hold_bool = hold.template cast<bool>();
        if (hold_bool.size() == 0) {
            throw std::runtime_error("cannot calculate scans for raster mode: Hold series is empty");
        }

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
        const auto map_coord_lower = boost::algorithm::to_lower_copy(map_coord);
        if (map_coord_lower == "ra" || map_coord_lower == "dec") {
            coord1_key = "ra_phys";
            coord2_key = "dec_phys";
        }
        else if (map_coord_lower == "az" || map_coord_lower == "el" || map_coord_lower == "alt") {
            coord1_key = "az_phys";
            coord2_key = "alt_phys";
        }
        else if (map_coord_lower == "gal" || map_coord_lower == "l" || map_coord_lower == "b") {
            coord1_key = "l_phys";
            coord2_key = "b_phys";
        }
        else {
            throw std::runtime_error(fmt::format(
                "cannot calculate scans for raster mode: unsupported Header.Map.MapCoord='{}'", map_coord));
        }

        auto &coord1 = require_tel_series(coord1_key);
        auto &coord2 = require_tel_series(coord2_key);
        if (coord1.size() != hold_bool.size() || coord2.size() != hold_bool.size()) {
            throw std::runtime_error(fmt::format(
                "cannot calculate scans for raster mode: coordinate sizes do not match Hold size "
                "(Hold={}, {}={}, {}={})",
                hold_bool.size(), coord1_key, coord1.size(), coord2_key, coord2.size()));
        }

        const double x_length = require_header_scalar("Header.Map.XLength");
        const double y_length = require_header_scalar("Header.Map.YLength");
        const double scan_angle = require_header_scalar("Header.Map.ScanAngle");

        for (Eigen::Index i = 0; i < hold_bool.size(); ++i) {
            if (!engine_utils::is_point_in_box(coord1(i), coord2(i),
                                              x_length, y_length, scan_angle)) {
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
        if (n_scans <= 0) {
            throw std::runtime_error(fmt::format(
                "cannot calculate scans for raster mode: found no in-bounds non-hold samples "
                "(map_coord='{}')", map_coord));
        }
        // resize matrix to hold scans
        scan_indices.resize(4,n_scans);

        // populate first scan
        int counter = -1;
        if (!hold_bool(0)) {
            scan_indices(0,0) = 0;
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
    else if (obs_pgm=="Lissajous" || (obs_pgm=="Map" && exec_mode==1) ||
             chunking.force) {
        logger->info("calculating scans for lissajous/rastajous mode");

        // index of first scan
        Eigen::Index first_scan_i = 0;
        // index of last scan
        Eigen::Index last_scan_i = require_tel_series("Hold").size() - 1;

        double period;
        Eigen::Index period_i;

        if (chunking.mode == "duration") {

            // period (time_chunk x fsmp in seconds x Hz)
            period_i = std::floor(chunking.value*fsmp);

            period = std::floor(chunking.value*fsmp);

            if (period > (last_scan_i - first_scan_i + 1)) {
                period = last_scan_i - first_scan_i + 1;
                period_i = last_scan_i - first_scan_i + 1;
            }

            if (period_i <= 0) {
                throw std::runtime_error(fmt::format(
                    "cannot calculate scans for lissajous/rastajous mode: invalid chunk duration "
                    "(chunking_value={}, fsmp={})", chunking.value, fsmp));
            }

            // calculate number of scans
            n_scans = std::floor((last_scan_i - first_scan_i + 1)*1./period);
        }
        else if (chunking.mode == "number") {
            n_scans = chunking.value;

            if (n_scans <= 0) {
                throw std::runtime_error(fmt::format(
                    "cannot calculate scans for lissajous/rastajous mode: invalid chunk count {}",
                    chunking.value));
            }

            period = (last_scan_i - first_scan_i + 1) / n_scans;
            period_i = (last_scan_i - first_scan_i + 1) / n_scans;
        }
        else {
            throw std::runtime_error(fmt::format(
                "cannot calculate scans for lissajous/rastajous mode: unsupported chunk_mode='{}'",
                chunking.mode));
        }

        if (period_i <= 0 || n_scans <= 0) {
            throw std::runtime_error(fmt::format(
                "cannot calculate scans for lissajous/rastajous mode: invalid period_i={} n_scans={}",
                period_i, n_scans));
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

    if (n_scans <= 0) {
        throw std::runtime_error("cannot calculate scan indices: all scans were rejected as too short");
    }

    const Eigen::Index inner_context = std::max<Eigen::Index>(0, inner_scans_chunk);
    const Eigen::Index outer_context =
        std::max<Eigen::Index>(inner_context, outer_scans_chunk);
    const Eigen::Index n_total_samples = require_tel_series("Hold").size();

    // Rows 0/1 define the science scan. Rows 2/3 define the larger data span
    // loaded around it for filters/PSD estimates. Keep these contexts separate:
    // large detector-notch PSD context must not shrink the science scan.
    for (Eigen::Index i = 0; i < n_scans; ++i) {
        scan_indices(2, i) =
            std::max<Eigen::Index>(0, scan_indices(0, i) - outer_context);
        scan_indices(3, i) =
            std::min<Eigen::Index>(n_total_samples - 1, scan_indices(1, i) + outer_context);
    }

    // When no pre/post samples exist at the observation boundary, keep the
    // legacy inner edge trim tied to the filter edge context only.
    if (inner_context > 0) {
        scan_indices(0, 0) =
            std::min<Eigen::Index>(scan_indices(0, 0) + inner_context,
                                   scan_indices(1, 0));
        scan_indices(1, n_scans - 1) =
            std::max<Eigen::Index>(scan_indices(0, n_scans - 1),
                                   scan_indices(1, n_scans - 1) - inner_context);
    }

    logger->info("scan_indices {}",scan_indices);
}

} // namespace engine
