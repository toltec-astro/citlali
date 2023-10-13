#include <boost/algorithm/string/trim.hpp>

#include <tula/logging.h>
#include <tula/algorithm/ei_stats.h>

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
            int obs_goal_size = vars.find("Header.Dcs.ObsGoal")->second.getDim(0).getSize();
            char obs_goal_char [obs_goal_size] = "";
            vars.find("Header.Dcs.ObsGoal")->second.getVar(&obs_goal_char);
            obs_goal = std::string(obs_goal_char);
        }

        std::string::iterator end_pos = std::remove(obs_goal.begin(), obs_goal.end(), ' ');
        obs_goal.erase(end_pos, obs_goal.end());

        // get mapping pattern
        vars.find("Header.Dcs.ObsPgm")->second.getVar(&obs_pgm_char);
        obs_pgm = std::string(obs_pgm_char);
        // try and remove end characters
        end_pos = std::remove(obs_pgm.begin(), obs_pgm.end(), ' ');
        obs_pgm.erase(end_pos, obs_pgm.end());

        // work around for files with bad ObsPgm
        if (!obs_pgm.find("Lissajous")) {
            obs_pgm = "Lissajous";
        }
        else if (!obs_pgm.find("Map")) {
            obs_pgm = "Map";
        }
        // if map pattern is unsupported
        else {
            logger->error("unsupported mapping pattern {}",obs_pgm);
            std::exit(EXIT_FAILURE);
        }
        // cannot reduce in lissajous mode if chunk less than or equal to zero
        if (std::strcmp("Lissajous", obs_pgm.c_str())==0 && time_chunk<=0) {
            logger->error("mapping mode is lissajous and time chunk size is zero");
            std::exit(EXIT_FAILURE);
        }

        // get source name
        int source_name_size = vars.find("Header.Source.SourceName")->second.getDim(0).getSize();
        char source_name_char [source_name_size] = "";

        vars.find("Header.Source.SourceName")->second.getVar(&source_name_char);
        source_name = std::string(source_name_char);
        // try and remove end characters
        end_pos = std::remove(source_name.begin(), source_name.end(), ' ');
        source_name.erase(end_pos, source_name.end());

        // loop through telescope data keys and populate vectors
        for (auto const& pair : tel_data_keys) {
            try {
                Eigen::Index n_pts = vars.find(pair.first)->second.getDim(0).getSize();
                tel_data[pair.second].resize(n_pts);
                vars.find(pair.first)->second.getVar(tel_data[pair.second].data());

            } catch (NcException &e) {
                logger->warn("cannot find {}", pair.first);
            }
        }

        // loop through telescope header keys and populate vectors
        for (auto const& pair : tel_header_keys) {
            // set for scalars
            Eigen::Index n_pts = 1;
            try {
                // try to get dimensions, otherwise keep n_pts at 1
                try {
                    n_pts = vars.find(pair.first)->second.getDim(0).getSize();
                } catch(...) {}

                tel_header[pair.second].resize(n_pts);
                vars.find(pair.first)->second.getVar(tel_header[pair.second].data());

            } catch (NcException &e) {
                // ignore if simulation
                if (!sim_obs) {
                    logger->warn("cannot find {}", pair.first);
                }
            }
        }

        // set tau 225 GHz
        tau_225_GHz = tel_header["Header.Radiometer.Tau"](0);

    } catch (NcException &e) {
        logger->warn("{}", e.what());
        throw DataIOError{fmt::format(
            "failed to load data from netCDF file {}", filepath)};
    }

    if (!sim_obs) {
        // convert TelUTC to unix time
        engine_utils::utc_to_unix(tel_data["TelUTC"],tel_header["Header.TimePlace.UTDate"]);

        // keys for fixing periodic boundary conditions
        std::vector<std::string> perodic_keys = {
            "TelRa","TelDec",
            "TelAzAct","TelElAct",
            "TelAzCor","TelElCor",
            "SourceAz","SourceEl",
        };

        // fix periodic boundary conditions
        for (const auto &key: perodic_keys) {
            engine_utils::fix_periodic_boundary(tel_data[key],pi,1.99*pi,2.0*pi);
        }
    }
}

void Telescope::calc_tan_pointing() {
    // get altaz tangent pointing
    calc_tan_altaz();

    // get icrs tangent pointing
    calc_tan_icrs();

    // set tangential projection to icrs
    if (std::strcmp("icrs", pixel_axes.c_str()) == 0) {
        logger->info("using icrs frame");
        tel_data["lat_phys"] = tel_data["dec_phys"];
        tel_data["lon_phys"] = tel_data["ra_phys"];
    }

    // set tangential projection to altaz
    else if (std::strcmp("altaz", pixel_axes.c_str()) == 0) {
        logger->info("using altaz frame");
        tel_data["lat_phys"] = tel_data["alt_phys"];
        tel_data["lon_phys"] = tel_data["az_phys"];
    }
}

void Telescope::calc_tan_icrs() {
    // size of data
    Eigen::Index n_pts = tel_data["TelRa"].size();

    // vectors to hold physical (tangent plane) coordinates
    tel_data["dec_phys"].resize(n_pts);
    tel_data["ra_phys"].resize(n_pts);

    // copy ra
    Eigen::VectorXd ra = tel_data["TelRa"];
    // copy dec
    Eigen::VectorXd dec = tel_data["TelDec"];

    // rescale ra
    (ra.array() > pi).select(tel_data["TelRa"].array() - 2.0*pi, tel_data["TelRa"].array());

    // center positions
    double center_ra = tel_header["Header.Source.Ra"](0);
    double center_dec = tel_header["Header.Source.Dec"](0);

    // rescale center ra
    center_ra = (center_ra > pi) ? center_ra - (2.0*pi) : center_ra;

    // denominator term
    Eigen::VectorXd cosc = sin(center_dec)*sin(dec.array()) +
                cos(center_dec)*cos(dec.array())*cos(ra.array() - center_ra);

    // calc tangent coordinates
    for (Eigen::Index i=0; i<n_pts; i++) {
        if (cosc(i)==0.) {
            tel_data["dec_phys"](i) = 0.;
            tel_data["ra_phys"](i) = 0.;
        }
        else {
            // tangent plane lat (dec)
            tel_data["dec_phys"](i) = (cos(center_dec)*sin(dec(i)) -
                                       sin(center_dec)*cos(dec(i))*cos(ra(i)-center_ra))/cosc(i);
            // tangent plane lon (ra)
            tel_data["ra_phys"](i) = cos(dec(i))*sin(ra(i)-center_ra)/cosc(i);
        }
    }
}

void Telescope::calc_tan_altaz() {
    // use loop to avoid annoying eigen aliasing issues with select
    for (Eigen::Index i=0; i<tel_data["TelAzAct"].size(); i++) {
        if ((tel_data["TelAzAct"](i) - tel_data["SourceAz"](i)) > 0.9*2.0*pi) {
            tel_data["TelAzAct"](i) = tel_data["TelAzAct"](i) - 2.0*pi;
        }
    }

    // tangent plane lat (alt)
    tel_data["alt_phys"] = tel_data["TelElAct"] - tel_data["SourceEl"] - tel_data["TelElCor"];

    // tangent plane lon (az)
    tel_data["az_phys"] = cos(tel_data["TelElAct"].array() -
                              tel_data["TelElCor"].array())*(tel_data["TelAzAct"].array() - tel_data["SourceAz"].array()) -
                          tel_data["TelAzCor"].array();

    // apply elevation corrections
    tel_data["TelElAct"] = tel_data["TelElAct"] - tel_data["TelElCor"];

    // apply azimuth corrections
    tel_data["TelAzAct"] = tel_data["TelAzAct"] - tel_data["TelAzCor"];
}

void Telescope::calc_scan_indices() {
    // number of scans
    Eigen::Index n_scans = 0;

    // get scans for raster pattern
    if (std::strcmp("Map", obs_pgm.c_str())==0 && !force_chunk) {
        logger->info("calculating scans for raster mode");

        // convert the hold signal to a bool
        Eigen::Matrix<bool,Eigen::Dynamic,1> hold_bool = tel_data["Hold"].template cast<bool>();

        // find where the change in the hold signal is 1 and increment scans
        for (Eigen::Index i=1; i<hold_bool.size(); i++) {
            if (hold_bool(i) - hold_bool(i-1) == 1) {
                n_scans++;
            }
        }

        // increment scan number if last element is zero
        if (hold_bool(hold_bool.size()-1) == 0){
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

        for (Eigen::Index i=1; i<hold_bool.size(); i++) {
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
        // populate final scan
        scan_indices(1,n_scans - 1) = hold_bool.size() - 1;
    }

    // get scan indices for Lissajous/Rastajous pattern
    else if (std::strcmp("Lissajous", obs_pgm.c_str())==0 || force_chunk) {
        logger->info("calculating scans for lissajous/rastajous mode");

        // index of first scan
        Eigen::Index first_scan_i = 0;
        // index of last scan
        Eigen::Index last_scan_i = tel_data["Hold"].size() - 1;

        // period (time_chunk x fsmp in seconds x Hz)
        Eigen::Index period_i = std::floor(time_chunk*fsmp);

        double period = std::floor(time_chunk*fsmp);

        // calculate number of scans
        n_scans = std::floor((last_scan_i - first_scan_i + 1)*1./period);

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

    // check for bad scans
    Eigen::Matrix<bool, Eigen::Dynamic, 1> is_bad_scan(n_scans);
    for (Eigen::Index i=0; i<n_scans; i++) {
        sum = 0;
        for (Eigen::Index j=scan_indices_temp(0,i); j<(scan_indices_temp(1,i)+1); j++) {
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
    for (Eigen::Index i=0; i<n_scans; i++) {
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
}

} // namespace engine
