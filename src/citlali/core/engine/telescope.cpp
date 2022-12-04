#include <tula/logging.h>

#include <citlali/core/engine/telescope.h>

namespace engine {

void Telescope::get_tel_data(std::string &filepath) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    try {
        // get telescope file
        NcFile fo(filepath, NcFile::read, NcFile::classic);
        auto vars = fo.getVars();

        // get mapping pattern
        vars.find("Header.Dcs.ObsPgm")->second.getVar(&obs_pgm_char);
        obs_pgm = obs_pgm_char;

        SPDLOG_INFO("obs_pgm {}", obs_pgm);
        SPDLOG_INFO("obs_pgm_char {}", obs_pgm_char);


        SPDLOG_INFO("obs_pgm len{}", obs_pgm.size());
        SPDLOG_INFO("obs_pgm find {}", obs_pgm.find("Map"));


        // work around for files with bad ObsPgm
        if (!obs_pgm.find("Lissajous")) {
            obs_pgm = "Lissajous";
        }

        else if (!obs_pgm.find("Map")) {
            obs_pgm = "Map";
        }

        SPDLOG_INFO("obs_pgm_after {}",obs_pgm);
        SPDLOG_INFO("obs_pgm_L {}",obs_pgm=="Lissajous");
        SPDLOG_INFO("obs_pgm_M{}",obs_pgm=="Map");


        if (std::strcmp("Lissajous", obs_pgm.c_str()) && time_chunk==0) {
            SPDLOG_ERROR("mapping mode is lissajous and time chunk size is zero");
            std::exit(EXIT_FAILURE);
        }

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
            // set for scalars
            Eigen::Index n_pts = 1;
            try {
                try {
                    n_pts = vars.find(pair.first)->second.getDim(0).getSize();
                } catch(...) {}

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

    // vectors to hold physical (tangent plane) coordinates
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
    Eigen::Index n_scans = 0;

    if (std::strcmp("Map", obs_pgm.c_str()) == 0) {
        SPDLOG_INFO("calculating scans for raster mode");
        // cast hold signal to boolean
        Eigen::Matrix<bool,Eigen::Dynamic,1> hold_bool = tel_data["Hold"].template cast<bool>();

        for (Eigen::Index i=1; i<hold_bool.size(); i++) {
            if (hold_bool(i) - hold_bool(i-1) == 1) {
                n_scans++;
            }
        }

        if (hold_bool(hold_bool.size()-1) == 0){
            n_scans++;
        }
        scan_indices.resize(4,n_scans);

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
        scan_indices(1,n_scans - 1) = hold_bool.size() - 1;
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
        n_scans = floor((last_scan_i - first_scan_i + 1)*1./period);

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

    Eigen::Matrix<bool, Eigen::Dynamic, 1> is_bad_scan(n_scans);
    for (Eigen::Index i=0; i<n_scans; i++) {
        sum = 0;
        for (Eigen::Index j=scan_indices_temp(0,i); j<(scan_indices_temp(1,i)+1); j++) {
            sum += 1;
        }
        if(sum < 2.*fsmp) {
            n_bad_scans++;
            is_bad_scan(i) = 1;
        }
        else {
            is_bad_scan(i) = 0;
        }
    }

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

    // set up the 3rd and 4th scan indices rows so that we don't lose data during lowpassing
    // inner_scans_chunk is zero if lowpassing is not enabled
    scan_indices.row(2) = scan_indices.row(0).array() - inner_scans_chunk;
    scan_indices.row(3) = scan_indices.row(1).array() + inner_scans_chunk;

    // set first and last outer scan positions to the same as inner scans
    scan_indices(2,0) = scan_indices(0,0);
    scan_indices(3,n_scans-1) = scan_indices(1,n_scans-1);

    scan_indices(0,0) = scan_indices(0,0) + inner_scans_chunk;
    scan_indices(1,n_scans-1) = scan_indices(1,n_scans-1) - inner_scans_chunk;
}

} // namespace engine
