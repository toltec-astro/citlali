#include <Eigen/Dense>
#include <boost/exception/diagnostic_information.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <grppi/grppi.h>
#include <netcdf>
#include <omp.h>
#include <yaml-cpp/yaml.h>

#include "../common_utils/src/utils/eigen.h"
#include "../common_utils/src/utils/enum.h"
#include "../common_utils/src/utils/formatter/enum.h"
#include "../common_utils/src/utils/formatter/matrix.h"
#include "../common_utils/src/utils/formatter/utils.h"
#include "../common_utils/src/utils/grppiex.h"
#include "../common_utils/src/utils/logging.h"
#include "kids/core/kidsdata.h"

#include "core/TCData.h"
#include "core/lali.h"

// namespaces
namespace po = boost::program_options;
namespace pt = boost::property_tree;
//namespace ln = lali;

// TcData is the data structure of which RTCData and PTCData are a part
using timestream::TCData;

// Selects the type of TCData
using timestream::LaliDataKind;

// Command line parser
void po2pt(po::variables_map &in, pt::ptree &out)
{
    for (auto it = in.begin(); it != in.end(); ++it) {
        const auto &t = it->second.value().type();
        if ((t == typeid(int)) || (t == typeid(size_t))) {
            out.put<int>(it->first, in[it->first].as<int>());
        } else if ((t == typeid(float)) || (t == typeid(double))) {
            out.put<double>(it->first, in[it->first].as<double>());
        } else if (t == typeid(std::string)) {
            out.put<std::string>(it->first, in[it->first].as<std::string>());
        } else {
            throw std::runtime_error(fmt::format("unknown type in config: {}", t.name()));
        }
    }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[])
{
    SPDLOG_INFO("Starting Lali");
    logging::scoped_timeit timer("Lali");

    /*
  Declare Lali class *testing (to be moved outside of lali.cpp)*
  */

    SPDLOG_INFO("Making Lali Class");
    lali::Lali LC;

    /*
  Get Test data *testing*
  */
    // LC.makeTestData<lali::Tester::UseAzTEC>(argc, argv);
    /*
  Setup lali *testing (to be moved outside of lali.cpp)*
      - Calculate map dimensions
      - Allocate map matrices/tensors
      - Calculates Lowpass+Highpass Filter if requested
  */

    /*
  Begin main pipeline
      - Timestream Processing Stage 1 (Despking, Replacing Flagged Data,
  Lowpassing and Highpassing, Downsampling, Calibration
      - Timestream Processing Stage 2 (PCA Clean)
      - Mapmakaing Stage 1 (Populate map pixels)
  */

    // Get GRPPI execution policy (seq, omp, tbb, etc.)
    auto ex_name = LC.config->get_typed<std::string>("pipeline_ex_policy");

    // Get map type (Ra & Dec, Az & El, etc)
    std::string maptype = LC.config->get_typed<std::string>("mapType").c_str();

    auto data_type = LC.config->get_str("data_type");
    if (data_type == "toltec") {
        LC.init_from_cli<lali::DataType::TolTEC>(argc, argv);
        // open the kids files here
        namespace kids_spec = kids::toltec;
        auto kidsdata = kids_spec::read_data<>,
                                  rc.get_str("source"));
        // solve kidsdata in its entirety
        // and populate scans_data
        LC.Data.scans.resize(10, 10);
        LC.ndetectors = scans_data.cols();
        std::vector<std::string> detector_names(LC.ndetectors);
    } else if (data_type == "aztec") {
        LC.init_from_cli<lali::DataType::AzTEC>(argc, argv);
    }
    SPDLOG_INFO("Setting up Maps and Variables");
    {
        logging::scoped_timeit timer("setup()");
        LC.setup();
        // populate detector offsets.
    }

    SPDLOG_INFO("Starting Main Pipeline in {} mode", ex_name);
    {
        logging::scoped_timeit timer("Main pipeline");
        grppi::pipeline(
            grppiex::dyn_ex(ex_name),
            [&]() -> std::optional<TCData<LaliDataKind::RTC, Eigen::MatrixXd>> {
                // Variable for current scan
                static auto scan = 0;
                // Current scanlength
                Eigen::Index scanlength;
                // Index of the start of the current scan
                Eigen::Index si = 0;

                // Loop through scans
                while (scan < LC.nscans) {
                    SPDLOG_INFO("----------------------------------------------------");
                    SPDLOG_INFO("Starting timestream -> map reduction for scan {}/{}",
                                scan + 1,
                                LC.nscans);
                    SPDLOG_INFO("----------------------------------------------------");
                    // Declare a TCData to hold data
                    TCData<LaliDataKind::RTC, Eigen::MatrixXd> rtc;

                    // First scan index for current scan
                    si = LC.Data.scanindex(2, scan);

                    // Get length of current scan (do we need the + 1?)
                    scanlength = LC.Data.scanindex(3, scan) - LC.Data.scanindex(2, scan) + 1;

                    // Get reference to BeammapData and put into the current RTC
                    if (auto data_type = LC.config->get_str("data_type");
                        data_type == "aztec") {
                        rtc.scans.data = LC.Data.scans.block(si, 0, scanlength,
                                                             LC.ndetectors);
                    } else if (data_type == "toltec") {
                        rtc.scans.data =
                            scans_data.block(si, 0, scanlength, ndetectors);
                    }

                    // Make flag matrix
                    rtc.flags.data.resize(rtc.scans.data.rows(),
                                          rtc.scans.data.cols());
                    rtc.flags.data.setOnes();

                    // Get scan indices and push into current RTC
                    rtc.scanindex.data = LC.Data.scanindex.col(scan);

                    // This index keeps track of which scan the RTC actually belongs to.
                    rtc.index.data = scan + 1;

                    // Get telescope pointings for scan (move to Eigen::Maps to save
                    // memory and time)
                    if (std::strcmp("RaDec", maptype.c_str()) == 0) {
                        rtc.telLat.data = LC.Data.telMetaData["TelRaPhys"].segment(si, scanlength);
                        rtc.telLon.data = LC.Data.telMetaData["TelDecPhys"].segment(si, scanlength);
                    }

                    else if (std::strcmp("AzEl", maptype.c_str()) == 0) {
                        rtc.telLat.data = LC.Data.telMetaData["TelAzPhys"].segment(si, scanlength);
                        rtc.telLon.data = LC.Data.telMetaData["TelElPhys"].segment(si, scanlength);
                    }

                    rtc.telElDes.data = LC.Data.telMetaData["TelElDes"].segment(si, scanlength);
                    rtc.ParAng.data = LC.Data.telMetaData["ParAng"].segment(si, scanlength);

                    // Increment scan
                    scan++;

                    // return RTC for the grppi:farm in Lali::run()
                    return rtc;
                }
                return {};
            },

            // Run the timestream - > map analysis.  This command contains a GRPPI farm that
            // parallelizes over the specified number of cores.
            LC.run());
    }

    /*
  Normalize maps
      - Mapmaking Stage 3 (Divide by weight map)
  */

    SPDLOG_INFO("Normalizing Maps by Weight Map");
    {
        logging::scoped_timeit timer("mapNormalize()");
        LC.Maps.mapNormalize();
    }

    /*
  Run after-mapmaking analyses
      - PSD
      - Histogram
  */

    /*
  Coadd maps
  */

    /*
  Run Wiener Filter
  */

    /*
  Additional Map Analyses
  */

    /*
  Output
  */

    SPDLOG_INFO("Outputing Maps to netCDF File");
    {
        logging::scoped_timeit timer("output()");
        LC.output(LC.config, LC.Maps);
    }

    return 0;
}
