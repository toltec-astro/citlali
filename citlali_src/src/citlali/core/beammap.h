#pragma once

#include <mutex>

#include "config.h"
#include "timestream/timestream.h"
#include "map/map_utils.h"
#include "map/map.h"
#include "fitting.h"
#include "gaussfit.h"
#include "result.h"


// TCData is the data structure of which RTCData and PTCData are a part
using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// Selects the type of TCData
using timestream::LaliDataKind;

namespace beammap {

class Beammap : public lali::Result {
public:
    // Config file
    YamlConfig config;

    lali::TelData telMD;

    // Total number of detectors and scans
    int n_detectors, nscans, obsid;

    // Sample rate
    double samplerate;

    std::string ex_name;
    double ncores;

    // Lowpass+Highpass filter class
    timestream::Filter filter;

    // Class to hold and populate maps
    mapmaking::MapStruct Maps;
    // Class to hold and populate coadded maps
    // mapmaking::CoaddedMapStruct CoaddedMaps;

    // Number of threads to use for the grppi::farm
    Eigen::Index nThreads = Eigen::nbThreads();

    // std::map for the detector Az/El offsets
    std::unordered_map<std::string, Eigen::VectorXd> offsets;

    // Eigen Vector for nws
    Eigen::VectorXd nw;

    // Eigen Vector for array names
    Eigen::VectorXd array_name;

    Eigen::VectorXd fluxscale;

    // Array indices
    std::vector<std::tuple<int,int>> array_index;

    // Detector indices
    std::vector<std::tuple<int,int>> det_index;

    // Vector for ptcs
    std::vector<TCData<LaliDataKind::PTC,Eigen::MatrixXd>> ptcs;

    // Mutex for populating ptcs in farm
    std::mutex farm_mutex;

    // Vectors for grppi maps
    std::vector<int> si, so, deti, deto;

    // Beammap iteration parameters
    Eigen::Index iteration, max_iterations;

    // Cutoff
    double cutoff;

    // n_params
    int n_params;

    // Vector for convergence check
    Eigen::Matrix<bool, Eigen::Dynamic, 1> converged;

    // Fitted Parameters
    Eigen::MatrixXd fittedParams, fittedParams_0;

    void setup();
    auto runTimestream();
    auto runLoop();

    template <typename Derived, class C, class RawObs>
    auto timestreamPipeline(Eigen::DenseBase<Derived> &, C &, RawObs &);

    template <typename Derived, class C, class RawObs>
    auto loopPipeline(Eigen::DenseBase<Derived> &, C &, RawObs &);

    template <typename Derived, class C, class RawObs>
    auto pipeline(Eigen::DenseBase<Derived> &, C &, RawObs &);

    void output();

};

// Sets up the map dimensions and lowpass+highpass filter
void Beammap::setup() {

  // Check if lowpass+highpass filter requested.
  // If so, make the filter once here and reuse
  // it later for each RTCData.

  if (config.get_typed<bool>(std::tuple{"tod", "filter", "enabled"})) {
    auto fLow = config.get_typed<double>(std::tuple{"tod", "filter", "flow"});
    auto fHigh = config.get_typed<double>(std::tuple{"tod", "filter", "fhigh"});
    auto aGibbs = config.get_typed<double>(std::tuple{"tod", "filter", "agibbs"});
    auto nTerms = config.get_typed<int>(std::tuple{"tod", "filter", "nterms"});

    filter.makefilter(fLow, fHigh, aGibbs, nTerms, samplerate);
  }

}

// Runs the timestream -> map analysis pipeline.
auto Beammap::runTimestream(){

    auto farm =  grppi::farm(nThreads,[&](auto in) -> TCData<LaliDataKind::PTC,Eigen::MatrixXd> {

        SPDLOG_INFO("Starting rtcproc {}", in.index.data);

        /*Stage 1: RTCProc*/
        RTCProc rtcproc(config);
        TCData<LaliDataKind::PTC,Eigen::MatrixXd> out;
        rtcproc.run(in, out, this);

        // push the ptc into the ptc vector
        {
            std::scoped_lock lock(farm_mutex);
            ptcs.push_back(std::move(out));
        }

        SPDLOG_INFO("Done with timestream processing for scan {}", out.index.data);

        return out;
    });

    return farm;

}

auto Beammap::runLoop() {

    auto loop = grppi::repeat_until([&](auto in) {

        // Make working copy of ptc vector.  Overwrite it on each loop.
        std::vector<TCData<LaliDataKind::PTC,Eigen::MatrixXd>> ptcs_working(ptcs);

        PTCProc ptcproc(config);
        grppi::map(grppiex::dyn_ex(ex_name), si, so, [&](auto s) {
            SPDLOG_INFO("s {}", s);
            SPDLOG_INFO("ptcs[s].scans.data {}", ptcs_working[s].scans.data);

            /*Subtract Gaussian for iterations > 0*/
            if (iteration > 0) {
                SPDLOG_INFO("Subtracting Gaussian");
                SPDLOG_INFO("fittedParams_0 {}", fittedParams_0);
                timestream::addGaussian(ptcs_working[s], offsets, fittedParams_0, config);
            }

            /*Stage 2: PTCProc*/
            SPDLOG_INFO("Running PTCProc");
            ptcproc.run(ptcs_working[s], ptcs_working[s], this);

            /*Add Gaussian for iterations > 0*/
            if (iteration > 0) {
                SPDLOG_INFO("Adding Gaussian");
                fittedParams_0.row(0) = -fittedParams_0.row(0);
                timestream::addGaussian(ptcs_working[s], offsets, fittedParams_0, config);
            }

            /*Stage 3 Populate Map*/
            SPDLOG_INFO("Populating Maps");
            Maps.mapPopulate(ptcs_working[s], offsets, config, det_index);

            return 0;});

        SPDLOG_INFO("Normalizing Maps by Weight Map");
        {
            logging::scoped_timeit timer("mapNormalize()");
            Maps.mapNormalize(config);
        }

        /*Stage 4 Fit Maps to Gaussian*/
        SPDLOG_INFO("Fitting maps");
        grppi::map(grppiex::dyn_ex(ex_name), deti, deto, [&](auto d) {

            if (converged(d) == 0) {
                Eigen::VectorXd init_p;
                init_p.setZero(6);

                // Set limits for fitting (temp)
                MatrixXd::Index maxRow, maxCol;
                double max = Maps.signal[d].maxCoeff(&maxRow, &maxCol);
                init_p(0) = max; // amp
                init_p(1) = Maps.rcphys(maxRow); // offset_y
                init_p(2) = Maps.ccphys(maxCol); // offset_x
                init_p(3) = 5.0*RAD_ASEC; // fwhm_y
                init_p(4) = 5.0*RAD_ASEC; // fwhm_x
                init_p(5) = pi/4.; // ang

                Eigen::MatrixXd limits(n_params, 2);
                limits.row(0) << 0, max;
                limits.row(1) << Maps.rcphys.minCoeff(), Maps.rcphys.maxCoeff();
                limits.row(2) << Maps.ccphys.minCoeff(), Maps.ccphys.maxCoeff();
                limits.row(3) << 0, 10.0*RAD_ASEC;
                limits.row(4) << 0, 10.0*RAD_ASEC;
                limits.row(5) << 0, pi/2.;

                SPDLOG_INFO("init_p {}", init_p);

                auto g = gaussfit::modelgen<gaussfit::Gaussian2D>(init_p);
                auto _p = g.params;
                auto xy = g.meshgrid(Maps.ccphys, Maps.rcphys);

                Eigen::Map<Eigen::MatrixXd> sigma(Maps.weight[d].data(), Maps.weight[d].rows(), Maps.weight[d].cols());
                (sigma.array() !=0).select(0, 1./sqrt(sigma.array()));
                auto g_fit = gaussfit::curvefit_ceres(g, _p, xy, Maps.signal[d], sigma, limits);

                fittedParams.col(d) = _p;
            }

            return 0;});
        SPDLOG_INFO("Done with fitting maps");

        return in;

    },

    [&](auto in) {
        SPDLOG_INFO("checking convergence");
        bool complete = 0;
        iteration++;
        if (iteration < max_iterations) {
            // do check for fit
            if ((converged.array() == 1).all()) {
                complete = 1;
            }

            else {
                grppi::map(grppiex::dyn_ex(ex_name), deti, deto, [&](auto d) {
                    if (converged(d) == 0) {
                        auto ratio = abs((fittedParams.col(d).array() - fittedParams_0.col(d).array())/fittedParams_0.col(d).array());
                        if ((ratio.array() <= cutoff).all()) {
                            converged(d) = 1;
                        }
                    }

                    return 0;});
            }

            fittedParams_0 = fittedParams;
        }

        else {
            complete = 1;
        }

        SPDLOG_INFO("complete {}", complete);
        return complete;
    });

    return loop;
}

template <typename Derived, class C, class RawObs>
auto Beammap::timestreamPipeline(Eigen::DenseBase<Derived> &scanindicies, C &kidsproc, RawObs &rawobs) {

    // do grppi reduction
    grppi::pipeline(grppiex::dyn_ex(ex_name),
        [&]() -> std::optional<TCData<LaliDataKind::RTC, Eigen::MatrixXd>> {
        // Variable for current scan
        static auto scan = 0;
        // Current scanlength
        Eigen::Index scanlength;
        // Index of the start of the current scan
        Eigen::Index si = 0;

        while (scan < scanindicies.cols()) {

            // First scan index for current scan
            si = scanindicies(2, scan);
            SPDLOG_INFO("si {}", si);
            // Get length of current scan (do we need the + 1?)
            scanlength = scanindicies(3, scan) - scanindicies(2, scan) + 1;
            SPDLOG_INFO("scanlength {}", scanlength);

            // Declare a TCData to hold data
            // predefs::TCData<predefs::LaliDataKind::RTC, Eigen::MatrixXd> rtc;
            TCData<LaliDataKind::RTC, Eigen::MatrixXd> rtc;

            // Get scan indices and push into current RTC
            rtc.scanindex.data = scanindicies.col(scan);

            // This index keeps track of which scan the RTC actually belongs to.
            rtc.index.data = scan + 1;

            // Get telescope pointings for scan (move to Eigen::Maps to save
            // memory and time)

            // Get the requested map type
            auto maptype = config.get_str(std::tuple{"map","type"});
            SPDLOG_INFO("mapy_type {}", maptype);

            // Put that scan's telescope pointing into RTC
            if (std::strcmp("RaDec", maptype.c_str()) == 0) {
                rtc.telLat.data = telMD.telMetaData["TelDecPhys"].segment(si, scanlength);
                rtc.telLon.data = telMD.telMetaData["TelRaPhys"].segment(si, scanlength);
            }

            else if (std::strcmp("AzEl", maptype.c_str()) == 0) {
                rtc.telLat.data = telMD.telMetaData["TelAzPhys"].segment(si, scanlength);
                rtc.telLon.data = telMD.telMetaData["TelElPhys"].segment(si, scanlength);
            }

            rtc.telElDes.data = telMD.telMetaData["TelElDes"].segment(si, scanlength);
            rtc.ParAng.data = telMD.telMetaData["ParAng"].segment(si, scanlength);

            rtc.scans.data.resize(scanlength, n_detectors);
            rtc.scans.data = kidsproc.populate_rtc(rawobs, rtc.scanindex.data, scanlength, n_detectors);

            rtc.flags.data.resize(scanlength, n_detectors);
            rtc.flags.data.setOnes();

            // Increment scan
            scan++;

            return rtc;
        }
        return {};

    },
        runTimestream());
}

template <typename Derived, class C, class RawObs>
auto Beammap::loopPipeline(Eigen::DenseBase<Derived> &scanindicies, C &kidsproc, RawObs &rawobs) {

    max_iterations = config.get_typed<int>(std::tuple{"beammap","max_iterations"});
    cutoff = config.get_typed<double>(std::tuple{"beammap","cutoff"});
    iteration = 0;
    n_params = 6;
    converged.setZero(n_detectors);

    // Vectors of size nscans for Grppi Map (move)
    si.resize(ptcs.size());
    std::iota(si.begin(), si.end(), 0);
    so.resize(ptcs.size());

    // Vectors of size n_detectors for Grppi Map (move)
    deti.resize(n_detectors);
    std::iota(deti.begin(), deti.end(), 0);
    deto.resize(n_detectors);

    fittedParams.resize(n_params, n_detectors);
    fittedParams_0.resize(n_params, n_detectors);
    fittedParams_0.setConstant(std::nan(""));

    // do grppi reduction
    grppi::pipeline(grppiex::dyn_ex(ex_name),
        [&]() -> std::optional<Eigen::Index> {

         static auto placeholder = 0;
         while (placeholder < 1) {
               return placeholder++;
         }

         return {};

        },

        runLoop(),

        [&](auto in) {
            // might not need
        return in;
        });
}

template <typename Derived, class C, class RawObs>
auto Beammap::pipeline(Eigen::DenseBase<Derived> &scanindicies, C &kidsproc, RawObs &rawobs) {

    ex_name = config.get_str(std::tuple{"runtime","policy"});
    ncores = config.get_typed<int>(std::tuple{"runtime","ncores"});

    timestreamPipeline(scanindicies, kidsproc, rawobs);
    SPDLOG_INFO("Done with beammap timestream pipeline");
    loopPipeline(scanindicies, kidsproc, rawobs);
}

void Beammap::output() {
    std::string filepath = config.get_str(std::tuple{"runtime","output_filepath"});
    for (int i = 0; i < array_index.size(); i++) {
        SPDLOG_INFO("array_index {}", i);
        auto filename = composeFilename<lali::TolTEC, lali::Simu, lali::Beammap>(this);
        writeMapsToFITS(this, filepath, filename, i, det_index);
    }
}


} // namespace
