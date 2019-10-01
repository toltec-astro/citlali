#include <boost/exception/diagnostic_information.hpp>
#include <boost/program_options.hpp> // po::options_description, po::variables_map, ...
#include <boost/property_tree/ptree.hpp> // pt::ptree
#include <boost/random.hpp>

#include <Eigen/Dense>

#include <grppi/grppi.h>
#include <iostream>
#include <fstream>

#include "../common_utils/src/utils/config.h"
#include "../common_utils/src/utils/enum.h"
#include "../common_utils/src/utils/formatter/enum.h"
#include "../common_utils/src/utils/formatter/matrix.h"
#include "../common_utils/src/utils/formatter/utils.h"
#include "../common_utils/src/utils/logging.h"
#include "../common_utils/src/utils/grppiex.h"

#include "../core/timestream/read.h"
#include "../core/timestream/timestream.h"
#include "../core/map/mapmaking.h"
#include "../core/timestream/mapresult.h"
#include "../core/map/map.h"

#include "../core/timestream/rtcdata.h"

#include <yaml-cpp/yaml.h>

// namespaces
namespace po = boost::program_options;
namespace pt = boost::property_tree;

void po2pt(po::variables_map &in, pt::ptree &out) {

    for (auto it = in.begin(); it != in.end(); ++it) {
        const auto &t = it->second.value().type();
        if ((t == typeid(int)) || (t == typeid(size_t))) {
            out.put<int>(it->first, in[it->first].as<int>());
        } else if ((t == typeid(float)) || (t == typeid(double))) {
            out.put<double>(it->first, in[it->first].as<double>());
        } else if (t == typeid(std::string)) {
            out.put<std::string>(it->first, in[it->first].as<std::string>());
        } else {
            throw std::runtime_error(
                fmt::format("unknown type in config: {}", t.name()));
        }
    }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
{

   logging::scoped_timeit timer("lali");

    using config::Config;
    using RC = pt::ptree;
    RC rc; // stores config
    SPDLOG_INFO("start lali process");
    try {
        po::options_description opts_desc{"Options"};
        opts_desc.add_options()("help,h", "Help screen")(
            "config_file,c", po::value<std::vector<std::string>>()
                                 ->multitoken()
                                 ->zero_tokens()
                                 ->composing())(
            "input_file,i", po::value<std::vector<std::string>>()
                                ->multitoken()
                                ->zero_tokens()
                                ->composing());
        po::positional_options_description inputs_desc{};
        inputs_desc.add("input_file", -1);
        po::command_line_parser parser{argc, argv};
        parser.options(opts_desc).positional(inputs_desc).allow_unregistered();
        po::variables_map vm;
        auto parsed = parser.run();
        po::store(parsed, vm);
        po::notify(vm);
        // handle command lines
        if (vm.empty() || vm.count("help")) {
            std::cout << opts_desc << '\n';
            return 0;
        }
        std::vector<std::string> config_files;
        if (vm.count("config_file")) {
            config_files = vm["config_file"].as<std::vector<std::string>>();
        }
        SPDLOG_INFO("number of config files: {}", config_files.size());
        for (const auto &f : config_files) {
            SPDLOG_INFO("   {}", f);
        }
        std::vector<std::string> input_files;
        if (vm.count("input_file")) {
            input_files = vm["input_file"].as<std::vector<std::string>>();
        }
        SPDLOG_INFO("number of input files: {}", input_files.size());
        for (const auto &f : input_files) {
            SPDLOG_INFO("   {}", f);
        }
        auto unparsed_opts =
            collect_unrecognized(parsed.options, po::exclude_positional);
        for (const auto &opt : unparsed_opts) {
            SPDLOG_INFO("unparsed options: {}", opt);
        }
        // setup config

        auto config = std::make_shared<Config>();

        YAML::Node root;
        try {
            root = YAML::LoadFile(config_files.front());
        } catch (const std::exception& e){
            SPDLOG_ERROR("{}", e.what());
            return 0;
        }

        for (YAML::const_iterator it = root.begin(); it != root.end(); ++it){
            if (aztec::is_number<int>(it->second.as<std::string>())){
                config->set(it->first.as<std::string>(),it->second.as<int>());
            }
            else if (aztec::is_number<double>(it->second.as<std::string>())){
                config->set(it->first.as<std::string>(),it->second.as<double>());
            }
            else{
                config->set(it->first.as<std::string>(),it->second.as<std::string>());
            }
        }

        //SPDLOG_INFO("use config: {}", config->pprint());
        //auto ex_name = config->get_typed<std::string>("pipeline.ex_policy");
        SPDLOG_INFO("available grppi execution modes:");
        //for (const auto &name : grppiex::) {
        //    SPDLOG_INFO("  {}", name);
        //}

        //SPDLOG_INFO("use grppi execution mode: {}", ex_name);

        using aztec::BeammapData;
        using aztec::mapResult;
        using aztec::DataIOError;

        //declare timestream classes before pipeline
        using timestream::RTCData;
        using timestream::PTCData;
        using timestream::KTCData;

        using timestream::RTCProc;
        using timestream::PTCProc;

        using timestream::LaliDataKind;
        //using timestream::KTCProc;

        std::vector<BeammapData> bds;
        std::vector<RTCData<LaliDataKind::SolvedTimeStream>> rtcs;
        std::vector<PTCData> ptcs;
        std::vector<mapResult> brs;

        std::mutex farm_mutex;

        static auto it = input_files.begin();

        //if bds vector is empty, get first BeammapData
        if (bds.empty()) {
            try{
            bds.push_back(BeammapData::fromNcFile(*(it)));
            }
            catch (const DataIOError &e) {
                SPDLOG_WARN("failed to read input {}: {}", *it, e.what());
            }
        }


        mapResult br;
        brs.push_back(std::move(br));

        //get number of detectors for later stages
        Eigen::Index ndet = bds.back().meta.template get_typed<int>("ndetectors");
        Eigen::Index nscans = bds.back().meta.template get_typed<int>("nscans");

        //Vectors for grppi::map iteration over detectors
        std::vector<int> detvec_in(ndet);
        std::iota(detvec_in.begin(), detvec_in.end(), 0);
        std::vector<int> detvec_out(ndet);

        //get config parameters for map generation
        auto pixelsize = config->get_typed<double>("pixel_size");
        auto mgrid_0 = config->get_typed<double>("mgrid0");
        auto mgrid_1 = config->get_typed<double>("mgrid1");

        auto samplerate = config->get_typed<double>("proc.rtc.samplerate");
        auto NNoiseMapsPerObs = config->get_typed<int>("noisemaps");

        brs.back().mapstruct.NNoiseMapsPerObs = NNoiseMapsPerObs;

        auto dsf = config->get_typed<int>("proc.rtc.downsample.downsamplefactor");

        boost::random::mt19937 rng;
        boost::random::uniform_int_distribution<> rands(0,1);

        Eigen::MatrixXi noisemaps = Eigen::MatrixXi::Zero(NNoiseMapsPerObs,1).unaryExpr([&](float dummy){return rands(rng);});

        noisemaps = (2.*(noisemaps.template cast<double>().array() - 0.5)).template cast<int>();

        SPDLOG_INFO("noisemaps {}",noisemaps);

        Eigen::MatrixXd offsets(2,ndet);
        offsets.setZero();

        offsets.row(0) << 15.4014410358,
                -7.09925493928,
                3.28362411614,
                30.316091789,
                25.5047214344,
                -6.84755689507,
                1.25557028847,
                -10.292873605,
                3.61305888678,
                34.538926963,
                23.100137326,
                11.9118760111,
                20.6256093364,
                37.5504845182,
                27.3281593083,
                18.4751081082,
                -43.964367463,
                -34.7160029766,
                -19.0780542196,
                -14.454087723,
                -5.73135909449,
                -32.6377892746,
                -34.100136509,
                -45.5622233329,
                5.04175221157,
                -26.3020460319,
                -25.8227091051,
                -9.35565364188,
                -12.7121739759,
                -36.7257591784,
                -28.9739563833,
                -21.0646997651,
                -16.0204743959,
                -16.5148289057,
                -30.1461307463,
                -52.0049081401,
                -35.9063095535,
                -36.736219045,
                -20.0278942775,
                -30.6141506637,
                -13.7715287325,
                -41.6224018127,
                -4.30312893782,
                -61.5385716709,
                -28.0588888576,
                -45.9811919777,
                -44.9377503458,
                -29.7528333112,
                -25.8538297088,
                -32.0274789884,
                -46.7244849525,
                -38.4041001013,
                -40.1669632242,
                -47.8618110227,
                -45.6588655581,
                -46.8719292253,
                -14.6955638653,
                12.551929275,
                2.44521821172,
                9.03770669574,
                -8.71167148389,
                -2.24032504659,
                -26.7320980369,
                -24.8789310376,
                11.3563144863,
                3.73827504253,
                13.2760069991,
                1.12828596536,
                7.61928805035,
                -34.3143769143,
                -9.66109300501,
                -18.7748515652,
                -6.12134931249,
                -7.80055090672,
                -17.1354542796,
                -35.0694724727,
                -25.7185774167,
                0.0487497427239,
                -15.9182630556,
                49.3288867463,
                49.6573664538,
                44.8943402419,
                31.2370027414,
                26.00988556,
                17.6222358798,
                20.2595246685,
                39.959463486,
                34.5557920845,
                22.45184644,
                31.5170349711,
                16.9985008187,
                18.4931098204,
                36.6994277814,
                27.9524132214,
                22.1121914953,
                29.8716178756,
                24.6184964325,
                36.9034224405,
                65.4123094785,
                38.8854309866,
                24.308153359,
                50.5922023325,
                61.1510172833,
                33.0451787677,
                55.4985582922,
                50.0436553762,
                35.435536381,
                18.0305206365,
                65.6514577941,
                30.3074437166,
                62.5065484941,
                61.5870911633,
                61.86613222;

        offsets.row(1) << 62.6053096671,
                51.6000098253,
                48.7957194431,
                38.2628472687,
                59.0161576063,
                40.950323707,
                59.1124495018,
                62.189502502,
                38.280291344,
                55.8376707684,
                31.153958332,
                45.5737955543,
                41.1042482647,
                45.7493745897,
                48.3064045663,
                51.9050183807,
                16.2352995574,
                23.4619401689,
                27.1463950433,
                44.6678840766,
                3.91128275811,
                50.8297429167,
                13.1265396764,
                26.4638689542,
                1.46936738531,
                19.9637479906,
                58.0954840218,
                24.0012009194,
                34.194037801,
                33.59215088,
                30.3207601894,
                37.5970299882,
                55.0962807141,
                16.9162989174,
                40.5903331785,
                -16.9883036725,
                -31.9189990558,
                -22.3227256625,
                -18.602613809,
                -6.04585348015,
                -2.28865826688,
                6.31408352429,
                -5.70931660084,
                -0.949807188211,
                -25.3744264074,
                -29.0879038325,
                -38.7806910169,
                -15.9887673347,
                0.527455510168,
                3.41702324765,
                -19.595567651,
                -12.9287447015,
                -3.63940472041,
                9.59762543981,
                -0.639365170668,
                -10.2989491451,
                -57.3658581623,
                -47.7706302742,
                -44.3729474574,
                -28.1043562104,
                -31.9186981265,
                -15.3410513902,
                -35.1040429057,
                -54.1894767544,
                -37.803459876,
                -54.044066086,
                -57.5772531322,
                -34.6749842593,
                -17.9498104069,
                -50.4992419132,
                -21.6940251966,
                -28.4047254139,
                -51.1936055277,
                -41.4772188193,
                -38.0446666686,
                -41.2977740477,
                -44.4768033981,
                -24.8860847951,
                -47.3335658961,
                -40.8332178006,
                -14.4215212364,
                -21.4878908465,
                -8.87077225222,
                -25.7342128816,
                -12.1735088488,
                -42.2475804117,
                -47.8793297395,
                -18.964429934,
                -5.57233993579,
                -54.638317073,
                -22.2158606047,
                -32.3770845558,
                -28.5527111494,
                -35.5826511446,
                -51.7970444917,
                -45.0566160318,
                -15.9686293027,
                -38.5110721165,
                19.7639394576,
                35.5343910653,
                20.9450339128,
                -3.99324532952,
                10.8011567492,
                28.2363452294,
                33.8527174653,
                43.5487068209,
                18.2534158968,
                13.9601413153,
                19.7691634542,
                1.46458021012,
                29.9558787063,
                -7.17151341041,
                2.76412325287;

        //offsets.row(0) = offsets.row(0).array() - offsets(0,16);
        //offsets.row(1) = offsets.row(1).array() - offsets(1,16);

        int n = config->get_typed<int>("proc.rtc.cores");
        int n2 = config->get_typed<int>("proc.ptc.cores");
        int n3 = config->get_typed<int>("proc.map.cores");

        brs.back().mapstruct.pixelsize = pixelsize;

         //Detector Lat and Lon
         Eigen::VectorXd lat, lon;
         double maxlat = 0;
         double minlat = 0;
         double maxlon = 0;
         double minlon = 0;

         //Get max and min lat and lon values out of all detectors.  Maybe parallelize?
         for (Eigen::Index i=0;i<ndet;i++) {
             mapmaking::internal::getPointing(bds.back().telescope_data, lat, lon, offsets,i);
             if(lat.maxCoeff() > maxlat){
                 maxlat = lat.maxCoeff();
             }
             if(lat.minCoeff() < minlat){
                 minlat = lat.minCoeff();
             }
             if(lon.maxCoeff() > maxlon){
                 maxlon = lon.maxCoeff();
             }
             if(lon.minCoeff() < minlon){
                 minlon = lon.minCoeff();
             }
         }

         //Get nrows, ncols
         mapmaking::internal::getRowCol(brs.back().mapstruct, mgrid_0, mgrid_1, maxlat, minlat, maxlon, minlon);

         //MapStruct class function to resize map tensors.
         brs.back().mapstruct.resize(ndet);

         brs.back().mapstruct.noisemaps.resize(NNoiseMapsPerObs,brs.back().mapstruct.nrows,brs.back().mapstruct.ncols);
         brs.back().mapstruct.noisemaps.setZero();

         //Parameter matrix
         brs.back().pp.resize(6,ndet);
         brs.back().pp.setOnes();

        grppi::pipeline(grppiex::dyn_ex("omp"), [&]() -> std::optional<RTCData<LaliDataKind::SolvedTimeStream>> {

            //scan index variable
            static auto x = 0;
            Eigen::Index scanlength = 0;
            int sdet = 0;
            Eigen::Index si = 0;
            int ns = 0;

            ns = bds.back().meta.template get_typed<int>("nscans");

            //ns = 180; //temp

            while (x != ns){
                SPDLOG_INFO("On scan {}/{}",x+1,ns);
                RTCData<LaliDataKind::SolvedTimeStream> rtc;
                //Put RTC at the end of the rtcs vector
                rtcs.push_back(rtc);
                //first scan index for current scan
                si = bds.back().scanindex(2,x);

                //si = bds.back().scanindex(2,0); //temp

                scanlength = bds.back().scanindex(3,x) - bds.back().scanindex(2,x);

                //scanlength = 4882; //temp

                sdet = bds.back().meta.template get_typed<int>("ndetectors");

                //get reference to BeammapData and put into the current RTC
                rtcs.back().scans.data = bds.back().scans.block(si,0,scanlength,sdet);
                //Get scan indices and push into current RTC
                rtcs.back().scanindex.data = bds.back().scanindex.col(x);

                //rtcs.back().scanindex.data = bds.back().scanindex.col(0); //temp

                //rtcs.back().scanindex.data.row(1) = rtcs.back().scanindex.data.row(0).array() + scanlength;
                //rtcs.back().scanindex.data.row(3) = rtcs.back().scanindex.data.row(2).array() + scanlength;

                //return RTC and increment current scan by one
                return rtcs[x++];
                }
                return {};
        },

        //grrppi call to parallelize inputs
        grppi::farm(n,[&](auto in) -> PTCData {
            //create an RTC to hold the reference to BeammapData scans
            RTCProc rtcproc(config);
            //Create a PTC to hold processed data
            PTCData out;
            //Process the data
            {
            logging::scoped_timeit timer("RTCProc");
            rtcproc.process(in,out);
            }
            Eigen::VectorXd beamSigAz(ndet);
            Eigen::VectorXd beamSigEl(ndet);

            beamSigAz.setConstant(4);
            beamSigEl.setConstant(4);

            {
            logging::scoped_timeit timer("makeKernelTimestream");
            for(Eigen::Index det=0;det<ndet;det++) {
                Eigen::VectorXd lat, lon;
                Eigen::Map<Eigen::VectorXd> scans(out.kernelscans.col(det).data(),out.kernelscans.rows());

                mapmaking::internal::getPointing(bds.back().telescope_data, lat, lon, offsets, det, out.scanindex(0), out.scanindex(1),dsf);
                timestream::makeKernelTimestream(scans,lat,lon,beamSigAz[det],beamSigEl[det]);
            }
            }
            return out;
         }),

        grppi::farm(n2,[&](auto in) -> PTCData {
            PTCData out;
            PTCProc ptcproc(config);
            {
            logging::scoped_timeit timer("PTCProc");
            ptcproc.process(in,out);
            }

            //Push the PTCs into a vector; scoped_lock is required to prevent racing condition
            {
                std::scoped_lock lock(farm_mutex);
                ptcs.push_back(out);
            }

        return out;
        }),

        grppi::farm(n3,[&](auto in) {
            double tmpwt = mapmaking::internal::calcScanWeight(in.scans, in.flags, samplerate);
            {
            logging::scoped_timeit timer("generatemaps");
            mapmaking::generatemaps(in,bds.back().telescope_data, mgrid_0, mgrid_1,
                                    samplerate, brs.back().mapstruct, offsets, tmpwt, NNoiseMapsPerObs, noisemaps, dsf);
            }
           }));

        double wt;

        double atmpix=0;
        for(int i=0;i<brs.back().mapstruct.nrows;i++){
            for(int j=0;j<brs.back().mapstruct.ncols;j++){
                wt = brs.back().mapstruct.wtt(i,j);
                if(wt != 0.){
                  //if (atmTemplate)
                  //atmpix = atmTemplate->image(i,j);
                  //brs.back().mapstruct.signal(i,j) = -(brs.back().mapstruct.signal(i,j)-atmpix)/wt;
                  brs.back().mapstruct.signal(i,j) = -(brs.back().mapstruct.signal(i,j))/wt;
                  //brs.back().mapstruct.wtt(i,j) = wt;
                  brs.back().mapstruct.kernel(i,j) = -(brs.back().mapstruct.kernel(i,j))/wt;
                  //brs.back().mapstruct.kernel(i,j) = wt;

                  for(int kk=0;kk<NNoiseMapsPerObs;kk++){
                      brs.back().mapstruct.noisemaps(kk,i,j) = brs.back().mapstruct.noisemaps(kk,i,j)/wt;
                  }
                }
                else{
                  brs.back().mapstruct.signal(i,j) = 0.;
                  brs.back().mapstruct.kernel(i,j) = 0.;
                  for(int kk=0;kk<NNoiseMapsPerObs;kk++){
                      brs.back().mapstruct.noisemaps(kk,i,j) = 0.;
                  }
                }
            }
        }

        auto output_filepath = config->get_typed<std::string>("output_filepath");
        brs.back().toNcFile(output_filepath, bds.back());

    } catch (const po::error &e) {
        SPDLOG_ERROR("{}", e.what());
        return 1;
    } catch (std::runtime_error &e) {
        SPDLOG_ERROR("{}", e.what());
        return 1;
    } catch (...) {
        auto what = boost::current_exception_diagnostic_information();
        SPDLOG_ERROR("unhandled exception: {}, abort", what);
        throw;
    }
    }

    return 0;
}
