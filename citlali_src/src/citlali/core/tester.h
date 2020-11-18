#pragma once

/*This file holds methods to simulate fake data in order to test performance or algorithms*/

// Namespaces for getting cmd line values
namespace po = boost::program_options;
namespace pt = boost::property_tree;

namespace lali {

class Tester {
public:
    // std::vectors for cmd line inputs
    std::vector<std::string> config_files;
    std::vector<std::string> input_files;

    std::shared_ptr<YamlConfig> config_;

    enum DataType { UseAzTEC = 0, UseTolTEC = 1 };

    // Temporary class for AzTEC data reading & storage
    aztec::BeammapData Data;

    int getInputs(int argc, char *argv[]);
    int getConfig();
    void getData();

    template <DataType datatype>
    static Tester getAztecData(int argc, char *argv[]) {
        Tester tester;
        tester.getInputs(argc, argv);
        tester.getConfig();

        if constexpr (datatype == UseAzTEC) {
            tester.getData();
        }

        return tester;
    }
};

class Tester1 {
public:
    // std::vectors for cmd line inputs
    std::vector<std::string> config_files;
    std::vector<std::string> input_files;

    std::shared_ptr<YamlConfig> config_;

    enum DataType { UseAzTEC = 0, UseTolTEC = 1 };

    // Temporary class for AzTEC data reading & storage
    aztec::BeammapData Data;

    int getInputs(int argc, char *argv[]);
    int getConfig();
    void getData();

    template<DataType datatype>
    static Tester getAztecData(int argc, char *argv[])
    {
        Tester tester;
        tester.getInputs(argc, argv);
        tester.getConfig();

        if constexpr (datatype == UseAzTEC) {
            tester.getData();
        }

        return tester;
    }
};

// Gets the config file from getInputs() and creates a config object
int Tester::getConfig()
{
    YAML::Node root;
    try {
        // Generalize to read multiple config files
        root = YAML::LoadFile(config_files.front());
        config_ = std::move(std::make_shared<lali::YamlConfig>(root));

    } catch (const std::exception &e) {
        SPDLOG_ERROR("{}", e.what());
        return 0;
    }

    return 0;
}

// Gets the data and config files specified at the cmd line and
// puts them into std vectors.
int Tester::getInputs(int argc, char *argv[])
{
    using RC = pt::ptree;
    RC rc;
    try {
        po::options_description opts_desc{"Options"};
        opts_desc.add_options()("help,h", "Help screen")("config_file,c",
                                                         po::value<std::vector<std::string>>()
                                                             ->multitoken()
                                                             ->zero_tokens()
                                                             ->composing())(
            "input_file,i",
            po::value<std::vector<std::string>>()->multitoken()->zero_tokens()->composing());
        po::positional_options_description inputs_desc{};
        inputs_desc.add("input_file", -1);
        po::command_line_parser parser{argc, argv};
        parser.options(opts_desc).positional(inputs_desc).allow_unregistered();
        po::variables_map vm;
        auto parsed = parser.run();
        po::store(parsed, vm);
        po::notify(vm);

        if (vm.empty() || vm.count("help")) {
            std::cout << opts_desc << '\n';
            return 0;
        }

        if (vm.count("config_file")) {
            config_files = vm["config_file"].as<std::vector<std::string>>();
        }
        SPDLOG_INFO("number of config files: {}", config_files.size());
        for (const auto &f : config_files) {
            SPDLOG_INFO("   {}", f);
        }

        if (vm.count("input_file")) {
            input_files = vm["input_file"].as<std::vector<std::string>>();
        }
        SPDLOG_INFO("number of input files: {}", input_files.size());
        for (const auto &f : input_files) {
            SPDLOG_INFO("   {}", f);
        }

        auto unparsed_opts = collect_unrecognized(parsed.options, po::exclude_positional);
        for (const auto &opt : unparsed_opts) {
            SPDLOG_INFO("unparsed options: {}", opt);
        }

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

    return 0;
}

// Loads the data from the input data files
void Tester::getData(){
    static auto it = input_files.begin();

    try{
        Data = aztec::BeammapData::fromNcFile(*(it));
    }
    catch (const aztec::DataIOError &e) {
        SPDLOG_WARN("failed to read input {}: {}", *it, e.what());
    }
}

} //namespace
