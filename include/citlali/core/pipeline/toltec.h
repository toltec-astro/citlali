# pragma once

#include <citlali/core/pipeline/hwpr.h>

class NetworkContainer {
public:
    // network interface and roach id
    int interface, roach_id;

    // adc snap data
    Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic> adc_snap_data;

    // APT for the detectors on this network
    ArrayPropertyTable apt;

    // default constructor
    NetworkContainer() {}
    NetworkContainer(int name, const ArrayPropertyTable& apt_data)
        : interface(name), apt(apt_data) {}
};

class ArrayContainer {
public:
    // array name
    int name;

    // APT for the detectors on this array
    ArrayPropertyTable apt;

    // List of networks in this array
    std::map<int, NetworkContainer> networks;

    // default constructor
    ArrayContainer() {}
    ArrayContainer(int name, const ArrayPropertyTable& apt_data)
        : name(name), apt(apt_data) {}
};

class Instrument {
public:
    std::string name = "TolTEC";
    ArrayPropertyTable apt;
    Hwpr hwpr;

    // map of arrays in this instrument
    std::map<int, ArrayContainer> arrays;

    // sample rate in Hz
    double data_fs_hz;

    // array names
    std::map<int, std::string> array_index_to_name = {
        {0, "a1100"},
        {1, "a1400"},
        {2, "a2000"}
    };

    // array names
    std::map<int, double> array_index_to_fwhm = {
        {0, 5.0},
        {1, 6.3},
        {2, 9.5}
    };

    // array wavelengths
    std::map<int, double> array_index_to_wavelength = {
        {0, 0.0011},
        {1, 0.0014},
        {2, 0.0020}
    };

    std::map<int, int> nw_to_array = {
        {0, 0}, {1, 0},
        {2, 0}, {3, 0},
        {4, 0}, {5, 0},
        {6, 0}, {7, 1},
        {8, 1}, {9, 1},
        {10, 1}, {11, 2},
        {12, 2}
    };

    // toltec array mounting angle
    std::map<int, double> array_index_to_install_angle = {
        {-1, -1},
        {0, pi/2},
        {1, -pi/2},
        {2, -pi/2},
    };

    // toltec detector orientation angles
    std::map<int, double> fg_to_detector_angle = {
        {-1, -1},
        {0, 0},
        {1, pi/4},
        {2, pi/2},
        {3, 3*pi/4}
    };

    // custom iterator to iterate over all networks across all arrays
    class NetworkIterator {
    public:
        using ArrayIterator = std::map<int, ArrayContainer>::const_iterator;
        using NetworkIteratorType = std::map<int, NetworkContainer>::const_iterator;

        NetworkIterator(ArrayIterator array_it, ArrayIterator array_end)
            : array_it_(array_it), array_end_(array_end) {
            if (array_it_ != array_end_) {
                network_it_ = array_it_->second.networks.begin();
                advance_to_next_network();
            }
        }

        const NetworkContainer& operator*() const { return network_it_->second; }
        const NetworkContainer* operator->() const { return &network_it_->second; }

        NetworkIterator& operator++() {
            ++network_it_;
            advance_to_next_network();
            return *this;
        }

        bool operator!=(const NetworkIterator& other) const {
            return array_it_ != other.array_it_ || network_it_ != other.network_it_;
        }

    private:
        ArrayIterator array_it_;
        ArrayIterator array_end_;
        NetworkIteratorType network_it_;

        void advance_to_next_network() {
            while (array_it_ != array_end_ && network_it_ == array_it_->second.networks.end()) {
                ++array_it_;
                if (array_it_ != array_end_) {
                    network_it_ = array_it_->second.networks.begin();
                }
            }
        }
    };

    // Methods to get the beginning and end of the network iterator
    NetworkIterator begin() const {
        return NetworkIterator(arrays.begin(), arrays.end());
    }

    NetworkIterator end() const {
        return NetworkIterator(arrays.end(), arrays.end());
    }

    std::string create_filename(const std::string& filepath,
                                const std::string& data_type,
                                const std::string& prod_type,
                                const std::string& filter_type,
                                const std::string& redu_type,
                                const std::string& array_name,
                                const std::string& obsnum,
                                const bool simu_obs) {
        std::string filename = filepath;

        // append data type
        filename += data_type;

        // append real data or simulation
        filename += simu_obs ? "_simu" : "_commissioning";

        // add array name, reduction type, and observation number if they are not empty
        if (!array_name.empty()) filename += "_" + array_name;
        if (!redu_type.empty()) filename += "_" + redu_type;
        if (!obsnum.empty()) filename += "_" + obsnum;
        if (!prod_type.empty()) filename += "_" + prod_type;

        // append filter type
        if (filter_type == "filtered") {
            filename += "_filtered";
        }

        // append pipeline information for maps or noise maps
        if (prod_type == "" || prod_type == "noise") {
            filename += "_citlali";
        }

        return filename;
    }
};

