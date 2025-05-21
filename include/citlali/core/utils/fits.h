#include <CCfits/CCfits>

#include <citlali/core/mapmaking/wcs.h>

// header object that can store different types of FITS metadata
class FitsHeader {
public:
    // variants to store different types of header values
    using HeaderValue = std::variant<int, double, std::string>;

    // Method to add a key-value pair to the header
    void add_key(const std::string& key, const HeaderValue& value, const std::string& comment = "") {
        keys.push_back(key);
        headers.push_back({value, comment});
    }

    // method to write header to a FITS file
    void write_to_fits(CCfits::HDU& hdu) const {
        for (int i = 0; i < keys.size(); ++i) {
            const auto& [value, comment] = headers[i];
            std::visit([&](auto&& arg) {
                // check for NaN in floating-point types
                if constexpr (std::is_floating_point_v<std::decay_t<decltype(arg)>>) {
                    if (std::isnan(arg)) {
                        throw std::runtime_error("Encountered NaN value while writing to FITS.");
                    }
                }
                add_key_to_fits(hdu, keys[i], arg, comment);
            }, value);
        }
    }

    template<typename T>
    void add_key_to_fits(CCfits::HDU& hdu, const std::string& key, const T& value, const std::string& comment) const {
        hdu.addKey(key, value, comment);
    }

    // store the headers, with the value and an optional comment
    std::vector<std::pair<HeaderValue, std::string>> headers;
    std::vector<std::string> keys;
};

enum class FitsMode {
    ReadFits = 0,
    WriteFits = 1
};

template<FitsMode mode, typename HduType>
class fitsIO {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // filepath
    std::string filepath;

    // pointer to fits file
    std::unique_ptr<CCfits::FITS> pfits;

    // vector of hdu's for easy access
    std::vector<HduType> hdus;

    // constructor
    fitsIO(std::string _f) : filepath(_f) {
        // read in file
        if constexpr (mode == FitsMode::ReadFits) {
            try {
                pfits.reset(new CCfits::FITS(filepath, CCfits::Read));
                logger->info("found fits file {}", filepath);
            } catch (const CCfits::FITS::CantOpen&) {
                throw std::runtime_error(fmt::format("unable to open file {}", filepath));
            }
        }

        // create file
        else if constexpr (mode == FitsMode::WriteFits) {
            try {
                // ! is the overwrite flag
                pfits.reset(new CCfits::FITS("!" + filepath, CCfits::Write));
                // write date
                pfits->pHDU().writeDate();
            } catch (const CCfits::FITS::CantCreate&) {
                throw std::runtime_error(fmt::format("unable to create file {}", filepath));
            }
        }
    }

    template <typename Derived, typename WcsType>
    void add_hdu(std::string hdu_name, const Eigen::DenseBase<Derived>& data, WcsType& wcs) {
        int n_rows = data.rows();
        int n_cols = data.cols();

        // axes in reverse order (cols, rows)
        std::vector<long> naxes{n_cols, n_rows};

        // add an extension hdu to vector
        hdus.push_back((pfits->addImage(hdu_name, DOUBLE_IMG, naxes)));

        // valarray to copy data into (seems to be necessary)
        std::valarray<double> temp_data(data.size());

        // copy the data (flip in x direction)
        int k = 0;
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                temp_data[k] = data(i, n_cols - j - 1);
                k++;
            }
        }

        // first pixel (starts at 1 I think)
        long first_pixel = 1;

        // write to the hdu
        hdus.back()->write(first_pixel, temp_data.size(), temp_data);

        // add wcs object
        add_wcs(hdus.back(), wcs);
    }

    auto get_hdu(std::string hdu_name) {
        try {
            // get extension
            CCfits::ExtHDU& hdu = pfits->extension(hdu_name);

            // hold image data
            std::valarray<double> contents;

            // read all user-specified, coordinate, and checksum keys in the image
            hdu.readAllKeys();
            hdu.read(contents);

            long ax1(hdu.axis(0));
            long ax2(hdu.axis(1));

            // holds the image data
            Eigen::MatrixXd data(ax2, ax1);

            // loop through and copy into eigen matrix
            int k = 0;
            for (int i = 0; i < ax2; ++i) {
                for (int j = 0; j < ax1; ++j) {
                    data(i, j) = contents[k];
                    k++;
                }
            }

            // flip to match internal orientation
            data.rowwise().reverseInPlace();

            return data;

        } catch (const CCfits::FITS::NoSuchHDU&) {
            throw std::runtime_error(fmt::format("failed to find {} in file {}", hdu_name, filepath));
        }
    }

    auto get_n_extensions() {
        // get number of extensions other than primary extension
        int n_extensions = 0;
        bool keep_going = true;
        while (keep_going) {
            try {
                // attempt to access an HDU (ignore primary hdu)
                CCfits::ExtHDU& ext = pfits->extension(n_extensions + 1);
                n_extensions++;
            } catch (CCfits::FITS::NoSuchHDU) {
                // NoSuchHDU exception is thrown when there are no more HDUs
                keep_going = false;
            }
        }

        return n_extensions;
    }

    template <class WcsType>
    void add_wcs(CCfits::ExtHDU* hdu, WcsType& wcs) {
        // add EQUINOX
        hdu->addKey("EQUINOX", wcs.epoch, "wcs: equinox");

        for (int i = 0; i < wcs.ctype.size(); ++i) {
            hdu->addKey("CTYPE" + std::to_string(i + 1), wcs.ctype[i], "wcs: projection type " + std::to_string(i + 1));
            hdu->addKey("CUNIT" + std::to_string(i + 1), wcs.cunit[i], "wcs: axis units " + std::to_string(i + 1));
            hdu->addKey("CRVAL" + std::to_string(i + 1), wcs.crval[i], "wcs: reference pixel value " + std::to_string(i + 1));
            hdu->addKey("CDELT" + std::to_string(i + 1), wcs.cdelt[i], "wcs: pixel scale " + std::to_string(i + 1));
            hdu->addKey("CRPIX" + std::to_string(i + 1), wcs.crpix[i], "wcs: reference pixel " + std::to_string(i + 1));
        }
    }

    auto get_wcs(std::string hdu_name) {
        // get extension
        CCfits::ExtHDU& hdu = pfits->extension(hdu_name);

        WCS wcs;
        wcs.ctype.resize(2);
        wcs.cunit.resize(2);
        wcs.crval.resize(2);
        wcs.cdelt.resize(2);
        wcs.crpix.resize(2);

        try {
            hdu.readKey("EQUINOX", wcs.epoch);

            std::vector<int> dims = {1, 2};
            for (const auto& dim : dims) {
                hdu.readKey("CTYPE" + std::to_string(dim), wcs.ctype[dim-1]);
                hdu.readKey("CUNIT" + std::to_string(dim), wcs.cunit[dim-1]);
                hdu.readKey("CRVAL" + std::to_string(dim), wcs.crval[dim-1]);
                hdu.readKey("CDELT" + std::to_string(dim), wcs.cdelt[dim-1]);
                hdu.readKey("CRPIX" + std::to_string(dim), wcs.crpix[dim-1]);
            }

        } catch (CCfits::HDU::NoSuchKeyword& e) {
            throw std::runtime_error(fmt::format("cannot find wcs header keys in {}: {}", hdu_name, e));
        }

        return wcs;
    }
};
