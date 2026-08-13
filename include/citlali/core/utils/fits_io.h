#pragma once

#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/atomic_yaml_output.h>

#include <CCfits/CCfits>
#include <Eigen/Core>
#include <fitsio.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <valarray>
#include <vector>

enum file_type_enum {
    read_fits = 0,
    write_fits = 1
};
template<file_type_enum file_type, typename ext_hdu_t>
class fitsIO {
public:
    enum class PublicationCheckpoint {
        after_hdu_write,
        after_library_flush,
        after_close,
        before_reopen,
        after_reopen
    };

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // filepath
    std::string filepath;

    // pointer to FITS file
    std::unique_ptr<CCfits::FITS> pfits;

    // vector of hdu's for easy access
    std::vector<ext_hdu_t> hdus;

    fitsIO() {}

    fitsIO(const fitsIO &) = delete;
    fitsIO &operator=(const fitsIO &) = delete;

    fitsIO(fitsIO &&other) noexcept
        : logger(std::move(other.logger)),
          filepath(std::move(other.filepath)),
          pfits(std::move(other.pfits)),
          hdus(std::move(other.hdus)),
          staged_path_(std::move(other.staged_path_)),
          expected_images_(std::move(other.expected_images_)),
          expected_calibration_join_(
              std::move(other.expected_calibration_join_)),
          write_failed_(other.write_failed_),
          published_(other.published_) {
        other.staged_path_.clear();
        other.published_ = true;
    }

    fitsIO &operator=(fitsIO &&other) noexcept {
        if (this != &other) {
            discard_staged_output_noexcept();
            logger = std::move(other.logger);
            filepath = std::move(other.filepath);
            pfits = std::move(other.pfits);
            hdus = std::move(other.hdus);
            staged_path_ = std::move(other.staged_path_);
            expected_images_ = std::move(other.expected_images_);
            expected_calibration_join_ =
                std::move(other.expected_calibration_join_);
            write_failed_ = other.write_failed_;
            published_ = other.published_;
            other.staged_path_.clear();
            other.published_ = true;
        }
        return *this;
    }

    ~fitsIO() {
        discard_staged_output_noexcept();
    }

    // constructor
    fitsIO(std::string _f) : filepath(_f) {
        // read in file
        if constexpr (file_type==file_type_enum::read_fits) {
            try {
                pfits.reset( new CCfits::FITS(filepath, CCfits::Read));
                logger->info("opened FITS file {}", filepath);
            }
            catch (const CCfits::FitsException &error) {
                throw citlali::error::io(
                    "unable to open FITS input file " + filepath + ": " +
                    error.message());
            }
        }

        // create file
        else if constexpr (file_type==file_type_enum::write_fits) {
            staged_path_ = std::filesystem::path(filepath + ".fits.tmp");
            std::error_code ignored;
            std::filesystem::remove(staged_path_, ignored);
            try {
                // ! is the overwrite flag
                pfits.reset(new CCfits::FITS(
                    "!" + staged_path_.string(), CCfits::Write));
                // write date
                pfits->pHDU().writeDate();
            }
            catch (const CCfits::FitsException &error) {
                std::filesystem::remove(staged_path_, ignored);
                throw citlali::error::output(
                    "unable to create required FITS output file " + filepath +
                    ".fits: " + error.message());
            }
        }
    }

    template <typename Derived>
    void add_hdu(std::string hdu_name,
                 const Eigen::DenseBase<Derived> &data) {
        using scalar_type =
            std::remove_cv_t<typename Derived::Scalar>;

        if constexpr (std::is_same_v<scalar_type, double>) {
            add_typed_hdu<double>(std::move(hdu_name), DOUBLE_IMG, data);
        }
        else if constexpr (std::is_same_v<scalar_type, std::int64_t>) {
            add_typed_hdu<long long>(std::move(hdu_name), LONGLONG_IMG, data);
        }
        else if constexpr (std::is_same_v<scalar_type, std::uint8_t>) {
            add_typed_hdu<unsigned char>(std::move(hdu_name), BYTE_IMG, data);
        }
        else {
            static_assert(std::is_same_v<scalar_type, double> ||
                              std::is_same_v<scalar_type, std::int64_t> ||
                              std::is_same_v<scalar_type, std::uint8_t>,
                          "fitsIO::add_hdu supports only double, signed int64, "
                          "and uint8 image planes");
        }
    }

private:
    struct ExpectedImage {
        std::string name;
        std::vector<long> axes;
    };

    std::filesystem::path staged_path_;
    std::vector<ExpectedImage> expected_images_;
    std::optional<std::pair<std::string, std::string>>
        expected_calibration_join_;
    bool write_failed_ = false;
    bool published_ = false;

    template <typename FitsScalar, typename Derived>
    void add_typed_hdu(std::string hdu_name, int image_type,
                       const Eigen::DenseBase<Derived> &data) {
        try {
            // axes in reverse order (cols, rows, pol, freq)
            std::vector<long> naxes{data.cols(), data.rows(), 1, 1};

            // add an extension hdu to vector
            hdus.push_back((pfits->addImage(hdu_name, image_type, naxes)));

            // valarray to copy data into (seems to be necessary)
            std::valarray<FitsScalar> temp_data(data.size());

            // copy the data (flip in x direction)
            int k = 0;
            for (int i=0; i<data.rows(); ++i){
                for (int j=0; j<data.cols(); ++j) {
                    temp_data[k] = static_cast<FitsScalar>(
                        data(i, data.cols() - j - 1));
                    k++;
                }
            }

            // first pixel (starts at 1 I think)
            long first_pixel = 1;

            // write to the hdu
            hdus.back()->write(first_pixel, temp_data.size(), temp_data);
            expected_images_.push_back({hdu_name, naxes});
        } catch (const CCfits::FitsException &e) {
            write_failed_ = true;
            throw citlali::error::output(
                "failed to add/write FITS HDU '" + hdu_name + "' in " + filepath + ": " + e.message());
        }
    }

    static std::string fits_status_message(int status) {
        char message[FLEN_STATUS] = {};
        fits_get_errstatus(status, message);
        return message;
    }

    static std::string read_long_string_key(fitsfile *file,
                                            const char *key) {
        char *raw = nullptr;
        char comment[FLEN_COMMENT] = {};
        int status = 0;
        if (fits_read_key_longstr(file, key, &raw, comment, &status) != 0) {
            throw std::runtime_error(
                "missing required FITS PHDU key " + std::string{key} +
                ": " + fits_status_message(status));
        }
        const std::string value = raw == nullptr ? "" : raw;
        int free_status = 0;
        if (raw != nullptr) {
            fits_free_memory(raw, &free_status);
        }
        if (free_status != 0) {
            throw std::runtime_error(
                "unable to release FITS PHDU key buffer " +
                std::string{key});
        }
        return value;
    }

    void validate_reopened_output() const {
        fitsfile *file = nullptr;
        int status = 0;
        if (fits_open_file(
                &file, staged_path_.c_str(), READONLY, &status) != 0) {
            throw std::runtime_error(
                "unable to reopen staged FITS output " +
                staged_path_.string() + ": " +
                fits_status_message(status));
        }
        try {
            int hdu_count = 0;
            if (fits_get_num_hdus(file, &hdu_count, &status) != 0) {
                throw std::runtime_error(
                    "unable to count reopened FITS HDUs: " +
                    fits_status_message(status));
            }
            if (hdu_count !=
                static_cast<int>(expected_images_.size() + 1)) {
                throw std::runtime_error(
                    "reopened FITS HDU count does not match completed writes");
            }

            if (expected_calibration_join_) {
                int hdu_type = 0;
                if (fits_movabs_hdu(file, 1, &hdu_type, &status) != 0) {
                    throw std::runtime_error(
                        "unable to reopen FITS PHDU: " +
                        fits_status_message(status));
                }
                if (read_long_string_key(file, "CALID") !=
                        expected_calibration_join_->first ||
                    read_long_string_key(file, "CALPKGID") !=
                        expected_calibration_join_->second) {
                    throw std::runtime_error(
                        "reopened FITS PHDU CALID/PKGID join mismatch");
                }
            }

            std::vector<double> buffer(65536);
            for (std::size_t index = 0;
                 index < expected_images_.size(); ++index) {
                int hdu_type = 0;
                status = 0;
                if (fits_movabs_hdu(
                        file, static_cast<int>(index + 2),
                        &hdu_type, &status) != 0 ||
                    hdu_type != IMAGE_HDU) {
                    throw std::runtime_error(
                        "reopened FITS image HDU is missing or malformed");
                }
                char extension_name[FLEN_VALUE] = {};
                if (fits_read_key(file, TSTRING, "EXTNAME", extension_name,
                                  nullptr, &status) != 0 ||
                    expected_images_[index].name != extension_name) {
                    throw std::runtime_error(
                        "reopened FITS extension identity mismatch");
                }
                int axis_count = 0;
                long axes[4] = {};
                if (fits_get_img_dim(file, &axis_count, &status) != 0 ||
                    fits_get_img_size(file, 4, axes, &status) != 0 ||
                    axis_count != 4 ||
                    !std::equal(expected_images_[index].axes.begin(),
                                expected_images_[index].axes.end(), axes)) {
                    throw std::runtime_error(
                        "reopened FITS extension shape mismatch");
                }
                LONGLONG remaining = 1;
                for (const auto axis : expected_images_[index].axes) {
                    remaining *= static_cast<LONGLONG>(axis);
                }
                LONGLONG first = 1;
                int any_null = 0;
                while (remaining > 0) {
                    const auto count = std::min<LONGLONG>(
                        remaining, static_cast<LONGLONG>(buffer.size()));
                    status = 0;
                    if (fits_read_img(file, TDOUBLE, first, count, nullptr,
                                      buffer.data(), &any_null, &status) != 0) {
                        throw std::runtime_error(
                            "reopened FITS image content is incomplete: " +
                            fits_status_message(status));
                    }
                    first += count;
                    remaining -= count;
                }
            }
        }
        catch (...) {
            int close_status = 0;
            fits_close_file(file, &close_status);
            throw;
        }
        status = 0;
        if (fits_close_file(file, &status) != 0) {
            throw std::runtime_error(
                "unable to close reopened FITS validation handle: " +
                fits_status_message(status));
        }
    }

    void discard_staged_output_noexcept() noexcept {
        if constexpr (file_type == file_type_enum::write_fits) {
            try {
                pfits.reset();
            }
            catch (...) {
            }
            if (!published_ && !staged_path_.empty()) {
                std::error_code ignored;
                std::filesystem::remove(staged_path_, ignored);
            }
        }
    }

public:

    const std::filesystem::path &staged_path() const {
        return staged_path_;
    }

    void require_calibration_join(std::string calibration_identity,
                                  std::string package_identity) {
        expected_calibration_join_ = std::make_pair(
            std::move(calibration_identity), std::move(package_identity));
    }

    void discard_staged_output() {
        discard_staged_output_noexcept();
    }

    template <class Checkpoint>
    void publish_atomically(Checkpoint &&checkpoint) {
        static_assert(file_type == file_type_enum::write_fits,
                      "only FITS writers publish outputs");
        if (published_) {
            return;
        }
        try {
            if (write_failed_) {
                throw std::runtime_error(
                    "cannot publish a FITS output after an HDU write failure");
            }
            std::invoke(checkpoint, PublicationCheckpoint::after_hdu_write);
            if (pfits && pfits->fitsPointer() != nullptr) {
                int status = 0;
                if (fits_flush_file(pfits->fitsPointer(), &status) != 0) {
                    throw std::runtime_error(
                        "unable to flush staged FITS output: " +
                        fits_status_message(status));
                }
            }
            citlali::pipeline::atomic_output::synchronize_file(staged_path_);
            std::invoke(
                checkpoint, PublicationCheckpoint::after_library_flush);
            pfits.reset();
            hdus.clear();
            std::invoke(checkpoint, PublicationCheckpoint::after_close);
            std::invoke(checkpoint, PublicationCheckpoint::before_reopen);
            validate_reopened_output();
            std::invoke(checkpoint, PublicationCheckpoint::after_reopen);
            citlali::pipeline::atomic_output::replace_atomically(
                staged_path_, std::filesystem::path(filepath + ".fits"));
            published_ = true;
        }
        catch (const std::exception &error) {
            discard_staged_output_noexcept();
            throw citlali::error::output(
                "failed to publish required FITS output " + filepath +
                ".fits: " + error.what());
        }
        catch (...) {
            discard_staged_output_noexcept();
            throw citlali::error::output(
                "failed to publish required FITS output " + filepath +
                ".fits");
        }
    }

    void publish_atomically() {
        publish_atomically([](PublicationCheckpoint) {});
    }

    auto get_hdu(std::string hdu_name) {
        try {
            // get extension
            CCfits::ExtHDU& hdu = pfits->extension(hdu_name);

            // hold image data
            std::valarray<double> contents;

            // read all user-specifed, coordinate, and checksum keys in the image
            hdu.readAllKeys();
            hdu.read(contents);

            // this doesn't print the data, just header info.
            long ax1(hdu.axis(0));
            long ax2(hdu.axis(1));

            // holds the image data
            Eigen::MatrixXd data(ax2,ax1);

            // loop through and copy into eigen matrix
            Eigen::Index k = 0;
            for (Eigen::Index i=0; i<ax2; ++i) {
                for (Eigen::Index j=0; j<ax1; ++j) {
                    data(i,j) = contents[k];
                    k++;
                }
            }

            // flip to match internal orientation
            data.rowwise().reverseInPlace();

            return data;

        } catch (const CCfits::FitsException &error) {
            throw citlali::error::io(
                "cannot read FITS HDU '" + hdu_name + "' in " + filepath +
                ": " + error.message());
        }
    }

    template <typename hdu_t, class wcs_t, typename epoch_t>
    void add_wcs(hdu_t *hdu, wcs_t &wcs, const epoch_t epoch) {
        try {
            // add equinox
            hdu->addKey("EQUINOX", epoch, "WCS: Equinox");

            for (Eigen::Index i=0; i<wcs.ctype.size(); ++i) {
                hdu->addKey("CTYPE"+std::to_string(i+1), wcs.ctype[i], "WCS: Projection Type " +std::to_string(i+1));
                hdu->addKey("CUNIT"+std::to_string(i+1), wcs.cunit[i], "WCS: Axis Unit " +std::to_string(i+1));
                hdu->addKey("CRVAL"+std::to_string(i+1), wcs.crval[i], "WCS: Ref Pixel Value " +std::to_string(i+1));
                hdu->addKey("CDELT"+std::to_string(i+1), wcs.cdelt[i], "WCS: Pixel Scale " +std::to_string(i+1));
                // add one to crpix due to FITS convention
                hdu->addKey("CRPIX"+std::to_string(i+1), wcs.crpix[i] + 1, "WCS: Ref Pixel " +std::to_string(i+1));
            }
        } catch (const CCfits::FitsException &e) {
            throw citlali::error::output(
                "failed to add WCS keywords in " + filepath + ": " + e.message());
        }
    }
};
