#include <kids/toltec/timestream.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>

auto main(int argc, char **argv) -> int
{
    if (argc != 2) {
        throw std::invalid_argument{"expected one raw timestream fixture"};
    }

    const std::filesystem::path fixture{argv[1]};
    const auto meta = kids::toltec::get_raw_timestream_meta(fixture);
    const auto data = kids::toltec::read_raw_timestream_slice(
        fixture, kids::toltec::SampleSlice{0, 2, 1});

    if (data.is.data.rows() != 2 || data.qs.data.rows() != 2 ||
        data.is.data.cols() <= 0 ||
        data.is.data.cols() != data.qs.data.cols()) {
        throw std::runtime_error{"invalid raw timestream slice shape"};
    }

    std::cout << "obsid=" << meta.get_typed<int>("obsid")
              << " samples=" << data.is.data.rows()
              << " tones=" << data.is.data.cols() << '\n';
    return 0;
}
