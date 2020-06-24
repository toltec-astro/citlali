#pragma once


#if defined(__GNUC__) && !defined(__clang__)
#if __GNUC__ < 10
#include <experimental/filesystem>
namespace std {
    namespace filesystem = ::std::experimental::filesystem;
} // namespace std
#else
#include <filesystem>
#endif
#endif
