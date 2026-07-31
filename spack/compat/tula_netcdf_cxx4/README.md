# NetCDF C++4 compatibility adapter

NetCDF C++ 4.3.1 installs `ncxx4-config`, its headers, and its library, but its
CMake build installs neither the `netcdf-cxx4.pc` file expected by the upstream
Tula adapter nor a complete imported-target configuration. This bounded
adapter preserves Tula's public `tula_deps::netcdf_cxx4` target while locating
the declared Spack package directly.

Remove this adapter when the upstream Tula adapter consumes an installed
NetCDF C++ target that both source-built and approved external packages
provide. It must not become a second provider-selection mechanism.
