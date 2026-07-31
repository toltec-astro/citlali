# TulaPerflibs OpenMP compatibility adapter

The upstream `TulaPerflibs` package discovers Homebrew LLVM OpenMP correctly
when it is built with explicit CMake hints, but its installed package config
discards those hints. Downstream packages using an older Spack-built CMake
therefore fail while rediscovering the same runtime.

This bounded adapter preserves the upstream `tula_deps::perflibs` target and
configuration header. Its only behavioral difference is that the installed
package config replays the exact OpenMP flags, library, and include directory
used to build the package while resolving `OpenMP::OpenMP_CXX`. Caller
variables are restored after discovery.

Remove this adapter when upstream `TulaPerflibs` exports a downstream package
config that can resolve its declared OpenMP runtime without ambient compiler
flags or package-specific overrides.
