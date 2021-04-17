# Citlali

`Citlali` is the core component of the TolTEC Data Analysis Pipeline
(TolTEC-DAP).


While `Citlali` is developed targeting LMT/TolTEC, it also works with
other telescope/detectors that shares similar architectural properties.


# Install

On Mac:

Use llvm compiler:
brew install llvm

Have cmake installed:
brew install cmake

git clone https://github.com/toltec-astro/citlali.git
cd citlali
git checkout sim_dev
git submodule update --recursive --remote --init
cd common_utils
git checkout kids_dev
cd cmake
git checkout master cd ../../
cd kidscpp_src
git checkout master
cd ../
mkdir build
cmake -DCMAKE_C_COMPILER=/path/to/llvm -DCMAKE_CXX_COMPILER=/path/to/llvm++ -DCMAKE_BUILD_TYPE=Release
make citlali




# Usage

<usage>


# Licence

3-Clause BSD.
