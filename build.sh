#!/bin/bash

if [ -z "${1}" ]; then
    echo -e "Usage: ./build_sh <preset>\n"
    cmake --list-presets
    echo
    exit 0
fi
preset=$1
echo -e "build with preset=${preset}\n"
builddir=build_${preset}
mkdir -p ${builddir}
cmake -S . -B ${builddir} --preset ${preset}
