#! /bin/sh
ROOT=`dirname $0`
PATCH_GLOB=$1
for PATCH in `ls ${ROOT}/${PATCH_GLOB} 2>/dev/null`
do
    echo "patch cmd: $(which patch) --verbose -u -N -p 1 < ${PATCH}"
    $(patch --verbose -u -N -p 1 < ${PATCH}) || echo "patch failed, as it maybe already patched."
done
