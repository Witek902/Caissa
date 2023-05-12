#!/bin/bash

# This script is used to build all configurations of Caissa for release

# release version
version=1.8

# architectures
archs=(x64-legacy x64-sse4-popcnt x64-avx2-bmi2 x64-avx512)

# create the release folder
mkdir ../release

# create build folder
mkdir build

# build with all architectures and put the binaries in the release folder
for arch in "${archs[@]}"
do
    echo "Building $arch ..."

    cmake .. -B build/$arch -DCMAKE_BUILD_TYPE=Final -DTARGET_ARCH=$arch
    if [ $? -eq 0 ]
    then
        echo "Build configuration success!"
    else
        echo "Build configuration failed!"
        exit 1
    fi

    cmake --build build/$arch -j
    if [ $? -eq 0 ]
    then
        echo "Build success!"
    else
        echo "Build failed!"
        exit 1
    fi

    # run unit tests
    ./build/$arch/bin/utils unittest
    if [ $? -eq 0 ]
    then
        echo "Unit tests passed!"
    else
        echo "Unit tests failed!"
    fi

    # copy the binary to the release folder
    cp build/$arch/bin/caissa ../release/caissa-$version-$arch
done
