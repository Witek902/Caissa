cmake . -B build_avx512 -G "Visual Studio 17 2022" -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DTARGET_ARCH=x64-avx512
cmake . -B build_bmi2 -G "Visual Studio 17 2022" -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DTARGET_ARCH=x64-bmi2
cmake . -B build_avx2 -G "Visual Studio 17 2022" -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DTARGET_ARCH=x64-avx2
cmake . -B build_sse4 -G "Visual Studio 17 2022" -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DTARGET_ARCH=x64-sse4-popcnt
cmake . -B build_legacy -G "Visual Studio 17 2022" -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DTARGET_ARCH=x64-legacy
