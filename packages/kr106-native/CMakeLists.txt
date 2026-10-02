cmake_minimum_required(VERSION 3.22)
project(kr106_native LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  kr106_dsp
  URL https://codeload.github.com/kayrockscreenprinting/ultramaster_kr106/tar.gz/bc15caee5843ab238a25d0969e68d57db2b1615f
  URL_HASH SHA256=eba003e0e6f295a5d884a6490b6055b1d1d0183ae74847c9bb2ffdc6e23809a5
  SOURCE_SUBDIR Source/DSP
)
FetchContent_MakeAvailable(kr106_dsp)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

pybind11_add_module(kr106_native src/kr106_native.cpp)
target_compile_features(kr106_native PRIVATE cxx_std_17)
target_include_directories(kr106_native PRIVATE
  ${kr106_dsp_SOURCE_DIR}/Source/DSP
  ${kr106_dsp_SOURCE_DIR}/Source
)
target_compile_definitions(kr106_native PRIVATE _USE_MATH_DEFINES)
install(TARGETS kr106_native LIBRARY DESTINATION .)
install(FILES kr106_native/PROVENANCE.md kr106_native/KR106-GPL-3.0.txt
        DESTINATION kr106_native)
