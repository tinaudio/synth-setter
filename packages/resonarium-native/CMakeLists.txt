cmake_minimum_required(VERSION 3.24)
project(resonarium_native LANGUAGES CXX C)

# Every archive is an immutable commit snapshot verified by SHA-256; the
# upstream git submodules and Perfetto's CPM checkout are replaced by these.
include(FetchContent)
set(_resonarium_rev d7f013f9b92839fdfd09cf0c0a1c6da90f331836)
# SOURCE_SUBDIR points at a missing directory so MakeAvailable only downloads.
FetchContent_Declare(resonarium
  URL https://codeload.github.com/gabrielsoule/resonarium/tar.gz/${_resonarium_rev}
  URL_HASH SHA256=c4d6292180a17145d7471599d8c0e2a92aea8ee7ccb399492cbf22cce20e5549
  SOURCE_SUBDIR _download_only
)
FetchContent_MakeAvailable(resonarium)

function(_resonarium_module name repo rev sha256)
  FetchContent_Declare(${name}
    URL https://codeload.github.com/${repo}/tar.gz/${rev}
    URL_HASH SHA256=${sha256}
    SOURCE_DIR ${resonarium_SOURCE_DIR}/modules/${name}
    SOURCE_SUBDIR _download_only
  )
  FetchContent_MakeAvailable(${name})
endfunction()

_resonarium_module(juce juce-framework/JUCE d6181bde38d858c283c3b7bf699ce6340c050b5d
  ab02a58f4cc0e42102a22e863e8bbe3aaa1c3e72899fd567154a5a1aa7898219)
_resonarium_module(gin gabrielsoule/Gin 6847ca1d9394ca48d3cb60d3185c87d7a5cc3f27
  41a3af8c12732d87285273e6df56d66072629c2cf8b4990a206eff7f2c0111d4)
_resonarium_module(melatonin_blur sudara/melatonin_blur 989a6e1f8d79b7183c9daa6271cb71f7f8800229
  3398e9d31cd63158031d7a8aba18d7adf7b73df3be98a115ee9ac6655f62e60a)
_resonarium_module(melatonin_inspector sudara/melatonin_inspector cd25631ede23fac747c72e898c37a4d68d8b5256
  bf2af417573d6826ca1cded9e259a69c8a4da02585972aefd2f0720e83f6cbfc)
_resonarium_module(melatonin_perfetto sudara/melatonin_perfetto e0d3f7eb40df1839a59a3878e56e93d3b201c132
  c0bdfcbff06f5b92443ad8a85e6ee8296851b6aab9eaa2f3dfb6a6a2eca258ef)
_resonarium_module(pybind11 pybind/pybind11 513d1f962e7a1ca156594bb4b58d2ddf106b74e5
  4a324bfeecd811a9a21300c363879ad0d8cc2f9f754e0a2a492e1e7c123b9e0e)

# Perfetto v50.1 (commit below); CPM uses this checkout instead of cloning the tag.
FetchContent_Declare(perfetto
  URL https://codeload.github.com/google/perfetto/tar.gz/5e2632131094c9c103e00abb37f48579e3e653ff
  URL_HASH SHA256=a00f3f1c9cf0866722e9d7912d7774d4d688bbbd6298c988719f7e9d64f5cd6e
  SOURCE_SUBDIR _download_only
)
FetchContent_MakeAvailable(perfetto)
set(CPM_perfetto_SOURCE ${perfetto_SOURCE_DIR} CACHE PATH "" FORCE)

add_subdirectory(${resonarium_SOURCE_DIR} resonarium EXCLUDE_FROM_ALL)

install(TARGETS Resonarium_Python LIBRARY DESTINATION . COMPONENT resonarium_python)
install(FILES resonarium_native/PROVENANCE.md resonarium_native/Resonarium-GPL-3.0.txt
        DESTINATION resonarium_native COMPONENT resonarium_python)
