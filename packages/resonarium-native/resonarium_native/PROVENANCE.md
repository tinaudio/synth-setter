# Resonarium provenance

This package compiles the unmodified upstream `Resonarium_Python` CMake target from [gabrielsoule/resonarium](https://github.com/gabrielsoule/resonarium), revision `d7f013f9b92839fdfd09cf0c0a1c6da90f331836` (release tag `v0.1.1`). The wheel installs the upstream extension module as `resonarium`.

CMake downloads and verifies one immutable archive per dependency at build time. They replace the upstream git submodules, plus the Perfetto checkout that `melatonin_perfetto` would otherwise clone by tag:

| Source                     | Revision                                   | SHA-256                                                            |
| -------------------------- | ------------------------------------------ | ------------------------------------------------------------------ |
| gabrielsoule/resonarium    | `d7f013f9b92839fdfd09cf0c0a1c6da90f331836` | `c4d6292180a17145d7471599d8c0e2a92aea8ee7ccb399492cbf22cce20e5549` |
| juce-framework/JUCE        | `d6181bde38d858c283c3b7bf699ce6340c050b5d` | `ab02a58f4cc0e42102a22e863e8bbe3aaa1c3e72899fd567154a5a1aa7898219` |
| gabrielsoule/Gin           | `6847ca1d9394ca48d3cb60d3185c87d7a5cc3f27` | `41a3af8c12732d87285273e6df56d66072629c2cf8b4990a206eff7f2c0111d4` |
| sudara/melatonin_blur      | `989a6e1f8d79b7183c9daa6271cb71f7f8800229` | `3398e9d31cd63158031d7a8aba18d7adf7b73df3be98a115ee9ac6655f62e60a` |
| sudara/melatonin_inspector | `cd25631ede23fac747c72e898c37a4d68d8b5256` | `bf2af417573d6826ca1cded9e259a69c8a4da02585972aefd2f0720e83f6cbfc` |
| sudara/melatonin_perfetto  | `e0d3f7eb40df1839a59a3878e56e93d3b201c132` | `c0bdfcbff06f5b92443ad8a85e6ee8296851b6aab9eaa2f3dfb6a6a2eca258ef` |
| pybind/pybind11            | `513d1f962e7a1ca156594bb4b58d2ddf106b74e5` | `4a324bfeecd811a9a21300c363879ad0d8cc2f9f754e0a2a492e1e7c123b9e0e` |
| google/perfetto (v50.1)    | `5e2632131094c9c103e00abb37f48579e3e653ff` | `a00f3f1c9cf0866722e9d7912d7774d4d688bbbd6298c988719f7e9d64f5cd6e` |

The upstream build links JUCE's GUI and audio-device modules, so Linux builds need the X11, ALSA, FreeType, fontconfig, OpenGL and libcurl development headers. Rendering itself is offline and opens no window or device.

Resonarium is distributed under the GNU GPL v3. The complete upstream license is included as `Resonarium-GPL-3.0.txt`, and the wheel includes this provenance and license.
