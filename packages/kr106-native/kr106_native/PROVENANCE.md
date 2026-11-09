# KR-106 DSP provenance

This package compiles the unmodified header-only DSP from [kayrockscreenprinting/ultramaster_kr106](https://github.com/kayrockscreenprinting/ultramaster_kr106), revision `bc15caee5843ab238a25d0969e68d57db2b1615f` (KR-106 2.5.13).

CMake downloads and verifies the immutable source archive at build time:

- URL: `https://codeload.github.com/kayrockscreenprinting/ultramaster_kr106/tar.gz/bc15caee5843ab238a25d0969e68d57db2b1615f`
- SHA-256: `eba003e0e6f295a5d884a6490b6055b1d1d0183ae74847c9bb2ffdc6e23809a5`

Only `Source/DSP/*.h` and `Source/KR106_Presets_JUCE.h` are compiled. The upstream JUCE build is not invoked. CMake's `FETCHCONTENT_SOURCE_DIR_KR106_DSP` override supports an already-verified source checkout for offline builds; callers must ensure it matches the pin.

The DSP is distributed under the upstream GNU GPL v3. The complete upstream license is included as `KR106-GPL-3.0.txt`. The wheel includes this provenance and license. Runtime rendering does not read the source checkout or use the network.
