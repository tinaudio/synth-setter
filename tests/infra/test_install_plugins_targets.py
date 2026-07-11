"""`make install-plugins` provisions every VST3 bundle the runtime docker image ships.

The image (docker/ubuntu22_04/Dockerfile) installs Surge XT plus three SHA256-pinned prebuilt
synths (Dexed, OB-Xf, Six Sines) and source-builds Ultramaster KR-106. The Makefile mirrors those
pins for local installs; these tests fail when either side drifts.

The download-path tests never touch the network: they seed the archive cache under a throwaway
``HOME`` and pass the fixture's real sha256 as a command-line make-variable override.
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import re
import shutil
import subprocess
import tarfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = PROJECT_ROOT / "Makefile"
DOCKERFILE = PROJECT_ROOT / "docker" / "ubuntu22_04" / "Dockerfile"

pytestmark = pytest.mark.infra

# Bound subprocess calls so a hung make can't wedge the suite.
_TIMEOUT_S = 60

# Every VST3 bundle staged into the runtime image, by plugins/ basename.
_IMAGE_BUNDLES = (
    "Surge XT.vst3",
    "Dexed.vst3",
    "OB-Xf.vst3",
    "Six Sines.vst3",
    "Ultramaster KR-106.vst3",
)

_LINUX_X86_64_PLUGIN_TARGETS = (
    "install-dexed",
    "install-obxf",
    "install-six-sines",
    "install-ultramaster-kr106",
)

# Pins that must stay identical between the Makefile and the Dockerfile ARGs.
_SHARED_PINS = (
    "DEXED_VERSION",
    "DEXED_SHA256",
    "OBXF_VERSION",
    "OBXF_SHA256",
    "SIX_SINES_VERSION",
    "SIX_SINES_ASSET",
    "SIX_SINES_SHA256",
    "ULTRAMASTER_KR106_VERSION",
    "ULTRAMASTER_KR106_GIT_REF",
)

# The prebuilt fetch and source-build recipes gate on x86_64 Linux, so their install branches are
# only reachable on such hosts.
requires_x86_64_linux = pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine() != "x86_64",
    reason="plugin install targets skip on non-x86_64 hosts",
)

if shutil.which("make") is None:
    pytest.skip("make not on PATH", allow_module_level=True)


def _makefile_var(name: str) -> str:
    """Return the value of a simple `NAME := value` Makefile assignment.

    :param name: variable name to look up.
    :returns: the assigned value, surrounding whitespace stripped.
    """
    match = re.search(rf"^{name}\s*:?=\s*(.+?)\s*$", MAKEFILE.read_text(), re.MULTILINE)
    assert match, f"Makefile does not define {name}"
    return match.group(1)


def _dockerfile_arg(name: str) -> str:
    """Return the default value of an `ARG NAME=value` Dockerfile instruction.

    :param name: build-arg name to look up.
    :returns: the default value, surrounding whitespace stripped.
    """
    match = re.search(rf"^ARG {name}=(.+?)\s*$", DOCKERFILE.read_text(), re.MULTILINE)
    assert match, f"Dockerfile does not define ARG {name}"
    return match.group(1)


def _dockerfile_stage_text(stage_name: str) -> str:
    """Return Dockerfile text from ``stage_name`` until the next stage.

    :param stage_name: Docker stage alias.
    :returns: the selected stage text.
    """
    text = DOCKERFILE.read_text()
    match = re.search(rf"^FROM .+ AS {re.escape(stage_name)}\n", text, re.MULTILINE)
    assert match, f"Dockerfile does not define stage {stage_name}"
    next_stage = re.search(r"^FROM ", text[match.end() :], re.MULTILINE)
    end = match.end() + next_stage.start() if next_stage else len(text)
    return text[match.start() : end]


def _run_make(
    cwd: Path,
    target: str,
    *makevars: str,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a make target in ``cwd`` without raising on failure.

    :param cwd: directory holding the Makefile under test.
    :param target: make target to invoke.
    :param *makevars: ``NAME=value`` command-line overrides for Makefile variables.
    :param env: full environment for the subprocess; inherits os.environ when None.
    :returns: the completed process, stdout/stderr captured as text.
    """
    git_ref_override = "CURRENT_LOCAL_GIT_REF=0000000000000000000000000000000000000000"
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["make", target, git_ref_override, *makevars],  # noqa: S607 — make on PATH
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )


def _zip_containing(inner_dir: str) -> bytes:
    """Build an in-memory .zip whose payload lives under ``inner_dir``.

    :param inner_dir: archive-internal directory path, e.g. ``x-lnx/Dexed.vst3``.
    :returns: the zip file's bytes.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{inner_dir}/Contents/x86_64-linux/plugin.so", b"fake plugin binary")
    return buf.getvalue()


def _targz_containing(inner_dir: str) -> bytes:
    """Build an in-memory .tgz whose payload lives under ``inner_dir``.

    :param inner_dir: archive-internal directory path, e.g. ``./Six Sines.vst3``.
    :returns: the gzipped tarball's bytes.
    """
    payload = io.BytesIO(b"fake plugin binary")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(f"{inner_dir}/Contents/x86_64-linux/plugin.so")
        info.size = len(payload.getvalue())
        tf.addfile(info, payload)
    return buf.getvalue()


def _home_env(checkout: Path) -> tuple[Path, dict[str, str]]:
    """Create a throwaway HOME under ``checkout`` and an env pointing at it.

    :param checkout: test checkout the HOME directory nests under.
    :returns: ``(home_path, env)`` for `_run_make` calls that must isolate the cache.
    """
    home = checkout / "home"
    return home, {**os.environ, "HOME": str(home)}


def _seed_cache(home: Path, asset_name: str, payload: bytes) -> str:
    """Place a fake cached archive under ``home`` and return its real sha256.

    :param home: throwaway HOME whose ``.cache/synth-setter/`` receives the archive.
    :param asset_name: archive filename the recipe will look for.
    :param payload: archive bytes to write.
    :returns: hex sha256 of ``payload``.
    """
    cache = home / ".cache" / "synth-setter"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / asset_name).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def makefile_checkout(tmp_path: Path) -> Path:
    """Provide a throwaway directory holding only the project Makefile.

    :param tmp_path: pytest scratch dir that receives the Makefile copy.
    :returns: the directory, ready for `make <target>` runs against it.
    """
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    return tmp_path


@pytest.mark.parametrize("pin", _SHARED_PINS)
def test_makefile_pin_matches_dockerfile_arg(pin: str) -> None:
    """Each fetched-synth pin in the Makefile equals the Dockerfile ARG default.

    :param pin: pin variable name present in both files.
    """
    assert _makefile_var(pin) == _dockerfile_arg(pin), (
        f"{pin} drifted between Makefile and {DOCKERFILE.relative_to(PROJECT_ROOT)}"
    )


def test_surge_version_matches_dockerfile_prebuilt_package() -> None:
    """The Makefile's Surge pin names the same release the image's prebuilt path installs."""
    version = _makefile_var("SURGE_XT_VERSION")
    assert f"{version}/surge-xt-linux-x64-{version}.deb" in DOCKERFILE.read_text(), (
        f"Dockerfile prebuilt Surge package does not match Makefile SURGE_XT_VERSION={version}"
    )


def test_runtime_image_installs_unzip_for_plugin_install_targets() -> None:
    """The runtime image has the zip extractor that Makefile plugin targets invoke."""
    stage = _dockerfile_stage_text("builder-install-synth-setter-deps")
    assert re.search(r"apt-get install\b[\s\S]*\bunzip\b", stage)


def test_install_plugins_all_bundles_present_skips_every_download(
    makefile_checkout: Path,
) -> None:
    """`make install-plugins` covers every image bundle and is a no-op when all exist.

    Pre-creating every bundle proves the aggregate target visits each image plugin without touching
    the network.

    :param makefile_checkout: throwaway checkout holding the Makefile.
    """
    plugins = makefile_checkout / "plugins"
    plugins.mkdir()
    for name in _IMAGE_BUNDLES:
        (plugins / name).mkdir()

    result = _run_make(makefile_checkout, "install-plugins")

    assert result.returncode == 0, result.stderr
    for name in _IMAGE_BUNDLES:
        assert f"plugins/{name} already exists" in result.stdout, f"{name} not visited"


@pytest.mark.parametrize("target", _LINUX_X86_64_PLUGIN_TARGETS)
def test_linux_x86_64_plugin_target_non_x86_64_skips_without_installing(
    makefile_checkout: Path, target: str
) -> None:
    """On a non-x86_64 host every x86_64 plugin target skips, mirroring the image gate.

    :param makefile_checkout: throwaway checkout holding the Makefile.
    :param target: plugin install make target under test.
    """
    bindir = makefile_checkout / "bin"
    bindir.mkdir()
    fake_uname = bindir / "uname"
    fake_uname.write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ]; then echo aarch64; else echo Linux; fi\n'
    )
    fake_uname.chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}

    result = _run_make(makefile_checkout, target, env=env)

    assert result.returncode == 0, result.stderr
    assert "skipping" in result.stdout
    assert not (makefile_checkout / "plugins").exists()


@requires_x86_64_linux
def test_install_dexed_cached_archive_verifies_and_installs_bundle(
    makefile_checkout: Path,
) -> None:
    """A cached archive with a matching sha256 is verified and extracted into plugins/.

    Exercises the full verify → extract → move path plus the 'Using cached' reuse branch, with the
    cache isolated under a throwaway HOME and the pin overridden to the fixture's real hash.

    :param makefile_checkout: throwaway checkout holding the Makefile.
    """
    home, env = _home_env(makefile_checkout)
    version = _makefile_var("DEXED_VERSION")
    payload = _zip_containing(f"dexed-{version}-lnx/Dexed.vst3")
    digest = _seed_cache(home, f"dexed-{version}-lnx.zip", payload)

    result = _run_make(makefile_checkout, "install-dexed", f"DEXED_SHA256={digest}", env=env)

    assert result.returncode == 0, result.stderr
    assert "Using cached" in result.stdout
    installed = makefile_checkout / "plugins" / "Dexed.vst3"
    assert (installed / "Contents" / "x86_64-linux" / "plugin.so").is_file()


@requires_x86_64_linux
def test_install_six_sines_cached_targz_verifies_and_installs_bundle(
    makefile_checkout: Path,
) -> None:
    """The .tgz extraction branch handles the space-containing Six Sines bundle name.

    :param makefile_checkout: throwaway checkout holding the Makefile.
    """
    home, env = _home_env(makefile_checkout)
    asset = _makefile_var("SIX_SINES_ASSET")
    payload = _targz_containing("./Six Sines.vst3")
    digest = _seed_cache(home, asset, payload)

    result = _run_make(
        makefile_checkout, "install-six-sines", f"SIX_SINES_SHA256={digest}", env=env
    )

    assert result.returncode == 0, result.stderr
    installed = makefile_checkout / "plugins" / "Six Sines.vst3"
    assert (installed / "Contents" / "x86_64-linux" / "plugin.so").is_file()


@requires_x86_64_linux
def test_install_plugins_mixed_presence_installs_only_missing_bundle(
    makefile_checkout: Path,
) -> None:
    """The aggregate target skips present bundles and installs the missing one in one run.

    :param makefile_checkout: throwaway checkout holding the Makefile.
    """
    home, env = _home_env(makefile_checkout)
    plugins = makefile_checkout / "plugins"
    plugins.mkdir()
    for name in (
        "Surge XT.vst3",
        "OB-Xf.vst3",
        "Six Sines.vst3",
        "Ultramaster KR-106.vst3",
    ):
        (plugins / name).mkdir()
    version = _makefile_var("DEXED_VERSION")
    payload = _zip_containing(f"dexed-{version}-lnx/Dexed.vst3")
    digest = _seed_cache(home, f"dexed-{version}-lnx.zip", payload)

    result = _run_make(makefile_checkout, "install-plugins", f"DEXED_SHA256={digest}", env=env)

    assert result.returncode == 0, result.stderr
    assert "plugins/Surge XT.vst3 already exists" in result.stdout
    assert "Installed plugins/Dexed.vst3" in result.stdout
    assert (plugins / "Dexed.vst3" / "Contents" / "x86_64-linux" / "plugin.so").is_file()


@requires_x86_64_linux
def test_install_ultramaster_existing_cache_refreshes_pinned_ref(
    makefile_checkout: Path,
) -> None:
    """An existing KR-106 cache still fetches and checks out the requested pin.

    :param makefile_checkout: throwaway checkout holding the Makefile.
    """
    home, env = _home_env(makefile_checkout)
    cache = home / ".cache" / "synth-setter" / "ultramaster-kr106-test"
    (cache / "src" / ".git").mkdir(parents=True)

    bindir = makefile_checkout / "bin"
    bindir.mkdir()
    log = makefile_checkout / "tool.log"
    fake_git = bindir / "git"
    fake_git.write_text('#!/bin/sh\nprintf "git %s\\n" "$*" >> "$TOOL_LOG"\n')
    fake_git.chmod(0o755)
    fake_cmake = bindir / "cmake"
    fake_cmake.write_text(
        "#!/bin/sh\n"
        'printf "cmake %s\\n" "$*" >> "$TOOL_LOG"\n'
        'if [ "$1" = "--build" ]; then\n'
        '  mkdir -p "$2/KR106_artefacts/Release/VST3/Ultramaster KR-106.vst3/Contents"\n'
        "fi\n"
    )
    fake_cmake.chmod(0o755)
    env = {**env, "PATH": f"{bindir}{os.pathsep}{env['PATH']}", "TOOL_LOG": str(log)}

    result = _run_make(
        makefile_checkout,
        "install-ultramaster-kr106",
        "ULTRAMASTER_KR106_VERSION=test",
        "ULTRAMASTER_KR106_GIT_REF=abc123",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    tool_log = log.read_text()
    assert "fetch --depth 1 origin abc123" in tool_log
    assert "checkout --detach FETCH_HEAD" in tool_log
    assert "reset --hard FETCH_HEAD" in tool_log
    assert (makefile_checkout / "plugins" / "Ultramaster KR-106.vst3").is_dir()


@requires_x86_64_linux
def test_install_ultramaster_invalid_cache_reinitializes_checkout(
    makefile_checkout: Path,
) -> None:
    """A malformed KR-106 cache is discarded before fetching the pinned ref.

    :param makefile_checkout: throwaway checkout holding the Makefile.
    """
    home, env = _home_env(makefile_checkout)
    cache = home / ".cache" / "synth-setter" / "ultramaster-kr106-test"
    (cache / "src" / ".git").mkdir(parents=True)

    bindir = makefile_checkout / "bin"
    bindir.mkdir()
    log = makefile_checkout / "tool.log"
    fake_git = bindir / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'workdir="$PWD"\n'
        'if [ "$1" = "-C" ]; then workdir="$2"; shift 2; fi\n'
        'printf "git -C %s %s\\n" "$workdir" "$*" >> "$TOOL_LOG"\n'
        'case "$1 $2" in\n'
        '  "rev-parse --git-dir") [ -f "$workdir/.git/valid" ]; exit $? ;;\n'
        '  "remote set-url") [ -f "$workdir/.git/valid" ]; exit $? ;;\n'
        "esac\n"
        'if [ "$1" = "init" ]; then mkdir -p "$workdir/.git"; touch "$workdir/.git/valid"; fi\n'
    )
    fake_git.chmod(0o755)
    fake_cmake = bindir / "cmake"
    fake_cmake.write_text(
        "#!/bin/sh\n"
        'printf "cmake %s\\n" "$*" >> "$TOOL_LOG"\n'
        'if [ "$1" = "--build" ]; then\n'
        '  mkdir -p "$2/KR106_artefacts/Release/VST3/Ultramaster KR-106.vst3/Contents"\n'
        "fi\n"
    )
    fake_cmake.chmod(0o755)
    env = {**env, "PATH": f"{bindir}{os.pathsep}{env['PATH']}", "TOOL_LOG": str(log)}

    result = _run_make(
        makefile_checkout,
        "install-ultramaster-kr106",
        "ULTRAMASTER_KR106_VERSION=test",
        "ULTRAMASTER_KR106_GIT_REF=abc123",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    tool_log = log.read_text()
    assert "rev-parse --git-dir" in tool_log
    assert "init" in tool_log
    assert (
        "remote set-url origin https://github.com/kayrockscreenprinting/ultramaster_kr106.git"
        in tool_log
    )
    assert "fetch --depth 1 origin abc123" in tool_log
    assert (makefile_checkout / "plugins" / "Ultramaster KR-106.vst3").is_dir()


@requires_x86_64_linux
def test_install_dexed_checksum_mismatch_fails_without_installing(
    makefile_checkout: Path,
) -> None:
    """A cached archive whose hash differs from the pin aborts before extraction.

    :param makefile_checkout: throwaway checkout holding the Makefile.
    """
    home, env = _home_env(makefile_checkout)
    version = _makefile_var("DEXED_VERSION")
    _seed_cache(home, f"dexed-{version}-lnx.zip", _zip_containing("x/Dexed.vst3"))

    result = _run_make(makefile_checkout, "install-dexed", f"DEXED_SHA256={'0' * 64}", env=env)

    assert result.returncode != 0
    assert "Remove the cached file and retry" in result.stderr
    assert not (makefile_checkout / "plugins" / "Dexed.vst3").exists()


@requires_x86_64_linux
def test_install_six_sines_unsupported_archive_type_fails(makefile_checkout: Path) -> None:
    """An asset name with an unknown extension fails after checksum, before extraction.

    :param makefile_checkout: throwaway checkout holding the Makefile.
    """
    home, env = _home_env(makefile_checkout)
    payload = b"not an archive"
    digest = _seed_cache(home, "six-sines.rar", payload)

    result = _run_make(
        makefile_checkout,
        "install-six-sines",
        "SIX_SINES_ASSET=six-sines.rar",
        f"SIX_SINES_SHA256={digest}",
        env=env,
    )

    assert result.returncode != 0
    assert "unsupported archive type" in result.stderr
    assert not (makefile_checkout / "plugins" / "Six Sines.vst3").exists()


@requires_x86_64_linux
def test_install_dexed_archive_missing_bundle_fails(makefile_checkout: Path) -> None:
    """An archive that lacks `<Bundle>.vst3` fails with a clear error and installs nothing.

    :param makefile_checkout: throwaway checkout holding the Makefile.
    """
    home, env = _home_env(makefile_checkout)
    version = _makefile_var("DEXED_VERSION")
    payload = _zip_containing(f"dexed-{version}-lnx/NotThePlugin.vst3")
    digest = _seed_cache(home, f"dexed-{version}-lnx.zip", payload)

    result = _run_make(makefile_checkout, "install-dexed", f"DEXED_SHA256={digest}", env=env)

    assert result.returncode != 0
    assert "Dexed.vst3 not found" in result.stderr
    assert not (makefile_checkout / "plugins" / "Dexed.vst3").exists()
