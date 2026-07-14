"""Minimal ctypes CLAP host that enumerates a plugin's parameters (#1787).

First-party replacement for ``free-audio/clap-info``: dlopens a ``.clap``
bundle, walks ``clap_entry`` → plugin factory → ``clap.params``, and returns
every parameter's id/name/module/range. Only the structs this walk touches are
declared; layouts follow the stable CLAP 1.x C ABI (github.com/free-audio/clap,
``include/clap``). No audio processing and no GUI — the plugin is created and
queried on the main thread, then destroyed.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from pydantic import BaseModel, ConfigDict

#: Where the surge-xt system package installs the CLAP on Linux.
SURGE_XT_CLAP_PATH = Path("/usr/lib/clap/Surge XT.clap")

_CLAP_NAME_SIZE = 256
_CLAP_PATH_SIZE = 1024
_CLAP_PARAM_IS_STEPPED = 1 << 0
_PLUGIN_FACTORY_ID = b"clap.plugin-factory"
_PARAMS_EXTENSION_ID = b"clap.params"


# DOC601/603: pydoclint can't see :ivar: docs on pydantic fields (#1787);
# both models mirror CLAP C structs field-for-field.
class ClapParamInfo(BaseModel):  # noqa: DOC601, DOC603
    """One entry of a raw CLAP param dump (mirrors ``clap_param_info``)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: int
    name: str
    module: str
    min_value: float
    max_value: float
    default_value: float
    flags: int
    is_stepped: bool


class ClapPluginInfo(BaseModel):  # noqa: DOC601, DOC603
    """Raw CLAP dump: plugin descriptor plus all params in enumeration order."""

    model_config = ConfigDict(strict=True, extra="forbid")

    plugin_id: str
    plugin_name: str
    vendor: str
    version: str
    clap_version: str
    params: list[ClapParamInfo]


class _ClapVersion(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("revision", ctypes.c_uint32),
    ]


class _ClapHost(ctypes.Structure):
    pass


_HOST_GET_EXTENSION = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(_ClapHost), ctypes.c_char_p)
_HOST_VOID_CB = ctypes.CFUNCTYPE(None, ctypes.POINTER(_ClapHost))

_ClapHost._fields_ = [
    ("clap_version", _ClapVersion),
    ("host_data", ctypes.c_void_p),
    ("name", ctypes.c_char_p),
    ("vendor", ctypes.c_char_p),
    ("url", ctypes.c_char_p),
    ("version", ctypes.c_char_p),
    ("get_extension", _HOST_GET_EXTENSION),
    ("request_restart", _HOST_VOID_CB),
    ("request_process", _HOST_VOID_CB),
    ("request_callback", _HOST_VOID_CB),
]


class _ClapPluginDescriptor(ctypes.Structure):
    _fields_ = [
        ("clap_version", _ClapVersion),
        ("id", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("vendor", ctypes.c_char_p),
        ("url", ctypes.c_char_p),
        ("manual_url", ctypes.c_char_p),
        ("support_url", ctypes.c_char_p),
        ("version", ctypes.c_char_p),
        ("description", ctypes.c_char_p),
        ("features", ctypes.POINTER(ctypes.c_char_p)),
    ]


class _ClapPlugin(ctypes.Structure):
    pass


_ClapPlugin._fields_ = [
    ("desc", ctypes.POINTER(_ClapPluginDescriptor)),
    ("plugin_data", ctypes.c_void_p),
    ("init", ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(_ClapPlugin))),
    ("destroy", ctypes.CFUNCTYPE(None, ctypes.POINTER(_ClapPlugin))),
    (
        "activate",
        ctypes.CFUNCTYPE(
            ctypes.c_bool,
            ctypes.POINTER(_ClapPlugin),
            ctypes.c_double,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ),
    ),
    ("deactivate", ctypes.CFUNCTYPE(None, ctypes.POINTER(_ClapPlugin))),
    ("start_processing", ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(_ClapPlugin))),
    ("stop_processing", ctypes.CFUNCTYPE(None, ctypes.POINTER(_ClapPlugin))),
    ("reset", ctypes.CFUNCTYPE(None, ctypes.POINTER(_ClapPlugin))),
    ("process", ctypes.c_void_p),
    (
        "get_extension",
        ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(_ClapPlugin), ctypes.c_char_p),
    ),
    ("on_main_thread", ctypes.CFUNCTYPE(None, ctypes.POINTER(_ClapPlugin))),
]


class _ClapPluginFactory(ctypes.Structure):
    pass


_ClapPluginFactory._fields_ = [
    ("get_plugin_count", ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(_ClapPluginFactory))),
    (
        "get_plugin_descriptor",
        ctypes.CFUNCTYPE(
            ctypes.POINTER(_ClapPluginDescriptor),
            ctypes.POINTER(_ClapPluginFactory),
            ctypes.c_uint32,
        ),
    ),
    (
        "create_plugin",
        ctypes.CFUNCTYPE(
            ctypes.POINTER(_ClapPlugin),
            ctypes.POINTER(_ClapPluginFactory),
            ctypes.POINTER(_ClapHost),
            ctypes.c_char_p,
        ),
    ),
]


class _ClapPluginEntry(ctypes.Structure):
    _fields_ = [
        ("clap_version", _ClapVersion),
        ("init", ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)),
        ("deinit", ctypes.CFUNCTYPE(None)),
        ("get_factory", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)),
    ]


class _ClapParamInfo(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("cookie", ctypes.c_void_p),
        ("name", ctypes.c_char * _CLAP_NAME_SIZE),
        ("module", ctypes.c_char * _CLAP_PATH_SIZE),
        ("min_value", ctypes.c_double),
        ("max_value", ctypes.c_double),
        ("default_value", ctypes.c_double),
    ]


class _ClapPluginParams(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(_ClapPlugin))),
        (
            "get_info",
            ctypes.CFUNCTYPE(
                ctypes.c_bool,
                ctypes.POINTER(_ClapPlugin),
                ctypes.c_uint32,
                ctypes.POINTER(_ClapParamInfo),
            ),
        ),
        (
            "get_value",
            ctypes.CFUNCTYPE(
                ctypes.c_bool,
                ctypes.POINTER(_ClapPlugin),
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_double),
            ),
        ),
        ("value_to_text", ctypes.c_void_p),
        ("text_to_value", ctypes.c_void_p),
        ("flush", ctypes.c_void_p),
    ]


def _shared_library_path(plugin_path: Path) -> Path:
    """Resolve the dlopen-able binary inside a ``.clap`` bundle.

    Linux ships the ``.clap`` as a bare shared object; macOS ships a bundle
    directory whose binary lives at ``Contents/MacOS/<bundle stem>``.

    :param plugin_path: Path to the ``.clap`` file or bundle directory.
    :returns: Path suitable for ``ctypes.CDLL``.
    :raises FileNotFoundError: when the plugin (or its bundle binary) is absent.
    """
    if not plugin_path.exists():
        raise FileNotFoundError(f"CLAP plugin not found: {plugin_path}")
    if plugin_path.is_dir():
        binary = plugin_path / "Contents" / "MacOS" / plugin_path.stem
        if not binary.is_file():
            raise FileNotFoundError(f"CLAP bundle has no binary at {binary}")
        return binary
    return plugin_path


def _make_host() -> _ClapHost:
    """Build the no-op host handed to ``create_plugin``.

    Returning ``None`` from ``get_extension`` is legal per the CLAP spec —
    plugins must tolerate hosts that provide no extensions.

    :returns: Host struct; the caller must keep it alive while the plugin exists.
    """
    return _ClapHost(
        clap_version=_ClapVersion(1, 2, 2),
        host_data=None,
        name=b"synth-setter clap introspector",
        vendor=b"synth-setter",
        url=b"https://github.com/tinaudio/synth-setter",
        version=b"1.0.0",
        get_extension=_HOST_GET_EXTENSION(lambda _host, _ext: None),
        request_restart=_HOST_VOID_CB(lambda _host: None),
        request_process=_HOST_VOID_CB(lambda _host: None),
        request_callback=_HOST_VOID_CB(lambda _host: None),
    )


def _read_param_infos(plugin: ctypes._Pointer, params: _ClapPluginParams) -> list[ClapParamInfo]:
    """Read every ``clap_param_info`` from an initialized plugin.

    :param plugin: Pointer to the created-and-initialized ``clap_plugin``.
    :param params: The plugin's ``clap.params`` extension vtable.
    :returns: One entry per param, in enumeration order.
    :raises RuntimeError: when ``get_info`` fails for any index.
    """
    infos = []
    for i in range(params.count(plugin)):
        raw = _ClapParamInfo()
        if not params.get_info(plugin, i, ctypes.byref(raw)):
            raise RuntimeError(f"clap.params get_info failed at index {i}")
        infos.append(
            ClapParamInfo(
                id=raw.id,
                name=raw.name.decode(errors="replace"),
                module=raw.module.decode(errors="replace"),
                min_value=raw.min_value,
                max_value=raw.max_value,
                default_value=raw.default_value,
                flags=raw.flags,
                is_stepped=bool(raw.flags & _CLAP_PARAM_IS_STEPPED),
            )
        )
    return infos


# DOC503: FileNotFoundError comes from the _shared_library_path helper.
def dump_clap_plugin(plugin_path: Path, plugin_index: int = 0) -> ClapPluginInfo:  # noqa: DOC503
    """Enumerate every parameter of a CLAP plugin without processing audio.

    :param plugin_path: Path to the ``.clap`` shared object (or macOS bundle).
    :param plugin_index: Factory index of the plugin class to open (Surge XT
        exposes the instrument at 0 and effects bundles separately).
    :returns: Descriptor fields plus all params in the plugin's enumeration
        order (this order matches pedalboard's VST3 parameter indices for
        Surge XT — the bridge ``tools/build_param_map.py`` relies on).
    :raises FileNotFoundError: when the plugin binary is absent.
    :raises RuntimeError: when any CLAP entry/factory/extension step fails.
    """
    library_path = _shared_library_path(plugin_path)
    # The dlopen handle is never released — fine for one-shot CLI use; don't
    # call this in a long-lived loop without adding dlclose handling.
    try:
        library = ctypes.CDLL(str(library_path))
        entry_symbol = library.clap_entry
    except (OSError, AttributeError) as exc:
        raise RuntimeError(f"failed to load CLAP shared library {library_path}: {exc}") from exc
    entry = ctypes.cast(entry_symbol, ctypes.POINTER(_ClapPluginEntry)).contents
    if not entry.init(str(plugin_path).encode()):
        raise RuntimeError(f"clap_entry.init failed for {plugin_path}")

    try:
        factory_ptr = entry.get_factory(_PLUGIN_FACTORY_ID)
        if not factory_ptr:
            raise RuntimeError(f"{plugin_path} exposes no clap.plugin-factory")
        factory = ctypes.cast(factory_ptr, ctypes.POINTER(_ClapPluginFactory))

        count = factory.contents.get_plugin_count(factory)
        if plugin_index >= count:
            raise RuntimeError(f"plugin index {plugin_index} out of range (factory has {count})")
        descriptor = factory.contents.get_plugin_descriptor(factory, plugin_index).contents

        # The host must outlive the plugin instance; keep the reference local
        # until destroy() below.
        host = _make_host()
        plugin = factory.contents.create_plugin(factory, ctypes.byref(host), descriptor.id)
        if not plugin:
            raise RuntimeError(f"create_plugin failed for {descriptor.id!r}")
        try:
            if not plugin.contents.init(plugin):
                raise RuntimeError(f"plugin init failed for {descriptor.id!r}")
            params_ptr = plugin.contents.get_extension(plugin, _PARAMS_EXTENSION_ID)
            if not params_ptr:
                raise RuntimeError(f"{descriptor.id!r} exposes no clap.params extension")
            params = ctypes.cast(params_ptr, ctypes.POINTER(_ClapPluginParams)).contents
            infos = _read_param_infos(plugin, params)
        finally:
            plugin.contents.destroy(plugin)

        clap_version = entry.clap_version
        return ClapPluginInfo(
            plugin_id=descriptor.id.decode(),
            plugin_name=descriptor.name.decode(),
            vendor=descriptor.vendor.decode(),
            version=descriptor.version.decode(),
            clap_version=f"{clap_version.major}.{clap_version.minor}.{clap_version.revision}",
            params=infos,
        )
    finally:
        entry.deinit()
