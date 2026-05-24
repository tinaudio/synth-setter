"""Hydra plugin: register ``synth_setter``'s shipped configs on the search path."""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class SynthSetterSearchPathPlugin(SearchPathPlugin):
    """Append ``pkg://synth_setter.configs`` to every Hydra search path.

    Auto-discovered by Hydra under the ``hydra_plugins`` namespace package
    when ``synth_setter`` is on the Python path. Downstream apps can then
    compose against ``dataset_render/wds`` / ``model/default`` etc.
    without knowing the on-disk location of the YAMLs.
    """

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Append the packaged-config URI to ``search_path``.

        :param search_path: Hydra's mutable search-path object.
        """
        search_path.append(
            provider="synth-setter-searchpath-plugin",
            path="pkg://synth_setter.configs",
        )
