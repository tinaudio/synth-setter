import pytest

from src.data.vst.core import load_plugin


def test_load_plugin_missing_file_raises_filenotfound(tmp_path):
    """load_plugin must raise FileNotFoundError when the plugin path does not exist."""
    missing = tmp_path / "does-not-exist.vst3"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        load_plugin(str(missing))
