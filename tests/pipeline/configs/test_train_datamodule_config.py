"""Config-composition tests for the ``train`` datamodule.

These compose ``train.yaml`` and instantiate the datamodule the way ``train``
does, without driving the entrypoint. They live here — not in
``tests/test_train.py`` — because they manage their own Hydra lifecycle via
``initialize_config_module``; the entrypoint suite must stay entrypoint-only
(`tests/_meta/test_entrypoint_e2e_only.py`).
"""

from __future__ import annotations

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate


def test_train_builds_vst_datamodule_with_ram_bounded_num_workers() -> None:
    """The datamodule train instantiates carries the RAM-bounded worker default.

    ``num_workers`` is applied per dataloader, so enabling validation doubles the
    live worker count. Lance workers are ~1.4 GB each, and the previous default
    of 11 put train+val pools past a 32 GB host, where the OOM killer reaped the
    run before its first checkpoint (#1916).

    Instantiates the datamodule the way ``train`` does rather than asserting the
    composed dict, so the default is checked where it is consumed. Composed
    explicitly rather than via ``cfg_train``: those fixtures pin ``num_workers``
    themselves, so no other train test would notice the default drifting up.
    """
    GlobalHydra.instance().clear()
    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="train.yaml",
                return_hydra_config=True,
                overrides=["datamodule=surge_simple", "model=ffn", "trainer=cpu"],
            )
        HydraConfig().set_config(cfg)
        datamodule = instantiate(cfg.datamodule)
    finally:
        GlobalHydra.instance().clear()
    assert datamodule.num_workers == 4
