window.BENCHMARK_DATA = {
  "lastUpdate": 1777506501763,
  "repoUrl": "https://github.com/tinaudio/synth-setter",
  "entries": {
    "VST fixed-params replay": [
      {
        "commit": {
          "author": {
            "email": "17952332+ktinubu@users.noreply.github.com",
            "name": "KT",
            "username": "ktinubu"
          },
          "committer": {
            "email": "17952332+ktinubu@users.noreply.github.com",
            "name": "KT",
            "username": "ktinubu"
          },
          "distinct": true,
          "id": "29809b37b19795f431cbe9a86fb72332312c3430",
          "message": "ci(test-vst): drop in-container symlink + add VST smoke + dummy fast-path\n\nThree changes to ``.github/workflows/test-vst-slow.yml``:\n\n1. Drop the ``mkdir -p plugins; ln -sf`` lines from the docker run. The\n   base image already places the VST3 at ``/usr/lib/vst3/Surge XT.vst3``,\n   and the bind mount over ``/home/build/synth-setter`` hides the\n   image-side symlink that the Dockerfile creates. Set\n   ``SYNTH_SETTER_PLUGIN_PATH=/usr/lib/vst3/Surge XT.vst3`` so the test\n   uses the absolute path the .deb installs to.\n\n2. Add a ``Smoke-test Surge XT plugin load`` step before the test step,\n   mirroring the local-runner smoke check in ``test-expensive.yml``.\n   Fails fast if the plugin / image / mount layout is broken before\n   committing to the much-longer pytest run.\n\n3. Add a ``dummy_only`` workflow_dispatch input + a\n   ``Write hardcoded dummy bench.json`` step gated on it. When set, the\n   pull / smoke / test / surface steps are skipped and a hand-crafted\n   ``bench.json`` is written directly to the workspace. Lets a maintainer\n   iterate on the publish-step gating in ~10 seconds instead of ~5\n   minutes per cycle. Implies ``publish_metrics``.\n\nAlso revert the ``skip-fetch-gh-pages: true`` flag now that the\n``gh-pages`` branch exists on the remote — the action's default fetch\npath now resolves it cleanly.\n\nRefs #703",
          "timestamp": "2026-04-29T23:41:55Z",
          "tree_id": "80a609507b4b288e2cb31042c0382c11d5101760",
          "url": "https://github.com/tinaudio/synth-setter/commit/29809b37b19795f431cbe9a86fb72332312c3430"
        },
        "date": 1777506501174,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-fixed-replay/mss-max",
            "value": 3.370206832885742,
            "unit": "dB"
          },
          {
            "name": "vst-fixed-replay/wmfcc-max",
            "value": 5.600218626610004,
            "unit": "L1"
          },
          {
            "name": "vst-fixed-replay/sot-max",
            "value": 0.023114768788218498,
            "unit": "W"
          },
          {
            "name": "vst-fixed-replay/rms-distance-max",
            "value": 0.013454079627990723,
            "unit": "1-cos"
          },
          {
            "name": "vst-fixed-replay/mel-mean-abs",
            "value": 2.0938022136688232,
            "unit": "dB"
          }
        ]
      }
    ]
  }
}