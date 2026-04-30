window.BENCHMARK_DATA = {
  "lastUpdate": 1777512284146,
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
    ],
    "VST noise floor": [
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
          "id": "41d64283e51f6f87a8f43eb0d2040d5022299c9e",
          "message": "ci(test-vst): rename benchmark bucket + use full metric names\n\nBucket: ``VST fixed-params replay`` → ``VST noise floor``. Reflects what\nthe test actually measures — the floor of how well two render passes of\nidentical params reproduce each other under the docker mitigation stack\n— rather than the now-misnamed historical reference to the\n``fixed_*_params_list`` API the test no longer uses.\n\nMetric series: drop project-internal abbreviations in favor of full\nnames so the chart's left-hand legend is self-explanatory.\n\n  mss-max          → multi-scale-spectral-loss-max\n  wmfcc-max        → dtw-aligned-mfcc-distance-max\n  sot-max          → spectral-optimal-transport-max  (unit: W → Wasserstein)\n  rms-distance-max → rms-envelope-cosine-distance-max\n  mel-mean-abs     → mel-spectrogram-mean-absolute-error\n\nAlso rename the ``benchmark_name_prefix`` argument from\n``vst-fixed-replay`` to ``vst-noise-floor`` so the on-chart series\nstrings are consistent with the bucket.\n\nThe single existing bootstrap data point on ``gh-pages`` will be\norphaned under the old bucket name — left for now since deleting it\nwould mean a force-push to ``gh-pages`` and the noise-floor chart only\nbecomes meaningful once a few runs land anyway.\n\nRefs #703",
          "timestamp": "2026-04-29T23:55:52Z",
          "tree_id": "bd4018372bd9ad435013f5a22c18ab30a96de364",
          "url": "https://github.com/tinaudio/synth-setter/commit/41d64283e51f6f87a8f43eb0d2040d5022299c9e"
        },
        "date": 1777507341505,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor/multi-scale-spectral-loss-max",
            "value": 3.4450809955596924,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor/dtw-aligned-mfcc-distance-max",
            "value": 5.758509016435128,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor/spectral-optimal-transport-max",
            "value": 0.019516294822096825,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor/rms-envelope-cosine-distance-max",
            "value": 0.01784980297088623,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor/mel-spectrogram-mean-absolute-error",
            "value": 1.9600520133972168,
            "unit": "dB"
          }
        ]
      }
    ],
    "VST noise floor (1 preset N renders)": [
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
          "id": "df4f983868854729dcb606167b380ea9d08d3ade",
          "message": "fix(test-vst): address PR #706 review feedback\n\n- Reword `render_params` reload references to present-tense bug-#489\n  descriptions; drop forward-references to the unmerged per-render\n  reload workaround (commits 086d80f / 9ff7f16, PR #702).\n- Sync hardcoded-params docstring `num_samples` and test-name\n  references to the actual `test_datasets_from_hardcoded_params_are_identical`\n  body (num_samples=6, all-pairs check rationale).\n- Sync sampled-params docstring rationale to match issue #489\n  framing (drop the workaround commit citations).\n- Cache `mel[...]` and `params[...]` reads in `_assert_h5_structure_is_valid`\n  to avoid double materialization.\n- Handle JSONDecodeError in `_emit_benchmark_metrics` by treating a\n  truncated bench file as an empty list.\n- Pin `benchmark-action/github-action-benchmark@v1` -> the v1.22.0\n  commit SHA in `test-vst-slow.yml` for supply-chain hygiene.\n- Update `docs/reference/audio-similarity-benchmarks.md` to drop the\n  forward-reference to the unmerged per-render reload workaround.\n\nRefs #489",
          "timestamp": "2026-04-30T00:18:11Z",
          "tree_id": "f5886d7abd107efb7a3fbe1eba3ca7f3fb5b86c4",
          "url": "https://github.com/tinaudio/synth-setter/commit/df4f983868854729dcb606167b380ea9d08d3ade"
        },
        "date": 1777508674459,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 4.453965663909912,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.547367088198662,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.035552944988012314,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.03994864225387573,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 2.4953606128692627,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 5.353806434583333,
            "unit": "seconds"
          }
        ]
      },
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
          "id": "ac73c774f513b1aad91784b923816262207d091e",
          "message": "fix(test-vst): address PR #706 review feedback (round 2)\n\nDoc/wording fixes only — no behavior change:\n\n- _assert_round_trip_matches docstring: ``BENCHMARK_OUTPUT_PATH`` →\n  ``BENCHMARK_OUTPUT_DIR`` (matches the actual env var read by\n  _emit_benchmark_metrics and set by test-vst-slow.yml). Comment\n  3164945781.\n- docs/reference/audio-similarity-benchmarks.md: \"six series\" → \"seven\n  series\" with explicit call-out of the two non-distance sentinels\n  (num-samples, wall-clock-seconds-per-render); the metric table\n  already listed seven rows. Comment 3164945796.\n- test-vst-slow.yml dummy_only fast-path: include num-samples and\n  wall-clock-seconds-per-render in the hardcoded bench JSON so the\n  debug-only payload mirrors what _assert_round_trip_matches actually\n  emits. Comment 3164945820.\n\nComment 3164945810 (temp branch in push.branches) is a duplicate of\nthe round-1 thread already justified at 3164936475 / 3164936515 — kept\nintentionally and gated by an in-file removal note; will be reverted\nin a follow-up before merge once the gh-pages chart is bootstrapped.\n\nxfail decorators, _HARDCODED_*_PARAMS, and gh-pages branch are not\ntouched.\n\nRefs #489\nRefs #703",
          "timestamp": "2026-04-30T00:29:33Z",
          "tree_id": "904910829e37bcabc324603891465ff49d0656ed",
          "url": "https://github.com/tinaudio/synth-setter/commit/ac73c774f513b1aad91784b923816262207d091e"
        },
        "date": 1777509343958,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.1963846683502197,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.8113422030210495,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.04953543841838837,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.020019829273223877,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 2.0885043144226074,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 5.2876903134166655,
            "unit": "seconds"
          }
        ]
      },
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
          "id": "d3b5b257b991d21a7fdb608abfa573b8f745e5e5",
          "message": "chore(test-vst): remove dummy fast-path debug code from workflow\n\nThe ``dummy_only`` workflow_dispatch input + ``Write hardcoded dummy\nbench JSON files (debug-only fast path)`` step + all\n``inputs.dummy_only`` references were scaffolding for iterating on the\npublish-step gating during the gh-pages bootstrap. The chart is live\nand the publish path is verified, so the dummy code is no longer\nload-bearing — it just adds noise to the workflow and gives operators\na footgun (publishing junk to gh-pages by accident).\n\nReverts:\n- ``dummy_only`` dispatch input\n- \"Write hardcoded dummy bench JSON files\" step\n- ``if: inputs.dummy_only != true`` gates on Pull image, Smoke-test,\n  Run VST tests, Surface\n- ``inputs.dummy_only == true`` clauses in both publish steps' ``if:``\n\nRefs #703",
          "timestamp": "2026-04-30T00:38:29Z",
          "tree_id": "fca565b9cd002e834bcc032bb343ac51df511643",
          "url": "https://github.com/tinaudio/synth-setter/commit/d3b5b257b991d21a7fdb608abfa573b8f745e5e5"
        },
        "date": 1777509908401,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 4.2995452880859375,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.297968615693971,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.039796262979507446,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.035881638526916504,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 2.600921392440796,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 5.0601840535,
            "unit": "seconds"
          }
        ]
      }
    ],
    "VST noise floor (random preset replay)": [
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
          "id": "df4f983868854729dcb606167b380ea9d08d3ade",
          "message": "fix(test-vst): address PR #706 review feedback\n\n- Reword `render_params` reload references to present-tense bug-#489\n  descriptions; drop forward-references to the unmerged per-render\n  reload workaround (commits 086d80f / 9ff7f16, PR #702).\n- Sync hardcoded-params docstring `num_samples` and test-name\n  references to the actual `test_datasets_from_hardcoded_params_are_identical`\n  body (num_samples=6, all-pairs check rationale).\n- Sync sampled-params docstring rationale to match issue #489\n  framing (drop the workaround commit citations).\n- Cache `mel[...]` and `params[...]` reads in `_assert_h5_structure_is_valid`\n  to avoid double materialization.\n- Handle JSONDecodeError in `_emit_benchmark_metrics` by treating a\n  truncated bench file as an empty list.\n- Pin `benchmark-action/github-action-benchmark@v1` -> the v1.22.0\n  commit SHA in `test-vst-slow.yml` for supply-chain hygiene.\n- Update `docs/reference/audio-similarity-benchmarks.md` to drop the\n  forward-reference to the unmerged per-render reload workaround.\n\nRefs #489",
          "timestamp": "2026-04-30T00:18:11Z",
          "tree_id": "f5886d7abd107efb7a3fbe1eba3ca7f3fb5b86c4",
          "url": "https://github.com/tinaudio/synth-setter/commit/df4f983868854729dcb606167b380ea9d08d3ade"
        },
        "date": 1777508676343,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 3.1836397647857666,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 4.369372892677784,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.01779405027627945,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.04124796390533447,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.8287198543548584,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 4.8635790631000075,
            "unit": "seconds"
          }
        ]
      },
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
          "id": "ac73c774f513b1aad91784b923816262207d091e",
          "message": "fix(test-vst): address PR #706 review feedback (round 2)\n\nDoc/wording fixes only — no behavior change:\n\n- _assert_round_trip_matches docstring: ``BENCHMARK_OUTPUT_PATH`` →\n  ``BENCHMARK_OUTPUT_DIR`` (matches the actual env var read by\n  _emit_benchmark_metrics and set by test-vst-slow.yml). Comment\n  3164945781.\n- docs/reference/audio-similarity-benchmarks.md: \"six series\" → \"seven\n  series\" with explicit call-out of the two non-distance sentinels\n  (num-samples, wall-clock-seconds-per-render); the metric table\n  already listed seven rows. Comment 3164945796.\n- test-vst-slow.yml dummy_only fast-path: include num-samples and\n  wall-clock-seconds-per-render in the hardcoded bench JSON so the\n  debug-only payload mirrors what _assert_round_trip_matches actually\n  emits. Comment 3164945820.\n\nComment 3164945810 (temp branch in push.branches) is a duplicate of\nthe round-1 thread already justified at 3164936475 / 3164936515 — kept\nintentionally and gated by an in-file removal note; will be reverted\nin a follow-up before merge once the gh-pages chart is bootstrapped.\n\nxfail decorators, _HARDCODED_*_PARAMS, and gh-pages branch are not\ntouched.\n\nRefs #489\nRefs #703",
          "timestamp": "2026-04-30T00:29:33Z",
          "tree_id": "904910829e37bcabc324603891465ff49d0656ed",
          "url": "https://github.com/tinaudio/synth-setter/commit/ac73c774f513b1aad91784b923816262207d091e"
        },
        "date": 1777509345908,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 1.0599759817123413,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 1.8633526645600795,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.008008966222405434,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.06360280513763428,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.0836472511291504,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 4.300568791100005,
            "unit": "seconds"
          }
        ]
      },
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
          "id": "d3b5b257b991d21a7fdb608abfa573b8f745e5e5",
          "message": "chore(test-vst): remove dummy fast-path debug code from workflow\n\nThe ``dummy_only`` workflow_dispatch input + ``Write hardcoded dummy\nbench JSON files (debug-only fast path)`` step + all\n``inputs.dummy_only`` references were scaffolding for iterating on the\npublish-step gating during the gh-pages bootstrap. The chart is live\nand the publish path is verified, so the dummy code is no longer\nload-bearing — it just adds noise to the workflow and gives operators\na footgun (publishing junk to gh-pages by accident).\n\nReverts:\n- ``dummy_only`` dispatch input\n- \"Write hardcoded dummy bench JSON files\" step\n- ``if: inputs.dummy_only != true`` gates on Pull image, Smoke-test,\n  Run VST tests, Surface\n- ``inputs.dummy_only == true`` clauses in both publish steps' ``if:``\n\nRefs #703",
          "timestamp": "2026-04-30T00:38:29Z",
          "tree_id": "fca565b9cd002e834bcc032bb343ac51df511643",
          "url": "https://github.com/tinaudio/synth-setter/commit/d3b5b257b991d21a7fdb608abfa573b8f745e5e5"
        },
        "date": 1777509911377,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 1.4437776803970337,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 1.5748991463705897,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.008739760145545006,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.017881572246551514,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.5714856386184692,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 4.034858916199999,
            "unit": "seconds"
          }
        ]
      },
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
          "id": "462bf63f0a9a4957632098a8c0bce889b5dcbc0d",
          "message": "refactor(test-vst): factor benchmark emission out of round-trip helper\n\nPer PR review feedback (r3165027905): the published \"1 preset N renders\"\nchart was wired to per-pair metrics, but the #489 reproducer is the\nall-pairs worst-case across the union of renders. The chart could look\nflat while the test xfails on the all-pairs assertion.\n\nRefactor:\n- New ``RoundTripMetrics`` and ``AllPairsMetrics`` frozen dataclasses\n  hold the four audio metrics + their respective extras (mel diff +\n  num_samples for round-trip; pair count for all-pairs).\n- ``_assert_round_trip_matches`` returns ``RoundTripMetrics`` and no\n  longer has any benchmark-emit logic. Drops ``benchmark_name_prefix``\n  and ``total_render_seconds`` params.\n- ``_assert_all_pairs_audio_metrics_within_thresholds`` returns\n  ``AllPairsMetrics``.\n- New ``_emit_audio_similarity_benchmark_metrics(prefix, round_trip,\n  all_pairs, total_render_seconds)`` consumes either or both structs\n  and writes the bench JSON. Round-trip series go under ``<prefix>/``;\n  all-pairs series go under ``<prefix>/all-pairs-`` so both can coexist\n  on the same chart bucket without name collisions.\n- Hardcoded test now emits BOTH structs — round-trip for context,\n  all-pairs as the primary regression signal for #489.\n- Sampled test still emits only round-trip (cross-row pairs differ\n  legitimately, no all-pairs check applies).\n\nAdds six unit tests for ``_emit_audio_similarity_benchmark_metrics``\ncovering: env-unset no-op, round-trip-only schema, all-pairs-only\nschema, both-structs namespace separation, no-args no-write, and\nappend-on-second-call. All run in <1s without the VST.\n\nUpdates ``docs/reference/audio-similarity-benchmarks.md`` to document\nthe new ``all-pairs-*`` series + their role as the primary #489 signal\non the hardcoded bucket.\n\nRefs #489\nRefs #703",
          "timestamp": "2026-04-30T01:11:47Z",
          "tree_id": "1b1da9859cfc8e5a09bd564307de7fa8a13ce321",
          "url": "https://github.com/tinaudio/synth-setter/commit/462bf63f0a9a4957632098a8c0bce889b5dcbc0d"
        },
        "date": 1777511910911,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 3.387610673904419,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 4.407728461921215,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.008398685604333878,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.003195464611053467,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.6361103057861328,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 4.252929640400009,
            "unit": "seconds"
          }
        ]
      },
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
          "id": "788d22f915a803f3afd51f94fef8bec9e8530ca0",
          "message": "docs(test-vst): make hardcoded-test docstring self-contained\n\nDrops the 'Variant of test_datasets_from_sampled_params_are_identical'\nframing and rewrites as a standalone description of what the test\nactually does.\n\nRefs #703",
          "timestamp": "2026-04-30T01:18:05Z",
          "tree_id": "10df6f91ab63787018fcb39bc4007e1f7f18abc9",
          "url": "https://github.com/tinaudio/synth-setter/commit/788d22f915a803f3afd51f94fef8bec9e8530ca0"
        },
        "date": 1777512283223,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.351823091506958,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.97067511998117,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.017272258177399635,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.016191601753234863,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.598201036453247,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 4.382360739800004,
            "unit": "seconds"
          }
        ]
      }
    ]
  }
}