window.BENCHMARK_DATA = {
  "lastUpdate": 1777676892401,
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
      },
      {
        "commit": {
          "author": {
            "email": "17952332+ktinubu@users.noreply.github.com",
            "name": "KT",
            "username": "ktinubu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ceaf0fc54f29e875edba3e60a7b575b39d8ec41c",
          "message": "fix(vst): reload plugin per render to eliminate every-other junk audio (#713)\n\n* fix(vst): reload plugin per render to eliminate every-other junk audio\n\nrender_params now takes a plugin_path and reloads the VST3 plugin on every\ncall, working around a stale-state bug where alternating renders produced\nsilent or repeated audio. load_plugin's editor-pump uses a threading.Event\n+ show_editor(stop_event) pattern (replacing the prior _thread.interrupt\nKeyboardInterrupt hack), which is what makes a per-call reload safe and\nfast enough to be the default.\n\ngenerate_sample, make_dataset, and scripts/predict_vst_audio.py are\nupdated to pass plugin_path through to render_params instead of\npre-loading the plugin.\n\nThe xfail decorator on\ntest_datasets_from_hardcoded_params_are_identical is removed: with this\nfix in place, the test no longer xpasses.\n\nCloses #489\nRefs #705\nRefs #702\n\n* docs(eval): update audio-similarity-benchmarks for #489 closure\n\nThe dashboard's framing described #489 as an open bug and called the\nall-pairs series its \"regression signal\". With #713 closing #489 via\nper-render plugin reload, the framing inverts: the all-pairs series is\nnow the regression guard against the fix.\n\nAlso fixes the stale module path `src/data/vst/render_params` →\n`src/data/vst/core.py § render_params()`.\n\nRefs #489\nRefs #713\n\n* test(vst): characterize that show_editor warm-up does not change rendered audio\n\nAdds test_show_editor_warmup_does_not_change_rendered_audio: renders the\nhardcoded #489 patch N times each with the show_editor warm-up enabled\nand disabled (by swapping VST3Plugin.show_editor to a no-op around the\nsecond batch), then asserts every cross-path pair is within the same\naudio-similarity thresholds the round-trip tests use.\n\nThis is the empirical justification for the macOS fix in #714 — if the\nwarm-up is not load-bearing for the per-render reload path, it can be\ndropped without changing output, which avoids the AppKit/CGS SIGTRAP\nthat show_editor accumulation triggers in unbundled python on macOS.\n\nRefs #489\nRefs #714\n\n* fix(vst): make load_plugin helper thread daemon + warn on stuck cleanup\n\nIf show_editor hangs past the join timeout, mark the helper thread\ndaemon so it can't block process exit, and log a warning so the\ncondition is visible. Cosmetic comment trim on test_preset_params\nexplaining the post-call parameter readback inversion.\n\nRefs #489\n\n* refactor(vst): use threading.Timer for show_editor close timing\n\nthreading.Timer is the right primitive for 'fire X after N seconds';\nhand-rolling it via Thread + time.sleep was reinventing it. Drops the\n_prepare_plugin helper and _PREPARE_PLUGIN_JOIN_TIMEOUT_SECONDS\nconstant. timer.cancel() + close_editor.set() in the finally block is\ndefensive against show_editor returning early for any reason.\n\nRefs #489 #714",
          "timestamp": "2026-04-30T03:26:17-04:00",
          "tree_id": "c5ca7f23bf1188ab84af12c9f2cd5ca12da53f22",
          "url": "https://github.com/tinaudio/synth-setter/commit/ceaf0fc54f29e875edba3e60a7b575b39d8ec41c"
        },
        "date": 1777534771618,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.86944842338562,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.892144585996866,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.02271847240626812,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.01628929376602173,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.303332805633545,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 12.150979840999996,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.419436931610107,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.867152560021059,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.02958657778799534,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.03589135408401489,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-pair-count",
            "value": 66,
            "unit": "count"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "13bfc624b277ca9f966ac897a290e26324383c3c",
          "message": "internal-feat(vst): add deterministic-render kwargs to make_dataset/generate_sample (#720)\n\n* internal-feat(vst): add deterministic-render kwargs to make_dataset/generate_sample\n\n`generate_sample` accepts optional `fixed_synth_params` / `fixed_note_params`\nthat take precedence over `param_spec.sample()`, and `make_dataset` accepts\n`fixed_synth_params_list` / `fixed_note_params_list` and indexes them per\nsample by `i - start_idx` after validating the lists are long enough. The\nkwargs are internal-only on this PR — they exist so a later act of the #702\nsplit (the `surge_xt_interactive.py` capture/replay flow) can render\ncaller-supplied patches deterministically. No public-facing surface changes.\n\nRefs #702 #719\n\n* internal-fix(vst): skip param_spec.sample() and bound retries when fully fixed\n\nAddress two Copilot review comments on PR #720:\n\n1. (#3166554305) When both fixed_synth_params and fixed_note_params are\n   supplied, skip the param_spec.sample() call entirely. The previous\n   code burned RNG state and paid the call overhead on every retry\n   even though the values were discarded — now param_spec.sample() only\n   runs when at least one half needs sampling.\n\n2. (#3166554339) When BOTH fixed dicts are supplied, render inputs are\n   fully deterministic, so retrying after a loudness fail is provably\n   futile. Raise ValueError with a clear caller-actionable message\n   instead of looping forever. When only one half is fixed, the other\n   is re-sampled each retry and the loop remains meaningful.\n\nPer-item shape validation of fixed_note_params (suggested by #3166554364)\nis intentionally not added — this is an internal-feat:, the caller is\ntrusted to produce well-formed dicts (same trust boundary as\nparam_spec.sample()), and the existing KeyError on\nnote_params['pitch'] is already actionable.\n\nRefs #720 #719 #702",
          "timestamp": "2026-04-30T08:35:59Z",
          "tree_id": "3d244bfe390ad2fd1fb1249bdfd33e8a53330295",
          "url": "https://github.com/tinaudio/synth-setter/commit/13bfc624b277ca9f966ac897a290e26324383c3c"
        },
        "date": 1777538896883,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 4.8057427406311035,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.420990044572391,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.034688860177993774,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.047290027141571045,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.7512216567993164,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 11.80401720758333,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.94356107711792,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.655967754672747,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.042190149426460266,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.05208402872085571,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-pair-count",
            "value": 66,
            "unit": "count"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "450cf0b05b9a6c516e4eea0e240fa2b335bb0bbd",
          "message": "build(deps): migrate lightning to pytorch_lightning (lightning quarantined on PyPI) + add docker deps for skypilot (#721)\n\n* build(deps): migrate lightning to pytorch_lightning\n\n* build(docker): drop ENTRYPOINT, default CMD to /bin/bash, install sky deps\n\nSkyPilot's RunPod backend launches the pod with `dockerArgs: \"bash -c\n'<base64-setup>'\"`, so a baked-in click-CLI ENTRYPOINT collides with the\nlauncher. Drop ENTRYPOINT and default CMD to /bin/bash so `docker run img`\nlands in a shell; callers invoke the click CLI explicitly.\n\nInstall rsync, openssh-client, and python3-pip — SkyPilot needs the SSH\ntoolchain to stage file_mounts and shells out to a system `pip3` that the\nuv-managed venv at /venv/main does not expose.\n\nSkip test_render_params_sets_preset_dependent_param on linux pending\nrefactor to use scripts/run-linux-vst-headless.sh.",
          "timestamp": "2026-04-30T13:07:41-04:00",
          "tree_id": "3d7d0591b758bf38112889d900bedcd4b57e5343",
          "url": "https://github.com/tinaudio/synth-setter/commit/450cf0b05b9a6c516e4eea0e240fa2b335bb0bbd"
        },
        "date": 1777569581574,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.8720703125,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.759646213936503,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.027715224772691727,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.018704593181610107,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.6998417377471924,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 11.830569734250004,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.420691013336182,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 7.029468371905386,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.03556937351822853,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.03645247220993042,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-pair-count",
            "value": 66,
            "unit": "count"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7ae7401f48eade9a3273ddf37519256c91dc6e0a",
          "message": "fix(ci): drop `passthrough` from remaining docker run invocations after #721 (#742)\n\n* fix(ci): drop `passthrough` from remaining docker run invocations after #721 dropped ENTRYPOINT\n\nPR #727 already dropped `passthrough` from `docker-build-validation.yml`\nand `spec-materialization.yml`, but `dataset-generation.yml` and the\n`validate-shard` job in `test-dataset-generation.yml` were missed and\nfail with `exec: \"passthrough\": executable file not found in $PATH`\nagainst the rebuilt `dev-snapshot` image.\n\nImage now has no ENTRYPOINT and `CMD=[\"/bin/bash\"]`, so trailing argv\nis exec'd directly:\n\n- `passthrough bash -c '…'`           → `bash -c '…'`\n- `passthrough rclone copy …`         → `rclone copy …`\n- `passthrough python3 -m …`          → `python3 -m …`\n- `generate_dataset --spec …`         → `python /usr/local/bin/entrypoint.py generate_dataset --spec …`\n  (matches `configs/compute/runpod-template.yaml` from #721)\n\n`flush-investigation.yml` still uses `passthrough` but is slated for\ndeletion, so leave it untouched.\n\nCloses #726\n\n* fix(ci): drop `passthrough` from test-vst-slow.yml after #721 dropped ENTRYPOINT\n\nSame pattern as the rest of #726: `docker run img passthrough bash -c '…'`\nfails with `exec: \"passthrough\": executable file not found in $PATH` against\nthe rebuilt `dev-snapshot` image (no ENTRYPOINT, `CMD=[\"/bin/bash\"]`).\nDrop the `passthrough` prefix so the trailing `bash -c '…'` is exec'd\ndirectly.\n\nRefs #726",
          "timestamp": "2026-05-01T18:55:38-04:00",
          "tree_id": "5d7518cc4f005ca49bd977a3bd47dd3ef2ddadd6",
          "url": "https://github.com/tinaudio/synth-setter/commit/7ae7401f48eade9a3273ddf37519256c91dc6e0a"
        },
        "date": 1777676891685,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.8175413608551025,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.716695620827377,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.024430369958281517,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.020220398902893066,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.612326145172119,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 11.88265226708333,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.472593307495117,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.825548760239035,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.044085100293159485,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.04849100112915039,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-pair-count",
            "value": 66,
            "unit": "count"
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
      },
      {
        "commit": {
          "author": {
            "email": "17952332+ktinubu@users.noreply.github.com",
            "name": "KT",
            "username": "ktinubu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9a33ed197268d916af8d7c3a83b96bc29b319da3",
          "message": "test(data-pipeline): reproduce round-trip reproducibility failure for VST dataset generation (#706)\n\n* test(data-pipeline): add xfail round-trip reproducibility tests for VST dataset generation\n\nTwo new e2e tests in tests/data/vst/test_generate_vst_dataset.py that exercise\nmake_dataset round-trip reproducibility via _patched_sample, plus a third\nrandom-sampling sanity test.\n\nThe two round-trip tests are marked @pytest.mark.xfail(strict=True, reason=\"bug #489\")\nbecause main does not yet carry the per-render plugin-reload workaround on\nfeat/surge-xt-interactive-load-prediction (commits 086d80f / 9ff7f16). Without\nthat workaround, ~50% of every-other render produces junk audio, and audio-metric\nassertions fail. strict=True ensures that an unexpected pass surfaces as a test\nfailure so the bug gets revisited.\n\nRefs #489\n\n* feat(ci-automation): track VST audio-similarity test metrics over time\n\nImplements #703.\n\nTest-side: ``_emit_benchmark_metrics`` writes the five summary metrics to\n``$BENCHMARK_OUTPUT_PATH`` when set (no-op locally).\n``_assert_audio_metrics_within_thresholds`` returns the metrics tuple so\n``_assert_round_trip_matches`` can accumulate per-pair values, and emits the\nworst-case (mss-max, wmfcc-max, sot-max, rms-distance-max, mel-mean-abs)\nunder ``vst-fixed-replay/`` when ``benchmark_name_prefix`` is passed.\n``test_datasets_from_hardcoded_params_are_identical`` opts in.\n\nWorkflow: ``.github/workflows/test-expensive.yml`` sets\n``BENCHMARK_OUTPUT_PATH`` on the pytest step and adds a\n``benchmark-action/github-action-benchmark@v1`` publish step gated to\n``push`` on ``refs/heads/main`` with ``hashFiles('bench.json') != ''``.\n``contents: write`` is granted at the *job* (not workflow) level so only\n``run_slow_tests`` can push to ``gh-pages``.\n\nAlso re-applies ``@pytest.mark.xfail(strict=True, reason=\"bug #489\")`` to\nthe two round-trip tests after the rename, and picks up the all-pairs\nworst-case check from the feature branch — the assertion that makes the\nxfail premise empirically true on main today.\n\nRefs #489\nRefs #703\n\n* ci(test-expensive): allow workflow_dispatch to publish benchmark history\n\nAdds a ``publish_metrics`` boolean input on the manual-dispatch trigger\n(default false) so a maintainer can bootstrap the ``gh-pages`` chart from\na feature branch before main has merged the workflow. Push-to-main still\nalways publishes; the new input is an explicit opt-in escape hatch.\n\nUsage:\n\n    gh workflow run test-expensive.yml \\\n      --ref test/vst-roundtrip-xfail-tests \\\n      -f publish_metrics=true\n\nRefs #703\n\n* ci(test-vst-slow): move VST slow tests + benchmark publish into Docker\n\nBare ``ubuntu-latest`` runners hit \"Timeout waiting for Xvfb to start\" in\n``test-expensive.yml``'s smoke-test step\n(https://github.com/tinaudio/synth-setter/actions/runs/25026506440), so\nthe slow VST tests never reach pytest there. The benchmark publish step\nin ``test-expensive.yml`` was therefore unreachable too.\n\nAdd a separate ``test-vst-slow.yml`` workflow that runs\n``tests/data/vst/test_generate_vst_dataset.py`` inside the\n``tinaudio/synth-setter:dev-snapshot`` Docker image, mirroring the working\ndocker-pull pattern in ``dataset-generation.yml``. ``BENCHMARK_OUTPUT_PATH``\nis set on the container; ``bench.json`` is mounted out via ``-v /tmp/bench``\nand copied to the runner workspace for the\n``benchmark-action/github-action-benchmark@v1`` publish step.\n\nTriggers: push-to-main on relevant paths, plus ``workflow_dispatch`` with\n``image_tag`` and ``publish_metrics`` inputs. The ``publish_metrics``\nopt-in lets a maintainer bootstrap the ``gh-pages`` chart from a feature\nbranch.\n\nReverts the benchmark instrumentation out of ``test-expensive.yml``: the\n``BENCHMARK_OUTPUT_PATH`` env var, the publish step, the dispatch input,\nand the job-level ``contents: write`` grant. ``test-expensive.yml`` goes\nback to its pre-#703 shape — its non-VST slow tests can remain there.\n\nRefs #489\nRefs #703\n\n* ci(test-vst-slow): TEMPORARY bootstrap push-trigger from PR branch\n\nAdds ``test/vst-roundtrip-xfail-tests`` to the push-trigger branch list\nand widens the publish step's ``if:`` to accept that ref. Lets us\nbootstrap the gh-pages benchmark chart from this PR branch before main\nhas the workflow.\n\nREVERT-ME: Roll back to ``branches: [main]`` and the main-only ``if:``\ngate once the chart exists. See follow-up revert commit.\n\nRefs #703\n\n* fix(test-vst): drop xfail from sampled-params test (not a #489 reproducer)\n\n``test_datasets_from_sampled_params_are_identical`` does NOT reproduce\n#489. Its rows use *different* random params per row (Stage 1 picks 5\nrandom samples), so it has no all-pairs cross-comparison — only per-row\n``expected[i]`` vs ``actual[i]`` checks. Per-row checks alone don't\nexpose every-other-render junk because they only ever compare a row to\nitself across stages, not row-vs-row within a stage.\n\nCI confirmed this on c69f985: the hardcoded test correctly XFAIL'd\n(all-pairs check caught the bug), the smoke test passed, but the\nsampled test XPASS'd against the strict marker.\n\nThe hardcoded test is the canonical #489 reproducer; the sampled test\nis a regression net for the round-trip API and should pass as-is.\n\nRefs #489\n\n* fix(test-vst): skip-fetch-gh-pages on first bootstrap\n\nThe benchmark action defaults to ``skip-fetch-gh-pages: false`` and runs\n``git fetch ... gh-pages:gh-pages`` before any other step. On a first\nbootstrap where the ``gh-pages`` branch doesn't exist yet, that fetch\nfails with \"couldn't find remote ref gh-pages\" instead of letting the\naction create the branch.\n\nRun 25138635107 (commit e0e191d) hit this — tests passed, publish step\ncrashed at the fetch.\n\nSetting ``skip-fetch-gh-pages: true`` lets the action take its\nlocal-only path: it generates ``data.js`` + ``index.html`` from\n``bench.json``, commits them on a fresh ``gh-pages`` worktree, and\n``auto-push`` creates the remote branch.\n\nRefs #703\n\n* ci: re-trigger after gh-pages bootstrap\n\n* ci: re-trigger after gh-pages bootstrap\n\n* ci(test-vst): drop in-container symlink + add VST smoke + dummy fast-path\n\nThree changes to ``.github/workflows/test-vst-slow.yml``:\n\n1. Drop the ``mkdir -p plugins; ln -sf`` lines from the docker run. The\n   base image already places the VST3 at ``/usr/lib/vst3/Surge XT.vst3``,\n   and the bind mount over ``/home/build/synth-setter`` hides the\n   image-side symlink that the Dockerfile creates. Set\n   ``SYNTH_SETTER_PLUGIN_PATH=/usr/lib/vst3/Surge XT.vst3`` so the test\n   uses the absolute path the .deb installs to.\n\n2. Add a ``Smoke-test Surge XT plugin load`` step before the test step,\n   mirroring the local-runner smoke check in ``test-expensive.yml``.\n   Fails fast if the plugin / image / mount layout is broken before\n   committing to the much-longer pytest run.\n\n3. Add a ``dummy_only`` workflow_dispatch input + a\n   ``Write hardcoded dummy bench.json`` step gated on it. When set, the\n   pull / smoke / test / surface steps are skipped and a hand-crafted\n   ``bench.json`` is written directly to the workspace. Lets a maintainer\n   iterate on the publish-step gating in ~10 seconds instead of ~5\n   minutes per cycle. Implies ``publish_metrics``.\n\nAlso revert the ``skip-fetch-gh-pages: true`` flag now that the\n``gh-pages`` branch exists on the remote — the action's default fetch\npath now resolves it cleanly.\n\nRefs #703\n\n* ci(test-vst): rename benchmark bucket + use full metric names\n\nBucket: ``VST fixed-params replay`` → ``VST noise floor``. Reflects what\nthe test actually measures — the floor of how well two render passes of\nidentical params reproduce each other under the docker mitigation stack\n— rather than the now-misnamed historical reference to the\n``fixed_*_params_list`` API the test no longer uses.\n\nMetric series: drop project-internal abbreviations in favor of full\nnames so the chart's left-hand legend is self-explanatory.\n\n  mss-max          → multi-scale-spectral-loss-max\n  wmfcc-max        → dtw-aligned-mfcc-distance-max\n  sot-max          → spectral-optimal-transport-max  (unit: W → Wasserstein)\n  rms-distance-max → rms-envelope-cosine-distance-max\n  mel-mean-abs     → mel-spectrogram-mean-absolute-error\n\nAlso rename the ``benchmark_name_prefix`` argument from\n``vst-fixed-replay`` to ``vst-noise-floor`` so the on-chart series\nstrings are consistent with the bucket.\n\nThe single existing bootstrap data point on ``gh-pages`` will be\norphaned under the old bucket name — left for now since deleting it\nwould mean a force-push to ``gh-pages`` and the noise-floor chart only\nbecomes meaningful once a few runs land anyway.\n\nRefs #703\n\n* feat(ci-automation): split benchmark dashboards + timing metrics + docs\n\nSplits the single benchmark dashboard into two\n(``test_datasets_from_hardcoded_params_are_identical`` →\n``VST noise floor (1 preset N renders)``,\n``test_datasets_from_sampled_params_are_identical`` →\n``VST noise floor (random preset replay)``), since the action keys all\nentries from one bench JSON under one chart bucket so multi-dashboard\nneeds separate files. ``_emit_benchmark_metrics`` now takes a\n``bench_filename`` arg and reads ``BENCHMARK_OUTPUT_DIR``; each test\npasses its prefix as the filename; the workflow's Surface step copies\nboth files; Publish is duplicated, one per bucket.\n\nAdds two new metrics per bucket:\n\n  num-samples                   sentinel for fixture-size regressions\n  wall-clock-seconds-per-render renderer perf drift\n\nEach test brackets its ``make_dataset`` calls with\n``time.perf_counter()`` and passes the elapsed total as\n``total_render_seconds``.\n\nNew doc ``docs/reference/audio-similarity-benchmarks.md`` covers\npurpose, where to find the live charts + raw data, the two dashboard\nsemantics, the seven metric series, threshold/alerting, workflow\nwiring, and operations (bootstrapping, pre-merge publishing, adding\nnew dashboards, pruning history).\n\nRefs #489\nRefs #703\n\n* fix(test-vst): address PR #706 review feedback\n\n- Reword `render_params` reload references to present-tense bug-#489\n  descriptions; drop forward-references to the unmerged per-render\n  reload workaround (commits 086d80f / 9ff7f16, PR #702).\n- Sync hardcoded-params docstring `num_samples` and test-name\n  references to the actual `test_datasets_from_hardcoded_params_are_identical`\n  body (num_samples=6, all-pairs check rationale).\n- Sync sampled-params docstring rationale to match issue #489\n  framing (drop the workaround commit citations).\n- Cache `mel[...]` and `params[...]` reads in `_assert_h5_structure_is_valid`\n  to avoid double materialization.\n- Handle JSONDecodeError in `_emit_benchmark_metrics` by treating a\n  truncated bench file as an empty list.\n- Pin `benchmark-action/github-action-benchmark@v1` -> the v1.22.0\n  commit SHA in `test-vst-slow.yml` for supply-chain hygiene.\n- Update `docs/reference/audio-similarity-benchmarks.md` to drop the\n  forward-reference to the unmerged per-render reload workaround.\n\nRefs #489\n\n* fix(test-vst): address PR #706 review feedback (round 2)\n\nDoc/wording fixes only — no behavior change:\n\n- _assert_round_trip_matches docstring: ``BENCHMARK_OUTPUT_PATH`` →\n  ``BENCHMARK_OUTPUT_DIR`` (matches the actual env var read by\n  _emit_benchmark_metrics and set by test-vst-slow.yml). Comment\n  3164945781.\n- docs/reference/audio-similarity-benchmarks.md: \"six series\" → \"seven\n  series\" with explicit call-out of the two non-distance sentinels\n  (num-samples, wall-clock-seconds-per-render); the metric table\n  already listed seven rows. Comment 3164945796.\n- test-vst-slow.yml dummy_only fast-path: include num-samples and\n  wall-clock-seconds-per-render in the hardcoded bench JSON so the\n  debug-only payload mirrors what _assert_round_trip_matches actually\n  emits. Comment 3164945820.\n\nComment 3164945810 (temp branch in push.branches) is a duplicate of\nthe round-1 thread already justified at 3164936475 / 3164936515 — kept\nintentionally and gated by an in-file removal note; will be reverted\nin a follow-up before merge once the gh-pages chart is bootstrapped.\n\nxfail decorators, _HARDCODED_*_PARAMS, and gh-pages branch are not\ntouched.\n\nRefs #489\nRefs #703\n\n* chore(test-vst): remove dummy fast-path debug code from workflow\n\nThe ``dummy_only`` workflow_dispatch input + ``Write hardcoded dummy\nbench JSON files (debug-only fast path)`` step + all\n``inputs.dummy_only`` references were scaffolding for iterating on the\npublish-step gating during the gh-pages bootstrap. The chart is live\nand the publish path is verified, so the dummy code is no longer\nload-bearing — it just adds noise to the workflow and gives operators\na footgun (publishing junk to gh-pages by accident).\n\nReverts:\n- ``dummy_only`` dispatch input\n- \"Write hardcoded dummy bench JSON files\" step\n- ``if: inputs.dummy_only != true`` gates on Pull image, Smoke-test,\n  Run VST tests, Surface\n- ``inputs.dummy_only == true`` clauses in both publish steps' ``if:``\n\nRefs #703\n\n* refactor(test-vst): factor benchmark emission out of round-trip helper\n\nPer PR review feedback (r3165027905): the published \"1 preset N renders\"\nchart was wired to per-pair metrics, but the #489 reproducer is the\nall-pairs worst-case across the union of renders. The chart could look\nflat while the test xfails on the all-pairs assertion.\n\nRefactor:\n- New ``RoundTripMetrics`` and ``AllPairsMetrics`` frozen dataclasses\n  hold the four audio metrics + their respective extras (mel diff +\n  num_samples for round-trip; pair count for all-pairs).\n- ``_assert_round_trip_matches`` returns ``RoundTripMetrics`` and no\n  longer has any benchmark-emit logic. Drops ``benchmark_name_prefix``\n  and ``total_render_seconds`` params.\n- ``_assert_all_pairs_audio_metrics_within_thresholds`` returns\n  ``AllPairsMetrics``.\n- New ``_emit_audio_similarity_benchmark_metrics(prefix, round_trip,\n  all_pairs, total_render_seconds)`` consumes either or both structs\n  and writes the bench JSON. Round-trip series go under ``<prefix>/``;\n  all-pairs series go under ``<prefix>/all-pairs-`` so both can coexist\n  on the same chart bucket without name collisions.\n- Hardcoded test now emits BOTH structs — round-trip for context,\n  all-pairs as the primary regression signal for #489.\n- Sampled test still emits only round-trip (cross-row pairs differ\n  legitimately, no all-pairs check applies).\n\nAdds six unit tests for ``_emit_audio_similarity_benchmark_metrics``\ncovering: env-unset no-op, round-trip-only schema, all-pairs-only\nschema, both-structs namespace separation, no-args no-write, and\nappend-on-second-call. All run in <1s without the VST.\n\nUpdates ``docs/reference/audio-similarity-benchmarks.md`` to document\nthe new ``all-pairs-*`` series + their role as the primary #489 signal\non the hardcoded bucket.\n\nRefs #489\nRefs #703\n\n* docs(test-vst): make hardcoded-test docstring self-contained\n\nDrops the 'Variant of test_datasets_from_sampled_params_are_identical'\nframing and rewrites as a standalone description of what the test\nactually does.\n\nRefs #703",
          "timestamp": "2026-04-29T21:19:24-04:00",
          "tree_id": "dce6896d60b602ae0db496f878ed96bc67631640",
          "url": "https://github.com/tinaudio/synth-setter/commit/9a33ed197268d916af8d7c3a83b96bc29b319da3"
        },
        "date": 1777512396155,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.312377691268921,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 3.0197676008939744,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.01443742960691452,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.014216125011444092,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.7976313829421997,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 4.092413352800003,
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c34930a8e51cc0f68b5beb9e2b5f904437f35770",
          "message": "workaround(vst): skip show_editor warmup on Darwin to avoid AppKit SIGTRAP (#715)\n\n* fix(vst): skip show_editor warmup on Darwin to avoid AppKit SIGTRAP\n\nOn macOS, pedalboard.VST3Plugin.show_editor accumulates AppKit/CGS\ncommit-handler state per call in the unbundled python process. After\n~3-4 calls the next show_editor invocation aborts with SIGTRAP, which\nbreaks any flow that reloads the plugin per render (#714).\n\nSkip the warmup on Darwin only. On Linux/Windows show_editor remains the\nestablished workaround for spotify/pedalboard#394 (preset state not\ncommitted until the editor opens).\n\nEmpirical justification (full audit logged on #714):\n- Cross-path equivalence test on the per-render-reload PR (#713) showed\n  0 audio-sample differences between renders that called show_editor and\n  renders that did not.\n- Preset-coverage audit across all 3 Surge XT presets and 770+\n  parameters per preset found 0 readback divergences between\n  (load_preset -> flush) and (show_editor -> load_preset -> flush).\n- The post-load process([], ...) flush already in render_params is\n  sufficient to commit Surge XT's preset state without show_editor.\n\nA new requires_vst test (tests/data/vst/test_preset_coverage.py) guards\nthis decision: it parametrizes over every .vstpreset and asserts the two\npatterns produce identical parameter readbacks. If a future preset,\npedalboard release, or Surge XT version ever diverges, the test fails\nloudly so the Darwin path doesn't silently fall back to Surge defaults.\n\nCloses #714\nRefs #489 #713 spotify/pedalboard#394\n\n* docs(eval): note macOS Darwin gate in eval-pipeline open questions; tighten load_plugin docstring\n\ndoc-drift findings on PR #715:\n- eval-pipeline.md Open Question #2 (macOS Apple Silicon support) was 'Needs\n  testing'; the plugin demonstrably loads, with the show_editor warmup skipped\n  on Darwin per #714. Status updated to reflect the partial answer + gating.\n- load_plugin docstring claimed the warmup populates the parameter dict; the\n  preset-coverage audit added in this PR proves the dict is identical with vs\n  without show_editor. Replaced with a pointer to the comment block, which\n  already explains the real rationale (pedalboard #394 ordering workaround +\n  Darwin SIGTRAP).\n\nRefs #714\n\n* test(vst): mark preset coverage test as slow\n\nEach parameter case constructs two VST3Plugin instances (one with the\nshow_editor warmup, one without) and reads ~770 parameters off each.\nTotal wall-clock per case is several seconds even on Linux + Xvfb, well\npast the make test budget. Aligns with the other VST-gated tests in\nthis directory.\n\nRefs #714\n\n* test(vst): skip preset coverage test on Darwin to avoid show_editor SIGTRAP\n\nThe test calls plugin.show_editor() which is the exact AppKit SIGTRAP trigger\nthis PR fixes. On a Darwin host with the Surge XT plugin installed, running\nthis test would crash pytest. Mirrors the sys.platform != \"darwin\" gate in\nsrc/data/vst/core.py:50.\n\nRefs #714",
          "timestamp": "2026-04-30T02:57:54-04:00",
          "tree_id": "75b2b583de9d382014c7eb04531e4a3472c6a4c1",
          "url": "https://github.com/tinaudio/synth-setter/commit/c34930a8e51cc0f68b5beb9e2b5f904437f35770"
        },
        "date": 1777532645197,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.5092296600341797,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.6222000133991243,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.021497655659914017,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.0040929317474365234,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.6025935411453247,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 4.418843707200006,
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ceaf0fc54f29e875edba3e60a7b575b39d8ec41c",
          "message": "fix(vst): reload plugin per render to eliminate every-other junk audio (#713)\n\n* fix(vst): reload plugin per render to eliminate every-other junk audio\n\nrender_params now takes a plugin_path and reloads the VST3 plugin on every\ncall, working around a stale-state bug where alternating renders produced\nsilent or repeated audio. load_plugin's editor-pump uses a threading.Event\n+ show_editor(stop_event) pattern (replacing the prior _thread.interrupt\nKeyboardInterrupt hack), which is what makes a per-call reload safe and\nfast enough to be the default.\n\ngenerate_sample, make_dataset, and scripts/predict_vst_audio.py are\nupdated to pass plugin_path through to render_params instead of\npre-loading the plugin.\n\nThe xfail decorator on\ntest_datasets_from_hardcoded_params_are_identical is removed: with this\nfix in place, the test no longer xpasses.\n\nCloses #489\nRefs #705\nRefs #702\n\n* docs(eval): update audio-similarity-benchmarks for #489 closure\n\nThe dashboard's framing described #489 as an open bug and called the\nall-pairs series its \"regression signal\". With #713 closing #489 via\nper-render plugin reload, the framing inverts: the all-pairs series is\nnow the regression guard against the fix.\n\nAlso fixes the stale module path `src/data/vst/render_params` →\n`src/data/vst/core.py § render_params()`.\n\nRefs #489\nRefs #713\n\n* test(vst): characterize that show_editor warm-up does not change rendered audio\n\nAdds test_show_editor_warmup_does_not_change_rendered_audio: renders the\nhardcoded #489 patch N times each with the show_editor warm-up enabled\nand disabled (by swapping VST3Plugin.show_editor to a no-op around the\nsecond batch), then asserts every cross-path pair is within the same\naudio-similarity thresholds the round-trip tests use.\n\nThis is the empirical justification for the macOS fix in #714 — if the\nwarm-up is not load-bearing for the per-render reload path, it can be\ndropped without changing output, which avoids the AppKit/CGS SIGTRAP\nthat show_editor accumulation triggers in unbundled python on macOS.\n\nRefs #489\nRefs #714\n\n* fix(vst): make load_plugin helper thread daemon + warn on stuck cleanup\n\nIf show_editor hangs past the join timeout, mark the helper thread\ndaemon so it can't block process exit, and log a warning so the\ncondition is visible. Cosmetic comment trim on test_preset_params\nexplaining the post-call parameter readback inversion.\n\nRefs #489\n\n* refactor(vst): use threading.Timer for show_editor close timing\n\nthreading.Timer is the right primitive for 'fire X after N seconds';\nhand-rolling it via Thread + time.sleep was reinventing it. Drops the\n_prepare_plugin helper and _PREPARE_PLUGIN_JOIN_TIMEOUT_SECONDS\nconstant. timer.cancel() + close_editor.set() in the finally block is\ndefensive against show_editor returning early for any reason.\n\nRefs #489 #714",
          "timestamp": "2026-04-30T03:26:17-04:00",
          "tree_id": "c5ca7f23bf1188ab84af12c9f2cd5ca12da53f22",
          "url": "https://github.com/tinaudio/synth-setter/commit/ceaf0fc54f29e875edba3e60a7b575b39d8ec41c"
        },
        "date": 1777534774673,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.2751128673553467,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.7308008645474913,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.008529347367584705,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.07157295942306519,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.869699478149414,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 14.891129689399985,
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "13bfc624b277ca9f966ac897a290e26324383c3c",
          "message": "internal-feat(vst): add deterministic-render kwargs to make_dataset/generate_sample (#720)\n\n* internal-feat(vst): add deterministic-render kwargs to make_dataset/generate_sample\n\n`generate_sample` accepts optional `fixed_synth_params` / `fixed_note_params`\nthat take precedence over `param_spec.sample()`, and `make_dataset` accepts\n`fixed_synth_params_list` / `fixed_note_params_list` and indexes them per\nsample by `i - start_idx` after validating the lists are long enough. The\nkwargs are internal-only on this PR — they exist so a later act of the #702\nsplit (the `surge_xt_interactive.py` capture/replay flow) can render\ncaller-supplied patches deterministically. No public-facing surface changes.\n\nRefs #702 #719\n\n* internal-fix(vst): skip param_spec.sample() and bound retries when fully fixed\n\nAddress two Copilot review comments on PR #720:\n\n1. (#3166554305) When both fixed_synth_params and fixed_note_params are\n   supplied, skip the param_spec.sample() call entirely. The previous\n   code burned RNG state and paid the call overhead on every retry\n   even though the values were discarded — now param_spec.sample() only\n   runs when at least one half needs sampling.\n\n2. (#3166554339) When BOTH fixed dicts are supplied, render inputs are\n   fully deterministic, so retrying after a loudness fail is provably\n   futile. Raise ValueError with a clear caller-actionable message\n   instead of looping forever. When only one half is fixed, the other\n   is re-sampled each retry and the loop remains meaningful.\n\nPer-item shape validation of fixed_note_params (suggested by #3166554364)\nis intentionally not added — this is an internal-feat:, the caller is\ntrusted to produce well-formed dicts (same trust boundary as\nparam_spec.sample()), and the existing KeyError on\nnote_params['pitch'] is already actionable.\n\nRefs #720 #719 #702",
          "timestamp": "2026-04-30T08:35:59Z",
          "tree_id": "3d244bfe390ad2fd1fb1249bdfd33e8a53330295",
          "url": "https://github.com/tinaudio/synth-setter/commit/13bfc624b277ca9f966ac897a290e26324383c3c"
        },
        "date": 1777538899381,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.0389244556427,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.5321727210655807,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.016979897394776344,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.07794207334518433,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.428880214691162,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 10.4250717613,
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "450cf0b05b9a6c516e4eea0e240fa2b335bb0bbd",
          "message": "build(deps): migrate lightning to pytorch_lightning (lightning quarantined on PyPI) + add docker deps for skypilot (#721)\n\n* build(deps): migrate lightning to pytorch_lightning\n\n* build(docker): drop ENTRYPOINT, default CMD to /bin/bash, install sky deps\n\nSkyPilot's RunPod backend launches the pod with `dockerArgs: \"bash -c\n'<base64-setup>'\"`, so a baked-in click-CLI ENTRYPOINT collides with the\nlauncher. Drop ENTRYPOINT and default CMD to /bin/bash so `docker run img`\nlands in a shell; callers invoke the click CLI explicitly.\n\nInstall rsync, openssh-client, and python3-pip — SkyPilot needs the SSH\ntoolchain to stage file_mounts and shells out to a system `pip3` that the\nuv-managed venv at /venv/main does not expose.\n\nSkip test_render_params_sets_preset_dependent_param on linux pending\nrefactor to use scripts/run-linux-vst-headless.sh.",
          "timestamp": "2026-04-30T13:07:41-04:00",
          "tree_id": "3d7d0591b758bf38112889d900bedcd4b57e5343",
          "url": "https://github.com/tinaudio/synth-setter/commit/450cf0b05b9a6c516e4eea0e240fa2b335bb0bbd"
        },
        "date": 1777569584049,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 1.5364614725112915,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.194485236611217,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.006318153813481331,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.037003517150878906,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.2827268838882446,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 10.508583992799998,
            "unit": "seconds"
          }
        ]
      }
    ]
  }
}