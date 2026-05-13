window.BENCHMARK_DATA = {
  "lastUpdate": 1778714283324,
  "repoUrl": "https://github.com/tinaudio/synth-setter",
  "entries": {
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
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 21.051050186157227,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 17.5240702496469,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.30374279618263245,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.15888166427612305,
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
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 21.084692001342773,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 17.81026928395033,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.31998199224472046,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.18377995491027832,
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
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 21.14731788635254,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 17.35396855942905,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.3002326488494873,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.12770217657089233,
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
          "id": "86a46d2f71c151ec8445e1b84dc2c3e4cf0af0c4",
          "message": "internal-feat(pipeline): renderer-version contract end-to-end + rclone-native upload bounds (#740)\n\n* internal-feat(pipeline): pin renderer_version to SURGE_XT_RENDERER_VERSION; expose extract_renderer_version\n\n`materialize_spec` previously extracted `renderer_version` from the VST3\nplugin bundle at materialization time, which required loading the plugin\nvia `pedalboard.VST3Plugin` when neither `Contents/moduleinfo.json` nor\n`Contents/Info.plist` was present — and that codepath needs an X display.\nThat blocks any caller that wants to materialize a spec without an X\nstack (e.g. the SkyPilot launcher, which runs on a GHA runner / dev\nlaptop and never loads the plugin itself).\n\nPin `renderer_version` to a single source of truth, the\n`SURGE_XT_RENDERER_VERSION = \"1.3.4\"` constant in this module, kept in\nlockstep with the dev-snapshot image's `SURGE_GIT_REF`. `materialize_spec`\nnow sets the pin directly and doesn't touch the plugin bundle.\n\nKeep `extract_renderer_version` as a public function — same static-metadata\n+ pedalboard-fallback shape — so the worker side can call it against the\nactual plugin and verify the pin matches reality before rendering. The\nworker-side cross-check is the next commit; the rclone-native upload\nbounds are the one after.\n\nRefs #534\n\n* internal-feat(pipeline): worker-side renderer_version cross-check in generate_dataset.run\n\nThe launcher pins `renderer_version` to `SURGE_XT_RENDERER_VERSION` blindly\n(its code path stays interpreter-only). The worker is where pedalboard is\navailable, so the worker is where the pin gets verified against reality.\n\n`run()` now calls `extract_renderer_version` against `spec.plugin_path`\nbefore any rclone or subprocess work and raises `RuntimeError` if the\nrunning plugin disagrees with the spec. The error message points at the\ntwo valid fixes (rebuild the image against the matching `SURGE_GIT_REF`\nor bump the constant), so failures are actionable rather than mysterious.\nOn match, a single `renderer_version OK: …` info log records the\nconfirmed pairing for forensics.\n\nTest fixture: tests/pipeline/fixtures/TestPlugin.vst3 (already on `main`)\nhas `Contents/moduleinfo.json` reporting Version=\"1.0.0-test\". Updated\n`_base_spec_kwargs` to use that fixture + that version so the spec/plugin\npair matches by default; new test asserts mismatch raises before any\nupload happens.\n\nRefs #534\n\n* internal-fix(pipeline): rclone-native upload bounds + 'rclone returned cleanly' sentinel\n\nTwo related observability fixes for the worker upload path:\n\n1. `_rclone_copy` was running `rclone copy --checksum src dst` with no\n   timeouts and no retries — a stuck TCP connect or a slow PUT could hold\n   the worker indefinitely. Switch to rclone's own bounds:\n     --contimeout=30s    bound TCP connect phase\n     --timeout=300s      bound any single HTTP request\n     --retries=3         retry the whole copy on transient failure\n     -vv                 emit per-request debug log so a failure leaves\n                         actionable evidence in the worker stdout\n   Letting rclone enforce these (vs. wrapping `subprocess.run(..., timeout=N)`\n   in Python) preserves the postcondition that a non-zero exit means the\n   upload genuinely failed, instead of \"we waited N seconds and gave up\".\n\n2. After `subprocess.check_call` returns from a successful rclone, log a\n   single `rclone returned cleanly: <src> -> <dst>` sentinel. Distinct\n   string so CI logs can be grepped to tell at a glance whether the rclone\n   subprocess actually exited vs. hanging post-upload (the bug-#2 hang\n   shape from #735, now believed gone but worth keeping the canary).\n\nAdds matching boundary logs around the upload path (`spec written:`,\n`spec uploaded ->`, `rendering shard …`, `shard rendered: … (N bytes)`,\n`shard uploaded: …`) so a `tail_logs(follow=False)` dump pinpoints which\nstep a hung run got to.\n\nRefs #534\nRefs #735\n\n* refactor: move extract_renderer_version to src.data.vst.core\n\nThe extractor reads VST3 plugin bundle metadata — that's a VST utility,\nnot a spec-schema concern. Move it next to the other VST helpers\n(`load_plugin`, `load_preset`, `render_params`) in `src/data/vst/core.py`\nand update the worker-side caller in `pipeline.entrypoints.generate_dataset`\nto import from the new location.\n\n`SURGE_XT_RENDERER_VERSION` stays in `pipeline.schemas.spec` because it\nis a spec-construction constant (consumed by `materialize_spec`); only\nthe extractor moves. Tests follow the source: `TestExtractRendererVersion`\nmoves from `tests/pipeline/test_schemas/test_spec.py` to a new\n`tests/data/vst/test_core.py` (matching the existing\n`tests/data/vst/{test_generate_vst_dataset,test_preset_*}.py` layout).\n\nNo behavior change. The function signature and error contract are\nidentical; tests are byte-for-byte the same as their previous\nlocation, just imported from the new path.\n\nRefs #534\n\n---------\n\nCo-authored-by: Your Name <you@example.com>",
          "timestamp": "2026-05-01T18:58:15-04:00",
          "tree_id": "1b380633afcd57b07636dddead1f765f334739f0",
          "url": "https://github.com/tinaudio/synth-setter/commit/86a46d2f71c151ec8445e1b84dc2c3e4cf0af0c4"
        },
        "date": 1777677711175,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 4.680042743682861,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.609619530038908,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.03496094048023224,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.04797077178955078,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.2701773643493652,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 12.03761984900001,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.680042743682861,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.771234348765574,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.03749045729637146,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.052183568477630615,
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
          "id": "adfd7ab4e1fdba639f812db874e1086dccddc471",
          "message": "internal-feat(skypilot): matrix-driven RunPod + OCI generate-dataset CI (#777)\n\n* feat(skypilot): add OCI x86 CPU as a second SkyPilot smoke target\n\nMirrors the existing RunPod path with a CPU-only Flex template\n(VM.Standard.E5.Flex), provider-neutral launcher (no code changes), a\nparallel `generate-oci` CI job (continue-on-error: true while bedding\nin), and a brief operator setup guide. The launcher's R2-uploaded spec\ncontract and the #735 os._exit(0) workaround are preserved across\nproviders.\n\nRegion lives only in ~/.oci/config so `sky check oci` and the launch\nagree on a single source of truth. ~/.oci paths are derived from $HOME\ninside the container so the cred-write step is portable across base\nimages.\n\nThe dev-snapshot Docker image must be rebuilt+pushed with skypilot[oci]\nin requirements-app.txt before the OCI CI job can pass.\n\nRefs #534\n\n* internal-feat(skypilot): add OCI debug noop template + temporarily switch debug workflow to OCI noop for iteration\n\nAdds configs/compute/oci-debug-template.yaml as the OCI/CPU sibling of\nrunpod-debug-template.yaml, updates the runner-side skypilot install in\ntest-skypilot-debug.yml to carry the [oci] extra, and TEMPORARILY:\n\n  - Comments out all RunPod debug matrix variants except 'noop'.\n  - Points 'noop' at configs/compute/oci-debug-template.yaml.\n  - Swaps the inline-sky cred-write step from ~/.runpod/config.toml to\n    ~/.oci/config + ~/.sky/config.yaml + 'sky check oci' fail-fast gate.\n\nThe temporary changes (matrix gating + cred-write swap) are iteration\nscaffolding for landing the OCI target. Re-enable variants progressively\nas OCI plumbing stabilises; back the gating out before marking PR #769\nready for review.\n\nRefs #768\n\n* fix(skypilot): rewrite OCI templates to docker-in-run; SkyPilot OCI rejects image_id\n\nOCI's SkyPilot backend rejects 'image_id: docker:<image>' with\n'Docker image is currently not supported on OCI'. Rewrite both OCI\ntemplates to provision a stock OCI Ubuntu VM (image_tag_general:\nskypilot:cpu-ubuntu-2204) and run the worker container ourselves\ninside the run: block:\n\n  - oci-debug-template.yaml: drop image_id entirely (noop probe just\n    echoes; no docker needed).\n  - oci-cpu-template.yaml: setup: installs docker.io, starts daemon,\n    pre-pulls worker image; run: invokes 'sudo docker run' with the\n    same env-injection contract the launcher uses on RunPod. Worker\n    image moved from a SkyPilot image_id to a WORKER_IMAGE env var\n    that the workflow sed-pins.\n  - test-dataset-generation.yml + test-skypilot-debug.yml: write\n    image_tag_general into ~/.sky/config.yaml; sed-pin updated to\n    rewrite WORKER_IMAGE (not image_id); test-dataset-generation also\n    runtime-installs skypilot[oci] inside dev-snapshot if 'oci' SDK is\n    missing (bridge until the post-rebuild dev-snapshot lands).\n\nRefs #768\n\n* fix(skypilot): pin OCI noop debug to VM.Standard.E2.1.Micro (Always Free)\n\nus-ashburn-1 returned ResourcesUnavailableError across all 3 ADs for\nSkyPilot's auto-picked VM.Standard.E4.Flex (cpus=2, mem=8). Likely\nzero E4.Flex compute quota in the operator's tenancy.\n\nPin the noop probe to VM.Standard.E2.1.Micro instead — it's OCI's\n'Always Free' shape (1 OCPU, 1 GB AMD64), available to every tenancy\nwithout a quota request. This lets us validate the OCI launcher\nplumbing (cred-write, sky check, provision, teardown) independently\nof whether the operator has paid compute quota for E4.Flex.\n\nProduction template (oci-cpu-template.yaml) still asks for cpus: 4+,\nmemory: 16+ (needed for the VST/numpy worker); a green test-dataset-\ngeneration OCI run depends on the operator having actual compute\nquota.\n\nRefs #768\n\n* diag(skypilot): list OCI region subscriptions + E-Flex compute limits in noop probe\n\nAdds a one-shot diagnostic to test-skypilot-debug.yml's inline-sky step\nto print the operator's tenancy region subscriptions and service limit\nvalues for any VM.Standard.E*.Flex compute in each region. Output guides\nwhich region the OCI templates should target (right now provisioning\nfails with ResourcesUnavailableError in us-ashburn-1, suggesting zero\nquota there for E4.Flex).\n\nRefs #768\n\n* diag(skypilot): expand OCI diagnostic to dump ALL compute limits + per-shape resource availability\n\nPrevious filter ('standard-e' AND 'flex' in limit name) returned empty\nacross the operator's home region — but the actual OCI limit names may\nnot match that regex. Print every compute limit verbatim, list ADs in\nthe region, and call get_resource_availability for E4.Flex / E5.Flex /\nA1.Flex / E2.1.Micro to surface used/available counts. This pinpoints\nwhether the tenancy has zero paid quota (so OCI is a non-starter for\nthe prod template) or just regional capacity issues.\n\nRefs #768\n\n* fix(skypilot): sudo -E to preserve env vars into nested docker run on OCI\n\nWorker container started successfully on OCI but failed at:\n  KeyError: 'WORKER_SPEC_URI'\ninside the inlined python -c. Root cause: bare 'sudo' strips the\ncaller's environment, so 'docker run -e WORKER_SPEC_URI' (no value;\ninherit from parent shell) reaches docker with WORKER_SPEC_URI unset.\nPass -E to sudo to preserve all caller env vars (RCLONE_CONFIG_R2_*,\nWORKER_SPEC_URI, WORKER_IMAGE) into the docker invocation.\n\nRefs #768\n\n* fix(skypilot): propagate SYNTH_SETTER_WORKER_RANK/NUM_WORKERS into OCI worker container\n\nThe launcher injects partition env vars per rank via task.update_envs().\nOn RunPod they reach the worker process directly because SkyPilot owns\nthe docker container. On OCI we run docker ourselves inside the run:\nblock, so we have to forward each env var explicitly via 'docker run\n-e'. Add the two partition vars to both the placeholder envs: block\n(so SkyPilot doesn't reject the task) and the docker -e list (so the\ninner python process inherits them).\n\nRefs #768\n\n* fix(skypilot): give OCI launcher a run-id-scoped cluster name to avoid R2 collision with RunPod\n\nBoth 'generate' (RunPod) and 'generate-oci' jobs in the same workflow run\ninvoke skypilot_launch_smoke concurrently. With the default cluster name\n('synth-setter-smoke-{config_id[:8]}' = 'synth-setter-smoke-runpod-s'),\nboth jobs upload their materialized spec to the SAME R2 key:\n'r2:.../skypilot-launcher-specs/synth-setter-smoke-runpod-s.json'.\nWhichever uploads last wins; both clouds' workers then download that\nspec and write shards under its r2_prefix. validate-shard reads RunPod's\nlocal /tmp/input_spec.json (the loser's run_id), gets the wrong prefix,\nand fails to find shards in R2.\n\nFix: pass --cluster-name explicitly for the OCI step, scoped to the\ngithub.run_id so it's distinct from RunPod's default and unique across\nPR pushes. RunPod keeps the default for backwards compat with existing\ndebug/dispatch tooling.\n\nRefs #768\n\n* fix(skypilot): wait for cloud-init + apt lock before installing docker on OCI VM\n\nSetup failed with:\n  E: Could not get lock /var/lib/apt/lists/lock. It is held by process 3178 (apt)\non a freshly-provisioned OCI Ubuntu VM. SkyPilot launches concurrently\nwith cloud-init's own apt activity. Wait for cloud-init to finish, then\npoll the apt+dpkg locks (up to 5 min) before our 'apt-get update' fires.\n\nRefs #768\n\n* fix(skypilot): give OCI worker docker container full privileges + raised nofile\n\nWorker exited on:\n  X Error of failed request:  BadWindow (invalid Window parameter)\n  Major opcode of failed request:  20 (X_GetProperty)\nduring pedalboard's Surge XT preset load on OCI. Preceded by:\n  dbus-daemon: Failed to set fd limit to 65536: Operation not permitted\n\nBoth symptoms are an under-privileged docker container. RunPod pods ARE\nthe SkyPilot container (RunPod's runtime grants full privileges); on\nOCI we run docker ourselves inside the VM, default-unprivileged, so\nthe dbus / Xvfb / pedalboard X-stack can't operate. Match RunPod's\nprivilege level: add --privileged and --ulimit nofile=65536:65536.\n\n--privileged is correct here even by least-privilege standards: the OCI\nVM is single-tenant per-job (sky.launch + down=True) and the inner\ncontainer is the entire workload — there's no other process or user\non the VM to escape to.\n\nRefs #768\n\n* chore(skypilot): drop redundant plugins/ symlink from OCI template run block\n\nThe Dockerfile pre-creates plugins/Surge XT.vst3 -> /usr/lib/vst3/Surge XT.vst3\ninside WORKDIR at build time (docker/ubuntu22_04/Dockerfile:322-323), and\ngit init/fetch/checkout is used (instead of clone) specifically to preserve\nthat symlink across the source layer. The OCI template's run block does\nnot task.workdir-override or volume-mount over that path, so the runtime\n'mkdir -p plugins && ln -sf ...' was dead code.\n\nThe workflow's launcher container still needs the runtime symlink because\ndocker run -v $github.workspace:/home/build/synth-setter masks the image's\nWORKDIR contents — leave that one alone.\n\nRefs #768\n\n* style(skypilot): expand single-line OCI python -c into multi-line form\n\nReplace the one-liner python -c with a properly-formatted multi-line\nblock. Comment block above run: documents why the python body lines\nsit at the YAML block-scalar minimum indent (2 spaces in source = 0\nafter YAML strip) instead of matching the surrounding bash indent —\nPython -c rejects leading whitespace on top-level statements even\nwhen uniform.\n\nNo behavioral change.\n\nRefs #768\n\n* fix(skypilot): tighten OCI setup — apt-native lock wait, drop dead usermod, hard timeout on cloud-init\n\nFive fixes from review:\n\n  - apt-get -o DPkg::Lock::Timeout=300 — apt itself waits for the lock\n    (no race between fuser and the next command). Drops the manual\n    fuser-poll loop.\n  - timeout 300 sudo cloud-init status --wait — bounds the wait\n    explicitly; --wait has no internal timeout and could hang ~10min\n    silently.\n  - Drop sudo systemctl enable --now docker || sudo service ... fallback.\n    SkyPilot's OCI Canonical Ubuntu 22.04 image is systemd; the service\n    fallback masks real failures (apt incomplete, dpkg lock, etc).\n  - Drop sudo usermod -aG docker \"$USER\" — dead code. Group membership\n    requires re-login and run: uses sudo -E docker throughout. Was only\n    useful for human SSH debugging on a VM that gets torn down post-job.\n  - Removes the \"$USER\" reference, which was fragile under set -u in\n    SkyPilot's run shell.\n\nRefs #768\n\n* docs(skypilot): link #776 follow-up next to OCI --privileged invocation\n\nIssue #776 tracks the work to drop --privileged and replace it with the\nminimal cap-add / shm-size / ulimit combination needed for Xvfb + dbus\n+ pedalboard's Surge XT preset load. Comment block above run: now\npoints the reader at it so the temporary nature of the privilege\nescalation is documented in-source.\n\nRefs #768, #776\n\n* ci(skypilot): drop OCI iteration scaffolding from debug workflow\n\nRestores the 12-variant RunPod debug matrix and the RunPod cred-write\nstep that c2e1030 temporarily gated to OCI noop only, and drops the\ndiagnostic dumps from 0d44582 + d549fb2 (transient quota false-alarm\nchase, no longer needed). Keeps:\n\n  - configs/compute/oci-debug-template.yaml — useful sibling reference\n    for future OCI debug variants.\n  - skypilot[runpod,oci] installer extra — the [oci] dep is harmless on\n    RunPod-only matrix cells and avoids a re-install when an OCI noop\n    is added back later.\n\nHeader banner updated to point readers at the OCI sibling template\nwithout making it part of the default matrix.\n\nRefs #768\n\n* internal-fix(skypilot): use empty-string env placeholders in OCI template\n\nSYNTH_SETTER_WORKER_RANK / SYNTH_SETTER_NUM_WORKERS were set to \"0\" /\n\"1\" defaults, which lied about the contract: the launcher's\ntask.update_envs(...) injects per-rank values, so the defaults were\nshadowed and never read. Switch to \"\" placeholders matching every\nother launcher-injected key (and matching runpod-template.yaml).\n\nNo runtime behavior change today (update_envs already overwrites), but\nthe bogus defaults would mask the missing-env failure mode at exactly\nthe worst time: a future regression where the launcher fails to inject\nper-rank values would silently render rank=0/1 on every worker\ninstead of raising in pipeline.partitioning.read_rank_world_from_env.\n\nRefs #768\n\n* ci(skypilot): collapse RunPod + OCI generate jobs into one matrix\n\nReplaces two parallel `generate` + `generate-oci` jobs with a single\nmatrix-driven `generate` job over [runpod, oci]. Both cells exercise\nthe same provider-neutral launcher\n(pipeline.entrypoints.skypilot_launch_smoke) against per-provider\ncompute templates.\n\nLoad-bearing changes vs the prior shape:\n\n  - Both cells now run --num-workers 3, so the shard partitioner is\n    exercised end-to-end on every PR (previously RunPod was passing\n    --num-workers 3 explicitly; OCI was implicitly 1).\n  - RunPod gets a run-id-scoped cluster name\n    (synth-setter-smoke-runpod-${run_id}) — fixes the same R2 spec-key\n    race that the OCI step was patched for in e371a13. Without this,\n    the launcher's R2 spec key would still collide if a future PR adds\n    a parallel generate-oci-style job.\n  - The launch step is one `docker run` whose bash heredoc switches on\n    $PROVIDER for cred-write (case \"$PROVIDER\" in runpod) ... ;; oci) ... ;;\n    esac), avoiding two divergent docker invocations.\n  - `continue-on-error` is per-matrix-cell (false for RunPod, true for\n    OCI while it accumulates a track record). Flip OCI to false once\n    3+ consecutive runs are green.\n  - `fail-fast: false` so a transient on one provider doesn't kill the\n    other.\n  - Artifacts renamed to test-run-metadata-${provider}; validate-spec\n    and validate-shard updated to reference test-run-metadata-runpod\n    (matrixing them follows in the next commit).\n\nThe OCI cell still carries the runtime `pip install skypilot[oci]`\nbridge — that's dropped once the post-merge dev-snapshot rebuild\nbakes in the [oci] extra.\n\nRefs #768\n\n* ci(skypilot): matrix validate-spec over RunPod + OCI\n\nvalidate_spec.py is provider-neutral (reads required fields from\ninput_spec.json structurally), so the only per-cell variation is the\nartifact name. fail-fast: false mirrors the generate matrix; OCI cell\nstays continue-on-error: true while it accumulates a track record.\n\nRefs #768\n\n* ci(skypilot): matrix validate-shard over RunPod + OCI\n\nSame pattern as the prior validate-spec matrixing. The per-shard\ndownload + h5py validation loop already iterates spec.shards[*] and\nparses r2_prefix from the spec, so it works as-is for both providers\nonce the artifact name is parameterized.\n\nAfter this lands, every PR exercises 6 matrix cells: 2 generate, 2\nvalidate-spec, 2 validate-shard.\n\nRefs #768\n\n* fix(skypilot): wire WORKER_GIT_REF through OCI worker container\n\nThe launcher already forwards WORKER_GIT_REF via task.update_envs (it's\nin pipeline.entrypoints.skypilot_launch_smoke._WORKER_ENV_KEYS), but\nthe OCI template's run: block was dropping it on the floor:\n\n  - envs: had no WORKER_GIT_REF placeholder, so SkyPilot's update_envs\n    wouldn't set it on the OCI VM.\n  - The nested `sudo -E docker run ...` lacked `-e WORKER_GIT_REF`, so\n    even if the VM had the value, it wouldn't reach the worker.\n  - The inner bash had no fetch/checkout logic.\n\nResult: OCI matrix cell ran whatever code was baked into the dev-\nsnapshot image, ignoring the PR's commit. RunPod and OCI cells gave\ninconsistent smoke signals on PR CI.\n\nMirror the RunPod template's contract: placeholder in envs:, forward\nvia -e, guarded fetch+checkout (validate ref looks like a 7-40 char\nhex SHA before passing to git, and use safe.directory + FETCH_HEAD to\navoid touching the working tree's index permissions).\n\nRefs #768\n\n* ci(skypilot): assert sed pin substitution and decouple per-provider validators\n\nTwo PR-feedback fixes bundled (both in the same workflow file):\n\n1. Pin step now asserts the sed substitution actually happened (drift-\n   resistance for Copilot review #3178403620). sed silently no-ops when\n   PIN_SEARCH stops matching the template text (e.g. someone reformats\n   the template, or renames the env key). Without this check, CI would\n   proceed against the dev-snapshot default tag instead of the\n   dispatched IMAGE_TAG. Now: fail the workflow if PIN_SEARCH is still\n   present after sed and REPLACE != PIN_SEARCH (PR CI's no-op case),\n   AND fail if REPLACE is not present.\n\n2. validate-spec / validate-shard now run with `if: ${{ !cancelled() }}`\n   so each provider's validator is decoupled from the OTHER provider's\n   generate outcome. Previously, a RunPod transient would skip BOTH\n   validate cells (needs: generate marks the whole job failed) — losing\n   OCI signal for reasons unrelated to OCI. Now: each provider's\n   validator runs as long as the workflow wasn't cancelled; the cell\n   whose generate didn't produce an artifact fails at download-artifact,\n   which is the right per-cell signal.\n\nRefs #768\n\n* refactor(skypilot): address PR #777 review feedback\n\nCode-health BLOCKs:\n- Trim multi-paragraph rationale comments in oci-cpu-template.yaml,\n  runpod-template.yaml, and test-dataset-generation.yml (CLAUDE.md\n  one-line rule). Canonical context lives in design doc / #735 / #776.\n- Extract shared worker run-block to scripts/skypilot_worker_run.sh\n  (RunPod + OCI both invoke). Removes the duplicated git-checkout +\n  python -c os._exit(0) block that had to be edited in two places.\n\nShell-style BLOCKs:\n- Add set -euo pipefail to outer GHA run: blocks (pin step, launch step,\n  validate-spec, validate-shard) and to oci-debug-template.yaml.\n- Replace single-bracket [ ] with [[ ]] in oci-cpu-template / workflow.\n- Move comment block out of \"Pin worker image tag\" run-scalar (CLAUDE.md\n  no-comments-inside-run rule); rationale now sits above the step.\n\nSynth-setter BLOCK:\n- Fix pin-assertion logic: previous logic short-circuited in the default\n  dev-snapshot PR-CI path because REPLACE == PIN_SEARCH made both checks\n  no-ops. Replace with pre-count assertion (PIN_SEARCH must occur\n  exactly once before sed) + post-state checks. Verified locally that\n  drift cases (missing/duplicated PIN_SEARCH) now fail loudly.\n\nTdd-refactor BLOCKs (doc drift caused by this PR):\n- Update docs/reference/github-actions.md: artifact name (now\n  per-provider), test-dataset-generation description, secrets table\n  (six new OCI_* secrets).\n- Update docs/reference/docker.md: per-provider artifact name + gh run\n  download examples.\n\nCode-health WARNs:\n- Drop redundant pin_grep matrix field; final grep prints the rewritten\n  line directly.\n- Consolidate continue-on-error pattern: all three jobs (generate,\n  validate-spec, validate-shard) now read continue_on_error from matrix\n  include for symmetry.\n- Add concurrency group at workflow level (cancel-in-progress) so\n  back-to-back PR pushes don't queue stacked billable RunPod/OCI runs.\n- Hoist the skypilot:cpu-ubuntu-2204 magic literal into matrix include\n  (oci_image_tag) so workflow + template comments share a single source.\n- Mark the WORKER_IMAGE default in oci-cpu-template.yaml as the CI sed\n  pin target (one-line comment) so readers don't mistake it for inert.\n- Bump cluster name to include github.run_attempt — re-running a failed\n  job no longer collides on the launcher's R2 spec key.\n\nShell-style WARNs:\n- Consistent braced quoting in pin step.\n- Separate decl from cmd-sub for R2_BUCKET / R2_PREFIX (SH10).\n\nGHA WARNs:\n- Bump actions/setup-python @v5 → @v6 in test-skypilot-debug.yml\n  (consistency with other workflows).\n- Assert ~/.oci/config region= and ~/.sky/config.yaml compartment_ocid\n  are non-empty before sky check oci — opaque empty-secret failures\n  surface a clear error instead.\n- pip install bridge wraps in explicit failure path; fall-through error\n  message is clearer than the downstream import error.\n\nSynth-setter WARNs:\n- Drop sibling-YAML cross-reference and OCPU/GB restatement comments in\n  oci-debug-template.yaml (CLAUDE.md \"don't bake values into comments\").\n- Update CLAUDE.md project blurb to mention SkyPilot-managed compute\n  (RunPod + OCI), not just RunPod.\n- Add OCI_COMPARTMENT_OCID + image_tag_general step to getting-started\n  §4e so local operators don't hit a missing-compartment failure.\n- Add three-places-in-sync invariant comment next to the skypilot pin\n  in requirements-app.txt.\n\nTdd-refactor WARNs:\n- Update docs/doc-map.yaml SkyPilot-integration block: add OCI templates,\n  scripts/skypilot_worker_run.sh, the new per-provider workflow shape,\n  and bump the requirements-app.txt extras string.\n\nJustified as-is (won't fix, with reasons posted on each thread):\n- oci-debug-template.yaml YAGNI: deferred per the linked comment in\n  test-skypilot-debug.yml until OCI cred-write lands in debug workflow.\n- \"|| true\" on cloud-init wait: deliberate fail-open documented above\n  the block; reviewer marked advisory.\n- git fetch retry: reviewer suggested \"consider\"; not adding.\n- Vast.ai drift in skypilot-compute-integration.md lines 277-278/362:\n  PR description explicitly defers to a follow-up doc PR.\n- Rename runpod-smoke-shard.yaml → smoke-shard.yaml: reviewer's own\n  suggestion is \"post-merge\"; deferred.\n\nRefs #777\nRefs #768\n\n* fix(skypilot): move WORKER_GIT_REF checkout out of shared worker script\n\nThe previous extraction (19db966) put the git-checkout *inside*\nscripts/skypilot_worker_run.sh, but PR CI invokes the script BEFORE\nthe checkout has run — and the dev-snapshot image hasn't been rebuilt\nyet, so the script doesn't exist on disk at invocation time. Worker\nexited 127 (command not found) on both providers.\n\nFix: keep the script for the python heredoc + #735 workaround only;\nmove the WORKER_GIT_REF git checkout back into each template's run:\nblock, before the script invocation. The checkout is what brings\nscripts/skypilot_worker_run.sh into the baked image's working tree\nuntil the next dev-snapshot rebuild bakes it in.\n\nRefs #777\n\n* refactor(skypilot): extract worker checkout logic to its own script\n\nSplits the WORKER_GIT_REF git checkout out of the templates' inline\nbootstrap into scripts/skypilot_worker_checkout.sh. Both compute\ntemplates now share one place for checkout logic too — symmetric with\nthe existing scripts/skypilot_worker_run.sh extraction.\n\nBootstrapping for the not-yet-rebuilt dev-snapshot image: the templates\nfetch the ref's git objects via the image's existing baked clone, then\ngit show <ref>:scripts/skypilot_worker_checkout.sh extracts the script\ncontent into /tmp without touching the working tree. bash that, which\ndoes the actual git checkout, after which scripts/skypilot_worker_run.sh\nis on disk for the worker invocation. No external endpoints involved.\n\nRefs #777\n\n* refactor(skypilot): collapse worker bootstrap into a single script\n\nscripts/skypilot_worker_run.sh now owns the full worker side: optional\ngit checkout to WORKER_GIT_REF + the python invocation with the #735\nos._exit(0) workaround. scripts/skypilot_worker_checkout.sh deleted.\n\nTemplates do the irreducible bootstrap (cd + git config + WORKER_GIT_REF\nformat-check + git fetch) and then `bash <(git show <ref>:scripts/skypilot_worker_run.sh)`,\nwhich streams the script straight from git's object DB through process\nsubstitution. No separate temp-file stage, no second extracted script.\n\nRefs #777\n\n* refactor(skypilot): keep bootstrap inline; script owns python only\n\nscripts/skypilot_worker_run.sh now owns just the python invocation +\n#735 os._exit(0) workaround — the original B2 review concern. Each\ntemplate's run: block keeps the inline bootstrap (cd + git config +\nWORKER_GIT_REF format-check + git fetch + git checkout FETCH_HEAD)\nbecause the script must be on disk for bash to run it, and the\nnot-yet-rebuilt dev-snapshot image doesn't have the script until the\ncheckout itself lands.\n\nReverts c75f7c2 + 1acf114 (separate checkout script + bash <(git show)\nprocess-substitution bootstrap).\n\nRefs #777\n\n* docs(skypilot): address PR #777 Copilot review nits\n\nDoc/comment-only fixes — no behavioral change.\n\n- docs/doc-map.yaml: skypilot_worker_run.sh `covers` no longer claims the\n  script does the WORKER_GIT_REF checkout (it doesn't — templates do).\n  oci-debug-template.yaml `covers` clarifies it's not currently in any\n  CI matrix.\n- docs/design/skypilot-compute-integration.md: replace incorrect \"the\n  run: block is overridden programmatically\" with the actual launcher\n  contract (instantiates Task from YAML, only calls update_envs).\n- configs/compute/oci-debug-template.yaml: header no longer claims the\n  template is \"used by test-skypilot-debug.yml\" — that workflow's matrix\n  is RunPod-only; the OCI cell lands in a follow-up PR.\n- scripts/skypilot_worker_run.sh: collapse stale \"see runpod-template\"\n  pointer to a one-line `# Workaround for #735.` per CLAUDE.md.\n\nRefs #777",
          "timestamp": "2026-05-03T14:43:59-04:00",
          "tree_id": "5c30d432274896b3231e65f249bdfc17660cc8e5",
          "url": "https://github.com/tinaudio/synth-setter/commit/adfd7ab4e1fdba639f812db874e1086dccddc471"
        },
        "date": 1777834634669,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.9457130432128906,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.223210551142692,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.02435869164764881,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.02904796600341797,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.3942902088165283,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 12.09089426708333,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.593145847320557,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.7639749541319905,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.0327242948114872,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.035304129123687744,
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
          "id": "b17b4c2264ca6d279a4edf56250971ec7308d3e0",
          "message": "refactor(pipeline): drop OCI bridge + collapse provider matrix (#803)\n\n`skypilot[runpod,oci]==0.12.0` ships in dev-snapshot via requirements-app.txt\n(Dockerfile installs requirements.txt, which includes requirements-app.txt),\nand #797 made the image rebuild on every merge to main, so the runtime\n\"bridge\" workarounds in test-dataset-generation.yml + skypilot_launch_smoke.py\nare dead weight.\n\nRemoves:\n- Conditional `pip install skypilot[oci]==0.12.0` + `sky check oci` block\n  inside the OCI launch step. `sky check oci` itself stays — useful as a\n  fast-fail probe of the cred file we just wrote.\n- `try/except ImportError` around `from sky.clouds import OCI` in\n  `_override_image_id` (now a direct module-level import inside the\n  function). The matching test_does_not_crash_when_oci_extras_missing\n  test goes with it.\n- Stale comment block in requirements-app.txt referring to the bridge.\n\nFolded in: collapse the dynamic-matrix setup script. Once `oci_image_tag`\nno longer needs to ride along, the matrix only needs the provider name —\ntemplate / cluster prefix / OCI image tag derive cleanly from\n`matrix.provider` via expressions in the consuming step. The `setup` job\nnow publishes a single `providers` JSON array; `generate_matrix`,\n`validate_matrix`, and `has_jobs` outputs are gone, as are the three\n`needs.setup.outputs.has_jobs == 'true'` gates (empty `fromJSON('[]')`\nalready skips a matrix job natively). Setup script: ~60 lines → ~15.\n\nCloses #800.",
          "timestamp": "2026-05-04T19:37:03-04:00",
          "tree_id": "9787aa628b823a54193284279684cc74034c8a2d",
          "url": "https://github.com/tinaudio/synth-setter/commit/b17b4c2264ca6d279a4edf56250971ec7308d3e0"
        },
        "date": 1777938546251,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 4.402127265930176,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.101785780694335,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.03251595422625542,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.0319744348526001,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.254378080368042,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 10.709790084,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.481884002685547,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.578242529882118,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.03251595422625542,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.043689608573913574,
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
          "id": "f940b9f16a7f39029eaca346ab50d1a5b752f150",
          "message": "build(deps): add oci sdk as standalone dep in requirements-app.txt (#825)\n\nCurrently pulled in transitively via skypilot[oci]. Adding it as a\ntop-level dep so we can import it directly without relying on the\nSkyPilot extra's resolution.\n\nRefs #785",
          "timestamp": "2026-05-06T08:58:50-04:00",
          "tree_id": "4e17e8814d1207ee40be40fc9538ae537bb1094b",
          "url": "https://github.com/tinaudio/synth-setter/commit/f940b9f16a7f39029eaca346ab50d1a5b752f150"
        },
        "date": 1778073000814,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.9993810653686523,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.437804043715587,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.025723451748490334,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.019142448902130127,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.5113372802734375,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 12.25392406291663,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.5465474128723145,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.437804043715587,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.035062275826931,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.0421527624130249,
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
          "id": "f32d49d889d4a80a63521c486272667a630d9a1f",
          "message": "feat(param-spec): SURGE_4 mini-example param spec and preset registry (#820)\n\n* internal-feat(vst): add SURGE_4_PARAM_SPEC mini-example and preset registry\n\nAdds a 4-parameter Surge XT spec (SURGE_4_PARAM_SPEC: amp envelope attack,\nfilter cutoff, LFO amplitude/rate) and a preset_paths registry mapping\nparam_spec names to their base preset files. The spec underlies the\nsmoke-test fixture and the predict_vst_audio end-to-end test.\n\n- param_specs[\"surge_4\"] registered alongside surge_xt/surge_simple.\n- preset_paths dict added so future code paths can look up the matching\n  preset by spec name (script wiring lands separately).\n- tests/conftest.py uses surge_4 + presets/surge-mini.vstpreset for the\n  surge fixture; cfg.model.net.d_out now derives from\n  len(param_specs[\"surge_4\"]) instead of being a literal 7 with a\n  comment that would drift when the spec changes.\n- presets/*.fxp gitignored — local-dev learned-model artifacts excluded\n  from version control by default; commit explicitly with git add -f when\n  one becomes a versioned base preset.\n- Docs cross-reference preset_paths from --param-spec-name and\n  --preset-path so users know the two flags should agree.\n\nRefs #811\n\n* test(surge): templatize cfg_surge_xt_global() over param_spec_name\n\nAdds a `param_spec_name` fixture (default \"surge_4\") that drives the surge\nfixtures: `cfg_surge_xt_global` propagates it to `model.net.d_out` and the\n`log_per_param_mse` callback; `surge_xt_smoke_datasets` derives the matching\n`--param_spec` and `--preset_path` from `preset_paths`. Tests can override\nvia indirect parametrization.\n\nAlso plumbs the spec through `predict_vst_audio.py` in the surge train+eval\ne2e test — the script previously defaulted to `--param_spec=surge_xt` while\nthe fixture trained on surge_4, so decode sliced past the end of the\npredicted tensor and crashed MPS CI with \"can only convert an array of size\n1 to a Python scalar\".\n\nAdds a fast cfg-composition test parametrized over surge_4, surge_simple,\nsurge_xt to lock the templating contract for every supported spec.\n\n* test(configs): add surge/test-mps experiment + cfg-equality guard\n\nAdds `configs/experiment/surge/test-mps.yaml`, a Hydra experiment that\nresolves to the same cfg `cfg_surge_xt_global(accelerator=\"mps\",\nparam_spec_name=\"surge_4\")` builds in `tests/conftest.py`. Inherits from\n`surge/base` and overrides `/trainer: mps`, `/callbacks: [default_surge,\neval_surge]` so the fixture's open_dict bake-ins (precision=32-true,\ndeterministic, max_steps=1, batch_size=1, lr_monitor null, etc.) are\nexpressed declaratively.\n\nTo pin the equality contract:\n\n- Extracts `_build_surge_xt_smoke_cfg(accelerator, param_spec_name)` from\n  the existing `cfg_surge_xt_global` fixture so the cfg can be built on\n  any host (the fixture's accelerator gate hardfails non-MPS runners\n  before composing). The fixture is now a thin wrapper.\n- Switches the lr_monitor cleanup from `del` to `= None`. `instantiate_callbacks`\n  skips entries without `_target_`, so runtime behavior is unchanged, and\n  the cfg now matches what `lr_monitor: null` produces on the YAML side.\n- Adds `test_test_mps_yaml_matches_cfg_surge_xt_global` in\n  `tests/test_configs.py`: composes both sides with `resolve=False`,\n  strips volatile top-level keys (`paths`, `hydra`, `task_name`), and\n  asserts deep equality with a human-readable diff on failure.\n\nFuture drift in either the fixture or test-mps.yaml fails fast.\n\n* internal-fix(vst): reformat param_specs/preset_paths dicts and annotate\n\nAddresses Copilot review comments #3192020841 and #3202813835 on PR #820:\n- Multi-line ``param_specs`` dict so ``ruff format`` (line-length 99)\n  stops complaining about the 119-char single-line literal.\n- Type-annotates both registries (``dict[str, ParamSpec]`` and\n  ``dict[str, str]``) so attribute access is type-checked at the call\n  sites and the ``preset_paths`` keys can't drift out of sync with\n  ``param_specs`` without lint surfacing it.\n\nThe third inline comment (#3192020859 — \"comment claims SURGE_4 is used\nby predict_vst_audio test, but the test uses defaults\") was already\nresolved by 2331be5, which plumbs ``--param_spec=surge_4\n--preset_path=presets/surge-mini.vstpreset`` through to the test's\n``predict_vst_audio.py`` invocation. No code change needed there.\n\n* test(surge): pin test_cfg_surge_xt_global_wires_param_spec to cpu\n\nConda CI runs ``pytest -m \"not slow\"`` which includes the (un-slow)\n``test_cfg_surge_xt_global_wires_param_spec`` test. The previous version\nwent through the ``cfg_surge_xt_global`` fixture, which depends on the\nparametrized ``accelerator`` fixture — and that fixture hardfails the\n``[mps-*]`` and ``[gpu-*]`` parametrizations on Linux runners with\n\"MPS not available\" / \"CUDA not available\", failing the conda job.\n\nThe cfg-shape contract this test asserts is accelerator-independent\n(``model.net.d_out`` and ``callbacks.log_per_param_mse.param_spec``\nare set by ``_build_surge_xt_smoke_cfg`` regardless of the ``accelerator``\nargument). Call the builder directly with ``accelerator=\"cpu\"`` and drop\nthe indirect parametrization so only the three param_spec cases run on\nevery CI runner.\n\n* fix(test): loosen SILENCE_PEAK_THRESHOLD in surge train+eval e2e\n\nLowers the ``SILENCE_PEAK_THRESHOLD`` from 1e-4 (~-80 dBFS) to 1e-6\n(~-120 dBFS) in ``test_train_eval_surge_xt``. The previous threshold was\nchosen with the rationale that ``compute_rms`` underflows below 1e-4, but\nthat's not actually true: ``compute_rms``'s NaN risk is the cosine-similarity\ndenominator collapsing to 0, which only happens on bit-zero audio.\n\nSymptom: MPS CI on ``faf2be1`` (and ``5b168b8``) failed with\n``sample_0/pred.wav is silent (peak=3.05e-05)`` even though peak\n3.05e-5 → ~-90 dBFS would not actually underflow downstream metric math.\nThe 1-step-trained smoke model's predicted params, rendered through\nSurge XT, can land in a quiet (but non-silent) region of param space — and\nthe dataset generator runs without a fixed seed, so the trained model and\nits predictions vary run-to-run.\n\nLoosening to 1e-6 keeps the original guard against truly silent (bit-zero)\naudio while letting the legitimate \"trained for one step on a randomly-sampled\n5-clip fixture\" prediction through. The downstream\n``np.isfinite(numeric).all()`` assertion on the metrics CSV remains the\nreal correctness check; the silence threshold is just an early-warning\nfast-fail.",
          "timestamp": "2026-05-07T21:19:07Z",
          "tree_id": "7231f8ac0b6c4223343729093421e1d7bccfbb81",
          "url": "https://github.com/tinaudio/synth-setter/commit/f32d49d889d4a80a63521c486272667a630d9a1f"
        },
        "date": 1778189387792,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.292647123336792,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.057547753052786,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.018409153446555138,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.006813645362854004,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.1730427742004395,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 10.66544505516667,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 3.723017692565918,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.279242483135313,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.02462758868932724,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.02509409189224243,
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
          "id": "81077272c042792a4441e6c11945529cf5f51878",
          "message": "refactor(workflows): split test-dataset-generation; rename launcher (#858)\n\n* refactor(workflows): extract generate-dataset-shards.yaml; rename skypilot_launch_smoke\n\nSplits test-dataset-generation.yml into a thin wrapper plus two reusable\nworkflows:\n\n* `generate-dataset-shards.yaml` — workflow_call only. Owns one provider's\n  launcher invocation (skypilot-local kind setup or runpod/oci in-container\n  launcher). Inputs: provider, dataset_config, image_tag, cluster_name,\n  num_workers, tail, api_server, local, artifact_name. Becomes the official\n  launcher entry point that follow-up PRs (R2-as-coordination, expanded\n  dispatch surface) build on.\n* `validate-dataset-shards.yaml` — workflow_call only. Owns validate-spec\n  + validate-shard jobs.\n* `test-dataset-generation.yml` keeps PR/dispatch triggers (3 inputs\n  unchanged) and computes the provider matrix; calls the two reusables\n  per provider. The docker-only `local` row stays inline (no launcher).\n\nAlso renames `pipeline/entrypoints/skypilot_launch_smoke.py` →\n`skypilot_launch.py` (and the matching test) since the launcher is no\nlonger smoke-specific. Updated all callers: test-skypilot-debug.yml,\ntest-dataset-generation.yml's paths filter, the compute templates'\nheader comments, scripts/sync_worker_checkout.sh, and the doc set.\n\nDeletes obsolete `dataset-generation.yml` (no callers, superseded by the\nunified launcher).\n\nBehavior-preserving — every flag the test wrapper passes to the reusable\nmatches the value today's inline blocks hardcoded (num_workers=1 +\nlocal=true for skypilot-local; defaults elsewhere).\n\nRefs #856\n\n* docs: fix stale dataset-generation.yml references after workflow split\n\nThe doc-drift agent surfaced doc references to the deleted `dataset-generation.yml`\nworkflow that the rename pass missed. Updated four files:\n\n* docs/doc-map.yaml — replace the deleted-workflow pattern with the two new\n  reusables (generate-dataset-shards.yaml + validate-dataset-shards.yaml).\n* docs/reference/github-actions.md — replace the `dataset-generation` row in\n  the Pipeline catalog with rows for both new reusables; refresh the\n  dependency map; replace `dataset-generation` in the Used-by columns of the\n  R2 + W&B secrets table; update the runtime-secrets and\n  mount-as-volume sections.\n* docs/design/storage-provenance-spec.md — update the workflow table row to\n  describe `generate-dataset-shards.yaml` (with its actual input set) and\n  add a sibling row for `validate-dataset-shards.yaml`.\n* .github/workflows/test-vst-slow.yml — update the comment that cites\n  `dataset-generation.yml` as the headless-X11 proof point to point at\n  `generate-dataset-shards.yaml`.\n\nRefs #856",
          "timestamp": "2026-05-08T00:19:24Z",
          "tree_id": "933671bdc7d3ac1f51cc8243a5873a3a13c94db7",
          "url": "https://github.com/tinaudio/synth-setter/commit/81077272c042792a4441e6c11945529cf5f51878"
        },
        "date": 1778200305639,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.594949722290039,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.987972955955192,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.02399531565606594,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.014654576778411865,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.207710027694702,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 12.17246782975,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.578157901763916,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.359780759289861,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.03254003822803497,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.03876692056655884,
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
          "id": "1eb0ef131e1dbfa4f2b8f2d3c0cede03349dd841",
          "message": "chore(deps): add ruff and pydantic-settings to requirements-app.txt (#894)\n\nruff is already configured (pyproject.toml [tool.ruff*]) and runs in\npre-commit, but isn't a direct dev dep — adding it lets contributors\ninvoke `ruff check` / `ruff format` from editors and the CLI without\nshelling out through the pre-commit harness.\n\npydantic-settings is required for the planned migration in #885\n(generate_vst_dataset CLI auto-generated from RenderConfig fields).\n\nRefs #885",
          "timestamp": "2026-05-11T05:43:37Z",
          "tree_id": "2f32f544751c56028dbb770cad1b7e8194814a3d",
          "url": "https://github.com/tinaudio/synth-setter/commit/1eb0ef131e1dbfa4f2b8f2d3c0cede03349dd841"
        },
        "date": 1778478998553,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 2.9823291301727295,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 4.28047621806385,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.021709546446800232,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.010849237442016602,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 2.8688931465148926,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 12.026773935666666,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.411105632781982,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.896789592169225,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.0330335795879364,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.041991591453552246,
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
          "id": "b4830f755d99be27f99869c4cb7067cbe5296864",
          "message": "fix(evaluation): clamp compute_rms denominator to defuse MPS pred.wav silence flake (#899)\n\n* fix(testing): clamp compute_rms denominator to defuse MPS pred.wav silence flake\n\n`test_train_eval_surge_xt[mps]` intermittently failed with `pred.wav is silent`\nbecause MPS has non-deterministic ops and a 1-step-trained model occasionally\npredicted params Surge XT renders below -120 dBFS. The silence assertion\nexisted only as a defensive proxy for `compute_rms`'s `0/0 → NaN` when\n`pred_norm = 0`.\n\nMove the protection into `compute_rms` itself (matches the epsilon-clip\npattern already used in `compute_sot`), so silent pred yields\n`cosine_sim = 0` rather than NaN. Drop the pred.wav silence assertion; keep\nthe target.wav check (target silence would be a real bug).\n\nReturning 0 is within the natural [0, 1] range of cosine similarity for\nnon-negative vectors and correctly penalizes silent predictions; it cannot\nbe gamed upward. No consumer relies on NaN-as-marker.\n\nCloses #898\n\n* fix(testing): short-circuit compute_rms underflow to actually return 0\n\nPer Copilot review on PR #899: the prior commit logged \"returning 0\" on\ndenominator underflow but still computed ``dot/np.clip(denom, 1e-12, None)``,\nwhich only collapsed to 0 when the numerator was exactly 0 (bit-silent pred).\nFor quiet-but-non-zero inputs the clamped division returned an unbounded\nsmall value, contradicting the warning text and the PR's documented intent.\n\nMove the clamp branch to an explicit ``return 0.0`` and add a regression test\nwith ``target = pred = uniform 1e-7`` that would have returned ~0.4 pre-fix.",
          "timestamp": "2026-05-11T02:33:58-04:00",
          "tree_id": "87433cbbb9491ed60b0129b4d29bc731c66dc02a",
          "url": "https://github.com/tinaudio/synth-setter/commit/b4830f755d99be27f99869c4cb7067cbe5296864"
        },
        "date": 1778482077656,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.6596744060516357,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.252902333587408,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.023899059742689133,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.027762949466705322,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.1610922813415527,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 10.985038660083335,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 3.8948307037353516,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.5721086424589155,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.027645153924822807,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.02806752920150757,
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
          "id": "1181351e9c287dfdd8f4f25e3acb88fd3fe8c3e5",
          "message": "internal-feat(schemas): unify DatasetConfig + DatasetPipelineSpec into DatasetSpec (#887)\n\n* internal-feat(schemas): unify DatasetConfig + DatasetPipelineSpec into DatasetSpec\n\nReplace the prior split between DatasetConfig (YAML-shaped config) and\nDatasetPipelineSpec (runtime-materialized artifact) with a single\nDatasetSpec model whose model_dump_json() is the artifact written to R2.\nRenderer-specific fields move to a nested RenderConfig sub-model.\nRuntime fields (git_sha, is_repo_dirty, created_at, run_id, r2_prefix)\nauto-fill via default_factory when missing and pass through when present\nin JSON-loaded input. shards / num_shards / num_params are computed\ndeterministically as @computed_field cached_properties.\n\nSURGE_XT_RENDERER_VERSION moves out of the schema into RenderConfig as\na config field; the worker still verifies the running plugin version\nmatches the pinned value before rendering.\n\nA legacy YAML loader (load_dataset_spec_yaml) keeps the launcher and\nci.materialize_spec working through this PR; both are removed in a\nfollow-up PR once the entrypoint migrates to @hydra.main.\n\nThe launcher's num_workers knob now lives on the CLI only (default 1);\nthe legacy YAML field is silently ignored.\n\noutput_format remains restricted to \"hdf5\" — wds support lands later\nin the chain.\n\nCloses #886\nPart of #882\n\n* fix(compute): invoke synced docker_entrypoint.py, not stale baked path\n\nThe skypilot templates execed /usr/local/bin/entrypoint.py — the copy baked\ninto the dev-snapshot image. After the pipeline/ → src/pipeline/ relocation\nand src/generate_dataset.py entrypoint move, the in-image script's imports\n('from pipeline.entrypoints.generate_dataset ...') stopped resolving and PR\nworkers failed with ModuleNotFoundError: No module named 'pipeline'.\n\nsync_worker_checkout.sh already updates /home/build/synth-setter to the PR\nhead ref before launch, so invoke scripts/docker_entrypoint.py from the\nsynced checkout instead. The Dockerfile still bakes the same script at\n/usr/local/bin/entrypoint.py for the no-sync (no WORKER_GIT_REF) fallback.\n\nRefs #882\n\n* ci(workflows): install pydantic + pyyaml + omegaconf for validate-spec runner step\n\nAfter unifying DatasetConfig + DatasetPipelineSpec into a single Pydantic\nDatasetSpec, `pipeline.ci.validate_spec` imports\n`pipeline.schemas.spec.DatasetSpec`. The spec module transitively imports\npydantic, pyyaml (for the load_dataset_spec_yaml bridge function), and\nomegaconf — none of which are on the runner's bare Python install.\n\nInstall only those three packages to keep the runner-side env minimal\ninstead of pulling the full requirements.txt (which would drag in torch\nand the rest of the training stack).\n\n* internal-fix(schemas): address Copilot review feedback on PR #887\n\n- pipeline/schemas/spec.py _strip_computed_field_keys: copy the input\n  mapping before popping computed keys so callers holding the dict\n  (logging, retries) see it unchanged (Copilot #3216318943).\n- pipeline/schemas/spec.py legacy YAML bridge: raise ValueError when\n  legacy 'num_shards' disagrees with sum(splits) instead of silently\n  computing a different shard count (Copilot #3216318975).\n- pipeline/schemas/prefix.py make_r2_prefix: strip leading/trailing\n  slashes from prefix_root so 'data/' and '/data' both produce a clean\n  prefix instead of doubled slashes; reject empty-after-strip with a\n  clear error (Copilot #3216319001).\n- pipeline/schemas/spec.py OUTPUT_FORMAT_TO_EXTENSION: rename from the\n  private '_OUTPUT_FORMAT_TO_EXTENSION' and add to __all__ so\n  pipeline.ci.validate_spec is no longer reaching across a private\n  boundary (Copilot #3216319015).\n- pipeline/ci/validate_spec.py: validate output_format in\n  validate_structure and look up extension via .get(...) in\n  validate_test_values so an unknown format produces a structural\n  error instead of a KeyError crash (Copilot #3216319025).\n\nAlso adds the missing docstrings required by interrogate (80% threshold)\non the touched files: the validator/computed-field methods in spec.py\nthat lacked them, and the existing tests in test_dataset_spec.py that\nwere previously undocumented.\n\n* internal-fix(schemas): defer param_specs import inside num_params\n\n`pipeline.schemas.spec` top-level imported `from src.data.vst import param_specs`,\nwhich transitively pulls `src.data.vst.core` → `mido` + `pedalboard`. The\nvalidate-spec runner doesn't install those, so `python -m pipeline.ci.validate_spec`\non the GitHub runner aborts with `ModuleNotFoundError: No module named 'mido'`\nbefore any validation runs.\n\nMove the import inside `num_params`'s body — the only call site. Side effects\nof `src.data.vst.__init__` (mido / pedalboard imports) now only happen when a\nspec's `num_params` is actually evaluated, not when the schema module is\nimported. `validate_spec` only consumes the module-level\n`OUTPUT_FORMAT_TO_EXTENSION` constant, so the deferred import is fine for that\ncode path.\n\n* internal-fix(schemas): address second Copilot review round on PR #887\n\n- pipeline/schemas/spec.py: add `frozen=True` to `DatasetSpec` and\n  `RenderConfig` so the `@cached_property` computed fields (`shards`,\n  `num_shards`, `num_params`) cannot go stale via post-construction field\n  mutation. The internal `_populate_derived_runtime_fields` validator\n  already uses `object.__setattr__`, which bypasses Pydantic's frozen\n  guard, so init-time runtime-field population still works.\n\n- pipeline/entrypoints/generate_dataset.py: replace the misleading\n  \"renderer CLI dispatches on filename suffix\" claim in both\n  `build_generate_args` and `run` docstrings with HDF5-only reality.\n  Drop the `configs/render/<spec>.yaml` / `configs/render/surge_xt.yaml`\n  references in the renderer-version inline comment and error message\n  (this PR keeps materialization in legacy `configs/dataset/*.yaml`;\n  the Hydra `configs/render/` group lands in PR-2).\n\n- src/data/vst/core.py: drop the `configs/render/<spec>.yaml` reference\n  in `extract_renderer_version`'s docstring.",
          "timestamp": "2026-05-11T02:47:18-04:00",
          "tree_id": "f514fd01bc7df5ec77f412c2b21fd73c322a895f",
          "url": "https://github.com/tinaudio/synth-setter/commit/1181351e9c287dfdd8f4f25e3acb88fd3fe8c3e5"
        },
        "date": 1778482852257,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.16964054107666,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.417810511142015,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.015445916913449764,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.00636821985244751,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.021010637283325,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 11.868632628249998,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 3.411895751953125,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 5.957260514153168,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.018764929845929146,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.011965036392211914,
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
          "id": "1f6eb7a34e4320b73ed5d42fd72c6a2b86b41167",
          "message": "internal-feat(vst): renderer signatures take RenderConfig + migrate CLI to pydantic-settings (#942)\n\n* internal-feat(vst): renderer signatures take RenderConfig + migrate CLI to pydantic-settings\n\n`make_dataset` now takes a single `render_cfg: RenderConfig` arg in place of\nnine separate kwargs. `param_spec_name` is resolved against the in-process\nregistry inside `make_dataset` (previously the launcher did the lookup);\n`num_samples` comes from `render_cfg.batch_per_shard`. The `fixed_*_params_list`\nkwarg-only args remain for `surge_xt_interactive` and the fixed-params tests.\n\nThe CLI on `generate_vst_dataset.py` is rewritten using pydantic-settings:\n`_GenerateCliArgs(RenderConfig, BaseSettings)` inherits every `RenderConfig`\nfield so the CLI flag set tracks the model automatically. Adding/removing a\nfield on `RenderConfig` extends/shrinks the CLI without a parallel update.\nA new test in `tests/data/vst/test_generate_vst_dataset_cli.py` pins the\nparity invariant.\n\n`pipeline/entrypoints/generate_dataset.py::build_generate_args` derives the\nflag set from `RenderConfig.model_fields` for the same reason — single source\nof truth for the renderer config surface.\n\n`scripts/surge_xt_interactive.py` constructs a `RenderConfig` for its\ncaptured-patches dataset write, with `batch_per_shard` set to the patch count\nand `renderer_version` pulled from the plugin's static metadata.\n\nCloses #885\nCloses #940\n\n* fix(vst): pin CLI flag style + harden round-trip + repair smoke fixture\n\nAddress PR #942 review round 1.\n\n- Pin `cli_kebab_case=False` on `_GenerateCliArgs.model_config` so a future\n  pydantic-settings minor flipping the default to kebab-case can't silently\n  desync the CLI from `build_generate_args`'s underscore output. (Copilot\n  comments on the producer + consumer sides.)\n- Add `test_build_generate_args_roundtrips_through_cli_parser`: builds args\n  with `build_generate_args`, parses them with `CliApp.run`, asserts the\n  reconstructed `RenderConfig` equals the original. Catches flag-spelling\n  and value-coercion drift the field-set parity tests miss. (Copilot\n  round-trip suggestion.)\n- Repair `tests/conftest.py::surge_xt_smoke_datasets`: the subprocess call\n  passed the old positional `num_samples` and `--param_spec`. The new\n  pydantic-settings CLI takes only `data_file` positional and the flag is\n  `--param_spec_name`, plus all other RenderConfig fields are required\n  (no model defaults). The fixture now passes every required flag. (doc-drift\n  follow-up flagging a likely VST-tier CI failure.)\n\nRefs #940\n\n* internal-fix(spec): gate unused train_val_test_seeds with NotImplementedError\n\ntrain_val_test_seeds was a required DatasetSpec field reserved for per-sample\nseeding (#884) but never consumed — yamls, fixtures, and worker payloads were\nforced to carry a dead `[42, 43, 44]` triple. Made it optional (default None)\nwith a model_validator(mode=\"before\") that raises NotImplementedError if any\nnon-None value is set, so the field can't quietly accumulate stale values\nbetween now and #884. Removed the boilerplate from configs/dataset.yaml,\nvalidate_spec's required-keys list, and all eight test fixtures that were\nplumbing the dead value through.\n\nAddresses ktinubu's self-comment on PR #942\n(https://github.com/tinaudio/synth-setter/pull/942#discussion_r3221956327).\n\nRefs #884\n\n* docs(conftest): align surge_xt_smoke_datasets docstring with new CLI flag\n\nThe docstring referenced the old `--param_spec` flag while the\nsubprocess invocation uses `--param_spec_name` (renamed in e73e0f4).",
          "timestamp": "2026-05-11T17:29:19-04:00",
          "tree_id": "f408503e7e68b78ba2dc332a2777ca998cd63abc",
          "url": "https://github.com/tinaudio/synth-setter/commit/1f6eb7a34e4320b73ed5d42fd72c6a2b86b41167"
        },
        "date": 1778535730528,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 4.48840856552124,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.452329257773235,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.03422567993402481,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.04677313566207886,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.9609272480010986,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 12.264664071666664,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.652944564819336,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.511173404343427,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.03683341667056084,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.05584162473678589,
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
          "id": "4dcb827f87e64b15a95f479416b85698acaa8ff5",
          "message": "refactor(pipeline): relocate pipeline/ → src/pipeline/ (#948)\n\nMirror src/data/, src/models/, etc. by moving the pipeline package\nunder src/. Hoist the dataset generation entrypoint to\nsrc/generate_dataset.py — the entrypoints/ subnamespace dissolves;\nskypilot_launch lives at src/pipeline/skypilot_launch.py.\n\nAll `from pipeline.*` imports rewritten to `from src.pipeline.*`;\n`pipeline.entrypoints.generate_dataset` references rewired to\n`src.generate_dataset`. Workflow YAMLs, compute YAMLs, pyproject.toml\npydoclint excludes, scripts, and doc-map.yaml updated mechanically.\nThe @hydra.main config_path on src/generate_dataset.py drops one level\n(`../configs`) since the file moved closer to repo root.\n\nRefs #882, refs #883.\nCloses #947.",
          "timestamp": "2026-05-11T18:42:50-04:00",
          "tree_id": "0abc2b31c53cf81a8aba2fde58612a36c221a61e",
          "url": "https://github.com/tinaudio/synth-setter/commit/4dcb827f87e64b15a95f479416b85698acaa8ff5"
        },
        "date": 1778540044190,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 5.257961273193359,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.47606325159315,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.05073601379990578,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.0701935887336731,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 4.053813457489014,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 10.66332305616667,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 5.257961273193359,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.655538489806349,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.05073601379990578,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.0701935887336731,
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
          "id": "94371d6751e61fc27162b18a99f53177d793a376",
          "message": "internal-fix(pipeline): code-health pass on skypilot_launch + pedalboard-free spec import (#963)\n\n* internal-fix(pipeline): code-health pass on skypilot_launch + pedalboard-free spec import\n\n- Defer sky.check import (avoids paying SkyPilot's import cost at module\n  load).\n- Extract _SECRET_WORKER_ENV_KEYS to a module-level constant.\n- Lift _launch_one_rank to module scope for testability.\n- Make src.pipeline.schemas.spec importable in pedalboard-free\n  environments (deferred param_specs import via param_spec_registry).\n- Migrate three call sites to import load_plugin / load_preset /\n  render_params directly from src.data.vst.core.\n\nRefs #882, refs #883.\nCloses #962.\n\n* internal-fix(pipeline): clarify pedalboard-free test class docstring\n\nCopilot review feedback: the original docstring blamed `tests/conftest.py`\nfor the in-session pedalboard load, but after this PR conftest only pulls\nthe pedalboard-free registry. The transitive load actually comes from\nearlier tests that import `src.data.vst.core`. Reword to match.\n\nRefs #962.\n\n* internal-fix(pipeline): tighten docstrings on registry + _SECRET_WORKER_ENV_KEYS\n\nCopilot review feedback:\n- param_spec_registry.py: the docstring still described pedalboard being\n  pulled via `src.data.vst.__init__`'s `from src.data.vst.core import ...`,\n  but `__init__` no longer imports `core` after this PR. Reword to describe\n  the registry as the canonical pedalboard-free entrypoint and call out\n  `src.data.vst.core` (not `__init__`) as the pedalboard pull point.\n- skypilot_launch.py: the comment called the residual subset \"real secrets,\"\n  but `WORKER_GIT_REF` is not a secret. Reword to describe the set by what\n  it actually is — keys not defaulted by `_R2_RCLONE_CONSTANTS`.\n\nRefs #962.",
          "timestamp": "2026-05-12T00:13:13Z",
          "tree_id": "ed0ac5099b6a7b1a776c1b632e367a3b7104bd7e",
          "url": "https://github.com/tinaudio/synth-setter/commit/94371d6751e61fc27162b18a99f53177d793a376"
        },
        "date": 1778545443221,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 4.488108158111572,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.907696331497282,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.035148248076438904,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.04487031698226929,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.29306697845459,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 11.932973464583332,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.488108158111572,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.427092907050101,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.035148248076438904,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.04487031698226929,
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
          "id": "6a4427d4aad89ad22a2a68b010defd7fb68f1c94",
          "message": "docs: convert remaining Google-style docstring sections to Sphinx (#952)\n\n* docs: convert remaining Google-style docstring sections to Sphinx\n\nThe repo's configured docstring style is Sphinx (`[tool.docformatter]` and\n`[tool.pydoclint]` both set to sphinx) and the bulk of the codebase\nalready uses `:param:` / `:return:` / `:raises:`. A handful of files in\n`src/` and `pipeline/` still had Google-style `Args:` / `Returns:` /\n`Raises:` / `Example:` section headers, showing up as DOC003 violations\nin pydoclint's audit (#938).\n\nThis converts them in place, matching the rest of the codebase:\n- `Args:` blocks → one `:param <name>: ...` line per arg\n- `Returns:` blocks → `:return: ...` (dominant form, 7 vs 2 over `:returns:`)\n- `Raises:` blocks → one `:raises <Exc>: ...` line per exception\n- `Example:` block in `src/utils/utils.py` → `.. code-block:: python` directive\n\nNo behavior changes; only docstring text. `scripts/` and `tests/` are out\nof scope per #938's chunked remediation plan.\n\nRefs #938.\n\n* docs(wandb-integration): shift line refs after src/utils/utils.py docstring conversion\n\nThe Google-→-Sphinx conversion in src/utils/utils.py shrank the\ntask_wrapper docstring by one line, shifting code below it up by one.\nTwo line-range refs in wandb-integration.md were now off by one:\n\n- task_wrapper wandb.finish() finally block: 102-107 → 101-106\n- watch_gradients source range: 138-149 → 137-148\n\nCaught by the doc-drift advisory on PR #952. Refs #938.\n\n* docs(skypilot-launch): clarify _run_workers :return: is a list\n\nThe Sphinx-style :return: introduced in the prior commit kept the\noriginal Google-style wording, which read like a scalar even though\nthe function returns list[int]. Spelled out that it's a list with one\nentry per rank, in cluster_names order, and called out the ``-1``\nsentinel and tail-mode behavior referenced elsewhere in the docstring.\n\nRefs #938.",
          "timestamp": "2026-05-11T21:55:39-04:00",
          "tree_id": "5eb292e7917966ef9dd341cb37fe3f84722833df",
          "url": "https://github.com/tinaudio/synth-setter/commit/6a4427d4aad89ad22a2a68b010defd7fb68f1c94"
        },
        "date": 1778551703106,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.5155773162841797,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.034173527186503,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.023132724687457085,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.023236572742462158,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.029608964920044,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 10.679147733249998,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.3425750732421875,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 7.035165253970772,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.03191940486431122,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.041259825229644775,
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
          "id": "dbb469ece61306fa036351a16a27178e9bb71628",
          "message": "refactor(layout)!: nest src/* under synth_setter/, declare console scripts (#991)\n\n* refactor(layout)!: nest src/* under synth_setter/, declare console scripts\n\nPhase 2 of the PEP src-layout migration (#989, parent #784).\n\nHoists `src/{data,models,utils,metrics.py}` to `src/synth_setter/` and\n`src/{train,eval,generate_dataset}.py` to `src/synth_setter/cli/`. Adds\nthe three `synth-setter-{train,eval,generate-dataset}` console scripts\nvia `[project.scripts]`. Sweeps all `from src.X` imports, `_target_:\nsrc.X` Hydra refs in configs, `python src/X.py` shell invocations in\njobs/ and sweeps/, and prose references in active docs.\n\nOut of scope: `src/pipeline/` (Phase 3, #784), `scripts/`\ndepopulation (Phase 4), `setup.py` deletion (Phase 5); only the\nlegacy `train_command` / `eval_command` `console_scripts` entries\nare dropped here.\n\nBreaking: any external consumer importing `src.{data,models,utils,\nmetrics,train,eval,generate_dataset}` must rewrite to\n`synth_setter.{...}` / `synth_setter.cli.{...}`. Legacy\n`train_command` / `eval_command` scripts are removed (replaced by\n`synth-setter-train` / `synth-setter-eval`).\n\n* test(baseline-configs): bump MODEL_BASELINE to Phase 2 src-layout SHA\n\nThe Phase 2 migration (#989) rewrote every Hydra `_target_:` from `src.X`\nto `synth_setter.X` and switched `jobs/train/{kosc,surge}/train.sh` from\n`python src/train.py` to `python -m synth_setter.cli.train`. The\nresolved Hydra YAMLs from the v0.0.0 baseline therefore literally\ncontain `_target_: src.X` keys while the live tree's resolved YAMLs\ncontain `_target_: synth_setter.X` — a 44-case failure (KOSC) plus a\nparallel SURGE failure pinned at the old tag.\n\nBumping MODEL_BASELINE to the Phase 2 commit captures the migration as\nthe new known-good model-config snapshot. FIXTURE_BASELINE is\nuntouched: the synthetic-fixture scripts under `tests/fixtures/` are\nself-contained and don't reference `src.X`.\n\nRefs #989.\n\n* fix(tests): set PYTHONPATH=src in CI subprocess + workflow probes\n\nThe Phase 2 migration's lazy import inside `DatasetSpec.num_params`\nswitched from `from src.data.vst.param_spec_registry` to\n`from synth_setter.data.vst.param_spec_registry`. That import is\ntriggered by `model_dump_json()` and is exercised by:\n\n  * `tests/pipeline/test_schemas/test_dataset_spec.py` —\n    two tests spawn fresh `sys.executable` subprocesses to verify the\n    spec stays pedalboard-free / launcher-pure. The subprocesses\n    don't inherit pytest's `pythonpath = [\"src\"]`, so\n    `synth_setter` isn't reachable without an editable install.\n    Fixed by passing `PYTHONPATH=<repo>:<repo>/src` to the subprocess\n    `env`.\n\n  * `.github/workflows/test-{mps,gpu,vst-slow}.yml` — the Surge XT\n    plugin-load smoke checks `python -c \"from synth_setter.data.vst.core\n    import load_plugin...\"` against a fresh interpreter (macOS host and\n    Docker container). Fixed by adding `src/` to the PYTHONPATH env\n    var the workflow already exports.\n\nBoth `make test-fast` (556/5) and the full slow `test_compare_baseline_configs`\nsuite (87 passed in 11m17s, including all 44 KOSC + 8 SURGE + 18 predict\ncases) pass locally against the bumped `MODEL_BASELINE=4e08950`.\n\nRefs #989.\n\n* docs(design): update stale src/* refs to synth_setter/* (Phase 2)\n\nSeven design docs (training-pipeline, eval-pipeline, data-pipeline,\nskypilot-compute-integration, storage-provenance-spec, plus the two\n*-implementation-plan docs) referenced legacy `src/train.py`,\n`src/eval.py`, `src/data/`, `src/utils/`, `src/models/` paths and a\n`_target_: src.X` YAML example that no longer resolve after the\nPhase 2 src-layout move.\n\nRewrote file paths to `src/synth_setter/cli/{train,eval}.py` and\n`src/synth_setter/{data,utils,models}/`; rewrote `_target_:` to\n`synth_setter.X`; rewrote `python src/train.py …` CLI invocations\nin code blocks to `python -m synth_setter.cli.train …` per the new\ncanonical surface.\n\nSurfaced by the doc-drift advisory on PR #991.\n\nRefs #989.\n\n* fix(tests): pass PYTHONPATH to VST subprocess in conftest fixture\n\nThe macOS MPS CI workflow does not run `pip install -e .` before\npytest, so the in-process `pythonpath = [\"src\"]` from pyproject.toml\ndoesn't propagate to subprocess.run children. The `surge_xt_smoke_datasets`\nfixture spawns `python src/synth_setter/data/vst/generate_vst_dataset.py`,\nwhich fails with `ModuleNotFoundError: No module named 'synth_setter'`\nwhen its `from synth_setter.data.vst import param_specs` import runs in\nthe child interpreter.\n\nMirrors the `_subprocess_env()` helper already in\n`tests/pipeline/test_schemas/test_dataset_spec.py` (added in b7c62c0):\nset PYTHONPATH=<repo>:<repo>/src on the child env so it can resolve\nboth `src.pipeline.*` and `synth_setter.*` without an install step.\n\nRefs #989.\n\n* fix(ci): install editable package in test workflows, drop PYTHONPATH workaround\n\nThe proper fix for \"subprocesses spawned from tests can't import\nsynth_setter\": install the package via `pip install -e .` in each\nworkflow's setup. Once installed, the import resolves naturally — no\nduplicated `_subprocess_env()` helper, no PYTHONPATH gymnastics.\n\nWorkflows updated: test.yml (3 jobs), test-mps.yml, test-conda.yml.\nEach now installs `synth_setter` as editable after the dependency\ninstall. test-mps.yml's \"Smoke-test Surge XT plugin load\" step drops\nits `PYTHONPATH: src` env which b7c62c0 added as a workaround — also\nno longer needed.\n\nThe `_subprocess_env()` helper in tests/conftest.py and\ntests/pipeline/test_schemas/test_dataset_spec.py is removed entirely.\nThat duplication was a code smell flagged by /repo-review-full as\nBLOCK; the real problem was the missing install step.\n\nAddresses BLOCK findings from review #4276527174:\n  - [code-health] _subprocess_env duplicated across two test files\n  - [gha] test-mps.yml Run MPS tests has no install / no PYTHONPATH\n\nRefs #989.\n\n* chore(review): address PR #991 review feedback round 1\n\n- src/synth_setter/cli/generate_dataset.py: add TODO(#784) above the\n  legacy `src.pipeline.*` import block flagging Phase 3 collapse.\n- tests/test_compare_baseline_configs.py: tighten the MODEL_BASELINE\n  prose to a 2-line pointer to #989 and correct the misleading\n  \"head of the Phase 2 PR\" wording — the SHA is the initial commit\n  of #989, not the head.\n- tests/pipeline/test_entrypoints/test_generate_dataset.py: switch\n  module-docstring header from file path to module form so it\n  doesn't drift if the file moves.\n\nRefs #989\n\n* fix(ci): install synth_setter editable in launcher workflows\n\nPhase 2 src-layout migration moved `synth_setter` from `src/`-on-PYTHONPATH\nto a properly-installed package. Two launcher workflows still invoke\n`python -m src.pipeline.skypilot_launch` (which imports\n`synth_setter.cli.generate_dataset` at module load) without installing the\npackage first, so they hit `ModuleNotFoundError: No module named 'synth_setter'`\nat src/pipeline/skypilot_launch.py:51.\n\nSame fix shape as 64ac16d (test.yml / test-mps.yml / test-conda.yml): add\n`pip install -e .` after the requirements install.\n\n- generate-dataset-shards.yaml: skypilot-local row's \"Install launcher deps\"\n  step. Fixes the PR-blocking `Test Dataset Generation /\n  Run generate_dataset (skypilot-local)` failure on #991.\n- test-skypilot-debug.yml: launcher-runner mode's \"Install launcher deps\"\n  step. workflow_dispatch only, same root cause.\n\nIn-container invocations (runpod / oci rows; launcher-docker mode) don't\nneed a change — the dev-snapshot Dockerfile already does\n`uv pip install --no-deps -e .` at build time.\n\ntest-skypilot-local.yml uses sky.launch directly with no synth_setter\nimports — no fix needed there.\n\nRefs #989\n\n* chore(review): address PR #991 review feedback round 2\n\n- src/synth_setter/cli/{train,eval,generate_dataset}.py: collapse the\n  copy-pasted 15-line rootutils explanatory block (and the variant in\n  generate_dataset.py) to a single one-liner pointing at the rootutils\n  README, per CLAUDE.md comment-hygiene (\"Keep comments terse — typically\n  one short line\").\n- src/synth_setter/cli/eval.py: add a one-line comment on the\n  `mode == \"val\" or mode == \"validate\"` branch documenting that both\n  spellings are accepted for backwards compatibility with older configs.\n- pyproject.toml: drop alignment whitespace on the three\n  `[project.scripts]` entries to match standard TOML formatting.\n- src/__init__.py: rephrase the docstring so it acknowledges that\n  src/pipeline/ is still part of the codebase, not just legacy residue.\n\nRefs #989",
          "timestamp": "2026-05-13T08:22:59-04:00",
          "tree_id": "1467e6d7bbbd20f15b0ad2bf1f5bc45a96b05449",
          "url": "https://github.com/tinaudio/synth-setter/commit/dbb469ece61306fa036351a16a27178e9bb71628"
        },
        "date": 1778675697176,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.2570199966430664,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.919081733282655,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.019176138564944267,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.01666039228439331,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.1069929599761963,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 10.808631060916673,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.1365509033203125,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.699236087696627,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.027401061728596687,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.03004610538482666,
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
          "id": "7d8a43877a722e382e76787f28f36e987917c420",
          "message": "refactor(layout)!: nest src/pipeline/ under synth_setter/, drop legacy src/ package (#1001)\n\n* refactor(layout)!: nest src/pipeline/ under synth_setter/, drop legacy src/ package\n\nMove `src/pipeline/` to `src/synth_setter/pipeline/` and remove the residual\n`src/__init__.py` so `src/` now contains only the `synth_setter` package.\n\nSweeps:\n- Python imports: `from src.pipeline.` -> `from synth_setter.pipeline.` across\n  tests/, scripts/, and the self-reference in materialize_spec.py's docstring.\n- YAML / Dockerfile / docs: `python -m src.pipeline.` ->\n  `python -m synth_setter.pipeline.` across .github/workflows, configs/compute,\n  configs/image, and docs/.\n- pyproject.toml [tool.setuptools].packages: drop `src`, `src.pipeline`,\n  `src.pipeline.ci`, `src.pipeline.schemas`; add `synth_setter.pipeline`,\n  `synth_setter.pipeline.ci`, `synth_setter.pipeline.schemas`. Delete the\n  Phase 2 transition comment that explained the dual registration.\n- CLAUDE.md Architecture section: collapse the separate `src/pipeline/` bullet\n  into a sub-bullet under `src/synth_setter/`.\n- tests/conftest.py: update the `RenderConfig` reference comment.\n\n`MODEL_BASELINE` in tests/test_compare_baseline_configs.py is intentionally\nunchanged — the project's stable-baseline-anchor rule applies and the\nresolved-YAML grep confirmed no `_target_: src.pipeline` rewrites surface\nunder the move.\n\nThe `!` flags this as breaking because `import src.pipeline` now raises\n`ModuleNotFoundError`. All in-tree callers have been migrated.\n\nCloses #995\nRefs #784\n\n* fix(ci,docs): pip install -e . in docker-build-validation; address Copilot review\n\nPhase 3 (#995) rewrote `.github/workflows/docker-build-validation.yml` to\ninvoke `python -m synth_setter.pipeline.ci.load_image_config`, but the\nworkflow's setup only installed `pyyaml pydantic` — the `synth_setter`\npackage itself was never installed on the runner. Pre-Phase-3 the call\nworked because `python -m src.pipeline.ci.load_image_config` resolved\nagainst the cwd-relative `src/` directory; post-Phase-3 the principled fix\nis `pip install -e .`, matching the pattern Phase 2 (#991) established\nfor every other CI workflow that spawns Python expecting `synth_setter`.\n\nAlso addresses three inline review findings from Copilot on PR #1001:\n\n- src/synth_setter/cli/generate_dataset.py: drop the `TODO(#784):\n  collapse to synth_setter.pipeline.* once Phase 3 hoists ...` comment;\n  the imports below are already on `synth_setter.pipeline.*` after this\n  PR, so the TODO is satisfied.\n- docs/doc-map.yaml: correct the `covers:` description for\n  `src/synth_setter/pipeline/constants.py` — the module defines only\n  `INPUT_SPEC_FILENAME`, no R2 bucket name constant.\n- docs/design/data-pipeline-implementation-plan.md: repoint the\n  `make_dataset` import example from the non-existent\n  `synth_setter.pipeline.vst` to the actual current location,\n  `synth_setter.data.vst.generate_vst_dataset`.\n\nRefs #995\nRefs #784\n\n* fix(ci): also install pyyaml + pydantic for docker-build-validation\n\nThe previous fix (`pip install -e .`) makes `synth_setter` importable but\ndoesn't pull in `pyyaml` or `pydantic` because neither is a declared\nruntime dependency in `pyproject.toml`. The Phase 3 sweep replaced the\nprior bare `pyyaml pydantic` install with `pip install -e .` alone, which\nre-broke the same step on a different `ModuleNotFoundError`. Pin both\nexplicitly alongside the editable install so the step has a self-contained\nenvironment for `python -m synth_setter.pipeline.ci.load_image_config`.\n\nRefs #995\nRefs #784\n\n* fix(ci,docs): install synth_setter for validate-dataset-shards; fix duplicate stale vst import\n\nTwo follow-up findings from Copilot's review of b9dd27d:\n\n1. .github/workflows/validate-dataset-shards.yaml — the validate-spec\n   job runs `python3 -m synth_setter.pipeline.ci.validate_spec` but its\n   install step only installed pydantic. Same regression as\n   docker-build-validation.yml in b9dd27d. Use `pip install --no-deps -e .`\n   alongside the explicit `pydantic>=2,<3` pin so the runner-side env\n   stays minimal (no torch) but synth_setter is importable. The comment\n   block above the step (which explains why this is a minimal install)\n   is preserved verbatim — the rationale still holds.\n\n2. docs/design/data-pipeline-implementation-plan.md L931 — the\n   \"Assumptions\" section had a second stale reference to the\n   non-existent `synth_setter.pipeline.vst.make_dataset` module that\n   c669164 only fixed at L562. Repointed to the actual current\n   location, `synth_setter.data.vst.generate_vst_dataset.make_dataset`,\n   matching the L562 fix.\n\nRefs #995\nRefs #784\n\n* fix(ci): install synth_setter for spec-materialization host-side validate\n\nThe host-side `Validate spec structure` step in spec-materialization.yml and\nthe `Assert test-specific values` step in test-spec-materialization.yml both\nrun `python3 -m synth_setter.pipeline.ci.validate_spec` outside the docker\ncontainer. Phase 3 made `synth_setter` only importable when installed (PEP\nsrc-layout, sources under src/), so both invocations would raise\nModuleNotFoundError on a fresh runner.\n\nSame fix pattern as c669164 / b9dd27d9 / b0c9cfd: add setup-python +\n`pip install --no-deps -e . \"pydantic>=2,<3\"` before the python invocation.\n--no-deps keeps the host env minimal (torch stays in the image).\n\nAddresses Copilot's suppressed low-confidence comment from review\n4282446908 on .github/workflows/test-spec-materialization.yml:35.\n\nRefs #995\n\n* fix(ci): drop --no-deps from host-side validate_spec installs\n\nThe `pip install --no-deps -e . \"pydantic>=2,<3\"` install pattern used by\nthe three host-side `Validate spec structure` / equivalent steps had a\nsubtle bug: `--no-deps` applies to *every* package in the pip command,\nnot just the editable install. As a result pydantic gets installed but\nits required `pydantic-core` (a separately-shipped C extension) does\nnot. The act-verify CI job caught this on PR #1001 with:\n\n    Successfully installed pydantic-2.13.4 synth-setter-3.0.0\n    ...\n    ModuleNotFoundError: No module named 'pydantic_core'\n\n`--no-deps` was originally added to keep the host env minimal (no torch).\nThe minimal-install goal is already met by `[project].dependencies = []`\nin pyproject.toml — the editable install adds nothing transitively for\nsynth_setter itself. Dropping `--no-deps` lets pydantic pull in its\nrequired pydantic-core, while torch still stays out of the env.\n\nAffects three workflows (each running `python3 -m\nsynth_setter.pipeline.ci.validate_spec` on a host runner, not in the\ndocker image):\n\n- .github/workflows/validate-dataset-shards.yaml\n- .github/workflows/spec-materialization.yml\n- .github/workflows/test-spec-materialization.yml\n\nComment blocks above each install step are updated to explain the\nnon-obvious interaction between `--no-deps` and pydantic's own\ndependency on pydantic-core, so the next refactor doesn't reintroduce\nthe flag.\n\nRefs #995\nRefs #784",
          "timestamp": "2026-05-13T15:20:28Z",
          "tree_id": "01a20ad39e9ec59ede0933d1188b34cee53d1769",
          "url": "https://github.com/tinaudio/synth-setter/commit/7d8a43877a722e382e76787f28f36e987917c420"
        },
        "date": 1778686443296,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.347397804260254,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.499598183147609,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.02188733033835888,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.018500566482543945,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.1008052825927734,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 12.068248456083333,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.602628231048584,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.7907643022947015,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.030180584639310837,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.05231422185897827,
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
          "id": "ebd0dfa6c5dc4a7c9b33c75858a1a980b6360cf2",
          "message": "chore(deps): consolidate requirements*.txt into pyproject.toml extras (#1008)\n\n* chore(deps): consolidate requirements*.txt into pyproject.toml extras\n\nMove requirements-torch.txt to [project.optional-dependencies].torch,\nrequirements-app.txt's runtime deps to [project.dependencies], and the\ndev tools (pytest, ruff, pre-commit, pyright, mutmut, pytest-benchmark,\npytest-xdist, hypothesis) to [project.optional-dependencies].dev. Add a\nconvenience [all] extra = [torch,dev]. Every pin (loguru==0.7.3,\nscipy==1.14.1, mutmut==3.5.*, pyright==1.1.408,\nskypilot[runpod,oci]==0.12.0, runpod==1.8.1, click<8.2, pesto-pitch,\ndtw-python, kymatio) is preserved verbatim.\n\nReplace the three requirements*.txt files with their pyproject equivalents\nacross all consumers: Makefile install, docker/ubuntu22_04/Dockerfile\n(now uv pip compile pyproject.toml --extra torch --extra dev so the\n~2.5 GB torch-wheels layer keeps surviving source edits — cache key\nnarrows to pyproject.toml + README.md), .devcontainer/Dockerfile,\nenvironment.yaml (.[dev]), scripts/sync_worker_checkout.sh, and every\nGitHub Actions workflow under .github/workflows/.\n\nTwo workflows that previously installed only pydantic on top of\npip install -e . (when [project.dependencies] was empty) — namely\ntest-spec-materialization.yml and validate-dataset-shards.yaml — switch\nto pip install --no-deps -e . + pip install pydantic so they continue\nto avoid pulling torch, librosa, skypilot, etc.\n\nVerified: uv pip compile --extra torch --extra dev resolves with every\npin honored; editable install dry-run produces the same direct-dep set.\nmake format passes (one pre-existing pyright failure on\ntests/pipeline/test_entrypoints/test_skypilot_launch.py is unrelated).\n\nCloses #533\nCloses #181\n\n* chore(deps): also update tart/macos.pkr.hcl install line\n\nDoc-drift review on PR #1008 caught a missed reference: the Packer\ntemplate's \"Clone the repo, use venv with all runtime deps\" provisioner\nstill ran `uv pip install -r requirements.txt && uv pip install --no-deps\n-e .`, which would break the next Tart image build now that\nrequirements.txt is gone.\n\nCollapse the two lines into the equivalent\n`uv pip install --torch-backend ${var.torch_backend} -e \".[torch,dev]\"`\nso the macOS VM ends up with the same dep set as before (torch backend\nhonored via uv's --torch-backend; project installed editably with the\ntorch and dev extras). docs/getting-started.md already advertises this\nbehavior — this brings the build script in line with the doc.\n\nRefs #533\n\n* docs(getting-started): clarify hydra-core lives in runtime deps, not torch extra\n\nCopilot review on #1008 flagged that the conda parenthetical claimed\nhydra-core ships in the `torch` extra. It actually lives in\n`[project.dependencies]`; the `torch` extra is just torch /\ntorchvision / torchaudio / lightning / torchmetrics. Reword to describe\nboth groups as the pip-only set the conda flow installs.\n\nRefs #533\n\n* ci: switch remaining minimal-install workflows to --no-deps\n\nNow that [project.dependencies] is populated, `pip install -e .`\n(and `uv pip install --system -e .`) drags in the full runtime\ndep set. Switch the two remaining minimal-install workflows to\nthe same `--no-deps` + explicit-deps pattern already used by\ntest-spec-materialization.yml and validate-dataset-shards.yaml:\n\n- spec-materialization.yml: `pip install -e . \"pydantic>=2,<3\"`\n  → `pip install --no-deps -e .` + `pip install \"pydantic>=2,<3\"`.\n  Update the inline rationale comment (it claimed `--no-deps`\n  would skip pydantic-core, which is no longer the reason — the\n  reason is now that the project's runtime deps are heavy).\n- docker-build-validation.yml: `uv pip install --system -e .\n  pyyaml pydantic` → `uv pip install --system --no-deps -e .`\n  + `uv pip install --system pyyaml pydantic`.\n\nRefs #533\n\n* chore(deps): collapse Docker uv pip compile+install into one pass\n\nCI Build-and-push failure on PR #1008 root-caused: the two-step\n`uv pip compile pyproject.toml --extra torch --extra dev → uv pip install\n--torch-backend ${TORCH_BACKEND} -r /tmp/requirements.lock` flow resolved\ntorch against the PyPI index in the compile step (no --torch-backend\nthere), pinning torch==2.12.0 (PyPI). The install step then asked the\ncu128 index for that exact version and got \"No solution found\" because\nthe cu128 index ships CUDA-tagged builds (e.g. 2.7.0+cu128), not the bare\n2.12.0 PyPI version.\n\nDrop the compile indirection entirely and use uv's direct support for\nreading deps out of pyproject.toml: `uv pip install -r pyproject.toml\n--extra torch --extra dev`. This resolves and installs in one pass\nagainst the cu128 index, matching the original requirements.txt flow,\nand removes the cross-index inconsistency.\n\nRefs #533\n\n* chore(deps): keep transitional requirements.txt stub for dev-snapshot bake lag\n\nCI Run-generate_dataset failure on PR #1008 root-cause: the\nskypilot-local worker runs `bash scripts/sync_worker_checkout.sh` from\nthe published `dev-snapshot` image, which was baked from main BEFORE\nthis PR. Bash buffers the script at open-time, so even though\n`git checkout WORKER_GIT_REF` succeeds and rewrites the worker's\nworking tree to this PR's HEAD (deleting requirements.txt in the\nprocess), the bash process is mid-execution of the OLD baked script\nlines. The next line in the old script is `uv pip install -r\nrequirements.txt`, which now errors with `File not found`.\n\nKeep a one-line requirements.txt stub that resolves to `-e .[torch]`\nso the OLD baked script's install still works. Once the dev-snapshot\nimage is rebuilt from main after this PR merges (the next push to\nmain triggers it), the baked script will be the updated one that\ndoes `uv pip install -e \".[torch]\"` directly — at which point this\nfile can be deleted. The stub has a sunset comment naming the\ndeletion criterion.\n\nRefs #533\n\n---------\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T14:21:01-04:00",
          "tree_id": "7321193c266ae8201b5ebe7d3c3e564c99a2dbd5",
          "url": "https://github.com/tinaudio/synth-setter/commit/ebd0dfa6c5dc4a7c9b33c75858a1a980b6360cf2"
        },
        "date": 1778697267928,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.4880497455596924,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.72107674703002,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.018643373623490334,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.012156248092651367,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.086259603500366,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 11.074583121166663,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.232139587402344,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.701745314151049,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.02903391420841217,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.03422945737838745,
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
          "id": "156164a30795b1ac89baad85ec3bec9ae911b911",
          "message": "internal-feat(schemas): add ShardMetadata + wds row in OUTPUT_FORMAT_TO_EXTENSION (#976)\n\n* internal-feat(schemas): add ShardMetadata + wds row in OUTPUT_FORMAT_TO_EXTENSION\n\nPlates the schema layer for the wds writer landing in PR-13:\n\n- New leaf module src/pipeline/schemas/shard_metadata.py holds the strict,\n  frozen ShardMetadata model (sidecar JSON for the wds tar's metadata.json\n  member). No project imports — consumers on either side of the src ↔\n  src/pipeline boundary can pick it up without forming a launcher-side\n  import cycle through pedalboard.\n- Extend OUTPUT_FORMAT_TO_EXTENSION from {\"hdf5\": \".h5\"} to\n  {\"hdf5\": \".h5\", \"wds\": \".tar\"} and widen DatasetSpec.output_format from\n  Literal[\"hdf5\"] to Literal[\"hdf5\", \"wds\"]. The existing\n  _shard_filenames_match_output_format model_validator now defends both\n  formats.\n\nJoins the existing schemas in the pydoclint exclude list (alongside\nspec.py, prefix.py, image_config.py) per the convention documented at\nthe exclude block — see #938 for the cleanup-as-we-go epic.\n\nInternal-only — no config / launcher / worker changes. PR-13 splits the\nwriter, PR-14 wires --wds-out end-to-end (closes #874).\n\nRefs #975\nPart of #72\n\n* docs(design): sync data-pipeline doc with ShardMetadata + Literal[\"hdf5\", \"wds\"]\n\nApply doc-drift findings from PR #976 review:\n\n- §14.1 spec sketch — drop the \"wds in a later PR\" trailer and widen the\n  Literal to match spec.py's new Literal[\"hdf5\", \"wds\"].\n- §14.7 directory tree — list the new shard_metadata.py leaf module.\n- §7.6 finalize step + §8 WDS shard structure — reference the metadata.json\n  sidecar (one per shard) and point readers at the ShardMetadata model.\n- doc-map.yaml — map src/pipeline/schemas/shard_metadata.py to the\n  data-pipeline design doc so future drift checks catch evolution of the\n  sidecar contract.\n\nRefs #975\nPart of #72\n\n* internal-fix(schemas): tighten ShardMetadata sample_rate type + AST-based leaf-import test\n\nAddresses Copilot review on PR #976:\n\n- ShardMetadata.sample_rate: float → int. The h5py audio attr is written\n  from RenderConfig.sample_rate (int), so the wds sidecar mirrors the\n  canonical type now rather than drifting at the format boundary. Test\n  payloads updated to match.\n- The leaf-module test now parses the module's AST and asserts no\n  ImportFrom/Import nodes targeting src.* — replaces the substring grep,\n  which would have false-failed on a docstring mentioning \"from src.\" and\n  missed alternative import phrasings.\n\nRefs #975\n\n* ci: re-trigger test-dataset-generation after transient VST X-server flake\n\n* internal-fix(schemas): clarify ShardMetadata is not yet read by validate_shard; UTF-8 source read in leaf-import test\n\nAddresses Copilot round 2 on PR #976:\n\n- ShardMetadata docstring + doc-map covers entry no longer claim the sidecar\n  is \"validated on read by validate_shard\". The wds writer and the wds branch\n  of validate_shard land in PR-13; PR-12 only plates the model. Reworded to\n  reflect current behavior (model exists, wiring in PR-13).\n- test_module_has_no_project_imports now uses Path(...).read_text(encoding=\"utf-8\")\n  instead of bare open().read(); shard_metadata.py contains the non-ASCII ↔\n  glyph, so a non-UTF-8 default locale (Windows) would have errored.\n\nRefs #975\n\n* test(pipeline): widen leaf-import check to flag all project-import forms\n\nAddresses Copilot round 3 on PR #976:\n\nThe AST check previously only caught ``import src.x`` and ``from src.x\nimport y``. Bare ``import src``, ``from src import x``, and any relative\n``from .x import y`` would have bypassed it. Now flags:\n\n- ast.Import: alias.name == \"src\" OR alias.name.startswith(\"src.\")\n- ast.ImportFrom: node.level > 0 (any relative import) OR node.module\n  starts with \"src.\" OR node.module == \"src\"\n\nThe wider check enforces the actual contract (no project-internal imports\nthat would form a launcher-side cycle), not a substring of one shape.\n\nRefs #975\n\n* internal-fix(schemas): add range validators on ShardMetadata + clarify leaf-test docstring\n\nAddresses Copilot round 4 on PR #976:\n\n- ShardMetadata now runs a _ranges_must_be_sane model_validator that mirrors\n  RenderConfig._ranges_must_be_sane: velocity ∈ [0, 127], sample_rate > 0,\n  channels >= 1, signal_duration_seconds > 0. The JSON-from-R2 path is a\n  trust boundary, so this catches corrupted/hand-edited sidecars at read\n  time rather than letting nonsensical values reach training. Tests pin\n  each rejection.\n- The leaf-import test docstring no longer claims generate_vst_dataset\n  imports ShardMetadata — that wiring lands in PR-13. Reworded to refer\n  to the future consumer.\n- Add tests/pipeline/test_schemas/test_shard_metadata.py to the pydoclint\n  exclude — its parametrized tests trip DOC101/DOC103 just like the\n  sibling test_dataset_spec.py / test_image_config.py / test_prefix.py\n  (which are all already excluded for the same reason).\n\nRefs #975\n\n* docs(design): clarify staged shards stay HDF5 regardless of output_format\n\nAddresses Copilot round 5 on PR #976. §7.6 hardcodes `.h5 + .valid` for the\nstaged-shard existence check (step 03), the structural-check open (step 04),\nand the promote copy (step 05). Now that `DatasetSpec.output_format` accepts\n`wds`, a casual reader might expect staging to flip to `.tar` for wds specs —\nbut it doesn't: workers always emit HDF5; only finalize's step 08 diverges per\nformat (transcoding to wds on demand). The rationale lives in §8's \"Why\ngeneration stays HDF5 regardless of output format\" but wasn't cross-referenced\nfrom §7.6.\n\nAdds a one-line clarifier at the top of §7.6 pointing readers to the §8 note,\nso the staging-stays-HDF5 contract is explicit without requiring the reader\nto find the other section.\n\nRefs #975\n\n* docs(design): revert §7.6 staged-HDF5 clarifier — conflicts with schema contract\n\nAddresses Copilot round 6 on PR #976. The clarifier added in 027a2df read:\n\"Staged shards are always HDF5 regardless of spec.output_format\". That's\ninternally inconsistent with PR-12's schema, where DatasetSpec.shards\nderives the shard filename from output_format via OUTPUT_FORMAT_TO_EXTENSION\n(wds → .tar). Reverting the clarifier keeps §7.6 matching the only working\ngeneration path today (hdf5); PR-13 will rewrite §7.6 + §8's \"Why generation\nstays HDF5\" section when the wds writer + extension dispatch land.\n\nRefs #975\n\n* docs(design): note §8 design-transition for wds — schema admits wds, writer lands PR-13\n\nAddresses Copilot round 7 on PR #976. Copilot rightly flagged that §8's \"Why\ngeneration stays HDF5 regardless of output format\" claim is inconsistent\nwith the schema's output_format → shard.filename wiring after PR-12. The\ntruth is the design IS changing across PR-12/PR-13: PR-12 widens the spec;\nPR-13 lands the wds writer + extension dispatch and will rewrite §8 to\nmatch the new pipeline shape.\n\nAdds a forward-looking note under §8's \"Why generation stays HDF5\" header\nthat:\n- points readers at §14.1's OUTPUT_FORMAT_TO_EXTENSION mapping,\n- states the eventual behavior (wds workers emit .tar directly),\n- says PR-13 lands the writer + section rewrite,\n- makes clear that on main today the schema admits wds but no writer is\n  wired.\n\nRefs #975\n\n---------\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
          "timestamp": "2026-05-13T16:00:22-04:00",
          "tree_id": "dcc6c23e6dcfb5f6652ca4ee848339819679c7af",
          "url": "https://github.com/tinaudio/synth-setter/commit/156164a30795b1ac89baad85ec3bec9ae911b911"
        },
        "date": 1778703207917,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.793736696243286,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.476662690639496,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.024236507713794708,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.030186951160430908,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.2639451026916504,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 12.20339379275,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.5194292068481445,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.677097791992128,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.03246081620454788,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.04671293497085571,
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
          "id": "ad5c853689e93005b8111b8ce96111725e2bb7ba",
          "message": "ci(docker): strip runtime PYTHONPATH from docker runs in workflows (#1017)\n\nPR #647 / #667 fixed setup.py so find_packages exposes the pipeline\npackage without a runtime PYTHONPATH override, and PR #797 wired\ndev-snapshot to rebuild on every push-to-main (merged 2026-05-04) so\nthe in-image package surface tracks main. The temporary\n-e PYTHONPATH=/home/build/synth-setter override added in 3529fae is no\nlonger needed; strip it from all docker run invocations.\n\nRemoves 15 -e PYTHONPATH=... lines across 10 workflow files:\n\n- .github/workflows/docker-build-validation.yml\n- .github/workflows/flush-investigation.yml\n- .github/workflows/generate-dataset-shards.yaml\n- .github/workflows/job-queue.yaml\n- .github/workflows/spec-materialization.yml\n- .github/workflows/test-dataset-generation.yml\n- .github/workflows/test-gpu.yml\n- .github/workflows/test-skypilot-debug.yml\n- .github/workflows/test-vst-slow.yml\n- .github/workflows/validate-dataset-shards.yaml\n\nThe integration check is the smoke tests in docker-build-validation.yml\nand test-dataset-generation.yml; a local docker probe was not run\nbecause docker was not available in the working environment.\n\nCloses #670\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
          "timestamp": "2026-05-13T20:31:26Z",
          "tree_id": "3c8bc483552ff70d508d62080e2457e3ddb06999",
          "url": "https://github.com/tinaudio/synth-setter/commit/ad5c853689e93005b8111b8ce96111725e2bb7ba"
        },
        "date": 1778705084439,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 4.175958633422852,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.949400022663176,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.030557164922356606,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.037357985973358154,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.3612630367279053,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 11.793521742750002,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.615119934082031,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.5907240908360105,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.03667474910616875,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.050898730754852295,
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
          "id": "e2c780ae3677279f07c832d6f4b8f90faea54558",
          "message": "chore(lint): close P1/P5/P6 pydoclint blind spots (#978)\n\n* chore(lint): close P1/P5/P6 pydoclint blind spots from adversarial probe\n\nThree concrete fixes for the slip categories the #939 adversarial probe\ndocumented. None of P2/P3/P4 are addressed here — those are inherent to\npydoclint and remain Open Questions on #938.\n\nP5 — flip pydoclint native-mode-noqa-location to \"definition\".\nThe CLI default is \"docstring\", so suppressions on the def line were\nsilently inert. Aligning with flake8/ruff convention means\n`# noqa: DOCxxx` next to `def` now does what every developer in this\nstack expects.\n\nP1 — add ruff D102/D103/D107 (missing-docstring rules).\nPydoclint defers \"must have a docstring\" to pydocstyle, which was not\nwired in. Ruff implements the same family. D102 = public method, D103 =\npublic function, D107 = __init__. Per-file-ignores cover the 40 tracked\nfiles that fail today; the list mirrors [tool.pydoclint].exclude.\n\nP6 — CI guard against new defs/classes in pydoclint-excluded files.\nscripts/check_no_new_funcs_in_pydoclint_excluded.py reads the pydoclint\nexclude regex from pyproject.toml and scans the PR diff for `+def`/\n`+class` lines whose file matches it. New tests pin its behaviour on\nsynthetic diffs; wired into code-quality-pr.yaml as a new step.\n\nCONTRIBUTING.md and .github/agents/lint-cleanup.md updated to describe\nthe new ruff D rules, the def-line noqa convention, and the guard.\n\nRefs #938\nRefs #939\n\n* docs(pydoclint): address doc-drift after P1/P5/P6 fixes\n\n- CONTRIBUTING.md: restore ANN001 to the ruff rule list it had been\n  dropped from when D102/D103/D107 were added.\n- CLAUDE.md: replace the inlined ruff rule list with a pointer to\n  [tool.ruff.lint].select so the drift clock does not reset on the next\n  rule addition; mention the new D rules.\n- docs/reference/github-actions.md: code-quality-pr now also runs the\n  pydoclint-excluded-file ratchet; document the new responsibility and\n  the fetch-depth: 0 requirement that goes with it.\n\n* chore(lint): address Copilot review on PR #978\n\n- Remove tests/scripts/test_check_no_new_funcs_in_pydoclint_excluded.py\n  from [tool.pydoclint].exclude and add `# noqa: DOC101,DOC103` to the\n  four test defs that take pytest fixtures. The previous setup made the\n  PR self-fail its own new P6 guard (verified: guard exit=1 against\n  origin/main, 12 findings before this commit; exit=0 after).\n  (comment #3223490913)\n\n- Replace `scripts/**` and `src/data/**` directory globs in\n  [tool.ruff.lint.per-file-ignores] with per-file entries mirroring\n  [tool.pydoclint].exclude. New files under those directories are no\n  longer silently exempt from D102/D103/D107.\n  (comment #3223490899)\n\n- Skip `\\\\ No newline at end of file` diff metadata in the guard's line\n  counter; add a pinning test. Without this, post-marker line numbers\n  in the guard's `path:line: name` report could be off-by-one.\n  (comment #3223490926)\n\n- Add an explicit `tomli; python_version < \"3.11\"` pin to\n  requirements-app.txt. The dep was already transitively available via\n  pytest/runpod, but pinning explicitly removes the fragility of relying\n  on a third-party transitive resolution.\n  (comment #3223490886)\n\nRefs #938\n\n* chore(lint): address Copilot post-push review on PR #978\n\nThree new Copilot comments after the merge from main:\n\n- pyproject.toml: flatten src/models/** for D102/D103/D107 the same way\n  scripts/** and src/data/** were already flattened. Keep ANN001 on the\n  glob (legacy, separate concern) but list each model file explicitly\n  for the D-rules so new files under src/models/ are not silently\n  exempt. (comment #3228263848)\n\n- scripts/check_no_new_funcs_in_pydoclint_excluded.py: fix module\n  docstring drift. The text said nested closures with \"six or more\n  leading spaces\" are ignored, but DEF_OR_CLASS_PATTERN matches 0-4\n  spaces, so anything >=5 is ignored. Rewrote to name the threshold\n  precisely and point at the regex where it lives. (comment #3228263810)\n\n- tests/scripts/test_check_no_new_funcs_in_pydoclint_excluded.py: import\n  the guard via importlib.util.spec_from_file_location instead of\n  mutating sys.path at module import time. Avoids leaking the change\n  into the rest of the test session. (comment #3228263867)\n\n* chore: trigger copilot review\n\nEmpty commit per CLAUDE.md step 6a — Copilot did not re-review 2625abd\nwithin the 15-min SLA and reviewers API rejects copilot-pull-request-reviewer\nas a non-collaborator. Push restarts the readiness loop.\n\n* docs(test): fix Copilot-flagged docstring typo on diff-header test\n\nCopilot review comment #3228504254 on PR #978: the test docstring said\n\"`+-+` headers\" — that is not a real unified-diff marker. The test\nactually guards against `+++ b/file.py` and `--- a/file.py` headers\nbeing mistaken for additions. Updated the docstring to name both\nmarkers correctly.\n\n* chore: resolve main merge conflicts in pydoclint follow-up PR\n\nAgent-Logs-Url: https://github.com/tinaudio/synth-setter/sessions/21c6c7a6-a341-4ed4-8ec7-ab11adad08ee\n\nCo-authored-by: ktinubu <17952332+ktinubu@users.noreply.github.com>\n\n* chore(lint): close P6 ratchet gap exposed by Phase 4 merge\n\nThe Phase 4 layout migration (#1009) moved files into\nsrc/synth_setter/{tools,models,metrics.py}/. The ruff per-file-ignores\nwere updated to mirror the new paths, but [tool.pydoclint].exclude\nwasn't, so 13 files had D102/D103/D107 ignored but were not in the\npydoclint exclude regex — re-opening the same blind spot this PR's\nP6 ratchet was supposed to close.\n\nRestores the maintainer's stated invariant from review round 2\n(comment #3228291103: \"D-rule ignores mirror pydoclint.exclude\")\nby adding the missing entries:\n\n  src/synth_setter/metrics.py\n  src/synth_setter/tools/model_from_wandb.py\n  src/synth_setter/tools/paramspec_to_table.py\n  src/synth_setter/tools/plot_param2tok.py\n  src/synth_setter/tools/sig_perf.py\n  src/synth_setter/models/components/cnn.py\n  src/synth_setter/models/components/embed_pool.py\n  src/synth_setter/models/components/residual_mlp.py\n  src/synth_setter/models/components/vector_field.py\n  src/synth_setter/models/ksin_ff_module.py\n  src/synth_setter/models/surge_ff_module.py\n  src/synth_setter/models/surge_flow_matching_module.py\n  src/synth_setter/models/surge_flowvae_module.py\n\nAfter this change, an adversarial probe (synthetic +def in\nsrc/synth_setter/{metrics,tools/sig_perf,models/components/cnn,\nmodels/ksin_ff_module}.py) makes the guard exit 1 in every case;\nthe 13 existing P6 tests still pass.\n\nRefs #938\n\n---------\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T16:42:04-04:00",
          "tree_id": "1697ba459f4799f7739afd06d9bc413baa440ca0",
          "url": "https://github.com/tinaudio/synth-setter/commit/e2c780ae3677279f07c832d6f4b8f90faea54558"
        },
        "date": 1778705809993,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.928321123123169,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.403775540776551,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.025144463405013084,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.024728119373321533,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.333155393600464,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 12.304192834166665,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.669558048248291,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.816596819856204,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.033946286886930466,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.05448567867279053,
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
          "id": "0a074b9e1a741a4cd6df8c1f7f2a49aaaebc9171",
          "message": "internal-feat(vst): promote writer shape helpers + DATASET_FIELD_NAMES to public (#1025)\n\n* internal-feat(vst): promote writer shape helpers + DATASET_FIELD_NAMES to public\n\nPromote the per-row array names and the audio/mel/param shape calculators\ninside synth_setter.data.vst.generate_vst_dataset to public module-level\nhelpers (DATASET_FIELD_NAMES, audio_dataset_shape, mel_dataset_shape,\nparam_array_dataset_shape) plus the mel-front-end constants and\nmel_hop_length / mel_n_fft / mel_n_frames helpers. make_spectrogram and\ncreate_datasets_and_get_start_idx now call the new helpers; behavior is\nbyte-identical for the existing render configs.\n\nFoundation for the upcoming WDS writer and shard-validator inner-shape\nchecks, which need to share these primitives with the validator side.\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): extract shape primitives to shapes.py to clear code-quality guard\n\nThe first commit added six new top-level helpers (DATASET_FIELD_NAMES,\nmel_hop_length, mel_n_fft, mel_n_frames, audio_dataset_shape,\nmel_dataset_shape, param_array_dataset_shape) and four module-level\nconstants to src/synth_setter/data/vst/generate_vst_dataset.py, which is\non the [tool.pydoclint].exclude list — and the code-quality CI guard\n(scripts/check_no_new_funcs_in_pydoclint_excluded.py, see #938) rejects\nany new top-level def in an excluded file. The preferred fix is to\nremove the source file from the exclude list, but generate_vst_dataset.py\nhas 12+ pre-existing pydoclint violations on neighbouring functions\n(make_spectrogram, generate_sample, make_dataset, _GenerateCliArgs) —\nall out of scope for this foundation PR.\n\nMove the new helpers to a fresh sibling module\nsrc/synth_setter/data/vst/shapes.py that was never on the exclude list,\nso pydoclint runs on it from day one and the guard sees the new defs\nland in an unexcluded file. generate_vst_dataset.py now imports the\nprimitives from the new module; behaviour is unchanged.\n\nAlso addresses the doc-drift advisory on the misleading \"single source\nof truth for the shard validator\" comment — the comment now lives in\nshapes.py's module docstring and hedges the validator/wds writer\nconsumers as \"(planned)\" since validate_shard.py still has its own\nprivate _EXPECTED_DATASETS tuple.\n\nRefs #874\nRefs #882\nRefs #938\n\n* internal-fix(vst): wire DATASET_FIELD_NAMES into the writer's HDF5 dataset names\n\nCopilot review on PR #1025 flagged the prior \"single source of truth\"\ncomment on DATASET_FIELD_NAMES as overpromising: save_samples and\ncreate_datasets_and_get_start_idx still hard-coded \"audio\", \"mel_spec\",\n\"param_array\" as string literals, so the constant was orthogonal to the\nwriter. The first follow-up commit (a2376d0) addressed half of that by\nmoving the constant to shapes.py and softening the comment.\n\nThis commit takes the other half — actually making the constant\nload-bearing on the writer side:\n\n- shapes.py exposes per-field constants AUDIO_FIELD, MEL_SPEC_FIELD,\n  PARAM_ARRAY_FIELD and builds DATASET_FIELD_NAMES from them, so the\n  tuple stays a derived view of the per-field constants.\n- create_datasets_and_get_start_idx now passes AUDIO_FIELD /\n  MEL_SPEC_FIELD / PARAM_ARRAY_FIELD into create_dataset instead of\n  the bare string literals.\n- A new shape-helpers test pins\n  DATASET_FIELD_NAMES == (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD)\n  and the literal triple so renaming any field still forces the\n  validator's expected tuple to update in lockstep.\n\nsave_samples doesn't reference dataset names (it operates on already-\ncreated h5py.Dataset handles), so no change there.\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): pass center=True explicitly to make_spectrogram's librosa call\n\nThe shape helpers in shapes.py (mel_n_frames) document the librosa\ncenter=True framing assumption, but make_spectrogram relied on the\nimplicit librosa default. Pinning center=True keeps the writer and the\n(planned) shard validator aligned on the same framing if librosa ever\nchanges its default.\n\nRefs #1025.\n\n* fix(vst): mel_hop_length raises on sample rates that would yield zero hop\n\nCopilot review on #1025 flagged that `mel_hop_length()` returns 0 when\n`sample_rate < MEL_FRAMES_PER_SECOND` (e.g., 50), and that 0 would later\ntrigger a `ZeroDivisionError` inside `mel_n_frames()`'s\n`audio_length // hop` floor-division. The schema doesn't lower-bound\nsample_rate at this depth, so guard at the leaf helper instead of\nrelying on upstream validation.\n\nRaises `ValueError` at the helper boundary so the failure surfaces with\na clear message instead of an opaque ZeroDivisionError downstream.\nNew test pins the raise.\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): guard mel_n_fft against sample rates that round to n_fft=0\n\nmel_n_fft now raises ValueError when int(0.025 * sample_rate) rounds down\nto 0 (e.g., sample_rate <= 39), mirroring the mel_hop_length guard so the\nfailure surfaces at the leaf helper instead of as an opaque librosa error\ndownstream. Also corrects a stale phrase in the mel_hop_length docstring\nthat still referenced the pre-guard ZeroDivisionError path.\n\n---------\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T22:11:52Z",
          "tree_id": "678e5b4d8aed880c1b20ae014808f476767541c5",
          "url": "https://github.com/tinaudio/synth-setter/commit/0a074b9e1a741a4cd6df8c1f7f2a49aaaebc9171"
        },
        "date": 1778711019163,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.929286241531372,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.725713217165321,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.023583507165312767,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.01666557788848877,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.565333604812622,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 11.894026241666666,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.517528057098389,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.891145356092602,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.032504044473171234,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.047752559185028076,
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
          "id": "0bcf7c6797e0bc4ca4b901537dd779122d3c06bb",
          "message": "ci(testing): wire Codecov gates and consolidate coverage collection (#1031)\n\nSteps 2-4 of the coverage-enforcement roadmap (#14). Builds the gating\ninfrastructure; activation of the Codecov GitHub App and CODECOV_TOKEN\nsecret happen separately (step 1, requires org admin in the UI).\n\n- Collapse the duplicate code-coverage job: every fast-suite leg\n  (ubuntu 3.10, ubuntu 3.11, macos 3.10) now produces a coverage.xml\n  and uploads under flag unit-cpu, instead of a fourth job re-running\n  the same suite.\n- Add [tool.coverage.run] (source, branch=true, parallel=true,\n  relative_files=true, omit) and [tool.coverage.paths] to pyproject.toml\n  so reports from different worktree paths merge cleanly.\n- New codecov.yml: project + patch status checks (informational for the\n  first week so we can observe before blocking), unit-cpu flag, and\n  per-directory component targets (pipeline 90%, models 85%, tools 50%,\n  rest auto). Validated against https://codecov.io/validate.\n- Align make coverage with CI flags (--cov-branch, xml + html reports,\n  same marker filter).\n\nFollow-ups (separate PRs): wire coverage into GPU/MPS/VST/slow workflows\nwith their own flags; add diff-cover as a fallback gate; add the Codecov\nbadge to README; flip status checks from informational to required once\nbaseline numbers stabilize.\n\nRefs #14, #149, #155, #30\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T22:25:09Z",
          "tree_id": "9ff00cb80a88d885f01328da660167e26c0e0cd9",
          "url": "https://github.com/tinaudio/synth-setter/commit/0bcf7c6797e0bc4ca4b901537dd779122d3c06bb"
        },
        "date": 1778711957961,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.6869418621063232,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.905548722790554,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.024676334112882614,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.02755880355834961,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.2969770431518555,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 10.80064003341667,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.233041763305664,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.433393930895254,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.03040858916938305,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.039746224880218506,
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
          "id": "b2cc2a1fa454f25d83707b539f5c2ba0949546ca",
          "message": "chore(ci): fix broken mutmut sandbox imports + document setup (#1026)\n\n* chore(ci): fix broken mutmut sandbox imports + document setup\n\nmutmut copies only `paths_to_mutate` into `mutants/` and strips the real\n`src/` off `sys.path`, so tests that transitively import un-mutated\nparts of the package (e.g. `synth_setter.cli.generate_dataset`,\n`synth_setter.pipeline.r2_io`) blow up during stats collection with\nImportError. PR #302 worked because it mutated `scripts/` only; the\nPhase 4 widen to `src/synth_setter/{evaluation,tools,pipeline/data}/`\nbroke this path and was never re-verified end-to-end.\n\nAdd `also_copy = [\"src/synth_setter/\"]` so the whole package lands in\nthe sandbox alongside the mutated subdirs, and document the moving\nparts in CLAUDE.md (Commands + a Mutation Testing section) so the next\ntime someone widens `paths_to_mutate` they know to recheck this.\n\nRefs #296\n\n* chore(ci): make mutmut run end-to-end (Linux CI workflow + subprocess fix)\n\nThree changes on top of the import-resolution fix in this PR's first\ncommit:\n\n1. **`tests/pipeline/data/test_stats.py`** — rewrite\n   `test_cli_help_advertises_mask_degenerate_bins_flag` to invoke\n   `_parse_args([\"--help\"])` in-process instead of shelling out via\n   `python -m`. Under `mutmut run`'s stats phase, the subprocess\n   inherited `MUTANT_UNDER_TEST=stats` and the mutated module's\n   trampoline tripped on `mutmut.config is None` in the fresh\n   interpreter, crashing stats collection. In-process avoids that\n   entirely and lets mutations of `_parse_args` actually be exercised\n   by this test (the subprocess form would have always run the\n   un-mutated function).\n\n2. **`.github/workflows/mutmut.yaml`** — new workflow_dispatch + weekly\n   cron job that runs `mutmut run` end-to-end on ubuntu-latest and\n   uploads the `mutants/` meta as an artifact. This is the\n   authoritative end-to-end gate for the `[tool.mutmut]` config.\n   macOS local runs cannot serve as that gate because\n   `tests/conftest.py` imports torch/h5py/hydra into the parent and\n   Apple's fork-safety check then SIGSEGVs every forked child.\n\n3. **`Makefile` + `CLAUDE.md`** — `make mutmut` sets\n   `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` (defensive on macOS,\n   no-op on Linux) and CLAUDE.md's \"Mutation Testing\" section now\n   covers (a) the subprocess pitfall, (b) the macOS caveat, and\n   (c) where the authoritative run lives.\n\nRefs #296\n\n* ci(mutmut): TEMP pull_request trigger to Level-1-verify the workflow on PR #1026 (revert before merge)\n\n* ci(mutmut): drop the temporary pull_request trigger\n\nRun 25829272616 (this branch's first commit with the workflow added)\ncompleted green on ubuntu-latest with the expected mix of statuses\n(🎉 810 killed, 🙁 341 survived, 🫥 1771 no tests, ⏰ 3 timeouts), so\nthe workflow is now Level-1-verified. Restore the trigger surface to\nworkflow_dispatch + weekly cron only.\n\nRefs #296\n\n---------\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T22:32:10Z",
          "tree_id": "3ed3a9d5082b21cb35deb38f66cd68b17c24f256",
          "url": "https://github.com/tinaudio/synth-setter/commit/b2cc2a1fa454f25d83707b539f5c2ba0949546ca"
        },
        "date": 1778712691538,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 4.068583965301514,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.933693888883572,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.02558779902756214,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.03581094741821289,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.519242286682129,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 12.274884934833329,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.311225891113281,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.65796631552279,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.03224911913275719,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.03608280420303345,
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
          "id": "da1327b625362cc1639518650075b3fe4572c8e9",
          "message": "internal-feat(pipeline): inner-shape checks in validate_shard (#1029)\n\n* internal-feat(vst): promote writer shape helpers + DATASET_FIELD_NAMES to public\n\nPromote the per-row array names and the audio/mel/param shape calculators\ninside synth_setter.data.vst.generate_vst_dataset to public module-level\nhelpers (DATASET_FIELD_NAMES, audio_dataset_shape, mel_dataset_shape,\nparam_array_dataset_shape) plus the mel-front-end constants and\nmel_hop_length / mel_n_fft / mel_n_frames helpers. make_spectrogram and\ncreate_datasets_and_get_start_idx now call the new helpers; behavior is\nbyte-identical for the existing render configs.\n\nFoundation for the upcoming WDS writer and shard-validator inner-shape\nchecks, which need to share these primitives with the validator side.\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): extract shape primitives to shapes.py to clear code-quality guard\n\nThe first commit added six new top-level helpers (DATASET_FIELD_NAMES,\nmel_hop_length, mel_n_fft, mel_n_frames, audio_dataset_shape,\nmel_dataset_shape, param_array_dataset_shape) and four module-level\nconstants to src/synth_setter/data/vst/generate_vst_dataset.py, which is\non the [tool.pydoclint].exclude list — and the code-quality CI guard\n(scripts/check_no_new_funcs_in_pydoclint_excluded.py, see #938) rejects\nany new top-level def in an excluded file. The preferred fix is to\nremove the source file from the exclude list, but generate_vst_dataset.py\nhas 12+ pre-existing pydoclint violations on neighbouring functions\n(make_spectrogram, generate_sample, make_dataset, _GenerateCliArgs) —\nall out of scope for this foundation PR.\n\nMove the new helpers to a fresh sibling module\nsrc/synth_setter/data/vst/shapes.py that was never on the exclude list,\nso pydoclint runs on it from day one and the guard sees the new defs\nland in an unexcluded file. generate_vst_dataset.py now imports the\nprimitives from the new module; behaviour is unchanged.\n\nAlso addresses the doc-drift advisory on the misleading \"single source\nof truth for the shard validator\" comment — the comment now lives in\nshapes.py's module docstring and hedges the validator/wds writer\nconsumers as \"(planned)\" since validate_shard.py still has its own\nprivate _EXPECTED_DATASETS tuple.\n\nRefs #874\nRefs #882\nRefs #938\n\n* internal-fix(vst): wire DATASET_FIELD_NAMES into the writer's HDF5 dataset names\n\nCopilot review on PR #1025 flagged the prior \"single source of truth\"\ncomment on DATASET_FIELD_NAMES as overpromising: save_samples and\ncreate_datasets_and_get_start_idx still hard-coded \"audio\", \"mel_spec\",\n\"param_array\" as string literals, so the constant was orthogonal to the\nwriter. The first follow-up commit (a2376d0) addressed half of that by\nmoving the constant to shapes.py and softening the comment.\n\nThis commit takes the other half — actually making the constant\nload-bearing on the writer side:\n\n- shapes.py exposes per-field constants AUDIO_FIELD, MEL_SPEC_FIELD,\n  PARAM_ARRAY_FIELD and builds DATASET_FIELD_NAMES from them, so the\n  tuple stays a derived view of the per-field constants.\n- create_datasets_and_get_start_idx now passes AUDIO_FIELD /\n  MEL_SPEC_FIELD / PARAM_ARRAY_FIELD into create_dataset instead of\n  the bare string literals.\n- A new shape-helpers test pins\n  DATASET_FIELD_NAMES == (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD)\n  and the literal triple so renaming any field still forces the\n  validator's expected tuple to update in lockstep.\n\nsave_samples doesn't reference dataset names (it operates on already-\ncreated h5py.Dataset handles), so no change there.\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): pass center=True explicitly to make_spectrogram's librosa call\n\nThe shape helpers in shapes.py (mel_n_frames) document the librosa\ncenter=True framing assumption, but make_spectrogram relied on the\nimplicit librosa default. Pinning center=True keeps the writer and the\n(planned) shard validator aligned on the same framing if librosa ever\nchanges its default.\n\nRefs #1025.\n\n* internal-feat(pipeline): inner-shape checks in validate_shard\n\nTightens validate_shard's HDF5 path so every dataset's full ``.shape`` is\nchecked against the writer's source-of-truth shape helpers in\n``synth_setter.data.vst.shapes`` — not just ``shape[0]``. The validator\nnow uses ``DATASET_FIELD_NAMES`` directly (deleting the private\n_EXPECTED_DATASETS mirror) and the new ``_expected_dataset_shapes`` helper\nto derive ``(N, C, time)`` for audio, ``(N, C, n_mels, n_frames)`` for\nmel, and ``(N, num_params)`` for the param array.\n\nA renderer change that drifts the audio / mel / param shapes now fails\nfast at validate time instead of silently shipping mis-shaped shards\ndownstream to training.\n\nHDF5-only; the wds tar branch is PR-E in the WDS port roadmap.\n\nRefs #874\nRefs #882\n\n* chore(pipeline): remove validate_shard from pydoclint excludes\n\nThe previous commit added _expected_dataset_shapes() to validate_shard.py\nwhile that file was on [tool.pydoclint].exclude — tripping the\ncheck_no_new_funcs_in_pydoclint_excluded guard. The guard's preferred\nremediation is to remove the file from the excludes list, which means\nmaking it pydoclint-clean.\n\nAdd sphinx :param: / :returns: sections to _expected_dataset_shapes,\nvalidate_shard, _load_spec, and validate_all_shards_from_r2, then drop\nvalidate_shard.py from the exclude list. Tightens lint coverage as a\nside benefit of the inner-shape work.\n\n* docs(design): update validate_shard description to match inner-shape checks\n\nAfter #1029 (this PR), validate_shard asserts the full per-dataset\n.shape against the writer's shape helpers from\nsynth_setter.data.vst.shapes, not just shape[0] row counts. Updates the\nfile-tree comment in data-pipeline.md to match.\n\nPicks up the post-PR doc-drift advisory.\n\nRefs #874\nRefs #882\n\n---------\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T22:53:21Z",
          "tree_id": "28c0e91dffdcb477b7368599004baad700276e52",
          "url": "https://github.com/tinaudio/synth-setter/commit/da1327b625362cc1639518650075b3fe4572c8e9"
        },
        "date": 1778713534305,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 4.233779430389404,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 6.850063925273717,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.028867458924651146,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.03103315830230713,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.534191131591797,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 12.102701674666667,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.233779430389404,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.861200887709856,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.032164886593818665,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.03721886873245239,
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
          "id": "9cf790ffd1e8700ad28d4de8a0a6e38b53ded6b5",
          "message": "internal-feat(vst): split make_dataset into make_hdf5_dataset + make_wds_dataset; dispatch CLI by suffix (#1030)\n\n* internal-feat(vst): split make_dataset into make_hdf5_dataset + make_wds_dataset; dispatch CLI by suffix\n\nSplits the single legacy ``make_dataset(h5py.File, render_cfg)`` into two\nwriter entrypoints dispatched by the renderer CLI on the output file's\nsuffix:\n\n- ``make_hdf5_dataset(hdf5_file: Path | str, render_cfg, ...)`` — keeps\n  the resumable HDF5 path; signature changes to take a path and open the\n  file internally. Writes the ``audio.attrs`` sidecar from a new\n  ``ShardMetadata`` instance.\n- ``make_wds_dataset(wds_file: Path | str, render_cfg, ...)`` — new wds\n  path using ``webdataset.TarWriter``. Emits per-batch ``.npy`` members\n  plus a ``metadata.json`` member from the same ``ShardMetadata`` instance.\n\nBoth paths share rendering logic via ``_validate_fixed_params_lengths`` /\n``_generate_sample_for_index`` / ``_shard_metadata_from_render`` /\n``_render_in_batches`` helpers. The new writer helpers live in a fresh\n``synth_setter.data.vst.writers`` module so the code-quality guard (#938)\nstays green without docstring-cleaning the legacy parts of\n``generate_vst_dataset.py``.\n\nCLI ``main`` in ``generate_vst_dataset.py`` dispatches by\n``Path(args.data_file).suffix`` via ``EXTENSION_TO_OUTPUT_FORMAT``\n(``.h5`` → HDF5, ``.tar`` → wds, unknown → ``SystemExit``).\n\nUpdates the one in-tree caller (``tools/surge_xt_interactive.py``) to\nthe new path-accepting signature. Drops the stale \"HDF5-only\" docstring\nclaims from ``cli/generate_dataset.py``.\n\nRefs #874\nRefs #882\n\n* docs(design,guides): sweep stale make_dataset refs after writer split\n\nPost-#1030 doc-drift cleanup. The PR renames make_dataset →\nmake_hdf5_dataset + adds make_wds_dataset, and lifts both into a new\nsynth_setter.data.vst.writers module. This commit ports the docs over.\n\n* docs/design/data-pipeline.md:\n  - Fix broken import example (was `from ...generate_vst_dataset import\n    make_hdf5_dataset` → now `from ...writers import make_hdf5_dataset`).\n  - Rename `make_dataset()` / `make_dataset` in §7.8.1 spawn-rationale\n    prose.\n  - Rewrite §7.10 head paragraph and the \"Why generation stays HDF5\"\n    sub-section: workers now emit the format selected by the spec; the\n    \"design in transition\" Note that pointed at PR-13 is replaced by a\n    factual HDF5-is-resumable / WDS-is-not paragraph since PR-13 has\n    landed.\n  - Fix stale src/pipeline/schemas/shard_metadata.py path (post-#1001\n    layout) at line 863.\n* docs/design/data-pipeline-implementation-plan.md — three\n  make_dataset / generate_vst_dataset import refs renamed to\n  make_hdf5_dataset / writers.\n* docs/reference/audio-similarity-benchmarks.md — two prose refs to\n  make_dataset renamed to make_hdf5_dataset.\n* docs/guides/surge-xt-interactive.md — seven refs renamed; module\n  path on the symbol-link swapped from generate_vst_dataset.py to\n  writers.py.\n* docs/doc-map.yaml — retarget the surge-xt-interactive\n  generate_vst_dataset.py entry to its remaining surface\n  (generate_sample / VSTDataSample / make_spectrogram) and add new\n  writers.py + shapes.py entries under both the data-pipeline design\n  doc and the surge-xt-interactive guide.\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): address Copilot review nits on PR #1030\n\n- Drop the unused `from synth_setter.data.vst import param_specs` import in\n  test_writers_wds_e2e.py — would trip ruff F401.\n- Rewrite the stale \"Inlined here to avoid a cross-test import dependency\"\n  comment that contradicted the actual import-from-sibling-test pattern.\n- Narrow `pytest.raises(Exception, ...)` → `pytest.raises(pydantic.ValidationError, ...)`\n  in test_make_wds_dataset_metadata_json_strict_rejects_extra so the test\n  actually pins the `extra=\"forbid\"` behavior its docstring promises.\n- Rename three test functions in test_generate_vst_dataset.py from\n  `test_make_dataset*` → `test_make_hdf5_dataset*` to match the post-split\n  public entrypoint name (better for grep/test selection).\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): address Copilot follow-ups on PR #1030\n\n- Drop unused hashlib and random imports from generate_vst_dataset.py\n  (left over after the writer split — would have failed ruff F401 once\n  the file isn't pydoclint-excluded).\n- Correct the WebDataset shard-structure snippet in the data-pipeline\n  design doc: real members are <start_idx:08d>.<audio|mel_spec|param_array>.npy\n  (per DATASET_FIELD_NAMES + save_wds_samples), each holds a whole\n  batch stacked on axis 0, and keys advance by sample_batch_size.\n\nRefs #874\n\n* chore(lint): remove test_generate_vst_dataset.py from pydoclint excludes\n\nThe 254c534 rename of three tests (test_make_dataset* → test_make_hdf5_dataset*)\nin tests/data/vst/test_generate_vst_dataset.py registers as new top-level\ndefs in the diff against main, which trips the\ncheck_no_new_funcs_in_pydoclint_excluded guard (#938).\n\nThe file is already lint-clean: 0 missing arg annotations, only 1 missing\ndocstring on a nested `fake_sample` closure — added a one-liner.\n\nDrop the file from [tool.pydoclint].exclude. pydoclint now runs against\nit and exits clean.\n\nRefs #25\nRefs #874\n\n* revert(vst): keep test_make_dataset names to satisfy code-quality guard\n\nThe 254c534 rename of three tests in test_generate_vst_dataset.py\n(`test_make_dataset` -> `test_make_hdf5_dataset` and friends) was a\nCopilot polish suggestion but tripped the\ncheck_no_new_funcs_in_pydoclint_excluded guard (#938) — the diff against\nmain shows them as new top-level defs, and the file is on\n[tool.pydoclint].exclude.\n\nThe previous attempted fix (remove the file from the exclude list)\nsilently failed locally because the `pre-commit run` output was piped\nthrough `tail -8`, which truncated pydoclint's 13+ DOC101/DOC103\nviolations on pytest-fixture-args (`tmp_path`, `monkeypatch`) — those\ndocstrings would all need `:param:` sections to clear pydoclint, which\nis out of scope for this PR.\n\nReverting just the three function names; the docstrings already say\n\"make_hdf5_dataset\" so future maintainers know the function under test.\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): address Copilot review nits on PR #1030\n\n- writers.py: tighten fixed_*_params_list validation from len() < to\n  len() != expected_fixed_len. The writer indexes fixed params by\n  ``i - start_idx``, so a shard-length list on a resumed run\n  (start_idx > 0) would silently shift indices (row start_idx using\n  list[0]). Exact-equality catches that mismatch at validation time\n  instead of letting it through. Existing too-short test still\n  matches via the \"fixed_synth_params_list has length\" prefix.\n- writers.py: rewrite make_hdf5_dataset / make_wds_dataset /\n  _validate_fixed_params_lengths docstrings to spell out the\n  tail-only contract on resumed runs (list[0] lands at row start_idx,\n  caller slices a full-length list themselves), so the indexing\n  semantics are visible at the public API surface.\n- generate_vst_dataset.py: rewrite the main() lazy-import rationale —\n  h5py is already a module-level import here, so it is not what the\n  lazy load defers; only webdataset is.\n\nRefs #874\n\n* Update writers.py\n\n* docs(claude-md): point doc-map's dispatch-maps covers at the actual dispatcher\n\nCopilot review on PR #1030 flagged that the doc-map.yaml line 85 entry\nfor spec.py says the suffix-dispatch maps are consumed by \"the renderer\nCLI's main() in writers.py\", but the dispatch actually lives in\ndata/vst/generate_vst_dataset.py::main() (which imports\nEXTENSION_TO_OUTPUT_FORMAT, reads Path(data_file).suffix, and routes to\nwriters.make_hdf5_dataset or writers.make_wds_dataset).\n\nUpdate the covers text to point at the correct module and spell out\nthe lookup so future doc-drift checks anchor on the right symbol.\n\nRefs #874\nRefs #882\n\n---------\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T19:02:45-04:00",
          "tree_id": "8d7c78e4f10f30a150409487d4f0ac9ec73a335e",
          "url": "https://github.com/tinaudio/synth-setter/commit/9cf790ffd1e8700ad28d4de8a0a6e38b53ded6b5"
        },
        "date": 1778714282213,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-1-preset-n-renders/multi-scale-spectral-loss-max",
            "value": 3.1501553058624268,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/dtw-aligned-mfcc-distance-max",
            "value": 5.343547191619873,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/spectral-optimal-transport-max",
            "value": 0.026436209678649902,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/rms-envelope-cosine-distance-max",
            "value": 0.014015793800354004,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/mel-spectrogram-mean-absolute-error",
            "value": 3.1869046688079834,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/num-samples",
            "value": 6,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/wall-clock-seconds-per-render",
            "value": 11.785817476249994,
            "unit": "seconds"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-multi-scale-spectral-loss-max",
            "value": 4.5833048820495605,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-dtw-aligned-mfcc-distance-max",
            "value": 6.7500910580158235,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-spectral-optimal-transport-max",
            "value": 0.039276544004678726,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-1-preset-n-renders/all-pairs-rms-envelope-cosine-distance-max",
            "value": 0.046715378761291504,
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
        "date": 1777676893927,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.5421438217163086,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 3.3441296565532683,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.03164781630039215,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.007090747356414795,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.5233615636825562,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 14.6120192707,
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
          "id": "86a46d2f71c151ec8445e1b84dc2c3e4cf0af0c4",
          "message": "internal-feat(pipeline): renderer-version contract end-to-end + rclone-native upload bounds (#740)\n\n* internal-feat(pipeline): pin renderer_version to SURGE_XT_RENDERER_VERSION; expose extract_renderer_version\n\n`materialize_spec` previously extracted `renderer_version` from the VST3\nplugin bundle at materialization time, which required loading the plugin\nvia `pedalboard.VST3Plugin` when neither `Contents/moduleinfo.json` nor\n`Contents/Info.plist` was present — and that codepath needs an X display.\nThat blocks any caller that wants to materialize a spec without an X\nstack (e.g. the SkyPilot launcher, which runs on a GHA runner / dev\nlaptop and never loads the plugin itself).\n\nPin `renderer_version` to a single source of truth, the\n`SURGE_XT_RENDERER_VERSION = \"1.3.4\"` constant in this module, kept in\nlockstep with the dev-snapshot image's `SURGE_GIT_REF`. `materialize_spec`\nnow sets the pin directly and doesn't touch the plugin bundle.\n\nKeep `extract_renderer_version` as a public function — same static-metadata\n+ pedalboard-fallback shape — so the worker side can call it against the\nactual plugin and verify the pin matches reality before rendering. The\nworker-side cross-check is the next commit; the rclone-native upload\nbounds are the one after.\n\nRefs #534\n\n* internal-feat(pipeline): worker-side renderer_version cross-check in generate_dataset.run\n\nThe launcher pins `renderer_version` to `SURGE_XT_RENDERER_VERSION` blindly\n(its code path stays interpreter-only). The worker is where pedalboard is\navailable, so the worker is where the pin gets verified against reality.\n\n`run()` now calls `extract_renderer_version` against `spec.plugin_path`\nbefore any rclone or subprocess work and raises `RuntimeError` if the\nrunning plugin disagrees with the spec. The error message points at the\ntwo valid fixes (rebuild the image against the matching `SURGE_GIT_REF`\nor bump the constant), so failures are actionable rather than mysterious.\nOn match, a single `renderer_version OK: …` info log records the\nconfirmed pairing for forensics.\n\nTest fixture: tests/pipeline/fixtures/TestPlugin.vst3 (already on `main`)\nhas `Contents/moduleinfo.json` reporting Version=\"1.0.0-test\". Updated\n`_base_spec_kwargs` to use that fixture + that version so the spec/plugin\npair matches by default; new test asserts mismatch raises before any\nupload happens.\n\nRefs #534\n\n* internal-fix(pipeline): rclone-native upload bounds + 'rclone returned cleanly' sentinel\n\nTwo related observability fixes for the worker upload path:\n\n1. `_rclone_copy` was running `rclone copy --checksum src dst` with no\n   timeouts and no retries — a stuck TCP connect or a slow PUT could hold\n   the worker indefinitely. Switch to rclone's own bounds:\n     --contimeout=30s    bound TCP connect phase\n     --timeout=300s      bound any single HTTP request\n     --retries=3         retry the whole copy on transient failure\n     -vv                 emit per-request debug log so a failure leaves\n                         actionable evidence in the worker stdout\n   Letting rclone enforce these (vs. wrapping `subprocess.run(..., timeout=N)`\n   in Python) preserves the postcondition that a non-zero exit means the\n   upload genuinely failed, instead of \"we waited N seconds and gave up\".\n\n2. After `subprocess.check_call` returns from a successful rclone, log a\n   single `rclone returned cleanly: <src> -> <dst>` sentinel. Distinct\n   string so CI logs can be grepped to tell at a glance whether the rclone\n   subprocess actually exited vs. hanging post-upload (the bug-#2 hang\n   shape from #735, now believed gone but worth keeping the canary).\n\nAdds matching boundary logs around the upload path (`spec written:`,\n`spec uploaded ->`, `rendering shard …`, `shard rendered: … (N bytes)`,\n`shard uploaded: …`) so a `tail_logs(follow=False)` dump pinpoints which\nstep a hung run got to.\n\nRefs #534\nRefs #735\n\n* refactor: move extract_renderer_version to src.data.vst.core\n\nThe extractor reads VST3 plugin bundle metadata — that's a VST utility,\nnot a spec-schema concern. Move it next to the other VST helpers\n(`load_plugin`, `load_preset`, `render_params`) in `src/data/vst/core.py`\nand update the worker-side caller in `pipeline.entrypoints.generate_dataset`\nto import from the new location.\n\n`SURGE_XT_RENDERER_VERSION` stays in `pipeline.schemas.spec` because it\nis a spec-construction constant (consumed by `materialize_spec`); only\nthe extractor moves. Tests follow the source: `TestExtractRendererVersion`\nmoves from `tests/pipeline/test_schemas/test_spec.py` to a new\n`tests/data/vst/test_core.py` (matching the existing\n`tests/data/vst/{test_generate_vst_dataset,test_preset_*}.py` layout).\n\nNo behavior change. The function signature and error contract are\nidentical; tests are byte-for-byte the same as their previous\nlocation, just imported from the new path.\n\nRefs #534\n\n---------\n\nCo-authored-by: Your Name <you@example.com>",
          "timestamp": "2026-05-01T18:58:15-04:00",
          "tree_id": "1b380633afcd57b07636dddead1f765f334739f0",
          "url": "https://github.com/tinaudio/synth-setter/commit/86a46d2f71c151ec8445e1b84dc2c3e4cf0af0c4"
        },
        "date": 1777677713198,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.481773853302002,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 3.106176937967539,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.020253485068678856,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.007873713970184326,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.6002923250198364,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 12.70911317560001,
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
          "id": "adfd7ab4e1fdba639f812db874e1086dccddc471",
          "message": "internal-feat(skypilot): matrix-driven RunPod + OCI generate-dataset CI (#777)\n\n* feat(skypilot): add OCI x86 CPU as a second SkyPilot smoke target\n\nMirrors the existing RunPod path with a CPU-only Flex template\n(VM.Standard.E5.Flex), provider-neutral launcher (no code changes), a\nparallel `generate-oci` CI job (continue-on-error: true while bedding\nin), and a brief operator setup guide. The launcher's R2-uploaded spec\ncontract and the #735 os._exit(0) workaround are preserved across\nproviders.\n\nRegion lives only in ~/.oci/config so `sky check oci` and the launch\nagree on a single source of truth. ~/.oci paths are derived from $HOME\ninside the container so the cred-write step is portable across base\nimages.\n\nThe dev-snapshot Docker image must be rebuilt+pushed with skypilot[oci]\nin requirements-app.txt before the OCI CI job can pass.\n\nRefs #534\n\n* internal-feat(skypilot): add OCI debug noop template + temporarily switch debug workflow to OCI noop for iteration\n\nAdds configs/compute/oci-debug-template.yaml as the OCI/CPU sibling of\nrunpod-debug-template.yaml, updates the runner-side skypilot install in\ntest-skypilot-debug.yml to carry the [oci] extra, and TEMPORARILY:\n\n  - Comments out all RunPod debug matrix variants except 'noop'.\n  - Points 'noop' at configs/compute/oci-debug-template.yaml.\n  - Swaps the inline-sky cred-write step from ~/.runpod/config.toml to\n    ~/.oci/config + ~/.sky/config.yaml + 'sky check oci' fail-fast gate.\n\nThe temporary changes (matrix gating + cred-write swap) are iteration\nscaffolding for landing the OCI target. Re-enable variants progressively\nas OCI plumbing stabilises; back the gating out before marking PR #769\nready for review.\n\nRefs #768\n\n* fix(skypilot): rewrite OCI templates to docker-in-run; SkyPilot OCI rejects image_id\n\nOCI's SkyPilot backend rejects 'image_id: docker:<image>' with\n'Docker image is currently not supported on OCI'. Rewrite both OCI\ntemplates to provision a stock OCI Ubuntu VM (image_tag_general:\nskypilot:cpu-ubuntu-2204) and run the worker container ourselves\ninside the run: block:\n\n  - oci-debug-template.yaml: drop image_id entirely (noop probe just\n    echoes; no docker needed).\n  - oci-cpu-template.yaml: setup: installs docker.io, starts daemon,\n    pre-pulls worker image; run: invokes 'sudo docker run' with the\n    same env-injection contract the launcher uses on RunPod. Worker\n    image moved from a SkyPilot image_id to a WORKER_IMAGE env var\n    that the workflow sed-pins.\n  - test-dataset-generation.yml + test-skypilot-debug.yml: write\n    image_tag_general into ~/.sky/config.yaml; sed-pin updated to\n    rewrite WORKER_IMAGE (not image_id); test-dataset-generation also\n    runtime-installs skypilot[oci] inside dev-snapshot if 'oci' SDK is\n    missing (bridge until the post-rebuild dev-snapshot lands).\n\nRefs #768\n\n* fix(skypilot): pin OCI noop debug to VM.Standard.E2.1.Micro (Always Free)\n\nus-ashburn-1 returned ResourcesUnavailableError across all 3 ADs for\nSkyPilot's auto-picked VM.Standard.E4.Flex (cpus=2, mem=8). Likely\nzero E4.Flex compute quota in the operator's tenancy.\n\nPin the noop probe to VM.Standard.E2.1.Micro instead — it's OCI's\n'Always Free' shape (1 OCPU, 1 GB AMD64), available to every tenancy\nwithout a quota request. This lets us validate the OCI launcher\nplumbing (cred-write, sky check, provision, teardown) independently\nof whether the operator has paid compute quota for E4.Flex.\n\nProduction template (oci-cpu-template.yaml) still asks for cpus: 4+,\nmemory: 16+ (needed for the VST/numpy worker); a green test-dataset-\ngeneration OCI run depends on the operator having actual compute\nquota.\n\nRefs #768\n\n* diag(skypilot): list OCI region subscriptions + E-Flex compute limits in noop probe\n\nAdds a one-shot diagnostic to test-skypilot-debug.yml's inline-sky step\nto print the operator's tenancy region subscriptions and service limit\nvalues for any VM.Standard.E*.Flex compute in each region. Output guides\nwhich region the OCI templates should target (right now provisioning\nfails with ResourcesUnavailableError in us-ashburn-1, suggesting zero\nquota there for E4.Flex).\n\nRefs #768\n\n* diag(skypilot): expand OCI diagnostic to dump ALL compute limits + per-shape resource availability\n\nPrevious filter ('standard-e' AND 'flex' in limit name) returned empty\nacross the operator's home region — but the actual OCI limit names may\nnot match that regex. Print every compute limit verbatim, list ADs in\nthe region, and call get_resource_availability for E4.Flex / E5.Flex /\nA1.Flex / E2.1.Micro to surface used/available counts. This pinpoints\nwhether the tenancy has zero paid quota (so OCI is a non-starter for\nthe prod template) or just regional capacity issues.\n\nRefs #768\n\n* fix(skypilot): sudo -E to preserve env vars into nested docker run on OCI\n\nWorker container started successfully on OCI but failed at:\n  KeyError: 'WORKER_SPEC_URI'\ninside the inlined python -c. Root cause: bare 'sudo' strips the\ncaller's environment, so 'docker run -e WORKER_SPEC_URI' (no value;\ninherit from parent shell) reaches docker with WORKER_SPEC_URI unset.\nPass -E to sudo to preserve all caller env vars (RCLONE_CONFIG_R2_*,\nWORKER_SPEC_URI, WORKER_IMAGE) into the docker invocation.\n\nRefs #768\n\n* fix(skypilot): propagate SYNTH_SETTER_WORKER_RANK/NUM_WORKERS into OCI worker container\n\nThe launcher injects partition env vars per rank via task.update_envs().\nOn RunPod they reach the worker process directly because SkyPilot owns\nthe docker container. On OCI we run docker ourselves inside the run:\nblock, so we have to forward each env var explicitly via 'docker run\n-e'. Add the two partition vars to both the placeholder envs: block\n(so SkyPilot doesn't reject the task) and the docker -e list (so the\ninner python process inherits them).\n\nRefs #768\n\n* fix(skypilot): give OCI launcher a run-id-scoped cluster name to avoid R2 collision with RunPod\n\nBoth 'generate' (RunPod) and 'generate-oci' jobs in the same workflow run\ninvoke skypilot_launch_smoke concurrently. With the default cluster name\n('synth-setter-smoke-{config_id[:8]}' = 'synth-setter-smoke-runpod-s'),\nboth jobs upload their materialized spec to the SAME R2 key:\n'r2:.../skypilot-launcher-specs/synth-setter-smoke-runpod-s.json'.\nWhichever uploads last wins; both clouds' workers then download that\nspec and write shards under its r2_prefix. validate-shard reads RunPod's\nlocal /tmp/input_spec.json (the loser's run_id), gets the wrong prefix,\nand fails to find shards in R2.\n\nFix: pass --cluster-name explicitly for the OCI step, scoped to the\ngithub.run_id so it's distinct from RunPod's default and unique across\nPR pushes. RunPod keeps the default for backwards compat with existing\ndebug/dispatch tooling.\n\nRefs #768\n\n* fix(skypilot): wait for cloud-init + apt lock before installing docker on OCI VM\n\nSetup failed with:\n  E: Could not get lock /var/lib/apt/lists/lock. It is held by process 3178 (apt)\non a freshly-provisioned OCI Ubuntu VM. SkyPilot launches concurrently\nwith cloud-init's own apt activity. Wait for cloud-init to finish, then\npoll the apt+dpkg locks (up to 5 min) before our 'apt-get update' fires.\n\nRefs #768\n\n* fix(skypilot): give OCI worker docker container full privileges + raised nofile\n\nWorker exited on:\n  X Error of failed request:  BadWindow (invalid Window parameter)\n  Major opcode of failed request:  20 (X_GetProperty)\nduring pedalboard's Surge XT preset load on OCI. Preceded by:\n  dbus-daemon: Failed to set fd limit to 65536: Operation not permitted\n\nBoth symptoms are an under-privileged docker container. RunPod pods ARE\nthe SkyPilot container (RunPod's runtime grants full privileges); on\nOCI we run docker ourselves inside the VM, default-unprivileged, so\nthe dbus / Xvfb / pedalboard X-stack can't operate. Match RunPod's\nprivilege level: add --privileged and --ulimit nofile=65536:65536.\n\n--privileged is correct here even by least-privilege standards: the OCI\nVM is single-tenant per-job (sky.launch + down=True) and the inner\ncontainer is the entire workload — there's no other process or user\non the VM to escape to.\n\nRefs #768\n\n* chore(skypilot): drop redundant plugins/ symlink from OCI template run block\n\nThe Dockerfile pre-creates plugins/Surge XT.vst3 -> /usr/lib/vst3/Surge XT.vst3\ninside WORKDIR at build time (docker/ubuntu22_04/Dockerfile:322-323), and\ngit init/fetch/checkout is used (instead of clone) specifically to preserve\nthat symlink across the source layer. The OCI template's run block does\nnot task.workdir-override or volume-mount over that path, so the runtime\n'mkdir -p plugins && ln -sf ...' was dead code.\n\nThe workflow's launcher container still needs the runtime symlink because\ndocker run -v $github.workspace:/home/build/synth-setter masks the image's\nWORKDIR contents — leave that one alone.\n\nRefs #768\n\n* style(skypilot): expand single-line OCI python -c into multi-line form\n\nReplace the one-liner python -c with a properly-formatted multi-line\nblock. Comment block above run: documents why the python body lines\nsit at the YAML block-scalar minimum indent (2 spaces in source = 0\nafter YAML strip) instead of matching the surrounding bash indent —\nPython -c rejects leading whitespace on top-level statements even\nwhen uniform.\n\nNo behavioral change.\n\nRefs #768\n\n* fix(skypilot): tighten OCI setup — apt-native lock wait, drop dead usermod, hard timeout on cloud-init\n\nFive fixes from review:\n\n  - apt-get -o DPkg::Lock::Timeout=300 — apt itself waits for the lock\n    (no race between fuser and the next command). Drops the manual\n    fuser-poll loop.\n  - timeout 300 sudo cloud-init status --wait — bounds the wait\n    explicitly; --wait has no internal timeout and could hang ~10min\n    silently.\n  - Drop sudo systemctl enable --now docker || sudo service ... fallback.\n    SkyPilot's OCI Canonical Ubuntu 22.04 image is systemd; the service\n    fallback masks real failures (apt incomplete, dpkg lock, etc).\n  - Drop sudo usermod -aG docker \"$USER\" — dead code. Group membership\n    requires re-login and run: uses sudo -E docker throughout. Was only\n    useful for human SSH debugging on a VM that gets torn down post-job.\n  - Removes the \"$USER\" reference, which was fragile under set -u in\n    SkyPilot's run shell.\n\nRefs #768\n\n* docs(skypilot): link #776 follow-up next to OCI --privileged invocation\n\nIssue #776 tracks the work to drop --privileged and replace it with the\nminimal cap-add / shm-size / ulimit combination needed for Xvfb + dbus\n+ pedalboard's Surge XT preset load. Comment block above run: now\npoints the reader at it so the temporary nature of the privilege\nescalation is documented in-source.\n\nRefs #768, #776\n\n* ci(skypilot): drop OCI iteration scaffolding from debug workflow\n\nRestores the 12-variant RunPod debug matrix and the RunPod cred-write\nstep that c2e1030 temporarily gated to OCI noop only, and drops the\ndiagnostic dumps from 0d44582 + d549fb2 (transient quota false-alarm\nchase, no longer needed). Keeps:\n\n  - configs/compute/oci-debug-template.yaml — useful sibling reference\n    for future OCI debug variants.\n  - skypilot[runpod,oci] installer extra — the [oci] dep is harmless on\n    RunPod-only matrix cells and avoids a re-install when an OCI noop\n    is added back later.\n\nHeader banner updated to point readers at the OCI sibling template\nwithout making it part of the default matrix.\n\nRefs #768\n\n* internal-fix(skypilot): use empty-string env placeholders in OCI template\n\nSYNTH_SETTER_WORKER_RANK / SYNTH_SETTER_NUM_WORKERS were set to \"0\" /\n\"1\" defaults, which lied about the contract: the launcher's\ntask.update_envs(...) injects per-rank values, so the defaults were\nshadowed and never read. Switch to \"\" placeholders matching every\nother launcher-injected key (and matching runpod-template.yaml).\n\nNo runtime behavior change today (update_envs already overwrites), but\nthe bogus defaults would mask the missing-env failure mode at exactly\nthe worst time: a future regression where the launcher fails to inject\nper-rank values would silently render rank=0/1 on every worker\ninstead of raising in pipeline.partitioning.read_rank_world_from_env.\n\nRefs #768\n\n* ci(skypilot): collapse RunPod + OCI generate jobs into one matrix\n\nReplaces two parallel `generate` + `generate-oci` jobs with a single\nmatrix-driven `generate` job over [runpod, oci]. Both cells exercise\nthe same provider-neutral launcher\n(pipeline.entrypoints.skypilot_launch_smoke) against per-provider\ncompute templates.\n\nLoad-bearing changes vs the prior shape:\n\n  - Both cells now run --num-workers 3, so the shard partitioner is\n    exercised end-to-end on every PR (previously RunPod was passing\n    --num-workers 3 explicitly; OCI was implicitly 1).\n  - RunPod gets a run-id-scoped cluster name\n    (synth-setter-smoke-runpod-${run_id}) — fixes the same R2 spec-key\n    race that the OCI step was patched for in e371a13. Without this,\n    the launcher's R2 spec key would still collide if a future PR adds\n    a parallel generate-oci-style job.\n  - The launch step is one `docker run` whose bash heredoc switches on\n    $PROVIDER for cred-write (case \"$PROVIDER\" in runpod) ... ;; oci) ... ;;\n    esac), avoiding two divergent docker invocations.\n  - `continue-on-error` is per-matrix-cell (false for RunPod, true for\n    OCI while it accumulates a track record). Flip OCI to false once\n    3+ consecutive runs are green.\n  - `fail-fast: false` so a transient on one provider doesn't kill the\n    other.\n  - Artifacts renamed to test-run-metadata-${provider}; validate-spec\n    and validate-shard updated to reference test-run-metadata-runpod\n    (matrixing them follows in the next commit).\n\nThe OCI cell still carries the runtime `pip install skypilot[oci]`\nbridge — that's dropped once the post-merge dev-snapshot rebuild\nbakes in the [oci] extra.\n\nRefs #768\n\n* ci(skypilot): matrix validate-spec over RunPod + OCI\n\nvalidate_spec.py is provider-neutral (reads required fields from\ninput_spec.json structurally), so the only per-cell variation is the\nartifact name. fail-fast: false mirrors the generate matrix; OCI cell\nstays continue-on-error: true while it accumulates a track record.\n\nRefs #768\n\n* ci(skypilot): matrix validate-shard over RunPod + OCI\n\nSame pattern as the prior validate-spec matrixing. The per-shard\ndownload + h5py validation loop already iterates spec.shards[*] and\nparses r2_prefix from the spec, so it works as-is for both providers\nonce the artifact name is parameterized.\n\nAfter this lands, every PR exercises 6 matrix cells: 2 generate, 2\nvalidate-spec, 2 validate-shard.\n\nRefs #768\n\n* fix(skypilot): wire WORKER_GIT_REF through OCI worker container\n\nThe launcher already forwards WORKER_GIT_REF via task.update_envs (it's\nin pipeline.entrypoints.skypilot_launch_smoke._WORKER_ENV_KEYS), but\nthe OCI template's run: block was dropping it on the floor:\n\n  - envs: had no WORKER_GIT_REF placeholder, so SkyPilot's update_envs\n    wouldn't set it on the OCI VM.\n  - The nested `sudo -E docker run ...` lacked `-e WORKER_GIT_REF`, so\n    even if the VM had the value, it wouldn't reach the worker.\n  - The inner bash had no fetch/checkout logic.\n\nResult: OCI matrix cell ran whatever code was baked into the dev-\nsnapshot image, ignoring the PR's commit. RunPod and OCI cells gave\ninconsistent smoke signals on PR CI.\n\nMirror the RunPod template's contract: placeholder in envs:, forward\nvia -e, guarded fetch+checkout (validate ref looks like a 7-40 char\nhex SHA before passing to git, and use safe.directory + FETCH_HEAD to\navoid touching the working tree's index permissions).\n\nRefs #768\n\n* ci(skypilot): assert sed pin substitution and decouple per-provider validators\n\nTwo PR-feedback fixes bundled (both in the same workflow file):\n\n1. Pin step now asserts the sed substitution actually happened (drift-\n   resistance for Copilot review #3178403620). sed silently no-ops when\n   PIN_SEARCH stops matching the template text (e.g. someone reformats\n   the template, or renames the env key). Without this check, CI would\n   proceed against the dev-snapshot default tag instead of the\n   dispatched IMAGE_TAG. Now: fail the workflow if PIN_SEARCH is still\n   present after sed and REPLACE != PIN_SEARCH (PR CI's no-op case),\n   AND fail if REPLACE is not present.\n\n2. validate-spec / validate-shard now run with `if: ${{ !cancelled() }}`\n   so each provider's validator is decoupled from the OTHER provider's\n   generate outcome. Previously, a RunPod transient would skip BOTH\n   validate cells (needs: generate marks the whole job failed) — losing\n   OCI signal for reasons unrelated to OCI. Now: each provider's\n   validator runs as long as the workflow wasn't cancelled; the cell\n   whose generate didn't produce an artifact fails at download-artifact,\n   which is the right per-cell signal.\n\nRefs #768\n\n* refactor(skypilot): address PR #777 review feedback\n\nCode-health BLOCKs:\n- Trim multi-paragraph rationale comments in oci-cpu-template.yaml,\n  runpod-template.yaml, and test-dataset-generation.yml (CLAUDE.md\n  one-line rule). Canonical context lives in design doc / #735 / #776.\n- Extract shared worker run-block to scripts/skypilot_worker_run.sh\n  (RunPod + OCI both invoke). Removes the duplicated git-checkout +\n  python -c os._exit(0) block that had to be edited in two places.\n\nShell-style BLOCKs:\n- Add set -euo pipefail to outer GHA run: blocks (pin step, launch step,\n  validate-spec, validate-shard) and to oci-debug-template.yaml.\n- Replace single-bracket [ ] with [[ ]] in oci-cpu-template / workflow.\n- Move comment block out of \"Pin worker image tag\" run-scalar (CLAUDE.md\n  no-comments-inside-run rule); rationale now sits above the step.\n\nSynth-setter BLOCK:\n- Fix pin-assertion logic: previous logic short-circuited in the default\n  dev-snapshot PR-CI path because REPLACE == PIN_SEARCH made both checks\n  no-ops. Replace with pre-count assertion (PIN_SEARCH must occur\n  exactly once before sed) + post-state checks. Verified locally that\n  drift cases (missing/duplicated PIN_SEARCH) now fail loudly.\n\nTdd-refactor BLOCKs (doc drift caused by this PR):\n- Update docs/reference/github-actions.md: artifact name (now\n  per-provider), test-dataset-generation description, secrets table\n  (six new OCI_* secrets).\n- Update docs/reference/docker.md: per-provider artifact name + gh run\n  download examples.\n\nCode-health WARNs:\n- Drop redundant pin_grep matrix field; final grep prints the rewritten\n  line directly.\n- Consolidate continue-on-error pattern: all three jobs (generate,\n  validate-spec, validate-shard) now read continue_on_error from matrix\n  include for symmetry.\n- Add concurrency group at workflow level (cancel-in-progress) so\n  back-to-back PR pushes don't queue stacked billable RunPod/OCI runs.\n- Hoist the skypilot:cpu-ubuntu-2204 magic literal into matrix include\n  (oci_image_tag) so workflow + template comments share a single source.\n- Mark the WORKER_IMAGE default in oci-cpu-template.yaml as the CI sed\n  pin target (one-line comment) so readers don't mistake it for inert.\n- Bump cluster name to include github.run_attempt — re-running a failed\n  job no longer collides on the launcher's R2 spec key.\n\nShell-style WARNs:\n- Consistent braced quoting in pin step.\n- Separate decl from cmd-sub for R2_BUCKET / R2_PREFIX (SH10).\n\nGHA WARNs:\n- Bump actions/setup-python @v5 → @v6 in test-skypilot-debug.yml\n  (consistency with other workflows).\n- Assert ~/.oci/config region= and ~/.sky/config.yaml compartment_ocid\n  are non-empty before sky check oci — opaque empty-secret failures\n  surface a clear error instead.\n- pip install bridge wraps in explicit failure path; fall-through error\n  message is clearer than the downstream import error.\n\nSynth-setter WARNs:\n- Drop sibling-YAML cross-reference and OCPU/GB restatement comments in\n  oci-debug-template.yaml (CLAUDE.md \"don't bake values into comments\").\n- Update CLAUDE.md project blurb to mention SkyPilot-managed compute\n  (RunPod + OCI), not just RunPod.\n- Add OCI_COMPARTMENT_OCID + image_tag_general step to getting-started\n  §4e so local operators don't hit a missing-compartment failure.\n- Add three-places-in-sync invariant comment next to the skypilot pin\n  in requirements-app.txt.\n\nTdd-refactor WARNs:\n- Update docs/doc-map.yaml SkyPilot-integration block: add OCI templates,\n  scripts/skypilot_worker_run.sh, the new per-provider workflow shape,\n  and bump the requirements-app.txt extras string.\n\nJustified as-is (won't fix, with reasons posted on each thread):\n- oci-debug-template.yaml YAGNI: deferred per the linked comment in\n  test-skypilot-debug.yml until OCI cred-write lands in debug workflow.\n- \"|| true\" on cloud-init wait: deliberate fail-open documented above\n  the block; reviewer marked advisory.\n- git fetch retry: reviewer suggested \"consider\"; not adding.\n- Vast.ai drift in skypilot-compute-integration.md lines 277-278/362:\n  PR description explicitly defers to a follow-up doc PR.\n- Rename runpod-smoke-shard.yaml → smoke-shard.yaml: reviewer's own\n  suggestion is \"post-merge\"; deferred.\n\nRefs #777\nRefs #768\n\n* fix(skypilot): move WORKER_GIT_REF checkout out of shared worker script\n\nThe previous extraction (19db966) put the git-checkout *inside*\nscripts/skypilot_worker_run.sh, but PR CI invokes the script BEFORE\nthe checkout has run — and the dev-snapshot image hasn't been rebuilt\nyet, so the script doesn't exist on disk at invocation time. Worker\nexited 127 (command not found) on both providers.\n\nFix: keep the script for the python heredoc + #735 workaround only;\nmove the WORKER_GIT_REF git checkout back into each template's run:\nblock, before the script invocation. The checkout is what brings\nscripts/skypilot_worker_run.sh into the baked image's working tree\nuntil the next dev-snapshot rebuild bakes it in.\n\nRefs #777\n\n* refactor(skypilot): extract worker checkout logic to its own script\n\nSplits the WORKER_GIT_REF git checkout out of the templates' inline\nbootstrap into scripts/skypilot_worker_checkout.sh. Both compute\ntemplates now share one place for checkout logic too — symmetric with\nthe existing scripts/skypilot_worker_run.sh extraction.\n\nBootstrapping for the not-yet-rebuilt dev-snapshot image: the templates\nfetch the ref's git objects via the image's existing baked clone, then\ngit show <ref>:scripts/skypilot_worker_checkout.sh extracts the script\ncontent into /tmp without touching the working tree. bash that, which\ndoes the actual git checkout, after which scripts/skypilot_worker_run.sh\nis on disk for the worker invocation. No external endpoints involved.\n\nRefs #777\n\n* refactor(skypilot): collapse worker bootstrap into a single script\n\nscripts/skypilot_worker_run.sh now owns the full worker side: optional\ngit checkout to WORKER_GIT_REF + the python invocation with the #735\nos._exit(0) workaround. scripts/skypilot_worker_checkout.sh deleted.\n\nTemplates do the irreducible bootstrap (cd + git config + WORKER_GIT_REF\nformat-check + git fetch) and then `bash <(git show <ref>:scripts/skypilot_worker_run.sh)`,\nwhich streams the script straight from git's object DB through process\nsubstitution. No separate temp-file stage, no second extracted script.\n\nRefs #777\n\n* refactor(skypilot): keep bootstrap inline; script owns python only\n\nscripts/skypilot_worker_run.sh now owns just the python invocation +\n#735 os._exit(0) workaround — the original B2 review concern. Each\ntemplate's run: block keeps the inline bootstrap (cd + git config +\nWORKER_GIT_REF format-check + git fetch + git checkout FETCH_HEAD)\nbecause the script must be on disk for bash to run it, and the\nnot-yet-rebuilt dev-snapshot image doesn't have the script until the\ncheckout itself lands.\n\nReverts c75f7c2 + 1acf114 (separate checkout script + bash <(git show)\nprocess-substitution bootstrap).\n\nRefs #777\n\n* docs(skypilot): address PR #777 Copilot review nits\n\nDoc/comment-only fixes — no behavioral change.\n\n- docs/doc-map.yaml: skypilot_worker_run.sh `covers` no longer claims the\n  script does the WORKER_GIT_REF checkout (it doesn't — templates do).\n  oci-debug-template.yaml `covers` clarifies it's not currently in any\n  CI matrix.\n- docs/design/skypilot-compute-integration.md: replace incorrect \"the\n  run: block is overridden programmatically\" with the actual launcher\n  contract (instantiates Task from YAML, only calls update_envs).\n- configs/compute/oci-debug-template.yaml: header no longer claims the\n  template is \"used by test-skypilot-debug.yml\" — that workflow's matrix\n  is RunPod-only; the OCI cell lands in a follow-up PR.\n- scripts/skypilot_worker_run.sh: collapse stale \"see runpod-template\"\n  pointer to a one-line `# Workaround for #735.` per CLAUDE.md.\n\nRefs #777",
          "timestamp": "2026-05-03T14:43:59-04:00",
          "tree_id": "5c30d432274896b3231e65f249bdfc17660cc8e5",
          "url": "https://github.com/tinaudio/synth-setter/commit/adfd7ab4e1fdba639f812db874e1086dccddc471"
        },
        "date": 1777834637107,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.4276366233825684,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.6477574996650217,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.008227573707699776,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.0026511549949645996,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.1070002317428589,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 11.73689364579999,
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
          "id": "b17b4c2264ca6d279a4edf56250971ec7308d3e0",
          "message": "refactor(pipeline): drop OCI bridge + collapse provider matrix (#803)\n\n`skypilot[runpod,oci]==0.12.0` ships in dev-snapshot via requirements-app.txt\n(Dockerfile installs requirements.txt, which includes requirements-app.txt),\nand #797 made the image rebuild on every merge to main, so the runtime\n\"bridge\" workarounds in test-dataset-generation.yml + skypilot_launch_smoke.py\nare dead weight.\n\nRemoves:\n- Conditional `pip install skypilot[oci]==0.12.0` + `sky check oci` block\n  inside the OCI launch step. `sky check oci` itself stays — useful as a\n  fast-fail probe of the cred file we just wrote.\n- `try/except ImportError` around `from sky.clouds import OCI` in\n  `_override_image_id` (now a direct module-level import inside the\n  function). The matching test_does_not_crash_when_oci_extras_missing\n  test goes with it.\n- Stale comment block in requirements-app.txt referring to the bridge.\n\nFolded in: collapse the dynamic-matrix setup script. Once `oci_image_tag`\nno longer needs to ride along, the matrix only needs the provider name —\ntemplate / cluster prefix / OCI image tag derive cleanly from\n`matrix.provider` via expressions in the consuming step. The `setup` job\nnow publishes a single `providers` JSON array; `generate_matrix`,\n`validate_matrix`, and `has_jobs` outputs are gone, as are the three\n`needs.setup.outputs.has_jobs == 'true'` gates (empty `fromJSON('[]')`\nalready skips a matrix job natively). Setup script: ~60 lines → ~15.\n\nCloses #800.",
          "timestamp": "2026-05-04T19:37:03-04:00",
          "tree_id": "9787aa628b823a54193284279684cc74034c8a2d",
          "url": "https://github.com/tinaudio/synth-setter/commit/b17b4c2264ca6d279a4edf56250971ec7308d3e0"
        },
        "date": 1777938548447,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 1.8589763641357422,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.7423885188996793,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.011311789974570274,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.005052447319030762,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.199535608291626,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 10.601110537300002,
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
          "id": "f940b9f16a7f39029eaca346ab50d1a5b752f150",
          "message": "build(deps): add oci sdk as standalone dep in requirements-app.txt (#825)\n\nCurrently pulled in transitively via skypilot[oci]. Adding it as a\ntop-level dep so we can import it directly without relying on the\nSkyPilot extra's resolution.\n\nRefs #785",
          "timestamp": "2026-05-06T08:58:50-04:00",
          "tree_id": "4e17e8814d1207ee40be40fc9538ae537bb1094b",
          "url": "https://github.com/tinaudio/synth-setter/commit/f940b9f16a7f39029eaca346ab50d1a5b752f150"
        },
        "date": 1778073003256,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 1.4508261680603027,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.0363807892939074,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.005627259612083435,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.0015625357627868652,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.3222322463989258,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 11.968921669400014,
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
          "id": "f32d49d889d4a80a63521c486272667a630d9a1f",
          "message": "feat(param-spec): SURGE_4 mini-example param spec and preset registry (#820)\n\n* internal-feat(vst): add SURGE_4_PARAM_SPEC mini-example and preset registry\n\nAdds a 4-parameter Surge XT spec (SURGE_4_PARAM_SPEC: amp envelope attack,\nfilter cutoff, LFO amplitude/rate) and a preset_paths registry mapping\nparam_spec names to their base preset files. The spec underlies the\nsmoke-test fixture and the predict_vst_audio end-to-end test.\n\n- param_specs[\"surge_4\"] registered alongside surge_xt/surge_simple.\n- preset_paths dict added so future code paths can look up the matching\n  preset by spec name (script wiring lands separately).\n- tests/conftest.py uses surge_4 + presets/surge-mini.vstpreset for the\n  surge fixture; cfg.model.net.d_out now derives from\n  len(param_specs[\"surge_4\"]) instead of being a literal 7 with a\n  comment that would drift when the spec changes.\n- presets/*.fxp gitignored — local-dev learned-model artifacts excluded\n  from version control by default; commit explicitly with git add -f when\n  one becomes a versioned base preset.\n- Docs cross-reference preset_paths from --param-spec-name and\n  --preset-path so users know the two flags should agree.\n\nRefs #811\n\n* test(surge): templatize cfg_surge_xt_global() over param_spec_name\n\nAdds a `param_spec_name` fixture (default \"surge_4\") that drives the surge\nfixtures: `cfg_surge_xt_global` propagates it to `model.net.d_out` and the\n`log_per_param_mse` callback; `surge_xt_smoke_datasets` derives the matching\n`--param_spec` and `--preset_path` from `preset_paths`. Tests can override\nvia indirect parametrization.\n\nAlso plumbs the spec through `predict_vst_audio.py` in the surge train+eval\ne2e test — the script previously defaulted to `--param_spec=surge_xt` while\nthe fixture trained on surge_4, so decode sliced past the end of the\npredicted tensor and crashed MPS CI with \"can only convert an array of size\n1 to a Python scalar\".\n\nAdds a fast cfg-composition test parametrized over surge_4, surge_simple,\nsurge_xt to lock the templating contract for every supported spec.\n\n* test(configs): add surge/test-mps experiment + cfg-equality guard\n\nAdds `configs/experiment/surge/test-mps.yaml`, a Hydra experiment that\nresolves to the same cfg `cfg_surge_xt_global(accelerator=\"mps\",\nparam_spec_name=\"surge_4\")` builds in `tests/conftest.py`. Inherits from\n`surge/base` and overrides `/trainer: mps`, `/callbacks: [default_surge,\neval_surge]` so the fixture's open_dict bake-ins (precision=32-true,\ndeterministic, max_steps=1, batch_size=1, lr_monitor null, etc.) are\nexpressed declaratively.\n\nTo pin the equality contract:\n\n- Extracts `_build_surge_xt_smoke_cfg(accelerator, param_spec_name)` from\n  the existing `cfg_surge_xt_global` fixture so the cfg can be built on\n  any host (the fixture's accelerator gate hardfails non-MPS runners\n  before composing). The fixture is now a thin wrapper.\n- Switches the lr_monitor cleanup from `del` to `= None`. `instantiate_callbacks`\n  skips entries without `_target_`, so runtime behavior is unchanged, and\n  the cfg now matches what `lr_monitor: null` produces on the YAML side.\n- Adds `test_test_mps_yaml_matches_cfg_surge_xt_global` in\n  `tests/test_configs.py`: composes both sides with `resolve=False`,\n  strips volatile top-level keys (`paths`, `hydra`, `task_name`), and\n  asserts deep equality with a human-readable diff on failure.\n\nFuture drift in either the fixture or test-mps.yaml fails fast.\n\n* internal-fix(vst): reformat param_specs/preset_paths dicts and annotate\n\nAddresses Copilot review comments #3192020841 and #3202813835 on PR #820:\n- Multi-line ``param_specs`` dict so ``ruff format`` (line-length 99)\n  stops complaining about the 119-char single-line literal.\n- Type-annotates both registries (``dict[str, ParamSpec]`` and\n  ``dict[str, str]``) so attribute access is type-checked at the call\n  sites and the ``preset_paths`` keys can't drift out of sync with\n  ``param_specs`` without lint surfacing it.\n\nThe third inline comment (#3192020859 — \"comment claims SURGE_4 is used\nby predict_vst_audio test, but the test uses defaults\") was already\nresolved by 2331be5, which plumbs ``--param_spec=surge_4\n--preset_path=presets/surge-mini.vstpreset`` through to the test's\n``predict_vst_audio.py`` invocation. No code change needed there.\n\n* test(surge): pin test_cfg_surge_xt_global_wires_param_spec to cpu\n\nConda CI runs ``pytest -m \"not slow\"`` which includes the (un-slow)\n``test_cfg_surge_xt_global_wires_param_spec`` test. The previous version\nwent through the ``cfg_surge_xt_global`` fixture, which depends on the\nparametrized ``accelerator`` fixture — and that fixture hardfails the\n``[mps-*]`` and ``[gpu-*]`` parametrizations on Linux runners with\n\"MPS not available\" / \"CUDA not available\", failing the conda job.\n\nThe cfg-shape contract this test asserts is accelerator-independent\n(``model.net.d_out`` and ``callbacks.log_per_param_mse.param_spec``\nare set by ``_build_surge_xt_smoke_cfg`` regardless of the ``accelerator``\nargument). Call the builder directly with ``accelerator=\"cpu\"`` and drop\nthe indirect parametrization so only the three param_spec cases run on\nevery CI runner.\n\n* fix(test): loosen SILENCE_PEAK_THRESHOLD in surge train+eval e2e\n\nLowers the ``SILENCE_PEAK_THRESHOLD`` from 1e-4 (~-80 dBFS) to 1e-6\n(~-120 dBFS) in ``test_train_eval_surge_xt``. The previous threshold was\nchosen with the rationale that ``compute_rms`` underflows below 1e-4, but\nthat's not actually true: ``compute_rms``'s NaN risk is the cosine-similarity\ndenominator collapsing to 0, which only happens on bit-zero audio.\n\nSymptom: MPS CI on ``faf2be1`` (and ``5b168b8``) failed with\n``sample_0/pred.wav is silent (peak=3.05e-05)`` even though peak\n3.05e-5 → ~-90 dBFS would not actually underflow downstream metric math.\nThe 1-step-trained smoke model's predicted params, rendered through\nSurge XT, can land in a quiet (but non-silent) region of param space — and\nthe dataset generator runs without a fixed seed, so the trained model and\nits predictions vary run-to-run.\n\nLoosening to 1e-6 keeps the original guard against truly silent (bit-zero)\naudio while letting the legitimate \"trained for one step on a randomly-sampled\n5-clip fixture\" prediction through. The downstream\n``np.isfinite(numeric).all()`` assertion on the metrics CSV remains the\nreal correctness check; the silence threshold is just an early-warning\nfast-fail.",
          "timestamp": "2026-05-07T21:19:07Z",
          "tree_id": "7231f8ac0b6c4223343729093421e1d7bccfbb81",
          "url": "https://github.com/tinaudio/synth-setter/commit/f32d49d889d4a80a63521c486272667a630d9a1f"
        },
        "date": 1778189390386,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.019289255142212,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 3.1009543400164694,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.02723713405430317,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.04806816577911377,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.7032185792922974,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 10.434988717600003,
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
          "id": "81077272c042792a4441e6c11945529cf5f51878",
          "message": "refactor(workflows): split test-dataset-generation; rename launcher (#858)\n\n* refactor(workflows): extract generate-dataset-shards.yaml; rename skypilot_launch_smoke\n\nSplits test-dataset-generation.yml into a thin wrapper plus two reusable\nworkflows:\n\n* `generate-dataset-shards.yaml` — workflow_call only. Owns one provider's\n  launcher invocation (skypilot-local kind setup or runpod/oci in-container\n  launcher). Inputs: provider, dataset_config, image_tag, cluster_name,\n  num_workers, tail, api_server, local, artifact_name. Becomes the official\n  launcher entry point that follow-up PRs (R2-as-coordination, expanded\n  dispatch surface) build on.\n* `validate-dataset-shards.yaml` — workflow_call only. Owns validate-spec\n  + validate-shard jobs.\n* `test-dataset-generation.yml` keeps PR/dispatch triggers (3 inputs\n  unchanged) and computes the provider matrix; calls the two reusables\n  per provider. The docker-only `local` row stays inline (no launcher).\n\nAlso renames `pipeline/entrypoints/skypilot_launch_smoke.py` →\n`skypilot_launch.py` (and the matching test) since the launcher is no\nlonger smoke-specific. Updated all callers: test-skypilot-debug.yml,\ntest-dataset-generation.yml's paths filter, the compute templates'\nheader comments, scripts/sync_worker_checkout.sh, and the doc set.\n\nDeletes obsolete `dataset-generation.yml` (no callers, superseded by the\nunified launcher).\n\nBehavior-preserving — every flag the test wrapper passes to the reusable\nmatches the value today's inline blocks hardcoded (num_workers=1 +\nlocal=true for skypilot-local; defaults elsewhere).\n\nRefs #856\n\n* docs: fix stale dataset-generation.yml references after workflow split\n\nThe doc-drift agent surfaced doc references to the deleted `dataset-generation.yml`\nworkflow that the rename pass missed. Updated four files:\n\n* docs/doc-map.yaml — replace the deleted-workflow pattern with the two new\n  reusables (generate-dataset-shards.yaml + validate-dataset-shards.yaml).\n* docs/reference/github-actions.md — replace the `dataset-generation` row in\n  the Pipeline catalog with rows for both new reusables; refresh the\n  dependency map; replace `dataset-generation` in the Used-by columns of the\n  R2 + W&B secrets table; update the runtime-secrets and\n  mount-as-volume sections.\n* docs/design/storage-provenance-spec.md — update the workflow table row to\n  describe `generate-dataset-shards.yaml` (with its actual input set) and\n  add a sibling row for `validate-dataset-shards.yaml`.\n* .github/workflows/test-vst-slow.yml — update the comment that cites\n  `dataset-generation.yml` as the headless-X11 proof point to point at\n  `generate-dataset-shards.yaml`.\n\nRefs #856",
          "timestamp": "2026-05-08T00:19:24Z",
          "tree_id": "933671bdc7d3ac1f51cc8243a5873a3a13c94db7",
          "url": "https://github.com/tinaudio/synth-setter/commit/81077272c042792a4441e6c11945529cf5f51878"
        },
        "date": 1778200307716,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.966935873031616,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 3.965617674589157,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.011576796881854534,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.022953331470489502,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 2.196953058242798,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 12.920366085599994,
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
          "id": "1eb0ef131e1dbfa4f2b8f2d3c0cede03349dd841",
          "message": "chore(deps): add ruff and pydantic-settings to requirements-app.txt (#894)\n\nruff is already configured (pyproject.toml [tool.ruff*]) and runs in\npre-commit, but isn't a direct dev dep — adding it lets contributors\ninvoke `ruff check` / `ruff format` from editors and the CLI without\nshelling out through the pre-commit harness.\n\npydantic-settings is required for the planned migration in #885\n(generate_vst_dataset CLI auto-generated from RenderConfig fields).\n\nRefs #885",
          "timestamp": "2026-05-11T05:43:37Z",
          "tree_id": "2f32f544751c56028dbb770cad1b7e8194814a3d",
          "url": "https://github.com/tinaudio/synth-setter/commit/1eb0ef131e1dbfa4f2b8f2d3c0cede03349dd841"
        },
        "date": 1778479000585,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 0.994428813457489,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 1.3908210581913591,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.01973121240735054,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.013765692710876465,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 0.5949540138244629,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 13.562267545599997,
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
          "id": "b4830f755d99be27f99869c4cb7067cbe5296864",
          "message": "fix(evaluation): clamp compute_rms denominator to defuse MPS pred.wav silence flake (#899)\n\n* fix(testing): clamp compute_rms denominator to defuse MPS pred.wav silence flake\n\n`test_train_eval_surge_xt[mps]` intermittently failed with `pred.wav is silent`\nbecause MPS has non-deterministic ops and a 1-step-trained model occasionally\npredicted params Surge XT renders below -120 dBFS. The silence assertion\nexisted only as a defensive proxy for `compute_rms`'s `0/0 → NaN` when\n`pred_norm = 0`.\n\nMove the protection into `compute_rms` itself (matches the epsilon-clip\npattern already used in `compute_sot`), so silent pred yields\n`cosine_sim = 0` rather than NaN. Drop the pred.wav silence assertion; keep\nthe target.wav check (target silence would be a real bug).\n\nReturning 0 is within the natural [0, 1] range of cosine similarity for\nnon-negative vectors and correctly penalizes silent predictions; it cannot\nbe gamed upward. No consumer relies on NaN-as-marker.\n\nCloses #898\n\n* fix(testing): short-circuit compute_rms underflow to actually return 0\n\nPer Copilot review on PR #899: the prior commit logged \"returning 0\" on\ndenominator underflow but still computed ``dot/np.clip(denom, 1e-12, None)``,\nwhich only collapsed to 0 when the numerator was exactly 0 (bit-silent pred).\nFor quiet-but-non-zero inputs the clamped division returned an unbounded\nsmall value, contradicting the warning text and the PR's documented intent.\n\nMove the clamp branch to an explicit ``return 0.0`` and add a regression test\nwith ``target = pred = uniform 1e-7`` that would have returned ~0.4 pre-fix.",
          "timestamp": "2026-05-11T02:33:58-04:00",
          "tree_id": "87433cbbb9491ed60b0129b4d29bc731c66dc02a",
          "url": "https://github.com/tinaudio/synth-setter/commit/b4830f755d99be27f99869c4cb7067cbe5296864"
        },
        "date": 1778482079792,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.944422721862793,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 3.720653077214956,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.012544241733849049,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.003511965274810791,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 2.2369418144226074,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 20.105641074799998,
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
          "id": "1181351e9c287dfdd8f4f25e3acb88fd3fe8c3e5",
          "message": "internal-feat(schemas): unify DatasetConfig + DatasetPipelineSpec into DatasetSpec (#887)\n\n* internal-feat(schemas): unify DatasetConfig + DatasetPipelineSpec into DatasetSpec\n\nReplace the prior split between DatasetConfig (YAML-shaped config) and\nDatasetPipelineSpec (runtime-materialized artifact) with a single\nDatasetSpec model whose model_dump_json() is the artifact written to R2.\nRenderer-specific fields move to a nested RenderConfig sub-model.\nRuntime fields (git_sha, is_repo_dirty, created_at, run_id, r2_prefix)\nauto-fill via default_factory when missing and pass through when present\nin JSON-loaded input. shards / num_shards / num_params are computed\ndeterministically as @computed_field cached_properties.\n\nSURGE_XT_RENDERER_VERSION moves out of the schema into RenderConfig as\na config field; the worker still verifies the running plugin version\nmatches the pinned value before rendering.\n\nA legacy YAML loader (load_dataset_spec_yaml) keeps the launcher and\nci.materialize_spec working through this PR; both are removed in a\nfollow-up PR once the entrypoint migrates to @hydra.main.\n\nThe launcher's num_workers knob now lives on the CLI only (default 1);\nthe legacy YAML field is silently ignored.\n\noutput_format remains restricted to \"hdf5\" — wds support lands later\nin the chain.\n\nCloses #886\nPart of #882\n\n* fix(compute): invoke synced docker_entrypoint.py, not stale baked path\n\nThe skypilot templates execed /usr/local/bin/entrypoint.py — the copy baked\ninto the dev-snapshot image. After the pipeline/ → src/pipeline/ relocation\nand src/generate_dataset.py entrypoint move, the in-image script's imports\n('from pipeline.entrypoints.generate_dataset ...') stopped resolving and PR\nworkers failed with ModuleNotFoundError: No module named 'pipeline'.\n\nsync_worker_checkout.sh already updates /home/build/synth-setter to the PR\nhead ref before launch, so invoke scripts/docker_entrypoint.py from the\nsynced checkout instead. The Dockerfile still bakes the same script at\n/usr/local/bin/entrypoint.py for the no-sync (no WORKER_GIT_REF) fallback.\n\nRefs #882\n\n* ci(workflows): install pydantic + pyyaml + omegaconf for validate-spec runner step\n\nAfter unifying DatasetConfig + DatasetPipelineSpec into a single Pydantic\nDatasetSpec, `pipeline.ci.validate_spec` imports\n`pipeline.schemas.spec.DatasetSpec`. The spec module transitively imports\npydantic, pyyaml (for the load_dataset_spec_yaml bridge function), and\nomegaconf — none of which are on the runner's bare Python install.\n\nInstall only those three packages to keep the runner-side env minimal\ninstead of pulling the full requirements.txt (which would drag in torch\nand the rest of the training stack).\n\n* internal-fix(schemas): address Copilot review feedback on PR #887\n\n- pipeline/schemas/spec.py _strip_computed_field_keys: copy the input\n  mapping before popping computed keys so callers holding the dict\n  (logging, retries) see it unchanged (Copilot #3216318943).\n- pipeline/schemas/spec.py legacy YAML bridge: raise ValueError when\n  legacy 'num_shards' disagrees with sum(splits) instead of silently\n  computing a different shard count (Copilot #3216318975).\n- pipeline/schemas/prefix.py make_r2_prefix: strip leading/trailing\n  slashes from prefix_root so 'data/' and '/data' both produce a clean\n  prefix instead of doubled slashes; reject empty-after-strip with a\n  clear error (Copilot #3216319001).\n- pipeline/schemas/spec.py OUTPUT_FORMAT_TO_EXTENSION: rename from the\n  private '_OUTPUT_FORMAT_TO_EXTENSION' and add to __all__ so\n  pipeline.ci.validate_spec is no longer reaching across a private\n  boundary (Copilot #3216319015).\n- pipeline/ci/validate_spec.py: validate output_format in\n  validate_structure and look up extension via .get(...) in\n  validate_test_values so an unknown format produces a structural\n  error instead of a KeyError crash (Copilot #3216319025).\n\nAlso adds the missing docstrings required by interrogate (80% threshold)\non the touched files: the validator/computed-field methods in spec.py\nthat lacked them, and the existing tests in test_dataset_spec.py that\nwere previously undocumented.\n\n* internal-fix(schemas): defer param_specs import inside num_params\n\n`pipeline.schemas.spec` top-level imported `from src.data.vst import param_specs`,\nwhich transitively pulls `src.data.vst.core` → `mido` + `pedalboard`. The\nvalidate-spec runner doesn't install those, so `python -m pipeline.ci.validate_spec`\non the GitHub runner aborts with `ModuleNotFoundError: No module named 'mido'`\nbefore any validation runs.\n\nMove the import inside `num_params`'s body — the only call site. Side effects\nof `src.data.vst.__init__` (mido / pedalboard imports) now only happen when a\nspec's `num_params` is actually evaluated, not when the schema module is\nimported. `validate_spec` only consumes the module-level\n`OUTPUT_FORMAT_TO_EXTENSION` constant, so the deferred import is fine for that\ncode path.\n\n* internal-fix(schemas): address second Copilot review round on PR #887\n\n- pipeline/schemas/spec.py: add `frozen=True` to `DatasetSpec` and\n  `RenderConfig` so the `@cached_property` computed fields (`shards`,\n  `num_shards`, `num_params`) cannot go stale via post-construction field\n  mutation. The internal `_populate_derived_runtime_fields` validator\n  already uses `object.__setattr__`, which bypasses Pydantic's frozen\n  guard, so init-time runtime-field population still works.\n\n- pipeline/entrypoints/generate_dataset.py: replace the misleading\n  \"renderer CLI dispatches on filename suffix\" claim in both\n  `build_generate_args` and `run` docstrings with HDF5-only reality.\n  Drop the `configs/render/<spec>.yaml` / `configs/render/surge_xt.yaml`\n  references in the renderer-version inline comment and error message\n  (this PR keeps materialization in legacy `configs/dataset/*.yaml`;\n  the Hydra `configs/render/` group lands in PR-2).\n\n- src/data/vst/core.py: drop the `configs/render/<spec>.yaml` reference\n  in `extract_renderer_version`'s docstring.",
          "timestamp": "2026-05-11T02:47:18-04:00",
          "tree_id": "f514fd01bc7df5ec77f412c2b21fd73c322a895f",
          "url": "https://github.com/tinaudio/synth-setter/commit/1181351e9c287dfdd8f4f25e3acb88fd3fe8c3e5"
        },
        "date": 1778482854811,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.207057237625122,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.440203178524971,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.029909037053585052,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.004652261734008789,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.1791973114013672,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 15.551011625699994,
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
          "id": "1f6eb7a34e4320b73ed5d42fd72c6a2b86b41167",
          "message": "internal-feat(vst): renderer signatures take RenderConfig + migrate CLI to pydantic-settings (#942)\n\n* internal-feat(vst): renderer signatures take RenderConfig + migrate CLI to pydantic-settings\n\n`make_dataset` now takes a single `render_cfg: RenderConfig` arg in place of\nnine separate kwargs. `param_spec_name` is resolved against the in-process\nregistry inside `make_dataset` (previously the launcher did the lookup);\n`num_samples` comes from `render_cfg.batch_per_shard`. The `fixed_*_params_list`\nkwarg-only args remain for `surge_xt_interactive` and the fixed-params tests.\n\nThe CLI on `generate_vst_dataset.py` is rewritten using pydantic-settings:\n`_GenerateCliArgs(RenderConfig, BaseSettings)` inherits every `RenderConfig`\nfield so the CLI flag set tracks the model automatically. Adding/removing a\nfield on `RenderConfig` extends/shrinks the CLI without a parallel update.\nA new test in `tests/data/vst/test_generate_vst_dataset_cli.py` pins the\nparity invariant.\n\n`pipeline/entrypoints/generate_dataset.py::build_generate_args` derives the\nflag set from `RenderConfig.model_fields` for the same reason — single source\nof truth for the renderer config surface.\n\n`scripts/surge_xt_interactive.py` constructs a `RenderConfig` for its\ncaptured-patches dataset write, with `batch_per_shard` set to the patch count\nand `renderer_version` pulled from the plugin's static metadata.\n\nCloses #885\nCloses #940\n\n* fix(vst): pin CLI flag style + harden round-trip + repair smoke fixture\n\nAddress PR #942 review round 1.\n\n- Pin `cli_kebab_case=False` on `_GenerateCliArgs.model_config` so a future\n  pydantic-settings minor flipping the default to kebab-case can't silently\n  desync the CLI from `build_generate_args`'s underscore output. (Copilot\n  comments on the producer + consumer sides.)\n- Add `test_build_generate_args_roundtrips_through_cli_parser`: builds args\n  with `build_generate_args`, parses them with `CliApp.run`, asserts the\n  reconstructed `RenderConfig` equals the original. Catches flag-spelling\n  and value-coercion drift the field-set parity tests miss. (Copilot\n  round-trip suggestion.)\n- Repair `tests/conftest.py::surge_xt_smoke_datasets`: the subprocess call\n  passed the old positional `num_samples` and `--param_spec`. The new\n  pydantic-settings CLI takes only `data_file` positional and the flag is\n  `--param_spec_name`, plus all other RenderConfig fields are required\n  (no model defaults). The fixture now passes every required flag. (doc-drift\n  follow-up flagging a likely VST-tier CI failure.)\n\nRefs #940\n\n* internal-fix(spec): gate unused train_val_test_seeds with NotImplementedError\n\ntrain_val_test_seeds was a required DatasetSpec field reserved for per-sample\nseeding (#884) but never consumed — yamls, fixtures, and worker payloads were\nforced to carry a dead `[42, 43, 44]` triple. Made it optional (default None)\nwith a model_validator(mode=\"before\") that raises NotImplementedError if any\nnon-None value is set, so the field can't quietly accumulate stale values\nbetween now and #884. Removed the boilerplate from configs/dataset.yaml,\nvalidate_spec's required-keys list, and all eight test fixtures that were\nplumbing the dead value through.\n\nAddresses ktinubu's self-comment on PR #942\n(https://github.com/tinaudio/synth-setter/pull/942#discussion_r3221956327).\n\nRefs #884\n\n* docs(conftest): align surge_xt_smoke_datasets docstring with new CLI flag\n\nThe docstring referenced the old `--param_spec` flag while the\nsubprocess invocation uses `--param_spec_name` (renamed in e73e0f4).",
          "timestamp": "2026-05-11T17:29:19-04:00",
          "tree_id": "f408503e7e68b78ba2dc332a2777ca998cd63abc",
          "url": "https://github.com/tinaudio/synth-setter/commit/1f6eb7a34e4320b73ed5d42fd72c6a2b86b41167"
        },
        "date": 1778535733562,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.1419403553009033,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.837198321595788,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.009967958554625511,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.0210573673248291,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.3858107328414917,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 13.901310724300004,
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
          "id": "4dcb827f87e64b15a95f479416b85698acaa8ff5",
          "message": "refactor(pipeline): relocate pipeline/ → src/pipeline/ (#948)\n\nMirror src/data/, src/models/, etc. by moving the pipeline package\nunder src/. Hoist the dataset generation entrypoint to\nsrc/generate_dataset.py — the entrypoints/ subnamespace dissolves;\nskypilot_launch lives at src/pipeline/skypilot_launch.py.\n\nAll `from pipeline.*` imports rewritten to `from src.pipeline.*`;\n`pipeline.entrypoints.generate_dataset` references rewired to\n`src.generate_dataset`. Workflow YAMLs, compute YAMLs, pyproject.toml\npydoclint excludes, scripts, and doc-map.yaml updated mechanically.\nThe @hydra.main config_path on src/generate_dataset.py drops one level\n(`../configs`) since the file moved closer to repo root.\n\nRefs #882, refs #883.\nCloses #947.",
          "timestamp": "2026-05-11T18:42:50-04:00",
          "tree_id": "0abc2b31c53cf81a8aba2fde58612a36c221a61e",
          "url": "https://github.com/tinaudio/synth-setter/commit/4dcb827f87e64b15a95f479416b85698acaa8ff5"
        },
        "date": 1778540047480,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.2058448791503906,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.790688357651234,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.010479730553925037,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.003342926502227783,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.1365809440612793,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 12.461853984399976,
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
          "id": "94371d6751e61fc27162b18a99f53177d793a376",
          "message": "internal-fix(pipeline): code-health pass on skypilot_launch + pedalboard-free spec import (#963)\n\n* internal-fix(pipeline): code-health pass on skypilot_launch + pedalboard-free spec import\n\n- Defer sky.check import (avoids paying SkyPilot's import cost at module\n  load).\n- Extract _SECRET_WORKER_ENV_KEYS to a module-level constant.\n- Lift _launch_one_rank to module scope for testability.\n- Make src.pipeline.schemas.spec importable in pedalboard-free\n  environments (deferred param_specs import via param_spec_registry).\n- Migrate three call sites to import load_plugin / load_preset /\n  render_params directly from src.data.vst.core.\n\nRefs #882, refs #883.\nCloses #962.\n\n* internal-fix(pipeline): clarify pedalboard-free test class docstring\n\nCopilot review feedback: the original docstring blamed `tests/conftest.py`\nfor the in-session pedalboard load, but after this PR conftest only pulls\nthe pedalboard-free registry. The transitive load actually comes from\nearlier tests that import `src.data.vst.core`. Reword to match.\n\nRefs #962.\n\n* internal-fix(pipeline): tighten docstrings on registry + _SECRET_WORKER_ENV_KEYS\n\nCopilot review feedback:\n- param_spec_registry.py: the docstring still described pedalboard being\n  pulled via `src.data.vst.__init__`'s `from src.data.vst.core import ...`,\n  but `__init__` no longer imports `core` after this PR. Reword to describe\n  the registry as the canonical pedalboard-free entrypoint and call out\n  `src.data.vst.core` (not `__init__`) as the pedalboard pull point.\n- skypilot_launch.py: the comment called the residual subset \"real secrets,\"\n  but `WORKER_GIT_REF` is not a secret. Reword to describe the set by what\n  it actually is — keys not defaulted by `_R2_RCLONE_CONSTANTS`.\n\nRefs #962.",
          "timestamp": "2026-05-12T00:13:13Z",
          "tree_id": "ed0ac5099b6a7b1a776c1b632e367a3b7104bd7e",
          "url": "https://github.com/tinaudio/synth-setter/commit/94371d6751e61fc27162b18a99f53177d793a376"
        },
        "date": 1778545445379,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.277176856994629,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 3.047481337040663,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.01836586557328701,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.007077038288116455,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.2923903465270996,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 10.595799098500004,
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
          "id": "6a4427d4aad89ad22a2a68b010defd7fb68f1c94",
          "message": "docs: convert remaining Google-style docstring sections to Sphinx (#952)\n\n* docs: convert remaining Google-style docstring sections to Sphinx\n\nThe repo's configured docstring style is Sphinx (`[tool.docformatter]` and\n`[tool.pydoclint]` both set to sphinx) and the bulk of the codebase\nalready uses `:param:` / `:return:` / `:raises:`. A handful of files in\n`src/` and `pipeline/` still had Google-style `Args:` / `Returns:` /\n`Raises:` / `Example:` section headers, showing up as DOC003 violations\nin pydoclint's audit (#938).\n\nThis converts them in place, matching the rest of the codebase:\n- `Args:` blocks → one `:param <name>: ...` line per arg\n- `Returns:` blocks → `:return: ...` (dominant form, 7 vs 2 over `:returns:`)\n- `Raises:` blocks → one `:raises <Exc>: ...` line per exception\n- `Example:` block in `src/utils/utils.py` → `.. code-block:: python` directive\n\nNo behavior changes; only docstring text. `scripts/` and `tests/` are out\nof scope per #938's chunked remediation plan.\n\nRefs #938.\n\n* docs(wandb-integration): shift line refs after src/utils/utils.py docstring conversion\n\nThe Google-→-Sphinx conversion in src/utils/utils.py shrank the\ntask_wrapper docstring by one line, shifting code below it up by one.\nTwo line-range refs in wandb-integration.md were now off by one:\n\n- task_wrapper wandb.finish() finally block: 102-107 → 101-106\n- watch_gradients source range: 138-149 → 137-148\n\nCaught by the doc-drift advisory on PR #952. Refs #938.\n\n* docs(skypilot-launch): clarify _run_workers :return: is a list\n\nThe Sphinx-style :return: introduced in the prior commit kept the\noriginal Google-style wording, which read like a scalar even though\nthe function returns list[int]. Spelled out that it's a list with one\nentry per rank, in cluster_names order, and called out the ``-1``\nsentinel and tail-mode behavior referenced elsewhere in the docstring.\n\nRefs #938.",
          "timestamp": "2026-05-11T21:55:39-04:00",
          "tree_id": "5eb292e7917966ef9dd341cb37fe3f84722833df",
          "url": "https://github.com/tinaudio/synth-setter/commit/6a4427d4aad89ad22a2a68b010defd7fb68f1c94"
        },
        "date": 1778551705653,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.572906255722046,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 3.044670149385929,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.010026505216956139,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.037553608417510986,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.8529987335205078,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 15.484668838300001,
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
          "id": "dbb469ece61306fa036351a16a27178e9bb71628",
          "message": "refactor(layout)!: nest src/* under synth_setter/, declare console scripts (#991)\n\n* refactor(layout)!: nest src/* under synth_setter/, declare console scripts\n\nPhase 2 of the PEP src-layout migration (#989, parent #784).\n\nHoists `src/{data,models,utils,metrics.py}` to `src/synth_setter/` and\n`src/{train,eval,generate_dataset}.py` to `src/synth_setter/cli/`. Adds\nthe three `synth-setter-{train,eval,generate-dataset}` console scripts\nvia `[project.scripts]`. Sweeps all `from src.X` imports, `_target_:\nsrc.X` Hydra refs in configs, `python src/X.py` shell invocations in\njobs/ and sweeps/, and prose references in active docs.\n\nOut of scope: `src/pipeline/` (Phase 3, #784), `scripts/`\ndepopulation (Phase 4), `setup.py` deletion (Phase 5); only the\nlegacy `train_command` / `eval_command` `console_scripts` entries\nare dropped here.\n\nBreaking: any external consumer importing `src.{data,models,utils,\nmetrics,train,eval,generate_dataset}` must rewrite to\n`synth_setter.{...}` / `synth_setter.cli.{...}`. Legacy\n`train_command` / `eval_command` scripts are removed (replaced by\n`synth-setter-train` / `synth-setter-eval`).\n\n* test(baseline-configs): bump MODEL_BASELINE to Phase 2 src-layout SHA\n\nThe Phase 2 migration (#989) rewrote every Hydra `_target_:` from `src.X`\nto `synth_setter.X` and switched `jobs/train/{kosc,surge}/train.sh` from\n`python src/train.py` to `python -m synth_setter.cli.train`. The\nresolved Hydra YAMLs from the v0.0.0 baseline therefore literally\ncontain `_target_: src.X` keys while the live tree's resolved YAMLs\ncontain `_target_: synth_setter.X` — a 44-case failure (KOSC) plus a\nparallel SURGE failure pinned at the old tag.\n\nBumping MODEL_BASELINE to the Phase 2 commit captures the migration as\nthe new known-good model-config snapshot. FIXTURE_BASELINE is\nuntouched: the synthetic-fixture scripts under `tests/fixtures/` are\nself-contained and don't reference `src.X`.\n\nRefs #989.\n\n* fix(tests): set PYTHONPATH=src in CI subprocess + workflow probes\n\nThe Phase 2 migration's lazy import inside `DatasetSpec.num_params`\nswitched from `from src.data.vst.param_spec_registry` to\n`from synth_setter.data.vst.param_spec_registry`. That import is\ntriggered by `model_dump_json()` and is exercised by:\n\n  * `tests/pipeline/test_schemas/test_dataset_spec.py` —\n    two tests spawn fresh `sys.executable` subprocesses to verify the\n    spec stays pedalboard-free / launcher-pure. The subprocesses\n    don't inherit pytest's `pythonpath = [\"src\"]`, so\n    `synth_setter` isn't reachable without an editable install.\n    Fixed by passing `PYTHONPATH=<repo>:<repo>/src` to the subprocess\n    `env`.\n\n  * `.github/workflows/test-{mps,gpu,vst-slow}.yml` — the Surge XT\n    plugin-load smoke checks `python -c \"from synth_setter.data.vst.core\n    import load_plugin...\"` against a fresh interpreter (macOS host and\n    Docker container). Fixed by adding `src/` to the PYTHONPATH env\n    var the workflow already exports.\n\nBoth `make test-fast` (556/5) and the full slow `test_compare_baseline_configs`\nsuite (87 passed in 11m17s, including all 44 KOSC + 8 SURGE + 18 predict\ncases) pass locally against the bumped `MODEL_BASELINE=4e08950`.\n\nRefs #989.\n\n* docs(design): update stale src/* refs to synth_setter/* (Phase 2)\n\nSeven design docs (training-pipeline, eval-pipeline, data-pipeline,\nskypilot-compute-integration, storage-provenance-spec, plus the two\n*-implementation-plan docs) referenced legacy `src/train.py`,\n`src/eval.py`, `src/data/`, `src/utils/`, `src/models/` paths and a\n`_target_: src.X` YAML example that no longer resolve after the\nPhase 2 src-layout move.\n\nRewrote file paths to `src/synth_setter/cli/{train,eval}.py` and\n`src/synth_setter/{data,utils,models}/`; rewrote `_target_:` to\n`synth_setter.X`; rewrote `python src/train.py …` CLI invocations\nin code blocks to `python -m synth_setter.cli.train …` per the new\ncanonical surface.\n\nSurfaced by the doc-drift advisory on PR #991.\n\nRefs #989.\n\n* fix(tests): pass PYTHONPATH to VST subprocess in conftest fixture\n\nThe macOS MPS CI workflow does not run `pip install -e .` before\npytest, so the in-process `pythonpath = [\"src\"]` from pyproject.toml\ndoesn't propagate to subprocess.run children. The `surge_xt_smoke_datasets`\nfixture spawns `python src/synth_setter/data/vst/generate_vst_dataset.py`,\nwhich fails with `ModuleNotFoundError: No module named 'synth_setter'`\nwhen its `from synth_setter.data.vst import param_specs` import runs in\nthe child interpreter.\n\nMirrors the `_subprocess_env()` helper already in\n`tests/pipeline/test_schemas/test_dataset_spec.py` (added in b7c62c0):\nset PYTHONPATH=<repo>:<repo>/src on the child env so it can resolve\nboth `src.pipeline.*` and `synth_setter.*` without an install step.\n\nRefs #989.\n\n* fix(ci): install editable package in test workflows, drop PYTHONPATH workaround\n\nThe proper fix for \"subprocesses spawned from tests can't import\nsynth_setter\": install the package via `pip install -e .` in each\nworkflow's setup. Once installed, the import resolves naturally — no\nduplicated `_subprocess_env()` helper, no PYTHONPATH gymnastics.\n\nWorkflows updated: test.yml (3 jobs), test-mps.yml, test-conda.yml.\nEach now installs `synth_setter` as editable after the dependency\ninstall. test-mps.yml's \"Smoke-test Surge XT plugin load\" step drops\nits `PYTHONPATH: src` env which b7c62c0 added as a workaround — also\nno longer needed.\n\nThe `_subprocess_env()` helper in tests/conftest.py and\ntests/pipeline/test_schemas/test_dataset_spec.py is removed entirely.\nThat duplication was a code smell flagged by /repo-review-full as\nBLOCK; the real problem was the missing install step.\n\nAddresses BLOCK findings from review #4276527174:\n  - [code-health] _subprocess_env duplicated across two test files\n  - [gha] test-mps.yml Run MPS tests has no install / no PYTHONPATH\n\nRefs #989.\n\n* chore(review): address PR #991 review feedback round 1\n\n- src/synth_setter/cli/generate_dataset.py: add TODO(#784) above the\n  legacy `src.pipeline.*` import block flagging Phase 3 collapse.\n- tests/test_compare_baseline_configs.py: tighten the MODEL_BASELINE\n  prose to a 2-line pointer to #989 and correct the misleading\n  \"head of the Phase 2 PR\" wording — the SHA is the initial commit\n  of #989, not the head.\n- tests/pipeline/test_entrypoints/test_generate_dataset.py: switch\n  module-docstring header from file path to module form so it\n  doesn't drift if the file moves.\n\nRefs #989\n\n* fix(ci): install synth_setter editable in launcher workflows\n\nPhase 2 src-layout migration moved `synth_setter` from `src/`-on-PYTHONPATH\nto a properly-installed package. Two launcher workflows still invoke\n`python -m src.pipeline.skypilot_launch` (which imports\n`synth_setter.cli.generate_dataset` at module load) without installing the\npackage first, so they hit `ModuleNotFoundError: No module named 'synth_setter'`\nat src/pipeline/skypilot_launch.py:51.\n\nSame fix shape as 64ac16d (test.yml / test-mps.yml / test-conda.yml): add\n`pip install -e .` after the requirements install.\n\n- generate-dataset-shards.yaml: skypilot-local row's \"Install launcher deps\"\n  step. Fixes the PR-blocking `Test Dataset Generation /\n  Run generate_dataset (skypilot-local)` failure on #991.\n- test-skypilot-debug.yml: launcher-runner mode's \"Install launcher deps\"\n  step. workflow_dispatch only, same root cause.\n\nIn-container invocations (runpod / oci rows; launcher-docker mode) don't\nneed a change — the dev-snapshot Dockerfile already does\n`uv pip install --no-deps -e .` at build time.\n\ntest-skypilot-local.yml uses sky.launch directly with no synth_setter\nimports — no fix needed there.\n\nRefs #989\n\n* chore(review): address PR #991 review feedback round 2\n\n- src/synth_setter/cli/{train,eval,generate_dataset}.py: collapse the\n  copy-pasted 15-line rootutils explanatory block (and the variant in\n  generate_dataset.py) to a single one-liner pointing at the rootutils\n  README, per CLAUDE.md comment-hygiene (\"Keep comments terse — typically\n  one short line\").\n- src/synth_setter/cli/eval.py: add a one-line comment on the\n  `mode == \"val\" or mode == \"validate\"` branch documenting that both\n  spellings are accepted for backwards compatibility with older configs.\n- pyproject.toml: drop alignment whitespace on the three\n  `[project.scripts]` entries to match standard TOML formatting.\n- src/__init__.py: rephrase the docstring so it acknowledges that\n  src/pipeline/ is still part of the codebase, not just legacy residue.\n\nRefs #989",
          "timestamp": "2026-05-13T08:22:59-04:00",
          "tree_id": "1467e6d7bbbd20f15b0ad2bf1f5bc45a96b05449",
          "url": "https://github.com/tinaudio/synth-setter/commit/dbb469ece61306fa036351a16a27178e9bb71628"
        },
        "date": 1778675699962,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 4.984283924102783,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 7.311417153775692,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.02956530638039112,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.02873861789703369,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 2.010673999786377,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 10.551055831200006,
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
          "id": "7d8a43877a722e382e76787f28f36e987917c420",
          "message": "refactor(layout)!: nest src/pipeline/ under synth_setter/, drop legacy src/ package (#1001)\n\n* refactor(layout)!: nest src/pipeline/ under synth_setter/, drop legacy src/ package\n\nMove `src/pipeline/` to `src/synth_setter/pipeline/` and remove the residual\n`src/__init__.py` so `src/` now contains only the `synth_setter` package.\n\nSweeps:\n- Python imports: `from src.pipeline.` -> `from synth_setter.pipeline.` across\n  tests/, scripts/, and the self-reference in materialize_spec.py's docstring.\n- YAML / Dockerfile / docs: `python -m src.pipeline.` ->\n  `python -m synth_setter.pipeline.` across .github/workflows, configs/compute,\n  configs/image, and docs/.\n- pyproject.toml [tool.setuptools].packages: drop `src`, `src.pipeline`,\n  `src.pipeline.ci`, `src.pipeline.schemas`; add `synth_setter.pipeline`,\n  `synth_setter.pipeline.ci`, `synth_setter.pipeline.schemas`. Delete the\n  Phase 2 transition comment that explained the dual registration.\n- CLAUDE.md Architecture section: collapse the separate `src/pipeline/` bullet\n  into a sub-bullet under `src/synth_setter/`.\n- tests/conftest.py: update the `RenderConfig` reference comment.\n\n`MODEL_BASELINE` in tests/test_compare_baseline_configs.py is intentionally\nunchanged — the project's stable-baseline-anchor rule applies and the\nresolved-YAML grep confirmed no `_target_: src.pipeline` rewrites surface\nunder the move.\n\nThe `!` flags this as breaking because `import src.pipeline` now raises\n`ModuleNotFoundError`. All in-tree callers have been migrated.\n\nCloses #995\nRefs #784\n\n* fix(ci,docs): pip install -e . in docker-build-validation; address Copilot review\n\nPhase 3 (#995) rewrote `.github/workflows/docker-build-validation.yml` to\ninvoke `python -m synth_setter.pipeline.ci.load_image_config`, but the\nworkflow's setup only installed `pyyaml pydantic` — the `synth_setter`\npackage itself was never installed on the runner. Pre-Phase-3 the call\nworked because `python -m src.pipeline.ci.load_image_config` resolved\nagainst the cwd-relative `src/` directory; post-Phase-3 the principled fix\nis `pip install -e .`, matching the pattern Phase 2 (#991) established\nfor every other CI workflow that spawns Python expecting `synth_setter`.\n\nAlso addresses three inline review findings from Copilot on PR #1001:\n\n- src/synth_setter/cli/generate_dataset.py: drop the `TODO(#784):\n  collapse to synth_setter.pipeline.* once Phase 3 hoists ...` comment;\n  the imports below are already on `synth_setter.pipeline.*` after this\n  PR, so the TODO is satisfied.\n- docs/doc-map.yaml: correct the `covers:` description for\n  `src/synth_setter/pipeline/constants.py` — the module defines only\n  `INPUT_SPEC_FILENAME`, no R2 bucket name constant.\n- docs/design/data-pipeline-implementation-plan.md: repoint the\n  `make_dataset` import example from the non-existent\n  `synth_setter.pipeline.vst` to the actual current location,\n  `synth_setter.data.vst.generate_vst_dataset`.\n\nRefs #995\nRefs #784\n\n* fix(ci): also install pyyaml + pydantic for docker-build-validation\n\nThe previous fix (`pip install -e .`) makes `synth_setter` importable but\ndoesn't pull in `pyyaml` or `pydantic` because neither is a declared\nruntime dependency in `pyproject.toml`. The Phase 3 sweep replaced the\nprior bare `pyyaml pydantic` install with `pip install -e .` alone, which\nre-broke the same step on a different `ModuleNotFoundError`. Pin both\nexplicitly alongside the editable install so the step has a self-contained\nenvironment for `python -m synth_setter.pipeline.ci.load_image_config`.\n\nRefs #995\nRefs #784\n\n* fix(ci,docs): install synth_setter for validate-dataset-shards; fix duplicate stale vst import\n\nTwo follow-up findings from Copilot's review of b9dd27d:\n\n1. .github/workflows/validate-dataset-shards.yaml — the validate-spec\n   job runs `python3 -m synth_setter.pipeline.ci.validate_spec` but its\n   install step only installed pydantic. Same regression as\n   docker-build-validation.yml in b9dd27d. Use `pip install --no-deps -e .`\n   alongside the explicit `pydantic>=2,<3` pin so the runner-side env\n   stays minimal (no torch) but synth_setter is importable. The comment\n   block above the step (which explains why this is a minimal install)\n   is preserved verbatim — the rationale still holds.\n\n2. docs/design/data-pipeline-implementation-plan.md L931 — the\n   \"Assumptions\" section had a second stale reference to the\n   non-existent `synth_setter.pipeline.vst.make_dataset` module that\n   c669164 only fixed at L562. Repointed to the actual current\n   location, `synth_setter.data.vst.generate_vst_dataset.make_dataset`,\n   matching the L562 fix.\n\nRefs #995\nRefs #784\n\n* fix(ci): install synth_setter for spec-materialization host-side validate\n\nThe host-side `Validate spec structure` step in spec-materialization.yml and\nthe `Assert test-specific values` step in test-spec-materialization.yml both\nrun `python3 -m synth_setter.pipeline.ci.validate_spec` outside the docker\ncontainer. Phase 3 made `synth_setter` only importable when installed (PEP\nsrc-layout, sources under src/), so both invocations would raise\nModuleNotFoundError on a fresh runner.\n\nSame fix pattern as c669164 / b9dd27d9 / b0c9cfd: add setup-python +\n`pip install --no-deps -e . \"pydantic>=2,<3\"` before the python invocation.\n--no-deps keeps the host env minimal (torch stays in the image).\n\nAddresses Copilot's suppressed low-confidence comment from review\n4282446908 on .github/workflows/test-spec-materialization.yml:35.\n\nRefs #995\n\n* fix(ci): drop --no-deps from host-side validate_spec installs\n\nThe `pip install --no-deps -e . \"pydantic>=2,<3\"` install pattern used by\nthe three host-side `Validate spec structure` / equivalent steps had a\nsubtle bug: `--no-deps` applies to *every* package in the pip command,\nnot just the editable install. As a result pydantic gets installed but\nits required `pydantic-core` (a separately-shipped C extension) does\nnot. The act-verify CI job caught this on PR #1001 with:\n\n    Successfully installed pydantic-2.13.4 synth-setter-3.0.0\n    ...\n    ModuleNotFoundError: No module named 'pydantic_core'\n\n`--no-deps` was originally added to keep the host env minimal (no torch).\nThe minimal-install goal is already met by `[project].dependencies = []`\nin pyproject.toml — the editable install adds nothing transitively for\nsynth_setter itself. Dropping `--no-deps` lets pydantic pull in its\nrequired pydantic-core, while torch still stays out of the env.\n\nAffects three workflows (each running `python3 -m\nsynth_setter.pipeline.ci.validate_spec` on a host runner, not in the\ndocker image):\n\n- .github/workflows/validate-dataset-shards.yaml\n- .github/workflows/spec-materialization.yml\n- .github/workflows/test-spec-materialization.yml\n\nComment blocks above each install step are updated to explain the\nnon-obvious interaction between `--no-deps` and pydantic's own\ndependency on pydantic-core, so the next refactor doesn't reintroduce\nthe flag.\n\nRefs #995\nRefs #784",
          "timestamp": "2026-05-13T15:20:28Z",
          "tree_id": "01a20ad39e9ec59ede0933d1188b34cee53d1769",
          "url": "https://github.com/tinaudio/synth-setter/commit/7d8a43877a722e382e76787f28f36e987917c420"
        },
        "date": 1778686447017,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 3.4932525157928467,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 4.092073093727231,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.028483038768172264,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.014163613319396973,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 2.1601274013519287,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 20.329702602499992,
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
          "id": "ebd0dfa6c5dc4a7c9b33c75858a1a980b6360cf2",
          "message": "chore(deps): consolidate requirements*.txt into pyproject.toml extras (#1008)\n\n* chore(deps): consolidate requirements*.txt into pyproject.toml extras\n\nMove requirements-torch.txt to [project.optional-dependencies].torch,\nrequirements-app.txt's runtime deps to [project.dependencies], and the\ndev tools (pytest, ruff, pre-commit, pyright, mutmut, pytest-benchmark,\npytest-xdist, hypothesis) to [project.optional-dependencies].dev. Add a\nconvenience [all] extra = [torch,dev]. Every pin (loguru==0.7.3,\nscipy==1.14.1, mutmut==3.5.*, pyright==1.1.408,\nskypilot[runpod,oci]==0.12.0, runpod==1.8.1, click<8.2, pesto-pitch,\ndtw-python, kymatio) is preserved verbatim.\n\nReplace the three requirements*.txt files with their pyproject equivalents\nacross all consumers: Makefile install, docker/ubuntu22_04/Dockerfile\n(now uv pip compile pyproject.toml --extra torch --extra dev so the\n~2.5 GB torch-wheels layer keeps surviving source edits — cache key\nnarrows to pyproject.toml + README.md), .devcontainer/Dockerfile,\nenvironment.yaml (.[dev]), scripts/sync_worker_checkout.sh, and every\nGitHub Actions workflow under .github/workflows/.\n\nTwo workflows that previously installed only pydantic on top of\npip install -e . (when [project.dependencies] was empty) — namely\ntest-spec-materialization.yml and validate-dataset-shards.yaml — switch\nto pip install --no-deps -e . + pip install pydantic so they continue\nto avoid pulling torch, librosa, skypilot, etc.\n\nVerified: uv pip compile --extra torch --extra dev resolves with every\npin honored; editable install dry-run produces the same direct-dep set.\nmake format passes (one pre-existing pyright failure on\ntests/pipeline/test_entrypoints/test_skypilot_launch.py is unrelated).\n\nCloses #533\nCloses #181\n\n* chore(deps): also update tart/macos.pkr.hcl install line\n\nDoc-drift review on PR #1008 caught a missed reference: the Packer\ntemplate's \"Clone the repo, use venv with all runtime deps\" provisioner\nstill ran `uv pip install -r requirements.txt && uv pip install --no-deps\n-e .`, which would break the next Tart image build now that\nrequirements.txt is gone.\n\nCollapse the two lines into the equivalent\n`uv pip install --torch-backend ${var.torch_backend} -e \".[torch,dev]\"`\nso the macOS VM ends up with the same dep set as before (torch backend\nhonored via uv's --torch-backend; project installed editably with the\ntorch and dev extras). docs/getting-started.md already advertises this\nbehavior — this brings the build script in line with the doc.\n\nRefs #533\n\n* docs(getting-started): clarify hydra-core lives in runtime deps, not torch extra\n\nCopilot review on #1008 flagged that the conda parenthetical claimed\nhydra-core ships in the `torch` extra. It actually lives in\n`[project.dependencies]`; the `torch` extra is just torch /\ntorchvision / torchaudio / lightning / torchmetrics. Reword to describe\nboth groups as the pip-only set the conda flow installs.\n\nRefs #533\n\n* ci: switch remaining minimal-install workflows to --no-deps\n\nNow that [project.dependencies] is populated, `pip install -e .`\n(and `uv pip install --system -e .`) drags in the full runtime\ndep set. Switch the two remaining minimal-install workflows to\nthe same `--no-deps` + explicit-deps pattern already used by\ntest-spec-materialization.yml and validate-dataset-shards.yaml:\n\n- spec-materialization.yml: `pip install -e . \"pydantic>=2,<3\"`\n  → `pip install --no-deps -e .` + `pip install \"pydantic>=2,<3\"`.\n  Update the inline rationale comment (it claimed `--no-deps`\n  would skip pydantic-core, which is no longer the reason — the\n  reason is now that the project's runtime deps are heavy).\n- docker-build-validation.yml: `uv pip install --system -e .\n  pyyaml pydantic` → `uv pip install --system --no-deps -e .`\n  + `uv pip install --system pyyaml pydantic`.\n\nRefs #533\n\n* chore(deps): collapse Docker uv pip compile+install into one pass\n\nCI Build-and-push failure on PR #1008 root-caused: the two-step\n`uv pip compile pyproject.toml --extra torch --extra dev → uv pip install\n--torch-backend ${TORCH_BACKEND} -r /tmp/requirements.lock` flow resolved\ntorch against the PyPI index in the compile step (no --torch-backend\nthere), pinning torch==2.12.0 (PyPI). The install step then asked the\ncu128 index for that exact version and got \"No solution found\" because\nthe cu128 index ships CUDA-tagged builds (e.g. 2.7.0+cu128), not the bare\n2.12.0 PyPI version.\n\nDrop the compile indirection entirely and use uv's direct support for\nreading deps out of pyproject.toml: `uv pip install -r pyproject.toml\n--extra torch --extra dev`. This resolves and installs in one pass\nagainst the cu128 index, matching the original requirements.txt flow,\nand removes the cross-index inconsistency.\n\nRefs #533\n\n* chore(deps): keep transitional requirements.txt stub for dev-snapshot bake lag\n\nCI Run-generate_dataset failure on PR #1008 root-cause: the\nskypilot-local worker runs `bash scripts/sync_worker_checkout.sh` from\nthe published `dev-snapshot` image, which was baked from main BEFORE\nthis PR. Bash buffers the script at open-time, so even though\n`git checkout WORKER_GIT_REF` succeeds and rewrites the worker's\nworking tree to this PR's HEAD (deleting requirements.txt in the\nprocess), the bash process is mid-execution of the OLD baked script\nlines. The next line in the old script is `uv pip install -r\nrequirements.txt`, which now errors with `File not found`.\n\nKeep a one-line requirements.txt stub that resolves to `-e .[torch]`\nso the OLD baked script's install still works. Once the dev-snapshot\nimage is rebuilt from main after this PR merges (the next push to\nmain triggers it), the baked script will be the updated one that\ndoes `uv pip install -e \".[torch]\"` directly — at which point this\nfile can be deleted. The stub has a sunset comment naming the\ndeletion criterion.\n\nRefs #533\n\n---------\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T14:21:01-04:00",
          "tree_id": "7321193c266ae8201b5ebe7d3c3e564c99a2dbd5",
          "url": "https://github.com/tinaudio/synth-setter/commit/ebd0dfa6c5dc4a7c9b33c75858a1a980b6360cf2"
        },
        "date": 1778697270834,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.5147171020507812,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 3.720987428314984,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.011726384051144123,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.03645879030227661,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.395329475402832,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 12.414048475200001,
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
          "id": "156164a30795b1ac89baad85ec3bec9ae911b911",
          "message": "internal-feat(schemas): add ShardMetadata + wds row in OUTPUT_FORMAT_TO_EXTENSION (#976)\n\n* internal-feat(schemas): add ShardMetadata + wds row in OUTPUT_FORMAT_TO_EXTENSION\n\nPlates the schema layer for the wds writer landing in PR-13:\n\n- New leaf module src/pipeline/schemas/shard_metadata.py holds the strict,\n  frozen ShardMetadata model (sidecar JSON for the wds tar's metadata.json\n  member). No project imports — consumers on either side of the src ↔\n  src/pipeline boundary can pick it up without forming a launcher-side\n  import cycle through pedalboard.\n- Extend OUTPUT_FORMAT_TO_EXTENSION from {\"hdf5\": \".h5\"} to\n  {\"hdf5\": \".h5\", \"wds\": \".tar\"} and widen DatasetSpec.output_format from\n  Literal[\"hdf5\"] to Literal[\"hdf5\", \"wds\"]. The existing\n  _shard_filenames_match_output_format model_validator now defends both\n  formats.\n\nJoins the existing schemas in the pydoclint exclude list (alongside\nspec.py, prefix.py, image_config.py) per the convention documented at\nthe exclude block — see #938 for the cleanup-as-we-go epic.\n\nInternal-only — no config / launcher / worker changes. PR-13 splits the\nwriter, PR-14 wires --wds-out end-to-end (closes #874).\n\nRefs #975\nPart of #72\n\n* docs(design): sync data-pipeline doc with ShardMetadata + Literal[\"hdf5\", \"wds\"]\n\nApply doc-drift findings from PR #976 review:\n\n- §14.1 spec sketch — drop the \"wds in a later PR\" trailer and widen the\n  Literal to match spec.py's new Literal[\"hdf5\", \"wds\"].\n- §14.7 directory tree — list the new shard_metadata.py leaf module.\n- §7.6 finalize step + §8 WDS shard structure — reference the metadata.json\n  sidecar (one per shard) and point readers at the ShardMetadata model.\n- doc-map.yaml — map src/pipeline/schemas/shard_metadata.py to the\n  data-pipeline design doc so future drift checks catch evolution of the\n  sidecar contract.\n\nRefs #975\nPart of #72\n\n* internal-fix(schemas): tighten ShardMetadata sample_rate type + AST-based leaf-import test\n\nAddresses Copilot review on PR #976:\n\n- ShardMetadata.sample_rate: float → int. The h5py audio attr is written\n  from RenderConfig.sample_rate (int), so the wds sidecar mirrors the\n  canonical type now rather than drifting at the format boundary. Test\n  payloads updated to match.\n- The leaf-module test now parses the module's AST and asserts no\n  ImportFrom/Import nodes targeting src.* — replaces the substring grep,\n  which would have false-failed on a docstring mentioning \"from src.\" and\n  missed alternative import phrasings.\n\nRefs #975\n\n* ci: re-trigger test-dataset-generation after transient VST X-server flake\n\n* internal-fix(schemas): clarify ShardMetadata is not yet read by validate_shard; UTF-8 source read in leaf-import test\n\nAddresses Copilot round 2 on PR #976:\n\n- ShardMetadata docstring + doc-map covers entry no longer claim the sidecar\n  is \"validated on read by validate_shard\". The wds writer and the wds branch\n  of validate_shard land in PR-13; PR-12 only plates the model. Reworded to\n  reflect current behavior (model exists, wiring in PR-13).\n- test_module_has_no_project_imports now uses Path(...).read_text(encoding=\"utf-8\")\n  instead of bare open().read(); shard_metadata.py contains the non-ASCII ↔\n  glyph, so a non-UTF-8 default locale (Windows) would have errored.\n\nRefs #975\n\n* test(pipeline): widen leaf-import check to flag all project-import forms\n\nAddresses Copilot round 3 on PR #976:\n\nThe AST check previously only caught ``import src.x`` and ``from src.x\nimport y``. Bare ``import src``, ``from src import x``, and any relative\n``from .x import y`` would have bypassed it. Now flags:\n\n- ast.Import: alias.name == \"src\" OR alias.name.startswith(\"src.\")\n- ast.ImportFrom: node.level > 0 (any relative import) OR node.module\n  starts with \"src.\" OR node.module == \"src\"\n\nThe wider check enforces the actual contract (no project-internal imports\nthat would form a launcher-side cycle), not a substring of one shape.\n\nRefs #975\n\n* internal-fix(schemas): add range validators on ShardMetadata + clarify leaf-test docstring\n\nAddresses Copilot round 4 on PR #976:\n\n- ShardMetadata now runs a _ranges_must_be_sane model_validator that mirrors\n  RenderConfig._ranges_must_be_sane: velocity ∈ [0, 127], sample_rate > 0,\n  channels >= 1, signal_duration_seconds > 0. The JSON-from-R2 path is a\n  trust boundary, so this catches corrupted/hand-edited sidecars at read\n  time rather than letting nonsensical values reach training. Tests pin\n  each rejection.\n- The leaf-import test docstring no longer claims generate_vst_dataset\n  imports ShardMetadata — that wiring lands in PR-13. Reworded to refer\n  to the future consumer.\n- Add tests/pipeline/test_schemas/test_shard_metadata.py to the pydoclint\n  exclude — its parametrized tests trip DOC101/DOC103 just like the\n  sibling test_dataset_spec.py / test_image_config.py / test_prefix.py\n  (which are all already excluded for the same reason).\n\nRefs #975\n\n* docs(design): clarify staged shards stay HDF5 regardless of output_format\n\nAddresses Copilot round 5 on PR #976. §7.6 hardcodes `.h5 + .valid` for the\nstaged-shard existence check (step 03), the structural-check open (step 04),\nand the promote copy (step 05). Now that `DatasetSpec.output_format` accepts\n`wds`, a casual reader might expect staging to flip to `.tar` for wds specs —\nbut it doesn't: workers always emit HDF5; only finalize's step 08 diverges per\nformat (transcoding to wds on demand). The rationale lives in §8's \"Why\ngeneration stays HDF5 regardless of output format\" but wasn't cross-referenced\nfrom §7.6.\n\nAdds a one-line clarifier at the top of §7.6 pointing readers to the §8 note,\nso the staging-stays-HDF5 contract is explicit without requiring the reader\nto find the other section.\n\nRefs #975\n\n* docs(design): revert §7.6 staged-HDF5 clarifier — conflicts with schema contract\n\nAddresses Copilot round 6 on PR #976. The clarifier added in 027a2df read:\n\"Staged shards are always HDF5 regardless of spec.output_format\". That's\ninternally inconsistent with PR-12's schema, where DatasetSpec.shards\nderives the shard filename from output_format via OUTPUT_FORMAT_TO_EXTENSION\n(wds → .tar). Reverting the clarifier keeps §7.6 matching the only working\ngeneration path today (hdf5); PR-13 will rewrite §7.6 + §8's \"Why generation\nstays HDF5\" section when the wds writer + extension dispatch land.\n\nRefs #975\n\n* docs(design): note §8 design-transition for wds — schema admits wds, writer lands PR-13\n\nAddresses Copilot round 7 on PR #976. Copilot rightly flagged that §8's \"Why\ngeneration stays HDF5 regardless of output format\" claim is inconsistent\nwith the schema's output_format → shard.filename wiring after PR-12. The\ntruth is the design IS changing across PR-12/PR-13: PR-12 widens the spec;\nPR-13 lands the wds writer + extension dispatch and will rewrite §8 to\nmatch the new pipeline shape.\n\nAdds a forward-looking note under §8's \"Why generation stays HDF5\" header\nthat:\n- points readers at §14.1's OUTPUT_FORMAT_TO_EXTENSION mapping,\n- states the eventual behavior (wds workers emit .tar directly),\n- says PR-13 lands the writer + section rewrite,\n- makes clear that on main today the schema admits wds but no writer is\n  wired.\n\nRefs #975\n\n---------\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
          "timestamp": "2026-05-13T16:00:22-04:00",
          "tree_id": "dcc6c23e6dcfb5f6652ca4ee848339819679c7af",
          "url": "https://github.com/tinaudio/synth-setter/commit/156164a30795b1ac89baad85ec3bec9ae911b911"
        },
        "date": 1778703210450,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.403386354446411,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.6680791029147803,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.011606983840465546,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.051624417304992676,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 2.0681002140045166,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 11.838391927400005,
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
          "id": "ad5c853689e93005b8111b8ce96111725e2bb7ba",
          "message": "ci(docker): strip runtime PYTHONPATH from docker runs in workflows (#1017)\n\nPR #647 / #667 fixed setup.py so find_packages exposes the pipeline\npackage without a runtime PYTHONPATH override, and PR #797 wired\ndev-snapshot to rebuild on every push-to-main (merged 2026-05-04) so\nthe in-image package surface tracks main. The temporary\n-e PYTHONPATH=/home/build/synth-setter override added in 3529fae is no\nlonger needed; strip it from all docker run invocations.\n\nRemoves 15 -e PYTHONPATH=... lines across 10 workflow files:\n\n- .github/workflows/docker-build-validation.yml\n- .github/workflows/flush-investigation.yml\n- .github/workflows/generate-dataset-shards.yaml\n- .github/workflows/job-queue.yaml\n- .github/workflows/spec-materialization.yml\n- .github/workflows/test-dataset-generation.yml\n- .github/workflows/test-gpu.yml\n- .github/workflows/test-skypilot-debug.yml\n- .github/workflows/test-vst-slow.yml\n- .github/workflows/validate-dataset-shards.yaml\n\nThe integration check is the smoke tests in docker-build-validation.yml\nand test-dataset-generation.yml; a local docker probe was not run\nbecause docker was not available in the working environment.\n\nCloses #670\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
          "timestamp": "2026-05-13T20:31:26Z",
          "tree_id": "3c8bc483552ff70d508d62080e2457e3ddb06999",
          "url": "https://github.com/tinaudio/synth-setter/commit/ad5c853689e93005b8111b8ce96111725e2bb7ba"
        },
        "date": 1778705086406,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.442899703979492,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 3.2709480833169073,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.018059860914945602,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.005213499069213867,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.75197172164917,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 11.426629607699994,
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
          "id": "e2c780ae3677279f07c832d6f4b8f90faea54558",
          "message": "chore(lint): close P1/P5/P6 pydoclint blind spots (#978)\n\n* chore(lint): close P1/P5/P6 pydoclint blind spots from adversarial probe\n\nThree concrete fixes for the slip categories the #939 adversarial probe\ndocumented. None of P2/P3/P4 are addressed here — those are inherent to\npydoclint and remain Open Questions on #938.\n\nP5 — flip pydoclint native-mode-noqa-location to \"definition\".\nThe CLI default is \"docstring\", so suppressions on the def line were\nsilently inert. Aligning with flake8/ruff convention means\n`# noqa: DOCxxx` next to `def` now does what every developer in this\nstack expects.\n\nP1 — add ruff D102/D103/D107 (missing-docstring rules).\nPydoclint defers \"must have a docstring\" to pydocstyle, which was not\nwired in. Ruff implements the same family. D102 = public method, D103 =\npublic function, D107 = __init__. Per-file-ignores cover the 40 tracked\nfiles that fail today; the list mirrors [tool.pydoclint].exclude.\n\nP6 — CI guard against new defs/classes in pydoclint-excluded files.\nscripts/check_no_new_funcs_in_pydoclint_excluded.py reads the pydoclint\nexclude regex from pyproject.toml and scans the PR diff for `+def`/\n`+class` lines whose file matches it. New tests pin its behaviour on\nsynthetic diffs; wired into code-quality-pr.yaml as a new step.\n\nCONTRIBUTING.md and .github/agents/lint-cleanup.md updated to describe\nthe new ruff D rules, the def-line noqa convention, and the guard.\n\nRefs #938\nRefs #939\n\n* docs(pydoclint): address doc-drift after P1/P5/P6 fixes\n\n- CONTRIBUTING.md: restore ANN001 to the ruff rule list it had been\n  dropped from when D102/D103/D107 were added.\n- CLAUDE.md: replace the inlined ruff rule list with a pointer to\n  [tool.ruff.lint].select so the drift clock does not reset on the next\n  rule addition; mention the new D rules.\n- docs/reference/github-actions.md: code-quality-pr now also runs the\n  pydoclint-excluded-file ratchet; document the new responsibility and\n  the fetch-depth: 0 requirement that goes with it.\n\n* chore(lint): address Copilot review on PR #978\n\n- Remove tests/scripts/test_check_no_new_funcs_in_pydoclint_excluded.py\n  from [tool.pydoclint].exclude and add `# noqa: DOC101,DOC103` to the\n  four test defs that take pytest fixtures. The previous setup made the\n  PR self-fail its own new P6 guard (verified: guard exit=1 against\n  origin/main, 12 findings before this commit; exit=0 after).\n  (comment #3223490913)\n\n- Replace `scripts/**` and `src/data/**` directory globs in\n  [tool.ruff.lint.per-file-ignores] with per-file entries mirroring\n  [tool.pydoclint].exclude. New files under those directories are no\n  longer silently exempt from D102/D103/D107.\n  (comment #3223490899)\n\n- Skip `\\\\ No newline at end of file` diff metadata in the guard's line\n  counter; add a pinning test. Without this, post-marker line numbers\n  in the guard's `path:line: name` report could be off-by-one.\n  (comment #3223490926)\n\n- Add an explicit `tomli; python_version < \"3.11\"` pin to\n  requirements-app.txt. The dep was already transitively available via\n  pytest/runpod, but pinning explicitly removes the fragility of relying\n  on a third-party transitive resolution.\n  (comment #3223490886)\n\nRefs #938\n\n* chore(lint): address Copilot post-push review on PR #978\n\nThree new Copilot comments after the merge from main:\n\n- pyproject.toml: flatten src/models/** for D102/D103/D107 the same way\n  scripts/** and src/data/** were already flattened. Keep ANN001 on the\n  glob (legacy, separate concern) but list each model file explicitly\n  for the D-rules so new files under src/models/ are not silently\n  exempt. (comment #3228263848)\n\n- scripts/check_no_new_funcs_in_pydoclint_excluded.py: fix module\n  docstring drift. The text said nested closures with \"six or more\n  leading spaces\" are ignored, but DEF_OR_CLASS_PATTERN matches 0-4\n  spaces, so anything >=5 is ignored. Rewrote to name the threshold\n  precisely and point at the regex where it lives. (comment #3228263810)\n\n- tests/scripts/test_check_no_new_funcs_in_pydoclint_excluded.py: import\n  the guard via importlib.util.spec_from_file_location instead of\n  mutating sys.path at module import time. Avoids leaking the change\n  into the rest of the test session. (comment #3228263867)\n\n* chore: trigger copilot review\n\nEmpty commit per CLAUDE.md step 6a — Copilot did not re-review 2625abd\nwithin the 15-min SLA and reviewers API rejects copilot-pull-request-reviewer\nas a non-collaborator. Push restarts the readiness loop.\n\n* docs(test): fix Copilot-flagged docstring typo on diff-header test\n\nCopilot review comment #3228504254 on PR #978: the test docstring said\n\"`+-+` headers\" — that is not a real unified-diff marker. The test\nactually guards against `+++ b/file.py` and `--- a/file.py` headers\nbeing mistaken for additions. Updated the docstring to name both\nmarkers correctly.\n\n* chore: resolve main merge conflicts in pydoclint follow-up PR\n\nAgent-Logs-Url: https://github.com/tinaudio/synth-setter/sessions/21c6c7a6-a341-4ed4-8ec7-ab11adad08ee\n\nCo-authored-by: ktinubu <17952332+ktinubu@users.noreply.github.com>\n\n* chore(lint): close P6 ratchet gap exposed by Phase 4 merge\n\nThe Phase 4 layout migration (#1009) moved files into\nsrc/synth_setter/{tools,models,metrics.py}/. The ruff per-file-ignores\nwere updated to mirror the new paths, but [tool.pydoclint].exclude\nwasn't, so 13 files had D102/D103/D107 ignored but were not in the\npydoclint exclude regex — re-opening the same blind spot this PR's\nP6 ratchet was supposed to close.\n\nRestores the maintainer's stated invariant from review round 2\n(comment #3228291103: \"D-rule ignores mirror pydoclint.exclude\")\nby adding the missing entries:\n\n  src/synth_setter/metrics.py\n  src/synth_setter/tools/model_from_wandb.py\n  src/synth_setter/tools/paramspec_to_table.py\n  src/synth_setter/tools/plot_param2tok.py\n  src/synth_setter/tools/sig_perf.py\n  src/synth_setter/models/components/cnn.py\n  src/synth_setter/models/components/embed_pool.py\n  src/synth_setter/models/components/residual_mlp.py\n  src/synth_setter/models/components/vector_field.py\n  src/synth_setter/models/ksin_ff_module.py\n  src/synth_setter/models/surge_ff_module.py\n  src/synth_setter/models/surge_flow_matching_module.py\n  src/synth_setter/models/surge_flowvae_module.py\n\nAfter this change, an adversarial probe (synthetic +def in\nsrc/synth_setter/{metrics,tools/sig_perf,models/components/cnn,\nmodels/ksin_ff_module}.py) makes the guard exit 1 in every case;\nthe 13 existing P6 tests still pass.\n\nRefs #938\n\n---------\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T16:42:04-04:00",
          "tree_id": "1697ba459f4799f7739afd06d9bc413baa440ca0",
          "url": "https://github.com/tinaudio/synth-setter/commit/e2c780ae3677279f07c832d6f4b8f90faea54558"
        },
        "date": 1778705812337,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.0836551189422607,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 2.8851351030170918,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.018047884106636047,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.022361278533935547,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 0.9098212718963623,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 12.997688747899996,
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
          "id": "0a074b9e1a741a4cd6df8c1f7f2a49aaaebc9171",
          "message": "internal-feat(vst): promote writer shape helpers + DATASET_FIELD_NAMES to public (#1025)\n\n* internal-feat(vst): promote writer shape helpers + DATASET_FIELD_NAMES to public\n\nPromote the per-row array names and the audio/mel/param shape calculators\ninside synth_setter.data.vst.generate_vst_dataset to public module-level\nhelpers (DATASET_FIELD_NAMES, audio_dataset_shape, mel_dataset_shape,\nparam_array_dataset_shape) plus the mel-front-end constants and\nmel_hop_length / mel_n_fft / mel_n_frames helpers. make_spectrogram and\ncreate_datasets_and_get_start_idx now call the new helpers; behavior is\nbyte-identical for the existing render configs.\n\nFoundation for the upcoming WDS writer and shard-validator inner-shape\nchecks, which need to share these primitives with the validator side.\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): extract shape primitives to shapes.py to clear code-quality guard\n\nThe first commit added six new top-level helpers (DATASET_FIELD_NAMES,\nmel_hop_length, mel_n_fft, mel_n_frames, audio_dataset_shape,\nmel_dataset_shape, param_array_dataset_shape) and four module-level\nconstants to src/synth_setter/data/vst/generate_vst_dataset.py, which is\non the [tool.pydoclint].exclude list — and the code-quality CI guard\n(scripts/check_no_new_funcs_in_pydoclint_excluded.py, see #938) rejects\nany new top-level def in an excluded file. The preferred fix is to\nremove the source file from the exclude list, but generate_vst_dataset.py\nhas 12+ pre-existing pydoclint violations on neighbouring functions\n(make_spectrogram, generate_sample, make_dataset, _GenerateCliArgs) —\nall out of scope for this foundation PR.\n\nMove the new helpers to a fresh sibling module\nsrc/synth_setter/data/vst/shapes.py that was never on the exclude list,\nso pydoclint runs on it from day one and the guard sees the new defs\nland in an unexcluded file. generate_vst_dataset.py now imports the\nprimitives from the new module; behaviour is unchanged.\n\nAlso addresses the doc-drift advisory on the misleading \"single source\nof truth for the shard validator\" comment — the comment now lives in\nshapes.py's module docstring and hedges the validator/wds writer\nconsumers as \"(planned)\" since validate_shard.py still has its own\nprivate _EXPECTED_DATASETS tuple.\n\nRefs #874\nRefs #882\nRefs #938\n\n* internal-fix(vst): wire DATASET_FIELD_NAMES into the writer's HDF5 dataset names\n\nCopilot review on PR #1025 flagged the prior \"single source of truth\"\ncomment on DATASET_FIELD_NAMES as overpromising: save_samples and\ncreate_datasets_and_get_start_idx still hard-coded \"audio\", \"mel_spec\",\n\"param_array\" as string literals, so the constant was orthogonal to the\nwriter. The first follow-up commit (a2376d0) addressed half of that by\nmoving the constant to shapes.py and softening the comment.\n\nThis commit takes the other half — actually making the constant\nload-bearing on the writer side:\n\n- shapes.py exposes per-field constants AUDIO_FIELD, MEL_SPEC_FIELD,\n  PARAM_ARRAY_FIELD and builds DATASET_FIELD_NAMES from them, so the\n  tuple stays a derived view of the per-field constants.\n- create_datasets_and_get_start_idx now passes AUDIO_FIELD /\n  MEL_SPEC_FIELD / PARAM_ARRAY_FIELD into create_dataset instead of\n  the bare string literals.\n- A new shape-helpers test pins\n  DATASET_FIELD_NAMES == (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD)\n  and the literal triple so renaming any field still forces the\n  validator's expected tuple to update in lockstep.\n\nsave_samples doesn't reference dataset names (it operates on already-\ncreated h5py.Dataset handles), so no change there.\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): pass center=True explicitly to make_spectrogram's librosa call\n\nThe shape helpers in shapes.py (mel_n_frames) document the librosa\ncenter=True framing assumption, but make_spectrogram relied on the\nimplicit librosa default. Pinning center=True keeps the writer and the\n(planned) shard validator aligned on the same framing if librosa ever\nchanges its default.\n\nRefs #1025.\n\n* fix(vst): mel_hop_length raises on sample rates that would yield zero hop\n\nCopilot review on #1025 flagged that `mel_hop_length()` returns 0 when\n`sample_rate < MEL_FRAMES_PER_SECOND` (e.g., 50), and that 0 would later\ntrigger a `ZeroDivisionError` inside `mel_n_frames()`'s\n`audio_length // hop` floor-division. The schema doesn't lower-bound\nsample_rate at this depth, so guard at the leaf helper instead of\nrelying on upstream validation.\n\nRaises `ValueError` at the helper boundary so the failure surfaces with\na clear message instead of an opaque ZeroDivisionError downstream.\nNew test pins the raise.\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): guard mel_n_fft against sample rates that round to n_fft=0\n\nmel_n_fft now raises ValueError when int(0.025 * sample_rate) rounds down\nto 0 (e.g., sample_rate <= 39), mirroring the mel_hop_length guard so the\nfailure surfaces at the leaf helper instead of as an opaque librosa error\ndownstream. Also corrects a stale phrase in the mel_hop_length docstring\nthat still referenced the pre-guard ZeroDivisionError path.\n\n---------\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T22:11:52Z",
          "tree_id": "678e5b4d8aed880c1b20ae014808f476767541c5",
          "url": "https://github.com/tinaudio/synth-setter/commit/0a074b9e1a741a4cd6df8c1f7f2a49aaaebc9171"
        },
        "date": 1778711021540,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.7576282024383545,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 3.067170107215643,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.018825657665729523,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.038400888442993164,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.6210427284240723,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 12.498737114300003,
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
          "id": "0bcf7c6797e0bc4ca4b901537dd779122d3c06bb",
          "message": "ci(testing): wire Codecov gates and consolidate coverage collection (#1031)\n\nSteps 2-4 of the coverage-enforcement roadmap (#14). Builds the gating\ninfrastructure; activation of the Codecov GitHub App and CODECOV_TOKEN\nsecret happen separately (step 1, requires org admin in the UI).\n\n- Collapse the duplicate code-coverage job: every fast-suite leg\n  (ubuntu 3.10, ubuntu 3.11, macos 3.10) now produces a coverage.xml\n  and uploads under flag unit-cpu, instead of a fourth job re-running\n  the same suite.\n- Add [tool.coverage.run] (source, branch=true, parallel=true,\n  relative_files=true, omit) and [tool.coverage.paths] to pyproject.toml\n  so reports from different worktree paths merge cleanly.\n- New codecov.yml: project + patch status checks (informational for the\n  first week so we can observe before blocking), unit-cpu flag, and\n  per-directory component targets (pipeline 90%, models 85%, tools 50%,\n  rest auto). Validated against https://codecov.io/validate.\n- Align make coverage with CI flags (--cov-branch, xml + html reports,\n  same marker filter).\n\nFollow-ups (separate PRs): wire coverage into GPU/MPS/VST/slow workflows\nwith their own flags; add diff-cover as a fallback gate; add the Codecov\nbadge to README; flip status checks from informational to required once\nbaseline numbers stabilize.\n\nRefs #14, #149, #155, #30\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T22:25:09Z",
          "tree_id": "9ff00cb80a88d885f01328da660167e26c0e0cd9",
          "url": "https://github.com/tinaudio/synth-setter/commit/0bcf7c6797e0bc4ca4b901537dd779122d3c06bb"
        },
        "date": 1778711960962,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 2.5336928367614746,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 4.0762998408079145,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.009413882158696651,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.004644632339477539,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.8985748291015625,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 14.586727661099996,
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
          "id": "b2cc2a1fa454f25d83707b539f5c2ba0949546ca",
          "message": "chore(ci): fix broken mutmut sandbox imports + document setup (#1026)\n\n* chore(ci): fix broken mutmut sandbox imports + document setup\n\nmutmut copies only `paths_to_mutate` into `mutants/` and strips the real\n`src/` off `sys.path`, so tests that transitively import un-mutated\nparts of the package (e.g. `synth_setter.cli.generate_dataset`,\n`synth_setter.pipeline.r2_io`) blow up during stats collection with\nImportError. PR #302 worked because it mutated `scripts/` only; the\nPhase 4 widen to `src/synth_setter/{evaluation,tools,pipeline/data}/`\nbroke this path and was never re-verified end-to-end.\n\nAdd `also_copy = [\"src/synth_setter/\"]` so the whole package lands in\nthe sandbox alongside the mutated subdirs, and document the moving\nparts in CLAUDE.md (Commands + a Mutation Testing section) so the next\ntime someone widens `paths_to_mutate` they know to recheck this.\n\nRefs #296\n\n* chore(ci): make mutmut run end-to-end (Linux CI workflow + subprocess fix)\n\nThree changes on top of the import-resolution fix in this PR's first\ncommit:\n\n1. **`tests/pipeline/data/test_stats.py`** — rewrite\n   `test_cli_help_advertises_mask_degenerate_bins_flag` to invoke\n   `_parse_args([\"--help\"])` in-process instead of shelling out via\n   `python -m`. Under `mutmut run`'s stats phase, the subprocess\n   inherited `MUTANT_UNDER_TEST=stats` and the mutated module's\n   trampoline tripped on `mutmut.config is None` in the fresh\n   interpreter, crashing stats collection. In-process avoids that\n   entirely and lets mutations of `_parse_args` actually be exercised\n   by this test (the subprocess form would have always run the\n   un-mutated function).\n\n2. **`.github/workflows/mutmut.yaml`** — new workflow_dispatch + weekly\n   cron job that runs `mutmut run` end-to-end on ubuntu-latest and\n   uploads the `mutants/` meta as an artifact. This is the\n   authoritative end-to-end gate for the `[tool.mutmut]` config.\n   macOS local runs cannot serve as that gate because\n   `tests/conftest.py` imports torch/h5py/hydra into the parent and\n   Apple's fork-safety check then SIGSEGVs every forked child.\n\n3. **`Makefile` + `CLAUDE.md`** — `make mutmut` sets\n   `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` (defensive on macOS,\n   no-op on Linux) and CLAUDE.md's \"Mutation Testing\" section now\n   covers (a) the subprocess pitfall, (b) the macOS caveat, and\n   (c) where the authoritative run lives.\n\nRefs #296\n\n* ci(mutmut): TEMP pull_request trigger to Level-1-verify the workflow on PR #1026 (revert before merge)\n\n* ci(mutmut): drop the temporary pull_request trigger\n\nRun 25829272616 (this branch's first commit with the workflow added)\ncompleted green on ubuntu-latest with the expected mix of statuses\n(🎉 810 killed, 🙁 341 survived, 🫥 1771 no tests, ⏰ 3 timeouts), so\nthe workflow is now Level-1-verified. Restore the trigger surface to\nworkflow_dispatch + weekly cron only.\n\nRefs #296\n\n---------\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T22:32:10Z",
          "tree_id": "3ed3a9d5082b21cb35deb38f66cd68b17c24f256",
          "url": "https://github.com/tinaudio/synth-setter/commit/b2cc2a1fa454f25d83707b539f5c2ba0949546ca"
        },
        "date": 1778712693323,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 1.3334201574325562,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 1.7023566967621446,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.008793829940259457,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.06220012903213501,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.1866557598114014,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 11.761744035699996,
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
          "id": "da1327b625362cc1639518650075b3fe4572c8e9",
          "message": "internal-feat(pipeline): inner-shape checks in validate_shard (#1029)\n\n* internal-feat(vst): promote writer shape helpers + DATASET_FIELD_NAMES to public\n\nPromote the per-row array names and the audio/mel/param shape calculators\ninside synth_setter.data.vst.generate_vst_dataset to public module-level\nhelpers (DATASET_FIELD_NAMES, audio_dataset_shape, mel_dataset_shape,\nparam_array_dataset_shape) plus the mel-front-end constants and\nmel_hop_length / mel_n_fft / mel_n_frames helpers. make_spectrogram and\ncreate_datasets_and_get_start_idx now call the new helpers; behavior is\nbyte-identical for the existing render configs.\n\nFoundation for the upcoming WDS writer and shard-validator inner-shape\nchecks, which need to share these primitives with the validator side.\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): extract shape primitives to shapes.py to clear code-quality guard\n\nThe first commit added six new top-level helpers (DATASET_FIELD_NAMES,\nmel_hop_length, mel_n_fft, mel_n_frames, audio_dataset_shape,\nmel_dataset_shape, param_array_dataset_shape) and four module-level\nconstants to src/synth_setter/data/vst/generate_vst_dataset.py, which is\non the [tool.pydoclint].exclude list — and the code-quality CI guard\n(scripts/check_no_new_funcs_in_pydoclint_excluded.py, see #938) rejects\nany new top-level def in an excluded file. The preferred fix is to\nremove the source file from the exclude list, but generate_vst_dataset.py\nhas 12+ pre-existing pydoclint violations on neighbouring functions\n(make_spectrogram, generate_sample, make_dataset, _GenerateCliArgs) —\nall out of scope for this foundation PR.\n\nMove the new helpers to a fresh sibling module\nsrc/synth_setter/data/vst/shapes.py that was never on the exclude list,\nso pydoclint runs on it from day one and the guard sees the new defs\nland in an unexcluded file. generate_vst_dataset.py now imports the\nprimitives from the new module; behaviour is unchanged.\n\nAlso addresses the doc-drift advisory on the misleading \"single source\nof truth for the shard validator\" comment — the comment now lives in\nshapes.py's module docstring and hedges the validator/wds writer\nconsumers as \"(planned)\" since validate_shard.py still has its own\nprivate _EXPECTED_DATASETS tuple.\n\nRefs #874\nRefs #882\nRefs #938\n\n* internal-fix(vst): wire DATASET_FIELD_NAMES into the writer's HDF5 dataset names\n\nCopilot review on PR #1025 flagged the prior \"single source of truth\"\ncomment on DATASET_FIELD_NAMES as overpromising: save_samples and\ncreate_datasets_and_get_start_idx still hard-coded \"audio\", \"mel_spec\",\n\"param_array\" as string literals, so the constant was orthogonal to the\nwriter. The first follow-up commit (a2376d0) addressed half of that by\nmoving the constant to shapes.py and softening the comment.\n\nThis commit takes the other half — actually making the constant\nload-bearing on the writer side:\n\n- shapes.py exposes per-field constants AUDIO_FIELD, MEL_SPEC_FIELD,\n  PARAM_ARRAY_FIELD and builds DATASET_FIELD_NAMES from them, so the\n  tuple stays a derived view of the per-field constants.\n- create_datasets_and_get_start_idx now passes AUDIO_FIELD /\n  MEL_SPEC_FIELD / PARAM_ARRAY_FIELD into create_dataset instead of\n  the bare string literals.\n- A new shape-helpers test pins\n  DATASET_FIELD_NAMES == (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD)\n  and the literal triple so renaming any field still forces the\n  validator's expected tuple to update in lockstep.\n\nsave_samples doesn't reference dataset names (it operates on already-\ncreated h5py.Dataset handles), so no change there.\n\nRefs #874\nRefs #882\n\n* internal-fix(vst): pass center=True explicitly to make_spectrogram's librosa call\n\nThe shape helpers in shapes.py (mel_n_frames) document the librosa\ncenter=True framing assumption, but make_spectrogram relied on the\nimplicit librosa default. Pinning center=True keeps the writer and the\n(planned) shard validator aligned on the same framing if librosa ever\nchanges its default.\n\nRefs #1025.\n\n* internal-feat(pipeline): inner-shape checks in validate_shard\n\nTightens validate_shard's HDF5 path so every dataset's full ``.shape`` is\nchecked against the writer's source-of-truth shape helpers in\n``synth_setter.data.vst.shapes`` — not just ``shape[0]``. The validator\nnow uses ``DATASET_FIELD_NAMES`` directly (deleting the private\n_EXPECTED_DATASETS mirror) and the new ``_expected_dataset_shapes`` helper\nto derive ``(N, C, time)`` for audio, ``(N, C, n_mels, n_frames)`` for\nmel, and ``(N, num_params)`` for the param array.\n\nA renderer change that drifts the audio / mel / param shapes now fails\nfast at validate time instead of silently shipping mis-shaped shards\ndownstream to training.\n\nHDF5-only; the wds tar branch is PR-E in the WDS port roadmap.\n\nRefs #874\nRefs #882\n\n* chore(pipeline): remove validate_shard from pydoclint excludes\n\nThe previous commit added _expected_dataset_shapes() to validate_shard.py\nwhile that file was on [tool.pydoclint].exclude — tripping the\ncheck_no_new_funcs_in_pydoclint_excluded guard. The guard's preferred\nremediation is to remove the file from the excludes list, which means\nmaking it pydoclint-clean.\n\nAdd sphinx :param: / :returns: sections to _expected_dataset_shapes,\nvalidate_shard, _load_spec, and validate_all_shards_from_r2, then drop\nvalidate_shard.py from the exclude list. Tightens lint coverage as a\nside benefit of the inner-shape work.\n\n* docs(design): update validate_shard description to match inner-shape checks\n\nAfter #1029 (this PR), validate_shard asserts the full per-dataset\n.shape against the writer's shape helpers from\nsynth_setter.data.vst.shapes, not just shape[0] row counts. Updates the\nfile-tree comment in data-pipeline.md to match.\n\nPicks up the post-PR doc-drift advisory.\n\nRefs #874\nRefs #882\n\n---------\n\nCo-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>",
          "timestamp": "2026-05-13T22:53:21Z",
          "tree_id": "28c0e91dffdcb477b7368599004baad700276e52",
          "url": "https://github.com/tinaudio/synth-setter/commit/da1327b625362cc1639518650075b3fe4572c8e9"
        },
        "date": 1778713536196,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "vst-noise-floor-random-preset-replay/multi-scale-spectral-loss-max",
            "value": 1.7549954652786255,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/dtw-aligned-mfcc-distance-max",
            "value": 4.176606944799423,
            "unit": "L1"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/spectral-optimal-transport-max",
            "value": 0.011366226710379124,
            "unit": "Wasserstein"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/rms-envelope-cosine-distance-max",
            "value": 0.01800459623336792,
            "unit": "1-cos"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/mel-spectrogram-mean-absolute-error",
            "value": 1.2564427852630615,
            "unit": "dB"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/num-samples",
            "value": 5,
            "unit": "count"
          },
          {
            "name": "vst-noise-floor-random-preset-replay/wall-clock-seconds-per-render",
            "value": 12.82355604999999,
            "unit": "seconds"
          }
        ]
      }
    ]
  }
}