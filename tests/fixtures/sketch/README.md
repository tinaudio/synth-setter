# Sketch tokenizer equivalence fixtures

Each `sketch_tokens_<sha>.pt` holds `SketchControlTokens` inputs, projection weights, and
train-mode outputs recorded by `make_sketch_tokens_fixture.py` in a worktree checked out at
`<sha>`, before the median-filter augmentation (#3730) existed. `tests/models/test_sketch_tokens.py`
replays the inputs on the current code with the filter at its disabled default and asserts
`torch.equal`.

| Fixture                      | Commit                                     | Grids recorded     | Command (from the worktree at that commit)                                                                                                            |
| ---------------------------- | ------------------------------------------ | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sketch_tokens_1416c2da.pt`  | `1416c2da270ffdc4592a078740b5ea12de421db3` | 32 -> 32           | `PYTHONPATH=$PWD/src uv run python <this dir>/make_sketch_tokens_fixture.py sketch_tokens_1416c2da.pt --frames 32 --commit $(git rev-parse HEAD)`     |
| `sketch_tokens_d37e80c8e.pt` | `d37e80c8ec672c10c491897aa03208c55af26ae7` | 32 -> 32, 64 -> 32 | `PYTHONPATH=$PWD/src uv run python <this dir>/make_sketch_tokens_fixture.py sketch_tokens_d37e80c8e.pt --frames 32 64 --commit $(git rev-parse HEAD)` |

`1416c2da` is the commit the published `flow_sketch_prelim-20261106T155616402Z` benchmark
trained at; it records only the identity 32-frame grid because #3552 (`d37e80c8e`, also
pre-#3730) deliberately changed pitch pooling from max to average on non-identity grids.
`d37e80c8e` is the direct parent of #3730 and covers both grids.
