
# Canonical MLB processed artifact

The paper MLB workflow requires the external file
`ProcessedData-From-GivenFiles.pkl` in this directory. It is intentionally not
tracked because it is 794,830,284 bytes.

Required SHA-256:

```text
4e1bb7e5412b1efce7f0ec08079164a5a85f3ff89bcabaa02a3ee847201a392c
```

Verify with `shasum -a 256` on macOS or `sha256sum` on Linux.

The artifact contains the training-compatible standardization and batter-index
mapping paired with `Environments/Baseball/final_OP`. A freshly downloaded and
refit Statcast dataset is not interchangeable. The runtime validates this file,
the model weights, schema, grid, and mapping against
`HJEEDS/data/baseball_processed_artifact_reference.json` before inference.

`scripts/prepare_baseball_data.py` can rebuild the historical pickle shape for
non-paper exploration only; it cannot reproduce this model-paired artifact.
