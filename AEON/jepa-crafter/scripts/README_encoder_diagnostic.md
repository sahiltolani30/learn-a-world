# AEON encoder diagnostic dataset collection

This is a small data-collection utility only. It does not train AEON or change its model.

## Run

From the repository root:

```bash
cd AEON/jepa-crafter
python scripts/collect_encoder_diagnostic.py \
  --env-id CrafterReward-v1 \
  --out artifacts/encoder_diagnostic \
  --seeds 101 102 103 104 \
  --transitions-per-seed 64
```

Expected output:

- `artifacts/encoder_diagnostic/images/*.png`: actual RGB observations from Crafter.
- `transitions.jsonl` and `transitions.csv`: current image, opaque action index, next image, reward, and episode flags.
- `images_index.json`: image index and episode/step metadata.
- `contact_sheet.png`: sampled screenshots for quick visual inspection.
- `collection_metadata.json`: environment, seeds, shapes, counts, and collection notes.

The default target is 256 transitions (64 per seed across four seeds). It may save more than 256 unique images because each transition has a current and next observation.

## Important safeguards

- Uses rendered RGB observations only; environment info dictionaries are not serialized.
- Does not read or save hidden inventory, coordinates, recipes, achievements, or semantic state.
- Action values are saved as opaque integer indices. Do not decode them using hidden environment mappings for the model.
- A transition is recorded before resetting an ended episode, so no transition should bridge two episodes.
- This is a diagnostic dataset, not yet a train/validation benchmark. Split by seed/episode for later evaluation; do not randomly split adjacent frames across train and test.
- The script expects the Crafter package and either Gymnasium or Gym, plus NumPy and Pillow, to be installed in the environment.

After collection, inspect the contact sheet and metadata before running any encoder or predictor tests.
