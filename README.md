# dlp_inference

Standalone DLP object extraction for Atari: one class, one call, a TensorDict
of objects out. Ships the trained weights for all 33 games (trained for 100 epochs each).

```python
from dlp_inference import DLPInference

model = DLPInference("Asterix")          # any game in new_weights_bg5/ or old_weights/
out = model(frame)                       # (H, W, 3) uint8 RGB game frame

out["position"]              # (N, 2) float - object centers, pixels (x, y)
out["size"]                  # (N, 2) float - object extents, pixels (w, h)
out["bbox"]                  # (N, 4) float - (x1, y1, x2, y2), pixels
out["confidence"]            # (N,)   float - obj_on in [0, 1]
out["depth"]                 # (N,)   float - relative occlusion depth
out["embedding"]             # (N, 8) float - per-object appearance latent
out["background_embedding"]  # (5,)   float - background latent (z_bg)
                             # (old_weights checkpoints: 5-d embedding)
```

A sequence works too - `model(frames)` with a `(T, H, W, 3)` array or a list
of frames returns a list of TensorDicts, one per frame (frames are processed
as a batch; the model is per-frame, no temporal state).

Visual check of any result:

```python
img = model.visualize(frame)                     # runs inference + draws
img = model.visualize(frame, out)                # reuse an existing result
model.visualize(frame, save_path="objects.png")  # also writes the PNG
```

returns the upscaled annotated RGB image (per-object colored boxes, center
dots, confidence labels). Also importable standalone:
`from dlp_inference import visualize`.

## Tracking

Hungarian-matching tracker that assigns
persistent ids to the detections across consecutive frames:

```python
from dlp_inference import DLPInference
from dlp_inference.tracking import HungarianTracker

model = DLPInference("Asterix")
tracker = HungarianTracker(frame_hw=(210, 160))   # (H, W) of your frames

for frame in frames:                              # consecutive frames!
    tracked = tracker(model(frame))               # TensorDict in -> out
    tracked["id"]            # (N,) int64 - persistent object ids
```

The output TensorDict has exactly the input attributes plus `"id"`.

How it works: each detection is matched to the active tracks by Hungarian
assignment on `0.3 * cosine_dist(embeddings) + 0.7 * center_distance`, gated
to |dy|, |dx| <= 0.2 of the frame (hence `frame_hw`). Unmatched detections
start new tracks; tracks unseen for `max_age=3` frames die; when a track
re-matches across a gap, the missed frames are back-filled into
`tracker.history` with linearly interpolated boxes (`"interpolated": True`).
`new_episode=True` resets active tracks while keeping ids globally unique.
Tunables: `cost_thresh` (match acceptance, default 0.5), `w_feat`
(appearance weight, 0.3), `max_age`, `max_dy`/`max_dx`.

Demo (longest test block -> mp4 with id-colored boxes):

```bash
python test_tracking.py --game Asterix    # -> tracking_test/<game>_hungarian.mp4
```

## Notes

- **Input**: RGB frames exactly as the emulator produces them (OCAtari
  `obs_mode="ori"`, `ale.getScreenRGB()`), any resolution - resized to the
  model's 128x128 internally, outputs mapped back to input pixels. The
  training-data channel quirk is handled inside; never pre-swap channels.
- **Confidence gate**: objects with `obj_on <= conf_thresh` (default 0.5) are
  dropped. `model(frame, conf_thresh=0.0)` returns all 90 particles.
- **Geometry source**: by default (`tight_boxes=True`) the per-particle
  alpha masks are decoded and each box is fitted tightly around the pixels
  the particle owns (per-pixel argmax over masks); `position`/`size` are the
  box center/extent. `tight_boxes=False` (constructor or per call) skips the
  decoder and uses the particle scale latent instead - roughly 2x faster,
  but boxes reflect the glimpse extent and run larger.
- **Per-game frame transforms** are applied automatically: Boxing's
  checkpoint was trained on recolored frames (near-black pixels -> red, so
  the black boxer stops being absorbed into the background), and
  `DLPInference("Boxing")` recolors incoming frames itself - always pass raw
  emulator frames. Registry: `dlp_inference/transforms.py`
  (`GAME_TRANSFORMS`); the transform is idempotent, so pre-recolored frames
  are fine too.
- DLP also detects HUD elements (score digits, lives) since it is fully
  unsupervised - filter by position if you don't want them.
- Weights layout: `new_weights_bg5/<Game>` (current recipe: pad 256, z_obj 8, z_bg 5) with fallback to `old_weights/<Game>` (original 34 checkpoints); add a new game
  by dropping a compatible run dir pair there. `list_games()` enumerates.

## Training a new game

```bash
python -m dlp_inference.train --game MyGame --root /path/to/dataset
```

Fully unsupervised - the dataset just needs frames in the OCAtari-PNG layout
`<root>/images/{train,val}/<Game>_<idx>.png` (128px training resolution is
handled internally). Writes `new_weights_bg5/<Game>/{hparams.json, best.pth}`, so the
new game is immediately available to `DLPInference`. Defaults reproduce the
shipped checkpoints (100 epochs, batch 8, Adam 2e-4; hours on a recent GPU);
`--epochs 5 --max-frames 400` gives a quick smoke run, `--out` redirects the
output elsewhere. Hyperparameters live in `dlp_inference/config_default.json`.

## Requirements

`pip install -r requirements.txt` (core: torch, numpy, tensordict,
opencv-python; the rest is pulled in by the vendored model code).

Model code under `dlp_inference/model/` is vendored from the
DLPv3 implementation (encoder/decoder; only the encoder is used here).
