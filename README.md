# dlp_inference

Standalone DLP object extraction for Atari: one class, one call, a TensorDict
of objects out. Ships trained weights for all 33 games of the OCAtari dataset
(100 epochs each) under `weights/<Game>/`.

```python
from dlp_inference import DLPInference

model = DLPInference("Asterix")          # any game in weights/ (list_games())
out = model(frame)                       # (H, W, 3) uint8 RGB game frame

out["position"]              # (N, 2) float - object centers, pixels (x, y)
out["size"]                  # (N, 2) float - object extents, pixels (w, h)
out["bbox"]                  # (N, 4) float - (x1, y1, x2, y2), pixels
out["confidence"]            # (N,)   float - obj_on in [0, 1]
out["depth"]                 # (N,)   float - relative occlusion depth
out["embedding"]             # (N, 8) float - per-object appearance latent
out["background_embedding"]  # (5,)   float - background latent (z_bg)
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

## The recipe

Every checkpoint was trained with the same, fully unsupervised recipe
(`dlp_inference/config_default.json`), and `DLPInference` reproduces its
preprocessing exactly from the files stored next to the weights:

1. **Recolor** (`palette_spread.py`, `weights/<Game>/palette.json`). Atari
   games use a handful of exact colors, and the model's loss is a plain RGB
   MSE, so what a missed sprite costs is its RGB distance to the background
   it is painted over. The palette is built from the game's training frames
   (pixel statistics only, no labels, no per-game rule): per frame the most
   frequent color becomes black, and every other color - black included when
   it is not the frame's dominant color, e.g. Boxing's black boxer - is
   mapped onto the bright faces of the RGB cube, smallest objects first, on
   hue directions kept apart from the colors that share a glimpse window
   with them.
2. **Downscale.** The recolored 210x160 frame is zero-padded to 256x256 and
   reduced to 128x128 by a 2x2 **max-pool**: on a black background every
   sprite pixel survives at full contrast instead of being averaged away,
   which is what makes 1-4 px objects recoverable at all.
3. **Model.** DLPv3 with 90 particles, 8-d object latents, a 5-d background
   latent (`z_bg`), pixel-sum MSE reconstruction, 100 epochs of Adam 2e-4,
   batch 8.

On the size-bucketed detection metrics (`eval/`) this recipe gives the best
small-object recall of every variant we trained (boxes of at most 100 px on
the native frame, IoU 0.25), at the price of precision on games with large
moving structure; see the notes below.

## Notes

- **Input**: RGB frames exactly as the emulator produces them (OCAtari
  `obs_mode="ori"`, `ale.getScreenRGB()`), any resolution - padded and
  downscaled to the model's 128x128 internally, outputs mapped back to input
  pixels. The recolor and the training-data channel quirk are handled
  inside; never pre-swap channels or pre-recolor.
- **Confidence gate**: objects with `obj_on <= conf_thresh` (default 0.5) are
  dropped. `model(frame, conf_thresh=0.0)` returns all 90 particles.
- **Geometry source**: by default (`tight_boxes=True`) the per-particle
  alpha masks are decoded, each input pixel is assigned to the particle
  whose alpha wins there, and the box is the pixel-exact extent of the
  non-black pixels the particle owns in the recolored input (the background
  is exactly black after the recolor, so boxes hug the sprite pixels and do
  not flicker with the soft alpha halo; a particle owning no sprite pixel is
  dropped). `position`/`size` are the box center/extent. `tight_boxes=False`
  (constructor or per call) skips the decoder and uses the particle scale
  latent instead - roughly 2x faster, looser boxes.
- DLP also detects HUD elements (score digits, lives) since it is fully
  unsupervised - filter by position if you don't want them.
- **Weights layout**: `weights/<Game>/{best.pth, hparams.json, palette.json}`.
  `hparams.json` records the downscale (`resize_mode`) and whether a palette
  applies (`palette_spread`), so a checkpoint trained with other settings
  (`--override`) is loaded correctly; add a game by dropping such a
  directory there. `list_games()` enumerates.

## Tracking

Hungarian-matching tracker that assigns persistent ids to the detections
across consecutive frames:

```python
from dlp_inference import DLPInference
from dlp_inference.tracking import HungarianTracker

model = DLPInference("Asterix")
tracker = HungarianTracker(frame_hw=(210, 160))   # (H, W) of your frames

for frame in frames:                              # consecutive frames!
    tracked = tracker(model(frame))               # TensorDict in -> out
    tracked["id"]            # (N,) int64 - persistent object ids
```

The output TensorDict has exactly the input attributes plus `"id"`. Each
detection is matched to the active tracks by Hungarian assignment on
`0.3 * cosine_dist(embeddings) + 0.7 * center_distance`, gated to
|dy|, |dx| <= 0.12 of the frame (25 px vertically and 19 px horizontally at
210x160, hence `frame_hw`). Unmatched detections start
new tracks; tracks unseen for `max_age=3` frames die; when a track re-matches
across a gap, the missed frames are back-filled into `tracker.history` with
linearly interpolated boxes (`"interpolated": True`). `new_episode=True`
resets active tracks while keeping ids globally unique. Tunables:
`cost_thresh` (match acceptance, default 0.5), `w_feat` (appearance weight,
0.3), `max_age`, `max_dy`/`max_dx`.

## Training a new game

```bash
python -m dlp_inference.train --game MyGame --root /path/to/dataset
```

The dataset just needs frames in the OCAtari-PNG layout
`<root>/images/{train,val}/<Game>_<idx>.png` (labels are not used). The run
builds the game's palette from `images/train`, trains with the recipe above
and writes `weights/<Game>/{hparams.json, palette.json, best.pth}` (best
validation loss), so the new game is immediately available to
`DLPInference`. About 2h40 per game on an H100 (`hpc_train.sbatch` is the
Slurm array job used for the shipped weights, one game per task);
`--epochs 5 --max-frames 400` gives a quick smoke run, `--out` redirects the
output, and any config key can be changed with `--override key=value`
(e.g. `resize_mode=interp` for the plain box average, `palette_spread=false`
for raw frames).

To look at a game's palette before training:

```bash
python -m dlp_inference.palette_spread --root /path/to/dataset --games MyGame --viz
#  -> palettes/MyGame.json, palette_viz/MyGame.png (swatches + before/after frames)
```

## Evaluation

`eval/eval_fgari.py` (foreground ARI of the particle masks against the
labels) and `eval/eval_prf.py` (precision / recall / F1 at IoU 0.25 and 0.5,
overall and per ground-truth box size: tiny <= 24 px, 25-100, 101-400,
> 400 px on the native frame) write one JSON per game under
`eval/results*/<weights-root>/`; `eval/compare_all.py` merges any number of
weight roots into one CSV. For small objects, read the per-bucket recall at
IoU 0.25 (did a particle land on it) next to the same bucket's precision;
FG-ARI is pixel-weighted and blind to false positives.

```bash
python eval/eval_fgari.py --game Asterix --dataset /path/to/dataset
python eval/eval_prf.py   --game Asterix --dataset /path/to/dataset
python eval/compare_all.py weights            # -> eval/compare_all.csv
```

## Requirements

`pip install -r requirements.txt` (core: torch, numpy, tensordict,
opencv-python; the rest is pulled in by the vendored model code).

Model code under `dlp_inference/model/` is vendored from the DLPv3
implementation (encoder/decoder; only the encoder runs at inference).
