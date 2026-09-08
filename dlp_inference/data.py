"""LPWM dataset wrapper for the Victor OCAtariDataset (YOLO-style layout).

Source layout
-------------
    <root>/
    ├── images/{train,val,test}/<Game>_<frame_id>.png
    └── labels/{train,val,test}/<Game>_<frame_id>.txt   (YOLO; ignored at train)

Two sampling modes:

* `window_half=0` (default): frames are independent stills. To fit LPWM's
  T-frame interface we treat each `__getitem__` as the same image repeated
  `sample_length` times so the dynamics module still receives a valid
  (T, 3, H, W) clip.

* `window_half=K>0`: each item is the temporal window [t-K, t+K] around the
  center frame t -- (2K+1, 3, H, W), in ascending frame order. Every window
  slot maps to the *nearest* frame (same game) actually present in this
  split, so at sequence/block boundaries the missing neighbours are filled
  by replicating the closest available frame (the center slot always maps
  exactly to t). This mode expects a split where consecutive video frames
  coexist -- use make_temporal_split.py to build one; on the original
  per-frame random split most slots would collapse onto the center.

Returns the same tuple shape as AsterixDataset:
    (video, pos, size, id_, in_camera)
with all GT fields empty (training is unsupervised; labels are only used at
eval time).
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class OCAtariDataset(Dataset):

    def __init__(
        self,
        root,                  # path to the dataset root (the dir with images/ and labels/)
        mode="train",          # "train" | "val" | "test"
        sample_length=9,       # T (= timestep_horizon + 1); ignored if window_half > 0
        image_size=128,
        games=None,            # optional list/set of game names to filter to
        max_frames=None,       # optional cap (smoke testing)
        window_half=0,         # K > 0 -> items are [t-K, t+K] temporal windows
        pad_to=0,              # >0: center the frame on a pad_to^2 zero canvas
                               # before resizing (aspect kept, exact-factor
                               # downscale, no fractional interpolation)
        resize_mode="maxpool", # padded downscale: "maxpool" (2x2 max, the shipped
                               # recipe) or "interp" (exact 2x2 box average)
        palette_spread=False,  # False | path to a palette.json (palette_spread.py):
                               # per-frame bg->black recolor applied to every frame
                               # of this dataset before padding and downscaling
        motion_weight=0.0,     # >0: each item also yields a per-pixel loss-weight
                               # map W = 1 + motion_weight * moved(p), where moved()
                               # is the dilated frame difference to the nearest
                               # +-1 video neighbour (looked up across ALL splits -
                               # pixels only, no labels). Rewards reconstructing
                               # what moves (the ball) without touching the model.
        motion_thresh=8,       # uint8 channel-diff threshold for "moved"
        motion_dilate=5,       # box dilation (native px) around moved pixels
    ):
        super().__init__()
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'valid':
            mode = 'val'
        img_dir = os.path.join(root, 'images', mode)
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f'{img_dir} does not exist (expected Victor layout: '
                f'<root>/images/<split>/<Game>_<id>.png)')

        keep_games = set(games) if games is not None else None
        entries = []
        for fname in os.listdir(img_dir):
            if not fname.endswith('.png'):
                continue
            stem = fname[:-4]
            if '_' not in stem:
                continue
            game, _, frame_id_str = stem.rpartition('_')
            if keep_games is not None and game not in keep_games:
                continue
            try:
                fid = int(frame_id_str)
            except ValueError:
                continue
            entries.append((game, fid, os.path.join(img_dir, fname)))
        entries.sort(key=lambda e: (e[0], e[1]))
        if max_frames is not None:
            entries = entries[:max_frames]

        self.paths       = [e[2] for e in entries]
        self.games       = [e[0] for e in entries]
        self.frame_ids   = [e[1] for e in entries]
        self.window_half = window_half
        self.pad_to = pad_to
        if resize_mode not in ("maxpool", "interp"):
            raise ValueError(f"resize_mode {resize_mode!r}: expected 'maxpool' or 'interp'")
        self.resize_mode = resize_mode
        self.palette_spread = palette_spread
        self._spread = None
        self.motion_weight = float(motion_weight)
        self.motion_thresh = motion_thresh
        self.motion_dilate = motion_dilate
        if self.motion_weight > 0:
            if not pad_to:
                raise ValueError("motion_weight needs pad_to (exact-factor grid)")
            # neighbour lookup across every split: the temporal +-1 frame of a
            # train frame usually landed in val/test (per-frame random split).
            self._nb_paths = {}
            for split in ("train", "val", "test"):
                d = os.path.join(root, "images", split)
                if not os.path.isdir(d):
                    continue
                for fname in os.listdir(d):
                    if not fname.endswith(".png"):
                        continue
                    game, _, fid = fname[:-4].rpartition("_")
                    if keep_games is not None and game not in keep_games:
                        continue
                    try:
                        self._nb_paths[(game, int(fid))] = os.path.join(d, fname)
                    except ValueError:
                        pass
        self.T           = (2 * window_half + 1) if window_half > 0 else sample_length
        self.image_size  = image_size
        self.root        = root
        self.mode        = mode
        if window_half > 0:
            self._frame_index = {(g, f): i for i, (g, f)
                                 in enumerate(zip(self.games, self.frame_ids))}
            # Windowed mode is only meaningful on a temporally-contiguous
            # split (make_temporal_split.py). On the original per-frame
            # random split most neighbours are missing and windows silently
            # collapse onto the center frame -- warn loudly.
            n_next = sum((g, f + 1) in self._frame_index
                         for g, f in zip(self.games, self.frame_ids))
            frac = n_next / max(1, len(self.paths))
            if frac < 0.9:
                print(f"[OCAtariDataset/{mode}] WARNING: window_half="
                      f"{window_half} but only {100 * frac:.0f}% of frames "
                      f"have their t+1 neighbour in this split. Windows will "
                      f"mostly replicate the center frame. Did you point "
                      f"`root` at a non-temporal split? "
                      f"(see make_temporal_split.py)")

        print(f"[OCAtariDataset/{mode}] {len(self.paths)} frames  "
              f"(T={self.T}, window_half={window_half}, "
              f"{image_size}x{image_size}, {len(set(self.games))} games)")

    def __len__(self):
        return len(self.paths)

    def _load_frame(self, idx):
        pil = Image.open(self.paths[idx]).convert('RGB')
        # palette recolor, applied at train time exactly as DLPInference
        # applies it at inference. PIL loads the dataset PNGs in the swapped
        # channel convention while the recolor is defined on true RGB, hence
        # the swap sandwich.
        sp = self._spread_for()
        if sp is not None:
            fr = sp(np.asarray(pil)[..., ::-1])             # -> true RGB -> recolored
            pil = Image.fromarray(np.ascontiguousarray(fr[..., ::-1]))
        if self.pad_to:
            fr = np.asarray(pil)                             # (H, W, 3)
            h, w = fr.shape[:2]
            canvas = np.zeros((self.pad_to, self.pad_to, 3), np.uint8)
            top, left = (self.pad_to - h) // 2, (self.pad_to - w) // 2
            canvas[top:top + h, left:left + w] = fr
            # exact-factor downscale, the same op as DLPInference._preprocess
            f = self.pad_to // self.image_size
            if f <= 1:
                arr = canvas.astype(np.float32) / 255.0
            elif self.resize_mode == "maxpool":
                # 2x2 max per channel: on the recolored frames (black
                # background, bright sprites) every sprite pixel survives the
                # downscale at full contrast instead of being diluted
                arr = canvas.reshape(self.image_size, f, self.image_size,
                                     f, 3).max(axis=(1, 3))
                arr = arr.astype(np.float32) / 255.0
            else:   # "interp": exact box average, matching _preprocess
                # Done in float here rather than with PIL: Image.BILINEAR is
                # NOT a box average (Pillow scales the filter support by the
                # reduction factor, so a 2x bilinear reduction is a 4-tap
                # (1,3,3,1)/8 triangle - a wider blur that smears thin
                # sprites), and Image.BOX rounds back to uint8. This is
                # bit-exact against inference.py's F.interpolate, which at an
                # integer factor of 2 with align_corners=False *is* the 2x2
                # mean, and it stays a true average for any factor f.
                arr = canvas.reshape(self.image_size, f, self.image_size,
                                     f, 3).mean(axis=(1, 3), dtype=np.float32)
                arr = arr / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)    # (3, H, W)
        # legacy squash for unpadded configs (PIL.resize takes (W, H))
        pil = pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0      # (H, W, 3)
        return torch.from_numpy(arr).permute(2, 0, 1)        # (3, H, W)

    def _spread_for(self):
        """The dataset's PaletteSpread (loaded once), or None when disabled."""
        if not self.palette_spread:
            return None
        if self._spread is None:
            from .palette_spread import PaletteSpread
            self._spread = PaletteSpread.from_file(self.palette_spread)
        return self._spread

    def _window_indices(self, idx):
        """Dataset indices for the [t-K, t+K] window around center `idx`.
        Each slot takes the nearest present frame of the same game (ties
        towards the past), so boundary slots replicate the closest edge
        frame and the center slot is always `idx` itself."""
        game, fid = self.games[idx], self.frame_ids[idx]
        K = self.window_half
        present = {d for d in range(-K, K + 1)
                   if (game, fid + d) in self._frame_index}
        out = []
        for d in range(-K, K + 1):
            nearest = min(present, key=lambda p: (abs(p - d), p))
            out.append(self._frame_index[(game, fid + nearest)])
        return out

    def _motion_map(self, idx):
        """(1, image_size, image_size) loss-weight map: 1 everywhere, plus
        motion_weight on pixels that differ from the nearest +-1 neighbour
        frame (dilated). Computed on raw PNGs - a fixed function of the data,
        so unlike mask-derived weights it cannot be gamed by the model. No
        neighbour -> all-ones (no bias, just no boost)."""
        from scipy.ndimage import maximum_filter
        game, fid = self.games[idx], self.frame_ids[idx]
        nb = self._nb_paths.get((game, fid - 1)) or self._nb_paths.get((game, fid + 1))
        H = W = self.image_size
        if nb is None:
            return torch.ones(1, H, W)
        a = np.asarray(Image.open(self.paths[idx]).convert("RGB"), np.int16)
        b = np.asarray(Image.open(nb).convert("RGB"), np.int16)
        m = (np.abs(a - b).max(-1) > self.motion_thresh)
        m = maximum_filter(m, size=self.motion_dilate).astype(np.float32)
        canvas = np.zeros((self.pad_to, self.pad_to), np.float32)
        top, left = (self.pad_to - m.shape[0]) // 2, (self.pad_to - m.shape[1]) // 2
        canvas[top:top + m.shape[0], left:left + m.shape[1]] = m
        f = self.pad_to // self.image_size
        if f > 1:   # same exact box average as the frame itself
            canvas = canvas.reshape(H, f, W, f).mean(axis=(1, 3))
        return torch.from_numpy(1.0 + self.motion_weight * canvas)[None]

    def __getitem__(self, idx):
        if self.window_half > 0:
            frames = [self._load_frame(i) for i in self._window_indices(idx)]
            video = torch.stack(frames, dim=0)               # (2K+1, 3, H, W)
        else:
            frame = self._load_frame(idx)
            video = frame.unsqueeze(0).repeat(self.T, 1, 1, 1)  # (T, 3, H, W)

        pos       = torch.zeros(0)
        size      = torch.zeros(0)
        id_       = torch.zeros(0)
        in_camera = torch.zeros(0)
        if self.motion_weight > 0:
            return video, pos, size, id_, in_camera, self._motion_map(idx)
        return video, pos, size, id_, in_camera
