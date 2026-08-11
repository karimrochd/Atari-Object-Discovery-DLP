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
        # per-game frame transform (e.g. Boxing black->red recolor), applied
        # at train time exactly as DLPInference applies it at inference.
        # PIL loads the dataset PNGs in the swapped channel convention while
        # the transforms are defined on true RGB, hence the swap sandwich
        # (recolor_black itself is order-invariant, but the paint color has
        # an orientation).
        from .transforms import GAME_TRANSFORMS
        tf = GAME_TRANSFORMS.get(self.games[idx])
        if tf is not None:
            fr = np.asarray(pil)
            pil = Image.fromarray(tf(fr[..., ::-1])[..., ::-1])
        if self.pad_to:
            fr = np.asarray(pil)                             # (H, W, 3)
            h, w = fr.shape[:2]
            canvas = np.zeros((self.pad_to, self.pad_to, 3), np.uint8)
            top, left = (self.pad_to - h) // 2, (self.pad_to - w) // 2
            canvas[top:top + h, left:left + w] = fr
            pil = Image.fromarray(canvas)
        # PIL.resize takes (W, H). With pad_to=256 -> 128 this is an exact
        # 2x box average (no fractional interpolation).
        pil = pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0      # (H, W, 3)
        return torch.from_numpy(arr).permute(2, 0, 1)        # (3, H, W)

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
        return video, pos, size, id_, in_camera
