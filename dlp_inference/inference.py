"""Ready-to-use DLP inference: game frame(s) in, object TensorDict out.

    from dlp_inference import DLPInference
    model = DLPInference("Asterix")
    out = model(frame)              # frame: (H, W, 3) uint8 RGB game image
    out["position"]                 # (N, 2) object centers, pixels (x, y)
    out["size"]                     # (N, 2) object extents, pixels (w, h)
    out["bbox"]                     # (N, 4) pixel boxes (x1, y1, x2, y2)
    out["embedding"]                # (N, D) per-object appearance latent
    out["confidence"]               # (N,)   obj_on in [0, 1]
    out["depth"]                    # (N,)   relative occlusion depth
    out["background_embedding"]     # (D_bg,) background latent

Frames are expected in RGB exactly as the emulator (OCAtari / ALE
``obs_mode="ori"``) produces them, any resolution (natively 210x160); the
channel-order quirk of the training data is handled internally. A (T, H, W, 3)
array or a list of frames returns a list of TensorDicts, one per frame.

Inference is encoder-only (no decoding), deterministic (posterior means).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .model.models import DLP

# weights/<Game>/{best.pth, hparams.json, palette.json}: the shipped recipe
# (pad 256 -> 2x2 max-pool to 128, z_obj 8, z_bg 5, per-frame bg->black recolor)
WEIGHTS_ROOT = Path(__file__).resolve().parents[1] / "weights"


def _resolve_game_dir(game, weights_root) -> Path:
    return Path(weights_root) / game

_PALETTE = [(80, 220, 80), (250, 120, 120), (110, 170, 255), (255, 200, 80),
            (220, 120, 255), (120, 240, 240), (255, 150, 60), (170, 255, 120),
            (255, 120, 190), (150, 150, 255)]


def visualize(frame: np.ndarray, out: "TensorDict", upscale: float = 4.0,
              show_conf: bool = True, save_path=None) -> np.ndarray:
    """Draw the objects of an inference result onto the frame.

    frame: the (H, W, 3) uint8 RGB image that produced ``out``.
    Returns the annotated RGB image (upscaled); optionally writes it to
    ``save_path`` (PNG/JPG).
    """
    import cv2

    h, w = frame.shape[:2]
    img = cv2.resize(frame, (int(w * upscale), int(h * upscale)),
                     interpolation=cv2.INTER_NEAREST)
    bbox = out["bbox"].detach().cpu().numpy() * upscale
    pos = out["position"].detach().cpu().numpy() * upscale
    conf = out["confidence"].detach().cpu().numpy()

    for i in range(len(bbox)):
        color = _PALETTE[i % len(_PALETTE)]
        x1, y1, x2, y2 = bbox[i].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, tuple(pos[i].astype(int)), 3, color, -1)
        if show_conf:
            label = f"{conf[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.4, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 1)

    if save_path is not None:
        cv2.imwrite(str(save_path), img[..., ::-1])  # cv2 writes BGR
    return img


def list_games(weights_root=WEIGHTS_ROOT) -> List[str]:
    root = Path(weights_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "best.pth").exists())


def _build_model(cfg) -> DLP:
    return DLP(
        cdim=cfg["ch"],
        image_size=cfg["image_size"],
        normalize_rgb=cfg["normalize_rgb"],
        n_kp_per_patch=cfg["n_kp_per_patch"],
        patch_size=cfg["patch_size"],
        anchor_s=cfg["anchor_s"],
        n_kp_enc=cfg["n_kp_enc"],
        n_kp_prior=cfg["n_kp_prior"],
        pad_mode=cfg["pad_mode"],
        dropout=cfg["dropout"],
        features_dist=cfg.get("features_dist", "gauss"),
        learned_feature_dim=cfg["learned_feature_dim"],
        learned_bg_feature_dim=cfg.get("learned_bg_feature_dim",
                                       cfg["learned_feature_dim"]),
        n_fg_categories=cfg.get("n_fg_categories", 8),
        n_fg_classes=cfg.get("n_fg_classes", 4),
        n_bg_categories=cfg.get("n_bg_categories", 4),
        n_bg_classes=cfg.get("n_bg_classes", 4),
        scale_std=cfg["scale_std"],
        offset_std=cfg["offset_std"],
        obj_on_alpha=cfg["obj_on_alpha"],
        obj_on_beta=cfg["obj_on_beta"],
        obj_res_from_fc=cfg["obj_res_from_fc"],
        obj_ch_mult_prior=cfg.get("obj_ch_mult_prior", cfg["obj_ch_mult"]),
        obj_ch_mult=cfg["obj_ch_mult"],
        obj_base_ch=cfg["obj_base_ch"],
        obj_final_cnn_ch=cfg["obj_final_cnn_ch"],
        bg_res_from_fc=cfg["bg_res_from_fc"],
        bg_ch_mult=cfg["bg_ch_mult"],
        bg_base_ch=cfg["bg_base_ch"],
        bg_final_cnn_ch=cfg["bg_final_cnn_ch"],
        share_bg_across_time=cfg.get("share_bg_across_time", False),
        use_resblock=cfg["use_resblock"],
        num_res_blocks=cfg["num_res_blocks"],
        cnn_mid_blocks=cfg.get("cnn_mid_blocks", False),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
        pint_enc_layers=cfg["pint_enc_layers"],
        pint_enc_heads=cfg["pint_enc_heads"],
        timestep_horizon=cfg.get("timestep_horizon", 1),
        n_static_frames=cfg.get("num_static_frames", 1),
        context_dim=cfg.get("context_dim", 0),
    )


def _boxes_from_alpha(alpha: torch.Tensor, alpha_floor=0.05, min_pixels=4):
    """Tight boxes from per-particle alpha canvases, fully on device.

    alpha: (K, S, S). Each pixel is owned by the particle whose alpha wins
    the argmax (if above alpha_floor); boxes are per-owner coordinate
    min/max via scatter reductions - no python loop over particles.
    Returns (keep_idx (M,), boxes (M, 4) xyxy in canvas pixels).
    """
    k, s, _ = alpha.shape
    device = alpha.device
    max_a, owner = alpha.max(dim=0)                       # (S, S)
    lab = torch.where(max_a >= alpha_floor, owner,
                      torch.full_like(owner, k)).flatten()  # invalid -> bin k
    ys = torch.arange(s, device=device).repeat_interleave(s)
    xs = torch.arange(s, device=device).repeat(s)

    xmin = torch.full((k + 1,), s, device=device, dtype=torch.long)
    ymin = torch.full((k + 1,), s, device=device, dtype=torch.long)
    xmax = torch.full((k + 1,), -1, device=device, dtype=torch.long)
    ymax = torch.full((k + 1,), -1, device=device, dtype=torch.long)
    xmin.scatter_reduce_(0, lab, xs, "amin")
    ymin.scatter_reduce_(0, lab, ys, "amin")
    xmax.scatter_reduce_(0, lab, xs, "amax")
    ymax.scatter_reduce_(0, lab, ys, "amax")
    counts = torch.bincount(lab, minlength=k + 1)[:k]

    keep = torch.nonzero(counts >= min_pixels).flatten()
    boxes = torch.stack([xmin[keep], ymin[keep],
                         xmax[keep] + 1, ymax[keep] + 1], dim=1).float()
    return keep, boxes


def _boxes_from_alpha_fg(alpha: torch.Tensor, fg: torch.Tensor, alpha_floor=0.05, min_fg_pixels=1):
    """Pixel-tight boxes: alpha ownership at model resolution, refined
    against the foreground of the model's own input at native resolution.

    With the bg->black recolor the background of the input is exactly black,
    so every non-black pixel is sprite (or structure). Each model cell is
    owned by the particle whose alpha wins the argmax (>= alpha_floor); the
    owner map is upsampled to the padded canvas and intersected with the
    non-black pixels, and each particle's box is the min/max of the native
    pixels it owns - no halo, no 2x2-cell rounding. Particles owning fewer
    than min_fg_pixels non-black pixels (a box over empty background) are
    dropped.

    alpha: (K, s, s); fg: (S, S) bool canvas, S a multiple of s.
    Returns (keep_idx (M,), boxes (M, 4) xyxy in canvas pixels)."""
    k, s, _ = alpha.shape
    S = fg.shape[-1]
    f = S // s
    device = alpha.device
    max_a, owner = alpha.max(dim=0)                                   # (s, s)
    own = torch.where(max_a >= alpha_floor, owner, torch.full_like(owner, k))
    own = own.repeat_interleave(f, 0).repeat_interleave(f, 1)         # (S, S)
    lab = torch.where(fg, own, torch.full_like(own, k)).flatten()     # background -> bin k
    ys = torch.arange(S, device=device).repeat_interleave(S)
    xs = torch.arange(S, device=device).repeat(S)
    xmin = torch.full((k + 1,), S, device=device, dtype=torch.long)
    ymin = torch.full((k + 1,), S, device=device, dtype=torch.long)
    xmax = torch.full((k + 1,), -1, device=device, dtype=torch.long)
    ymax = torch.full((k + 1,), -1, device=device, dtype=torch.long)
    xmin.scatter_reduce_(0, lab, xs, "amin")
    ymin.scatter_reduce_(0, lab, ys, "amin")
    xmax.scatter_reduce_(0, lab, xs, "amax")
    ymax.scatter_reduce_(0, lab, ys, "amax")
    counts = torch.bincount(lab, minlength=k + 1)[:k]
    keep = torch.nonzero(counts >= min_fg_pixels).flatten()
    boxes = torch.stack([xmin[keep], ymin[keep], xmax[keep] + 1, ymax[keep] + 1], dim=1).float()
    return keep, boxes


class DLPInference:
    """Per-game DLP object extractor. See module docstring for the output."""

    def __init__(self, game: str, weights_root=WEIGHTS_ROOT, device=None,
                 conf_thresh: float = 0.5, tight_boxes: bool = True,
                 compile_model: bool = True, k_max: int = 64):
        """compile_model (default True): CUDA-graph compile (reduce-overhead)
        of the encoder and the alpha decode: ~11.5 -> ~6.2 ms single-frame,
        detections identical (boxes exact; float attributes differ ~1e-3 and
        particle ordering may change). Costs ~30-60 s one-time compilation on
        the first call (per batch shape) - pass compile_model=False for quick
        interactive scripts where that warmup is not worth it.
        k_max: static cap on decoded particles per frame (tight boxes). Must
        be >= the number of simultaneously confident objects (64 covers every
        game in the dataset; SpaceInvaders peaks at ~45).
        """
        game_dir = _resolve_game_dir(game, weights_root)
        if not (game_dir / "best.pth").exists():
            raise FileNotFoundError(
                f"no weights for {game!r} under {weights_root} "
                f"(available: {', '.join(list_games(weights_root))})")
        self.game = game
        self.conf_thresh = conf_thresh
        self.tight_boxes = tight_boxes
        self.k_max = k_max
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.cfg = json.loads((game_dir / "hparams.json").read_text())
        self.image_size = self.cfg["image_size"]
        # checkpoints trained on padded canvases (cfg pad_to > 0) get the
        # same padding at inference; boxes are mapped back to input pixels
        self.pad_to = self.cfg.get("pad_to", 0)
        # how the padded canvas is downscaled to image_size - must match what
        # the checkpoint was trained on: "maxpool" (the shipped recipe) or
        # "interp" (exact box average)
        self.resize_mode = self.cfg.get("resize_mode", "interp")
        if self.resize_mode not in ("maxpool", "interp"):
            raise ValueError(f"{game}: hparams resize_mode {self.resize_mode!r} is not supported by this package")
        if self.cfg.get("game_transforms", False):
            raise ValueError(f"{game}: checkpoint trained with legacy per-game frame transforms, not supported by this package")
        # checkpoints trained with the recolor (hparams palette_spread: true)
        # carry their palette.json: apply it to every incoming frame, exactly
        # as the training loader did
        self.frame_transform = None
        self.palette_spread = None
        if self.cfg.get("palette_spread"):
            from .palette_spread import PaletteSpread
            self.palette_spread = PaletteSpread.from_file(game_dir / "palette.json")
            self.frame_transform = self.palette_spread
        self.model = _build_model(self.cfg).to(self.device)
        self.model.load_state_dict(
            torch.load(game_dir / "best.pth", map_location=self.device))
        self.model.eval()

        self._alpha_fn = self._make_alpha_fn()
        self._compiled = compile_model
        if compile_model:
            # reduce-overhead = CUDA graphs: replays the recorded kernel
            # sequence with a single launch - the big win for online batch=1
            self.model.encoder_module = torch.compile(
                self.model.encoder_module, mode="reduce-overhead")
            self._alpha_fn = torch.compile(self._alpha_fn,
                                           mode="reduce-overhead")

    def _make_alpha_fn(self):
        """Alpha-mask decode for a fixed-size particle set (static shapes,
        so it is compilable/graphable): glimpse decode (alpha channel only),
        placement on canvas, depth-importance stitching."""
        dec = self.model.decoder_module

        def alpha_fn(feats, z, z_scale, conf_w, depth):
            glimpses = dec.particle_dec(feats.unsqueeze(0))   # (K', 4, p, p)
            alpha_glimpse = glimpses[:, :1].unsqueeze(0)      # (1, K', 1, p, p)
            a_obj = dec.translate_patches(z.unsqueeze(0), alpha_glimpse,
                                          z_scale.unsqueeze(0))
            a_obj = conf_w[None, :, None, None, None] * a_obj
            importance = a_obj * torch.sigmoid(-depth[None, :, :, None, None])
            importance = importance / (importance.sum(dim=1, keepdim=True)
                                       + 1e-5)
            return (importance * a_obj)[0, :, 0]              # (K', S, S)

        return alpha_fn

    # ------------------------------------------------------------------ #
    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """(T, H, W, 3) uint8 RGB -> (T, 1, 3, s, s) float on device.

        The checkpoints were trained on channel-swapped PNG loads, so the
        RGB input is swapped to that convention here.
        """
        arr = np.ascontiguousarray(frames[..., ::-1])
        if self.pad_to:
            t, h, w = arr.shape[:3]
            canvas = np.zeros((t, self.pad_to, self.pad_to, 3), arr.dtype)
            top, left = (self.pad_to - h) // 2, (self.pad_to - w) // 2
            canvas[:, top:top + h, left:left + w] = arr
            arr = canvas
        x = torch.from_numpy(arr)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        if self.pad_to:
            # exact-factor downscale in the checkpoint's trained resize_mode.
            # Done on CPU so it is bit-identical to the training loader
            # (argmax tie-breaking can differ between CPU and CUDA).
            factor = self.pad_to // self.image_size
            if factor > 1:
                if self.resize_mode == "maxpool":
                    x = F.max_pool2d(x, kernel_size=factor, stride=factor)
                else:   # "interp": exact box average
                    x = F.interpolate(x, size=(self.image_size, self.image_size),
                                      mode="bilinear", align_corners=False)
        else:
            # legacy squash for the old (unpadded) checkpoints
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        return x.unsqueeze(1).to(self.device)  # frames as batch, T=1 each

    def _frame_tensordict(self, enc, i, orig_hw, conf_thresh) -> TensorDict:
        h, w = orig_hw
        z = enc["z"][i, 0]                              # (K, 2) (y, x) in (-1, 1)
        scale = torch.sigmoid(enc["z_scale"][i, 0])     # (K, 2) (sy, sx) in (0, 1)
        conf = enc["obj_on"][i, 0].squeeze(-1)          # (K,)
        depth = enc["z_depth"][i, 0].squeeze(-1)        # (K,)
        feats = enc["z_features"][i, 0]                 # (K, D)

        keep = conf > conf_thresh

        if self.pad_to:
            # latents span the padded canvas; undo the centered pad
            top, left = (self.pad_to - h) // 2, (self.pad_to - w) // 2
            cx = (0.5 + z[keep, 1] / 2) * self.pad_to - left
            cy = (0.5 + z[keep, 0] / 2) * self.pad_to - top
            sw = scale[keep, 1] * self.pad_to
            sh = scale[keep, 0] * self.pad_to
        else:
            cx = (0.5 + z[keep, 1] / 2) * w
            cy = (0.5 + z[keep, 0] / 2) * h
            sw = scale[keep, 1] * w
            sh = scale[keep, 0] * h
        bbox = torch.stack([(cx - sw / 2).clamp(0, w),
                            (cy - sh / 2).clamp(0, h),
                            (cx + sw / 2).clamp(0, w),
                            (cy + sh / 2).clamp(0, h)], dim=-1)

        return TensorDict({
            "position": torch.stack([cx, cy], dim=-1),
            "size": torch.stack([sw, sh], dim=-1),
            "bbox": bbox,
            "confidence": conf[keep],
            "depth": depth[keep],
            "embedding": feats[keep],
            "background_embedding": enc["z_bg_features"][i, 0].clone(),
        }, batch_size=[])

    def _frame_tensordict_tight(self, enc, i, orig_hw, conf_thresh, fg=None,
                                alpha_floor=0.05, min_pixels=4) -> TensorDict:
        """Tight boxes from decoded alpha masks - fast path.

        Only the particles that pass the confidence gate are decoded, only
        their alpha channel is placed on the canvas (no RGB compositing, no
        background decode), and the blob->box extraction runs on device.
        With ``fg`` (the non-black pixels of the recolored input on the padded
        canvas) boxes are pixel-tight (_boxes_from_alpha_fg); without it they
        follow the alpha ownership at model resolution (_boxes_from_alpha)."""
        h, w = orig_hw
        conf_all = enc["obj_on"][i, 0].squeeze(-1)      # (K,)
        # static top-k selection instead of a data-dependent nonzero gate:
        # shapes stay fixed (CUDA-graph friendly) and no GPU->CPU sync is
        # forced mid-pipeline. Sub-threshold particles get weight 0, which
        # reproduces the gated arithmetic exactly for surviving particles.
        k = min(self.k_max, conf_all.shape[0])
        conf, gate = conf_all.topk(k)                   # (K',), (K',)

        z = enc["z"][i, 0][gate]                        # (K', 2)
        z_scale = enc["z_scale"][i, 0][gate]            # (K', 2) pre-sigmoid
        depth = enc["z_depth"][i, 0][gate]              # (K', 1)
        feats = enc["z_features"][i, 0][gate]           # (K', D)
        conf_w = conf * (conf > conf_thresh)            # masked alpha weight

        alpha = self._alpha_fn(feats, z, z_scale, conf_w, depth)  # (K', S, S)
        s = alpha.shape[-1]
        if fg is not None:
            keep, boxes = _boxes_from_alpha_fg(alpha, fg, alpha_floor)   # canvas px already
            f = 1.0
        else:
            keep, boxes = _boxes_from_alpha(alpha, alpha_floor, min_pixels)
            f = self.pad_to / s if self.pad_to else 1.0
        if self.pad_to:
            # model px -> padded px -> original px (undo the centered pad)
            top, left = (self.pad_to - h) // 2, (self.pad_to - w) // 2
            bbox = boxes * f - torch.tensor([left, top, left, top],
                                            device=boxes.device)
            lim = torch.tensor([w, h, w, h], device=boxes.device,
                               dtype=bbox.dtype)
            bbox = bbox.clamp(min=torch.zeros_like(lim), max=lim)
            good = (bbox[:, 2] - bbox[:, 0] > 0) & (bbox[:, 3] - bbox[:, 1] > 0)
            keep, bbox = keep[good], bbox[good]
        else:
            bbox = boxes * torch.tensor([w / s, h / s, w / s, h / s],
                                        device=boxes.device)

        return TensorDict({
            "position": (bbox[:, :2] + bbox[:, 2:]) / 2,
            "size": bbox[:, 2:] - bbox[:, :2],
            "bbox": bbox,
            "confidence": conf[keep],
            "depth": depth.squeeze(-1)[keep],
            "embedding": feats[keep],
            "background_embedding": enc["z_bg_features"][i, 0].clone(),
        }, batch_size=[])

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(self, frames: Union[np.ndarray, List[np.ndarray]],
                 conf_thresh: float = None, tight_boxes: bool = None
                 ) -> Union[TensorDict, List[TensorDict]]:
        """Run DLP on one frame or a sequence.

        frames: (H, W, 3) uint8 RGB, or (T, H, W, 3) / list of frames.
        Returns one TensorDict per frame (a bare TensorDict for a single
        frame). Objects with obj_on <= conf_thresh are dropped; pass
        conf_thresh=0 to keep every particle.

        tight_boxes=True (default) decodes the per-particle alpha masks and
        fits boxes to the owned pixels (tight, a bit slower). False skips the
        decoder and uses the particle scale latent (fast, looser boxes).
        """
        if conf_thresh is None:
            conf_thresh = self.conf_thresh
        if tight_boxes is None:
            tight_boxes = self.tight_boxes
        if self._compiled:
            # new CUDA-graph step: previous call's graph outputs may now be
            # overwritten (our returned TensorDicts hold copies, never views)
            torch.compiler.cudagraph_mark_step_begin()
        arr = np.stack(frames) if isinstance(frames, (list, tuple)) \
            else np.asarray(frames)
        if self.frame_transform is not None:
            arr = self.frame_transform(arr)
        single = arr.ndim == 3
        if single:
            arr = arr[None]
        orig_hw = arr.shape[1:3]

        x = self._preprocess(arr)
        enc = self.model.encode_all(x, deterministic=True)  # encoder only
        if tight_boxes:
            fg = self._foreground(arr) if (self.palette_spread is not None and self.pad_to) else None
            out = [self._frame_tensordict_tight(enc, i, orig_hw, conf_thresh,
                                                fg=None if fg is None else fg[i])
                   for i in range(len(arr))]
        else:
            out = [self._frame_tensordict(enc, i, orig_hw, conf_thresh) for i in range(len(arr))]
        return out[0] if single else out

    def _foreground(self, arr: np.ndarray) -> torch.Tensor:
        """(T, pad_to, pad_to) bool: non-black pixels of the recolored frames
        on the padded canvas (the background is exactly black after the
        recolor), used to fit boxes to the sprite pixels."""
        t, h, w = arr.shape[:3]
        fg = np.zeros((t, self.pad_to, self.pad_to), bool)
        top, left = (self.pad_to - h) // 2, (self.pad_to - w) // 2
        fg[:, top:top + h, left:left + w] = arr.max(axis=-1) > 0
        return torch.from_numpy(fg).to(self.device)

    def visualize(self, frame: np.ndarray, out: TensorDict = None,
                  **kwargs) -> np.ndarray:
        """Annotated RGB image of the detections on ``frame``. Runs inference
        when ``out`` is not given. See module-level ``visualize`` for options."""
        if out is None:
            out = self(frame)
        return visualize(frame, out, **kwargs)
