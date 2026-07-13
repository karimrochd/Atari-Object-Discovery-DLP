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

WEIGHTS_ROOT = Path(__file__).resolve().parents[1] / "weights"

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
    pos = ((bbox[:, :2] + bbox[:, 2:]) / 2).detach().cpu().numpy() * upscale
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
    return sorted(p.name for p in Path(weights_root).iterdir()
                  if (p / "best.pth").exists())


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


def _align_to_render(t: torch.Tensor, z_base_var: torch.Tensor,
                     k_render: int) -> torch.Tensor:
    """Reproduce the decoder's K_full -> K_render particle selection (lowest
    positional variance) so latent attributes line up with alpha_masks."""
    if t.shape[1] == k_render:
        return t
    key = z_base_var.sum(-1) if z_base_var.dim() == 3 else z_base_var
    _, top_idx = torch.topk(key, k=k_render, dim=-1, largest=False)
    while top_idx.dim() < t.dim():
        top_idx = top_idx.unsqueeze(-1)
    expand = [-1] * t.dim()
    for d in range(2, t.dim()):
        expand[d] = t.shape[d]
    return torch.gather(t, dim=1, index=top_idx.expand(*expand))


class DLPInference:
    """Per-game DLP object extractor. See module docstring for the output."""

    def __init__(self, game: str, weights_root=WEIGHTS_ROOT, device=None,
                 conf_thresh: float = 0.5, tight_boxes: bool = True):
        game_dir = Path(weights_root) / game
        if not (game_dir / "best.pth").exists():
            raise FileNotFoundError(
                f"no weights for {game!r} under {weights_root} "
                f"(available: {', '.join(list_games(weights_root))})")
        self.game = game
        self.conf_thresh = conf_thresh
        self.tight_boxes = tight_boxes
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.cfg = json.loads((game_dir / "hparams.json").read_text())
        self.image_size = self.cfg["image_size"]
        self.model = _build_model(self.cfg).to(self.device)
        self.model.load_state_dict(
            torch.load(game_dir / "best.pth", map_location=self.device))
        self.model.eval()

    # ------------------------------------------------------------------ #
    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """(T, H, W, 3) uint8 RGB -> (T, 1, 3, s, s) float on device.

        The checkpoints were trained on channel-swapped PNG loads, so the
        RGB input is swapped to that convention here.
        """
        x = torch.from_numpy(np.ascontiguousarray(frames[..., ::-1]))
        x = x.permute(0, 3, 1, 2).float() / 255.0
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

        # Testing stuff
        assert z.isin([-1, 1]).all(), f"z is not in [-1, 1]: {z}"
        assert scale.isin([0, 1]).all(), f"scale is not in [0, 1]: {scale}"
        assert conf.isin([0, 1]).all(), f"conf is not in [0, 1]: {conf}"
        assert depth.isin([0, 1]).all(), f"depth is not in [0, 1]: {depth}"
        assert feats.isin([0, 1]).all(), f"feats is not in [0, 1]: {feats}"

        keep = conf > conf_thresh
        n_objects = keep.sum().item()

        cx = (0.5 + z[keep, 1] / 2) * w
        cy = (0.5 + z[keep, 0] / 2) * h
        sw = scale[keep, 1] * w
        sh = scale[keep, 0] * h
        bbox = torch.stack([(cx - sw / 2).clamp(0, w),
                            (cy - sh / 2).clamp(0, h),
                            (cx + sw / 2).clamp(0, w),
                            (cy + sh / 2).clamp(0, h)], dim=-1)

        return TensorDict({
            "position": z[keep],
            "size": scale[keep],
            "bbox": bbox,
            "confidence": conf[keep],
            "depth": depth[keep],
            "embedding": feats[keep],
        }, batch_size=[n_objects])

    def _frame_tensordict_tight(self, out, i, orig_hw, conf_thresh,
                                alpha_floor=0.05, min_pixels=4) -> TensorDict:
        """Boxes from the decoded alpha masks: each particle owns the pixels
        where its mask wins the argmax; box = tight box of that blob."""
        h, w = orig_hw
        alpha = out["alpha_masks"]                      # (B, K_r, 1, s, s)
        k_render = alpha.shape[1]
        zbv = out["z_base_var"][:, 0]
        conf = _align_to_render(out["obj_on"][:, 0], zbv, k_render)[i].squeeze(-1)
        depth = _align_to_render(out["z_depth"][:, 0], zbv, k_render)[i].squeeze(-1)
        feats = _align_to_render(out["mu_features"][:, 0], zbv, k_render)[i]

        a = alpha[i].squeeze(1).cpu().numpy()           # (K_r, s, s)
        on = conf.cpu().numpy()
        a_gated = a.copy()
        a_gated[on <= conf_thresh] = 0.0
        valid = a_gated.max(axis=0) >= alpha_floor
        owner = a_gated.argmax(axis=0)

        s = a.shape[-1]
        sx, sy = w / s, h / s
        keep, boxes = [], []
        for k in range(k_render):
            if on[k] <= conf_thresh:
                continue
            mask = (owner == k) & valid
            if mask.sum() < min_pixels:
                continue
            ys, xs = np.nonzero(mask)
            boxes.append([xs.min() * sx, ys.min() * sy,
                          (xs.max() + 1) * sx, (ys.max() + 1) * sy])
            keep.append(k)

        keep = torch.as_tensor(keep, dtype=torch.long, device=conf.device)
        bbox = torch.as_tensor(np.asarray(boxes, np.float32).reshape(-1, 4),
                               device=conf.device)

        # normalize position to [-1, 1] and size to [0, 1]
        pos = (bbox[:, :2] + bbox[:, 2:]) / torch.tensor([w, h]) * 2 - 1
        size = (bbox[:, 2:] - bbox[:, :2]) / torch.tensor([w, h]) * 2
        return TensorDict({
            "position": pos,
            "size": size,
            "bbox": bbox,
            "confidence": conf[keep],
            "depth": depth[keep],
            "embedding": feats[keep],
        }, batch_size=[k_render])

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
        arr = np.stack(frames) if isinstance(frames, (list, tuple)) \
            else np.asarray(frames)
        single = arr.ndim == 3
        if single:
            arr = arr[None]
        orig_hw = arr.shape[1:3]

        x = self._preprocess(arr)
        if tight_boxes:
            dec = self.model(x, deterministic=True, with_loss=False)
            alpha = dec["alpha_masks"]
            if alpha.dim() == 4:                        # (B*T, ...) -> (B, K, 1, s, s)
                dec["alpha_masks"] = alpha.view(len(arr), *alpha.shape[1:])
            obj_out = [self._frame_tensordict_tight(dec, i, orig_hw, conf_thresh)
                   for i in range(len(arr))]
            bg_out = dec["mu_bg_features"]
        else:
            enc = self.model.encode_all(x, deterministic=True)
            obj_out = [self._frame_tensordict(enc, i, orig_hw, conf_thresh)
                   for i in range(len(arr))]
            bg_out = enc["z_bg_features"]
        obj_out = obj_out[0] if single else obj_out
        return obj_out, bg_out

    def visualize(self, frame: np.ndarray, out: TensorDict = None,
                  **kwargs) -> np.ndarray:
        """Annotated RGB image of the detections on ``frame``. Runs inference
        when ``out`` is not given. See module-level ``visualize`` for options."""
        if out is None:
            out = self(frame)
        return visualize(frame, out, **kwargs)
