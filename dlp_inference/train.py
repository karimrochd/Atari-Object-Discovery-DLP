"""Train a DLP model for a new game, directly into the weights/ layout.

    python -m dlp_inference.train --game Asterix --root /path/to/dataset

The dataset must follow the OCAtari-PNG layout used for the shipped weights:
``<root>/images/{train,val}/<Game>_<idx>.png`` (labels are not needed -
training is fully unsupervised). Frames are loaded with PIL; the shipped
checkpoints' channel convention is reproduced automatically as long as the
PNGs were written the same way as the OCAtari dataset.

Output: ``weights/<Game>/{hparams.json, palette.json, best.pth}`` - immediately
loadable with ``DLPInference(game)``. The recipe (config_default.json): frames
recolored with the per-frame bg->black palette built from the game's own
training frames (palette_spread.py), zero-padded to 256 and 2x2 max-pooled to
128, z_obj 8, z_bg 5, plain pixel MSE, 100 epochs (~2.6 h on an H100).
``--epochs 5 --max-frames 400`` gives a quick smoke run; any config key can
be overridden with ``--override key=value``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import OCAtariDataset
from .inference import WEIGHTS_ROOT, _build_model
from .model.loss_functions import calc_reconstruction_loss


def _align_to_render(t, z_base_var, k_render):
    """Reproduce the decoder's K_full -> K_render particle selection (lowest
    positional variance) so per-particle attributes line up with alpha_masks."""
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

DEFAULT_CONFIG = Path(__file__).parent / "config_default.json"


def _to_5d(x):
    return x.unsqueeze(1) if x.dim() == 4 else x


@torch.no_grad()
def save_epoch_viz(model, samples, path, conf_thresh=0.5, alpha_floor=0.05):
    """Fixed val samples -> grid PNG: input | reconstruction | particle seg.

    Same images every call so training progress is easy to eyeball."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.RandomState(0)
    palette = rng.rand(256, 3)
    palette[0] = 0.0

    model.eval()
    n = samples.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.5 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        x = samples[i : i + 1].unsqueeze(1)                # (1,1,3,H,W)
        out = model(x, deterministic=True, with_loss=False)
        rec = out["rec_rgb"]
        rec = (rec[0, 0] if rec.dim() == 5 else rec[0]).clamp(0, 1)
        alpha = out["alpha_masks"]
        alpha = (alpha[0] if alpha.dim() == 5 else alpha[0, 0])  # (K,1,H,W)
        k_render = alpha.shape[0]
        obj_on = _align_to_render(out["obj_on"][:, 0],
                                  out["z_base_var"][:, 0], k_render)[0]
        a = alpha.squeeze(1).cpu().numpy()
        on = obj_on.squeeze(-1).cpu().numpy()
        a[on <= conf_thresh] = 0.0
        seg = (a.argmax(0) + 1) * (a.max(0) >= alpha_floor)

        # [:3] = display the box-average view when the input is 6-channel
        panels = [samples[i][:3].permute(1, 2, 0).cpu().numpy().clip(0, 1),
                  rec[:3].permute(1, 2, 0).cpu().numpy(),
                  palette[seg % 256]]
        for j, (p, title) in enumerate(zip(panels, ("input", "recon", "seg"))):
            axes[i, j].imshow(p)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    model.train()


def padded_valid_mask(pad_to, image_size, h, w):
    """(1,1,image_size,image_size) float mask: 1 = real game pixels,
    0 = padding. A downscaled cell counts as valid only if EVERY source
    pixel of its box average lies inside the frame (conservative at
    fractional boundaries) - same rule as pair_tracker."""
    import numpy as np
    v = np.zeros((pad_to, pad_to), np.float32)
    top, left = (pad_to - h) // 2, (pad_to - w) // 2
    v[top:top + h, left:left + w] = 1.0
    f = pad_to // image_size
    m = v.reshape(image_size, f, image_size, f).mean(axis=(1, 3))
    return torch.from_numpy((m >= 0.999).astype("float32"))[None, None]


def _epoch_loss(model, loader, cfg, device, valid_mask=None):
    losses = []
    with torch.no_grad():
        for batch in loader:
            x = _to_5d(batch[0]).to(device)
            pw = batch[5].to(device) if len(batch) > 5 else None
            out = model(x, warmup=False, with_loss=True,
                        beta_kl=cfg["beta_kl"], beta_rec=cfg["beta_rec"],
                        kl_balance=cfg["kl_balance"],
                        recon_loss_type=cfg["recon_loss_type"],
                        recon_loss_func=calc_reconstruction_loss,
                        beta_obj=cfg.get("beta_obj", 0.0),
                        valid_mask=valid_mask,
                        rec_norm=cfg.get("rec_norm", "pixel"),
                        rec_full_weight=cfg.get("rec_full_weight", 1.0),
                        rec_obj_min_mass=cfg.get("rec_obj_min_mass", 1.0),
                        rec_obj_bg=cfg.get("rec_obj_bg", True),
                        rec_obj_max_amp=cfg.get("rec_obj_max_amp"),
                        pixel_weight=pw)
            losses.append(out["loss_dict"]["loss"].item())
    return sum(losses) / max(1, len(losses))


def train(game, root, out_dir=None, num_epochs=None, batch_size=None,
          device=None, max_frames=None, **overrides):
    """Train DLP for ``game`` on the dataset at ``root``.

    Writes hparams.json + best.pth (best val loss) into
    ``out_dir`` (default: the package's weights/<game>/). Returns out_dir.
    """
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    cfg.update(overrides)
    cfg["root"] = str(root)
    cfg["game"] = game
    if num_epochs is not None:
        cfg["num_epochs"] = num_epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir) if out_dir else WEIGHTS_ROOT / game
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hparams.json").write_text(json.dumps(cfg, indent=2))

    # recolor (palette_spread.py): the palette is derived from the TRAINING
    # frames of this dataset alone (pixel statistics, no labels) and stored
    # next to the checkpoint so DLPInference applies the identical mapping
    palette_file = False
    if cfg.get("palette_spread", True):
        from .palette_spread import build_palette
        palette_file = out_dir / "palette.json"
        spec = build_palette(game, cfg["root"], palette_file, split="train")
        mp = spec.get("miss_penalty", {}).get("after", {})
        print(f"[train] recolor ON (built from {cfg['root']}/images/train): {spec['n_colors']} colors, "
              f"miss-penalty proxy all {mp.get('all')} smallest {mp.get('smallest')} -> {palette_file}")

    model = _build_model(cfg).to(device)
    print(f"[train] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"params | game={game} epochs={cfg['num_epochs']} device={device}")

    def split(mode):
        return OCAtariDataset(root=cfg["root"], mode=mode, sample_length=1,
                              image_size=cfg["image_size"], games=[game],
                              max_frames=max_frames,
                              pad_to=cfg.get("pad_to", 0),
                              resize_mode=cfg.get("resize_mode", "maxpool"),
                              palette_spread=str(palette_file) if palette_file else False,
                              motion_weight=cfg.get("motion_weight", 0.0))

    ds_train, ds_val = split("train"), split("val")
    if len(ds_train) == 0:
        raise SystemExit(f"[train] no {game!r} frames in {root}/images/train/")

    valid_mask = None
    if cfg.get("pad_to", 0):
        from PIL import Image as _Image
        w0, h0 = _Image.open(ds_train.paths[0]).size   # native frame size
        valid_mask = padded_valid_mask(cfg["pad_to"], cfg["image_size"],
                                       h0, w0).to(device)
        print(f"[train] pad_to={cfg['pad_to']}: frames {h0}x{w0} centered, "
              f"loss masked to {int(valid_mask.sum())} / "
              f"{cfg['image_size'] ** 2} cells")
    train_loader = DataLoader(ds_train, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg.get("num_workers", 4),
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg.get("num_workers", 4))

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"],
                           betas=tuple(cfg["adam_betas"]), eps=cfg["adam_eps"],
                           weight_decay=cfg["weight_decay"])

    best_val = float("inf")
    eval_freq = cfg.get("eval_epoch_freq", 5)

    # fixed validation samples for the periodic viz (same frames every time)
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    viz_samples = None
    if len(ds_val):
        n_viz = min(6, len(ds_val))
        viz_samples = torch.stack(
            [_to_5d(ds_val[i][0].unsqueeze(0))[0, 0] for i in range(n_viz)]
        ).to(device)

    # curriculum: the per-object loss punishes badly reconstructed objects
    # hard enough that untrained glimpses get switched off instead of learned,
    # so train on the plain pixel loss until glimpses exist, then switch.
    # Validation always uses the final loss so best.pth is picked on one metric.
    final_rec_norm = cfg.get("rec_norm", "pixel")
    rec_obj_start = cfg.get("rec_obj_start_epoch", 0) if final_rec_norm != "pixel" else 0

    for epoch in range(cfg["num_epochs"]):
        model.train()
        warmup = epoch < cfg.get("warmup_epoch", 0)
        rec_norm = final_rec_norm if epoch >= rec_obj_start else "pixel"
        if epoch == rec_obj_start and rec_obj_start > 0:
            print(f"[train] epoch {epoch}: switching reconstruction loss "
                  f"pixel -> {final_rec_norm}")
        pbar = tqdm(train_loader, desc=f"[train] epoch {epoch} ({rec_norm})", leave=False)
        for batch in pbar:
            x = _to_5d(batch[0]).to(device)
            pw = batch[5].to(device) if len(batch) > 5 else None
            out = model(x, warmup=warmup, with_loss=True,
                        beta_kl=cfg["beta_kl"], beta_rec=cfg["beta_rec"],
                        kl_balance=cfg["kl_balance"],
                        recon_loss_type=cfg["recon_loss_type"],
                        recon_loss_func=calc_reconstruction_loss,
                        beta_obj=cfg.get("beta_obj", 0.0),
                        valid_mask=valid_mask,
                        rec_norm=rec_norm,
                        rec_full_weight=cfg.get("rec_full_weight", 1.0),
                        rec_obj_min_mass=cfg.get("rec_obj_min_mass", 1.0),
                        rec_obj_bg=cfg.get("rec_obj_bg", True),
                        rec_obj_max_amp=cfg.get("rec_obj_max_amp"),
                        pixel_weight=pw)
            loss = out["loss_dict"]["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        if epoch % eval_freq == 0 or epoch == cfg["num_epochs"] - 1:
            if viz_samples is not None:
                viz_path = viz_dir / f"epoch_{epoch:03d}.png"
                save_epoch_viz(model, viz_samples, viz_path)
                print(f"[train] viz -> {viz_path}")
            model.eval()
            val = _epoch_loss(model, val_loader, cfg, device, valid_mask)
            print(f"[train] epoch {epoch}  val_loss = {val:.4f}")
            if val < best_val:  # only the best checkpoint is kept
                best_val = val
                torch.save(model.state_dict(), out_dir / "best.pth")

    print(f"[train] done: {out_dir} (best val {best_val:.4f})")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", required=True)
    ap.add_argument("--root", required=True,
                    help="dataset root with images/{train,val}/<Game>_<idx>.png")
    ap.add_argument("--out", default=None,
                    help="output dir (default: weights/<game>/ in this package)")
    ap.add_argument("--epochs", type=int, default=None, help="default 100")
    ap.add_argument("--batch-size", type=int, default=None, help="default 8")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames per split (smoke runs)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VAL",
                    help="config overrides (config_default.json keys), e.g. "
                         "resize_mode=interp palette_spread=false "
                         "(values parsed as JSON, else kept as strings)")
    args = ap.parse_args()
    overrides = {}
    for kv in args.override:
        key, _, val = kv.partition("=")
        try:
            overrides[key] = json.loads(val)
        except json.JSONDecodeError:
            overrides[key] = val
    train(args.game, args.root, out_dir=args.out, num_epochs=args.epochs,
          batch_size=args.batch_size, device=args.device,
          max_frames=args.max_frames, **overrides)


if __name__ == "__main__":
    main()
