"""Train a DLP model for a new game, directly into the weights/ layout.

    python -m dlp_inference.train --game Asterix --root /path/to/dataset

The dataset must follow the OCAtari-PNG layout used for the shipped weights:
``<root>/images/{train,val}/<Game>_<idx>.png`` (labels are not needed -
training is fully unsupervised). Frames are loaded with PIL; the shipped
checkpoints' channel convention is reproduced automatically as long as the
PNGs were written the same way as the OCAtari dataset.

Output: ``weights/<Game>/{hparams.json, best.pth}`` - immediately loadable
with ``DLPInference(game)``. Full quality takes ~100 epochs (hours on a
recent GPU); pass --epochs 5 --max-frames 400 for a quick smoke run.
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

DEFAULT_CONFIG = Path(__file__).parent / "config_default.json"


def _to_5d(x):
    return x.unsqueeze(1) if x.dim() == 4 else x


def _epoch_loss(model, loader, cfg, device):
    losses = []
    with torch.no_grad():
        for batch in loader:
            x = _to_5d(batch[0]).to(device)
            out = model(x, warmup=False, with_loss=True,
                        beta_kl=cfg["beta_kl"], beta_rec=cfg["beta_rec"],
                        kl_balance=cfg["kl_balance"],
                        recon_loss_type=cfg["recon_loss_type"],
                        recon_loss_func=calc_reconstruction_loss,
                        beta_obj=cfg.get("beta_obj", 0.0))
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

    model = _build_model(cfg).to(device)
    print(f"[train] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"params | game={game} epochs={cfg['num_epochs']} device={device}")

    def split(mode):
        return OCAtariDataset(root=cfg["root"], mode=mode, sample_length=1,
                              image_size=cfg["image_size"], games=[game],
                              max_frames=max_frames)

    ds_train, ds_val = split("train"), split("val")
    if len(ds_train) == 0:
        raise SystemExit(f"[train] no {game!r} frames in {root}/images/train/")
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
    for epoch in range(cfg["num_epochs"]):
        model.train()
        warmup = epoch < cfg.get("warmup_epoch", 0)
        pbar = tqdm(train_loader, desc=f"[train] epoch {epoch}", leave=False)
        for batch in pbar:
            x = _to_5d(batch[0]).to(device)
            out = model(x, warmup=warmup, with_loss=True,
                        beta_kl=cfg["beta_kl"], beta_rec=cfg["beta_rec"],
                        kl_balance=cfg["kl_balance"],
                        recon_loss_type=cfg["recon_loss_type"],
                        recon_loss_func=calc_reconstruction_loss,
                        beta_obj=cfg.get("beta_obj", 0.0))
            loss = out["loss_dict"]["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        if epoch % eval_freq == 0 or epoch == cfg["num_epochs"] - 1:
            model.eval()
            val = _epoch_loss(model, val_loader, cfg, device)
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
    args = ap.parse_args()
    train(args.game, args.root, out_dir=args.out, num_epochs=args.epochs,
          batch_size=args.batch_size, device=args.device,
          max_frames=args.max_frames)


if __name__ == "__main__":
    main()
