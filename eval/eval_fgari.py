"""FGARI evaluation of a DLP checkpoint on the test split of one game.

FGARI (foreground Adjusted Rand Index): for every test frame,
  - prediction: per-pixel object ownership from the decoded per-particle
    alpha masks (argmax over particles, the exact ownership rule the tight
    boxes of DLPInference use; pixels below --alpha-floor are background),
    mapped back to input-frame pixels;
  - ground truth: the YOLO-format label boxes painted as an instance map
    (larger boxes painted first, so smaller/on-top objects survive overlaps);
  - score: ARI between the two label maps restricted to the ground-truth
    foreground pixels. Frames without any GT box are skipped.
The game score is the mean over frames.

Usage (paths relative to the repo root):
  python eval/eval_fgari.py --game Asterix --weights-root weights \
      --dataset $SCRATCH/dataset

Writes eval/results/<weights-root-name>/<game>.json (mean/std + per-frame).
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dlp_inference import DLPInference                      # noqa: E402
from dlp_inference.inference import _resolve_game_dir       # noqa: E402


# --------------------------------------------------------------------------- #
def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """ARI between two 1D integer label arrays (contingency-table formula)."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    m = np.zeros((a.max() + 1, b.max() + 1), np.int64)
    np.add.at(m, (a, b), 1)

    def comb2(x):
        return x * (x - 1) // 2

    sum_ij = comb2(m).sum()
    sum_a = comb2(m.sum(axis=1)).sum()
    sum_b = comb2(m.sum(axis=0)).sum()
    n_pairs = comb2(m.sum())
    if n_pairs == 0:
        return 1.0
    expected = sum_a * sum_b / n_pairs
    max_index = (sum_a + sum_b) / 2.0
    denom = max_index - expected
    if denom == 0:            # single cluster on both sides -> perfect match
        return 1.0
    return float((sum_ij - expected) / denom)


def gt_instance_map(label_path: str, h: int, w: int) -> np.ndarray:
    """(h, w) int32 map: 0 = background, i>=1 = GT object i (YOLO boxes)."""
    boxes = []
    for line in Path(label_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1 = int(round((cx - bw / 2) * w))
        y1 = int(round((cy - bh / 2) * h))
        x2 = int(round((cx + bw / 2) * w))
        y2 = int(round((cy + bh / 2) * h))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    seg = np.zeros((h, w), np.int32)
    order = sorted(range(len(boxes)),                       # big first ->
                   key=lambda i: -(boxes[i][2] - boxes[i][0])
                   * (boxes[i][3] - boxes[i][1]))           # small stay on top
    for lab, i in enumerate(order, start=1):
        x1, y1, x2, y2 = boxes[i]
        seg[y1:y2, x1:x2] = lab
    return seg


@torch.no_grad()
def owner_maps(dlp: DLPInference, frames, conf_thresh: float,
               alpha_floor: float = 0.05):
    """Per-pixel predicted ownership for a batch of frames.

    Mirrors DLPInference._frame_tensordict_tight up to (and including) the
    alpha argmax, then maps the (S, S) label canvas back to input pixels.
    Returns a list of (h, w) int32 maps, 0 = background.
    """
    arr = np.stack(frames)
    if dlp.frame_transform is not None:
        arr = dlp.frame_transform(arr)
    h, w = arr.shape[1:3]
    x = dlp._preprocess(arr)
    enc = dlp.model.encode_all(x, deterministic=True)

    maps = []
    for i in range(len(arr)):
        conf_all = enc["obj_on"][i, 0].squeeze(-1)          # (K,)
        k = min(dlp.k_max, conf_all.shape[0])
        conf, gate = conf_all.topk(k)
        alpha = dlp._alpha_fn(enc["z_features"][i, 0][gate],
                              enc["z"][i, 0][gate],
                              enc["z_scale"][i, 0][gate],
                              conf * (conf > conf_thresh),
                              enc["z_depth"][i, 0][gate])    # (K', S, S)
        max_a, owner = alpha.max(dim=0)
        lab = torch.where(max_a >= alpha_floor, owner + 1,
                          torch.zeros_like(owner))
        lab = lab.cpu().numpy().astype(np.int32)
        if dlp.pad_to:      # model canvas -> padded px -> undo centered pad
            lab = cv2.resize(lab, (dlp.pad_to, dlp.pad_to),
                             interpolation=cv2.INTER_NEAREST)
            top, left = (dlp.pad_to - h) // 2, (dlp.pad_to - w) // 2
            lab = lab[top:top + h, left:left + w]
        else:
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        maps.append(lab)
    return maps


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--weights-root", default="weights",
                    help="weights dir (name relative to the repo root, or an "
                         "absolute path), default: weights")
    ap.add_argument("--dataset",
                    default=os.path.join(os.environ.get("SCRATCH", "."),
                                         "dataset"),
                    help="OCAtari dataset root with images/ and labels/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--conf-thresh", type=float, default=0.5)
    ap.add_argument("--alpha-floor", type=float, default=0.05)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap on evaluated frames (0 = all)")
    ap.add_argument("--out-dir", default=None,
                    help="default: eval/results/<weights-root-name>")
    args = ap.parse_args()

    root = Path(args.weights_root)
    if not root.is_absolute():
        root = REPO / root
    game_dir = _resolve_game_dir(args.game, root)
    if not (game_dir / "best.pth").exists():
        print(f"SKIP {args.game}: no weights under {root}")
        return

    img_dir = Path(args.dataset) / "images" / args.split
    lbl_dir = Path(args.dataset) / "labels" / args.split
    paths = sorted(glob.glob(str(img_dir / f"{args.game}_*.png")),
                   key=lambda p: int(Path(p).stem.split("_")[-1]))
    pairs = [(p, lbl_dir / (Path(p).stem + ".txt")) for p in paths]
    pairs = [(p, l) for p, l in pairs if l.exists()]
    if args.max_frames > 0:
        pairs = pairs[:args.max_frames]
    if not pairs:
        raise SystemExit(f"no {args.split} frames+labels for {args.game} "
                         f"under {args.dataset}")

    dlp = DLPInference(args.game, weights_root=root, compile_model=False,
                       conf_thresh=args.conf_thresh)
    print(f"{args.game}: {len(pairs)} frames, weights {game_dir}")

    scores, skipped = [], 0
    for j in range(0, len(pairs), args.batch):
        chunk = pairs[j:j + args.batch]
        frames = [cv2.imread(p) for p, _ in chunk]   # cv2 read -> true RGB
        preds = owner_maps(dlp, frames, args.conf_thresh, args.alpha_floor)
        for (p, lbl), pred in zip(chunk, preds):
            h, w = pred.shape
            gt = gt_instance_map(lbl, h, w)
            fg = gt > 0
            if not fg.any():
                skipped += 1
                continue
            scores.append(adjusted_rand_index(gt[fg], pred[fg]))
        done = min(j + args.batch, len(pairs))
        if done % (args.batch * 10) < args.batch or done == len(pairs):
            print(f"  {done}/{len(pairs)}  running FGARI "
                  f"{np.mean(scores):.4f}", flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else \
        REPO / "eval" / "results" / root.name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "game": args.game,
        "weights_root": root.name,
        "weights_dir": str(game_dir),
        "split": args.split,
        "n_frames": len(scores),
        "n_skipped_no_gt": skipped,
        "conf_thresh": args.conf_thresh,
        "alpha_floor": args.alpha_floor,
        "fgari_mean": float(np.mean(scores)),
        "fgari_std": float(np.std(scores)),
        "fgari_median": float(np.median(scores)),
        "fgari_per_frame": [round(s, 6) for s in scores],
    }
    out_path = out_dir / f"{args.game}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"{args.game} [{root.name}]  FGARI {result['fgari_mean']:.4f} "
          f"+- {result['fgari_std']:.4f}  ({len(scores)} frames) "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
