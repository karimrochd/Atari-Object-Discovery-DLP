"""Object-level detection metrics: precision / recall / F1 with IoU matching.

Unlike FG-ARI (pixel-weighted, biased toward large objects), this treats
every GT object equally: predicted boxes are matched one-to-one to GT boxes
greedily by descending IoU; a match counts if IoU >= threshold.

Usage:
  python eval/eval_prf.py --game Asterix --weights-root weights \
      --dataset /path/to/dataset [--frames 500] [--ious 0.25 0.5]

Writes eval/results_prf/<weights-root-name>/<game>.json.

Size-stratified metrics ("by_size" in the JSON): every GT object is bucketed
by its box area at native 210x160 resolution and recall is reported per
bucket (and precision per predicted-box bucket), so small particles count
exactly as much as big ones - FG-ARI is pixel-weighted and hides them.
Default buckets (px): tiny <=24, small 25-100, medium 101-400, large >400.
"size_balanced_f1" is the mean of the per-bucket F1s (buckets with GT).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dlp_inference import DLPInference           # noqa: E402
from dlp_inference.inference import _resolve_game_dir  # noqa: E402


def gt_boxes(label_path, w=160, h=210):
    out = []
    for line in Path(label_path).read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, np.float32).reshape(-1, 4)


def iou_matrix(a, b):
    """(N,4) x (M,4) -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(iou, thresh, pairs=None):
    """One-to-one greedy matching by descending IoU. Returns n_matched; if
    ``pairs`` (a list) is given, the matched (pred_idx, gt_idx) are appended."""
    iou = iou.copy()
    n = 0
    if pairs is None:
        pairs = []
    while True:
        i, j = np.unravel_index(iou.argmax(), iou.shape) if iou.size else (0, 0)
        if iou.size == 0 or iou[i, j] < thresh:
            return n
        n += 1
        pairs.append((int(i), int(j)))
        iou[i, :] = -1
        iou[:, j] = -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--weights-root", default="weights")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--frames", type=int, default=500,
                    help="frames evaluated, spread over the split (0 = all)")
    ap.add_argument("--ious", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--conf-thresh", type=float, default=0.5)
    ap.add_argument("--size-buckets", type=float, nargs="+", default=[24, 100, 400],
                    help="GT box area thresholds (px at 210x160) separating the size buckets")
    args = ap.parse_args()
    edges = [0.0] + list(args.size_buckets) + [float("inf")]
    bucket_names = ([f"tiny<={int(edges[1])}"] +
                    [f"{int(edges[i]) + 1}-{int(edges[i + 1])}" for i in range(1, len(edges) - 2)] +
                    [f"large>{int(edges[-2])}"])

    def bucket_of(area):
        for b in range(len(edges) - 1):
            if edges[b] <= area <= edges[b + 1] if b == 0 else edges[b] < area <= edges[b + 1]:
                return b
        return len(edges) - 2

    root = Path(args.weights_root)
    if not root.is_absolute():
        root = REPO / root
    if not (_resolve_game_dir(args.game, root) / "best.pth").exists():
        print(f"SKIP {args.game}: no weights under {root}")
        return

    img_dir = Path(args.dataset) / "images" / args.split
    lbl_dir = Path(args.dataset) / "labels" / args.split
    paths = sorted(glob.glob(str(img_dir / f"{args.game}_*.png")),
                   key=lambda p: int(Path(p).stem.split("_")[-1]))
    pairs = [(p, lbl_dir / (Path(p).stem + ".txt")) for p in paths]
    pairs = [(p, l) for p, l in pairs if l.exists()]
    if args.frames > 0 and len(pairs) > args.frames:
        step = len(pairs) / args.frames
        pairs = [pairs[int(i * step)] for i in range(args.frames)]

    model = DLPInference(args.game, weights_root=root,
                         conf_thresh=args.conf_thresh, compile_model=False)
    stats = {t: dict(tp=0, fp=0, fn=0) for t in args.ious}
    nb = len(bucket_names)
    # per size bucket: GT side (tp, fn) for recall, predicted side (tp, fp) for precision
    by_size = {t: [dict(gt=0, gt_matched=0, pred=0, pred_matched=0) for _ in range(nb)] for t in args.ious}
    for p, l in pairs:
        gt = gt_boxes(l)
        out = model(cv2.imread(p))
        pred = (out["bbox"].cpu().numpy().reshape(-1, 4)
                if out["bbox"].numel() else np.zeros((0, 4), np.float32))
        iou = iou_matrix(pred, gt)
        gt_b = [bucket_of((b[2] - b[0]) * (b[3] - b[1])) for b in gt]
        pr_b = [bucket_of((b[2] - b[0]) * (b[3] - b[1])) for b in pred]
        for t in args.ious:
            matched = []
            m = greedy_match(iou, t, matched)
            stats[t]["tp"] += m
            stats[t]["fp"] += len(pred) - m
            stats[t]["fn"] += len(gt) - m
            for b in gt_b:
                by_size[t][b]["gt"] += 1
            for b in pr_b:
                by_size[t][b]["pred"] += 1
            for i, j in matched:
                by_size[t][pr_b[i]]["pred_matched"] += 1
                by_size[t][gt_b[j]]["gt_matched"] += 1

    result = {"game": args.game, "weights_root": root.name,
              "split": args.split, "n_frames": len(pairs),
              "conf_thresh": args.conf_thresh, "size_buckets_px": args.size_buckets,
              "metrics": {}}
    for t, s in stats.items():
        prec = s["tp"] / max(1, s["tp"] + s["fp"])
        rec = s["tp"] / max(1, s["tp"] + s["fn"])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        result["metrics"][str(t)] = dict(precision=round(prec, 4),
                                         recall=round(rec, 4),
                                         f1=round(f1, 4), **s)
        sizes = {}
        f1s = []
        for b, name in enumerate(bucket_names):
            c = by_size[t][b]
            r_b = c["gt_matched"] / c["gt"] if c["gt"] else None
            p_b = c["pred_matched"] / c["pred"] if c["pred"] else None
            f1_b = (2 * p_b * r_b / (p_b + r_b) if (p_b and r_b) else (0.0 if c["gt"] else None))
            sizes[name] = dict(gt=c["gt"], recall=None if r_b is None else round(r_b, 4),
                               pred=c["pred"], precision=None if p_b is None else round(p_b, 4),
                               f1=None if f1_b is None else round(f1_b, 4))
            if c["gt"]:
                f1s.append(f1_b or 0.0)
        result["metrics"][str(t)]["by_size"] = sizes
        result["metrics"][str(t)]["size_balanced_f1"] = round(sum(f1s) / len(f1s), 4) if f1s else None
        small_r = [sizes[n]["recall"] for n in bucket_names[:2] if sizes[n]["recall"] is not None]
        result["metrics"][str(t)]["small_recall"] = round(sum(small_r) / len(small_r), 4) if small_r else None
        print(f"{args.game} [{root.name}] IoU>={t}: "
              f"P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f} | size-balanced F1 "
              f"{result['metrics'][str(t)]['size_balanced_f1']} | recall by size: "
              + "  ".join(f"{n} {sizes[n]['recall']}({sizes[n]['gt']})" for n in bucket_names))

    out_dir = REPO / "eval" / "results_prf" / root.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.game}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
