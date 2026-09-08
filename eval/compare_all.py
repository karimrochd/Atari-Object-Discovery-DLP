"""One CSV comparing every family: FG-ARI, detection P/R/F1 at one IoU, and the size buckets.

  python eval/compare_all.py weights [other_weights_root ...]

Reads eval/results/<root>/<game>.json (eval_fgari.py) and eval/results_prf/<root>/<game>.json (eval_prf.py).
Writes eval/compare_all.csv: one row per game (union of games; a family without a result leaves blanks) plus a
mean row over the games present in every family. small_recall and size_balanced_f1 follow compare_size.py:
the tiny bucket is ignored for games with more than --max-tiny-per-frame tiny GT boxes per frame (texture).
"""
import argparse
import json
from pathlib import Path

EVAL = Path(__file__).resolve().parent
METRICS = ["fgari", "precision", "recall", "f1", "small_recall", "size_balanced_f1",
           "recall_tiny", "recall_25_100", "recall_101_400", "recall_large",
           # small-object detection at the loose IoU: "did a particle land on it", localization aside
           "small_recall_iou25", "small_precision_iou25", "recall_tiny_iou25", "recall_25_100_iou25"]
LOOSE_IOU = "0.25"


def load(root, iou):
    out = {}
    for f in sorted((EVAL / "results" / root).glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("n_frames", 0) > 0:
            out.setdefault(r["game"], {})["fgari"] = r["fgari_mean"]
    for f in sorted((EVAL / "results_prf" / root).glob("*.json")):
        r = json.loads(f.read_text())
        m = r["metrics"].get(iou)
        if r.get("n_frames", 0) > 0 and m and "by_size" in m:
            out.setdefault(r["game"], {})["prf"] = m
            out[r["game"]]["n_frames"] = r["n_frames"]
            out[r["game"]]["prf_loose"] = r["metrics"].get(LOOSE_IOU)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--iou", default="0.5")
    ap.add_argument("--max-tiny-per-frame", type=float, default=20.0)
    ap.add_argument("--out", default=str(EVAL / "compare_all.csv"))
    ap.add_argument("--min-games", type=int, default=30, help="families with fewer games are averaged over their own subset of the common games")
    args = ap.parse_args()
    fams = {r: load(r, args.iou) for r in args.roots}
    short = {r: r.replace("new_weights_", "") for r in args.roots}
    games = sorted(set().union(*[set(v) for v in fams.values()]))
    has = lambda r, g: g in fams[r] and "prf" in fams[r][g] and "fgari" in fams[r][g]
    full = [r for r in args.roots if sum(has(r, g) for g in games) >= args.min_games]   # incomplete families don't shrink the common set
    common = [g for g in games if all(has(r, g) for r in full)]
    own = {r: [g for g in common if has(r, g)] for r in args.roots}

    # GT counts are family-independent: take them from any family that has the game
    def gt_of(g):
        for r in args.roots:
            if g in fams[r] and "prf" in fams[r][g]:
                return fams[r][g]["prf"]["by_size"], fams[r][g]["n_frames"]
        return None, None
    buckets = None
    for g in games:
        bs, _ = gt_of(g)
        if bs:
            buckets = list(bs); break
    tiny, mid = buckets[0], buckets[1]
    ignored = set()
    for g in games:
        bs, nf = gt_of(g)
        if bs and bs[tiny]["gt"] / max(1, nf) > args.max_tiny_per_frame:
            ignored.add(g)

    def derived(m, g):
        bs = m["by_size"]
        small_b = [b for b in (tiny, mid) if not (g in ignored and b == tiny)]
        sr = mean([bs[b]["recall"] for b in small_b])
        f1s = [bs[b]["f1"] if bs[b]["f1"] is not None else 0.0 for b in buckets if bs[b]["gt"] and not (g in ignored and b == tiny)]
        bf = mean(f1s)
        out = {"precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "small_recall": sr, "size_balanced_f1": bf,
               "recall_tiny": None if g in ignored else bs[tiny]["recall"], "recall_25_100": bs[mid]["recall"],
               "recall_101_400": bs[buckets[2]]["recall"], "recall_large": bs[buckets[3]]["recall"],
               "small_recall_iou25": None, "small_precision_iou25": None, "recall_tiny_iou25": None, "recall_25_100_iou25": None}
        lo = m.get("_loose")
        if lo and "by_size" in lo:
            lb = lo["by_size"]
            out["small_recall_iou25"] = mean([lb[b]["recall"] for b in small_b])
            out["small_precision_iou25"] = mean([lb[b]["precision"] for b in small_b if lb[b]["gt"]])
            out["recall_tiny_iou25"] = None if g in ignored else lb[tiny]["recall"]
            out["recall_25_100_iou25"] = lb[mid]["recall"]
        return out

    table = {}   # (root, game) -> {metric: value}
    for r in args.roots:
        for g in games:
            e = fams[r].get(g, {})
            row = {"fgari": e.get("fgari")}
            if "prf" in e:
                e["prf"]["_loose"] = e.get("prf_loose")
            row.update(derived(e["prf"], g) if "prf" in e else {k: None for k in METRICS[1:]})
            table[(r, g)] = row

    cols = ["game", "n_gt_small", "n_gt_total", "tiny_is_texture"] + [f"{m}_{short[r]}" for m in METRICS for r in args.roots]
    lines = [",".join(cols)]
    fmt = lambda v: "" if v is None else f"{v:.4f}"
    for g in games:
        bs, _ = gt_of(g)
        n_small = sum(bs[b]["gt"] for b in (tiny, mid)) if bs else ""
        n_tot = sum(bs[b]["gt"] for b in buckets) if bs else ""
        lines.append(",".join([g, str(n_small), str(n_tot), "yes" if g in ignored else "no"]
                              + [fmt(table[(r, g)][m]) for m in METRICS for r in args.roots]))
    lines.append(",".join([f"mean_over_{len(common)}_common_games", "", "", ""]
                          + [fmt(mean([table[(r, g)][m] for g in own[r]])) for m in METRICS for r in args.roots]))
    lines.append(",".join(["n_games_in_mean", "", "", ""] + [str(len(own[r])) for m in METRICS for r in args.roots]))
    Path(args.out).write_text("\n".join(lines) + "\n")

    # terminal summary
    print(f"IoU {args.iou} (columns *_iou25 use IoU {LOOSE_IOU}); {len(games)} games in CSV, means over the {len(common)} games present in every family"
          + (f"; tiny bucket ignored as texture for: {', '.join(sorted(ignored))}" if ignored else ""))
    print(f"{'metric':18s}" + "".join(f"{short[r]:>10s}" for r in args.roots))
    print(f"{'n games':18s}" + "".join(f"{len(own[r]):>10d}" for r in args.roots))
    for m in METRICS:
        print(f"{m:18s}" + "".join(f"{(mean([table[(r, g)][m] for g in own[r]]) or 0):>10.3f}" for r in args.roots))
    last = args.roots[-1]
    for r in args.roots[:-1]:
        parts = []
        for m, tol in (("fgari", 0.005), ("f1", 0.01), ("small_recall", 0.01)):
            gs = [g for g in common if has(r, g) and has(last, g)]
            w = sum(1 for g in gs if (table[(last, g)][m] or 0) - (table[(r, g)][m] or 0) > tol)
            l = sum(1 for g in gs if (table[(r, g)][m] or 0) - (table[(last, g)][m] or 0) > tol)
            parts.append(f"{m}: better on {w}, worse on {l}")
        print(f"{short[last]} vs {short[r]}: " + "; ".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
