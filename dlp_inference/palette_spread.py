"""Per-game recolor ("bg->black") that makes small sprites expensive to miss.

Atari frames use a tiny exact palette (5-70 colors per game). The model's
reconstruction loss is a plain RGB MSE, so the penalty for *not* explaining a
sprite with a particle is the squared RGB distance between the sprite and the
background it is painted over. This transform maximizes that penalty:

  * per frame, the most frequent color (the background) becomes **black**;
  * every other color of the game - source black included when it is not
    the frame's dominant color (Boxing's black boxer on the ring) - is
    remapped onto the **bright faces** of the RGB cube (max channel = 255),
    smallest objects first, on hue directions kept apart from the colors that
    share a glimpse window with them, so that no two sprites that can meet
    collapse onto the same color after the downscale.

Everything is measured on the training frames alone - pixel statistics, no
labels, no per-game rule - and the resulting mapping is stored next to the
checkpoint (``weights/<Game>/palette.json``) so training and inference apply
the identical transform (``PaletteSpread.from_file``).

Building palettes by hand (train.py does this automatically):

  python -m dlp_inference.palette_spread --root <dataset> --games all --viz
     -> palettes/<Game>.json     the mapping and the statistics behind it
     -> palette_viz/<Game>.png   swatch strip + original/recolored frames

Runtime:

  from dlp_inference.palette_spread import PaletteSpread
  tf = PaletteSpread.from_file("weights/Asterix/palette.json")
  frames_rgb = tf(frames_rgb)        # (H,W,3) or (T,H,W,3) uint8, true RGB in and out
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PALETTE_DIR = REPO / "palettes"
VIZ_DIR = REPO / "palette_viz"

PAD_TO, IMAGE_SIZE = 256, 128            # training canvas / model resolution (config_default.json)
GLIMPSE_PX = 32                          # one 16-px glimpse at 128 = 32 frame px (anchor_s 0.125)
MAX_PENALTY = 3 * 255.0 ** 2             # white vs black, per pixel, summed over 3 channels


# --------------------------------------------------------------------------- #
# pixels <-> int24 color codes
def _codes(frames: np.ndarray) -> np.ndarray:
    f = frames.reshape(-1, 3).astype(np.int64)
    return (f[:, 0] << 16) | (f[:, 1] << 8) | f[:, 2]


def _decode(codes: np.ndarray) -> np.ndarray:
    c = np.asarray(codes, np.int64)
    return np.stack([(c >> 16) & 255, (c >> 8) & 255, c & 255], -1).astype(np.uint8)


def load_rgb(path: str) -> np.ndarray:
    """Dataset PNG -> TRUE-RGB (H, W, 3) uint8. The OCAtari PNGs are stored
    channel-swapped (cv2.imread yields true RGB), so the PIL load is flipped."""
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))[..., ::-1].copy()


def _effective_area(mask, top, left):
    """Miss-penalty weight of one component after a 2x2 downscale: a block
    covered by fraction f of the object contributes f^2 * ||T(obj) - T(bg)||^2,
    so the penalty scales with A_eff = sum_blocks f^2 (<= area / 4)."""
    ys, xs = np.nonzero(mask)
    by, bx = (ys + top) // 2, (xs + left) // 2
    _, cnt = np.unique(by * (PAD_TO // 2) + bx, return_counts=True)
    return float(((cnt / 4.0) ** 2).sum())


# --------------------------------------------------------------------------- #
# statistics of a game's colors, from its training frames only
def collect_palette(paths, max_frames: int = 300, cc_frames: int = 60, static_thresh: float = 0.8):
    """Exact colors used across ``paths`` (K,3 uint8, by descending pixel
    count), their counts (K,), and per-color statistics:

      dominant_frac  fraction of the sampled frames in which the color is the
                     most frequent one (the frame's background)
      static_frac    fraction of the color's pixels sitting at positions that
                     hold the color in >= static_thresh of the frames
      eff_area       median effective area (sum f^2 after the 2x2 downscale)
                     of the color's connected components
      med_area, med_extent   median component area / max(width, height), px
      context        histogram of the colors in the 1-px ring around the
                     color's pixels (which colors touch it)
      cooc           fraction of frames in which another color lies within
                     one glimpse window of the color (which sprites can meet)
    Component statistics are measured on every ``len(paths) // cc_frames``-th
    sampled frame."""
    import cv2
    if len(paths) > max_frames:
        idx = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
        paths = [paths[i] for i in idx]
    cc_every = max(1, len(paths) // cc_frames)
    acc: dict[int, int] = {}
    areas: dict[int, list] = {}
    extents: dict[int, list] = {}
    pos_count: dict[int, np.ndarray] = {}
    context: dict[int, dict] = {}
    ring_k = np.ones((3, 3), np.uint8)
    win_k = np.ones((GLIMPSE_PX + 1, GLIMPSE_PX + 1), np.uint8)
    eff: dict[int, list] = {}
    cooc: dict[int, dict] = {}
    dom_count: dict[int, int] = {}
    n_cc = 0
    for fi, p in enumerate(paths):
        fr = load_rgb(p)
        codes = _codes(fr)
        u, c = np.unique(codes, return_counts=True)
        for k, n in zip(u.tolist(), c.tolist()):
            acc[k] = acc.get(k, 0) + n
        kdom = int(u[np.argmax(c)])
        dom_count[kdom] = dom_count.get(kdom, 0) + 1
        if fi % cc_every:
            continue
        n_cc += 1
        cmap = codes.reshape(fr.shape[:2])
        h, w = cmap.shape
        top, left = (PAD_TO - h) // 2, (PAD_TO - w) // 2
        masks = {}
        for k in u.tolist():
            mask = (cmap == k)
            masks[k] = mask
            n_lab, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            if n_lab > 1:
                areas.setdefault(k, []).extend(st[1:, cv2.CC_STAT_AREA].tolist())
                extents.setdefault(k, []).extend(
                    np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]).tolist())
                if n_lab <= 64:                      # skip texture colors with hundreds of components
                    eff.setdefault(k, []).extend(_effective_area(lab == j, top, left) for j in range(1, n_lab))
            if k not in pos_count:
                pos_count[k] = np.zeros(mask.shape, np.uint16)
            pos_count[k] += mask
            ring = cv2.dilate(mask.astype(np.uint8), ring_k) > 0
            ring &= ~mask
            if ring.any():
                rc, rn = np.unique(cmap[ring], return_counts=True)
                ctx = context.setdefault(k, {})
                for cc_, n in zip(rc.tolist(), rn.tolist()):
                    ctx[cc_] = ctx.get(cc_, 0) + n
        # co-occurrence within one glimpse window, counted per frame
        small = [k for k in u.tolist() if masks[k].sum() <= 4000]
        dil = {k: cv2.dilate(masks[k].astype(np.uint8), win_k) > 0 for k in small}
        for a in small:
            ca = cooc.setdefault(a, {})
            for b in small:
                if b != a and (dil[a] & masks[b]).any():
                    ca[b] = ca.get(b, 0) + 1
    codes = np.array(sorted(acc, key=lambda k: -acc[k]), np.int64)
    counts = np.array([acc[k] for k in codes], np.int64)
    static_frac = []
    for k in codes.tolist():
        pc = pos_count.get(k)
        if pc is None or pc.sum() == 0:
            static_frac.append(0.0)
        else:
            static_frac.append(float(pc[pc >= static_thresh * n_cc].sum() / pc.sum()))
    stats = {
        "codes": codes,
        "dominant_frac": np.array([dom_count.get(int(k), 0) / len(paths) for k in codes.tolist()]),
        "static_frac": np.array(static_frac),
        "eff_area": np.array([float(np.median(eff[int(k)])) if eff.get(int(k)) else 0.0 for k in codes.tolist()]),
        "med_area": np.array([float(np.median(areas.get(int(k), [0]))) for k in codes]),
        "med_extent": np.array([float(np.median(extents.get(int(k), [0]))) for k in codes]),
        "context": [context.get(int(k), {}) for k in codes.tolist()],
        "cooc": [{c: n / n_cc for c, n in cooc.get(int(k), {}).items()} for k in codes.tolist()],
        "n_cc_frames": n_cc,
    }
    return _decode(codes), counts, stats


# --------------------------------------------------------------------------- #
# target placement
def _rgb_grid(step):
    g = np.arange(0, 256, step, dtype=np.int64)
    if g[-1] != 255:
        g = np.append(g, 255)
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)


def _union_find_groups(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def build_mapping(colors, counts, stats, size_emphasis=1.0, touch_group=0.25,
                  cooc_full=0.2, cooc_half=0.05, max_group=8, face_step=15,
                  min_spacing_all=0.3, max_spacing=0.35, max_obj_static=0.7) -> dict:
    """Place every color of the game on the bright faces of the RGB cube.

    The background is black (the frame's dominant color, applied per frame at
    runtime), so a color's miss penalty is simply its squared norm and every
    target is chosen as far from black as the spacing constraints allow.
    Colors that touch inside one blob (measured ring context) form a sprite
    group placed together; groups are placed smallest effective area first
    (moving before static), each one as far as possible from the targets of
    the groups it shares a glimpse window with (measured co-occurrence:
    ``cooc_full`` -> full spacing, ``cooc_half`` -> half), and at least
    ``min_spacing_all`` from every other target. Spacing is angular
    (min(norm) * sin(angle)): after the downscale, partial coverage slides a
    color along its ray toward black, so only a difference in direction
    survives. Colors too rare to be measured inherit the target of their
    nearest placed source color."""
    colors = np.asarray(colors, np.uint8)
    K = len(colors)
    share = counts / counts.sum()
    codes = stats["codes"]; code2i = {int(c): i for i, c in enumerate(codes)}
    rgb = colors.astype(np.float64) / 255.0
    eff = stats["eff_area"]; static = stats["static_frac"]
    T = np.zeros((K, 3), np.float64)
    black = np.zeros(3)

    # sprite groups: colors that touch each other inside one blob
    touch = np.zeros((K, K))
    for i in range(K):
        tot = sum(stats["context"][i].values()) or 1
        for k, n in stats["context"][i].items():
            if k in code2i:
                touch[i, code2i[k]] = n / tot
    edges = [(i, j) for i in range(K) for j in range(K) if i < j and max(touch[i, j], touch[j, i]) >= touch_group]
    groups = _union_find_groups(K, edges)
    groups = sum(([[m] for m in g] if len(g) > max_group else [g] for g in groups), [])

    def gstat(g):
        a = float(sum(eff[m] for m in g))
        w = np.array([counts[m] for m in g], float); w = w / w.sum()
        return a, float((w * static[[*g]]).sum())
    ginfo = {tuple(g): gstat(g) for g in groups}
    measured = [g for g in groups if ginfo[tuple(g)][0] > 0]
    unmeasured = [g for g in groups if ginfo[tuple(g)][0] <= 0]
    # placement order: moving before static (the background decoder memorizes
    # static objects whatever their color), then smallest effective area, then rarest
    order = sorted(measured, key=lambda g: (ginfo[tuple(g)][1] > max_obj_static, ginfo[tuple(g)][0],
                                            sum(share[m] for m in g)))
    n = len(order)
    cooc = np.zeros((K, K))
    for i in range(K):
        for k, f in stats["cooc"][i].items():
            if k in code2i:
                cooc[i, code2i[k]] = f

    def gcooc(g, h):
        return max(cooc[a, b] for a in g for b in h) if g and h else 0.0

    # candidates: the bright faces of the cube (max channel = 255)
    faces = _rgb_grid(face_step); faces = faces[faces.max(1) == 255].astype(np.float64) / 255.0

    def miss_score(cands):
        return ((cands - black) ** 2).sum(1)
    # spacing cap: above ~0.35 any required spacing evicts every later object
    # from the high-penalty region and defeats the small-object priority
    s_base = float(np.clip(0.9 * np.sqrt(3.0 / max(1, n)), 0.15, 0.25))

    def sep(cands, t):
        nc = np.sqrt((cands ** 2).sum(1)) + 1e-9; nt = np.sqrt((t ** 2).sum()) + 1e-9
        cos = np.clip((cands @ t) / (nc * nt), -1, 1)
        return np.minimum(nc, nt) * np.sqrt(1 - cos ** 2)
    placed = []                                    # (group, target)
    group_target = {}
    used = np.zeros(len(faces), bool)              # face candidates already handed out

    def feasible(req, relax):
        ok = ~used
        for th, s_h in req:
            ok &= sep(faces, th) >= (min(s_h, min_spacing_all) if relax < 1.0 and s_h > min_spacing_all else s_h) * (relax if s_h > min_spacing_all else 1.0)
        return ok
    for g in order:
        score = miss_score(faces)
        req = []
        for j, (h, th) in enumerate(placed):
            co = gcooc(g, h)
            s_h = min(max_spacing, s_base * (1.0 + size_emphasis * (1.0 - j / max(1, n - 1))))   # smaller -> more room, capped
            if co >= cooc_full:
                req.append((th, s_h))
            elif co >= cooc_half:
                req.append((th, 0.5 * s_h))
            else:                                   # never share a glimpse: only keep them distinct
                req.append((th, min_spacing_all))
        # co-occurrence spacing may be relaxed when the faces are crowded; the
        # distinctness floor and exact-target reuse never are
        relax = 1.0
        while True:
            ok = feasible(req, relax)
            if ok.any() or relax < 0.2:
                break
            relax *= 0.85
        if not ok.any():                           # only the floor left unsatisfiable: best unused face
            ok = ~used
        cand_idx = np.where(ok)[0]
        k = cand_idx[np.lexsort((-(faces[cand_idx] ** 2).sum(1), -score[cand_idx]))][0]
        used[k] = True
        tg = faces[k]
        placed.append((g, tg)); group_target[tuple(g)] = tg
        # members: the biggest gets tg; the others nearby on the faces, distinct
        members = sorted(g, key=lambda m: -counts[m])
        T[members[0]] = tg
        sib = [tg]
        for m in members[1:]:
            d = sep(faces, tg)
            ok = (d >= 0.12) & (d <= 0.35)               # same sprite: near the base hue, still distinct
            for sv in sib[1:]:
                ok &= sep(faces, sv) >= 0.12
            for h, th in placed[:-1]:
                if gcooc([m], h) >= cooc_half:
                    ok &= sep(faces, th) >= 0.5 * s_base
            ok &= ~used
            idx = np.where(ok)[0] if ok.any() else np.where(~used)[0]
            sc = miss_score(faces[idx])
            km = idx[int(np.argmax(sc))]
            used[km] = True
            T[m] = faces[km]
            sib.append(T[m])
    # unmeasured / ultra-rare colors: inherit the target of the nearest placed source color
    done = np.array([any(i in g for g in order) for i in range(K)])
    for g in unmeasured:
        for m in g:
            src = np.where(done)[0]
            j = src[int(np.argmin(((rgb[src] - rgb[m]) ** 2).sum(1)))]
            T[m] = T[j]

    dst = np.clip(np.round(T * 255), 0, 255).astype(np.uint8)
    gid = {m: gi for gi, g in enumerate(groups) for m in g}
    inherited = {m for g in unmeasured for m in g}
    mapping = [{"src": colors[j].tolist(), "dst": dst[j].tolist(),
                "dominant_frac": round(float(stats["dominant_frac"][j]), 4),
                "inherited": bool(j in inherited),     # too rare to measure: takes its nearest source color's target
                "count": int(counts[j]), "share": round(float(share[j]), 5),
                "median_component_px": round(float(stats["med_area"][j]), 1),
                "median_extent_px": round(float(stats["med_extent"][j]), 1),
                "static_frac": round(float(static[j]), 3),
                "eff_area": round(float(eff[j]), 2), "group": gid.get(j)} for j in range(K)]
    return {"mode": "bgblack", "mapping": mapping, "n_colors": K,
            "n_groups": len(groups), "groups": [[int(m) for m in g] for g in groups],
            "n_inherited": len(inherited),
            "thresholds": {"size_emphasis": size_emphasis, "touch_group": touch_group,
                           "cooc_full": cooc_full, "cooc_half": cooc_half, "max_group": max_group,
                           "face_step": face_step, "min_spacing_all": min_spacing_all,
                           "max_spacing": max_spacing, "max_obj_static": max_obj_static}}


def miss_penalty(colors, is_bg, stats, mapping=None):
    """Offline proxy for what the recolor buys: the RGB-MSE penalty the model
    pays, per pixel, when a color is NOT captured by a particle and is
    reconstructed as its local background instead (``is_bg`` marks the
    background colors, i.e. those that are the frame's dominant color
    somewhere; ``mapping`` is src -> dst, identity when None). Returns one
    dict per non-background color, smallest first, with the penalty as a
    fraction of MAX_PENALTY."""
    colors = np.asarray(colors, np.uint8)
    codes = stats["codes"]
    code2rgb = {int(c): colors[i].astype(np.float64) for i, c in enumerate(codes)}
    code2bg = {int(c): bool(is_bg[i]) for i, c in enumerate(codes)}
    T = (lambda v: np.asarray(mapping.get(tuple(int(x) for x in v), v), np.float64)) if mapping else (lambda v: np.asarray(v, np.float64))
    out = []
    for i in [i for i in np.argsort(stats["med_area"]) if not is_bg[i]]:
        ctx = stats["context"][i]
        tot = sum(n for c, n in ctx.items() if code2bg.get(c, False))
        if not tot:
            continue
        bgs = [(c, n / tot) for c, n in ctx.items() if code2bg.get(c, False)]
        o = T(colors[i])
        pen = sum(f * ((o - T(code2rgb[c])) ** 2).sum() for c, f in bgs)
        out.append({"src": colors[i].tolist(), "dst": [int(v) for v in o], "px": float(stats["med_area"][i]),
                    "penalty_frac": float(pen / MAX_PENALTY)})
    return out


def penalty_summary(pen_list, n_small=4):
    """Mean penalty fraction over all object colors and over the n smallest."""
    if not pen_list:
        return {"all": None, "smallest": None}
    return {"all": round(float(np.mean([p["penalty_frac"] for p in pen_list])), 4),
            "smallest": round(float(np.mean([p["penalty_frac"] for p in pen_list[:n_small]])), 4)}


# --------------------------------------------------------------------------- #
class PaletteSpread:
    """Callable frame transform (true RGB in, true RGB out): the palette's
    LUT, then the frame's most frequent source color -> black. Colors never
    seen while building the palette snap to their nearest known source color."""

    def __init__(self, path):
        path = Path(path)
        spec = json.loads(path.read_text())
        if spec.get("mode", "bgblack") != "bgblack":
            raise ValueError(f"{path}: palette mode {spec.get('mode')!r} is not supported (expected 'bgblack')")
        self.game = spec.get("game")
        self.path = path
        self.src = np.array([m["src"] for m in spec["mapping"]], np.uint8)
        self.dst = np.array([m["dst"] for m in spec["mapping"]], np.uint8)
        self.lut = {int(c): i for i, c in enumerate(_codes(self.src[:, None, :]))}
        self.src_f = self.src.astype(np.float32)
        self.unseen = 0                                # unseen colors met so far

    @classmethod
    def from_file(cls, path):
        return cls(path)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        codes = _codes(frames)
        u, inv = np.unique(codes, return_inverse=True)
        idx = np.empty(len(u), np.int64)
        for j, c in enumerate(u.tolist()):
            k = self.lut.get(c)
            if k is None:                             # unseen color -> nearest source
                v = _decode(np.array([c]))[0].astype(np.float32)
                k = int(np.argmin(((self.src_f - v) ** 2).sum(1)))
                self.unseen += 1
            idx[j] = k
        out = self.dst[idx[inv]].reshape(frames.shape)
        return self._dominant_to_black(frames, codes, out)

    @staticmethod
    def _dominant_to_black(frames, codes, out):
        """Per frame: the most frequent SOURCE color of that frame becomes black
        (ties -> the smaller int24 code), so a game whose background changes
        with the level follows along; every other color keeps its LUT target."""
        single = frames.ndim == 3
        cm = codes.reshape((1,) + frames.shape[:2]) if single else codes.reshape(frames.shape[:3])
        out = out.reshape(cm.shape + (3,)).copy()
        for t in range(cm.shape[0]):
            u, c = np.unique(cm[t], return_counts=True)
            out[t][cm[t] == u[np.argmax(c)]] = 0
        return out.reshape(frames.shape)


# --------------------------------------------------------------------------- #
def _frames_for(game: str, roots) -> list[str]:
    for root, split in roots:
        paths = sorted(glob.glob(str(Path(root) / "images" / split / f"{game}_*.png")),
                       key=lambda p: int(Path(p).stem.split("_")[-1]))
        if paths:
            return paths
    return []


def build_palette(game: str, root, out_path, split: str = "train", frames: int = 300,
                  cc_frames: int = 60, **knobs) -> dict:
    """Build <game>'s palette from the ``split`` images of ``root`` alone and
    write it to ``out_path``. train.py calls this so a training run derives its
    recolor from exactly the frames it trains on. Returns the spec."""
    paths = _frames_for(game, [(root, split)])
    if not paths:
        raise FileNotFoundError(f"no {split} frames for {game} under {root}")
    colors, counts, stats = collect_palette(paths, frames, cc_frames)
    spec = build_mapping(colors, counts, stats, **knobs)
    # audit: colors that are the frame's dominant color somewhere are the
    # background (black at runtime); the proxy is what every other color pays
    is_bg = np.array([m["dominant_frac"] > 0 for m in spec["mapping"]])
    mapping = {tuple(m["src"]): ([0, 0, 0] if is_bg[i] else m["dst"]) for i, m in enumerate(spec["mapping"])}
    spec["miss_penalty"] = {"before": penalty_summary(miss_penalty(colors, is_bg, stats)),
                            "after": penalty_summary(miss_penalty(colors, is_bg, stats, mapping))}
    spec.update({"game": game, "n_frames_sampled": min(len(paths), frames), "split": split,
                 "source": str(Path(paths[0]).parent)})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(spec, indent=1))
    return spec


def render_game(game: str, paths, spec: dict, tf: PaletteSpread, out: Path, n_show: int = 3):
    """Swatch strip (source over target, by pixel share) + n_show frames before/after."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.linspace(0, len(paths) - 1, n_show).round().astype(int)
    frames = [load_rgb(paths[i]) for i in idx]
    mp = sorted(spec["mapping"], key=lambda m: -m["share"])
    fig = plt.figure(figsize=(6, 1.6 + 3.1 * n_show))
    gs = fig.add_gridspec(n_show + 1, 2, height_ratios=[1.1] + [3] * n_show, hspace=0.15, wspace=0.05)
    ax = fig.add_subplot(gs[0, :])
    strip = np.zeros((2, len(mp), 3), np.uint8)
    strip[0] = [m["src"] for m in mp]; strip[1] = [m["dst"] for m in mp]
    ax.imshow(strip, interpolation="nearest", aspect="auto")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["source", "target"], fontsize=8); ax.xaxis.tick_top()
    ax.set_xticks(range(len(mp)))
    ax.set_xticklabels([f"dom {100 * m['dominant_frac']:.0f}%" if m["dominant_frac"] >= 0.05 else f"{m['median_component_px']:.0f}px"
                        for m in mp], fontsize=6, rotation=90)
    pen = spec.get("miss_penalty", {})
    ax.set_title(f"{game}: {spec['n_colors']} colors; the frame's dominant color -> black, the rest -> bright faces. "
                 f"miss-penalty proxy {pen.get('before', {}).get('all')} -> {pen.get('after', {}).get('all')}", fontsize=8)
    for r, fr in enumerate(frames):
        for c, (img, name) in enumerate([(fr, "original"), (tf(fr), "recolored")]):
            a = fig.add_subplot(gs[r + 1, c]); a.imshow(img, interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(name, fontsize=9)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="dataset root (images/{train,val,test})")
    ap.add_argument("--extra-root", default=None, help="second dataset root tried when a game has no frames under --root")
    ap.add_argument("--games", nargs="+", default=["all"], help="game names, or 'all' for games.txt")
    ap.add_argument("--frames", type=int, default=300, help="frames sampled per game")
    ap.add_argument("--cc-frames", type=int, default=60, help="frames used for component statistics")
    ap.add_argument("--viz", action="store_true", help="write <viz-dir>/<Game>.png")
    ap.add_argument("--palette-dir", default=str(PALETTE_DIR))
    ap.add_argument("--viz-dir", default=str(VIZ_DIR))
    args = ap.parse_args()

    games = (REPO / "games.txt").read_text().split() if args.games == ["all"] else args.games
    roots = [(args.root, "train"), (args.root, "test")]
    if args.extra_root:
        roots += [(args.extra_root, "train"), (args.extra_root, "test")]
    pdir, vdir = Path(args.palette_dir), Path(args.viz_dir)
    pdir.mkdir(exist_ok=True, parents=True)
    if args.viz:
        vdir.mkdir(exist_ok=True, parents=True)
    print(f"{'game':17s}{'colors':>7s}{'groups':>7s}{'rare':>6s}{'miss-penalty all':>18s}{'smallest4':>12s}   (% of max, before -> after)")
    for g in games:
        paths = _frames_for(g, roots)
        if not paths:
            print(f"{g:17s}   no frames found - skipped")
            continue
        root, split = next((r, s) for r, s in roots if glob.glob(str(Path(r) / "images" / s / f"{g}_*.png")))
        spec = build_palette(g, root, pdir / f"{g}.json", split=split, frames=args.frames, cc_frames=args.cc_frames)
        mp = spec["miss_penalty"]

        def pct(x):
            return "-" if x is None else f"{100 * x:.0f}%"
        print(f"{g:17s}{spec['n_colors']:>7d}{spec['n_groups']:>7d}{spec['n_inherited']:>6d}"
              f"{pct(mp['before']['all']) + ' -> ' + pct(mp['after']['all']):>18s}"
              f"{pct(mp['before']['smallest']) + ' -> ' + pct(mp['after']['smallest']):>12s}", flush=True)
        if args.viz:
            render_game(g, paths, spec, PaletteSpread(pdir / f"{g}.json"), vdir / f"{g}.png")
    print(f"mappings in {pdir}/" + (f", images in {vdir}/" if args.viz else ""))


if __name__ == "__main__":
    main()
