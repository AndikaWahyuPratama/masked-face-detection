#!/usr/bin/env python3
"""Validasi dataset YOLO hasil prepare_fmld.py + buat grafik & contoh gambar untuk laporan.

Pakai:
  pip install matplotlib pillow
  python check_dataset.py --data .\\fmld_yolo --out .\\reports
"""
import argparse, random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

CLASSES = ["masked_face", "unmasked_face", "incorrectly_masked_face"]
COLORS = ["#2a9d8f", "#e76f51", "#264653"]  # hijau-biru, oranye, biru tua (beda jelas)
SPLITS = ["train", "val", "test"]


def read_labels(path):
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            rows.append(ln.split())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n_samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    counts, issues, n_img = {}, [], {}
    for sp in SPLITS:
        c, imgs = Counter(), list((data / "images" / sp).glob("*"))
        n_img[sp] = len(imgs)
        lbl_dir = data / "labels" / sp
        lbls = {p.stem: p for p in lbl_dir.glob("*.txt")}
        for im in imgs:
            if im.stem not in lbls:
                issues.append(f"{sp}: label hilang untuk {im.name}")
        for stem, p in lbls.items():
            rows = read_labels(p)
            if not rows:
                issues.append(f"{sp}: label kosong {p.name}")
            for r in rows:
                try:
                    cid = int(r[0]); vals = list(map(float, r[1:5]))
                except Exception:
                    issues.append(f"{sp}: format salah {p.name}: {r}"); continue
                if len(r) != 5: issues.append(f"{sp}: kolom != 5 di {p.name}")
                if cid not in (0, 1, 2): issues.append(f"{sp}: class id {cid} di {p.name}")
                if any(v < 0 or v > 1 for v in vals): issues.append(f"{sp}: nilai di luar 0-1 di {p.name}")
                if vals[2] <= 0 or vals[3] <= 0: issues.append(f"{sp}: lebar/tinggi <= 0 di {p.name}")
                c[CLASSES[cid] if cid in (0, 1, 2) else "?"] += 1
        counts[sp] = c

    # ---- ringkasan teks
    lines = ["Split | gambar | " + " | ".join(CLASSES) + " | % incorrectly_masked"]
    for sp in SPLITS:
        tot = sum(counts[sp].values()) or 1
        lines.append(f"{sp} | {n_img[sp]} | " + " | ".join(str(counts[sp][k]) for k in CLASSES)
                     + f" | {100 * counts[sp]['incorrectly_masked_face'] / tot:.2f}%")
    lines.append(f"\nMasalah ditemukan: {len(issues)}")
    lines += issues[:20]
    (out / "dataset_check.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # ---- grafik distribusi kelas (per split)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.26
    for i, k in enumerate(CLASSES):
        xs = [j + (i - 1) * w for j in range(3)]
        vals = [counts[sp][k] for sp in SPLITS]
        bars = ax.bar(xs, vals, w, label=k, color=COLORS[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(SPLITS)
    ax.set_ylabel("Jumlah bounding box"); ax.set_title("Distribusi kelas FMLD per split")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=150); plt.close(fig)

    # ---- contoh gambar berlabel (utamakan gambar yang punya kelas langka)
    rng = random.Random(a.seed)
    lbl_dir = data / "labels" / "train"
    rare, other = [], []
    for p in lbl_dir.glob("*.txt"):
        (rare if any(r[0] == "2" for r in read_labels(p)) else other).append(p)
    pick = rng.sample(rare, min(3, len(rare))) + rng.sample(other, min(a.n_samples - 3, len(other)))
    cols = 4
    rows_n = (len(pick) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.4 * cols, 3.4 * rows_n))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes: ax.axis("off")
    for ax, p in zip(axes, pick):
        img_path = next((data / "images" / "train").glob(p.stem + ".*"))
        im = Image.open(img_path).convert("RGB"); W, H = im.size
        d = ImageDraw.Draw(im); lw = max(3, W // 120)
        for r in read_labels(p):
            c = int(r[0]); x, y, bw, bh = map(float, r[1:5])
            box = [(x - bw / 2) * W, (y - bh / 2) * H, (x + bw / 2) * W, (y + bh / 2) * H]
            d.rectangle(box, outline=COLORS[c], width=lw)
        ax.imshow(im); ax.set_title(p.stem, fontsize=7)
    handles = [plt.Line2D([0], [0], color=COLORS[i], lw=3, label=CLASSES[i]) for i in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out / "samples.png", dpi=130); plt.close(fig)
    print(f"\n[ok] hasil di: {out.resolve()}")


if __name__ == "__main__":
    main()
