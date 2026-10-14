#!/usr/bin/env python3
"""FMLD (PASCAL VOC XML) -> YOLO format + train/val/test split + data.yaml.

Split:
  train/val : dari anotasi FMLD 'train' (val diambil 10% secara acak per-gambar,
              distribusi gambar yang memuat 'incorrectly_masked_face' dijaga
              proporsinya agar kelas langka tetap ada di val).
  test      : FMLD 'test' resmi (tidak disentuh saat training).

Contoh:
  python prepare_fmld.py \
    --ann_dir  fmld_raw/ann/FMLD_annotations \
    --images_root data/MAFA data/WIDER \
    --out datasets/fmld_yolo
"""
import argparse, os, random, shutil, sys, json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["masked_face", "unmasked_face", "incorrectly_masked_face"]
CID = {c: i for i, c in enumerate(CLASSES)}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def index_images(roots):
    idx, dup = {}, 0
    for r in roots:
        for dp, _, fns in os.walk(r):
            for fn in fns:
                if Path(fn).suffix.lower() in IMG_EXT:
                    if fn in idx:
                        dup += 1
                    idx[fn] = os.path.join(dp, fn)
    return idx, dup


def parse_xml(path, stats, drop_difficult):
    root = ET.parse(path).getroot()
    fn = root.findtext("filename")
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    boxes = []
    for o in root.findall("object"):
        name = (o.findtext("name") or "").strip()
        if name not in CID:                      # mis. 'invalid_face'
            stats["skip_" + name] += 1
            continue
        if drop_difficult and (o.findtext("difficult") or "0").strip() == "1":
            stats["skip_difficult"] += 1
            continue
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))   # clip ke gambar
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            stats["skip_degenerate"] += 1
            continue
        boxes.append((CID[name], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                      (x2 - x1) / w, (y2 - y1) / h))
    return fn, w, h, boxes


def put(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def build(split_name, items, out, mode, counters):
    for fn, src, boxes in items:
        stem = Path(fn).stem
        put(src, out / "images" / split_name / (stem + Path(fn).suffix.lower()), mode)
        lbl = out / "labels" / split_name / (stem + ".txt")
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes), encoding="utf-8")
        for b in boxes:
            counters[split_name][CLASSES[b[0]]] += 1
        counters[split_name]["_images"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True, help="folder berisi train/ dan test/ (hasil unzip FMLD_annotations.zip)")
    ap.add_argument("--images_root", nargs="+", required=True, help="folder gambar MAFA & WIDER (dicari rekursif)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", choices=["symlink", "copy"], default="copy" if os.name == "nt" else "symlink")
    ap.add_argument("--test_difficult", choices=["drop", "keep"], default="drop",
                    help="drop: buang box difficult=1 di test (default)")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    idx, dup = index_images(a.images_root)
    print(f"[info] {len(idx)} gambar ditemukan ({dup} nama file duplikat)")

    stats = Counter()
    data = {"train": [], "test": []}
    missing = Counter()
    for sp in data:
        for xml in sorted((Path(a.ann_dir) / sp).glob("*.xml")):
            fn, w, h, boxes = parse_xml(xml, stats, drop_difficult=(sp == "test" and a.test_difficult == "drop"))
            if fn not in idx:
                missing[sp] += 1
                continue
            if not boxes:
                stats[f"skip_empty_image_{sp}"] += 1
                continue
            data[sp].append((fn, idx[fn], boxes))
    print("[info] gambar tidak ditemukan:", dict(missing))
    if missing:
        print("[warn] pastikan MAFA (train+test) dan WIDER (train+val) sudah diunduh lengkap.")

    if not data["train"] or not data["test"]:
        sys.exit("[error] Tidak ada gambar yang cocok dengan anotasi. Pastikan gambar MAFA dan WIDER "
                 "sudah diekstrak ke folder --images_root, lalu jalankan ulang. Tidak ada folder output yang dibuat.")

    # train -> train/val, stratified berdasarkan keberadaan kelas langka
    rng = random.Random(a.seed)
    has_rare = lambda it: any(b[0] == CID["incorrectly_masked_face"] for b in it[2])
    rare = [it for it in data["train"] if has_rare(it)]
    rest = [it for it in data["train"] if not has_rare(it)]
    val, train = [], []
    for grp in (rare, rest):
        rng.shuffle(grp)
        k = int(len(grp) * a.val_ratio)
        val += grp[:k]
        train += grp[k:]

    counters = defaultdict(Counter)
    build("train", train, out, a.link, counters)
    build("val", val, out, a.link, counters)
    build("test", data["test"], out, a.link, counters)

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASSES)}\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)), encoding="utf-8")
    (out / "dataset_summary.json").write_text(json.dumps(
        {"splits": {k: dict(v) for k, v in counters.items()}, "skipped": dict(stats), "missing_images": dict(missing)}, indent=2), encoding="utf-8")

    print("\n=== RINGKASAN ===")
    for sp in ("train", "val", "test"):
        c = counters[sp]
        print(f"{sp:5s} gambar={c['_images']:6d} | " + " | ".join(f"{k}={c[k]}" for k in CLASSES))
    print("dilewati:", dict(stats))
    print("[ok] selesai. data.yaml ada di folder output.")


if __name__ == "__main__":
    sys.exit(main())
