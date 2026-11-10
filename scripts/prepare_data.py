"""Split data_raw/progan_val into data/train + data/val and copy GANGen test sets.
Usage: python scripts/prepare_data.py --src data_raw --dst data --val_ratio 0.1 --seed 42
Expects data_raw/progan_val/<class>/{0_real,1_fake}/*  and data_raw/gangen/<GAN>/{0_real,1_fake}/*
Produces data/{train,val}/<class>/{0_real,1_fake}, data/test/<GAN>/{0_real,1_fake}, data/stats.json
"""
import argparse, json, random, shutil
from pathlib import Path

IMG = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def imgs(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG) if d.is_dir() else []


def copy(files, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f.name)


def find_root(base):
    """locate the folder whose children are class folders containing 0_real/1_fake"""
    for p in [base] + [q for q in base.rglob("*") if q.is_dir()]:
        kids = [k for k in p.iterdir() if k.is_dir()] if p.is_dir() else []
        if kids and all((k / "0_real").is_dir() for k in kids):
            return p
    raise SystemExit(f"no <class>/0_real layout found under {base}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data_raw")
    ap.add_argument("--dst", default="data")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    rng = random.Random(a.seed)
    stats = {"train": {"0_real": 0, "1_fake": 0}, "val": {"0_real": 0, "1_fake": 0}, "test": {}}

    root = find_root(src / "progan_val")
    classes = sorted(k.name for k in root.iterdir() if k.is_dir())
    for c in classes:
        for lab in ("0_real", "1_fake"):
            fs = imgs(root / c / lab)
            rng.shuffle(fs)
            nv = max(1, round(len(fs) * a.val_ratio))
            copy(fs[:nv], dst / "val" / c / lab)
            copy(fs[nv:], dst / "train" / c / lab)
            stats["val"][lab] += nv
            stats["train"][lab] += len(fs) - nv

    g = src / "gangen"
    for gan in sorted(k for k in g.iterdir() if k.is_dir()) if g.is_dir() else []:
        s = {}
        for lab in ("0_real", "1_fake"):
            fs = imgs(gan / lab)
            copy(fs, dst / "test" / gan.name / lab)
            s[lab] = len(fs)
        stats["test"][gan.name] = s

    stats["classes"] = classes
    (dst / "stats.json").parent.mkdir(parents=True, exist_ok=True)
    (dst / "stats.json").write_text(json.dumps(stats, indent=2))
    print("--classes", ",".join(classes))
    print(json.dumps({k: v for k, v in stats.items() if k != "classes"}, indent=2))


if __name__ == "__main__":
    main()
