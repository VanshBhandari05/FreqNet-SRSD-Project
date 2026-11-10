"""Download ProGAN val (Hugging Face) and GANGen-Detection test sets (Google Drive).
Usage: python scripts/download_data.py --dst data_raw [--gans BEGAN,STGAN,SNGAN,InfoMaxGAN]
Needs: pip install requests
GANGen Drive file ids: fill GANGEN_IDS (folder https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj).
"""
import argparse, re, tarfile, zipfile
from pathlib import Path
import requests

PROGAN_URL = "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip"
PROGAN_SIZE = 830_792_545
GANGEN_IDS = {  # name -> Google Drive file id (TO BE FILLED from the Drive folder listing)
    "BEGAN": None, "STGAN": None, "SNGAN": None, "InfoMaxGAN": None,
}


def stream(url, out, expected=None, session=None):
    s = session or requests.Session()
    pos = out.stat().st_size if out.exists() else 0
    if expected and pos == expected:
        return
    h = {"Range": f"bytes={pos}-"} if pos else {}
    with s.get(url, headers=h, stream=True, timeout=60) as r:
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        done = pos if mode == "ab" else 0
        with open(out, mode) as f:
            for ch in r.iter_content(1 << 20):
                f.write(ch)
                done += len(ch)
                if expected and done % (50 << 20) < (1 << 20):
                    print(f"\r{out.name}: {done/1e6:.0f}/{expected/1e6:.0f} MB", end="", flush=True)
    print()
    if expected and out.stat().st_size != expected:
        raise SystemExit(f"size mismatch for {out}: {out.stat().st_size} != {expected}")


def drive(fid, out):
    s = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    r = s.get(url, params={"id": fid}, stream=True)
    if "text/html" in r.headers.get("content-type", ""):
        html = r.text
        m = re.search(r'name="uuid" value="([^"]+)"', html)
        params = {"id": fid, "export": "download", "confirm": "t"}
        if m:
            params["uuid"] = m.group(1)
        if "quota" in html.lower():
            raise SystemExit("Google Drive quota exceeded; retry later or download manually")
        stream("https://drive.usercontent.google.com/download?" + "&".join(f"{k}={v}" for k, v in params.items()), out, session=s)
    else:
        with open(out, "wb") as f:
            for ch in r.iter_content(1 << 20):
                f.write(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="data_raw")
    ap.add_argument("--gans", default="BEGAN,STGAN,SNGAN,InfoMaxGAN")
    a = ap.parse_args()
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    if not (dst / "progan_val").is_dir():
        z = dst / "progan_val.zip"
        stream(PROGAN_URL, z, PROGAN_SIZE)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dst)
        z.unlink()
        print("ProGAN val extracted")

    (dst / "gangen").mkdir(exist_ok=True)
    for g in a.gans.split(","):
        if (dst / "gangen" / g).is_dir():
            continue
        fid = GANGEN_IDS.get(g)
        if not fid:
            print(f"[skip] no Drive id for {g}; fill GANGEN_IDS or place files manually")
            continue
        t = dst / "gangen" / f"{g}.tar.gz"
        drive(fid, t)
        with tarfile.open(t) as tf:
            tf.extractall(dst / "gangen")
        t.unlink()
    print("DONE")


if __name__ == "__main__":
    main()
