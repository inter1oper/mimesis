"""
Inline everything into one self-contained HTML file.

render/index.html fetches ./data/*.json, which needs a server -- browsers refuse
fetch() from file://. This produces one file that opens by double-click with no
server and no network: the JSON bundle, the serif, and the two painting
photographs as backdrops.

The backdrops are a PREVIEW aid. In the installation the projector shows only
the overlay and the paint is physical, so `p` toggles them off.

    python tools/make_standalone.py [--no-images]
"""
import argparse
import base64
import json
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent.parent


def b64(path: pathlib.Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def painting_datauri(panel: str, width: int) -> str | None:
    """Downscale the rectified painting to a backdrop-sized JPEG."""
    src = root / f"out/panel_{panel}_rectified.png"
    if not src.exists():
        return None
    dst = root / f"out/_backdrop_{panel}.jpg"
    code = (
        "import cv2;"
        f"im=cv2.imread(r'{src}');"
        f"s={width}/im.shape[1];"
        "im=cv2.resize(im,None,fx=s,fy=s,interpolation=cv2.INTER_AREA);"
        f"cv2.imwrite(r'{dst}',im,[cv2.IMWRITE_JPEG_QUALITY,80])"
    )
    subprocess.run([str(root / ".venv/bin/python"), "-c", code], check=True)
    return b64(dst, "image/jpeg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--backdrop-width", type=int, default=1400)
    ap.add_argument("--audio-dir", type=pathlib.Path,
                    default=root / "render/audio",
                    help="Directory holding panel_a_voice.mp3 and panel_b_voice.mp3. "
                         "The two narrations total roughly 12 MB, which base64 "
                         "inflates past what belongs in one html file, so they are "
                         "referenced and must travel beside it.")
    args = ap.parse_args()

    src = (root / "render/index.html").read_text()
    names = re.findall(r'"([^"]+)"',
                       re.search(r'const FILES=\[(.*?)\];', src, re.S).group(1))
    data = {n: json.loads((root / "render/data" / f"{n}.json").read_text()) for n in names}

    out = src.replace("FONT_LIGHT", b64(root / "render/fonts/SourceSerif4-Light.ttf", "font/ttf"))
    out = out.replace("FONT_REGULAR", b64(root / "render/fonts/SourceSerif4-Regular.ttf", "font/ttf"))

    images = {}
    if not args.no_images:
        for p in ("a", "b"):
            uri = painting_datauri(p, args.backdrop_width)
            if uri:
                images["panel_" + p] = uri

    audio_js, voices = "", {}
    if args.audio_dir and args.audio_dir.exists():
        for P in ("A", "B"):
            f = args.audio_dir / f"panel_{P.lower()}_voice.mp3"
            if f.exists():
                voices[P] = f"audio/{f.name}"
                print(f"  voice {P}: {f.name} "
                      f"({f.stat().st_size/1024/1024:.1f} MB, referenced)")
    if voices:
        audio_js = "window.MIMESIS_AUDIO=" + json.dumps(voices) + ";"

    head = ("<script>" + audio_js + "window.MIMESIS_DATA="
            + json.dumps(data, separators=(",", ":")) + ";"
            + "window.MIMESIS_IMAGES=" + json.dumps(images, separators=(",", ":"))
            + ";</script>\n")
    out = out.replace('<script>\n"use strict";', head + '<script>\n"use strict";', 1)

    dest = root / "render/mimesis_standalone.html"
    dest.write_text(out)
    print(f"{dest}  {len(out)/1024/1024:.2f} MB  "
          f"({len(names)} json, {len(images)} backdrops, serif inlined)")


if __name__ == "__main__":
    main()
