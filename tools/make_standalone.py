"""
Inline the JSON bundle into a single self-contained HTML file.

render/index.html fetches ./data/*.json, which means it needs to be served over
http -- browsers refuse fetch() from file://. In a gallery that is one more
thing to go wrong at 9am. This produces one file that opens by double-click,
with no server and no network.

    python tools/make_standalone.py
"""
import json
import pathlib
import re

root = pathlib.Path(__file__).resolve().parent.parent
src = (root / "render/index.html").read_text()
names = re.findall(r'"([^"]+)"', re.search(r'const FILES = \[(.*?)\];', src, re.S).group(1))

data = {n: json.loads((root / "render/data" / f"{n}.json").read_text()) for n in names}
blob = json.dumps(data, separators=(",", ":"))
out = src.replace('<script>\n"use strict";',
                  "<script>window.MIMESIS_DATA=" + blob + ";</script>\n<script>\n\"use strict\";", 1)
dest = root / "render/mimesis_standalone.html"
dest.write_text(out)
print(f"{dest}  {len(out)/1024/1024:.2f} MB  ({len(names)} json files inlined)")
