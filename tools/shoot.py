"""Screenshot the renderer at chosen points in the loop, for review."""
import pathlib, sys
from playwright.sync_api import sync_playwright

root = pathlib.Path(__file__).resolve().parent.parent
page_url = (root / "render/mimesis_standalone.html").as_uri()
times = [float(t) for t in (sys.argv[1:] or [40, 200, 420, 640, 830])]
outdir = root / "out/shots"; outdir.mkdir(parents=True, exist_ok=True)

with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
                           args=["--force-device-scale-factor=1"])
    for t in times:
        pg = b.new_page(viewport={"width": 3840, "height": 2160})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append("console:" + m.text) if m.type == "error" else None)
        # freeze the clock: state must be a pure function of it
        pg.add_init_script(f"performance.now = () => {t * 1000};")
        pg.goto(page_url, wait_until="load")
        pg.wait_for_timeout(900)
        st = pg.eval_on_selector("#status", "e=>e.textContent")
        pg.screenshot(path=str(outdir / f"t{int(t):04d}.png"))
        print(f"t={t:7.1f}  {st.strip()}" + (f"   ERRORS: {errs[:2]}" if errs else ""))
        pg.close()
    b.close()
print("shots in", outdir)
