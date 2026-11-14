"""从 quantum-mark.svg 渲染三版 512x512 PNG（仅标准库 + 本机 Chrome/Edge headless）。

用法：python render_assets.py

背景：本机无 cairosvg / ImageMagick，用浏览器渲染保真度最高。
沙箱下 headless Chrome 只能写工作区内的路径，所以 HTML 与 user-data-dir
都放在同目录的 _render/ 下，渲完即删。
"""
from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SVG_FILE = ROOT / "quantum-mark.svg"
WORK = ROOT / "_render"

VARIANTS = [
    # (输出文件名, CSS 背景, CSS 前景色, 是否透明背景)
    ("quantum-mark.png", "transparent", "#1f2328", True),
    ("quantum-avatar.png", "#ffffff", "#1f2328", False),
    ("quantum-avatar-dark.png", "#0d1117", "#f0f6fc", False),
]

BROWSERS = [
    r"C:/Program Files/Google/Chrome/Application/chrome.exe",
    r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    r"C:/Program Files/Microsoft/Edge/Application/msedge.exe",
]


def find_browser() -> str:
    for p in BROWSERS:
        if Path(p).exists():
            return p
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("找不到 Chrome / Edge，无法渲染 SVG")


def main() -> None:
    svg = SVG_FILE.read_text(encoding="utf-8")
    exe = find_browser()
    WORK.mkdir(exist_ok=True)
    try:
        for name, bg, fg, transparent in VARIANTS:
            html = (
                "<!doctype html><html><head><meta charset='utf-8'><style>"
                "html,body{margin:0;width:100%;height:100%;}"
                f"body{{background:{bg};color:{fg};display:flex;"
                "align-items:center;justify-content:center;}"
                "svg{width:90vmin;height:90vmin;display:block;}"
                f"</style></head><body>{svg}</body></html>"
            )
            hp = WORK / f"{name}.html"
            hp.write_text(html, encoding="utf-8")
            out = ROOT / name
            cmd = [exe, "--headless=new", "--disable-gpu", "--no-sandbox",
                   "--hide-scrollbars", "--no-first-run", "--disable-extensions",
                   "--force-device-scale-factor=1",
                   f"--user-data-dir={WORK / ('ud-' + name)}",
                   "--window-size=512,512"]
            if transparent:
                cmd.append("--default-background-color=00000000")
            cmd += [f"--screenshot={out}", hp.as_uri()]
            proc = subprocess.run(
                cmd,
                check=False,
                cwd=WORK,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()[:300]
                msg = (
                    f"Headless render failed for {name} "
                    f"(exit {proc.returncode}): {stderr}"
                )
                raise RuntimeError(msg)
            if not out.exists():
                stderr = (proc.stderr or "").strip()[:300]
                raise RuntimeError(f"Rendered output missing for {name}: {stderr}")
            data = out.read_bytes()
            w, h = struct.unpack(">II", data[16:24])
            print(f"{name}: {w}x{h}, {len(data)} bytes")
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
