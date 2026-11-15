"""Render the captured console logs (reports/logs/console/*.log) as terminal screenshots.

    python reports/build/render_console.py        # needs Playwright + Google Chrome
"""

from __future__ import annotations

import html
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "reports" / "logs" / "console"
OUT = ROOT / "reports" / "screenshots"

SHOTS = [
    ("stage_01_ingestion.png", ["stage_ingest.log", "ingestion_manifest.log"], None),
    ("stage_02_validation.png", ["stage_validate.log"], None),
    ("stage_02b_error_dataset.png", ["errors_dataset.log"], None),
    ("stage_03_preparation.png", ["stage_prepare.log"], None),
    ("stage_04_transformation.png", ["stage_transform.log", "data_layout.log"], None),
    ("stage_05_features.png", ["stage_features.log"], None),
    ("stage_06_training.png", ["stage_train.log"], None),
    ("stage_07_evaluation.png", ["stage_evaluate.log"], None),
    ("stage_08_orchestration_cli.png", ["prefect_flow_run.log"], ("$", "Flow run", "Task run")),
    ("stage_09_prediction.png", ["predict.log"], None),
    ("stage_10_unit_tests.png", ["tests.log"], None),
]

CSS = """
body { margin: 0; background: #ffffff; }
.term { display: inline-block; background: #1e1e2e; color: #cdd6f4; border-radius: 10px; padding: 0 0 14px 0;
        font: 12.5px/1.45 Menlo, Monaco, monospace; min-width: 900px; box-shadow: 0 4px 18px rgba(0,0,0,.25); }
.bar { background: #313244; border-radius: 10px 10px 0 0; padding: 8px 12px; color: #a6adc8; font-size: 12px; }
.dot { display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 6px; }
pre { margin: 10px 16px 0 16px; white-space: pre; }
.cmd { color: #a6e3a1; font-weight: bold; } .err { color: #f38ba8; } .warn { color: #f9e2af; } .ok { color: #89dceb; }
"""


def colorize(line: str) -> str:
    esc = html.escape(line)
    if line.startswith("$"):
        return f'<span class="cmd">{esc}</span>'
    if "[ERROR" in line or "FAIL" in line:
        return f'<span class="err">{esc}</span>'
    if "[WARNING" in line:
        return f'<span class="warn">{esc}</span>'
    if any(k in line for k in ("PASS", "Completed", "finished", "passed", "registered")):
        return f'<span class="ok">{esc}</span>'
    return esc


def build_html(files: list[str], keep: tuple | None) -> str:
    lines = []
    for f in files:
        for line in (LOGS / f).read_text().splitlines():
            if "WARNING mlflow" in line or (keep and not line.startswith(keep) and not any(k in line for k in keep)):
                continue
            lines.append(line)
        lines.append("")
    body = "\n".join(colorize(x) for x in lines).rstrip()
    dots = "".join(f'<span class="dot" style="background:{c}"></span>' for c in ("#f38ba8", "#f9e2af", "#a6e3a1"))
    return f'<html><head><style>{CSS}</style></head><body><div class="term"><div class="bar">{dots} zsh — recommendation-pipeline</div><pre>{body}</pre></div></body></html>'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
        for name, files, keep in SHOTS:
            page.set_content(build_html(files, keep))
            page.locator(".term").screenshot(path=str(OUT / name))
            print("saved", name)
        browser.close()


if __name__ == "__main__":
    main()
