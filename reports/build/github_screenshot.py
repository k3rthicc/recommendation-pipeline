"""Screenshot of the public GitHub repository page (reports/screenshots/github_repo.png)."""

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[2] / "reports" / "screenshots"

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1300}, device_scale_factor=1.5)
    page.goto("https://github.com/k3rthicc/recommendation-pipeline")
    page.get_by_role("heading", name="End-to-End Data Management Pipeline for a Recommendation System").first.wait_for(timeout=30000)
    time.sleep(3)
    page.screenshot(path=str(OUT / "github_repo.png"))
    browser.close()
print("saved github_repo.png")
