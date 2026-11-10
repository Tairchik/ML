"""Optional: genuine browser captures. Requires Playwright + Chromium (not inference deps)."""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"


def main():
    OUT.mkdir(exist_ok=True)
    metadata_path = OUT / 'capture_metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    captures = metadata.setdefault('files', {})

    def save_capture(page, filename):
        # Keep the user's supplied originals when automatic captures are refreshed.
        if captures.get(filename, {}).get('source') == 'user':
            print(f'Preserved user screenshot: {filename}', flush=True)
            return
        path = OUT / filename
        page.screenshot(path=str(path), full_page=True)
        captures[filename] = {'source': 'playwright',
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                              'captured_at_utc': datetime.now(timezone.utc).isoformat()}
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')

    summary = json.loads((ROOT / "results/summary.json").read_text())
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1600, "height": 1050}, device_scale_factor=1)
        page.goto(f"http://127.0.0.1:5000/#/experiments/{summary['experiment_id']}")
        page.wait_for_timeout(5000)
        for column in ("Dataset", "Source", "Models", "val_f1_spam", "val_precision_spam", "val_recall_spam", "alpha", "C"):
            if not page.get_by_placeholder("Search columns").is_visible():
                page.get_by_role("button", name="Columns", exact=True).click()
            search = page.get_by_placeholder("Search columns")
            search.fill(column)
            page.get_by_text(column, exact=True).last.click()
        page.mouse.click(1100, 100)
        page.wait_for_timeout(500)
        save_capture(page, '01_mlflow_experiments.png')
        print("EXPERIMENT PAGE:", page.locator("body").inner_text()[:10000], flush=True)
        page.goto("http://127.0.0.1:5000/#/models/sms-spam-filter")
        page.wait_for_timeout(3000)
        save_capture(page, '02_registry.png')
        print("REGISTRY PAGE:", page.locator("body").inner_text()[:6000], flush=True)
        page.goto("http://127.0.0.1:8000/docs")
        page.get_by_text("/predict", exact=True).click()
        page.get_by_role("button", name="Try it out").click()
        editor = page.locator("textarea")
        editor.fill((ROOT / "examples/valid.json").read_text())
        page.get_by_role("button", name="Execute", exact=True).click()
        page.wait_for_timeout(2000)
        save_capture(page, '03_swagger_valid.png')
        editor.fill('{"texts": []}')
        page.get_by_role("button", name="Execute", exact=True).click()
        page.wait_for_timeout(1000)
        save_capture(page, '04_swagger_invalid.png')
        page.goto("http://127.0.0.1:8000/health")
        save_capture(page, '05_health.png')
        browser.close()
    metadata.update(captured_at_utc=datetime.now(timezone.utc).isoformat(),
                    method='Original user screenshots and Playwright captures of local MLflow/FastAPI',
                    model_version=summary['releases']['v2']['model_version'])
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')


if __name__ == "__main__":
    main()
