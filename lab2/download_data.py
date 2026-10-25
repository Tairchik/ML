"""Download the original UCI Rice files; subsequent experiments work offline."""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/545/rice+cammeo+and+osmancik.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "Python/UCI-lab2"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    files = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if name in {"Rice_Cammeo_Osmancik.arff", "Citation_Request.txt"}:
                content = archive.read(member)
                (DATA_DIR / name).write_bytes(content)
                files[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    if "Rice_Cammeo_Osmancik.arff" not in files:
        raise RuntimeError("The UCI archive does not contain the expected ARFF file.")
    metadata = {
        "name": "Rice (Cammeo and Osmancik)", "uci_id": 545,
        "dataset_url": DATASET_URL, "archive_url": ARCHIVE_URL,
        "doi": "10.24432/C5MW4Z", "license": "CC BY 4.0",
        "download_date": "2026-10-25", "files": files,
    }
    (DATA_DIR / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
