"""Paths and serialization shared by training and verification."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sms-spam-filter"
SEED = 42
SOURCE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()
