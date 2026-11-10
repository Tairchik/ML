"""Download once; retain raw bytes, deduplication decisions and split indices."""
import hashlib
import io
import re
import unicodedata
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from .common import DATA_URL, ROOT, SEED, SOURCE, read_json, sha256, write_json


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def prepare_data():
    raw = ROOT / "data/raw/SMSSpamCollection"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        # No implicit download when the committed source file is already present.
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            archive_bytes = response.read()
        (raw.parent / "sms_spam_collection.zip").write_bytes(archive_bytes)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for name in ("SMSSpamCollection", "readme"):
                (raw.parent / name).write_bytes(archive.read(name))
    # split on the first TAB only: SMS may contain quotes and additional TABs.
    rows = [line.split("\t", 1) for line in raw.read_text(encoding="utf-8").splitlines() if line]
    if any(len(row) != 2 for row in rows):
        raise ValueError("Invalid source TSV row")
    df = pd.DataFrame(rows, columns=["label", "text"])
    df.index.name = "row_id"
    assert df.label.isin(["ham", "spam"]).all()
    assert df.text.str.strip().ne("").all()
    df["text_hash"] = df.text.map(lambda text: hashlib.sha256(normalized(text).encode()).hexdigest())
    conflicts = df.groupby("text_hash").label.nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    clean = df[~df.text_hash.isin(conflicts)].drop_duplicates("text_hash").copy()
    clean["target"] = clean.label.map({"ham": 0, "spam": 1})
    split_path = ROOT / "data/splits.json"
    expected_hash = sha256(raw)
    if split_path.exists():
        split = read_json(split_path)
        if split["raw_sha256"] != expected_hash or split["seed"] != SEED:
            raise ValueError("Source/seed changed; saved split must be reviewed explicitly")
    else:
        train_ids, rest = train_test_split(
            clean.index.to_numpy(), test_size=0.4, random_state=SEED, stratify=clean.target
        )
        val_ids, test_ids = train_test_split(
            rest, test_size=0.5, random_state=SEED, stratify=clean.loc[rest, "target"]
        )
        split = {"seed": SEED, "raw_sha256": expected_hash,
                 "method": "normalized-text deduplication, stratified 60/20/20",
                 "train": train_ids.tolist(), "validation": val_ids.tolist(), "test": test_ids.tolist()}
        write_json(split_path, split)
    sets = [set(split[name]) for name in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(clean.index)
    frames = {name: clean.loc[split[name]].copy() for name in ("train", "validation", "test")}
    hashes = [set(frame.text_hash) for frame in frames.values()]
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
    clean.to_csv(ROOT / "data/clean.csv", index=True)
    # This file explains every omitted source row, including conflicting labels.
    excluded = df.loc[~df.index.isin(clean.index)].copy()
    excluded["reason"] = excluded.text_hash.map(
        lambda value: "conflicting labels" if value in conflicts else "duplicate normalized text"
    )
    excluded.to_csv(ROOT / "data/excluded.csv", index=True)
    summary = {"source": SOURCE, "download_url": DATA_URL, "license": "CC BY 4.0",
               "raw_sha256": expected_hash, "raw_rows": len(df), "clean_rows": len(clean),
               "excluded_rows": len(excluded), "conflicting_hashes": len(conflicts),
               "raw_classes": df.label.value_counts().to_dict(),
               "clean_classes": clean.label.value_counts().to_dict(),
               "split_sha256": sha256(split_path),
               "splits": {name: {"rows": len(frame), **frame.label.value_counts().to_dict()}
                          for name, frame in frames.items()}}
    write_json(ROOT / "data/manifest.json", summary)
    return frames, summary


if __name__ == "__main__":
    _, summary = prepare_data()
    print(summary)
