"""Rebase MLflow local artifact paths when the whole lab is moved to another folder."""
import sqlite3
from pathlib import Path

from .common import ROOT


def relocate_store(root=ROOT):
    root = Path(root).resolve()
    database = root / "mlflow-data/mlflow.db"
    if not database.exists():
        return 0
    new_artifacts = str(root / "mlflow-data/artifacts")
    # SQL identifiers are fixed, values are always bound parameters.
    fields = [("experiments", "artifact_location"), ("runs", "artifact_uri"),
              ("model_versions", "source"), ("model_versions", "storage_location")]
    changes = []
    with sqlite3.connect(database) as connection:
        for table, column in fields:
            exists = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if not exists:
                continue
            for (old,) in connection.execute(f"SELECT DISTINCT {column} FROM {table}"):
                if not old or "/mlflow-data/artifacts" not in old:
                    continue
                old_base, tail = old.split("/mlflow-data/artifacts", 1)
                prefix = "file://" if old_base.startswith("file://") else ""
                new = prefix + new_artifacts + tail
                if old != new:
                    changes.append((table, column, old, new))
        if changes:
            backup_path = database.with_name("mlflow.pre-relocation.db.bak")
            with sqlite3.connect(backup_path) as backup:
                connection.backup(backup)
            for table, column, old, new in changes:
                connection.execute(f"UPDATE {table} SET {column}=? WHERE {column}=?", (new, old))
    return len(changes)


if __name__ == "__main__":
    print("Rebased artifact paths:", relocate_store())
