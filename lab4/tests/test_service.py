"""Startup integrity and export/API contract regressions."""
import json
import shutil

import pytest
from fastapi.testclient import TestClient

from sms_mlops.common import ROOT
from sms_mlops.service import create_app
from sms_mlops.verify import check_client


@pytest.mark.parametrize("release", ["v1", "v2"])
def test_export_and_api_contract(release):
    with TestClient(create_app(ROOT / "models" / release)) as client:
        report = check_client(client, release)
    assert report["status"] == "passed"


def test_corrupt_export_prevents_startup(tmp_path):
    for name in ("pipeline.joblib", "metadata.json"):
        shutil.copy(ROOT / "models/v2" / name, tmp_path / name)
    with (tmp_path / "pipeline.joblib").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        with TestClient(create_app(tmp_path)):
            pass
