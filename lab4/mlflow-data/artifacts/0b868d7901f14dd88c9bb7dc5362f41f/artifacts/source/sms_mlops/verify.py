"""The same API checks run against TestClient and a real Docker HTTP endpoint."""
import argparse
import time

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from .common import ROOT, read_json, sha256, utc_now, write_json
from .service import create_app

INVALID = [
    ("missing_texts", {}), ("empty_batch", {"texts": []}),
    ("empty_sms", {"texts": [""]}), ("blank_sms", {"texts": ["  \t\n"]}),
    ("number", {"texts": [42]}), ("boolean", {"texts": [True]}),
    ("null_sms", {"texts": [None]}), ("mixed_types", {"texts": ["Hello", 1]}),
    ("string_instead_of_batch", {"texts": "Hello"}),
    ("null_batch", {"texts": None}), ("wrong_top_level", ["Hello"]),
    ("extra_field", {"texts": ["Hello"], "unexpected": True}),
    ("oversized_batch", {"texts": ["Hello"] * 101}),
    ("oversized_sms", {"texts": ["x" * 5001]}),
]


def check_client(client, release):
    meta = read_json(ROOT / "models" / release / "metadata.json")
    model = joblib.load(ROOT / "models" / release / "pipeline.joblib")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": meta["model_version"], "release": release}
    api_meta = client.get("/metadata").json()
    assert api_meta["pipeline_sha256"] == meta["pipeline_sha256"]
    assert api_meta["run_id"] == meta["run_id"]
    assert client.get("/openapi.json").status_code == 200
    request = read_json(ROOT / "examples/valid.json")
    response = client.post("/predict", json=request)
    assert response.status_code == 200, response.text
    example = response.json()
    tests = []
    for name, payload in INVALID:
        bad = client.post("/predict", json=payload)
        assert bad.status_code == 422, (name, bad.status_code, bad.text)
        tests.append({"case": name, "expected_status": 422, "actual_status": bad.status_code})
    bad = client.post("/predict", content="{broken", headers={"Content-Type": "application/json"})
    assert bad.status_code == 422
    tests.append({"case": "malformed_json", "expected_status": 422, "actual_status": bad.status_code})
    # All held-out SMS, not just one hand-picked request; no refitting happens here.
    data = pd.read_csv(ROOT / "data/clean.csv", index_col="row_id", keep_default_na=False)
    ids = read_json(ROOT / "data/splits.json")["test"]
    texts = data.loc[ids, "text"].tolist()
    texts.extend(["Привет, как дела?", "🙂", "!@#$%", "Hello", "x" * 5000])
    # Check exact batch-size boundary as well as one-object requests.
    maximum_difference = 0.0
    for offset in range(0, len(texts), 100):
        batch = texts[offset:offset + 100]
        answer = client.post("/predict", json={"texts": batch})
        assert answer.status_code == 200, answer.text
        body = answer.json()
        expected = ["spam" if p == 1 else "ham" for p in model.predict(batch)]
        assert body["predictions"] == expected
        probabilities = model.predict_proba(batch)[:, 1]
        np.testing.assert_allclose(body["probabilities_spam"], probabilities, rtol=1e-10, atol=1e-12)
        maximum_difference = max(maximum_difference,
                                 float(np.max(np.abs(np.asarray(body["probabilities_spam"]) - probabilities))))
        assert body["model_version"] == meta["model_version"] and body["release"] == release
        assert body["run_id"] == meta["run_id"]
    single = client.post("/predict", json={"texts": ["See you at six!"]})
    assert single.status_code == 200 and len(single.json()["predictions"]) == 1
    wall_ms, prediction_ms = [], []
    for _ in range(10):
        started = time.perf_counter()
        response = client.post("/predict", json=request)
        wall_ms.append((time.perf_counter() - started) * 1000)
        assert response.status_code == 200
        prediction_ms.append(response.json()["latency_ms"])
    return {"release": release, "model_version": meta["model_version"], "pipeline_sha256": meta["pipeline_sha256"],
            "status": "passed", "parity_rows": len(texts), "max_probability_difference": maximum_difference,
            "invalid_requests": tests, "example_request": request, "example_response": example,
            "latency": {"requests": 10, "batch_size": len(request["texts"]),
                        "roundtrip_median_ms": float(np.median(wall_ms)),
                        "roundtrip_p95_ms": float(np.percentile(wall_ms, 95)),
                        "handler_median_ms": float(np.median(prediction_ms))}}


def run_local():
    reports = []
    # Starts a fresh app per release, exercising the startup loader on every switch.
    for release in ("v1", "v2", "v1"):
        with TestClient(create_app(ROOT / "models" / release)) as client:
            reports.append(check_client(client, release))
    assert reports[0]["example_response"]["predictions"] == reports[2]["example_response"]["predictions"]
    result = {"status": "passed", "transport": "FastAPI TestClient, in-process",
              "created_at": utc_now(), "sequence": reports,
              "note": "Local checks exercise API and exports. Docker results are recorded separately."}
    write_json(ROOT / "results/local_checks.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Real running HTTP endpoint, e.g. http://127.0.0.1:8000")
    parser.add_argument("--release", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()
    if args.url:
        import httpx
        with httpx.Client(base_url=args.url, timeout=30, trust_env=False) as client:
            report = check_client(client, args.release)
        write_json(ROOT / "results/http_checks.json", report)
        print(report)
    else:
        result = run_local()
        print("PASS: API, invalid requests and export parity; lifecycle v1 → v2 → v1")
        for item in result["sequence"]:
            print(item["release"], item["model_version"], item["parity_rows"], item["latency"])
