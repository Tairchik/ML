"""Build two releases, run/update/rollback without training, then Compose healthcheck."""
import argparse
import os
import subprocess
import time

import httpx

from .common import MODEL_NAME, ROOT, read_json, sha256, utc_now, write_json
from .train import setup_tracking
from .verify import check_client


def run(require=False, no_build=False):
    report_path = ROOT / "results/docker_checks.json"
    report = {"status": "running", "created_at": utc_now(), "commands": [], "sequence": []}
    write_json(report_path, report)

    def command(args, env=None, timeout=600):
        started = time.perf_counter()
        result = subprocess.run(args, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
        entry = {"argv": args, "returncode": result.returncode,
                 "seconds": time.perf_counter() - started, "stdout": result.stdout, "stderr": result.stderr}
        report["commands"].append(entry)
        write_json(report_path, report)
        if result.returncode:
            raise RuntimeError(f"Command failed: {args}\n{result.stderr or result.stdout}")
        return result.stdout.strip()

    try:
        report["docker_version"] = command(["docker", "version"], timeout=30)
        report["compose_version"] = command(["docker", "compose", "version"], timeout=30)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        report.update(status="blocked", reason=str(error),
                      instruction="Start Docker Desktop and enable Resources > WSL Integration > Ubuntu, then rerun this module.")
        write_json(report_path, report)
        print("Docker недоступен:", str(error), flush=True)
        if require:
            raise
        return report
    client, experiment_id = setup_tracking()
    summary = read_json(ROOT / "results/summary.json")
    before_runs = len(client.search_runs([experiment_id], max_results=50000))
    before_hashes = {r: sha256(ROOT / "models" / r / "pipeline.joblib") for r in ("v1", "v2")}
    created = False
    container = "lab4-sms-demo"
    try:
        # Refuse to delete any pre-existing container with the same name.
        occupied = command(["docker", "ps", "-aq", "--filter", f"name=^/{container}$"])
        if occupied:
            raise RuntimeError(f"Container {container} already exists. Choose another name or remove it yourself.")
        report["images"] = {}
        for release in ("v1", "v2"):
            if not no_build:
                command(["docker", "build", "--build-arg", f"MODEL_RELEASE={release}", "-t", f"sms-spam:{release}", "."])
                entry = report["commands"][-1]
                (ROOT / "results" / f"docker_build_{release}.txt").write_text(entry["stdout"] + entry["stderr"])
            import json
            image = json.loads(command(["docker", "image", "inspect", f"sms-spam:{release}"]))[0]
            report["images"][release] = {"tag": f"sms-spam:{release}", "id": image["Id"],
                                          "repo_digests": image.get("RepoDigests", []), "size_bytes": image["Size"]}
        for release in ("v1", "v2", "v1"):
            version = summary["releases"][release]["model_version"]
            client.set_registered_model_alias(MODEL_NAME, "champion", version)
            assert str(client.get_model_version_by_alias(MODEL_NAME, "champion").version) == str(version)
            command(["docker", "run", "--rm", "-d", "--name", container,
                     "-p", "127.0.0.1:18000:8000", f"sms-spam:{release}"])
            created = True
            with httpx.Client(base_url="http://127.0.0.1:18000", timeout=5, trust_env=False) as http:
                deadline = time.monotonic() + 90
                while True:
                    try:
                        health = http.get("/health")
                        if health.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError("Container did not become ready")
                    time.sleep(0.5)
                checked = check_client(http, release)
                checked["image_tag"] = f"sms-spam:{release}"
                checked["image_id"] = report["images"][release]["id"]
                checked["champion_alias"] = version
                report["sequence"].append(checked)
            command(["docker", "logs", container])
            command(["docker", "stop", container])
            created = False
        assert report["sequence"][0]["example_response"]["predictions"] == report["sequence"][2]["example_response"]["predictions"]
        # After the rollback demonstration restore the validation champion, then serve it via Compose.
        client.set_registered_model_alias(MODEL_NAME, "champion", summary["releases"]["v2"]["model_version"])
        environment = {**os.environ, "MODEL_RELEASE": "v2"}
        command(["docker", "compose", "up", "-d", "--no-build", "--wait", "--wait-timeout", "90"], env=environment)
        report["compose_ps"] = command(["docker", "compose", "ps", "--format", "json"], env=environment)
        cid = command(["docker", "compose", "ps", "-q", "api"], env=environment)
        report["compose_health"] = command(["docker", "inspect", "--format", "{{.State.Health.Status}}", cid])
        assert report["compose_health"] == "healthy"
        with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30, trust_env=False) as http:
            report["compose_checks"] = check_client(http, "v2")
        after_hashes = {r: sha256(ROOT / "models" / r / "pipeline.joblib") for r in ("v1", "v2")}
        after_runs = len(client.search_runs([experiment_id], max_results=50000))
        assert before_runs == after_runs and before_hashes == after_hashes
        report["no_retraining"] = {"runs_before": before_runs, "runs_after": after_runs,
                                    "hashes_before": before_hashes, "hashes_after": after_hashes}
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", reason=repr(error))
        raise
    finally:
        if created:
            command(["docker", "stop", container])
        client.set_registered_model_alias(MODEL_NAME, "champion", summary["releases"]["v2"]["model_version"])
        write_json(report_path, report)
    print("PASS: Docker v1 → v2 → v1, Compose healthy; no retraining", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require", action="store_true", help="Fail if Docker is unavailable")
    parser.add_argument("--no-build", action="store_true", help="Use already built v1/v2 images and verify them against exports")
    args = parser.parse_args()
    run(require=args.require, no_build=args.no_build)
