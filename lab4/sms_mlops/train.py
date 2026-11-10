"""Six validation experiments, Registry versions, immutable exports, final test."""
import argparse
import importlib.metadata
import os
import platform
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path(__file__).resolve().parents[1] / ".runtime/matplotlib"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .common import MODEL_NAME, ROOT, SEED, SOURCE, read_json, sha256, utc_now, write_json
from .data import prepare_data
from .relocate import relocate_store

CONFIGS = [("nb", "alpha", value) for value in (0.1, 0.5, 1.0)] + [
    ("lr", "C", value) for value in (0.5, 2.0, 10.0)]


def setup_tracking():
    # No external server is needed to train. The UI opens this same SQLite store.
    folder = ROOT / "mlflow-data"
    folder.mkdir(exist_ok=True)
    relocate_store()
    mlflow.set_tracking_uri(f"sqlite:///{folder / 'mlflow.db'}")
    mlflow.set_registry_uri(mlflow.get_tracking_uri())
    client = MlflowClient()
    exp = client.get_experiment_by_name("lab4-sms-spam")
    if exp is None:
        # Relative artifact root keeps the whole lab movable. Always run UI from lab4.
        experiment_id = client.create_experiment("lab4-sms-spam", artifact_location="./mlflow-data/artifacts")
    else:
        experiment_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return client, experiment_id


def metrics(y_true, predicted, prefix):
    return {f"{prefix}_f1_spam": f1_score(y_true, predicted, pos_label=1, zero_division=0),
            f"{prefix}_precision_spam": precision_score(y_true, predicted, pos_label=1, zero_division=0),
            f"{prefix}_recall_spam": recall_score(y_true, predicted, pos_label=1, zero_division=0),
            f"{prefix}_accuracy": accuracy_score(y_true, predicted)}


def export_version(client, version, release, dataset):
    # Resolve the exact numbered version: later alias changes cannot alter the export.
    mv = client.get_model_version(MODEL_NAME, str(version))
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    folder = ROOT / "models" / release
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "pipeline.joblib", compress=3)
    meta = {"model_name": MODEL_NAME, "model_version": str(mv.version), "release": release,
            "run_id": mv.run_id, "model_uri": f"models:/{MODEL_NAME}/{mv.version}",
            "pipeline_sha256": sha256(folder / "pipeline.joblib"),
            "raw_sha256": dataset["raw_sha256"], "split_sha256": dataset["split_sha256"],
            "classes": {"0": "ham", "1": "spam"}, "input": {"texts": "1..100 nonblank strings; each <=5000 characters"},
            "threshold": 0.5, "python": platform.python_version(),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "algorithm": type(model.named_steps["classifier"]).__name__, "exported_at": utc_now()}
    write_json(folder / "metadata.json", meta)
    return meta


def run(force=False):
    import os
    os.chdir(ROOT)
    frames, dataset = prepare_data()
    client, experiment_id = setup_tracking()
    summary_path = ROOT / "results/summary.json"
    if summary_path.exists() and not force:
        summary = read_json(summary_path)
        if summary["dataset"]["split_sha256"] != dataset["split_sha256"]:
            raise ValueError("Saved results use a different split")
        for item in summary["releases"].values():
            client.get_model_version(MODEL_NAME, item["model_version"])
            assert sha256(ROOT / "models" / item["release"] / "pipeline.joblib") == item["pipeline_sha256"]
        print("Используются сохранённые результаты. Для новых запусков: --force.")
        return summary
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "figures").mkdir(exist_ok=True)
    train, val = frames["train"], frames["validation"]
    records = []
    for order, (algorithm, parameter, value) in enumerate(CONFIGS):
        classifier = (MultinomialNB(alpha=value) if algorithm == "nb" else
                      LogisticRegression(C=value, max_iter=2000, solver="liblinear", random_state=SEED))
        model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                                                   sublinear_tf=True, min_df=1)),
                          ("classifier", classifier)])
        config_id = f"{algorithm}_{parameter}_{value:g}"
        with mlflow.start_run(run_name=config_id) as active:
            mlflow.log_params({"algorithm": type(classifier).__name__, parameter: value,
                               "seed": SEED, "train_rows": len(train), "validation_rows": len(val),
                               "tfidf_ngram_range": "1,2", "tfidf_max_features": 20000,
                               "tfidf_sublinear_tf": True, "tfidf_min_df": 1,
                               "tfidf_lowercase": True, "tfidf_norm": "l2", "threshold": 0.5})
            mlflow.set_tags({"variant": "2", "purpose": "lab4", "config_id": config_id,
                             "dataset_sha256": dataset["raw_sha256"],
                             "split_sha256": dataset["split_sha256"],
                             "selection_metric": "val_f1_spam", "positive_class": "spam"})
            started = time.perf_counter()
            model.fit(train.text.tolist(), train.target)
            fit_seconds = time.perf_counter() - started
            pred = model.predict(val.text.tolist())
            scores = metrics(val.target, pred, "val")
            scores["fit_seconds"] = fit_seconds
            scores["vocabulary_size"] = len(model.named_steps["tfidf"].vocabulary_)
            mlflow.log_metrics(scores)
            for name in ("train", "validation"):
                mlflow.log_input(mlflow.data.from_pandas(frames[name][["text", "target"]],
                                                       source=SOURCE, targets="target"), context=name)
            mlflow.log_artifact(str(ROOT / "data/splits.json"), artifact_path="data")
            mlflow.log_artifact(str(ROOT / "data/manifest.json"), artifact_path="data")
            for code in sorted((ROOT / "sms_mlops").glob("*.py")):
                mlflow.log_artifact(str(code), artifact_path="source/sms_mlops")
            for name in ("requirements.in", "requirements.txt", "requirements-runtime.txt"):
                if (ROOT / name).exists():
                    mlflow.log_artifact(str(ROOT / name), artifact_path="source")
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(val.target, pred, labels=[0, 1],
                                                    display_labels=["ham", "spam"], ax=ax, colorbar=False)
            ax.set_title(f"Validation: {config_id}")
            fig.tight_layout()
            mlflow.log_figure(fig, "validation_confusion_matrix.png")
            plt.close(fig)
            example = np.array(train.text.iloc[:2].tolist(), dtype=str)
            model_info = mlflow.sklearn.log_model(
                model, artifact_path="model", input_example=example,
                signature=infer_signature(example, model.predict(example)),
                pip_requirements=[line for line in (ROOT / "requirements-runtime.txt").read_text().splitlines()
                                  if line and not line.startswith("#")])
            record = {"config_id": config_id, "algorithm": type(classifier).__name__,
                      "parameter": parameter, "value": value, "run_id": active.info.run_id,
                      "model_uri": model_info.model_uri, "order": order, **scores}
            records.append(record)
            print(config_id, active.info.run_id, scores, flush=True)
    # Fixed rule: highest validation F1, then precision, then first config in declared order.
    ranked = sorted(records, key=lambda x: (-x["val_f1_spam"], -x["val_precision_spam"], x["order"]))
    champion = ranked[0]
    alternative = next(row for row in ranked if row["algorithm"] != champion["algorithm"])
    pd.DataFrame(records).to_csv(results / "experiments.csv", index=False)
    releases = {}
    for release, selected in (("v1", alternative), ("v2", champion)):
        mv = mlflow.register_model(selected["model_uri"], MODEL_NAME)
        client.set_model_version_tag(MODEL_NAME, mv.version, "release", release)
        client.set_model_version_tag(MODEL_NAME, mv.version, "config_id", selected["config_id"])
        client.update_model_version(MODEL_NAME, mv.version,
                                    description=f"{release}: {selected['config_id']}; selected using validation only")
        if release == "v2":
            client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
            resolved = client.get_model_version_by_alias(MODEL_NAME, "champion")
            assert resolved.version == mv.version
            version = resolved.version
        else:
            version = mv.version
        releases[release] = export_version(client, version, release, dataset)
    # Final test evaluates ONLY the selected frozen model; no tuning after this point.
    model = joblib.load(ROOT / "models/v2/pipeline.joblib")
    test = frames["test"]
    pred = model.predict(test.text.tolist())
    final = {**metrics(test.target, pred, "test"),
             "confusion_matrix": confusion_matrix(test.target, pred, labels=[0, 1]).tolist(),
             "classification_report": classification_report(test.target, pred, target_names=["ham", "spam"],
                                                              output_dict=True, zero_division=0),
             "rows": len(test), "run_id": champion["run_id"],
             "model_version": releases["v2"]["model_version"]}
    with mlflow.start_run(run_id=champion["run_id"]):
        mlflow.log_metrics({key: value for key, value in final.items() if key.startswith("test_")})
        mlflow.log_dict(final, "final_test_metrics.json")
    write_json(results / "test_metrics.json", final)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(test.target, pred, display_labels=["ham", "spam"], ax=ax, colorbar=False)
    ax.set_title(f"Final test: {champion['config_id']}")
    fig.tight_layout()
    fig.savefig(results / "figures/test_confusion_matrix.png", dpi=150)
    plt.close(fig)
    df = pd.DataFrame(records).set_index("config_id")
    ax = df[["val_f1_spam", "val_precision_spam", "val_recall_spam"]].plot.bar(figsize=(11, 5), ylim=(0, 1.05))
    ax.set_ylabel("Metric for spam")
    ax.set_xlabel("Experiment")
    ax.figure.tight_layout()
    ax.figure.savefig(results / "figures/experiments.png", dpi=150)
    plt.close(ax.figure)
    # Include both labels plus a validation SMS on which releases disagree, if present.
    samples = [val[val.label == label].text.iloc[0] for label in ("ham", "spam")]
    old_model = joblib.load(ROOT / "models/v1/pipeline.joblib")
    different = np.flatnonzero(old_model.predict(val.text.tolist()) != model.predict(val.text.tolist()))
    if len(different):
        samples.append(val.text.iloc[int(different[0])])
    write_json(ROOT / "examples/valid.json", {"texts": samples})
    for release in releases:
        exported = joblib.load(ROOT / "models" / release / "pipeline.joblib")
        write_json(ROOT / "examples" / f"expected_{release}.json",
                   {"predictions": ["spam" if p == 1 else "ham" for p in exported.predict(samples)],
                    "probabilities_spam": exported.predict_proba(samples)[:, 1].tolist(),
                    "model_version": releases[release]["model_version"]})
    summary = {"created_at": utc_now(), "experiment_id": experiment_id, "dataset": dataset,
               "selection_rule": "max val_f1_spam; ties: max val_precision_spam, then config order",
               "champion": champion, "alternative": alternative, "releases": releases, "test": final,
               "environment": {"python": sys.version, "platform": platform.platform(),
                               "packages": {p: importlib.metadata.version(p) for p in
                                            ("mlflow", "scikit-learn", "numpy", "pandas", "fastapi", "pydantic")}}}
    write_json(summary_path, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Create six NEW runs and two NEW Registry versions")
    args = parser.parse_args()
    result = run(force=args.force)
    print("Champion:", result["champion"]["config_id"], "Registry", result["releases"]["v2"]["model_version"])
