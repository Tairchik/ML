"""FastAPI inference; loading never imports MLflow or contacts the Registry."""
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Text = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=5000)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    texts: list[Text] = Field(min_length=1, max_length=100)

    @field_validator("texts")
    @classmethod
    def reject_blank(cls, texts):
        if any(not text.strip() for text in texts):
            raise ValueError("SMS must contain at least one non-whitespace character")
        return texts


class PredictResponse(BaseModel):
    predictions: list[str]
    probabilities_spam: list[float]
    model_version: str
    release: str
    run_id: str
    latency_ms: float


def create_app(model_dir=None):
    folder = Path(model_dir or os.environ.get("MODEL_DIR", "models/v2")).resolve()

    @asynccontextmanager
    async def lifespan(app):
        metadata = json.loads((folder / "metadata.json").read_text())
        pipeline_path = folder / "pipeline.joblib"
        digest = hashlib.sha256(pipeline_path.read_bytes()).hexdigest()
        if digest != metadata["pipeline_sha256"]:
            raise RuntimeError("Exported model SHA-256 mismatch")
        pipeline = joblib.load(pipeline_path)
        if pipeline.classes_.tolist() != [0, 1]:
            raise RuntimeError("Expected ordered classes [ham=0, spam=1]")
        app.state.pipeline = pipeline
        app.state.metadata = metadata
        yield

    app = FastAPI(title="SMS spam filter — вариант 2", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": app.state.metadata["model_version"],
                "release": app.state.metadata["release"]}

    @app.get("/metadata")
    def metadata():
        return app.state.metadata

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest):
        started = time.perf_counter()
        probabilities = app.state.pipeline.predict_proba(body.texts)[:, 1]
        # Same threshold and tie rule as sklearn's predict (class 0 wins a tie).
        predicted = ["spam" if probability > 0.5 else "ham" for probability in probabilities]
        meta = app.state.metadata
        return {"predictions": predicted, "probabilities_spam": probabilities.tolist(),
                "model_version": meta["model_version"], "release": meta["release"],
                "run_id": meta["run_id"], "latency_ms": (time.perf_counter() - started) * 1000}

    return app


app = create_app()
