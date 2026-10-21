import logging
import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from support_router.api.schemas import (
    BatchRequest,
    BatchResponseItem,
    PredictRequest,
    PredictResponse,
)
from support_router.common import read_json, setup_compute, sha
from support_router.config import load_config

logger = logging.getLogger("support_router.api")


class BodyTooLarge(Exception):
    pass


class RequestLimits:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        size, status = 0, 500
        started = time.perf_counter()

        async def tracked_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"].append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            content_length = dict(scope["headers"]).get(b"content-length", b"0")
            try:
                too_large = int(content_length) > self.limit
            except ValueError:
                too_large = True
            if too_large:
                raise BodyTooLarge()
            # Buffer at most the configured limit before JSON parsing. This also handles
            # chunked requests, whose parse errors FastAPI would otherwise turn into 400.
            messages = []
            while True:
                message = await receive()
                size += len(message.get("body", b""))
                if size > self.limit:
                    raise BodyTooLarge()
                messages.append(message)
                if not message.get("more_body", False):
                    break
            position = 0

            async def replay():
                nonlocal position
                if position < len(messages):
                    message = messages[position]
                    position += 1
                    return message
                return await receive()

            await self.app(scope, replay, tracked_send)
        except BodyTooLarge:
            response = JSONResponse(
                status_code=413, content={"error": "body_too_large", "request_id": request_id}
            )
            await response(scope, receive, tracked_send)
        finally:
            logger.info(
                "request_id=%s status=%s duration_ms=%.2f input_bytes=%s",
                request_id,
                status,
                (time.perf_counter() - started) * 1000,
                size,
            )


def create_app(config=None, router_factory=None):
    config = config or load_config()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.router = None
        app.state.reason = "artifacts_missing"
        app.state.controller = setup_compute(config)
        try:
            if router_factory:
                app.state.router = router_factory()
            else:
                from support_router.runtime import Router

                app.state.router = Router(config, os.getenv("RUN_ID"))
        except FileNotFoundError:
            logger.warning("Artifact files missing; API alive but not ready")
        except Exception as exc:
            app.state.reason = "artifact_validation_failed"
            # No input text or absolute paths are exposed in HTTP responses.
            logger.error("Artifact validation failed: %s", type(exc).__name__)
        yield

    app = FastAPI(title="Support Router", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestLimits, limit=config["limits"]["max_body_bytes"])

    @app.exception_handler(BodyTooLarge)
    async def large_body(request, exc):
        return JSONResponse(
            status_code=413,
            content={"error": "body_too_large", "request_id": request.state.request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        # FastAPI's default validation detail contains the submitted raw text.
        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_request",
                "request_id": request.state.request_id,
                "details": [{"location": list(e["loc"]), "type": e["type"]} for e in exc.errors()],
            },
        )

    @app.exception_handler(Exception)
    async def unexpected(request, exc):
        logger.error(
            "request_id=%s unexpected=%s",
            getattr(request.state, "request_id", "unknown"),
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    def get_router():
        if app.state.router is None:
            raise HTTPException(503, detail={"code": app.state.reason})
        return app.state.router

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        router = get_router()
        return {"status": "ready", "run_id": router.run_id}

    @app.get("/metadata")
    def metadata():
        return get_router().metadata()

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request):
        result = get_router().predict(body.text, body.top_k, body.include_similar)
        result["request_id"] = request.state.request_id
        return result

    @app.post("/predict-batch", response_model=list[BatchResponseItem])
    def batch(body: BatchRequest):
        router = get_router()
        rows, valid, positions = [], [], []
        for item in body.items:
            rows.append({"id": item.id, "result": None, "error": None})
            try:
                parsed = PredictRequest(
                    text=item.text, top_k=body.top_k, include_similar=body.include_similar
                )
                valid.append(parsed.text)
                positions.append(len(rows) - 1)
            except ValidationError:
                rows[-1]["error"] = {
                    "code": "invalid_text",
                    "message": "Expected text with 1..4000 characters",
                }
        if valid:
            results = router.predict_many(valid, body.top_k, body.include_similar)
            for index, result in zip(positions, results, strict=True):
                rows[index]["result"] = result
        return rows

    @app.get("/reports")
    def reports():
        router = get_router()
        path = router.report
        if not (path / "test_metrics.json").exists():
            raise HTTPException(503, detail={"code": "evaluation_not_available"})
        import pandas as pd

        result = read_json(path / "test_metrics.json")
        if result["run_id"] != router.run_id or result["manifest_sha256"] != sha(
            router.artifact / "manifest.json"
        ):
            raise HTTPException(503, detail={"code": "report_run_mismatch"})
        return {
            "run_id": router.run_id,
            "test": result,
            "validation": [{"id": r["id"], **r["metrics"]} for r in router.selection["candidates"]],
            "policy": router.policy,
            "latency": read_json(path / "latency.json"),
            "confusions": pd.read_csv(path / "top_confusions.csv").head(20).to_dict("records"),
            "errors": pd.read_csv(path / "error_review_sample.csv", keep_default_na=False).to_dict(
                "records"
            ),
        }

    return app


app = create_app()
