FROM ghcr.io/astral-sh/uv:0.13.4 AS uv
FROM python:3.11-slim
COPY --from=uv /uv /uvx /bin/
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock .python-version ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --no-cache
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "support_router.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-config", "configs/logging.json"]
