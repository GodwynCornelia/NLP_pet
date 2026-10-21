.PHONY: bootstrap test integration lint api ui docker-up
bootstrap:
	uv sync --frozen
	uv run --frozen python -m support_router bootstrap
test:
	uv run --frozen pytest -q
integration:
	uv run --frozen pytest -q -m integration
lint:
	uv run --frozen ruff check src tests scripts
	uv run --frozen ruff format --check src tests scripts
api:
	uv run --frozen uvicorn support_router.api.main:app --host 127.0.0.1 --port 8000 --workers 1 --log-config configs/logging.json
ui:
	uv run --frozen streamlit run src/support_router/ui/app.py --server.address=127.0.0.1 --server.maxUploadSize=5 --browser.gatherUsageStats=false
docker-up:
	docker compose up -d
