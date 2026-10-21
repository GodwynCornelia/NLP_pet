# Запуск и диагностика

## Подготовка

Python 3.11 и uv; зависимости устанавливаются `uv sync --frozen`. В PowerShell
сначала задайте `$env:PYTHONUTF8 = "1"`, особенно если путь содержит кириллицу.
Если uv не установлен: `python -m pip install --user uv`; вместо команды `uv`
можно использовать `python -m uv`. GNU Make необязателен.

Полный цикл на CPU:

```bash
uv run --frozen python -m support_router bootstrap
```

Он выполняет download → prepare → train → tune-policy → build-index → evaluate →
verify-artifacts. При повторном запуске используются сохранённые splits, embeddings,
кандидаты и selection. Не запускайте меняющие один run команды одновременно.
Чтобы повторить опубликованный протокол, используйте новую копию проекта с пустыми
`data/`, `artifacts/`, `reports/` и кешами проекта; переносить веса вручную не требуется.
Исторические публичные отчёты находятся отдельно в `reports/public/`.

Этапы доступны по отдельности:

```bash
uv run --frozen python -m support_router download
uv run --frozen python -m support_router prepare
uv run --frozen python -m support_router train
uv run --frozen python -m support_router tune-policy
uv run --frozen python -m support_router build-index
uv run --frozen python -m support_router evaluate
uv run --frozen python -m support_router verify-artifacts
```

Общие опции CLI: `--config <toml>` и `--run-id <id>`. После prepare команды используют
локальный `artifacts/current.json`, если run не указан. `prepare --run-id <new-id>`
создаёт новый цикл. `download --refresh` разрешает новые версии ресурсов и предназначен
для отдельного эксперимента; для закреплённого воспроизведения его не используйте.

Пути в TOML разрешаются относительно корня проекта. `ARTIFACT_DIR`, `REPORTS_DIR`,
`SUPPORT_ROUTER_CONFIG`, `API_BASE_URL`, `LOG_LEVEL` можно задать через окружение.
`.env.example` — шаблон; приложение не загружает `.env` автоматически. Offline-флаги
включайте после загрузки ресурсов. Датасеты, веса и собственные CSV в Git не добавляются.

## API и интерфейс

Команды двух терминалов приведены в [README](../README.md#быстрый-запуск).
API загружает один classifier и один encoder. UI обращается к API по HTTP.
`Ctrl+C` останавливает процесс и сохраняет артефакты на диске.

- `/health` — живой процесс; `/ready` — модель, policy и индекс готовы.
- `/metadata` — run, модель и 77 категорий; `/reports` — отчёты того же run.
- `/predict` — текст, top_k=1..5; include_similar по умолчанию true.
- `/predict-batch` — 1..100 items с уникальными техническими id; ошибочная строка
  сохраняется с error, include_similar по умолчанию false.

UI принимает UTF-8 CSV с колонкой `text`, необязательным `id` и дополнительными
колонками. До 5 MiB и 1000 строк. Исходные id могут повторяться: для API создаётся
уникальный row_id. `NA` читается как строка. Порядок и ошибки строк сохраняются.
Зарезервированные имена выходных колонок заменяются результатами. Формулоподобные
текстовые ячейки при экспорте получают ведущий апостроф; вход модели не меняется.
CSV хранится в памяти. Запросы ограничены 2 MiB, текст — 1..4000 символов после trim.

## Docker

В отдельной копии проекта с Docker Linux Engine:

```bash
docker compose --profile tools config --quiet
docker compose build api
docker compose --profile tools run --rm prepare
docker compose up -d --wait api ui
docker compose exec ui python -c "import urllib.request; print(urllib.request.urlopen('http://api:8000/ready').read().decode())"
```

Все три сервиса используют один образ; UI не требует отдельной сборки. Порты
публикуются на 127.0.0.1:8000 и 127.0.0.1:8501, UI использует `http://api:8000`.
prepare обучает; runtime API/UI не скачивают веса и не обучают. Данные API монтируются
read-only. Изменение портов только в host mapping не меняет внутренний URL API.
`docker compose down` останавливает стенд, bind-каталоги с результатами остаются.

## Проверки и типовые проблемы

```bash
uv run --frozen pytest -q
uv run --frozen ruff check src tests scripts
uv run --frozen ruff format --check src tests scripts
uv run --frozen pytest -q -m integration
```

Integration требует реальных локальных артефактов после bootstrap. Он блокирует
внешние socket-соединения, сохраняя loopback. Для запущенных API/UI доступны
`python scripts/smoke_http.py` и `python scripts/smoke_ui.py` через `uv run --frozen`.
Эти smoke обновляют локальные отчёты текущего run; публичные файлы не меняются.
Внутри Compose: `docker compose exec -T ui python scripts/container_smoke.py`.

Для проверки изолированного локального инференса в собранном образе:

```bash
docker compose exec -T api python scripts/offline_local_smoke.py
```

Этот скрипт запрещает все socket connects; отдельный запуск с Docker `--network none`
также выполнен в сохранённой [проверке воспроизводимости](reproducibility.md).

- **API не готов:** проверьте `/ready`, наличие завершённого bootstrap и run pointer.
  Без артефактов health=200, ready/predict=503. Неверный ввод=422, большое тело=413.
- **Несовместимые версии или SHA-256:** не отключайте проверку; используйте lock и
  артефакты собственного bootstrap. Joblib загружается только из доверенного pipeline.
- **Encoder недоступен:** для первой загрузки нужны GitHub, Hugging Face/CDN и PyTorch
  CPU index. Токен не требуется; проверяйте сеть и место под загрузку/распаковку слоёв.
- **Длинный текст:** MiniLM использует максимум 256 токенов; API возвращает warning,
  интерфейс показывает предупреждение. Это отдельный лимит от 4000 символов ввода.
- **Порт занят:** остановите старый процесс либо согласованно задайте другой порт API
  и `API_BASE_URL` для UI. Не запускайте второй training поверх работающего run.

Минимальные RAM и диск экспериментально не определялись. Замеры оборудования и памяти
конкретного запуска находятся в [latency.json](../reports/public/latency.json).
