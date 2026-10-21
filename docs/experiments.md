# Эксперименты

Run `2026-10-20T16-06-04.937473Z-76b6db38`. Сводка программно получена из сохранённых первичных отчётов:
[происхождение и SHA-256](../reports/public/provenance.json).
Генератор `scripts/write_run_summary.py` не обучает модели и не переписывает README.

## Шесть конфигураций — validation_model, n=1499

| Модель | Macro-F1 | Accuracy | Top-3 | Fit, с | Median / p95, мс |
| --- | ---: | ---: | ---: | ---: | ---: |
| tfidf-C0.5 | 0.7955 | 0.8152 | 0.9306 | 3.70 | 1.18 / 1.92 |
| tfidf-C1 | 0.8328 | 0.8406 | 0.9460 | 4.58 | 1.09 / 1.34 |
| tfidf-C4 | 0.8737 | 0.8706 | 0.9546 | 6.54 | 1.63 / 2.75 |
| minilm-C0.5 | 0.8899 | 0.8919 | 0.9613 | 1.80 | 23.52 / 33.29 |
| minilm-C1 | 0.9054 | 0.9033 | 0.9686 | 2.16 | 23.09 / 31.91 |
| minilm-C4 | 0.9180 | 0.9153 | 0.9740 | 2.70 | 19.02 / 26.94 |

TF-IDF: word n-grams (1,2), min_df=2, max_features=50000, sublinear_tf=True.
MiniLM: frozen all-MiniLM-L6-v2, нормализованные float32 embeddings размерности 384,
максимум 256 токенов. LR: lbfgs, max_iter=2000, seed=42, C={0.5,1,4}.
Все кандидаты сошлись. Fit TF-IDF включает fit преобразователя и LR; fit MiniLM
здесь означает только LR на готовых embeddings. Вычисление embeddings в fit не входит.

Выбор до test: максимум macro-F1 по всем 77 классам; варианты в пределах 0.005
сравниваются по median latency. При разнице latency до 5% предпочтение TF-IDF,
затем меньшему C. Выбран `minilm-C4`; tie-break не понадобился.
Baseline test отдельно не оценивался. [Полные числа](../reports/public/selection.json).

## Policy — отдельный validation_policy

Порог `0.41993725299835205` выбран после фиксации модели: максимальная coverage
при accepted accuracy ≥ 95% и accepted count ≥ 100.
На policy split приняты 1380 из 1499,
правильных 1311. После выбора refit нет.
Confidence = max predict_proba; threshold — сравниваемый с ним порог, не вероятность.
[Параметры policy и привязка к SHA-256 модели](../reports/public/policy.json).

## Official test — n=3080

Macro-F1 0.917388; accuracy 0.917532;
top-3 0.982468 (3026 правильных).

| Решение | Всего | Правильных top-1 | Ошибок top-1 |
| --- | ---: | ---: | ---: |
| auto_route | 2776 | 2657 | 119 |
| manual_review | 304 | 169 | 135 |
| all | 3080 | 2826 | 254 |

Порог перехватывает 53.15% ошибок, но оставляет
4.29% ошибок среди автоматических решений. Top-3 не измеряет
точность человека. Confidence не калиброван; Wilson 95% для accepted accuracy
[94.89%, 96.41%]
пересекает 95%, поэтому SLA не доказан.
[Test JSON](../reports/public/benchmark.json), [per-class](../reports/public/per_class.csv),
[путаницы](../reports/public/top_confusions.csv), [разбор ошибок](error_analysis.md).

## Производительность

Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz, Windows-10-10.0.19045-SP0, Python 3.11.17,
4 вычислительных потока, RAM 16.0 GiB.
10 warmup + 100 фиксированных одиночных test-запросов, batch_size=1.
Классификация: median 24.20 мс,
p95 32.07 мс; с retrieval:
median 25.96 мс, p95 31.47 мс.
Это замер внутри процесса, без HTTP, сети и конкурентной нагрузки. Серии независимы:
немного меньший p95 полного ответа не означает ускорение за счёт retrieval.
RSS 811.5 MiB — снимок, не peak.
HTTP latency и QPS не измерялись. [Условия, sample IDs и версии](../reports/public/latency.json).
