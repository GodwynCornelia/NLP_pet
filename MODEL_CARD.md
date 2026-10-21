# MODEL_CARD — Support Router v1

Назначение: локальная учебная маршрутизация английских банковских обращений по 77
intents. Не генератор ответов и не готовая банковская интеграция.

Run `2026-10-20T16-06-04.937473Z-76b6db38`, модель `minilm-C4`. Признаки классификатора: нормализованные MiniLM embeddings.
Для семантического поиска используется frozen
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (Apache 2.0), CPU,
нормализованные float32 эмбеддинги размерности 384.
Максимум 256 токенов, batch size=32.
LogisticRegression: C=4.0, lbfgs, max_iter=2000, seed=42.
Калибровки нет. Все fit завершились без ConvergenceWarning; iterations в selection.
Train: только train_fit (6994 примеров); [DATA_CARD](DATA_CARD.md).

Шесть конфигураций и выбор до test:
[validation_results.csv](reports/public/validation_results.csv). Выбран minilm-C4 по validation macro-F1; tie-break не понадобился. Test не менял выбор. Модель сохраняется
без refit после выбора порога. Веса привязаны к policy через SHA-256:
`f4dd877e2f74f43ebfef2b10c359a17f5341ff65a982ceed2ceb12d4f5e814f0`.

| Split | n | Macro-F1 | Accuracy | Top-3 accuracy |
| --- | ---: | ---: | ---: | ---: |
| validation_model | 1499 | 0.9180 | 0.9153 | 0.9740 |
| official test | 3080 | 0.9174 | 0.9175 | 0.9825 |

Policy split не участвовал в выборе C/представления. Порог `0.41993725299835205`
выбран на validation_policy (n=1499): accepted=1380,
coverage=92.06%, accepted accuracy=95.00%.
Цель: >=95% и >=100 принятых, максимум coverage; это учебное правило.
Параметры и итог подбора: [policy.json](reports/public/policy.json).
На test coverage=90.13%, accepted=2776,
accepted accuracy=95.71%; Wilson 95% [94.89%, 96.41%].
Интервал на отдельной выборке не позволяет заявлять гарантированный SLA.
Policy confidence использует >=; если цели нет, auto_routing_enabled=false,
threshold=null, accepted accuracy=null. API проверяет policy/model hash до readiness.

Слабые test-категории (автоматическая сводка, не результат ручного анализа студента):

| Category | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| pending_transfer | 0.929 | 0.650 | 0.765 | 40 |
| balance_not_updated_after_bank_transfer | 0.795 | 0.775 | 0.785 | 40 |
| declined_transfer | 0.966 | 0.700 | 0.812 | 40 |
| cash_withdrawal_not_recognised | 0.761 | 0.875 | 0.814 | 40 |
| transfer_not_received_by_recipient | 0.825 | 0.825 | 0.825 | 40 |
| card_payment_not_recognised | 0.886 | 0.775 | 0.827 | 40 |
| direct_debit_payment_not_recognised | 0.865 | 0.800 | 0.831 | 40 |
| transfer_timing | 0.755 | 0.925 | 0.831 | 40 |

Ошибки и путаницы: [per_class.csv](reports/public/per_class.csv),
[частые путаницы](reports/public/top_confusions.csv),
[20 ошибок для самостоятельного разбора](reports/public/error_analysis.csv).
Это зафиксированная оценка; настройка под эти ошибки означала бы новый
исследовательский цикл с уже просмотренным test.

Hardware: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz, 16.0 GiB RAM,
4 потока, Python 3.11.17. Замер полного ответа без HTTP:
median=25.96 ms, p95=31.47 ms.
10 warmup + фиксированные 100 test-примеров по одному, batch_size=1.
RSS при замере=811.5 MiB, не peak RAM.
[Протокол, размеры и версии](reports/public/latency.json), lock в корне.

Ограничения: один домен и язык; возможен высокий score для русского текста,
бессмыслицы и неизвестной темы — OOD-детектора нет. Confidence не является
калиброванной вероятностью правильности. Длинные тексты усечены по token limit;
warning возвращается при использовании encoder. Retrieval содержит только train_fit;
similarity — контекст для оператора, не причинное объяснение классификатора.
Нет мониторинга drift, оценки на современных запросах банка, интеграции с CRM
или автоматического обучения на пользовательских данных. Для портфолио и локальных
экспериментов, не для решений с финансовыми последствиями.

## Проверка завершения, 21 октября 2026

Независимый инференс сохранённых весов подтвердил 2826 правильных top-1,
3026 top-3 и 254 ошибки из 3080. Среди auto_route: 2657 правильных и 119 ошибок;
в manual_review: 169 правильных и 135 ошибок. Порог перехватывает 53,15% ошибок,
но оставляет 4,29% ошибочных решений в автоматической группе.
[Первичный аудит](reports/public/reproducibility.json).

Чистое контейнерное воспроизведение Linux/amd64 без прежних ресурсов проекта:
run `2026-10-21T04-28-44.192246Z-949706ba`, те же закреплённые commit/revision,
шесть конфигураций и прежний протокол. Champion minilm-C4, test macro-F1
0.9173880677241317, accuracy 0.9175324675324675, top-3 0.9824675324675325;
accepted 2657/2776. Эти агрегаты совпали с исходным run. Linux threshold
`0.41994646191596985` сохранён как получен; исходный threshold и веса не заменены.
Это проверка воспроизводимости на уже просмотренном test, не новый подбор или
независимый holdout. Бинарное совпадение повторно обученных файлов между ОС не требовалось.

Docker/API/UI, повторный старт и offline-инференс проверены, включая загрузку
исходных Windows-артефактов в Linux с сетью none. Хеши исходных selection, policy,
classifier, manifest, test_metrics и active pointer совпали до/после проверок.
[Полный отчёт завершения](docs/reproducibility.md).
[Анализ 20 ошибок агентом](docs/error_analysis.md); выводы студента не заполнены.
Ручной просмотр частично подтверждён скриншотами пользователя: predict, CSV с ошибками
строк и вкладка качества. Скачанный CSV и визуальное предупреждение об усечении пока
не показаны. [Фактически подтверждённые шаги](docs/manual_ui_checklist.md).
