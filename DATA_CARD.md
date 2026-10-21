# DATA_CARD — BANKING77

Источник: [PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets).
Авторы: Iñigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson, Ivan Vulić.
[Efficient Intent Detection with Dual Sentence Encoders, ACL 2020](https://arxiv.org/abs/2003.04807).
Лицензия: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Атрибуция и изменения: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Язык: английский. Схема исходных CSV: text, category (строки). Категорий 77;
канонические имена берутся из данных, включая регистр и пунктуацию.
До обучения проверены схема, пустые поля, число записей CSV-парсером и train-классы.
Обнаружено пустых train-текстов: 0; пропусков: 0.
Размеры классов исходного train: 35–187.
Длины исходного train: 13–433 символов,
среднее 59.47.

Run: `2026-10-20T16-06-04.937473Z-76b6db38`. Источник: [reports/public/data.json](reports/public/data.json).
Train: 10003 исходных → 9992 очищенных.
Официальный test: 3080, байты и записи не менялись.
Причины исключений: `{"official_test_overlap": 7, "duplicate_same_label": 4}`.
Групп конфликтующих меток не обнаружено; код исключает всю такую группу при появлении.
Исключённые sample_id и тексты сохраняются локально после prepare. В Git включена
только [сводка аудита и split](reports/public/data.json).

Общий вход: Unicode NFC, trim, свёртка пробелов. Исходный text сохраняется отдельно.
Ключ группировки: нормализованный текст + casefold, SHA-256. Метки test для
исключения train-overlap не используются. Перефразы семантически не удаляются.
Это вариант benchmark с очищенным train; не все опубликованные результаты сопоставимы.

Разбиение стратифицированное, seed=42, train_fit=6994,
validation_model=1499, validation_policy=1499.
В каждой части 77 классов, group_id/sample_id между частями не пересекаются.
Fit TF-IDF/LR и retrieval — только train_fit; остальные части используются раздельно.
Список ID и хеши: `artifacts/<run_id>/splits.json` (локально после prepare).
Проверяемые сводки: [reports/public/data.json](reports/public/data.json).

Dataset commit: `57ec275d8078af65b7731c2a98be812d844a6d6b`; commit pinning: True.
Загрузка UTC: `2026-10-20T16:04:41.208619+00:00`. Исходные байты:

- `raw/57ec275d8078af65b7731c2a98be812d844a6d6b/train.csv`: SHA-256 `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`, 839073 bytes.
- `raw/57ec275d8078af65b7731c2a98be812d844a6d6b/test.csv`: SHA-256 `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`, 239961 bytes.

MiniLM revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; Apache 2.0.
Хеши файлов encoder и CSV: [resources.lock.json](configs/resources.lock.json).
Происхождение публичных чисел: [provenance.json](reports/public/provenance.json).
Source manifest SHA-256: `47310139106aa80626af5b190d220e5663b48c1910fa82ff8476d4dc9c5ce218`.
Split manifest SHA-256: `4e0e34da541ddcb1600cbf7a85402e51e564c5d841b1b32f7ce7c874e24c2f03`.

Пользовательские CSV не пополняют данные автоматически, не сохраняются в отчёты/Git/логи.
Код предполагает локальные доверенные данные, не заявляет удаление всех возможных
чувствительных сведений из публичного корпуса. До реального использования нужна
отдельная оценка приватности и актуальности данных.
