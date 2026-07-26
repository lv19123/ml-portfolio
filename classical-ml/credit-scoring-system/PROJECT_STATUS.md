# Статус проекта

## Итоговый статус

**READY WITH LIMITATIONS**

Проект готов как воспроизводимый portfolio pet-project с полностью
работающим baseline-контуром: train → bundle → holdout evaluation →
prediction CSV. Ограничение относится к research champion: полноценный
новый CatBoost+POS bundle не переобучался на CPU, а старые CatBoost notebook
results не содержат валидного PR-AUC.

## Что было исправлено

- `src/config.py` — общие seed и пути к split, default bundle, metadata и
  evaluation.
- `src/validation.py` — отдельные train/inference-контракты, обязательные
  признаки, null/duplicate/empty checks.
- `src/features.py` — воспроизводимые агрегаты bureau, previous,
  installments, POS/CASH и credit card.
- `src/dataset.py` — загрузка CSV, сборка feature sets, фиксированный split,
  безопасный merge и stratified smoke sampling.
- `src/preprocessing.py` — единый Logistic Pipeline и CatBoost preprocessing.
- `src/model_bundle.py` — атомарное сохранение/загрузка модели вместе со
  schema, threshold и risk bands; поддержка legacy baseline.
- `src/metrics.py` — ROC-AUC, Average Precision, Brier, KS, accuracy,
  precision, recall, F1 и confusion counts.
- `src/train.py` — OOF CV, train-only выбор F1 threshold, final fit,
  holdout evaluation и CLI.
- `src/evaluate.py` — независимая оценка и сохранение таблиц/figures.
- `src/predict.py` — inference CLI без обучения.
- `src/utils.py` — strict JSON и версии runtime-библиотек.
- `pyproject.toml` и `requirements*.txt` — editable package, console scripts
  и зафиксированные версии.
- `.gitignore` — защита данных, secrets, caches, legacy моделей и больших
  построчных отчётов; разрешены компактные portfolio artifacts.
- `tests/` — добавлены тесты validation, preprocessing, bundle,
  probabilities, unknown categories, prediction schema, feature assembly,
  metrics и evaluation. Некорректный тест очистки notebook outputs заменён
  проверкой отсутствия error outputs.
- `README.md` — полностью обновлены бизнес-контекст, реальные результаты,
  архитектура и команды.
- `models/credit_scoring_model.joblib` и metadata — новый полный Logistic
  bundle в проверенном окружении.
- `reports/tables/` и `reports/figures/` — реальные holdout-артефакты.

Ноутбуки `01–08` не изменялись.

## Что проверено

| Команда/проверка | Результат |
| --- | --- |
| `python3 -m pytest -q` | `63 passed` |
| `python3 -m compileall -q src tests scripts` | Успешно |
| Импорт всех модулей `src.*` | 14 модулей импортированы |
| `python3 scripts/check_colab_compatibility.py --skip-data` | Passed, CPU fallback |
| `python3 scripts/check_colab_compatibility.py` | Passed со всеми raw CSV |
| Logistic smoke: 2 folds, 2 000/500 | Train, save, load, evaluate успешно |
| CatBoost+POS smoke: 2 folds, 20 iterations, 2 000/500 | Train, save, load успешно; smoke metrics не публикуются как финальные |
| `python3 -m src.train --model logistic --features application --cv-folds 3` | Полный train 246 008, holdout 61 503, bundle сохранён |
| `python3 -m src.evaluate` | Реальные holdout-таблицы и 4 figures сохранены |
| `python3 -m src.predict --input data/raw/application_test.csv ...` | 48 744 predictions, probability в `[0,1]` |
| Editable install в отдельном venv | Package и console scripts установлены |
| `pip install --dry-run -r requirements.txt` | Все pinned runtime/dev/notebook dependencies разрешены |
| `credit-predict` из `/tmp` | 48 744 predictions; cwd-independent |
| SHA-256 notebooks `01–08` до/после | Все восемь хэшей совпадают |
| Ignore rules через временный Git work tree | Данные/caches/legacy artifacts ignored; default bundle и compact reports не ignored |
| Поиск secrets/user-specific paths вне notebooks/data/models/reports | Секреты и пользовательские абсолютные пути не найдены |

`git status` и `git diff` выполнить нельзя: в предоставленном каталоге нет
`.git`.

## Итоговые метрики

### Выбранный запускаемый bundle: Logistic Regression

| Метрика | 3-fold CV | Holdout |
| --- | ---: | ---: |
| ROC-AUC | 0.7449006368 ± 0.0025252810 | 0.7486001109 |
| Average Precision | 0.2179283762 ± 0.0052747105 | 0.2285411788 |
| Brier score | — | 0.2025357722 |
| KS statistic | — | 0.3724979451 |
| Accuracy | — | 0.8304635546 |
| Precision | — | 0.2221204721 |
| Recall | — | 0.4396777442 |
| F1 | — | 0.2951395931 |

OOF F1 threshold: `0.6433644304`.

Confusion matrix: TN `48 893`, FP `7 645`, FN `2 782`, TP `2 183`.

### Research experiments

Лучший полный notebook CV ROC-AUC: CatBoost `application + POS_CASH`,
`0.7612492839 ± 0.0019830282`. Его PR-AUC в сохранённом notebook output
равен `NaN`, поэтому эксперимент не выбран как запускаемый final bundle.

All-features CatBoost `cpu_debug`: CV ROC-AUC `0.7586087863`, holdout
ROC-AUC `0.7655073854`, holdout AP `0.2498360063`. Эти значения не
финальные из-за 50 000/15 000 sampling, 2 folds и 150 iterations.

## Оставшиеся ограничения

- Полный новый CatBoost+POS bundle не обучен: CPU full CV и final fit
  потенциально длительны; выполнен только smoke test.
- Notebook CatBoost PR-AUC равен `NaN` из-за сохранённого результата
  `catboost.cv` на GPU. Новый CLI считает AP по реальным fold
  probabilities, но полный CatBoost CLI run не выполнялся.
- Logistic scores не проходили отдельную calibration. Brier и calibration
  curve опубликованы; значения следует считать model score, а не банковским
  PD.
- Нет out-of-time/external validation, cost matrix, fairness и production
  drift monitoring.
- Локальный Jupyter runtime заново не устанавливался и notebooks не
  перезапускались, чтобы сохранить их код и outputs. Их сохранённые Colab
  outputs и статические контракты проверены.
- В рабочем каталоге нет Git metadata и `LICENSE`.

## Как запустить проект

```bash
git clone <repository-url>
cd credit-scoring-system
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Скачайте Home Credit Default Risk с Kaggle и положите
`application_train.csv` и `application_test.csv` в `data/raw/`.
Для CatBoost default также нужен `POS_CASH_balance.csv`.

Получить predictions из включённого компактного baseline bundle:

```bash
python3 -m src.predict \
  --input data/raw/application_test.csv \
  --output predictions.csv
```

Переобучить и оценить baseline:

```bash
python3 -m src.train \
  --model logistic \
  --features application \
  --cv-folds 3
python3 -m src.evaluate
```

Обучить CatBoost+POS research configuration:

```bash
python3 -m src.train
python3 -m src.evaluate
```

Запустить тесты:

```bash
python3 -m pytest -q
```

Windows: используйте `py -3.10`, активируйте
`.\.venv\Scripts\Activate.ps1`, затем выполняйте те же команды через
`python -m`.
