# Как устроен Fashion Recommender System

Этот документ помогает пройти проект в том же порядке, в котором строится рекомендательная система. Главная идея notebook-first версии: важные ML-действия видны непосредственно в ячейках; `src` используется для повторяющейся технической логики, а не для сокрытия полного pipeline.

## Задача

Для каждого клиента нужно выбрать до 12 товаров на следующую неделю. В данных есть покупки, но нет явных оценок. Поэтому покупка является positive implicit signal, а отсутствие покупки — слабым negative: клиент мог не увидеть товар или товар мог быть недоступен.

Архитектура не менялась:

```text
Popularity + Personal History + ALS + Content-Based
→ candidates
→ user-item features
→ CatBoostClassifier probability
→ Top-12
```

## Путь по notebooks

### 01 — Data Overview

- Вход: три CSV из `data/raw`.
- Видимые шаги: `shape`, `head`, `dtypes`, `isna`, `nunique`, проверки ключей и диапазона дат.
- Технические функции: только `load_transactions`, `load_articles`, `load_customers`.
- Выход: проверенные DataFrame с `t_dat` типа datetime и строковым `article_id`.
- Артефакт: отдельный файл не создаётся.
- Нужно уметь объяснить: почему ID нельзя хранить как число и почему leading zero важен для join.

### 02 — EDA

- Вход: загруженные transactions, articles, customers.
- Видимые шаги: `groupby`, `value_counts`, `describe`, ограниченные `merge`, histogram и bar plots.
- Выход: наблюдения о long tail, активности клиентов, категориях, цене, каналах и пропусках.
- Артефакт: отдельный обязательный файл не создаётся.
- Нужно уметь объяснить: как EDA обосновывает sparse matrix, popularity fallback и выбранные признаки.

### 03 — Temporal Validation

- Вход: `transactions_train.csv`.
- Видимые шаги: `last_date`, `cutoff_date`, фильтры history/future, проверка границ, history users, cold-start filter, ground truth через `sort_values → drop_duplicates → groupby → list`.
- Метрики: ручные формулы Recall и AP для одного пользователя; затем общие функции Recall, MAP, HitRate и Candidate Recall.
- Выход: основной test split и три последовательных окна для следующих notebooks.
- Артефакт: `data/processed/temporal_windows.json`.
- Нужно уметь объяснить: почему history заканчивается строго до cutoff и почему random split создал бы leakage.

### 04 — Baselines

- Вход: transactions и test-границы из `temporal_windows.json`.
- Видимые шаги: Popularity через `groupby.size` с детерминированным tie-break `count DESC, article_id ASC`; Recent History через сортировку и `drop_duplicates`; Frequent History через `groupby.size`; небольшой popularity fallback.
- Выход: Top-12 каждого baseline и одинаковые offline-метрики.
- Артефакт: `reports/tables/baseline_metrics.csv`.
- Нужно уметь объяснить: почему сложную модель нельзя оценивать без простого baseline.

### 05 — ALS

- Вход: transactions и три temporal windows.
- Видимые шаги: user-item `groupby`, `purchase_count`, confidence, четыре mappings, `.map`, `csr_matrix`, форма/nnz/density, создание `AlternatingLeastSquares`, `.fit`, `.recommend`, преобразование indices обратно в `article_id`.
- Повторение: train и test оформлены отдельными блоками; массовый цикл существует только внутри технического user inference.
- Выход: до 200 ALS candidates для train/validation/test; notebook 07 использует первые 150.
- Артефакты: `data/processed/als_candidates_{train,validation,test}.parquet`, alias `als_candidates.parquet`, `models/als_model.npz`, `models/mappings/als_user_ids.json`, `models/mappings/als_article_ids.json`, `reports/tables/als_metrics.csv`. Финальная API-модель обучается отдельно на всей доступной истории после offline test.
- Нужно уметь объяснить: implicit feedback, log confidence, ориентацию user-item matrix и назначение mappings.

### 06 — Content-Based

- Вход: articles, transactions и temporal windows.
- Видимые шаги: шесть категорий, заполнение `Unknown`, `OneHotEncoder.fit_transform`, sparse item matrix, user-item history, recency/frequency/purchase weights, sparse profile одного пользователя, `cosine_similarity(profile, article_matrix)`, Top items без seen purchases.
- Повторение: после примера используется техническая batched-функция массового inference; train и test не находятся в общем цикле. Dense item-item matrix не создаётся.
- Выход: 50 content candidates для train/validation/test.
- Артефакты: `data/processed/content_candidates_{train,validation,test}.parquet`, alias `content_candidates.parquet`, `models/content_encoder.joblib`, `models/article_feature_matrix.npz`, `models/content_config.json`, article mappings, `reports/tables/content_metrics.csv`.
- Нужно уметь объяснить: почему профиль делится на сумму весов и что измеряет cosine similarity.

### 07 — Candidate Generation

- Вход: готовые ALS и Content Parquet каждого окна, transactions и window boundaries.
- Видимые шаги: проверка файлов, чтение Parquet, ограничение ALS до 150, Recent/Frequent History, Popularity, `pd.concat`, `groupby` по паре, scores, ranks, source flags, source count, лимит 250, проверки дубликатов и Candidate Recall.
- Здесь нет: `AlternatingLeastSquares`, `.fit` ALS, `OneHotEncoder` или построения content profiles.
- Выход: одна строка на candidate pair и Simple Hybrid test baseline.
- Артефакты: `data/processed/merged_candidates_{train,validation,test}.parquet`, alias `merged_candidates.parquet`, `candidate_source_analysis.csv`, `simple_hybrid_metrics.csv`.
- Нужно уметь объяснить: почему Candidate Recall ограничивает максимальное качество ranking model.

### 08 — Feature Engineering

- Вход: `merged_candidates_{train,validation,test}.parquet`, transactions, articles, customers и boundaries.
- Видимые шаги: отдельные короткие блоки user features, item features, user-item history, category affinity; обычные `merge`; target через positive pairs из соответствующей future-недели; заполнение пропусков и проверки.
- Здесь нет: обучения ALS, Content-Based и генерации candidates с нуля.
- Выход: leakage-safe candidate tables с признаками и target.
- Артефакты: `data/processed/train_ranking_table.parquet`, `validation_ranking_table.parquet`, `test_ranking_table.parquet`.
- Нужно уметь объяснить: откуда взят каждый признак, какую дату он видит и почему не строится user × catalog.

### 09 — CatBoost Ranking

- Вход: три готовые ranking tables.
- Видимые шаги: явные `FEATURE_COLUMNS` и `CATEGORICAL_FEATURES`, раздельные X/y, negative sampling, создание `CatBoostClassifier`, `.fit` с validation/early stopping, `.predict_proba`, `sort_values`, group rank, Top-12 и popularity fallback.
- Здесь нет: повторного ALS, OneHotEncoder, candidate merge или полного feature engineering.
- Выход: final test metrics, feature importance и готовая выдача.
- Артефакты: `models/catboost_recommender.cbm`, feature/category JSON, popularity JSON, metadata JSON, `artifacts/final_recommendations.parquet`, `reports/tables/catboost_feature_importance.csv`, `catboost_metrics.csv`.
- Нужно уметь объяснить: почему classifier probability можно использовать как score внутри пользователя и почему test не участвует в fit.

### 10 — Model Comparison

- Вход: реально созданные `baseline_metrics.csv`, `als_metrics.csv`, `content_metrics.csv`, `simple_hybrid_metrics.csv`, `catboost_metrics.csv`.
- Видимые шаги: `read_csv`, `concat`, выбор столбцов, сортировка и один bar chart.
- Выход: единая таблица без подстановки нулей за отсутствующие метрики.
- Артефакты: `reports/tables/model_metrics.csv`, `reports/figures/model_comparison.png`.
- Нужно уметь объяснить: почему все модели должны сравниваться на одном cohort и одном временном окне.

### 11 — Дополнительный batch inference

- Вход: сохранённый CatBoost, готовая test ranking table, feature/category JSON, popularity и metadata.
- Видимые шаги: загрузка `.cbm`, подготовка X, `predict_proba`, сортировка, Top-12, fallback и сохранение.
- Здесь нет: обучения ALS, Content-Based или CatBoost и генерации кандидатов с нуля.
- Выход: пример повторного scoring готовой candidate table.
- Артефакт: необязательный `artifacts/batch_demo_recommendations.parquet`; основной API-файл остаётся результатом notebook 09.
- Нужно уметь объяснить: различие training, batch scoring готовых признаков и HTTP lookup.

## Контракты артефактов

| Producer | Файл | Consumer | Минимальные ключи |
|---|---|---|---|
| 03 | `temporal_windows.json` | 04–08 | cutoff и target end каждого окна |
| 05 | `als_candidates_<split>.parquet` | 07 | customer, article, ALS score/rank |
| 06 | `content_candidates_<split>.parquet` | 07 | customer, article, cosine score/rank |
| 07 | `merged_candidates_<split>.parquet` | 08 | уникальная pair + source scores/ranks/flags |
| 08 | `<split>_ranking_table.parquet` | 09; test table также 11 | уникальная pair + features + target |
| 09 | `catboost_recommender.cbm` и feature JSON | 11 | готовая модель и порядок признаков |
| 09 | `final_recommendations.parquet` | API | customer, article, rank, score |

Ошибка отсутствующего файла должна указывать notebook-producer. Это делает зависимость явной и предотвращает скрытое повторное обучение.

## Ключевые понятия

### Sparse matrix

Плотная матрица `users × items` почти полностью состояла бы из нулей. CSR хранит только позиции и значения покупок. В проекте user-item и item-feature matrices остаются sparse; полная item-item similarity matrix не строится.

### ALS и confidence

ALS приближает наблюдаемые implicit interactions произведением user и item factors. Значение:

```text
confidence = 1 + log1p(purchase_count)
```

усиливает повторную покупку, но логарифм ограничивает влияние очень больших counts. Factor indices не содержат исходных ID, поэтому mappings являются частью модели.

### Weighted content profile

```text
recency_weight   = exp(-days_since_purchase / decay_days)
frequency_weight = 1 + log1p(purchase_count)
purchase_weight  = recency_weight × frequency_weight
profile          = Σ(weight × item_vector) / Σ(weight)
```

Cosine similarity сравнивает направление user profile и item vector. Для one-hot категорий это сходство набора товарных свойств, а не понимание текста или изображения.

### Candidate Recall

Candidate Recall@K — средняя доля future items, присутствующих среди K кандидатов пользователя. Если товар отсутствует в candidates, CatBoost не может вернуть его независимо от качества ranking.

### Classification-based ranking

CatBoost решает бинарную задачу на каждой generated pair. Probability не интерпретируется как причинный эффект; она используется для сортировки кандидатов одного пользователя. Это намеренно не `CatBoostRanker`.

## Контроль leakage

Для каждого split notebook 08 выполняет одинаковое правило:

1. `history = transactions[t_dat < cutoff_date]`.
2. Popularity и Personal History используют только history.
3. ALS matrix обучается только на history своего окна.
4. Content weights и profiles используют только history своего окна.
5. User/item/pair/category features агрегируют только history.
6. Future появляется только при построении ground truth и `positive_pairs` для target.
7. Validation используется в `eval_set` и early stopping.
8. Test используется только в `.predict_proba` и итоговых метриках.

Список evaluation users можно определить по наличию события в future, но future items и future aggregates не должны попадать в признаки или candidate scores.

## Что делает `src`

- `data`: единая безопасная загрузка CSV и нормализация `article_id`.
- `evaluation`: повторяемая договорённость о Recall/AP/MAP/HitRate/Candidate Recall.
- `persistence`: компактное сохранение/загрузка JSON, Parquet, ALS, CatBoost, encoder и CSR.
- `als`, `content_based`, `baselines`, `candidates`, `features`, `ranking`: повторяющиеся операции дополнительного batch inference; notebook-обучение их крупные wrappers не вызывает.
- `batch`: необязательная обработка пользователей порциями из сохранённых models/mappings.
- `api`: lookup готового Parquet, health, recommendations и model metadata.

Удалён модуль `pipeline.py`, потому что его функции только объединяли метрики и скрывали понятные операции. Также удалены одноразовые wrappers для baseline recommendations, target, negative sampling, ALS/CatBoost fit, simple hybrid, DataFrame-to-dict и feature-list getters. Их реальный код теперь находится рядом с объяснением в соответствующем notebook.

## Вопросы для самопроверки

1. Почему cutoff-строка должна принадлежать future, а не history?
2. Почему `article_id` хранится строкой?
3. Чем implicit confidence отличается от rating?
4. Почему CSR критична для user-item matrix?
5. Как проверить ориентацию matrix для установленной версии `implicit`?
6. Зачем сохранять ALS mappings вместе с factor matrices?
7. Почему log1p используется в confidence и frequency weight?
8. Почему content profile делится на сумму весов?
9. Что именно означает cosine similarity для one-hot категорий?
10. Почему нельзя строить полную item-item dense matrix?
11. Какие разные сигналы дают четыре candidate sources?
12. Почему Candidate Recall измеряется до CatBoost?
13. Почему target создаётся только для generated pairs?
14. Как вычислить category affinity без future data?
15. Почему negative sampling выполняется только для train?
16. Почему CatBoost validation разрешена, а test в `fit` запрещён?
17. Как probability превращается в Top-12?
18. Почему fallback должен удалять повторы?
19. Чем Recall@12 отличается от HitRate@12 и MAP@12?
20. Почему API не должен запускать ALS или CatBoost на каждый запрос?

## Рекомендуемый порядок изучения

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11
→ src/data + evaluation + persistence
→ tests
→ optional batch/API
```

При разборе каждого этапа задайте пять вопросов: какие данные он видит, какой результат создаёт, где хранится артефакт, как ограничена память и где возможен leakage.
