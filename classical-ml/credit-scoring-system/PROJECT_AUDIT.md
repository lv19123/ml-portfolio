# Технический аудит credit-scoring-system

Дата аудита: 2026-07-27.

Исходный вердикт до исправлений: **NOT READY**.

Проект содержит последовательный исследовательский ML-процесс, единый
фиксированный holdout, baseline и несколько CatBoost-экспериментов. Однако
законченного сценария обучения, оценки и инференса вне ноутбуков пока нет.
Полный финальный CatBoost не обучен, а PR-AUC во всех CatBoost CV-запусках
сохранён как `NaN`. Единственная оценка финального набора признаков получена
в техническом `cpu_debug`-режиме и не является финальным результатом.

## Матрица готовности

| Компонент | Статус | Проблема | Что нужно сделать |
| --------- | ------ | -------- | ----------------- |
| Структура репозитория | PARTIAL | Основные каталоги есть, но присутствуют локальные кэши и `catboost_info`; каталог не содержит `.git`, поэтому нельзя определить фактически отслеживаемые файлы. | Исключить генерируемые файлы, добавить недостающие placeholders и явно зафиксировать ограничение проверки Git. |
| Ноутбук 01: обзор данных | PARTIAL | Код EDA есть, но сохранённый ноутбук не содержит выполненных outputs. | Не изменять ноутбук; описать этап и реальные известные свойства данных в README. |
| Ноутбук 02: baseline | PARTIAL | Logistic Regression корректно использует sklearn Pipeline; CatBoost CV вернул `NaN` для PR-AUC. Holdout намеренно не оценён. | Сохранить notebook как есть; реализовать воспроизводимый baseline в `src/`, рассчитывать PR-AUC по реальным вероятностям. |
| Ноутбуки 03–07: агрегаты | PARTIAL | Пять таблиц признаков созданы, даты/месяцы после заявки фильтруются, но CatBoost PR-AUC во всех экспериментах равен `NaN`; код построения признаков доступен только в ноутбуках. | Повторно использовать сохранённые агрегаты через модуль сборки матрицы и исправить оценку вне ноутбуков. |
| Ноутбук 08: финальная модель | PARTIAL | Выполнен только `cpu_debug`: 50 000 train, 15 000 holdout, 2 folds, 150 итераций. Файлов `catboost_all_features.cbm` и metadata полного запуска нет. | Не менять notebook; добавить CLI обучения/оценки и явно отделить debug-артефакт от финального. |
| Загрузка данных | PARTIAL | Поиск файлов независим от cwd, но нет отдельного контракта train/inference и проверки схемы до чтения моделью. | Добавить явные функции загрузки и проверку обязательных колонок/файлов. |
| Валидация application train | PARTIAL | Проверяются ключ и бинарный `TARGET`, но не пустой датасет, null/тип ключа, дубликаты колонок и режим инференса. | Расширить `src.validation` и тесты для train и inference. |
| Валидация агрегатов | READY | Проверяются ключ, уникальность, отсутствие `TARGET`/`split`, префиксы и бесконечности; merge защищён `one_to_one`. | Использовать эти проверки в новом CLI-пайплайне. |
| Train/holdout split | READY | Фиксированный стратифицированный split с `random_state=42` сохраняется один раз; CV выполняется только на train, holdout открывается в notebook 08. | Вынести повторно используемый контракт split в `src/` без изменения ноутбуков. |
| Независимый validation внутри train | READY | Используется `StratifiedKFold(shuffle=True, random_state=42)`; preprocessing Logistic Regression обучается внутри каждого fold. | Сохранить методологию в CLI. |
| Проверка утечки TARGET | READY | `TARGET`, `SK_ID_CURR` и `split` исключены из матрицы; агрегаты не содержат `TARGET`; исторические даты фильтруются значениями `<= 0`. Явной утечки в изученном коде не найдено. | Закрепить проверками сборки матрицы и тестами. |
| Preprocessing Logistic Regression | READY | Imputation, scaling и one-hot encoding находятся в едином sklearn Pipeline; неизвестные категории игнорируются. | Переиспользовать эту архитектуру в `src.preprocessing`. |
| Preprocessing CatBoost | PARTIAL | Категориальные null заменяются `Unknown`, порядок и список признаков есть только в metadata; единого инференс-компонента нет. | Реализовать детерминированное выравнивание схемы и bundle/metadata-контракт. |
| Baseline-модель | READY | Реальные 3-fold результаты Logistic Regression сохранены: ROC-AUC `0.7448406488`, PR-AUC `0.2178652004`; Pipeline сохранён. | Добавить CLI-воспроизведение и полную holdout-оценку. |
| Основная модель | PARTIAL | CatBoost сравнивался на нескольких наборах признаков, но лучший полноценный CV-эксперимент (`application + POS_CASH`, ROC-AUC `0.7612492839`) не сохранён как используемый артефакт. All-features модель есть только в debug-варианте. | Реализовать обучение основной модели через CLI и сохранять полный bundle; не выдавать debug за финальный. |
| Реестр метрик | BROKEN | `model_metrics.csv` содержит `NaN` PR-AUC для всех CatBoost-строк; debug-результат смешан с полными CV-экспериментами. Файл сейчас исключён `.gitignore`. | Валидировать конечность метрик, вынести расширенную итоговую оценку в отдельную публикуемую таблицу. |
| Сравнение моделей | PARTIAL | Сравнение по CV ROC-AUC возможно, но сопоставимого PR-AUC для CatBoost нет; holdout использован только debug-моделью. | Строить сравнение по ROC-AUC и Average Precision из одинаковых folds; не выбирать модель по debug holdout. |
| Accuracy, precision, recall, F1 | MISSING | Метрики отсутствуют. | Добавить в `src.evaluate` как threshold-dependent дополнительные метрики. |
| Confusion matrix | MISSING | Не рассчитывается и не сохраняется. | Добавить таблицу и figure при оценке. |
| Калибровка и Brier score | MISSING | Не проверены; class weighting делает интерпретацию вероятности особенно важной. | Добавить Brier score и calibration curve; калибровку модели оставить будущим улучшением, если она не подтверждена validation. |
| KS statistic | MISSING | Не рассчитывается. | Добавить стандартный KS по распределениям score для классов. |
| Выбор порога | MISSING | Используемый порог не зафиксирован и не обоснован. | Реализовать настраиваемый порог с безопасным default `0.5`; отдельно документировать, что бизнес-порог требует cost matrix. |
| Анализ ошибок | MISSING | Нет таблицы false positive / false negative. | Сохранять компактный error analysis с ID, target, score и классом. |
| Интерпретация модели | MISSING | Нет feature importance или SHAP для итогового артефакта. | Добавить model-agnostic/нативную feature importance там, где модель её поддерживает; не добавлять тяжёлый SHAP без необходимости. |
| Распределение score и сегменты риска | MISSING | Нет histogram и категории риска в предсказаниях. | Добавить score distribution и прозрачные конфигурируемые risk bands. |
| Сохранение модели | PARTIAL | Baseline Pipeline и CatBoost `.cbm` сохранены; полноценной all-features модели нет. Debug metadata содержит нестандартные JSON `NaN`. | Добавить атомарное сохранение bundle и строго валидный JSON metadata. |
| Инференс | MISSING | Нет `src.predict`, нет проверки схемы и нет команды получения CSV. | Добавить CLI без обучения внутри predict; выводить ID, вероятность, класс и risk category. |
| Обучение из CLI | MISSING | Обучение возможно только через выполненные ноутбуки. | Добавить `python3 -m src.train` с smoke-параметрами и фиксированными seed. |
| Оценка из CLI | MISSING | Нет независимой загрузки артефакта и holdout-оценки. | Добавить `python3 -m src.evaluate` и сохранение таблиц/figures. |
| Тесты | PARTIAL | Есть тесты конфигурации, notebook-контрактов, tracking и базовой валидации. Нет тестов preprocessing, prediction, model loading, неизвестных категорий, диапазона вероятностей и missing files в CLI. | Добавить unit/integration tests требуемых сценариев. |
| Работа независимо от cwd | READY | `src.config` определяет корень от расположения модуля; текущие тесты это проверяют. | Проверить новые CLI из корня и посторонней cwd. |
| Локальные абсолютные пути | PARTIAL | Пользовательских Windows/macOS-путей в коде не найдено. В legacy bootstrap ноутбуков и Colab-конфигурации есть `/content/...`; это платформенный Colab fallback, но он остаётся жёстким путём. | Не добавлять новые абсолютные пути; использовать repo-relative `pathlib` в CLI. Ноутбуки 01–08 не менять из-за ограничения задачи. |
| Импорты | PARTIAL | `src` импортируется от корня; ноутбуки содержат собственный bootstrap `sys.path`. Запуск `python` в текущей локальной среде невозможен, доступен `python3`. | Документировать `python3`; проверить все новые модули через `python3 -m`. |
| Google Colab | PARTIAL | Bootstrap и отдельный compatibility-check есть, но зависимости устанавливаются без фиксации версии, а notebooks 02–07 требуют GPU. | Зафиксировать совместимые версии и сохранить понятный CPU smoke-сценарий через CLI. |
| Зависимости | PARTIAL | `requirements.txt` не фиксирует версии и смешивает runtime/dev/notebook зависимости. | Зафиксировать проверенные совместимые диапазоны или версии; не добавлять лишние библиотеки. |
| `.gitignore` | PARTIAL | Данные и модели исключены корректно, но не исключены `.pytest_cache/`, `catboost_info/` и типовые prediction outputs; полезная таблица метрик тоже полностью исключена. | Обновить правила, публиковать компактные отчёты, не публиковать сырые данные и тяжёлые модели. |
| README | PARTIAL | Есть описание notebook-flow и Colab, но нет полноценного бизнес-резюме, реальной итоговой таблицы, CLI train/evaluate/predict, ограничений и сценария от clone до prediction. | Переписать README на основе реально проверенных результатов. |
| Код-стиль и надёжность | PARTIAL | Основные функции имеют docstring/type hints, но CLI-слоя и logging нет; `model_config` подавляет любое исключение при GPU detection; часть логики дублируется в notebooks. | Добавить небольшие модули с logging и целевыми исключениями без рефакторинга notebooks. |
| Секреты | READY | Файлы credentials/tokens по именам не обнаружены; Kaggle credentials не используются в коде. | Сохранить запрет в `.gitignore` и инструкции. |
| Git/GitHub готовность | BROKEN | В проверяемом каталоге отсутствует `.git`; невозможно выполнить `git status`, `git diff`, доказать неизменность tracked notebooks или проверить попадание 2.7 ГБ данных в индекс. | В финале сравнить SHA-256 notebooks; пользователю инициализировать/подключить Git и перед публикацией проверить `git status`/`git ls-files`. |
| Лицензия | MISSING | `LICENSE` отсутствует. | Добавить лицензию только после выбора владельцем; до этого указать ограничение, не выбирать юридические условия автоматически. |

## Реальные результаты, найденные до исправлений

| Эксперимент | Режим | CV ROC-AUC | CV PR-AUC | Holdout ROC-AUC | Holdout PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Logistic Regression, application | full, 3 folds | 0.7448406488 | 0.2178652004 | — | — |
| CatBoost, application | full GPU, 3 folds | 0.7541763385 | `NaN` | — | — |
| CatBoost, application + bureau | full GPU, 3 folds | 0.7583623131 | `NaN` | — | — |
| CatBoost, application + previous | full GPU, 3 folds | 0.7602856557 | `NaN` | — | — |
| CatBoost, application + installments | full GPU, 3 folds | 0.7599574228 | `NaN` | — | — |
| CatBoost, application + POS_CASH | full GPU, 3 folds | 0.7612492839 | `NaN` | — | — |
| CatBoost, application + credit card | full GPU, 3 folds | 0.7580373685 | `NaN` | — | — |
| CatBoost, all features | `cpu_debug`, 2 folds | 0.7586087863 | `NaN` | 0.7655073854 | 0.2498360063 |

Значения `cpu_debug` нельзя использовать как финальные portfolio-метрики.
Лучший сопоставимый **полный CV ROC-AUC** среди сохранённых экспериментов —
`0.7612492839` у `application + POS_CASH`. Этого недостаточно для объявления
финальной модели: PR-AUC отсутствует, сам артефакт этого эксперимента не
сохранён, а независимая holdout-оценка для него не выполнялась.

## Зафиксированные ограничения аудита

- Каталог проекта не является Git working tree; исходный Git diff недоступен.
- Полное переобучение до первичного smoke test не запускалось.
- Ноутбуки `01–08` на этапе аудита не изменялись. Их исходные SHA-256
  сохранены отдельно в рабочем журнале проверки и будут сверены в финале.
- Сырые данные присутствуют локально (около 2.6 ГБ), но должны оставаться
  вне Git.

## Закрытие аудита после исправлений

Итоговый технический вердикт: **READY WITH LIMITATIONS**.

Исходная таблица выше сохранена как evidence состояния до изменений.
Фактическое закрытие пунктов:

| Компонент | Финальный статус | Результат |
| --- | --- | --- |
| Data loading и schema validation | READY | Добавлены `src.dataset` и расширенный `src.validation` для train/inference, missing files, обязательных колонок и запрета `TARGET` при prediction. |
| Feature engineering | READY | Агрегаты пяти исторических источников вынесены в тестируемые функции `src.features`; CLI может построить их без выполненных notebook state. |
| Split и leakage controls | READY | Сохранён фиксированный client split; OOF выполняется только на train; holdout используется однократно после выбора threshold. |
| Preprocessing | READY | Logistic preprocessing входит в sklearn Pipeline; CatBoost categorical contract хранится в bundle. Unknown categories протестированы. |
| Model bundle | READY | Создан `models/credit_scoring_model.joblib` размером около 24 КБ и строгий JSON metadata с версиями, feature schema, threshold и risk cutoffs. |
| Train CLI | READY | `python3 -m src.train`; проверены полный Logistic run и CatBoost+POS smoke. |
| Evaluate CLI | READY | `python3 -m src.evaluate`; сохраняются ROC-AUC, AP, Brier, KS, accuracy, precision, recall, F1, confusion matrix, calibration, errors, importance и risk segments. |
| Predict CLI | READY | `python3 -m src.predict`; проверены 48 744 строки application_test, ID/probability/class/risk schema и запуск из посторонней cwd после editable install. |
| Tests | READY | Финальный результат: `63 passed`; notebook outputs сохраняются, проверяется отсутствие error outputs. |
| Dependencies/package | READY | Добавлен `pyproject.toml`, exact runtime versions и editable console scripts; сборка и импорт из `/tmp` проверены. |
| `.gitignore` | READY | Raw/interim/processed data, legacy/heavy models, caches, secrets и error analysis игнорируются; компактный default bundle и portfolio reports разрешены. |
| Документация | READY | README содержит бизнес-задачу, реальные метрики, model choice, setup/data/train/evaluate/predict/test/Colab и ограничения. |
| Основная CatBoost-модель | PARTIAL | Ветка CatBoost+POS работает в smoke test, но полный новый bundle не обучался на CPU; старый лучший full CV имеет только ROC-AUC, notebook PR-AUC равен `NaN`. |
| Probability calibration | PARTIAL | Brier и calibration curve добавлены. Отдельная calibration намеренно не применена без train-only сравнительного эксперимента. |
| Git diff | BROKEN | `.git` по-прежнему отсутствует. Ignore-правила проверены через отдельный временный Git work tree; tracked state проверить невозможно. |
| LICENSE | MISSING | Юридические условия не выбирались автоматически; решение остаётся владельцу репозитория. |

Ноутбуки `01–08` после всех изменений имеют те же SHA-256, что и до
аудита.
