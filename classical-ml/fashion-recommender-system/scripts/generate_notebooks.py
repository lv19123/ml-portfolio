"""Generate the Junior-friendly notebook sequence 03–11.

The generator deliberately creates many small cells.  Each code cell answers one
question, and train/validation/test pipelines are never hidden inside one loop.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def step(title: str, what: str, why: str, result: str, source: str) -> list[dict]:
    explanation = (
        f"### {title}\n\n"
        f"**Что делаем:** {what}  \n"
        f"**Зачем:** {why}  \n"
        f"**Что получим:** {result}"
    )
    return [md(explanation), code(source)]


def setup(title: str, description: str) -> list[dict]:
    return [
        md(f"# {title}\n\n{description}"),
        md(
            "## Подключение проекта\n\n"
            "**Что делаем:** определяем корень проекта.  \n"
            "**Зачем:** одинаковые пути должны работать локально и в Colab.  \n"
            "**Что получим:** `PROJECT_ROOT` и доступный пакет из `src`."
        ),
        code(
            """from pathlib import Path
import sys

try:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_ROOT = Path("/content/drive/MyDrive/fashion-recommender-system")
except ImportError:
    PROJECT_ROOT = Path.cwd().resolve()

if not (PROJECT_ROOT / "src").is_dir():
    raise FileNotFoundError(f"Не найдена папка src: {PROJECT_ROOT / 'src'}")
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
print("Корень проекта:", PROJECT_ROOT)"""
        ),
        md(
            "### Зависимости\n\n"
            "**Что делаем:** устанавливаем requirements только в Colab.  \n"
            "**Зачем:** локальное окружение не должно изменяться при каждом запуске.  \n"
            "**Что получим:** готовые библиотеки для следующих ячеек."
        ),
        code(
            """import subprocess

if "google.colab" in sys.modules:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r",
         str(PROJECT_ROOT / "requirements.txt")],
        check=True,
    )"""
        ),
    ]


def raw_paths() -> str:
    return """RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
TRANSACTIONS_PATH = RAW_DIR / "transactions_train.csv"
ARTICLES_PATH = RAW_DIR / "articles.csv"
CUSTOMERS_PATH = RAW_DIR / "customers.csv"
for directory in [PROCESSED_DIR, MODEL_DIR, REPORT_DIR, ARTIFACT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)"""


def check_file(variable: str, notebook_name: str) -> str:
    return f"""if not {variable}.is_file():
    raise FileNotFoundError(
        f"Не найден файл: {{{variable}}}\\n"
        "Сначала выполните notebook {notebook_name}."
    )"""


def refine_existing_01_02() -> None:
    """Add explanations only where old notebooks had adjacent code cells."""
    additions = {
        "01_data_overview_colab.ipynb": {
            7: "### Пути к CSV\n\n**Что делаем:** задаём три входных файла.  \n**Зачем:** все загрузчики должны читать один каталог.  \n**Что получим:** понятные пути к исходным таблицам.",
            8: "### Чтение данных\n\n**Что делаем:** загружаем три таблицы.  \n**Зачем:** загрузчики проверят схемы и типы ID.  \n**Что получим:** `transactions`, `articles`, `customers`.",
        },
        "02_eda_colab.ipynb": {
            5: "### Библиотеки EDA\n\n**Что делаем:** импортируем pandas, NumPy и matplotlib.  \n**Зачем:** анализ и графики остаются прямо в notebook.  \n**Что получим:** инструменты для следующих ячеек.",
            6: "### Пути к данным\n\n**Что делаем:** задаём пути к CSV.  \n**Зачем:** отделяем настройку путей от загрузки.  \n**Что получим:** три входных пути.",
            7: "### Загрузка таблиц\n\n**Что делаем:** читаем transactions, articles и customers.  \n**Зачем:** EDA использует проверенные типы и ключи.  \n**Что получим:** три DataFrame.",
            14: "### Гистограмма активности\n\n**Что делаем:** ограничиваем хвост только для графика.  \n**Зачем:** несколько очень активных клиентов не должны скрыть основную массу.  \n**Что получим:** читаемое распределение покупок.",
            17: "### Категории товаров\n\n**Что делаем:** строим небольшие bar charts по категориям.  \n**Зачем:** сравниваем популярность товарных признаков.  \n**Что получим:** шесть компактных графиков.",
        },
    }
    for filename, markdown_by_index in additions.items():
        path = NOTEBOOK_DIR / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        existing_markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        marker = next(iter(markdown_by_index.values())).splitlines()[0]
        if marker in existing_markdown:
            continue
        new_cells = []
        for index, cell in enumerate(notebook["cells"]):
            if index in markdown_by_index:
                new_cells.append(md(markdown_by_index[index]))
            new_cells.append(cell)
        notebook["cells"] = new_cells
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def notebook_03() -> list[dict]:
    cells = setup(
        "03. Temporal validation и метрики",
        "Разделяем прошлое и будущее небольшими шагами, строим ground truth и сохраняем границы трёх окон.",
    )
    cells += step(
        "Импорты",
        "подключаем pandas, метрики и JSON persistence",
        "в notebook не нужны модели",
        "минимальный набор импортов",
        """import numpy as np
import pandas as pd
from IPython.display import display

from fashion_recommender.data import load_transactions
from fashion_recommender.evaluation import (
    average_precision_at_k, candidate_recall_at_k, hit_rate_at_k,
    map_at_k, mean_recall_at_k, recall_at_k,
)
from fashion_recommender.persistence import save_json""",
    )
    cells += step(
        "Пути и параметры",
        "задаём каталоги и длину target-периода",
        "границы должны использоваться одинаково дальше",
        "`FUTURE_DAYS = 7` и reproducible seed",
        raw_paths() + "\n\nFUTURE_DAYS = 7\nRANDOM_STATE = 42",
    )
    cells += step(
        "Загрузка транзакций",
        "читаем историю покупок",
        "cutoff определяется по фактической последней дате",
        "DataFrame с корректным datetime",
        """transactions = load_transactions(TRANSACTIONS_PATH)
print("Форма transactions:", transactions.shape)
display(transactions.head())""",
    )
    cells += step(
        "Максимальная дата",
        "находим последнюю дату данных",
        "от неё отсчитывается последняя неделя",
        "`last_date`",
        """last_date = transactions["t_dat"].max()
print("Последняя дата:", last_date)""",
    )
    cells += step(
        "Начало target-периода",
        "вычисляем cutoff для семи календарных дней",
        "обе границы недели включительны",
        "`cutoff_date`",
        """cutoff_date = last_date - pd.Timedelta(days=FUTURE_DAYS - 1)
print("Target начинается:", cutoff_date)""",
    )
    cells += step(
        "History",
        "берём строки строго раньше cutoff",
        "модели не должны видеть target-неделю",
        "таблицу прошлого",
        """history = transactions[
    transactions["t_dat"] < cutoff_date
].copy()
print("History:", history.shape)""",
    )
    cells += step(
        "Future",
        "берём строки начиная с cutoff",
        "это ответы для offline-оценки",
        "таблицу будущего",
        """future = transactions[
    transactions["t_dat"] >= cutoff_date
].copy()
print("Future:", future.shape)""",
    )
    cells += step(
        "Проверка диапазонов",
        "сравниваем крайние даты и число строк",
        "так обнаруживается пересечение или потеря данных",
        "явные leakage-checks",
        """print("History заканчивается:", history["t_dat"].max())
print("Future начинается:", future["t_dat"].min())
print("Строк после split:", len(history) + len(future))

assert history["t_dat"].max() < future["t_dat"].min()
assert len(history) + len(future) == len(transactions)""",
    )
    cells += step(
        "Пользователи с историей",
        "собираем ID из history",
        "персональные модели работают только для известных клиентов",
        "множество `history_users`",
        """history_users = set(history["customer_id"])
print("Пользователей в history:", len(history_users))""",
    )
    cells += step(
        "Cold start",
        "находим future-пользователей без history",
        "их нельзя смешивать с обычной персональной оценкой",
        "отдельный список cold-start ID",
        """future_users = set(future["customer_id"])
cold_start_users = future_users - history_users
print("Пользователей в future:", len(future_users))
print("Cold-start пользователей:", len(cold_start_users))""",
    )
    cells += step(
        "Evaluation future",
        "оставляем future только известных пользователей",
        "ground truth должен соответствовать доступной истории",
        "`future_evaluation`",
        """future_evaluation = future[
    future["customer_id"].isin(history_users)
].copy()
print("Evaluation future:", future_evaluation.shape)""",
    )
    cells += step(
        "Уникальные будущие пары",
        "сортируем и удаляем повторные user-item покупки",
        "один товар учитывается в метриках один раз",
        "`future_unique`",
        """future_unique = (
    future_evaluation
    .sort_values("t_dat")
    .drop_duplicates(["customer_id", "article_id"])
)
display(future_unique.head())""",
    )
    cells += step(
        "Ground truth",
        "собираем список будущих товаров каждого пользователя",
        "метрики сравнивают рекомендации с этим словарём",
        "полный `ground_truth` без искусственного обрезания",
        """ground_truth = (
    future_unique
    .groupby("customer_id", sort=False)["article_id"]
    .apply(list)
    .to_dict()
)
print("Пользователей в ground truth:", len(ground_truth))""",
    )
    cells += step(
        "Пример пользователя",
        "показываем один будущий список",
        "проверяем смысл структуры до расчёта метрик",
        "один customer ID и его товары",
        """example_customer = next(iter(ground_truth))
print("Пользователь:", example_customer)
print("Future items:", ground_truth[example_customer])""",
    )
    cells += step(
        "Маленький пример",
        "задаём actual и recommended items вручную",
        "формулы проще проверить на четырёх позициях",
        "данные для ручного Recall и AP",
        """actual_items = ["A", "B", "C"]
recommended_items = ["A", "X", "B", "Y"]
K = 4

actual_set = set(actual_items)
top_k = list(dict.fromkeys(recommended_items))[:K]
print("Actual:", actual_items)
print("Top-K:", top_k)""",
    )
    cells += step(
        "Recall вручную",
        "делим число найденных товаров на число actual items",
        "Recall отвечает за полноту",
        "ручное значение Recall@4",
        """hits = actual_set & set(top_k)
manual_recall = len(hits) / len(actual_set)
print("Попадания:", hits)
print("Recall@4:", manual_recall)""",
    )
    cells += step(
        "AP вручную",
        "суммируем precision только на позициях попаданий",
        "AP учитывает порядок рекомендаций",
        "ручное значение AP@4",
        """precision_sum = 0.0
hits_so_far = 0
for rank, article_id in enumerate(top_k, start=1):
    if article_id in actual_set:
        hits_so_far += 1
        precision_sum += hits_so_far / rank

manual_ap = precision_sum / min(len(actual_set), K)
print("AP@4:", manual_ap)""",
    )
    cells += step(
        "Проверка готовыми метриками",
        "сравниваем ручные значения с функциями из src",
        "единые функции нужны во всех следующих notebooks",
        "совпадающие Recall и AP",
        """print("Recall function:", recall_at_k(actual_items, recommended_items, K))
print("AP function:", average_precision_at_k(actual_items, recommended_items, K))

assert np.isclose(manual_recall, recall_at_k(actual_items, recommended_items, K))
assert np.isclose(manual_ap, average_precision_at_k(actual_items, recommended_items, K))""",
    )
    cells += step(
        "Групповые метрики",
        "считаем Recall, MAP, HitRate и Candidate Recall на двух пользователях",
        "видим различие метрик до экспериментов",
        "четыре проверочных значения",
        """example_truth = {"u1": ["A", "B"], "u2": ["C"]}
example_recommendations = {"u1": ["A", "X"], "u2": ["Y", "C"]}

print("Mean Recall@2:", mean_recall_at_k(example_truth, example_recommendations, 2))
print("MAP@2:", map_at_k(example_truth, example_recommendations, 2))
print("HitRate@2:", hit_rate_at_k(example_truth, example_recommendations, 2))
print("Candidate Recall@2:", candidate_recall_at_k(example_truth, example_recommendations, 2))""",
    )
    cells += step(
        "Границы трёх окон",
        "задаём train, validation и test явно",
        "следующие notebooks используют одинаковый protocol",
        "шесть дат без цикла по pipeline",
        """train_cutoff = last_date - pd.Timedelta(days=20)
train_end = train_cutoff + pd.Timedelta(days=6)
validation_cutoff = last_date - pd.Timedelta(days=13)
validation_end = validation_cutoff + pd.Timedelta(days=6)
test_cutoff = last_date - pd.Timedelta(days=6)
test_end = last_date

print("Train:", train_cutoff.date(), "—", train_end.date())
print("Validation:", validation_cutoff.date(), "—", validation_end.date())
print("Test:", test_cutoff.date(), "—", test_end.date())""",
    )
    cells += step(
        "Конфигурация окон",
        "собираем даты в компактный JSON-словарь",
        "следующие этапы не должны вычислять границы заново",
        "`window_config`",
        """window_config = {
    "train": {"cutoff_date": train_cutoff, "target_end_date": train_end},
    "validation": {
        "cutoff_date": validation_cutoff,
        "target_end_date": validation_end,
    },
    "test": {"cutoff_date": test_cutoff, "target_end_date": test_end},
}
display(pd.DataFrame(window_config).T)""",
    )
    cells += step(
        "Сохранение окон",
        "записываем только даты",
        "notebooks 04–08 загрузят один общий контракт",
        "`temporal_windows.json`",
        """windows_path = PROCESSED_DIR / "temporal_windows.json"
save_json(window_config, windows_path)
print("Сохранено:", windows_path)""",
    )
    return cells


def notebook_04() -> list[dict]:
    cells = setup(
        "04. Понятные baseline-модели",
        "Popularity, Recent History и Frequent History рассчитываются отдельно и сравниваются на одном test cohort.",
    )
    cells += step(
        "Импорты",
        "подключаем pandas, загрузчики и три итоговые метрики",
        "baseline не требует ML-библиотек",
        "короткий набор импортов",
        """import numpy as np
import pandas as pd
from IPython.display import display

from fashion_recommender.data import load_transactions
from fashion_recommender.baselines import popular_items
from fashion_recommender.evaluation import hit_rate_at_k, map_at_k, mean_recall_at_k
from fashion_recommender.persistence import load_json""",
    )
    cells += step(
        "Пути и параметры",
        "проверяем JSON временных окон",
        "baseline должен использовать test history, а не все данные",
        "готовые пути и константы",
        raw_paths()
        + "\n\nWINDOWS_PATH = PROCESSED_DIR / \"temporal_windows.json\"\n"
        + check_file("WINDOWS_PATH", "03_temporal_validation_colab.ipynb")
        + "\n\nK = 12\nMAX_EVALUATION_USERS = 2_000\nRANDOM_STATE = 42",
    )
    cells += step(
        "Загрузка входов",
        "читаем транзакции и границы",
        "подготовка отделена от baseline-логики",
        "`transactions` и `windows`",
        """transactions = load_transactions(TRANSACTIONS_PATH)
windows = load_json(WINDOWS_PATH)
print("Transactions:", transactions.shape)""",
    )
    cells += step(
        "Test-границы",
        "выбираем test cutoff и конец target",
        "все три baseline сравниваются на одной неделе",
        "две даты",
        """test_cutoff = pd.Timestamp(windows["test"]["cutoff_date"])
test_end = pd.Timestamp(windows["test"]["target_end_date"])
print("Test:", test_cutoff.date(), "—", test_end.date())""",
    )
    cells += step(
        "History и target",
        "разделяем test-окно",
        "baseline видит только прошлое",
        "две таблицы и leakage assert",
        """history = transactions[transactions["t_dat"] < test_cutoff].copy()
target = transactions[
    transactions["t_dat"].between(test_cutoff, test_end)
].copy()

assert history["t_dat"].max() < test_cutoff
print("History:", history.shape)
print("Target:", target.shape)""",
    )
    cells += step(
        "Ground truth",
        "оставляем известных пользователей и уникальные future pairs",
        "cold-start не смешивается с персональными baseline",
        "полный словарь ответов",
        """known_users = set(history["customer_id"])
target_evaluation = target[target["customer_id"].isin(known_users)]
target_unique = (
    target_evaluation
    .sort_values("t_dat")
    .drop_duplicates(["customer_id", "article_id"])
)
ground_truth = target_unique.groupby("customer_id", sort=False)["article_id"].apply(list).to_dict()
print("Ground-truth users:", len(ground_truth))""",
    )
    cells += step(
        "Воспроизводимый cohort",
        "случайно выбираем до 2 000 пользователей",
        "результат не зависит от порядка строк словаря",
        "`evaluation_users` и `ground_truth_sample`",
        """all_users = np.array(sorted(ground_truth))
sample_size = min(MAX_EVALUATION_USERS, len(all_users))
rng = np.random.default_rng(RANDOM_STATE)
evaluation_users = rng.choice(all_users, size=sample_size, replace=False).tolist()
ground_truth_sample = {
    customer_id: ground_truth[customer_id]
    for customer_id in evaluation_users
}
print("Evaluation users:", len(evaluation_users))""",
    )
    cells += step(
        "Popularity items",
        "считаем детерминированный глобальный Top-12 по history",
        "count убывает, а article ID разрешает ties одинаково во всех environments",
        "список популярных article ID",
        """popular_table = popular_items(history, limit=K)
popular_article_ids = popular_table["article_id"].tolist()
print("Popularity Top-12:", popular_article_ids)""",
    )
    cells += step(
        "Popularity рекомендации",
        "копируем один список каждому пользователю",
        "модель не использует customer history",
        "словарь рекомендаций",
        """popularity_recommendations = {
    customer_id: popular_article_ids.copy()
    for customer_id in evaluation_users
}
print(popularity_recommendations[evaluation_users[0]])""",
    )
    cells += step(
        "Popularity метрики",
        "оцениваем первый baseline",
        "он задаёт минимальный ориентир качества",
        "`popularity_metrics`",
        """popularity_metrics = {
    "model": "Popularity",
    "Recall@12": mean_recall_at_k(ground_truth_sample, popularity_recommendations, K),
    "MAP@12": map_at_k(ground_truth_sample, popularity_recommendations, K),
    "HitRate@12": hit_rate_at_k(ground_truth_sample, popularity_recommendations, K),
    "users_evaluated": len(ground_truth_sample),
    "average_candidates": K,
    "notes": "Top-12 baseline; common cohort",
}
display(pd.Series(popularity_metrics))""",
    )
    cells += step(
        "Небольшой fallback",
        "дополняем короткий персональный список popularity",
        "каждый baseline должен вернуть K уникальных товаров",
        "простую вспомогательную функцию",
        """def fill_with_popularity(personal_items, fallback_items, k=12):
    result = []
    for article_id in list(personal_items) + list(fallback_items):
        if article_id not in result:
            result.append(article_id)
        if len(result) == k:
            break
    return result""",
    )
    cells += step(
        "Recent History сортировка",
        "ставим самые свежие покупки первыми",
        "recency является персональным сигналом",
        "отсортированный history cohort",
        """cohort_history = history[
    history["customer_id"].isin(evaluation_users)
].copy()
recent_history = cohort_history.sort_values("t_dat", ascending=False)
display(recent_history.head())""",
    )
    cells += step(
        "Уникальные recent items",
        "убираем повтор одной пары и собираем списки",
        "один товар не должен занимать несколько позиций",
        "`recent_items_by_user`",
        """recent_unique = recent_history.drop_duplicates(
    ["customer_id", "article_id"]
)
recent_items_by_user = (
    recent_unique
    .groupby("customer_id", sort=False)["article_id"]
    .apply(list)
    .to_dict()
)
display(recent_unique.head())""",
    )
    cells += step(
        "Recent рекомендации",
        "берём recent items и дополняем popularity",
        "неактивные пользователи тоже получают K товаров",
        "Top-12 Recent History",
        """recent_recommendations = {}
for customer_id in evaluation_users:
    personal_items = recent_items_by_user.get(customer_id, [])
    recent_recommendations[customer_id] = fill_with_popularity(
        personal_items, popular_article_ids, K
    )

print(recent_recommendations[evaluation_users[0]])""",
    )
    cells += step(
        "Recent метрики",
        "оцениваем второй baseline отдельно",
        "его результат не смешан с Popularity",
        "`recent_metrics`",
        """recent_metrics = {
    "model": "Recent Personal History",
    "Recall@12": mean_recall_at_k(ground_truth_sample, recent_recommendations, K),
    "MAP@12": map_at_k(ground_truth_sample, recent_recommendations, K),
    "HitRate@12": hit_rate_at_k(ground_truth_sample, recent_recommendations, K),
    "users_evaluated": len(ground_truth_sample),
    "average_candidates": K,
    "notes": "Top-12 baseline; common cohort",
}
display(pd.Series(recent_metrics))""",
    )
    cells += step(
        "Frequent History counts",
        "считаем число покупок каждой user-item пары",
        "частые повторные покупки получают больший приоритет",
        "таблицу `frequent_history`",
        """frequent_history = (
    cohort_history
    .groupby(["customer_id", "article_id"], as_index=False)
    .size()
    .rename(columns={"size": "purchase_count"})
)
display(frequent_history.head())""",
    )
    cells += step(
        "Frequent сортировка",
        "сортируем по count, затем по article ID",
        "tie-break делает результат воспроизводимым",
        "ранжированные пары",
        """frequent_ranked = frequent_history.sort_values(
    ["customer_id", "purchase_count", "article_id"],
    ascending=[True, False, True],
)
frequent_items_by_user = (
    frequent_ranked
    .groupby("customer_id", sort=False)["article_id"]
    .apply(list)
    .to_dict()
)
display(frequent_ranked.head())""",
    )
    cells += step(
        "Frequent рекомендации",
        "дополняем персональные списки popularity",
        "получаем третий самостоятельный baseline",
        "Top-12 Frequent History",
        """frequent_recommendations = {}
for customer_id in evaluation_users:
    personal_items = frequent_items_by_user.get(customer_id, [])
    frequent_recommendations[customer_id] = fill_with_popularity(
        personal_items, popular_article_ids, K
    )

print(frequent_recommendations[evaluation_users[0]])""",
    )
    cells += step(
        "Frequent метрики",
        "оцениваем третий baseline отдельно",
        "теперь три строки готовы к сравнению",
        "`frequent_metrics`",
        """frequent_metrics = {
    "model": "Frequent Personal History",
    "Recall@12": mean_recall_at_k(ground_truth_sample, frequent_recommendations, K),
    "MAP@12": map_at_k(ground_truth_sample, frequent_recommendations, K),
    "HitRate@12": hit_rate_at_k(ground_truth_sample, frequent_recommendations, K),
    "users_evaluated": len(ground_truth_sample),
    "average_candidates": K,
    "notes": "Top-12 baseline; common cohort",
}
display(pd.Series(frequent_metrics))""",
    )
    cells += step(
        "Итоговая таблица",
        "собираем три заранее рассчитанных словаря метрик",
        "здесь нет цикла, скрывающего различия baseline",
        "`baseline_metrics`",
        """baseline_metrics = pd.DataFrame([
    popularity_metrics,
    recent_metrics,
    frequent_metrics,
])
display(baseline_metrics)""",
    )
    cells += step(
        "Сохранение baseline",
        "записываем компактную CSV",
        "notebook 10 загрузит готовые результаты",
        "`baseline_metrics.csv`",
        """baseline_path = REPORT_DIR / "baseline_metrics.csv"
baseline_metrics.to_csv(baseline_path, index=False)
print("Сохранено:", baseline_path)""",
    )
    return cells


def als_repeated_split_cells(prefix: str, label: str) -> list[dict]:
    title = label.capitalize()
    return (
        step(
            f"{title}: границы",
            f"выбираем {label}-окно",
            "validation уже показал устройство ALS подробно",
            f"`{prefix}_cutoff` и `{prefix}_end`",
            f'''{prefix}_cutoff = pd.Timestamp(windows["{prefix}"]["cutoff_date"])
{prefix}_end = pd.Timestamp(windows["{prefix}"]["target_end_date"])
print("{title}:", {prefix}_cutoff.date(), "—", {prefix}_end.date())''',
        )
        + step(
            f"{title}: history и target",
            "делим транзакции по выбранным датам",
            "ALS должен видеть только строки до cutoff",
            f"`{prefix}_history` и `{prefix}_target`",
            f'''{prefix}_history = transactions[
    transactions["t_dat"] < {prefix}_cutoff
].copy()
{prefix}_target = transactions[
    transactions["t_dat"].between({prefix}_cutoff, {prefix}_end)
].copy()

assert {prefix}_history["t_dat"].max() < {prefix}_cutoff
print("History:", {prefix}_history.shape, "Target:", {prefix}_target.shape)''',
        )
        + step(
            f"{title}: ground truth",
            "повторяем уже разобранную подготовку ответов",
            "evaluation использует полный future список",
            f"`{prefix}_ground_truth`",
            f'''{prefix}_ground_truth = build_ground_truth(
    {prefix}_target,
    set({prefix}_history["customer_id"]),
)
print("Ground-truth users:", len({prefix}_ground_truth))''',
        )
        + step(
            f"{title}: cohort",
            "делаем воспроизводимую выборку пользователей",
            "ALS и Content-Based должны работать на одинаковом cohort",
            f"`{prefix}_users` и `{prefix}_ground_truth_sample`",
            f'''{prefix}_users, {prefix}_ground_truth_sample = sample_ground_truth(
    {prefix}_ground_truth,
    MAX_EVALUATION_USERS,
    RANDOM_STATE,
)
print("Evaluation users:", len({prefix}_users))''',
        )
        + step(
            f"{title}: sparse inputs",
            "повторяем показанные groupby, mappings и CSR технической функцией",
            "подробная реализация уже была видна на validation",
            f"`{prefix}_interactions`",
            f'''{prefix}_interactions = prepare_user_item_matrix({prefix}_history)
print("Matrix:", {prefix}_interactions.matrix.shape)
print("NNZ:", {prefix}_interactions.matrix.nnz)''',
        )
        + step(
            f"{title}: модель",
            "создаём отдельный ALS для этого cutoff",
            "разные окна не должны использовать future interactions",
            f"`{prefix}_model` до обучения",
            f'''{prefix}_model = AlternatingLeastSquares(
    factors=FACTORS,
    regularization=REGULARIZATION,
    iterations=ITERATIONS,
    random_state=RANDOM_STATE,
)
print({prefix}_model)''',
        )
        + step(
            f"{title}: обучение",
            "вызываем `.fit()` только на history matrix",
            "оценка и сохранение остаются в следующих ячейках",
            f"обученный `{prefix}_model`",
            f'''{prefix}_fit_started = time.perf_counter()
{prefix}_model.fit({prefix}_interactions.matrix, show_progress=False)
{prefix}_training_time = time.perf_counter() - {prefix}_fit_started
print("Training seconds:", round({prefix}_training_time, 2))''',
        )
        + step(
            f"{title}: массовые кандидаты",
            "получаем Top-K для evaluation cohort",
            "массовый inference является разрешённой технической функцией",
            f"`{prefix}_candidates`",
            f'''{prefix}_candidates = generate_als_candidates(
    {prefix}_model,
    {prefix}_interactions,
    customer_ids={prefix}_users,
    limit=CANDIDATE_LIMIT,
)
print("Candidate rows:", len({prefix}_candidates))
display({prefix}_candidates.head())''',
        )
        + step(
            f"{title}: метрики",
            "считаем Candidate Recall и standalone Top-12",
            "оценка не смешана с fit или сохранением",
            f"`{prefix}_metrics`",
            f'''{prefix}_candidate_lists = {prefix}_candidates.groupby(
    "customer_id", sort=False
)["article_id"].apply(list).to_dict()
{prefix}_top12_lists = {prefix}_candidates[
    {prefix}_candidates["als_rank"] <= 12
].groupby("customer_id", sort=False)["article_id"].apply(list).to_dict()
{prefix}_metrics = {{
    "Candidate Recall": candidate_recall_at_k(
        {prefix}_ground_truth_sample, {prefix}_candidate_lists, CANDIDATE_LIMIT
    ),
    "Recall@12": mean_recall_at_k({prefix}_ground_truth_sample, {prefix}_top12_lists, 12),
    "MAP@12": map_at_k({prefix}_ground_truth_sample, {prefix}_top12_lists, 12),
    "HitRate@12": hit_rate_at_k({prefix}_ground_truth_sample, {prefix}_top12_lists, 12),
}}
display(pd.Series({prefix}_metrics))''',
        )
        + step(
            f"{title}: сохранение кандидатов",
            "записываем только candidate table",
            "notebook 07 загрузит файл без повторного ALS fit",
            f"`als_candidates_{prefix}.parquet`",
            f'''{prefix}_candidates_path = PROCESSED_DIR / "als_candidates_{prefix}.parquet"
{prefix}_candidates.to_parquet({prefix}_candidates_path, index=False)
print("Сохранено:", {prefix}_candidates_path)''',
        )
    )


def notebook_05() -> list[dict]:
    cells = setup(
        "05. ALS шаг за шагом",
        "Сначала подробно разбираем validation: interactions, confidence, mappings, CSR, fit и recommend. Train и test повторяют уже понятные шаги без цикла по окнам.",
    )
    cells += step(
        "Импорты",
        "подключаем sparse matrix, implicit ALS и технический массовый inference",
        "основная подготовка validation всё равно выполняется напрямую",
        "необходимые классы и метрики",
        """import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from IPython.display import display

from fashion_recommender.als import (
    InteractionMatrix, generate_als_candidates, prepare_user_item_matrix,
)
from fashion_recommender.data import load_transactions
from fashion_recommender.evaluation import (
    candidate_recall_at_k, hit_rate_at_k, map_at_k, mean_recall_at_k,
)
from fashion_recommender.persistence import load_json, save_als_model, save_json""",
    )
    cells += step(
        "Пути и ALS-параметры",
        "проверяем temporal windows и задаём один набор гиперпараметров",
        "никаких скрытых настроек между окнами быть не должно",
        "пути и константы",
        raw_paths()
        + "\n\nWINDOWS_PATH = PROCESSED_DIR / \"temporal_windows.json\"\n"
        + check_file("WINDOWS_PATH", "03_temporal_validation_colab.ipynb")
        + "\n\nFACTORS = 64\nREGULARIZATION = 0.05\nITERATIONS = 20"
        + "\nCANDIDATE_LIMIT = 200\nMAX_EVALUATION_USERS = 2_000\nRANDOM_STATE = 42",
    )
    cells += step(
        "Загрузка входов",
        "читаем transactions и JSON окон",
        "данные загружаются один раз",
        "`transactions` и `windows`",
        """transactions = load_transactions(TRANSACTIONS_PATH)
windows = load_json(WINDOWS_PATH)
print("Transactions:", transactions.shape)""",
    )
    cells += step(
        "Validation-границы",
        "выбираем одно окно для подробного разбора",
        "validation не является final test",
        "`validation_cutoff` и `validation_end`",
        """validation_cutoff = pd.Timestamp(windows["validation"]["cutoff_date"])
validation_end = pd.Timestamp(windows["validation"]["target_end_date"])
print("Validation:", validation_cutoff.date(), "—", validation_end.date())""",
    )
    cells += step(
        "Validation history",
        "берём покупки строго раньше cutoff",
        "ALS matrix не должна видеть validation target",
        "`validation_history`",
        """validation_history = transactions[
    transactions["t_dat"] < validation_cutoff
].copy()
print("Validation history:", validation_history.shape)
assert validation_history["t_dat"].max() < validation_cutoff""",
    )
    cells += step(
        "Validation target",
        "берём только семь дней validation",
        "эта таблица используется для ground truth, не для fit",
        "`validation_target`",
        """validation_target = transactions[
    transactions["t_dat"].between(validation_cutoff, validation_end)
].copy()
print("Validation target:", validation_target.shape)""",
    )
    cells += step(
        "Известные validation users",
        "фильтруем cold-start клиентов",
        "ALS не имеет factor для пользователя без history",
        "`validation_target_evaluation`",
        """validation_known_users = set(validation_history["customer_id"])
validation_target_evaluation = validation_target[
    validation_target["customer_id"].isin(validation_known_users)
].copy()
print("Known target users:", validation_target_evaluation["customer_id"].nunique())""",
    )
    cells += step(
        "Уникальные validation pairs",
        "сортируем future и удаляем повторные пары",
        "ground truth учитывает товар один раз",
        "`validation_target_unique`",
        """validation_target_unique = (
    validation_target_evaluation
    .sort_values("t_dat")
    .drop_duplicates(["customer_id", "article_id"])
)
display(validation_target_unique.head())""",
    )
    cells += step(
        "Validation ground truth",
        "собираем будущие товары каждого пользователя",
        "словарь пока остаётся полным",
        "`validation_ground_truth`",
        """validation_ground_truth = (
    validation_target_unique
    .groupby("customer_id", sort=False)["article_id"]
    .apply(list)
    .to_dict()
)
print("Ground-truth users:", len(validation_ground_truth))""",
    )
    cells += step(
        "Validation cohort",
        "случайно выбираем до 2 000 ID с фиксированным seed",
        "порядок словаря не влияет на эксперимент",
        "отдельный `validation_ground_truth_sample`",
        """validation_all_users = np.array(sorted(validation_ground_truth))
validation_sample_size = min(MAX_EVALUATION_USERS, len(validation_all_users))
validation_rng = np.random.default_rng(RANDOM_STATE)
validation_users = validation_rng.choice(
    validation_all_users,
    size=validation_sample_size,
    replace=False,
).tolist()
validation_ground_truth_sample = {
    customer_id: validation_ground_truth[customer_id]
    for customer_id in validation_users
}
print("Evaluation users:", len(validation_users))""",
    )
    cells += step(
        "User-item counts",
        "агрегируем повторные покупки пары",
        "implicit ALS работает с силой наблюдаемого сигнала",
        "`validation_interaction_counts`",
        """validation_interaction_counts = (
    validation_history
    .groupby(["customer_id", "article_id"])
    .size()
    .reset_index(name="purchase_count")
)
display(validation_interaction_counts.head())""",
    )
    cells += step(
        "Confidence",
        "преобразуем purchase count логарифмом",
        "повторы усиливают сигнал без линейного роста",
        "столбец `confidence`",
        """validation_interaction_counts["confidence"] = (
    1 + np.log1p(validation_interaction_counts["purchase_count"])
)
display(validation_interaction_counts.head())""",
    )
    cells += step(
        "Списки ID",
        "создаём детерминированные списки users и items",
        "номер строки/столбца должен иметь обратное отображение",
        "`validation_index_to_user` и `validation_index_to_item`",
        """validation_index_to_user = sorted(
    validation_interaction_counts["customer_id"].astype(str).unique()
)
validation_index_to_item = sorted(
    validation_interaction_counts["article_id"].astype(str).unique()
)
print("Users:", len(validation_index_to_user))
print("Items:", len(validation_index_to_item))""",
    )
    cells += step(
        "Mappings",
        "назначаем числовой индекс каждому исходному ID",
        "sparse matrix принимает только числовые координаты",
        "два словаря ID → index",
        """validation_user_to_index = {
    customer_id: index
    for index, customer_id in enumerate(validation_index_to_user)
}
validation_item_to_index = {
    article_id: index
    for index, article_id in enumerate(validation_index_to_item)
}
print("User mapping example:", next(iter(validation_user_to_index.items())))
print("Item mapping example:", next(iter(validation_item_to_index.items())))""",
    )
    cells += step(
        "Числовые индексы",
        "применяем mappings к таблице взаимодействий",
        "каждая пара получает координату CSR",
        "`user_index` и `item_index`",
        """validation_interaction_counts["user_index"] = (
    validation_interaction_counts["customer_id"].astype(str).map(validation_user_to_index)
)
validation_interaction_counts["item_index"] = (
    validation_interaction_counts["article_id"].astype(str).map(validation_item_to_index)
)
display(validation_interaction_counts.head())""",
    )
    cells += step(
        "CSR matrix",
        "передаём confidence и координаты в `csr_matrix`",
        "нули не хранятся в памяти",
        "`validation_user_item_matrix`",
        """validation_user_item_matrix = csr_matrix(
    (
        validation_interaction_counts["confidence"].astype("float32"),
        (
            validation_interaction_counts["user_index"],
            validation_interaction_counts["item_index"],
        ),
    ),
    shape=(len(validation_index_to_user), len(validation_index_to_item)),
    dtype=np.float32,
)""",
    )
    cells += step(
        "Размер sparse matrix",
        "смотрим shape, nnz и density",
        "density должна быть очень маленькой",
        "проверку памяти и ориентации users × items",
        """validation_density = (
    validation_user_item_matrix.nnz
    / (validation_user_item_matrix.shape[0] * validation_user_item_matrix.shape[1])
)
print("Type:", type(validation_user_item_matrix))
print("Shape:", validation_user_item_matrix.shape)
print("NNZ:", validation_user_item_matrix.nnz)
print("Density:", validation_density)""",
    )
    cells += step(
        "Создание ALS",
        "задаём модель до обучения",
        "создание объекта и fit — разные действия",
        "`validation_model`",
        """validation_model = AlternatingLeastSquares(
    factors=FACTORS,
    regularization=REGULARIZATION,
    iterations=ITERATIONS,
    random_state=RANDOM_STATE,
)
print(validation_model)""",
    )
    cells += step(
        "Обучение ALS",
        "вызываем `.fit()` на user-item CSR",
        "target не передаётся модели",
        "обученные user/item factors",
        """validation_fit_started = time.perf_counter()
validation_model.fit(validation_user_item_matrix, show_progress=False)
validation_training_time = time.perf_counter() - validation_fit_started
print("User factors:", validation_model.user_factors.shape)
print("Item factors:", validation_model.item_factors.shape)
print("Training seconds:", round(validation_training_time, 2))""",
    )
    cells += step(
        "Один пользователь",
        "находим matrix index первого evaluation user",
        "сначала проверяем API recommend на одном примере",
        "`validation_example_user_index`",
        """validation_example_user = validation_users[0]
validation_example_user_index = validation_user_to_index[validation_example_user]
print("Customer:", validation_example_user)
print("Matrix row:", validation_example_user_index)""",
    )
    cells += step(
        "Один вызов recommend",
        "получаем item indices и scores",
        "проверяем фактический API установленного implicit",
        "два NumPy-массива",
        """validation_example_indices, validation_example_scores = validation_model.recommend(
    validation_example_user_index,
    validation_user_item_matrix[validation_example_user_index],
    N=10,
    filter_already_liked_items=True,
)
print("Indices:", validation_example_indices[:5])
print("Scores:", validation_example_scores[:5])""",
    )
    cells += step(
        "Обратное преобразование ID",
        "заменяем item indices исходными article ID",
        "API и метрики работают с бизнес-идентификаторами",
        "таблицу десяти рекомендаций",
        """validation_example_articles = [
    validation_index_to_item[int(item_index)]
    for item_index in validation_example_indices
]
validation_example_table = pd.DataFrame({
    "article_id": validation_example_articles,
    "als_score": validation_example_scores,
})
display(validation_example_table)""",
    )
    cells += step(
        "Технический контейнер",
        "упаковываем matrix и mappings для массового recommend",
        "эта операция не скрывает уже показанную подготовку",
        "`validation_interactions`",
        """validation_interactions = InteractionMatrix(
    matrix=validation_user_item_matrix,
    user_to_index=validation_user_to_index,
    item_to_index=validation_item_to_index,
    index_to_user=validation_index_to_user,
    index_to_item=validation_index_to_item,
)""",
    )
    cells += step(
        "Массовые validation candidates",
        "вызываем переиспользуемый batched recommend",
        "цикл по пользователям является техническим inference, не циклом по окнам",
        "`validation_candidates`",
        """validation_candidates = generate_als_candidates(
    validation_model,
    validation_interactions,
    customer_ids=validation_users,
    limit=CANDIDATE_LIMIT,
)
print("Candidate rows:", len(validation_candidates))
display(validation_candidates.head())""",
    )
    cells += step(
        "Validation Candidate Recall",
        "собираем candidate lists и оцениваем покрытие",
        "ранжировщик не найдёт товар вне candidates",
        "Candidate Recall@CANDIDATE_LIMIT",
        """validation_candidate_lists = validation_candidates.groupby(
    "customer_id", sort=False
)["article_id"].apply(list).to_dict()
validation_candidate_recall = candidate_recall_at_k(
    validation_ground_truth_sample,
    validation_candidate_lists,
    CANDIDATE_LIMIT,
)
print(f"Candidate Recall@{CANDIDATE_LIMIT}:", validation_candidate_recall)""",
    )
    cells += step(
        "Validation Top-12",
        "оцениваем первые 12 ALS items",
        "standalone качество сравнимо с baseline",
        "Recall/MAP/HitRate",
        """validation_top12_lists = validation_candidates[
    validation_candidates["als_rank"] <= 12
].groupby("customer_id", sort=False)["article_id"].apply(list).to_dict()

validation_metrics = {
    "Recall@12": mean_recall_at_k(validation_ground_truth_sample, validation_top12_lists, 12),
    "MAP@12": map_at_k(validation_ground_truth_sample, validation_top12_lists, 12),
    "HitRate@12": hit_rate_at_k(validation_ground_truth_sample, validation_top12_lists, 12),
}
display(pd.Series(validation_metrics))""",
    )
    cells += step(
        "Сохранение validation candidates",
        "записываем candidate table отдельно от модели",
        "notebook 07 сможет загрузить её напрямую",
        "`als_candidates_validation.parquet`",
        """validation_candidates_path = PROCESSED_DIR / "als_candidates_validation.parquet"
validation_candidates.to_parquet(validation_candidates_path, index=False)
print("Сохранено:", validation_candidates_path)""",
    )
    cells += step(
        "Ground truth helper",
        "фиксируем уже показанные pandas-шаги для других окон",
        "функция не обучает и ничего не сохраняет",
        "небольшую `build_ground_truth`",
        """def build_ground_truth(target, known_users):
    target_evaluation = target[target["customer_id"].isin(known_users)]
    target_unique = (
        target_evaluation
        .sort_values("t_dat")
        .drop_duplicates(["customer_id", "article_id"])
    )
    return (
        target_unique
        .groupby("customer_id", sort=False)["article_id"]
        .apply(list)
        .to_dict()
    )""",
    )
    cells += step(
        "Sampling helper",
        "фиксируем воспроизводимый выбор cohort",
        "полный ground truth не перезаписывается",
        "users и отдельный sample dictionary",
        """def sample_ground_truth(ground_truth, max_users, random_state):
    all_users = np.array(sorted(ground_truth))
    sample_size = min(max_users, len(all_users))
    rng = np.random.default_rng(random_state)
    users = rng.choice(all_users, size=sample_size, replace=False).tolist()
    ground_truth_sample = {
        customer_id: ground_truth[customer_id]
        for customer_id in users
    }
    return users, ground_truth_sample""",
    )
    cells += als_repeated_split_cells("train", "train")
    cells += als_repeated_split_cells("test", "test")
    cells += step(
        "Test alias",
        "сохраняем удобное имя test candidates",
        "старые consumers остаются совместимыми",
        "`als_candidates.parquet`",
        """als_alias_path = PROCESSED_DIR / "als_candidates.parquet"
test_candidates.to_parquet(als_alias_path, index=False)
print("Сохранено:", als_alias_path)""",
    )
    cells += step(
        "Финальные ALS interactions",
        "строим matrix по всей доступной истории отдельно от offline test",
        "API-модель не должна быть test-моделью, обученной только до cutoff",
        "`final_interactions`",
        """final_interactions = prepare_user_item_matrix(transactions)
print("Final matrix:", final_interactions.matrix.shape)
print("Final NNZ:", final_interactions.matrix.nnz)""",
    )
    cells += step(
        "Финальная ALS-модель",
        "создаём новый объект для batch/API",
        "offline metrics уже рассчитаны test-моделью",
        "необученный `final_als_model`",
        """final_als_model = AlternatingLeastSquares(
    factors=FACTORS,
    regularization=REGULARIZATION,
    iterations=ITERATIONS,
    random_state=RANDOM_STATE,
)
print(final_als_model)""",
    )
    cells += step(
        "Обучение финальной ALS",
        "обучаем модель на всех доступных transactions",
        "она сохраняется для дополнительного batch inference",
        "обученный `final_als_model`",
        """final_fit_started = time.perf_counter()
final_als_model.fit(final_interactions.matrix, show_progress=False)
final_training_time = time.perf_counter() - final_fit_started
print("Final training seconds:", round(final_training_time, 2))""",
    )
    cells += step(
        "Сохранение ALS model",
        "записываем модель отдельно от mappings",
        "batch inference загрузит factor matrices",
        "`models/als_model.npz`",
        """als_model_path = save_als_model(
    final_als_model,
    MODEL_DIR / "als_model.npz",
)
print("Сохранено:", als_model_path)""",
    )
    cells += step(
        "Сохранение ALS mappings",
        "записываем исходные user/item ID",
        "factor indices без mappings не имеют смысла",
        "два JSON-файла",
        """save_json(
    final_interactions.index_to_user,
    MODEL_DIR / "mappings" / "als_user_ids.json",
)
save_json(
    final_interactions.index_to_item,
    MODEL_DIR / "mappings" / "als_article_ids.json",
)
print("ALS mappings сохранены")""",
    )
    cells += step(
        "ALS test report",
        "добавляем model name и техническую статистику к test metrics",
        "notebook 10 загрузит один компактный CSV",
        "`als_metrics.csv`",
        """als_test_report = {
    "model": "ALS",
    **test_metrics,
    "users_evaluated": len(test_ground_truth_sample),
    "average_candidates": test_candidates.groupby("customer_id").size().mean(),
    "training_time": test_training_time,
    "notes": f"ALS Top-{CANDIDATE_LIMIT} candidates",
}
als_metrics_path = REPORT_DIR / "als_metrics.csv"
pd.DataFrame([als_test_report]).to_csv(als_metrics_path, index=False)
display(pd.Series(als_test_report))""",
    )
    return cells


def content_repeated_split_cells(prefix: str, label: str) -> list[dict]:
    title = label.capitalize()
    return (
        step(
            f"{title}: границы",
            f"выбираем {label}-окно",
            "validation уже показал расчёт весов и одного профиля",
            f"`{prefix}_cutoff` и `{prefix}_end`",
            f'''{prefix}_cutoff = pd.Timestamp(windows["{prefix}"]["cutoff_date"])
{prefix}_end = pd.Timestamp(windows["{prefix}"]["target_end_date"])
print("{title}:", {prefix}_cutoff.date(), "—", {prefix}_end.date())''',
        )
        + step(
            f"{title}: history и target",
            "разделяем прошлое и будущую неделю",
            "content profiles используют только history",
            f"`{prefix}_history` и `{prefix}_target`",
            f'''{prefix}_history = transactions[
    transactions["t_dat"] < {prefix}_cutoff
].copy()
{prefix}_target = transactions[
    transactions["t_dat"].between({prefix}_cutoff, {prefix}_end)
].copy()

assert {prefix}_history["t_dat"].max() < {prefix}_cutoff
print("History:", {prefix}_history.shape, "Target:", {prefix}_target.shape)''',
        )
        + step(
            f"{title}: ground truth",
            "повторяем уже показанную подготовку ответов",
            "оценка использует полный target каждого пользователя",
            f"`{prefix}_ground_truth`",
            f'''{prefix}_ground_truth = build_ground_truth(
    {prefix}_target,
    set({prefix}_history["customer_id"]),
)
print("Ground-truth users:", len({prefix}_ground_truth))''',
        )
        + step(
            f"{title}: cohort",
            "выбираем пользователей с тем же seed, что в ALS",
            "candidate sources должны покрывать одинаковый cohort",
            f"`{prefix}_users` и sample ground truth",
            f'''{prefix}_users, {prefix}_ground_truth_sample = sample_ground_truth(
    {prefix}_ground_truth,
    MAX_EVALUATION_USERS,
    RANDOM_STATE,
)
print("Evaluation users:", len({prefix}_users))''',
        )
        + step(
            f"{title}: profile matrix",
            "повторяем показанные weights и sparse aggregation",
            "техническая функция не выполняет cosine или Top-K",
            f"`{prefix}_profiles`",
            f'''{prefix}_profiles = build_user_profiles(
    {prefix}_history,
    content_artifacts,
    reference_date={prefix}_cutoff,
    decay_days=DECAY_DAYS,
)
print("Profiles:", {prefix}_profiles.matrix.shape)''',
        )
        + step(
            f"{title}: seen items",
            "собираем покупки, которые нельзя рекомендовать повторно",
            "filtering отделён от profile building",
            f"`{prefix}_seen_items`",
            f'''{prefix}_seen_items = seen_items_by_user({prefix}_history)
print("Users with seen items:", len({prefix}_seen_items))''',
        )
        + step(
            f"{title}: массовые кандидаты",
            "считаем user-to-items cosine batched-функцией",
            "один пользователь уже был полностью разобран вручную",
            f"`{prefix}_candidates`",
            f'''{prefix}_candidates = generate_content_candidates(
    {prefix}_profiles,
    content_artifacts,
    customer_ids={prefix}_users,
    seen_items={prefix}_seen_items,
    limit=CANDIDATE_LIMIT,
)
print("Candidate rows:", len({prefix}_candidates))
display({prefix}_candidates.head())''',
        )
        + step(
            f"{title}: метрики",
            "считаем Candidate Recall и standalone Top-12",
            "метрики не смешаны с inference или сохранением",
            f"`{prefix}_metrics`",
            f'''{prefix}_candidate_lists = {prefix}_candidates.groupby(
    "customer_id", sort=False
)["article_id"].apply(list).to_dict()
{prefix}_top12_lists = {prefix}_candidates[
    {prefix}_candidates["content_rank"] <= 12
].groupby("customer_id", sort=False)["article_id"].apply(list).to_dict()
{prefix}_metrics = {{
    "Candidate Recall": candidate_recall_at_k(
        {prefix}_ground_truth_sample, {prefix}_candidate_lists, CANDIDATE_LIMIT
    ),
    "Recall@12": mean_recall_at_k({prefix}_ground_truth_sample, {prefix}_top12_lists, 12),
    "MAP@12": map_at_k({prefix}_ground_truth_sample, {prefix}_top12_lists, 12),
    "HitRate@12": hit_rate_at_k({prefix}_ground_truth_sample, {prefix}_top12_lists, 12),
}}
display(pd.Series({prefix}_metrics))''',
        )
        + step(
            f"{title}: сохранение кандидатов",
            "записываем готовую candidate table",
            "notebook 07 не будет строить profiles повторно",
            f"`content_candidates_{prefix}.parquet`",
            f'''{prefix}_candidates_path = PROCESSED_DIR / "content_candidates_{prefix}.parquet"
{prefix}_candidates.to_parquet({prefix}_candidates_path, index=False)
print("Сохранено:", {prefix}_candidates_path)''',
        )
    )


def notebook_06() -> list[dict]:
    cells = setup(
        "06. Content-Based шаг за шагом",
        "Сначала вручную строим weighted sparse profile одного validation-пользователя. Только после этого используем технические функции для массового inference и других окон.",
    )
    cells += step(
        "Импорты",
        "подключаем sparse operations, OneHotEncoder и cosine similarity",
        "полная item-item matrix не нужна",
        "библиотеки и технические inference-функции",
        """import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from IPython.display import display

from fashion_recommender.content_based import (
    ContentArtifacts, build_user_profiles, generate_content_candidates,
    seen_items_by_user,
)
from fashion_recommender.data import load_articles, load_transactions
from fashion_recommender.evaluation import (
    candidate_recall_at_k, hit_rate_at_k, map_at_k, mean_recall_at_k,
)
from fashion_recommender.persistence import load_json, save_content_artifacts, save_json""",
    )
    cells += step(
        "Пути и параметры",
        "задаём decay, candidate limit и reproducible cohort",
        "те же параметры используются во всех окнах",
        "пути и константы",
        raw_paths()
        + "\n\nWINDOWS_PATH = PROCESSED_DIR / \"temporal_windows.json\"\n"
        + check_file("WINDOWS_PATH", "03_temporal_validation_colab.ipynb")
        + "\n\nDECAY_DAYS = 30.0\nCANDIDATE_LIMIT = 50"
        + "\nMAX_EVALUATION_USERS = 2_000\nRANDOM_STATE = 42",
    )
    cells += step(
        "Признаки товаров",
        "явно задаём шесть объяснимых категорий",
        "они определяют пространство content vectors",
        "`ITEM_FEATURE_COLUMNS`",
        """ITEM_FEATURE_COLUMNS = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "department_name",
    "section_name",
    "garment_group_name",
]
print(ITEM_FEATURE_COLUMNS)""",
    )
    cells += step(
        "Загрузка входов",
        "читаем transactions, articles и windows",
        "item metadata статичны для всех temporal windows",
        "три входных объекта",
        """transactions = load_transactions(TRANSACTIONS_PATH)
articles = load_articles(ARTICLES_PATH)
windows = load_json(WINDOWS_PATH)
print("Transactions:", transactions.shape)
print("Articles:", articles.shape)""",
    )
    cells += step(
        "Заполнение категорий",
        "выбираем нужные столбцы и заменяем пропуски",
        "OneHotEncoder должен получать строки без NaN",
        "`article_features`",
        """article_features = articles[
    ["article_id", *ITEM_FEATURE_COLUMNS]
].drop_duplicates("article_id").copy()
article_features[ITEM_FEATURE_COLUMNS] = (
    article_features[ITEM_FEATURE_COLUMNS]
    .fillna("Unknown")
    .astype(str)
)
display(article_features.head())""",
    )
    cells += step(
        "Создание encoder",
        "задаём OneHotEncoder до fit",
        "неизвестная категория не должна ломать inference",
        "`encoder`",
        """encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True,
)
print(encoder)""",
    )
    cells += step(
        "Sparse item matrix",
        "вызываем `fit_transform` на категориях",
        "матрица остаётся sparse",
        "`article_matrix`",
        """article_matrix = encoder.fit_transform(
    article_features[ITEM_FEATURE_COLUMNS]
)
article_matrix = article_matrix.tocsr().astype("float32")
print("Type:", type(article_matrix))""",
    )
    cells += step(
        "Размер item matrix",
        "смотрим shape, nnz и имена первых one-hot признаков",
        "число строк должно совпадать с числом товаров",
        "проверку encoder output",
        """encoded_feature_names = encoder.get_feature_names_out(ITEM_FEATURE_COLUMNS)
print("Shape:", article_matrix.shape)
print("NNZ:", article_matrix.nnz)
print("Первые признаки:", encoded_feature_names[:10])
assert article_matrix.shape[0] == len(article_features)""",
    )
    cells += step(
        "Article mapping",
        "связываем article ID со строкой matrix",
        "после cosine numeric index нужно вернуть в article ID",
        "`article_to_index` и `index_to_article`",
        """index_to_article = article_features["article_id"].astype(str).tolist()
article_to_index = {
    article_id: index
    for index, article_id in enumerate(index_to_article)
}
print("Mapping example:", next(iter(article_to_index.items())))""",
    )
    cells += step(
        "Content artifacts в памяти",
        "упаковываем encoder, matrix и mapping",
        "технические функции массового inference принимают один согласованный объект",
        "`content_artifacts`",
        """content_artifacts = ContentArtifacts(
    encoder=encoder,
    article_feature_matrix=article_matrix,
    article_to_index=article_to_index,
    index_to_article=index_to_article,
    feature_columns=ITEM_FEATURE_COLUMNS,
)""",
    )
    cells += step(
        "Validation-границы",
        "выбираем одно окно для подробного профиля",
        "weights должны измеряться относительно validation cutoff",
        "две даты",
        """validation_cutoff = pd.Timestamp(windows["validation"]["cutoff_date"])
validation_end = pd.Timestamp(windows["validation"]["target_end_date"])
print("Validation:", validation_cutoff.date(), "—", validation_end.date())""",
    )
    cells += step(
        "Validation history и target",
        "делим транзакции по cutoff",
        "target не участвует в профилях",
        "две таблицы",
        """validation_history = transactions[
    transactions["t_dat"] < validation_cutoff
].copy()
validation_target = transactions[
    transactions["t_dat"].between(validation_cutoff, validation_end)
].copy()

assert validation_history["t_dat"].max() < validation_cutoff
print("History:", validation_history.shape, "Target:", validation_target.shape)""",
    )
    cells += step(
        "Validation ground truth",
        "оставляем известных users и собираем уникальные future items",
        "content и ALS должны сравниваться на одном protocol",
        "полный `validation_ground_truth`",
        """validation_known_users = set(validation_history["customer_id"])
validation_target_evaluation = validation_target[
    validation_target["customer_id"].isin(validation_known_users)
]
validation_target_unique = (
    validation_target_evaluation
    .sort_values("t_dat")
    .drop_duplicates(["customer_id", "article_id"])
)
validation_ground_truth = validation_target_unique.groupby(
    "customer_id", sort=False
)["article_id"].apply(list).to_dict()
print("Ground-truth users:", len(validation_ground_truth))""",
    )
    cells += step(
        "Validation cohort",
        "выбираем ID случайно с тем же seed",
        "candidate files ALS и Content должны иметь один cohort",
        "sample ground truth",
        """validation_all_users = np.array(sorted(validation_ground_truth))
validation_sample_size = min(MAX_EVALUATION_USERS, len(validation_all_users))
validation_rng = np.random.default_rng(RANDOM_STATE)
validation_users = validation_rng.choice(
    validation_all_users,
    size=validation_sample_size,
    replace=False,
).tolist()
validation_ground_truth_sample = {
    customer_id: validation_ground_truth[customer_id]
    for customer_id in validation_users
}
print("Evaluation users:", len(validation_users))""",
    )
    cells += step(
        "Purchase count",
        "считаем частоту каждой user-item пары",
        "frequency weight использует число повторов",
        "`validation_purchase_counts`",
        """validation_purchase_counts = (
    validation_history
    .groupby(["customer_id", "article_id"])
    .size()
    .reset_index(name="purchase_count")
)
display(validation_purchase_counts.head())""",
    )
    cells += step(
        "Последняя покупка пары",
        "отдельно находим максимальную дату user-item",
        "recency зависит от последнего события",
        "`validation_last_purchases`",
        """validation_last_purchases = (
    validation_history
    .groupby(["customer_id", "article_id"], as_index=False)["t_dat"]
    .max()
    .rename(columns={"t_dat": "last_purchase"})
)
display(validation_last_purchases.head())""",
    )
    cells += step(
        "User-item history",
        "объединяем count и last date",
        "два сигнала остаются видимыми отдельными столбцами",
        "`validation_user_item_history`",
        """validation_user_item_history = validation_purchase_counts.merge(
    validation_last_purchases,
    on=["customer_id", "article_id"],
    how="inner",
)
validation_user_item_history = validation_user_item_history[
    validation_user_item_history["article_id"].isin(article_to_index)
].copy()
display(validation_user_item_history.head())""",
    )
    cells += step(
        "Days since purchase",
        "вычитаем last purchase из cutoff",
        "future dates здесь недопустимы",
        "неотрицательный `days_since_purchase`",
        """validation_user_item_history["days_since_purchase"] = (
    validation_cutoff
    - validation_user_item_history["last_purchase"]
).dt.days
assert validation_user_item_history["days_since_purchase"].min() >= 0
display(validation_user_item_history.head())""",
    )
    cells += step(
        "Recency weight",
        "экспоненциально уменьшаем вклад старой покупки",
        "недавние товары сильнее влияют на профиль",
        "столбец `recency_weight`",
        """validation_user_item_history["recency_weight"] = np.exp(
    -validation_user_item_history["days_since_purchase"] / DECAY_DAYS
)
display(validation_user_item_history[
    ["days_since_purchase", "recency_weight"]
].head())""",
    )
    cells += step(
        "Frequency weight",
        "логарифмически усиливаем повторные покупки",
        "один очень частый товар не должен полностью доминировать",
        "столбец `frequency_weight`",
        """validation_user_item_history["frequency_weight"] = (
    1 + np.log1p(validation_user_item_history["purchase_count"])
)
display(validation_user_item_history[
    ["purchase_count", "frequency_weight"]
].head())""",
    )
    cells += step(
        "Итоговый вес",
        "перемножаем recency и frequency",
        "профиль учитывает оба объяснимых сигнала",
        "`purchase_weight`",
        """validation_user_item_history["purchase_weight"] = (
    validation_user_item_history["recency_weight"]
    * validation_user_item_history["frequency_weight"]
)
display(validation_user_item_history[
    ["purchase_count", "days_since_purchase", "purchase_weight"]
].sample(5, random_state=RANDOM_STATE))""",
    )
    cells += step(
        "Один подходящий пользователь",
        "выбираем evaluation user с известными article rows",
        "профиль сначала разбирается на одном примере",
        "`example_customer`",
        """eligible_profile_users = set(validation_user_item_history["customer_id"])
example_customer = None
for customer_id in validation_users:
    if customer_id in eligible_profile_users:
        example_customer = customer_id
        break

if example_customer is None:
    raise ValueError("Не найден пользователь для примера профиля")
print("Example customer:", example_customer)""",
    )
    cells += step(
        "Покупки пользователя",
        "фильтруем weighted user-item history",
        "видим товары, даты, counts и weights",
        "`example_purchases`",
        """example_purchases = validation_user_item_history[
    validation_user_item_history["customer_id"] == example_customer
].copy()
example_purchases["article_index"] = example_purchases["article_id"].map(
    article_to_index
)
display(example_purchases)""",
    )
    cells += step(
        "Строки item matrix",
        "выбираем sparse vectors купленных товаров",
        "матрица не превращается в dense",
        "`example_item_rows`",
        """example_indices = example_purchases["article_index"].to_numpy()
example_item_rows = article_matrix[example_indices]
print("Item rows shape:", example_item_rows.shape)
print("Item rows nnz:", example_item_rows.nnz)""",
    )
    cells += step(
        "Веса примера",
        "берём purchase weights в том же порядке",
        "каждый item vector получает свой коэффициент",
        "sparse row с весами",
        """example_weights = example_purchases["purchase_weight"].to_numpy(
    dtype="float32"
)
example_weight_row = csr_matrix(example_weights.reshape(1, -1))
print("Weights:", example_weights)
print("Weight sum:", example_weights.sum())""",
    )
    cells += step(
        "Взвешенная сумма",
        "умножаем weights на item rows",
        "получаем сумму признаков без dense conversion",
        "`example_weighted_sum`",
        """example_weighted_sum = example_weight_row @ example_item_rows
print("Weighted sum shape:", example_weighted_sum.shape)
print("Weighted sum nnz:", example_weighted_sum.nnz)""",
    )
    cells += step(
        "User profile",
        "делим сумму признаков на сумму весов",
        "профиль становится взвешенным средним",
        "sparse `example_profile`",
        """example_profile = example_weighted_sum / example_weights.sum()
example_profile = example_profile.tocsr()
print("Profile shape:", example_profile.shape)
print("Profile nnz:", example_profile.nnz)""",
    )
    cells += step(
        "Cosine similarity",
        "сравниваем один profile со всеми item rows",
        "не строим квадратную item-item matrix",
        "один similarity vector",
        """example_similarities = cosine_similarity(
    example_profile,
    article_matrix,
).ravel()
print("Similarities shape:", example_similarities.shape)
print("Max similarity:", example_similarities.max())""",
    )
    cells += step(
        "Исключение seen items",
        "понижаем score уже купленных товаров",
        "кандидаты должны предлагать новые позиции",
        "filtered similarity vector",
        """example_seen_items = set(example_purchases["article_id"])
example_seen_indices = [
    article_to_index[article_id]
    for article_id in example_seen_items
]
example_similarities[example_seen_indices] = -np.inf
print("Исключено товаров:", len(example_seen_indices))""",
    )
    cells += step(
        "Top-10 похожих товаров",
        "сортируем similarity и возвращаем article ID",
        "завершаем ручной путь одного пользователя",
        "таблицу Top-10",
        """example_top_indices = np.argsort(example_similarities)[-10:][::-1]
example_top_articles = [
    index_to_article[int(item_index)]
    for item_index in example_top_indices
]
example_top10 = pd.DataFrame({
    "article_id": example_top_articles,
    "cosine_similarity": example_similarities[example_top_indices],
})
display(example_top10)""",
    )
    cells += step(
        "Все validation profiles",
        "повторяем показанные weights и sparse aggregation для всех users",
        "функция не выполняет Top-K",
        "`validation_profiles`",
        """validation_profiles = build_user_profiles(
    validation_history,
    content_artifacts,
    reference_date=validation_cutoff,
    decay_days=DECAY_DAYS,
)
print("Profiles shape:", validation_profiles.matrix.shape)""",
    )
    cells += step(
        "Validation seen items",
        "собираем историю покупок каждого user",
        "mass inference исключит эти article ID",
        "`validation_seen_items`",
        """validation_seen_items = seen_items_by_user(validation_history)
print("Users with seen items:", len(validation_seen_items))""",
    )
    cells += step(
        "Mass validation candidates",
        "вызываем batched user-to-item cosine",
        "ручной алгоритм одного user уже показан",
        "`validation_candidates`",
        """validation_candidates = generate_content_candidates(
    validation_profiles,
    content_artifacts,
    customer_ids=validation_users,
    seen_items=validation_seen_items,
    limit=CANDIDATE_LIMIT,
)
print("Candidate rows:", len(validation_candidates))
display(validation_candidates.head())""",
    )
    cells += step(
        "Validation Candidate Recall",
        "оцениваем покрытие Top-50",
        "это верхняя граница для последующего ranking",
        "Candidate Recall",
        """validation_candidate_lists = validation_candidates.groupby(
    "customer_id", sort=False
)["article_id"].apply(list).to_dict()
validation_candidate_recall = candidate_recall_at_k(
    validation_ground_truth_sample,
    validation_candidate_lists,
    CANDIDATE_LIMIT,
)
print(f"Candidate Recall@{CANDIDATE_LIMIT}:", validation_candidate_recall)""",
    )
    cells += step(
        "Validation Top-12",
        "оцениваем standalone Content-Based",
        "позиционные метрики рассчитаны отдельно от сохранения",
        "Recall/MAP/HitRate",
        """validation_top12_lists = validation_candidates[
    validation_candidates["content_rank"] <= 12
].groupby("customer_id", sort=False)["article_id"].apply(list).to_dict()

validation_metrics = {
    "Recall@12": mean_recall_at_k(validation_ground_truth_sample, validation_top12_lists, 12),
    "MAP@12": map_at_k(validation_ground_truth_sample, validation_top12_lists, 12),
    "HitRate@12": hit_rate_at_k(validation_ground_truth_sample, validation_top12_lists, 12),
}
display(pd.Series(validation_metrics))""",
    )
    cells += step(
        "Сохранение validation candidates",
        "записываем отдельный Parquet",
        "notebook 07 загрузит готовый результат",
        "`content_candidates_validation.parquet`",
        """validation_candidates_path = PROCESSED_DIR / "content_candidates_validation.parquet"
validation_candidates.to_parquet(validation_candidates_path, index=False)
print("Сохранено:", validation_candidates_path)""",
    )
    cells += step(
        "Ground truth helper",
        "фиксируем уже разобранную pandas-логику",
        "функция не строит profile и не считает cosine",
        "`build_ground_truth`",
        """def build_ground_truth(target, known_users):
    target_evaluation = target[target["customer_id"].isin(known_users)]
    target_unique = (
        target_evaluation
        .sort_values("t_dat")
        .drop_duplicates(["customer_id", "article_id"])
    )
    return target_unique.groupby(
        "customer_id", sort=False
    )["article_id"].apply(list).to_dict()""",
    )
    cells += step(
        "Sampling helper",
        "повторяем deterministic cohort",
        "полный ground truth остаётся отдельным объектом",
        "users и sample dictionary",
        """def sample_ground_truth(ground_truth, max_users, random_state):
    all_users = np.array(sorted(ground_truth))
    sample_size = min(max_users, len(all_users))
    rng = np.random.default_rng(random_state)
    users = rng.choice(all_users, size=sample_size, replace=False).tolist()
    ground_truth_sample = {
        customer_id: ground_truth[customer_id]
        for customer_id in users
    }
    return users, ground_truth_sample""",
    )
    cells += content_repeated_split_cells("train", "train")
    cells += content_repeated_split_cells("test", "test")
    cells += step(
        "Test alias",
        "сохраняем совместимое имя test candidates",
        "старые consumers не требуют изменения пути",
        "`content_candidates.parquet`",
        """content_alias_path = PROCESSED_DIR / "content_candidates.parquet"
test_candidates.to_parquet(content_alias_path, index=False)
print("Сохранено:", content_alias_path)""",
    )
    cells += step(
        "Сохранение encoder и matrix",
        "записываем статичные content artifacts",
        "batch inference не должен повторно fit OneHotEncoder",
        "joblib, NPZ, mapping и config",
        """content_paths = save_content_artifacts(
    encoder,
    article_matrix,
    index_to_article,
    MODEL_DIR,
    {"feature_columns": ITEM_FEATURE_COLUMNS, "decay_days": DECAY_DAYS},
)
print(content_paths)""",
    )
    cells += step(
        "Сохранение явного mapping",
        "записываем article ID → row index",
        "mapping можно проверить без загрузки matrix",
        "`article_index_mapping.json`",
        """article_mapping_path = MODEL_DIR / "mappings" / "article_index_mapping.json"
save_json(article_to_index, article_mapping_path)
print("Сохранено:", article_mapping_path)""",
    )
    cells += step(
        "Content test report",
        "добавляем model name и статистику к test metrics",
        "notebook 10 загрузит компактный CSV",
        "`content_metrics.csv`",
        """content_test_report = {
    "model": "Content-Based",
    **test_metrics,
    "users_evaluated": len(test_ground_truth_sample),
    "average_candidates": test_candidates.groupby("customer_id").size().mean(),
    "notes": f"Content-Based Top-{CANDIDATE_LIMIT} candidates",
}
content_metrics_path = REPORT_DIR / "content_metrics.csv"
pd.DataFrame([content_test_report]).to_csv(content_metrics_path, index=False)
display(pd.Series(content_test_report))""",
    )
    return cells


def candidate_repeated_split_cells(prefix: str, label: str) -> list[dict]:
    title = label.capitalize()
    return (
        step(
            f"{title}: границы и данные",
            f"выбираем {label}-history и target",
            "кандидаты должны оцениваться на своём temporal cutoff",
            f"`{prefix}_history` и `{prefix}_target`",
            f'''{prefix}_cutoff = pd.Timestamp(windows["{prefix}"]["cutoff_date"])
{prefix}_end = pd.Timestamp(windows["{prefix}"]["target_end_date"])
{prefix}_history = transactions[
    transactions["t_dat"] < {prefix}_cutoff
].copy()
{prefix}_target = transactions[
    transactions["t_dat"].between({prefix}_cutoff, {prefix}_end)
].copy()
assert {prefix}_history["t_dat"].max() < {prefix}_cutoff
print("History:", {prefix}_history.shape, "Target:", {prefix}_target.shape)''',
        )
        + step(
            f"{title}: ground truth и cohort",
            "повторяем уже показанную подготовку evaluation users",
            "все candidate sources используют одинаковые ID",
            f"`{prefix}_users` и sample ground truth",
            f'''{prefix}_ground_truth = build_ground_truth(
    {prefix}_target,
    set({prefix}_history["customer_id"]),
)
{prefix}_users, {prefix}_ground_truth_sample = sample_ground_truth(
    {prefix}_ground_truth,
    MAX_EVALUATION_USERS,
    RANDOM_STATE,
)
print("Evaluation users:", len({prefix}_users))''',
        )
        + step(
            f"{title}: ALS candidates",
            "загружаем готовый Parquet и оставляем Top-150",
            "ALS здесь не обучается",
            f"`{prefix}_als_candidates`",
            f'''{prefix}_als_path = PROCESSED_DIR / "als_candidates_{prefix}.parquet"
{prefix}_als_candidates = pd.read_parquet({prefix}_als_path)
{prefix}_als_candidates = {prefix}_als_candidates[
    {prefix}_als_candidates["als_rank"] <= ALS_LIMIT
].copy()
print("ALS rows:", len({prefix}_als_candidates))''',
        )
        + step(
            f"{title}: Content candidates",
            "загружаем готовый Content-Based Parquet",
            "OneHotEncoder и profiles здесь не создаются",
            f"`{prefix}_content_candidates`",
            f'''{prefix}_content_path = PROCESSED_DIR / "content_candidates_{prefix}.parquet"
{prefix}_content_candidates = pd.read_parquet({prefix}_content_path)
print("Content rows:", len({prefix}_content_candidates))''',
        )
        + step(
            f"{title}: Personal History",
            "повторяем уже разобранные recent и frequent операции",
            "функция не объединяет остальные candidate sources",
            f"`{prefix}_personal_candidates`",
            f'''{prefix}_personal_candidates = build_personal_candidates(
    {prefix}_history,
    {prefix}_users,
    PERSONAL_LIMIT,
)
print("Personal rows:", len({prefix}_personal_candidates))''',
        )
        + step(
            f"{title}: Popularity",
            "строим небольшой глобальный список и повторяем его для cohort",
            "операция быстрая и не использует target",
            f"`{prefix}_popularity_candidates`",
            f'''{prefix}_popular_items = popular_items(
    {prefix}_history,
    limit=POPULARITY_LIMIT,
)
{prefix}_popularity_candidates = popularity_candidates(
    {prefix}_users,
    {prefix}_popular_items,
    limit=POPULARITY_LIMIT,
)
print("Popularity rows:", len({prefix}_popularity_candidates))''',
        )
        + step(
            f"{title}: объединение источников",
            "повторяем уже показанные concat/groupby/flags технической функцией",
            "validation выше показывает реальную реализацию",
            f"`{prefix}_merged_candidates`",
            f'''{prefix}_merged_candidates = merge_candidate_sources(
    als={prefix}_als_candidates,
    content_based={prefix}_content_candidates,
    personal_history={prefix}_personal_candidates,
    popularity={prefix}_popularity_candidates,
    limit_per_user=FINAL_LIMIT,
)
print("Merged rows:", len({prefix}_merged_candidates))
print("Average candidates:", {prefix}_merged_candidates.groupby("customer_id").size().mean())''',
        )
        + step(
            f"{title}: Candidate Recall",
            "сравниваем merged candidates с полным ground truth",
            "оценка отделена от merge и сохранения",
            f"`{prefix}_candidate_recall`",
            f'''{prefix}_candidate_lists = {prefix}_merged_candidates.groupby(
    "customer_id", sort=False
)["article_id"].apply(list).to_dict()
{prefix}_candidate_recall = candidate_recall_at_k(
    {prefix}_ground_truth_sample,
    {prefix}_candidate_lists,
    FINAL_LIMIT,
)
print(f"Candidate Recall@{{FINAL_LIMIT}}:", {prefix}_candidate_recall)''',
        )
        + step(
            f"{title}: сохранение",
            "записываем одну строку на user-item pair",
            "notebook 08 загрузит candidates без генерации с нуля",
            f"`merged_candidates_{prefix}.parquet`",
            f'''{prefix}_merged_path = PROCESSED_DIR / "merged_candidates_{prefix}.parquet"
{prefix}_merged_candidates.to_parquet({prefix}_merged_path, index=False)
print("Сохранено:", {prefix}_merged_path)''',
        )
    )


def notebook_07() -> list[dict]:
    cells = setup(
        "07. Объединение кандидатов",
        "ALS и Content-Based загружаются из Parquet. Validation подробно показывает History, Popularity, flags, concat и groupby; train/test не спрятаны в цикл.",
    )
    cells += step(
        "Импорты",
        "подключаем pandas, метрики и технические helpers только для повторения",
        "модели и encoder здесь не импортируются",
        "минимальный набор для candidate merge",
        """import numpy as np
import pandas as pd
from IPython.display import display

from fashion_recommender.baselines import popular_items
from fashion_recommender.candidates import merge_candidate_sources, popularity_candidates
from fashion_recommender.data import load_transactions
from fashion_recommender.evaluation import (
    candidate_recall_at_k, hit_rate_at_k, map_at_k, mean_recall_at_k,
)
from fashion_recommender.persistence import load_json""",
    )
    cells += step(
        "Пути и лимиты",
        "задаём входные файлы и размеры источников",
        "лимиты должны быть видны до объединения",
        "пути и пять констант",
        raw_paths()
        + "\n\nWINDOWS_PATH = PROCESSED_DIR / \"temporal_windows.json\"\n"
        + check_file("WINDOWS_PATH", "03_temporal_validation_colab.ipynb")
        + "\n\nALS_LIMIT = 150\nCONTENT_LIMIT = 50\nPERSONAL_LIMIT = 20"
        + "\nPOPULARITY_LIMIT = 30\nFINAL_LIMIT = 250"
        + "\nMAX_EVALUATION_USERS = 2_000\nRANDOM_STATE = 42",
    )
    cells += step(
        "Проверка candidate artifacts",
        "проверяем шесть файлов от notebooks 05 и 06",
        "отсутствие файла не должно запускать повторное обучение",
        "понятную ошибку с producer notebook",
        """required_candidate_paths = [
    PROCESSED_DIR / "als_candidates_train.parquet",
    PROCESSED_DIR / "als_candidates_validation.parquet",
    PROCESSED_DIR / "als_candidates_test.parquet",
    PROCESSED_DIR / "content_candidates_train.parquet",
    PROCESSED_DIR / "content_candidates_validation.parquet",
    PROCESSED_DIR / "content_candidates_test.parquet",
]
missing_candidate_paths = [
    path for path in required_candidate_paths if not path.is_file()
]
if missing_candidate_paths:
    raise FileNotFoundError(
        f"Не найдены candidate artifacts: {missing_candidate_paths}. "
        "Сначала выполните notebooks 05 и 06."
    )""",
    )
    cells += step(
        "Загрузка общих входов",
        "читаем transactions и temporal windows один раз",
        "Personal History и Popularity используют history каждого окна",
        "`transactions` и `windows`",
        """transactions = load_transactions(TRANSACTIONS_PATH)
windows = load_json(WINDOWS_PATH)
print("Transactions:", transactions.shape)""",
    )
    cells += step(
        "Validation history и target",
        "выбираем одно окно для подробного merge",
        "никакая модель здесь не обучается",
        "две таблицы",
        """validation_cutoff = pd.Timestamp(windows["validation"]["cutoff_date"])
validation_end = pd.Timestamp(windows["validation"]["target_end_date"])
validation_history = transactions[
    transactions["t_dat"] < validation_cutoff
].copy()
validation_target = transactions[
    transactions["t_dat"].between(validation_cutoff, validation_end)
].copy()
assert validation_history["t_dat"].max() < validation_cutoff
print("History:", validation_history.shape, "Target:", validation_target.shape)""",
    )
    cells += step(
        "Validation ground truth",
        "оставляем известных users и уникальные future items",
        "source coverage измеряется на полном target",
        "полный словарь ответов",
        """validation_known_users = set(validation_history["customer_id"])
validation_target_evaluation = validation_target[
    validation_target["customer_id"].isin(validation_known_users)
]
validation_target_unique = (
    validation_target_evaluation
    .sort_values("t_dat")
    .drop_duplicates(["customer_id", "article_id"])
)
validation_ground_truth = validation_target_unique.groupby(
    "customer_id", sort=False
)["article_id"].apply(list).to_dict()
print("Ground-truth users:", len(validation_ground_truth))""",
    )
    cells += step(
        "Validation cohort",
        "повторяем deterministic sample",
        "загруженные ALS/Content files построены на тех же users",
        "`validation_users` и sample ground truth",
        """validation_all_users = np.array(sorted(validation_ground_truth))
validation_sample_size = min(MAX_EVALUATION_USERS, len(validation_all_users))
validation_rng = np.random.default_rng(RANDOM_STATE)
validation_users = validation_rng.choice(
    validation_all_users,
    size=validation_sample_size,
    replace=False,
).tolist()
validation_ground_truth_sample = {
    customer_id: validation_ground_truth[customer_id]
    for customer_id in validation_users
}
print("Evaluation users:", len(validation_users))""",
    )
    cells += step(
        "Загрузка ALS validation",
        "читаем готовые ALS candidates",
        "никакого `AlternatingLeastSquares.fit` здесь нет",
        "`validation_als_candidates`",
        """validation_als_path = PROCESSED_DIR / "als_candidates_validation.parquet"
validation_als_candidates = pd.read_parquet(validation_als_path)
display(validation_als_candidates.head())""",
    )
    cells += step(
        "Проверка ALS columns",
        "проверяем contract и оставляем Top-150",
        "notebook 05 сохраняет до 200 для анализа recall curve",
        "корректную ALS source table",
        """required_als_columns = {
    "customer_id", "article_id", "als_score", "als_rank"
}
assert required_als_columns <= set(validation_als_candidates.columns)
validation_als_candidates = validation_als_candidates[
    validation_als_candidates["als_rank"] <= ALS_LIMIT
].copy()
print("ALS rows:", len(validation_als_candidates))""",
    )
    cells += step(
        "Загрузка Content validation",
        "читаем готовые cosine candidates",
        "OneHotEncoder здесь не создаётся",
        "`validation_content_candidates`",
        """validation_content_path = PROCESSED_DIR / "content_candidates_validation.parquet"
validation_content_candidates = pd.read_parquet(validation_content_path)
display(validation_content_candidates.head())""",
    )
    cells += step(
        "Проверка Content columns",
        "проверяем score и rank",
        "ошибка схемы должна быть обнаружена до concat",
        "корректную Content source table",
        """required_content_columns = {
    "customer_id", "article_id", "content_similarity_score", "content_rank"
}
assert required_content_columns <= set(validation_content_candidates.columns)
assert validation_content_candidates["content_rank"].max() <= CONTENT_LIMIT
print("Content rows:", len(validation_content_candidates))""",
    )
    cells += step(
        "Recent History candidates",
        "сортируем history и удаляем повторные пары",
        "последние покупки дают один персональный источник",
        "Top-20 recent pairs",
        """validation_cohort_history = validation_history[
    validation_history["customer_id"].isin(validation_users)
].copy()
validation_recent = (
    validation_cohort_history
    .sort_values("t_dat", ascending=False)
    .drop_duplicates(["customer_id", "article_id"])
)
validation_recent["personal_history_rank"] = (
    validation_recent.groupby("customer_id").cumcount() + 1
)
validation_recent = validation_recent[
    validation_recent["personal_history_rank"] <= PERSONAL_LIMIT
].copy()
validation_recent["personal_history_score"] = 1 / validation_recent["personal_history_rank"]
display(validation_recent.head())""",
    )
    cells += step(
        "Frequent History candidates",
        "считаем count и last purchase каждой пары",
        "частота даёт второй персональный порядок",
        "Top-20 frequent pairs",
        """validation_frequent = validation_cohort_history.groupby(
    ["customer_id", "article_id"], as_index=False
).agg(
    purchase_count=("article_id", "size"),
    last_purchase=("t_dat", "max"),
)
validation_frequent = validation_frequent.sort_values(
    ["customer_id", "purchase_count", "last_purchase", "article_id"],
    ascending=[True, False, False, True],
)
validation_frequent["personal_history_rank"] = (
    validation_frequent.groupby("customer_id").cumcount() + 1
)
validation_frequent = validation_frequent[
    validation_frequent["personal_history_rank"] <= PERSONAL_LIMIT
].copy()
validation_frequent["personal_history_score"] = validation_frequent["purchase_count"]
display(validation_frequent.head())""",
    )
    cells += step(
        "Общий Personal History",
        "складываем recent и frequent pairs, затем удаляем дубли",
        "оба baseline-сигнала сохраняются в одном source",
        "одну строку на персональную пару",
        """validation_personal_parts = [
    validation_recent[
        ["customer_id", "article_id", "personal_history_score", "personal_history_rank"]
    ],
    validation_frequent[
        ["customer_id", "article_id", "personal_history_score", "personal_history_rank"]
    ],
]
validation_personal_candidates = pd.concat(
    validation_personal_parts,
    ignore_index=True,
)
validation_personal_candidates = validation_personal_candidates.groupby(
    ["customer_id", "article_id"], as_index=False
).agg(
    personal_history_score=("personal_history_score", "max"),
    personal_history_rank=("personal_history_rank", "min"),
)
display(validation_personal_candidates.head())""",
    )
    cells += step(
        "Popularity table",
        "считаем детерминированные глобальные counts и ranks",
        "source использует только validation history и стабильный article-ID tie-break",
        "Top-30 товаров",
        """validation_popular_table = popular_items(
    validation_history,
    limit=POPULARITY_LIMIT,
)
display(validation_popular_table.head())""",
    )
    cells += step(
        "Popularity candidates",
        "повторяем небольшой Top-30 для каждого evaluation user",
        "полного user × catalog произведения нет",
        "60 000 строк для cohort 2 000",
        """validation_users_table = pd.DataFrame({
    "customer_id": validation_users,
    "_key": 1,
})
validation_popularity_candidates = validation_users_table.merge(
    validation_popular_table.assign(_key=1),
    on="_key",
).drop(columns="_key")
print("Popularity rows:", len(validation_popularity_candidates))""",
    )
    cells += step(
        "Флаги источников",
        "добавляем по одному бинарному флагу к каждой таблице",
        "после concat будет видно происхождение пары",
        "четыре source frames",
        """validation_als_source = validation_als_candidates.assign(from_als=1)
validation_content_source = validation_content_candidates.assign(
    from_content_based=1
)
validation_personal_source = validation_personal_candidates.assign(
    from_personal_history=1
)
validation_popularity_source = validation_popularity_candidates.assign(
    from_popularity=1
)""",
    )
    cells += step(
        "Concat источников",
        "складываем четыре таблицы вертикально",
        "до groupby одна пара может встретиться несколько раз",
        "`validation_all_candidates`",
        """validation_all_candidates = pd.concat(
    [
        validation_als_source,
        validation_content_source,
        validation_personal_source,
        validation_popularity_source,
    ],
    ignore_index=True,
    sort=False,
)
print("Строк до groupby:", len(validation_all_candidates))""",
    )
    cells += step(
        "Одна строка на пару",
        "агрегируем scores, ranks и flags",
        "одинаковая user-item pair не должна дублироваться",
        "`validation_merged_candidates`",
        """validation_merged_candidates = validation_all_candidates.groupby(
    ["customer_id", "article_id"], as_index=False
).agg(
    als_score=("als_score", "max"),
    als_rank=("als_rank", "min"),
    content_similarity_score=("content_similarity_score", "max"),
    content_rank=("content_rank", "min"),
    personal_history_score=("personal_history_score", "max"),
    personal_history_rank=("personal_history_rank", "min"),
    popularity_score=("popularity_score", "max"),
    popularity_rank=("popularity_rank", "min"),
    from_als=("from_als", "max"),
    from_content_based=("from_content_based", "max"),
    from_personal_history=("from_personal_history", "max"),
    from_popularity=("from_popularity", "max"),
)
print("Уникальных пар:", len(validation_merged_candidates))""",
    )
    cells += step(
        "Заполнение scores",
        "заменяем отсутствующий score нулём",
        "NaN означает, что source не предложил пару",
        "четыре числовых score columns",
        """score_columns = [
    "als_score",
    "content_similarity_score",
    "personal_history_score",
    "popularity_score",
]
validation_merged_candidates[score_columns] = (
    validation_merged_candidates[score_columns].fillna(0.0)
)
display(validation_merged_candidates[score_columns].head())""",
    )
    cells += step(
        "Заполнение ranks и flags",
        "заменяем отсутствующие ranks/flags нулём и задаём типы",
        "значение 0 явно означает отсутствие source",
        "готовые rank и flag columns",
        """rank_columns = [
    "als_rank", "content_rank", "personal_history_rank", "popularity_rank",
]
flag_columns = [
    "from_als", "from_content_based", "from_personal_history", "from_popularity",
]
validation_merged_candidates[rank_columns] = (
    validation_merged_candidates[rank_columns].fillna(0).astype("int32")
)
validation_merged_candidates[flag_columns] = (
    validation_merged_candidates[flag_columns].fillna(0).astype("int8")
)""",
    )
    cells += step(
        "Число источников",
        "суммируем четыре бинарных флага",
        "пары из нескольких sources получают отдельный признак",
        "`number_of_candidate_sources`",
        """validation_merged_candidates["number_of_candidate_sources"] = (
    validation_merged_candidates[flag_columns].sum(axis=1)
)
display(validation_merged_candidates[
    ["customer_id", "article_id", "number_of_candidate_sources"]
].head())""",
    )
    cells += step(
        "Ограничение 250",
        "сортируем по source count и reciprocal ranks",
        "ranking table остаётся ограниченного размера",
        "до 250 pairs на пользователя",
        """validation_rank_priority = sum(
    np.where(validation_merged_candidates[column] > 0,
             1 / validation_merged_candidates[column], 0)
    for column in rank_columns
)
validation_merged_candidates["_rank_priority"] = validation_rank_priority
validation_merged_candidates = validation_merged_candidates.sort_values(
    ["customer_id", "number_of_candidate_sources", "_rank_priority"],
    ascending=[True, False, False],
)
validation_merged_candidates = validation_merged_candidates[
    validation_merged_candidates.groupby("customer_id").cumcount() < FINAL_LIMIT
].drop(columns="_rank_priority").reset_index(drop=True)""",
    )
    cells += step(
        "Проверки merged table",
        "смотрим дубликаты и среднее число candidates",
        "одна пара должна встречаться один раз",
        "две контрольные статистики",
        """validation_duplicate_pairs = validation_merged_candidates.duplicated(
    ["customer_id", "article_id"]
).sum()
validation_average_candidates = validation_merged_candidates.groupby(
    "customer_id"
).size().mean()
print("Duplicate pairs:", validation_duplicate_pairs)
print("Average candidates:", validation_average_candidates)
assert validation_duplicate_pairs == 0""",
    )
    cells += step(
        "Validation Candidate Recall",
        "оцениваем объединённое покрытие",
        "метрика рассчитана после merge и до ranking",
        "Candidate Recall@250",
        """validation_candidate_lists = validation_merged_candidates.groupby(
    "customer_id", sort=False
)["article_id"].apply(list).to_dict()
validation_candidate_recall = candidate_recall_at_k(
    validation_ground_truth_sample,
    validation_candidate_lists,
    FINAL_LIMIT,
)
print(f"Candidate Recall@{FINAL_LIMIT}:", validation_candidate_recall)""",
    )
    cells += step(
        "Вклад validation sources",
        "считаем coverage каждого source отдельно",
        "видим, какой генератор добавляет future items",
        "таблицу из четырёх строк",
        """validation_source_analysis = pd.DataFrame([
    {
        "source": "ALS",
        "pairs": len(validation_als_candidates),
        "candidate_recall": candidate_recall_at_k(
            validation_ground_truth_sample,
            validation_als_candidates.groupby("customer_id")["article_id"].apply(list).to_dict(),
            ALS_LIMIT,
        ),
    },
    {
        "source": "Content-Based",
        "pairs": len(validation_content_candidates),
        "candidate_recall": candidate_recall_at_k(
            validation_ground_truth_sample,
            validation_content_candidates.groupby("customer_id")["article_id"].apply(list).to_dict(),
            CONTENT_LIMIT,
        ),
    },
])
display(validation_source_analysis)""",
    )
    cells += step(
        "Дополнение source analysis",
        "добавляем Personal History и Popularity",
        "эти источники рассчитывались прямо в notebook",
        "полную таблицу из четырёх строк",
        """validation_source_analysis.loc[len(validation_source_analysis)] = {
    "source": "Personal History",
    "pairs": len(validation_personal_candidates),
    "candidate_recall": candidate_recall_at_k(
        validation_ground_truth_sample,
        validation_personal_candidates.groupby("customer_id")["article_id"].apply(list).to_dict(),
        PERSONAL_LIMIT,
    ),
}
validation_source_analysis.loc[len(validation_source_analysis)] = {
    "source": "Popularity",
    "pairs": len(validation_popularity_candidates),
    "candidate_recall": candidate_recall_at_k(
        validation_ground_truth_sample,
        validation_popularity_candidates.groupby("customer_id")["article_id"].apply(list).to_dict(),
        POPULARITY_LIMIT,
    ),
}
display(validation_source_analysis)""",
    )
    cells += step(
        "Сохранение validation merge",
        "записываем готовые candidates",
        "notebook 08 не повторяет candidate generation",
        "`merged_candidates_validation.parquet`",
        """validation_merged_path = PROCESSED_DIR / "merged_candidates_validation.parquet"
validation_merged_candidates.to_parquet(validation_merged_path, index=False)
print("Сохранено:", validation_merged_path)""",
    )
    cells += step(
        "Ground truth helper",
        "фиксируем уже показанную подготовку ответов",
        "функция не создаёт candidates",
        "`build_ground_truth`",
        """def build_ground_truth(target, known_users):
    target_evaluation = target[target["customer_id"].isin(known_users)]
    target_unique = (
        target_evaluation
        .sort_values("t_dat")
        .drop_duplicates(["customer_id", "article_id"])
    )
    return target_unique.groupby(
        "customer_id", sort=False
    )["article_id"].apply(list).to_dict()""",
    )
    cells += step(
        "Sampling helper",
        "фиксируем deterministic cohort",
        "полный ground truth остаётся отдельным",
        "users и sample dictionary",
        """def sample_ground_truth(ground_truth, max_users, random_state):
    all_users = np.array(sorted(ground_truth))
    sample_size = min(max_users, len(all_users))
    rng = np.random.default_rng(random_state)
    users = rng.choice(all_users, size=sample_size, replace=False).tolist()
    sample = {customer_id: ground_truth[customer_id] for customer_id in users}
    return users, sample""",
    )
    cells += step(
        "Personal History helper",
        "фиксируем уже показанные recent и frequent шаги",
        "функция не объединяет ALS, Content или Popularity",
        "одну personal source table",
        """def build_personal_candidates(history, customer_ids, limit):
    cohort = history[history["customer_id"].isin(customer_ids)].copy()
    recent = cohort.sort_values("t_dat", ascending=False).drop_duplicates(
        ["customer_id", "article_id"]
    )
    recent["personal_history_rank"] = recent.groupby("customer_id").cumcount() + 1
    recent = recent[recent["personal_history_rank"] <= limit].copy()
    recent["personal_history_score"] = 1 / recent["personal_history_rank"]
    frequent = cohort.groupby(["customer_id", "article_id"], as_index=False).agg(
        purchase_count=("article_id", "size"), last_purchase=("t_dat", "max")
    )
    frequent = frequent.sort_values(
        ["customer_id", "purchase_count", "last_purchase", "article_id"],
        ascending=[True, False, False, True],
    )
    frequent["personal_history_rank"] = frequent.groupby("customer_id").cumcount() + 1
    frequent = frequent[frequent["personal_history_rank"] <= limit].copy()
    frequent["personal_history_score"] = frequent["purchase_count"]
    columns = ["customer_id", "article_id", "personal_history_score", "personal_history_rank"]
    combined = pd.concat([recent[columns], frequent[columns]], ignore_index=True)
    return combined.groupby(["customer_id", "article_id"], as_index=False).agg(
        personal_history_score=("personal_history_score", "max"),
        personal_history_rank=("personal_history_rank", "min"),
    )""",
    )
    cells += candidate_repeated_split_cells("train", "train")
    cells += candidate_repeated_split_cells("test", "test")
    cells += step(
        "Test alias",
        "сохраняем совместимое имя test merge",
        "следующие consumers могут использовать короткий путь",
        "`merged_candidates.parquet`",
        """merged_alias_path = PROCESSED_DIR / "merged_candidates.parquet"
test_merged_candidates.to_parquet(merged_alias_path, index=False)
print("Сохранено:", merged_alias_path)""",
    )
    cells += step(
        "Test source analysis",
        "сохраняем coverage фактического final test",
        "model comparison должен ссылаться на test, не validation",
        "`candidate_source_analysis.csv`",
        """test_source_analysis = pd.DataFrame([
    {"source": "ALS", "candidate_pairs": len(test_als_candidates),
     "candidate_recall": candidate_recall_at_k(
         test_ground_truth_sample,
         test_als_candidates.groupby("customer_id")["article_id"].apply(list).to_dict(), ALS_LIMIT)},
    {"source": "Content-Based", "candidate_pairs": len(test_content_candidates),
     "candidate_recall": candidate_recall_at_k(
         test_ground_truth_sample,
         test_content_candidates.groupby("customer_id")["article_id"].apply(list).to_dict(), CONTENT_LIMIT)},
    {"source": "Personal History", "candidate_pairs": len(test_personal_candidates),
     "candidate_recall": candidate_recall_at_k(
         test_ground_truth_sample,
         test_personal_candidates.groupby("customer_id")["article_id"].apply(list).to_dict(), PERSONAL_LIMIT)},
    {"source": "Popularity", "candidate_pairs": len(test_popularity_candidates),
     "candidate_recall": candidate_recall_at_k(
         test_ground_truth_sample,
         test_popularity_candidates.groupby("customer_id")["article_id"].apply(list).to_dict(), POPULARITY_LIMIT)},
])
test_source_analysis.to_csv(REPORT_DIR / "candidate_source_analysis.csv", index=False)
display(test_source_analysis)""",
    )
    cells += step(
        "Нормализация hybrid scores",
        "масштабируем ALS, History и Popularity внутри пользователя",
        "scores разных источников имеют разные диапазоны",
        "три normalized columns",
        """test_hybrid = test_merged_candidates.copy()

als_min = test_hybrid.groupby("customer_id")["als_score"].transform("min")
als_max = test_hybrid.groupby("customer_id")["als_score"].transform("max")
test_hybrid["normalized_als_score"] = (
    (test_hybrid["als_score"] - als_min) / (als_max - als_min).replace(0, np.nan)
).fillna(0)

history_min = test_hybrid.groupby("customer_id")["personal_history_score"].transform("min")
history_max = test_hybrid.groupby("customer_id")["personal_history_score"].transform("max")
test_hybrid["normalized_history_score"] = (
    (test_hybrid["personal_history_score"] - history_min)
    / (history_max - history_min).replace(0, np.nan)
).fillna(0)""",
    )
    cells += step(
        "Popularity normalization",
        "масштабируем последний ненормированный source",
        "после этого фиксированная формула читается напрямую",
        "`normalized_popularity_score`",
        """popularity_min = test_hybrid.groupby("customer_id")["popularity_score"].transform("min")
popularity_max = test_hybrid.groupby("customer_id")["popularity_score"].transform("max")
test_hybrid["normalized_popularity_score"] = (
    (test_hybrid["popularity_score"] - popularity_min)
    / (popularity_max - popularity_min).replace(0, np.nan)
).fillna(0)""",
    )
    cells += step(
        "Simple Hybrid score",
        "применяем фиксированные веса 0.5/0.3/0.1/0.1",
        "это понятный baseline перед CatBoost",
        "`hybrid_score`",
        """test_hybrid["hybrid_score"] = (
    0.5 * test_hybrid["normalized_als_score"]
    + 0.3 * test_hybrid["content_similarity_score"]
    + 0.1 * test_hybrid["normalized_history_score"]
    + 0.1 * test_hybrid["normalized_popularity_score"]
)
display(test_hybrid[["customer_id", "article_id", "hybrid_score"]].head())""",
    )
    cells += step(
        "Simple Hybrid Top-12",
        "сортируем score внутри пользователя",
        "получаем standalone рекомендации без CatBoost",
        "`hybrid_recommendations`",
        """hybrid_top12 = test_hybrid.sort_values(
    ["customer_id", "hybrid_score", "article_id"],
    ascending=[True, False, True],
).groupby("customer_id", sort=False).head(12)
hybrid_recommendations = hybrid_top12.groupby(
    "customer_id", sort=False
)["article_id"].apply(list).to_dict()
display(hybrid_top12.head())""",
    )
    cells += step(
        "Simple Hybrid metrics",
        "оцениваем test Top-12 и сохраняем CSV",
        "notebook 10 сравнит baseline с CatBoost",
        "`simple_hybrid_metrics.csv`",
        """hybrid_metrics = {
    "model": "Simple Hybrid",
    "Recall@12": mean_recall_at_k(test_ground_truth_sample, hybrid_recommendations, 12),
    "MAP@12": map_at_k(test_ground_truth_sample, hybrid_recommendations, 12),
    "HitRate@12": hit_rate_at_k(test_ground_truth_sample, hybrid_recommendations, 12),
    "Candidate Recall": test_candidate_recall,
    "users_evaluated": len(test_ground_truth_sample),
    "average_candidates": test_merged_candidates.groupby("customer_id").size().mean(),
    "notes": "Fixed 0.5/0.3/0.1/0.1 weights",
}
pd.DataFrame([hybrid_metrics]).to_csv(
    REPORT_DIR / "simple_hybrid_metrics.csv", index=False
)
display(pd.Series(hybrid_metrics))""",
    )
    return cells


def feature_repeated_split_cells(prefix: str, label: str) -> list[dict]:
    title = label.capitalize()
    return (
        step(
            f"{title}: history, future и candidates",
            f"загружаем входы {label}-окна",
            "готовые candidates приходят из notebook 07",
            f"три `{prefix}_...` таблицы",
            f'''{prefix}_cutoff = pd.Timestamp(windows["{prefix}"]["cutoff_date"])
{prefix}_end = pd.Timestamp(windows["{prefix}"]["target_end_date"])
{prefix}_history = transactions[
    transactions["t_dat"] < {prefix}_cutoff
].copy()
{prefix}_future = transactions[
    transactions["t_dat"].between({prefix}_cutoff, {prefix}_end)
].copy()
{prefix}_candidates = pd.read_parquet(
    PROCESSED_DIR / "merged_candidates_{prefix}.parquet"
)
assert {prefix}_history["t_dat"].max() < {prefix}_cutoff
print("History:", {prefix}_history.shape)
print("Future:", {prefix}_future.shape)
print("Candidates:", {prefix}_candidates.shape)''',
        )
        + step(
            f"{title}: user features",
            "повторяем уже показанные user aggregations",
            "функция возвращает только признаки пользователя",
            f"`{prefix}_user_features`",
            f'''{prefix}_user_features = build_user_features(
    {prefix}_history,
    {prefix}_cutoff,
    customers,
)
print("User features:", {prefix}_user_features.shape)''',
        )
        + step(
            f"{title}: item features",
            "повторяем item counts, recency и popularity windows",
            "future не участвует в агрегатах",
            f"`{prefix}_item_features`",
            f'''{prefix}_item_features = build_item_features(
    {prefix}_history,
    {prefix}_cutoff,
)
print("Item features:", {prefix}_item_features.shape)''',
        )
        + step(
            f"{title}: user-item features",
            "повторяем count и last purchase пары",
            "эти признаки описывают конкретного кандидата",
            f"`{prefix}_pair_features`",
            f'''{prefix}_pair_features = build_user_item_features(
    {prefix}_history,
    {prefix}_cutoff,
)
print("Pair features:", {prefix}_pair_features.shape)''',
        )
        + step(
            f"{title}: category affinity",
            "считаем четыре уже объяснённых category counts",
            "один пример product type был показан вручную",
            f"`{prefix}_affinity_features`",
            f'''{prefix}_affinity_features = build_affinity_features(
    {prefix}_history,
    article_attributes,
    {prefix}_candidates,
)
print("Affinity features:", {prefix}_affinity_features.shape)''',
        )
        + step(
            f"{title}: user и item merge",
            "присоединяем первые два блока к candidate pairs",
            "каждая строка по-прежнему является одной user-item pair",
            f"начало `{prefix}_ranking_table`",
            f'''{prefix}_ranking_table = {prefix}_candidates.merge(
    {prefix}_user_features,
    on="customer_id",
    how="left",
)
{prefix}_ranking_table = {prefix}_ranking_table.merge(
    {prefix}_item_features,
    on="article_id",
    how="left",
)
print({prefix}_ranking_table.shape)''',
        )
        + step(
            f"{title}: pair и category merge",
            "добавляем article attributes, pair history и affinity",
            "target пока не присоединяется",
            f"полную feature table без label",
            f'''{prefix}_ranking_table = {prefix}_ranking_table.merge(
    article_attributes,
    on="article_id",
    how="left",
)
{prefix}_ranking_table = {prefix}_ranking_table.merge(
    {prefix}_pair_features,
    on=["customer_id", "article_id"],
    how="left",
)
{prefix}_ranking_table = {prefix}_ranking_table.merge(
    {prefix}_affinity_features,
    on=["customer_id", "article_id"],
    how="left",
)
print({prefix}_ranking_table.shape)''',
        )
        + step(
            f"{title}: positive pairs",
            "создаём target=1 только из соответствующей future-недели",
            "label не влияет на features",
            f"`{prefix}_positive_pairs`",
            f'''{prefix}_positive_pairs = (
    {prefix}_future[["customer_id", "article_id"]]
    .drop_duplicates()
    .assign(target=1)
)
print("Future positive pairs:", len({prefix}_positive_pairs))''',
        )
        + step(
            f"{title}: target merge",
            "left-join positives к generated candidates",
            "непокупка generated pair получает target=0",
            f"labelled `{prefix}_ranking_table`",
            f'''{prefix}_ranking_table = {prefix}_ranking_table.merge(
    {prefix}_positive_pairs,
    on=["customer_id", "article_id"],
    how="left",
)
{prefix}_ranking_table["target"] = (
    {prefix}_ranking_table["target"].fillna(0).astype("int8")
)
print({prefix}_ranking_table["target"].value_counts())''',
        )
        + step(
            f"{title}: пропуски и проверки",
            "заполняем categories и numeric NaN, затем проверяем пары",
            "CatBoost table не должна содержать missing или duplicates",
            f"готовую `{prefix}_ranking_table`",
            f'''{prefix}_ranking_table[ITEM_FEATURE_COLUMNS] = (
    {prefix}_ranking_table[ITEM_FEATURE_COLUMNS].fillna("Unknown").astype(str)
)
{prefix}_numeric_columns = {prefix}_ranking_table.select_dtypes(
    include="number"
).columns
{prefix}_ranking_table[{prefix}_numeric_columns] = (
    {prefix}_ranking_table[{prefix}_numeric_columns].fillna(0)
)
assert not {prefix}_ranking_table.duplicated(
    ["customer_id", "article_id"]
).any()
assert not {prefix}_ranking_table.isna().any().any()
print("Positive share:", {prefix}_ranking_table["target"].mean())''',
        )
        + step(
            f"{title}: сохранение ranking table",
            "записываем готовые features и target",
            "notebook 09 только загрузит эти таблицы",
            f"`{prefix}_ranking_table.parquet`",
            f'''{prefix}_ranking_path = PROCESSED_DIR / "{prefix}_ranking_table.parquet"
{prefix}_ranking_table.to_parquet({prefix}_ranking_path, index=False)
print("Сохранено:", {prefix}_ranking_path)''',
        )
    )


def notebook_08() -> list[dict]:
    cells = setup(
        "08. Признаки и target",
        "Validation подробно показывает user, item, user-item и category features. Train/test повторяют эти уже объяснённые блоки без цикла и без обучения моделей.",
    )
    cells += step(
        "Импорты",
        "подключаем pandas, загрузчики и Candidate Recall",
        "в feature notebook нет ALS, encoder или CatBoost",
        "минимальный набор для агрегаций",
        """import numpy as np
import pandas as pd
from IPython.display import display

from fashion_recommender.data import load_articles, load_customers, load_transactions
from fashion_recommender.evaluation import candidate_recall_at_k
from fashion_recommender.persistence import load_json""",
    )
    cells += step(
        "Пути",
        "задаём общие каталоги и проверяем temporal windows",
        "настройка путей отделена от проверки candidate files",
        "базовые пути",
        raw_paths()
        + "\n\nWINDOWS_PATH = PROCESSED_DIR / \"temporal_windows.json\"\n"
        + check_file("WINDOWS_PATH", "03_temporal_validation_colab.ipynb"),
    )
    cells += step(
        "Проверка merged candidates",
        "проверяем три Parquet от notebook 07",
        "feature engineering не должен запускать candidate generation",
        "явные artifact contracts",
        """CANDIDATE_PATHS = {
    split: PROCESSED_DIR / f"merged_candidates_{split}.parquet"
    for split in ["train", "validation", "test"]
}
missing_paths = [
    path for path in CANDIDATE_PATHS.values() if not path.is_file()
]
if missing_paths:
    raise FileNotFoundError(
        f"Не найдены merged candidates: {missing_paths}. "
        "Сначала выполните notebook 07_candidate_generation_colab.ipynb."
    )""",
    )
    cells += step(
        "Категории",
        "задаём item attributes и четыре affinity categories",
        "названия должны совпасть с notebook 09",
        "два понятных списка",
        """ITEM_FEATURE_COLUMNS = [
    "product_type_name", "product_group_name", "colour_group_name",
    "department_name", "section_name", "garment_group_name",
]
AFFINITY_CATEGORIES = [
    "product_type_name",
    "colour_group_name",
    "section_name",
    "garment_group_name",
]
print("Item categories:", ITEM_FEATURE_COLUMNS)""",
    )
    cells += step(
        "Загрузка данных",
        "читаем исходные таблицы и windows один раз",
        "каждый split дальше использует свой cutoff",
        "четыре входных объекта",
        """transactions = load_transactions(TRANSACTIONS_PATH)
articles = load_articles(ARTICLES_PATH)
customers = load_customers(CUSTOMERS_PATH)
windows = load_json(WINDOWS_PATH)
print("Transactions:", transactions.shape)
print("Articles:", articles.shape)
print("Customers:", customers.shape)""",
    )
    cells += step(
        "Article attributes",
        "готовим только ID и шесть категорий",
        "эти столбцы присоединяются к candidate pair",
        "`article_attributes`",
        """article_attributes = articles[
    ["article_id", *ITEM_FEATURE_COLUMNS]
].drop_duplicates("article_id").copy()
article_attributes[ITEM_FEATURE_COLUMNS] = (
    article_attributes[ITEM_FEATURE_COLUMNS]
    .fillna("Unknown")
    .astype(str)
)
display(article_attributes.head())""",
    )
    cells += step(
        "Validation inputs",
        "выбираем validation history/future и загружаем candidates",
        "на одном окне подробно строятся все признаки",
        "три validation tables",
        """validation_cutoff = pd.Timestamp(windows["validation"]["cutoff_date"])
validation_end = pd.Timestamp(windows["validation"]["target_end_date"])
validation_history = transactions[
    transactions["t_dat"] < validation_cutoff
].copy()
validation_future = transactions[
    transactions["t_dat"].between(validation_cutoff, validation_end)
].copy()
validation_candidates = pd.read_parquet(
    CANDIDATE_PATHS["validation"]
)
assert validation_history["t_dat"].max() < validation_cutoff
print("History:", validation_history.shape)
print("Future:", validation_future.shape)
print("Candidates:", validation_candidates.shape)""",
    )
    cells += step(
        "User count features",
        "считаем покупки, unique items и active days",
        "это базовые признаки активности клиента",
        "`validation_user_features`",
        """validation_user_features = validation_history.groupby(
    "customer_id", as_index=False
).agg(
    user_total_purchases=("article_id", "size"),
    user_unique_items=("article_id", "nunique"),
    user_active_days=("t_dat", "nunique"),
)
display(validation_user_features.head())""",
    )
    cells += step(
        "Последняя покупка user",
        "находим max date отдельно",
        "из неё рассчитывается user recency",
        "`validation_user_last_purchase`",
        """validation_user_last_purchase = validation_history.groupby(
    "customer_id", as_index=False
)["t_dat"].max().rename(columns={"t_dat": "user_last_purchase"})
display(validation_user_last_purchase.head())""",
    )
    cells += step(
        "User recency",
        "присоединяем дату и считаем дни до cutoff",
        "future не участвует в признаке",
        "`user_days_since_last_purchase`",
        """validation_user_features = validation_user_features.merge(
    validation_user_last_purchase,
    on="customer_id",
    how="left",
)
validation_user_features["user_days_since_last_purchase"] = (
    validation_cutoff - validation_user_features.pop("user_last_purchase")
).dt.days
display(validation_user_features.head())""",
    )
    cells += step(
        "Средняя цена user",
        "считаем mean price покупок",
        "признак описывает привычный ценовой уровень",
        "`user_average_price`",
        """validation_user_price = validation_history.groupby(
    "customer_id", as_index=False
)["price"].mean().rename(columns={"price": "user_average_price"})
validation_user_features = validation_user_features.merge(
    validation_user_price,
    on="customer_id",
    how="left",
)
display(validation_user_features.head())""",
    )
    cells += step(
        "Online share user",
        "помечаем канал 2 и считаем среднее",
        "получаем долю online-покупок",
        "`user_online_share`",
        """validation_online_share = (
    validation_history
    .assign(_online=validation_history["sales_channel_id"].eq(2).astype(float))
    .groupby("customer_id", as_index=False)["_online"]
    .mean()
    .rename(columns={"_online": "user_online_share"})
)
validation_user_features = validation_user_features.merge(
    validation_online_share,
    on="customer_id",
    how="left",
)""",
    )
    cells += step(
        "Возраст user",
        "присоединяем age из customer profile",
        "статичный профиль не агрегируется из future",
        "`user_age` в user features",
        """validation_age = customers[
    ["customer_id", "age"]
].drop_duplicates("customer_id").rename(columns={"age": "user_age"})
validation_user_features = validation_user_features.merge(
    validation_age,
    on="customer_id",
    how="left",
)
display(validation_user_features.head())""",
    )
    cells += step(
        "Item count features",
        "считаем покупки и уникальных покупателей товара",
        "это базовая item popularity",
        "`validation_item_features`",
        """validation_item_features = validation_history.groupby(
    "article_id", as_index=False
).agg(
    item_total_purchases=("customer_id", "size"),
    item_unique_customers=("customer_id", "nunique"),
)
display(validation_item_features.head())""",
    )
    cells += step(
        "Item price",
        "считаем среднюю историческую цену",
        "товары разных ценовых уровней различаются",
        "`item_average_price`",
        """validation_item_price = validation_history.groupby(
    "article_id", as_index=False
)["price"].mean().rename(columns={"price": "item_average_price"})
validation_item_features = validation_item_features.merge(
    validation_item_price,
    on="article_id",
    how="left",
)""",
    )
    cells += step(
        "Последняя покупка item",
        "находим max date каждого товара",
        "item recency рассчитывается относительно cutoff",
        "`item_days_since_last_purchase`",
        """validation_item_last = validation_history.groupby(
    "article_id", as_index=False
)["t_dat"].max().rename(columns={"t_dat": "item_last_purchase"})
validation_item_features = validation_item_features.merge(
    validation_item_last,
    on="article_id",
    how="left",
)
validation_item_features["item_days_since_last_purchase"] = (
    validation_cutoff - validation_item_features.pop("item_last_purchase")
).dt.days""",
    )
    cells += step(
        "Популярность item за 7 дней",
        "считаем покупки в коротком history-окне",
        "признак отражает свежий спрос",
        "`item_popularity_7d`",
        """validation_recent_7d = validation_history[
    validation_history["t_dat"] >= validation_cutoff - pd.Timedelta(days=7)
]
validation_popularity_7d = validation_recent_7d.groupby(
    "article_id", as_index=False
).size().rename(columns={"size": "item_popularity_7d"})
validation_item_features = validation_item_features.merge(
    validation_popularity_7d,
    on="article_id",
    how="left",
)""",
    )
    cells += step(
        "Популярность item за 30 дней",
        "считаем более устойчивый месячный спрос",
        "7d и 30d дают разные временные масштабы",
        "`item_popularity_30d`",
        """validation_recent_30d = validation_history[
    validation_history["t_dat"] >= validation_cutoff - pd.Timedelta(days=30)
]
validation_popularity_30d = validation_recent_30d.groupby(
    "article_id", as_index=False
).size().rename(columns={"size": "item_popularity_30d"})
validation_item_features = validation_item_features.merge(
    validation_popularity_30d,
    on="article_id",
    how="left",
)
display(validation_item_features.head())""",
    )
    cells += step(
        "User-item count",
        "считаем число прошлых покупок каждой пары",
        "так модель узнаёт повторную покупку",
        "`validation_pair_features`",
        """validation_pair_features = validation_history.groupby(
    ["customer_id", "article_id"], as_index=False
).size().rename(columns={"size": "user_item_purchase_count"})
validation_pair_features["user_bought_item_before"] = 1
display(validation_pair_features.head())""",
    )
    cells += step(
        "User-item last purchase",
        "находим последнюю дату пары",
        "pair recency отличается от общей user recency",
        "`days_since_user_bought_item`",
        """validation_pair_last = validation_history.groupby(
    ["customer_id", "article_id"], as_index=False
)["t_dat"].max().rename(columns={"t_dat": "user_item_last_purchase"})
validation_pair_features = validation_pair_features.merge(
    validation_pair_last,
    on=["customer_id", "article_id"],
    how="left",
)
validation_pair_features["days_since_user_bought_item"] = (
    validation_cutoff - validation_pair_features.pop("user_item_last_purchase")
).dt.days
display(validation_pair_features.head())""",
    )
    cells += step(
        "History с категориями",
        "добавляем item attributes к прошлым покупкам",
        "affinity считается только из history",
        "`validation_history_categories`",
        """validation_history_categories = validation_history[
    ["customer_id", "article_id"]
].merge(
    article_attributes,
    on="article_id",
    how="left",
)
display(validation_history_categories.head())""",
    )
    cells += step(
        "Пример Product Type affinity",
        "считаем покупки user в каждом product type",
        "этот пример объясняет остальные category counts",
        "`validation_product_affinity`",
        """validation_product_affinity = validation_history_categories.groupby(
    ["customer_id", "product_type_name"], as_index=False
).size().rename(columns={"size": "user_product_type_count"})
display(validation_product_affinity.head())""",
    )
    cells += step(
        "Остальные affinity",
        "повторяем тот же groupby для colour, section и garment",
        "логика уже показана на product type",
        "три небольшие таблицы",
        """validation_colour_affinity = validation_history_categories.groupby(
    ["customer_id", "colour_group_name"], as_index=False
).size().rename(columns={"size": "user_colour_count"})
validation_section_affinity = validation_history_categories.groupby(
    ["customer_id", "section_name"], as_index=False
).size().rename(columns={"size": "user_section_count"})
validation_garment_affinity = validation_history_categories.groupby(
    ["customer_id", "garment_group_name"], as_index=False
).size().rename(columns={"size": "user_garment_group_count"})
display(validation_colour_affinity.head())""",
    )
    cells += step(
        "Merge user и item features",
        "присоединяем два агрегатных блока к candidates",
        "число candidate pairs не должно измениться",
        "начало `validation_ranking_table`",
        """validation_ranking_table = validation_candidates.merge(
    validation_user_features,
    on="customer_id",
    how="left",
)
validation_ranking_table = validation_ranking_table.merge(
    validation_item_features,
    on="article_id",
    how="left",
)
print("Ranking rows:", len(validation_ranking_table))""",
    )
    cells += step(
        "Merge attributes и pair features",
        "добавляем категории товара и историю пары",
        "каждый merge имеет понятный ключ",
        "расширенную feature table",
        """validation_ranking_table = validation_ranking_table.merge(
    article_attributes,
    on="article_id",
    how="left",
)
validation_ranking_table = validation_ranking_table.merge(
    validation_pair_features,
    on=["customer_id", "article_id"],
    how="left",
)
print("Columns:", len(validation_ranking_table.columns))""",
    )
    cells += step(
        "Merge affinity features",
        "присоединяем четыре category counts по соответствующему значению",
        "affinity относится к конкретному candidate item",
        "четыре новых признака",
        """validation_ranking_table = validation_ranking_table.merge(
    validation_product_affinity,
    on=["customer_id", "product_type_name"],
    how="left",
).merge(
    validation_colour_affinity,
    on=["customer_id", "colour_group_name"],
    how="left",
).merge(
    validation_section_affinity,
    on=["customer_id", "section_name"],
    how="left",
).merge(
    validation_garment_affinity,
    on=["customer_id", "garment_group_name"],
    how="left",
)
display(validation_ranking_table.head())""",
    )
    cells += step(
        "Positive future pairs",
        "создаём отдельную таблицу target=1",
        "future используется только на этапе label",
        "`validation_positive_pairs`",
        """validation_positive_pairs = (
    validation_future[["customer_id", "article_id"]]
    .drop_duplicates()
    .assign(target=1)
)
print("Future positive pairs:", len(validation_positive_pairs))""",
    )
    cells += step(
        "Target merge",
        "left-join positives к generated pairs",
        "товары вне candidates не создают новые строки",
        "столбец `target`",
        """validation_ranking_table = validation_ranking_table.merge(
    validation_positive_pairs,
    on=["customer_id", "article_id"],
    how="left",
)
validation_ranking_table["target"] = (
    validation_ranking_table["target"].fillna(0).astype("int8")
)
print(validation_ranking_table["target"].value_counts())
print("Positive share:", validation_ranking_table["target"].mean())""",
    )
    cells += step(
        "Заполнение пропусков",
        "categories заменяем Unknown, numeric NaN — нулём",
        "CatBoost table должна иметь полную схему",
        "таблицу без missing values",
        """validation_ranking_table[ITEM_FEATURE_COLUMNS] = (
    validation_ranking_table[ITEM_FEATURE_COLUMNS].fillna("Unknown").astype(str)
)
validation_numeric_columns = validation_ranking_table.select_dtypes(
    include="number"
).columns
validation_ranking_table[validation_numeric_columns] = (
    validation_ranking_table[validation_numeric_columns].fillna(0)
)
print("Missing cells:", validation_ranking_table.isna().sum().sum())""",
    )
    cells += step(
        "Проверки validation table",
        "проверяем duplicates, infinity и temporal boundary",
        "ошибка должна появиться до сохранения",
        "leakage-safe table",
        """validation_duplicates = validation_ranking_table.duplicated(
    ["customer_id", "article_id"]
).sum()
validation_numeric_values = validation_ranking_table.select_dtypes(
    include="number"
).astype("float64").to_numpy()

assert validation_duplicates == 0
assert not np.isinf(validation_numeric_values).any()
assert not validation_ranking_table.isna().any().any()
assert validation_history["t_dat"].max() < validation_cutoff
print("Duplicate pairs:", validation_duplicates)""",
    )
    cells += step(
        "Validation Candidate Recall",
        "сравниваем candidates с полным future ground truth",
        "target share и coverage отвечают на разные вопросы",
        "Candidate Recall@250",
        """validation_users = set(validation_ranking_table["customer_id"])
validation_future_evaluation = validation_future[
    validation_future["customer_id"].isin(validation_users)
]
validation_ground_truth = validation_future_evaluation.sort_values(
    "t_dat"
).drop_duplicates(
    ["customer_id", "article_id"]
).groupby("customer_id", sort=False)["article_id"].apply(list).to_dict()
validation_candidate_lists = validation_ranking_table.groupby(
    "customer_id", sort=False
)["article_id"].apply(list).to_dict()
validation_candidate_recall = candidate_recall_at_k(
    validation_ground_truth, validation_candidate_lists, 250
)
print("Candidate Recall@250:", validation_candidate_recall)""",
    )
    cells += step(
        "Сохранение validation table",
        "записываем features и target",
        "notebook 09 загрузит файл напрямую",
        "`validation_ranking_table.parquet`",
        """validation_ranking_path = PROCESSED_DIR / "validation_ranking_table.parquet"
validation_ranking_table.to_parquet(validation_ranking_path, index=False)
print("Сохранено:", validation_ranking_path)""",
    )
    cells += step(
        "User feature helper",
        "фиксируем уже показанные user aggregations",
        "функция не добавляет item, pair или target",
        "`build_user_features`",
        """def build_user_features(history, reference_date, customers):
    result = history.groupby("customer_id", as_index=False).agg(
        user_total_purchases=("article_id", "size"),
        user_unique_items=("article_id", "nunique"),
        user_active_days=("t_dat", "nunique"),
        user_last_purchase=("t_dat", "max"),
        user_average_price=("price", "mean"),
    )
    result["user_days_since_last_purchase"] = (
        reference_date - result.pop("user_last_purchase")
    ).dt.days
    online = (
        history.assign(_online=history["sales_channel_id"].eq(2).astype(float))
        .groupby("customer_id", as_index=False)["_online"].mean()
        .rename(columns={"_online": "user_online_share"})
    )
    ages = customers[["customer_id", "age"]].drop_duplicates("customer_id")
    ages = ages.rename(columns={"age": "user_age"})
    return result.merge(online, on="customer_id", how="left").merge(
        ages, on="customer_id", how="left"
    )""",
    )
    cells += step(
        "Item feature helper",
        "фиксируем уже показанные item aggregations",
        "функция использует только history до reference date",
        "`build_item_features`",
        """def build_item_features(history, reference_date):
    result = history.groupby("article_id", as_index=False).agg(
        item_total_purchases=("customer_id", "size"),
        item_unique_customers=("customer_id", "nunique"),
        item_average_price=("price", "mean"),
        item_last_purchase=("t_dat", "max"),
    )
    result["item_days_since_last_purchase"] = (
        reference_date - result.pop("item_last_purchase")
    ).dt.days
    recent_7d = history[history["t_dat"] >= reference_date - pd.Timedelta(days=7)]
    counts_7d = recent_7d.groupby("article_id", as_index=False).size()
    counts_7d = counts_7d.rename(columns={"size": "item_popularity_7d"})
    recent_30d = history[history["t_dat"] >= reference_date - pd.Timedelta(days=30)]
    counts_30d = recent_30d.groupby("article_id", as_index=False).size()
    counts_30d = counts_30d.rename(columns={"size": "item_popularity_30d"})
    return result.merge(counts_7d, on="article_id", how="left").merge(
        counts_30d, on="article_id", how="left"
    )""",
    )
    cells += step(
        "Pair feature helper",
        "фиксируем count и recency одной пары",
        "функция возвращает только user-item признаки",
        "`build_user_item_features`",
        """def build_user_item_features(history, reference_date):
    result = history.groupby(
        ["customer_id", "article_id"], as_index=False
    ).agg(
        user_item_purchase_count=("article_id", "size"),
        user_item_last_purchase=("t_dat", "max"),
    )
    result["days_since_user_bought_item"] = (
        reference_date - result.pop("user_item_last_purchase")
    ).dt.days
    result["user_bought_item_before"] = 1
    return result""",
    )
    cells += step(
        "Affinity helper",
        "повторяем product type пример для четырёх categories",
        "функция не выполняет остальные feature merges",
        "одну affinity table",
        """def build_affinity_features(history, article_attributes, candidates):
    history_categories = history[["customer_id", "article_id"]].merge(
        article_attributes,
        on="article_id",
        how="left",
    )
    output_names = {
        "product_type_name": "user_product_type_count",
        "colour_group_name": "user_colour_count",
        "section_name": "user_section_count",
        "garment_group_name": "user_garment_group_count",
    }
    result = candidates[["customer_id", "article_id"]].merge(
        article_attributes,
        on="article_id",
        how="left",
    )
    for category, output_name in output_names.items():
        counts = history_categories.groupby(
            ["customer_id", category], as_index=False
        ).size().rename(columns={"size": output_name})
        result = result.merge(counts, on=["customer_id", category], how="left")
    output_columns = ["customer_id", "article_id", *output_names.values()]
    return result[output_columns]""",
    )
    cells += feature_repeated_split_cells("train", "train")
    cells += feature_repeated_split_cells("test", "test")
    return cells


def notebook_09() -> list[dict]:
    cells = setup(
        "09. CatBoostClassifier и Top-12",
        "Notebook загружает три готовые ranking tables. Здесь нет ALS, Content-Based, candidate generation или повторного feature engineering.",
    )
    cells += step(
        "Импорты",
        "подключаем CatBoost, метрики и persistence",
        "ranking model является единственной обучаемой моделью notebook",
        "необходимые библиотеки",
        """import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from IPython.display import display

from fashion_recommender.baselines import popular_items as rank_popular_items
from fashion_recommender.data import load_transactions
from fashion_recommender.evaluation import (
    candidate_recall_at_k, hit_rate_at_k, map_at_k, mean_recall_at_k,
)
from fashion_recommender.persistence import (
    load_json, save_catboost_model, save_json, save_recommendations,
)""",
    )
    cells += step(
        "Пути и параметры",
        "задаём три ranking files и CatBoost settings",
        "все входы должны существовать до запуска fit",
        "пути и reproducible параметры",
        raw_paths()
        + "\n\nWINDOWS_PATH = PROCESSED_DIR / \"temporal_windows.json\"\n"
        + "TRAIN_PATH = PROCESSED_DIR / \"train_ranking_table.parquet\"\n"
        + "VALIDATION_PATH = PROCESSED_DIR / \"validation_ranking_table.parquet\"\n"
        + "TEST_PATH = PROCESSED_DIR / \"test_ranking_table.parquet\"\n\n"
        + "RANDOM_STATE = 42\nNEGATIVE_RATIO = 10\nITERATIONS = 500"
        + "\nDEPTH = 6\nLEARNING_RATE = 0.05\nEARLY_STOPPING_ROUNDS = 50",
    )
    cells += step(
        "Проверка ranking artifacts",
        "проверяем JSON окон и три Parquet",
        "отсутствие файла не должно запускать предыдущие этапы",
        "понятную ошибку с notebook 08",
        """required_paths = [
    WINDOWS_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
]
missing_paths = [path for path in required_paths if not path.is_file()]
if missing_paths:
    raise FileNotFoundError(
        f"Не найдены ranking artifacts: {missing_paths}. "
        "Сначала выполните notebook 08_feature_engineering_colab.ipynb."
    )""",
    )
    cells += step(
        "Загрузка train",
        "читаем готовую train ranking table",
        "кандидаты, признаки и target уже рассчитаны",
        "`train_table`",
        """train_table = pd.read_parquet(TRAIN_PATH)
print("Train:", train_table.shape)
display(train_table.head())""",
    )
    cells += step(
        "Загрузка validation",
        "читаем validation ranking table",
        "она используется только для early stopping",
        "`validation_table`",
        """validation_table = pd.read_parquet(VALIDATION_PATH)
print("Validation:", validation_table.shape)
display(validation_table.head())""",
    )
    cells += step(
        "Загрузка test",
        "читаем final test ranking table",
        "test не участвует в fit",
        "`test_table`",
        """test_table = pd.read_parquet(TEST_PATH)
print("Test:", test_table.shape)
display(test_table.head())""",
    )
    cells += step(
        "Проверка target",
        "смотрим число positives и долю класса в каждом split",
        "пустой класс сделал бы обучение некорректным",
        "таблицу class balance",
        """target_summary = pd.DataFrame({
    "split": ["train", "validation", "test"],
    "rows": [len(train_table), len(validation_table), len(test_table)],
    "positives": [
        train_table["target"].sum(),
        validation_table["target"].sum(),
        test_table["target"].sum(),
    ],
    "positive_share": [
        train_table["target"].mean(),
        validation_table["target"].mean(),
        test_table["target"].mean(),
    ],
})
display(target_summary)
assert train_table["target"].nunique() == 2""",
    )
    cells += step(
        "Числовые feature columns",
        "явно перечисляем user, item, pair и generator signals",
        "ID и target не должны попасть в X",
        "33 numeric feature names",
        """NUMERIC_FEATURES = [
    "user_total_purchases", "user_unique_items", "user_active_days",
    "user_days_since_last_purchase", "user_average_price", "user_online_share",
    "user_age", "item_total_purchases", "item_unique_customers",
    "item_popularity_7d", "item_popularity_30d", "item_average_price",
    "item_days_since_last_purchase", "user_bought_item_before",
    "user_item_purchase_count", "days_since_user_bought_item",
    "user_product_type_count", "user_colour_count", "user_section_count",
    "user_garment_group_count", "als_score", "als_rank",
    "content_similarity_score", "content_rank", "personal_history_score",
    "personal_history_rank", "popularity_score", "popularity_rank",
    "from_als", "from_content_based", "from_personal_history",
    "from_popularity", "number_of_candidate_sources",
]""",
    )
    cells += step(
        "Категориальные features",
        "отдельно перечисляем шесть item categories",
        "CatBoost получает их строками без ручного one-hot",
        "`CATEGORICAL_FEATURES` и полный список",
        """CATEGORICAL_FEATURES = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "department_name",
    "section_name",
    "garment_group_name",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
print("Всего features:", len(FEATURE_COLUMNS))""",
    )
    cells += step(
        "Проверка feature contract",
        "сравниваем ожидаемые столбцы со всеми tables",
        "ошибка schema должна появиться до sampling",
        "подтверждённый contract",
        """for table_name, table in [
    ("train", train_table),
    ("validation", validation_table),
    ("test", test_table),
]:
    missing_features = set(FEATURE_COLUMNS) - set(table.columns)
    if missing_features:
        raise ValueError(
            f"В {table_name} отсутствуют features: {sorted(missing_features)}"
        )
print("Feature contract: OK")""",
    )
    cells += step(
        "Положительные train pairs",
        "выбираем все строки target=1",
        "ни один positive не должен потеряться при sampling",
        "`positive_train`",
        """positive_train = train_table[
    train_table["target"] == 1
].copy()
print("Positive train rows:", len(positive_train))""",
    )
    cells += step(
        "Отрицательные train pairs",
        "отдельно выбираем target=0",
        "negative sampling применяется только к train",
        "`negative_train`",
        """negative_train = train_table[
    train_table["target"] == 0
].copy()
print("Negative train rows:", len(negative_train))""",
    )
    cells += step(
        "Negative sampling",
        "берём не больше десяти negatives на positive",
        "фиксированный seed делает выбор воспроизводимым",
        "`sampled_negatives`",
        """number_of_negatives = min(
    len(negative_train),
    len(positive_train) * NEGATIVE_RATIO,
)
sampled_negatives = negative_train.sample(
    n=number_of_negatives,
    random_state=RANDOM_STATE,
)
print("Sampled negatives:", len(sampled_negatives))""",
    )
    cells += step(
        "Итоговый train sample",
        "объединяем positives и sampled negatives, затем перемешиваем",
        "validation и test остаются полными",
        "`sampled_train`",
        """sampled_train = pd.concat(
    [positive_train, sampled_negatives],
    ignore_index=True,
)
sampled_train = sampled_train.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)
print("Sampled train:", sampled_train.shape)
print("Positive share:", sampled_train["target"].mean())""",
    )
    cells += step(
        "X_train",
        "выбираем только feature columns",
        "customer/article ID не являются model features",
        "`X_train`",
        """X_train = sampled_train[FEATURE_COLUMNS].copy()
print("X_train:", X_train.shape)
display(X_train.head())""",
    )
    cells += step(
        "y_train",
        "выбираем target отдельно",
        "явное разделение X/y упрощает проверку",
        "`y_train`",
        """y_train = sampled_train["target"].copy()
print("y_train:", y_train.shape)
print(y_train.value_counts())""",
    )
    cells += step(
        "Validation X/y",
        "готовим полный validation split",
        "он используется для AUC и early stopping",
        "`X_validation` и `y_validation`",
        """X_validation = validation_table[FEATURE_COLUMNS].copy()
y_validation = validation_table["target"].copy()
print("X_validation:", X_validation.shape)
print("y_validation:", y_validation.shape)""",
    )
    cells += step(
        "Test X/y",
        "готовим final test, не передавая его в fit",
        "y_test нужен только для диагностики таблицы",
        "`X_test` и `y_test`",
        """X_test = test_table[FEATURE_COLUMNS].copy()
y_test = test_table["target"].copy()
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)""",
    )
    cells += step(
        "Тип категорий",
        "приводим шесть category columns к строкам",
        "CatBoost должен видеть одинаковый тип во всех splits",
        "три согласованных X tables",
        """for column in CATEGORICAL_FEATURES:
    X_train[column] = X_train[column].astype(str)
    X_validation[column] = X_validation[column].astype(str)
    X_test[column] = X_test[column].astype(str)
print(X_train[CATEGORICAL_FEATURES].dtypes)""",
    )
    cells += step(
        "Создание CatBoostClassifier",
        "задаём модель и гиперпараметры",
        "создание объекта отделено от обучения",
        "необученный `model`",
        """model = CatBoostClassifier(
    iterations=ITERATIONS,
    depth=DEPTH,
    learning_rate=LEARNING_RATE,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=50,
    allow_writing_files=False,
)
print(model)""",
    )
    cells += step(
        "Обучение CatBoost",
        "вызываем `.fit()` с train и validation",
        "test не участвует ни в fit, ни в early stopping",
        "обученную модель и training time",
        """training_started = time.perf_counter()
model.fit(
    X_train,
    y_train,
    cat_features=CATEGORICAL_FEATURES,
    eval_set=(X_validation, y_validation),
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)
training_time = time.perf_counter() - training_started
print("Trees:", model.tree_count_)
print("Training seconds:", round(training_time, 2))""",
    )
    cells += step(
        "История обучения",
        "читаем AUC по итерациям",
        "видим работу early stopping без нового fit",
        "последние значения validation AUC",
        """evaluation_history = model.get_evals_result()
validation_auc = evaluation_history["validation"]["AUC"]
history_table = pd.DataFrame({
    "iteration": np.arange(1, len(validation_auc) + 1),
    "validation_auc": validation_auc,
})
display(history_table.tail(10))""",
    )
    cells += step(
        "Predict probability",
        "получаем вероятность positive class для test candidates",
        "probability становится ranking score",
        "столбец `prediction`",
        """inference_started = time.perf_counter()
test_predictions = model.predict_proba(X_test)[:, 1]
inference_time = time.perf_counter() - inference_started

scored_test = test_table.copy()
scored_test["prediction"] = test_predictions
display(scored_test[["customer_id", "article_id", "prediction", "target"]].head())""",
    )
    cells += step(
        "Сортировка candidates",
        "сортируем probability отдельно внутри пользователя",
        "высокий score должен иметь меньший rank",
        "`ranked_test`",
        """ranked_test = scored_test.sort_values(
    ["customer_id", "prediction", "article_id"],
    ascending=[True, False, True],
).drop_duplicates(["customer_id", "article_id"])
ranked_test["model_rank"] = ranked_test.groupby(
    "customer_id"
).cumcount() + 1
display(ranked_test.head())""",
    )
    cells += step(
        "Первые 12 candidates",
        "оставляем model_rank не больше 12",
        "fallback добавляется только в следующем шаге",
        "`top12_scored`",
        """top12_scored = ranked_test[
    ranked_test["model_rank"] <= 12
].copy()
print("Top candidate rows:", len(top12_scored))
print(top12_scored.groupby("customer_id").size().describe())""",
    )
    cells += step(
        "Test history и future",
        "загружаем raw transactions только для ground truth и fallback",
        "features и candidates здесь не пересчитываются",
        "две temporal tables",
        """transactions = load_transactions(TRANSACTIONS_PATH)
windows = load_json(WINDOWS_PATH)
test_cutoff = pd.Timestamp(windows["test"]["cutoff_date"])
test_end = pd.Timestamp(windows["test"]["target_end_date"])
test_history = transactions[
    transactions["t_dat"] < test_cutoff
].copy()
test_future = transactions[
    transactions["t_dat"].between(test_cutoff, test_end)
].copy()
assert test_history["t_dat"].max() < test_cutoff""",
    )
    cells += step(
        "Popularity fallback",
        "считаем детерминированный Top-100 только по test history",
        "короткие списки дополняются без target information и unstable ties",
        "`popular_items`",
        """popular_items = rank_popular_items(
    test_history,
    limit=100,
)["article_id"].tolist()
print("Первые fallback items:", popular_items[:12])""",
    )
    cells += step(
        "Scored lists",
        "собираем article/score каждого user из Top-12",
        "порядок уже задан model_rank",
        "`scored_items_by_user`",
        """scored_items_by_user = {}
for customer_id, group in top12_scored.groupby("customer_id", sort=False):
    scored_items_by_user[customer_id] = list(
        zip(group["article_id"], group["prediction"])
    )
print("Users with scored lists:", len(scored_items_by_user))""",
    )
    cells += step(
        "Заполнение Top-12",
        "дополняем каждый список популярными товарами без повторов",
        "получаем ровно 12 рекомендаций для каждого candidate user",
        "`recommendation_rows`",
        """evaluation_users = test_table["customer_id"].drop_duplicates().tolist()
recommendation_rows = []

for customer_id in evaluation_users:
    chosen = list(scored_items_by_user.get(customer_id, []))
    chosen_ids = {article_id for article_id, _ in chosen}
    for article_id in popular_items:
        if article_id not in chosen_ids:
            chosen.append((article_id, 0.0))
            chosen_ids.add(article_id)
        if len(chosen) == 12:
            break
    for rank, (article_id, score) in enumerate(chosen[:12], start=1):
        recommendation_rows.append({
            "customer_id": customer_id,
            "article_id": article_id,
            "rank": rank,
            "score": float(score),
        })""",
    )
    cells += step(
        "Итоговая recommendation table",
        "создаём DataFrame и проверяем длины списков",
        "API ожидает customer, article, rank и score",
        "`final_recommendations`",
        """final_recommendations = pd.DataFrame(recommendation_rows)
list_lengths = final_recommendations.groupby("customer_id").size()
print("Recommendations:", final_recommendations.shape)
print("Min/Max list length:", list_lengths.min(), list_lengths.max())
assert list_lengths.eq(12).all()
assert not final_recommendations.duplicated(
    ["customer_id", "article_id"]
).any()
display(final_recommendations.head(12))""",
    )
    cells += step(
        "Test ground truth",
        "собираем полные future items evaluation users",
        "denominator Recall не ограничивается candidate positives",
        "`test_ground_truth`",
        """test_known_users = set(test_history["customer_id"])
test_future_evaluation = test_future[
    test_future["customer_id"].isin(test_known_users)
    & test_future["customer_id"].isin(evaluation_users)
]
test_future_unique = (
    test_future_evaluation
    .sort_values("t_dat")
    .drop_duplicates(["customer_id", "article_id"])
)
test_ground_truth = test_future_unique.groupby(
    "customer_id", sort=False
)["article_id"].apply(list).to_dict()
print("Ground-truth users:", len(test_ground_truth))""",
    )
    cells += step(
        "Dictionaries для метрик",
        "преобразуем рекомендации и candidates в списки",
        "Top-12 и Candidate Recall используют разные dictionaries",
        "два model outputs",
        """recommendations_dict = final_recommendations.sort_values(
    ["customer_id", "rank"]
).groupby("customer_id", sort=False)["article_id"].apply(list).to_dict()

candidate_dict = test_table.groupby(
    "customer_id", sort=False
)["article_id"].apply(list).to_dict()
print("Recommendation users:", len(recommendations_dict))
print("Candidate users:", len(candidate_dict))""",
    )
    cells += step(
        "Финальные метрики",
        "считаем Recall, MAP, HitRate и Candidate Recall",
        "это final test, не участвовавший в fit",
        "`final_metrics`",
        """final_metrics = {
    "model": "CatBoost Hybrid",
    "Recall@12": mean_recall_at_k(test_ground_truth, recommendations_dict, 12),
    "MAP@12": map_at_k(test_ground_truth, recommendations_dict, 12),
    "HitRate@12": hit_rate_at_k(test_ground_truth, recommendations_dict, 12),
    "Candidate Recall": candidate_recall_at_k(
        test_ground_truth, candidate_dict, 250
    ),
    "users_evaluated": len(test_ground_truth),
    "average_candidates": test_table.groupby("customer_id").size().mean(),
    "training_time": training_time,
    "inference_time": inference_time,
    "notes": "CatBoostClassifier; common test cohort",
}
display(pd.Series(final_metrics))""",
    )
    cells += step(
        "Feature importance",
        "получаем важности из уже обученной модели",
        "importance описывает использование признака, не причинность",
        "отсортированную таблицу",
        """feature_importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.get_feature_importance(),
}).sort_values("importance", ascending=False).reset_index(drop=True)
display(feature_importance.head(20))""",
    )
    cells += step(
        "Сохранение CatBoost",
        "записываем обученную модель отдельно",
        "batch demo загрузит `.cbm` без fit",
        "`catboost_recommender.cbm`",
        """catboost_path = save_catboost_model(
    model,
    MODEL_DIR / "catboost_recommender.cbm",
)
print("Сохранено:", catboost_path)""",
    )
    cells += step(
        "Сохранение feature config",
        "записываем порядок features и categories",
        "inference должен передавать колонки в том же порядке",
        "два JSON-файла",
        """save_json(FEATURE_COLUMNS, MODEL_DIR / "feature_columns.json")
save_json(CATEGORICAL_FEATURES, MODEL_DIR / "categorical_features.json")
save_json(popular_items, MODEL_DIR / "popular_items.json")
print("Feature config и popularity сохранены")""",
    )
    cells += step(
        "Сохранение metadata",
        "записываем фактические даты, cohort и metrics",
        "API не должен придумывать сведения о модели",
        "`model_metadata.json`",
        """model_metadata = {
    "architecture": "Popularity + Personal History + ALS + Content-Based -> CatBoostClassifier",
    "trained_at": pd.Timestamp.now(tz="UTC"),
    "prediction_horizon_days": 7,
    "recommendation_size": 12,
    "evaluation_user_limit": len(evaluation_users),
    "final_test_metrics": final_metrics,
    "final_test_window_start": test_cutoff,
    "final_test_window_end": test_end,
}
save_json(model_metadata, MODEL_DIR / "model_metadata.json")""",
    )
    cells += step(
        "Сохранение рекомендаций",
        "записываем готовый serving Parquet",
        "API выполняет lookup, а не model inference",
        "`final_recommendations.parquet`",
        """recommendations_path = save_recommendations(
    final_recommendations,
    ARTIFACT_DIR / "final_recommendations.parquet",
)
print("Сохранено:", recommendations_path)""",
    )
    cells += step(
        "Сохранение отчётов",
        "записываем importance и test metrics",
        "notebook 10 загрузит компактные CSV",
        "два report files",
        """feature_importance.to_csv(
    REPORT_DIR / "catboost_feature_importance.csv",
    index=False,
)
pd.DataFrame([final_metrics]).to_csv(
    REPORT_DIR / "catboost_metrics.csv",
    index=False,
)
print("CatBoost reports сохранены")""",
    )
    return cells


def notebook_10() -> list[dict]:
    cells = setup(
        "10. Сравнение моделей",
        "Загружаем только сохранённые метрики, объединяем их и строим один понятный график. Повторного обучения нет.",
    )
    cells += step(
        "Импорты",
        "подключаем pandas и matplotlib",
        "notebook работает только с компактными CSV",
        "два инструмента",
        """import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display""",
    )
    cells += step(
        "Пути к метрикам",
        "перечисляем пять результатов notebooks 04–09",
        "отсутствующая модель не заменяется нулём",
        "`METRIC_FILES`",
        raw_paths()
        + "\n\nFIGURE_DIR = PROJECT_ROOT / \"reports\" / \"figures\"\n"
        + "FIGURE_DIR.mkdir(parents=True, exist_ok=True)\n"
        + "METRIC_FILES = [\n"
        + "    REPORT_DIR / \"baseline_metrics.csv\",\n"
        + "    REPORT_DIR / \"als_metrics.csv\",\n"
        + "    REPORT_DIR / \"content_metrics.csv\",\n"
        + "    REPORT_DIR / \"simple_hybrid_metrics.csv\",\n"
        + "    REPORT_DIR / \"catboost_metrics.csv\",\n"
        + "]",
    )
    cells += step(
        "Проверка файлов",
        "находим отсутствующие CSV",
        "сравнение должно опираться только на реальные результаты",
        "понятную ошибку с порядком notebooks",
        """missing_metric_files = [
    path for path in METRIC_FILES if not path.is_file()
]
if missing_metric_files:
    raise FileNotFoundError(
        f"Не найдены metric files: {missing_metric_files}. "
        "Выполните notebooks 04–09 по порядку."
    )""",
    )
    cells += step(
        "Загрузка метрик",
        "читаем каждый CSV и объединяем строки",
        "новые функции и модели здесь не нужны",
        "`model_metrics`",
        """metric_tables = [pd.read_csv(path) for path in METRIC_FILES]
model_metrics = pd.concat(
    metric_tables,
    ignore_index=True,
    sort=False,
)
print("Rows:", len(model_metrics))""",
    )
    cells += step(
        "Проверка моделей",
        "смотрим названия и дубликаты",
        "каждая model должна иметь одну строку, кроме трёх baseline в общем CSV",
        "список фактических моделей",
        """print(model_metrics["model"].tolist())
duplicate_models = model_metrics["model"].duplicated().sum()
print("Duplicate model names:", duplicate_models)
assert duplicate_models == 0""",
    )
    cells += step(
        "Таблица сравнения",
        "выбираем одинаковые метрики и сортируем по Recall@12",
        "NaN сохраняется как неприменимая метрика",
        "понятную итоговую таблицу",
        """comparison_columns = [
    "model", "Recall@12", "MAP@12", "HitRate@12",
    "Candidate Recall", "users_evaluated", "notes",
]
model_comparison = model_metrics.reindex(columns=comparison_columns)
model_comparison = model_comparison.sort_values(
    "Recall@12",
    ascending=False,
).reset_index(drop=True)
display(model_comparison)""",
    )
    cells += step(
        "График Top-12 метрик",
        "строим Recall, MAP и HitRate одним grouped bar chart",
        "позиционные и непозиционные метрики видны рядом",
        "`model_comparison.png`",
        """plot_table = model_comparison.set_index("model")[[
    "Recall@12", "MAP@12", "HitRate@12"
]]
axis = plot_table.plot.bar(figsize=(12, 5))
axis.set_title("Top-12 metrics on the common test cohort")
axis.set_ylabel("Metric value")
axis.set_xlabel("")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
figure_path = FIGURE_DIR / "model_comparison.png"
plt.savefig(figure_path, dpi=150)
plt.show()
print("Сохранено:", figure_path)""",
    )
    cells += step(
        "Сохранение и вывод",
        "записываем итоговую таблицу и называем лучший Recall",
        "вывод основан на фактических значениях",
        "`model_metrics.csv` и короткое заключение",
        """comparison_path = REPORT_DIR / "model_metrics.csv"
model_comparison.to_csv(comparison_path, index=False)

best_model = model_comparison.iloc[0]
print("Лучший Recall@12:", best_model["model"])
print("Recall@12:", best_model["Recall@12"])
print("Сохранено:", comparison_path)""",
    )
    return cells


def notebook_11() -> list[dict]:
    cells = setup(
        "Дополнительный этап: пакетный inference и API",
        "Notebook повторно оценивает готовую test ranking table сохранённым CatBoost. Он не обучает ALS, Content-Based или CatBoost и не является обязательной частью учебного pipeline.",
    )
    cells += step(
        "Импорты",
        "подключаем pandas и функции загрузки/сохранения",
        "в notebook нет training imports",
        "минимальный inference-набор",
        """import pandas as pd
from IPython.display import display

from fashion_recommender.persistence import (
    load_catboost_model, load_json, load_recommendations, save_recommendations,
)""",
    )
    cells += step(
        "Пути к artifacts",
        "задаём модель, ranking table и config files",
        "batch demo полностью зависит от notebook 09",
        "шесть входных путей",
        raw_paths()
        + "\n\nCATBOOST_PATH = MODEL_DIR / \"catboost_recommender.cbm\"\n"
        + "TEST_TABLE_PATH = PROCESSED_DIR / \"test_ranking_table.parquet\"\n"
        + "FEATURES_PATH = MODEL_DIR / \"feature_columns.json\"\n"
        + "CATEGORIES_PATH = MODEL_DIR / \"categorical_features.json\"\n"
        + "POPULARITY_PATH = MODEL_DIR / \"popular_items.json\"\n"
        + "METADATA_PATH = MODEL_DIR / \"model_metadata.json\"\n"
        + "DEMO_OUTPUT_PATH = ARTIFACT_DIR / \"batch_demo_recommendations.parquet\"",
    )
    cells += step(
        "Проверка artifacts",
        "проверяем модель, table и JSON",
        "batch inference не должен незаметно переобучать отсутствующую модель",
        "понятную ошибку с notebook 09",
        """required_paths = [
    CATBOOST_PATH, TEST_TABLE_PATH, FEATURES_PATH,
    CATEGORIES_PATH, POPULARITY_PATH, METADATA_PATH,
]
missing_paths = [path for path in required_paths if not path.is_file()]
if missing_paths:
    raise FileNotFoundError(
        f"Не найдены inference artifacts: {missing_paths}. "
        "Сначала выполните notebook 09_catboost_ranking_colab.ipynb."
    )""",
    )
    cells += step(
        "Загрузка модели",
        "читаем сохранённый CatBoost `.cbm`",
        "метод `.fit()` здесь не вызывается",
        "`batch_model`",
        """batch_model = load_catboost_model(CATBOOST_PATH)
print("Loaded trees:", batch_model.tree_count_)""",
    )
    cells += step(
        "Загрузка ranking table",
        "читаем готовые test candidates и features",
        "ALS/Content/candidate generation не повторяются",
        "`batch_table`",
        """batch_table = pd.read_parquet(TEST_TABLE_PATH)
print("Batch table:", batch_table.shape)
display(batch_table.head())""",
    )
    cells += step(
        "Загрузка config",
        "читаем порядок features, categories и fallback items",
        "inference должен повторять training schema",
        "три небольших списка",
        """feature_columns = load_json(FEATURES_PATH)
categorical_features = load_json(CATEGORIES_PATH)
popular_items = load_json(POPULARITY_PATH)
print("Features:", len(feature_columns))
print("Categories:", categorical_features)
print("Fallback items:", len(popular_items))""",
    )
    cells += step(
        "Batch X",
        "выбираем features и приводим categories к строкам",
        "ID и target не передаются модели",
        "`batch_x`",
        """batch_x = batch_table[feature_columns].copy()
for column in categorical_features:
    batch_x[column] = batch_x[column].astype(str)
print("Batch X:", batch_x.shape)""",
    )
    cells += step(
        "Batch predict_proba",
        "получаем scores сохранённой моделью",
        "это inference без обучения",
        "`batch_score`",
        """batch_scored = batch_table[["customer_id", "article_id"]].copy()
batch_scored["batch_score"] = batch_model.predict_proba(batch_x)[:, 1]
display(batch_scored.head())""",
    )
    cells += step(
        "Сортировка и Top-12",
        "сортируем scores внутри user и берём первые позиции",
        "fallback будет отдельным шагом",
        "`batch_top12`",
        """batch_ranked = batch_scored.sort_values(
    ["customer_id", "batch_score", "article_id"],
    ascending=[True, False, True],
).drop_duplicates(["customer_id", "article_id"])
batch_top12 = batch_ranked.groupby(
    "customer_id", sort=False
).head(12)
print(batch_top12.groupby("customer_id").size().describe())""",
    )
    cells += step(
        "Batch fallback",
        "дополняем короткие lists сохранённой popularity",
        "результат имеет тот же contract, что notebook 09",
        "`batch_rows`",
        """batch_rows = []
for customer_id, group in batch_top12.groupby("customer_id", sort=False):
    chosen = list(zip(group["article_id"], group["batch_score"]))
    chosen_ids = {article_id for article_id, _ in chosen}
    for article_id in popular_items:
        if article_id not in chosen_ids:
            chosen.append((article_id, 0.0))
            chosen_ids.add(article_id)
        if len(chosen) == 12:
            break
    for rank, (article_id, score) in enumerate(chosen[:12], start=1):
        batch_rows.append({
            "customer_id": customer_id,
            "article_id": article_id,
            "rank": rank,
            "score": float(score),
        })""",
    )
    cells += step(
        "Сохранение demo batch",
        "создаём DataFrame и сохраняем отдельный Parquet",
        "основной `final_recommendations.parquet` из notebook 09 не перезаписывается",
        "`batch_demo_recommendations.parquet`",
        """batch_recommendations = pd.DataFrame(batch_rows)
saved_demo_path = save_recommendations(
    batch_recommendations,
    DEMO_OUTPUT_PATH,
)
print("Сохранено:", saved_demo_path)
print("Rows:", len(batch_recommendations))""",
    )
    cells += step(
        "Демонстрация API-файла",
        "загружаем сохранённый Parquet проверенным loader",
        "API использует такой же формат для lookup",
        "несколько готовых рекомендаций",
        """loaded_demo = load_recommendations(DEMO_OUTPUT_PATH)
example_users = loaded_demo["customer_id"].drop_duplicates().head(2)
display(loaded_demo[loaded_demo["customer_id"].isin(example_users)])""",
    )
    cells += step(
        "Metadata",
        "показываем фактическую информацию сохранённой модели",
        "endpoint `/model-info` возвращает этот объект",
        "словарь model metadata",
        """model_metadata = load_json(METADATA_PATH)
display(model_metadata)""",
    )
    return cells


NOTEBOOKS = {
    "03_temporal_validation_colab.ipynb": notebook_03,
    "04_baselines_colab.ipynb": notebook_04,
    "05_als_colab.ipynb": notebook_05,
    "06_content_based_colab.ipynb": notebook_06,
    "07_candidate_generation_colab.ipynb": notebook_07,
    "08_feature_engineering_colab.ipynb": notebook_08,
    "09_catboost_ranking_colab.ipynb": notebook_09,
    "10_model_comparison_colab.ipynb": notebook_10,
    "11_batch_inference_colab.ipynb": notebook_11,
}


def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    refine_existing_01_02()
    for filename, builder in NOTEBOOKS.items():
        path = NOTEBOOK_DIR / filename
        path.write_text(
            json.dumps(make_notebook(builder()), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print("Generated", filename)


if __name__ == "__main__":
    main()
