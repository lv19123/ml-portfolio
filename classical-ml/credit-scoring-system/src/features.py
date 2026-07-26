"""Детерминированные агрегаты из исторических таблиц Home Credit."""

import numpy as np
import pandas as pd

from src.validation import validate_feature_table


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Выполнить деление без бесконечностей при нулевом знаменателе."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _require_columns(
    data: pd.DataFrame,
    required: set[str],
    *,
    source_name: str,
) -> None:
    """Проверить минимальную схему сырой таблицы."""
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            f"{source_name}: отсутствуют обязательные колонки {missing}"
        )


def build_bureau_features(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
) -> pd.DataFrame:
    """Построить bureau/bureau_balance-агрегаты на уровне клиента."""
    _require_columns(
        bureau,
        {
            "SK_ID_CURR",
            "SK_ID_BUREAU",
            "DAYS_CREDIT",
            "CREDIT_ACTIVE",
            "CREDIT_DAY_OVERDUE",
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
        },
        source_name="bureau",
    )
    _require_columns(
        bureau_balance,
        {"SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"},
        source_name="bureau_balance",
    )
    if not bureau["SK_ID_BUREAU"].is_unique:
        raise ValueError("SK_ID_BUREAU должен быть уникальным в bureau")

    bureau_known = bureau.loc[bureau["DAYS_CREDIT"].le(0)].copy()
    bureau_known["IS_ACTIVE"] = (
        bureau_known["CREDIT_ACTIVE"].eq("Active").astype(int)
    )
    bureau_known["HAS_OVERDUE"] = (
        bureau_known["CREDIT_DAY_OVERDUE"].gt(0).astype(int)
    )
    bureau_known["DEBT_TO_CREDIT_RATIO"] = safe_divide(
        bureau_known["AMT_CREDIT_SUM_DEBT"],
        bureau_known["AMT_CREDIT_SUM"],
    )
    bureau_features = (
        bureau_known.groupby("SK_ID_CURR")
        .agg(
            BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
            BUREAU_ACTIVE_SHARE=("IS_ACTIVE", "mean"),
            BUREAU_OVERDUE_SHARE=("HAS_OVERDUE", "mean"),
            BUREAU_CREDIT_AMOUNT_TOTAL=("AMT_CREDIT_SUM", "sum"),
            BUREAU_CREDIT_AMOUNT_MEAN=("AMT_CREDIT_SUM", "mean"),
            BUREAU_DEBT_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
            BUREAU_DEBT_TO_CREDIT_MEAN=("DEBT_TO_CREDIT_RATIO", "mean"),
            BUREAU_MAX_DAYS_OVERDUE=("CREDIT_DAY_OVERDUE", "max"),
            BUREAU_MOST_RECENT_CREDIT_DAYS=("DAYS_CREDIT", "max"),
            BUREAU_OLDEST_CREDIT_DAYS=("DAYS_CREDIT", "min"),
        )
        .reset_index()
    )
    bureau_features["BUREAU_MOST_RECENT_CREDIT_DAYS"] *= -1
    bureau_features["BUREAU_OLDEST_CREDIT_DAYS"] *= -1

    balance_known = bureau_balance.loc[
        bureau_balance["MONTHS_BALANCE"].le(0)
    ].copy()
    status_map = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "C": 0,
        "X": np.nan,
    }
    balance_known["STATUS_NUMERIC"] = balance_known["STATUS"].map(status_map)
    balance_known["HAS_OVERDUE_MONTH"] = (
        balance_known["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
    )
    balance_by_credit = (
        balance_known.groupby("SK_ID_BUREAU")
        .agg(
            BB_MONTH_COUNT=("MONTHS_BALANCE", "count"),
            BB_MAX_STATUS=("STATUS_NUMERIC", "max"),
            BB_OVERDUE_MONTH_SHARE=("HAS_OVERDUE_MONTH", "mean"),
        )
        .reset_index()
    )
    balance_with_client = balance_by_credit.merge(
        bureau_known[["SK_ID_BUREAU", "SK_ID_CURR"]],
        on="SK_ID_BUREAU",
        how="inner",
        validate="one_to_one",
    )
    balance_features = (
        balance_with_client.groupby("SK_ID_CURR")
        .agg(
            BB_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
            BB_MONTH_COUNT_TOTAL=("BB_MONTH_COUNT", "sum"),
            BB_MAX_STATUS=("BB_MAX_STATUS", "max"),
            BB_OVERDUE_MONTH_SHARE_MEAN=("BB_OVERDUE_MONTH_SHARE", "mean"),
        )
        .reset_index()
    )
    result = bureau_features.merge(
        balance_features,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    validate_feature_table(result, allowed_prefixes=("BUREAU_", "BB_"))
    return result


def build_previous_application_features(
    previous: pd.DataFrame,
) -> pd.DataFrame:
    """Построить агрегаты предыдущих заявок на уровне клиента."""
    _require_columns(
        previous,
        {
            "SK_ID_CURR",
            "SK_ID_PREV",
            "DAYS_DECISION",
            "NAME_CONTRACT_STATUS",
            "AMT_APPLICATION",
            "AMT_CREDIT",
            "AMT_DOWN_PAYMENT",
            "RATE_DOWN_PAYMENT",
        },
        source_name="previous_application",
    )
    known = previous.loc[previous["DAYS_DECISION"].le(0)].copy()
    date_columns = [
        column
        for column in [
            "DAYS_FIRST_DRAWING",
            "DAYS_FIRST_DUE",
            "DAYS_LAST_DUE_1ST_VERSION",
            "DAYS_LAST_DUE",
            "DAYS_TERMINATION",
        ]
        if column in known.columns
    ]
    known[date_columns] = known[date_columns].replace(365243, np.nan)
    known["IS_APPROVED"] = (
        known["NAME_CONTRACT_STATUS"].eq("Approved").astype(int)
    )
    known["IS_REFUSED"] = (
        known["NAME_CONTRACT_STATUS"].eq("Refused").astype(int)
    )
    known["CREDIT_TO_APPLICATION_RATIO"] = safe_divide(
        known["AMT_CREDIT"],
        known["AMT_APPLICATION"],
    )
    result = (
        known.groupby("SK_ID_CURR")
        .agg(
            PREV_APPLICATION_COUNT=("SK_ID_PREV", "count"),
            PREV_APPROVED_SHARE=("IS_APPROVED", "mean"),
            PREV_REFUSED_SHARE=("IS_REFUSED", "mean"),
            PREV_REQUESTED_AMOUNT_MEAN=("AMT_APPLICATION", "mean"),
            PREV_REQUESTED_AMOUNT_TOTAL=("AMT_APPLICATION", "sum"),
            PREV_GRANTED_AMOUNT_MEAN=("AMT_CREDIT", "mean"),
            PREV_GRANTED_AMOUNT_TOTAL=("AMT_CREDIT", "sum"),
            PREV_CREDIT_TO_APPLICATION_MEAN=(
                "CREDIT_TO_APPLICATION_RATIO",
                "mean",
            ),
            PREV_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
            PREV_RATE_DOWN_PAYMENT_MEAN=("RATE_DOWN_PAYMENT", "mean"),
            PREV_MOST_RECENT_DECISION_DAYS=("DAYS_DECISION", "max"),
        )
        .reset_index()
    )
    result["PREV_MOST_RECENT_DECISION_DAYS"] *= -1
    validate_feature_table(result, allowed_prefixes=("PREV_",))
    return result


def build_installments_features(
    installments: pd.DataFrame,
) -> pd.DataFrame:
    """Построить платёжные агрегаты на уровне клиента."""
    _require_columns(
        installments,
        {
            "SK_ID_CURR",
            "SK_ID_PREV",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        },
        source_name="installments_payments",
    )
    known_mask = (
        installments["DAYS_INSTALMENT"].le(0)
        & installments["DAYS_ENTRY_PAYMENT"].notna()
        & installments["DAYS_ENTRY_PAYMENT"].le(0)
    )
    known = installments.loc[known_mask].copy()
    known["PAYMENT_DELAY_DAYS"] = (
        known["DAYS_ENTRY_PAYMENT"] - known["DAYS_INSTALMENT"]
    ).clip(lower=0)
    known["IS_LATE"] = known["PAYMENT_DELAY_DAYS"].gt(0).astype(int)
    known["IS_UNDERPAID"] = (
        known["AMT_PAYMENT"].lt(known["AMT_INSTALMENT"]).astype(int)
    )
    result = (
        known.groupby("SK_ID_CURR")
        .agg(
            INST_PAYMENT_COUNT=("SK_ID_PREV", "size"),
            INST_PLANNED_AMOUNT_TOTAL=("AMT_INSTALMENT", "sum"),
            INST_PAID_AMOUNT_TOTAL=("AMT_PAYMENT", "sum"),
            INST_PAYMENT_AMOUNT_MEAN=("AMT_PAYMENT", "mean"),
            INST_UNDERPAID_SHARE=("IS_UNDERPAID", "mean"),
            INST_LATE_PAYMENT_SHARE=("IS_LATE", "mean"),
            INST_DELAY_DAYS_MEAN=("PAYMENT_DELAY_DAYS", "mean"),
            INST_DELAY_DAYS_MAX=("PAYMENT_DELAY_DAYS", "max"),
        )
        .reset_index()
    )
    result["INST_PAID_TO_PLANNED_RATIO"] = safe_divide(
        result["INST_PAID_AMOUNT_TOTAL"],
        result["INST_PLANNED_AMOUNT_TOTAL"],
    )
    validate_feature_table(result, allowed_prefixes=("INST_",))
    return result


def build_pos_cash_features(pos_cash: pd.DataFrame) -> pd.DataFrame:
    """Построить POS/CASH-агрегаты на уровне клиента."""
    _require_columns(
        pos_cash,
        {
            "SK_ID_CURR",
            "SK_ID_PREV",
            "MONTHS_BALANCE",
            "SK_DPD",
            "NAME_CONTRACT_STATUS",
            "CNT_INSTALMENT_FUTURE",
        },
        source_name="POS_CASH_balance",
    )
    known = pos_cash.loc[pos_cash["MONTHS_BALANCE"].le(0)].copy()
    known["HAS_DELINQUENCY"] = known["SK_DPD"].gt(0).astype(int)
    known["ACTIVE_CREDIT_ID"] = known["SK_ID_PREV"].where(
        known["NAME_CONTRACT_STATUS"].eq("Active")
    )
    result = (
        known.groupby("SK_ID_CURR")
        .agg(
            POS_RECORD_COUNT=("MONTHS_BALANCE", "count"),
            POS_DPD_MEAN=("SK_DPD", "mean"),
            POS_DPD_MAX=("SK_DPD", "max"),
            POS_DELINQUENCY_SHARE=("HAS_DELINQUENCY", "mean"),
            POS_REMAINING_INSTALMENTS_MEAN=(
                "CNT_INSTALMENT_FUTURE",
                "mean",
            ),
            POS_ACTIVE_CREDIT_COUNT=("ACTIVE_CREDIT_ID", "nunique"),
            POS_LAST_OBSERVATION_MONTH=("MONTHS_BALANCE", "max"),
        )
        .reset_index()
    )
    result["POS_LAST_OBSERVATION_MONTH"] *= -1
    validate_feature_table(result, allowed_prefixes=("POS_",))
    return result


def build_credit_card_features(
    credit_card: pd.DataFrame,
) -> pd.DataFrame:
    """Построить агрегаты кредитных карт на уровне клиента."""
    _require_columns(
        credit_card,
        {
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "AMT_PAYMENT_CURRENT",
            "AMT_DRAWINGS_CURRENT",
            "SK_DPD",
        },
        source_name="credit_card_balance",
    )
    known = credit_card.loc[credit_card["MONTHS_BALANCE"].le(0)].copy()
    known["BALANCE_TO_LIMIT_RATIO"] = safe_divide(
        known["AMT_BALANCE"],
        known["AMT_CREDIT_LIMIT_ACTUAL"],
    )
    known["HAS_DELINQUENCY"] = known["SK_DPD"].gt(0).astype(int)
    result = (
        known.groupby("SK_ID_CURR")
        .agg(
            CC_RECORD_COUNT=("MONTHS_BALANCE", "count"),
            CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
            CC_BALANCE_MAX=("AMT_BALANCE", "max"),
            CC_CREDIT_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            CC_BALANCE_TO_LIMIT_MEAN=("BALANCE_TO_LIMIT_RATIO", "mean"),
            CC_BALANCE_TO_LIMIT_MAX=("BALANCE_TO_LIMIT_RATIO", "max"),
            CC_PAYMENT_TOTAL=("AMT_PAYMENT_CURRENT", "sum"),
            CC_DRAWING_AMOUNT_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
            CC_DPD_MEAN=("SK_DPD", "mean"),
            CC_DPD_MAX=("SK_DPD", "max"),
            CC_DELINQUENCY_SHARE=("HAS_DELINQUENCY", "mean"),
        )
        .reset_index()
    )
    validate_feature_table(result, allowed_prefixes=("CC_",))
    return result
