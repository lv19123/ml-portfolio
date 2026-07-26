import pandas as pd
import pytest

from src.config import find_data_file
from src.validation import validate_application_data


try:
    APPLICATION_PATH = find_data_file("application_train.csv")
except FileNotFoundError:
    APPLICATION_PATH = None


@pytest.mark.skipif(
    APPLICATION_PATH is None,
    reason="application_train.csv не добавлен",
)
def test_application_train_contract():
    application = pd.read_csv(
        APPLICATION_PATH,
        usecols=["SK_ID_CURR", "TARGET"],
    )
    validate_application_data(application)
