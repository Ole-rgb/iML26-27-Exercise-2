"""Dataset loading.

Every dataset is returned as a :class:`Dataset` with numeric, imputed features so that all
models in :mod:`iml_feature_effects.models` can be fit on it directly. Add further loaders
(e.g. other OpenML or Kaggle datasets) following the pattern of :func:`load_bike_sharing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

# OpenML downloads are cached here, relative to the current working directory.
DATA_HOME = Path("data")


@dataclass(frozen=True)
class Dataset:
    """A preprocessed tabular classification dataset.

    :param name: Human-readable dataset name, used in result tables and plots.
    :param X: Numeric feature matrix without missing values.
    :param y: Target vector.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


def load_bike_sharing(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the hourly Bike Sharing Demand dataset from OpenML (data id 42712).

    The target is the number of rented bikes per hour. The download is cached in ``data_home``, so
    network access is only needed on the first call.

    :param data_home: Cache directory for the download, relative paths are resolved against the
        current working directory.
    :return: The preprocessed Bike Sharing dataset.
    :rtype: Dataset
    """
    bunch = fetch_openml(data_id=42712, as_frame=True, parser="liac-arff", data_home=data_home)
    # This OpenML version already excludes ``casual`` and ``registered``, which sum to the target.
    X = pd.get_dummies(bunch.data, drop_first=True, dtype=float)
    y = bunch.target.astype(float).rename("count")
    return Dataset(name="bike_sharing", X=X.astype(float), y=y)


def load_parole_violators() -> Dataset:
    """Load the Parole Violators dataset (Kaggle: econdata/predicting-parole-violators).

    Parolees released in 2004 from the National Corrections Reporting Program (as used in MIT's
    *The Analytics Edge*). The target ``violator`` is 1 if the parolee violated parole. The CSV is
    shipped in ``data/kaggle/`` so that no Kaggle API key is needed.

    Preprocessing:

    - ``race`` (1 = white, 2 = other) is recoded to ``non.white`` (0 = white, 1 = non-white).
    - ``state`` and ``crime`` are integer codes for categories, so they are one-hot encoded.

    :return: The preprocessed Parole Violators dataset.
    :rtype: Dataset
    """
    df = pd.read_csv("data/kaggle/parole.csv")
    df["non.white"] = (df.pop("race") - 1).astype(float)  # original is encoded with values [1,2] -> [white, other]
    y = df.pop("violator").astype(int)
    X = pd.get_dummies(df, columns=["state", "crime"], drop_first=True, dtype=float)
    return Dataset(name="parole", X=X.astype(float), y=y)


def load_heart_failure(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the Heart Failure Clinical Records dataset from OpenML (data id 43008).

    299 heart-failure patients; the target ``DEATH_EVENT`` is 1 if the patient died during follow-up.

    :param data_home: Cache directory for the download.
    :return: The preprocessed Heart Failure dataset.
    :rtype: Dataset
    """
    bunch = fetch_openml(
        data_id=43008, as_frame=True, parser="liac-arff", data_home=data_home, target_column="DEATH_EVENT"
    )
    X = bunch.data.astype(float)
    X = X.drop(columns="time")
    y = bunch.target.astype(float).astype(int).rename("death_event")
    return Dataset(name="heart_failure", X=X, y=y)


def load_german_credit(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the German Credit dataset from OpenML (data id 31, ``credit-g``).

    1000 loan applicants; the target is 1 for a *bad* credit risk and 0 for a good one.
    Note: ``personal_status`` mixes marital status and sex in one feature.

    :param data_home: Cache directory for the download.
    :return: The preprocessed German Credit dataset.
    :rtype: Dataset
    """
    bunch = fetch_openml(data_id=31, as_frame=True, parser="liac-arff", data_home=data_home)
    X = pd.get_dummies(bunch.data, drop_first=True, dtype=float)
    y = (bunch.target == "bad").astype(int).rename("bad_credit")
    return Dataset(name="german_credit", X=X.astype(float), y=y)
