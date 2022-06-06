import datetime
import pathlib
import pickle
from typing import Any, List, Optional, Tuple

import pandas as pd
import prefect
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

DATA_PATH = pathlib.Path("./data/")
ARTIFACTS_PATH = pathlib.Path("./models/")
FULL_FORMAT = "%Y-%m-%d"
SHORT_FORMAT = "%Y-%m"

logger = prefect.logging.get_logger()


@prefect.task
def read_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


@prefect.task
def prepare_features(
    df: pd.DataFrame,
    categorical: List[str],
    train: bool = True,
) -> pd.DataFrame:
    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


@prefect.task
def train_model(
    df: pd.DataFrame,
    categorical: List[str],
) -> Tuple[LinearRegression, DictVectorizer]:
    train_dicts = df[categorical].to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")

    return lr, dv


@prefect.task
def run_model(
    df: pd.DataFrame,
    categorical: List[str],
    dv: DictVectorizer,
    lr: LinearRegression,
) -> None:
    val_dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


def _format_date(date: datetime.date, full: bool = False) -> str:
    if not full:
        return date.strftime(SHORT_FORMAT)
    return date.strftime(FULL_FORMAT)


def _get_date_lag(base: datetime.date, months_back: int) -> datetime.date:
    total_base_months = 12 * base.year + base.month
    total_months_back = total_base_months - months_back
    return datetime.date(
        total_months_back // 12,
        total_months_back % 12,
        1,
    )


@prefect.task
def get_paths(date_string: str) -> Tuple[str, str]:
    date = datetime.datetime.strptime(date_string, FULL_FORMAT).date()
    train_month = _get_date_lag(date, 2)
    valid_month = _get_date_lag(date, 1)

    logger.debug("Given month: %s", date)
    logger.debug("Train month: %s", train_month)
    logger.debug("Valid month: %s", valid_month)

    return (
        DATA_PATH / f"fhv_tripdata_{_format_date(train_month)}.parquet",
        DATA_PATH / f"fhv_tripdata_{_format_date(valid_month)}.parquet",
    )


@prefect.task
def save(obj: Any, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


@prefect.flow(name="03-hw-tripdata")
def main(date: Optional[str] = None):
    if date is None:
        date = datetime.datetime.strftime(datetime.date.today(), FULL_FORMAT)

    train_path, val_path = get_paths(date).result()
    categorical = ["PUlocationID", "DOlocationID"]

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    save(lr, str(ARTIFACTS_PATH / f"model-{date}.pkl"))
    save(dv, str(ARTIFACTS_PATH / f"dv-{date}.pkl"))

    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    flow=main,
    name="hw-03",
    flow_runner=SubprocessFlowRunner(),
    schedule=CronSchedule(cron="0 9 15 * *"),
    tags=["ml"],
)
