import pickle
import pandas as pd
import argparse


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    categorical = ["PUlocationID", "DOlocationID"]
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


def save_results(df_result: pd.DataFrame, output_file: str) -> None:
    df_result.to_parquet(
        output_file,
        engine="pyarrow",
        compression=None,
        index=False,
    )


def main(year: int, month: int):
    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(
        f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet"
    )

    categorical = ["PUlocationID", "DOlocationID"]
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    results = df[["ride_id"]].copy()
    results["duration"] = y_pred
    print(f"Mean duration: {results['duration'].mean()}")
    save_results(results, output_file=f"{year:04d}_{month:02d}_preds.parquet")


if __name__ == "__main__":
    year = 2021
    month = 2
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--month",
        required=True,
        type=int,
    )
    args = parser.parse_args()
    print(f"Launch script with: {args}")
    main(args.year, args.month)
