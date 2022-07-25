from typing import List
import pandas as pd
from datetime import datetime

from batch_olegtaratukhin import main, get_output_path, process_data


S3_ENDPOINT_URL = 'http://localhost:4566'


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def output_file():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)

    df_prepared = process_data(df, ['PUlocationID', 'DOlocationID'])

    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }
    
    df_prepared.to_parquet(
            's3://nyc-durationaws/test/test.parquet',
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options)


def test_integration():
    YEAR = 2021
    MONTH = 1

    main(YEAR, MONTH)
    
    categorical = ['PUlocationID', 'DOlocationID']
    input_file = get_output_path(YEAR, MONTH)

    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }
    df_output = pd.read_parquet(input_file, storage_options=options)

    sum_predictions = df_output['predicted_duration'].sum()

    assert sum_predictions == 34, f"sum: {sum_predictions}"
