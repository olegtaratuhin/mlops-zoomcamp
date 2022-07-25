import pandas as pd
from datetime import datetime
from batch_olegtaratukhin import process_data


YEAR = 2021
MONTH = 1
DAY = 1


def dt(hour: int, minute: int, second: int=0):
    return datetime(YEAR, MONTH, DAY, hour, minute, second, microsecond=0)


def test_preprocessing():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)

    result = process_data(df, ['PUlocationID', 'DOlocationID'])
    expected = pd.DataFrame(
        data={
            'PUlocationID': ["-1", "1"],
            'DOlocationID': ["-1", "1"],
            'pickup_datetime': [dt(1, 2, 0), dt(1, 2, 0)],
            'dropOff_datetime': [dt(1, 10, 0), dt(1, 10, 0)],
            'duration': [8.0, 8.0],
        }
    )
    expected['duration'] = expected['duration'].round(8)
    result['duration'] = result['duration'].round(8)
    assert expected.equals(result)
