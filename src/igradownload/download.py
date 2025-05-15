import pathlib
import logging
import zipfile

import numpy as np
import pandas as pd
import requests

import tqdm
from aq3 import task, workflow, iotask, get_path

_logger = logging.getLogger(__name__)

ROOT_URL = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive"
DEFAULT_DESTINATION = "test"


@iotask(path="raw/{0}", backend='thread', key="raw/{0}")
def download_igra_file(resource_name: str) -> None:
    full_url = f"{ROOT_URL}/{resource_name}"

    output_path = pathlib.Path(get_path())
    tmp_file_path = pathlib.Path(output_path).with_suffix(
        output_path.suffix + ".tmp"
    )

    response = requests.get(full_url, stream=True)
    response.raise_for_status()

    tmp_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tmp_file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    tmp_file_path.rename(output_path)


def download_station_list():
    return download_igra_file(
        "doc/igra2-station-list.txt"
    )


@workflow()
def download_docs():
    docs_files = [
        'igra2-country-list.txt',
        'igra2-data-format.txt',
        'igra2-derived-format.txt',
        'igra2-list-format.txt',
        'igra2-metadata-readme.txt',
        'igra2-monthly-format.txt',
        'igra2-product-description-supplement.pdf',
        'igra2-product-description.pdf',
        'igra2-readme.txt',
        'igra2-station-list.txt',
        'igra2-us-states.txt',
        'status.txt',
        'wmo-history-format.txt',
        'wmo-sonde-history.txt',
        'wmo-wndeq-history.txt',
    ]

    tasks = []
    for file in docs_files:
        resource_name = f"doc/{file}"
        tasks.append(download_igra_file(resource_name))

    return tasks


@iotask(path="station_list.parquet")
def station_list_to_dataframe(station_list_path: str) -> None:
    df = pd.read_fwf(station_list_path, colspec='infer', infer_nrows=2000, header=None, names=[
        'id', 'latitude', 'longitude', 'elevation', 'state', 'name', 'first_year', 'last_year', 'n_obs'
    ])

    df['latitude'] = np.where(
        df['latitude'] == -98.8888, np.nan, df['latitude'])
    df['longitude'] = np.where(
        df['longitude'] == -998.8888, np.nan, df['longitude'])
    df['elevation'] = np.where(
        (df['elevation'] == -999.9) | (df['elevation'] == -998.8), np.nan, df['elevation'])

    df['state'] = df['state'].astype('string')
    df['name'] = df['name'].astype('string')

    output_path = get_path()

    print(output_path)

    df.to_parquet(output_path, index=False)


@workflow()
def station_list():
    station_list_path = download_station_list()

    df = station_list_to_dataframe(station_list_path)

    return df


@workflow()
def download_one_station_data_raw(id: str):
    return download_igra_file("access/data-por/{0}-data.txt.zip".format(id))


@iotask(backend='process', path="station_data/{0}.parquet")
def parse_station_datafile(station_name: str, station_data_path: str):
    sounding_data = []
    sounding_level_data = []

    with zipfile.ZipFile(station_data_path, "r") as zip_ref:
        [filename] = zip_ref.namelist()

        with zip_ref.open(filename, "r") as f:
            while True:
                # Read sounding header.
                header_txt = f.readline()

                if not header_txt:
                    break

                header_tuple = (
                    header_txt[1:12].strip(),
                    header_txt[13:17].strip(),
                    header_txt[18:20].strip(),
                    header_txt[21:23].strip(),
                    header_txt[24:26].strip(),
                    header_txt[27:31].strip(),
                    header_txt[32:36].strip(),
                    header_txt[37:45].strip(),
                    header_txt[46:54].strip(),
                    header_txt[55:62].strip(),
                    header_txt[63:71].strip(),
                )

                sounding_data.append(header_tuple)

                n_levels = int(header_tuple[6])

                # Read sounding levels.
                level_record_lines = [f.readline() for _ in range(n_levels)]

                levels_of_sounding = []
                for level_record in level_record_lines:

                    level_tuple = (
                        level_record[0],
                        level_record[1],
                        level_record[3:8],
                        level_record[9:15],
                        level_record[15:16],
                        level_record[16:21],
                        level_record[21:22],
                        level_record[22:27],
                        level_record[27:28],
                        level_record[28:33],
                        level_record[34:39],
                        level_record[40:45],
                        level_record[46:51],
                    )
                    levels_of_sounding.append(level_tuple)

                sounding_level_data.append(levels_of_sounding)

    # Convert to DataFrames
    sounding_df = pd.DataFrame(
        sounding_data,
        columns=[
            "station_id",
            "year",
            "month",
            "day",
            "hour",
            "release_time",
            "n_levels",
            "pressure_level_data_source",
            "non_pressure_level_data_source",
            "latitude",
            "longitude",
        ],
    )

    # Process the sounding index columns.
    sounding_df['time_nominal'] = pd.to_datetime(
        sounding_df[['year', 'month', 'day', 'hour']]
    )

    sounding_df['latitude'] = sounding_df['latitude'].astype(float) / 10000.0
    sounding_df['longitude'] = sounding_df['longitude'].astype(float) / 10000.0

    sounding_df['release_time_hour'] = sounding_df['release_time'].str[0:2].astype(
        float)
    sounding_df['release_time_minute'] = sounding_df['release_time'].str[2:4].astype(
        float)

    sounding_df['release_time_hour'] = np.where(
        sounding_df['release_time_hour'] == 99,
        np.nan,
        sounding_df['release_time_hour'],
    )

    sounding_df['release_time_minute'] = np.where(
        sounding_df['release_time_minute'] == 99,
        np.nan,
        sounding_df['release_time_minute'],
    )

    utc_time_template = sounding_df[[
        'year', 'month', 'day', 'release_time_hour', 'release_time_minute']]
    utc_time_template.columns = ['year', 'month', 'day', 'hour', 'minute']

    sounding_df['time'] = pd.to_datetime(utc_time_template)
    sounding_df.drop(columns=['year', 'month', 'day', 'hour',
                     'release_time_minute', 'release_time_hour'], inplace=True)

    sounding_level_dfs = []
    for i in tqdm.tqdm(list(range(len(sounding_level_data)))):
        level_data = sounding_level_data[i]
        level_df = pd.DataFrame(
            level_data,
            columns=[
                "level_type_1",
                "level_type_2",
                "elapsed_time",
                "pressure",
                "pressure_qc_flag",
                "geopotential_height",
                "geopotential_height_qc_flag",
                "temperature",
                "temperature_qc_flag",
                "relative_humidity",
                "dewpoint_depression",
                "wind_direction",
                "wind_speed",
            ],
        )

        # Add sounding index to each level
        level_df["sounding_index"] = i

        sounding_level_dfs.append(level_df)

    all_levels_df = pd.concat(sounding_level_dfs, ignore_index=True)

    # Convert the columns to appropriate types
    all_levels_df['elapsed_time'] = all_levels_df['elapsed_time'].astype(int)
    all_levels_df['elapsed_time_seconds'] = all_levels_df['elapsed_time'] % 100
    all_levels_df['elapsed_time_minutes'] = all_levels_df['elapsed_time'] // 100

    all_levels_df['elapsed_time_minutes'] = np.where(
        (all_levels_df['elapsed_time'] == -9999) | (all_levels_df['elapsed_time'] == -8888), np.nan, all_levels_df['elapsed_time_minutes'])
    all_levels_df['elapsed_time_seconds'] = np.where(
        (all_levels_df['elapsed_time'] == -
         9999) | (all_levels_df['elapsed_time'] == -8888),
        np.nan,
        all_levels_df['elapsed_time_seconds'])

    all_levels_df['elapsed_time'] = pd.to_timedelta(all_levels_df['elapsed_time_minutes'] *
                                                    60 + all_levels_df['elapsed_time_seconds'], unit='s')

    all_levels_df.drop(
        columns=['elapsed_time_minutes', 'elapsed_time_seconds'], inplace=True)

    # Following fields have same treatment.
    column_ratio_pairs = [
        ("pressure", 100.0),
        ("geopotential_height", 10.0),
        ("relative_humidity", 10.0),
        ("dewpoint_depression", 10.0),
        ("wind_direction", 1.0),
        ("wind_speed", 10.0),
    ]

    for column, ratio in column_ratio_pairs:
        all_levels_df[column] = all_levels_df[column].astype(int)
        all_levels_df[column] = np.where(
            (all_levels_df[column] == -9999) | (all_levels_df[column] == -8888), np.nan, all_levels_df[column])
        all_levels_df[column] = all_levels_df[column] / ratio

    sounding_df.drop(columns=["n_levels"], inplace=True)
    merged_df = pd.merge(all_levels_df, sounding_df, left_on="sounding_index", right_index=True)

    output_path = pathlib.Path(get_path())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = pathlib.Path(output_path).with_suffix(
        output_path.suffix + ".tmp"
    )
    merged_df.to_parquet(tmp_path, index=False)
    tmp_path.rename(output_path)




@workflow()
def acquire_process_one_station(
    station_id: str,
):
    station_data_path = download_one_station_data_raw(station_id)
    station_data = parse_station_datafile(station_id, station_data_path)

    return station_data


@workflow()
def download_igra():
    list_binding = station_list()
    stns_path = yield list_binding

    stns_df = pd.read_parquet(stns_path)

    station_download_bindings = []
    for stn_id in stns_df['id']:
        station_download_bindings.append(
            parse_station_datafile(stn_id, download_one_station_data_raw(stn_id))
        )

    yield station_download_bindings
