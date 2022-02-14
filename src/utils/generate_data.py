import cx_Oracle
import pandas as pd
from src.utils.utils import init_oracle_client, dataframe_cacher, fyi
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor
import os


def df_from_oracle(filename="", query_string=""):
    init_oracle_client()
    print(filename)
    if filename and query_string:
        raise Exception

    if filename:
        with open(filename, "r") as f:
            query_string = f.read()
    c = cx_Oracle.connect(
        "LHAUPT",
        "GSFkfNoZTKyC23971",
        "exadata2-cluster.stats.com:1521/bladerac_usr.stats.com",
    )
    return pd.read_sql(query_string, con=c)


def df_from_athena(filename="", query_string=""):
    print(filename)
    if filename and query_string:
        raise Exception

    if filename:
        with open(filename, "r") as f:
            query_string = f.read()

    c = connect(
        aws_access_key_id="AKIAIRYXCD73CAZ5A76A",
        aws_secret_access_key="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",
        s3_staging_dir="s3://aws-athena-query-results-323906537337-us-east-1/",
        region_name="us-east-1",
        cursor_class=PandasCursor,
    ).cursor()

    return c.execute(query_string).as_pandas()


@fyi
@dataframe_cacher(filename="data/event_data.parquet")
def get_event_data():
    return df_from_oracle("queries/event_data.sql")


@fyi
@dataframe_cacher(filename="data/odds_data.parquet")
def get_odds_data():
    return df_from_oracle("queries/odds_data.sql")


@fyi
@dataframe_cacher(filename="data/game_data.parquet")
def get_game_data():
    return df_from_oracle("queries/game_data.sql")


@fyi
@dataframe_cacher(filename="data/schedule_data.parquet")
def get_schedule_data():
    return df_from_oracle("queries/schedule_data.sql")


@fyi
@dataframe_cacher(filename="data/division_data.parquet")
def get_division_data():
    return df_from_oracle("queries/division_data.sql")


if __name__ == "__main__":
    get_event_data(cache=False)
    get_odds_data(cache=False)
    get_game_data(cache=False)
    get_schedule_data(cache=False)
    get_division_data(cache=False)
    os.system('say "done"')
