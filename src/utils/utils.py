import cx_Oracle
from functools import wraps
import pandas as pd
import os
import time
import scipy
from sklearn.metrics import brier_score_loss, log_loss
import numpy as np


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


@run_once
def init_oracle_client(config_file="oracle_cfg.txt"):
    lib_dir = open(config_file).read()
    cx_Oracle.init_oracle_client(lib_dir=lib_dir)


def dataframe_cacher(filename):
    """used to allow command-line arguments to force storing/returning cached dataframe outputs of any function.

    i.e.
    @dataframe_cacher("../data/my_filename.csv")
    def download_dataframe():
        output = expensive_function_that_returns_dataframe()
        return output

    download_dataframe(cache=True)
    """

    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            file_exists = os.path.isfile(filename)
            cache = kwargs.pop("cache", False)
            if file_exists and cache:
                print(f"{func.__name__} is returning cached df")
                return pd.read_parquet(filename)
            else:
                if cache:
                    print("No cached file exists")
                print(f"Running {func.__name__}")
                df = func(*args, **kwargs)
                df.to_parquet(filename, index=False)
                return df

        return wrapper

    return actual_decorator


def fyi(func):
    # Tells you what function is running, and when it is complete.
    @wraps(func)
    def wrapper_decorator(*args, **kwargs):
        function_name = func.__name__
        print(f"Running {function_name}... \n")
        init_time = time.time()
        value = func(*args, **kwargs)
        elapsed_time = time.time() - init_time
        print(f"{function_name} done in {elapsed_time:.0f} sec.")
        return value

    return wrapper_decorator


def uniform_distribution(lo, hi):
    return scipy.stats.uniform(lo, hi - lo)


def get_probability_metrics(df, target_col, probability_a_col, probability_b_col):
    return (
        brier_score_loss(df[target_col], df[probability_a_col]),
        log_loss(df[target_col], df[[probability_a_col, probability_b_col]]),
    )


def make_scores_dict(dataset_name, metric_brier_score, metric_log_loss):
    return {
        "dataset": dataset_name,
        "brier_score": metric_brier_score,
        "log_loss": metric_log_loss,
    }


def get_metrics_dict(
    dataset_name, df, target_col, probability_a_col, probability_b_col
):
    metric_brier_score, metric_log_loss = get_probability_metrics(
        df, target_col, probability_a_col, probability_b_col
    )
    return make_scores_dict(dataset_name, metric_brier_score, metric_log_loss)


def add_features(df):
    df["is_home_team"] = 1
    df["is_away_team"] = 0
    df["away_team_win"] = 1 - df["home_team_win"]
    df["inning_clipped"] = np.clip(df["inning"], 1, 10)
    df["new_extra_innings_rule"] = np.where(
        (df["season"].isin([2020, 2021]))
        & (df["inning_clipped"] == 10)
        & (df["GAME_TYPE_DESC"] == "Regular Season"),
        1,
        0,
    )
    df["current_run_differential"] = (
        df["home_team_current_runs"] - df["away_team_current_runs"]
    )
    df["current_run_differential_clipped"] = np.clip(
        df["away_team_current_runs"], -12, 12
    )
    df["away_team_current_runs_clipped"] = np.clip(df["away_team_current_runs"], 0, 15)
    df["home_team_current_runs_clipped"] = np.clip(df["home_team_current_runs"], 0, 15)
    df["away_team_runs_rest_of_game"] = (
        df["away_runs_scored_end_of_game"] - df["away_team_current_runs"]
    )
    df["home_team_runs_rest_of_game"] = (
        df["home_runs_scored_end_of_game"] - df["home_team_current_runs"]
    )
    return df
