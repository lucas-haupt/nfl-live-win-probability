from asyncio import events
from copy import deepcopy
from sklearn import model_selection
import streamlit as st
import pandas as pd
import pickle
import os
from notebooks.utils import get_model_outputs
import numpy as np
import datetime

data_dir = "data/"
apptitle = "Stats Perform 4th Down Bot"
CHARTS = ["Games", "Experiment"]


st.set_page_config(
    page_title=apptitle,
    page_icon=":football:",
    # layout="wide",
    # initial_sidebar_state="expanded",
)
st.title("Stats Perform 4th Down Bot")
st.write("by Lucas Haupt and Evan Boyd")


def ordinaltg(n):
    return n.replace({1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"})


def add_timeouts_remaining(df):
    df["half"] = round((df["quarter"] + 0.01) / 2)
    df["home_timeout"] = np.where(
        ((df["event_id"] == 57) & (df["home_team_has_ball"] == 1))
        | ((df["event_id"] == 58) & (df["home_team_has_ball"] == 0)),
        1,
        0,
    )
    df["away_timeout"] = np.where(
        ((df["event_id"] == 57) & (df["home_team_has_ball"] == 0))
        | ((df["event_id"] == 58) & (df["home_team_has_ball"] == 1)),
        1,
        0,
    )
    df = df.sort_values(["game_code", "nevent"])
    df["home_timeouts_remaining"] = np.clip(
        3 - df.groupby(["game_code", "half"])["home_timeout"].cumsum(), 0, 3
    )
    df["away_timeouts_remaining"] = np.clip(
        3 - df.groupby(["game_code", "half"])["away_timeout"].cumsum(), 0, 3
    )
    return df


def add_game_info(df):
    df["game_info"] = (
        df["away_team_abbrev"]
        + " at "
        + df["home_team_abbrev"]
        + " Week "
        + df["week"].apply(str)
        + " "
        + df["season"].apply(str)
        + " ("
        + (df["game_code"]).apply(str)
        + ")"
    )
    return df


def add_play_description(df):
    df["absolute_score_diff"] = abs(df["home_start_score"] - df["away_start_score"])

    df["minutes"] = (df["play_start_time"] // 60).apply(int)
    df["seconds"] = (df["play_start_time"] - (df["play_start_time"] // 60) * 60).apply(
        int
    )
    df["seconds_str"] = np.where(
        df["seconds"] > 10, df["seconds"].apply(str), "0" + df["seconds"].apply(str)
    )

    df["team_score_desc"] = np.where(
        df["home_team_has_ball"] == 1,
        np.where(
            df["home_start_score"] > df["away_start_score"],
            "Up by " + df["absolute_score_diff"].apply(str),
            np.where(
                df["home_start_score"] < df["away_start_score"],
                "Down by " + df["absolute_score_diff"].apply(str),
                "Tied",
            ),
        ),
        np.where(
            df["home_start_score"] < df["away_start_score"],
            "Up by " + df["absolute_score_diff"].apply(str),
            np.where(
                df["home_start_score"] > df["away_start_score"],
                "Down by " + df["absolute_score_diff"].apply(str),
                "Tied",
            ),
        ),
    )
    df["play_description"] = (
        ordinaltg(df["quarter"])
        + " Qtr "
        + df["minutes"].apply(str)
        + ":"
        + df["seconds_str"]
        + ", "
        + df["team_score_desc"]
        + ", "
        + ordinaltg(df["down"]).apply(str)
        + " & "
        + df["ytg"].apply(str)
        + ", "
        + df["yd_from_goal"].apply(str)
        + " Yards From Goal, "
        + np.where(
            df["home_team_has_ball"] == 1,
            df["home_team_abbrev"],
            df["away_team_abbrev"],
        )
        + " has ball ("
        + df["nevent"].apply(str)
        + ")"
    )

    return df


def run():
    events_df = pd.read_parquet(os.path.join(data_dir, "event_data.parquet"))
    game_df = pd.read_parquet(os.path.join(data_dir, "game_data.parquet"))
    # print(game_df.head())
    prior_df = pd.read_csv(os.path.join(data_dir, "game_priors.csv"))
    full_df = (
        events_df.merge(prior_df, on="game_code")
        .merge(game_df, on="game_code", suffixes=["", "_y"])
        .pipe(add_timeouts_remaining)
    )

    input_names = [
        "prior_home",
        "prior_away",
        "home_team_has_ball",
        "home_start_score",
        "away_start_score",
        "quarter",
        "overtime",
        "play_start_time",
        "yd_from_goal",
        "from_scrimmage",
        "kick_off",
        "punt",
        "point_after_kick",
        "two_point_attempt",
        "field_goal_attempt",
        "down",
        "ytg",
        "home_timeouts_remaining",
        "away_timeouts_remaining",
    ]

    full_df = full_df[full_df[input_names].notna().all(axis=1)]
    full_df = full_df.pipe(add_game_info).pipe(add_play_description)
    full_df = full_df.sort_values(["game_date", "nevent"], ascending=[False, True])
    games = full_df["game_info"].drop_duplicates()
    # print(full_df.head())
    fourth_downs_only = full_df.loc[full_df.down == 4]
    chart_select_box = st.selectbox("Chart", CHARTS)
    clf = pickle.load(open(os.path.join("models/game_score_new_4.sav"), "rb"))
    rf = pickle.load(open(os.path.join("models/game_score_random_forest.p"), "rb"))

    if chart_select_box == "Games":
        game_selection = st.selectbox("Pick a Game", games)
        plays = fourth_downs_only[fourth_downs_only["game_info"] == game_selection][
            "play_description"
        ]
        play_selection = st.selectbox("Play", plays)
        play_example = fourth_downs_only[
            (fourth_downs_only["play_description"] == play_selection)
            & (fourth_downs_only["game_info"] == game_selection)
        ]
        example_input = play_example[input_names]
        example_input_rf = play_example[input_names]
        example_input_go_for_it = deepcopy(example_input)
        example_input_go_for_it_rf = deepcopy(example_input)
        example_input_punt = deepcopy(example_input)
        example_input_punt_rf = deepcopy(example_input)
        example_input_field_goal = deepcopy(example_input)
        example_input_field_goal_rf = deepcopy(example_input)
        # print(example_input_go_for_it)
        example_input_go_for_it["punt"] = 0
        example_input_go_for_it["field_goal_attempt"] = 0
        # print(example_input_go_for_it)
        example_input_punt["punt"] = 1
        example_input_punt["field_goal_attempt"] = 0
        example_input_field_goal["punt"] = 0
        example_input_field_goal["field_goal_attempt"] = 1
        # print(example_input_go_for_it)
        example_input_go_for_it_rf["punt"] = 0
        example_input_go_for_it_rf["field_goal_attempt"] = 0
        # print(example_input_go_for_it)
        example_input_punt_rf["punt"] = 1
        example_input_punt_rf["field_goal_attempt"] = 0
        example_input_field_goal_rf["punt"] = 0
        example_input_field_goal_rf["field_goal_attempt"] = 1
        example_running_score = play_example[
            ["home_start_score", "away_start_score"]
        ].values
        example_output_go_for_it = get_model_outputs(
            clf, example_input_go_for_it, example_running_score
        )
        example_output_punt = get_model_outputs(
            clf, example_input_punt, example_running_score
        )
        example_output_field_goal = get_model_outputs(
            clf, example_input_field_goal, example_running_score
        )
        example_output_go_for_it_rf = get_model_outputs(
            rf, example_input_go_for_it_rf, example_running_score
        )
        example_output_punt_rf = get_model_outputs(
            rf, example_input_punt_rf, example_running_score
        )
        example_output_field_goal_rf = get_model_outputs(
            rf, example_input_field_goal_rf, example_running_score
        )

        example_input = pd.concat(
            [
                pd.DataFrame(example_input_go_for_it),
                pd.DataFrame(example_input_punt),
                pd.DataFrame(example_input_field_goal),
            ]
        )
        example_input_rf = pd.concat(
            [
                pd.DataFrame(example_input_go_for_it_rf),
                pd.DataFrame(example_input_punt_rf),
                pd.DataFrame(example_input_field_goal_rf),
            ]
        )
        team_lwp_index = np.where(play_example["home_team_has_ball"] == 1, 0, 2)
        example_output = pd.concat(
            [
                pd.DataFrame(
                    example_output_go_for_it["ft_outcome"],
                    index=["Go For It"],
                )[team_lwp_index],
                pd.DataFrame(example_output_punt["ft_outcome"], index=["Punt"])[
                    team_lwp_index
                ],
                pd.DataFrame(
                    example_output_field_goal["ft_outcome"],
                    index=["Field Goal"],
                )[team_lwp_index],
            ]
        )
        example_output_rf = pd.concat(
            [
                pd.DataFrame(
                    example_output_go_for_it_rf["ft_outcome"],
                    index=["Go For It"],
                )[team_lwp_index],
                pd.DataFrame(example_output_punt_rf["ft_outcome"], index=["Punt"])[
                    team_lwp_index
                ],
                pd.DataFrame(
                    example_output_field_goal_rf["ft_outcome"],
                    index=["Field Goal"],
                )[team_lwp_index],
            ]
        )
        MODEL_TYPE_SELECT = ["MLP", "RF"]
        model_type_selection = st.selectbox("Model", MODEL_TYPE_SELECT)
        if model_type_selection == "MLP":
            st.dataframe(
                example_output,
                width=10000,
            )

        elif model_type_selection == "RF":
            st.dataframe(
                example_output_rf,
                width=10000,
            )


if __name__ == "__main__":
    run()
