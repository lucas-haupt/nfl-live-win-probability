from asyncio import events
from copy import deepcopy
from functools import cache
from operator import index
from pyexpat import model
from sklearn import model_selection
import streamlit as st
import pandas as pd
import pickle
import os
from scipy.linalg import lstsq
from notebooks.utils import get_model_outputs, normalize_df
from src.utils.utils import fyi
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from src.utils.generate_data import (
    get_event_data,
    get_game_data,
    get_odds_data,
    get_schedule_data,
)

data_dir = "data/"
apptitle = "Stats Perform Football Predictions"
CHARTS = ["Games", "4th Down Bot Situation", "LWP Slider"]


st.set_page_config(
    page_title=apptitle,
    page_icon=":football:",
    layout="wide",
    # initial_sidebar_state="expanded",
)
st.title("Stats Perform Football Predictions")
# st.write("by Lucas Haupt and Evan Boyd")


# @st.cache
# def load_models():
#     scrimmage_plays_we_want = [1, 2, 3, 4, 7, 9, 14, 17, 18, 35]
#     # search_mlp_play_outcome = pickle.load(open("models/search_mlp_play_outcome.p", "rb"))
#     search_rf_play_outcome = pickle.load(open("models/search_rf_play_outcome.p", "rb"))
#     search_rf_drive_outcome = pickle.load(
#         open("models/search_rf_drive_outcome.p", "rb")
#     )
#     search_mlp_play_outcome_normalized_new_hpo = pickle.load(
#         open("models/search_mlp_play_outcome_normalized_new_hpo.p", "rb")
#     )
#     search_mlp_score_diff_clipped_rf_drive_preds = pickle.load(
#         open("models/search_mlp_score_diff_clipped_rf_drive_preds.p", "rb")
#     )
#     search_mlp_total_score = pickle.load(open("models/search_mlp_total_score.p", "rb"))
#     input_names_play_and_drive_preds = search_rf_play_outcome.feature_names_in_
#     input_names_score_diff_pred = (
#         search_mlp_score_diff_clipped_rf_drive_preds.feature_names_in_
#     )
#     search_rf_play_class_names = [
#         "search_rf_play_" + x for x in search_rf_play_outcome.classes_
#     ]
#     search_mlp_play_class_names = [
#         "search_mlp_play_" + x for x in search_rf_play_outcome.classes_
#     ]
#     search_rf_drive_class_names = [
#         "search_rf_drive_" + x for x in search_rf_drive_outcome.classes_
#     ]

#     search_rf_play_class_names_home = [x + "_home" for x in search_rf_play_class_names]
#     search_rf_play_class_names_away = [x + "_away" for x in search_rf_play_class_names]
#     search_rf_drive_class_names_home = [
#         x + "_home" for x in search_rf_drive_class_names
#     ]
#     search_rf_drive_class_names_away = [
#         x + "_away" for x in search_rf_drive_class_names
#     ]
#     return (
#         search_rf_play_outcome,
#         search_rf_drive_outcome,
#         search_rf_play_class_names,
#         search_rf_drive_class_names,
#         search_mlp_score_diff_clipped_rf_drive_preds,
#         scrimmage_plays_we_want,
#         search_mlp_play_outcome_normalized_new_hpo,
#         search_mlp_total_score,
#         input_names_play_and_drive_preds,
#         input_names_score_diff_pred,
#         search_mlp_play_class_names,
#         search_rf_play_class_names_home,
#         search_rf_play_class_names_away,
#         search_rf_drive_class_names_home,
#         search_rf_drive_class_names_away,
#     )


# (
#     search_rf_play_outcome,
#     search_rf_drive_outcome,
#     search_rf_play_class_names,
#     search_rf_drive_class_names,
#     search_mlp_score_diff_clipped_rf_drive_preds,
#     scrimmage_plays_we_want,
#     search_mlp_play_outcome_normalized_new_hpo,
#     search_mlp_total_score,
#     input_names_play_and_drive_preds,
#     input_names_score_diff_pred,
#     search_mlp_play_class_names,
#     search_rf_play_class_names_home,
#     search_rf_play_class_names_away,
#     search_rf_drive_class_names_home,
#     search_rf_drive_class_names_away,
# ) = load_models()
# home_score_cols_go_for_it = ["home_score_go_for_it_" + str(x) for x in list(range(63))]
# away_score_cols_go_for_it = ["away_score_go_for_it_" + str(x) for x in list(range(60))]
# home_score_cols_punt = ["home_score_punt_" + str(x) for x in list(range(63))]
# away_score_cols_punt = ["away_score_punt_" + str(x) for x in list(range(60))]
# home_score_cols_field_goal = [
#     "home_score_field_goal_" + str(x) for x in list(range(63))
# ]
# away_score_cols_field_goal = [
#     "away_score_field_goal_" + str(x) for x in list(range(60))
# ]
# home_score_cols_go_for_it_rf = [
#     "home_score_go_for_it_rf_" + str(x) for x in list(range(63))
# ]
# away_score_cols_go_for_it_rf = [
#     "away_score_go_for_it_rf_" + str(x) for x in list(range(60))
# ]
# home_score_cols_punt_rf = ["home_score_punt_rf_" + str(x) for x in list(range(63))]
# away_score_cols_punt_rf = ["away_score_punt_rf_" + str(x) for x in list(range(60))]
# home_score_cols_field_goal_rf = [
#     "home_score_field_goal_rf_" + str(x) for x in list(range(63))
# ]
# away_score_cols_field_goal_rf = [
#     "away_score_field_goal_rf_" + str(x) for x in list(range(60))
# ]
@st.cache
def ordinaltg(n):
    return n.replace({1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"})


# def convert_to_dataframe(df):
#     return pd.DataFrame(df)


@st.cache
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
    df["time_left_in_game"] = np.where(
        df["quarter"] <= 4,
        df["play_start_time"] + (4 - df["quarter"]) * 900,
        df["play_start_time"],
    )

    return df


@st.cache
def add_play_description(df):
    df["absolute_score_diff"] = abs(df["home_start_score"] - df["away_start_score"])

    df["minutes"] = (df["play_start_time"].fillna(0) // 60).apply(int)
    df["seconds"] = (
        df["play_start_time"].fillna(0) - (df["play_start_time"].fillna(0) // 60) * 60
    ).apply(int)
    df["seconds_str"] = np.where(
        df["seconds"] >= 10, df["seconds"].apply(str), "0" + df["seconds"].apply(str)
    )
    df["time_str"] = df["minutes"].apply(str) + ":" + df["seconds_str"]

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
        + " has ball, "
        + "Off TO: "
        + np.where(
            df["home_team_has_ball"] == 1,
            df["home_timeouts_remaining"],
            df["away_timeouts_remaining"],
        ).astype(str)
        + ", Def TO: "
        + np.where(
            df["home_team_has_ball"] == 0,
            df["home_timeouts_remaining"],
            df["away_timeouts_remaining"],
        ).astype(str)
        + " ("
        + df["nevent"].apply(str)
        + ")"
    )

    return df

chart_select_box = st.selectbox("Chart", CHARTS)

@st.cache
def get_fourth_down_predictions(df):
    scrimmage_plays_we_want = [1, 2, 3, 4, 7, 9, 14, 17, 18, 35]
    # search_mlp_play_outcome = pickle.load(open("models/search_mlp_play_outcome.p", "rb"))
    search_rf_play_outcome = pickle.load(open("models/search_rf_play_outcome.p", "rb"))
    search_rf_drive_outcome = pickle.load(
        open("models/search_rf_drive_outcome.p", "rb")
    )
    search_mlp_play_outcome_normalized_new_hpo = pickle.load(
        open("models/search_mlp_play_outcome_normalized_new_hpo.p", "rb")
    )
    search_mlp_score_diff_clipped_rf_drive_preds = pickle.load(
        open("models/search_mlp_score_diff_clipped_rf_drive_preds.p", "rb")
    )
    search_mlp_total_score = pickle.load(open("models/search_mlp_total_score.p", "rb"))
    input_names_play_and_drive_preds = search_rf_play_outcome.feature_names_in_
    input_names_score_diff_pred = (
        search_mlp_score_diff_clipped_rf_drive_preds.feature_names_in_
    )
    search_rf_play_class_names = [
        "search_rf_play_" + x for x in search_rf_play_outcome.classes_
    ]
    search_mlp_play_class_names = [
        "search_mlp_play_" + x for x in search_rf_play_outcome.classes_
    ]
    search_rf_drive_class_names = [
        "search_rf_drive_" + x for x in search_rf_drive_outcome.classes_
    ]

    search_rf_play_class_names_home = [x + "_home" for x in search_rf_play_class_names]
    search_rf_play_class_names_away = [x + "_away" for x in search_rf_play_class_names]
    search_rf_drive_class_names_home = [
        x + "_home" for x in search_rf_drive_class_names
    ]
    search_rf_drive_class_names_away = [
        x + "_away" for x in search_rf_drive_class_names
    ]
    fourth_down_data = df[df["down"] == 4]
    mlp_punt_field_position = pickle.load(
        open("models/mlp_punt_field_position.p", "rb")
    )
    logit_field_goal_made = pickle.load(open("models/logit_field_goal_made.p", "rb"))
    mlp_go_for_it_success_next_yds = pickle.load(
        open("models/mlp_go_for_it_success_next_yds.p", "rb")
    )
    fourth_down_data["home_vegas_score_pred"] = (
        fourth_down_data["cur_over_under"] / 2 - 0.5 * fourth_down_data["cur_spread"]
    )
    fourth_down_data["away_vegas_score_pred"] = (
        fourth_down_data["cur_over_under"] / 2 + 0.5 * fourth_down_data["cur_spread"]
    )
    fourth_down_data["kicking_vegas_pred"] = np.where(
        fourth_down_data["home_team_has_ball"] == 1,
        fourth_down_data["home_vegas_score_pred"],
        fourth_down_data["away_vegas_score_pred"],
    )
    fourth_down_data["receiving_vegas_pred"] = np.where(
        fourth_down_data["home_team_has_ball"] == 0,
        fourth_down_data["home_vegas_score_pred"],
        fourth_down_data["away_vegas_score_pred"],
    )
    fourth_down_data[
        "xnext_field_position_if_first_down"
    ] = mlp_go_for_it_success_next_yds.predict(
        fourth_down_data[mlp_go_for_it_success_next_yds.feature_names_in_]
    )

    fourth_down_go_for_it_success_data = deepcopy(fourth_down_data)
    fourth_down_go_for_it_success_data["current_score_diff"] = np.where(
        (fourth_down_data["ytg"] >= fourth_down_data["yd_from_goal"])
        | (fourth_down_data["xnext_field_position_if_first_down"] <= 0.5),
        (2 * fourth_down_data["home_team_has_ball"] - 1) * 6
        + fourth_down_data["current_score_diff"],
        fourth_down_data["current_score_diff"],
    )
    fourth_down_go_for_it_success_data["current_score_total"] = np.where(
        (fourth_down_data["ytg"] >= fourth_down_data["yd_from_goal"])
        | (fourth_down_data["xnext_field_position_if_first_down"] <= 0.5),
        6 + fourth_down_data["current_score_total"],
        fourth_down_data["current_score_total"],
    )
    fourth_down_go_for_it_success_data["ytg"] = np.where(
        (fourth_down_data["ytg"] >= fourth_down_data["yd_from_goal"])
        | (fourth_down_data["xnext_field_position_if_first_down"] <= 0.5),
        -1,
        np.where(
            fourth_down_data["xnext_field_position_if_first_down"] >= 10,
            10,
            fourth_down_data["xnext_field_position_if_first_down"].apply(int),
        ),
    )

    fourth_down_go_for_it_success_data["yd_from_goal"] = np.where(
        (fourth_down_data["ytg"] >= fourth_down_data["yd_from_goal"])
        | (fourth_down_data["xnext_field_position_if_first_down"] <= 0.5),
        -1,
        fourth_down_data["xnext_field_position_if_first_down"]
        .fillna(fourth_down_data["yd_from_goal"] - fourth_down_data["ytg"])
        .apply(int),
    )
    fourth_down_go_for_it_success_data["down"] = np.where(
        (fourth_down_data["ytg"] >= fourth_down_data["yd_from_goal"])
        | (fourth_down_data["xnext_field_position_if_first_down"] <= 0.5),
        0,
        1,
    )
    fourth_down_go_for_it_success_data["point_after_play"] = np.where(
        (fourth_down_data["ytg"] >= fourth_down_data["yd_from_goal"])
        | (fourth_down_data["xnext_field_position_if_first_down"] <= 0.5),
        1,
        0,
    )
    fourth_down_go_for_it_success_data[search_rf_play_class_names] = pd.DataFrame(
        search_rf_play_outcome.predict_proba(
            fourth_down_go_for_it_success_data[
                fourth_down_go_for_it_success_data.down != 0
            ][input_names_play_and_drive_preds]
        ),
        fourth_down_go_for_it_success_data[
            fourth_down_go_for_it_success_data.down != 0
        ].index,
    )
    fourth_down_go_for_it_success_data[
        search_rf_play_class_names
    ] = fourth_down_go_for_it_success_data[search_rf_play_class_names].fillna(0)
    fourth_down_go_for_it_success_data[search_rf_drive_class_names] = pd.DataFrame(
        search_rf_drive_outcome.predict_proba(
            fourth_down_go_for_it_success_data[
                fourth_down_go_for_it_success_data.down != 0
            ][input_names_play_and_drive_preds]
        ),
        fourth_down_go_for_it_success_data[
            fourth_down_go_for_it_success_data.down != 0
        ].index,
    )
    fourth_down_go_for_it_success_data[
        search_rf_drive_class_names
    ] = fourth_down_go_for_it_success_data[search_rf_drive_class_names].fillna(0)
    fourth_down_go_for_it_success_data[
        search_rf_play_class_names_home
    ] = fourth_down_go_for_it_success_data[search_rf_play_class_names].where(
        fourth_down_go_for_it_success_data["home_team_has_ball"] == 1, 0
    )
    fourth_down_go_for_it_success_data[
        search_rf_play_class_names_away
    ] = fourth_down_go_for_it_success_data[search_rf_play_class_names].where(
        fourth_down_go_for_it_success_data["home_team_has_ball"] == 0, 0
    )
    fourth_down_go_for_it_success_data[
        search_rf_drive_class_names_home
    ] = fourth_down_go_for_it_success_data[search_rf_drive_class_names].where(
        fourth_down_go_for_it_success_data["home_team_has_ball"] == 1, 0
    )
    fourth_down_go_for_it_success_data[
        search_rf_drive_class_names_away
    ] = fourth_down_go_for_it_success_data[search_rf_drive_class_names].where(
        fourth_down_go_for_it_success_data["home_team_has_ball"] == 0, 0
    )
    fourth_down_go_for_it_fail_data = deepcopy(fourth_down_data)
    fourth_down_go_for_it_fail_data["event_name"].value_counts()

    fourth_down_go_for_it_fail_data["yd_from_goal"] = (
        100 - fourth_down_data["yd_from_goal"]
    )
    fourth_down_go_for_it_fail_data["ytg"] = np.where(
        fourth_down_go_for_it_fail_data["yd_from_goal"] <= 10,
        fourth_down_go_for_it_fail_data["yd_from_goal"],
        10,
    )
    fourth_down_go_for_it_fail_data["down"] = 1
    fourth_down_go_for_it_fail_data["home_team_has_ball"] = (
        1 - fourth_down_data["home_team_has_ball"]
    )

    # fourth_down_go_for_it_fail_data[search_rf_play_class_names] = np.where(
    #     (fourth_down_data["ytg"]>=fourth_down_data["yd_from_goal"])| (fourth_down_data["xnext_field_position_if_first_down"] <=.5),
    #     search_rf_play_outcome.predict_proba(fourth_down_go_for_it_fail_data[input_names_play_and_drive_preds]), [0, 0, 0, 0, 0, 0, 0]
    # )
    fourth_down_go_for_it_fail_data[search_rf_play_class_names] = pd.DataFrame(
        search_rf_play_outcome.predict_proba(
            fourth_down_go_for_it_fail_data[fourth_down_go_for_it_fail_data.down != 0][
                input_names_play_and_drive_preds
            ]
        ),
        fourth_down_go_for_it_fail_data[
            fourth_down_go_for_it_fail_data.down != 0
        ].index,
    )
    fourth_down_go_for_it_fail_data[
        search_rf_play_class_names
    ] = fourth_down_go_for_it_fail_data[search_rf_play_class_names].fillna(0)
    fourth_down_go_for_it_fail_data[
        search_rf_play_class_names_home
    ] = fourth_down_go_for_it_fail_data[search_rf_play_class_names].where(
        fourth_down_go_for_it_fail_data["home_team_has_ball"] == 1, 0
    )
    fourth_down_go_for_it_fail_data[
        search_rf_play_class_names_away
    ] = fourth_down_go_for_it_fail_data[search_rf_play_class_names].where(
        fourth_down_go_for_it_fail_data["home_team_has_ball"] == 0, 0
    )

    fourth_down_go_for_it_fail_data[search_rf_drive_class_names] = pd.DataFrame(
        search_rf_drive_outcome.predict_proba(
            fourth_down_go_for_it_fail_data[fourth_down_go_for_it_fail_data.down != 0][
                input_names_play_and_drive_preds
            ]
        ),
        fourth_down_go_for_it_fail_data[
            fourth_down_go_for_it_fail_data.down != 0
        ].index,
    )
    fourth_down_go_for_it_fail_data[
        search_rf_drive_class_names
    ] = fourth_down_go_for_it_fail_data[search_rf_drive_class_names].fillna(0)
    fourth_down_go_for_it_fail_data[
        search_rf_drive_class_names_home
    ] = fourth_down_go_for_it_fail_data[search_rf_drive_class_names].where(
        fourth_down_go_for_it_fail_data["home_team_has_ball"] == 1, 0
    )
    fourth_down_go_for_it_fail_data[
        search_rf_drive_class_names_away
    ] = fourth_down_go_for_it_fail_data[search_rf_drive_class_names].where(
        fourth_down_go_for_it_fail_data["home_team_has_ball"] == 0, 0
    )

    fourth_down_field_goal_success_data = deepcopy(fourth_down_data)
    fourth_down_field_goal_success_data["event_name"].value_counts()

    fourth_down_field_goal_success_data["current_score_diff"] = (
        2 * fourth_down_data["home_team_has_ball"] - 1
    ) * 3 + fourth_down_data["current_score_diff"]
    fourth_down_field_goal_success_data["current_score_total"] = (
        3 + fourth_down_data["current_score_total"]
    )
    fourth_down_field_goal_success_data["ytg"] = -1
    fourth_down_field_goal_success_data["yd_from_goal"] = -1
    fourth_down_field_goal_success_data["down"] = 0
    fourth_down_field_goal_success_data["home_team_has_ball"] = (
        1 - fourth_down_data["home_team_has_ball"]
    )
    fourth_down_field_goal_success_data["kick_off"] = 1
    # fourth_down_field_goal_success_data[search_rf_play_class_names] = np.where(
    #     (fourth_down_data["ytg"]>=fourth_down_data["yd_from_goal"])| (fourth_down_data["xnext_field_position_if_first_down"] <=.5),
    #     search_rf_play_outcome.predict_proba(fourth_down_field_goal_success_data[input_names_play_and_drive_preds]), [0, 0, 0, 0, 0, 0, 0]
    # )
    fourth_down_field_goal_success_data[search_rf_play_class_names_away] = 0
    fourth_down_field_goal_success_data[search_rf_play_class_names_home] = 0
    fourth_down_field_goal_success_data[search_rf_drive_class_names_away] = 0
    fourth_down_field_goal_success_data[search_rf_drive_class_names_home] = 0
    fourth_down_field_goal_fail_data = deepcopy(fourth_down_data)

    # fourth_down_field_goal_fail_data["yd_from_goal"] = 100 - (fourth_down_data["yd_from_goal"] + 7)
    fourth_down_field_goal_fail_data["yd_from_goal"] = np.where(
        fourth_down_data["yd_from_goal"] >= 20,
        100 - (fourth_down_data["yd_from_goal"] + 7),
        80,
    )
    fourth_down_field_goal_fail_data["ytg"] = np.where(
        fourth_down_field_goal_fail_data["yd_from_goal"] <= 10,
        fourth_down_field_goal_fail_data["yd_from_goal"],
        10,
    )
    fourth_down_field_goal_fail_data["down"] = 1
    fourth_down_field_goal_fail_data["home_team_has_ball"] = (
        1 - fourth_down_data["home_team_has_ball"]
    )

    # fourth_down_field_goal_fail_data[search_rf_play_class_names] = np.where(
    #     (fourth_down_data["ytg"]>=fourth_down_data["yd_from_goal"])| (fourth_down_data["xnext_field_position_if_first_down"] <=.5),
    #     search_rf_play_outcome.predict_proba(fourth_down_field_goal_fail_data[input_names_play_and_drive_preds]), [0, 0, 0, 0, 0, 0, 0]
    # )
    fourth_down_field_goal_fail_data[search_rf_play_class_names] = pd.DataFrame(
        search_rf_play_outcome.predict_proba(
            fourth_down_field_goal_fail_data[
                fourth_down_field_goal_fail_data.down != 0
            ][input_names_play_and_drive_preds]
        ),
        fourth_down_field_goal_fail_data[
            fourth_down_field_goal_fail_data.down != 0
        ].index,
    )
    fourth_down_field_goal_fail_data[
        search_rf_play_class_names
    ] = fourth_down_field_goal_fail_data[search_rf_play_class_names].fillna(0)
    fourth_down_field_goal_fail_data[
        search_rf_play_class_names_home
    ] = fourth_down_field_goal_fail_data[search_rf_play_class_names].where(
        fourth_down_field_goal_fail_data["home_team_has_ball"] == 1, 0
    )
    fourth_down_field_goal_fail_data[
        search_rf_play_class_names_away
    ] = fourth_down_field_goal_fail_data[search_rf_play_class_names].where(
        fourth_down_field_goal_fail_data["home_team_has_ball"] == 0, 0
    )

    fourth_down_field_goal_fail_data[search_rf_drive_class_names] = pd.DataFrame(
        search_rf_drive_outcome.predict_proba(
            fourth_down_field_goal_fail_data[
                fourth_down_field_goal_fail_data.down != 0
            ][input_names_play_and_drive_preds]
        ),
        fourth_down_field_goal_fail_data[
            fourth_down_field_goal_fail_data.down != 0
        ].index,
    )
    fourth_down_field_goal_fail_data[
        search_rf_drive_class_names
    ] = fourth_down_field_goal_fail_data[search_rf_drive_class_names].fillna(0)
    fourth_down_field_goal_fail_data[
        search_rf_drive_class_names_home
    ] = fourth_down_field_goal_fail_data[search_rf_drive_class_names].where(
        fourth_down_field_goal_fail_data["home_team_has_ball"] == 1, 0
    )
    fourth_down_field_goal_fail_data[
        search_rf_drive_class_names_away
    ] = fourth_down_field_goal_fail_data[search_rf_drive_class_names].where(
        fourth_down_field_goal_fail_data["home_team_has_ball"] == 0, 0
    )
    fourth_down_punt_data = deepcopy(fourth_down_data)
    punt_prediction_inputs = normalize_df(
        fourth_down_data[mlp_punt_field_position.feature_names_in_],
        fourth_down_data[
            (fourth_down_data["punt"] == 1) & (fourth_down_data["season"] < 2020)
        ],
    ).dropna()
    fourth_down_punt_data["xpunt_opp_field_position"] = mlp_punt_field_position.predict(
        punt_prediction_inputs
    )

    fourth_down_punt_data["yd_from_goal"] = fourth_down_punt_data[
        "xpunt_opp_field_position"
    ].fillna(80)
    fourth_down_punt_data["ytg"] = np.where(
        fourth_down_punt_data["yd_from_goal"] <= 10,
        fourth_down_punt_data["yd_from_goal"],
        10,
    )
    fourth_down_punt_data["down"] = 1
    fourth_down_punt_data["home_team_has_ball"] = (
        1 - fourth_down_data["home_team_has_ball"]
    )

    # fourth_down_punt_data[search_rf_play_class_names] = np.where(
    #     (fourth_down_data["ytg"]>=fourth_down_data["yd_from_goal"])| (fourth_down_data["xnext_field_position_if_first_down"] <=.5),
    #     search_rf_play_outcome.predict_proba(fourth_down_punt_data[input_names_play_and_drive_preds]), [0, 0, 0, 0, 0, 0, 0]
    # )
    fourth_down_punt_data[search_rf_play_class_names] = pd.DataFrame(
        search_rf_play_outcome.predict_proba(
            fourth_down_punt_data[fourth_down_punt_data.down != 0][
                input_names_play_and_drive_preds
            ]
        ),
        fourth_down_punt_data[fourth_down_punt_data.down != 0].index,
    )
    fourth_down_punt_data[search_rf_play_class_names] = fourth_down_punt_data[
        search_rf_play_class_names
    ].fillna(0)
    fourth_down_punt_data[search_rf_play_class_names_home] = fourth_down_punt_data[
        search_rf_play_class_names
    ].where(fourth_down_punt_data["home_team_has_ball"] == 1, 0)
    fourth_down_punt_data[search_rf_play_class_names_away] = fourth_down_punt_data[
        search_rf_play_class_names
    ].where(fourth_down_punt_data["home_team_has_ball"] == 0, 0)

    fourth_down_punt_data[search_rf_drive_class_names] = pd.DataFrame(
        search_rf_drive_outcome.predict_proba(
            fourth_down_punt_data[fourth_down_punt_data.down != 0][
                input_names_play_and_drive_preds
            ]
        ),
        fourth_down_punt_data[fourth_down_punt_data.down != 0].index,
    )
    fourth_down_punt_data[search_rf_drive_class_names] = fourth_down_punt_data[
        search_rf_drive_class_names
    ].fillna(0)
    fourth_down_punt_data[search_rf_drive_class_names_home] = fourth_down_punt_data[
        search_rf_drive_class_names
    ].where(fourth_down_punt_data["home_team_has_ball"] == 1, 0)
    fourth_down_punt_data[search_rf_drive_class_names_away] = fourth_down_punt_data[
        search_rf_drive_class_names
    ].where(fourth_down_punt_data["home_team_has_ball"] == 0, 0)
    mask_model = (
        (df.continuation == 0)
        & (df[input_names_score_diff_pred].notna().all(axis=1))
        & ~(df.event_id.isin([12, 57, 58, 13]))
        & (df["overtime"] == 0)
    )

    anchor_df = df[mask_model & (df.season < 2020)]

    score_diff_change_list_clipped = list(
        df.end_of_regulation_score_diff_change_clipped.drop_duplicates().sort_values()
    )
    score_diff_clipped_mlp_drive_preds_matrix = pd.DataFrame(
        np.zeros((len(fourth_down_data), len(score_diff_change_list_clipped))),
        index=fourth_down_data.index,
    )

    for column in score_diff_clipped_mlp_drive_preds_matrix.columns:
        score_diff_clipped_mlp_drive_preds_matrix[column] = (
            score_diff_change_list_clipped[column]
            + fourth_down_data["current_score_diff"]
        )

    normalized_data_dicts = {}
    predictions_dicts = {}
    win_prob_dict = {}
    for outcome in [
        "fourth_down_go_for_it_success_data",
        "fourth_down_go_for_it_fail_data",
        "fourth_down_field_goal_success_data",
        "fourth_down_field_goal_fail_data",
        "fourth_down_punt_data",
    ]:
        # print(outcome)
        normalized_data_dicts[outcome + "_normalized"] = normalize_df(
            eval(outcome)[input_names_score_diff_pred], anchor_df
        )
        predictions_dicts[outcome] = pd.DataFrame(
            search_mlp_score_diff_clipped_rf_drive_preds.predict_proba(
                normalized_data_dicts[outcome + "_normalized"]
            ),
            normalized_data_dicts[outcome + "_normalized"].index,
        )
        score_diff_clipped_mlp_drive_preds_matrix = pd.DataFrame(
            np.zeros((len(fourth_down_data), len(score_diff_change_list_clipped))),
            index=fourth_down_data.index,
        )
        for column in score_diff_clipped_mlp_drive_preds_matrix.columns:
            score_diff_clipped_mlp_drive_preds_matrix[column] = (
                score_diff_change_list_clipped[column]
                + eval(outcome)["current_score_diff"]
            )
        win_prob_dict[outcome] = pd.DataFrame()
        win_prob_dict[outcome]["xhome_win_mlp_search_clipped_mlp_drive_preds"] = np.sum(
            predictions_dicts[outcome].T[
                score_diff_clipped_mlp_drive_preds_matrix.T > 0
            ],
            axis=0,
        )
        win_prob_dict[outcome]["xovertime_mlp_search_clipped_mlp_drive_preds"] = np.sum(
            predictions_dicts[outcome].T[
                score_diff_clipped_mlp_drive_preds_matrix.T == 0
            ],
            axis=0,
        )
        win_prob_dict[outcome]["xaway_win_mlp_search_clipped_mlp_drive_preds"] = np.sum(
            predictions_dicts[outcome].T[
                score_diff_clipped_mlp_drive_preds_matrix.T < 0
            ],
            axis=0,
        )
    outcome_list = []
    for outcome in [
        "fourth_down_go_for_it_success_data",
        "fourth_down_go_for_it_fail_data",
        "fourth_down_field_goal_success_data",
        "fourth_down_field_goal_fail_data",
        "fourth_down_punt_data",
    ]:
        fourth_down_data[
            [outcome + x for x in ["_home_win", "_overtime", "_away_win"]]
        ] = win_prob_dict[outcome]
        outcome_list = outcome_list + [
            outcome + x for x in ["_home_win", "_overtime", "_away_win"]
        ]
    mask_model = (
        (df.continuation == 0)
        & (df.down != 0)
        & (df[input_names_play_and_drive_preds].notna().all(axis=1))
        & (df["from_scrimmage"] == 1)
        & (df["overtime"] == 0)
    )
    anchor_df_mlp_plays = df[
        mask_model
        & (df["season"] < 2020)
        & (df.play_counts == 1)
        & (df.event_id.isin(scrimmage_plays_we_want))
    ]
    fourth_down_data_normalized = normalize_df(
        fourth_down_data[input_names_play_and_drive_preds], anchor_df_mlp_plays
    )
    fourth_down_data[
        search_mlp_play_class_names
    ] = search_mlp_play_outcome_normalized_new_hpo.predict_proba(
        fourth_down_data_normalized
    )

    fourth_down_success_rates = pd.DataFrame()
    fourth_down_success_rates["go_for_it"] = np.sum(
        fourth_down_data[
            ["search_mlp_play_first_down", "search_mlp_play_offensive_touchdown"]
        ],
        axis=1,
    ) / np.sum(
        fourth_down_data[
            [
                "search_mlp_play_first_down",
                "search_mlp_play_offensive_touchdown",
                "search_mlp_play_turnover",
                "search_mlp_play_none",
            ]
        ],
        axis=1,
    )
    fourth_down_data["offense_point_diff"] = np.where(
        fourth_down_data["home_team_has_ball"] == 1,
        fourth_down_data["current_score_diff"],
        -fourth_down_data["current_score_diff"],
    )
    fourth_down_data["yd_from_goal_sq"] = fourth_down_data["yd_from_goal"] ** 2
    fourth_down_data["yd_from_goal_cu"] = fourth_down_data["yd_from_goal"] ** 3
    field_goal_data = fourth_down_data[
        (fourth_down_data["field_goal_attempt"] == 1)
        & (fourth_down_data["play_counts"] == 1)
    ].reset_index(drop=True)
    field_goal_prediction_inputs = normalize_df(
        fourth_down_data[logit_field_goal_made.feature_names_in_],
        field_goal_data[field_goal_data["season"] < 2020],
    ).dropna()
    fourth_down_data[
        ["xfield_goal_missed", "xfield_goal_made"]
    ] = logit_field_goal_made.predict_proba(field_goal_prediction_inputs)

    fourth_down_success_rates["field_goal"] = fourth_down_data["xfield_goal_made"]
    fourth_down_data[
        ["go_for_it_success", "field_goal_success"]
    ] = fourth_down_success_rates[["go_for_it", "field_goal"]]
    fourth_down_data["punt_success"] = np.nan
    outcome_list_team_with_ball = [
        "fourth_down_go_for_it_success_data_win",
        "fourth_down_go_for_it_fail_data_win",
        "fourth_down_field_goal_success_data_win",
        "fourth_down_field_goal_fail_data_win",
        "fourth_down_punt_data_win",
    ]
    outcome_list_home = [
        "fourth_down_go_for_it_success_data_home_win",
        "fourth_down_go_for_it_fail_data_home_win",
        "fourth_down_field_goal_success_data_home_win",
        "fourth_down_field_goal_fail_data_home_win",
        "fourth_down_punt_data_home_win",
    ]
    outcome_list_away = [
        "fourth_down_go_for_it_success_data_away_win",
        "fourth_down_go_for_it_fail_data_away_win",
        "fourth_down_field_goal_success_data_away_win",
        "fourth_down_field_goal_fail_data_away_win",
        "fourth_down_punt_data_away_win",
    ]
    outcome_list_overtime = [
        "fourth_down_go_for_it_success_data_overtime",
        "fourth_down_go_for_it_fail_data_overtime",
        "fourth_down_field_goal_success_data_overtime",
        "fourth_down_field_goal_fail_data_overtime",
        "fourth_down_punt_data_overtime",
    ]
    for x in range(len(outcome_list_team_with_ball)):
        fourth_down_data[outcome_list_team_with_ball[x]] = np.where(
            fourth_down_data["home_team_has_ball"] == 1,
            fourth_down_data[outcome_list_home[x]],
            fourth_down_data[outcome_list_away[x]],
        ) + (0.5 * fourth_down_data[outcome_list_overtime[x]])
    fourth_down_data["x_win_go_for_it"] = fourth_down_data[
        "fourth_down_go_for_it_success_data_win"
    ] * fourth_down_success_rates["go_for_it"] + fourth_down_data[
        "fourth_down_go_for_it_fail_data_win"
    ] * (
        1 - fourth_down_success_rates["go_for_it"]
    )
    fourth_down_data["x_win_field_goal"] = fourth_down_data[
        "fourth_down_field_goal_success_data_win"
    ] * fourth_down_success_rates["field_goal"] + fourth_down_data[
        "fourth_down_field_goal_fail_data_win"
    ] * (
        1 - fourth_down_success_rates["field_goal"]
    )
    fourth_down_data["x_win_punt"] = fourth_down_data["fourth_down_punt_data_win"]

    return fourth_down_data


@st.cache
def load_predictions(cache=True):
    scrimmage_plays_we_want = [1, 2, 3, 4, 7, 9, 14, 17, 18, 35]
    # search_mlp_play_outcome = pickle.load(open("models/search_mlp_play_outcome.p", "rb"))
    search_rf_play_outcome = pickle.load(open("models/search_rf_play_outcome.p", "rb"))
    search_rf_drive_outcome = pickle.load(
        open("models/search_rf_drive_outcome.p", "rb")
    )
    search_mlp_play_outcome_normalized_new_hpo = pickle.load(
        open("models/search_mlp_play_outcome_normalized_new_hpo.p", "rb")
    )
    search_mlp_score_diff_clipped_rf_drive_preds = pickle.load(
        open("models/search_mlp_score_diff_clipped_rf_drive_preds.p", "rb")
    )
    search_mlp_total_score = pickle.load(open("models/search_mlp_total_score.p", "rb"))
    input_names_play_and_drive_preds = search_rf_play_outcome.feature_names_in_
    input_names_score_diff_pred = (
        search_mlp_score_diff_clipped_rf_drive_preds.feature_names_in_
    )
    search_rf_play_class_names = [
        "search_rf_play_" + x for x in search_rf_play_outcome.classes_
    ]
    search_mlp_play_class_names = [
        "search_mlp_play_" + x for x in search_rf_play_outcome.classes_
    ]
    search_rf_drive_class_names = [
        "search_rf_drive_" + x for x in search_rf_drive_outcome.classes_
    ]

    search_rf_play_class_names_home = [x + "_home" for x in search_rf_play_class_names]
    search_rf_play_class_names_away = [x + "_away" for x in search_rf_play_class_names]
    search_rf_drive_class_names_home = [
        x + "_home" for x in search_rf_drive_class_names
    ]
    search_rf_drive_class_names_away = [
        x + "_away" for x in search_rf_drive_class_names
    ]
    event_df = get_event_data(cache=cache)
    game_df = get_game_data(cache=cache)
    prior_df = pd.read_csv(os.path.join(data_dir, "game_priors.csv"))
    odds_df = pd.read_parquet(os.path.join(data_dir, "odds_data.parquet"))
    odds_df = odds_df.drop_duplicates("game_code")
    event_df[["cur_spread", "cur_over_under"]] = event_df.merge(
        odds_df, how="left", on="game_code"
    )[["cur_spread", "cur_over_under"]].fillna(
        {
            "cur_spread": np.mean(odds_df["cur_spread"]),
            "cur_over_under": np.mean(odds_df["cur_over_under"]),
        }
    )
    event_df = event_df.pipe(add_timeouts_remaining)
    # event_df["end_of_regulation_score_total_diff"] =

    model_df = deepcopy(event_df)
    # print(event_df.columns)
    model_df["time_left_in_half"] = event_df["time_left_in_game"] - (
        (2 - event_df["half"]) * 1800
    )
    model_df["from_scrimmage"] = np.where(
        event_df["event_id"].isin([22, 47, 52, 53, 54, 55, 56]),
        0,
        event_df["from_scrimmage"],
    )
    model_df["point_after_play"] = np.where(
        model_df["point_after_kick"] + model_df["two_point_attempt"] == 1, 1, 0
    )
    model_df["down"] = np.where(model_df["from_scrimmage"] == 0, 0, event_df["down"])
    model_df["ytg"] = np.where(model_df["from_scrimmage"] == 0, -1, event_df["ytg"])
    model_df["yd_from_goal"] = np.where(
        model_df["from_scrimmage"] == 0, -1, event_df["yd_from_goal"]
    )
    model_df["home_team_has_ball"] = np.where(
        event_df["event_id"].isin([5]),
        1 - event_df["home_team_has_ball"],
        event_df["home_team_has_ball"],
    )
    mask_model_predict = (
        (model_df.continuation == 0)
        & (model_df.down != 0)
        & (model_df[input_names_play_and_drive_preds].notna().all(axis=1))
        & (model_df["from_scrimmage"] == 1)
        & (model_df["overtime"] == 0)
    )
    model_df = model_df.sort_values(["game_date", "nevent"], ascending=[False, True])

    # model_df = model_df[model_df["game_code"]==2337728]
    model_df["game_info"] = (
        model_df["home_team"]
        + " "
        + model_df["away_team"]
        + " "
        + model_df["game_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        + " "
        + model_df["season"].apply(str)
        + " ("
        + (model_df["game_code"]).apply(str)
        + ")"
    )
    model_df["absolute_score_diff"] = abs(
        model_df["home_start_score"] - model_df["away_start_score"]
    )

    model_df["minutes"] = (model_df["play_start_time"].fillna(0) // 60).apply(int)
    model_df["seconds"] = (
        model_df["play_start_time"].fillna(0)
        - (model_df["play_start_time"].fillna(0) // 60) * 60
    ).apply(int)
    model_df["seconds_str"] = np.where(
        model_df["seconds"] >= 10,
        model_df["seconds"].apply(str),
        "0" + model_df["seconds"].apply(str),
    )
    model_df["time_str"] = (
        model_df["minutes"].apply(str) + ":" + model_df["seconds_str"]
    )

    model_df["team_score_desc"] = np.where(
        model_df["home_team_has_ball"] == 1,
        np.where(
            model_df["home_start_score"] > model_df["away_start_score"],
            "Up by " + model_df["absolute_score_diff"].apply(str),
            np.where(
                model_df["home_start_score"] < model_df["away_start_score"],
                "Down by " + model_df["absolute_score_diff"].apply(str),
                "Tied",
            ),
        ),
        np.where(
            model_df["home_start_score"] < model_df["away_start_score"],
            "Up by " + model_df["absolute_score_diff"].apply(str),
            np.where(
                model_df["home_start_score"] > model_df["away_start_score"],
                "Down by " + model_df["absolute_score_diff"].apply(str),
                "Tied",
            ),
        ),
    )
    model_df["play_description"] = (
        ordinaltg(model_df["quarter"])
        + " Qtr "
        + model_df["minutes"].apply(str)
        + ":"
        + model_df["seconds_str"]
        + ", "
        + model_df["team_score_desc"]
        + ", "
        + ordinaltg(model_df["down"]).apply(str)
        + " & "
        + model_df["ytg"].apply(str)
        + ", "
        + model_df["yd_from_goal"].apply(str)
        + " Yards From Goal, "
        + np.where(
            model_df["home_team_has_ball"] == 1,
            model_df["home_team_abbrev"],
            model_df["away_team_abbrev"],
        )
        + " has ball, "
        + "Off TO: "
        + np.where(
            model_df["home_team_has_ball"] == 1,
            model_df["home_timeouts_remaining"],
            model_df["away_timeouts_remaining"],
        ).astype(str)
        + ", Def TO: "
        + np.where(
            model_df["home_team_has_ball"] == 0,
            model_df["home_timeouts_remaining"],
            model_df["away_timeouts_remaining"],
        ).astype(str)
        + " ("
        + model_df["nevent"].apply(str)
        + ")"
    )

    # print(model_df.head())
    # fourth_downs_only = model_df.loc[model_df.down == 4]
    # breakpoint()
    model_df[search_rf_play_class_names] = pd.DataFrame(
        search_rf_play_outcome.predict_proba(
            model_df[mask_model_predict][input_names_play_and_drive_preds]
        ),
        index=model_df[mask_model_predict].index,
    )
    model_df[search_rf_play_class_names] = model_df[search_rf_play_class_names].fillna(
        0
    )
    model_df[search_rf_drive_class_names] = pd.DataFrame(
        search_rf_drive_outcome.predict_proba(
            model_df[mask_model_predict][input_names_play_and_drive_preds]
        ),
        index=model_df[mask_model_predict].index,
    )
    model_df[search_rf_drive_class_names] = model_df[
        search_rf_drive_class_names
    ].fillna(0)

    model_df[search_rf_play_class_names_home] = model_df[
        search_rf_play_class_names
    ].where(model_df.home_team_has_ball == 1, 0)
    model_df[search_rf_play_class_names_away] = model_df[
        search_rf_play_class_names
    ].where(model_df.home_team_has_ball == 0, 0)
    model_df[search_rf_drive_class_names_home] = model_df[
        search_rf_drive_class_names
    ].where(model_df.home_team_has_ball == 1, 0)
    model_df[search_rf_drive_class_names_away] = model_df[
        search_rf_drive_class_names
    ].where(model_df.home_team_has_ball == 0, 0)
    mask_model_score_diff = (
        (model_df.continuation == 0)
        & (model_df[input_names_score_diff_pred].notna().all(axis=1))
        & ~(model_df.event_id.isin([12, 57, 58, 13]))
        & (model_df["overtime"] == 0)
    )
    normalized_score_pred_df = normalize_df(
        model_df[mask_model_score_diff][input_names_score_diff_pred],
        model_df[mask_model_score_diff & (model_df.season < 2020)][
            input_names_score_diff_pred
        ],
    )
    # st.dataframe(normalized_score_pred_df)
    mlp_search_score_diff_clipped_rf_drive_preds_preds = pd.DataFrame(
        search_mlp_score_diff_clipped_rf_drive_preds.predict_proba(
            normalized_score_pred_df.values
        ),
        index=model_df[mask_model_score_diff].index,
    )
    score_diff_clipped_rf_drive_preds_matrix = pd.DataFrame(
        np.zeros(mlp_search_score_diff_clipped_rf_drive_preds_preds.shape),
        index=mlp_search_score_diff_clipped_rf_drive_preds_preds.index,
    )
    model_df["end_of_regulation_score_diff_change_clipped"] = np.clip(
        model_df["end_of_regulation_score_diff_change"], -35, 35
    )
    score_diff_change_list_clipped = list(
        model_df["end_of_regulation_score_diff_change_clipped"]
        .drop_duplicates()
        .sort_values()
    )
    for column in score_diff_clipped_rf_drive_preds_matrix.columns:
        score_diff_clipped_rf_drive_preds_matrix[column] = (
            score_diff_change_list_clipped[column] + model_df["current_score_diff"]
        )

    model_df["xhome_win_mlp_search_clipped_rf_drive_preds"] = np.sum(
        mlp_search_score_diff_clipped_rf_drive_preds_preds.T[
            score_diff_clipped_rf_drive_preds_matrix.T > 0
        ],
        axis=0,
    )
    model_df["xovertime_mlp_search_clipped_rf_drive_preds"] = np.sum(
        mlp_search_score_diff_clipped_rf_drive_preds_preds.T[
            score_diff_clipped_rf_drive_preds_matrix.T == 0
        ],
        axis=0,
    )
    model_df["xaway_win_mlp_search_clipped_rf_drive_preds"] = np.sum(
        mlp_search_score_diff_clipped_rf_drive_preds_preds.T[
            score_diff_clipped_rf_drive_preds_matrix.T < 0
        ],
        axis=0,
    )
    model_df[
        "xend_of_regulation_score_diff_mlp_search_clipped_rf_drive_preds"
    ] = np.sum(
        score_diff_clipped_rf_drive_preds_matrix
        * mlp_search_score_diff_clipped_rf_drive_preds_preds,
        axis=1,
    )
    fourth_down_data = get_fourth_down_predictions(model_df)
    # st.dataframe(model_df.head(50))
    return model_df, fourth_down_data


model_df, fourth_down_data = load_predictions()

# model_df.to_csv("model_df.csv")
if chart_select_box == "4th Down Bot Situation":
    games = fourth_down_data["game_info"].drop_duplicates()
    game_selection = st.selectbox("Pick a Game", games)
    plays = fourth_down_data[fourth_down_data["game_info"] == game_selection][
        "play_description"
    ]
    play_selection = st.selectbox("Pick a Play", plays)

    output_df = pd.DataFrame()
    play_df = fourth_down_data.loc[fourth_down_data["play_description"]== play_selection]
    print(play_df["event_name"].values[0], "for", int(play_df["yards_gained"].values[0]), "yards")
    print(abs(play_df["cur_spread"].values[0]), "Point", np.where((play_df["home_team_has_ball"] * 2 - 1) * play_df["cur_spread"].values[0]>0, "Underdogs", "favorites")[0])
    output_df["Win %"] = play_df[["x_win_go_for_it", "x_win_field_goal", "x_win_punt"]].values[0].round(3) * 100
    output_df["Success %"] = play_df[["go_for_it_success", "field_goal_success", "punt_success"]].values[0].round(3) * 100
    output_df.loc[2] = np.where(play_df["yd_from_goal"]<=30, np.nan, output_df.loc[2])
    output_df = output_df.sort_values("Win %", ascending=False)
    conditional_df = pd.DataFrame()
    conditional_df["Fail"] = play_df[["fourth_down_go_for_it_fail_data_win", "fourth_down_field_goal_fail_data_win", "punt_success"]].values[0].round(3) * 100
    conditional_df["Success"] = play_df[["fourth_down_go_for_it_success_data_win", "fourth_down_field_goal_success_data_win", "punt_success"]].values[0].round(3) * 100
    # output_df.loc["Field Goal Attempt"] = np.where(play_df["yd_from_goal"]>=55, None, output_df.loc["Field Goal Attempt"])
    output_df = pd.concat([output_df.T, conditional_df.T], keys=["", "Win % if"])
    output_df = output_df.rename(columns={0: "Go For it", 1: "Field Goal Attempt", 2: "Punt"})
    st.dataframe(output_df.T)

elif chart_select_box == "Games":
    games = model_df["game_info"].drop_duplicates()
    game_selection = st.selectbox("Pick a Game", games)
    game_df = model_df[
        (model_df["game_info"] == game_selection)
        & (
            model_df["event_id"].isin(
                [1, 2, 3, 4, 5, 7, 9, 14, 17, 18, 22, 35, 41, 47, 52, 53, 54, 55, 56]
            )
        )
        & (model_df["continuation"] == 0)
        & (model_df["overtime"] == 0)
    ]
    # MODEL_TYPE_SELECT = ["MLP", "RF"]
    game_df["nevent"] = range(len(game_df))
    # model_type_selection = st.selectbox("Model", MODEL_TYPE_SELECT)
    x = "nevent"
    hover_values = [
        "home_team_has_ball",
        "home_start_score",
        "away_start_score",
        "quarter",
        "time_str",
        "yd_from_goal",
        "down",
        "ytg",
        "home_timeouts_remaining",
        "away_timeouts_remaining",
    ]
    v = pd.DataFrame(game_df[["quarter", "nevent"]]).reset_index(drop=True)
    # print(v)
    # breakpoint()
    mask_ticks = v["quarter"][1:].reset_index(drop=True) == v["quarter"][
        :-1
    ].reset_index(drop=True)
    # print(mask_ticks)
    # breakpoint()
    ticks_idx = [min(v["nevent"])] + list(v[:-1][~mask_ticks]["nevent"])
    if len(ticks_idx) == 4:
        ticks_values = [1, 2, 3, 4]
    else:
        ticks_values = [1, 2, 3, 4, "OT"]

    # print("ticks_values: ", ticks_values)
    # print("ticks_idx: ", ticks_idx)

    fig = go.Figure()
    # game_df["xhome_win_mlp_no_tie"] = game_df["xhome_win_mlp"] / (
    #     game_df["xhome_win_mlp"] + game_df["xaway_win_mlp"]
    # )
    # game_df["xhome_win_rf_no_tie"] = game_df["xhome_win_rf"] / (
    #     game_df["xhome_win_rf"] + game_df["xaway_win_rf"]
    # )
    # game_df["xaway_win_mlp_no_tie"] = 1 - game_df["xhome_win_mlp_no_tie"]
    # game_df["xaway_win_rf_no_tie"] = 1 - game_df["xhome_win_rf_no_tie"]
    # if model_type_selection == "MLP":
    #     if game_df["game_type_id"].tolist()[0] == 1:
    #         y = ["xhome_win_mlp", "xdraw_mlp", "xaway_win_mlp"]
    #     else:
    #         y = ["xhome_win_mlp_no_tie", "xaway_win_mlp_no_tie"]
    #     # y = "xhome_win_mlp"
    # elif model_type_selection == "RF":
    #     if game_df["game_type_id"].tolist()[0] == 1:
    #         y = ["xhome_win_rf", "xdraw_rf", "xaway_win_rf"]
    #     else:
    #         y = ["xhome_win_rf_no_tie", "xaway_win_rf_no_tie"]
    #     # y = "xhome_win_rf"
    y = [
        "xhome_win_mlp_search_clipped_rf_drive_preds",
        "xovertime_mlp_search_clipped_rf_drive_preds",
        "xaway_win_mlp_search_clipped_rf_drive_preds",
    ]
    colors = ["khaki", "lightgray", "lightskyblue"]
    # print(pd.to_numeric(game_df["yards_gained"], errors="coerce"))

    game_df["yards_description"] = (
        game_df["event_name"]
        + " for "
        + game_df["yards_gained"].fillna(0).apply(int).apply(str)
        + " yards"
    )
    fig.add_trace(
        go.Scatter(
            x=game_df["nevent"],
            y=game_df[y[0]],
            customdata=game_df[["yards_description", "play_description"]],
            stackgroup="one",
            mode="lines",
            line=dict(width=0.5, color=colors[0]),
            name=game_df["home_team_abbrev"].tolist()[0],
            hovertemplate="<br>".join(
                [
                    "%{y}",
                    "%{customdata[0]}",
                    "%{customdata[1]}",
                ]
            ),
        )
    )
    if True:
        fig.add_trace(
            go.Scatter(
                x=game_df["nevent"],
                y=game_df[y[1]],
                hovertext=game_df[hover_values],
                stackgroup="one",
                mode="lines",
                line=dict(width=0.5, color=colors[1]),
                name="overtime",
            )
        )
        # print(y)
        fig.add_trace(
            go.Scatter(
                x=game_df["nevent"],
                y=game_df[y[2]],
                hovertext=game_df[hover_values],
                stackgroup="one",
                mode="lines",
                line=dict(width=0.5, color=colors[2]),
                name=game_df["away_team_abbrev"].tolist()[0],
            )
        )

    # fig.add_trace(go.Scatter(text=game_df))
    # fig = px.area(
    #     game_df,
    #     x="nevent",
    #     y=y,
    #     custom_data=hover_values,
    #     # hover_name="event_name",
    #     # hover_data=hover_values,
    # )
    # fig.add_trace(
    #     go.Scatter(hovertext=game_df[hover_values], hovertype="text")
    # hovertemplate="<br>".join(
    #     [
    #         "Col1: %{customdata[0]}",
    #         "Col2: %{customdata[1]}",
    #         "Col3: %{customdata[2]}",
    #     ]
    # ),
    # )
    # # fig.add_trace(
    # #     go.Scatter(customdata=game_df[hover_values], hovertext=game_df[hover_values])
    # # )
    fig.update_layout(hovermode="x unified")
    fig.update_layout(
        xaxis=dict(
            tickvals=ticks_idx, ticktext=ticks_values, gridcolor="black", gridwidth=2
        ),
        yaxis=dict(tick0=0, dtick=0.25, gridcolor="black", gridwidth=2),
    )
    # print(game_df.columns.tolist())
    game_df["score_change"] = game_df["home_score_added"] + game_df["away_score_added"]
    game_df["score_str"] = (
        (game_df["away_score_added"] + game_df["away_start_score"]).apply(str)
        + "-"
        + (game_df["home_score_added"] + game_df["home_start_score"]).apply(str)
    )
    scores_idx_home = game_df[game_df["home_score_added"] >= 3]["nevent"].tolist()
    score_y_home = game_df[game_df["home_score_added"] >= 3][y[0]].tolist()
    score_home_value = game_df[game_df["home_score_added"] >= 3]["home_score_added"]
    score_home_str = np.where(score_home_value == 3, " FG", " TD")
    score_display_home = game_df[game_df["home_score_added"] >= 3]["score_str"].tolist()
    scores_idx_away = game_df[game_df["away_score_added"] >= 3]["nevent"].tolist()
    score_y_away = (game_df[game_df["away_score_added"] >= 3][y[0]]).tolist()
    score_away_value = game_df[game_df["away_score_added"] >= 3]["away_score_added"]
    score_away_str = np.where(score_away_value == 3, " FG", " TD")
    score_display_away = game_df[game_df["away_score_added"] >= 3]["score_str"].tolist()
    mask_poss_change = (
        (game_df["home_team_has_ball"].shift(-1) != game_df["home_team_has_ball"])
        & (game_df["score_change"] == 0)
        & (game_df["score_change"].shift(-1) == 0)
    )
    ball_change_idx = game_df[mask_poss_change]["nevent"].tolist()
    ball_change_cause = game_df[mask_poss_change]["event_name"].tolist()
    ball_change_y = (game_df[mask_poss_change][y[0]]).tolist()
    for x in range(len(scores_idx_home)):
        fig.add_annotation(
            x=scores_idx_home[x],
            y=score_y_home[x],
            text=game_df["home_team_abbrev"].tolist()[0]
            + score_home_str[x]
            + " "
            + score_display_home[x],
            showarrow=True,
        )
    for x in range(len(scores_idx_away)):
        fig.add_annotation(
            x=scores_idx_away[x],
            y=score_y_away[x],
            text=game_df["away_team_abbrev"].tolist()[0]
            + score_away_str[x]
            + " "
            + score_display_away[x],
            showarrow=True,
        )
    fig.add_trace(
        go.Scatter(
            x=ball_change_idx,
            y=ball_change_y,
            text=ball_change_cause,
            mode="markers+text",
            name="Possession Change",
            textposition="top center",
            marker=dict(color="blue"),
        )
    )
    # for x in range(len(ball_change_idx)):
    #     fig.add_annotation(
    #         x=ball_change_idx[x],
    #         y=ball_change_idx[x],
    #         text=ball_change_cause[x],
    #         showarrow=True,
    #     )

    # # fig.add_annotation(hovertext=game_df["game_code"])
    # # print((game_df))
    fig.update_xaxes(range=[0, len(game_df) - 1])
    fig.update_yaxes(range=[0, 1])
    # # fig.update_annotations(hover_lable=game_df[hover_values])
    # fig.update_yaxes(range=[0, 1], row=1, col=1)
    fig.update_layout(height=700, width=1600)
    st.plotly_chart(fig, use_container_width=True)
