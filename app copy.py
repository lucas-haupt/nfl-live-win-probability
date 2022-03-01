from asyncio import events
from copy import deepcopy
from operator import index
from sklearn import model_selection
import streamlit as st
import pandas as pd
import pickle
import os
from notebooks.utils import get_model_outputs
from src.utils.utils import fyi
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
    return df


@st.cache
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


@st.cache
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
home_score_cols_go_for_it = ["home_score_go_for_it_" + str(x) for x in list(range(63))]
away_score_cols_go_for_it = ["away_score_go_for_it_" + str(x) for x in list(range(60))]
home_score_cols_punt = ["home_score_punt_" + str(x) for x in list(range(63))]
away_score_cols_punt = ["away_score_punt_" + str(x) for x in list(range(60))]
home_score_cols_field_goal = [
    "home_score_field_goal_" + str(x) for x in list(range(63))
]
away_score_cols_field_goal = [
    "away_score_field_goal_" + str(x) for x in list(range(60))
]
home_score_cols_go_for_it_rf = [
    "home_score_go_for_it_rf_" + str(x) for x in list(range(63))
]
away_score_cols_go_for_it_rf = [
    "away_score_go_for_it_rf_" + str(x) for x in list(range(60))
]
home_score_cols_punt_rf = ["home_score_punt_rf_" + str(x) for x in list(range(63))]
away_score_cols_punt_rf = ["away_score_punt_rf_" + str(x) for x in list(range(60))]
home_score_cols_field_goal_rf = [
    "home_score_field_goal_rf_" + str(x) for x in list(range(63))
]
away_score_cols_field_goal_rf = [
    "away_score_field_goal_rf_" + str(x) for x in list(range(60))
]


@st.cache
def load_predictions():
    events_df = pd.read_parquet(os.path.join(data_dir, "event_data.parquet"))
    game_df = pd.read_parquet(os.path.join(data_dir, "game_data.parquet"))
    prior_df = pd.read_csv(os.path.join(data_dir, "game_priors.csv"))
    odds_df = pd.read_parquet(os.path.join(data_dir, "odds_data.parquet"))
    odds_df = odds_df.drop_duplicates("game_code")
    full_df = (
        events_df.merge(prior_df, suffixes=["", "_y"], on="game_code", how="left")
        .merge(game_df, on="game_code", suffixes=["", "_y"], how="left")
        .merge(odds_df, on="game_code", suffixes=["", "_y"], how="left")
        .pipe(add_timeouts_remaining)
    ).sort_values(["game_date", "nevent"], ascending=[False, True])[1:10000]
    # full_df = events_df.merge(prior_df, on="game_code", how="left").merge(odds_df, on="game_code", how="left")
    full_df["cur_spread"].fillna((full_df["cur_spread"].mean()), inplace=True)
    full_df["cur_over_under"].fillna((full_df["cur_over_under"].mean()), inplace=True)
    # full_df = full_df[full_df[input_names+[output_name]].notna().all(axis=1)]

    # input_names = [
    #     "prior_home",
    #     "prior_away",
    #     "home_team_has_ball",
    #     "home_start_score",
    #     "away_start_score",
    #     "quarter",
    #     "overtime",
    #     "play_start_time",
    #     "yd_from_goal",
    #     "from_scrimmage",
    #     "kick_off",
    #     "punt",
    #     "point_after_kick",
    #     "two_point_attempt",
    #     "field_goal_attempt",
    #     "down",
    #     "ytg",
    #     "home_timeouts_remaining",
    #     "away_timeouts_remaining",
    # ]
    score_change_columns = [
        "away_increase_6",
        "away_increase_3",
        "away_increase_2",
        "away_increase_1",
        "no_increase",
        "home_increase_1",
        "home_increase_2",
        "home_increase_3",
        "home_increase_6",
    ]
    clf = pickle.load(open(os.path.join("models/game_score_new_4.sav"), "rb"))
    clf_input_names = clf.feature_names_in_.tolist()
    rf = pickle.load(
        open(
            os.path.join("models/game_score_random_forest_100_10_new_features.p"), "rb"
        )
    )
    rf_score_change = pickle.load(
        open(os.path.join("models/score_change_random_forest_100_10.p"), "rb")
    )
    input_names_score_change = rf_score_change.feature_names_in_.tolist()
    full_df = full_df[full_df[input_names_score_change].notna().all(axis=1)]
    full_df = full_df.reset_index()
    # print(full_df[input_names_score_change].head())
    input_names = rf.feature_names_in_.tolist()
    # full_df.to_csv("help2.csv")
    # breakpoint()
    score_change_predictions = pd.DataFrame(
        rf_score_change.predict_proba(full_df[input_names_score_change])
    )
    full_df[score_change_columns] = score_change_predictions
    full_df = full_df.pipe(add_game_info).pipe(add_play_description)
    full_df = full_df.sort_values(["game_date", "nevent"], ascending=[False, True])

    # print(full_df.head())
    # fourth_downs_only = full_df.loc[full_df.down == 4]
    example_input = full_df.loc[full_df.down == 4]
    example_running_score = example_input[
        ["home_start_score", "away_start_score"]
    ].values
    # print(input_names)
    # print(example_input.shape())
    example_input_go_for_it = deepcopy(example_input[clf_input_names])
    example_input_go_for_it_rf = deepcopy(example_input[input_names])
    example_input_punt = deepcopy(example_input[clf_input_names])
    example_input_punt_rf = deepcopy(example_input[input_names])
    example_input_field_goal = deepcopy(example_input[clf_input_names])
    example_input_field_goal_rf = deepcopy(example_input[input_names])
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
    example_input_go_for_it_rf.to_csv("help.csv")
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
    # breakpoint()
    example_input[
        [
            "xhome_win_go_for_it_mlp",
            "xdraw_go_for_it_mlp",
            "xaway_win_go_for_it_mlp",
        ]
    ] = pd.DataFrame(example_output_go_for_it["ft_outcome"])
    example_input[
        [
            "xhome_win_punt_mlp",
            "xdraw_punt_mlp",
            "xaway_win_punt_mlp",
        ]
    ] = pd.DataFrame(example_output_punt["ft_outcome"])
    example_input[
        [
            "xhome_win_field_goal_mlp",
            "xdraw_field_goal_mlp",
            "xaway_win_field_goal_mlp",
        ]
    ] = pd.DataFrame(example_output_field_goal["ft_outcome"])
    example_input[
        [
            "xhome_win_go_for_it_rf",
            "xdraw_go_for_it_rf",
            "xaway_win_go_for_it_rf",
        ]
    ] = pd.DataFrame(example_output_go_for_it_rf["ft_outcome"])
    example_input[
        [
            "xhome_win_punt_rf",
            "xdraw_punt_rf",
            "xaway_win_punt_rf",
        ]
    ] = pd.DataFrame(example_output_punt_rf["ft_outcome"])
    example_input[
        [
            "xhome_win_field_goal_rf",
            "xdraw_field_goal_rf",
            "xaway_win_field_goal_rf",
        ]
    ] = pd.DataFrame(example_output_field_goal_rf["ft_outcome"])

    example_input[home_score_cols_go_for_it] = pd.DataFrame(
        example_output_go_for_it["home_score"]
    )
    example_input[away_score_cols_go_for_it] = pd.DataFrame(
        example_output_go_for_it["away_score"]
    )
    example_input[home_score_cols_go_for_it_rf] = pd.DataFrame(
        example_output_go_for_it_rf["home_score"]
    )
    example_input[away_score_cols_go_for_it_rf] = pd.DataFrame(
        example_output_go_for_it_rf["away_score"]
    )
    example_input[home_score_cols_punt] = pd.DataFrame(
        example_output_punt["home_score"]
    )
    example_input[away_score_cols_punt] = pd.DataFrame(
        example_output_punt["away_score"]
    )
    example_input[home_score_cols_punt_rf] = pd.DataFrame(
        example_output_punt_rf["home_score"]
    )
    example_input[away_score_cols_punt_rf] = pd.DataFrame(
        example_output_punt_rf["away_score"]
    )
    example_input[home_score_cols_field_goal] = pd.DataFrame(
        example_output_field_goal["home_score"]
    )
    example_input[away_score_cols_field_goal] = pd.DataFrame(
        example_output_field_goal["away_score"]
    )
    example_input[home_score_cols_field_goal_rf] = pd.DataFrame(
        example_output_field_goal_rf["home_score"]
    )
    example_input[away_score_cols_field_goal_rf] = pd.DataFrame(
        example_output_field_goal_rf["away_score"]
    )

    return (
        full_df,
        example_input,
        example_output_go_for_it,
        example_output_punt,
        example_output_field_goal,
        example_output_go_for_it_rf,
        example_output_punt_rf,
        example_output_field_goal_rf,
        score_change_columns,
    )


(
    full_df,
    example_input,
    example_output_go_for_it,
    example_output_punt,
    example_output_field_goal,
    example_output_go_for_it_rf,
    example_output_punt_rf,
    example_output_field_goal_rf,
    score_change_columns,
) = load_predictions()
if chart_select_box == "Games":
    games = full_df["game_info"].drop_duplicates()
    game_selection = st.selectbox("Pick a Game", games)
    plays = example_input[example_input["game_info"] == game_selection][
        "play_description"
    ]
    play_selection = st.selectbox("Play", plays)
    play_example = example_input[
        (example_input["game_info"] == game_selection)
        & (example_input["play_description"] == play_selection)
    ]
    if play_example["home_team_has_ball"].values == 1:
        example_output = play_example[
            [
                "xhome_win_go_for_it_mlp",
                "xhome_win_punt_mlp",
                "xhome_win_field_goal_mlp",
            ]
        ].transpose()
        print_scores = pd.concat(
            [
                pd.DataFrame(
                    play_example[home_score_cols_go_for_it].values, index=["go_for_it"]
                ),
                pd.DataFrame(play_example[home_score_cols_punt].values, index=["punt"]),
                pd.DataFrame(
                    play_example[home_score_cols_field_goal].values,
                    index=["field_goal"],
                ),
            ],
        )
        print_scores_rf = pd.concat(
            [
                pd.DataFrame(
                    play_example[home_score_cols_go_for_it_rf].values,
                    index=["go_for_it"],
                ),
                pd.DataFrame(
                    play_example[home_score_cols_punt_rf].values, index=["punt"]
                ),
                pd.DataFrame(
                    play_example[home_score_cols_field_goal_rf].values,
                    index=["field_goal"],
                ),
                pd.DataFrame(
                    play_example[score_change_columns].values,
                    index=[score_change_columns],
                ),
            ],
        )
        # print_scores.rename(index=["go_for_it", "punt", "field_goal"])

        example_output_rf = play_example[
            [
                "xhome_win_go_for_it_rf",
                "xhome_win_punt_rf",
                "xhome_win_field_goal_rf",
            ]
            + score_change_columns,
        ].transpose()
    else:
        example_output = play_example[
            [
                "xaway_win_go_for_it_mlp",
                "xaway_win_punt_mlp",
                "xaway_win_field_goal_mlp",
            ]
        ].transpose()

        example_output_rf = play_example[
            [
                "xaway_win_go_for_it_rf",
                "xaway_win_punt_rf",
                "xaway_win_field_goal_rf",
            ]
            + score_change_columns
        ].transpose()
        print_scores = pd.concat(
            [
                pd.DataFrame(
                    play_example[away_score_cols_go_for_it].values, index=["go_for_it"]
                ),
                pd.DataFrame(play_example[away_score_cols_punt].values, index=["punt"]),
                pd.DataFrame(
                    play_example[away_score_cols_field_goal].values,
                    index=["field_goal"],
                ),
            ],
        )
        print_scores_rf = pd.concat(
            [
                pd.DataFrame(
                    play_example[away_score_cols_go_for_it_rf].values,
                    index=["go_for_it"],
                ),
                pd.DataFrame(
                    play_example[away_score_cols_punt_rf].values, index=["punt"]
                ),
                pd.DataFrame(
                    play_example[away_score_cols_field_goal_rf].values,
                    index=["field_goal"],
                ),
            ],
        )

    MODEL_TYPE_SELECT = ["MLP", "RF"]
    model_type_selection = st.selectbox("Model", MODEL_TYPE_SELECT)
    if model_type_selection == "MLP":
        # print(print_scores)
        st.dataframe(
            example_output,
            width=10000,
        )
        st.dataframe(
            print_scores.transpose(),
            width=10000,
        )

        # st.dataframe(print_scores)
        # st.dataframe(play_example["away_score"])

    elif model_type_selection == "RF":
        st.dataframe(
            example_output_rf,
            width=10000,
        )
        st.dataframe(
            print_scores_rf.transpose(),
            width=10000,
        )
