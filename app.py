from asyncio import events
from copy import deepcopy
from functools import cache
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
CHARTS = [
    "4th Down Bot",
    "Games",
]


st.set_page_config(
    page_title=apptitle,
    page_icon=":football:",
    layout="wide",
    # initial_sidebar_state="expanded",
)
st.title("Stats Perform Football Predictions")
st.write("by Lucas Haupt and Evan Boyd")


def ordinaltg(n):
    return n.replace({1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"})


def convert_to_dataframe(df):
    return pd.DataFrame(df)


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
def load_predictions(cache=True):
    events_df = get_event_data(cache=cache)
    game_df = get_game_data(cache=cache)
    # print(game_df.head())
    prior_df = pd.read_csv(os.path.join(data_dir, "game_priors.csv"))
    odds_df = pd.read_parquet(os.path.join(data_dir, "odds_data.parquet"))
    odds_df = odds_df.drop_duplicates("game_code")
    full_df = (
        (
            events_df.merge(prior_df, on="game_code")
            .merge(game_df, on="game_code", suffixes=["", "_y"])
            .merge(odds_df, on="game_code", how="left", suffixes=["", "_y"])
            .pipe(add_timeouts_remaining)
        )
        .sort_values(["game_date", "nevent"], ascending=[False, True])
        .reset_index()
    )
    full_df = full_df[(full_df["season"] == 2021)][1:1000]

    clf = pickle.load(open(os.path.join("models/game_score_new_4.sav"), "rb"))
    rf = pickle.load(
        open(
            os.path.join("models/game_score_random_forest_100_10_vegas_spread.p"), "rb"
        )
    )
    rf.verbose = 0
    input_names_mlp = [
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
    input_names_rf = rf.feature_names_in_.tolist()

    full_df = full_df[full_df[input_names_mlp].notna().all(axis=1)]
    full_df = full_df.pipe(add_game_info).pipe(add_play_description)
    full_df = full_df.sort_values(["game_date", "nevent"], ascending=[False, True])
    # print(full_df.head())
    # fourth_downs_only = full_df.loc[full_df.down == 4]
    # breakpoint()
    example_input = full_df
    example_running_score = example_input[
        ["home_start_score", "away_start_score"]
    ].values
    example_input_go_for_it = deepcopy(example_input[input_names_mlp])
    example_input_go_for_it_rf = deepcopy(example_input[input_names_rf])
    example_input_punt = deepcopy(example_input[input_names_mlp])
    example_input_punt_rf = deepcopy(example_input[input_names_rf])
    example_input_field_goal = deepcopy(example_input[input_names_mlp])
    example_input_field_goal_rf = deepcopy(example_input[input_names_rf])
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
    example_output_original = get_model_outputs(
        clf, example_input[input_names_mlp], example_running_score
    )
    example_output_original_rf = get_model_outputs(
        rf, example_input[input_names_rf], example_running_score
    )
    # breakpoint()
    example_input[
        [
            "xhome_win_mlp",
            "xdraw_mlp",
            "xaway_win_mlp",
        ]
    ] = pd.DataFrame(example_output_original["ft_outcome"])
    example_input[
        [
            "xhome_win_rf",
            "xdraw_rf",
            "xaway_win_rf",
        ]
    ] = pd.DataFrame(example_output_original_rf["ft_outcome"])
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
        input_names_mlp,
        input_names_rf,
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
    input_names_mlp,
    input_names_rf,
) = load_predictions()
full_df.to_csv("full_df.csv")
if chart_select_box == "4th Down Bot":
    games = full_df["game_info"].drop_duplicates()
    game_selection = st.selectbox("Pick a Game", games)
    plays = example_input[
        (example_input["game_info"] == game_selection) & (example_input["down"] == 4)
    ]["play_description"]
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
            ],
        )
        # print_scores.rename(index=["go_for_it", "punt", "field_goal"])

        example_output_rf = play_example[
            [
                "xhome_win_go_for_it_rf",
                "xhome_win_punt_rf",
                "xhome_win_field_goal_rf",
            ]
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
    print(play_example[input_names_rf].values.tolist())
    print(play_example[input_names_rf])
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

elif chart_select_box == "Games":
    games = full_df["game_info"].drop_duplicates()
    game_selection = st.selectbox("Pick a Game", games)
    game_df = example_input[example_input["game_info"] == game_selection]
    MODEL_TYPE_SELECT = ["MLP", "RF"]
    model_type_selection = st.selectbox("Model", MODEL_TYPE_SELECT)
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
    print(v)
    # breakpoint()
    mask_ticks = v["quarter"][1:].reset_index(drop=True) == v["quarter"][
        :-1
    ].reset_index(drop=True)
    print(mask_ticks)
    # breakpoint()
    ticks_idx = [min(v["nevent"])] + list(v[:-1][~mask_ticks]["nevent"] + 1)
    if len(ticks_idx) == 4:
        ticks_values = [1, 2, 3, 4]
    else:
        ticks_values = [1, 2, 3, 4, "OT"]

    print("ticks_values: ", ticks_values)
    print("ticks_idx: ", ticks_idx)

    fig = go.Figure()
    game_df["xhome_win_mlp_no_tie"] = game_df["xhome_win_mlp"] / (
        game_df["xhome_win_mlp"] + game_df["xaway_win_mlp"]
    )
    game_df["xhome_win_rf_no_tie"] = game_df["xhome_win_rf"] / (
        game_df["xhome_win_rf"] + game_df["xaway_win_rf"]
    )
    game_df["xaway_win_mlp_no_tie"] = 1 - game_df["xhome_win_mlp_no_tie"]
    game_df["xaway_win_rf_no_tie"] = 1 - game_df["xhome_win_rf_no_tie"]
    if model_type_selection == "MLP":
        if game_df["game_type_id"].tolist()[0] == 1:
            y = ["xhome_win_mlp", "xdraw_mlp", "xaway_win_mlp"]
        else:
            y = ["xhome_win_mlp_no_tie", "xaway_win_mlp_no_tie"]
        # y = "xhome_win_mlp"
    elif model_type_selection == "RF":
        if game_df["game_type_id"].tolist()[0] == 1:
            y = ["xhome_win_rf", "xdraw_rf", "xaway_win_rf"]
        else:
            y = ["xhome_win_rf_no_tie", "xaway_win_rf_no_tie"]
        # y = "xhome_win_rf"
    colors = ["khaki", "lightgray", "lightskyblue"]
    print(pd.to_numeric(game_df["yards_gained"], errors="coerce"))

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
    if game_df["game_type_id"].tolist()[0] == 1:
        print(y)
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

    else:
        fig.add_trace(
            go.Scatter(
                customdata=game_df[["event_name", "play_description"]],
                stackgroup="one",
                mode="lines",
                hovertemplate="<br>".join(
                    [
                        "%{y}",
                        "%{customdata[0]}",
                        "%{customdata[1]}",
                    ]
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=game_df["nevent"],
                y=game_df[y[1]],
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
    print(game_df.columns.tolist())
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

    # # fig.add_annotation(hovertext=game_df["game_code"])
    # # print((game_df))
    fig.update_xaxes(range=[0, len(game_df)])
    fig.update_yaxes(range=[0, 1])
    # # fig.update_annotations(hover_lable=game_df[hover_values])
    # fig.update_yaxes(range=[0, 1], row=1, col=1)
    fig.update_layout(height=700, width=1600)
    st.plotly_chart(fig, use_container_width=True)
