import numpy as np
import time
import pandas as pd
from functools import wraps

max_home_score = 62
max_away_score = 59
n_categories = (max_home_score + 1) * (max_away_score + 1)
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain


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


@fyi
def get_model_outputs(model, input_data, running_scores):
    start_time = time.time()
    raw_output = model.predict_proba(input_data)
    # Exact score outputs
    score_probs = np.zeros((input_data.shape[0], n_categories))
    score_probs[:, model.classes_] = raw_output
    # 1X2 prediction & team score outputs
    outcome_probs = np.zeros((input_data.shape[0], 3))
    home_score_probs = np.zeros((input_data.shape[0], max_home_score + 1))
    away_score_probs = np.zeros((input_data.shape[0], max_away_score + 1))
    for home_score in range(max_home_score + 1):
        ft_home_score = home_score + running_scores[:, 0]
        for away_score in range(max_away_score + 1):
            # print(home_score, "-", away_score)
            ft_away_score = away_score + running_scores[:, 1]
            remaining_prob = score_probs[
                :, away_score + (max_away_score + 1) * home_score
            ]
            # 1X2 - Home win
            outcome_probs[:, 0] = np.where(
                ft_home_score > ft_away_score,
                outcome_probs[:, 0] + remaining_prob,
                outcome_probs[:, 0],
            )
            # 1X2 - Draw
            outcome_probs[:, 1] = np.where(
                ft_home_score == ft_away_score,
                outcome_probs[:, 1] + remaining_prob,
                outcome_probs[:, 1],
            )
            # 1X2 - Away win
            outcome_probs[:, 2] = np.where(
                ft_home_score < ft_away_score,
                outcome_probs[:, 2] + remaining_prob,
                outcome_probs[:, 2],
            )
            # Team scores
            home_score_probs[:, home_score] += remaining_prob
            away_score_probs[:, away_score] += remaining_prob
    return {
        "remaining_score": score_probs,
        "home_score": home_score_probs,
        "away_score": away_score_probs,
        "ft_outcome": outcome_probs,
    }


def aggregate_match_results_by_team(results, team_names, return_detail=False):

    results_df = pd.DataFrame(results)
    results_df["outcome"] = (
        np.sign(results_df["home_score"] - results_df["away_score"]) + 1
    )
    home_df = results_df[
        [
            "home_team_id",
            "away_team_id",
            "game_code",
            "outcome",
            "home_score",
            "away_score",
        ]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "away_team_id": "opponent_id",
            "home_score": "GF",
            "away_score": "GA",
        }
    )
    home_df["is_home"] = 1
    away_df = results_df[
        [
            "away_team_id",
            "home_team_id",
            "game_code",
            "outcome",
            "home_score",
            "away_score",
        ]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "home_team_id": "opponent_id",
            "away_score": "GF",
            "home_score": "GA",
        }
    )
    away_df["outcome"] = away_df["outcome"].map({0: 2, 1: 1, 2: 0})
    away_df["is_home"] = 0
    results_df = pd.concat([home_df, away_df], axis=0)

    results_df = results_df.merge(
        team_names[["division_id", "conference_id"]],
        left_on="team_id",
        right_index=True,
        sort=False,
    )
    results_df = results_df.merge(
        team_names[["division_id", "conference_id"]],
        left_on="opponent_id",
        right_index=True,
        sort=False,
        suffixes=["_team", "_opponent"],
    )

    results_df["G"] = 1
    results_df["W"] = (results_df["outcome"] == 2).astype(int)
    results_df["D"] = (results_df["outcome"] == 1).astype(int)
    results_df["L"] = (results_df["outcome"] == 0).astype(int)

    results_df["points"] = results_df["outcome"]
    results_df["GF_away"] = results_df["GF"].where(results_df["is_home"] == 0, 0)

    results_df["points_conference"] = results_df["points"].where(
        results_df["conference_id_team"] == results_df["conference_id_opponent"], 0
    )
    results_df["points_division"] = results_df["points"].where(
        (results_df["conference_id_team"] == results_df["conference_id_opponent"])
        & (results_df["division_id_team"] == results_df["division_id_opponent"]),
        0,
    )

    results_df["games_conference"] = results_df["G"].where(
        results_df["conference_id_team"] == results_df["conference_id_opponent"], 0
    )
    results_df["games_division"] = results_df["G"].where(
        (results_df["conference_id_team"] == results_df["conference_id_opponent"])
        & (results_df["division_id_team"] == results_df["division_id_opponent"]),
        0,
    )
    agg_df = results_df.groupby("team_id")[
        [
            "G",
            "W",
            "D",
            "L",
            "points",
            "GF",
            "GA",
            "GF_away",
            "points_conference",
            "points_division",
            "games_conference",
            "games_division",
        ]
    ].sum()
    if return_detail:
        return agg_df, results_df
    else:
        return agg_df


def simulate_season_standing_with_tiebreakers(
    match_preds, match_results, team_names, pred_params, n_runs
):
    team_names = pd.DataFrame(team_names).set_index("id")[
        ["name", "conference_id", "division_id"]
    ]
    start_time = time.time()
    print("Simulating season ({0:d} runs)".format(n_runs))
    run_list = []
    no_games_left = len(match_preds) == 0
    for match in match_preds:
        exact_score_probs = np.array(match["pred_exact_score"]) / np.sum(
            match["pred_exact_score"]
        )
        scores = np.random.multinomial(n=1, pvals=exact_score_probs, size=n_runs)
        ind_scores = np.argmax(scores, axis=1)
        aux_home = np.tile(
            np.arange(0, pred_params["max_home_score"] + 1),
            (pred_params["max_away_score"] + 1, 1),
        ).flatten(order="F")
        aux_away = np.tile(
            np.arange(0, pred_params["max_away_score"] + 1),
            (pred_params["max_home_score"] + 1),
        )
        home_scores = aux_home[ind_scores] + match["current_score"][0]
        away_scores = aux_away[ind_scores] + match["current_score"][1]
        away_outcomes = np.zeros((n_runs, 3)).astype(int)
        away_outcomes[np.arange(n_runs), np.sign(home_scores - away_scores) + 1] = 1
        home_outcomes = np.fliplr(away_outcomes)

        home_df = pd.DataFrame(
            data=np.hstack(
                (
                    np.arange(1, n_runs + 1).reshape(-1, 1),
                    home_outcomes,
                    home_scores.reshape(-1, 1),
                    away_scores.reshape(-1, 1),
                )
            ),
            columns=["run", "W", "D", "L", "GF", "GA"],
        )
        # home_df = home_df.merge(team_names)
        home_df["team_id"] = match["home_team_id"]
        home_df["opponent_id"] = match["away_team_id"]
        home_df["is_home"] = 1
        away_df = pd.DataFrame(
            data=np.hstack(
                (
                    np.arange(1, n_runs + 1).reshape(-1, 1),
                    away_outcomes,
                    away_scores.reshape(-1, 1),
                    home_scores.reshape(-1, 1),
                )
            ),
            columns=["run", "W", "D", "L", "GF", "GA"],
        )
        away_df["team_id"] = match["away_team_id"]
        away_df["opponent_id"] = match["home_team_id"]
        away_df["is_home"] = 0
        run_df = pd.concat([home_df, away_df], axis=0)
        run_df = run_df.merge(
            team_names[["division_id", "conference_id"]],
            left_on="team_id",
            right_index=True,
            sort=False,
        )
        run_df = run_df.merge(
            team_names[["division_id", "conference_id"]],
            left_on="opponent_id",
            right_index=True,
            sort=False,
            suffixes=["_team", "_opponent"],
        )
        run_list.append(run_df)

    if no_games_left:
        empty_run_df_cols = [
            "run",
            "team_id",
            "opponent_id",
            "W",
            "D",
            "L",
            "GF",
            "GA",
            "is_home",
            "G",
            "points",
            "GF_away",
        ]
        all_runs_df = pd.DataFrame(
            {
                col: pd.Series(
                    [], dtype="str" if col in ["team_id", "opponent_id"] else "int64"
                )
                for col in empty_run_df_cols
            }
        )
        missing_team_ids = team_names.index.unique()
    else:
        all_runs_df = pd.concat(run_list, axis=0).reset_index(drop=True)
        all_runs_df["G"] = 1
        all_runs_df["points"] = (
            all_runs_df[["W", "D", "L"]] * np.array([2, 1, 0])
        ).sum(axis=1)
        all_runs_df["points_conference"] = all_runs_df["points"].where(
            all_runs_df["conference_id_team"] == all_runs_df["conference_id_opponent"],
            0,
        )
        all_runs_df["points_division"] = all_runs_df["points"].where(
            (all_runs_df["conference_id_team"] == all_runs_df["conference_id_opponent"])
            & (all_runs_df["division_id_team"] == all_runs_df["division_id_opponent"]),
            0,
        )

        all_runs_df["games_conference"] = all_runs_df["G"].where(
            all_runs_df["conference_id_team"] == all_runs_df["conference_id_opponent"],
            0,
        )
        all_runs_df["games_division"] = all_runs_df["G"].where(
            (all_runs_df["conference_id_team"] == all_runs_df["conference_id_opponent"])
            & (all_runs_df["division_id_team"] == all_runs_df["division_id_opponent"]),
            0,
        )
        all_runs_df["GF_away"] = all_runs_df["GF"].where(all_runs_df["is_home"] == 0, 0)
        # Aggregate simulated game results by run
        agg_runs_df = (
            all_runs_df.drop(columns="opponent_id")
            .groupby(["run", "team_id"])
            .sum()
            .reset_index(drop=False)
        )
        missing_team_ids = np.setdiff1d(team_names.index, agg_runs_df.team_id)

    # Ranking criteria
    ranking_criteria = [
        "points",
        "H2H_points",
        "points_division",
        "points_conference",
        "GD",
        "GF",
        "H2H_GF_away",
    ]

    # Preliminary criteria (anything that does not involve head-to-head subsetting)
    preliminary_criteria = []
    for i in ranking_criteria:
        if i.find("H2H_") == 0:
            break
        else:
            preliminary_criteria.append(i)

    # Aggregate previous results
    if match_results == []:
        agg_match_results = pd.DataFrame(team_names)
        agg_match_results[
            [
                "G",
                "W",
                "D",
                "L",
                "points",
                "GF",
                "GA",
                "GF_away",
                "points_conference",
                "points_division",
                "games_conference",
                "games_division",
            ]
        ] = 0
        match_team_results = []
    else:
        agg_match_results, match_team_results = aggregate_match_results_by_team(
            match_results, team_names, return_detail=True
        )

    # Add missing teams (teams present in observed results but without any remaining games)
    if missing_team_ids.size > 0:
        aux_ind = pd.MultiIndex.from_product(
            [np.arange(1, n_runs + 1), missing_team_ids], names=["run", "team_id"]
        )
        missing_df = pd.DataFrame(
            columns=np.setdiff1d(
                all_runs_df.columns, ["run", "team_id", "opponent_id"]
            ).tolist(),
            data=0,
            index=aux_ind,
        ).reset_index(drop=False)
        if no_games_left:
            agg_runs_df = missing_df
        else:
            agg_runs_df = pd.concat([agg_runs_df, missing_df], axis=0)

    # Add observed results from past games
    sum_cols = [
        "G",
        "W",
        "D",
        "L",
        "GF",
        "GA",
        "points",
        "GF_away",
        "points_conference",
        "points_division",
        "games_conference",
        "games_division",
    ]
    prev_df = (
        agg_match_results.reindex(agg_runs_df.team_id.values, columns=sum_cols)
        .fillna(0)
        .astype(int)
    )
    agg_runs_df[sum_cols] = agg_runs_df[sum_cols].values + prev_df.values
    agg_runs_df["GD"] = agg_runs_df["GF"] - agg_runs_df["GA"]
    agg_runs_df = agg_runs_df.sort_values(
        by=["run"] + preliminary_criteria, ascending=False
    ).reset_index(drop=True)

    # Identify ties (2 or more teams with identical values in the (preliminary) sorting features)
    aux = agg_runs_df.drop_duplicates(
        subset=["run"] + preliminary_criteria
    ).index.values
    agg_runs_df["tie_id"] = 0
    agg_runs_df.loc[aux, "tie_id"] = 1
    agg_runs_df["tie_id"] = np.cumsum(agg_runs_df["tie_id"])
    tie_size = (
        agg_runs_df.groupby(["tie_id"])["team_id"]
        .count()
        .rename("tie_size")
        .reset_index(drop=False)
    )
    agg_runs_df = agg_runs_df.merge(right=tie_size, on="tie_id", how="left")

    agg_runs_df = agg_runs_df.merge(
        team_names, left_on=["team_id"], right_index=True, how="left"
    )
    agg_runs_df["division_id_unique"] = (
        agg_runs_df["conference_id"] * 10 + agg_runs_df["division_id"]
    )

    agg_runs_df = agg_runs_df.sort_values(
        ["run", "division_id_unique"] + preliminary_criteria
    )
    aux_division = agg_runs_df.drop_duplicates(
        subset=["run", "division_id_unique"] + preliminary_criteria
    ).index.values
    agg_runs_df["division_tie_id"] = 0
    agg_runs_df.loc[aux_division, "division_tie_id"] = 1
    agg_runs_df["division_tie_id"] = np.cumsum(agg_runs_df["division_tie_id"])
    tie_size_division = (
        agg_runs_df.groupby(["division_tie_id"])["team_id"]
        .count()
        .rename("division_tie_size")
        .reset_index(drop=False)
    )
    agg_runs_df = agg_runs_df.merge(
        right=tie_size_division, on="division_tie_id", how="left"
    )
    agg_runs_df_tie_values = agg_runs_df

    agg_runs_df = agg_runs_df.sort_values(
        ["run", "conference_id"] + preliminary_criteria
    )
    aux_conference = agg_runs_df.drop_duplicates(
        subset=["run", "conference_id"] + preliminary_criteria
    ).index.values
    agg_runs_df["conference_tie_id"] = 0
    agg_runs_df.loc[aux_conference, "conference_tie_id"] = 1
    agg_runs_df["conference_tie_id"] = np.cumsum(agg_runs_df["conference_tie_id"])
    tie_size_conference = (
        agg_runs_df.groupby(["conference_tie_id"])["team_id"]
        .count()
        .rename("conference_tie_size")
        .reset_index(drop=False)
    )
    agg_runs_df = agg_runs_df.merge(
        right=tie_size_conference, on="conference_tie_id", how="left"
    )
    agg_runs_df_tie_values = agg_runs_df

    # Head-to-head data
    h2h_df = agg_runs_df[
        ["run", "tie_id", "division_tie_id", "conference_tie_id", "team_id"]
    ]
    h2h_df = h2h_df.merge(
        right=h2h_df,
        on=["run", "tie_id", "division_tie_id", "conference_tie_id"],
        how="left",
    )
    h2h_df = h2h_df.loc[h2h_df.team_id_x != h2h_df.team_id_y]
    h2h_df.rename(
        columns={"team_id_x": "team_id", "team_id_y": "opponent_id"}, inplace=True
    )
    # Relevant results
    h2h_past_df = h2h_df.merge(
        right=match_team_results[["team_id", "opponent_id", "is_home"] + sum_cols],
        on=["team_id", "opponent_id"],
        how="inner",
    )
    # Relevant simulations
    h2h_pred_df = h2h_df.merge(
        right=all_runs_df[["run", "team_id", "opponent_id", "is_home"] + sum_cols],
        on=["run", "team_id", "opponent_id"],
        how="inner",
    )
    # Join results and predictions
    h2h_df = pd.concat([h2h_past_df, h2h_pred_df], axis=0).reset_index(drop=True)
    h2h_df["GD"] = h2h_df["GF"] - h2h_df["GA"]
    h2h_df["GF_away"] = h2h_df["GF"].where(h2h_df["is_home"] == 0, 0)
    h2h_df.drop(columns="opponent_id", inplace=True)
    # Aggregate "head-to-head runs"
    agg_h2h_df = (
        h2h_df.groupby(
            ["run", "tie_id", "division_tie_id", "conference_tie_id", "team_id"]
        )
        .sum()
        .reset_index(drop=False)
    )
    agg_h2h_df.rename(
        columns={i: "H2H_" + i for i in ["points", "GD", "GF", "GF_away"]}, inplace=True
    )

    # Add tie breaker values to the main data frame
    h2h_cols = [i for i in agg_h2h_df.columns if i.find("H2H_") == 0]
    agg_runs_df = agg_runs_df.merge(
        right=agg_h2h_df[
            ["run", "tie_id", "division_tie_id", "conference_tie_id", "team_id"]
            + h2h_cols
        ],
        on=["run", "tie_id", "division_tie_id", "conference_tie_id", "team_id"],
        how="left",
    )
    agg_runs_df[h2h_cols] = agg_runs_df[h2h_cols].fillna(0)

    # Sort main data frame again, now with all tie-breaking data included

    agg_runs_df = agg_runs_df.sort_values(
        by=["run"] + ranking_criteria, ascending=False
    ).reset_index(drop=True)

    # Calculate rank ("cheap" method, repeating a [1, 2, 3, ..., N] array as many times as runs)
    agg_runs_df["rank"] = np.arange(1, team_names.size / 3 + 1).tolist() * n_runs

    agg_runs_df = agg_runs_df.sort_values(
        by=["run"] + ["conference_id"] + ranking_criteria, ascending=False
    ).reset_index(drop=True)
    agg_runs_df["conference_rank"] = (
        np.arange(1, team_names.size / 3 / 2 + 1).tolist() * n_runs * 2
    )

    agg_runs_df = agg_runs_df.sort_values(
        by=["run"] + ["division_id_unique"] + ranking_criteria, ascending=False
    ).reset_index(drop=True)
    agg_runs_df["division_rank"] = (
        np.arange(1, team_names.size / 3 / 8 + 1).tolist() * n_runs * 8
    )
    # for division_id in agg_runs_df
    # agg_runs_df['conference_rank'] = np.arange(1, team_names.size/3 + 1).tolist() * n_runs
    # agg_runs_df['division_rank'] = np.arange(1, team_names.size/3 + 1).tolist() * n_runs
    agg_runs_df["first_round_bye"] = np.where(agg_runs_df["conference_rank"] == 1, 1, 0)
    agg_runs_df["won_division"] = np.where(agg_runs_df["division_rank"] == 1, 1, 0)
    agg_runs_df["made_playoffs"] = np.where(agg_runs_df["conference_rank"] <= 7, 1, 0)
    # Distribution of simulated end-of-season rankings
    rank_dist = (
        agg_runs_df.groupby(["team_id", "rank"])[["run"]]
        .count()
        .rename(columns={"run": "n"})
    )
    rank_dist["p"] = rank_dist["n"] / n_runs
    # Distribution of simulated end-of-season league points
    points_dist = (
        agg_runs_df.groupby(["team_id", "points"])[["run"]]
        .count()
        .rename(columns={"run": "n"})
    )
    points_dist["p"] = points_dist["n"] / n_runs

    wins_dist = (
        agg_runs_df.groupby(["team_id", "W"])[["run"]]
        .count()
        .rename(columns={"run": "n"})
    )
    wins_dist["p"] = wins_dist["n"] / n_runs

    first_round_bye_dist = (
        agg_runs_df.groupby(["team_id", "first_round_bye"])[["run"]]
        .count()
        .rename(columns={"run": "n"})
    )
    first_round_bye_dist["p"] = first_round_bye_dist["n"] / n_runs

    won_division_dist = (
        agg_runs_df.groupby(["team_id", "won_division"])[["run"]]
        .count()
        .rename(columns={"run": "n"})
    )
    won_division_dist["p"] = won_division_dist["n"] / n_runs

    made_playoffs_dist = (
        agg_runs_df.groupby(["team_id", "made_playoffs"])[["run"]]
        .count()
        .rename(columns={"run": "n"})
    )
    made_playoffs_dist["p"] = won_division_dist["n"] / n_runs

    # Average number of points per team
    avg_points_df = (
        agg_runs_df.groupby("team_id")[
            [
                "points",
                "rank",
                "W",
                "L",
                "D",
                "first_round_bye",
                "won_division",
                "made_playoffs",
            ]
        ]
        .mean()
        .sort_values(by="points", ascending=False)
        .astype(float)
    )
    avg_points_df[["team_name", "conference_id", "division_id"]] = team_names.loc[
        avg_points_df.index
    ].values

    if agg_match_results.shape[0] == team_names.shape[0]:
        aux = agg_match_results[["points", "GF", "GA"]].copy()
        aux["GD"] = aux["GF"] - aux["GA"]
        ordered_team_ids = aux.sort_values(
            by=["points", "GD", "GF"], ascending=False
        ).index.values
    else:
        ordered_team_ids = avg_points_df.index.values

    # Prepare "agg_match_results" before formatting output
    for team_id in ordered_team_ids:
        if team_id not in agg_match_results.index:
            agg_match_results.loc[team_id] = 0
    agg_match_results = (
        agg_match_results.drop(columns="GF_away")
        .rename(
            columns={
                "G": "g",
                "W": "w",
                "D": "d",
                "L": "l",
                "GF": "gF",
                "GA": "gA",
            }
        )
        .astype(int)
    )

    print("   Done! Time elapsed: {0:.4f} seconds".format(time.time() - start_time))

    # Merge results into a single dictionary
    final_list = [
        {
            "id": int(team_id),
            "name": avg_points_df.loc[team_id, "team_name"],
            "current": agg_match_results.loc[team_id].to_dict(),
            "predicted": {
                "averagePoints": avg_points_df.loc[team_id, "points"],
                "averageRank": avg_points_df.loc[team_id, "rank"],
                "averageWins": avg_points_df.loc[team_id, "W"],
                "first_round_bye": avg_points_df.loc[team_id, "first_round_bye"],
                "won_division": avg_points_df.loc[team_id, "won_division"],
                "made_playoffs": avg_points_df.loc[team_id, "made_playoffs"],
                "averageLosses": avg_points_df.loc[team_id, "L"],
                "averageTies": avg_points_df.loc[team_id, "D"],
                "rank": {},
                "points": {},
                "wins": {},
            },
        }
        for team_id in ordered_team_ids
    ]
    for (team_id, rank), sim_results in rank_dist.to_dict(orient="index").items():
        idx_team = np.flatnonzero(ordered_team_ids == team_id)[0]
        final_list[idx_team]["predicted"]["rank"][rank] = sim_results["p"]
    for (team_id, rank), sim_results in points_dist.to_dict(orient="index").items():
        idx_team = np.flatnonzero(ordered_team_ids == team_id)[0]
        final_list[idx_team]["predicted"]["points"][rank] = sim_results["p"]
    for (team_id, rank), sim_results in wins_dist.to_dict(orient="index").items():
        idx_team = np.flatnonzero(ordered_team_ids == team_id)[0]
        final_list[idx_team]["predicted"]["wins"][rank] = sim_results["p"]
    return final_list

"""
This module implements multioutput regression and classification.

The estimators provided in this module are meta-estimators: they require
a base estimator to be provided in their constructor. The meta-estimator
extends single output estimators to multioutput estimators.
"""

# Author: Tim Head <betatim@gmail.com>
# Author: Hugo Bowne-Anderson <hugobowne@gmail.com>
# Author: Chris Rivera <chris.richard.rivera@gmail.com>
# Author: Michael Williamson
# Author: James Ashton Nichols <james.ashton.nichols@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
from joblib import Parallel

from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.base import RegressorMixin, ClassifierMixin, is_classifier
from sklearn.model_selection import cross_val_predict
from sklearn.utils.metaestimators import available_if
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, has_fit_parameter, _check_fit_params
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.fixes import delayed

__all__ = [
    "MultiOutputRegressor",
    "MultiOutputClassifier",
    "TennisClassifierChain",
    "TennisRegressorChain",
]


def _fit_estimator(estimator, X, y, sample_weight=None, **fit_params):
    estimator = clone(estimator)
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
    else:
        estimator.fit(X, y, **fit_params)
    return estimator


def _partial_fit_estimator(
    estimator, X, y, classes=None, sample_weight=None, first_time=True
):
    if first_time:
        estimator = clone(estimator)

    if sample_weight is not None:
        if classes is not None:
            estimator.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
        else:
            estimator.partial_fit(X, y, sample_weight=sample_weight)
    else:
        if classes is not None:
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)
    return estimator


def _available_if_estimator_has(attr):
    """Return a function to check if `estimator` or `estimators_` has `attr`.

    Helper for Chain implementations.
    """

    def _check(self):
        return hasattr(self.estimator, attr) or all(
            hasattr(est, attr) for est in self.estimators_
        )

    return available_if(_check)


class _MultiOutputEstimator(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, estimator, *, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs

    @_available_if_estimator_has("partial_fit")
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incrementally fit a separate model for each class output.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets.

        classes : list of ndarray of shape (n_outputs,), default=None
            Each array is unique classes for one output in str/int.
            Can be obtained via
            ``[np.unique(y[:, i]) for i in range(y.shape[1])]``, where `y`
            is the target matrix of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that `y` doesn't need to contain all labels in `classes`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        first_time = not hasattr(self, "estimators_")
        y = self._validate_data(X="no_validation", y=y, multi_output=True)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        if sample_weight is not None and not has_fit_parameter(
            self.estimator, "sample_weight"
        ):
            raise ValueError("Underlying estimator does not support sample weights.")

        first_time = not hasattr(self, "estimators_")

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_estimator)(
                self.estimators_[i] if not first_time else self.estimator,
                X,
                y[:, i],
                classes[i] if classes is not None else None,
                sample_weight,
                first_time,
            )
            for i in range(y.shape[1])
        )

        if first_time and hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if first_time and hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the model to data, separately for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        y = self._validate_data(X="no_validation", y=y, multi_output=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        if sample_weight is not None and not has_fit_parameter(
            self.estimator, "sample_weight"
        ):
            raise ValueError("Underlying estimator does not support sample weights.")

        fit_params_validated = _check_fit_params(X, fit_params)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator, X, y[:, i], sample_weight, **fit_params_validated
            )
            for i in range(y.shape[1])
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict(self, X):
        """Predict multi-output variable using model for each target variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self)
        if not hasattr(self.estimators_[0], "predict"):
            raise ValueError("The base estimator should implement a predict method")

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X) for e in self.estimators_
        )

        return np.asarray(y).T

    def _more_tags(self):
        return {"multioutput_only": True}


class MultiOutputRegressor(RegressorMixin, _MultiOutputEstimator):
    """Multi target regression.

    This strategy consists of fitting one regressor per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.

    .. versionadded:: 0.18

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and :term:`predict`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        :meth:`fit`, :meth:`predict` and :meth:`partial_fit` (if supported
        by the passed estimator) will be parallelized for each target.

        When individual estimators are fast to train or predict,
        using ``n_jobs > 1`` can result in slower performance due
        to the parallelism overhead.

        ``None`` means `1` unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all available processes / threads.
        See :term:`Glossary <n_jobs>` for more details.

        .. versionchanged:: 0.20
            `n_jobs` default changed from `1` to `None`.

    Attributes
    ----------
    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    RegressorChain : A multi-label model that arranges regressions into a
        chain.
    MultiOutputClassifier : Classifies each output independently rather than
        chaining.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_linnerud
    >>> from sklearn.multioutput import MultiOutputRegressor
    >>> from sklearn.linear_model import Ridge
    >>> X, y = load_linnerud(return_X_y=True)
    >>> clf = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
    >>> clf.predict(X[[0]])
    array([[176..., 35..., 57...]])
    """

    def __init__(self, estimator, *, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)

    @_available_if_estimator_has("partial_fit")
    def partial_fit(self, X, y, sample_weight=None):
        """Incrementally fit the model to data, for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        super().partial_fit(X, y, sample_weight=sample_weight)


class MultiOutputClassifier(ClassifierMixin, _MultiOutputEstimator):
    """Multi target classification.

    This strategy consists of fitting one classifier per target. This is a
    simple strategy for extending classifiers that do not natively support
    multi-target classification.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit`, :term:`score` and
        :term:`predict_proba`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        :meth:`fit`, :meth:`predict` and :meth:`partial_fit` (if supported
        by the passed estimator) will be parallelized for each target.

        When individual estimators are fast to train or predict,
        using ``n_jobs > 1`` can result in slower performance due
        to the parallelism overhead.

        ``None`` means `1` unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all available processes / threads.
        See :term:`Glossary <n_jobs>` for more details.

        .. versionchanged:: 0.20
            `n_jobs` default changed from `1` to `None`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.

    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    ClassifierChain : A multi-label model that arranges binary classifiers
        into a chain.
    MultiOutputRegressor : Fits one regressor per target variable.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> X, y = make_multilabel_classification(n_classes=3, random_state=0)
    >>> clf = MultiOutputClassifier(KNeighborsClassifier()).fit(X, y)
    >>> clf.predict(X[-2:])
    array([[1, 1, 0], [1, 1, 1]])
    """

    def __init__(self, estimator, *, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)

    def fit(self, X, Y, sample_weight=None, **fit_params):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying classifier supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        super().fit(X, Y, sample_weight, **fit_params)
        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self

    def _check_predict_proba(self):
        if hasattr(self, "estimators_"):
            # raise an AttributeError if `predict_proba` does not exist for
            # each estimator
            [getattr(est, "predict_proba") for est in self.estimators_]
            return True
        # raise an AttributeError if `predict_proba` does not exist for the
        # unfitted estimator
        getattr(self.estimator, "predict_proba")
        return True

    @available_if(_check_predict_proba)
    def predict_proba(self, X):
        """Return prediction probabilities for each class of each output.

        This method will raise a ``ValueError`` if any of the
        estimators do not have ``predict_proba``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        p : array of shape (n_samples, n_classes), or a list of n_outputs \
                such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

            .. versionchanged:: 0.19
                This function now returns a list of arrays where the length of
                the list is ``n_outputs``, and each array is (``n_samples``,
                ``n_classes``) for that particular output.
        """
        check_is_fitted(self)
        results = [estimator.predict_proba(X) for estimator in self.estimators_]
        return results

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples, n_outputs)
            True values for X.

        Returns
        -------
        scores : float
            Mean accuracy of predicted target versus true target.
        """
        check_is_fitted(self)
        n_outputs_ = len(self.estimators_)
        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi target classification but has only one"
            )
        if y.shape[1] != n_outputs_:
            raise ValueError(
                "The number of outputs of Y for fit {0} and"
                " score {1} should be same".format(n_outputs_, y.shape[1])
            )
        y_pred = self.predict(X)
        return np.mean(np.all(y == y_pred, axis=1))

    def _more_tags(self):
        # FIXME
        return {"_skip_test": True}


def _available_if_base_estimator_has(attr):
    """Return a function to check if `base_estimator` or `estimators_` has `attr`.

    Helper for Chain implementations.
    """

    def _check(self):
        return hasattr(self.base_estimator, attr) or all(
            hasattr(est, attr) for est in self.estimators_
        )

    return available_if(_check)


class _BaseChain(BaseEstimator, metaclass=ABCMeta):
    def __init__(self, base_estimator, *, order=None, cv=None, random_state=None):
        self.base_estimator = base_estimator
        self.order = order
        self.cv = cv
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, Y, **fit_params):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        **fit_params : dict of string -> object
            Parameters passed to the `fit` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        X, Y = self._validate_data(X, Y, multi_output=True, accept_sparse=True)

        random_state = check_random_state(self.random_state)
        self.order_ = self.order
        if isinstance(self.order_, tuple):
            self.order_ = np.array(self.order_)

        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif isinstance(self.order_, str):
            if self.order_ == "random":
                self.order_ = random_state.permutation(Y.shape[1])
        elif sorted(self.order_) != list(range(Y.shape[1])):
            raise ValueError("invalid order")

        self.estimators_ = [clone(self.base_estimator) for _ in range(Y.shape[1])]

        if self.cv is None:
            Y_pred_chain = Y[:, self.order_]
            if sp.issparse(X):
                X_aug = sp.hstack((X, Y_pred_chain), format="lil")
                X_aug = X_aug.tocsr()
            else:
                X_aug = np.hstack((X, Y_pred_chain))

        elif sp.issparse(X):
            Y_pred_chain = sp.lil_matrix((X.shape[0], Y.shape[1]))
            X_aug = sp.hstack((X, Y_pred_chain), format="lil")

        else:
            Y_pred_chain = np.zeros((X.shape[0], Y.shape[1]))
            X_aug = np.hstack((X, Y_pred_chain))

        del Y_pred_chain

        for chain_idx, estimator in enumerate(self.estimators_):
            y = Y[:, self.order_[chain_idx]]
            estimator.fit(X_aug[:, : (X.shape[1] + chain_idx)], y, **fit_params)
            if self.cv is not None and chain_idx < len(self.estimators_) - 1:
                col_idx = X.shape[1] + chain_idx
                cv_result_proba = cross_val_predict(
                    self.base_estimator, X_aug[:, :col_idx], y=y, cv=self.cv, method='predict_proba'
                )[:, 1]
                if sp.issparse(X_aug):
                    X_aug[:, col_idx] = np.expand_dims(cv_result_proba, 1)
                else:
                    X_aug[:, col_idx] = cv_result_proba

        return self

    def predict(self, X):
        """Predict on the data matrix X using the ClassifierChain model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_pred : array-like of shape (n_samples, n_classes)
            The predicted values.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=True, reset=False)
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        Y_prob_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            # use previous probabilities as additional feature in X_aug but return predictions
            previous_predictions = Y_prob_chain[:, :chain_idx]
            if sp.issparse(X):
                if chain_idx == 0:
                    X_aug = X
                else:
                    X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)
            Y_prob_chain[:, chain_idx] = estimator.predict_proba(X_aug)[:,1]

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_pred = Y_pred_chain[:, inv_order]

        return Y_pred


class TennisClassifierChain(MetaEstimatorMixin, ClassifierMixin, _BaseChain):
    """A multi-label model that arranges binary classifiers into a chain.

    Each model makes a prediction in the order specified by the chain using
    all of the available features provided to the model plus the predictions
    of models that are earlier in the chain.

    Read more in the :ref:`User Guide <classifierchain>`.

    .. versionadded:: 0.19

    Parameters
    ----------
    base_estimator : estimator
        The base estimator from which the classifier chain is built.

    order : array-like of shape (n_outputs,) or 'random', default=None
        If `None`, the order will be determined by the order of columns in
        the label matrix Y.::

            order = [0, 1, 2, ..., Y.shape[1] - 1]

        The order of the chain can be explicitly set by providing a list of
        integers. For example, for a chain of length 5.::

            order = [1, 3, 2, 4, 0]

        means that the first model in the chain will make predictions for
        column 1 in the Y matrix, the second model will make predictions
        for column 3, etc.

        If order is `random` a random ordering will be used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines whether to use cross validated predictions or true
        labels for the results of previous estimators in the chain.
        Possible inputs for cv are:

        - None, to use true labels when fitting,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

    random_state : int, RandomState instance or None, optional (default=None)
        If ``order='random'``, determines random number generation for the
        chain order.
        In addition, it controls the random seed given at each `base_estimator`
        at each chaining iteration. Thus, it is only used when `base_estimator`
        exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    classes_ : list
        A list of arrays of length ``len(estimators_)`` containing the
        class labels for each estimator in the chain.

    estimators_ : list
        A list of clones of base_estimator.

    order_ : list
        The order of labels in the classifier chain.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `base_estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    RegressorChain : Equivalent for regression.
    MultioutputClassifier : Classifies each output independently rather than
        chaining.

    References
    ----------
    Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank, "Classifier
    Chains for Multi-label Classification", 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.multioutput import ClassifierChain
    >>> X, Y = make_multilabel_classification(
    ...    n_samples=12, n_classes=3, random_state=0
    ... )
    >>> X_train, X_test, Y_train, Y_test = train_test_split(
    ...    X, Y, random_state=0
    ... )
    >>> base_lr = LogisticRegression(solver='lbfgs', random_state=0)
    >>> chain = ClassifierChain(base_lr, order='random', random_state=0)
    >>> chain.fit(X_train, Y_train).predict(X_test)
    array([[1., 1., 0.],
           [1., 0., 0.],
           [0., 1., 0.]])
    >>> chain.predict_proba(X_test)
    array([[0.8387..., 0.9431..., 0.4576...],
           [0.8878..., 0.3684..., 0.2640...],
           [0.0321..., 0.9935..., 0.0625...]])
    """

    def fit(self, X, Y):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        Returns
        -------
        self : object
            Class instance.
        """
        super().fit(X, Y)
        self.classes_ = [
            estimator.classes_ for chain_idx, estimator in enumerate(self.estimators_)
        ]
        return self

    @_available_if_base_estimator_has("predict_proba")
    def predict_proba(self, X):
        """Predict probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_prob : array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        X = self._validate_data(X, accept_sparse=True, reset=False)
        Y_prob_chain = np.zeros((X.shape[0], len(self.estimators_)))
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            # use previous predicted probabilities instead of predictions
            previous_predictions = Y_prob_chain[:, :chain_idx]
            if sp.issparse(X):
                X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_prob_chain[:, chain_idx] = estimator.predict_proba(X_aug)[:, 1]
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)
        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_prob = Y_prob_chain[:, inv_order]

        return Y_prob

    @_available_if_base_estimator_has("decision_function")
    def decision_function(self, X):
        """Evaluate the decision_function of the models in the chain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_decision : array-like of shape (n_samples, n_classes)
            Returns the decision function of the sample for each model
            in the chain.
        """
        X = self._validate_data(X, accept_sparse=True, reset=False)
        Y_decision_chain = np.zeros((X.shape[0], len(self.estimators_)))
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_decision_chain[:, chain_idx] = estimator.decision_function(X_aug)
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_decision = Y_decision_chain[:, inv_order]

        return Y_decision

    def _more_tags(self):
        return {"_skip_test": True, "multioutput_only": True}


class TennisRegressorChain(MetaEstimatorMixin, RegressorMixin, _BaseChain):
    """A multi-label model that arranges regressions into a chain.

    Each model makes a prediction in the order specified by the chain using
    all of the available features provided to the model plus the predictions
    of models that are earlier in the chain.

    Read more in the :ref:`User Guide <regressorchain>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    base_estimator : estimator
        The base estimator from which the classifier chain is built.

    order : array-like of shape (n_outputs,) or 'random', default=None
        If `None`, the order will be determined by the order of columns in
        the label matrix Y.::

            order = [0, 1, 2, ..., Y.shape[1] - 1]

        The order of the chain can be explicitly set by providing a list of
        integers. For example, for a chain of length 5.::

            order = [1, 3, 2, 4, 0]

        means that the first model in the chain will make predictions for
        column 1 in the Y matrix, the second model will make predictions
        for column 3, etc.

        If order is 'random' a random ordering will be used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines whether to use cross validated predictions or true
        labels for the results of previous estimators in the chain.
        Possible inputs for cv are:

        - None, to use true labels when fitting,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

    random_state : int, RandomState instance or None, optional (default=None)
        If ``order='random'``, determines random number generation for the
        chain order.
        In addition, it controls the random seed given at each `base_estimator`
        at each chaining iteration. Thus, it is only used when `base_estimator`
        exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    estimators_ : list
        A list of clones of base_estimator.

    order_ : list
        The order of labels in the classifier chain.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `base_estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    ClassifierChain : Equivalent for classification.
    MultiOutputRegressor : Learns each output independently rather than
        chaining.

    Examples
    --------
    >>> from sklearn.multioutput import RegressorChain
    >>> from sklearn.linear_model import LogisticRegression
    >>> logreg = LogisticRegression(solver='lbfgs',multi_class='multinomial')
    >>> X, Y = [[1, 0], [0, 1], [1, 1]], [[0, 2], [1, 1], [2, 0]]
    >>> chain = RegressorChain(base_estimator=logreg, order=[0, 1]).fit(X, Y)
    >>> chain.predict(X)
    array([[0., 2.],
           [1., 1.],
           [2., 0.]])
    """

    def fit(self, X, Y, **fit_params):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        **fit_params : dict of string -> object
            Parameters passed to the `fit` method at each step
            of the regressor chain.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        super().fit(X, Y, **fit_params)
        return self

    def _more_tags(self):
        return {"multioutput_only": True}

class TennisLiveScoreMultiClassifier(MultiOutputClassifier):
    def predict_proba(self, X):
        predict_proba = super().predict_proba(X)
        return predict_proba


class TennisLiveWinnerClassifierChainEnhanced(TennisClassifierChain):
    def predict_proba(self, X):
        predict_proba = super().predict_proba(X)
        predict_proba = self._adjust_set_win_prob_to_game_win_prob_if_tiebreak(
            X, predict_proba
        )
        predict_proba = self._adjust_match_win_prob_to_set_win_prob_if_3rd_set(
            X, predict_proba
        )
        return predict_proba

    @staticmethod
    def _adjust_set_win_prob_to_game_win_prob_if_tiebreak(X, predicted_proba):
        set_index = 2
        game_index = 1
        X = X.reset_index(drop=True)
        tiebreak_indices = list(X[X["isTiebreak"]].index.values)
        for i in tiebreak_indices:
            predicted_proba[i][set_index] = predicted_proba[i][game_index]
        return predicted_proba

    @staticmethod
    def _adjust_match_win_prob_to_set_win_prob_if_3rd_set(X, predicted_proba):
        match_index = 3
        set_index = 2
        X = X.reset_index(drop=True)
        third_set_indices = list(
            X[
                (X["matchSetsRemainingTeam1"] == 1)
                & (X["matchSetsRemainingTeam2"] == 1)
                & (X["matchSetDiffTeam1"] == 0)
            ].index.values
        )
        # third_set_indices = list(X[(X['matchSetsRemaining'] == 1) & (X['matchSetDiffTeam1'] == 0)].index.values)
        for i in third_set_indices:
            predicted_proba[i][match_index] = predicted_proba[i][set_index]
        return predicted_proba
def normalize_df(df, anchor_df=None):
    for col in df.columns:
        data = df[col]
        if anchor_df is None:
            df[col] = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            df[col] = (data - np.min(anchor_df[col])) / (np.max(anchor_df[col]) - np.min(anchor_df[col]))
    return df

def create_train_test_val_df(
    df,
    input_names,
    output_name,
    group_col="game_code",
    mask_test_season=2021,
    mask_val_season=2020,
    normalize=False
):
    mask_train = ~(df.season.isin([mask_test_season, mask_val_season]))
    mask_test = (df.season == mask_test_season)
    mask_val = (df.season == mask_val_season)
    if normalize==False:
        X_train = df.loc[mask_train, input_names]
        X_test = df.loc[mask_test, input_names]
        X_val = df.loc[mask_val, input_names]
    else:
        X_train = normalize_df(df.loc[mask_train, input_names])
        X_test = normalize_df(df.loc[mask_test, input_names], df.loc[mask_train, input_names])
        X_val = normalize_df(df.loc[mask_val, input_names], df.loc[mask_train, input_names])
    y_train = df.loc[mask_train, output_name]
    group_train = df.loc[mask_train, group_col]
    y_test = df.loc[mask_test, output_name]
    group_test = df.loc[mask_test, group_col]
    y_val = df.loc[mask_val, output_name]
    group_val = df.loc[mask_val, group_col]
    return X_train, y_train, group_train, X_test, y_test, group_test, X_val, y_val, group_val