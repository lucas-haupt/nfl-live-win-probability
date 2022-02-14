import numpy as np
import time
import pandas as pd

max_home_score = 62
max_away_score = 59
n_categories = (max_home_score + 1) * (max_away_score + 1)


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
