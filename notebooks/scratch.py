    model_df["absolute_score_diff"] = abs(model_df["home_start_score"] - model_df["away_start_score"])

    model_df["minutes"] = (model_df["play_start_time"].fillna(0) // 60).apply(int)
    model_df["seconds"] = (
        model_df["play_start_time"].fillna(0) - (model_df["play_start_time"].fillna(0) // 60) * 60
    ).apply(int)
    model_df["seconds_str"] = np.where(
        model_df["seconds"] >= 10, model_df["seconds"].apply(str), "0" + model_df["seconds"].apply(str)
    )
    model_df["time_str"] = model_df["minutes"].apply(str) + ":" + model_df["seconds_str"]

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
