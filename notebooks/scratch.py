    results_df = results_df.merge(team_names[["division_id", "conference_id"]], left_on="team_id", right_index=True, sort=False)
    results_df = results_df.merge(team_names[["division_id", "conference_id"]], left_on="opponent_id", right_index=True, sort=False, suffixes=["_team", "_opponent"])
