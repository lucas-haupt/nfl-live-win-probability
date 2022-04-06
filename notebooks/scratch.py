full_df["game_info"] = (
    full_df["home_team"]
    + " "
    + full_df["away_team"]
    + " "
    + full_df["game_date_x"].apply(lambda x: x.strftime("%Y-%m-%d"))
    + " "
    + full_df["season_x"].apply(str)
    + " ("
    + (full_df["game_code"]).apply(str)
    + ")"
)
