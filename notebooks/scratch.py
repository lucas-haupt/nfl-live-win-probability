average_velo = (
    pd.concat(
        [
            df_past_season[df_past_season["PITCH_TYPE_ID"].isin([11, 12, 9])],
            knn_current_season,
        ]
    )
    .groupby(["PITCHER_ID", "PITCH_TYPE_ID"], as_index=False)
    .mean()[["PITCHER_ID", "PITCH_TYPE_ID", "start_speed"]]
    .sort_values("start_speed", ascending=False)
    .drop_duplicates("PITCHER_ID")
)
average_velo
