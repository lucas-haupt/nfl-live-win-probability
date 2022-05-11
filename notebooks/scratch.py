model_df["home_team_win_in_regulation"] = np.where(model_df["end_of_regulation_score_diff"]>0, 1, 0)
model_df["away_team_win_in_regulation"] = np.where(model_df["end_of_regulation_score_diff"]<0, 1, 0)
model_df["tie_at_end_of_regulation"] = np.where(model_df["end_of_regulation_score_diff"]==0, 1, 0)
calb_df = model_df[(model_df["season"]==2020)&~(model_df.xhome_win_mlp_search_clipped_rf_drive_preds_vegas_adjusted.isna())]
# calb = calibration.calibration_curve(calb_df["home_team_win"], np.clip(calb_df["xhome_win_score_diff_mlp"], 0, 1), n_bins=10, strategy="quantile")
# plt.plot(calb[1], calb[0], marker="o")
bins = 20

calb = calibration.calibration_curve(calb_df["home_team_win_in_regulation"], np.clip(calb_df["xhome_win_mlp_search_clipped_rf_drive_preds_vegas_adjusted"], 0, 1), n_bins=bins, strategy="quantile")
plt.plot(calb[1], calb[0], marker="o")
calb = calibration.calibration_curve(calb_df["home_team_win_in_regulation"], np.clip(calb_df["xhome_win_mlp_search_clipped_mlp_drive_preds_vegas_adjusted"], 0, 1), n_bins=bins, strategy="quantile")
plt.plot(calb[1], calb[0], marker="o")
calb = calibration.calibration_curve(calb_df["home_team_win_in_regulation"], np.clip(calb_df["xhome_win_rf_search_clipped_rf_drive_preds_vegas_adjusted"], 0, 1), n_bins=bins, strategy="quantile")
plt.plot(calb[1], calb[0], marker="o")
calb = calibration.calibration_curve(calb_df["home_team_win_in_regulation"], np.clip(calb_df["xhome_win_rf_search_clipped_mlp_drive_preds_vegas_adjusted"], 0, 1), n_bins=bins, strategy="quantile")
plt.plot(calb[1], calb[0], marker="o")
plt.plot([0, 1], [0, 1])
