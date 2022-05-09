print(brier_score_loss(model_df[mask_model]["home_team_win_in_regulation"], model_df[mask_model]["xhome_win_rf_search_clipped_rf_drive_preds"]))
print(brier_score_loss(model_df[mask_model]["home_team_win_in_regulation"], model_df[mask_model]["xhome_win_rf_search_clipped_mlp_drive_preds"]))
print(brier_score_loss(model_df[mask_model]["home_team_win_in_regulation"], model_df[mask_model]["xhome_win_mlp_search_clipped_rf_drive_preds"]))
print(brier_score_loss(model_df[mask_model]["home_team_win_in_regulation"], model_df[mask_model]["xhome_win_mlp_search_clipped_mlp_drive_preds"]))
