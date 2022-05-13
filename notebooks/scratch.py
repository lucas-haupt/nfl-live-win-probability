for key in search_mlp_score_diff_clipped_mlp_drive_preds.best_params_.keys():
    rf_grid[key] = [search_rf_score_diff_clipped_mlp_drive_preds.best_params_[key]]
for key in search_mlp_score_diff_clipped_mlp_drive_preds.best_params_.keys():
    mlp_grid[key] = [search_mlp_score_diff_clipped_mlp_drive_preds.best_params_[key]]
