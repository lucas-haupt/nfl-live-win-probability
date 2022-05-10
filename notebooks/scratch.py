    elif model =="rf_search_rf_vegas_adjusted":
        y = ["xhome_win_rf_search_clipped_rf_drive_preds_vegas_adjusted", "xaway_win_rf_search_clipped_rf_drive_preds_vegas_adjusted", "xovertime_rf_search_clipped_rf_drive_preds_vegas_adjusted"]
        y1 = "xend_of_regulation_score_diff_rf_search_clipped_rf_drive_preds_vegas_adjusted"
        y2 = "xend_of_regulation_score_total_rf_search"
        y3 = search_rf_drive_class_names
        y4 = "search_rf_play_first_down"
    elif model =="rf_search_mlp_vegas_adjusted":
        y = ["xhome_win_rf_search_clipped_mlp_drive_preds_vegas_adjusted", "xaway_win_rf_search_clipped_mlp_drive_preds_vegas_adjusted", "xovertime_rf_search_clipped_mlp_drive_preds_vegas_adjusted"]
        y1 = "xend_of_regulation_score_diff_rf_search_clipped_mlp_drive_preds_vegas_adjusted"
        y2 = "xend_of_regulation_score_total_rf_search"
        y3 = search_mlp_drive_class_names
        y4 = "search_mlp_play_first_down"
    elif model =="mlp_search_rf_vegas_adjusted":
        y = ["xhome_win_mlp_search_clipped_rf_drive_preds_vegas_adjusted", "xaway_win_mlp_search_clipped_rf_drive_preds_vegas_adjusted", "xovertime_mlp_search_clipped_rf_drive_preds_vegas_adjusted"]
        y1 = "xend_of_regulation_score_diff_mlp_search_clipped_rf_drive_preds_vegas_adjusted"
        y2 = "xend_of_regulation_score_total_mlp_search"
        y3 = search_rf_drive_class_names
        y4 = "search_rf_play_first_down"
    elif model =="mlp_search_mlp_vegas_adjusted":
        y = ["xhome_win_mlp_search_clipped_mlp_drive_preds_vegas_adjusted", "xaway_win_mlp_search_clipped_mlp_drive_preds_vegas_adjusted", "xovertime_mlp_search_clipped_mlp_drive_preds_vegas_adjusted"]
        y1 = "xend_of_regulation_score_diff_mlp_search_clipped_mlp_drive_preds_vegas_adjusted"
        y2 = "xend_of_regulation_score_total_mlp_search"
        y3 = search_mlp_drive_class_names
        y4 = "search_mlp_play_first_down"
