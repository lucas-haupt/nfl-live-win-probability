search_mlp_score_diff_clipped_rf_drive_preds = pickle.load(open(os.path.join(root_dir, "models/search_mlp_score_diff_clipped_rf_drive_preds.p"), 'rb'))


input_names_score_pred = [item for item in input_names if item not in ["punt", "field_goal_attempt"]] + ["kick_off", "point_after_kick", "two_point_attempt"] + ["search_rf_play_first_down_home", "search_rf_play_first_down_away"] + search_rf_drive_class_names_home[1:] + search_rf_drive_class_names_away[1:]
output_name = "end_of_regulation_score_diff_change_clipped"

mask_model_score_diff = (
    (model_df.continuation==0)&
    (model_df[input_names_score_pred+[output_name]].notna().all(axis=1))&
    ~(model_df.event_id.isin([12,57,58,13]))&
    (model_df["overtime"]==0)
)


normalized_score_pred_df = normalize_df(model_df[mask_model_score_diff][input_names_score_pred], model_df[mask_model_score_diff & (model_df.season<2020)][input_names_score_pred])
mlp_search_score_diff_clipped_rf_drive_preds_preds = pd.DataFrame(search_mlp_score_diff_clipped_rf_drive_preds.predict_proba(normalized_score_pred_df), index=model_df[mask_model_score_diff].index)
score_diff_clipped_rf_drive_preds_matrix = pd.DataFrame(np.zeros(mlp_search_score_diff_clipped_rf_drive_preds_preds.shape), index=mlp_search_score_diff_clipped_rf_drive_preds_preds.index)
score_diff_change_list_clipped = list(model_df.end_of_regulation_score_diff_change_clipped.drop_duplicates().sort_values())

for column in score_diff_clipped_rf_drive_preds_matrix.columns:
    score_diff_clipped_rf_drive_preds_matrix[column] = score_diff_change_list_clipped[column] + model_df["current_score_diff"]



model_df["xhome_win_mlp_search_clipped_rf_drive_preds"] = np.sum(mlp_search_score_diff_clipped_rf_drive_preds_preds.T[score_diff_clipped_rf_drive_preds_matrix.T>0], axis=0)
model_df["xovertime_mlp_search_clipped_rf_drive_preds"] = np.sum(mlp_search_score_diff_clipped_rf_drive_preds_preds.T[score_diff_clipped_rf_drive_preds_matrix.T==0], axis=0)
model_df["xaway_win_mlp_search_clipped_rf_drive_preds"] = np.sum(mlp_search_score_diff_clipped_rf_drive_preds_preds.T[score_diff_clipped_rf_drive_preds_matrix.T<0], axis=0)
model_df["xend_of_regulation_score_diff_mlp_search_clipped_rf_drive_preds"] = np.sum(score_diff_clipped_rf_drive_preds_matrix * mlp_search_score_diff_clipped_rf_drive_preds_preds, axis=1)
