big_cv = GroupKFold(n_splits=5)
final_model = cb.CatBoostClassifier(n_estimators=search_game_outcome.best_estimator_.tree_count_,**search_game_outcome.best_params_,cat_features=categoricals,verbose=1)
predictions_game_outcome = cross_val_predict(final_model,X_all, y_all_game_outcome,groups=group_all_game_outcome,cv=big_cv,verbose=1,n_jobs=1, method="predict_proba")