def create_train_test_val_df(
    df,
    input_names,
    output_name,
    mask_train=(event_df.season <= 2019) & (event_df.continuation == 0),
    mask_test=(event_df.season == 2021) & (event_df.continuation == 0),
    mask_val=(event_df.season == 2020) & (event_df.continuation == 0),
):
    X_train = 0
