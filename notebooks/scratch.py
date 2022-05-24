        if show_rf == True:
            ax[0].plot(score_difference_list, pred_output, 'green', label='rf', linewidth=3, marker="o")
        if show_log_basic == True:
            ax[0].plot(score_difference_list, pred_output_basic_loglin, '#AABA00', label='log_basic', linewidth=3, marker="o")
        if show_mlp_basic == True:
            ax[0].plot(score_difference_list, pred_output_basic_mlp, '#A200BA', label='mlp_basic', linewidth=3, marker="o")
        if show_mlp_vegas == True:
            ax[0].plot(score_difference_list, pred_output_basic_mlp_vegas, '#A200BA', label='mlp_vegas', linewidth=3, marker="o")
        if show_mlp_advanced == True:
            ax[0].plot(score_difference_list, pred_output_advanced_mlp, 'blue', label='mlp_advanced', linewidth=3, marker="o")
        if show_log_advanced == True:
            ax[0].plot(score_difference_list, pred_output_advanced_log, '#009B9C', label='loglin_advanced', linewidth=3, marker="o")