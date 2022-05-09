from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain

from TennisClassifierChain import TennisClassifierChain


class TennisLiveWinnerClassifierChain(ClassifierChain):
    def predict_proba(self, X):
        predict_proba = super().predict_proba(X)
        return predict_proba

    @staticmethod
    def _adjust_set_win_prob_to_game_win_prob_if_tiebreak(X, predicted_proba):
        set_index = 2
        game_index = 1
        X = X.reset_index(drop=True)
        tiebreak_indices = list(X[X["isTiebreak"]].index.values)
        for i in tiebreak_indices:
            predicted_proba[i][set_index] = predicted_proba[i][game_index]
        return predicted_proba

    @staticmethod
    def _adjust_match_win_prob_to_set_win_prob_if_3rd_set(X, predicted_proba):
        match_index = 3
        set_index = 2
        X = X.reset_index(drop=True)
        third_set_indices = list(
            X[
                (X["matchSetsRemainingTeam1"] == 1)
                & (X["matchSetsRemainingTeam2"] == 1)
                & (X["matchSetDiffTeam1"] == 0)
            ].index.values
        )
        # third_set_indices = list(X[(X['matchSetsRemaining'] == 1) & (X['matchSetDiffTeam1'] == 0)].index.values)
        for i in third_set_indices:
            predicted_proba[i][match_index] = predicted_proba[i][set_index]
        return predicted_proba


class TennisLiveWinnerClassifierChainEnhanced(TennisClassifierChain):
    def predict_proba(self, X):
        predict_proba = super().predict_proba(X)
        return predict_proba

    @staticmethod
    def _adjust_set_win_prob_to_game_win_prob_if_tiebreak(X, predicted_proba):
        set_index = 2
        game_index = 1
        X = X.reset_index(drop=True)
        tiebreak_indices = list(X[X["isTiebreak"]].index.values)
        for i in tiebreak_indices:
            predicted_proba[i][set_index] = predicted_proba[i][game_index]
        return predicted_proba

    @staticmethod
    def _adjust_match_win_prob_to_set_win_prob_if_3rd_set(X, predicted_proba):
        match_index = 3
        set_index = 2
        X = X.reset_index(drop=True)
        third_set_indices = list(
            X[
                (X["matchSetsRemainingTeam1"] == 1)
                & (X["matchSetsRemainingTeam2"] == 1)
                & (X["matchSetDiffTeam1"] == 0)
            ].index.values
        )
        # third_set_indices = list(X[(X['matchSetsRemaining'] == 1) & (X['matchSetDiffTeam1'] == 0)].index.values)
        for i in third_set_indices:
            predicted_proba[i][match_index] = predicted_proba[i][set_index]
        return predicted_proba


class TennisLiveWinnerClassifier(RandomForestClassifier):
    def predict_proba(self, X):
        predict_proba = super().predict_proba(X)
        return predict_proba

    @staticmethod
    def _adjust_set_win_prob_to_game_win_prob_if_tiebreak(X, predicted_proba):
        set_index = 2
        game_index = 1
        X = X.reset_index(drop=True)
        tiebreak_indices = list(X[X["isTiebreak"]].index.values)
        for i in tiebreak_indices:
            predicted_proba[set_index][i] = predicted_proba[game_index][i]
        return predicted_proba

    @staticmethod
    def _adjust_match_win_prob_to_set_win_prob_if_3rd_set(X, predicted_proba):
        match_index = 3
        set_index = 2
        X = X.reset_index(drop=True)
        third_set_indices = list(
            X[
                (X["matchSetsRemainingTeam1"] == 1)
                & (X["matchSetsRemainingTeam2"] == 1)
                & (X["matchSetDiffTeam1"] == 0)
            ].index.values
        )
        # third_set_indices = list(X[(X['matchSetsRemaining'] == 1) & (X['matchSetDiffTeam1'] == 0)].index.values)
        for i in third_set_indices:
            predicted_proba[match_index][i] = predicted_proba[set_index][i]
        return predicted_proba


class TennisLiveWinnerMultiClassifier(MultiOutputClassifier):
    def predict_proba(self, X):
        predict_proba = super().predict_proba(X)
        return predict_proba

    @staticmethod
    def _adjust_set_win_prob_to_game_win_prob_if_tiebreak(X, predicted_proba):
        set_index = 2
        game_index = 1
        X = X.reset_index(drop=True)
        tiebreak_indices = list(X[X["isTiebreak"]].index.values)
        for i in tiebreak_indices:
            predicted_proba[set_index][i] = predicted_proba[game_index][i]
        return predicted_proba

    @staticmethod
    def _adjust_match_win_prob_to_set_win_prob_if_3rd_set(X, predicted_proba):
        match_index = 3
        set_index = 2
        X = X.reset_index(drop=True)
        third_set_indices = list(
            X[
                (X["matchSetsRemainingTeam1"] == 1)
                & (X["matchSetsRemainingTeam2"] == 1)
                & (X["matchSetDiffTeam1"] == 0)
            ].index.values
        )
        # third_set_indices = list(X[(X['matchSetsRemaining'] == 1) & (X['matchSetDiffTeam1'] == 0)].index.values)
        for i in third_set_indices:
            predicted_proba[match_index][i] = predicted_proba[set_index][i]
        return predicted_proba


class TennisLiveScoreMultiClassifier(MultiOutputClassifier):
    def predict_proba(self, X):
        predict_proba = super().predict_proba(X)

        return predict_proba
