from collections import defaultdict
from numba.core.decorators import njit
import numpy as np
import mlflow
import boto3
import json
import scipy.stats as st
from operator import itemgetter
import random
import math
import copy
from typing import Dict
from helper import elo_update, get_win_prob_from_elo
from generate_test_data import load_dataset
import pickle
import itertools
import math
import time
import numba as nb
from functools import wraps


@nb.njit
def _create_dict(items):
    return {k: v for k, v in items}


def dict_lists_to_numpy_arrays(dict_list):
    """Convert dictionary whose values are lists into numpy arrays"""
    return {k: np.array(v) for k, v in dict_list.items()}


def dict_to_numba(d):
    # HACK
    if isinstance(list(d.values())[0], list):
        return _create_dict(tuple(dict_lists_to_numpy_arrays(d).items()))
    return _create_dict(tuple(d.items()))


def cls_njit_with_dicts(func):
    @wraps(func)
    def wrapper_decorator(self, *args, **kwargs):
        for i, arg in enumerate(args):
            if isinstance(arg, dict):
                if isinstance(list(arg.values())[0], dict):
                    continue
                args[i] = dict_to_numba(arg)
        for k, v in kwargs.items():
            if isinstance(v, dict):
                if isinstance(list(k.values())[0], dict):
                    continue
                kwargs[k] = dict_to_numba(v)
        return func(*args, **kwargs)

    return wrapper_decorator


def njit_with_dicts(func):
    @wraps(func)
    def wrapper_decorator(*args, **kwargs):
        new_args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, dict):
                if isinstance(list(arg.values())[0], dict):
                    continue
                try:
                    new_args[i] = dict_to_numba(arg)
                except:
                    breakpoint()
        for k, v in kwargs.items():
            if isinstance(v, dict):
                if isinstance(list(k.values())[0], dict):
                    continue
                kwargs[k] = dict_to_numba(v)
        return func(*new_args, **kwargs)

    return wrapper_decorator


@njit_with_dicts
def optimized_dictionary_averager(d):
    """d = __win_games"""
    return {k: np.average(v) for k, v in d.items()}


# def team_make_playoff_prob(self):
#     for team_id in self.__win_games.keys():
#         self.__team_make_playoff_prob[team_id] = sum(
#             self.__make_playoff[team_id]
#         ) / len(
#             self.__make_playoff[team_id]
#         )  # average
#     return self.__team_make_playoff_prob


OLD_ELO = {
    323: 1436.7492822543,
    324: 1617.11121643316,
    325: 1411.97901199442,
    326: 1384.62942219053,
    327: 1489.82063632216,
    329: 1560.29866635805,
    331: 1482.56657581514,
    332: 1457.89916745627,
    334: 1333.82771062622,
    335: 1623.46777066143,
    336: 1543.85123456974,
    338: 1536.24571392503,
    339: 1513.36509762446,
    341: 1560.53264343334,
    343: 1570.67971499915,
    345: 1382.46496334842,
    347: 1517.50820167989,
    348: 1498.50775614176,
    350: 1574.03926486236,
    351: 1464.25469451229,
    352: 1395.43059396555,
    354: 1442.1873834701,
    355: 1552.54019743416,
    356: 1477.38104716702,
    357: 1466.41789646329,
    359: 1582.50829161309,
    361: 1506.11662186044,
    362: 1666.97097623008,
    363: 1472.73673168795,
    364: 1460.88728570045,
    365: 1366.49849987053,
    366: 1622.4306357813,
}


CURRENT_ELO = {
    1: 1575.87757569949,
    2: 1480.3830709241574,
    3: 1457.1890873647035,
    4: 1475.335650764823,
    5: 1286.4240542566952,
    6: 1532.4708046538499,
    7: 1582.307569237937,
    8: 1325.779304637848,
    9: 1519.2938016413668,
    10: 1277.7188892758572,
    11: 1462.6109211419907,
    12: 1642.4345107271258,
    13: 1545.0134212687979,
    14: 1495.7944512483225,
    15: 1689.9491550496966,
    16: 1399.1706403086823,
    17: 1620.6947855387734,
    18: 1545.8133575326733,
    19: 1256.4749744450796,
    20: 1626.647986469997,
    21: 1677.971533415999,
    22: 1584.3516292999896,
    23: 1418.4753804366617,
    24: 1448.4592820606458,
    25: 1229.092784520263,
    26: 1634.282424203115,
    27: 1476.105645887519,
    28: 1413.5611077959925,
    29: 1532.679083724397,
    5312: 1401.5259467421654,
}


CURRENT_ELO = {int(k): v for k, v in CURRENT_ELO.items()}


# @njit_with_dicts
def update_current_season_ratings(ratings):
    """Regress team Elo for some reason.
    # TODO: Why?"""
    for team in ratings.keys():
        ratings[team] = ratings[team] * 0.6 + 1500.0 * 0.4
        return ratings


class RegularSeasonRecord:
    def __init__(self, team_info):
        self.record = {}
        # self.win_games = {}
        self.win_games = defaultdict(lambda: 0)
        # self.team_games_result = numba_dict({})
        # self.head_to_head_team_win_games = {}
        self.head_to_head_team_win_games = defaultdict(
            lambda: defaultdict(lambda: {"win_number": 0, "game_number": 0})
        )
        # {team_id:{opponent_team_id_1:{win_number:1, game_number:2},...},...}
        self.team_info = team_info
        self.conf_standing = {
            conf_id: [] for conf_id in set(self.team_info["conf_id"].values())
        }
        self.sub_div_standing = {
            sub_div_id: [] for sub_div_id in set(self.team_info["sub_div_id"].values())
        }
        # TODO: sub_div = division?
        self.playoff_bracket = {}

    def add_game(self, game_id, game_results):
        """Add a single game to the simulation record.
        Includes: win count and head-to-head count"""
        self.record[game_id] = game_results
        team_ids = list(game_results.keys())
        for team_id in team_ids:
            result = game_results[team_id]["result"]
            # if team_id not in self.win_games.keys():
            #     self.win_games[team_id] = 0

            self.win_games[team_id] += result
            opponent_team_id = team_ids[0] if team_ids[1] == team_id else team_ids[1]
            # if team_id not in self.head_to_head_team_win_games.keys():
            #     self.head_to_head_team_win_games[team_id] = {}

            # if opponent_team_id not in self.head_to_head_team_win_games[team_id].keys():

            #     self.head_to_head_team_win_games[team_id][opponent_team_id] = {
            #         "win_number": 0,
            #         "game_number": 0,
            #     }

            self.head_to_head_team_win_games[team_id][opponent_team_id][
                "game_number"
            ] += 1
            self.head_to_head_team_win_games[team_id][opponent_team_id][
                "win_number"
            ] += result

    def two_team_tie_breaker_div_winner(self, team_1, team_2):

        if (
            self.head_to_head_team_win_games[team_1][team_2]["win_number"]
            / self.head_to_head_team_win_games[team_1][team_2]["game_number"]
        ) > 0.5:
            return [team_1, team_2]
        elif (
            self.head_to_head_team_win_games[team_1][team_2]["win_number"]
            / self.head_to_head_team_win_games[team_1][team_2]["game_number"]
        ) < 0.5:
            return [team_2, team_1]
        else:
            return random.sample([team_1, team_2], 2)

        # Two-Team Tiebreaker:

        # 1. Better record in head-to-head games
        # 2. Division winner (this criterion is applied regardless of whether the tied teams are in the same division)
        # Todo 3. Higher winning percentage within division (if teams are in the same division)
        # Todo 4. Higher winning percentage in conference games
        # Todo 5. Higher winning percentage against playoff teams in own conference
        # Todo 6. Higher winning percentage against playoff teams in opposite conference
        # Todo 7. Higher point differential between points scored and points allowed

        # Todo Multiple-Team Tiebreaker

    def two_team_tie_breaker_conf_1(self, team_1, team_2):

        if (
            self.head_to_head_team_win_games[team_1][team_2]["win_number"]
            / self.head_to_head_team_win_games[team_1][team_2]["game_number"]
            > 0.5
        ):
            return [team_1, team_2]
        elif (
            self.head_to_head_team_win_games[team_1][team_2]["win_number"]
            / self.head_to_head_team_win_games[team_1][team_2]["game_number"]
            < 0.5
        ):
            return [team_2, team_1]
        else:
            return self.two_team_tie_breaker_conf_2(team_1, team_2)

    def two_team_tie_breaker_conf_2(self, team_1, team_2):

        if (
            team_1 == self.sub_div_standing[self.team_info["sub_div_id"][team_1]][0]
            and team_2 != self.sub_div_standing[self.team_info["sub_div_id"][team_2]][0]
        ):
            return [team_1, team_2]
        elif (
            team_1 != self.sub_div_standing[self.team_info["sub_div_id"][team_1]][0]
            and team_2 == self.sub_div_standing[self.team_info["sub_div_id"][team_2]][0]
        ):
            return [team_2, team_1]
        else:
            return random.sample([team_1, team_2], 2)

    def get_sub_div_standing(self):

        standing = sorted(self.win_games.items(), key=itemgetter(1), reverse=True)

        for team_id in range(len(standing)):
            self.sub_div_standing[
                self.team_info["sub_div_id"][standing[team_id][0]]
            ].append(standing[team_id][0])

        for sub_div_id in self.sub_div_standing.keys():
            sub_div_most_wins = self.win_games[self.sub_div_standing[sub_div_id][0]]
            if (
                self.win_games[self.sub_div_standing[sub_div_id][4]]
                == sub_div_most_wins
            ):
                self.sub_div_standing[sub_div_id] = random.sample(
                    self.sub_div_standing[sub_div_id], 5
                )
            elif (
                self.win_games[self.sub_div_standing[sub_div_id][3]]
                == sub_div_most_wins
            ):
                sub_div_standing_1 = self.sub_div_standing[sub_div_id][:4]
                sub_div_standing_2 = self.sub_div_standing[sub_div_id][4]
                self.sub_div_standing[sub_div_id] = random.sample(sub_div_standing_1, 4)
                self.sub_div_standing[sub_div_id].append(sub_div_standing_2)
            elif (
                self.win_games[self.sub_div_standing[sub_div_id][2]]
                == sub_div_most_wins
            ):
                sub_div_standing_1 = self.sub_div_standing[sub_div_id][:3]
                sub_div_standing_2 = self.sub_div_standing[sub_div_id][3:]
                self.sub_div_standing[sub_div_id] = random.sample(sub_div_standing_1, 3)
                self.sub_div_standing[sub_div_id].extend(sub_div_standing_2)
            elif (
                self.win_games[self.sub_div_standing[sub_div_id][1]]
                == sub_div_most_wins
            ):
                sub_div_standing_1, sub_div_standing_2 = (
                    self.sub_div_standing[sub_div_id][0],
                    self.sub_div_standing[sub_div_id][1],
                )
                sub_div_standing_3 = self.sub_div_standing[sub_div_id][2:]
                self.sub_div_standing[
                    sub_div_id
                ] = self.two_team_tie_breaker_div_winner(
                    sub_div_standing_1, sub_div_standing_2
                )
                self.sub_div_standing[sub_div_id].extend(sub_div_standing_3)

    def get_conf_standing(self):

        standing = sorted(self.win_games.items(), key=itemgetter(1), reverse=True)

        conf_standing = {
            conf_id: [] for conf_id in set(self.team_info["conf_id"].values())
        }
        for team_id in range(len(standing)):
            conf_standing[self.team_info["conf_id"][standing[team_id][0]]].append(
                standing[team_id][0]
            )

        for conf_id in conf_standing.keys():
            idx = 0
            while idx < len(conf_standing[conf_id]):
                last_idx = self.find_last(
                    conf_standing[conf_id][idx:],
                    self.win_games[conf_standing[conf_id][idx]],
                )
                if last_idx == 0:
                    self.conf_standing[conf_id].append(conf_standing[conf_id][idx])
                elif last_idx == 1:
                    self.conf_standing[conf_id].extend(
                        self.two_team_tie_breaker_conf_1(
                            conf_standing[conf_id][idx], conf_standing[conf_id][idx + 1]
                        )
                    )
                else:
                    self.conf_standing[conf_id].extend(
                        random.sample(
                            conf_standing[conf_id][idx : (idx + last_idx + 1)],
                            last_idx + 1,
                        )
                    )
                idx += last_idx + 1

    def find_last(self, standing, wins):
        """TODO: Find last what? Most recent game?"""
        for r_idx, team_id in enumerate(reversed(standing)):
            if self.win_games[team_id] == wins:
                return len(standing) - 1 - r_idx

    def create_playoff_bracket(self):

        for conf_id in set(self.team_info["conf_id"].values()):
            self.playoff_bracket[conf_id] = [
                [self.conf_standing[conf_id][0], self.conf_standing[conf_id][7]],
                [self.conf_standing[conf_id][3], self.conf_standing[conf_id][4]],
                [self.conf_standing[conf_id][1], self.conf_standing[conf_id][6]],
                [self.conf_standing[conf_id][2], self.conf_standing[conf_id][5]],
            ]


class Sport:
    def __init__(
        self,
        home_court_advantage=92,
        dist_std=12.158544804925146,
        model_uri=None,
        model_use=True,
        team_rating_init=None,
        coef=None,
        intercept=None,
    ):
        # if model_uri is None:
        #     model_uri = "s3://sp-mlplatform-eks-us-east-1-prod/mlflow/artifacts/212/bcd0eab4567a4d70a4e848e0f65fd6a8/artifacts/"
        if team_rating_init is None:
            team_rating_init = update_current_season_ratings(CURRENT_ELO)
        self.home_court_advantage = home_court_advantage
        self.dist_std = dist_std
        self.model_uri = model_uri

        if model_uri is not None:
            print(f"Model URI: {self.model_uri}")
            self.model = mlflow.sklearn.load_model(self.model_uri)
            self.coef = self.model.coef_[0]  # 0.0375340333507193
            self.intercept = self.model.intercept_  # -0.6562550135266716
        else:
            self.coef = coef
            self.intercept = intercept

        self.model_use = model_use
        # TODO: True = Elo. What is the False case?
        self.team_rating = team_rating_init

    def update(self, game: Dict, random_normal_samples):
        pred_mean, pred_std = self.get_std_and_mean_for_game(game)
        if game["team_1_score"] and game["team_2_score"]:
            mov = game["team_1_score"] - game["team_2_score"]

        else:
            mov = self.simulate_results(
                next(random_normal_samples), pred_mean, pred_std
            )

        team_1_elo_before, team_2_elo_before = self.update_team_strengths(game, mov)
        return self.get_new_elo_dict(game, mov, team_1_elo_before, team_2_elo_before)

    def get_std_and_mean_for_game(self, game):
        # returns the expected score diff

        # TODO: what's the if/else for?
        if game["team_1_expected_win_prob"] and game["team_1_expected_score_diff"]:
            mean = game["team_1_expected_score_diff"]
            std = -mean / st.norm.ppf(1 - game["team_1_expected_win_prob"])

        else:
            if not self.model_use:
                mean = (
                    self.team_rating[game["team_1"]]
                    + self.home_court_advantage
                    - self.team_rating[game["team_2"]]
                ) / 28.0

                win_prob = get_win_prob_from_elo(
                    self.team_rating[game["team_1"]] + self.home_court_advantage,
                    self.team_rating[game["team_2"]],
                )
                std = -mean / st.norm.ppf(1 - win_prob)

            else:
                mean = (
                    self.coef
                    * (
                        self.team_rating[game["team_1"]]
                        + self.home_court_advantage
                        - self.team_rating[game["team_2"]]
                    )
                    + self.intercept
                )
                std = self.dist_std

        return mean, std

    @staticmethod
    @nb.njit
    def simulate_results(random_sample, pred_mean, pred_std):
        """I have checked the output of this, and it's not the issue"""
        score_diff = round(random_sample * pred_std + pred_mean)
        return score_diff if score_diff != 0 else int(math.copysign(1, score_diff))

    def get_new_elo_dict(
        self, game, point_differential, team_1_elo_before, team_2_elo_before
    ):
        if point_differential > 0:
            result_1, result_2 = 1, 0
        else:
            result_1, result_2 = 0, 1
        return {
            game["team_1"]: {
                "result": result_1,
                "score_differential": point_differential,
                "elo_before": team_1_elo_before,
                "elo_after": self.team_rating[game["team_1"]],
            },
            game["team_2"]: {
                "result": result_2,
                "score_differential": -point_differential,
                "elo_before": team_2_elo_before,
                "elo_after": self.team_rating[game["team_2"]],
            },
        }

    def update_team_strengths(self, game, res):
        team_1_elo_before, team_2_elo_before = (
            self.team_rating[game["team_1"]],
            self.team_rating[game["team_2"]],
        )
        self.team_rating[game["team_1"]] = (
            team_1_elo_before
            + elo_update(
                res, team_1_elo_before, team_2_elo_before, self.home_court_advantage
            )[0]
        )
        self.team_rating[game["team_2"]] = (
            team_2_elo_before
            + elo_update(
                res, team_1_elo_before, team_2_elo_before, self.home_court_advantage
            )[1]
        )

        return team_1_elo_before, team_2_elo_before


class Schedule:
    def __init__(self, games: Dict):
        self.games = games
        # games = {game_id: {team_1: _, team_2:_, team_1_score: _, team_2_score:_, team_1_expected_win_prob:_, team_1_expected_score_diff:_}}
        # team_1 need to be the home team

    def get_game(self, game_id):
        return self.games[game_id]


# class Schedule(ABC):
#     def __init__(self, regular_season_games):
#         self.regular_season_games = regular_season_games
#         self.rsg_index = 0
#         self.season_over = False
#         # games = {game_id: {date: _, type:normal/tournament, team_1: _, team_2:_}} OR dataframe
#         # One consideration is how
#         pass
#     def games(self):
#         while not self.season_over:
#             if self.rsg_index < len(self.regular_season_games):
#                 yield self.regular_season_games[self.rsg_index]
#             else:
#                 # Yield next playoff game, for now testing with regular season
#                 self.season_over = True
#                 yield None
#     @abstractmethod
#     def playoffs(self, season_record:SeasonRecord):
#         # yield next tournament game
#         pass


class SeasonSimulation:
    def __init__(
        self, sport: Sport, schedule: Schedule, team_info
    ):  # TODO: WTF is team_info?
        self.team_info = team_info
        self.season_record = RegularSeasonRecord(self.team_info)
        self.simulation_record = SimulationRecord()
        try:
            self.sport = copy.deepcopy(sport)
        except:
            breakpoint()
        self.sport_copy = copy.deepcopy(sport)
        self.schedule = schedule
        self.random_normal_samples = itertools.cycle(np.random.normal(size=250000))

    def playoff_matchup_simulation(self, matchup):
        win_games = {}
        win_games[matchup[0]] = 0
        win_games[matchup[1]] = 0
        # win_games = {matchup[0]: 0, matchup[1]: 0}
        home_team_dict = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1, 7: 0}

        for id in range(7):
            game = {
                "team_1": matchup[home_team_dict[id + 1]],
                "team_2": matchup[1 - home_team_dict[id + 1]],
                "team_1_score": None,
                "team_2_score": None,
                "team_1_expected_win_prob": None,
                "team_1_expected_score_diff": None,
            }
            if win_games[game["team_1"]] < 4 and win_games[game["team_2"]] < 4:
                results = self.sport.update(game, self.random_normal_samples)
                for team_id in matchup:
                    win_games[team_id] += results[team_id]["result"]

        if win_games[matchup[0]] == 4:
            return matchup[0]
        else:
            return matchup[1]

    def playoff_simulation(self, playoff_bracket):

        for conf_id in playoff_bracket.keys():
            num_rounds = int(math.log(len(playoff_bracket[conf_id]), 2)) + 1
            for round in range(num_rounds):
                winners = []
                for matchup in playoff_bracket[conf_id]:
                    winner = self.playoff_matchup_simulation(matchup)
                    winners.append(winner)
                playoff_bracket[conf_id] = [
                    winners[i : i + 2] for i in range(0, len(winners), 2)
                ]

        final_matchup = self.season_record.two_team_tie_breaker_conf_1(
            playoff_bracket[1][0][0], playoff_bracket[2][0][0]
        )
        champion = self.playoff_matchup_simulation(final_matchup)

        return final_matchup, champion

    def run(self, iterations):
        random_normal_samples = itertools.cycle(np.random.normal(size=250000))
        for _ in range(iterations):
            self.season_record = RegularSeasonRecord(self.team_info)
            self.sport = copy.deepcopy(self.sport_copy)

            games = [
                self.schedule.get_game(game_id)
                for game_id in self.schedule.games.keys()
            ]
            for game_id, game in zip(self.schedule.games.keys(), games):

                results = self.sport.update(game, random_normal_samples)
                self.season_record.add_game(game_id, results)

            self.season_record.get_sub_div_standing()
            self.season_record.get_conf_standing()
            self.season_record.create_playoff_bracket()
            final_matchup, champion = self.playoff_simulation(
                self.season_record.playoff_bracket
            )
            self.simulation_record.add_season(
                self.season_record, final_matchup, champion
            )


class SimulationRecord:
    """TODO: Does everything simulation-related?"""

    def __init__(self):
        self.record = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: np.empty(4)))
        )
        # frame = {game_id_1: {team_id_1: [], team_id_2: []}, ...}
        self.__win_games = {}
        self.__score_difference = {}
        self.__elo_distribution = {}
        self.__reg_season_final_elo_distribution = {}
        self.__make_playoff = {}
        self.__make_finals = {}
        self.__win_finals = {}
        self.__team_make_playoff_prob = {}
        self.__team_make_finals_prob = {}
        self.__team_win_finals_prob = {}
        self.__team_average_win_games = {}

    def add_season(
        self, season_simulation_record: SeasonSimulation, final_matchup, champion
    ):
        """Store the outputs of a single season simulation.
        i.e. season_record = {12345: }
        """

        season_record = season_simulation_record.record

        for game_id in season_record.keys():
            self.record[game_id] = {
                team_id: season_record[game_id][team_id]
                for team_id in season_record[game_id].keys()
            }

        # record_make_play_off
        self.record_playoff_simulation_outcomes(
            season_simulation_record.conf_standing, final_matchup, champion
        )

        # record_team_wins
        self.record_wins(season_simulation_record.win_games)

        # record_team_score_diff
        self.record_team_score_diff(season_simulation_record.record)

        # record_team_elo
        self.record_team_elo(season_simulation_record.record)

        self.record_team_regular_season_final_elo_dist()

    def record_playoff_simulation_outcomes(
        self, conf_standing, final_matchup, champion
    ):

        for conf_id in conf_standing:
            for idx in range(len(conf_standing[conf_id])):
                team_id = conf_standing[conf_id][idx]
                if team_id not in self.__make_playoff.keys():
                    self.__make_playoff[team_id] = []
                if idx <= 7:
                    self.__make_playoff[team_id].append(1)
                else:
                    self.__make_playoff[team_id].append(0)
                if team_id not in self.__make_finals.keys():
                    self.__make_finals[team_id] = []
                if team_id in final_matchup:
                    self.__make_finals[team_id].append(1)
                else:
                    self.__make_finals[team_id].append(0)
                if team_id not in self.__win_finals.keys():
                    self.__win_finals[team_id] = []
                if team_id == champion:
                    self.__win_finals[team_id].append(1)
                else:
                    self.__win_finals[team_id].append(0)

    def record_wins(self, wins):
        for team_id in wins.keys():
            if team_id not in self.__win_games.keys():
                self.__win_games[team_id] = []

            self.__win_games[team_id].append(wins[team_id])

    def record_team_score_diff(self, season_record):
        for game_id in season_record.keys():
            for team_id in season_record[game_id].keys():
                if team_id not in self.__score_difference.keys():
                    self.__score_difference[team_id] = []

                self.__score_difference[team_id].append(
                    season_record[game_id][team_id]["score_differential"]
                )

    def record_team_elo(self, season_record):
        for game_id in season_record.keys():
            for team_id in season_record[game_id].keys():
                if team_id not in self.__elo_distribution.keys():
                    self.__elo_distribution[team_id] = {}

                if game_id not in self.__elo_distribution[team_id].keys():
                    self.__elo_distribution[team_id][game_id] = []

                self.__elo_distribution[team_id][game_id].append(
                    season_record[game_id][team_id]["elo_after"]
                )

    @staticmethod
    # @lru_cache()
    def get_final_game_ids_from_elo_distribution(elo_distribution):
        return np.array([list(v.keys()) for v in elo_distribution.values()]).max(axis=1)

    def record_team_regular_season_final_elo_dist(self):
        final_game_ids = self.get_final_game_ids_from_elo_distribution(
            self.__elo_distribution
        )
        for (team_id, elo_distribution), final_reg_game_id in zip(
            self.__elo_distribution.items(), final_game_ids
        ):
            self.__reg_season_final_elo_distribution[team_id] = elo_distribution[
                final_reg_game_id
            ]

    def team_make_playoff_prob(self):
        self.__team_make_playoff_prob = optimized_dictionary_averager(
            self.__make_playoff
        )
        return self.__team_make_playoff_prob

    # def team_make_playoff_prob(self):
    #     for team_id in self.__win_games.keys():
    #         self.__team_make_playoff_prob[team_id] = sum(
    #             self.__make_playoff[team_id]
    #         ) / len(
    #             self.__make_playoff[team_id]
    #         )  # average
    #     return self.__team_make_playoff_prob

    def team_make_finals_prob(self):
        for team_id in self.__win_games.keys():
            self.__team_make_finals_prob[team_id] = sum(
                self.__make_finals[team_id]
            ) / len(self.__make_finals[team_id])

        return self.__team_make_finals_prob

    def team_win_finals_prob(self):
        for team_id in self.__win_games.keys():
            self.__team_win_finals_prob[team_id] = sum(
                self.__win_finals[team_id]
            ) / len(self.__win_finals[team_id])

        return self.__team_win_finals_prob

    def team_average_win_games(self):
        for team_id in self.__win_games.keys():
            self.__team_average_win_games[team_id] = sum(
                self.__win_games[team_id]
            ) / len(self.__win_games[team_id])

        return self.__team_average_win_games

    def team_game_accuracy(self, team_id):
        # return percent of games correctly predicted for a team
        pass

    def season_game_accuracy(self):
        # return percent of games correctly predicted in season
        pass

    def team_game_results(self, team_id):
        # return dictionary of game_id: prob winning
        pass

    def season_game_results(self):
        # return dictionary of game_id: expected_winner_team_id
        pass

    def team_games_won_dist(self):
        # return distribution of # of games a team wins in a season
        return self.__win_games

    def team_score_difference_dist(self):
        return self.__score_difference

    def team_elo_dist(self):
        return self.__elo_distribution

    def team_regular_season_final_elo_dist(self):
        return self.__reg_season_final_elo_distribution

    def game_results(self, game_id):
        # return prob of each team winning game (can be more than two teams)
        return self.record[game_id]


if __name__ == "__main__":

    model_uri = "s3://sp-mlplatform-eks-us-east-1-prod/mlflow/artifacts/212/bcd0eab4567a4d70a4e848e0f65fd6a8/artifacts/"

    s3 = boto3.resource("s3")
    team_ratings_object = s3.Object(
        "sp-mlplatform-eks-us-east-1-prod",
        "mlflow/artifacts/212/bcd0eab4567a4d70a4e848e0f65fd6a8/artifacts/ratings.txt",
    )
    dist_std_object = s3.Object(
        "sp-mlplatform-eks-us-east-1-prod",
        "mlflow/artifacts/212/bcd0eab4567a4d70a4e848e0f65fd6a8/artifacts/dist_std.txt",
    )

    team_rating_init = json.loads(
        team_ratings_object.get()["Body"].read().decode("utf-8")
    )
    team_rating_init = update_current_season_ratings(
        dict((int(key), value) for (key, value) in team_rating_init.items())
    )
    dist_std = json.loads(dist_std_object.get()["Body"].read().decode("utf-8"))

    sport = Sport(team_rating_init=None, dist_std=dist_std, model_uri=model_uri)

    df_schedule = load_dataset("df_schedule").fillna(0)

    games = {}
    for game in df_schedule.iterrows():
        games[game[1].gamecode] = {
            "team_1": game[1].teamid,
            "team_2": game[1].opponentteamid,
            "team_1_score": game[1].points,
            "team_2_score": game[1].pointsconceded,
            "team_1_expected_win_prob": None,
            "team_1_expected_score_diff": None,
        }

    schedule = Schedule(games)

    df_team_info = load_dataset("df_team_info")
    team_info = (
        df_team_info[["teamid", "sub_div_id", "conf_id"]]
        .set_index("teamid")
        .to_dict("dict")
    )

    nicknames = dict(zip(df_team_info["teamid"], df_team_info["nickname"]))

    def map_nicknames(d, mapper):
        return {mapper[k]: round(v, 3) for k, v in d.items()}

    simulation = SeasonSimulation(sport=sport, schedule=schedule, team_info=team_info)

    start = time.time()
    simulation.run(1000)
    end = time.time()
    print(end - start)

    score_diffs = simulation.simulation_record.team_score_difference_dist()
    team_games_won_dist = simulation.simulation_record.team_games_won_dist()
    team_elo_dist = simulation.simulation_record.team_elo_dist()
    team_elo_dist_season = (
        simulation.simulation_record.team_regular_season_final_elo_dist()
    )

    file = open("score_diffs.pkl", "wb")
    pickle.dump(score_diffs, file)
    file.close()

    file = open("team_games_won_dist.pkl", "wb")
    pickle.dump(team_games_won_dist, file)
    file.close()

    file = open("team_elo_dist.pkl", "wb")
    pickle.dump(team_elo_dist, file)
    file.close()

    file = open("team_elo_dist_season.pkl", "wb")
    pickle.dump(team_elo_dist_season, file)
    file.close()

    team_average_win_games = map_nicknames(
        simulation.simulation_record.team_average_win_games(), nicknames
    )
    team_make_playoff_prob = map_nicknames(
        simulation.simulation_record.team_make_playoff_prob(), nicknames
    )
    team_make_finals_prob = map_nicknames(
        simulation.simulation_record.team_make_finals_prob(), nicknames
    )
    team_win_finals_prob = map_nicknames(
        simulation.simulation_record.team_win_finals_prob(), nicknames
    )
    print("average games won", team_average_win_games)
    print()
    print("make playoffs probabilities", team_make_playoff_prob)
    print()
    print("make finals probabilities", team_make_finals_prob)
    print()
    print("win finals probabilities", team_win_finals_prob)
    # model = mlflow.sklearn.load_model(model_uri)
    # print(model.predict(np.array([0]).reshape(-1, 1)))
    # print(model.predict(np.array([1]).reshape(-1, 1)) - model.predict(np.array([0]).reshape(-1, 1)))
