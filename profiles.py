from datetime import datetime
import random
import pandas as pd


class GameStats:
    def __init__(self):
        # Historical game statistics are stored in a dataFrame
        # stores information on the player id, opponent id, match id,
        # date, tournament_id, result, and game statistics.
        # The game statistics include the points scored by the player,
        # the points allowed by the player, the fastest ball speed,
        # the average ball speed, the number of aces, the number of rallies,
        # longest rally length, and the average rally length.
        self.game_stats = pd.DataFrame(
            columns=[
                "player_id",
                "opponent_id",
                "match_id",
                "date",
                "tournament_id",
                "result",
                "points_scored",
                "points_allowed",
            ]
        )
        self.tournament_stats = pd.DataFrame(
            columns=["winner_id", "runner_up_id", "tournament_id", "match_id"]
        )
        self.player_elo = {}

    def tournament_pairing(self, player_ids):
        if len(player_ids) % 2 != 0:
            raise ValueError("Number of players in a tournament must be even.")
        players_by_rankings = sorted(
            player_ids, key=lambda x: self.player_elo[x], reverse=True
        )
        half_len = len(players_by_rankings) // 2
        elite_tier = players_by_rankings[:half_len]
        challenger_tier = players_by_rankings[half_len:]
        random.shuffle(challenger_tier)
        pairings = list(zip(elite_tier, challenger_tier))
        return pairings

    def simulate_tournament(self, player_ids, tournament_id, tournament_year):
        tournament_pairings = [
            player for pair in self.tournament_pairing(player_ids) for player in pair
        ]
        round_players = tournament_pairings[:]
        match_id_counter = 1

        # Tournament start date
        match_date = datetime(tournament_year, 7, 1)

        while len(round_players) > 1:
            next_round_players = []
            match_date += pd.DateOffset(days=1)

            for i in range(0, len(round_players), 2):
                player1, player2 = round_players[i], round_players[i + 1]
                p1_rank, p2_rank = self.player_elo[player1], self.player_elo[player2]
                rank_diff = p2_rank - p1_rank

                if match_id_counter % 8 == 0:
                    match_date += pd.DateOffset(days=1)

                p1_win_probability = 1 / (1 + 10 ** (rank_diff / 400))
                p2_win_probability = 1 - p1_win_probability

                if random.random() < p1_win_probability:
                    winner, loser = player1, player2
                    k_winner = max(1, 32 - 0.04 * (p1_rank - 2000))
                    k_loser = max(32, 32 - 0.04 * (p2_rank - 2000))
                    self.player_elo[winner] += k_winner * (1 - p1_win_probability)
                    self.player_elo[loser] += k_loser * (0 - p2_win_probability)
                else:
                    winner, loser = player2, player1
                    k_winner = max(1, 32 - 0.04 * (p2_rank - 2000))
                    k_loser = max(32, 32 - 0.04 * (p1_rank - 2000))
                    self.player_elo[winner] += k_winner * (1 - p2_win_probability)
                    self.player_elo[loser] += k_loser * (0 - p1_win_probability)

                winner_points = 21
                loser_points = random.randint(0, winner_points - 1)

                self.game_stats.loc[len(self.game_stats)] = {
                    "player_id": winner,
                    "opponent_id": loser,
                    "match_id": match_id_counter,
                    "date": match_date,
                    "tournament_id": tournament_id,
                    "result": "W",
                    "points_scored": winner_points,
                    "points_allowed": loser_points,
                }

                self.game_stats.loc[len(self.game_stats)] = {
                    "player_id": loser,
                    "opponent_id": winner,
                    "match_id": match_id_counter,
                    "date": match_date,
                    "tournament_id": tournament_id,
                    "result": "L",
                    "points_scored": loser_points,
                    "points_allowed": winner_points,
                }

                match_id_counter += 1
                next_round_players.append(winner)

            if len(next_round_players) == 2:
                self.tournament_stats.loc[len(self.tournament_stats)] = {
                    "winner_id": next_round_players[0],
                    "runner_up_id": next_round_players[1],
                    "tournament_id": tournament_id,
                    "match_id": match_id_counter - 1,
                }
            round_players = next_round_players

    def assign_init_elo(self, player_ids, player_rankings):
        upper_bound = 2700
        lower_bound = 2400
        delta = (upper_bound - lower_bound) / len(player_ids)
        for player_id in player_ids:
            self.player_elo[player_id] = upper_bound - (
                delta * (player_rankings[player_id - 1] - 1)
            )

    def show_top_bottom_elo_stats(self):
        player_elo_df = pd.DataFrame(
            sorted_players, columns=["Player ID", "ELO Rating"]
        )
        print(player_elo_df.head(5))
        print(player_elo_df.tail(5))
        print(
            "Delta is: ",
            player_elo_df["ELO Rating"].max() - player_elo_df["ELO Rating"].min(),
        )


if __name__ == "__main__":
    simul = GameStats()

    player_ids = list(range(1, 65))
    player_rankings = list(range(1, 65))
    random.shuffle(player_rankings)

    simul.assign_init_elo(player_ids, player_rankings)

    number_of_tournaments = 15
    tournament_year = 2025 - number_of_tournaments
    for tournament_id in range(0, number_of_tournaments):
        simul.simulate_tournament(player_ids, tournament_id, tournament_year)
        tournament_year += 1

    sorted_players = sorted(simul.player_elo.items(), key=lambda x: x[1], reverse=True)
    print(simul.tournament_stats.head(15))
    simul.game_stats.to_csv("game_stats.csv", index=False)
