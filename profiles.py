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
        self.game_stats = []
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

    def simulate_tournament(self, player_ids, tournament_id):
        tournament_pairings = [
            player for pair in self.tournament_pairing(player_ids) for player in pair
        ]
        round_players = tournament_pairings[:]
        match_id_counter = 1

        while len(round_players) > 1:
            next_round_players = []

            for i in range(0, len(round_players), 2):
                player1, player2 = round_players[i], round_players[i + 1]
                p1_rank, p2_rank = self.player_elo[player1], self.player_elo[player2]
                rank_diff = p2_rank - p1_rank

                p1_win_probability = 1 / (1 + 10 ** (rank_diff / 400))
                p2_win_probability = 1 - p1_win_probability

                if random.random() < p1_win_probability:
                    winner, loser = player1, player2
                    k_winner = max(1, 32 - 0.04 * (p1_rank - 2000))
                    k_loser = max(1, 32 - 0.04 * (p2_rank - 2000))
                    self.player_elo[winner] += k_winner * (1 - p1_win_probability)
                    self.player_elo[loser] += k_loser * (1 - (1 - p1_win_probability))
                else:
                    winner, loser = player2, player1
                    k_winner = max(1, 32 - 0.04 * (p1_rank - 2000))
                    k_loser = max(1, 32 - 0.04 * (p2_rank - 2000))
                    self.player_elo[winner] += k_winner * (1 - p2_win_probability)
                    self.player_elo[loser] += k_loser * (1 - (1 - p2_win_probability))

                winner_points = 21
                loser_points = random.randint(0, winner_points - 1)

                fastest_speed = random.randint(100, 250)
                avg_speed = random.randint(80, fastest_speed)
                aces = random.randint(0, 3)
                rallies = random.randint(10, 30)
                longest_rally = random.randint(5, rallies)
                avg_rally = round(rallies / random.uniform(1.0, 3.0), 1)

                self.game_stats.append(
                    {
                        "player_id": winner,
                        "opponent_id": loser,
                        "match_id": match_id_counter,
                        "date": datetime.now(),
                        "tournament_id": tournament_id,
                        "result": "W",
                        "points_scored": winner_points,
                        "points_allowed": loser_points,
                        "fastest_ball_speed": fastest_speed,
                        "average_ball_speed": avg_speed,
                        "aces": aces,
                        "rallies": rallies,
                        "longest_rally": longest_rally,
                        "average_rally": avg_rally,
                    },
                )

                self.game_stats.append(
                    {
                        "player_id": loser,
                        "opponent_id": winner,
                        "match_id": match_id_counter,
                        "date": datetime.now(),
                        "tournament_id": tournament_id,
                        "result": "L",
                        "points_scored": loser_points,
                        "points_allowed": winner_points,
                        "fastest_ball_speed": fastest_speed,
                        "average_ball_speed": avg_speed,
                        "aces": random.randint(0, aces),
                        "rallies": rallies,
                        "longest_rally": random.randint(0, longest_rally),
                        "average_rally": avg_rally,
                    },
                )

                match_id_counter += 1
                next_round_players.append(winner)

            round_players = next_round_players

        print("Tournament complete! Winner is:", round_players[0])

    def assign_init_elo(self, player_ids, player_rankings):
        upper_bound = 2500
        lower_bound = 2100
        delta = (upper_bound - lower_bound) / len(player_ids)
        for player_id in player_ids:
            self.player_elo[player_id] = upper_bound - (
                delta * (player_rankings[player_id - 1] - 1)
            )


if __name__ == "__main__":
    game_stats_obj = GameStats()

    player_ids = list(range(1, 65))
    player_rankings = list(range(1, 65))
    random.shuffle(player_rankings)

    game_stats_obj.assign_init_elo(player_ids, player_rankings)

    for tournament_id in range(1, 100):
        game_stats_obj.simulate_tournament(player_ids, tournament_id)

    sorted_players = sorted(
        game_stats_obj.player_elo.items(), key=lambda x: x[1], reverse=True
    )
    print("Top 10 players after tournament", tournament_id)
    print(sorted_players[:10])
