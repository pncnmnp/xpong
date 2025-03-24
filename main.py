import ast
from datetime import datetime
import random
import math
import os

import pandas as pd
from scipy.stats import truncnorm
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

env_file = find_dotenv(os.path.join(os.getcwd(), ".env"))
load_dotenv(env_file)


class GPTClient:
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def chat(self, messages, temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Failed to generate chat response: {e}")


class GPTPrompts:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.gpt_client = GPTClient(api_key)

    def generate_player_info(self):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise and structured data generator. "
                    "You always return clean, raw JSON. Do not include any explanations, "
                    "markdown formatting, or code block symbols like ```."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Generate exactly 64 rows of fictional person data. "
                    "Each row should be a JSON array (not an object) containing exactly 4 values: "
                    "[autoincremented_id, full_name, country, date_of_birth]. "
                    "The date of birth should be a string in 'YYYY-MM-DD' format, "
                    "and ages should be between 18 and 40 as of the year 2025. "
                    "Ensure names are culturally consistent with their country. "
                    "Not all countries need to be represented; it's okay if some repeat. "
                    "Return only a single raw JSON array of arrays. "
                    "Do not include any commentary, preamble, or markdown formatting like triple backticks."
                ),
            },
        ]
        chat_response = self.gpt_client.chat(messages)
        chat_response = ast.literal_eval(chat_response)
        player_info = {}
        for player in chat_response:
            player_id, full_name, country, dob = player
            player_info[player_id] = {
                "full_name": full_name,
                "country": country,
                "dob": dob,
            }
        return player_info


class GameStats:
    def __init__(self):
        # Historical game statistics are stored in a dataFrame
        # stores information on the player id, opponent id, match id,
        # date, tournament_id, result, and game statistics.
        # The game statistics include the points scored by the player,
        # the points allowed by the player, the fastest ball speed,
        # and the number of aces
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
                "fastest_ball_speed",
                "aces",
            ]
        )
        self.tournament_stats = pd.DataFrame(
            columns=["winner_id", "runner_up_id", "tournament_id", "match_id"]
        )
        self.player_elo = {}
        self.player_stats = pd.DataFrame(
            columns=[
                "player_id",
                "name",
                "country",
                "dob",
                "style",
                "player_elo",
                "total_games",
                "win_rate",
                "avg_points_scored",
                "avg_points_allowed",
                "avg_points_diff",
                "longest_winning_streak",
                "longest_losing_streak",
                "fastest_ball_speed",
                "avg_aces",
                "tournaments_won",
                "tournaments_runner_up",
            ]
        )

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

    def simulate_aces(self, num_points):
        threshold = num_points // 3
        if random.random() < 0.9:
            return random.randint(0, threshold)
        else:
            return math.floor(random.random() * num_points)

    def simulate_fastest_ball_speed(self):
        lower, upper = 130, 160
        mu = (lower + upper) / 2
        sigma = (upper - lower) / 6
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        ball_speed = truncnorm.rvs(a, b, loc=mu, scale=sigma)
        return round(ball_speed, 1)

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
                    "fastest_ball_speed": self.simulate_fastest_ball_speed(),
                    "aces": self.simulate_aces(winner_points),
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
                    "fastest_ball_speed": self.simulate_fastest_ball_speed(),
                    "aces": self.simulate_aces(loser_points),
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

    def show_top_bottom_elo_stats(self, sorted_players):
        player_elo_df = pd.DataFrame(
            sorted_players, columns=["Player ID", "ELO Rating"]
        )
        print(player_elo_df.head(3))
        print(player_elo_df.tail(3))
        print(
            "Delta is: ",
            player_elo_df["ELO Rating"].max() - player_elo_df["ELO Rating"].min(),
        )

    def player_statistics(self, player_ids, player_info):
        for player_id in player_ids:
            name = player_info[player_id]["full_name"]
            country = player_info[player_id]["country"]
            dob = player_info[player_id]["dob"]
            style = "Right Hand" if random.random() > 0.3 else "Left Hand"

            player_stats = self.game_stats[self.game_stats["player_id"] == player_id]
            total_games = len(player_stats)
            total_wins = len(player_stats[player_stats["result"] == "W"])
            win_rate = total_wins / total_games
            avg_points_scored = round(player_stats["points_scored"].mean(), 2)
            avg_points_allowed = round(player_stats["points_allowed"].mean(), 2)
            points_diff = round(avg_points_scored - avg_points_allowed, 2)
            fastest_ball_speed = player_stats["fastest_ball_speed"].max()
            avg_aces = round(player_stats["aces"].mean(), 2)
            tournaments_won = len(
                self.tournament_stats[self.tournament_stats["winner_id"] == player_id]
            )
            tournaments_runner_up = len(
                self.tournament_stats[
                    self.tournament_stats["runner_up_id"] == player_id
                ]
            )
            win_streak, lose_streak = 0, 0
            max_win_streak, max_lose_streak = 0, 0
            for result in player_stats["result"]:
                if result == "W":
                    win_streak += 1
                    lose_streak = 0
                    max_win_streak = max(max_win_streak, win_streak)
                else:
                    lose_streak += 1
                    win_streak = 0
                    max_lose_streak = max(max_lose_streak, lose_streak)
            self.player_stats.loc[len(self.player_stats)] = [
                player_id,
                name,
                country,
                dob,
                style,
                self.player_elo[player_id],
                total_games,
                win_rate,
                avg_points_scored,
                avg_points_allowed,
                points_diff,
                max_win_streak,
                max_lose_streak,
                fastest_ball_speed,
                avg_aces,
                tournaments_won,
                tournaments_runner_up,
            ]

    def head_to_head_statistics(self, player_id, opponent_id):
        player_games = self.game_stats[self.game_stats["player_id"] == player_id]
        head_to_head = player_games[player_games["opponent_id"] == opponent_id]
        total_games = len(head_to_head)
        total_wins = len(head_to_head[head_to_head["result"] == "W"])
        win_rate = total_wins / total_games
        avg_points_scored = round(head_to_head["points_scored"].mean(), 2)
        avg_points_allowed = round(head_to_head["points_allowed"].mean(), 2)
        return (
            player_id,
            opponent_id,
            total_games,
            win_rate,
            avg_points_scored,
            avg_points_allowed,
            head_to_head["result"].values,
        )


if __name__ == "__main__":
    simul = GameStats()
    gpt_prompts = GPTPrompts()
    num_players_per_tournament = 64

    player_ids = list(range(1, num_players_per_tournament + 1))
    player_rankings = list(range(1, num_players_per_tournament + 1))
    player_info = gpt_prompts.generate_player_info()
    random.shuffle(player_rankings)

    simul.assign_init_elo(player_ids, player_rankings)

    number_of_tournaments = 15
    tournament_year = 2025 - number_of_tournaments
    for tournament_id in range(0, number_of_tournaments):
        simul.simulate_tournament(player_ids, tournament_id, tournament_year)
        tournament_year += 1

    sorted_players = sorted(simul.player_elo.items(), key=lambda x: x[1], reverse=True)
    simul.player_statistics(player_ids, player_info)
    simul.player_stats.to_csv("player_stats.csv", index=False)
