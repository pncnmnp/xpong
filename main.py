import ast
import asyncio
from datetime import datetime
import random
import math
import os
import time
import threading
import logging

import pandas as pd
from scipy.stats import truncnorm
from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer
from dotenv import load_dotenv, find_dotenv
import eel

env_file = find_dotenv(os.path.join(os.getcwd(), ".env"))
load_dotenv(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        self.async_openai = AsyncOpenAI(api_key=api_key)

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

    def generate_opening_script(self, h2h):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an insightful and engaging sports commentator assistant. "
                    "You always return structured, clear, and realistic commentary in JSON format, "
                    "as an ordered list of turns. Each turn must be represented as an object with two keys: "
                    "'speaker' (alternating strictly between 'Tony McCrae' and 'Nina Novak') and 'text' (the commentary). "
                    "Allow each speaker to speak multiple consecutive sentences in a single turn before switching to the other. "
                    "Never use markdown formatting or additional explanations in the JSON response."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the head-to-head statistics for today's Pong game:\n"
                    f"Player ID: {h2h['player_id']} vs Opponent ID: {h2h['opponent_id']}\n"
                    f"Player Names: {h2h['player_name']} vs {h2h['opponent_name']}\n"
                    f"Player Ranks: World Number {h2h['player_rank'] + 1} vs World Number {h2h['opponent_rank'] + 1}\n"
                    f"Countries: {h2h['player_country']} vs {h2h['opponent_country']}\n"
                    f"Dates of Birth: {h2h['player_dob']} vs {h2h['opponent_dob']}\n"
                    f"Playing Styles: {h2h['player_style']} vs {h2h['opponent_style']}\n"
                    f"Total Games faced against each other: {h2h['total_games']}\n"
                    f"Win Rate of Player {h2h['player_name']}: {h2h['win_rate']*100}%\n"
                    f"Win Rate of Player {h2h['opponent_name']}: {100 - h2h['win_rate']*100}%\n"
                    f"Average Points Scored by {h2h['player_name']} against {h2h['opponent_name']}: {h2h['avg_points_scored']}\n"
                    f"Average Points Allowed by {h2h['player_name']} against {h2h['opponent_name']}: {h2h['avg_points_allowed']}\n"
                    f"Recent Match Outcomes (from {h2h['player_name']}'s perspective - against {h2h['opponent_name']}): "
                    f"{', '.join(h2h['head_to_head'])}\n\n"
                    f"Generate a commentary opening script structured exactly as an ordered list in JSON format, with each item containing:\n"
                    f"- 'speaker': alternating strictly between 'Tony McCrae' and 'Nina Novak'\n"
                    f"- 'text': multiple consecutive sentences of commentary under one speaker before switching.\n\n"
                    f"Include explicitly:\n"
                    f"- Extend a warm, lively welcome to our global audience, setting an enthusiastic tone right from the start.\n"
                    f"- Briefly introduce the two commentators.\n"
                    f"- Spotlight Madrid, Spain as our grand host, highlighting its global sporting significance and cultural vibrancy.\n"
                    f"- Reveal and discuss both players' world rankings and how those standings elevate the competitive stakes.\n"
                    f"- Provide an engaging, data-driven breakdown of their head-to-head match history and notable statistics.\n"
                    f"- Capture the excitement of each player's entrance, describing the crowd's anticipation and overall atmosphere.\n"
                    f"- Begin the match with an official opening line that includes the phrase: 'Paddles out and away we pong!'\n\n"
                    f"No additional explanations or markdown should appear outside this JSON-formatted list."
                ),
            },
        ]
        chat_response = self.gpt_client.chat(messages)
        chat_response = ast.literal_eval(chat_response)
        return chat_response

    async def speak(self, input, voice):
        instructions = (
            ""
            "Personality/affect: A high-energy sports commentator guiding users through administrative tasks.\n\n"
            "Voice: Dynamic, passionate, and engaging, with an exhilarating and motivating quality.\n\n"
            "Tone: Excited and enthusiastic, turning routine tasks into thrilling and high-stakes events.\n\n"
            "Dialect: Informal, fast-paced, and energetic, using sports terminology, vibrant metaphors, and enthusiastic phrasing.\n\n"
            "Pronunciation: Clear, powerful, and emphatic, emphasizing key actions and successes as if celebrating a game-winning moment.\n\n"
            "Features: Incorporates vivid sports analogies, enthusiastic exclamations, "
            "and rapid-fire commentary style to build excitement and maintain a lively pace throughout interactions.\n"
            ""
        )

        async with self.async_openai.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=input,
            speed=1.4,
            instructions=instructions,
            response_format="pcm",
        ) as response:
            await LocalAudioPlayer().play(response)

    async def speak_opening_script(self, head_to_head_stats):
        opening_script = self.generate_opening_script(head_to_head_stats)
        print(opening_script)

        for line in opening_script:
            voice = "ballad" if line["speaker"] == "Tony McCrae" else "coral"
            await self.speak(line["text"], voice)


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
            win_rate = total_wins / total_games if total_games > 0 else 0
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
        player_name = self.player_stats[self.player_stats["player_id"] == player_id][
            "name"
        ].values[0]
        opponent_name = self.player_stats[
            self.player_stats["player_id"] == opponent_id
        ]["name"].values[0]
        player_country, player_dob, player_style = self.player_stats.loc[
            self.player_stats["player_id"] == player_id, ["country", "dob", "style"]
        ].iloc[0]
        opponent_country, opponent_dob, opponent_style = self.player_stats.loc[
            self.player_stats["player_id"] == opponent_id, ["country", "dob", "style"]
        ].iloc[0]

        head_to_head = player_games[player_games["opponent_id"] == opponent_id]
        total_games = len(head_to_head)
        total_wins = len(head_to_head[head_to_head["result"] == "W"])
        win_rate = total_wins / total_games if total_games > 0 else 0
        avg_points_scored = round(head_to_head["points_scored"].mean(), 2)
        avg_points_allowed = round(head_to_head["points_allowed"].mean(), 2)
        # rank of player_id and opponent_id using player_elo
        player_rank = sorted(
            self.player_elo.items(), key=lambda x: x[1], reverse=True
        ).index((player_id, self.player_elo[player_id]))
        opponent_rank = sorted(
            self.player_elo.items(), key=lambda x: x[1], reverse=True
        ).index((opponent_id, self.player_elo[opponent_id]))

        return {
            "player_id": player_id,
            "opponent_id": opponent_id,
            "player_name": player_name,
            "opponent_name": opponent_name,
            "player_country": player_country,
            "opponent_country": opponent_country,
            "player_dob": player_dob,
            "opponent_dob": opponent_dob,
            "player_style": player_style,
            "opponent_style": opponent_style,
            "total_games": total_games,
            "win_rate": win_rate,
            "avg_points_scored": avg_points_scored,
            "avg_points_allowed": avg_points_allowed,
            "head_to_head": head_to_head["result"].values,
            "player_rank": player_rank,
            "opponent_rank": opponent_rank,
        }


class PongGame:
    def __init__(self):
        self.width = 800
        self.height = 600

        self.ball = {
            "x": self.width / 2,
            # this is the center of the ball
            "y": self.height / 2,
            # speed and dir in x dir
            "vx": 5,
            # speed and dir in y dir
            "vy": 3,
            "radius": 10,
        }

        self.left_paddle = {
            "x": 10,
            # this is the top of the paddle
            "y": self.height / 2 - 50,
            "width": 10,
            "height": 100,
            "speed": 6,
        }
        self.right_paddle = {
            "x": self.width - 20,
            "y": self.height / 2 - 50,
            "width": 10,
            "height": 100,
            "speed": 6,
        }

        self.left_score = 0
        self.right_score = 0

        # Approximately 60 FPS
        self.game_speed = 0.016

        self.ball_in_play = True

        eel.init("./web")

    def init_ball(self, direction):
        # If left side just lost, we serve on left side
        if direction == 1:
            # Position ball next to the left paddle
            self.ball["x"] = (
                self.left_paddle["x"] + self.left_paddle["width"] + self.ball["radius"]
            )
            # Place the ball vertically at a random part of the left paddle
            self.ball["y"] = self.left_paddle["y"] + random.randrange(
                math.ceil(self.left_paddle["height"] * 0.25),
                math.ceil(self.left_paddle["height"] * 0.75),
            )
        # If right side just lost, we serve on right side
        elif direction == -1:
            self.ball["x"] = self.right_paddle["x"] - self.ball["radius"]
            self.ball["y"] = self.right_paddle["y"] + random.randrange(
                math.ceil(self.right_paddle["height"] * 0.25),
                math.ceil(self.right_paddle["height"] * 0.75),
            )

        self.shot_velocity_x(direction)
        self.play_paddle_shot_sound()
        self.shot_velocity_y()

    async def reset_ball(self, direction=1):
        await asyncio.sleep(5)
        self.init_ball(direction)
        self.ball_in_play = True

    def shot_velocity_x(self, direction):
        self.ball["vx"] = direction * random.choice([i for i in range(5, 15)])

    def shot_velocity_y(self):
        self.ball["vy"] = random.choice(
            [i for i in range(-4, -10, -1)] + [i for i in range(4, 10)]
        )

    def play_paddle_shot_sound(self):
        # variable volume between 0.1 and 1, based on the new shot speed
        # when the shot is softest, the velocity is 5, and volume is 0.1
        # when the shot is hardest, the velocity is 15, and volume is 1
        volume = ((abs(self.ball["vx"]) - 5) / 10) * 0.9 + 0.1
        eel.play_sound(volume)

    def predict_ball_y(self, paddle_x):
        net_line = self.width / 2
        # if paddle is on the left and ball is moving to the right,
        # stand at the center mark, and vice versa
        if (paddle_x < net_line and self.ball["vx"] >= 0) or (
            paddle_x > net_line and self.ball["vx"] <= 0
        ):
            return self.height / 2

        # reaction time
        # if the paddle is on left, and ball is moving to the left,
        # we start the prediction after ball crosses the the net line
        # and vice versa
        # tuning this, changes how strong the AI can predict the ball
        if (paddle_x < net_line and self.ball["x"] > net_line) or (
            paddle_x > net_line and self.ball["x"] < net_line
        ):
            return self.height / 2

        # how long it will take for the ball to reach the paddle
        time_to_reach = abs((paddle_x - self.ball["x"]) / self.ball["vx"])
        # estimate where the paddle should be,
        # by adding up the ball's current y position
        # with the y dist it will take to reach the paddle
        y = self.ball["y"] + self.ball["vy"] * time_to_reach

        # if the ball is out of bounds, we need to reflect it
        while y < 0 or y > self.height:
            if y < 0:
                y = -y
            elif y > self.height:
                y = 2 * self.height - y
        return y

    def move_paddle_toward(self, paddle, target_y):
        center = paddle["y"] + paddle["height"] / 2
        # paddle speed determines the accuracy of the prediction
        # and the speed of the paddle AI
        if abs(center - target_y) < paddle["speed"]:
            return
        if center < target_y:
            # if the paddle is not overshooting across the bottom of the screen,
            # move it closer to the predicted y position
            if paddle["y"] + paddle["height"] < self.height:
                paddle["y"] += paddle["speed"]
        elif center > target_y:
            # if the paddle is not overshooting across the top of the screen,
            # move it closer to the predicted y position
            if paddle["y"] > 0:
                paddle["y"] -= paddle["speed"]

    def update_game(self):
        if not self.ball_in_play:
            return

        self.ball["x"] += self.ball["vx"]
        self.ball["y"] += self.ball["vy"]

        # collision with top and bottom walls
        if (
            self.ball["y"] - self.ball["radius"] <= 0
            or self.ball["y"] + self.ball["radius"] >= self.height
        ):
            self.ball["vy"] *= -1

        # AI paddle movement for left paddle
        # There is no fancy AI here.
        # We simply check where the ball is going to be in the y direction,
        # based on the ball's velocity,
        # and move the paddle to the predicted y position (within an approx range)
        left_target = (
            self.predict_ball_y(self.left_paddle["x"])
            if self.ball["vx"] < 0
            else self.height / 2
        )
        self.move_paddle_toward(self.left_paddle, left_target)

        # AI paddle movement for right paddle
        # Similar to the left paddle
        right_target = (
            self.predict_ball_y(self.right_paddle["x"])
            if self.ball["vx"] > 0
            else self.height / 2
        )
        self.move_paddle_toward(self.right_paddle, right_target)

        # if the ball hits the left paddle
        # we need the left paddle to make a new shot with a new velocity
        if (
            self.ball["x"] - self.ball["radius"]
            <= self.left_paddle["x"] + self.left_paddle["width"]
            and self.left_paddle["y"]
            <= self.ball["y"]
            <= self.left_paddle["y"] + self.left_paddle["height"]
        ):
            self.shot_velocity_x(direction=1)
            self.play_paddle_shot_sound()

        # ball hitting the right paddle
        # similar to the left paddle but in the opposite direction
        if (
            self.ball["x"] + self.ball["radius"] >= self.right_paddle["x"]
            and self.right_paddle["y"]
            <= self.ball["y"]
            <= self.right_paddle["y"] + self.right_paddle["height"]
        ):
            self.shot_velocity_x(direction=-1)
            self.play_paddle_shot_sound()

        # ball's out of bounds
        if self.ball["x"] < 0:
            self.ball_in_play = False
            self.right_score += 1
            asyncio.create_task(self.reset_ball(direction=1))
        elif self.ball["x"] > self.width:
            self.ball_in_play = False
            self.left_score += 1
            asyncio.create_task(self.reset_ball(direction=-1))

    async def game_loop(self):
        while True:
            self.update_game()
            # invoke index.html's update_pong function
            eel.update_pong(
                self.ball,
                self.left_paddle,
                self.right_paddle,
                self.left_score,
                self.right_score,
                self.width,
                self.height,
            )
            # wait for the next frame
            await asyncio.sleep(self.game_speed)

    def start_game(self):
        def start_eel():
            eel.start("index.html", size=(self.width, self.height), block=True)

        eel_thread = threading.Thread(target=start_eel)
        eel_thread.daemon = True
        eel_thread.start()

        time.sleep(1)
        self.init_ball(direction=random.choice([-1, 1]))
        asyncio.run(self.game_loop())


if __name__ == "__main__":
    logger.info("Generating tournament data...")
    simul = GameStats()
    gpt_prompts = GPTPrompts()
    num_players_per_tournament = 64
    logger.info(f"{num_players_per_tournament} players in the tournament...")

    player_ids = list(range(1, num_players_per_tournament + 1))
    player_rankings = list(range(1, num_players_per_tournament + 1))

    logger.info("Generating player information from GPT...")
    player_info = gpt_prompts.generate_player_info()

    logger.info("Shuffle player rankings and assign ELO ratings...")
    random.shuffle(player_rankings)
    simul.assign_init_elo(player_ids, player_rankings)

    number_of_tournaments = 15
    tournament_year = 2025 - number_of_tournaments
    logger.info(
        f"Simulating {number_of_tournaments} tournaments starting with year {tournament_year}..."
    )
    for tournament_id in range(0, number_of_tournaments):
        simul.simulate_tournament(player_ids, tournament_id, tournament_year)
        tournament_year += 1

    logger.info(f"Generate player statistics based on tournament data...")
    sorted_players = sorted(simul.player_elo.items(), key=lambda x: x[1], reverse=True)
    simul.player_statistics(player_ids, player_info)
    simul.player_stats.to_csv("./player_stats.csv", index=False)

    logger.info(f"Generate head-to-head statistics for the top 2 players...")
    head_to_head_stats = simul.head_to_head_statistics(
        sorted_players[0][0], sorted_players[1][0]
    )

    logger.info(f"Start the opening commentary script...")
    asyncio.run(gpt_prompts.speak_opening_script(head_to_head_stats))

    logger.info("Starting Pong Game...")
    PongGame().start_game()
