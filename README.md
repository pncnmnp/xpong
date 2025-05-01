# xPong

![ChatGPT Image Apr 28, 2025, 12_15_40 AM](https://github.com/user-attachments/assets/d3035aab-ad72-45cd-b84f-3b0a6d0a9541)

As the name suggests, **xPong** is all about Pong — but with a twist: there’s LLM-based commentary generation integrated into the game.    

I’ve wanted to do this for the longest time — in fact, back in my undergrad days, 
I used to carry around a GitHub Gist to store such ideas, and an entry for this one dates back to Feb 16 2020. 
That’s five years! 
Anyway, for the longest time we never had the right technology to pull this off in a cost-effective, near-realtime manner.

However, late last month, with <a href="https://news.ycombinator.com/item?id=43426022">OpenAI’s `gpt-40-mini-tts`</a>, we finally have the resources to do it! 
I envision a future where `gpt-40-mini-tts`-like software is integrated on the edge — that is, edge LLMs. 
When that happens, next-generation gaming consoles will leverage such technology, opening up a whole universe of possibilities in gaming, especially in sports-simulation titles.

Hope you have a blast tinkering with this game and its source code!

## Demo

## Installation

Before installing the dependencies, I encourage you to [create and activate a virtual environment](https://docs.python.org/3/library/venv.html) - for example, run `python3 -m venv .venv` and then `source .venv/bin/activate` on macos/linux.

To install the Python dependencies, use:
```bash
pip3 install -r requirements.txt
```

Next, create an API key in OpenAI’s dashboard.
For this, follow the steps in OpenAI’s [Create and export an API key](https://platform.openai.com/docs/libraries#create-and-export-an-api-key) docs.
Once you have the key, store it in a `.env` file at the root of this repository:
```env
OPENAI_API_KEY=your-api-key
```

Because this code leverages an Electron-like library [Eel](https://github.com/python-eel/Eel), you will also need a [Chromium-based browser](https://www.chromium.org/getting-involved/download-chromium/).

When everything is set up, run:
```bash
python3 main.py
```

Add `--cache` to cache LLM-generated player metadata for later runs, and `--no-opening` to skip the opening commentary.

## What’s in it?

It’s packed with ideas!

* There’s a tournament simulator that simulates 15 years' worth of tournaments. Each year includes four majors — sixty events in all. Players and matches are simulated using **Elo**.
* In the **16th year**, when the main match starts, the world’s top two players face off in the World Championship Final.
* The commentary logic has **three layers** - opening commentary with a scorecard, in-game ball-by-ball commentary, and closing commentary.
* Two commentators take turns talking to each other and the audience.
* Commentary is interrupted when an important in-game event occurs, then resumes from where it left off afterward.
* In-game commentary is driven by an **event-based pipeline**: actions are logged as events, periodically parsed into metrics,
  those metrics are ranked by priority to decide which deserve mention next, and the resulting text is fed through TTS.
* With 15 years of historical data, the game uses **nearest-neighbor search** to find similar past games and share insights.
* It even goofs around - comparing the current playing style to past legends and sprinkling in a few locker-room whispers.

## License and acknowledgements

This code is licensed under **the MIT License**.

The paddle-hitting sound was generated using SoundReality's ["Tennis Ball Hit"](https://pixabay.com/sound-effects/tennis-ball-hit-151257/) sound.  
It is licensed under Pixabay's ["Content License"](https://pixabay.com/service/license-summary/).

The filler sounds were generated with <https://www.openai.fm/> and abide by OpenAI's [terms and policies](https://openai.com/policies/).
