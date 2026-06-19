# Nova Career Coach

Nova Career Coach is an AI-powered conversational coaching application for the **DRIVEN** program—a six-week curriculum that helps adults with mental health challenges build career skills, resilience, and job-search confidence. Users work through guided weekly sessions with **NOVA**, an OpenAI-backed coach that asks structured questions, validates responses, and adapts follow-ups based on what the user shares.

## Features

- **Six-week curriculum** with sequential unlocking (complete Week *N* to access Week *N+1*)
- **Structured coaching flows** with scenario-based prompts and validation per question
- **Shared progress tracking** across all weeks via `progress_tracker.py` and `user_progress.json`
- **Single-page web UI** (`index.html`) with week selection via URL parameter
- **Unified dev server** (`app.py`) that serves the frontend and proxies each week's Flask API under one origin

## Program Weeks

| Week | Module | Backend | Status |
|------|--------|---------|--------|
| 1 | Thinking Flexibly and Goal Setting | `week1_main.py` | Implemented |
| 2 | Building Resilience | `week2_main.py` | Implemented |
| 3 | Career Exploration | `week3_main.py` | Implemented |
| 4 | Interview Immersion | `week4_main.py` | Implemented |
| 5 | Storytelling for Impact | `week5_main.py` | Implemented |
| 6 | Launch & Celebrate | `week6_main.py` | Not yet created |

## Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

1. **Clone the repository**

   ```bash
   git clone git@github.com:jsphzhao/careercoachai.git
   cd careercoachai
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   # venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install flask flask-cors openai python-dotenv
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   FLASK_SECRET_KEY=your_secret_key_here
   PORT=5007
   ```

   `OPENAI_API_KEY` is required. The other variables are optional.

## Running the App

### Recommended: unified server

Start the frontend and all mounted week backends together:

```bash
python app.py
```

Then open a week in your browser:

- Week 1: [http://localhost:5007/?week=1](http://localhost:5007/?week=1)
- Week 2: [http://localhost:5007/?week=2](http://localhost:5007/?week=2)
- Week 3: [http://localhost:5007/?week=3](http://localhost:5007/?week=3)
- Week 4: [http://localhost:5007/?week=4](http://localhost:5007/?week=4)
- Week 5: [http://localhost:5007/?week=5](http://localhost:5007/?week=5)

The default port is **5007** (override with the `PORT` environment variable).

### Debug mode: run weeks individually

Each week can also run as its own Flask server on ports 5001–5005:

```bash
python week1_main.py   # http://localhost:5001
python week2_main.py   # http://localhost:5002
python week3_main.py   # http://localhost:5003
python week4_main.py   # http://localhost:5004
python week5_main.py   # http://localhost:5005
```

Use this mode when developing or debugging a single week in isolation.

## Architecture

```
Browser (index.html)
       │
       ▼
   app.py  ──►  /week1  →  week1_main.py
   (port 5007)   /week2  →  week2_main.py
                 /week3  →  week3_main.py
                 /week4  →  week4_main.py
                 /week5  →  week5_main.py
                       │
                       ▼
              progress_tracker.py
                       │
                       ▼
              user_progress.json
```

- **`app.py`** — Serves static assets and mounts each week's Flask app under `/weekN` using Werkzeug's `DispatcherMiddleware`.
- **`weekN_main.py`** — Self-contained backend for a single week: conversation state, OpenAI calls, question flow, and progress API endpoints.
- **`progress_tracker.py`** — Central module for reading/writing user progress and enforcing sequential week unlocking.
- **`index.html`** — Frontend chat UI; routes API calls to `/weekN/api/...` based on the `?week=` URL parameter.

## Project Structure

```
NovaCareerCoach/
├── app.py                  # Unified development server
├── index.html              # Frontend UI
├── progress_tracker.py     # Shared progress tracking
├── user_progress.json      # Persisted user progress (generated at runtime)
├── week1_main.py           # Week 1 backend
├── week2_main.py           # Week 2 backend
├── week3_main.py           # Week 3 backend
├── week4_main.py           # Week 4 backend
├── week5_main.py           # Week 5 backend
├── LIGHTRAG_FOLDER/        # LightRAG experiments (separate from main app)
├── WEEK_INTEGRATION_README.md
└── PORT_CONFIGURATION.md
```

## Additional Documentation

- [WEEK_INTEGRATION_README.md](WEEK_INTEGRATION_README.md) — How week-by-week integration, progress storage, and API endpoints work
- [PORT_CONFIGURATION.md](PORT_CONFIGURATION.md) — Port assignments and debug-mode server setup

## Notes

- Progress is tied to Flask session IDs; use the same browser session to preserve progress across visits.
- Do not commit `.env` or `user_progress.json` if they contain secrets or user data.
- The `LIGHTRAG_FOLDER/` directory contains a vendored LightRAG setup for retrieval-augmented generation experiments and is not required to run the main coaching app.
