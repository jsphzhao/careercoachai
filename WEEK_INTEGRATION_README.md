# Week-by-Week Integration Guide

This document explains how the week-by-week integration works and what was added to connect `week1_main.py` (Week 1) and `week2_main.py` (currently serving Week 2 via WEEK2_TEMP) with shared progress tracking.

## Overview

The system now supports:
- **Shared progress tracking** across all weeks using a JSON file
- **Week unlocking** - users must complete previous weeks before accessing new ones
- **Dynamic API routing** - the frontend automatically connects to the correct week's server
- **Week completion detection** - automatically prompts users to move to the next week

## Files Added/Modified

### New Files

1. **`progress_tracker.py`** - Centralized progress tracking module
   - Manages user progress across all weeks
   - Stores progress in `user_progress.json`
   - Provides functions to check week unlock status
   - Tracks question and week completion

### Modified Files

1. **`week2_main.py`** - Week 2 backend (temporary via WEEK2_TEMP)
   - Added `progress_tracker` import
   - Added progress tracking API endpoints
   - Saves progress when questions/weeks are completed
   - Checks if week is unlocked before allowing access

2. **`week1_main.py`** - Week 1 backend
   - Added `progress_tracker` import
   - Added progress tracking API endpoints
   - Saves progress when questions/weeks are completed
   - Week 1 is always unlocked (first week)

3. **`index.html`** - Frontend
   - Added dynamic week selection via URL parameter (`?week=1` or `?week=3`)
   - Automatically routes API calls to the correct port based on week
   - Loads and displays progress status
   - Shows week completion message and prompts to continue

## How It Works

### 1. Week Configuration

Weeks are configured in `progress_tracker.py`:
```python
WEEK_CONFIG = {
    1: {
        "port": 5001,
        "file": "week1_main.py",
        "name": "Week 1",
        "title": "Thinking Flexibly and Goal Setting"
    },
    2: {
        "port": 5002,
        "file": "week2_main.py",  # WEEK2_TEMP
        "name": "Week 2",
        "title": "Building Resilience"
    },
    3: {
        "port": 5003,
        "file": "week3_main.py",
        "name": "Week 3",
        "title": "Career Exploration"
    },
    4: {
        "port": 5004,
        "file": "week4_main.py",
        "name": "Week 4",
        "title": "Interview Immersion"
    },
    5: {
        "port": 5005,
        "file": "week5_main.py",
        "name": "Week 5",
        "title": "Storytelling for Impact"
    },
    6: {
        "port": 5006,
        "file": "week6_main.py",
        "name": "Week 6",
        "title": "Launch & Celebrate"
    }
}
```

### 2. Progress Storage

**One JSON file for all weeks:** Progress is stored in a **single** `user_progress.json` file. Each week gets its own dictionary entry within each user's session data.

**Structure:**
- **One file:** `user_progress.json` (contains all users and all weeks)
- **One entry per user session:** Keyed by `session_id`
- **One dictionary per week:** Each week (1-6) has its own entry in the `"weeks"` object

**Example structure:**
```json
{
  "session_id_123": {
    "name": "User Name",
    "current_week": 3,
    "created_at": "2024-01-01T00:00:00",
    "last_updated": "2024-01-20T14:45:00",
    "weeks": {
      "1": {
        "completed": true,
        "questions_completed": {"1": true, "2": true, "3": true, ...},
        "started_at": "2024-01-01T00:00:00",
        "completed_at": "2024-01-02T12:00:00"
      },
      "2": {
        "completed": true,
        "questions_completed": {"1": true, "2": true, ...},
        "started_at": "2024-01-03T09:00:00",
        "completed_at": "2024-01-04T15:30:00"
      },
      "3": {
        "completed": false,
        "questions_completed": {"1": true, "2": false, ...},
        "started_at": "2024-01-05T10:00:00",
        "completed_at": null
      },
      "4": {
        "completed": false,
        "questions_completed": {},
        "started_at": null,
        "completed_at": null
      },
      "5": {
        "completed": false,
        "questions_completed": {},
        "started_at": null,
        "completed_at": null
      },
      "6": {
        "completed": false,
        "questions_completed": {},
        "started_at": null,
        "completed_at": null
      }
    }
  }
}
```

**Key Points:**
- ✅ **One file** (`user_progress.json`) stores everything
- ✅ **Each week (1-6)** has its own dictionary entry in `"weeks"`
- ✅ **All weeks** are tracked for each user session
- ✅ Weeks that haven't been started have `null` values for `started_at` and `completed_at`
- ✅ See `PROGRESS_STRUCTURE_EXAMPLE.json` for a complete example with multiple users

### 3. API Endpoints Added

Both week files now have these endpoints:

- `GET /api/progress/status` - Get progress for all weeks
- `GET /api/progress/week/<week_number>` - Get status for specific week
- `POST /api/progress/check-unlock` - Check if a week is unlocked

### 4. Frontend Integration

The frontend (`index.html`) now:
- Reads the `?week=X` URL parameter to determine which week to load
- Dynamically sets `API_BASE` to the correct `/weekX` path on the unified server (falls back to per-port mode only when debugging)
- Loads progress status on page load
- Shows completion message and prompts to continue to next week

## Usage

### Starting the Servers

**Recommended (single origin):**

```bash
python app.py   # Frontend + proxy at http://localhost:5000
```

**Debug mode (run weeks individually):**

```bash
python week1_main.py        # Port 5001
python week2_main.py        # Port 5002
python week3_main.py        # Port 5003
```

### Accessing Weeks

- **Week 1:** `http://localhost:5000/?week=1`
- **Week 2:** `http://localhost:5000/?week=2` *(served by `week2_main.py` for now)*
- **Week 3:** `http://localhost:5000/?week=3` *(served by `week3_main.py`)*
- **Week 4-6:** `http://localhost:5000/?week=4/5/6` (once their backends are wired up)
- *(Debug mode)* you can still hit the legacy ports (5001-5006) if you run each week standalone.

The frontend will automatically route API calls to the correct server based on the week parameter.

### Progress Flow

1. User starts Week 1 (always unlocked)
2. As they complete questions, progress is saved to `user_progress.json`
3. When Week 1 is completed, it's marked as complete
4. User can now access Week 2 (unlocked after Week 1 completion)
5. When Week 2 is completed, user sees completion message

## Adding New Weeks

To add the next dedicated week backend (e.g., Week 4 or a refreshed Week 3 in the future):

1. **Create the new week file** following the same structure as `week3_main.py` (or `week2_main.py` if you need a lighter template).

2. **Update `progress_tracker.py`:**
   ```python
   WEEK_CONFIG = {
       1: {...},
       2: {
           "port": 5002,
           "file": "week2_main.py",  # WEEK2_TEMP
           "name": "Week 2",
           "title": "Building Resilience"
       },
       3: {
           "port": 5003,
           "file": "<future_week3_file>",
           "name": "Week 3",
           "title": "Career Exploration"
       },
       ...
   }
   ```

3. **Confirm `index.html` `WEEK_CONFIG`** lists the new `/weekX` path so the frontend can hit `/weekX/api/...`.

4. **Add progress tracking** to the new week file (mirror the helpers used here).

5. **Start the new server** on the configured port and test the flow end-to-end.

## Important Notes

- **Session Management:** Progress is tied to Flask session IDs. Users need to use the same browser/session to maintain progress.
- **File Location:** `user_progress.json` is created in the same directory as the Python files.
- **Week Unlocking:** Weeks are unlocked sequentially. Week N requires Week N-1 to be completed.
- **Port Conflicts:** Make sure each week uses a unique port to avoid conflicts.

## Troubleshooting

### Week Not Unlocking
- Check that `user_progress.json` exists and has the correct structure
- Verify that the previous week is marked as `"completed": true`
- Check server logs for progress tracking errors

### API Connection Issues
- Ensure both servers are running on their respective ports
- Verify the `?week=X` parameter matches the server you're accessing
- Check browser console for CORS or connection errors

### Progress Not Saving
- Check file permissions for `user_progress.json`
- Verify session IDs are consistent (check browser cookies)
- Look for errors in server logs related to progress tracking

