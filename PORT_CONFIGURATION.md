# Port Configuration for 6-Week DRIVEN Program

## Port Assignment

The system uses **6 ports** (5001-5006) for the 6 weeks:

| Week | Port | File | Status |
|------|------|------|--------|
| Week 1 | 5001 | `week1_main.py` | ✅ Implemented |
| Week 2 | 5002 | `week2_main.py` | ✅ Temporarily Implemented |
| Week 3 | 5003 | `week3_main.py` | ✅ Implemented |
| Week 4 | 5004 | `week4_main.py` | ⚠️ TODO: Create |
| Week 5 | 5005 | `week5_main.py` | ⚠️ TODO: Create |
| Week 6 | 5006 | `week6_main.py` | ⚠️ TODO: Create |

## Port Availability

**Answer: Yes, there are plenty of ports available!**

- **Total available ports:** 65,535 (ports 0-65535)
- **System reserved ports:** 0-1023 (should not be used)
- **Commonly used range:** 1024-4999 (may conflict with other services)
- **Recommended range:** 5000-65535 (safe for development)

**We're using ports 5001-5006**, which is well within the safe range and leaves **65,529 ports** available for other uses.

## Current Configuration

### `progress_tracker.py`
All 6 weeks are configured with their respective ports.

### `index.html`
Frontend is configured to route to all 6 weeks based on the `?week=X` parameter.

### Week Files
- ✅ `week1_main.py` (Week 1) - Port 5001
- ✅ `week2_main.py` *(WEEK2_TEMP)* (serving Week 2) - Port 5002
- ✅ `week3_main.py` (Week 3) - Port 5003
- ⚠️ Need to create: `week6_main.py` (future weeks may still evolve)

## Unified Development Server (Recommended)

Run everything through `app.py`, which serves the frontend and proxies each week app under a single origin (`http://localhost:5000`):

```bash
python app.py  # visit http://localhost:5000/?week=1 (or 2, 3, ...)
```

- `/week1` → `week1_main.py`
- `/week2` → `week2_main.py`
- `/week3` → `week3_main.py`
- Future weeks can be mounted the same way once their files exist.

## Starting All Servers Individually (Debug Mode)

To run each week separately (useful for debugging), start them in separate terminals:

```bash
# Terminal 1 - Week 1
python week1_main.py  # Port 5001

# Terminal 2 - Week 2 (TEMP: run week2_main.py)
python week2_main.py      # Port 5002  # WEEK2_TEMP

# Terminal 3 - Week 3
python week3_main.py   # Port 5003

# Terminal 4 - Week 4
python week4_main.py      # Port 5004

# Terminal 5 - Week 5
python week5_main.py      # Port 5005

# Terminal 6 - Week 6
python week6_main.py      # Port 5006
```

## Notes

- Ports 5001-5006 remain available if you want to start an individual week manually.
- The unified server runs on port 5000 by default (configurable via `PORT` env var in `app.py`).
- All weeks share the same `progress_tracker` data, so users can hop between weeks without switching browser origins.

## Port Conflicts

If you encounter port conflicts:

1. **Check what's using the port:**
   ```bash
   # macOS/Linux
   lsof -i :5001
   
   # Windows
   netstat -ano | findstr :5001
   ```

2. **Kill the process** (if safe to do so):
   ```bash
   # macOS/Linux
   kill -9 <PID>
   
   # Windows
   taskkill /PID <PID> /F
   ```

3. **Or change the port** in the configuration files if needed.

## Notes

- Ports 5001-5006 are commonly used for Flask development, so they're a good choice
- Each week server runs independently, so you can start/stop them individually
- The frontend automatically routes to the correct port based on the week parameter
- Progress is shared across all weeks via `user_progress.json`

