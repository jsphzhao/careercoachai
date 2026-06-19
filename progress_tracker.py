"""
Shared Progress Tracker for Multi-Week DRIVEN Program

This module provides a centralized way to track progress across all weeks.
Progress is stored in a JSON file that can be shared across different week servers.
"""

import os
import json
import uuid
from typing import Dict, Optional, List
from datetime import datetime

PROGRESS_FILE = "user_progress.json"

# Week configuration - maps week numbers to their server ports and file names
WEEK_CONFIG = {
    1: {
        "port": 5001,
        "file": "week1_main.py",
        "name": "Week 1",
        "title": "Thinking Flexibly and Goal Setting"
    },
    2: {
        "port": 5002,
        "file": "week2_main.py",
        "name": "Week 2",
        "title": "Building Resilience"  # TODO: Update with actual title
    },
    3: {
        "port": 5003,
        "file": "week3_main.py",
        "name": "Week 3",
        "title": "Career Exploration"
    },
    4: {
        "port": 5004,
        "file": "week4_main.py",  # TODO: Create this file
        "name": "Week 4",
        "title": "Interview Immersion"  # TODO: Update with actual title
    },
    5: {
        "port": 5005,
        "file": "week5_main.py",  # TODO: Create this file
        "name": "Week 5",
        "title": "Storytelling for Impact"  # TODO: Update with actual title
    },
    6: {
        "port": 5006,
        "file": "week6_main.py",  # TODO: Create this file
        "name": "Week 6",
        "title": "Launch & Celebrate"  # TODO: Update with actual title
    }
}


class ProgressTracker:
    """Manages user progress across all weeks."""
    
    def __init__(self, progress_file: str = PROGRESS_FILE):
        self.progress_file = progress_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create progress file if it doesn't exist."""
        if not os.path.exists(self.progress_file):
            with open(self.progress_file, 'w') as f:
                json.dump({}, f)
    
    def _load_progress(self) -> Dict:
        """Load progress from JSON file."""
        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_progress(self, progress: Dict):
        """Save progress to JSON file."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def get_user_progress(self, session_id: str) -> Dict:
        """Get progress for a specific user session."""
        progress = self._load_progress()
        return progress.get(session_id, {
            "name": None,
            "weeks": {},
            "current_week": 1,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        })
    
    def update_user_progress(self, session_id: str, week_number: int, 
                           question_number: int, completed: bool = False,
                           week_completed: bool = False):
        """Update progress for a specific user and week."""
        progress = self._load_progress()
        
        if session_id not in progress:
            progress[session_id] = {
                "name": None,
                "weeks": {},
                "current_week": week_number,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        
        user_progress = progress[session_id]
        
        # Initialize week if not exists
        if str(week_number) not in user_progress["weeks"]:
            user_progress["weeks"][str(week_number)] = {
                "completed": False,
                "questions_completed": {},
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "selected_problem": None,
                "selected_corner_piece": None
            }
        
        week_progress = user_progress["weeks"][str(week_number)]
        
        # Update question completion
        if completed:
            week_progress["questions_completed"][str(question_number)] = True
        
        # Update week completion
        if week_completed:
            week_progress["completed"] = True
            week_progress["completed_at"] = datetime.now().isoformat()
            # Update current week to next available week
            user_progress["current_week"] = week_number + 1
        
        # Update last updated timestamp
        user_progress["last_updated"] = datetime.now().isoformat()
        user_progress["current_week"] = max(user_progress["current_week"], week_number)
        
        self._save_progress(progress)
    
    def set_user_name(self, session_id: str, name: str):
        """Set the user's name in their progress."""
        progress = self._load_progress()
        
        if session_id not in progress:
            progress[session_id] = {
                "name": name,
                "weeks": {},
                "current_week": 1,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        else:
            progress[session_id]["name"] = name
            progress[session_id]["last_updated"] = datetime.now().isoformat()
        
        self._save_progress(progress)
    
    def is_week_unlocked(self, session_id: str, week_number: int) -> bool:
        """Check if a week is unlocked (previous week completed)."""
        # TEMPORARY: Unlock all weeks for testing
        # TODO: Re-enable sequential unlocking later
        return True
        
        # Original sequential unlocking logic (commented out for testing):
        # if week_number == 1:
        #     return True  # Week 1 is always unlocked
        # 
        # progress = self._load_progress()
        # user_progress = progress.get(session_id, {})
        # 
        # # Check if previous week is completed
        # prev_week = week_number - 1
        # while prev_week >= 1:
        #     if str(prev_week) in user_progress.get("weeks", {}):
        #         week_data = user_progress["weeks"][str(prev_week)]
        #         if week_data.get("completed", False):
        #             return True
        #         # If previous week exists but not completed, check if it's the immediate previous
        #         if prev_week == week_number - 1:
        #             return False
        #     prev_week -= 1
        # 
        # # If no previous week data, only allow week 1
        # return week_number == 1
    
    def get_week_status(self, session_id: str, week_number: int) -> Dict:
        """Get detailed status for a specific week."""
        progress = self._load_progress()
        user_progress = progress.get(session_id, {})
        week_data = user_progress.get("weeks", {}).get(str(week_number), {})
        
        return {
            "unlocked": self.is_week_unlocked(session_id, week_number),
            "completed": week_data.get("completed", False),
            "questions_completed": week_data.get("questions_completed", {}),
            "started_at": week_data.get("started_at"),
            "completed_at": week_data.get("completed_at"),
            "selected_problem": week_data.get("selected_problem"),
            "selected_corner_piece": week_data.get("selected_corner_piece")
        }
    
    def get_all_weeks_status(self, session_id: str) -> Dict:
        """Get status for all weeks."""
        result = {}
        for week_num in WEEK_CONFIG.keys():
            result[week_num] = {
                **self.get_week_status(session_id, week_num),
                "config": WEEK_CONFIG[week_num]
            }
        return result
    
    def get_current_week(self, session_id: str) -> int:
        """Get the current week for a user."""
        progress = self._load_progress()
        user_progress = progress.get(session_id, {})
        return user_progress.get("current_week", 1)
    
    def get_week_port(self, week_number: int) -> Optional[int]:
        """Get the port number for a specific week's server."""
        return WEEK_CONFIG.get(week_number, {}).get("port")
    
    def get_available_weeks(self) -> List[int]:
        """Get list of all available week numbers."""
        return sorted(WEEK_CONFIG.keys())
    
    def save_selected_problem(self, session_id: str, week_number: int, problem_text: Optional[str]):
        """Persist the user's selected problem for a given week."""
        if not problem_text:
            return
        
        cleaned_problem = problem_text.strip()
        if not cleaned_problem:
            return
        
        progress = self._load_progress()
        
        if session_id not in progress:
            progress[session_id] = {
                "name": None,
                "weeks": {},
                "current_week": week_number,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        
        user_progress = progress[session_id]
        
        if str(week_number) not in user_progress["weeks"]:
            user_progress["weeks"][str(week_number)] = {
                "completed": False,
                "questions_completed": {},
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "selected_problem": None,
                "selected_corner_piece": None
            }
        
        week_progress = user_progress["weeks"][str(week_number)]
        week_progress["selected_problem"] = cleaned_problem
        user_progress["last_updated"] = datetime.now().isoformat()
        
        self._save_progress(progress)

    def save_selected_corner_piece(self, session_id: str, week_number: int, corner_piece_text: Optional[str]):
        """Persist the user's selected corner piece for a given week."""
        if not corner_piece_text:
            return
        
        cleaned_text = corner_piece_text.strip()
        if not cleaned_text:
            return
        
        progress = self._load_progress()
        
        if session_id not in progress:
            progress[session_id] = {
                "name": None,
                "weeks": {},
                "current_week": week_number,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        
        user_progress = progress[session_id]
        
        if str(week_number) not in user_progress["weeks"]:
            user_progress["weeks"][str(week_number)] = {
                "completed": False,
                "questions_completed": {},
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "selected_problem": None,
                "selected_corner_piece": None
            }
        
        week_progress = user_progress["weeks"][str(week_number)]
        week_progress["selected_corner_piece"] = cleaned_text
        user_progress["last_updated"] = datetime.now().isoformat()
        
        self._save_progress(progress)


# Global instance
progress_tracker = ProgressTracker()

