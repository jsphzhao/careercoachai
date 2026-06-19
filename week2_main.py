"""
Week 2 Main Application (TEMPORARY)

# WEEK2_TEMP: This file has been temporarily repurposed to serve the Week 2 experience
# until a dedicated Week 3 backend is provided. All references within the file (and
# associated UI plumbing) have been updated to reflect Week 2 content.

This file contains the complete Week 2 implementation for the DRIVEN program.
It includes:
- Week 2 questions
- Week 2 system prompts
- Validation logic
- All conversation flow logic

This is a self-contained file - all Week 2 content is embedded here (temporarily).
"""

import os
import uuid
import subprocess
import sys
from flask import Flask, request, jsonify, session, Response
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from progress_tracker import progress_tracker

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-" + uuid.uuid4().hex)

# Configure session cookie settings
# Since frontend and backend are on same origin (same Flask app), Lax works fine
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Works for same-origin requests
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# CORS configuration (for API calls from same origin, this should work fine)
CORS(app, supports_credentials=True)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set your OPENAI_API_KEY in the .env file.")

# Global state (in production, use a database or session management)
conversation_states = {}


class ConversationState:
    """Manages state for a single conversation."""
    
    def __init__(self, name):
        self.name = name
        self.current_question = 1  # Start with question 1
        self.answers = {}  # qnum -> list of user responses
        self.nova_responses = {}  # qnum -> list of NOVA responses
        self.iteration_count = {}  # qnum -> iteration count (for 2-iteration loop)
        self.question_completed = {}  # qnum -> bool (whether question is fully completed)
        self.q1_scenario = None  # Store Q1 scenario classification: "SCENARIO_1" or "SCENARIO_2"
        self.q2_scenario = None  # Store Q2 scenario classification: "SCENARIO_1" or "SCENARIO_2"
        self.q3_scenario = None  # Store Q3 scenario classification: "SCENARIO_1" or "SCENARIO_2"
        self.q4_scenario = None  # Store Q4 scenario classification: "SCENARIO_1" or "SCENARIO_2"
        self.q5_scenario = None  # Store Q5 scenario classification: "SCENARIO_1" or "SCENARIO_2"
        self.selected_problem = None  # Persist the problem the user selected in Q4
        self.selected_corner_piece = None  # Persist the corner piece the user selected in Q5
    
    def get_iteration(self, qnum):
        """Get current iteration count for a question."""
        return self.iteration_count.get(qnum, 0)


def get_or_create_state(name=None):
    """Get or create a conversation state using Flask session."""
    # Make session permanent to ensure it persists
    session.permanent = True
    
    session_id = session.get('session_id')
    if session_id is None:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        # Explicitly mark session as modified to ensure cookie is sent
        session.modified = True
        print(f"[DEBUG STATE] Created new session_id: {session_id}")
    else:
        print(f"[DEBUG STATE] Using existing session_id: {session_id}")
    
    if session_id not in conversation_states:
        if name is None:
            # Try to get name from existing state or use default
            name = "Friend"
        conversation_states[session_id] = ConversationState(name)
        print(f"[DEBUG STATE] Created new state for session_id: {session_id}, name: {name}")
    else:
        print(f"[DEBUG STATE] Retrieved existing state for session_id: {session_id}, current_question: {conversation_states[session_id].current_question}")
    
    # Ensure selected problem/corner piece stay in sync with persisted progress
    state = conversation_states[session_id]
    progress_data = progress_tracker.get_user_progress(session_id)
    week_data = progress_data.get("weeks", {}).get(str(2), {})  # WEEK2_TEMP
    persisted_problem = week_data.get("selected_problem")
    persisted_corner_piece = week_data.get("selected_corner_piece")
    if persisted_problem:
        state.selected_problem = persisted_problem
    if persisted_corner_piece:
        state.selected_corner_piece = persisted_corner_piece
    
    return conversation_states[session_id]


def save_question_progress(question_number: int, week_number: int = 3):
    """Helper function to save question completion to progress tracker."""
    session_id = session.get('session_id')
    if session_id:
        progress_tracker.update_user_progress(session_id, week_number, question_number, completed=True)


def call_llm(system_prompt, user_message):
    """Call OpenAI API with system prompt and user message."""
    try:
        # Format system prompt with name if needed
        if "{name}" in system_prompt:
            state = get_or_create_state()
            system_prompt = system_prompt.replace("{name}", state.name)
        
        response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I apologize, but I encountered an error processing your response. Please try again. Error: {str(e)}"


def _extract_latest_substantive_response(responses, min_chars: int = 10):
    """Return the most recent substantive response that meets a minimum character threshold."""
    if not responses:
        return None
    for response in reversed(responses):
        cleaned = response.strip()
        if len(cleaned) >= min_chars:
            return cleaned
    return responses[-1].strip() if responses else None


def save_selected_problem_to_state(state):
    """Persist the selected problem in both state and shared progress."""
    if not state:
        return
    responses = state.answers.get(4, [])
    problem_summary = _extract_latest_substantive_response(responses)
    if not problem_summary:
        return
    state.selected_problem = problem_summary
    session_id = session.get('session_id')
    if session_id:
        progress_tracker.save_selected_problem(session_id, 2, problem_summary)  # WEEK2_TEMP


def save_selected_corner_piece_to_state(state):
    """Persist the selected corner piece in both state and shared progress."""
    if not state:
        return
    responses = state.answers.get(5, [])
    corner_piece_summary = _extract_latest_substantive_response(responses)
    if not corner_piece_summary:
        return
    state.selected_corner_piece = corner_piece_summary
    session_id = session.get('session_id')
    if session_id:
        progress_tracker.save_selected_corner_piece(session_id, 2, corner_piece_summary)  # WEEK2_TEMP


def validate_completeness(question_number, nova_response, user_responses, conversation_history,
                          q1_scenario=None, q2_scenario=None, q3_scenario=None,
                          q4_scenario=None, q5_scenario=None):
    """
    Use LLM to validate if user has provided all required information.
    Returns (is_complete: bool, missing_items: str)
    q1_scenario, q2_scenario, q3_scenario, q4_scenario, q5_scenario: Pass the scenario classification for each question
    """
    # Build context of what was requested
    context = f"Question {question_number}: {QUESTIONS.get(question_number, '')}\n\n"
    context += f"NOVA's latest response: {nova_response}\n\n"
    context += f"User's responses so far: {len(user_responses)} response(s)\n"
    for i, resp in enumerate(user_responses, 1):
        context += f"  Response {i}: {resp[:200]}...\n"
    
    # WEEK2_TEMP: Validation prompts for Week 2 questions
    if question_number == 1:
        # Q1: "What did you think about the second week's materials?"
        # Both scenarios are valid responses - mark as complete
        if q1_scenario and ("SCENARIO_1" in q1_scenario or "SCENARIO_2" in q1_scenario):
            print(f"[DEBUG Q1 VALIDATION] Scenario 1 or 2 detected - marking complete")
            return True, "None"
        else:
            # No scenario set yet - validate if user provided a response
            validation_prompt = """You are validating if a user has provided a response to "What did you think about the second week's materials?"

The user has provided at least one response. Determine if they have given ANY answer about their thoughts on the Week 2 materials, even if it's brief.
- If they shared any thoughts, opinions, or reflections about the materials → COMPLETE
- If they expressed confusion or said materials were unclear → COMPLETE (this is a valid response)
- If they gave a positive, neutral, or negative reflection → COMPLETE
- If they gave a non-meaningful response → COMPLETE (will be handled by Scenario 2)
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any response: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about what the user thought about the second week's materials"
"""
    
    elif question_number == 2:
        # Q2: "What was one main idea you took away?"
        # If Scenario 1 is classified, user provided a main takeaway - mark complete
        if q2_scenario and "SCENARIO_1" in q2_scenario:
            print(f"[DEBUG Q2 VALIDATION] Scenario 1 detected - user provided main takeaway, marking complete")
            return True, "None"
        elif q2_scenario and "SCENARIO_2" in q2_scenario:
            # Scenario 2: User confused or non-meaningful response - NOVA will provide summary, mark complete
            print(f"[DEBUG Q2 VALIDATION] Scenario 2 detected - NOVA will provide summary, marking complete")
            return True, "None"
        else:
            # No scenario set yet - validate if user provided a main idea
            print(f"[DEBUG Q2 VALIDATION] No scenario set - validating main takeaway")
            validation_prompt = """You are validating if a user has provided a main idea they took away from the second week's materials.

The original question was: "What was one main idea you took away?"

The user has provided at least one response. Determine if they have given ANY main takeaway idea, even if it's brief.
- If they mentioned a specific concept, strategy, or idea from Week 2 materials → COMPLETE
- If they expressed confusion or gave non-meaningful response → COMPLETE (will be handled by Scenario 2)
- If they gave any response related to what they learned or understood → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided a main idea: "COMPLETE: Yes\nMISSING: None"
- If no main idea provided: "COMPLETE: No\nMISSING: A main idea the user took away from the second week's materials"
"""
    
    elif question_number == 3:
        # Q3: "Were there any parts that felt unclear or tricky?"
        # Both scenarios are valid responses - just need to check if user answered
        if q3_scenario and ("SCENARIO_1" in q3_scenario or "SCENARIO_2" in q3_scenario):
            print(f"[DEBUG Q3 VALIDATION] Scenario detected - user answered, marking complete")
            return True, "None"
        else:
            # No scenario set yet - validate if user provided an answer
            validation_prompt = """You are validating if a user has provided a response to "Were there any parts that felt unclear or tricky?"

The user has provided at least one response. Determine if they have answered the question, even if briefly.
- If they said yes/no or indicated confusion → COMPLETE
- If they said no confusion or that everything was clear → COMPLETE
- If they mentioned specific parts that were unclear → COMPLETE
- If they said nothing was unclear → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user answered yes/no or provided clarification: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response indicating whether parts felt unclear or tricky"
"""
    
    elif question_number == 4:
        # Q4: "What problem did you pick and why?"
        # Scenario 1: User named the problem and reason → complete immediately
        # Scenario 2: User still needs support selecting a problem or explaining why
        if q4_scenario and "SCENARIO_1" in q4_scenario:
            print(f"[DEBUG Q4 VALIDATION] Scenario 1 detected - user named problem and reason, marking complete")
            return True, "None"
        elif q4_scenario and "SCENARIO_2" in q4_scenario:
            print(f"[DEBUG Q4 VALIDATION] Scenario 2 detected - validating problem + reason")
            validation_prompt = """You are validating if the user has now named the specific problem they chose for the corner-piece exercise AND explained why they picked it.

The user initially indicated they did not do the exercise. NOVA has been helping them complete it.

- If they mentioned a specific problem AND explained why it matters (why they chose it) → COMPLETE
- If they only gave a problem without the reason, or a reason without the problem → INCOMPLETE
- If they still haven't picked a problem → INCOMPLETE

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing (problem or reason or both), or "None" if all provided]

Example responses:
- If user selected a specific problem and gave a reason: "COMPLETE: Yes\nMISSING: None"
- If missing either item: "COMPLETE: No\nMISSING: A specific problem and/or the reason they picked it"
"""
        else:
            validation_prompt = """You are validating if a user has provided a response to "What problem did you pick and why?"

The user has provided at least one response. Determine if they have answered the question.
- If they provided the problem AND why they chose it → COMPLETE
- If they clearly stated they have not completed the exercise yet → INCOMPLETE (NOVA needs to help)
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing (problem or reason or both), or "None" if all provided]

Example responses:
- If user provided both items: "COMPLETE: Yes\nMISSING: None"
- If missing: "COMPLETE: No\nMISSING: The problem they picked and/or why they picked it"
"""
    elif question_number == 5:
        # Q5: "What corner piece did you pick, and why did you pick it to start solving the problem?"
        if q5_scenario and "SCENARIO_1" in q5_scenario:
            print(f"[DEBUG Q5 VALIDATION] Scenario 1 detected - user explained corner piece choice, marking complete")
            return True, "None"
        else:
            validation_prompt = """You are validating if a user has responded to "What corner piece did you pick, and why did you pick it to start solving the problem?"

The user must provide BOTH:
- The specific corner piece or starter action they will use
- A clear reason why that corner piece is important to them. Be lenient with the user and do not ask follow up questions if they have provided a response that is meaningful to the question asked.

If either the corner piece or the reason is missing, mark INCOMPLETE.

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing (corner piece, reason, or both), or "None" if all provided]

Example responses:
- If user shared both items: "COMPLETE: Yes\nMISSING: None"
- If missing the reason: "COMPLETE: No\nMISSING: The reason they chose that corner piece"
- If missing the corner piece: "COMPLETE: No\nMISSING: The specific corner piece they plan to use"
"""
    elif question_number == 6:
        # Q6 always completes once the user responds
        return True, "None"
    else:
        # Generic validation for other questions - only use for Q2-like scenarios
        # For simple questions, rely on iteration limits
        return True, "None"  # Skip validation for other questions
    
    try:
        validation_response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": validation_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.3  # Lower temperature for more consistent validation
        )
        
        validation_text = validation_response.choices[0].message.content.strip()
        
        print(f"[DEBUG] Validation LLM response: {validation_text}")
        
        # Parse the response
        is_complete = False
        missing_items = "Unknown"
        
        if "COMPLETE: Yes" in validation_text or "COMPLETE:YES" in validation_text.upper():
            is_complete = True
            print(f"[DEBUG] ✓ Validation result: COMPLETE")
        elif "COMPLETE: No" in validation_text or "COMPLETE:NO" in validation_text.upper():
            is_complete = False
            print(f"[DEBUG] ✗ Validation result: INCOMPLETE")
        
        # Extract missing items
        missing_match = None
        if "MISSING:" in validation_text:
            missing_match = validation_text.split("MISSING:")[-1].strip()
            missing_items = missing_match if missing_match else "Unknown"
            print(f"[DEBUG] Missing items: {missing_items}")
        
        print(f"[DEBUG] Final validation decision: is_complete={is_complete}, missing_items={missing_items}")
        print(f"{'='*80}\n")
        
        return is_complete, missing_items
        
    except Exception as e:
        # On error, assume incomplete to be safe (continue the loop)
        print(f"Validation error: {e}")
        return False, "Validation error occurred"


# WEEK2_TEMP: Week 2 specific data and helper content
# Week 2 Videos + Exercises content (used in Week 2 questions)
WEEK2_VIDEOS_EXERCISES = """Session 1

Exercise 1

Exercise 1 blurb: Think about a specific problem related to your job search. Ideally, this problem is something that has been difficult to start taking action on. For example: Problem: I really need a new interview outfit, but I don't like shopping and don't know where I'll find the money.
When we're feeling low, it can often be hard to take action and tackle problems. One of the best ways to get started is by breaking down problems into smaller steps.
So, imagine your problem is a jigsaw puzzle. One way to start a puzzle is to look for the easy corner pieces and go from there. Thinking about your problem, what might your easy corner pieces be? For example, you could start cracking the interview outfit problem "puzzle" by A) looking online for discount clothing options, or B) asking a friend to go shopping with you as moral support. Both A and B are corner pieces. You haven't completely solved the problem of the interview outfit, but you've gotten a good start!

Exercise 1 Questions:
1. Write your problem here: (example: I really need a new interview outfit, but I don't like shopping and don't know where I'll find the money).
2. Write one corner piece you will place on your next problem puzzle over the next week. Talk with your coach about it the next time you meet.


Session 2

Exercise 2

Exercise 2 Blurb: Sometimes, we respond to distress by thinking over and over again about a negative event and how badly we feel about it. This is called repetitive thinking. The trouble is: repetitive thinking often doesn't solve the problem, and instead makes you feel more stressed out, particularly when you feel like you can't slow down or get "unstuck" from a thought. Try the following exercise to break free from a thought that may be bothering you at the moment.

Exercise 2 Questions: 
1. What is the repetitive thought? (Sample: That last conversation I had with HR.)
2. When did you start thinking about this? (Sample: On my last day of work.)
3. How long have you been thinking about this? (Sample: Two weeks)
4. How do you usually feel when you're thinking about this, or afterwards? (Sample: I feel angry at my old company and sorry for myself.)
5. In the moments that you are thinking about this, what else could you do to distract yourself and focus on something else? (Sample: If I'm in bed, I could read a book until I get sleepy. If I'm walking to the train, I could people watch or listen to music on my phone.)

Week 2 covered recognizing symptoms of stress and using strategies for thinking flexibly + taking action to reduce them. It also reviewed types of negative thoughts and how to challenge them so you can stay focused on your job search."""

# Corner piece method explanation
CORNER_PIECE_METHOD = """When we're feeling low, it can often be hard to take action and tackle problems. One of the best ways to get started is by breaking down problems into smaller steps.
So, imagine your problem is a jigsaw puzzle. One way to start a puzzle is to look for the easy corner pieces and go from there. Thinking about your problem, what might your easy corner pieces be? For example, you could start cracking the interview outfit problem "puzzle" by A) looking online for discount clothing options, or B) asking a friend to go shopping with you as moral support. Both A and B are corner pieces. You haven't completely solved the problem of the interview outfit, but you've gotten a good start!"""

Q6_CANNED_RESPONSE = """Great work on the exercise and coming up with ways to start solving your problem. If you don't have a complete fix yet, that's okay. Just keep adding solutions - or pieces - one at a time until the puzzle is complete.

# WEEK2_TEMP: Mention Week 2 instead of Week 3 while this file is repurposed.
I'll check in later this week to discuss Week 2's content and exercises. Talk with you later and keep up the amazing work!"""

# Global instruction used to ensure no extra questions once a question is complete
NO_FOLLOWUPS_INSTRUCTION = (
    "Important: If the user's response already satisfies the criteria for this question and it is considered complete (or the classified scenario instructs to move on), do not ask any additional questions. Conclude your reply without further questions so we can proceed to the next question."
)

# WEEK2_TEMP: Week 2 Questions
QUESTIONS = {
    1: "What did you think about the second week's materials?",
    2: "What was one main idea you took away?",
    3: "Were there any parts that felt unclear or tricky?",
    4: "What problem did you pick and why?",
    5: "What corner piece did you pick, and why did you pick it to start solving the problem?",
    6: "Were you able to take any action on that corner piece in the last few weeks?",
}

# WEEK2_TEMP: Week 2 System Prompts
SYSTEM_PROMPTS = {
    1: {
        "classifier": f"""Based on the user's response to "What did you think about the second week's materials?", determine which scenario applies:

Scenario 1: User shares a general positive or neutral reflection about the Week 2 materials.

Scenario 2: User expresses confusion, says materials were unclear, OR does not provide a response that is meaningful to the question asked.

Respond with ONLY one of these: "SCENARIO_1" or "SCENARIO_2".""",
        "scenario_1_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. Above is the user's thoughts on the second session's material for the DRIVEN program. The user's name is {{name}}.

The user shares a general positive or neutral reflection.

In this scenario respond with a short summary + reinforcement of Week 2 themes {WEEK2_VIDEOS_EXERCISES}.""",
        "scenario_2_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. Above is the user's response to "What did you think about the second week's materials?" for the DRIVEN program. The user's name is {{name}}.

The user either: (1) expresses confusion or says materials were unclear, OR (2) does not provide a response that is meaningful to the question asked.

In this scenario, provide a comprehensive summary of Week 2 content using {WEEK2_VIDEOS_EXERCISES}. Cover the main themes: recognizing symptoms of stress, using strategies for thinking flexibly and taking action to reduce them, types of negative thoughts and how to challenge them. 

IMPORTANT: Do NOT ask the user if they need clarifications or if they have questions. Do NOT ask any follow-up questions. Simply provide the comprehensive summary and conclude your response. We will move on to the next question."""
    },
    
    2: {
        "classifier": f"""Based on the user's response to "What was one main idea you took away?", determine which scenario applies:

Scenario 1: User provides a main takeaway idea.

Scenario 2: User expresses confusion, does not provide a main takeaway idea, OR does not provide a response that is meaningful to the question asked.

Respond with ONLY one of these: "SCENARIO_1" or "SCENARIO_2".""",
        "scenario_1_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. Above is the main idea the user learned from the second session's material for the DRIVEN program. The user's name is {{name}}.

The user provides a main takeaway idea.

In this scenario, respond by congratulating the user.""",
        "scenario_2_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. Above is the user's response to "What was one main idea you took away?" from the second session's material for the DRIVEN program. The user's name is {{name}}.

The user either: (1) expresses confusion or does not provide a main takeaway idea, OR (2) does not provide a response that is meaningful to the question asked.

In this scenario, provide a comprehensive summary of the main ideas in Week 2 content using {WEEK2_VIDEOS_EXERCISES}. Cover the key concepts: recognizing symptoms of stress, using strategies for thinking flexibly and taking action, types of negative thoughts and how to challenge them.

IMPORTANT: Do NOT ask the user if they need clarifications or if they have questions. Do NOT ask any follow-up questions. Simply provide the comprehensive summary and conclude your response. We will move on to the next question."""
    },
    
    3: {
        "classifier": f"""Based on the user's response to "Were there any parts that felt unclear or tricky?", determine which scenario applies:

Scenario 1: User reports confusion.

Scenario 2: User reports no confusion.

Respond with ONLY one of these: "SCENARIO_1" or "SCENARIO_2".""",
        "scenario_1_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. Above are the parts of the second session's material for the DRIVEN program that the user felt unclear about or felt was tricky. The user's name is {{name}}.

The user reports confusion.

Offer tailored clarification using {WEEK2_VIDEOS_EXERCISES}.""",
        "scenario_2_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. Above are the parts of the second session's material for the DRIVEN program that the user felt unclear about or felt was tricky. The user's name is {{name}}.

The user reports no confusion.

Reinforce confidence and bridge to the exercise review."""
    },
    
    4: {
        "classifier": f"""Based on the user's response to "What problem did you pick and why?", determine which scenario applies:

Scenario 1: User response indicates they completed the exercise, clearly naming the problem they chose AND why it matters.

Scenario 2: User response indicates they did not complete the exercise, did not name a specific problem, or did not explain why they picked it.

Respond with ONLY one of these: "SCENARIO_1" or "SCENARIO_2".""",
        "scenario_1_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. Above is the problem the user selected and why it matters to them. The user's name is {{name}}.

The user response indicates that they completed the exercise, named the problem, and explained why it matters.

Reflect their problem choice and reasoning back empathetically. Reinforce how focusing on that problem will help them keep momentum. {NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_2_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. Above is the user's response to "What problem did you pick and why?" The user's name is {{name}}.

The user response indicates that they did not name a specific problem, did not explain why they picked it, or they have not completed the exercise yet.

Spend up to two turns helping them complete it. Use the corner piece method explanation: {CORNER_PIECE_METHOD}. Ask concise follow-up questions to help them identify (1) the specific problem they will work on and (2) why that problem is important right now. Once they have clearly shared both the problem and the reason, reflect it back, let them know you'll keep this in mind for future steps, and conclude your response."""
    },
    
    5: {
        "classifier": f"""Based on the user's response to "What corner piece did you pick, and why did you pick it to start solving the problem?", determine which scenario applies:

Scenario 1: User response clearly states which corner piece they chose for the selected problem AND why they chose it (why it feels like the right starting point).

Scenario 2: User response does NOT explain the corner piece choice, does not mention a corner piece, or does not explain why they chose this cornerpiece. Be lenient with the user and do not ask follow up questions if they have provided a response that is meaningful to the question asked.

Respond with ONLY one of these: "SCENARIO_1" or "SCENARIO_2".""",
        "scenario_1_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's selected problem is {{Selected_problem}}. Above is the corner piece they chose and why it matters. The user's name is {{name}}.

The user response indicates they clearly stated the corner piece and their reasoning.

Reflect their corner piece choice and reasoning back empathetically. Emphasize how it will help them get momentum on {{Selected_problem}}. {NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_2_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's selected problem is {{Selected_problem}}. The user's name is {{name}}.

The user response does not yet explain which corner piece they are choosing or why it matters.

Spend up to two turns helping them complete it. Reiterate the corner piece method using this explanation: {CORNER_PIECE_METHOD}. Ask concise follow-up questions to help them identify (1) the specific corner piece they will use for {{Selected_problem}} and (2) why that corner piece is a good starting place. Once they share both pieces, reflect it back, let them know you'll remember this corner piece, and conclude your response."""
    }
}

WELCOME_MESSAGE = """Welcome to your second coaching session! You've made it through half of the content in the DRIVEN program.

Our goal for the second week of DRIVEN is for participants to learn to recognize the symptoms of stress and combat them using our core strategies of thinking flexibly and taking action.

Our goal for the third week of DRIVEN is for participants to better understand and build their own skills based on what employers are looking for and the types of jobs that best fit into their life.

If you haven't had a chance yet, this is your gentle reminder to complete your weekly check-in.

In today's session, we will: Recap the second and third week's ideas and guidance. Review your exercise on solving a big problem using corner pieces from Week 2. If you didn't have a chance to finish something, we can look at it together.

Week 2 covered recognizing symptoms of stress and using strategies for thinking flexibly + taking action to reduce them. It also reviewed types of negative thoughts and how to challenge them so you can stay focused on your job search."""


@app.route('/')
def index():
    """Serve the index.html file."""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return Response(f.read(), mimetype='text/html')
    except FileNotFoundError:
        return "index.html not found. Please make sure the file exists in the same directory as week2_main.py.", 404


@app.route('/api/progress/status', methods=['GET'])
def get_progress_status():
    """Get progress status for all weeks."""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"success": False, "error": "No session found"}), 400
    
    weeks_status = progress_tracker.get_all_weeks_status(session_id)
    current_week = progress_tracker.get_current_week(session_id)
    
    return jsonify({
        "success": True,
        "current_week": current_week,
        "weeks": weeks_status
    })


@app.route('/api/progress/week/<int:week_number>', methods=['GET'])
def get_week_status(week_number):
    """Get status for a specific week."""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"success": False, "error": "No session found"}), 400
    
    status = progress_tracker.get_week_status(session_id, week_number)
    return jsonify({
        "success": True,
        "week": week_number,
        "status": status
    })


@app.route('/api/progress/check-unlock', methods=['POST'])
def check_week_unlock():
    """Check if a week is unlocked for the current user."""
    data = request.get_json()
    week_number = data.get('week_number', 2)  # WEEK2_TEMP: default to Week 2
    
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"success": False, "error": "No session found"}), 400
    
    is_unlocked = progress_tracker.is_week_unlocked(session_id, week_number)
    return jsonify({
        "success": True,
        "week": week_number,
        "unlocked": is_unlocked
    })


@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize a new conversation."""
    data = request.get_json()
    name = data.get('name', '').strip()
    week_number = 2  # WEEK2_TEMP: force progress saves to Week 2
    
    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400
    
    # Get or create session
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    # TEMPORARY: All weeks unlocked for testing
    # TODO: Re-enable unlock check later
    # if not progress_tracker.is_week_unlocked(session_id, week_number):
    #     return jsonify({
    #         "success": False,
    #         "error": f"Week {week_number} is locked. Please complete previous weeks first.",
    #         "unlocked": False
    #     }), 403
    
    # Save name to progress tracker
    progress_tracker.set_user_name(session_id, name)
    
    # Create new state (will create session if needed)
    state = get_or_create_state(name)
    state.name = name
    state.current_question = 1  # Start with question 1
    state.question_completed = {}
    
    # Return welcome message
    welcome_text = WELCOME_MESSAGE.replace("{name}", name)
    return jsonify({
        "success": True,
        "message": welcome_text,
        "week": week_number
    })


@app.route('/api/get_next_message', methods=['POST'])
def get_next_message():
    """Get the next message in the dialogue flow."""
    state = get_or_create_state()
    
    print(f"[DEBUG GET_NEXT_MESSAGE] session_id={session.get('session_id')}, current_question={state.current_question}, name={state.name}")
    print(f"[DEBUG GET_NEXT_MESSAGE] Answers: {list(state.answers.keys())}, Completed: {list(state.question_completed.keys())}")
    print(f"[DEBUG GET_NEXT_MESSAGE] Available questions: {list(QUESTIONS.keys())}")
    
    # Safety check: if current_question is None or invalid, default to 1
    if state.current_question is None:
        print(f"[DEBUG GET_NEXT_MESSAGE] WARNING: current_question is None, defaulting to 1")
        state.current_question = 1
    
    # If we've completed all questions, return completion message
    if state.current_question > len(QUESTIONS):
        print(f"[DEBUG GET_NEXT_MESSAGE] All questions completed (current_question={state.current_question} > {len(QUESTIONS)})")
        
        # Mark week as completed in progress tracker
        session_id = session.get('session_id')
        if session_id:
            progress_tracker.update_user_progress(
                session_id, 
                week_number=2,  # WEEK2_TEMP
                question_number=len(QUESTIONS),
                week_completed=True
            )
        
        return jsonify({
            "success": True,
            "message": "Thank you for completing Week 2! Great job diving into the second week's materials.",
            "is_complete": True,
            "awaiting_response": False,
            "week_completed": True
        })
    
    # Return the current question
    qnum = state.current_question
    
    # Safety check: ensure question number is valid
    if qnum not in QUESTIONS:
        print(f"[ERROR GET_NEXT_MESSAGE] Invalid question number {qnum}, available questions: {list(QUESTIONS.keys())}")
        return jsonify({
            "success": False,
            "error": f"Invalid question number: {qnum}"
        }), 400
    
    question_text = QUESTIONS[qnum]
    
    # Add transition or contextual messaging for later questions
    if qnum == 4:
        transition_message = "Let's now take a look at the exercise you completed using the corner-piece method from Week 2!"
        question_text = f"{transition_message}\n\n{question_text}"
    elif qnum == 5:
        selected_problem = state.selected_problem or "the problem you identified earlier"
        contextual_intro = (
            f"Earlier you said you're working on this problem: {selected_problem}.\n"
            "Think back to the exercise where you identified small \"corner pieces\" to get started."
        )
        question_text = f"{contextual_intro}\n\n{CORNER_PIECE_METHOD}\n\n{question_text}"
    elif qnum == 6:
        selected_corner_piece = state.selected_corner_piece or "the corner piece you described earlier"
        contextual_intro = (
            f"You listed this corner piece to get started: {selected_corner_piece}."
        )
        question_text = f"{contextual_intro}\n\n{CORNER_PIECE_METHOD}\n\n{question_text}"
    
    print(f"[DEBUG GET_NEXT_MESSAGE] ✓ Returning question {qnum}: {question_text[:50]}...")
    print(f"[DEBUG GET_NEXT_MESSAGE] ✓ State after return - current_question={state.current_question}, session_id={session.get('session_id')}")
    
    return jsonify({
        "success": True,
        "message": f"{question_text}",
        "is_complete": False,
        "awaiting_response": True,
        "question_number": qnum
    })


@app.route('/api/process_response', methods=['POST'])
def process_response():
    """Process a user response to a question."""
    data = request.get_json()
    user_message = data.get('message', '').strip()
    question_number = data.get('question_number')
    
    if not user_message:
        return jsonify({"success": False, "error": "Message is required"}), 400
    
    state = get_or_create_state()
    
    print(f"[DEBUG PROCESS_RESPONSE START] session_id={session.get('session_id')}, state.current_question={state.current_question}, provided question_number={question_number}")
    print(f"[DEBUG PROCESS_RESPONSE START] Answers so far: {dict(state.answers)}, Iterations: {dict(state.iteration_count)}")
    
    # Use current question if not provided
    if question_number is None:
        question_number = state.current_question
    
    if question_number is None or question_number not in QUESTIONS:
        return jsonify({"success": False, "error": "No active question"}), 400
    
    print(f"[DEBUG PROCESS_RESPONSE] Processing response for question {question_number}")
    
    # Store user response
    if question_number not in state.answers:
        state.answers[question_number] = []
    state.answers[question_number].append(user_message)
    
    # Get current iteration before incrementing
    iteration = state.iteration_count.get(question_number, 0)
    
    # WEEK2_TEMP: Week 2 Question Processing
    # For Q1, classify first, then use hardcoded logic based on scenario
    if question_number == 1:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q1] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Step 1: Classify the scenario (only on first response to Q1)
        if not hasattr(state, 'q1_scenario') or state.q1_scenario is None:
            q1_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q1_prompts, dict) or "classifier" not in q1_prompts:
                return jsonify({"success": False, "error": "Q1 prompts not configured correctly"}), 400
            
            classifier_prompt = q1_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q1...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q1_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q1_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q1_prompts = SYSTEM_PROMPTS.get(question_number, {})
        
        if "SCENARIO_1" in state.q1_scenario:
            print(f"[DEBUG] SCENARIO 1 detected - positive/neutral reflection")
            system_prompt = q1_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q1_scenario:
            # Scenario 2: User expresses confusion OR provides non-meaningful response
            # Provide comprehensive summary and move on (no follow-ups)
            print(f"[DEBUG] SCENARIO 2 detected - confusion or non-meaningful response")
            system_prompt = q1_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 1 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 1")
            state.q1_scenario = "SCENARIO_1"
            system_prompt = q1_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
    
    # For Q2, classify first, then use hardcoded logic based on scenario
    elif question_number == 2:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q2] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Step 1: Classify the scenario (only on first response to Q2)
        if state.q2_scenario is None:
            q2_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q2_prompts, dict) or "classifier" not in q2_prompts:
                return jsonify({"success": False, "error": "Q2 prompts not configured correctly"}), 400
            
            classifier_prompt = q2_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q2...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q2_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q2_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q2_prompts = SYSTEM_PROMPTS.get(question_number, {})
        
        if "SCENARIO_1" in state.q2_scenario:
            # Scenario 1: User provided main takeaway - congratulate
            print(f"[DEBUG] SCENARIO 1 detected - congratulating user")
            system_prompt = q2_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q2_scenario:
            # Scenario 2: User confused, no takeaway, OR non-meaningful response
            # Provide comprehensive summary and move on (no follow-ups)
            print(f"[DEBUG] SCENARIO 2 detected - confusion, no takeaway, or non-meaningful response")
            system_prompt = q2_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 1 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 1")
            state.q2_scenario = "SCENARIO_1"
            system_prompt = q2_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
    
    # For Q3, classify first, then use hardcoded logic based on scenario
    elif question_number == 3:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q3] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Step 1: Classify the scenario (only on first response to Q3)
        if state.q3_scenario is None:
            q3_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q3_prompts, dict) or "classifier" not in q3_prompts:
                return jsonify({"success": False, "error": "Q3 prompts not configured correctly"}), 400
            
            classifier_prompt = q3_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q3...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q3_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q3_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q3_prompts = SYSTEM_PROMPTS.get(question_number, {})
        
        if "SCENARIO_1" in state.q3_scenario:
            # Scenario 1: User reports confusion - offer clarification
            print(f"[DEBUG] SCENARIO 1 detected - offering clarification")
            system_prompt = q3_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q3_scenario:
            # Scenario 2: User reports no confusion - reinforce confidence
            print(f"[DEBUG] SCENARIO 2 detected - reinforcing confidence")
            system_prompt = q3_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 2 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 2")
            state.q3_scenario = "SCENARIO_2"
            system_prompt = q3_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
    
    # For Q4, classify first, then use hardcoded logic based on scenario
    elif question_number == 4:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q4] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Step 1: Classify the scenario (only on first response to Q4)
        if state.q4_scenario is None:
            q4_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q4_prompts, dict) or "classifier" not in q4_prompts:
                return jsonify({"success": False, "error": "Q4 prompts not configured correctly"}), 400
            
            classifier_prompt = q4_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q4...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q4_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q4_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q4_prompts = SYSTEM_PROMPTS.get(question_number, {})
        
        if "SCENARIO_1" in state.q4_scenario:
            # Scenario 1: User did exercise and provided problem - reflect reasoning empathetically
            print(f"[DEBUG] SCENARIO 1 detected - user did homework, reflecting reasoning empathetically")
            system_prompt = q4_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q4_scenario:
            # Scenario 2: User did not do exercise - help them complete it (NO NO_FOLLOWUPS_INSTRUCTION - need follow-ups)
            print(f"[DEBUG] SCENARIO 2 detected - user didn't do homework, helping complete exercise")
            system_prompt = q4_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            # Don't add NO_FOLLOWUPS_INSTRUCTION - we need follow-ups to help them select a problem
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 2 if unclear (safer to help them complete it)
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 2")
            state.q4_scenario = "SCENARIO_2"
            system_prompt = q4_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            # Don't add NO_FOLLOWUPS_INSTRUCTION - we need follow-ups
            nova_response = call_llm(system_prompt, user_message)
    elif question_number == 5:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q5] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        selected_problem = state.selected_problem or "the problem you identified earlier"
        
        # Step 1: Classify the scenario (only on first response to Q5)
        if state.q5_scenario is None:
            q5_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q5_prompts, dict) or "classifier" not in q5_prompts:
                return jsonify({"success": False, "error": "Q5 prompts not configured correctly"}), 400
            
            classifier_prompt = q5_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q5...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q5_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q5_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q5_prompts = SYSTEM_PROMPTS.get(question_number, {})
        if "SCENARIO_1" in (state.q5_scenario or ""):
            print(f"[DEBUG] Q5 SCENARIO 1 detected - reflecting reasoning back")
            system_prompt = q5_prompts.get("scenario_1_respond", "")
            system_prompt = system_prompt.replace("{name}", state.name).replace("{Selected_problem}", selected_problem)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in (state.q5_scenario or ""):
            print(f"[DEBUG] Q5 SCENARIO 2 detected - helping user identify corner piece + reason")
            system_prompt = q5_prompts.get("scenario_2_respond", "")
            system_prompt = system_prompt.replace("{name}", state.name).replace("{Selected_problem}", selected_problem)
            nova_response = call_llm(system_prompt, user_message)
        else:
            print(f"[DEBUG] Q5 classification unclear, defaulting to Scenario 2 guidance")
            state.q5_scenario = "SCENARIO_2"
            system_prompt = q5_prompts.get("scenario_2_respond", "")
            system_prompt = system_prompt.replace("{name}", state.name).replace("{Selected_problem}", selected_problem)
            nova_response = call_llm(system_prompt, user_message)
    elif question_number == 6:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q6] User's message: {user_message}")
        print(f"{'#'*80}\n")
        nova_response = Q6_CANNED_RESPONSE
    else:
        # WEEK2_TEMP: For other questions (should not happen in Week 2, but handle gracefully)
        system_prompt = SYSTEM_PROMPTS.get(question_number, "")
        if not system_prompt:
            return jsonify({"success": False, "error": f"System prompt not found for question {question_number}"}), 400
        
        # Handle both string and dictionary prompts
        if isinstance(system_prompt, dict):
            # If it's a dictionary, use the first scenario response as default
            system_prompt = system_prompt.get("scenario_1_respond", "") or system_prompt.get(list(system_prompt.keys())[0], "")
        
        # Format system prompt with name
        system_prompt = system_prompt.replace("{name}", state.name)
        nova_response = call_llm(system_prompt, user_message)
    
    # DEBUG: Print NOVA's response
    if question_number == 1:
        print(f"[DEBUG PROCESS_RESPONSE Q1] NOVA's response: {nova_response}")
        print(f"{'#'*80}\n")
    elif question_number == 2:
        print(f"[DEBUG PROCESS_RESPONSE Q2] NOVA's response: {nova_response}")
        print(f"{'#'*80}\n")
    elif question_number == 3:
        print(f"[DEBUG PROCESS_RESPONSE Q3] NOVA's response: {nova_response}")
        print(f"{'#'*80}\n")
    elif question_number == 4:
        print(f"[DEBUG PROCESS_RESPONSE Q4] NOVA's response: {nova_response}")
        print(f"{'#'*80}\n")
    elif question_number == 5:
        print(f"[DEBUG PROCESS_RESPONSE Q5] NOVA's response: {nova_response}")
        print(f"{'#'*80}\n")
    elif question_number == 6:
        print(f"[DEBUG PROCESS_RESPONSE Q6] NOVA's response: {nova_response}")
        print(f"{'#'*80}\n")
    
    # Store NOVA response
    if question_number not in state.nova_responses:
        state.nova_responses[question_number] = []
    state.nova_responses[question_number].append(nova_response)
    
    # Increment iteration
    state.iteration_count[question_number] = state.iteration_count.get(question_number, 0) + 1
    iteration = state.get_iteration(question_number)
    
    # Build conversation history for validation
    conversation_history = []
    if question_number in state.nova_responses:
        conversation_history.extend(state.nova_responses[question_number])
    if question_number in state.answers:
        conversation_history.extend(state.answers[question_number])
    
    # Use LLM-based completeness validation
    # Pass scenario parameters for each question
    is_complete, missing_items = validate_completeness(
        question_number, 
        nova_response, 
        state.answers.get(question_number, []),
        conversation_history,
        q1_scenario=state.q1_scenario if question_number == 1 else None,
        q2_scenario=state.q2_scenario if question_number == 2 else None,
        q3_scenario=state.q3_scenario if question_number == 3 else None,
        q4_scenario=state.q4_scenario if question_number == 4 else None,
        q5_scenario=state.q5_scenario if question_number == 5 else None
    )
    
    # Check if we need another iteration
    needs_followup = False
    followup_question = None
    move_to_next = False
    
    # Check if NOVA's response contains a question (indicating follow-up needed)
    is_followup_question = "?" in nova_response
    
    # Continue loop if:
    # 1. Information is not complete (based on LLM validation) AND it's Q2 (homework), OR
    # 2. NOVA is asking a follow-up question AND we haven't exceeded max iterations
    max_iterations = 5 if question_number == 2 else 3  # More iterations for Q2 (homework)
    
    # For Q1, only use validation if NOVA is not asking follow-ups
    # For Q2, always use validation
    if question_number == 1:
        # Q1: Logic based on scenario
        print(f"[DEBUG] Q1: Current iteration={iteration}, is_complete={is_complete}, scenario={state.q1_scenario}, current_question={state.current_question}")
        print(f"[DEBUG] Q1: User responses: {state.answers.get(question_number, [])}")
        
        # For Scenario 2 (confusion or non-meaningful response), provide summary and move on immediately
        if state.q1_scenario and "SCENARIO_2" in state.q1_scenario:
            # Scenario 2: NOVA provided comprehensive summary, now move on
            print(f"[DEBUG] Q1 Scenario 2: Summary provided - moving to next question (no follow-ups)")
            move_to_next = True
            state.question_completed[question_number] = True
            next_question = question_number + 1
            state.current_question = next_question
            needs_followup = False
            # Mark session as modified to ensure cookie is sent
            session.modified = True
            # Save progress
            session_id = session.get('session_id')
            if session_id:
                progress_tracker.update_user_progress(session_id, 2, question_number, completed=True)  # WEEK2_TEMP
            print(f"[DEBUG] Q1 Scenario 2: ✓ Updated current_question from {question_number} to {next_question}")
        # CRITICAL: After 2 iterations (which means user has responded twice), always advance
        elif iteration >= 2:
            # After 2 iterations, move on regardless of validation
            print(f"[DEBUG] Q1: ✓ Max iterations reached ({iteration}) - FORCING advance to next question")
            move_to_next = True
            state.question_completed[question_number] = True
            next_question = question_number + 1
            state.current_question = next_question
            needs_followup = False
            # Mark session as modified to ensure cookie is sent
            session.modified = True
            # Save progress
            session_id = session.get('session_id')
            if session_id:
                progress_tracker.update_user_progress(session_id, 2, question_number, completed=True)  # WEEK2_TEMP
            print(f"[DEBUG] Q1: ✓ Updated current_question from {question_number} to {next_question}")
            print(f"[DEBUG] Q1: ✓ Next question exists in QUESTIONS: {next_question in QUESTIONS}")
            if next_question in QUESTIONS:
                print(f"[DEBUG] Q1: ✓ Next question text: {QUESTIONS[next_question][:50]}...")
        elif is_complete:
            # If validation says complete, move on (even after just 1 iteration)
            print(f"[DEBUG] Q1: ✓ Validation complete after {iteration} iterations - moving to next question")
            move_to_next = True
            state.question_completed[question_number] = True
            next_question = question_number + 1
            state.current_question = next_question
            needs_followup = False
            # Mark session as modified to ensure cookie is sent
            session.modified = True
            # Save progress
            session_id = session.get('session_id')
            if session_id:
                progress_tracker.update_user_progress(session_id, 2, question_number, completed=True)  # WEEK2_TEMP
            print(f"[DEBUG] Q1: ✓ Updated current_question from {question_number} to {next_question}")
            print(f"[DEBUG] Q1: ✓ Next question exists in QUESTIONS: {next_question in QUESTIONS}")
            if next_question in QUESTIONS:
                print(f"[DEBUG] Q1: ✓ Next question text: {QUESTIONS[next_question][:50]}...")
        else:
            # Continue with follow-up only if we haven't reached 2 iterations yet
            print(f"[DEBUG] Q1: Continuing loop (iteration {iteration}, needs follow-up)")
            needs_followup = True
            move_to_next = False
            # Ensure we don't get stuck - if we somehow have many responses but low iteration count, force advance
            if len(state.answers.get(question_number, [])) >= 3:
                print(f"[DEBUG] Q1: WARNING - Too many responses ({len(state.answers.get(question_number, []))}) but iteration is {iteration}, forcing advance")
                move_to_next = True
                state.question_completed[question_number] = True
                next_question = question_number + 1
                state.current_question = next_question
                needs_followup = False
                # Mark session as modified to ensure cookie is sent
                session.modified = True
                print(f"[DEBUG] Q1: ✓ Updated current_question from {question_number} to {next_question}")
                print(f"[DEBUG] Q1: ✓ Next question exists in QUESTIONS: {next_question in QUESTIONS}")
    elif question_number == 2:
        # Q2: Hardcoded logic based on scenario classification
        if state.q2_scenario and "SCENARIO_2" in state.q2_scenario:
            # Scenario 2: User confused, no takeaway, OR non-meaningful response
            # NOVA provided comprehensive summary, now move on immediately
            print(f"[DEBUG] Q2 Scenario 2: Summary provided - moving to next question (no follow-ups)")
            move_to_next = True
            state.question_completed[question_number] = True
            next_question = question_number + 1
            state.current_question = next_question
            needs_followup = False
            # Mark session as modified to ensure cookie is sent
            session.modified = True
            # Save progress
            session_id = session.get('session_id')
            if session_id:
                progress_tracker.update_user_progress(session_id, 2, question_number, completed=True)  # WEEK2_TEMP
            print(f"[DEBUG] Q2 Scenario 2: ✓ Updated current_question from {question_number} to {next_question}")
        elif state.q2_scenario and "SCENARIO_1" in state.q2_scenario:
            # Scenario 1: User provided main takeaway - loop to verify all 4 homework questions are answered
            print(f"[DEBUG] SCENARIO 1: Validating homework completeness...")
            
            if not is_complete:
                # Homework incomplete - continue loop to get all answers
                needs_followup = True
                move_to_next = False
                if missing_items and missing_items != "None" and missing_items != "Unknown":
                    nova_response += f"\n\n[Note: I still need: {missing_items}]"
                print(f"[DEBUG] Homework incomplete - continuing loop")
            else:
                # All homework questions answered - complete
                needs_followup = False
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
                print(f"[DEBUG] All homework complete - moving to next question")
            
        else:
            # Fallback: should not reach here, but handle gracefully
            print(f"[DEBUG] WARNING: Q2 scenario not set, defaulting to complete")
            needs_followup = False
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
    elif question_number == 3:
        # Q3: Hardcoded logic based on scenario classification
        # Check validation - if incomplete, ask for follow-up
        print(f"[DEBUG] Q3: Validating response completeness...")
        
        if not is_complete:
            # Response incomplete - ask for follow-up
            needs_followup = True
            move_to_next = False
            if missing_items and missing_items != "None" and missing_items != "Unknown":
                nova_response += f"\n\n[Note: I still need: {missing_items}]"
            print(f"[DEBUG] Q3 incomplete - continuing to get response")
        else:
            # Response complete - move on
            print(f"[DEBUG] Q3: Completing question and moving on")
            needs_followup = False
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
    elif question_number == 4:
        # Q4: Hardcoded logic based on scenario classification
        print(f"[DEBUG] Q4: Validating problem selection...")
        print(f"[DEBUG] Q4: Scenario = {state.q4_scenario}, is_complete = {is_complete}, iteration = {iteration}")
        
        if state.q4_scenario and "SCENARIO_1" in state.q4_scenario:
            # Scenario 1: User did homework and provided problem - complete immediately
            print(f"[DEBUG] Q4 Scenario 1: User did homework - completing immediately")
            needs_followup = False
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
            save_selected_problem_to_state(state)
        elif state.q4_scenario and "SCENARIO_2" in state.q4_scenario:
            # Scenario 2: User didn't do homework - validate they've selected a problem
            # Run 2-3 loops to help them complete the exercise
            max_iterations_scenario2 = 3  # Allow 2-3 loops to help them select a problem
            
            if not is_complete and iteration < max_iterations_scenario2:
                # Problem not selected yet - continue loop to help them complete exercise
                needs_followup = True
                move_to_next = False
                if missing_items and missing_items != "None" and missing_items != "Unknown":
                    nova_response += f"\n\n[Note: I still need: {missing_items}]"
                print(f"[DEBUG] Q4 Scenario 2: Problem not selected yet (iteration {iteration}/{max_iterations_scenario2}) - continuing to help complete exercise")
            elif is_complete:
                # Problem selected - can complete
                print(f"[DEBUG] Q4 Scenario 2: Problem selected - completing question and moving on")
                needs_followup = False
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
                save_selected_problem_to_state(state)
            else:
                # Max iterations reached - force complete (safety measure)
                print(f"[DEBUG] Q4 Scenario 2: Max iterations reached ({iteration}) - forcing completion")
                needs_followup = False
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
        else:
            # No scenario set yet - use generic validation
            if not is_complete:
                needs_followup = True
                move_to_next = False
                if missing_items and missing_items != "None" and missing_items != "Unknown":
                    nova_response += f"\n\n[Note: I still need: {missing_items}]"
                print(f"[DEBUG] Q4: Response incomplete - continuing")
            else:
                print(f"[DEBUG] Q4: Response complete - completing question and moving on")
                needs_followup = False
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
    elif question_number == 5:
        # Q5: Validate that the user shared the corner piece AND why it matters
        print(f"[DEBUG] Q5: Validating corner piece selection...")
        max_iterations_q5 = 3
        
        if state.q5_scenario and "SCENARIO_1" in state.q5_scenario:
            print(f"[DEBUG] Q5 Scenario 1: User clearly explained corner piece - completing immediately")
            needs_followup = False
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
            save_selected_corner_piece_to_state(state)
        else:
            if not is_complete and iteration < max_iterations_q5:
                needs_followup = True
                move_to_next = False
                if missing_items and missing_items not in ("None", "Unknown"):
                    nova_response += f"\n\n[Note: I still need: {missing_items}]"
                print(f"[DEBUG] Q5 Scenario 2: Missing info (iteration {iteration}/{max_iterations_q5}) - continuing")
            elif is_complete:
                print(f"[DEBUG] Q5: Corner piece + reason captured - moving on")
                needs_followup = False
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
                save_selected_corner_piece_to_state(state)
            else:
                print(f"[DEBUG] Q5: Max iterations reached ({iteration}) - forcing completion")
                needs_followup = False
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
    elif question_number == 6:
        # Q6: Always complete after delivering the canned encouragement
        print(f"[DEBUG] Q6: Providing closing encouragement and wrapping up")
        needs_followup = False
        move_to_next = True
        state.question_completed[question_number] = True
        state.current_question = question_number + 1
    else:
        # Other questions: rely on iteration limit and follow-up detection
        if is_followup_question and iteration < max_iterations:
            needs_followup = True
            move_to_next = False
        else:
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
    
    # CRITICAL: Mark session as modified to ensure cookie is sent back to client
    session.modified = True
    
    # Save progress when question is completed
    if move_to_next and question_number in state.question_completed:
        save_question_progress(question_number, week_number=3)
        
        # Check if all questions are completed (week is done)
        all_completed = all(state.question_completed.get(q, False) for q in QUESTIONS.keys())
        if all_completed:
            session_id = session.get('session_id')
            if session_id:
                progress_tracker.update_user_progress(
                    session_id, 
                    week_number=2,  # WEEK2_TEMP
                    question_number=len(QUESTIONS),
                    week_completed=True
                )
    
    print(f"[DEBUG PROCESS_RESPONSE END] Question {question_number}, iteration={iteration}, move_to_next={move_to_next}, needs_followup={needs_followup}")
    print(f"[DEBUG PROCESS_RESPONSE END] Final state.current_question={state.current_question}, session_id={session.get('session_id')}")
    print(f"[DEBUG PROCESS_RESPONSE END] Answers: {list(state.answers.keys())}, Completed: {list(state.question_completed.keys())}")
    print(f"[DEBUG PROCESS_RESPONSE END] Session modified: {session.modified}")
    
    # If moving to next question, verify it exists
    if move_to_next and state.current_question is not None:
        next_q = state.current_question
        if next_q in QUESTIONS:
            print(f"[DEBUG PROCESS_RESPONSE END] ✓ Next question ({next_q}) exists: {QUESTIONS[next_q][:50]}...")
        else:
            print(f"[ERROR PROCESS_RESPONSE END] ✗ Next question ({next_q}) does NOT exist in QUESTIONS!")
            # Week is complete
            all_completed = all(state.question_completed.get(q, False) for q in QUESTIONS.keys())
            if all_completed:
                print(f"[DEBUG PROCESS_RESPONSE END] ✓ All questions completed - Week 2 is done!  # WEEK2_TEMP")
    
    print(f"{'='*80}\n")
    
    return jsonify({
        "success": True,
        "response": nova_response,
        "needs_followup": needs_followup,
        "followup_question": followup_question,
        "move_to_next": move_to_next,
        "iteration": iteration,
        "next_question": state.current_question if move_to_next else None,  # Include next question number for debugging
        "week_completed": all(state.question_completed.get(q, False) for q in QUESTIONS.keys()) if move_to_next and state.current_question not in QUESTIONS else False
    })


def kill_process_on_port(port):
    """Kill any process running on the specified port."""
    try:
        # For macOS and Linux
        if sys.platform in ['darwin', 'linux']:
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                # First, try graceful shutdown (SIGTERM)
                for pid in pids:
                    if pid and pid.strip():
                        try:
                            print(f"Found process {pid} on port {port}. Stopping it gracefully...")
                            subprocess.run(['kill', '-TERM', pid.strip()], 
                                         capture_output=True, timeout=2)
                            import time
                            time.sleep(0.5)
                        except:
                            pass
                
                # Check if any processes are still running
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    # Force kill remaining processes
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid and pid.strip():
                            print(f"Force stopping process {pid}...")
                            subprocess.run(['kill', '-9', pid.strip()], 
                                         capture_output=True)
                            print(f"Process {pid} stopped.")
                    return True
                return True
        # For Windows (if needed in future)
        elif sys.platform == 'win32':
            result = subprocess.run(['netstat', '-ano'], 
                                  capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) > 0:
                        pid = parts[-1]
                        print(f"Found process {pid} on port {port}. Stopping it...")
                        subprocess.run(['taskkill', '/F', '/PID', pid], 
                                     capture_output=True)
                        print(f"Process {pid} stopped.")
                        return True
    except Exception as e:
        print(f"Warning: Could not check for processes on port {port}: {e}")
    return False


if __name__ == '__main__':
    print("Starting NOVA Career Coach Week 2 Server (via repurposed file)...  # WEEK2_TEMP")
    print(f"Loaded {len(QUESTIONS)} Week 2 questions  # WEEK2_TEMP")
    
    # WEEK2_TEMP: Use port 5002 for Week 2 (Week 1 uses 5001)
    port = int(os.getenv('PORT', 5002))
    
    # Check and kill any existing process on the port
    print(f"Checking port {port}...")
    if kill_process_on_port(port):
        import time
        time.sleep(1)  # Give it a moment to fully stop
    
    print(f"Server starting on http://localhost:{port}")
    print("Test scenarios available:")
    for qnum, question in QUESTIONS.items():
        print(f"  Question {qnum}: {question}")
    # Set use_reloader=False to avoid multiprocessing warnings when killing processes
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

