"""
Week 5 Main Application

This file contains the complete Week 5 implementation for the DRIVEN program.
It includes:
- Week 5 questions
- Week 5 system prompts
- Validation logic
- All conversation flow logic

All Week 5 content is embedded here.
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
        self.q1_scenario = None  # Store Q1 scenario classification
        self.q2_scenario = None  # Store Q2 scenario classification
        self.q3_scenario = None  # Store Q3 scenario classification
        self.q4_scenario = None  # Store Q4 scenario classification
        self.q5_scenario = None  # Store Q5 scenario classification
    
    def get_iteration(self, qnum):
        """Get current iteration count for a question."""
        return self.iteration_count.get(qnum, 0)


def get_or_create_state(name=None):
    """Get or create a conversation state using Flask session."""
    session_id = session.get('session_id')
    if session_id is None:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    if session_id not in conversation_states:
        if name is None:
            # Try to get name from existing state or use default
            name = "Friend"
        conversation_states[session_id] = ConversationState(name)
    return conversation_states[session_id]


def save_question_progress(question_number: int, week_number: int = 5):
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
        # Log the full error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"\n{'='*80}")
        print(f"ERROR in call_llm:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Full traceback:\n{error_details}")
        print(f"{'='*80}\n")
        return f"I apologize, but I encountered an error processing your response. Please try again. Error: {str(e)}"


def validate_completeness(question_number, nova_response, user_responses, conversation_history):
    """
    Use LLM to validate if user has provided all required information.
    Returns (is_complete: bool, missing_items: str)
    """
    # Build context of what was requested
    context = f"Question {question_number}: {QUESTIONS.get(question_number, '')}\n\n"
    context += f"NOVA's latest response: {nova_response}\n\n"
    context += f"User's responses so far: {len(user_responses)} response(s)\n"
    for i, resp in enumerate(user_responses, 1):
        context += f"  Response {i}: {resp[:200]}...\n"

    # Generic validation for Week 5 questions
    validation_prompt = f"""You are validating if a user has provided a meaningful response to the question.

The user has provided at least one response. Determine if they have given a meaningful answer.
- If they provided any relevant response related to the question → COMPLETE
- If they gave vague or unrelated responses → INCOMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided meaningful response: "COMPLETE: Yes\nMISSING: None"
- If no meaningful response provided: "COMPLETE: No\nMISSING: A meaningful response to the question"
"""
    
    try:
        validation_response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": validation_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.3
        )
        
        validation_text = validation_response.choices[0].message.content.strip()
        
        # Parse the response
        is_complete = False
        missing_items = "Unknown"
        
        if "COMPLETE: Yes" in validation_text or "COMPLETE:YES" in validation_text.upper():
            is_complete = True
        elif "COMPLETE: No" in validation_text or "COMPLETE:NO" in validation_text.upper():
            is_complete = False
        
        # Extract missing items
        if "MISSING:" in validation_text:
            missing_match = validation_text.split("MISSING:")[-1].strip()
            missing_items = missing_match if missing_match else "Unknown"
        
        return is_complete, missing_items
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False, "Validation error occurred"


# Global instruction used to ensure no extra questions once a question is complete
NO_FOLLOWUPS_INSTRUCTION = (
    "Important: If the user's response already satisfies the criteria for this question and it is considered complete (or the classified scenario instructs to move on), do not ask any additional questions. Conclude your reply without further questions so we can proceed to the next question."
)

# Week 5 content data
WEEK5_VIDEOS_EXERCISES = """The fifth week of DRIVEN is about how to identify and contact people who are in a position to help you in your job search, which is what we commonly refer to as networking. The videos also covered how to prepare and then put your best foot forward in an interview situation and then how to effectively negotiate a job offer."""

WEEK5_EXERCISE1 = """Exercise 1 focused on networking and identifying potential job search contacts. The exercise asked you to jot down potential job search contacts and/or places you might meet people who could help you, including friends, family members, acquaintances, prior managers, prior coworkers, school contacts, and organizations. For each entry, you were asked to write a deadline for reaching out."""

WEEK5_EXERCISE2 = """Exercise 2 focused on preparing for interviews. The exercise asked you to practice answering common interview questions such as: Tell me about yourself, What are your greatest weaknesses?, Why do you want to work here?, Where do you see yourself in five years?, What kind of work environment do you like best?, Tell me about the last time a customer or coworker got angry at you, and What was the biggest challenge you had in your last position. You were asked to write 2-3 sentence responses to each question and practice saying them out loud."""

# Questions from week5.py
QUESTIONS = {
    1: "What was one thing you remember about the fifth week's content on networking and interviewing?",
    2: "Were you able to jot down some brief responses to each of the common interview questions?",
    3: "Which question was the most challenging for you?",
    4: "Have you made any progress on the job search goal we talked about at the beginning of the program, or any of the job seeking activities we talked about in the last few sessions?",
    5: "What concerns do you have about your job search moving forward?"
}

# System prompts from week5.py
SYSTEM_PROMPTS = {
    1: {
        "classifier": """Based on the user's response to "What was one thing you remember about the fifth week's content on networking and interviewing?", determine which scenario applies:

Scenario 1: User explains a takeaway relevant to the material (e.g., mentions networking, interviewing, job search contacts, interview preparation, etc.).
Scenario 2: User did not complete exercises for the week (e.g., "I didn't do it", "I haven't completed the exercises", "I didn't watch the videos").
Scenario 3: User's response is irrelevant to course material (e.g., completely unrelated response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. The user explains a takeaway relevant to Week 5's material.

Praise user for completing the exercise and expound upon the point they've brought up based on the week 5 exercise and video content:

{WEEK5_VIDEOS_EXERCISES}
{WEEK5_EXERCISE1}
{WEEK5_EXERCISE2}

Be warm and encouraging.""",
        "scenario_2_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user did not complete exercises for the week.

Tell user to go back and finish exercises, then return and answer the question. Repeat the question again. Be encouraging and supportive.""",
        "scenario_3_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user's response is irrelevant to course material.

Remind user to stay on topic and repeat the question. Be patient and understanding."""
    },
    2: {
        "classifier": """Based on the user's response to "Were you able to jot down some brief responses to each of the common interview questions?", determine which scenario applies:

Scenario 1: User completed the exercise and has no questions (e.g., "Yes", "I did it", "I completed it", "No questions").
Scenario 2: User has a question about one of the interview questions and did not complete it (e.g., mentions being stuck on a specific question, asks for help with a question, expresses confusion about how to answer).
Scenario 3: User's response is irrelevant to course material (e.g., completely unrelated response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user completed the exercise and has no questions.

Praise the user for completing the exercise and move on. Do not ask any follow up questions in this case. Be warm and encouraging.""",
        "scenario_2_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. The user has a question about one of the interview questions and did not complete it.

Based on video and exercise content, address the user's question and advise them on how to answer. After this, follow up and ask if they have any other questions, and then address those in the same manner. Repeat until the user says they have no more questions.

{WEEK5_VIDEOS_EXERCISES}
{WEEK5_EXERCISE2}

Be supportive and helpful.""",
        "scenario_3_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user's response is irrelevant to course material.

Remind the user to stay on topic and repeat the question. Be patient."""
    },
    3: {
        "classifier": """Based on the user's response to "Which question was the most challenging for you?", determine which scenario applies:

Scenario 1: User points to a particular exercise question they struggled with (e.g., mentions a specific interview question, says which one was hard, identifies a particular challenge).
Scenario 2: User did not have any questions or challenges (e.g., "None", "I didn't find any challenging", "They were all easy", "No questions").
Scenario 3: User's response is irrelevant to course material (e.g., completely unrelated response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. The user points to a particular exercise question they struggled with.

Based on video and exercise content, address the user's concern and provide advice on how to answer that specific question.

{WEEK5_VIDEOS_EXERCISES}
{WEEK5_EXERCISE2}

Be supportive and helpful.""",
        "scenario_2_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user did not have any questions or challenges.

Praise the user for being confident and remind them they can consult the DRIVEN videos and exercises if they have questions. Be warm and encouraging.""",
        "scenario_3_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user's response is irrelevant to course material.

Remind user to stay on topic and repeat the question. Be patient."""
    },
    4: {
        "classifier": """Based on the user's response to "Have you made any progress on the job search goal we talked about at the beginning of the program, or any of the job seeking activities we talked about in the last few sessions?", determine which scenario applies:

Scenario 1: User made progress and references a specific goal or activity in their response (e.g., mentions a specific goal, activity, or step they took).
Scenario 2: User states they made progress but does not reference a specific goal or activity in their response (e.g., "Yes I made progress" without details).
Scenario 3: User did not make progress (e.g., "No", "Not really", "I haven't", "Nothing yet").
Scenario 4: User's response is irrelevant to course material (e.g., completely unrelated response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", "SCENARIO_3", or "SCENARIO_4".""",
        "scenario_1_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user made progress and references a specific goal or activity in their response.

Praise user for making progress in that area and provide positive reinforcement for their future endeavors. Do not ask any follow up questions after this, and move onto the next question. Be warm and encouraging.""",
        "scenario_2_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user states they made progress but does not reference a specific goal or activity in their response.

Praise the user for making progress and clarify that NOVA needs to know the specific goal or activity they have addressed. Be supportive and encouraging.""",
        "scenario_3_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. The user did not make progress.

Respond empathetically, reminding user that making progress does not always happen overnight, and reinforce the ideas of taking little steps and thinking flexibly to eventually reach their career goals.

{WEEK5_VIDEOS_EXERCISES}

Be empathetic and supportive.""",
        "scenario_4_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user's response is irrelevant to course material.

Remind user to stay on topic and repeat the question. Be patient."""
    },
    5: {
        "classifier": """Based on the user's response to "What concerns do you have about your job search moving forward?", determine which scenario applies:

Scenario 1: User has no concerns about their job search moving forward (e.g., "None", "No concerns", "I'm confident", "I feel good about it").
Scenario 2: User does have concerns about their job search going forward (e.g., mentions specific worries, fears, or concerns).
Scenario 3: User's response is irrelevant to course material (e.g., completely unrelated response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user has no concerns about their job search moving forward.

Praise the user for their confidence and ability to use their flexible thinking skills to get to a good place in their job search. Remind them that, if any issues do arise and they want to consult DRIVEN resources, they still have access to their account for six months. Be warm and encouraging.""",
        "scenario_2_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. The user does have concerns about their job search going forward.

Address the user's concerns using the week 5 video and exercise material, responding empathetically. Do not ask any follow up questions after addressing user's concerns.

{WEEK5_VIDEOS_EXERCISES}
{WEEK5_EXERCISE1}
{WEEK5_EXERCISE2}

Be empathetic and supportive.""",
        "scenario_3_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user's response is irrelevant to course material.

Remind user to stay on topic and repeat the question. Be patient."""
    }
}

WELCOME_MESSAGE = """You're almost at the end of the DRIVEN program - just one week to go! 

Also, thanks again for all of the incredible effort you've been putting toward your job search. This isn't an easy process and your effort towards finishing this program has been amazing!

In today's session, we will:

💡 Review the fifth weeks' ideas and guidance.

❓ Review your exercises on answering common interview questions from Week 5. 

🗣️ Talk about the status of your job search today, and the most important issue you're facing going into your last week of DRIVEN.

👣 Plan the next steps so that you can keep practicing what you've learned once you've completed the program.

Let's get started!

The fifth week of DRIVEN is about how to identify and contact people who are in a position to help you in your job search, which is what we commonly refer to as networking. The videos also covered how to prepare and then put your best foot forward in an interview situation and then how to effectively negotiate a job offer."""

FINAL_RESPONSE = """Great work today! Looking ahead, next week is your last week of the program! I'll be in touch to talk about how you can continue to use your new skills as the DRIVEN program draws to a close.
Talk to you soon!"""


@app.route('/')
def index():
    """Serve the index.html file."""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return Response(f.read(), mimetype='text/html')
    except FileNotFoundError:
        return "index.html not found. Please make sure the file exists in the same directory as week5_main.py.", 404


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
    week_number = data.get('week_number', 5)
    
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
    week_number = 5
    
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
    
    # If we've completed all questions, return completion message
    if state.current_question is None or state.current_question > len(QUESTIONS):
        # Mark week as completed in progress tracker
        session_id = session.get('session_id')
        if session_id:
            progress_tracker.update_user_progress(
                session_id, 
                week_number=5,
                question_number=len(QUESTIONS),
                week_completed=True
            )
        
        return jsonify({
            "success": True,
            "message": FINAL_RESPONSE,
            "is_complete": True,
            "awaiting_response": False,
            "week_completed": True
        })
    
    # Return the current question
    qnum = state.current_question
    question_text = QUESTIONS[qnum]
    
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
    
    # Use current question if not provided
    if question_number is None:
        question_number = state.current_question
    
    if question_number is None or question_number not in QUESTIONS:
        return jsonify({"success": False, "error": "No active question"}), 400
    
    # Store user response
    if question_number not in state.answers:
        state.answers[question_number] = []
    state.answers[question_number].append(user_message)
    
    # Get current iteration before incrementing
    iteration = state.get_iteration(question_number)
    
    # Process each question with scenario classification
    nova_response = None
    
    # For all questions, classify scenario first, then use appropriate prompt
    if question_number in [1, 2, 3, 4, 5]:
        q_prompts = SYSTEM_PROMPTS.get(question_number, {})
        if not isinstance(q_prompts, dict) or "classifier" not in q_prompts:
            return jsonify({"success": False, "error": f"Q{question_number} prompts not configured correctly"}), 400
        
        # Classify scenario (only on first response)
        scenario_attr = f"q{question_number}_scenario"
        if not hasattr(state, scenario_attr) or getattr(state, scenario_attr) is None:
            classifier_prompt = q_prompts["classifier"]
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            scenario = classification_response.choices[0].message.content.strip().upper()
            setattr(state, scenario_attr, scenario)
            print(f"[DEBUG] Q{question_number} scenario classification: {scenario}")
        
        scenario = getattr(state, scenario_attr)
        
        # Get the appropriate response prompt based on scenario
        # Extract scenario number (e.g., "SCENARIO_1" -> "1")
        scenario_num = scenario.split('_')[1] if '_' in scenario else scenario.replace('SCENARIO', '').strip()
        scenario_key = f"scenario_{scenario_num.lower()}_respond"
        
        # Handle special cases for Q4 (which has SCENARIO_4)
        if question_number == 4:
            if "SCENARIO_1" in scenario:
                scenario_key = "scenario_1_respond"
            elif "SCENARIO_2" in scenario:
                scenario_key = "scenario_2_respond"
            elif "SCENARIO_3" in scenario:
                scenario_key = "scenario_3_respond"
            elif "SCENARIO_4" in scenario:
                scenario_key = "scenario_4_respond"
        
        system_prompt = q_prompts.get(scenario_key, "")
        if not system_prompt:
            # Fallback to first available scenario
            for key in q_prompts.keys():
                if key.startswith("scenario_") and key.endswith("_respond"):
                    system_prompt = q_prompts[key]
                    break
        
        if not system_prompt:
            return jsonify({"success": False, "error": f"System prompt not found for Q{question_number} scenario {scenario}"}), 400
        
        # Replace placeholders
        system_prompt = system_prompt.replace("{name}", state.name)
        
        # Replace WEEK5 placeholders
        if "{Week5_Videos+Exercises}" in system_prompt:
            system_prompt = system_prompt.replace("{Week5_Videos+Exercises}", WEEK5_VIDEOS_EXERCISES)
        if "{Week5_Videos+Exercise 1}" in system_prompt:
            system_prompt = system_prompt.replace("{Week5_Videos+Exercise 1}", f"{WEEK5_VIDEOS_EXERCISES}\n\n{WEEK5_EXERCISE1}")
        if "{Week5_Videos+Exercise 2}" in system_prompt:
            system_prompt = system_prompt.replace("{Week5_Videos+Exercise 2}", f"{WEEK5_VIDEOS_EXERCISES}\n\n{WEEK5_EXERCISE2}")
        if "{WEEK5_EXERCISES}" in system_prompt:
            system_prompt = system_prompt.replace("{WEEK5_EXERCISES}", f"{WEEK5_VIDEOS_EXERCISES}\n\n{WEEK5_EXERCISE1}\n\n{WEEK5_EXERCISE2}")
        
        system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
        nova_response = call_llm(system_prompt, user_message)
        
        # For Q4 and Q5 Scenario 3 (irrelevant responses), append the question text to repeat it
        if question_number in [4, 5] and scenario and "SCENARIO_3" in scenario:
            question_text = QUESTIONS.get(question_number, "")
            if question_text:
                nova_response += f"\n\n{question_text}"
    else:
        # Fallback for any other questions
        system_prompt = SYSTEM_PROMPTS.get(question_number, "")
        if not system_prompt:
            return jsonify({"success": False, "error": "System prompt not found"}), 400
        system_prompt = system_prompt.replace("{name}", state.name)
        nova_response = call_llm(system_prompt, user_message)
    
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
    is_complete, missing_items = validate_completeness(
        question_number, 
        nova_response, 
        state.answers.get(question_number, []),
        conversation_history
    )
    
    # Check if we need another iteration
    needs_followup = False
    followup_question = None
    move_to_next = False
    
    # Check if NOVA's response contains a question (indicating follow-up needed)
    is_followup_question = "?" in nova_response
    
    max_iterations = 3
    
    # Special handling for Q2 (may need multiple iterations if user has questions)
    if question_number == 2:
        if state.q2_scenario and "SCENARIO_2" in state.q2_scenario:
            # User has questions - continue until they say no more questions
            if is_followup_question and iteration < 5:  # Allow more iterations for Q2
                needs_followup = True
                move_to_next = False
            else:
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
        else:
            # Scenario 1 or 3 - standard flow
            if is_complete or iteration >= max_iterations:
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
            elif not is_complete and iteration < max_iterations:
                needs_followup = True
                move_to_next = False
                if missing_items and missing_items != "None" and missing_items != "Unknown":
                    nova_response += f"\n\n[Note: I still need: {missing_items}]"
    elif question_number in [4, 5]:
        # Special handling for Q4 and Q5 Scenario 3 (irrelevant responses) - must repeat same question
        scenario_attr = f"q{question_number}_scenario"
        scenario = getattr(state, scenario_attr, None)
        if scenario and "SCENARIO_3" in scenario:
            # For irrelevant responses, stay on the same question and repeat it
            # Allow up to 3 tries for irrelevant responses
            if iteration < 3:
                needs_followup = True
                move_to_next = False
                # Don't mark as complete - stay on same question
            else:
                # After 3 tries, move on even if irrelevant
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
        else:
            # Not Scenario 3 - use standard logic
            if not is_complete and iteration < max_iterations:
                needs_followup = True
                move_to_next = False
                if missing_items and missing_items != "None" and missing_items != "Unknown":
                    nova_response += f"\n\n[Note: I still need: {missing_items}]"
            elif is_complete or iteration >= max_iterations:
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
    else:
        # Determine if we should continue or move to next question
        if not is_complete and iteration < max_iterations:
            needs_followup = True
            move_to_next = False
            if missing_items and missing_items != "None" and missing_items != "Unknown":
                nova_response += f"\n\n[Note: I still need: {missing_items}]"
        elif is_complete or iteration >= max_iterations:
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
    
    # Save progress when question is completed
    if move_to_next and question_number in state.question_completed:
        save_question_progress(question_number, week_number=5)
        
        # Check if all questions are completed (week is done)
        all_completed = all(state.question_completed.get(q, False) for q in QUESTIONS.keys())
        if all_completed:
            session_id = session.get('session_id')
            if session_id:
                progress_tracker.update_user_progress(
                    session_id, 
                    week_number=5,
                    question_number=len(QUESTIONS),
                    week_completed=True
                )
    
    return jsonify({
        "success": True,
        "response": nova_response,
        "needs_followup": needs_followup,
        "followup_question": followup_question,
        "move_to_next": move_to_next,
        "iteration": iteration,
        "week_completed": all(state.question_completed.get(q, False) for q in QUESTIONS.keys()) if move_to_next and (state.current_question is None or state.current_question > len(QUESTIONS)) else False
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
    print("Starting NOVA Career Coach Week 5 Server...")
    print(f"Loaded {len(QUESTIONS)} questions for Week 5")
    
    port = int(os.getenv('PORT', 5005))
    
    # Check and kill any existing process on the port
    print(f"Checking port {port}...")
    if kill_process_on_port(port):
        import time
        time.sleep(1)  # Give it a moment to fully stop
    
    print(f"Server starting on http://localhost:{port}")
    print("Week 5 questions available:")
    for qnum, question in QUESTIONS.items():
        print(f"  Question {qnum}: {question}")
    # Set use_reloader=False to avoid multiprocessing warnings when killing processes
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

