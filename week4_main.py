"""
Week 4 Main Application

This file contains the complete Week 4 implementation for the DRIVEN program.
It includes:
- Week 4 questions
- Week 4 system prompts
- Validation logic
- All conversation flow logic

All Week 4 content is embedded here.
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
        self.q6_scenario = None  # Store Q6 scenario classification
        self.q7_scenario = None  # Store Q7 scenario classification
        self.q7_scenario = None  # Store Q7 scenario classification
    
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


def save_question_progress(question_number: int, week_number: int = 4):
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

    # Generic validation for Week 4 questions
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

# Week 4 content data
WEEK4_VIDEOS_EXERCISES = """Week 4 of DRIVEN talked about the basics you'll need when applying for jobs, namely three components you're already familiar with: the application, the resume, and the cover letter. The videos helped you think flexibly about how your achievements and prior experiences can be useful to employers, and about who in your life might be able to review and provide feedback on your materials."""

WEEK4_EXERCISE1 = """Exercise 1 focused on identifying keywords from job descriptions. Many companies use software called 'applicant tracking systems' to automatically scan resumes for certain keywords. The exercise asked you to list 5-10 words you keep seeing over and over in job descriptions."""

WEEK4_EXERCISE2 = """Exercise 2 focused on asking for help with job materials. People sometimes avoid asking others for help because they are worried that they are burdening the other person or that the other person will let them down. When you're job searching, tapping into your social support network is incredibly valuable. The exercise asked you to identify who you can ask for help with reviewing your resume and cover letter, and to work through any concerns you might have about asking for help."""

# Questions from week4.py
QUESTIONS = {
    1: "What was one main idea you took away?",
    2: "How was your experience completing this exercise?",
    3: "Which individuals did you choose and why?",
    4: "Did you have any trouble coming up with people who might help you?",
    5: "What were some of the concerns you wrote down about asking for help?",
    6: "Were you able to use this exercise to overcome them?",
    7: "Did you end up asking anyone for help? If so, what was the outcome?"
}

# System prompts from week4.py
SYSTEM_PROMPTS = {
    1: {
        "classifier": """Based on the user's response to "What was one main idea you took away?" from Week 4 recap, determine which scenario applies:

Scenario 1: User provides a main takeaway idea (e.g., mentions learning something, understanding a concept, taking away an idea).
Scenario 2: User expresses confusion or does not provide a main takeaway idea (e.g., "I don't know", "I didn't have meaningful takeaways", "I'm confused").
Scenario 3: User does not provide a response that is meaningful to the question asked (e.g., completely unrelated response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user has provided a main takeaway idea from Week 4's material.

Respond by congratulating the user for their engagement with the material. Be warm and encouraging.""",
        "scenario_2_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. The user expresses confusion or did not have meaningful takeaways from Week 4.

Provide a summary of some of the main ideas in Week 4 content:

{WEEK4_VIDEOS_EXERCISES}

Be supportive and help them understand the key concepts.""",
        "scenario_3_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user did not provide a response that is meaningful to the question asked.

Kindly ask the user to reconsider the question and provide any needed clarification. Be patient and understanding."""
    },
    2: {
        "classifier": """Based on the user's response to "How was your experience completing this exercise?", determine which scenario applies:

Scenario 1: User shares a general positive or neutral reflection on their experience (e.g., "I enjoyed it", "It was helpful", "It was okay", "It was fine").
Scenario 2: User gives a negative response (e.g., "I didn't like it", "It was difficult", "It was frustrating").
Scenario 3: User did not complete the exercise or gives a response that is irrelevant (e.g., "I didn't do it", completely unrelated response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. The user shares a general positive or neutral reflection on their experience completing Exercise 1.

Respond with a short summary + reinforcement of Week 4 themes:

{WEEK4_VIDEOS_EXERCISES}
{WEEK4_EXERCISE1}

Be warm and encouraging.""",
        "scenario_2_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. The user gives a negative response about completing Exercise 1.

Provide positive reassurance that addresses the user's concern while emphasizing the importance of the exercises:

{WEEK4_VIDEOS_EXERCISES}

Be empathetic and supportive.""",
        "scenario_3_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user did not complete the exercise or gives a response that is irrelevant.

Prompt the user to go back and watch the videos + do the exercises in Week 4 about applying to jobs, crafting a resume, and writing a cover letter. Be encouraging."""
    },
    3: {
        "classifier": """Based on the user's response to "Which individuals did you choose and why?", determine which scenario applies:

Scenario 1: User did choose an individual and explains why they chose them (e.g., mentions a specific person and reason).
Scenario 2: User did not pick an individual (e.g., "I didn't choose anyone", "I don't know who I could ask").
Scenario 3: User's response is irrelevant (e.g., completely unrelated response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user did choose an individual and explains why they chose them.

Provide positive reassurance and commend user for asking for help. Do not ask more follow up questions about the selected individual after this. Be warm and encouraging.""",
        "scenario_2_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. The user did not pick an individual.

Reinforce the importance of tapping into your social network for support during the job search. Remind user that, for example, they could ask a friend to attend a local job fair with you because they're good at starting conversations. Or they could ask a family member to provide feedback on their resume because they are good at writing. Tell user to try and come up with someone before continuing, but don't ask any follow up questions after this.

{WEEK4_EXERCISE2}

Be supportive and encouraging.""",
        "scenario_3_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user's response is irrelevant.

Repeat the question, and move onto the next one if 2 more tries are unsuccessful. Be patient and understanding."""
    },
    4: {
        "classifier": """Based on the user's response to "Did you have any trouble coming up with people who might help you?", determine which scenario applies:

Scenario 1: User did not have trouble picking a person (e.g., "No", "Not really", "It was easy").
Scenario 2: User did have trouble picking a person (e.g., "Yes", "I had trouble", "It was difficult").
Scenario 3: User's response is irrelevant to the course material (e.g., completely unrelated response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. 

The current question is: "Did you have any trouble coming up with people who might help you?"

The user did not have trouble picking a person.

Provide a short positive response and remind the user that negative feelings around asking for help are normal, but commend them for having the ability to ask. Be warm and supportive. Do NOT mention or reference any other questions.""",
        "scenario_2_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. 

The current question is: "Did you have any trouble coming up with people who might help you?"

The user did have trouble picking a person.

Respond empathetically that you understand asking for help can be daunting in this situation. Again prompt user to try to think of someone they could ask to keep in mind for the future, but don't ask any follow up questions after this.

{WEEK4_EXERCISE2}

Be empathetic and supportive. Do NOT mention or reference any other questions.""",
        "scenario_3_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. 

The current question is: "Did you have any trouble coming up with people who might help you?"

The user's response is irrelevant to the course material.

Remind user to stay on topic and repeat the question. Be patient. Do NOT mention or reference any other questions."""
    },
    5: {
        "classifier": """Based on the user's response to "What were some of the concerns you wrote down about asking for help?", determine which scenario applies:

Scenario 1: User had concerns (e.g., mentions specific worries, fears, or concerns).
Scenario 2: The user did not have any concerns (e.g., "No concerns", "I didn't have any", "Nothing").
Scenario 3: User's response is irrelevant to course material (e.g., completely unrelated response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. The user had concerns about asking for help.

Respond empathetically and address the user's concerns about reaching out to their social network using the videos and exercises provided:

{WEEK4_EXERCISE2}

Be empathetic and supportive.""",
        "scenario_2_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user did not have any concerns.

Commend the user for their positive attitude and reinforce that they can express their concerns to NOVA if they have any. Be warm and encouraging.""",
        "scenario_3_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user's response is irrelevant to course material.

Remind user to stay on topic and repeat the question. Be patient."""
    },
    6: {
        "classifier": """Based on the user's response to "Were you able to use this exercise to overcome them?", determine which scenario applies:

Scenario 1: User did not have any concerns (from Q5) - they're responding that they didn't have concerns to overcome.
Scenario 2: The user did have concerns but overcame them (e.g., "Yes", "I was able to", "It helped").
Scenario 3: The user had concerns but did not feel like they overcame them (e.g., "No", "Not really", "I still feel worried").
Scenario 4: User gives response that is irrelevant (e.g., completely unrelated response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", "SCENARIO_3", or "SCENARIO_4".""",
        "scenario_1_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user did not have any concerns.

Commend user for their courage in asking for help. Be warm and encouraging.""",
        "scenario_2_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user did have concerns but overcame them.

Reference the concern they stated in question 5 and praise them for overcoming that. Do not ask any follow up questions and move onto the next question. Be warm and encouraging.""",
        "scenario_3_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. The user had concerns but did not feel like they overcame them.

Reference the concern stated in question 5 and offer suggestions on how user could overcome, as well as the importance of asking for help. Do not ask follow up questions after this and move onto the next question.

{WEEK4_EXERCISE2}

Be empathetic and supportive.""",
        "scenario_4_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user gives response that is irrelevant.

Repeat the question again, and move on after 2 irrelevant responses. Be patient."""
    },
    7: {
        "classifier": """Based on the user's response to "Did you end up asking anyone for help? If so, what was the outcome?", determine which scenario applies:

Scenario 1: User did not ask for help because of their concerns from question 5 (e.g., mentions their concerns as the reason).
Scenario 2: User did not ask for help because they did not have time (e.g., "I didn't have time", "I was too busy").
Scenario 3: User did ask for help and shares the outcome (e.g., mentions asking someone and what happened).
Scenario 4: User states they did ask for help but does not share the outcome (e.g., "Yes" without details).
Scenario 5: User's response is irrelevant to question (e.g., completely unrelated response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", "SCENARIO_3", "SCENARIO_4", or "SCENARIO_5".""",
        "scenario_1_respond": f"""Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {{name}}. The user did not ask for help because of their concerns from question 5.

Respond empathetically and address the user's concerns about reaching out to their social network using the videos and exercises provided:

{WEEK4_EXERCISE2}

Be empathetic and supportive.""",
        "scenario_2_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user did not ask for help because they did not have time.

Recommend user schedule time on their calendar to reach out this week before they finish their week 4 session. Be encouraging and supportive.""",
        "scenario_3_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user did ask for help and shares the outcome.

Comment on outcome and commend user for their courage in reaching out. Do not ask follow up questions after this and end the session. Be warm and encouraging.""",
        "scenario_4_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user states they did ask for help but does not share the outcome.

Prompt user again to share the outcome. Move on and end session after 2 tries if they do not share additional information. Be patient and encouraging.""",
        "scenario_5_respond": """Imagine you are a trained career coach helping adults with mental health challenges find jobs. The user's name is {name}. The user's response is irrelevant to question.

Repeat the question. Move on and end session after 2 tries if they do not share any additional information. Be patient."""
    }
}

WELCOME_MESSAGE = """Nova here! Congrats on wrapping up Week 4 🥳 Let's chat about what you learned.
In today's session, we will:

💡 Recap the fourth weeks' ideas and guidance.

🔎 Review your exercises on asking for help with job materials from Week 4.
Let's dive in!
The fourth week of DRIVEN talked about the basics you'll need when applying for jobs, namely three components you're already familiar with: the application, the resume, and the cover letter. Hopefully, the videos helped you think flexibly about how your achievements and prior experiences can be useful to employers, and about who in your life might be able to review and provide feedback on your materials."""

FINAL_RESPONSE = """Great work on this exercise. It's not easy to ask people for help, but when you think about the situation flexibly and have the courage to reach out, you'll usually find that both you and the other person are glad you did. 
I'll be in touch again next week. Take care! 👋"""


@app.route('/')
def index():
    """Serve the index.html file."""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return Response(f.read(), mimetype='text/html')
    except FileNotFoundError:
        return "index.html not found. Please make sure the file exists in the same directory as week4_main.py.", 404


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
    week_number = data.get('week_number', 4)
    
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
    week_number = 4
    
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
                week_number=4,
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
    
    # Add transition message for Q2 (now "How was your experience...")
    if qnum == 2:
        transition_message = "Let's now take a look at the exercise from this week: asking for help with job materials."
        question_text = f"{transition_message}\n\n{question_text}"
    
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
    if question_number in [1, 2, 3, 4, 5, 6, 7]:
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
        
        # Handle special cases for Q6 and Q7 (which have SCENARIO_4 and SCENARIO_5)
        if question_number == 6:
            if "SCENARIO_1" in scenario:
                scenario_key = "scenario_1_respond"
            elif "SCENARIO_2" in scenario:
                scenario_key = "scenario_2_respond"
            elif "SCENARIO_3" in scenario:
                scenario_key = "scenario_3_respond"
            elif "SCENARIO_4" in scenario:
                scenario_key = "scenario_4_respond"
        elif question_number == 7:
            if "SCENARIO_1" in scenario:
                scenario_key = "scenario_1_respond"
            elif "SCENARIO_2" in scenario:
                scenario_key = "scenario_2_respond"
            elif "SCENARIO_3" in scenario:
                scenario_key = "scenario_3_respond"
            elif "SCENARIO_4" in scenario:
                scenario_key = "scenario_4_respond"
            elif "SCENARIO_5" in scenario:
                scenario_key = "scenario_5_respond"
        
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
        
        # For Q4, explicitly include the question text in the system prompt to prevent confusion
        if question_number == 4:
            question_text = QUESTIONS.get(question_number, "")
            if question_text and "The current question is:" not in system_prompt:
                # Add question context at the beginning of the prompt
                system_prompt = f"""The current question you are responding to is: "{question_text}"

{system_prompt}"""
        
        # Replace answer placeholders for Q6 and Q7
        if question_number == 6 or question_number == 7:
            # Get Q5 answer (concerns)
            q5_answer = ""
            if 5 in state.answers and state.answers[5]:
                q5_answer = " ".join(state.answers[5])
            if q5_answer:
                system_prompt = system_prompt.replace("{Answer to question 5}", q5_answer)
        
        if question_number == 7:
            # Get Q5 answer (concerns)
            q5_answer = ""
            if 5 in state.answers and state.answers[5]:
                q5_answer = " ".join(state.answers[5])
            if q5_answer:
                system_prompt = system_prompt.replace("{Answer to question 5}", q5_answer)
        
        # Replace WEEK4_EXERCISES placeholder
        if "{Week4_Videos+Exercises}" in system_prompt:
            system_prompt = system_prompt.replace("{Week4_Videos+Exercises}", WEEK4_VIDEOS_EXERCISES)
        if "{Week4_Videos+Exercise 1}" in system_prompt:
            system_prompt = system_prompt.replace("{Week4_Videos+Exercise 1}", f"{WEEK4_VIDEOS_EXERCISES}\n\n{WEEK4_EXERCISE1}")
        if "{Week4_Videos+Exercise 2}" in system_prompt:
            system_prompt = system_prompt.replace("{Week4_Videos+Exercise 2}", f"{WEEK4_VIDEOS_EXERCISES}\n\n{WEEK4_EXERCISE2}")
        if "{WEEK4_EXERCISES}" in system_prompt:
            system_prompt = system_prompt.replace("{WEEK4_EXERCISES}", f"{WEEK4_VIDEOS_EXERCISES}\n\n{WEEK4_EXERCISE2}")
        
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
    
    # Special handling for Q4 and Q5 Scenario 3 (irrelevant responses) - must repeat same question
    if question_number in [4, 5]:
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
        # For other questions, use standard logic
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
        save_question_progress(question_number, week_number=4)
        
        # Check if all questions are completed (week is done)
        all_completed = all(state.question_completed.get(q, False) for q in QUESTIONS.keys())
        if all_completed:
            session_id = session.get('session_id')
            if session_id:
                progress_tracker.update_user_progress(
                    session_id, 
                    week_number=4,
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
    print("Starting NOVA Career Coach Week 4 Server...")
    print(f"Loaded {len(QUESTIONS)} questions for Week 4")
    
    port = int(os.getenv('PORT', 5004))
    
    # Check and kill any existing process on the port
    print(f"Checking port {port}...")
    if kill_process_on_port(port):
        import time
        time.sleep(1)  # Give it a moment to fully stop
    
    print(f"Server starting on http://localhost:{port}")
    print("Week 4 questions available:")
    for qnum, question in QUESTIONS.items():
        print(f"  Question {qnum}: {question}")
    # Set use_reloader=False to avoid multiprocessing warnings when killing processes
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

