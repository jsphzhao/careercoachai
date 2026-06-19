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
        self.q2_scenario = None  # Store Q2 scenario classification: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3"
        self.q3_scenario = None  # Store Q3 scenario classification: "SCENARIO_1" or "SCENARIO_2"
        self.q4_scenario = None  # Store Q4 scenario classification: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3"
        self.q5_scenario = None  # Store Q5 scenario classification: "SCENARIO_1" or "SCENARIO_2"
        self.q6_scenario = None  # Store Q6 scenario classification: "SCENARIO_1" or "SCENARIO_2"
        self.q7_scenario = None  # Store Q7 scenario classification: "SCENARIO_1" or "SCENARIO_2"
        self.q8_scenario = None  # Store Q8 scenario classification: "SCENARIO_1" or "SCENARIO_2"
        self.q9_scenario = None  # Store Q9 scenario classification: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3"
        self.q10_scenario = None  # Store Q10 scenario classification: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3"
        self.q11_scenario = None  # Store Q11 scenario classification: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3"
        self.q12_scenario = None  # Store Q12 scenario classification: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3"
        self.q13_scenario = None  # Store Q13 scenario classification: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3"
        self.skip_q14 = False  # Track if user declined to share next-session topics
        self.q13_closing_message_index = 0  # Track which closing message to show (0 = first, 1 = second)
    
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


def save_question_progress(question_number: int, week_number: int = 1):
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


def validate_completeness(question_number, nova_response, user_responses, conversation_history, q2_scenario=None, q6_scenario=None, q6_iteration=None, q7_scenario=None, q8_scenario=None):
    """
    Use LLM to validate if user has provided all required information.
    Returns (is_complete: bool, missing_items: str)
    q2_scenario: For Q2, pass the scenario classification to skip validation for Scenario 2
    q6_scenario: For Q6, pass the scenario classification to handle Scenario 1 properly
    q6_iteration: For Q6, pass the iteration count to skip validation on first "I don't know" response
    q7_scenario: For Q7, pass the scenario classification to handle scenarios properly
    q8_scenario: For Q8, pass the scenario classification to handle scenarios properly
    """
    # Build context of what was requested
    context = f"Question {question_number}: {QUESTIONS.get(question_number, '')}\n\n"
    context += f"NOVA's latest response: {nova_response}\n\n"
    context += f"User's responses so far: {len(user_responses)} response(s)\n"
    for i, resp in enumerate(user_responses, 1):
        # Include full response for Q2 to ensure numbered answers are visible
        if question_number == 2:
            context += f"  Response {i}: {resp}\n"
        else:
            context += f"  Response {i}: {resp[:200]}...\n"

    # Special validation prompt for Q2 (homework questions)
    if question_number == 2:
        # If Scenario 2 is already classified, skip validation (it's complete)
        if q2_scenario and "SCENARIO_2" in q2_scenario:
            print(f"[DEBUG Q2 VALIDATION] Scenario 2 detected - skipping validation, marking complete")
            return True, "None"
        
        # If Scenario 1 or 3 is classified, proceed with homework validation
        if q2_scenario and ("SCENARIO_1" in q2_scenario or "SCENARIO_3" in q2_scenario):
            scenario_num = "1" if "SCENARIO_1" in q2_scenario else "3"
            print(f"[DEBUG Q2 VALIDATION] Scenario {scenario_num} detected - validating homework questions")
            # DEBUG: Print user's response
            user_resp = user_responses[-1] if user_responses else ""
            print(f"\n{'='*80}")
            print(f"[DEBUG Q2 VALIDATION] User's response: {user_resp}")
            print(f"[DEBUG Q2 VALIDATION] NOVA's response: {nova_response[:200]}...")
            print(f"{'='*80}\n")
            
            # Only validate homework questions for Scenario 1
            print(f"[DEBUG] Proceeding to validate homework questions...")
            validation_prompt = f"""You are validating if a user has provided all required homework information.

IMPORTANT: The homework REQUIRES answers to these EXACT 4 questions (and ONLY these 4):
1. Situation that bothered you - where were you, what were you doing, and who were you with?
2. How it made you feel? (Rate the intensity of your emotions from 0-10, with 0 being an insignificant emotion and 10 being an extremely intense emotion)
3. Thoughts (What were you thinking during the event?)
4. Alternate viewpoints (How might your coach see the situation?)

IGNORE any other questions that NOVA might have asked. ONLY validate against these 4 specific questions above.

The user may provide answers in numbered format (1. 2. 3. 4.) or in any other format. Look for answers that address each of the 4 questions above, regardless of format.

CRITICAL: If the user has provided numbered responses (1. 2. 3. 4.) that correspond to the 4 homework questions, they are COMPLETE. Do NOT require additional details beyond what they've provided.

Look carefully at the user's responses. If you see responses numbered 1, 2, 3, and 4 that address:
- Question 1: Situation details (where, what, who)
- Question 2: Feelings/emotions (with or without intensity rating)
- Question 3: Thoughts during the event
- Question 4: Alternate viewpoints (coach's or other perspective)

Then the user has completed ALL 4 homework questions and you should respond: "COMPLETE: Yes\nMISSING: None"

Based on the conversation history above, determine if the user has provided answers to ALL 4 of these specific homework questions. Do NOT check for any other questions that might appear in NOVA's response.

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items from the 4 questions above that are missing, or "None" if all provided]

Example responses:
- If user provided numbered answers 1-4 addressing all questions: "COMPLETE: Yes\nMISSING: None"
- If all 4 are provided in any format: "COMPLETE: Yes\nMISSING: None"
- If missing items: "COMPLETE: No\nMISSING: 1) Situation details, 3) Thoughts"
- DO NOT mention any questions that are not in the 4 questions listed above
"""
        else:
            # If scenario not set or unclear, skip validation (shouldn't happen but handle gracefully)
            print(f"[DEBUG Q2 VALIDATION] Scenario not set or unclear - skipping validation")
            return True, "None"
    elif question_number == 1:
        # For Q1, we only need to check if user provided a reason for participating
        # Don't validate follow-up questions - just check if original question was answered
        validation_prompt = """You are validating if a user has provided a reason for participating in the DRIVEN program.
The original question was: "Why did you decide to participate in the DRIVEN program?"

The user has provided at least one response. Determine if they have given ANY reasonable answer to why they want to participate, even if it's brief or indirect. 
- If they mentioned wanting a job, looking for work, career help → COMPLETE
- If they mentioned feeling down/hopeless AND NOVA asked follow-up → COMPLETE (the follow-up is part of the conversation flow)
- If they gave any reason related to mental health, job search, or personal growth → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any reason: "COMPLETE: Yes\nMISSING: None"
- If no reason provided: "COMPLETE: No\nMISSING: A clear explanation of why the user decided to participate"
"""
    elif question_number == 3:
        # For Q3, check if user provided a response about how viewpoints might help in the future
        validation_prompt = """You are validating if a user has provided a response to "How might these viewpoints help you navigate a similar situation in the future?"

The user has provided at least one response. Determine if they have given ANY reasonable answer, even if it's brief. The response should relate to how thinking flexibly or the viewpoints discussed might help them in future situations.
- If they express optimism, confidence, or mention using flexible thinking → COMPLETE
- If they express doubt but still provide a response (even if negative) → COMPLETE (the system will handle the doubt)
- If they give any response related to future application, learning, or perspectives → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any response: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about how viewpoints might help in the future"
"""
    elif question_number == 4:
        # For Q4, check if user provided a specific goal
        validation_prompt = """You are validating if a user has provided a response to "What goal did you select?"

The user has provided at least one response. Determine if they have selected a specific goal.
- If they mentioned a specific goal (e.g., "Take a fitness class", "Update my resume", "Read a book") → COMPLETE
- If they only mentioned a category (e.g., "personal goal", "professional goal") → INCOMPLETE (need specific goal)
- If they said they didn't select a goal or haven't chosen yet → INCOMPLETE
- Only mark as complete if they provided a concrete, actionable goal

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided specific goal: "COMPLETE: Yes\nMISSING: None"
- If user only provided category or no goal: "COMPLETE: No\nMISSING: A specific goal (not just the category)"
"""
    elif question_number == 5:
        # For Q5, check if user provided a Yes or No response
        validation_prompt = """You are validating if a user has provided a response to "Have you scheduled it in your calendar already? Please respond Yes or No."

The user has provided at least one response. Determine if they have given a clear Yes or No answer.
- If they said "Yes", "I have", "I scheduled it", "Already done", etc. → COMPLETE
- If they said "No", "Not yet", "I haven't", "I didn't", etc. → COMPLETE (both Yes and No are valid answers)
- If the response is unclear or doesn't answer the question → INCOMPLETE
- Only mark as incomplete if they gave no clear Yes/No indication

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided Yes or No: "COMPLETE: Yes\nMISSING: None"
- If user response is unclear: "COMPLETE: No\nMISSING: A clear Yes or No answer"
"""
    elif question_number == 6:
        # For Q6, check if user provided meaningful steps
        # For Scenario 1, check if user provided concrete steps after NOVA's breakdown
        if q6_scenario and "SCENARIO_1" in q6_scenario:
            # Check if this is the first response (user said "I don't know") or subsequent (user responding to breakdown)
            # We need to check if user provided concrete steps after NOVA's suggestions
            validation_prompt = """You are validating if a user has provided a concrete, specific step to achieve their goal after NOVA provided suggestions.

NOVA has provided a breakdown of steps and suggestions. The user should now respond with at least one specific, actionable step they will take.

Determine if the user has provided a concrete step:
- If they mentioned a specific action (e.g., "I will do X", "My first step is to...", "I'll start by...", "I plan to...", "I want to try...") → COMPLETE
- If they selected/mentioned one of NOVA's suggestions → COMPLETE
- If they provided a concrete plan with actionable items → COMPLETE
- If they gave vague responses like "I'll try", "Maybe", "I'll think about it" without specifics → INCOMPLETE
- If they gave NO answer at all or completely unrelated response → INCOMPLETE

Be lenient with wording - focus on whether they've indicated a concrete action they will take, not the exact phrasing.

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided a concrete step: "COMPLETE: Yes\nMISSING: None"
- If user gave vague response: "COMPLETE: No\nMISSING: A specific, concrete step they will take to achieve their goal"
"""
        else:
            # For Q6 Scenario 2, check if user provided meaningful steps
            validation_prompt = """You are validating if a user has provided meaningful steps to achieve their goal.

The user has provided at least one response. Determine if they have given specific, actionable steps.
- If they provided specific steps (e.g., "I will do X on Y date", "First step is to...", "I plan to...") → COMPLETE
- If they provided a concrete plan with actionable items → COMPLETE
- If they gave vague responses like "I'll try" or "Maybe" without specifics → INCOMPLETE
- If they gave NO answer at all or completely unrelated response → INCOMPLETE

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided specific steps: "COMPLETE: Yes\nMISSING: None"
- If user gave vague response: "COMPLETE: No\nMISSING: At least one specific, actionable step"
"""
    elif question_number == 7:
        # For Q7, check if user provided a response about barriers
        validation_prompt = """You are validating if a user has provided a response to "What barriers do you think you might encounter?"

The user has provided at least one response. Determine if they have given any meaningful answer.
- If they mentioned barriers or obstacles → COMPLETE
- If they said they don't see any barriers or everything is manageable → COMPLETE (the system will provide clarifying questions)
- If they gave any response related to challenges, obstacles, or barriers (or lack thereof) → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided barriers or said no barriers: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about barriers they might encounter"
"""
    elif question_number == 8:
        # For Q8, check if user provided a response about how to overcome barriers
        validation_prompt = """You are validating if a user has provided a response to "How can you think flexibly to overcome these barriers?"

The user has provided at least one response. Determine if they have given any meaningful answer.
- If they provided strategies, ideas, or ways to overcome barriers → COMPLETE
- If they expressed uncertainty but still provided some response → COMPLETE (the system will provide suggestions)
- If they gave any response related to overcoming barriers or flexible thinking → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Important: If the user's response already satisfies the criteria and is considered complete, do not require additional information. Mark as COMPLETE so we can proceed to the next question.

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided strategies or ideas: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about how to overcome barriers"
"""
    elif question_number == 9:
        # For Q9, check if user provided types of roles they are looking for
        validation_prompt = """You are validating if a user has provided a response to "What types of roles are you looking for?"

The user has provided at least one response. Determine if they have given any meaningful answer about job roles or types of positions.
- If they mentioned specific roles, job titles, or types of positions → COMPLETE
- If they mentioned industries, fields, or general categories → COMPLETE
- If they expressed uncertainty but still provided some response (e.g., "I'm not sure", "I'm exploring") → COMPLETE (the system will help narrow down)
- If they gave any response related to job roles, career interests, or positions → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided roles or types: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about types of roles they are looking for"
"""
    elif question_number == 10:
        # For Q10, check if user provided a response about difficult aspects of job searching
        validation_prompt = """You are validating if a user has provided a response to "And what is - or what do you think will be - the most difficult aspect of job searching for you?"

The user has provided at least one response. Determine if they have given any meaningful answer about challenges or difficulties in job searching.
- If they mentioned specific challenges (e.g., interviews, applications, networking, confidence) → COMPLETE
- If they said they don't see any difficulties or feel confident → COMPLETE (the system will provide appropriate response)
- If they expressed uncertainty but still provided some response → COMPLETE (the system will help identify challenges)
- If they gave any response related to job search difficulties, challenges, or obstacles → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided challenges or said no challenges: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about difficult aspects of job searching"
"""
    elif question_number == 11:
        # For Q11, check if user provided a response about barriers to completing activities
        validation_prompt = """You are validating if a user has provided a response to "Is there anything happening in your life that might prevent you from completing the activities?"

The user has provided at least one response. Determine if they have given any meaningful answer about potential barriers or obstacles.
- If they mentioned barriers, obstacles, or challenges → COMPLETE
- If they said there are no barriers or everything is fine → COMPLETE (the system will acknowledge this)
- If they mentioned serious mental health concerns or crisis → COMPLETE (the system will provide appropriate resources)
- If they expressed uncertainty but still provided some response → COMPLETE
- If they gave any response related to barriers, obstacles, or life circumstances → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided barriers or said no barriers: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about potential barriers to completing activities"
"""
    elif question_number == 12:
        # For Q12, check if user provided a response about motivation
        validation_prompt = """You are validating if a user has provided a response to "Thinking about the ideas we've discussed so far, are you feeling more motivated to proceed with your job search?"

The user has provided at least one response. Determine if they have given any meaningful answer about their motivation level.
- If they expressed increased motivation, enthusiasm, or positive feelings → COMPLETE
- If they expressed low motivation, discouragement, or negative feelings → COMPLETE (the system will provide support)
- If they expressed mixed or uncertain feelings → COMPLETE (the system will acknowledge ambivalence)
- If they gave any response related to motivation, feelings, or readiness → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any response about motivation: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about their motivation level"
"""
    elif question_number == 13:
        # For Q13, check if user provided a response about topics for next session
        validation_prompt = """You are validating if a user has provided a response to "Is there anything specific you'd like to talk about the next time we get together?"

The user has provided at least one response. Determine if they have given any meaningful answer about topics for future sessions.
- If they mentioned specific topics or areas they want to discuss → COMPLETE
- If they said "No" or indicated they don't have specific topics → COMPLETE (the system will offer suggestions)
- If they expressed uncertainty but still provided some response → COMPLETE (the system will clarify)
- If they gave any response related to future topics, preferences, or next steps → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided topics or said no: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about topics for next session"
"""
    elif question_number == 14:
        # For Q14, check if user provided a response (this is a follow-up/closing question)
        validation_prompt = """You are validating if a user has provided a meaningful response to the closing question or transition.

The user has provided at least one response. Determine if they have given any meaningful answer.
- If they provided any response related to the conversation or next steps → COMPLETE
- If they gave any acknowledgment or indication they understood → COMPLETE
- If they gave any response at all → COMPLETE (be lenient for closing questions)
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any response: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response to the closing question"
"""
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


# Define the homework questions explicitly (must be before SYSTEM_PROMPTS that uses it)
HOMEWORK_QUESTIONS_Q2 = """1. Situation that bothered you - where were you, what were you doing, and who were you with?
2. How it made you feel? (Rate the intensity of your emotions from 0-10, with 0 being an insignificant emotion and 10 being an extremely intense emotion)
3. Thoughts (What were you thinking during the event?)
4. Alternate viewpoints (How might your coach see the situation?)"""

# Data required for Q3 prompts
THINKING_FLEXIBLY_NOTES = """Notes on thinking flexibly from videos and homework: Why think flexibly: It helps you adapt to change, solve problems creatively, and understand different perspectives, making you more resilient and effective in an unpredictable world. How to practice: Actively challenge your initial reactions by asking "What's another way to see this?" or "What would someone I respect but disagree with say?" How to strengthen it: Regularly expose yourself to unfamiliar ideas, disciplines, and experiences, as mental flexibility grows strongest when you step outside familiar patterns and genuinely consider viewpoints that initially feel uncomfortable."""

THINKING_FLEXIBLY_JOB_MARKET = """How thinking flexibly can help one navigate the job market:
There isn't just one right path—try a few good options and you'll create more chances.
Each "no" is data—tweak one small thing and you're already improving for next time.
On hard days, shrink the task; five minutes forward still counts.
Progress has many forms—networking, a draft, a message; stack small wins for future searches.
Experiment like a scientist—test, learn, adjust, and momentum will follow in the job market."""

# Data required for Q4 prompts
GOAL_CATEGORIES = """Dictionary of goal categories:

Personal examples:
- Take a fitness class
- Read a book for pleasure
- Try a new recipe

Professional examples:
- Update your resume
- Refresh your LinkedIn profile
- Search online for job openings"""

# Data required for Q9 prompts
JOB_ROLE_CATEGORIES = """List of example job role categories:
- Entry-level roles (e.g., retail, customer service, food service)
- Administrative or office support roles
- Peer support, caregiving, or human services roles
- Skilled trades or hands-on roles
- Remote or flexible-schedule roles"""

# Data required for Q10 prompts
JOB_SEARCH_CHALLENGES = """Dict of common job search challenges:
- Not knowing where to look for jobs
- Difficulty writing resumes or filling out applications
- Anxiety about interviews or talking to employers
- Limited work history, gaps in employment, or legal history
- Transportation, childcare, or health-related challenges
- Fear of rejection, low confidence, or feeling overwhelmed"""

# Data required for Q11 prompts
LIFE_CIRCUMSTANCES = """List of example life circumstances that might affect participation:
- Work, school, or caregiving responsibilities
- Housing or financial instability
- Physical health concerns or disability
- Mental health symptoms (e.g., depression, anxiety, substance use)
- Limited access to internet, technology, or a quiet space
- Legal, transportation, or safety issues"""

# Data required for Q12 prompts
BEHAVIORAL_ACTIVATION_NOTES = """Notes on behavioral activation and motivation from week 1:
- Taking small, goal-directed actions can improve mood and motivation over time.
- Action often comes before motivation; doing one small step can help you feel more hopeful.
- Tracking small wins (like sending a message or starting an application) can build momentum."""

# Data required for Q6 prompts
CLARIFYING_QUESTIONS_Q6 = """List of Clarifying questions:
- What is step 1, exactly when and where will you do it?
- What skills/info are missing, and who/what can help?
- What's the smallest workable version of the goal (MVP)?
- What milestone can you hit in 1 hour? In 2 weeks?"""

# Data required for Q7 prompts
CLARIFYING_QUESTIONS_Q7 = """List of Clarifying questions:
- Are there any financial constraints?
- Are there any time constraints?
- Do you have trouble finding the motivation to accomplish the steps toward your goal?"""

# Data required for Q8 prompts
CLARIFYING_QUESTIONS_Q8 = """List of Clarifying questions:
- Which part of the expected barriers is within your control this week?
- What's the smallest one-hour experiment you can run to move forward?
- Name two alternate paths to the same outcome that avoid the constraint.
- What single resource/skill would remove most friction—and how will you get it?
- Which barrier may be an assumption that might be wrong—what's the opposite you could test?"""

# Global instruction used to ensure no extra questions once a question is complete
NO_FOLLOWUPS_INSTRUCTION = (
    "Important: If the user's response already satisfies the criteria for this question and it is considered complete (or the classified scenario instructs to move on), do not ask any additional questions. Conclude your reply without further questions so we can proceed to the next question."
)

WEEK1_AFTER_Q8_MESSAGE = """Hey there! It's Nova 👋

As you heard about in the first week's videos, the rest of the DRIVEN course will offer guidance on how to stay motivated while working on your job search, and how to deal with some of the stress and challenges along the way.

I'd like to use these coaching sessions to build on what you learn through the course, and help you solve your specific job search challenges.

Let's jump into it!"""

WEEK1_AFTER_Q13_MESSAGE = "Lastly, while you're working on Week 2's content about stress and Week 3's content about skills, please write down your questions so we can cover them when we meet after Week 3."

WEEK1_FINAL_CLOSING_MESSAGE = "It has been terrific meeting with you today. It's clear that you're working hard and your willingness to have an open mind about new ways to approach your job search is awesome. Talk with you soon!"

NEXT_STEPS_TEXT = """Before we meet next, you will complete the activities in Weeks 2 and 3 in the DRIVEN app. I'll also look forward to hearing about your progress toward your goal from this week's Exercise 2."""

CLOSING_TEXT = """Lastly, while you're working on Week 2's content about stress and Week 3's content about skills, please write down your questions so we can cover them when we meet after Week 3.

It has been terrific meeting with you today. It’s clear that you’re working hard and your willingness to have an open mind about new ways to approach your job search is awesome. Talk with you in a few weeks!"""


def normalize_yes_no(message: str):
    """Return True for yes, False for no, or None if unclear."""
    if not message:
        return None
    cleaned = message.strip().lower()
    yes_values = {"yes", "y", "yeah", "yep", "sure", "absolutely"}
    no_values = {"no", "n", "nope", "nah", "not really"}
    if cleaned in yes_values:
        return True
    if cleaned in no_values:
        return False
    return None

def generate_user_summary(state):
    """Generate a summary of user's priorities and challenges from questions 1-12."""
    summary_parts = []
    
    # Q1: Reason for participating
    if 1 in state.answers and state.answers[1]:
        q1_answer = " ".join(state.answers[1])
        summary_parts.append(f"Reason for participating in DRIVEN: {q1_answer}")
    
    # Q4: Goal selected
    if 4 in state.answers and state.answers[4]:
        q4_answer = " ".join(state.answers[4])
        summary_parts.append(f"Selected goal: {q4_answer}")
    
    # Q6: Steps to achieving goal
    if 6 in state.answers and state.answers[6]:
        q6_answer = " ".join(state.answers[6])
        summary_parts.append(f"Steps planned: {q6_answer}")
    
    # Q7: Barriers
    if 7 in state.answers and state.answers[7]:
        q7_answer = " ".join(state.answers[7])
        summary_parts.append(f"Expected barriers: {q7_answer}")
    
    # Q8: Overcoming barriers
    if 8 in state.answers and state.answers[8]:
        q8_answer = " ".join(state.answers[8])
        summary_parts.append(f"Strategies to overcome barriers: {q8_answer}")
    
    # Q9: Types of roles
    if 9 in state.answers and state.answers[9]:
        q9_answer = " ".join(state.answers[9])
        summary_parts.append(f"Types of roles interested in: {q9_answer}")
    
    # Q10: Difficult aspects of job searching
    if 10 in state.answers and state.answers[10]:
        q10_answer = " ".join(state.answers[10])
        summary_parts.append(f"Most difficult aspect of job searching: {q10_answer}")
    
    # Q11: Life circumstances
    if 11 in state.answers and state.answers[11]:
        q11_answer = " ".join(state.answers[11])
        summary_parts.append(f"Life circumstances that might affect participation: {q11_answer}")
    
    # Q12: Motivation level
    if 12 in state.answers and state.answers[12]:
        q12_answer = " ".join(state.answers[12])
        summary_parts.append(f"Current motivation level: {q12_answer}")
    
    if summary_parts:
        return "\n".join(summary_parts)
    else:
        return "User is early in the program and has not yet provided extensive information about their priorities and challenges."

QUESTIONS = {
    1: "Why did you decide to participate in the DRIVEN program?",
    2: "What was the main idea you took away from the homework?",
    3: "How might these viewpoints help you navigate a similar situation in the future?",
    4: "What goal did you select? (activity)",
    5: "Have you scheduled it in your calendar already? Please respond Yes or No.",
    6: "What are the steps you'll need to take to achieve this goal?",
    7: "What barriers do you think you might encounter?",
    8: "How can you think flexibly to overcome these barriers?",
    9: "What types of roles are you looking for?",
    10: "And what is - or what do you think will be - the most difficult aspect of job searching for you?",
    11: "Is there anything happening in your life that might prevent you from completing the activities?",
    12: "Thinking about the ideas we've discussed so far, are you feeling more motivated to proceed with your job search?",
    13: "Is there anything specific you'd like to talk about the next time we get together?"
}

# System prompts from follow.py
SYSTEM_PROMPTS = {
    1: """Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. Above is the user's reason for participating in the DRIVEN program. The user's name is {name}. The user's response will fall into the following scenarios. 1) The user is look for a job; 2) User is feeling down or hopeless. 3) The user provides a reason that is not related to mental health or job opportunities. Please respond to each potential scenario in accordance with the following guidelines:
Scenario 1: Provide motivation that the DRIVEN program will teach them how to regain control of their lives. 
Scenario 2: First, inquire if they are feeling that way because of unemployment - If they are, provide motivation that the DRIVEN 6-week program will teach them how to regain control of their lives and steer conversation back to the day's session. If they are not feeling down because of unemployment, direct them to mental health services. 
Scenario 3: provide a summary of what the DRIVEN program is, and tell the user that Nova is excited to take them through the program.""",
    
    # Q2 classifier prompt - separate from response generation
    2: {
        "classifier": f"""Based on the user's response to "What was the main idea you took away from the homework?", determine which scenario applies:

Scenario 1: User responds that they did NOT do the exercise yet (e.g., "I didn't do it", "I haven't done it", "I didn't watch the video", "I forgot", "I didn't complete it")

Scenario 2: User responds suggesting they completed the exercise and their response is insightful and related to the homework (e.g., mentions "learned", "takeaway", "thinking flexibly", "adapting to change", "different perspectives", "challenging reactions", or any meaningful reflection on what they learned from the homework)

Scenario 3: User responds indicating they did the homework but their response is NOT related to the goal of the homework questions (e.g., "I learned nothing", "It was boring", "Nothing useful", vague or unrelated responses that don't reflect understanding of thinking flexibly or the homework content)

IMPORTANT: 
- If the user mentions ANYTHING meaningful about what they learned, took away, or understood from the homework (related to thinking flexibly, adapting, perspectives), that is Scenario 2
- If the user explicitly says they didn't do it, that is Scenario 1
- If the user says they did it but gives an unhelpful/unrelated response, that is Scenario 3

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_ask": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. The user has indicated they did not do the homework exercise yet.

Ask the user to answer the questions in the homework. Present these 4 questions exactly as listed:
{HOMEWORK_QUESTIONS_Q2}

Ask the user to number their responses to each question.""",
        "scenario_2_congratulate": """Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {name}. The user has indicated they completed the homework and provided their main takeaway.

Congratulate the user for having done a good job on the homework. If they mention learning about thinking flexibly or any related concepts, acknowledge their understanding and congratulate them. Be warm and encouraging.""",
        "scenario_3_ask": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. The user has indicated they did the homework but their response suggests they may not have fully engaged with or understood the homework exercise.

Ask the user to answer the questions in the homework. Present these 4 questions exactly as listed:
{HOMEWORK_QUESTIONS_Q2}

Ask the user to number their responses to each question. Be encouraging and help them understand that these questions will help them practice thinking flexibly."""
    },
    
    # Q3 classifier prompt - separate from response generation
    3: {
        "classifier": f"""Based on the user's response to "How might these viewpoints help you navigate a similar situation in the future?", determine which scenario applies:

Scenario 1: User thinks these new viewpoints might help them in the future (e.g., expresses optimism, confidence, or mentions using flexible thinking to handle future challenges).

Scenario 2: User does not think these viewpoints will help them in the future (e.g., expresses doubt, discouragement, or uncertainty about the usefulness of the exercise).

Respond with ONLY one of these: "SCENARIO_1" or "SCENARIO_2".""",
        "scenario_1_respond": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. The user believes that thinking flexibly might help them in the future.

Congratulate the user for their optimistic perspective and acknowledge how they are applying what they learned in the program to their outlook on life.""",
        "scenario_2_respond": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. The user does not think the exercise will help them in the future.

Provide a summary of why thinking flexibly can help the user achieve their goals for participating in DRIVEN. Use the following information:

{THINKING_FLEXIBLY_NOTES}

{THINKING_FLEXIBLY_JOB_MARKET}

If the user provided a reason for participating in DRIVEN (from Question 1), reference that specific goal. If no specific goal was provided, give general examples of how flexible thinking helps job seekers—such as adapting to change, staying motivated, and finding creative solutions in uncertain situations."""
    },
    
    # Q4 classifier prompt - separate from response generation
    4: {
        "classifier": f"""Based on the user's response to "What goal did you select?", determine which scenario applies:

Scenario 1: User clearly selected a goal (personal or professional).

Scenario 2: User did not select a goal.

Scenario 3: User responded with only the category (e.g., 'personal goal' or 'professional goal') rather than a specific one.

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. They have selected a specific goal.

Acknowledge their goal selection positively and provide a brief encouraging comment on how it could support their mental health or job search motivation. Be warm and supportive.""",
        "scenario_2_prompt": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. The user has not selected a goal.

Ask them to choose either a personal or professional goal. Provide examples from the following goal categories:

{GOAL_CATEGORIES}

Also suggest goals to set based on {{Response from 1}}. Once they respond with a specific goal, provide a brief comment on how it could support their mental health or job search motivation.""",
        "scenario_3_followup": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. The user has only named the category of goal.

Prompt them to specify a concrete goal within that category, offering examples from the following goal categories:

{GOAL_CATEGORIES}

Also provide suggestions aligned with {{Response from 1}}. After they specify a goal, provide a short encouraging reflection on how it supports their wellbeing or professional growth."""
    },
    
    # Q5 classifier prompt - separate from response generation
    5: {
        "classifier": f"""Based on the user's response to "Have you scheduled it in your calendar already?", determine which scenario applies:

Scenario 1: User has scheduled the goal.

Scenario 2: User has NOT scheduled the goal.

Respond with ONLY one of these: "SCENARIO_1" or "SCENARIO_2".""",
        "scenario_1_respond": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. 

Respond in this exact format: "Great work, {{name}}! I'm so glad to hear that you've scheduled your goal in your calendar. This is a fantastic step towards taking charge of your career journey. Remember, every little step counts and your commitment to this process is truly commendable. " UNDER NO CIRCUMSTANCES SHOULD YOU ASK A FOLLOW UP QUESTION IN THIS RESPONSE.""",
        "scenario_2_prompt": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. They have not yet scheduled their goal.

Thank them for sharing and gently encourage them to add it to their calendar. For example: Thanks for letting me know! I encourage you to take a moment to add it to your calendar so it's more likely to happen. Let me know when you've added it to your calendar and we can continue the session!"""
    },
    
    # Q6 classifier prompt - separate from response generation
    6: {
        "classifier": f"""Based on the user's response to "What are the steps you'll need to take to achieve this goal?", determine which scenario applies:

Scenario 1: User does not know how to achieve the goal.

Scenario 2: User provides clear steps.

Respond with ONLY one of these: "SCENARIO_1" or "SCENARIO_2".""",
        "scenario_1_assist": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. The user does not know how to achieve their goal and has asked for help.

Break down their goal ({{Answer to question 4 (goal selected)}}) into simple, achievable steps. Use the following clarifying questions to guide them in identifying their first small actions:

{CLARIFYING_QUESTIONS_Q6}

After providing the breakdown, ask the user to select one of the steps or suggestions you provided and specify when and where they will do it. Make sure they provide an answer to the question.""",
        "scenario_2_reinforce": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. The user provides clear steps and confidence.

Acknowledge their plan and reinforce their motivation with encouragement.  
If the steps are vague or overly ambitious, provide constructive feedback — for example, suggest setting specific times or simplifying the goal — while keeping the tone supportive and positive."""
    },
    
    # Q7 classifier prompt - separate from response generation
    7: {
        "classifier": f"""Based on the user's response to "What barriers do you think you might encounter?", determine which scenario applies:

Scenario 1: User presents barriers.

Scenario 2: User does not present any barriers (e.g., says they don't anticipate obstacles or that everything is manageable).

Respond with ONLY one of these: "SCENARIO_1" or "SCENARIO_2".""",
        "scenario_1_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues learn how to think more flexibly. The user's name is {{name}}. The user has shared barriers they might face while pursuing their goal ({{Answer to question 4 (goal selected)}}).

Based on their steps to achieve this goal ({{Answer to question 6 (steps to achieving goal)}}), provide a positive and motivational response that acknowledges their feelings and reinforces their ability to overcome these challenges. Focus on encouragement and validation. Do not give advice yet on how the user can overcome these barriers; that will be the next question.""",
        "scenario_2_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues learn how to think more flexibly. The user's name is {{name}}. The user does not see any barriers to achieving their goal.

Provide clarifying questions based on their goal ({{Answer to question 4 (goal selected)}}) and their steps ({{Answer to question 6 (steps to achieving goal)}}) to encourage them to consider potential challenges and develop contingency plans. Use the following clarifying questions: {CLARIFYING_QUESTIONS_Q7}

Do not give advice yet on how the user can overcome these barriers or ask any follow up questions under any circumstances; that will be the next question."""
    },
    
    # Q8 classifier prompt - separate from response generation
    8: {
        "classifier": f"""Based on the user's response to "How can you think flexibly to overcome these barriers?", determine which scenario applies:

Scenario 1: User provides a confident and thoughtful answer to the question of how to overcome barriers.

Scenario 2: User does not express a way to overcome barriers.

Respond with ONLY one of these: "SCENARIO_1" or "SCENARIO_2".""",
        "scenario_1_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user provides confident strategies for overcoming their barriers ({{Answers to questions 7 (expected barriers)}}).

Based on their goal ({{Answer to questions 4 (goal selected)}}) and their steps ({{Answer to questions 6 (steps to achieving goal)}}), offer positive reinforcement — acknowledge their problem-solving mindset and affirm their flexible thinking. Then, summarize their plan in bullet points to help them see their progress clearly.""",
        "scenario_2_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user is unsure how to overcome their barriers ({{Answers to questions 7 (expected barriers)}}) .

Analyze their listed barriers and provide tailored suggestions for overcoming them by thinking flexibly. Use the following information:

{THINKING_FLEXIBLY_NOTES}
Referencing their goal {{Answer to questions 4 (goal selected)}} and their steps {{Answer to questions 6 (steps to achieving goal)}} when relevant), use the following clarifying questions to guide their next steps when overcoming barriers:
{CLARIFYING_QUESTIONS_Q8}

End with a brief summary of their updated plan in bullet points."""
    },
    
    # Q9 classifier prompt - separate from response generation
    9: {
        "classifier": f"""Based on the user's response to "What types of roles are you looking for?", determine which scenario applies:

Scenario 1: User provides a clear and realistic description of the roles they are interested in (e.g., specific job titles, types of work, industries, or categories from the list).

Scenario 2: User is unsure, very broad (e.g., "anything", "I don't know", "whatever is available"), or gives conflicting role interests (e.g., mentions multiple unrelated fields without clarity).

Scenario 3: User describes roles that may be unrealistic for their current situation, experience, or constraints (e.g., very senior positions, requires extensive education/experience they don't have, or roles that conflict with known constraints).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. The user has provided a clear and realistic description of the types of roles they are interested in.

Reflect back the roles the user mentioned and briefly connect how exploring these roles through DRIVEN can help them move toward the goals they shared in {{Answer to question 1 (reason for participating in DRIVEN)}}. Offer encouragement and, if appropriate, invite them to share one or two skills or experiences that might fit those roles.

{NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_2_respond": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. The user is unsure, very broad, or has conflicting role interests.

Normalize that it is okay not to be sure yet, and use the following job role categories to help them narrow down what sounds most interesting or doable:
{JOB_ROLE_CATEGORIES}

Ask one or two gentle questions to clarify what kind of environment or tasks they might prefer, and reassure them that DRIVEN will continue to help them explore options.

{NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_3_respond": f"""Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {{name}}. The user has described roles that may be unrealistic for their current situation, experience, or constraints.

Acknowledge their interests respectfully and explore what appeals to them about those roles. Gently introduce the idea of "stepping-stone" roles or related positions that might be more realistic right now, based on their situation and the DRIVEN goals {{Answer to question 1 (reason for participating in DRIVEN)}}. Emphasize that starting with a realistic role can still move them toward their longer-term dreams.

{NO_FOLLOWUPS_INSTRUCTION}"""
    },
    
    # Q10 classifier prompt - separate from response generation
    10: {
        "classifier": f"""Based on the user's response to "And what is - or what do you think will be - the most difficult aspect of job searching for you?", determine which scenario applies:

Scenario 1: User identifies one or more concrete challenges (e.g., mentions specific difficulties like interviews, resumes, applications, transportation, confidence, etc.).

Scenario 2: User says that "everything" feels hard, expresses strong discouragement or hopelessness about job searching (e.g., "everything is difficult", "it all seems impossible", "I don't think I can do this", very negative/disheartened tone).

Scenario 3: User says they do not expect any major difficulties or feels very confident (e.g., "I don't think anything will be too hard", "I'm pretty confident", "I'm not worried about it", very positive/confident tone).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user has identified one or more concrete challenges about job searching.

Validate and normalize the challenge(s) they mentioned, linking them to one or more items from the following common challenges:
{JOB_SEARCH_CHALLENGES}

Briefly explain how the DRIVEN program can help them build skills or strategies for those difficulties (for example, breaking tasks down, practicing interviews, or learning where to search), and offer one small next step they can focus on. Reference their roles of interest {{Answer to question 9 (types of roles you are looking for)}} when relevant.

{NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_2_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user says that "everything" feels hard or expresses strong discouragement or hopelessness about job searching.

Acknowledge that job searching can feel overwhelming, especially when many things feel hard at once. Gently help them break "everything" into smaller parts by referencing the following common challenges:
{JOB_SEARCH_CHALLENGES}

Ask which one or two feel like the biggest obstacles right now. Provide reassurance that the program will help them work on these challenges step-by-step, so they do not have to solve everything at once. Reference their roles of interest {{Answer to question 9 (types of roles you are looking for)}} when relevant.

{NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_3_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user says they do not expect any major difficulties or feels very confident.

Affirm their confidence and positive expectations. Encourage them to stay open to noticing any challenges that might arise and remind them that it is okay to ask for support if things become more difficult later. Invite them to think about one area (such as applications, interviews, or routines) where preparing ahead could help them maintain their confidence. Reference their roles of interest {{Answer to question 9 (types of roles you are looking for)}} when relevant.

{NO_FOLLOWUPS_INSTRUCTION}"""
    },
    
    # Q11 classifier prompt - separate from response generation
    11: {
        "classifier": f"""Based on the user's response to "Is there anything happening in your life that might prevent you from completing the activities?", determine which scenario applies:

Scenario 1: User describes practical or situational barriers but appears generally safe and stable (e.g., mentions work responsibilities, time constraints, financial issues, transportation, health concerns, but does NOT mention self-harm, suicide, abuse, or immediate danger).

Scenario 2: User describes serious mental health symptoms, mentions self-harm, thoughts of suicide, harm to others, or current abuse or danger (e.g., mentions suicidal thoughts, self-harm, wanting to hurt others, being in an abusive relationship, feeling unsafe, experiencing a mental health crisis).

Scenario 3: User reports no major barriers at this time (e.g., says "no", "nothing", "I'm good", "everything is fine", indicates they don't anticipate barriers).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user has described practical or situational barriers but appears generally safe and stable.

Thank the user for sharing and normalize that life circumstances can make it harder to complete activities. Show empathy for the barriers they mentioned and, using the following example life circumstances, suggest one or two simple adjustments or supports (such as breaking tasks into smaller steps, choosing lower-energy activities, or planning around their schedule) that could help them still make progress toward their goal {{Answer to question 4 (goal selected)}}:

{LIFE_CIRCUMSTANCES}

Reference their steps {{Answer to question 6 (steps to achieving goal)}} when relevant.

{NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_2_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user has described serious mental health symptoms, mentions self-harm, thoughts of suicide, harm to others, or current abuse or danger.

Respond with warmth and concern. Acknowledge the seriousness of what they are going through and clearly encourage them to reach out to appropriate mental health or crisis supports in their area (for example, a crisis line, emergency services, or a trusted professional), emphasizing that their safety and well-being come first. Clarify that DRIVEN and Nova cannot provide emergency or crisis care, but can still support them with gentle steps toward their goals once they are safe. Avoid giving medical advice; instead, focus on validation, safety, and connecting them with professional help.

{NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_3_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user reports no major barriers at this time.

Acknowledge that it is helpful they are not noticing major barriers right now and encourage them to keep checking in with themselves each week. Invite them to consider any small routines or supports that could help them stay on track with their goal {{Answer to question 4 (goal selected)}}, so if new challenges arise, they already have a plan. Reference their steps {{Answer to question 6 (steps to achieving goal)}} when relevant.

{NO_FOLLOWUPS_INSTRUCTION}"""
    },
    
    # Q12 classifier prompt - separate from response generation
    12: {
        "classifier": f"""Based on the user's response to "Thinking about the ideas we've discussed so far, are you feeling more motivated to proceed with your job search?", determine which scenario applies:

Scenario 1: User responds "Yes" or expresses feeling more motivated (e.g., "yes", "I feel more motivated", "definitely", "I'm excited", positive/encouraged tone).

Scenario 2: User responds "No" or expresses less or low motivation (e.g., "no", "not really", "I'm still not feeling it", "it's hard", negative/discouraged tone).

Scenario 3: User gives a mixed, uncertain, or "maybe" type response (e.g., "maybe", "kind of", "a little bit", "somewhat", "I'm not sure", ambiguous/mixed tone).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user responds "Yes" or expresses feeling more motivated.

Celebrate their increased motivation and connect it back to the goals they shared in {{Answer to question 1 (reason for participating in DRIVEN)}} and {{Answer to question 4 (goal selected)}}. Highlight how the small steps they are taking fit with the following notes on behavioral activation and motivation:
{BEHAVIORAL_ACTIVATION_NOTES}

{NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_2_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user responds "No" or expresses less or low motivation.

Validate their honesty and normalize that motivation can go up and down, especially when someone is dealing with stress or mental health challenges. Briefly remind them of the ideas from the following notes on behavioral activation and motivation:
{BEHAVIORAL_ACTIVATION_NOTES}

Emphasize that they do not have to feel highly motivated to take one tiny step. Invite them to share what is making it hard to feel motivated and, if appropriate, suggest one very small, manageable action that still moves them gently toward their goal {{Answer to question 4 (goal selected)}}.

{NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_3_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user gives a mixed, uncertain, or "maybe" type response.

Acknowledge that it is completely okay to have mixed feelings about job searching. Reflect their ambivalence and connect it to the idea that taking small actions from the following notes can sometimes help clarify how they feel:
{BEHAVIORAL_ACTIVATION_NOTES}

Suggest a small next step—such as spending a few minutes exploring roles, updating a tiny part of their resume, or planning one activity—that feels doable even with uncertainty to the user.

{NO_FOLLOWUPS_INSTRUCTION}"""
    },
    
    # Q13 classifier prompt - separate from response generation
    13: {
        "classifier": f"""Based on the user's response to "Is there anything specific you'd like to talk about the next time we get together?", determine which scenario applies:

Scenario 1: User response provides what they would like to talk about next time (e.g., mentions specific topics, areas, or questions they want to discuss).

Scenario 2: User responds "No" or indicates they don't have anything specific (e.g., "no", "nothing", "not really", "I don't have anything specific").

Scenario 3: User's response is unclear or does not directly answer (e.g., vague responses, doesn't clearly say yes or no, ambiguous).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user has provided what they would like to talk about next time.

Affirm that it is helpful they already have ideas for what they want to focus on next time. Reassure them that you will remember and use it to plan your next session together. You may offer gentle examples based on {{Summary of user's priorities and challenges from questions 1–12}} (for example, job goals, applications, interviews, or coping with stress) to support their thinking.

{NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_2_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user responds "No" or indicates they don't have anything specific to talk about next time.

Normalize that it is okay not to have a specific topic in mind yet. Let them know you can suggest ideas next time based on what you have learned about them so far {{Summary of user's priorities and challenges from questions 1–12}}. Mention that you might focus on areas like clarifying job goals, building application materials, or managing motivation and stress, and reassure them that you will check in again about their preferences in future sessions.

{NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_3_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user's response is unclear or does not directly answer whether they have something specific to talk about next time.

Gently clarify whether they would like to choose a topic now or prefer that you suggest one at the next meeting. Provide a couple of simple examples drawn from {{Summary of user's priorities and challenges from questions 1–12}} (for instance, practicing interviews, planning job search steps, or talking more about barriers) and invite them to say which sounds most helpful.

{NO_FOLLOWUPS_INSTRUCTION}"""
    }
}

WELCOME_MESSAGE = """Hello {name}!
I'm Nova, a virtual coach here to help you through the DRIVEN program.
To help you as best I can, I want to get to know you a bit better."""


@app.route('/')
def index():
    """Serve the index.html file."""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return Response(f.read(), mimetype='text/html')
    except FileNotFoundError:
        return "index.html not found. Please make sure the file exists in the same directory as week1_main.py.", 404


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
    week_number = data.get('week_number', 1)
    
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
    week_number = 1  # Week 1
    
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

    # Automatically skip Q14 if the user indicated they have no additional topics
    while state.current_question is not None and state.current_question == 14 and getattr(state, "skip_q14", False):
        if 14 not in state.question_completed:
            state.question_completed[14] = True
            state.answers.setdefault(14, []).append("User indicated no specific topic for the next session.")
            save_question_progress(14, week_number=1)
        state.skip_q14 = False
        state.current_question += 1
        if state.current_question is None or state.current_question > len(QUESTIONS):
            break
    
    # Special handling: After Q8 is completed, show the transition message
    if state.current_question == 9 and 8 in state.question_completed and not getattr(state, "q8_transition_shown", False):
        state.q8_transition_shown = True
        # Keep current_question as 9 so next call will show Q9
        return jsonify({
            "success": True,
            "message": WEEK1_AFTER_Q8_MESSAGE,
            "is_complete": False,
            "awaiting_response": False,
            "is_print_message": True,
            "week_completed": False
        })
    
    # Special handling: After Q13 is completed, show the closing messages sequentially
    if state.current_question == 14 and 13 in state.question_completed:
        # Initialize message index if not set
        if not hasattr(state, "q13_closing_message_index"):
            state.q13_closing_message_index = 0
        
        # Show first message
        if state.q13_closing_message_index == 0:
            state.q13_closing_message_index = 1
            return jsonify({
                "success": True,
                "message": WEEK1_AFTER_Q13_MESSAGE,
                "is_complete": False,
                "awaiting_response": False,
                "is_print_message": True,
                "week_completed": False
            })
        # Show second message
        elif state.q13_closing_message_index == 1:
            # Mark week as completed after showing the final message
            session_id = session.get('session_id')
            if session_id:
                progress_tracker.update_user_progress(
                    session_id, 
                    week_number=1,
                    question_number=len(QUESTIONS),
                    week_completed=True
                )
            # Set current_question to None to mark completion
            state.current_question = None
            return jsonify({
                "success": True,
                "message": WEEK1_FINAL_CLOSING_MESSAGE,
                "is_complete": True,
                "awaiting_response": False,
                "is_print_message": True,
                "week_completed": True
            })
    
    # If we've completed all questions, return completion message
    if state.current_question is None or state.current_question > len(QUESTIONS):
        # Mark week as completed in progress tracker
        session_id = session.get('session_id')
        if session_id:
            progress_tracker.update_user_progress(
                session_id, 
                week_number=1,
                question_number=len(QUESTIONS),
                week_completed=True
            )
        
        return jsonify({
            "success": True,
            "message": "Thank you for completing Week 1! Great job working through the first week's materials.",
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

    # Helpers for new questions
    q11_missing_requirements = []
    q12_valid = True
    q13_valid = True
    
    # For Q2, classify first, then use hardcoded logic based on scenario
    if question_number == 2:
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
        if "SCENARIO_1" in state.q2_scenario or "SCENARIO_3" in state.q2_scenario:
            # Scenario 1 or 3: Loop to verify homework answers
            scenario_num = "1" if "SCENARIO_1" in state.q2_scenario else "3"
            print(f"[DEBUG] SCENARIO {scenario_num} detected - entering homework verification loop")
            q2_prompts = SYSTEM_PROMPTS.get(question_number, {})
            
            # Check if this is the first time asking for homework
            if iteration == 0:
                # First time - ask for homework questions
                if "SCENARIO_1" in state.q2_scenario:
                    system_prompt = q2_prompts.get("scenario_1_ask", "").replace("{name}", state.name)
                else:  # Scenario 3
                    system_prompt = q2_prompts.get("scenario_3_ask", "").replace("{name}", state.name)
                system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
                nova_response = call_llm(system_prompt, user_message)
            else:
                # Subsequent iterations - user is providing homework answers
                # Check if they've already provided all 4 numbered answers
                # Build context for validation
                conversation_history = []
                if question_number in state.nova_responses:
                    conversation_history.extend(state.nova_responses[question_number])
                if question_number in state.answers:
                    conversation_history.extend(state.answers[question_number])
                
                # Check if user's latest response contains numbered answers (1. 2. 3. 4.)
                latest_response = user_message.lower()
                has_numbered_answers = any(f"{i}." in latest_response or f"{i})" in latest_response for i in [1, 2, 3, 4])
                
                if has_numbered_answers:
                    # User provided numbered answers - acknowledge and confirm completeness
                    system_prompt = f"""You are a professional career coach. The user's name is {state.name}. 
The user has provided numbered responses (1. 2. 3. 4.) to the 4 homework questions you asked. Acknowledge their responses positively and confirm that they've completed the homework exercise. Be encouraging and warm."""
                else:
                    # User is providing additional information - acknowledge and continue
                    system_prompt = f"""You are a professional career coach. The user's name is {state.name}. 
You asked the user to answer 4 homework questions. The user is providing additional information. Acknowledge their responses and ask for any remaining information if needed."""
                system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
                nova_response = call_llm(system_prompt, user_message)
                
        elif "SCENARIO_2" in state.q2_scenario:
            # Scenario 2: Congratulate and move on
            print(f"[DEBUG] SCENARIO 2 detected - congratulating and moving on")
            q2_prompts = SYSTEM_PROMPTS.get(question_number, {})
            system_prompt = q2_prompts.get("scenario_2_congratulate", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 2 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 2")
            state.q2_scenario = "SCENARIO_2"
            q2_prompts = SYSTEM_PROMPTS.get(question_number, {})
            system_prompt = q2_prompts.get("scenario_2_congratulate", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
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
            # Scenario 1: Congratulate for optimistic perspective
            print(f"[DEBUG] SCENARIO 1 detected - congratulating user")
            system_prompt = q3_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q3_scenario:
            # Scenario 2: Provide summary of why thinking flexibly helps
            print(f"[DEBUG] SCENARIO 2 detected - providing summary of benefits")
            
            # Get user's answer to Q1 (reason for participating)
            q1_answer = ""
            if 1 in state.answers and state.answers[1]:
                q1_answer = " ".join(state.answers[1])
            
            # Build the prompt with Q1 answer
            system_prompt = q3_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            
            # Replace placeholder for Q1 answer if it exists
            # The prompt mentions "{Answer to 2 (reason for participating in DRIVEN)}" but it's actually Q1
            placeholder = "{Answer to 2 (reason for participating in DRIVEN)}"
            if q1_answer:
                system_prompt = system_prompt.replace(placeholder, q1_answer)
            else:
                # Remove placeholder if no Q1 answer
                system_prompt = system_prompt.replace(placeholder, "")
            
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 1 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 1")
            state.q3_scenario = "SCENARIO_1"
            system_prompt = q3_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
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
            # Scenario 1: User selected a specific goal - acknowledge it
            print(f"[DEBUG] SCENARIO 1 detected - acknowledging goal selection")
            system_prompt = q4_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q4_scenario:
            # Scenario 2: User did not select a goal - ask them to choose
            print(f"[DEBUG] SCENARIO 2 detected - asking user to select a goal")
            
            # Get user's answer to Q1 (reason for participating)
            q1_answer = ""
            if 1 in state.answers and state.answers[1]:
                q1_answer = " ".join(state.answers[1])
            
            # Build the prompt with Q1 answer and goal categories
            system_prompt = q4_prompts.get("scenario_2_prompt", "").replace("{name}", state.name)
            
            # Replace placeholder for Q1 answer if it exists
            if q1_answer:
                system_prompt = system_prompt.replace("{Response from 1}", q1_answer)
            else:
                # Remove placeholder if no Q1 answer
                system_prompt = system_prompt.replace("based on their reason for participating in DRIVEN (if they provided one). ", "")
                system_prompt = system_prompt.replace("Also suggest goals to set based on their reason for participating in DRIVEN (if they provided one). ", "")
            
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_3" in state.q4_scenario:
            # Scenario 3: User only named category - prompt for specific goal
            print(f"[DEBUG] SCENARIO 3 detected - prompting for specific goal within category")
            
            # Get user's answer to Q1 (reason for participating)
            q1_answer = ""
            if 1 in state.answers and state.answers[1]:
                q1_answer = " ".join(state.answers[1])
            
            # Build the prompt with Q1 answer and goal categories
            system_prompt = q4_prompts.get("scenario_3_followup", "").replace("{name}", state.name)
            
            # Replace placeholder for Q1 answer if it exists
            if q1_answer:
                system_prompt = system_prompt.replace("{Response from 1}", q1_answer)
            else:
                # Remove placeholder if no Q1 answer
                system_prompt = system_prompt.replace("Also provide suggestions aligned with their reason for participating in DRIVEN (if they provided one). ", "")
            
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 2 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 2")
            state.q4_scenario = "SCENARIO_2"
            q1_answer = ""
            if 1 in state.answers and state.answers[1]:
                q1_answer = " ".join(state.answers[1])
            system_prompt = q4_prompts.get("scenario_2_prompt", "").replace("{name}", state.name)
            if q1_answer:
                system_prompt = system_prompt.replace("{Response from 1}", q1_answer)
            else:
                system_prompt = system_prompt.replace("based on their reason for participating in DRIVEN (if they provided one). ", "")
                system_prompt = system_prompt.replace("Also suggest goals to set based on their reason for participating in DRIVEN (if they provided one). ", "")
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
    
    # For Q5, classify first, then use hardcoded logic based on scenario
    elif question_number == 5:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q5] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
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
        
        if "SCENARIO_1" in state.q5_scenario:
            # Scenario 1: User has scheduled the goal - congratulate
            print(f"[DEBUG] SCENARIO 1 detected - congratulating user for scheduling")
            system_prompt = q5_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q5_scenario:
            # Scenario 2: User has not scheduled - encourage them
            print(f"[DEBUG] SCENARIO 2 detected - encouraging user to schedule")
            system_prompt = q5_prompts.get("scenario_2_prompt", "").replace("{name}", state.name)
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 2 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 2")
            state.q5_scenario = "SCENARIO_2"
            system_prompt = q5_prompts.get("scenario_2_prompt", "").replace("{name}", state.name)
            nova_response = call_llm(system_prompt, user_message)
    
    # For Q6, classify first, then use hardcoded logic based on scenario
    elif question_number == 6:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q6] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Step 1: Classify the scenario (only on first response to Q6)
        if state.q6_scenario is None:
            q6_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q6_prompts, dict) or "classifier" not in q6_prompts:
                return jsonify({"success": False, "error": "Q6 prompts not configured correctly"}), 400
            
            classifier_prompt = q6_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q6...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q6_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q6_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q6_prompts = SYSTEM_PROMPTS.get(question_number, {})
        
        if "SCENARIO_1" in state.q6_scenario:
            # Scenario 1: User doesn't know how to achieve goal - assist them
            iteration = state.get_iteration(question_number)
            
            if iteration == 0:
                # First iteration: User said they don't know - provide breakdown
                print(f"[DEBUG] SCENARIO 1 detected - assisting user with goal breakdown (iteration {iteration})")
                
                # Get user's answer to Q4 (goal selected)
                q4_answer = ""
                if 4 in state.answers and state.answers[4]:
                    q4_answer = " ".join(state.answers[4])
                
                # Build the prompt with Q4 answer
                system_prompt = q6_prompts.get("scenario_1_assist", "").replace("{name}", state.name)
                
                # Replace placeholder for Q4 answer if it exists
                if q4_answer:
                    system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
                else:
                    # Remove placeholder if no Q4 answer
                    system_prompt = system_prompt.replace("({Answer to question 4 (goal selected)})", "")
                
                nova_response = call_llm(system_prompt, user_message)
            else:
                # Subsequent iterations: User is responding to NOVA's breakdown
                # Acknowledge their response and encourage them to select a specific step if they haven't
                print(f"[DEBUG] SCENARIO 1 - User responding to breakdown (iteration {iteration})")
                
                # Check if user provided a concrete step
                system_prompt = f"""You are a professional career coach. The user's name is {state.name}. 
You just provided them with a breakdown of steps to achieve their goal. The user has responded.

Acknowledge their response. If they've selected a specific step or mentioned a concrete action they will take, confirm it positively and encourage them. If their response is vague or they haven't selected a specific step yet, gently encourage them to pick one of the suggestions you provided and specify when/where they'll do it.

Be warm and supportive."""
                
                nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q6_scenario:
            # Scenario 2: User provided clear steps - reinforce and provide SMART feedback
            print(f"[DEBUG] SCENARIO 2 detected - reinforcing user's plan")
            system_prompt = q6_prompts.get("scenario_2_reinforce", "").replace("{name}", state.name)
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 2 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 2")
            state.q6_scenario = "SCENARIO_2"
            system_prompt = q6_prompts.get("scenario_2_reinforce", "").replace("{name}", state.name)
            nova_response = call_llm(system_prompt, user_message)
    
    # For Q7, classify first, then use hardcoded logic based on scenario
    elif question_number == 7:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q7] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Step 1: Classify the scenario (only on first response to Q7)
        if state.q7_scenario is None:
            q7_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q7_prompts, dict) or "classifier" not in q7_prompts:
                return jsonify({"success": False, "error": "Q7 prompts not configured correctly"}), 400
            
            classifier_prompt = q7_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q7...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q7_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q7_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q7_prompts = SYSTEM_PROMPTS.get(question_number, {})
        
        if "SCENARIO_1" in state.q7_scenario:
            # Scenario 1: User presented barriers - acknowledge and motivate
            print(f"[DEBUG] SCENARIO 1 detected - acknowledging barriers and providing motivation")
            
            # Get user's answers to Q4 and Q6
            q4_answer = ""
            if 4 in state.answers and state.answers[4]:
                q4_answer = " ".join(state.answers[4])
            
            q6_answer = ""
            if 6 in state.answers and state.answers[6]:
                q6_answer = " ".join(state.answers[6])
            
            # Build the prompt with Q4 and Q6 answers
            system_prompt = q7_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            
            # Replace placeholders for Q4 and Q6 answers
            if q4_answer:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
                system_prompt = system_prompt.replace("{Answer to question 4}", q4_answer)
            else:
                # Remove placeholder if no Q4 answer
                system_prompt = system_prompt.replace("their goal ({Answer to question 4 (goal selected)})", "their goal")
                system_prompt = system_prompt.replace("their goal ({Answer to question 4})", "their goal")
            
            if q6_answer:
                system_prompt = system_prompt.replace("{Answer to question 6 (steps to achieving goal)}", q6_answer)
                system_prompt = system_prompt.replace("{Answer to question 6}", q6_answer)
            else:
                # Remove placeholder if no Q6 answer
                system_prompt = system_prompt.replace("Based on their steps to achieve this goal ({Answer to question 6 (steps to achieving goal)}), ", "")
                system_prompt = system_prompt.replace("Based on their steps to achieve this goal ({Answer to question 6}), ", "")
            
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q7_scenario:
            # Scenario 2: User doesn't see barriers - provide clarifying questions
            # Check if this is the first iteration (asking clarifying questions) or subsequent (user responded)
            iteration = state.get_iteration(question_number)
            
            if iteration == 0:
                # First iteration: NOVA provides clarifying questions
                print(f"[DEBUG] SCENARIO 2 detected - providing clarifying questions (iteration {iteration})")
                
                # Get user's answers to Q4 and Q6
                q4_answer = ""
                if 4 in state.answers and state.answers[4]:
                    q4_answer = " ".join(state.answers[4])
                
                q6_answer = ""
                if 6 in state.answers and state.answers[6]:
                    q6_answer = " ".join(state.answers[6])
                
                print(f"[DEBUG Q7 Scenario 2] Q4 answer: {q4_answer[:100] if q4_answer else 'None'}...")
                print(f"[DEBUG Q7 Scenario 2] Q6 answer: {q6_answer[:100] if q6_answer else 'None'}...")
                
                # Build the prompt with Q4 and Q6 answers
                system_prompt = q7_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
                
                # Replace placeholders for Q4 answer (handle both with and without parentheses)
                if q4_answer:
                    system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
                    system_prompt = system_prompt.replace("({Answer to question 4 (goal selected)})", f"({q4_answer})")
                    system_prompt = system_prompt.replace("{Answer to question 4}", q4_answer)
                else:
                    # Remove placeholder if no Q4 answer
                    system_prompt = system_prompt.replace("their goal ({Answer to question 4 (goal selected)})", "their goal")
                    system_prompt = system_prompt.replace("their goal {Answer to question 4 (goal selected)}", "their goal")
                    system_prompt = system_prompt.replace("their goal ({Answer to question 4})", "their goal")
                    system_prompt = system_prompt.replace("their goal {Answer to question 4}", "their goal")
                
                # Replace placeholders for Q6 answer (handle both with and without parentheses)
                if q6_answer:
                    system_prompt = system_prompt.replace("{Answer to question 6 (steps to achieving goal)}", q6_answer)
                    system_prompt = system_prompt.replace("({Answer to question 6 (steps to achieving goal)})", f"({q6_answer})")
                    system_prompt = system_prompt.replace("{Answer to question 6}", q6_answer)
                else:
                    # Remove placeholder if no Q6 answer
                    system_prompt = system_prompt.replace("their steps ({Answer to question 6 (steps to achieving goal)})", "their steps")
                    system_prompt = system_prompt.replace("based on their ({Answer to question 6 (steps to achieving goal)})", "based on their steps")
                    system_prompt = system_prompt.replace("based on their {Answer to question 6 (steps to achieving goal)}", "based on their steps")
                    system_prompt = system_prompt.replace("their steps ({Answer to question 6})", "their steps")
                    system_prompt = system_prompt.replace("based on their ({Answer to question 6})", "based on their steps")
                    system_prompt = system_prompt.replace("based on their {Answer to question 6}", "based on their steps")
                
                # Add no-followups instruction
                system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
                
                nova_response = call_llm(system_prompt, user_message)
            else:
                # Subsequent iterations: User has responded with a barrier - just praise and move on
                print(f"[DEBUG] SCENARIO 2 - User responded with barrier (iteration {iteration}) - praising and moving on")
                
                system_prompt = f"""Imagine that you are a trained career coach that helps adults with mental health issues learn how to think more flexibly. The user's name is {state.name}. 

The user has provided a response about barriers they might encounter. Acknowledge their response positively with praise and motivation. Be warm and encouraging. 

CRITICAL: Do NOT ask any follow-up questions. Simply provide praise and motivation, then conclude your response. The goal is to move on to the next question."""
                
                nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 1 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 1")
            state.q7_scenario = "SCENARIO_1"
            q4_answer = ""
            if 4 in state.answers and state.answers[4]:
                q4_answer = " ".join(state.answers[4])
            q6_answer = ""
            if 6 in state.answers and state.answers[6]:
                q6_answer = " ".join(state.answers[6])
            system_prompt = q7_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            if q4_answer:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
                system_prompt = system_prompt.replace("{Answer to question 4}", q4_answer)
            else:
                system_prompt = system_prompt.replace("their goal ({Answer to question 4 (goal selected)})", "their goal")
                system_prompt = system_prompt.replace("their goal ({Answer to question 4})", "their goal")
            if q6_answer:
                system_prompt = system_prompt.replace("{Answer to question 6 (steps to achieving goal)}", q6_answer)
                system_prompt = system_prompt.replace("{Answer to question 6}", q6_answer)
            else:
                system_prompt = system_prompt.replace("Based on their steps to achieve this goal ({Answer to question 6 (steps to achieving goal)}), ", "")
                system_prompt = system_prompt.replace("Based on their steps to achieve this goal ({Answer to question 6}), ", "")
            nova_response = call_llm(system_prompt, user_message)
    
    # For Q8, classify first, then use hardcoded logic based on scenario
    elif question_number == 8:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q8] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Step 1: Classify the scenario (only on first response to Q8)
        if state.q8_scenario is None:
            q8_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q8_prompts, dict) or "classifier" not in q8_prompts:
                return jsonify({"success": False, "error": "Q8 prompts not configured correctly"}), 400
            
            classifier_prompt = q8_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q8...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q8_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q8_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q8_prompts = SYSTEM_PROMPTS.get(question_number, {})
        
        if "SCENARIO_1" in state.q8_scenario:
            # Scenario 1: User provided confident strategies - reinforce and summarize
            print(f"[DEBUG] SCENARIO 1 detected - reinforcing user's strategies and summarizing plan")
            
            # Get user's answers to Q4, Q6, and Q7
            q4_answer = ""
            if 4 in state.answers and state.answers[4]:
                q4_answer = " ".join(state.answers[4])
            
            q6_answer = ""
            if 6 in state.answers and state.answers[6]:
                q6_answer = " ".join(state.answers[6])
            
            q7_answer = ""
            if 7 in state.answers and state.answers[7]:
                q7_answer = " ".join(state.answers[7])
            
            print(f"[DEBUG Q8 Scenario 1] Q4 answer: {q4_answer[:100] if q4_answer else 'None'}...")
            print(f"[DEBUG Q8 Scenario 1] Q6 answer: {q6_answer[:100] if q6_answer else 'None'}...")
            print(f"[DEBUG Q8 Scenario 1] Q7 answer: {q7_answer[:100] if q7_answer else 'None'}...")
            
            # Build the prompt with Q4, Q6, and Q7 answers
            system_prompt = q8_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            
            # Replace placeholders for Q4 answer
            if q4_answer:
                system_prompt = system_prompt.replace("{Answer to questions 4 (goal selected)}", q4_answer)
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
                system_prompt = system_prompt.replace("({Answer to questions 4 (goal selected)})", f"({q4_answer})")
                system_prompt = system_prompt.replace("({Answer to question 4 (goal selected)})", f"({q4_answer})")
                system_prompt = system_prompt.replace("{Answer to question 4}", q4_answer)
            else:
                system_prompt = system_prompt.replace("their goal ({Answer to questions 4 (goal selected)})", "their goal")
                system_prompt = system_prompt.replace("their goal {Answer to questions 4 (goal selected)}", "their goal")
            
            # Replace placeholders for Q6 answer
            if q6_answer:
                system_prompt = system_prompt.replace("{Answer to questions 6 (steps to achieving goal)}", q6_answer)
                system_prompt = system_prompt.replace("{Answer to question 6 (steps to achieving goal)}", q6_answer)
                system_prompt = system_prompt.replace("({Answer to questions 6 (steps to achieving goal)})", f"({q6_answer})")
                system_prompt = system_prompt.replace("({Answer to question 6 (steps to achieving goal)})", f"({q6_answer})")
                system_prompt = system_prompt.replace("{Answer to question 6}", q6_answer)
            else:
                system_prompt = system_prompt.replace("their steps ({Answer to questions 6 (steps to achieving goal)})", "their steps")
                system_prompt = system_prompt.replace("based on their ({Answer to questions 6 (steps to achieving goal)})", "based on their steps")
            
            # Replace placeholders for Q7 answer
            if q7_answer:
                system_prompt = system_prompt.replace("{Answers to questions 7 (expected barriers)}", q7_answer)
                system_prompt = system_prompt.replace("{Answers to question 7 (expected barriers)}", q7_answer)
                system_prompt = system_prompt.replace("({Answers to questions 7 (expected barriers)})", f"({q7_answer})")
                system_prompt = system_prompt.replace("({Answers to question 7 (expected barriers)})", f"({q7_answer})")
                system_prompt = system_prompt.replace("{{Answers to question 7 (expected barriers)}}", q7_answer)
                system_prompt = system_prompt.replace("{Answer to question 7}", q7_answer)
            else:
                system_prompt = system_prompt.replace("their barriers ({Answers to questions 7 (expected barriers)})", "their barriers")
                system_prompt = system_prompt.replace("their barriers {Answers to questions 7 (expected barriers)}", "their barriers")
                system_prompt = system_prompt.replace("{{Answers to question 7 (expected barriers)}}", "their barriers")
            
            # Add no-followups instruction
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q8_scenario:
            # Scenario 2: User is unsure - provide tailored suggestions
            print(f"[DEBUG] SCENARIO 2 detected - providing tailored suggestions for overcoming barriers")
            
            # Get user's answers to Q4, Q6, and Q7
            q4_answer = ""
            if 4 in state.answers and state.answers[4]:
                q4_answer = " ".join(state.answers[4])
            
            q6_answer = ""
            if 6 in state.answers and state.answers[6]:
                q6_answer = " ".join(state.answers[6])
            
            q7_answer = ""
            if 7 in state.answers and state.answers[7]:
                q7_answer = " ".join(state.answers[7])
            
            print(f"[DEBUG Q8 Scenario 2] Q4 answer: {q4_answer[:100] if q4_answer else 'None'}...")
            print(f"[DEBUG Q8 Scenario 2] Q6 answer: {q6_answer[:100] if q6_answer else 'None'}...")
            print(f"[DEBUG Q8 Scenario 2] Q7 answer: {q7_answer[:100] if q7_answer else 'None'}...")
            
            # Build the prompt with Q4, Q6, and Q7 answers
            system_prompt = q8_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            
            # Replace placeholders for Q4 answer
            if q4_answer:
                system_prompt = system_prompt.replace("{Answer to questions 4 (goal selected)}", q4_answer)
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
                system_prompt = system_prompt.replace("({Answer to questions 4 (goal selected)})", f"({q4_answer})")
                system_prompt = system_prompt.replace("({Answer to question 4 (goal selected)})", f"({q4_answer})")
                system_prompt = system_prompt.replace("{Answer to question 4}", q4_answer)
            else:
                system_prompt = system_prompt.replace("their goal ({Answer to questions 4 (goal selected)})", "their goal")
                system_prompt = system_prompt.replace("their goal {Answer to questions 4 (goal selected)}", "their goal")
            
            # Replace placeholders for Q6 answer
            if q6_answer:
                system_prompt = system_prompt.replace("{Answer to questions 6 (steps to achieving goal)}", q6_answer)
                system_prompt = system_prompt.replace("{Answer to question 6 (steps to achieving goal)}", q6_answer)
                system_prompt = system_prompt.replace("({Answer to questions 6 (steps to achieving goal)})", f"({q6_answer})")
                system_prompt = system_prompt.replace("({Answer to question 6 (steps to achieving goal)})", f"({q6_answer})")
                system_prompt = system_prompt.replace("{Answer to question 6}", q6_answer)
            else:
                system_prompt = system_prompt.replace("their steps ({Answer to questions 6 (steps to achieving goal)})", "their steps")
                system_prompt = system_prompt.replace("based on their ({Answer to questions 6 (steps to achieving goal)})", "based on their steps")
            
            # Replace placeholders for Q7 answer
            if q7_answer:
                system_prompt = system_prompt.replace("{Answers to questions 7 (expected barriers)}", q7_answer)
                system_prompt = system_prompt.replace("{Answers to question 7 (expected barriers)}", q7_answer)
                system_prompt = system_prompt.replace("({Answers to questions 7 (expected barriers)})", f"({q7_answer})")
                system_prompt = system_prompt.replace("({Answers to question 7 (expected barriers)})", f"({q7_answer})")
                system_prompt = system_prompt.replace("{{Answers to question 7 (expected barriers)}}", q7_answer)
                system_prompt = system_prompt.replace("{Answer to question 7}", q7_answer)
            else:
                system_prompt = system_prompt.replace("their barriers ({Answers to questions 7 (expected barriers)})", "their barriers")
                system_prompt = system_prompt.replace("their barriers {Answers to questions 7 (expected barriers)}", "their barriers")
                system_prompt = system_prompt.replace("{{Answers to question 7 (expected barriers)}}", "their barriers")
            
            # Add no-followups instruction
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 2 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 2")
            state.q8_scenario = "SCENARIO_2"
            q4_answer = ""
            if 4 in state.answers and state.answers[4]:
                q4_answer = " ".join(state.answers[4])
            q6_answer = ""
            if 6 in state.answers and state.answers[6]:
                q6_answer = " ".join(state.answers[6])
            q7_answer = ""
            if 7 in state.answers and state.answers[7]:
                q7_answer = " ".join(state.answers[7])
            system_prompt = q8_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            if q4_answer:
                system_prompt = system_prompt.replace("{Answer to questions 4 (goal selected)}", q4_answer)
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
                system_prompt = system_prompt.replace("({Answer to questions 4 (goal selected)})", f"({q4_answer})")
            else:
                system_prompt = system_prompt.replace("their goal ({Answer to questions 4 (goal selected)})", "their goal")
            if q6_answer:
                system_prompt = system_prompt.replace("{Answer to questions 6 (steps to achieving goal)}", q6_answer)
                system_prompt = system_prompt.replace("{Answer to question 6 (steps to achieving goal)}", q6_answer)
                system_prompt = system_prompt.replace("({Answer to questions 6 (steps to achieving goal)})", f"({q6_answer})")
            else:
                system_prompt = system_prompt.replace("their steps ({Answer to questions 6 (steps to achieving goal)})", "their steps")
            if q7_answer:
                system_prompt = system_prompt.replace("{Answers to questions 7 (expected barriers)}", q7_answer)
                system_prompt = system_prompt.replace("{Answers to question 7 (expected barriers)}", q7_answer)
                system_prompt = system_prompt.replace("({Answers to questions 7 (expected barriers)})", f"({q7_answer})")
                system_prompt = system_prompt.replace("{{Answers to question 7 (expected barriers)}}", q7_answer)
            else:
                system_prompt = system_prompt.replace("their barriers ({Answers to questions 7 (expected barriers)})", "their barriers")
                system_prompt = system_prompt.replace("{{Answers to question 7 (expected barriers)}}", "their barriers")
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
    
    # For Q9, classify first, then use hardcoded logic based on scenario
    elif question_number == 9:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q9] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Initialize q9_scenario if not already set
        if not hasattr(state, 'q9_scenario'):
            state.q9_scenario = None
        
        # Step 1: Classify the scenario (only on first response to Q9)
        if state.q9_scenario is None:
            q9_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q9_prompts, dict) or "classifier" not in q9_prompts:
                return jsonify({"success": False, "error": "Q9 prompts not configured correctly"}), 400
            
            classifier_prompt = q9_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q9...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q9_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q9_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q9_prompts = SYSTEM_PROMPTS.get(question_number, {})
        
        # Get user's answer to Q1 (reason for participating)
        q1_answer = ""
        if 1 in state.answers and state.answers[1]:
            q1_answer = " ".join(state.answers[1])
        
        if "SCENARIO_1" in state.q9_scenario:
            # Scenario 1: User provided clear and realistic roles
            print(f"[DEBUG] SCENARIO 1 detected - reflecting back roles and connecting to goals")
            system_prompt = q9_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            if q1_answer:
                system_prompt = system_prompt.replace("{Answer to question 1 (reason for participating in DRIVEN)}", q1_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 1 (reason for participating in DRIVEN)}", "")
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q9_scenario:
            # Scenario 2: User is unsure or very broad
            print(f"[DEBUG] SCENARIO 2 detected - normalizing uncertainty and helping narrow down")
            system_prompt = q9_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_3" in state.q9_scenario:
            # Scenario 3: User described unrealistic roles
            print(f"[DEBUG] SCENARIO 3 detected - acknowledging interests and suggesting stepping-stone roles")
            system_prompt = q9_prompts.get("scenario_3_respond", "").replace("{name}", state.name)
            if q1_answer:
                system_prompt = system_prompt.replace("{Answer to question 1 (reason for participating in DRIVEN)}", q1_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 1 (reason for participating in DRIVEN)}", "")
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 1 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 1")
            state.q9_scenario = "SCENARIO_1"
            system_prompt = q9_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            if q1_answer:
                system_prompt = system_prompt.replace("{Answer to question 1 (reason for participating in DRIVEN)}", q1_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 1 (reason for participating in DRIVEN)}", "")
            nova_response = call_llm(system_prompt, user_message)
    
    # For Q10, classify first, then use hardcoded logic based on scenario
    elif question_number == 10:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q10] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Initialize q10_scenario if not already set
        if not hasattr(state, 'q10_scenario'):
            state.q10_scenario = None
        
        # Step 1: Classify the scenario (only on first response to Q10)
        if state.q10_scenario is None:
            q10_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q10_prompts, dict) or "classifier" not in q10_prompts:
                return jsonify({"success": False, "error": "Q10 prompts not configured correctly"}), 400
            
            classifier_prompt = q10_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q10...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q10_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q10_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q10_prompts = SYSTEM_PROMPTS.get(question_number, {})
        
        # Get user's answer to Q9 (types of roles)
        q9_answer = ""
        if 9 in state.answers and state.answers[9]:
            q9_answer = " ".join(state.answers[9])
        
        if "SCENARIO_1" in state.q10_scenario:
            # Scenario 1: User identified concrete challenges
            print(f"[DEBUG] SCENARIO 1 detected - validating challenges and offering next steps")
            system_prompt = q10_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            if q9_answer:
                system_prompt = system_prompt.replace("{Answer to question 9 (types of roles you are looking for)}", q9_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 9 (types of roles you are looking for)}", "")
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q10_scenario:
            # Scenario 2: User feels everything is hard or discouraged
            print(f"[DEBUG] SCENARIO 2 detected - acknowledging overwhelm and breaking down challenges")
            system_prompt = q10_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            if q9_answer:
                system_prompt = system_prompt.replace("{Answer to question 9 (types of roles you are looking for)}", q9_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 9 (types of roles you are looking for)}", "")
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_3" in state.q10_scenario:
            # Scenario 3: User feels confident
            print(f"[DEBUG] SCENARIO 3 detected - affirming confidence and encouraging preparation")
            system_prompt = q10_prompts.get("scenario_3_respond", "").replace("{name}", state.name)
            if q9_answer:
                system_prompt = system_prompt.replace("{Answer to question 9 (types of roles you are looking for)}", q9_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 9 (types of roles you are looking for)}", "")
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 1 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 1")
            state.q10_scenario = "SCENARIO_1"
            system_prompt = q10_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            if q9_answer:
                system_prompt = system_prompt.replace("{Answer to question 9 (types of roles you are looking for)}", q9_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 9 (types of roles you are looking for)}", "")
            nova_response = call_llm(system_prompt, user_message)
    
    # For Q11, classify first, then use hardcoded logic based on scenario
    elif question_number == 11:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q11] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Initialize q11_scenario if not already set
        if not hasattr(state, 'q11_scenario'):
            state.q11_scenario = None
        
        # Step 1: Classify the scenario (only on first response to Q11)
        if state.q11_scenario is None:
            q11_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q11_prompts, dict) or "classifier" not in q11_prompts:
                return jsonify({"success": False, "error": "Q11 prompts not configured correctly"}), 400
            
            classifier_prompt = q11_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q11...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q11_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q11_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q11_prompts = SYSTEM_PROMPTS.get(question_number, {})
        
        # Get user's answers to Q4 (goal) and Q6 (steps)
        q4_answer = ""
        if 4 in state.answers and state.answers[4]:
            q4_answer = " ".join(state.answers[4])
        
        q6_answer = ""
        if 6 in state.answers and state.answers[6]:
            q6_answer = " ".join(state.answers[6])
        
        if "SCENARIO_1" in state.q11_scenario:
            # Scenario 1: Practical/situational barriers
            print(f"[DEBUG] SCENARIO 1 detected - normalizing barriers and suggesting adjustments")
            system_prompt = q11_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            if q4_answer:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", "")
            if q6_answer:
                system_prompt = system_prompt.replace("{Answer to question 6 (steps to achieving goal)}", q6_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 6 (steps to achieving goal)}", "")
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q11_scenario:
            # Scenario 2: Serious mental health crisis or danger
            print(f"[DEBUG] SCENARIO 2 detected - responding with concern and safety resources")
            system_prompt = q11_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_3" in state.q11_scenario:
            # Scenario 3: No major barriers
            print(f"[DEBUG] SCENARIO 3 detected - acknowledging and encouraging planning")
            system_prompt = q11_prompts.get("scenario_3_respond", "").replace("{name}", state.name)
            if q4_answer:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", "")
            if q6_answer:
                system_prompt = system_prompt.replace("{Answer to question 6 (steps to achieving goal)}", q6_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 6 (steps to achieving goal)}", "")
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 1 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 1")
            state.q11_scenario = "SCENARIO_1"
            system_prompt = q11_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            if q4_answer:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", "")
            if q6_answer:
                system_prompt = system_prompt.replace("{Answer to question 6 (steps to achieving goal)}", q6_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 6 (steps to achieving goal)}", "")
            nova_response = call_llm(system_prompt, user_message)
    
    # For Q12, classify first, then use hardcoded logic based on scenario
    elif question_number == 12:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q12] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Initialize q12_scenario if not already set
        if not hasattr(state, 'q12_scenario'):
            state.q12_scenario = None
        
        # Step 1: Classify the scenario (only on first response to Q12)
        if state.q12_scenario is None:
            q12_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q12_prompts, dict) or "classifier" not in q12_prompts:
                return jsonify({"success": False, "error": "Q12 prompts not configured correctly"}), 400
            
            classifier_prompt = q12_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q12...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q12_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q12_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q12_prompts = SYSTEM_PROMPTS.get(question_number, {})
        
        # Get user's answers to Q1 (reason for participating) and Q4 (goal selected)
        q1_answer = ""
        if 1 in state.answers and state.answers[1]:
            q1_answer = " ".join(state.answers[1])
        
        q4_answer = ""
        if 4 in state.answers and state.answers[4]:
            q4_answer = " ".join(state.answers[4])
        
        if "SCENARIO_1" in state.q12_scenario:
            # Scenario 1: User feels more motivated
            print(f"[DEBUG] SCENARIO 1 detected - celebrating motivation and connecting to goals")
            system_prompt = q12_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            if q1_answer:
                system_prompt = system_prompt.replace("{Answer to question 1 (reason for participating in DRIVEN)}", q1_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 1 (reason for participating in DRIVEN)}", "")
            if q4_answer:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", "")
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q12_scenario:
            # Scenario 2: User has low motivation
            print(f"[DEBUG] SCENARIO 2 detected - validating and normalizing, suggesting small steps")
            system_prompt = q12_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            if q4_answer:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", "")
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_3" in state.q12_scenario:
            # Scenario 3: User has mixed/uncertain feelings
            print(f"[DEBUG] SCENARIO 3 detected - acknowledging ambivalence, encouraging small steps")
            system_prompt = q12_prompts.get("scenario_3_respond", "").replace("{name}", state.name)
            if q4_answer:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", "")
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 1 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 1")
            state.q12_scenario = "SCENARIO_1"
            system_prompt = q12_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            if q1_answer:
                system_prompt = system_prompt.replace("{Answer to question 1 (reason for participating in DRIVEN)}", q1_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 1 (reason for participating in DRIVEN)}", "")
            if q4_answer:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", q4_answer)
            else:
                system_prompt = system_prompt.replace("{Answer to question 4 (goal selected)}", "")
            nova_response = call_llm(system_prompt, user_message)
    
    # For Q13, classify first, then use hardcoded logic based on scenario
    elif question_number == 13:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q13] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Initialize q13_scenario if not already set
        if not hasattr(state, 'q13_scenario'):
            state.q13_scenario = None
        
        # Step 1: Classify the scenario (only on first response to Q13)
        if state.q13_scenario is None:
            q13_prompts = SYSTEM_PROMPTS.get(question_number, {})
            if not isinstance(q13_prompts, dict) or "classifier" not in q13_prompts:
                return jsonify({"success": False, "error": "Q13 prompts not configured correctly"}), 400
            
            classifier_prompt = q13_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q13...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.1
            )
            
            state.q13_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Scenario classification result: {state.q13_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q13_prompts = SYSTEM_PROMPTS.get(question_number, {})
        
        # Generate summary of user's priorities and challenges from Q1-12
        user_summary = generate_user_summary(state)
        
        if "SCENARIO_1" in state.q13_scenario:
            # Scenario 1: User provided what they want to talk about
            print(f"[DEBUG] SCENARIO 1 detected - affirming their topics and offering examples")
            system_prompt = q13_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            system_prompt = system_prompt.replace("{Summary of user's priorities and challenges from questions 1–12}", user_summary)
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q13_scenario:
            # Scenario 2: User said "No"
            print(f"[DEBUG] SCENARIO 2 detected - normalizing and offering to suggest topics")
            system_prompt = q13_prompts.get("scenario_2_respond", "").replace("{name}", state.name)
            system_prompt = system_prompt.replace("{Summary of user's priorities and challenges from questions 1–12}", user_summary)
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_3" in state.q13_scenario:
            # Scenario 3: User's response is unclear
            print(f"[DEBUG] SCENARIO 3 detected - clarifying and providing examples")
            system_prompt = q13_prompts.get("scenario_3_respond", "").replace("{name}", state.name)
            system_prompt = system_prompt.replace("{Summary of user's priorities and challenges from questions 1–12}", user_summary)
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 1 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 1")
            state.q13_scenario = "SCENARIO_1"
            system_prompt = q13_prompts.get("scenario_1_respond", "").replace("{name}", state.name)
            system_prompt = system_prompt.replace("{Summary of user's priorities and challenges from questions 1–12}", user_summary)
            nova_response = call_llm(system_prompt, user_message)
            
    else:
        # For other questions (Q1), use the standard prompt
        system_prompt = SYSTEM_PROMPTS.get(question_number, "")
        if not system_prompt:
            return jsonify({"success": False, "error": "System prompt not found"}), 400
        # Format system prompt with name
        system_prompt = system_prompt.replace("{name}", state.name)
        nova_response = call_llm(system_prompt, user_message)
    
    # DEBUG: Print NOVA's response
    if question_number == 2:
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
    elif question_number == 7:
        print(f"[DEBUG PROCESS_RESPONSE Q7] NOVA's response: {nova_response}")
        print(f"{'#'*80}\n")
    elif question_number == 8:
        print(f"[DEBUG PROCESS_RESPONSE Q8] NOVA's response: {nova_response}")
        print(f"{'#'*80}\n")
    elif question_number == 9:
        print(f"[DEBUG PROCESS_RESPONSE Q9] NOVA's response: {nova_response}")
        print(f"{'#'*80}\n")
    elif question_number == 10:
        print(f"[DEBUG PROCESS_RESPONSE Q10] NOVA's response: {nova_response}")
        print(f"{'#'*80}\n")
    elif question_number == 11:
        print(f"[DEBUG PROCESS_RESPONSE Q11] NOVA's response: {nova_response}")
        print(f"{'#'*80}\n")
    elif question_number == 12:
        print(f"[DEBUG PROCESS_RESPONSE Q12] NOVA's response: {nova_response}")
        print(f"{'#'*80}\n")
    elif question_number == 13:
        print(f"[DEBUG PROCESS_RESPONSE Q13] NOVA's response: {nova_response}")
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
    # Pass q2_scenario to skip validation for Scenario 2
    # Pass q6_scenario and iteration to handle Scenario 1 properly
    is_complete, missing_items = validate_completeness(
        question_number, 
        nova_response, 
        state.answers.get(question_number, []),
        conversation_history,
        q2_scenario=state.q2_scenario if question_number == 2 else None,
        q6_scenario=state.q6_scenario if question_number == 6 else None,
        q6_iteration=iteration if question_number == 6 else None,
        q7_scenario=state.q7_scenario if question_number == 7 else None,
        q8_scenario=state.q8_scenario if question_number == 8 else None
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
        # Q1: Use validation only if NOVA isn't asking follow-ups, otherwise rely on iteration limit
        if not is_followup_question and not is_complete:
            # NOVA finished responding and validation says incomplete
            needs_followup = True
            move_to_next = False
            if missing_items and missing_items != "None" and missing_items != "Unknown":
                nova_response += f"\n\n[Note: I still need: {missing_items}]"
        elif is_followup_question and iteration < max_iterations:
            # NOVA is asking follow-up - continue naturally
            needs_followup = True
            move_to_next = False
        else:
            # Done with Q1
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
    elif question_number == 2:
        # Q2: Hardcoded logic based on scenario classification
        if state.q2_scenario and ("SCENARIO_1" in state.q2_scenario or "SCENARIO_3" in state.q2_scenario):
            # Scenario 1 or 3: Loop to verify all 4 homework questions are answered
            scenario_num = "1" if "SCENARIO_1" in state.q2_scenario else "3"
            print(f"[DEBUG] SCENARIO {scenario_num}: Validating homework completeness...")
            
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
                
        elif state.q2_scenario and "SCENARIO_2" in state.q2_scenario:
            # Scenario 2: Congratulate and move on immediately
            print(f"[DEBUG] SCENARIO 2: Congratulating and moving on")
            needs_followup = False
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
            
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
        # Check validation - if incomplete, ask for follow-up
        print(f"[DEBUG] Q4: Validating goal selection...")
        
        if not is_complete:
            # Goal incomplete - continue loop (scenarios 2 and 3 already asked for goal)
            needs_followup = True
            move_to_next = False
            if missing_items and missing_items != "None" and missing_items != "Unknown":
                nova_response += f"\n\n[Note: I still need: {missing_items}]"
            print(f"[DEBUG] Q4 incomplete - continuing to get goal")
        else:
            # Goal complete - move on
            print(f"[DEBUG] Q4: Goal selected - completing question and moving on")
            needs_followup = False
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
    elif question_number == 5:
        # Q5: Hardcoded logic based on scenario classification
        # If Scenario 1 (scheduled), praise and move on immediately (no follow-ups)
        if state.q5_scenario and "SCENARIO_1" in state.q5_scenario:
            print(f"[DEBUG] Q5 Scenario 1: Scheduled - praising and moving to next question")
            needs_followup = False
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
        else:
            # Otherwise (Scenario 2 or unknown), use validation
            print(f"[DEBUG] Q5: Validating response completeness...")
            if not is_complete:
                # Response incomplete - ask for follow-up
                needs_followup = True
                move_to_next = False
                if missing_items and missing_items != "None" and missing_items != "Unknown":
                    nova_response += f"\n\n[Note: I still need: {missing_items}]"
                print(f"[DEBUG] Q5 incomplete - continuing to get response")
            else:
                # Response complete - move on
                print(f"[DEBUG] Q5: Completing question and moving on")
                needs_followup = False
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
    elif question_number == 6:
        # Q6: Hardcoded logic based on scenario classification
        print(f"[DEBUG] Q6: Validating response completeness...")
        
        if state.q6_scenario and "SCENARIO_1" in state.q6_scenario:
            # Scenario 1: Check if user provided concrete steps after NOVA's breakdown
            iteration = state.get_iteration(question_number)
            
            if iteration == 0:
                # First iteration: NOVA just provided breakdown - wait for user response
                print(f"[DEBUG] Q6 Scenario 1: NOVA provided breakdown - waiting for user to provide concrete step")
                needs_followup = True
                move_to_next = False
            else:
                # Subsequent iterations: Check if user provided concrete step
                if not is_complete:
                    # User hasn't provided concrete step yet - continue loop
                    needs_followup = True
                    move_to_next = False
                    if missing_items and missing_items != "None" and missing_items != "Unknown":
                        nova_response += f"\n\n[Note: I still need: {missing_items}]"
                    print(f"[DEBUG] Q6 Scenario 1: User hasn't provided concrete step yet - continuing")
                else:
                    # User provided concrete step - complete and move on
                    print(f"[DEBUG] Q6 Scenario 1: User provided concrete step - completing question and moving on")
                    needs_followup = False
                    move_to_next = True
                    state.question_completed[question_number] = True
                    state.current_question = question_number + 1
        else:
            # Scenario 2: User provided clear steps initially - move on immediately after motivation
            print(f"[DEBUG] Q6 Scenario 2: Provided motivation - moving to next question")
            needs_followup = False
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
    elif question_number == 7:
        # Q7: Hardcoded logic based on scenario classification
        print(f"[DEBUG] Q7: Validating response completeness...")
        
        # Special handling for Scenario 2: After user responds with a barrier, move on immediately
        if state.q7_scenario and "SCENARIO_2" in state.q7_scenario:
            iteration = state.get_iteration(question_number)
            if iteration > 0:
                # User has responded to NOVA's clarifying questions - move on immediately
                print(f"[DEBUG] Q7 Scenario 2: User responded with barrier - moving to next question immediately")
                needs_followup = False
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
            else:
                # First iteration: NOVA just asked clarifying questions - wait for user response
                print(f"[DEBUG] Q7 Scenario 2: NOVA asked clarifying questions - waiting for user response")
                needs_followup = True
                move_to_next = False
        else:
            # Scenario 1 or unknown: Use standard validation
            if not is_complete:
                # Response incomplete - ask for follow-up
                needs_followup = True
                move_to_next = False
                if missing_items and missing_items != "None" and missing_items != "Unknown":
                    nova_response += f"\n\n[Note: I still need: {missing_items}]"
                print(f"[DEBUG] Q7 incomplete - continuing to get response")
            else:
                # Response complete - move on
                print(f"[DEBUG] Q7: Completing question and moving on")
                needs_followup = False
                move_to_next = True
                state.question_completed[question_number] = True
                state.current_question = question_number + 1
    elif question_number == 8:
        # Q8: Hardcoded logic based on scenario classification
        # Check validation - if incomplete, ask for follow-up
        print(f"[DEBUG] Q8: Validating response completeness...")
        
        if not is_complete:
            # Response incomplete - ask for follow-up
            needs_followup = True
            move_to_next = False
            if missing_items and missing_items != "None" and missing_items != "Unknown":
                nova_response += f"\n\n[Note: I still need: {missing_items}]"
            print(f"[DEBUG] Q8 incomplete - continuing to get response")
        else:
            # Response complete - move on
            print(f"[DEBUG] Q8: Completing question and moving on")
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
    
    # Q9 completion logic (after NOVA response is generated)
    if question_number == 9:
        # Check if NOVA is asking follow-up questions
        is_followup_question_q9 = "?" in nova_response
        
        if is_followup_question_q9 and iteration < max_iterations:
            # NOVA is asking follow-up - continue naturally
            needs_followup = True
            move_to_next = False
            print(f"[DEBUG] Q9: NOVA asking follow-up question - continuing")
        else:
            # Response complete - move on
            print(f"[DEBUG] Q9: Completing question and moving on")
            needs_followup = False
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
    elif question_number == 10:
        # Q10: Hardcoded logic based on scenario classification
        # Check if NOVA is asking follow-up questions
        print(f"[DEBUG] Q10: Validating response completeness...")
        
        is_followup_question = "?" in nova_response
        
        if is_followup_question and iteration < max_iterations:
            # NOVA is asking follow-up - continue naturally
            needs_followup = True
            move_to_next = False
            print(f"[DEBUG] Q10: NOVA asking follow-up question - continuing")
        else:
            # Response complete - move on
            print(f"[DEBUG] Q10: Completing question and moving on")
            needs_followup = False
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
    elif question_number == 11:
        # Q11: Hardcoded logic based on scenario classification
        # Check if NOVA is asking follow-up questions
        print(f"[DEBUG] Q11: Validating response completeness...")
        
        is_followup_question = "?" in nova_response
        
        if is_followup_question and iteration < max_iterations:
            # NOVA is asking follow-up - continue naturally
            needs_followup = True
            move_to_next = False
            print(f"[DEBUG] Q11: NOVA asking follow-up question - continuing")
        else:
            # Response complete - move on
            print(f"[DEBUG] Q11: Completing question and moving on")
            needs_followup = False
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
    elif question_number == 12:
        # Q12: Hardcoded logic based on scenario classification
        # Check if NOVA is asking follow-up questions
        print(f"[DEBUG] Q12: Validating response completeness...")
        
        is_followup_question = "?" in nova_response
        
        if is_followup_question and iteration < max_iterations:
            # NOVA is asking follow-up - continue naturally
            needs_followup = True
            move_to_next = False
            print(f"[DEBUG] Q12: NOVA asking follow-up question - continuing")
        else:
            # Response complete - move on
            print(f"[DEBUG] Q12: Completing question and moving on")
            needs_followup = False
            move_to_next = True
            state.question_completed[question_number] = True
            state.current_question = question_number + 1
    elif question_number == 13:
        # Q13: Hardcoded logic based on scenario classification
        # Check if NOVA is asking follow-up questions
        print(f"[DEBUG] Q13: Validating response completeness...")
        
        is_followup_question = "?" in nova_response
        
        if is_followup_question and iteration < max_iterations:
            # NOVA is asking follow-up - continue naturally
            needs_followup = True
            move_to_next = False
            print(f"[DEBUG] Q13: NOVA asking follow-up question - continuing")
        else:
            # Response complete - move on
            print(f"[DEBUG] Q13: Completing question and moving on")
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
    
    # Save progress when question is completed
    if move_to_next and question_number in state.question_completed:
        save_question_progress(question_number, week_number=1)
        
        # Check if all questions are completed (week is done)
        all_completed = all(state.question_completed.get(q, False) for q in QUESTIONS.keys())
        if all_completed:
            session_id = session.get('session_id')
            if session_id:
                progress_tracker.update_user_progress(
                    session_id, 
                    week_number=1,
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
    print("Starting NOVA Career Coach Test Server (follow.py integration)...")
    print(f"Loaded {len(QUESTIONS)} test questions")
    
    port = int(os.getenv('PORT', 5001))
    
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

