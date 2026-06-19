import os
import uuid
import subprocess
import sys
from typing import List, Set
from flask import Flask, request, jsonify, session, Response
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from progress_tracker import progress_tracker

# Constants
WEEK_NUMBER = 3  # Week 3 backend

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
        self.current_question = 15  # Start with question 15
        self.answers = {}  # qnum -> list of user responses
        self.nova_responses = {}  # qnum -> list of NOVA responses
        self.iteration_count = {}  # qnum -> iteration count (for 2-iteration loop)
        self.question_completed = {}  # qnum -> bool (whether question is fully completed)
        self.pending_print_message = None  # Track if there's a print message to show before next question
        self.show_final_message = False  # Track if final message should be shown after Q22
        self.final_message_index = 0  # Track which final message index we're on
        self.print_messages_shown = set()  # Track which questions have had their print messages shown
        self.print_message_index = {}  # Track which print message index we're on for each question
        self.q17_categories_identified = set()  # Track which skill categories user has already covered
        self.q17_missing_categories = set(SKILL_CATEGORIES)  # Track remaining skill categories for Q17
        # Store scenario classifications for Q15-Q22
        self.q15_scenario = None
        self.q16_scenario = None
        self.q17_scenario = None
        self.q18_scenario = None
        self.q19_scenario = None
        self.q20_scenario = None
        self.q21_scenario = None
        self.q22_scenario = None
        
        # Data store for required variables
        self.data_store = {
            "Name": name,
            "skill_list": "hard skills, soft skills, technology skills, growth skills, and experiential skills",
        }
    
    def get_iteration(self, qnum):
        """Get current iteration count for a question."""
        return self.iteration_count.get(qnum, 0)
    
    def get_answer(self, qnum):
        """Get the last answer for a question number."""
        answers = self.answers.get(qnum, [])
        return answers[-1] if answers else ""
    
    def substitute_variables(self, text):
        """Substitute variables in text with actual values."""
        # Replace {Name} and {name}
        text = text.replace("{Name}", self.name)
        text = text.replace("{name}", self.name)
        
        # Replace data store variables
        for key, value in self.data_store.items():
            if isinstance(value, dict):
                formatted = ""
                for k, v in value.items():
                    if isinstance(v, list):
                        formatted += f"{k}:\n" + "\n".join(f"  - {item}" for item in v) + "\n"
                    else:
                        formatted += f"{k}: {v}\n"
                text = text.replace(f"{{{key}}}", formatted.strip())
            elif isinstance(value, list):
                formatted = "\n".join(f"* {item}" for item in value)
                text = text.replace(f"{{{key}}}", formatted)
            else:
                text = text.replace(f"{{{key}}}", str(value))
        
        # Replace answer references (e.g., {Answer to question 15})
        for qnum in range(15, 23):
            answer = self.get_answer(qnum)
            if answer:
                text = text.replace(f"{{Answer to question {qnum}}}", answer)
                text = text.replace(f"{{Answer to questions {qnum}}}", answer)
        
        # Replace {user motivation} with Q21 answer
        q21_answer = self.get_answer(21)
        if q21_answer:
            text = text.replace("{user motivation}", q21_answer)
        else:
            text = text.replace("{user motivation}", "their motivation")
        
        return text


def get_or_create_state(name=None):
    """Get or create a conversation state using Flask session."""
    session_id = session.get('session_id')
    if session_id is None:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    if session_id not in conversation_states:
        if name is None:
            name = "Friend"
        conversation_states[session_id] = ConversationState(name)
    return conversation_states[session_id]


def call_llm(system_prompt, user_message):
    """Call OpenAI API with system prompt and user message."""
    try:
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


def save_question_progress(question_number):
    """Persist question completion to the shared progress tracker."""
    session_id = session.get('session_id')
    if session_id:
        progress_tracker.update_user_progress(
            session_id,
            WEEK_NUMBER,
            question_number,
            completed=True
        )


def mark_week_completion():
    """Mark Week 3 as completed in the shared progress tracker."""
    session_id = session.get('session_id')
    if session_id:
        progress_tracker.update_user_progress(
            session_id,
            WEEK_NUMBER,
            question_number=22,
            completed=True,
            week_completed=True
        )


def validate_completeness(question_number, nova_response, user_responses, conversation_history, q15_scenario=None, q16_scenario=None):
    """
    Use LLM to validate if user has provided all required information.
    Returns (is_complete: bool, missing_items: str)
    """
    context = f"Question {question_number}: {QUESTIONS.get(question_number, '')}\n\n"
    context += f"NOVA's latest response: {nova_response}\n\n"
    context += f"User's responses so far: {len(user_responses)} response(s)\n"
    for i, resp in enumerate(user_responses, 1):
        context += f"  Response {i}: {resp[:200]}...\n"
    
    if question_number == 15:
        # For Q15, check if user provided a response about what stood out
        validation_prompt = """You are validating if a user has provided a response to "What stood out to you about the third week's content on skills?"

The user has provided at least one response. Determine if they have given ANY reasonable answer, even if it's brief.
- If they mention something specific that stood out → COMPLETE
- If they say nothing stood out → COMPLETE
- If they express confusion or discouragement → COMPLETE
- If they give any response related to the Week 3 skills content → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any response: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about what stood out from the skills content"
"""
    elif question_number == 16:
        # For Q16, check if user provided a response about confusion
        validation_prompt = """You are validating if a user has provided a response to "Was there anything that felt a little off or confusing?"

The user has provided at least one response. Determine if they have given ANY reasonable answer, even if it's brief.
- If they mention something confusing or unclear → COMPLETE
- If they say everything made sense → COMPLETE
- If they give any response related to the Week 3 materials → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any response: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about whether anything felt confusing"
"""
    elif question_number == 17:
        # For Q17, check if user provided a response about skills
        validation_prompt = """You are validating if a user has provided a response to "Were you able to come up with a skill or two for all five areas (hard skills, soft skills, technology skills, growth skills, and experiential skills)?"

The five areas are: hard skills, soft skills, technology skills, growth skills, and experiential skills.

The user has provided at least one response. Determine if they have given ANY reasonable answer, even if it's brief.
- If they mention skills in any category → COMPLETE
- If they say yes/no and provide examples → COMPLETE
- If they acknowledge they did/didn't complete the exercise → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any response: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about whether they identified skills for all five areas"
"""
    elif question_number == 18:
        # For Q18, check if user provided a response about which area was challenging
        validation_prompt = """You are validating if a user has provided a response to "Which area was most challenging for you?"

The five skill areas are: hard skills, soft skills, technology skills, growth skills, and experiential skills.

The user has provided at least one response. Determine if they have given ANY reasonable answer, even if it's brief.
- If they mention a specific skill area → COMPLETE
- If they say none were challenging → COMPLETE
- If they give any response related to the skill areas → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any response: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about which skill area was most challenging"
"""
    elif question_number == 19:
        # For Q19, check if user provided a response about their job search
        validation_prompt = """You are validating if a user has provided a response to "How has your job search been going?"

The user has provided at least one response. Determine if they have given ANY reasonable answer, even if it's brief.
- If they mention progress, challenges, or status of job search → COMPLETE
- If they say it's going well or poorly → COMPLETE
- If they mention they haven't started → COMPLETE
- If they give any response related to job searching → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any response: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about how their job search is going"
"""
    elif question_number == 20:
        # For Q20, check if user provided a response about job-seeking activities
        validation_prompt = """You are validating if a user has provided a response to "What job seeking activities have you been working on?"

The user has provided at least one response. Determine if they have given ANY reasonable answer, even if it's brief.
- If they list specific activities → COMPLETE
- If they say they haven't been doing any activities → COMPLETE
- If they give any response related to job seeking activities → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any response: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about what job-seeking activities they've been working on"
"""
    elif question_number == 21:
        # For Q21, check if user provided a response about their motivation
        validation_prompt = """You are validating if a user has provided a response to "What is motivating you to finish this program strong?"

The user has provided at least one response. Determine if they have given ANY reasonable answer, even if it's brief.
- If they express clear motivation → COMPLETE
- If they express uncertainty or low motivation → COMPLETE
- If they express discouragement or negative feelings → COMPLETE
- If they give any response related to motivation → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any response: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about what is motivating them to finish the program strong"
"""
    elif question_number == 22:
        # For Q22, check if user provided a response about their concerns
        validation_prompt = """You are validating if a user has provided a response to "What concerns do you have going into Week 4?"

The user has provided at least one response. Determine if they have given ANY reasonable answer, even if it's brief.
- If they share concerns → COMPLETE
- If they report no concerns → COMPLETE
- If they give any response related to concerns about Week 4 → COMPLETE
- Only mark as incomplete if they gave NO answer at all or completely unrelated response

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]

Example responses:
- If user provided any response: "COMPLETE: Yes\nMISSING: None"
- If no response provided: "COMPLETE: No\nMISSING: A response about what concerns they have going into Week 4"
"""
    else:
        # Generic validation for other questions
        validation_prompt = f"""You are validating if a user has provided a complete response to question {question_number}.

Based on the conversation history above, determine if the user has provided a reasonable answer to the question.

Respond in this exact format:
COMPLETE: Yes or No
MISSING: [List specific items that are missing, or "None" if all provided]
"""
    
    try:
        validation_response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a validation assistant. Respond only in the specified format."},
                {"role": "user", "content": f"{context}\n\n{validation_prompt}"}
            ],
            temperature=0.3
        )
        
        result = validation_response.choices[0].message.content.strip()
        is_complete = "COMPLETE: Yes" in result
        missing_items = result.split("MISSING:")[1].strip() if "MISSING:" in result else "Unknown"
        
        return is_complete, missing_items
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        return True, "Validation error"  # Default to complete on error


# Global instruction used to ensure no extra questions once a question is complete
NO_FOLLOWUPS_INSTRUCTION = (
    "Important: If the user's response already satisfies the criteria for this question and it is considered complete (or the classified scenario instructs to move on), do not ask any additional questions. Conclude your reply without further questions so we can proceed to the next question."
)

SKILL_CATEGORIES = [
    "hard skills",
    "soft skills",
    "technology skills",
    "growth skills",
    "experiential skills"
]

SKILL_CATEGORY_EXAMPLES = {
    "hard skills": ["Excel modeling", "CNC machining", "statistical analysis"],
    "soft skills": ["facilitating team meetings", "conflict resolution", "client communication"],
    "technology skills": ["SQL queries", "Salesforce automations", "Adobe Creative Cloud"],
    "growth skills": ["learning a new certification", "attending a workshop", "seeking mentorship"],
    "experiential skills": ["leading volunteer projects", "managing events", "coordinating internships"]
}

# Data required for Q15 prompts
WEEK_3_SKILLS_CONTENT = """Notes on week 3 skills content, including:

Key ideas about skills and strengths (for example, hard skills vs. soft skills, transferable skills)

Examples of job-related skills discussed in DRIVEN (for example, communication, reliability, customer service, computer use, organization, problem solving)"""

def extract_skill_categories_from_text(text: str) -> Set[str]:
    """Return a set of skill categories mentioned in the provided text."""
    text_lower = text.lower()
    matches = set()
    for category in SKILL_CATEGORIES:
        if category in text_lower:
            matches.add(category)
    return matches

def format_category_list(categories: List[str]) -> str:
    """Return a human-friendly list string from category names."""
    if not categories:
        return ""
    formatted = [category.title() for category in categories]
    if len(formatted) == 1:
        return formatted[0]
    return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"

def build_q17_missing_skills_prompt(
    name: str,
    known_categories: List[str],
    missing_categories: List[str],
    prompt_all: bool = False
) -> str:
    """Craft a coaching message guiding the user to fill missing skill categories."""
    lines = []
    if known_categories:
        lines.append(f"Great job identifying skills for {format_category_list(known_categories)} so far, {name}.")
    else:
        lines.append(f"Thanks for letting me know where you're at with the skills list, {name}.")
    
    lines.append("It's totally normal for some categories to take longer to nail down.")
    
    if prompt_all or not known_categories:
        lines.append("Let's list one skill you can claim for each of the five categories. For each area below, share your own example of something you've done or can do:")
    else:
        lines.append("Could you add a skill or two for these remaining areas? Feel free to describe your own examples:")
    
    for category in missing_categories:
        examples = SKILL_CATEGORY_EXAMPLES.get(category, [])
        if examples:
            example_text = ", ".join(examples[:2])
            lines.append(f"- {category.title()}: for example, {example_text}")
        else:
            lines.append(f"- {category.title()}")
    
    lines.append("Send your own skills for each area above, and we'll keep moving once the full list feels complete.")
    return "\n".join(lines)

QUESTIONS = {
    15: "What stood out to you about the third week's content on skills?",
    16: "Was there anything that felt a little off or confusing?",
    17: "Were you able to come up with a skill or two for all five areas (hard skills, soft skills, technology skills, growth skills, and experiential skills)?",
    18: "Which area was most challenging for you?",
    19: "How has your job search been going?",
    20: "What job seeking activities have you been working on?",
    21: "What is motivating you to finish this program strong?",
    22: "What concerns do you have going into Week 4?",
}

# Print messages to show between questions
# Can be a single string or a list of strings (for multiple messages)
PRINT_MESSAGES = {
    17: "Now, let's take a moment to review your Week 3 exercise on current skills.",
    19: [
        "Some skill areas will be more relevant than others for the types of jobs you're looking for today. But it's always a good idea to have a complete inventory of the skills you have because you never know when you might have an opportunity to use them in a role.",
        "In Week 4, you will use this skills list to create a resume that will show potential employers everything you have to offer.",
        "Nice work today! I'll be in touch tomorrow to chat about your general goals with this program and discuss next steps.",
        "Hey there! Hope you've been having a good day so far.",
        "In today's session, we will:\n\nTalk about where you are with the goal you set in Week 1 and your job search today.\n\nPlan the next steps until we meet again in next week.\n\nLet's jump into it!",
    ],
}

# Final messages to show after Q22 (as a list for separate messages)
FINAL_MESSAGES = [
    "While you're working on Week 4's content about online applications and resumes, feel free to write down any questions so we can talk about them when we chat next.",
    "I am incredibly proud of you for giving this program your all. While job searching is never easy, this effort will make all the difference. See you next week!"
]

# System prompts for Q15-Q22
SYSTEM_PROMPTS = {
    15: {
        "classifier": f"""Based on the user's response to "What stood out to you about the third week's content on skills?", determine which scenario applies:

Scenario 1: User identifies one or more specific ideas or exercises from the skills content that felt meaningful or helpful (e.g., mentions specific skills, concepts, exercises, or activities that resonated with them).

Scenario 2: User is unsure, does not remember much, or says that nothing really stood out (e.g., "I don't remember", "nothing really", "I'm not sure", "I didn't notice anything specific", vague or uncertain responses).

Scenario 3: User reports that the skills content felt confusing, discouraging, or not relevant to them (e.g., mentions confusion, discouragement, or that it didn't apply to them).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        "scenario_1_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user has identified one or more specific ideas or exercises from the skills content that felt meaningful or helpful.

Reflect back the specific skills ideas or activities they mentioned and validate why those might have stood out (for example, helping them notice strengths, transferable skills, or new ways to describe what they can do). Briefly connect these insights to how understanding their skills can support their job search and participation in DRIVEN. Use the following notes on week 3 skills content as context:
{WEEK_3_SKILLS_CONTENT}

{NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_2_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user is unsure, does not remember much, or says that nothing really stood out.

Normalize that it can be hard to remember or connect with new information, especially when someone has a lot going on. Gently summarize one or two of the main points from the following notes on week 3 skills content:
{WEEK_3_SKILLS_CONTENT}

Reassure them that you can keep revisiting skills together and focus on small, concrete examples from their own life so it feels easier to connect with.

{NO_FOLLOWUPS_INSTRUCTION}""",
        "scenario_3_respond": f"""Imagine that you are a trained career coach that helps adults with mental health issues to find jobs. The user's name is {{name}}. The user reports that the skills content felt confusing, discouraging, or not relevant to them.

Acknowledge their feelings and validate that it is okay if the skills content did not feel helpful or felt overwhelming at first. Look for any part of the following notes on week 3 skills content that might still fit their experiences or strengths, even if it is just one small piece:
{WEEK_3_SKILLS_CONTENT}

Emphasize that the goal is not to force them into a box, but to help them recognize skills they already have, and offer to go more slowly, use different examples, or focus on a smaller part of the content next time so it feels more manageable and relevant.

{NO_FOLLOWUPS_INSTRUCTION}"""
    },
    
    16: {
        "classifier": """Based on the user's response to "Was there anything that felt a little off or confusing?", determine which scenario applies:

Scenario 1: The user found something confusing (e.g., mentions something unclear, confusing, or difficult to understand about the Week 3 materials).

Scenario 2: The user felt everything made sense (e.g., says "no", "nothing", "everything was clear", "it all made sense", "no confusion").

Scenario 3: The user gave a response unrelated to the question (e.g., talks about something completely different, doesn't address confusion or clarity).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        
        "scenario_1_respond": """Imagine that you are a trained career coach that helps adults with mental health challenges navigate job searching. The user's name is {name}. The user found something confusing in the Week 3 material.

Acknowledge their confusion, normalize their experience (it's common to find some parts challenging), and briefly clarify the part of the material that may have felt unclear. Be supportive and encouraging.""",
        
        "scenario_2_respond": """Imagine that you are a trained career coach that helps adults with mental health challenges navigate job searching. The user's name is {name}. The user felt everything in the Week 3 material made sense.

Congratulate the user for moving through the material with clarity and reinforce how understanding the concepts supports their job-search journey. Be warm and encouraging.""",
        
        "scenario_3_respond": """Imagine that you are a trained career coach that helps adults with mental health challenges navigate job searching. The user's name is {name}. The user gave a response unrelated to whether anything felt confusing.

Gently steer the user back to the question and ask them whether anything felt confusing or unclear about the Week 3 material. Be friendly and supportive."""
    },
    
    17: {
        "classifier": """Based on the user's response to "Were you able to come up with a skill or two for all five areas (hard skills, soft skills, technology skills, growth skills, and experiential skills)?", determine which scenario applies:

The five areas are: hard skills, soft skills, technology skills, growth skills, and experiential skills.

Scenario 1: User identified skills for all five areas (e.g., mentions skills in all categories, lists skills across all areas, says "yes" and provides examples from multiple categories).

Scenario 2: User identified some areas but not all (e.g., mentions 2-4 categories but not all 5, says "some" or "a few", provides examples but acknowledges missing some areas).

Scenario 3: User did not identify any skills or avoided the exercise (e.g., says "no", "I don't know", "I didn't do it", "I couldn't think of any").

Scenario 4: User gave a response unrelated to the question (e.g., talks about something completely different, does not address skills at all).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", "SCENARIO_3", or "SCENARIO_4".""",
        
        "scenario_1_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user identified skills across all five areas {skill_list}.

Praise the user for completing the exercise and reinforce how this complete skills foundation will help them navigate the job market. Be warm and encouraging.""",
        
        "scenario_2_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user identified some — but not all — of the areas in {skill_list}.

Normalize the difficulty of identifying certain skill areas. If the user clearly mentions which areas they identified, offer concrete examples or prompts for the remaining categories they did not mention so they can fill the gaps. If they only say they completed “some” areas without specifying which ones, briefly give one or two simple example skills for each of the five categories in {skill_list} so they have ideas for every area. Encourage them to revisit their skills list and finish the exercise. Be supportive and helpful.""",

        "scenario_3_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user did not identify any skills or avoided the exercise for the five areas {skill_list}.

Encourage the user to complete the skills assignment in the DRIVEN app by identifying at least one skill in each of the five areas, then invite them to come back to Nova afterward to talk about what they wrote. Be patient and supportive.""",

        "scenario_4_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user shared something unrelated to the question about skills {skill_list}.

Gently steer the user back to the question by asking them to let you know which skill areas they've covered so far and which ones still need examples. Be friendly and supportive."""
    },
    
    18: {
        "classifier": """Based on the user's response to "Which area was most challenging for you?", determine which scenario applies:

The five skill areas are: hard skills, soft skills, technology skills, growth skills, and experiential skills.

Scenario 1: User identifies a specific challenging skill area (e.g., mentions "hard skills", "technology skills", "soft skills", "growth skills", "experiential skills", or any specific area from the list).

Scenario 2: User reports that none of the areas were challenging (e.g., says "none", "nothing was challenging", "all were easy", "I didn't find any challenging", "they were all fine").

Scenario 3: User gives an unclear or unrelated response (e.g., doesn't mention any skill area, talks about something completely different, gives vague or confusing response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        
        "scenario_1_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user identified a specific challenging skill area from {skill_list}.

Validate that the area they identified can be difficult and briefly explain why. Provide a small, approachable strategy to help them continue developing that skill area. Be supportive and encouraging.""",
        
        "scenario_2_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user reported that none of the areas from {skill_list} were challenging.

Reinforce the user's confidence. Acknowledge their self-awareness and capability. Be warm and encouraging.""",
        
        "scenario_3_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user gave an unclear or unrelated response about which skill area was most challenging from {skill_list}.

Re-ask the question gently and invite the user to share which skill area—if any—felt harder to identify. Be friendly and supportive."""
    },
    
    19: {
        "classifier": """Based on the user's response to "How has your job search been going?", determine which scenario applies:

Scenario 1: User reports their job search is going well (e.g., mentions progress, positive updates, interviews, applications sent, networking success, feeling optimistic, making headway).

Scenario 2: User reports difficulty, frustration, or discouragement (e.g., mentions challenges, rejections, feeling stuck, no responses, frustration, discouragement, difficulty finding opportunities, feeling overwhelmed).

Scenario 3: User reports they have not started a job search yet (e.g., says "I haven't started", "not yet", "I haven't begun", "I haven't applied", "I'm not searching", "I haven't looked").

Scenario 4: User gives a response unrelated to the question (e.g., talks about an unrelated topic, does not address how their job search is going).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", "SCENARIO_3", or "SCENARIO_4".""",
        
        "scenario_1_respond": """Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {name}. The user reports their job search is going well.

Congratulate the user on their progress and highlight how continued skill {skill_list} development will support their next steps. Be warm and encouraging.""",
        
        "scenario_2_respond": """Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {name}. The user reports difficulty, frustration, or discouragement with their job search.

Validate their feelings and difficulties, normalize their experience, and suggest that the skills {skill_list} they identified can help rebuild confidence and direction. Be empathetic and supportive.""",
        
        "scenario_3_respond": """Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {name}. The user reports they have not started a job search yet.

Normalize that starting can be hard, and suggest one or two very small, manageable steps to help them begin. Be patient and encouraging.""",

        "scenario_4_respond": """Imagine that you are a professional career coach who helps adults with mental health issues to find jobs. The user's name is {name}. The user shared something unrelated to how their job search is going.

Gently steer them back to the question and ask for a quick update on how the job search feels right now. Be supportive and curious."""
    },
    
    20: {
        "classifier": """Based on the user's response to "What job seeking activities have you been working on?", determine which scenario applies:

Scenario 1: User lists job-seeking activities (e.g., mentions applying to jobs, networking, updating resume, searching online, attending events, reaching out to contacts, practicing interviews, researching companies, etc.).

Scenario 2: User says they have not been doing any job-seeking activities (e.g., says "none", "nothing", "I haven't been doing anything", "I haven't worked on anything", "I haven't been active", "no activities").

Scenario 3: User gives an off-topic response (e.g., talks about something completely unrelated to job seeking, doesn't address the question, gives vague or confusing response).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", or "SCENARIO_3".""",
        
        "scenario_1_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user lists job-seeking activities they have been working on.

Praise their effort and connect their activities to the skills {skill_list} they identified in earlier exercises. Be warm and encouraging.""",
        
        "scenario_2_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user says they have not been doing any job-seeking activities.

Normalize the difficulty of staying motivated, then suggest one or two small, realistic job-seeking actions they could take this week. Be supportive and understanding.""",
        
        "scenario_3_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user gives an off-topic response about job-seeking activities.

Gently repeat the question and ask what job-seeking actions—if any—they've been working on. Be friendly and supportive."""
    },
    
    21: {
        "classifier": """Based on the user's response to "What is motivating you to finish this program strong?", determine which scenario applies:

Scenario 1: User expresses a clear motivation (e.g., mentions specific goals, reasons, aspirations, positive outcomes they want, clear purpose, determination, specific benefits they're seeking).

Scenario 2: User expresses uncertainty or low motivation (e.g., says "I'm not sure", "I don't know", "it's hard to stay motivated", "I'm struggling", "I'm not very motivated", expresses doubt or uncertainty about continuing).

Scenario 3: User expresses discouragement or negative feelings (e.g., mentions feeling discouraged, hopeless, negative about the program, doubts about it working, frustration, disappointment, or expresses giving up).

Scenario 4: User gives a response unrelated to the question (e.g., talks about something else entirely, does not mention motivation or goals).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", "SCENARIO_3", or "SCENARIO_4".""",
        
        "scenario_1_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user expresses a clear motivation to finish the program strong. Their motivation is: {user motivation}.

Reinforce and reflect the user's stated motivation and explain how the program supports that goal. Be warm and encouraging.""",
        
        "scenario_2_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user expresses uncertainty or low motivation about finishing the program strong.

Acknowledge that staying motivated can be difficult and encourage the user to identify even a small reason for continuing. Be supportive and understanding.""",
        
        "scenario_3_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user expresses discouragement or negative feelings about finishing the program.

Acknowledge their feelings, validate their experience, and gently encourage them to identify even a small reason for continuing. Remind them that progress can be gradual and that small steps forward still matter. Be empathetic and supportive.""",

        "scenario_4_respond": """Imagine that you are a trained career coach that helps adults with mental health issues find jobs. The user's name is {name}. The user shared something unrelated to what is motivating them to finish the program strong.

Gently steer them back by asking what is currently motivating them—or what they would like their motivation to be—as they move through DRIVEN. Be kind and curious."""
    },
    
    22: {
        "classifier": """Based on the user's response to "What concerns do you have going into Week 4?", determine which scenario applies:

Scenario 1: User shares concerns and seems discouraged (e.g., expresses worry, anxiety, fear, doubt, feeling overwhelmed, negative feelings about Week 4, seems worried or anxious about what's coming).

Scenario 2: User shares concerns but appears confident or matter-of-fact (e.g., mentions concerns but in a calm, practical way, acknowledges potential challenges but seems ready to handle them, expresses concerns but with confidence, matter-of-fact tone).

Scenario 3: User reports no concerns (e.g., says "none", "nothing", "I don't have any concerns", "I'm not worried", "no concerns", expresses confidence about Week 4).

Scenario 4: User gives a response unrelated to the question (e.g., brings up another topic, does not mention concerns or readiness for Week 4).

Respond with ONLY one of these: "SCENARIO_1", "SCENARIO_2", "SCENARIO_3", or "SCENARIO_4".""",
        
        "scenario_1_respond": """Imagine that you are a trained career coach who helps adults with mental health issues to find jobs. The user's name is {name}. The user shares concerns about going into Week 4 and seems discouraged.

Validate and normalize their concerns, provide encouragement, and offer brief, relevant guidance to help reduce their worry. Be empathetic and supportive.""",
        
        "scenario_2_respond": """Imagine that you are a trained career coach who helps adults with mental health issues to find jobs. The user's name is {name}. The user shares concerns about going into Week 4 but appears confident or matter-of-fact.

Affirm the user's readiness and offer a few proactive tips to help them maintain momentum. Be encouraging and supportive.""",
        
        "scenario_3_respond": """Imagine that you are a trained career coach who helps adults with mental health issues to find jobs. The user's name is {name}. The user reports no concerns about going into Week 4.

Reinforce the user's confidence and let them know they can bring up concerns anytime they arise. Be warm and encouraging.""",

        "scenario_4_respond": """Imagine that you are a trained career coach who helps adults with mental health issues to find jobs. The user's name is {name}. The user responded with something unrelated to their Week 4 concerns.

Gently steer them back to the question and invite them to share any worries—or confirm that they have none—about Week 4 so you can support them appropriately. Be encouraging."""
    }
}

WELCOME_MESSAGE = """Hello {name}!
I'm Nova, a virtual coach here to help you through the DRIVEN program.
We're continuing with Week 3 materials. Let's get started!"""


@app.route('/')
def index():
    """Serve the index.html file."""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return Response(f.read(), mimetype='text/html')
    except FileNotFoundError:
        return "index.html not found. Please make sure the file exists in the same directory as week3_main.py.", 404


@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize a new conversation."""
    data = request.get_json()
    name = data.get('name', '').strip()
    
    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400
    
    # Create new state (will create session if needed)
    state = get_or_create_state(name)
    state.name = name
    state.current_question = 15  # Start with question 15
    state.question_completed = {}
    
    # Persist user name for progress tracking
    session_id = session.get('session_id')
    if session_id:
        progress_tracker.set_user_name(session_id, name)
    
    # Return welcome message
    welcome_text = WELCOME_MESSAGE.replace("{name}", name)
    return jsonify({
        "success": True,
        "message": welcome_text
    })


@app.route('/api/get_next_message', methods=['POST'])
def get_next_message():
    """Get the next message in the dialogue flow."""
    state = get_or_create_state()
    
    # Check if final message should be shown (after Q22)
    if state.show_final_message:
        # Show messages one at a time
        if state.final_message_index < len(FINAL_MESSAGES):
            final_message = FINAL_MESSAGES[state.final_message_index]
            state.final_message_index += 1
            # If this is the last message, mark as complete
            is_complete = state.final_message_index >= len(FINAL_MESSAGES)
            if is_complete:
                state.show_final_message = False  # Clear the flag after all messages shown
                mark_week_completion()
            return jsonify({
                "success": True,
                "message": final_message,
                "is_complete": is_complete,
                "awaiting_response": False,
                "is_print_message": True,
                "week_completed": is_complete
            })
        else:
            # All final messages shown
            state.show_final_message = False
            mark_week_completion()
            return jsonify({
                "success": True,
                "message": "Thank you for completing this session!",
                "is_complete": True,
                "awaiting_response": False,
                "week_completed": True
            })
    
    # Check if we have a question to ask
    if state.current_question in QUESTIONS:
        # Check if there's a print message before this question
        if state.current_question in PRINT_MESSAGES:
            print_messages = PRINT_MESSAGES[state.current_question]
            # Handle both single string and list of strings
            if isinstance(print_messages, str):
                print_messages = [print_messages]
            
            # Get the current index for this question
            current_index = state.print_message_index.get(state.current_question, 0)
            
            # If we haven't shown all messages yet
            if current_index < len(print_messages):
                print_message = print_messages[current_index]
                # Increment the index for next time
                state.print_message_index[state.current_question] = current_index + 1
                return jsonify({
                    "success": True,
                    "message": print_message,
                    "is_complete": False,
                    "awaiting_response": False,
                    "is_print_message": True
                })
            else:
                # All print messages shown, mark as complete and show the question
                state.print_messages_shown.add(state.current_question)
        
        question_text = QUESTIONS[state.current_question]
        return jsonify({
            "success": True,
            "message": question_text,
            "is_complete": False,
            "awaiting_response": True,
            "question_number": state.current_question,
            "week_completed": False
        })
    
    # No more questions
    mark_week_completion()
    return jsonify({
        "success": True,
        "message": "Thank you for completing this session! I'll be in touch soon.",
        "is_complete": True,
        "awaiting_response": False,
        "week_completed": True
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
    
    # Handle Q15 with scenario classification
    if question_number == 15:
        print(f"\n{'#'*80}")
        print(f"[DEBUG PROCESS_RESPONSE Q15] User's message: {user_message}")
        print(f"{'#'*80}\n")
        
        # Initialize q15_scenario if not already set
        if not hasattr(state, 'q15_scenario'):
            state.q15_scenario = None
        
        # Step 1: Classify the scenario (only on first response to Q15 or if Scenario 3 needs reclassification)
        if state.q15_scenario is None or "SCENARIO_3" in state.q15_scenario:
            q15_prompts = SYSTEM_PROMPTS.get(15, {})
            if not isinstance(q15_prompts, dict) or "classifier" not in q15_prompts:
                return jsonify({"success": False, "error": "Q15 prompts not configured correctly"}), 400
            
            classifier_prompt = q15_prompts["classifier"]
            print(f"[DEBUG] Classifying scenario for Q15...")
            
            classification_response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                    {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                ],
                temperature=0.3
            )
            
            state.q15_scenario = classification_response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Q15 Scenario classification result: {state.q15_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q15_prompts = SYSTEM_PROMPTS.get(15, {})
        
        if "SCENARIO_1" in state.q15_scenario:
            # Scenario 1: User identified specific ideas/exercises
            print(f"[DEBUG] Q15 SCENARIO 1 detected - reflecting back and validating insights")
            system_prompt = q15_prompts.get("scenario_1_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_2" in state.q15_scenario:
            # Scenario 2: User is unsure or doesn't remember
            print(f"[DEBUG] Q15 SCENARIO 2 detected - normalizing and summarizing key points")
            system_prompt = q15_prompts.get("scenario_2_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_3" in state.q15_scenario:
            # Scenario 3: User found content confusing/discouraging
            print(f"[DEBUG] Q15 SCENARIO 3 detected - acknowledging feelings and offering support")
            system_prompt = q15_prompts.get("scenario_3_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 1 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 1")
            state.q15_scenario = "SCENARIO_1"
            system_prompt = q15_prompts.get("scenario_1_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            nova_response = call_llm(system_prompt, user_message)
        
        # Store NOVA response
        if question_number not in state.nova_responses:
            state.nova_responses[question_number] = []
        state.nova_responses[question_number].append(nova_response)
        
        # Increment iteration
        state.iteration_count[question_number] = state.iteration_count.get(question_number, 0) + 1
        iteration = state.get_iteration(question_number)
        
        # Validate completeness
        user_responses = state.answers.get(question_number, [])
        conversation_history = []
        for qnum in sorted(state.answers.keys()):
            if state.answers[qnum]:
                conversation_history.append(f"Q{qnum}: {state.answers[qnum][-1]}")
        
        is_complete, missing = validate_completeness(
            question_number, 
            nova_response, 
            user_responses, 
            "\n".join(conversation_history),
            q15_scenario=None,
            q16_scenario=None
        )
        
        # Check if we need another iteration (2-iteration loop)
        needs_followup = False
        move_to_next = False
        
        # If Scenario 3, we're offering support, so stay on this question
        if "SCENARIO_3" in state.q15_scenario:
            needs_followup = True
            move_to_next = False
        elif iteration < 2 and not is_complete:
            # Need another iteration if not complete and under 2 iterations
            needs_followup = True
            move_to_next = False
        else:
            # Done with this question - move to next
            move_to_next = True
            state.question_completed[question_number] = True
            save_question_progress(question_number)
            state.current_question += 1
            print(f"[DEBUG] Q{question_number} marked as complete. Iteration: {iteration}, is_complete: {is_complete}")
        
        return jsonify({
            "success": True,
            "response": nova_response,
            "needs_followup": needs_followup,
            "move_to_next": move_to_next,
            "iteration": iteration,
            "is_complete": is_complete
        })
    
    # Handle Q17 with scenario classification
    elif question_number == 17:
        # Initialize tracking for skill categories on first iteration
        if iteration == 0:
            state.q17_categories_identified = set()
            state.q17_missing_categories = set(SKILL_CATEGORIES)
        
        # Step 1: Classify the scenario
        # Reclassify if scenario is None or if user is in Scenario 2/3 (to allow them to complete)
        if state.q17_scenario is None or "SCENARIO_2" in state.q17_scenario or "SCENARIO_3" in state.q17_scenario or "SCENARIO_4" in state.q17_scenario:
            q17_prompts = SYSTEM_PROMPTS.get(17, {})
            if isinstance(q17_prompts, dict) and "classifier" in q17_prompts:
                classifier_prompt = q17_prompts["classifier"]
                
                classification_response = openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                        {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                    ],
                    temperature=0.3
                )
                
                state.q17_scenario = classification_response.choices[0].message.content.strip().upper()
                print(f"[DEBUG] Q17 Scenario classification result: {state.q17_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q17_prompts = SYSTEM_PROMPTS.get(17, {})
        
        if "SCENARIO_1" in state.q17_scenario:
            # Scenario 1: User identified skills for all five areas
            print(f"[DEBUG] Q17 SCENARIO 1 detected - praising user for completing exercise")
            
            system_prompt = q17_prompts.get("scenario_1_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_2" in state.q17_scenario:
            # Scenario 2: User identified some areas but not all
            print(f"[DEBUG] Q17 SCENARIO 2 detected - helping complete missing categories")
            
            # Ensure tracking structures exist
            if not hasattr(state, "q17_categories_identified"):
                state.q17_categories_identified = set()
            if not hasattr(state, "q17_missing_categories") or not state.q17_missing_categories:
                state.q17_missing_categories = set(SKILL_CATEGORIES)
            
            identified_categories = extract_skill_categories_from_text(user_message)
            if identified_categories:
                print(f"[DEBUG] Q17 categories identified in this response: {identified_categories}")
            else:
                print("[DEBUG] Q17 no specific categories detected in this response")
            
            state.q17_categories_identified.update(identified_categories)
            state.q17_missing_categories -= identified_categories
            
            known_categories = [cat for cat in SKILL_CATEGORIES if cat in state.q17_categories_identified]
            missing_categories = [cat for cat in SKILL_CATEGORIES if cat in state.q17_missing_categories]
            
            if missing_categories:
                # Keep guiding the user to fill in the remaining categories
                prompt_all = not known_categories
                nova_response = build_q17_missing_skills_prompt(
                    state.name,
                    known_categories,
                    missing_categories,
                    prompt_all=prompt_all
                )
                # Ensure scenario remains Scenario 2 for the next iteration
                state.q17_scenario = "SCENARIO_2"
            else:
                # All categories captured – celebrate and fall through to Scenario 1 completion
                print("[DEBUG] Q17 all categories captured – transitioning to Scenario 1 completion")
                state.q17_scenario = "SCENARIO_1"
                state.q17_categories_identified = set(SKILL_CATEGORIES)
                system_prompt = q17_prompts.get("scenario_1_respond", "")
                system_prompt = state.substitute_variables(system_prompt)
                system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
                nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_3" in state.q17_scenario:
            # Scenario 3: User did not identify any skills
            print(f"[DEBUG] Q17 SCENARIO 3 detected - encouraging user to start")
            
            if not hasattr(state, "q17_categories_identified"):
                state.q17_categories_identified = set()
            if not hasattr(state, "q17_missing_categories") or not state.q17_missing_categories:
                state.q17_missing_categories = set(SKILL_CATEGORIES)
            
            identified_categories = extract_skill_categories_from_text(user_message)
            if identified_categories:
                print(f"[DEBUG] Q17 categories identified in this response: {identified_categories}")
            else:
                print("[DEBUG] Q17 no specific categories detected in this response")
            
            state.q17_categories_identified.update(identified_categories)
            state.q17_missing_categories -= identified_categories
            
            known_categories = [cat for cat in SKILL_CATEGORIES if cat in state.q17_categories_identified]
            missing_categories = [cat for cat in SKILL_CATEGORIES if cat in state.q17_missing_categories]
            
            if missing_categories:
                nova_response = build_q17_missing_skills_prompt(
                    state.name,
                    known_categories,
                    missing_categories,
                    prompt_all=True
                )
                state.q17_scenario = "SCENARIO_3"
            else:
                print("[DEBUG] Q17 Scenario 3 user completed all categories – transitioning to Scenario 1 completion")
                state.q17_scenario = "SCENARIO_1"
                state.q17_categories_identified = set(SKILL_CATEGORIES)
                system_prompt = q17_prompts.get("scenario_1_respond", "")
                system_prompt = state.substitute_variables(system_prompt)
                system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
                nova_response = call_llm(system_prompt, user_message)
        elif "SCENARIO_4" in state.q17_scenario:
            # Scenario 4: User response unrelated – gently steer back
            print(f"[DEBUG] Q17 SCENARIO 4 detected - re-centering on skills question")
            
            system_prompt = q17_prompts.get("scenario_4_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            nova_response = call_llm(system_prompt, user_message)
            # Keep scenario 4 until user provides relevant info (will reclassify next iteration)
            
        else:
            # Default to Scenario 2 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 2")
            state.q17_scenario = "SCENARIO_2"
            
            system_prompt = q17_prompts.get("scenario_2_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            
            nova_response = call_llm(system_prompt, user_message)
        
        # Store NOVA response
        if question_number not in state.nova_responses:
            state.nova_responses[question_number] = []
        state.nova_responses[question_number].append(nova_response)
        
        # Increment iteration
        state.iteration_count[question_number] = state.iteration_count.get(question_number, 0) + 1
        iteration = state.get_iteration(question_number)
        
        # Validate completeness
        user_responses = state.answers.get(question_number, [])
        conversation_history = []
        for qnum in sorted(state.answers.keys()):
            if state.answers[qnum]:
                conversation_history.append(f"Q{qnum}: {state.answers[qnum][-1]}")
        
        is_complete, missing = validate_completeness(
            question_number, 
            nova_response, 
            user_responses, 
            "\n".join(conversation_history),
            q16_scenario=None  # Q17 doesn't use q16_scenario
        )
        
        # Check if we need another iteration (2-iteration loop)
        needs_followup = False
        move_to_next = False
        
        # If Scenario 4, stay until user provides relevant info
        if "SCENARIO_4" in state.q17_scenario:
            needs_followup = True
            move_to_next = False
        # If Scenario 1, we're done - move to next
        elif "SCENARIO_1" in state.q17_scenario:
            move_to_next = True
            state.question_completed[question_number] = True
            save_question_progress(question_number)
            state.current_question += 1
            print(f"[DEBUG] Q{question_number} marked as complete (Scenario 1). Iteration: {iteration}")
        # If Scenario 2 or 3, we're helping/encouraging, so stay on this question
        elif "SCENARIO_2" in state.q17_scenario or "SCENARIO_3" in state.q17_scenario:
            needs_followup = True
            move_to_next = False
        elif iteration < 2 and not is_complete:
            # Need another iteration if not complete and under 2 iterations
            needs_followup = True
            move_to_next = False
        else:
            # Done with this question - move to next
            # Force completion after 2 iterations or if validation says complete
            move_to_next = True
            state.question_completed[question_number] = True
            save_question_progress(question_number)
            state.current_question += 1
            print(f"[DEBUG] Q{question_number} marked as complete. Iteration: {iteration}, is_complete: {is_complete}")
        
        return jsonify({
            "success": True,
            "response": nova_response,
            "needs_followup": needs_followup,
            "move_to_next": move_to_next,
            "iteration": iteration,
            "is_complete": is_complete
        })
    
    elif question_number == 18:
        # Step 1: Classify the scenario
        # Reclassify if scenario is None or if user is in Scenario 3 (to allow them to provide clearer answer)
        if state.q18_scenario is None or "SCENARIO_3" in state.q18_scenario:
            q18_prompts = SYSTEM_PROMPTS.get(18, {})
            if isinstance(q18_prompts, dict) and "classifier" in q18_prompts:
                classifier_prompt = q18_prompts["classifier"]
                
                classification_response = openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                        {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                    ],
                    temperature=0.3
                )
                
                state.q18_scenario = classification_response.choices[0].message.content.strip().upper()
                print(f"[DEBUG] Q18 Scenario classification result: {state.q18_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q18_prompts = SYSTEM_PROMPTS.get(18, {})
        
        if "SCENARIO_1" in state.q18_scenario:
            # Scenario 1: User identifies a specific challenging skill area
            print(f"[DEBUG] Q18 SCENARIO 1 detected - validating challenge and providing strategy")
            
            system_prompt = q18_prompts.get("scenario_1_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_2" in state.q18_scenario:
            # Scenario 2: User reports none were challenging
            print(f"[DEBUG] Q18 SCENARIO 2 detected - reinforcing confidence")
            
            system_prompt = q18_prompts.get("scenario_2_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_3" in state.q18_scenario:
            # Scenario 3: User gives unclear or unrelated response
            print(f"[DEBUG] Q18 SCENARIO 3 detected - re-asking question gently")
            
            system_prompt = q18_prompts.get("scenario_3_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            # Don't add NO_FOLLOWUPS_INSTRUCTION here since we're re-asking the question
            
            nova_response = call_llm(system_prompt, user_message)
            
        else:
            # Default to Scenario 3 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 3")
            state.q18_scenario = "SCENARIO_3"
            
            system_prompt = q18_prompts.get("scenario_3_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            
            nova_response = call_llm(system_prompt, user_message)
        
        # Store NOVA response
        if question_number not in state.nova_responses:
            state.nova_responses[question_number] = []
        state.nova_responses[question_number].append(nova_response)
        
        # Increment iteration
        state.iteration_count[question_number] = state.iteration_count.get(question_number, 0) + 1
        iteration = state.get_iteration(question_number)
        
        # Validate completeness
        user_responses = state.answers.get(question_number, [])
        conversation_history = []
        for qnum in sorted(state.answers.keys()):
            if state.answers[qnum]:
                conversation_history.append(f"Q{qnum}: {state.answers[qnum][-1]}")
        
        is_complete, missing = validate_completeness(
            question_number, 
            nova_response, 
            user_responses, 
            "\n".join(conversation_history),
            q16_scenario=None  # Q18 doesn't use q16_scenario
        )
        
        # Check if we need another iteration (2-iteration loop)
        needs_followup = False
        move_to_next = False
        
        # If Scenario 3, we're re-asking the question, so stay on this question
        if "SCENARIO_3" in state.q18_scenario:
            needs_followup = True
            move_to_next = False
        elif iteration < 2 and not is_complete:
            # Need another iteration if not complete and under 2 iterations
            needs_followup = True
            move_to_next = False
        else:
            # Done with this question - move to next
            # Force completion after 2 iterations or if validation says complete
            move_to_next = True
            state.question_completed[question_number] = True
            save_question_progress(question_number)
            state.current_question += 1
            print(f"[DEBUG] Q{question_number} marked as complete. Iteration: {iteration}, is_complete: {is_complete}")
        
        return jsonify({
            "success": True,
            "response": nova_response,
            "needs_followup": needs_followup,
            "move_to_next": move_to_next,
            "iteration": iteration,
            "is_complete": is_complete
        })
    
    elif question_number == 19:
        # Step 1: Classify the scenario
        if state.q19_scenario is None or "SCENARIO_4" in state.q19_scenario:
            q19_prompts = SYSTEM_PROMPTS.get(19, {})
            if isinstance(q19_prompts, dict) and "classifier" in q19_prompts:
                classifier_prompt = q19_prompts["classifier"]
                
                classification_response = openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                        {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                    ],
                    temperature=0.3
                )
                
                state.q19_scenario = classification_response.choices[0].message.content.strip().upper()
                print(f"[DEBUG] Q19 Scenario classification result: {state.q19_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q19_prompts = SYSTEM_PROMPTS.get(19, {})
        
        if "SCENARIO_1" in state.q19_scenario:
            # Scenario 1: User reports job search is going well
            print(f"[DEBUG] Q19 SCENARIO 1 detected - congratulating progress")
            
            system_prompt = q19_prompts.get("scenario_1_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_2" in state.q19_scenario:
            # Scenario 2: User reports difficulty, frustration, or discouragement
            print(f"[DEBUG] Q19 SCENARIO 2 detected - validating feelings and providing support")
            
            system_prompt = q19_prompts.get("scenario_2_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_3" in state.q19_scenario:
            # Scenario 3: User reports they haven't started job search yet
            print(f"[DEBUG] Q19 SCENARIO 3 detected - normalizing and suggesting small steps")
            
            system_prompt = q19_prompts.get("scenario_3_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_4" in state.q19_scenario:
            # Scenario 4: User response unrelated – gently steer back
            print(f"[DEBUG] Q19 SCENARIO 4 detected - re-centering on job search question")
            
            system_prompt = q19_prompts.get("scenario_4_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            nova_response = call_llm(system_prompt, user_message)
            # Keep scenario 4 active until reclassified on next iteration
        else:
            # Default to Scenario 2 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 2")
            state.q19_scenario = "SCENARIO_2"
            
            system_prompt = q19_prompts.get("scenario_2_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
        
        # Store NOVA response
        if question_number not in state.nova_responses:
            state.nova_responses[question_number] = []
        state.nova_responses[question_number].append(nova_response)
        
        # Increment iteration
        state.iteration_count[question_number] = state.iteration_count.get(question_number, 0) + 1
        iteration = state.get_iteration(question_number)
        
        # Validate completeness
        user_responses = state.answers.get(question_number, [])
        conversation_history = []
        for qnum in sorted(state.answers.keys()):
            if state.answers[qnum]:
                conversation_history.append(f"Q{qnum}: {state.answers[qnum][-1]}")
        
        is_complete, missing = validate_completeness(
            question_number, 
            nova_response, 
            user_responses, 
            "\n".join(conversation_history),
            q16_scenario=None  # Q19 doesn't use q16_scenario
        )
        
        # Check if we need another iteration (2-iteration loop)
        needs_followup = False
        move_to_next = False
        
        # Scenario 4 should never advance until user provides a relevant answer
        if state.q19_scenario and "SCENARIO_4" in state.q19_scenario:
            print(f"[DEBUG] Q19 SCENARIO 4 active - awaiting relevant job-search response")
            needs_followup = True
            move_to_next = False
        elif iteration < 2 and not is_complete:
            # Need another iteration if not complete and under 2 iterations
            needs_followup = True
            move_to_next = False
        else:
            # Done with this question - move to next
            # Force completion after 2 iterations or if validation says complete
            move_to_next = True
            state.question_completed[question_number] = True
            save_question_progress(question_number)
            state.current_question += 1
            print(f"[DEBUG] Q{question_number} marked as complete. Iteration: {iteration}, is_complete: {is_complete}")
        
        return jsonify({
            "success": True,
            "response": nova_response,
            "needs_followup": needs_followup,
            "move_to_next": move_to_next,
            "iteration": iteration,
            "is_complete": is_complete
        })
    
    elif question_number == 20:
        # Step 1: Classify the scenario
        # Reclassify if scenario is None or if user is in Scenario 3 (to allow them to provide clearer answer)
        if state.q20_scenario is None or "SCENARIO_3" in state.q20_scenario:
            q20_prompts = SYSTEM_PROMPTS.get(20, {})
            if isinstance(q20_prompts, dict) and "classifier" in q20_prompts:
                classifier_prompt = q20_prompts["classifier"]
                
                classification_response = openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                        {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                    ],
                    temperature=0.3
                )
                
                state.q20_scenario = classification_response.choices[0].message.content.strip().upper()
                print(f"[DEBUG] Q20 Scenario classification result: {state.q20_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q20_prompts = SYSTEM_PROMPTS.get(20, {})
        
        if "SCENARIO_1" in state.q20_scenario:
            # Scenario 1: User lists job-seeking activities
            print(f"[DEBUG] Q20 SCENARIO 1 detected - praising effort and connecting to skills")
            
            system_prompt = q20_prompts.get("scenario_1_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_2" in state.q20_scenario:
            # Scenario 2: User says they haven't been doing any activities
            print(f"[DEBUG] Q20 SCENARIO 2 detected - normalizing difficulty and suggesting actions")
            
            system_prompt = q20_prompts.get("scenario_2_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_3" in state.q20_scenario:
            # Scenario 3: User gives off-topic response
            print(f"[DEBUG] Q20 SCENARIO 3 detected - gently repeating question")
            
            system_prompt = q20_prompts.get("scenario_3_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            # Don't add NO_FOLLOWUPS_INSTRUCTION here since we're re-asking the question
            
            nova_response = call_llm(system_prompt, user_message)
            
        else:
            # Default to Scenario 3 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 3")
            state.q20_scenario = "SCENARIO_3"
            
            system_prompt = q20_prompts.get("scenario_3_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            
            nova_response = call_llm(system_prompt, user_message)
        
        # Store NOVA response
        if question_number not in state.nova_responses:
            state.nova_responses[question_number] = []
        state.nova_responses[question_number].append(nova_response)
        
        # Increment iteration
        state.iteration_count[question_number] = state.iteration_count.get(question_number, 0) + 1
        iteration = state.get_iteration(question_number)
        
        # Validate completeness
        user_responses = state.answers.get(question_number, [])
        conversation_history = []
        for qnum in sorted(state.answers.keys()):
            if state.answers[qnum]:
                conversation_history.append(f"Q{qnum}: {state.answers[qnum][-1]}")
        
        is_complete, missing = validate_completeness(
            question_number, 
            nova_response, 
            user_responses, 
            "\n".join(conversation_history),
            q16_scenario=None  # Q20 doesn't use q16_scenario
        )
        
        # Check if we need another iteration (2-iteration loop)
        needs_followup = False
        move_to_next = False
        
        # If Scenario 3, we're re-asking the question, so stay on this question
        # But only if we haven't exceeded 2 iterations
        if "SCENARIO_3" in state.q20_scenario and iteration < 2:
            needs_followup = True
            move_to_next = False
        elif iteration < 2 and not is_complete:
            # Need another iteration if not complete and under 2 iterations
            needs_followup = True
            move_to_next = False
        else:
            # Done with this question - move to next
            # Force completion after 2 iterations or if validation says complete
            move_to_next = True
            state.question_completed[question_number] = True
            save_question_progress(question_number)
            state.current_question += 1
            print(f"[DEBUG] Q{question_number} marked as complete. Iteration: {iteration}, is_complete: {is_complete}")
        
        return jsonify({
            "success": True,
            "response": nova_response,
            "needs_followup": needs_followup,
            "move_to_next": move_to_next,
            "iteration": iteration,
            "is_complete": is_complete
        })
    
    elif question_number == 21:
        # Step 1: Classify the scenario
        if state.q21_scenario is None or "SCENARIO_4" in state.q21_scenario:
            q21_prompts = SYSTEM_PROMPTS.get(21, {})
            if isinstance(q21_prompts, dict) and "classifier" in q21_prompts:
                classifier_prompt = q21_prompts["classifier"]
                
                classification_response = openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                        {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                    ],
                    temperature=0.3
                )
                
                state.q21_scenario = classification_response.choices[0].message.content.strip().upper()
                print(f"[DEBUG] Q21 Scenario classification result: {state.q21_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q21_prompts = SYSTEM_PROMPTS.get(21, {})
        
        if "SCENARIO_1" in state.q21_scenario:
            # Scenario 1: User expresses a clear motivation
            print(f"[DEBUG] Q21 SCENARIO 1 detected - reinforcing motivation")
            
            system_prompt = q21_prompts.get("scenario_1_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_2" in state.q21_scenario:
            # Scenario 2: User expresses uncertainty or low motivation
            print(f"[DEBUG] Q21 SCENARIO 2 detected - acknowledging difficulty and encouraging")
            
            system_prompt = q21_prompts.get("scenario_2_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_3" in state.q21_scenario:
            # Scenario 3: User expresses discouragement or negative feelings
            print(f"[DEBUG] Q21 SCENARIO 3 detected - acknowledging feelings and providing support")
            
            system_prompt = q21_prompts.get("scenario_3_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_4" in state.q21_scenario:
            # Scenario 4: User response unrelated – gently steer back
            print(f"[DEBUG] Q21 SCENARIO 4 detected - re-centering on motivation question")
            
            system_prompt = q21_prompts.get("scenario_4_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 2 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 2")
            state.q21_scenario = "SCENARIO_2"
            
            system_prompt = q21_prompts.get("scenario_2_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
        
        # Store NOVA response
        if question_number not in state.nova_responses:
            state.nova_responses[question_number] = []
        state.nova_responses[question_number].append(nova_response)
        
        # Increment iteration
        state.iteration_count[question_number] = state.iteration_count.get(question_number, 0) + 1
        iteration = state.get_iteration(question_number)
        
        # Validate completeness
        user_responses = state.answers.get(question_number, [])
        conversation_history = []
        for qnum in sorted(state.answers.keys()):
            if state.answers[qnum]:
                conversation_history.append(f"Q{qnum}: {state.answers[qnum][-1]}")
        
        is_complete, missing = validate_completeness(
            question_number, 
            nova_response, 
            user_responses, 
            "\n".join(conversation_history),
            q16_scenario=None  # Q21 doesn't use q16_scenario
        )
        
        # Check if we need another iteration (2-iteration loop)
        needs_followup = False
        move_to_next = False
        
        if state.q21_scenario and "SCENARIO_4" in state.q21_scenario:
            print("[DEBUG] Q21 SCENARIO 4 active - awaiting relevant motivation response")
            needs_followup = True
            move_to_next = False
        elif iteration < 2 and not is_complete:
            # Need another iteration if not complete and under 2 iterations
            needs_followup = True
            move_to_next = False
        else:
            # Done with this question - move to next
            # Force completion after 2 iterations or if validation says complete
            move_to_next = True
            state.question_completed[question_number] = True
            save_question_progress(question_number)
            state.current_question += 1
            print(f"[DEBUG] Q{question_number} marked as complete. Iteration: {iteration}, is_complete: {is_complete}")
        
        return jsonify({
            "success": True,
            "response": nova_response,
            "needs_followup": needs_followup,
            "move_to_next": move_to_next,
            "iteration": iteration,
            "is_complete": is_complete
        })
    
    elif question_number == 22:
        # Step 1: Classify the scenario
        if state.q22_scenario is None or "SCENARIO_4" in state.q22_scenario:
            q22_prompts = SYSTEM_PROMPTS.get(22, {})
            if isinstance(q22_prompts, dict) and "classifier" in q22_prompts:
                classifier_prompt = q22_prompts["classifier"]
                
                classification_response = openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                        {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                    ],
                    temperature=0.3
                )
                
                state.q22_scenario = classification_response.choices[0].message.content.strip().upper()
                print(f"[DEBUG] Q22 Scenario classification result: {state.q22_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q22_prompts = SYSTEM_PROMPTS.get(22, {})
        
        if "SCENARIO_1" in state.q22_scenario:
            # Scenario 1: User shares concerns and seems discouraged
            print(f"[DEBUG] Q22 SCENARIO 1 detected - validating concerns and providing encouragement")
            
            system_prompt = q22_prompts.get("scenario_1_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_2" in state.q22_scenario:
            # Scenario 2: User shares concerns but appears confident or matter-of-fact
            print(f"[DEBUG] Q22 SCENARIO 2 detected - affirming readiness and offering tips")
            
            system_prompt = q22_prompts.get("scenario_2_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_3" in state.q22_scenario:
            # Scenario 3: User reports no concerns
            print(f"[DEBUG] Q22 SCENARIO 3 detected - reinforcing confidence")
            
            system_prompt = q22_prompts.get("scenario_3_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_4" in state.q22_scenario:
            # Scenario 4: User response unrelated – gently steer back
            print(f"[DEBUG] Q22 SCENARIO 4 detected - re-centering on Week 4 concerns")
            
            system_prompt = q22_prompts.get("scenario_4_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            nova_response = call_llm(system_prompt, user_message)
        else:
            # Default to Scenario 2 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 2")
            state.q22_scenario = "SCENARIO_2"
            
            system_prompt = q22_prompts.get("scenario_2_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
        
        # Store NOVA response
        if question_number not in state.nova_responses:
            state.nova_responses[question_number] = []
        state.nova_responses[question_number].append(nova_response)
        
        # Increment iteration
        state.iteration_count[question_number] = state.iteration_count.get(question_number, 0) + 1
        iteration = state.get_iteration(question_number)
        
        # Validate completeness
        user_responses = state.answers.get(question_number, [])
        conversation_history = []
        for qnum in sorted(state.answers.keys()):
            if state.answers[qnum]:
                conversation_history.append(f"Q{qnum}: {state.answers[qnum][-1]}")
        
        is_complete, missing = validate_completeness(
            question_number, 
            nova_response, 
            user_responses, 
            "\n".join(conversation_history),
            q16_scenario=None  # Q22 doesn't use q16_scenario
        )
        
        # Check if we need another iteration (2-iteration loop)
        needs_followup = False
        move_to_next = False
        
        if state.q22_scenario and "SCENARIO_4" in state.q22_scenario:
            print("[DEBUG] Q22 SCENARIO 4 active - awaiting relevant Week 4 concerns response")
            needs_followup = True
            move_to_next = False
        elif iteration < 2 and not is_complete:
            # Need another iteration if not complete and under 2 iterations
            needs_followup = True
            move_to_next = False
        else:
            # Done with this question
            state.question_completed[question_number] = True
            save_question_progress(question_number)
            # If this is Q22 (the last question), set flag to show final message
            if question_number == 22:
                state.show_final_message = True
                move_to_next = True  # Set to True so frontend calls get_next_message to show final message
                # Don't increment current_question since we're done
            else:
                # Move to next question
                move_to_next = True
                state.current_question += 1
        
        return jsonify({
            "success": True,
            "response": nova_response,
            "needs_followup": needs_followup,
            "move_to_next": move_to_next,
            "iteration": iteration,
            "is_complete": is_complete
        })
    
    elif question_number == 16:
        # Step 1: Classify the scenario
        # Reclassify if scenario is None or if user is in Scenario 3 (to allow them to provide clearer answer)
        if state.q16_scenario is None or "SCENARIO_3" in state.q16_scenario:
            q16_prompts = SYSTEM_PROMPTS.get(16, {})
            if isinstance(q16_prompts, dict) and "classifier" in q16_prompts:
                classifier_prompt = q16_prompts["classifier"]
                
                classification_response = openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a scenario classifier. Respond with only the scenario identifier."},
                        {"role": "user", "content": f"{classifier_prompt}\n\nUser's response: {user_message}"}
                    ],
                    temperature=0.3
                )
                
                state.q16_scenario = classification_response.choices[0].message.content.strip().upper()
                print(f"[DEBUG] Q16 Scenario classification result: {state.q16_scenario}")
        
        # Step 2: Use hardcoded logic based on classification
        q16_prompts = SYSTEM_PROMPTS.get(16, {})
        
        if "SCENARIO_1" in state.q16_scenario:
            # Scenario 1: User found something confusing
            print(f"[DEBUG] Q16 SCENARIO 1 detected - acknowledging confusion and clarifying")
            
            system_prompt = q16_prompts.get("scenario_1_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_2" in state.q16_scenario:
            # Scenario 2: User felt everything made sense
            print(f"[DEBUG] Q16 SCENARIO 2 detected - congratulating user")
            
            system_prompt = q16_prompts.get("scenario_2_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
            
        elif "SCENARIO_3" in state.q16_scenario:
            # Scenario 3: User gave unrelated response
            print(f"[DEBUG] Q16 SCENARIO 3 detected - steering back to question")
            
            system_prompt = q16_prompts.get("scenario_3_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            # Don't add NO_FOLLOWUPS_INSTRUCTION here since we're asking a follow-up
            
            nova_response = call_llm(system_prompt, user_message)
            
        else:
            # Default to Scenario 2 if unclear
            print(f"[DEBUG] Unclear classification, defaulting to Scenario 2")
            state.q16_scenario = "SCENARIO_2"
            
            system_prompt = q16_prompts.get("scenario_2_respond", "")
            system_prompt = state.substitute_variables(system_prompt)
            system_prompt = f"{system_prompt}\n\n{NO_FOLLOWUPS_INSTRUCTION}"
            
            nova_response = call_llm(system_prompt, user_message)
        
        # Store NOVA response
        if question_number not in state.nova_responses:
            state.nova_responses[question_number] = []
        state.nova_responses[question_number].append(nova_response)
        
        # Increment iteration
        state.iteration_count[question_number] = state.iteration_count.get(question_number, 0) + 1
        iteration = state.get_iteration(question_number)
        
        # Validate completeness
        user_responses = state.answers.get(question_number, [])
        conversation_history = []
        for qnum in sorted(state.answers.keys()):
            if state.answers[qnum]:
                conversation_history.append(f"Q{qnum}: {state.answers[qnum][-1]}")
        
        is_complete, missing = validate_completeness(
            question_number, 
            nova_response, 
            user_responses, 
            "\n".join(conversation_history),
            q16_scenario=state.q16_scenario
        )
        
        # Check if we need another iteration (2-iteration loop)
        needs_followup = False
        move_to_next = False
        
        # If Scenario 3, we're asking a follow-up, so stay on this question
        if "SCENARIO_3" in state.q16_scenario:
            needs_followup = True
            move_to_next = False
        elif iteration < 2 and not is_complete:
            # Need another iteration if not complete and under 2 iterations
            needs_followup = True
            move_to_next = False
        else:
            # Done with this question - move to next
            # Force completion after 2 iterations or if validation says complete
            move_to_next = True
            state.question_completed[question_number] = True
            save_question_progress(question_number)
            state.current_question += 1
            print(f"[DEBUG] Q{question_number} marked as complete. Iteration: {iteration}, is_complete: {is_complete}")
        
        return jsonify({
            "success": True,
            "response": nova_response,
            "needs_followup": needs_followup,
            "move_to_next": move_to_next,
            "iteration": iteration,
            "is_complete": is_complete
        })
    
    else:
        # Generic handling for questions not yet implemented
        return jsonify({
            "success": False,
            "error": f"Question {question_number} not yet implemented"
        })


def kill_process_on_port(port):
    """Kill any process running on the specified port."""
    try:
        if sys.platform in ['darwin', 'linux']:
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
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
                
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid and pid.strip():
                            print(f"Force stopping process {pid}...")
                            subprocess.run(['kill', '-9', pid.strip()], 
                                         capture_output=True)
                            print(f"Process {pid} stopped.")
                    return True
                return True
    except Exception as e:
        print(f"Warning: Could not check for processes on port {port}: {e}")
    return False


if __name__ == '__main__':
    print("Starting NOVA Career Coach API server (Questions 16-22)...")
    print(f"Loaded {len(QUESTIONS)} questions")
    
    port = int(os.getenv('PORT', 5003))  # Week 3 runs on port 5003
    
    # Check and kill any existing process on the port
    print(f"Checking port {port}...")
    if kill_process_on_port(port):
        import time
        time.sleep(1)
    
    print(f"Server starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

