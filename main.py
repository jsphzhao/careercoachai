import os
import re
import json
import uuid
import subprocess
import sys
from flask import Flask, request, jsonify, session, Response
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

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

# Path to prompts file
PROMPTS_FILE = "Copy of Week 1 prompts.txt"


class DialogueManager:
    """Manages the dialogue flow from the prompts document."""
    
    def __init__(self, prompts_file):
        self.prompts_file = prompts_file
        self.steps = []
        self.system_prompts = {}
        self.data_requirements = {}
        self.load_document()
    
    def load_document(self):
        """Parse the prompts document into structured steps."""
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # First, extract all QX SYSTEM PROMPT sections
        system_prompt_pattern = r'Q(\d+)\s+SYSTEM PROMPT:\s*(.*?)(?=Q\d+\s+SYSTEM PROMPT:|ASK THIS QUESTION:|PRINT THIS:|\Z)'
        for match in re.finditer(system_prompt_pattern, content, re.DOTALL):
            qnum = int(match.group(1))
            prompt_text = match.group(2).strip()
            self.system_prompts[qnum] = prompt_text
        
        # Extract all steps in order (PRINT THIS and ASK THIS QUESTION)
        # Handle both "ASK THIS QUESTION:" and "ASK THIS QUESTION" (without colon)
        # Use a more precise pattern that handles the document structure
        step_pattern = r'(PRINT THIS:|ASK THIS QUESTION:?)\s*\n\s*(.*?)(?=\n\s*(?:PRINT THIS:|ASK THIS QUESTION:?|Q\d+\s+SYSTEM PROMPT:)|\Z)'
        for match in re.finditer(step_pattern, content, re.DOTALL):
            step_type = match.group(1).strip().rstrip(':')
            step_content = match.group(2).strip()
            
            if step_type == "PRINT THIS":
                # Clean up the text (remove quotes if present)
                text = step_content.strip().strip('"').strip()
                self.steps.append({
                    "type": "print",
                    "text": text
                })
            elif step_type == "ASK THIS QUESTION":
                # Extract question number and question text
                # Question text is everything until "Data required" or end
                question_match = re.match(r'(\d+)\.\s*(.+?)(?:\s+Data required|$)', step_content, re.DOTALL | re.IGNORECASE)
                if question_match:
                    qnum = int(question_match.group(1))
                    question_text = question_match.group(2).strip()
                    
                    # Extract data requirements if present (case insensitive)
                    data_required = []
                    data_section = re.search(r'Data required[:\s]*(.*?)(?=Q\d+\s+SYSTEM PROMPT:|ASK THIS QUESTION:?|PRINT THIS:|\Z)', 
                                            step_content, re.DOTALL | re.IGNORECASE)
                    if data_section:
                        data_text = data_section.group(1)
                        # Extract all {variable} patterns
                        data_vars = re.findall(r'\{([^}]+)\}', data_text)
                        data_required = data_vars
                    
                    self.data_requirements[qnum] = data_required
                    
                    self.steps.append({
                        "type": "ask",
                        "qnum": qnum,
                        "question": question_text,
                        "system_prompt": self.system_prompts.get(qnum, ""),
                        "data_required": data_required
                    })
    
    def get_steps(self):
        return self.steps


# Initialize dialogue manager
dialogue_manager = DialogueManager(PROMPTS_FILE)


class ConversationState:
    """Manages state for a single conversation."""
    
    def __init__(self, name):
        self.name = name
        self.current_step = 0
        self.current_question = None  # Current question number being answered
        self.answers = {}  # qnum -> list of user responses
        self.nova_responses = {}  # qnum -> list of NOVA responses
        self.iteration_count = {}  # qnum -> iteration count (for 2-iteration loop)
        self.data_store = {
            "Name": name,
            "Notes on thinking flexibly from videos and homework: Why think flexibly: It helps you adapt to change, solve problems creatively, and understand different perspectives, making you more resilient and effective in an unpredictable world. How to practice: Actively challenge your initial reactions by asking \"What's another way to see this?\" or \"What would someone I respect but disagree with say?\" How to strengthen it: Regularly expose yourself to unfamiliar ideas, disciplines, and experiences, as mental flexibility grows strongest when you step outside familiar patterns and genuinely consider viewpoints that initially feel uncomfortable.": "Why think flexibly: It helps you adapt to change, solve problems creatively, and understand different perspectives, making you more resilient and effective in an unpredictable world. How to practice: Actively challenge your initial reactions by asking \"What's another way to see this?\" or \"What would someone I respect but disagree with say?\" How to strengthen it: Regularly expose yourself to unfamiliar ideas, disciplines, and experiences, as mental flexibility grows strongest when you step outside familiar patterns and genuinely consider viewpoints that initially feel uncomfortable.",
            "List of homework questions:": "Situation that bothered you - where were you, what were you doing, and who were you with?\nHow it made you feel? (Rate the intensity of your emotions from 0-10, with 0 being an insignificant emotion and 10 being an extremely intense emotion)\nThoughts (What were you thinking during the event?)\nAlternate viewpoints (How might your coach see the situation?)",
            "Homework questions:": "Situation that bothered you - where were you, what were you doing, and who were you with?\nHow it made you feel? (Rate the intensity of your emotions from 0-10, with 0 being an insignificant emotion and 10 being an extremely intense emotion)\nThoughts (What were you thinking during the event?)\nAlternate viewpoints (How might your coach see the situation?)",
            "Dictionary of Clarifying questions:": {
                "Personal examples": ["Take a fitness class", "Read a book for pleasure", "Try a new recipe"],
                "Professional examples": ["Update your resume", "Refresh your linkedin profile", "Search online for job openings"]
            },
            "Dict of goal categories": {
                "Personal examples": ["Take a fitness class", "Read a book for pleasure", "Try a new recipe"],
                "Professional examples": ["Update your resume", "Refresh your linkedin profile", "Search online for job openings"]
            },
            "Notes on breaking down tasks from videos and homework": "Break down large goals into smaller, manageable steps. Set specific times and locations for each step. Identify what skills or information you might need and where to get them.",
            "List of Clarifying questions:": [
                "What is step 1, exactly when and where will you do it?",
                "What skills/info are missing, and who/what can help?",
                "What's the smallest workable version of the goal (MVP)?",
                "What milestone can you hit in 1 hour? In 2 weeks?"
            ],
            "How thinking flexibly can help one navigate the job market (clarifying questions)": """How thinking flexibly can help one navigate the job market:
* There isn't just one right path—try a few good options and you'll create more chances.
* Each "no" is data—tweak one small thing and you're already improving for next time.
* On hard days, shrink the task; five minutes forward still counts.
* Progress has many forms—networking, a draft, a message; stack small wins for future searches.
* Experiment like a scientist—test, learn, adjust, and momentum will follow in the job market.""",
            "Dict of Article URLs": {
                "Asana - What Are SMART Goals? Examples and Templates [2025]": "https://asana.com/resources/smart-goals",
                "Atlassian - How to write SMART goals (with examples)": "https://www.atlassian.com/blog/productivity/how-to-write-smart-goals",
                "Indeed - How To Write SMART Goals (With Examples)": "https://www.indeed.com/career-advice/career-development/how-to-write-smart-goals"
            }
        }
    
    def get_answer(self, qnum):
        """Get the last answer for a question number."""
        answers = self.answers.get(qnum, [])
        return answers[-1] if answers else ""
    
    def substitute_variables(self, text):
        """Substitute variables in text with actual values."""
        # Replace {Name}
        text = text.replace("{Name}", self.name)
        text = text.replace("{name}", self.name)
        
        # Replace answer references
        text = re.sub(r'\{Answer to 2 \(reason for participating in DRIVEN\)\}', self.get_answer(1), text)
        text = re.sub(r'\{Answer from 2 \(reason for participating in DRIVEN\)\}', self.get_answer(1), text)
        text = re.sub(r'\{Answer to question 4 \(goal selected\)\}', self.get_answer(4), text)
        text = re.sub(r'\{Answer to questions 4 \(goal selected\)\}', self.get_answer(4), text)
        text = re.sub(r'\{Answer to question 6 \(steps to achieving goal\)\}', self.get_answer(6), text)
        text = re.sub(r'\{Answer to questions 6 \(steps to achieving goal\)\}', self.get_answer(6), text)
        text = re.sub(r'\{Answers to questions 7 \(expected barriers\)\}', self.get_answer(7), text)
        text = re.sub(r'\{Answer to questions 7 \(expected barriers\)\}', self.get_answer(7), text)
        
        # Replace data store variables
        for key, value in self.data_store.items():
            if isinstance(value, dict):
                # Format dictionary as text
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
        
        return text
    
    def check_data_requirements(self, qnum):
        """Check if all required data is available for a question."""
        if qnum not in dialogue_manager.data_requirements:
            return True, []
        
        required = dialogue_manager.data_requirements[qnum]
        missing = []
        
        for req in required:
            # Check if it's a Name requirement
            if req == "Name":
                if not self.name or self.name == "Friend":
                    missing.append(req)
            # Check if it's an answer requirement
            elif "Answer to" in req or "Answer from" in req:
                # Extract question number from requirement
                match = re.search(r'(\d+)', req)
                if match:
                    req_qnum = int(match.group(1))
                    if not self.get_answer(req_qnum):
                        missing.append(req)
            # Check if it's in data store
            elif req not in self.data_store:
                missing.append(req)
        
        return len(missing) == 0, missing
    
    def increment_iteration(self, qnum):
        """Increment iteration count for a question."""
        self.iteration_count[qnum] = self.iteration_count.get(qnum, 0) + 1
    
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


@app.route('/')
def index():
    """Serve the index.html file."""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return Response(f.read(), mimetype='text/html')
    except FileNotFoundError:
        return "index.html not found. Please make sure the file exists in the same directory as main.py.", 404


@app.route('/debug/steps')
def debug_steps():
    """Debug endpoint to view all parsed dialogue steps."""
    steps = dialogue_manager.get_steps()
    output = []
    output.append(f"<h1>Parsed Dialogue Steps ({len(steps)} total)</h1>")
    output.append("<style>body { font-family: monospace; padding: 20px; } .step { margin: 20px 0; padding: 10px; border: 1px solid #ccc; } .print { background: #e8f4f8; } .ask { background: #fff4e8; }</style>")
    
    for i, step in enumerate(steps):
        step_class = step["type"]
        output.append(f'<div class="step {step_class}">')
        output.append(f'<strong>Step {i+1}: {step["type"].upper()}</strong><br>')
        
        if step["type"] == "print":
            output.append(f'<pre>{step["text"]}</pre>')
        elif step["type"] == "ask":
            output.append(f'<strong>Question {step["qnum"]}:</strong> {step["question"]}<br>')
            output.append(f'<strong>Data Required:</strong> {step.get("data_required", [])}<br>')
            output.append(f'<strong>System Prompt:</strong> <pre style="max-height: 200px; overflow: auto;">{step.get("system_prompt", "N/A")[:500]}...</pre>')
        
        output.append('</div>')
    
    return Response('\n'.join(output), mimetype='text/html')


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
    state.current_step = 0
    state.current_question = None
    
    # Get first message (should be the welcome PRINT THIS)
    steps = dialogue_manager.get_steps()
    if steps and steps[0]["type"] == "print":
        welcome_text = state.substitute_variables(steps[0]["text"])
        return jsonify({
            "success": True,
            "message": welcome_text
        })
    
    return jsonify({"success": True, "message": f"Hello {name}! Welcome to DRIVEN."})


@app.route('/api/get_next_message', methods=['POST'])
def get_next_message():
    """Get the next message in the dialogue flow."""
    state = get_or_create_state()
    
    steps = dialogue_manager.get_steps()
    
    # Process steps until we find one that needs user input or is complete
    while state.current_step < len(steps):
        step = steps[state.current_step]
        
        if step["type"] == "print":
            # Print messages don't require user response
            message = state.substitute_variables(step["text"])
            state.current_step += 1
            # Return this print message - frontend will call again for next message
            return jsonify({
                "success": True,
                "message": message,
                "is_complete": False,
                "awaiting_response": False
            })
        
        elif step["type"] == "ask":
            qnum = step["qnum"]
            
            # Check data requirements
            data_ready, missing = state.check_data_requirements(qnum)
            if not data_ready:
                # Skip this question for now (shouldn't happen in proper flow)
                state.current_step += 1
                continue
            
            # Ask the question
            question_text = state.substitute_variables(step["question"])
            state.current_question = qnum
            state.iteration_count[qnum] = 0  # Reset iteration count for new question
            
            return jsonify({
                "success": True,
                "message": f"{qnum}. {question_text}",
                "is_complete": False,
                "awaiting_response": True,
                "question_number": qnum
            })
    
    # No more steps
    return jsonify({
        "success": True,
        "message": "Thank you for completing Week 1! I'll be in touch on Friday to see how it went.",
        "is_complete": True,
        "awaiting_response": False
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
    
    if question_number is None:
        return jsonify({"success": False, "error": "No active question"}), 400
    
    # Store user response
    if question_number not in state.answers:
        state.answers[question_number] = []
    state.answers[question_number].append(user_message)
    
    # Find the system prompt for this question
    steps = dialogue_manager.get_steps()
    system_prompt = None
    for step in steps:
        if step["type"] == "ask" and step["qnum"] == question_number:
            system_prompt = step["system_prompt"]
            break
    
    if not system_prompt:
        return jsonify({"success": False, "error": "System prompt not found"}), 400
    
    # Substitute variables in system prompt
    system_prompt = state.substitute_variables(system_prompt)
    
    # Get current iteration before incrementing
    iteration = state.get_iteration(question_number)
    
    # Call LLM to analyze response
    nova_response = call_llm(system_prompt, user_message)
    
    # Store NOVA response
    if question_number not in state.nova_responses:
        state.nova_responses[question_number] = []
    state.nova_responses[question_number].append(nova_response)
    
    # Increment iteration
    state.increment_iteration(question_number)
    iteration = state.get_iteration(question_number)
    
    # Check if we need another iteration (2-iteration loop)
    needs_followup = False
    followup_question = None
    move_to_next = False
    
    # Check if response contains a question (indicating follow-up needed)
    # Also check if LLM is asking for clarification
    is_followup_question = "?" in nova_response or any(
        phrase in nova_response.lower() 
        for phrase in ["can you", "could you", "please", "what", "how", "when", "where", "would you"]
    )
    
    if iteration < 2 and is_followup_question:
        # Need another iteration - stay on this question
        needs_followup = True
        followup_question = None  # The nova_response itself is the follow-up
        move_to_next = False
    else:
        # Done with this question - move to next step
        move_to_next = True
        state.current_question = None
        # Advance to next step (skip the current ask step)
        steps = dialogue_manager.get_steps()
        while state.current_step < len(steps):
            if steps[state.current_step]["type"] == "ask" and steps[state.current_step]["qnum"] == question_number:
                state.current_step += 1
                break
            state.current_step += 1
    
    return jsonify({
        "success": True,
        "response": nova_response,
        "needs_followup": needs_followup,
        "followup_question": followup_question,
        "move_to_next": move_to_next,
        "iteration": iteration
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
    print("Starting NOVA Career Coach API server...")
    print(f"Loaded {len(dialogue_manager.get_steps())} dialogue steps")
    
    port = int(os.getenv('PORT', 5001))
    
    # Check and kill any existing process on the port
    print(f"Checking port {port}...")
    if kill_process_on_port(port):
        import time
        time.sleep(1)  # Give it a moment to fully stop
    
    print(f"Server starting on http://localhost:{port}")
    # Set use_reloader=False to avoid multiprocessing warnings when killing processes
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
