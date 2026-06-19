# app.py
import os, re, json, uuid
from flask import Flask, request, session, jsonify, make_response
from flask import send_file
from PyPDF2 import PdfReader

# --- OpenAI (simple usage) ---
try:
    from openai import OpenAI
    OPENAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    OPENAI = None  # so the file still imports without the SDK installed

APP = Flask(__name__)
APP.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-" + uuid.uuid4().hex)

PDF_PATH = os.getenv("PROMPTS_PDF", "/mnt/data/Copy of Week 1 prompts.pdf")
HTML_PATH = os.getenv("INDEX_HTML", "/mnt/data/index.html")

# ---------- Utilities ----------
def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def build_flow_from_pdf(txt: str):
    """
    Parse the document into an ordered list of steps:
      {"type":"print","text":...}
      {"type":"ask","qnum":1,"question": "...", "sys_prompt": "..."}
    """
    # Normalize spacing
    t = re.sub(r"[ \t]+", " ", txt)
    t = re.sub(r"\n{2,}", "\n\n", t)

    # Capture all Qx system prompts first into a dict
    sys_prompts = {}
    for m in re.finditer(r"(Q(\d+)\s*SYSTEM PROMPT:)(.*?)(?=Q\d+\s*SYSTEM PROMPT:|ASK THIS QUESTION:|PRINT THIS:|\Z)", t, flags=re.S):
        qnum = int(m.group(2))
        body = m.group(3).strip()
        sys_prompts[qnum] = body

    # Walk the doc in order, picking PRINT/ASK blocks
    # We’ll scan sequentially, emitting steps as they appear
    steps = []
    i = 0
    tokens = list(re.finditer(r"(PRINT THIS:|ASK THIS QUESTION:)", t))
    for idx, tok in enumerate(tokens):
        label = tok.group(1)
        start = tok.end()
        end = tokens[idx + 1].start() if idx + 1 < len(tokens) else len(t)
        chunk = t[start:end].strip()

        if label == "PRINT THIS:":
            # Use the raw chunk, keep line breaks
            steps.append({"type": "print", "text": chunk})
        else:
            # ASK THIS QUESTION: → expect a leading "X. question"
            q_match = re.search(r"^\s*(\d+)\s*\.\s*(.+?)(?:\n|$)", chunk, flags=re.S)
            if not q_match:
                continue
            qnum = int(q_match.group(1))
            question_line = q_match.group(2).strip()
            sys_prompt = sys_prompts.get(qnum, "").strip()
            steps.append({
                "type": "ask",
                "qnum": qnum,
                "question": question_line,
                "sys_prompt": sys_prompt
            })
    # Keep only the first 8 asks as specified
    asks_seen = 0
    pruned = []
    for s in steps:
        if s["type"] == "ask":
            asks_seen += 1
            if asks_seen > 8:
                continue
        pruned.append(s)
    return pruned

def load_flow():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Missing PDF at {PDF_PATH}")
    txt = read_pdf_text(PDF_PATH)
    return build_flow_from_pdf(txt)

def load_index_html():
    if not os.path.exists(HTML_PATH):
        raise FileNotFoundError(f"Missing index.html at {HTML_PATH}")
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()

def ensure_session():
    if "chat" not in session:
        session["chat"] = []  # list of {"role":"nova/user", "content": "..."}
    if "state" not in session:
        session["state"] = {
            "Name": None,
            "answers": {},  # qnum -> {"user": [...], "nova": [...]}
        }
    if "flow" not in session:
        session["flow"] = load_flow()
    if "cursor" not in session:
        session["cursor"] = 0
    if "ask_iter" not in session:
        session["ask_iter"] = 0

def add_msg(role, text):
    session["chat"].append({"role": role, "content": text})

def substitute_tokens(text: str, state: dict) -> str:
    # Replace {Name}, {Answer to question X ...}, and other curly placeholders we have in state
    def get_answer(qn):
        a = state["answers"].get(qn, {})
        parts = a.get("user", [])
        return parts[-1] if parts else ""
    # Simple subs
    replacements = {
        "{Name}": state.get("Name") or "Friend"
    }
    # Common answer placeholders in the PDF
    text = text.replace("{Answer from 2 (reason for participating in DRIVEN)}", get_answer(2))
    text = text.replace("{Answer to 2 (reason for participating in DRIVEN)}", get_answer(2))
    text = text.replace("{Answer to question 4 (goal selected)}", get_answer(4))
    text = text.replace("{Answer to questions 4 (goal selected)}", get_answer(4))
    text = text.replace("{Answer to question 6 (steps to achieving goal)}", get_answer(6))
    text = text.replace("{Answers to questions 7 (expected barriers)}", get_answer(7))

    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def llm_complete(system_prompt, user_input):
    if OPENAI is None:
        # Fallback stub to let UI function without the SDK; echoes a gentle message
        return "(LLM unavailable) Thanks! I captured your response. Let's keep going."
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    resp = OPENAI.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=msgs,
        temperature=0.6,
    )
    return resp.choices[0].message.content.strip()

def push_until_next_ask():
    """Emit any consecutive PRINT THIS blocks as NOVA messages."""
    ensure_session()
    pushed = []
    flow = session["flow"]
    cur = session["cursor"]
    while cur < len(flow) and flow[cur]["type"] == "print":
        text = substitute_tokens(flow[cur]["text"], session["state"])
        add_msg("nova", text)
        pushed.append({"role": "nova", "content": text})
        cur += 1
    session["cursor"] = cur
    session.modified = True
    return pushed

def get_current_ask():
    ensure_session()
    flow = session["flow"]
    cur = session["cursor"]
    if cur < len(flow) and flow[cur]["type"] == "ask":
        return flow[cur]
    return None

def start_if_needed():
    ensure_session()
    # Capture name up-front if the very first PRINT includes "Hello {name}!"
    if session["state"]["Name"] is None:
        session["state"]["Name"] = "Friend"
    return push_until_next_ask()

# ---------- Routes ----------
@APP.route("/")
def home():
    html = load_index_html()

    # inject a tiny, non-intrusive helper script to wire up the chat
    helper_js = r"""
    <script>
    (function(){
      const stream = document.querySelector(".chat-stream");
      const textarea = document.querySelector("textarea");
      const sendBtn = document.querySelector(".primary-button");
      const voiceText = document.querySelector(".voice-status-text");

      function renderMessage(role, text){
        const article = document.createElement("article");
        article.className = "message " + (role === "nova" ? "from-nova" : "from-user");
        const header = document.createElement("header");
        const who = document.createElement("span");
        who.textContent = role === "nova" ? "NOVA" : "You";
        header.appendChild(who);
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.textContent = text;
        article.appendChild(header);
        article.appendChild(bubble);
        stream.appendChild(article);
        stream.scrollTop = stream.scrollHeight;
      }

      async function fetchJSON(url, body){
        const res = await fetch(url, {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify(body || {})
        });
        return await res.json();
      }

      async function boot(){
        const data = await fetchJSON("/start");
        data.messages.forEach(m => renderMessage(m.role, m.content));
        if (data.pending_question){
          renderMessage("nova", data.pending_question);
        }
      }

      sendBtn?.addEventListener("click", async () => {
        const text = textarea.value.trim();
        if (!text) return;
        renderMessage("user", text);
        textarea.value = "";
        const data = await fetchJSON("/send", { text });
        (data.messages || []).forEach(m => renderMessage(m.role, m.content));
        if (data.pending_question){
          renderMessage("nova", data.pending_question);
        }
      });

      window.addEventListener("load", boot);
    })();
    </script>
    </body></html>
    """

    # place just before closing tags
    html = re.sub(r"</body>\s*</html>\s*$", helper_js, html, flags=re.I)
    resp = make_response(html)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp

@APP.route("/start", methods=["POST"])
def start():
    msgs = start_if_needed()
    ask = get_current_ask()
    pending_question = None
    if ask:
        pending_question = substitute_tokens(f"{ask['qnum']}. {ask['question']}", session["state"])
    return jsonify({"messages": msgs, "pending_question": pending_question})

@APP.route("/send", methods=["POST"])
def send():
    ensure_session()
    payload = request.get_json(force=True) or {}
    user_text = (payload.get("text") or "").strip()
    if not user_text:
        return jsonify({"messages": []})

    # Must have an active ASK step
    ask = get_current_ask()
    if not ask:
        # if no current ask, just echo and push prints
        add_msg("user", user_text)
        more = push_until_next_ask()
        return jsonify({"messages": more})

    qn = ask["qnum"]
    add_msg("user", user_text)
    ans = session["state"]["answers"].setdefault(qn, {"user": [], "nova": []})
    ans["user"].append(user_text)

    # Fill the system prompt with any available tokens
    sys_prompt = substitute_tokens(ask["sys_prompt"], session["state"])

    # Two-iteration loop: first LLM response; if it asks a follow-up, we allow one more user reply later
    iteration = session.get("ask_iter", 0)
    nova_reply = llm_complete(sys_prompt, user_text)
    add_msg("nova", nova_reply)
    ans["nova"].append(nova_reply)
    session["ask_iter"] = iteration + 1
    session.modified = True

    messages_out = [{"role": "nova", "content": nova_reply}]

    # If we just completed the second iteration OR LLM reply doesn't look like a question, move to next step
    is_question = ("?" in nova_reply)
    if session["ask_iter"] >= 2 or not is_question:
        # If this question captures the Name, try a simple heuristic on Q1
        if qn == 1 and session["state"].get("Name") in (None, "Friend"):
            # naive name capture from first sentence
            m = re.search(r"\b(?:I am|I'm|My name is)\s+([A-Z][a-zA-Z\-']+)", user_text)
            if m: session["state"]["Name"] = m.group(1)

        # advance cursor past this ASK
        session["cursor"] += 1
        session["ask_iter"] = 0

        # After advancing, push any immediate PRINT blocks
        messages_out += push_until_next_ask()

        # If the next step is an ASK, present the question
        nxt = get_current_ask()
        pending_question = None
        if nxt:
            pending_question = substitute_tokens(f"{nxt['qnum']}. {nxt['question']}", session["state"])
        else:
            pending_question = None
        return jsonify({"messages": messages_out, "pending_question": pending_question})

    # Else, stay on this question and wait for the user's follow-up
    return jsonify({"messages": messages_out})

# Optional: simple download of the raw transcript
@APP.route("/transcript.txt")
def transcript():
    ensure_session()
    lines = []
    for m in session["chat"]:
        who = "NOVA" if m["role"] == "nova" else "You"
        lines.append(f"{who}: {m['content']}")
    data = "\n\n".join(lines).encode("utf-8")
    fname = "transcript.txt"
    return make_response((data, 200, {
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Disposition": f'attachment; filename="{fname}"'
    }))

if __name__ == "__main__":
    print("Starting NOVA (Week 1) — http://127.0.0.1:5000")
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
