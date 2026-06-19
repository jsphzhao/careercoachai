import os

from flask import Flask, send_from_directory, abort
from dotenv import load_dotenv
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

# Load environment variables once for all sub-apps
load_dotenv()

# Import each week's Flask app (they remain independent for debugging)
from week1_main import app as week1_app
from week2_main import app as week2_app
from week3_main import app as week3_app
from week4_main import app as week4_app
from week5_main import app as week5_app


def create_root_app():
    """Root Flask app that serves the shared frontend assets."""
    root = Flask(__name__)

    def _serve_file(path):
        try:
            return send_from_directory(os.getcwd(), path)
        except FileNotFoundError:
            abort(404)

    @root.route('/')
    def serve_index():
        return _serve_file('index.html')

    @root.route('/<path:path>')
    def serve_static(path):
        return _serve_file(path)

    @root.route('/health')
    def health():
        return {"status": "ok"}

    return root


root_app = create_root_app()

# Mount each week's Flask app under its own prefix so their routes remain unchanged
application = DispatcherMiddleware(root_app, {
    '/week1': week1_app,
    '/week2': week2_app,
    '/week3': week3_app,
    '/week4': week4_app,
    '/week5': week5_app,
})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5007))
    run_simple('0.0.0.0', port, application, use_reloader=False, use_debugger=os.getenv('FLASK_DEBUG') == '1')

