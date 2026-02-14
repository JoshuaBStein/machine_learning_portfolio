import os
import json
import time
import uuid
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# --- 🔑 CONFIGURATION ---
SLACK_BOT_TOKEN = "xoxb-add-code-here"
SLACK_APP_TOKEN = "xapp-add-code-here"

# Path logic: Matches the Worker's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUEUE_DIR = os.path.join(BASE_DIR, "queue")

if not os.path.exists(QUEUE_DIR):
    os.makedirs(QUEUE_DIR)

app = App(token=SLACK_BOT_TOKEN)

@app.event("app_mention")
@app.event("message")
def handle_incoming(body, say, context):
    event = body["event"]
    
    # 1. Ignore messages from the bot itself
    if event.get("bot_id"):
        return
    
    # 2. Extract metadata
    thread_ts = event.get("thread_ts", event["ts"])
    user_id = event.get("user")
    
    # Fix the NameError: Define user_text by stripping the bot's handle
    raw_text = event.get("text", "")
    bot_user_id = context.get("bot_user_id")
    user_text = raw_text.replace(f"<@{bot_user_id}>", "").strip()
    
    # 3. Handle PDF Links
    # Slack stores files in a list; we grab the 'url_private' for any PDFs
    pdf_links = []
    if "files" in event:
        pdf_links = [f["url_private"] for f in event["files"] if f.get("filetype") == "pdf"]

    # 4. Save Request to Disk (The "Task")
    task_id = f"{int(time.time())}_{uuid.uuid4().hex[:4]}"
    task_data = {
        "channel": event["channel"],
        "thread_ts": thread_ts,
        "text": user_text,
        "user": user_id,
        "pdf_urls": pdf_links  # This matches the Worker's expected key
    }
    
    task_path = os.path.join(QUEUE_DIR, f"{task_id}.json")
    with open(task_path, "w") as f:
        json.dump(task_data, f)

    # 5. Immediate Feedback
    # Calculate queue position by counting JSON files in the directory
    queue_pos = len([f for f in os.listdir(QUEUE_DIR) if f.endswith(".json")])
    
    status_msg = f"🧬 *Request Received.* You are position *#{queue_pos}* in the queue."
    if pdf_links:
        status_msg += f"\n📄 _Detected {len(pdf_links)} PDF(s) for analysis._"
    
    say(text=status_msg, thread_ts=thread_ts)

if __name__ == "__main__":
    print("⚡️ Broker is running and taking requests...")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
