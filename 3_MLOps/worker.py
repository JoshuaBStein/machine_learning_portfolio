import os, json, time, io, requests, PyPDF2, psutil
from slack_sdk import WebClient
from mlx_lm import load, generate

# --- 🔑 CONFIGURATION ---
SLACK_BOT_TOKEN = "xoxb-add-code-here"
RESPONSIBLE_PERSON_ID = "add_technician_slack_member_id" # Your ID
MODEL_NAME = "mlx-community/gemma-3-27b-it-4bit"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUEUE_DIR = os.path.join(BASE_DIR, "queue")

client = WebClient(token=SLACK_BOT_TOKEN)
print(f"⏳ Loading {MODEL_NAME}...")
model, tokenizer = load(MODEL_NAME)
print("✅ Worker Online.")

# --- 🔄 MAIN PROTECTION LOOP ---
try:
    while True:
        files = sorted([f for f in os.listdir(QUEUE_DIR) if f.endswith(".json")])
        if not files:
            time.sleep(2)
            continue
        
        task_file = files[0]
        task_path = os.path.join(QUEUE_DIR, task_file)
        
        try:
            with open(task_path, "r") as f:
                task = json.load(f)
            
            # --- STRESS TEST TRIGGER ---
            if "SYSTEM_STRESS_TEST" in task['text']:
                print("🚨 Stress test triggered! Forcing crash...")
                raise Exception("Manual Stress Test Triggered by User")

            # PDF Extraction
            context = ""
            for url in task.get("pdf_urls", []):
                context += read_pdf(url, SLACK_BOT_TOKEN)

            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": "You are a lab assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {task['text']}" if context else task['text']}
            ], tokenize=False, add_generation_prompt=True)

            response = generate(model, tokenizer, prompt=prompt, max_tokens=1000000)
            
            # Final Message back to thread
            client.chat_postMessage(channel=task["channel"], thread_ts=task["thread_ts"], text=response)

        except Exception as e:
            if "Manual Stress Test" in str(e): raise e
            client.chat_postMessage(channel=task["channel"], thread_ts=task["thread_ts"], text=f"⚠️ *Task Error:* `{e}`")
        
        finally:
            if os.path.exists(task_path): os.remove(task_path)

except Exception as fatal_error:
    # --- 🚨 CRASH SOS (BROADCAST & PERSONAL DM) 🚨 ---
    print(f"🛑 FATAL ERROR: {fatal_error}")
    
    crash_report = (
        f"🚨 *CRITICAL SYSTEM FAILURE*\n"
        f"The LLM system has crashed and is now *OFFLINE*.\n"
        f"Responsible Party: <@{RESPONSIBLE_PERSON_ID}>\n"
        f"📑 *Error:* `{str(fatal_error)}`"
    )

    # 1. SEND PERSONAL DM TO YOU
    try:
        client.chat_postMessage(
            channel=RESPONSIBLE_PERSON_ID, 
            text=f"⚠️ *URGENT:* The LLM Worker has crashed on the Mac Studio. \nError: `{str(fatal_error)}`"
        )
    except Exception as dm_err:
        print(f"Could not send DM: {dm_err}")

    # 2. Notify the person whose task caused the crash in the thread
    if 'task' in locals():
        client.chat_postMessage(channel=task["channel"], thread_ts=task["thread_ts"], text=crash_report)

    # 3. Notify everyone else currently in line
    remaining_tasks = [f for f in os.listdir(QUEUE_DIR) if f.endswith(".json")]
    for f_name in remaining_tasks:
        try:
            with open(os.path.join(QUEUE_DIR, f_name), "r") as f:
                waiting_task = json.load(f)
            client.chat_postMessage(
                channel=waiting_task["channel"], 
                thread_ts=waiting_task["thread_ts"], 
                text=f"🛑 *System Offline:* A fatal error occurred. The queue is cleared. Contact <@{RESPONSIBLE_PERSON_ID}> for a reboot."
            )
        except:
            pass
