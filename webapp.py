import time
import os
import threading
import json
from flask import Flask, render_template, jsonify, request
from orchestrator.forgejo_client import ForgejoClient
from orchestrator.agent_engine import BusinessAnalyst, HRManager
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Globaler State für das Dashboard
logs = []
status = {
    "version": "1.0.1-fixed",
    "org": os.getenv("FORGEJO_ORG", "Not Configured"),
    "repos": 0,
    "agents": ["Business Analyst", "HR Manager"]
}

def add_log(msg):
    timestamp = time.strftime("%H:%M:%S")
    logs.append(f"[{timestamp}] {msg}")
    if len(logs) > 100:
        logs.pop(0)
    print(msg)

class OnlineOrchestrator(threading.Thread):
    def __init__(self):
        super().__init__()
        self.client = ForgejoClient()
        self.org = os.getenv("FORGEJO_ORG")
        self.inbox_repo = f"{self.org}/inbox"
        self.running = True
        self.daemon = True
        self.analyst = BusinessAnalyst()
        self.hr = HRManager()

    def run(self):
        add_log(f"Orchestrator gestartet. Überwache {self.inbox_repo}...")
        while self.running:
            try:
                # Update Status
                repos = self.client.get_org_repos(self.org)
                status["repos"] = len(repos)
                
                # Check Inbox
                issues = self.client.get_issues(self.inbox_repo)
                for issue in issues:
                    if issue['state'] == 'open' and issue['comments'] == 0:
                        self.process_task(issue)
                
                time.sleep(10)
            except Exception as e:
                add_log(f"Fehler: {e}")
                time.sleep(10)

    def process_task(self, issue):
        add_log(f"Neu: {issue['title']} (#{issue['number']})")
        
        # 1. Analyst
        add_log("Alex (Analyst) analysiert...")
        analysis_raw = self.analyst.query(issue['body'] or issue['title'])
        analysis_text = analysis_raw
        try:
            json_part = analysis_raw.split("]: ", 1)[1] if "]: " in analysis_raw else analysis_raw
            data = json.loads(json_part)
            if isinstance(data, list) and len(data) > 0:
                args = data[0].get("arguments", {})
                analysis_text = f"**Ziel:** {args.get('goal', 'N/A')}\n**Skills:** {args.get('skills', 'N/A')}"
        except:
            pass
        
        # 2. HR
        add_log("Jordan (HR) prüft Experten...")
        hr_raw = self.hr.check_hiring(analysis_raw, "config/specialists.json")
        hr_text = hr_raw
        try:
            json_part = hr_raw.split("]: ", 1)[1] if "]: " in hr_raw else hr_raw
            data = json.loads(json_part)
            if isinstance(data, list) and len(data) > 0:
                args = data[0].get("arguments", {})
                hr_text = f"**Zuweisung:** {args.get('name', 'N/A')}"
        except:
            pass
        
        response = f"### Alex (Analyst):\n{analysis_text}\n\n### Jordan (HR):\n{hr_text}"
        self.client.post_comment(self.inbox_repo, issue['number'], response)
        add_log(f"Task #{issue['number']} verarbeitet.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify(status)

@app.route('/api/logs')
def get_logs():
    return jsonify({"logs": logs})

@app.route('/api/logs/clear', methods=['POST'])
def clear_logs():
    global logs
    logs = []
    return jsonify({"status": "ok"})

@app.route('/api/trigger', methods=['POST'])
def trigger():
    data = request.json
    client = ForgejoClient()
    org = os.getenv("FORGEJO_ORG")
    client.create_issue(f"{org}/inbox", data['title'], "Manueller Task via Dashboard.")
    add_log(f"Manueller Task ausgelöst: {data['title']}")
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    # Start Orchestrator
    orchestrator = OnlineOrchestrator()
    orchestrator.start()
    
    # Start Webserver
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
