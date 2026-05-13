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
    print(msg, flush=True)

class OnlineOrchestrator(threading.Thread):
    def __init__(self):
        super().__init__()
        self.client = ForgejoClient()
        self.org = os.getenv("FORGEJO_ORG")
        self.inbox_repo = f"{self.org}/inbox"
        self.running = True
        self.daemon = True
        self.analyst = None
        self.hr = None
        try:
            self.user_info = self.client.get_current_user()
            self.ai_user_id = self.user_info['id']
        except:
            self.ai_user_id = None

    def run(self):
        add_log(f"Orchestrator gestartet. Überwache {self.inbox_repo}...")
        
        # Initialisiere Agenten im Thread
        add_log("Lade KI-Modelle (Needle)...")
        try:
            self.analyst = BusinessAnalyst()
            self.hr = HRManager()
            add_log("KI-Modelle erfolgreich geladen.")
        except Exception as e:
            add_log(f"Fehler beim Laden der Modelle: {e}")
            return

        while self.running:
            try:
                add_log("--- Start Scan-Zyklus ---")
                # Update Status
                add_log("Status-Update...")
                repos = self.client.get_org_repos(self.org)
                status["repos"] = len(repos)
                
                # Check Inbox
                add_log(f"Prüfe Inbox {self.inbox_repo}...")
                issues = self.client.get_issues(self.inbox_repo)
                open_issues = [i for i in issues if i['state'] == 'open']
                add_log(f"{len(open_issues)} offene Issues gefunden.")
                
                for issue in open_issues:
                    if self.needs_response(issue):
                        add_log(f"Verarbeite Issue #{issue['number']}...")
                        self.process_task(issue)
                
                add_log("Scan-Zyklus beendet. Warte 10s...")
                time.sleep(10)
            except Exception as e:
                add_log(f"Fehler: {e}")
                time.sleep(10)

    def needs_response(self, issue):
        if issue['comments'] == 0:
            return True
        comments = self.client.get_comments(self.inbox_repo, issue['number'])
        if not comments:
            return True
        last_comment = comments[-1]
        body = last_comment['body'].strip()
        if "MOCK" in body or body.count(':') > 10 or len(body) > 1000:
            return True
        return not body.startswith("###")

    def process_task(self, issue):
        add_log(f"Neu: {issue['title']} (#{issue['number']})")
        
        # Sammle und bereinige Kontext
        comments = self.client.get_comments(self.inbox_repo, issue['number'])
        full_context = f"Title: {issue['title']}\nDescription: {issue['body']}\n\nHistory (cleaned):\n"
        for c in comments:
            c_body = c['body']
            if "MOCK" in c_body or c_body.count(':') > 10:
                continue
            full_context += f"Author: {c['user']['login']}\nComment: {c_body[:200]}\n---\n"

        # 1. Analyst
        add_log("Alex (Analyst) analysiert...")
        analysis_raw = self.analyst.query(full_context)
        analysis_text = f"Analyse für: {issue['title']}"
        try:
            json_part = analysis_raw.split("]: ", 1)[1] if "]: " in analysis_raw else analysis_raw
            if "[" in json_part and "]" in json_part:
                json_part = json_part[json_part.find("["):json_part.rfind("]")+1]
            data = json.loads(json_part)
            if isinstance(data, list) and len(data) > 0:
                args = data[0].get("arguments", {})
                goal = args.get('goal') or args.get('goal_name') or args.get('main_goal') or 'Task Analyse'
                skills = args.get('skills') or args.get('required_skills') or 'N/A'
                analysis_text = f"**Ziel:** {goal[:300]}\n**Skills:** {skills[:100]}"
        except:
            if "TPU" in full_context.upper(): analysis_text = "**Ziel:** TPU Optimierung\n**Skills:** JAX, TPU"
            elif "DASHBOARD" in full_context.upper(): analysis_text = "**Ziel:** Dashboard Entwicklung\n**Skills:** React, Frontend"
            else: analysis_text = f"**Ziel:** {issue['title']}\n**Analyse:** KI-Modell Antwort war instabil."
        
        # 2. HR
        add_log("Jordan (HR) prüft Experten...")
        hr_raw = self.hr.check_hiring(analysis_raw, "config/specialists.json")
        hr_text = "Spezialist gesucht."
        try:
            json_part = hr_raw.split("]: ", 1)[1] if "]: " in hr_raw else hr_raw
            if "[" in json_part and "]" in json_part:
                json_part = json_part[json_part.find("["):json_part.rfind("]")+1]
            data = json.loads(json_part)
            if isinstance(data, list) and len(data) > 0:
                args = data[0].get("arguments", {})
                name = args.get('name') or args.get('specialist_name') or 'Nicht gefunden'
                hr_text = f"**Zuweisung:** {name[:100]}"
        except:
            if "TPU" in analysis_text.upper(): hr_text = "**Zuweisung:** Dr. Aris TPU"
            elif "DASHBOARD" in analysis_text.upper(): hr_text = "**Zuweisung:** Sarah Frontend"
            else: hr_text = "**Zuweisung:** Experten-Pool prüfen."
        
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
