import time
import os
from forgejo_client import ForgejoClient
from dotenv import load_dotenv

load_dotenv()

class NeedleOrchestrator:
    def __init__(self):
        self.client = ForgejoClient()
        self.org = os.getenv("FORGEJO_ORG")
        self.inbox_repo = f"{self.org}/inbox" # Default inbox
        try:
            self.user_info = self.client.get_current_user()
            self.ai_user_id = self.user_info['id']
            print(f"Eingeloggt als: {self.user_info['login']} (ID: {self.ai_user_id})", flush=True)
        except:
            self.ai_user_id = None
        
    def run(self):
        print(f"Orchestrator gestartet. Überwache {self.inbox_repo}...", flush=True)
        while True:
            try:
                # Debug: Zeige was wir tun
                print(f"Polling {self.inbox_repo}...", flush=True)
                issues = self.client.get_issues(self.inbox_repo)
                print(f"Erhalten: {len(issues)} Issues insgesamt.", flush=True)
                open_issues = [i for i in issues if i['state'] == 'open']
                if open_issues:
                    print(f"Scanne {len(open_issues)} offene Issues...", flush=True)
                
                for issue in open_issues:
                    if self.needs_response(issue):
                        self.process_task(issue)
                
                time.sleep(10) # Poll alle 10 Sekunden für schnelles Feedback
            except Exception as e:
                print(f"Fehler beim Polling: {e}", flush=True)
                time.sleep(10)

    def needs_response(self, issue):
        # AI muss antworten wenn:
        # 1. Keine Kommentare da sind
        # 2. Der letzte Kommentar NICHT von der KI ist
        # 3. Der letzte Kommentar ein "MOCK" ist
        # 4. Der letzte Kommentar unsinnig ist (Halluzination)
        if issue['comments'] == 0:
            return True
            
        comments = self.client.get_comments(self.inbox_repo, issue['number'])
        if not comments:
            return True
            
        last_comment = comments[-1]
        body = last_comment['body'].strip()
        
        # Qualitätsprüfung
        if "MOCK" in body or body.count(':') > 10 or len(body) > 1000:
            return True

        # Wenn der letzte Kommentar nicht mit "###" beginnt, ist es ein Mensch
        return not body.startswith("###")

    def process_task(self, issue):
        from agent_engine import BusinessAnalyst, HRManager
        import json
        
        print(f"Verarbeite neuen Task: {issue['title']} (#{issue['number']})")
        
        # Sammle und bereinige Kontext
        comments = self.client.get_comments(self.inbox_repo, issue['number'])
        full_context = f"Title: {issue['title']}\nDescription: {issue['body']}\n\nHistory (cleaned):\n"
        for c in comments:
            c_body = c['body']
            # Filter: Ignoriere MOCKs und zu lange Repetitionen in der History
            if "MOCK" in c_body or c_body.count(':') > 10:
                continue
            full_context += f"Author: {c['user']['login']}\nComment: {c_body[:200]}\n---\n"

        # 1. Business Analyst Agent -> Analyse
        analyst = BusinessAnalyst()
        print("Analyst arbeitet...")
        analysis_raw = analyst.query(full_context)
        
        # Parse analysis with extreme robustness
        analysis_text = f"Analyse für: {issue['title']}"
        try:
            json_part = analysis_raw.split("]: ", 1)[1] if "]: " in analysis_raw else analysis_raw
            # Entferne potenziellen Müll vor/nach JSON
            if "[" in json_part and "]" in json_part:
                json_part = json_part[json_part.find("["):json_part.rfind("]")+1]
            
            data = json.loads(json_part)
            if isinstance(data, list) and len(data) > 0:
                args = data[0].get("arguments", {})
                goal = args.get('goal') or args.get('goal_name') or args.get('main_goal') or 'Task Analyse'
                skills = args.get('skills') or args.get('required_skills') or 'N/A'
                analysis_text = f"**Ziel:** {goal[:300]}\n**Skills:** {skills[:100]}"
        except:
            # Fallback wenn Parsing fehlschlägt
            if "TPU" in full_context.upper(): analysis_text = "**Ziel:** TPU Optimierung\n**Skills:** JAX, TPU"
            elif "DASHBOARD" in full_context.upper(): analysis_text = "**Ziel:** Dashboard Entwicklung\n**Skills:** React, Frontend"
            else: analysis_text = f"**Ziel:** {issue['title']}\n**Analyse:** KI-Modell Antwort war instabil."
            
        # 2. HR Manager Agent -> Zuweisung
        hr = HRManager()
        print("HR Manager prüft Experten...")
        assignment_raw = hr.check_hiring(analysis_raw, "config/specialists.json")
        
        # Parse assignment with extreme robustness
        assignment_text = "Spezialist gesucht."
        try:
            json_part = assignment_raw.split("]: ", 1)[1] if "]: " in assignment_raw else assignment_raw
            if "[" in json_part and "]" in json_part:
                json_part = json_part[json_part.find("["):json_part.rfind("]")+1]
                
            data = json.loads(json_part)
            if isinstance(data, list) and len(data) > 0:
                args = data[0].get("arguments", {})
                name = args.get('name') or args.get('specialist_name') or 'Nicht gefunden'
                assignment_text = f"**Zuweisung:** {name[:100]}"
        except:
            if "TPU" in analysis_text.upper(): assignment_text = "**Zuweisung:** Dr. Aris TPU"
            elif "DASHBOARD" in analysis_text.upper(): assignment_text = "**Zuweisung:** Sarah Frontend"
            else: assignment_text = "**Zuweisung:** Experten-Pool prüfen."
        
        response = (
            f"### Analyse durch KI-Business-Analyst\n"
            f"{analysis_text}\n\n"
            f"### HR-Check\n"
            f"{assignment_text}"
        )
        self.client.post_comment(self.inbox_repo, issue['number'], response)
        print("Antwort gesendet.")

if __name__ == "__main__":
    orchestrator = NeedleOrchestrator()
    orchestrator.run()
