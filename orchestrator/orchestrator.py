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
            print(f"Eingeloggt als: {self.user_info['login']} (ID: {self.ai_user_id})")
        except:
            self.ai_user_id = None
        
    def run(self):
        print(f"Orchestrator gestartet. Überwache {self.inbox_repo}...")
        while True:
            try:
                issues = self.client.get_issues(self.inbox_repo)
                for issue in issues:
                    if issue['state'] == 'open' and self.needs_response(issue):
                        self.process_task(issue)
                
                time.sleep(30) # Poll alle 30 Sekunden
            except Exception as e:
                print(f"Fehler beim Polling: {e}")
                time.sleep(30)

    def needs_response(self, issue):
        # AI muss antworten wenn:
        # 1. Keine Kommentare da sind
        # 2. Der letzte Kommentar NICHT von der KI selbst ist
        if issue['comments'] == 0:
            return True
            
        comments = self.client.get_comments(self.inbox_repo, issue['number'])
        if not comments:
            return True
            
        last_comment = comments[-1]
        return last_comment['user']['id'] != self.ai_user_id

    def process_task(self, issue):
        from agent_engine import BusinessAnalyst, HRManager
        import json
        
        print(f"Verarbeite neuen Task: {issue['title']} (#{issue['number']})")
        
        # Sammle gesamten Kontext (Issue Body + Kommentare)
        comments = self.client.get_comments(self.inbox_repo, issue['number'])
        full_context = f"Title: {issue['title']}\nDescription: {issue['body']}\n\nHistory:\n"
        for c in comments:
            full_context += f"Author: {c['user']['login']}\nComment: {c['body']}\n---\n"

        # 1. Business Analyst Agent -> Analyse
        analyst = BusinessAnalyst()
        print("Analyst arbeitet...")
        analysis_raw = analyst.query(full_context)
        
        # Parse analysis
        analysis_text = analysis_raw
        try:
            json_part = analysis_raw.split("]: ", 1)[1] if "]: " in analysis_raw else analysis_raw
            data = json.loads(json_part)
            if isinstance(data, list) and len(data) > 0:
                args = data[0].get("arguments", {})
                analysis_text = f"**Ziel:** {args.get('goal', 'N/A')}\n**Skills:** {args.get('skills', 'N/A')}"
        except:
            pass
            
        # 2. HR Manager Agent -> Zuweisung
        hr = HRManager()
        print("HR Manager prüft Experten...")
        assignment_raw = hr.check_hiring(analysis_raw, "config/specialists.json")
        
        # Parse assignment
        assignment_text = assignment_raw
        try:
            json_part = assignment_raw.split("]: ", 1)[1] if "]: " in assignment_raw else assignment_raw
            data = json.loads(json_part)
            if isinstance(data, list) and len(data) > 0:
                args = data[0].get("arguments", {})
                assignment_text = f"**Zuweisung:** {args.get('name', 'N/A')}"
        except:
            pass
        
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
