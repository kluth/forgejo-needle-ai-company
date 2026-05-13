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
        
    def run(self):
        print(f"Orchestrator gestartet. Überwache {self.inbox_repo}...")
        while True:
            try:
                issues = self.client.get_issues(self.inbox_repo)
                for issue in issues:
                    if issue['state'] == 'open' and not self.is_processed(issue):
                        self.process_task(issue)
                
                time.sleep(60) # Poll alle 60 Sekunden
            except Exception as e:
                print(f"Fehler beim Polling: {e}")
                time.sleep(30)

    def is_processed(self, issue):
        # Einfache Prüfung: Hat bereits jemand (die KI) kommentiert?
        # In einer echten App würde man Labels nutzen (z.B. 'analyzed')
        return issue['comments'] > 0

    def process_task(self, issue):
        from agent_engine import BusinessAnalyst, HRManager
        import json
        
        print(f"Verarbeite neuen Task: {issue['title']} (#{issue['number']})")
        
        # 1. Business Analyst Agent -> Analyse
        analyst = BusinessAnalyst()
        print("Analyst arbeitet...")
        analysis_raw = analyst.query(issue['body'] or issue['title'])
        
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
