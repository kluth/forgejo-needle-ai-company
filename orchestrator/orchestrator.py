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
        print(f"Verarbeite neuen Task: {issue['title']} (#{issue['number']})")
        
        # Hier würde der Aufruf an Needle erfolgen
        # 1. Business Analyst Agent -> Analyse
        # 2. HR Manager Agent -> Zuweisung
        
        # Mock-Antwort für Phase 1
        response = (
            "### Analyse durch KI-Business-Analyst\n"
            "Task erkannt. Ich analysiere die Anforderungen...\n\n"
            "### HR-Check\n"
            "Suche passenden Spezialisten..."
        )
        self.client.post_comment(self.inbox_repo, issue['number'], response)
        print("Mock-Antwort gesendet.")

if __name__ == "__main__":
    orchestrator = NeedleOrchestrator()
    orchestrator.run()
