import os
import json
import sys

# Ensure needle-repo is in path if not installed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "needle-repo"))

try:
    from needle import SimpleAttentionNetwork, load_checkpoint, generate, get_tokenizer
    HAS_NEEDLE = True
except ImportError:
    HAS_NEEDLE = False

class NeedleAgent:
    def __init__(self, role, prompt_template):
        self.role = role
        self.prompt_template = prompt_template
        self.params = None
        self.config = None
        self.model = None
        self.tokenizer = None
        
        if HAS_NEEDLE:
            checkpoint = os.getenv("NEEDLE_CHECKPOINT_PATH", "checkpoints/needle.pkl")
            if os.path.exists(checkpoint):
                self.params, self.config = load_checkpoint(checkpoint)
                self.model = SimpleAttentionNetwork(self.config)
                self.tokenizer = get_tokenizer()

    def query(self, context, tools=None):
        full_prompt = self.prompt_template.format(context=context)
        
        if HAS_NEEDLE and self.model:
            # Hier würde die tatsächliche Generierung stattfinden
            # return generate(self.model, self.params, self.tokenizer, query=full_prompt, tools=tools)
            return f"[Needle {self.role}]: (Simulierte Antwort basierend auf Needle Logik) Die Analyse von '{context[:30]}...' ist abgeschlossen."
        else:
            return f"[MOCK {self.role}]: Da keine Needle-Weights unter {os.getenv('NEEDLE_CHECKPOINT_PATH')} gefunden wurden, antworte ich im Simulationsmodus."

class BusinessAnalyst(NeedleAgent):
    def __init__(self):
        template = """
        Du bist ein Senior Business Analyst in einer KI-Softwarefirma.
        Analysiere den folgenden Forgejo-Issue und extrahiere:
        1. Hauptziel
        2. Technische Anforderungen
        3. Erforderliche Skills

        Kontext: {context}
        """
        super().__init__("Business Analyst", template)

class HRManager(NeedleAgent):
    def __init__(self):
        template = """
        Du bist der HR-Manager der KI-Firma. 
        Basierend auf der Analyse des Business Analysten, prüfe ob wir einen Experten haben.
        Verfügbare Experten: {specialists}
        
        Analyse: {context}
        
        Entscheidung: 
        - Zuweisen an [Name]
        - ODER: Neuer Experte muss 'eingestellt' (konfiguriert) werden.
        """
        super().__init__("HR Manager", template)
        
    def check_hiring(self, analysis, specialists_path):
        try:
            with open(specialists_path, 'r') as f:
                specialists = f.read()
        except FileNotFoundError:
            specialists = "{}"
            
        full_context = f"Spezialisten: {specialists}\nAnalyse: {analysis}"
        return self.query(full_context)
