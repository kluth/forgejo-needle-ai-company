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

def download_weights():
    try:
        from huggingface_hub import hf_hub_download
        local_dir = "data"
        os.makedirs(local_dir, exist_ok=True)
        print("Lade Needle-Weights von HuggingFace herunter...")
        path = hf_hub_download(
            repo_id="Cactus-Compute/needle",
            filename="needle.pkl",
            repo_type="model",
            local_dir=local_dir
        )
        print(f"Weights heruntergeladen: {path}")
        return path
    except Exception as e:
        print(f"Download-Fehler: {e}")
        return None

class NeedleAgent:
    def __init__(self, role, prompt_template):
        self.role = role
        self.prompt_template = prompt_template
        self.params = None
        self.config = None
        self.model = None
        self.tokenizer = None
        
        if HAS_NEEDLE:
            checkpoint = os.getenv("NEEDLE_CHECKPOINT_PATH", "data/needle.pkl")
            if not os.path.exists(checkpoint):
                checkpoint = download_weights() or checkpoint
            
            if os.path.exists(checkpoint):
                try:
                    self.params, self.config = load_checkpoint(checkpoint)
                    self.model = SimpleAttentionNetwork(self.config)
                    self.tokenizer = get_tokenizer()
                except Exception as e:
                    print(f"Fehler beim Laden von Needle: {e}")

    def query(self, context, **kwargs):
        try:
            full_prompt = self.prompt_template.format(context=context, **kwargs)
        except KeyError as e:
            # Fallback if templates differ
            full_prompt = f"System Error: Missing key {e} in template. Context: {context}"
        
        if HAS_NEEDLE and self.model:
            try:
                # Actual generation using Needle
                result = generate(self.model, self.params, self.tokenizer, query=full_prompt)
                return f"[Needle {self.role}]: {result}"
            except Exception as e:
                return f"[Needle {self.role} Error]: {e}"
        else:
            return f"[MOCK {self.role}]: {context[:100]}..."

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
            
        return self.query(context=analysis, specialists=specialists)
