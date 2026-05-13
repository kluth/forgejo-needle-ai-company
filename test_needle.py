import os
import sys

# Add needle to path
sys.path.append(os.path.join(os.getcwd(), "needle-repo"))

try:
    from needle import SimpleAttentionNetwork, load_checkpoint
    print("Needle import erfolgreich.")
    # Check if we can load default checkpoint
    # path = "checkpoints/needle.pkl"
    # if os.path.exists(path):
    #    params, config = load_checkpoint(path)
    #    print("Weights gefunden.")
except Exception as e:
    print(f"Fehler: {e}")
