import os
import subprocess
import time

env = os.environ.copy()
with open(".env") as f:
    for line in f:
        if "=" in line:
            key, val = line.strip().split("=", 1)
            env[key] = val

proc = subprocess.Popen(["./venv/bin/python", "webapp.py"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(15)
output = proc.stdout.read(4096)
print(output)
proc.terminate()
