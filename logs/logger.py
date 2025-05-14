# logs/logger.py
import os
import json
from datetime import datetime

def log_json(obj, name, subfolder="configs"):
    os.makedirs(f"logs/{subfolder}", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"logs/{subfolder}/{name}_{timestamp}.json", "w") as f:
        json.dump(obj, f, indent=4)

def log_note(text, name, subfolder="notes"):
    os.makedirs(f"logs/{subfolder}", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"logs/{subfolder}/{name}_{timestamp}.txt", "w") as f:
        f.write(text)