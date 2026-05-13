import requests
import json

url = "https://ai-company.kluth.cloud/api/logs"
try:
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        logs = response.json().get("logs", [])
        print("Dashboard Logs:")
        for log in logs:
            print(log)
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
