import requests
import json

url = "https://ai-company.kluth.cloud/api/status"
try:
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        print("Status:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
