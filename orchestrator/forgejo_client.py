import requests
import os
from dotenv import load_dotenv

load_dotenv()

class ForgejoClient:
    def __init__(self):
        self.base_url = os.getenv("FORGEJO_URL").rstrip('/')
        self.token = os.getenv("FORGEJO_TOKEN")
        self.headers = {
            "Authorization": f"token {self.token}",
            "Content-Type": "application/json"
        }

    def get_issues(self, repo):
        url = f"{self.base_url}/api/v1/repos/{repo}/issues"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def post_comment(self, repo, issue_index, body):
        url = f"{self.base_url}/api/v1/repos/{repo}/issues/{issue_index}/comments"
        data = {"body": body}
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

    def create_issue(self, repo, title, body):
        url = f"{self.base_url}/api/v1/repos/{repo}/issues"
        data = {"title": title, "body": body}
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

    def get_org_repos(self, org):
        url = f"{self.base_url}/api/v1/orgs/{org}/repos"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
