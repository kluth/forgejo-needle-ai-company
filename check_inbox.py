from orchestrator.forgejo_client import ForgejoClient
import os

client = ForgejoClient()
repo = "project-freedom/inbox"
try:
    issues = client.get_issues(repo)
    open_issues = [i for i in issues if i['state'] == 'open']
    print(f"Total Open Issues: {len(open_issues)}")
    for i in open_issues:
        print(f" - #{i['number']}: {i['title']} ({i['comments']} comments)")
        if i['comments'] > 0:
            comments = client.get_comments(repo, i['number'])
            last_comment = comments[-1]
            print(f"   Last comment by: {last_comment['user']['login']}")
except Exception as e:
    print(f"Error: {e}")
