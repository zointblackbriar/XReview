import requests
import json

#sample json object
data = {
    "id": "<item_id>", 
    "summary": "<one-line summary>"
    }
todo_item = json.dumps(data)

resp = requests.get('192.168.75.138:5000/tasks')
if resp.status_code != 200:
    raise print('GET /tasks/ {}'.format(resp.status_code))
else:
    print(resp)
    