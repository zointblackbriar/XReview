import requests
import json
from flask import Flask
from flask import request

app = Flask(__name__)

  
  
stringParam = "It is a good product"
 
def WebCrawlJSONCall():
    api_token = '472d294f4a4986ed807635eeabbd1ec0'
    api_url_base = 'https://api.dexi.io/'
    api_url = format(api_url_base)
    headers = {"X-DexiIO-Access": api_token,
           "X-DexiIO-Account": "13c85c77-14d2-4da3-952e-45e343e330fc",
                 "Content-Type":"application/json",
             "Accept":"application/json",
           "Accept-Encoding" : "gzip"}
    try:
        response = requests.post('https://api.dexi.io/executions', headers=headers)
        print(headers)
        if response.status_code == 200:
            print("Status 200")
        else:
            print("Status code:" + json.dumps(response.status_code))
    except ValueError:
        print("Oops there is an error about preparation of package")
     
 
def JSONParseFromFile():
    jsonData = []
    with open('test.json', encoding='utf-8') as data_file:
        jsonData = json.loads(data_file.read())
    #print ("data" + json.dumps(jsonData))
    print("Entry 1 : " + json.dumps(jsonData['firstEntry']))
    testFunction(json.dumps(jsonData['firstEntry']))
    print("\n")
    print("Entry 2 : " + json.dumps(jsonData['secondEntry']))
    testFunction(json.dumps(jsonData['secondEntry']))
    print("\n")
    print("Entry 3 : " + json.dumps(jsonData['thirdEntry']))
    testFunction(json.dumps(jsonData['thirdEntry']))
    print("\n")
    print("Entry 4 : " + json.dumps(jsonData['fourthEntry']))
    testFunction(json.dumps(jsonData['fourthEntry']))
    print("\n")
    print("Entry 5 : " + json.dumps(jsonData['fifthEntry']))
    testFunction(json.dumps(jsonData['fifthEntry']))
    print("\n")
    print("Entry 6 : " + stringParam)
    testFunction(stringParam)
    print("\n")
    inputEntry = input("Do you want to define your sentence whether negative or positive?")
    testFunction(inputEntry)
 
def todoRESTCall():
    print ('Call is waiting')
    resp = requests.get('https://192.168.1.3/tasks/')
    if resp.status_code != 200:
        #This means something wrong
#             raise ApiError('GET /tasks& {}'.format(resp.status_code))
              raise print("Error")
    for todo_item in resp.json():
            print('{}{}'.format(todo_item['id'], todo_item['summary']))
 
 
def parsingJsonRequest(json_string):
    parsed_json = json.loads(json_string)
    print(parsed_json['text'])
    testFunction(parsed_json['text'])
        
@app.route('/posneg')
def testFunction():

    text = request.args.get('text')
    data = json.dumps(text)
 
    response = requests.post('http://text-processing.com/api/sentiment/', data=data,
                  headers={'Content-Type': 'application/json'})
 
    if response.status_code == 200:
        print ('Status code 200')
        jsonDataTest = json.loads(response.content.decode('utf-8'))
        #return json.loads(response.content.decode('utf-8'))
        print('Result:' + json.dumps(jsonDataTest))
        return jsonDataTest
    else:
        print('Status code something else')
        return None
    return "hello"
    
if __name__=="__main__":
    app.run(host='127.0.0.1', port=4888, debug=True)


