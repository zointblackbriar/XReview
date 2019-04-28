# import requests
import json
from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def main():
    return "Welcome!"

@app.route('/tasks')
def test():
    return "Another app point"
test()

if __name__=="__main__":
    app.run(host='127.0.0.1', port=4999)