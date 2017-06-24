from flask import Flask
from flask import request
import json
app = Flask(__name__)

@app.route('/api/check_message', methods=['POST'])
def check_message():
    # message = request.args.get('message')
    print(request.data)
    # print(json.load(request.))
    print(request.get_json()['message'])

    # pass message to model, return predicted quality of message

    return 'Hello, World!'

def main():
    print("heijsndf")

if __name__ == '__main__':
    main()
    app.run()
    