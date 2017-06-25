from flask import Flask
from flask import request
import json
from model.git_good_model import GitGudModel
app = Flask(__name__)

model = None

@app.route('/api/check_message', methods=['POST'])
def check_message():
    # message = request.args.get('message')
    print(request.data)
    # print(json.load(request.))
    # print(request.get_json()['message'])

    msg = request.get_json()['message']
    print(msg)

    # pass message to model, return predicted quality of message
    model.predict_on_msg(msg)

    return 'Hello, World!'

def main():

    # Instantiate the wrapper model
    nonlocal model
    model = GitGudModel()

    print("Getting data!")

    # Read in the scrapped CSV data
    df = get_csv_data()

    print("Labeling data")

    # Label the data
    df = label_data(df)
    
    print("Engineer features!")
    
    # Engineer Features
    df = model.engineer_features(df)

    print("Parse and split!")

    X_train, X_test, y_train, y_test = model.parse_and_split_data(df)

    print("Training")

    # Train the model
    model = model.train(X_train, y_train)

    print("Predict")

    # Predict on the model
    pred = model.predict(X_test)

    model.report_scores(pred, y_test)


if __name__ == '__main__':
    main()
    app.run()
    