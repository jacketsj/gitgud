from flask import Flask
from flask import request
import json
from model.git_good_model import GitGudModel
app = Flask(__name__)

model = None

@app.route('/api/check_message', methods=['POST'])
def check_message():

    print(request.data)

    msg = request.get_json()['message']
    print(msg)

    # Pass message to model, return predicted quality of message
    print(model)
    model.predict_on_msg(msg)

    return 'Hello, World!'


if __name__ == '__main__':

    # Instantiate the wrapper model
    model = GitGudModel()

    print("Getting data!")

    # Read in the scrapped CSV data
    df = model.get_csv_data()

    print("Labeling data")

    # Label the data
    df = model.label_data(df)
    
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

    app.run()
    