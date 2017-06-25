import pandas as pd
import numpy as np
import http.client, urllib.request, urllib.parse, urllib.error, base64
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
import random
import json

GOOD = 1
BAD = 0

class GitGudModel():

    def __init__(self, dataframe=None):
        self.model = MultinomialNB()
        self.dataframe = dataframe
        pass

    def parse_and_split_data(self, df, *, star_thresh=200, fork_thresh=50, watchers_thresh=30):
        # go though datframe and parse data and assign labels
        X = df['msg']

        #TODO Account for first character capitalization.

        self._vect = CountVectorizer(analyzer='word', stop_words='english', max_features = 850, ngram_range=(1, 1),
                           binary=False, lowercase=True)
        mcl_transformed = self._vect.fit_transform(X)
        # print(type(mcl_transformed))
        # print(X)
        # exit()

        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(mcl_transformed, y, test_size=0.2, random_state=random.randint(1, 10))

        return (X_train, X_test, y_train, y_test)


    def engineer_features(self, df):
        keyword_counts = self._get_keywords(df['msg'])
        df['kwcount'] = pd.Series(keyword_counts)
        print(df)
        return df


    def _get_keywords(self, msgs):
        headers = { 'Content-Type': 'application/json', 'Ocp-Apim-Subscription-Key': '40bbff912e3a454bb5d07403932d6d88' }

        params = urllib.parse.urlencode({
        })

        document_list = []
        for index, msg in msgs.iteritems():
            msg_dict = {"language": "en", "id": str(index), "text": msg[:2500]}
            document_list.append(msg_dict)

        counts = np.zeros(len(document_list))

        start = 0
        end = 500
        # Cannot make API calls with massive data: "Limit request size to: 1048576 bytes"
        while(start < len(document_list)):
            body = {"documents" : document_list[start:end]}
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')

            try:
                conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
                conn.request("POST", "/text/analytics/v2.0/keyPhrases?%s" % params, body, headers)
                response = conn.getresponse()
                data = response.read()
                response_dict = json.loads(data)
                returned_docs = response_dict["documents"]
                for dictionary in returned_docs:
                    counts[int(dictionary["id"])] = len(dictionary["keyPhrases"])
                # print(response_dict)
                conn.close()
            except Exception as e:
                print("Error: {0}, {1}".format(e.errno, e.strerror))

            start = end
            end += 500

        return counts


    def train(self, X, y):
        self.model = self.model.fit(X,y)
        return self
        pass

    def predict(self, X):
        return list(self.model.predict(X))


    def report_scores(self, y_pred, y_actual):

        precision, recall, fscore, _ = score(y_actual, y_pred)

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))

        # correct = 0
        # for index, y in enumerate(y_pred):
        #     if y == y_actual[index]:
        #         correct += 1
        #
        # print("Accuracy: ",correct/len(y_pred))


def get_data():
    # Dummy get data function
    column_names = [
        "id",
        "msg",
        "stars",
        "forks",
        "watchers",
        "adds",
        "dels"
    ]

    df = pd.DataFrame([
                        ['22342', 'Hey good fix', '23', '43', '2', '23', '43'],
                        ['5435', 'Hey great fix', '300', '300', '300', '23', '43'],
                        ['6546', 'Hey poop fix', '200', '200', '2000', '23', '43'],
                        ['7453534', 'Hey man fix', '15', '64', '2', '23', '43'],
                    ], columns=column_names)

    return df


def get_csv_data():
    # Dummy get data function
    column_names = [
        "msg",
        "id",
        "stars",
        "forks",
        "watchers"
    ]

    df = pd.read_csv('../../data/linux-commits.csv', names=column_names)

    return df


def has_swear(msg, weight):

    outcome = 1
    return outcome * weight


def label_data(df, good_thresh = .5):
    label = []

    functions = [(has_swear, .6)]

    for index, row in df.iterrows():
        score = 0
        for funct, weight in functions:
            score += funct(row['msg'], weight)
        label.append(GOOD if score > good_thresh else BAD)
    df['label'] = pd.Series(label)
    return df




def main():
    # Instantiate the wrapper model
    model = GitGudModel()

    # Read in the scrapped CSV data
    df = get_csv_data()

    # Label the data
    df = label_data(df)

    # Engineer Features
    df = model.engineer_features(df)

    X_train, X_test, y_train, y_test = model.parse_and_split_data(df)

    # Train the model
    model = model.train(X_train, y_train)

    # Predict on the model
    pred = model.predict(X_test)

    model.report_scores(pred, y_test)

    pass


if __name__ == '__main__':
    main()
