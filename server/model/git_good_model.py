import pandas as pd
import numpy as np
import http.client, urllib.request, urllib.parse, urllib.error, base64
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
import random
import nltk
from nltk import word_tokenize, pos_tag
import json
import time

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

        self._vect = CountVectorizer(analyzer='word', stop_words='english', max_features = 850, ngram_range=(1, 1),
                           binary=False, lowercase=True)
        mcl_transformed = self._vect.fit_transform(X)

        # Adding Keyword Counts to X
        mcl_transformed = hstack((mcl_transformed, np.array(df['kwcount'])[:,None]))

        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(mcl_transformed, y, test_size=0.2, random_state=random.randint(1, 10))

        return (X_train, X_test, y_train, y_test)


    def engineer_features(self, df):
        keyword_counts = self._get_keywords(df['msg'])
        df['kwcount'] = pd.Series(keyword_counts)
        # df['kwcount'] = pd.Series(np.zeros(df.shape[0]))

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

    def predict(self, X):
        return list(self.model.predict(X))


    def report_scores(self, y_pred, y_actual):

        precision, recall, fscore, _ = score(y_actual, y_pred, labels=[0, 1])

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))

        # correct = 0
        # for index, y in enumerate(y_pred):
        #     if y == y_actual[index]:
        #         correct += 1
        #
        # print("Accuracy: ",correct/len(y_pred))


def get_csv_data():
    # Dummy get data function
    column_names = [
        "msg",
        "id",
        "stars",
        "forks",
        "watchers"
    ]

    df = pd.read_csv('../../git-csv-scraper/linux-commits2.csv', names=column_names)

    return df


def commit_body_has_swear(msg, weight):
    with open('../../data_attr/3ds_badwordlist0.txt', 'r') as content_file:
        content = content_file.read().replace("\r", "")
        content = content.split("\n")
        s = set(content)
        msg = msg.split(' ')
        s_msg = set(msg)
        union = set.intersection(*[s, s_msg])
        outcome = expon((1 - len(union) / len(s_msg)), 25)
    return outcome * weight


def commit_subject_doesnt_end_with_period(msg, weight):
    # Split by \s\s
    msg = msg.split('  ')

    # Assume subject doesnt end with period
    outcome = 1

    if msg[0].endswith('.'):
        outcome = 0

    return outcome * weight


def commit_subject_is_50chars_long(msg, weight):
    # Assume subject is under 50 chars
    outcome = 1

    # Split by \s\s
    msg = msg.split('  ')

    msglen = len(msg[0])

    if msglen > 50:
        outcome = 0

    return outcome * weight


def commit_body_has_bullet_points(msg, weight):
    outcome = 0

    delimiter = [' - ', ' * ']

    def has_delimiter(msg):
        for d in delimiter:
            # If delimiter exist
            if msg.find(d) is not -1:
                return True

    def has_list(msg):
        for d in delimiter:
            # If message contains the delimiter more than once
            if msg.find(d) > 1:
                return True

    if has_delimiter(msg) and has_list(msg):
        outcome = 1

    return outcome * weight


def expon(x, k):
    return x**k

def commit_subject_starts_with_capital(msg, weight):
     if msg[0].isupper():
         outcome = 1
     else:
         outcome = 0
     return outcome * weight


def commit_body_has_present_tense(msg, weight):
    text = word_tokenize(msg)
    tagged = pos_tag(text)
    present = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG", "VB"]])
    total = len([word for word in tagged if word[1] in ["VBD", "VBN"]]) + present
    if total == 0:
        return weight

    return expon((weight * present / total), 5)


def label_data(df, good_thresh=.5):
    label = []

    negfunctions = [(commit_body_has_swear, 1)]

    functions = [
        (commit_body_has_present_tense, .2),
        (commit_body_has_bullet_points, .2),
        (commit_subject_is_50chars_long, .2),
        (commit_subject_doesnt_end_with_period, .2),
        (commit_subject_starts_with_capital, .2),
    ]

    for index, row in df.iterrows():
        score = 0
        for funct, weight in functions:
            score += funct(row['msg'], weight)
        for funct, weight in negfunctions:
            score *= funct(row['msg'], weight)
        label.append(GOOD if score > good_thresh else BAD)

    df['label'] = pd.Series(label)
    return df


def main():
    # Instantiate the wrapper model
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
    start = time.time()
    main()
    print("Execution time: ", time.time() - start)
