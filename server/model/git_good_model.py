import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import random
import re

GOOD = 0
BAD = 1

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
        # print(type(mcl_transformed))
        # print(X)
        # exit()

        y = []
        for index, row in df.iterrows():
            has_stars = False
            has_fork = False
            has_watchers = False
            if int(row['stars']) > star_thresh:
                has_stars = True
            if int(row['forks']) > fork_thresh:
                has_fork = True
            if int(row['watchers']) > watchers_thresh:
                has_watchers = True

            if has_stars and has_fork and has_watchers:
                y.append(GOOD)
            else:
                y.append(BAD)

        # print(y)
        # exit()

        X_train, X_test, y_train, y_test = train_test_split(mcl_transformed, y, test_size=0.5, random_state=random.randint(1, 10))

        return (X_train, X_test, y_train, y_test)

    def train(self, X, y):
        self.model = self.model.fit(X,y)
        return self
        pass

    def predict(self, X):
        return list(self.model.predict(X))

        # return

    def report_scores(self, y_pred, y_actual):
        right = 0
        for index, y in enumerate(y_pred):
            if y == y_actual[index]:
                right += 1

        print(right/len(y_pred))


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

    df = pd.read_csv('../../data/google_commits.csv', names=column_names)

    return df

def has_swear(msg, weight):

    outcome = 1
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

def label_data(df, good_thresh = .5):
    label = []

    functions = [
        (has_swear, .6),
        (commit_subject_doesnt_end_with_period, .5),
        (commit_subject_is_50chars_long, .5),
    ]

    for index, row in df.iterrows():
        score = 0
        for funct, weight in functions:
            score += funct(row['msg'], weight)
        label.append(1 if score > good_thresh else 0)
    df['label'] = pd.Series(label)
    return df

def main():
    # Instantiate the wrapper model
    model = GitGudModel()

    # Read in the scrapped CSV data
    df = get_csv_data()

    # Label the data
    df = label_data(df)
    print(df)
    exit()



    X_train, X_test, y_train, y_test = model.parse_and_split_data(df)

    # print(X_train)
    # print(y_train)
    # exit()

    model = model.train(X_train, y_train)

    pred = model.predict(X_test)
    print(list(pred))


    print(y_test)

    model.report_scores(pred, y_test)




    # print(X_test)

    pass


if __name__ == '__main__':
    main()
