import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import random
import nltk
from nltk import word_tokenize, pos_tag

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
    with open('../../data_attr/3ds_badwordlist0.txt', 'r') as content_file:
        content = content_file.read().replace("\r", "")
        content = content.split("\n")
        s = set(content)
        msg = msg.split(' ')
        s_msg = set(msg)
        union = set.intersection(*[s, s_msg])
        outcome = expon(len(union) / len(s_msg))
    return outcome * weight

def expon(x):
    pow = 1-x
    for i in range (25): #constant time, 50 is arbitrarily chosen
        pow *= 1-x
    return pow

def commit_subject_starts_with_capital(msg, weight):
     if msg[0].isupper():
         outcome = 1
     else:
         outcome = 0
     return outcome * weight

def determine_tense_input(msg, weight):
    text = word_tokenize(msg)
    tagged = pos_tag(text)
    present = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
    total = len([word for word in tagged if word[1] in ["VBD", "VB", "VBN"]]) + present
    if total == 0:
        print("none")
        return weight
    return(weight * present / total)

def label_data(df, good_thresh = .5):
    label = []

    functions = [(has_swear, .6)]

    for index, row in df.iterrows():
        score = 0
        for funct, weight in functions:
            score += funct(row['msg'], weight)
        label.append(1 if score > good_thresh else 0)
    df['label'] = pd.Series(label)
    return df

def main():
    a = determine_tense_input("learnt", 1)
    print (a)
    exit()


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
