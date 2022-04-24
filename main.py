import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# https://github.com/datsoftlyngby/soft2022spring-DS/blob/main/Code/E10-1-Iris-Bayes.ipynb

def console_setup():
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 20)


def preprocessing(df):
    print(df.head())
    print(df.columns)
    print(df.shape)
    print(df.describe())


def plot_graph_full_model(df):
    plt.hist(df)
    plt.xlabel("art's")
    plt.ylabel("number")
    plt.show()

    plt.hist(df['Atr1'], label="art1")
    plt.legend()
    plt.show()

    plt.matshow(df.corr())
    plt.show()
    #
    # pd.plotting.scatter_matrix(df, alpha=0.2)
    # plt.show()


def train_bayes_df_full_model(df):
    # every column except class column
    X = df.iloc[:, df.columns != "Class"]
    y = df['Class']

    # Split the dataset into two:
    #   80% of it as training data
    #   20% as a validation dataset
    # Let Python split the set into four, we tell the proportion of splitting
    test_set_size = 0.2

    # Initial value for randomization
    seed = 7
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_set_size, random_state=seed)

    model = GaussianNB()
    model.fit(X_train, y_train)

    print("score")
    print(model.score(X_test, y_test))
    # Test on the test data, try prediction
    prediction = model.predict(X_test)
    print("Prediction")
    print(prediction.shape)

    # tester score
    tt = X.iloc[0]
    tt = tt.array.reshape(1, -1)
    # print(tt)
    print("tester score")
    print(model.predict(tt))

    # Evaluation
    # Set the metrix
    scoring = 'accuracy'

    # print(X_test)
    # Calculated accuracy of the model over the validation set
    print(accuracy_score(y_test, prediction))
    # Confusion matrix provides an indication of the the errors of prediction
    print(confusion_matrix(y_test, prediction))
    # Classification report provides a breakdown of each class by precision, recall, f1-score and support
    print(classification_report(y_test, prediction))

    # testing model
    print("test the model")
    k = X.iloc[0].array.reshape(1, -1)
    my_prediction = model.predict(k)
    print("testing prediction")
    print(my_prediction)

    # testing model married
    print("test the model- married ##############")
    df = df.iloc[:, df.columns != "Class"]
    k = df.iloc[90].array.reshape(1, -1)
    my_prediction = model.predict(k)
    print("testing prediction- married==")
    print(my_prediction)



def two_data_sets_split(df):
    df1married = df[df['Class'] == 1]
    df0divorced = df[df['Class'] == 0]
    # print(df1married.tail())
    # print(df0divorced.tail())
    plot_married(df0divorced)
    plot_divorced(df0divorced)


def plot_married(df):
    df = df.iloc[:, df.columns != "Class"]
    plt.hist(df)
    plt.xlabel("art's")
    plt.ylabel("number")
    plt.title("married")
    plt.show()

    sns.heatmap(df.corr())
    plt.title("heapmap married")
    plt.show()


def plot_divorced(df):
    df = df.iloc[:, df.columns != "Class"]
    plt.hist(df)
    plt.xlabel("art's")
    plt.ylabel("number")
    plt.title("divorced")
    plt.show()

    sns.heatmap(df.corr())
    plt.title("heapmap divorced")
    plt.show()


if __name__ == '__main__':

    console_setup()
    URL = "./data/divorce.csv"
    df = pd.read_csv(URL, delimiter=';')
    # preprocessing(df)
    # plot_graph(df)
    train_bayes_df_full_model(df)
    two_data_sets_split(df)
