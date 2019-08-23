import sys
import os

# find filepath for data sets
if os.path.exists("../data"):
    fp = "../data/"
else:
    fp = "data/"

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="ticks", color_codes=True)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class Data(object):
    def __init__(self):

        # pull in data, drop "day" column for now
        self.raw_data = pd.read_csv(fp + "Smarket.csv").drop("day", axis=1)
        self.raw_data.replace({"Direction": {"Up": 1, "Down": -1}}, inplace=True)
        self.raw_features = self.raw_data.loc[:, self.raw_data.columns != "Direction"]

        breakpoint()


def sec_4_6_1(data):
    """
        describe data, make plot to show correlation between features
    """
    print(data.raw_data.describe())

    print(data.raw_features.corr())

    # parir plot
    # sns.pairplot(data.raw_features,diag_kind="hist")
    # plt.show()


def sec_4_6_2(data):
    """
        logistic regression model to predict Direction or stock market on any given day based
        on lag1-5 and volume (previous day's)
    """
    logmodel = LogisticRegression()

    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=0
    )


if __name__ == "__main__":
    data = Data()

    sec_4_6_1(data)
