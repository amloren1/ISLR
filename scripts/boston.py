import sys
import os

# find filepath for data sets
if os.path.exists("../data"):
    fp = "../data/"
else:
    fp = "data/"


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.linear_model import LinearRegression



class Model(object):

    def __init__(self):

        self.raw_data = pd.read_csv(fp+"Boston.csv")
        breakpoint()
        self.raw_features = self.raw_data.loc[:, self.raw_data.columns != "medv"]
        self.med_val = self.raw_data.loc[:, self.raw_data.columns == "medv"]


    @staticmethod
    def pair_plot(self, y = None, fn =None):
        if y:
            sns.pairplot(self.raw_features, hue="cylinders")

    def linear_regresison(X=self.raw_features, y=self.med_val):
        LinearRegression().fit(X, y)


if __name__ == "__main__":
    model = Model()
    pass