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
plt.style.use('ggplot')


class Model(object):

    def __init__(self):

        self.raw_data = pd.read_csv(fp+"Boston.csv")
        self.raw_features = self.raw_data.loc[:, self.raw_data.columns != "medv"]
        self.med_val = self.raw_data.loc[:, self.raw_data.columns == "medv"]


    @staticmethod
    def pair_plot(self, y = None, fn =None):
        if y:
            sns.pairplot(self.raw_features, hue="cylinders")

    def linear_regresison(self, X=None, y=None):

        model = LinearRegression().fit(X.values.reshape(-1,1), y.values.reshape(-1,1))
        y_hat = model.fit(
            X.values.reshape(-1,1),
            y.values.reshape(-1,1)).predict(X.values.reshape(-1,1))
        return y_hat

    @staticmethod
    def plot_1p(X, y):
        fig, ax = plt.subplots()
        ax.plot(X, y)
        ax.set(xlabel= "X1", ylabel= "Median value",
        title="")
        plt.show()

if __name__ == "__main__":
    model = Model()
    y_hat = model.linear_regresison(X = model.raw_features["lstat"], y = model.med_val)
    breakpoint()
    pass