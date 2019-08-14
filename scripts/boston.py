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
from sklearn.metrics import mean_squared_error, r2_score

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

        regression = LinearRegression().fit(X.values.reshape(-1,1), y.values.reshape(-1,1))
        regression.fit(X.values.reshape(-1,1),
                    y.values.reshape(-1,1))
        y_hat = regression.predict(X.values.reshape(-1,1))


        return regression, y_hat

    def summary(self, y_hat, y_test):

        r2 = r2_score(y_test,y_hat)
        mse =  mean_squared_error(y_test, y_hat)


    @staticmethod
    def plot_1p(X, y_test, y_pred):
        fig, ax = plt.subplots()
        ax.scatter(X, y_test, color = "b")
        ax.plot(X, y_pred, color = "r")
        ax.set(xlabel= "X1", ylabel= "Median value",
        title="")
        plt.show()

if __name__ == "__main__":
    model = Model()
    regression, y_hat = model.linear_regresison(X = model.raw_features["lstat"], y = model.med_val)
    breakpoint()
    model.plot_1p(model.raw_features["lstat"], model.med_val, y_hat)
    pass