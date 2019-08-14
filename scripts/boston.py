import sys
import os

# find filepath for data sets
if os.path.exists("../data"):
    fp = "../data/"
else:
    fp = "data/"


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
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

    def summary(self, model, X, y_pred):
        y_test = self.med_val.values
        r2 = r2_score(y_test,y_pred)
        mse =  mean_squared_error(y_test, y_pred)

        b1_std_err = self.get_slopes_se(X, y_pred)

        t_statistic = model.intercept_[0]/b1_std_err
        breakpoint()
        p_val = scipy.stats.norm.sf(abs(t_statistic)) #one-sided

    def get_rss(self, y_pred):
        rss = 0
        y_test = self.med_val.values
        for i in range(len(y_test)):
            rss+= (y_test[i][0]-y_pred[i][0])**2

        return (rss/(len(y_test)-2))**0.5

    def get_slopes_se(self, X, y_pred):
        rss = self.get_rss(y_pred)
        x_mean = np.mean(X)
        x_var = list(map(lambda x: (x-x_mean)**2, X))
        x_var = sum(x_var)

        return (rss**2/x_var)**0.5

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
    regression, y_pred = model.linear_regresison(X = model.raw_features["lstat"], y = model.med_val)
    breakpoint()

    model.summary(regression, model.raw_features["lstat"].values, y_pred)
    model.plot_1p(model.raw_features["lstat"], model.med_val, y_pred)
    pass