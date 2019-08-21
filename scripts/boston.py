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
import seaborn as sns

sns.set(style="ticks", color_codes=True)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

plt.style.use("ggplot")


class Model(object):
    def __init__(self):

        self.raw_data = pd.read_csv(fp + "Boston.csv")
        self.raw_features = self.raw_data.loc[:, self.raw_data.columns != "medv"]
        self.med_val = self.raw_data.loc[:, self.raw_data.columns == "medv"]

    @staticmethod
    def pair_plot(self, y=None, fn=None):
        if y:
            sns.pairplot(self.raw_features, hue="cylinders")

    def simple_linear_regresison(self, X=None, y=None):

        regression = LinearRegression()
        regression.fit(X, y.values.reshape(-1, 1))
        y_hat = regression.predict(X)

        return regression, y_hat

    def non_linear_transformations(self):
        """
         Lab: 3.6.5
        """

        lstat = self.raw_features.loc[:, self.raw_features.columns == "lstat"]

    def coef_summary(self, X, y_pred, coef):
        b1_std_err = self.get_slopes_se(X, y_pred)

        t_statistic = coef / b1_std_err

        p_val = scipy.stats.norm.sf(abs(t_statistic))  # one-sided
        print(p_val)
        breakpoint()
        return b1_std_err, t_statistic, p_val

    def summary(self, model, X, y_pred):
        """
            summary showing results of coefficients
        """
        y_test = self.med_val.values
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        summary_list = []
        i = 0
        print("coef     value    SE      t-stat     p-value     ")
        for x in X.columns:
            #breakpoint()
            coef = model.coef_[0][i]
            b1_std_err, t_statistic, p_val = self.coef_summary(X[x], y_pred, coef)

            summary_list.append((b1_std_err, t_statistic, p_val))
            i +=1
            # breakpoint()

            print(
                f"beta_1  {coef:.3f}    {b1_std_err:.3f}    {t_statistic:.3f}    {p_val:.5e}"
            )
        breakpoint()
    def get_rss(self, y_pred):
        rss = 0
        y_test = self.med_val.values
        for i in range(len(y_test)):
            rss += (y_test[i][0] - y_pred[i][0]) ** 2

        return (rss / (len(y_test) - 2)) ** 0.5

    def get_slopes_se(self, X, y_pred):
        rss = self.get_rss(y_pred)
        x_mean = np.mean(X)
        x_var = list(map(lambda x: (x - x_mean) ** 2, X))
        x_var = sum(x_var)

        return (rss ** 2 / x_var) ** 0.5

    @staticmethod
    def plot_1p(X, y_test, y_pred):
        fig, ax = plt.subplots()
        ax.scatter(X, y_test, color="b")
        ax.plot(X, y_pred, color="r")
        ax.set(xlabel="X1", ylabel="Median value", title="")
        plt.show()


if __name__ == "__main__":
    model = Model()

    # Lab 3.6.5
    X = pd.concat(
        [model.raw_features["lstat"], model.raw_features["lstat"] ** 2],
        axis=1,
        keys=["lstat", "lstat^2"],
    )

    regression, y_pred = model.simple_linear_regresison(X=X.values, y=model.med_val)
    model.summary(regression, X, y_pred)
    breakpoint()
    # Lab 3.6.4 interaction terms
    X = pd.concat(
        [
            model.raw_features["lstat"],
            model.raw_features["age"],
            model.raw_features["lstat"] * model.raw_features["age"],
        ],
        axis=1,
        keys=["lstat", "age", "lstat x age"],
    )
    regression, y_pred = model.simple_linear_regresison(X=X.values, y=model.med_val)
    model.summary(regression, model.raw_features["lstat"].values, y_pred)
    model.plot_1p(model.raw_features["lstat"], model.med_val, y_pred)
    pass
